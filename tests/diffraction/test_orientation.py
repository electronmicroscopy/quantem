"""Round-trip tests for quantem.diffraction.orientation."""

import numpy as np
import pytest
import torch
from ase.build import bulk

from quantem.core.datastructures.vector import Vector
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.orientation import OrientationMap
from quantem.diffraction.rotations import misorientation_angle_deg, qnormalize


def _make_peaks(xtl, q_true, sigma=0.02):
    N = q_true.shape[0]
    peaks = Vector.from_shape(
        (1, N), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(N):
        p = xtl.generate_pattern(q_true[i], energy_ev=200e3, sigma_excitation=sigma)
        peaks[0, i] = np.stack([p["qx"].numpy(), p["qy"].numpy(), p["intensity"].numpy()], axis=1)
    return peaks


@pytest.mark.parametrize(
    "builder,kwargs",
    [
        (bulk, dict(name="Ti", crystalstructure="bcc", a=3.31, cubic=True)),
        (bulk, dict(name="Ti", crystalstructure="hcp", a=2.95, c=4.686)),
    ],
)
def test_roundtrip_matching(builder, kwargs):
    torch.manual_seed(3)
    xtl = Crystal.from_ase(builder(**kwargs))
    xtl.calculate_structure_factors(k_max=1.5)
    N = 15
    q_true = qnormalize(torch.randn(N, 4, dtype=torch.float64))
    peaks = _make_peaks(xtl, q_true)

    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=2.0, angle_step_in_plane_deg=2.0, power_intensity=0.0)
    om.match_orientations(progress_bar=False)
    # noiseless synthetic data: the envelope tilt is exact, so allow the
    # full grid-scale correction (the default trust region is sized for
    # noisy measured intensities)
    om.refine_orientations(zone_max_total_deg=1.5, progress_bar=False)

    err = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()
    # majority recovered to well below the grid step; a small number of
    # kinematically (near-)degenerate orientations may land elsewhere
    assert np.median(err) < 0.1
    assert (err < 1.0).mean() >= 0.7


def test_normalized_scores_and_reliability():
    torch.manual_seed(0)
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True))
    xtl.calculate_structure_factors(k_max=1.5)
    q_true = qnormalize(torch.randn(6, 4, dtype=torch.float64))
    peaks = _make_peaks(xtl, q_true)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=3.0, angle_step_in_plane_deg=3.0)
    om.match_orientations(progress_bar=False)

    assert float(om.corr.max()) <= 1.0 + 1e-9
    assert float(om.corr.min()) >= 0.0
    assert om.reliability is not None
    assert (om.reliability[0] > 0).all()


def test_mirror_channel():
    """Orientations in the opposite hemisphere are matched via the mirror."""
    torch.manual_seed(5)
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True))
    xtl.calculate_structure_factors(k_max=1.5)
    q_true = qnormalize(torch.randn(10, 4, dtype=torch.float64))
    peaks = _make_peaks(xtl, q_true)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=2.0, angle_step_in_plane_deg=2.0)
    om.match_orientations(progress_bar=False)
    om.refine_orientations(progress_bar=False)
    err = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()
    used_mirror = om.mirror[0, :, 0].numpy()
    # both channels appear and mirror matches are as accurate as direct ones
    assert used_mirror.any()
    assert (~used_mirror).any()
    ok = err < 5
    assert ok.mean() >= 0.7
    assert np.median(err[ok & used_mirror]) < 0.5


def test_square_detector_correction():
    """Peaks clipped by a square detector: the aperture-normalized match
    recovers the orientation as well as the unclipped case."""
    torch.manual_seed(7)
    xtl = Crystal.from_ase(bulk("Ti", "hcp", a=2.95, c=4.686))
    xtl.calculate_structure_factors(k_max=1.5)
    q_true = qnormalize(torch.randn(10, 4, dtype=torch.float64))
    q_det = 0.9  # detector half-width < k_max: corners clipped

    peaks = Vector.from_shape(
        (1, 10), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(10):
        p = xtl.generate_pattern(q_true[i], energy_ev=200e3, sigma_excitation=0.02)
        keep = (p["qx"].abs() < q_det) & (p["qy"].abs() < q_det)
        peaks[0, i] = np.stack(
            [p["qx"][keep].numpy(), p["qy"][keep].numpy(), p["intensity"][keep].numpy()],
            axis=1,
        )

    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(
        angle_step_zone_axis_deg=2.0,
        angle_step_in_plane_deg=2.0,
        detector_q_max=q_det,
    )
    om.match_orientations(progress_bar=False)
    err = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()
    assert (err < 5).mean() >= 0.7
    # with the aperture correction, kernel leakage at the hard detector edge
    # can push the normalized score a few percent above 1
    assert float(om.corr.max()) <= 1.05


def _ase(spacegroup, symbols, basis, cellpar):
    from ase.spacegroup import crystal as ase_crystal

    return ase_crystal(symbols, basis=basis, spacegroup=spacegroup, cellpar=cellpar)


@pytest.mark.parametrize(
    "label,atoms,step",
    [
        (
            "Bi -3m",
            lambda: _ase(166, ["Bi"], [(0, 0, 0.234)], [4.55, 4.55, 11.86, 90, 90, 120]),
            2.0,
        ),
        # ilmenite's projections are nearly mirror symmetric, so the flipped
        # orientation is a close rival and needs a finer zone grid than Bi
        (
            "ilmenite -3",
            lambda: _ase(
                148,
                ["Fe", "Ti", "O"],
                [(0, 0, 0.355), (0, 0, 0.146), (0.317, 0.023, 0.245)],
                [5.09, 5.09, 14.09, 90, 90, 120],
            ),
            1.5,
        ),
    ],
)
def test_roundtrip_low_symmetry(label, atoms, step):
    # low-symmetry crystals see errors that cubic and hexagonal symmetry
    # hides: a wrong wedge (trigonal) or a redundant library
    torch.manual_seed(5)
    xtl = Crystal.from_ase(atoms(), name=label, verbose=False)
    xtl.calculate_structure_factors(k_max=1.3)
    N = 20
    q_true = qnormalize(torch.randn(N, 4, dtype=torch.float64))
    peaks = _make_peaks(xtl, q_true)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=step, angle_step_in_plane_deg=2.0, verbose=False)
    om.match_orientations(progress_bar=False)
    err = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()
    assert (err < 2.5).mean() >= 0.8
    # symmetry copies of the best zone must not count as the second best
    assert float(np.median(om.reliability[0].numpy())) > 0.02


def test_reliability_with_hemisphere_library():
    # Laue 2/m has no wedge: the hemisphere library holds every zone twice,
    # and reliability must still see past the symmetry copy
    torch.manual_seed(2)
    atoms = _ase(
        14,
        ["Zr", "O", "O"],
        [(0.275, 0.040, 0.208), (0.070, 0.332, 0.345), (0.450, 0.758, 0.479)],
        [5.15, 5.21, 5.32, 90, 99.2, 90],
    )
    xtl = Crystal.from_ase(atoms, name="ZrO2", pseudo_symmetry_tol=None, verbose=False)
    xtl.calculate_structure_factors(k_max=1.2)
    assert xtl.zone_axis_wedge() is None
    q_true = qnormalize(torch.randn(8, 4, dtype=torch.float64))
    peaks = _make_peaks(xtl, q_true)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=3.0, angle_step_in_plane_deg=3.0, verbose=False)
    om.match_orientations(progress_bar=False)
    assert float(np.median(om.reliability[0].numpy())) > 0.02


def test_pseudo_symmetry_warning_on_plan():
    import warnings

    from ase import Atoms

    ortho = Atoms("Au", positions=[[0, 0, 0]], cell=[4.000, 4.001, 4.002], pbc=True)
    xtl = Crystal.from_ase(ortho, verbose=False)
    xtl.calculate_structure_factors(k_max=1.2)
    q_true = qnormalize(torch.randn(2, 4, dtype=torch.float64))
    peaks = _make_peaks(xtl, q_true)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        om.build_plan(angle_step_zone_axis_deg=5.0, angle_step_in_plane_deg=5.0, verbose=False)
    assert any("pseudo-symmetry" in str(x.message) for x in w)


def test_metadata_inheritance():
    # each stage records its hyperparameters; later stages inherit what is
    # left as None, so one tuned value propagates through the whole chain
    from quantem.diffraction import bloch
    from quantem.diffraction.phase import PhaseMap

    torch.manual_seed(1)
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True), verbose=False)
    xtl.calculate_structure_factors(k_max=1.5)
    q_true = qnormalize(torch.randn(4, 4, dtype=torch.float64))
    peaks = _make_peaks(xtl, q_true)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3, precession_deg=0.7)
    om.build_plan(
        angle_step_zone_axis_deg=3.0, corr_kernel_size=0.04, power_intensity=0.3, verbose=False
    )
    om.match_orientations(progress_bar=False)
    om.refine_orientations(progress_bar=False)
    assert om.metadata["plan"]["pair_distance"] == 0.04
    assert om.metadata["refine"]["pair_distance"] == 0.04
    assert om.metadata["match"]["min_number_peaks"] == 5
    pm = PhaseMap.from_orientation_maps([om])
    pm.fit(progress_bar=False)
    assert pm.metadata["fit"]["pair_distance"] == 0.04
    assert pm.metadata["fit"]["power_intensity"] == 0.3
    assert pm.metadata["fit"]["min_number_peaks"] == 5
    xtl.calculate_dynamical_structure_factors(energy_ev=200e3, k_max=2.0)
    res = bloch.refine_dynamical(
        pm,
        thicknesses_A=[300.0, 400.0],
        tilt_stages=((0.1, 0.1),),
        n_precession=4,
        k_max=1.0,
        mask=np.array([[True, False, False, False]]),
        progress_bar=False,
    )
    md = res["metadata"]
    assert md["precession_deg"] == 0.7 and md["pair_distance"] == 0.04
    assert md["power_intensity"] == 0.3 and md["min_number_peaks"] == 5
    assert pm.metadata["dynamical"] is md
    # explicit values still win
    res2 = bloch.refine_dynamical(
        pm,
        thicknesses_A=[300.0],
        tilt_stages=((0.1, 0.1),),
        n_precession=4,
        precession_deg=0.0,
        pair_distance=0.06,
        k_max=1.0,
        mask=np.array([[True, False, False, False]]),
        progress_bar=False,
    )
    assert res2["metadata"]["precession_deg"] == 0.0 and res2["metadata"]["pair_distance"] == 0.06


def test_precession_envelope_matches_quadrature():
    # the analytic ring-averaged envelope equals the positive quadrature of
    # the static envelope over the exact excitation errors on the ring
    from quantem.diffraction.illumination import ring_disk_quadrature
    from quantem.diffraction.rotations import qrotate

    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True), verbose=False)
    xtl.calculate_structure_factors(k_max=1.5)
    torch.manual_seed(4)
    q = qnormalize(torch.randn(4, dtype=torch.float64))
    energy_ev, sigma, prec = 200e3, 0.04, 0.6
    from quantem.core.utils.utils import electron_wavelength_angstrom

    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    pat = xtl.generate_pattern(q, energy_ev, sigma_excitation=sigma, precession_deg=prec)
    g = qrotate(q, xtl.g_vec)
    hkl_map = {tuple(h): i for i, h in enumerate(xtl.hkl.tolist())}
    idx = torch.tensor([hkl_map[tuple(h)] for h in pat["hkl"].tolist()])
    gs = g[idx].numpy()
    r = k0 * np.sin(np.deg2rad(prec))
    t, w = ring_disk_quadrature(r, 0.0, n_phi=256)
    kz = np.sqrt(k0**2 - (t**2).sum(1))[:, None]
    s = (2 * kz * gs[:, 2] - 2 * (t @ gs[:, :2].T) - (gs**2).sum(1)) / (2 * (kz - gs[:, 2]))
    ref = (w[:, None] * np.exp(-0.5 * (s / sigma) ** 2)).sum(0) * xtl.struct_factors_int[
        idx
    ].numpy()
    assert np.allclose(pat["intensity"].numpy(), ref, rtol=1e-6, atol=1e-9)
    # without precession the static envelope is recovered exactly
    pat0 = xtl.generate_pattern(q, energy_ev, sigma_excitation=sigma)
    s0 = pat0["s_g"].numpy()
    assert np.allclose(
        pat0["intensity"].numpy(),
        xtl.struct_factors_int[[hkl_map[tuple(h)] for h in pat0["hkl"].tolist()]].numpy()
        * np.exp(-0.5 * (s0 / sigma) ** 2),
    )


def test_roundtrip_with_precession():
    # library, matching and refinement with the precession-averaged
    # envelope: patterns simulated with precession are recovered
    torch.manual_seed(6)
    xtl = Crystal.from_ase(bulk("Ti", "hcp", a=2.95, c=4.686), verbose=False)
    xtl.calculate_structure_factors(k_max=1.5)
    N = 12
    q_true = qnormalize(torch.randn(N, 4, dtype=torch.float64))
    peaks = Vector.from_shape(
        (1, N), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(N):
        p = xtl.generate_pattern(
            q_true[i], energy_ev=200e3, sigma_excitation=0.02, precession_deg=0.7
        )
        peaks[0, i] = np.stack([p["qx"].numpy(), p["qy"].numpy(), p["intensity"].numpy()], axis=1)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3, precession_deg=0.7)
    om.build_plan(angle_step_zone_axis_deg=2.0, verbose=False)
    om.match_orientations(progress_bar=False)
    om.refine_orientations(zone_max_total_deg=1.5, progress_bar=False)
    err = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()
    assert np.median(err) < 0.3
    assert (err < 1.5).mean() >= 0.75
    assert om.metadata["precession_deg"] == 0.7


def test_staged_positions_subset():
    """Matching a few positions leaves the rest untouched, and the later
    stages follow the subset without repeating it."""
    torch.manual_seed(5)
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True))
    xtl.calculate_structure_factors(k_max=1.5)
    N = 6
    q_true = qnormalize(torch.randn(N, 4, dtype=torch.float64))
    peaks = _make_peaks(xtl, q_true)

    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=2.0, angle_step_in_plane_deg=2.0, power_intensity=0.0)

    test_pos = [(0, 1), (0, 4)]
    om.match_orientations(positions=test_pos, progress_bar=False)
    assert om.computed.sum() == len(test_pos)
    assert bool(om.computed[0, 1]) and bool(om.computed[0, 4])
    assert float(om.corr[0, 0, 0]) == 0.0  # not requested, untouched
    assert float(om.corr[0, 1, 0]) > 0.5

    # refinement follows `computed` with no position list of its own
    before = om.quats.clone()
    om.refine_orientations(progress_bar=False, zone_max_total_deg=1.5)
    untouched = torch.allclose(before[0, 0], om.quats[0, 0])
    assert untouched
    err = misorientation_angle_deg(q_true[[1, 4]], om.quats[0, [1, 4], 0], xtl.sym_quats).numpy()
    assert np.all(err < 1.0)

    # the full run then covers everything
    om.match_orientations(progress_bar=False)
    assert bool(om.computed.all())
    assert float(om.corr[0, 0, 0]) > 0.5


def test_fiber_zone_axis_range():
    """A fiber plan of zero half angle samples one zone axis and still
    recovers the in-plane angle exactly."""
    torch.manual_seed(7)
    xtl = Crystal.from_ase(bulk("Ti", "hcp", a=2.9505, c=4.6855))
    xtl.calculate_structure_factors(k_max=1.5)
    N = 6
    gam = torch.rand(N, dtype=torch.float64) * 2 * np.pi
    q_true = qnormalize(
        torch.stack(
            [torch.cos(gam / 2), torch.zeros(N), torch.zeros(N), torch.sin(gam / 2)], dim=1
        )
    )
    peaks = _make_peaks(xtl, q_true)

    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(
        zone_axis_range="fiber",
        fiber_axis=[0, 0, 0, 1],  # Miller-Bravais [0001]
        fiber_angle_deg=0.0,
        angle_step_in_plane_deg=2.0,
        power_intensity=0.0,
        verbose=False,
    )
    assert om.zone_axes.shape[0] == 1
    om.match_orientations(progress_bar=False)
    om.refine_orientations(progress_bar=False, zone_max_total_deg=0.5)
    err = misorientation_angle_deg(q_true, om.quats[0, :, 0], xtl.sym_quats).numpy()
    assert np.all(err < 0.2)

    # a cap of a few degrees covers a spread of tilts, and the hemisphere
    # fallback and the symmetry wedge both stay available
    om2 = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om2.build_plan(
        zone_axis_range="fiber", fiber_axis=[0, 0, 1], fiber_angle_deg=5.0, verbose=False
    )
    assert om2.zone_axes.shape[0] > 1
    om3 = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om3.build_plan(zone_axis_range="full", angle_step_zone_axis_deg=4.0, verbose=False)
    om4 = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om4.build_plan(angle_step_zone_axis_deg=4.0, verbose=False)
    assert om3.zone_axes.shape[0] > om4.zone_axes.shape[0]


def test_power_intensity_experiment_is_separate():
    """The measured-intensity exponent defaults to the library one and can
    be set independently."""
    torch.manual_seed(11)
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True))
    xtl.calculate_structure_factors(k_max=1.5)
    q_true = qnormalize(torch.randn(3, 4, dtype=torch.float64))
    peaks = _make_peaks(xtl, q_true)

    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=3.0, power_intensity=0.25, verbose=False)
    assert om.power_intensity_experiment == 0.25

    om2 = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om2.build_plan(
        angle_step_zone_axis_deg=3.0,
        power_intensity=0.25,
        power_intensity_experiment=0.0,
        verbose=False,
    )
    assert om2.power_intensity_experiment == 0.0
    assert om2.metadata["plan"]["power_intensity_experiment"] == 0.0
    om2.match_orientations(progress_bar=False)
    assert float(om2.corr[0, 0, 0]) > 0.3


def test_in_plane_angle_auto_fold():
    """The automatic fold removes the in-plane ambiguity of a <111> zone.

    Two orientations 60 degrees apart about a body-centered cubic <111> beam
    give the same zero-layer pattern, so matching returns one or the other at
    random. Folding by the projected order makes the reported angle the same
    for both, which is what keeps an in-plane map continuous.
    """
    from quantem.diffraction.rotations import (
        qmult,
        quat_from_axis_angle,
        quat_from_zone_axis,
    )

    torch.manual_seed(2)
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.26, cubic=True))
    xtl.calculate_structure_factors(k_max=1.5)
    d = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64) @ xtl.lat_real
    q0 = quat_from_zone_axis(d / torch.linalg.norm(d))
    beam = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

    # N in-plane angles, each also present as its 60 degree twin
    N = 5
    spin = torch.linspace(0.0, 1.0, N, dtype=torch.float64)
    q_a = qnormalize(qmult(quat_from_axis_angle(beam, spin), q0))
    q_b = qnormalize(qmult(quat_from_axis_angle(beam, torch.tensor(np.deg2rad(60.0))), q_a))
    q_true = torch.stack([q_a, q_b], dim=0)  # (2, N, 4)

    peaks = Vector.from_shape(
        (2, N), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(2):
        for j in range(N):
            p = xtl.generate_pattern(q_true[i, j], energy_ev=200e3, sigma_excitation=0.02)
            peaks[i, j] = np.stack(
                [p["qx"].numpy(), p["qy"].numpy(), p["intensity"].numpy()], axis=1
            )

    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=2.0, angle_step_in_plane_deg=2.0, power_intensity=0.0)
    om.match_orientations(progress_bar=False)
    om.refine_orientations(progress_bar=False, zone_max_total_deg=1.5)

    assert xtl.projected_rotation_order((d / torch.linalg.norm(d)).numpy()) == 6
    folded = om.in_plane_angle_deg(mod_deg="auto").numpy()
    assert folded.max() <= 60.0 + 1e-6
    # the twin rows must agree once folded, to well under the library step
    delta = np.abs(folded[0] - folded[1]) % 60.0
    delta = np.minimum(delta, 60.0 - delta)
    assert np.all(delta < 1.0), delta
    # explicit values and None still behave as before
    assert om.in_plane_angle_deg(mod_deg=90.0).max() <= 90.0
    assert om.in_plane_angle_deg(mod_deg=None).max() > 60.0


def test_fold_in_plane_collapses_degenerate_variants():
    """Two orientations with the same zero-layer pattern get the same color."""
    from quantem.diffraction.orientation_visualization import fold_in_plane, ipf_color
    from quantem.diffraction.rotations import (
        qmult,
        quat_from_axis_angle,
        quat_from_zone_axis,
    )

    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.26, cubic=True), verbose=False)
    xtl.calculate_structure_factors(k_max=1.5)
    d = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64) @ xtl.lat_real
    q0 = quat_from_zone_axis(d / torch.linalg.norm(d))
    beam = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    tilt_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

    torch.manual_seed(4)
    for tilt in (0.0, 1.8, 3.0):
        base = qnormalize(
            qmult(quat_from_axis_angle(tilt_axis, torch.tensor(np.deg2rad(tilt))), q0)
        )
        spin = torch.rand(8, dtype=torch.float64) * 2 * np.pi
        q_a = qnormalize(qmult(quat_from_axis_angle(beam, spin), base))
        q_b = qnormalize(qmult(quat_from_axis_angle(beam, torch.tensor(np.deg2rad(60.0))), q_a))
        folded_a = fold_in_plane(q_a, xtl)
        folded_b = fold_in_plane(q_b, xtl)
        assert torch.allclose(torch.abs(folded_a), torch.abs(folded_b), atol=1e-8)
        c_a = ipf_color(folded_a, xtl, "r")
        c_b = ipf_color(folded_b, xtl, "r")
        assert np.abs(c_a - c_b).max() < 1e-6, tilt
        # the out-of-plane color never depended on the in-plane angle
        assert np.abs(ipf_color(q_a, xtl, "z") - ipf_color(q_b, xtl, "z")).max() < 1e-6

    # far from the pole the ambiguity is gone and nothing is folded
    far = qnormalize(qmult(quat_from_axis_angle(tilt_axis, torch.tensor(np.deg2rad(20.0))), q0))
    assert torch.allclose(fold_in_plane(far[None], xtl)[0], far, atol=1e-12)


def test_smooth_orientations():
    """Bilateral smoothing averages noise inside a grain, not across variants."""
    from quantem.diffraction.rotations import qmult, quat_from_axis_angle

    torch.manual_seed(6)
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True))
    xtl.calculate_structure_factors(k_max=1.5)

    # a 6 x 6 patch of one orientation with half a degree of scatter, plus a
    # second grain 30 degrees away filling the right-hand columns
    R, C = 6, 6
    base = qnormalize(torch.randn(4, dtype=torch.float64))
    axis = torch.randn(R, C, 3, dtype=torch.float64)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    noise = quat_from_axis_angle(axis.reshape(-1, 3), torch.deg2rad(0.5 * torch.randn(R * C)))
    q = qnormalize(qmult(noise, base)).reshape(R, C, 4)
    other = qnormalize(
        qmult(
            quat_from_axis_angle(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(np.deg2rad(30.0))),
            base,
        )
    )
    q[:, 4:] = other

    peaks = Vector.from_shape(
        (R, C), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(R):
        for j in range(C):
            p = xtl.generate_pattern(q[i, j], energy_ev=200e3, sigma_excitation=0.02)
            peaks[i, j] = np.stack(
                [p["qx"].numpy(), p["qy"].numpy(), p["intensity"].numpy()], axis=1
            )
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=3.0, verbose=False)
    om.quats = q.clone()[..., None, :]
    om.corr = torch.ones((R, C, 1), dtype=torch.float64)
    om.computed = torch.ones((R, C), dtype=torch.bool)

    before = misorientation_angle_deg(base, om.quats[:, :4, 0].reshape(-1, 4), xtl.sym_quats)
    om.smooth_orientations(sigma_px=1.0, sigma_deg=1.0, max_angle_deg=5.0)
    after = misorientation_angle_deg(base, om.quats[:, :4, 0].reshape(-1, 4), xtl.sym_quats)
    assert float(after.mean()) < float(before.mean()), (float(before.mean()), float(after.mean()))

    # the second grain is 30 degrees away, beyond max_angle_deg, so it is
    # neither pulled toward the first nor allowed to pull on it
    kept = misorientation_angle_deg(other, om.quats[:, 5, 0], xtl.sym_quats)
    assert float(kept.max()) < 1e-6


def test_display_smoothing_needs_both_widths():
    """A bare number is refused: the angular tolerance must not default silently."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(8)
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True))
    xtl.calculate_structure_factors(k_max=1.5)
    R, C = 4, 4
    q = qnormalize(torch.randn(4, dtype=torch.float64)).expand(R, C, 4).clone()
    peaks = Vector.from_shape(
        (R, C), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(R):
        for j in range(C):
            p = xtl.generate_pattern(q[i, j], energy_ev=200e3, sigma_excitation=0.02)
            peaks[i, j] = np.stack(
                [p["qx"].numpy(), p["qy"].numpy(), p["intensity"].numpy()], axis=1
            )
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=4.0, verbose=False)
    om.quats = q.clone()[..., None, :]
    om.corr = torch.ones((R, C, 1), dtype=torch.float64)
    om.computed = torch.ones((R, C), dtype=torch.bool)

    with pytest.raises(TypeError, match="angular tolerance"):
        om.plot_orientation(smooth=1.0)

    before = om.quats.clone()
    om.plot_orientation(smooth={"sigma_px": 1.0, "sigma_deg": 1.0, "max_angle_deg": 5.0})
    om.plot_orientation(smooth=True)
    plt.close("all")
    # smoothing for display must never touch the stored orientations
    assert torch.equal(before, om.quats)


def test_rescue_breaks_friedel_ties_by_neighbours():
    """At a zone axis the pattern rotated 180 degrees about the beam is the
    same pattern, so a scattered twin can only be undone by its neighbours."""
    from quantem.diffraction.rotations import qmult, quat_from_zone_axis

    xtl = Crystal.from_ase(bulk("Ti", "hcp", a=2.95, c=4.686))
    xtl.calculate_structure_factors(k_max=1.5)
    zone = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64) @ xtl.lat_real.to(torch.float64)
    q = quat_from_zone_axis(zone, 20.0)
    twin = qmult(torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64), q)
    assert float(misorientation_angle_deg(q, twin, xtl.sym_quats_matching)) > 10.0

    R, C = 5, 5
    peaks = Vector.from_shape(
        (R, C), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    p = xtl.generate_pattern(q, energy_ev=200e3, sigma_excitation=0.02)
    for i in range(R):
        for j in range(C):
            peaks[i, j] = np.stack(
                [p["qx"].numpy(), p["qy"].numpy(), p["intensity"].numpy()], axis=1
            )
    flipped = torch.zeros((R, C), dtype=torch.bool)
    flipped[1, 1] = flipped[2, 3] = flipped[3, 1] = True

    def run(consensus_tol):
        om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
        om.build_plan(angle_step_zone_axis_deg=2.0, verbose=False, progress_bar=False)
        om.quats = torch.where(flipped[..., None], twin, q).clone()[..., None, :]
        om.corr = torch.ones((R, C, 1), dtype=torch.float64)
        om.computed = torch.ones((R, C), dtype=torch.bool)
        om.refine_orientations(consensus_tol=consensus_tol, progress_bar=False)
        return misorientation_angle_deg(q, om.quats[..., 0, :], xtl.sym_quats_matching)

    # each pattern alone cannot tell the twin apart ...
    assert float(run(0.0)[flipped].min()) > 10.0
    # ... but it is a tie, and every neighbour holds the other variant
    assert float(run(0.01).max()) < 2.0


def test_ipf_key_is_smooth_with_exact_corners_and_white_centre():
    from quantem.diffraction.orientation_visualization import IPF_CORNER_COLORS, _bary_to_rgb

    assert np.allclose(_bary_to_rgb(np.eye(3)), IPF_CORNER_COLORS)
    assert np.allclose(_bary_to_rgb(np.ones(3) / 3), 1)
    # no creases: along lines across the wedge, including across the lines
    # where one corner takes over from another, the color turns gently
    t = np.linspace(0, 1, 801)[:, None]
    for p0, p1 in (
        ((0.55, 0.40, 0.05), (0.20, 0.10, 0.70)),
        ((0.90, 0.05, 0.05), (0.05, 0.90, 0.05)),
        ((0.70, 0.30, 0.00), (0.00, 0.30, 0.70)),
    ):
        c = _bary_to_rgb(np.array(p0) * (1 - t) + np.array(p1) * t)
        assert np.abs(np.diff(c, 2, axis=0)).max() < 1e-4
    # the edges stay colored all along: midway between two corners is vivid
    for i, j in ((0, 1), (1, 2), (2, 0)):
        w = np.zeros(3)
        w[i] = w[j] = 0.5
        rgb = _bary_to_rgb(w)
        assert rgb.max() - rgb.min() > 0.5


def test_wedge_labels_name_zone_axes_with_positive_leading_index():
    import re

    for xtl in (
        Crystal.from_ase(bulk("Cu", "fcc", a=3.6, cubic=True), verbose=False),
        Crystal.from_ase(bulk("Ti", "hcp", a=2.95, c=4.68), verbose=False),
    ):
        for label in xtl.zone_axis_wedge_labels(mathtext=False):
            # the first nonzero digit carries no overbar
            m = re.search("[1-9]", label)
            assert m is not None
            assert label[m.end() : m.end() + 1] != "\u0305", label


def test_plot_matches_background_norm():
    import matplotlib

    matplotlib.use("Agg")
    from types import SimpleNamespace

    from quantem.diffraction.orientation_visualization import plot_pattern_matches

    torch.manual_seed(0)
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True))
    xtl.calculate_structure_factors(k_max=1.5)
    peaks = _make_peaks(xtl, qnormalize(torch.randn(2, 4, dtype=torch.float64)))
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=3.0, angle_step_in_plane_deg=3.0)
    om.match_orientations(progress_bar=False)
    img = np.random.default_rng(0).random((1, 2, 32, 32)) ** 4
    dataset = SimpleNamespace(array=img, shape=img.shape)
    shown = []
    for norm in (None, {"power": 0.5, "upper_quantile": 0.9}):
        fig, axs = plot_pattern_matches(
            om, [(0, 0)], dataset=dataset, pixel_size=0.05, matches=(0,), norm=norm
        )
        # show_2d draws the pattern in the panel, extended to q units
        ax = axs[0, 0]
        im = ax.images[0]
        assert np.allclose(im.get_extent()[:2], (-0.5 * 0.05 - 16 * 0.05, 31.5 * 0.05 - 16 * 0.05))
        # the panel shows the recorded area and no marker outside it
        assert np.allclose(ax.get_xlim(), im.get_extent()[:2])
        x0, x1, y1, y0 = im.get_extent()
        for coll in ax.collections:
            xy = coll.get_offsets()
            assert (
                (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
            ).all()
        shown.append(np.asarray(im.get_array())[..., 0])
        matplotlib.pyplot.close(fig)
    # gray_r: a lower upper quantile saturates more of the pattern to black
    assert (shown[1] <= shown[1].min() + 1e-6).mean() > (shown[0] <= shown[0].min() + 1e-6).mean()


def _quat_deg(axis, angle_deg):
    from quantem.diffraction.rotations import quat_from_axis_angle

    a = torch.tensor(axis, dtype=torch.float64)
    return quat_from_axis_angle(a / a.norm(), torch.tensor(np.deg2rad(angle_deg)))


def _two_grain_peaks(xtl, q1, q2, n=3):
    """n positions, each the sum of two grains' patterns."""
    peaks = Vector.from_shape(
        (1, n), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(n):
        rows = []
        for q in (q1, q2):
            p = xtl.generate_pattern(q, energy_ev=200e3, sigma_excitation=0.02)
            rows.append(np.stack([p["qx"].numpy(), p["qy"].numpy(), p["intensity"].numpy()], 1))
        peaks[0, i] = np.concatenate(rows)
    return peaks


def test_second_match_indexes_second_grain():
    """With deflation, the second match fits the peaks the first leaves."""
    from quantem.diffraction.rotations import quat_from_zone_axis

    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True), verbose=False)
    xtl.calculate_structure_factors(k_max=1.5)
    q1 = quat_from_zone_axis(xtl.direction_vector((0, 0, 1)), 10.0)
    q2 = quat_from_zone_axis(xtl.direction_vector((1, 1, 1)), 35.0)
    peaks = _two_grain_peaks(xtl, q1, q2)
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=2.0, angle_step_in_plane_deg=2.0, verbose=False)
    om.match_orientations(num_matches=2, suppress_matched=1.0, progress_bar=False)
    assert om.quats.shape[2] == 2 and om.corr_residual.shape == om.corr.shape
    sym = xtl.sym_quats
    for i in range(peaks.shape[1]):
        e = [
            [float(misorientation_angle_deg(q, om.quats[0, i, m], sym)) for q in (q1, q2)]
            for m in range(2)
        ]
        # the two matches are the two grains, one each
        assert min(e[0][0] + e[1][1], e[0][1] + e[1][0]) < 6.0, e


def test_match_residual():
    """The residual of one crystal is re-matched by the other."""
    torch.manual_seed(1)
    ti_a = Crystal.from_ase(bulk("Ti", "hcp", a=2.9505, c=4.6855), name="a", verbose=False)
    ti_b = Crystal.from_ase(bulk("Ti", "bcc", a=3.26, cubic=True), name="b", verbose=False)
    for x in (ti_a, ti_b):
        x.calculate_structure_factors(k_max=1.5)
    q_a = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    q_b = _quat_deg((1.0, 0.0, 0.0), 20.0)
    N = 3
    peaks = Vector.from_shape(
        (1, N), fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    for i in range(N):
        rows = []
        for x, q, s in ((ti_a, q_a, 1.0), (ti_b, q_b, 0.5)):
            p = x.generate_pattern(q, energy_ev=200e3, sigma_excitation=0.02)
            rows.append(
                np.stack([p["qx"].numpy(), p["qy"].numpy(), s * p["intensity"].numpy()], 1)
            )
        peaks[0, i] = np.concatenate(rows)
    oms = {}
    for x in (ti_a, ti_b):
        om = OrientationMap.from_vectors(peaks, x, energy_ev=200e3)
        om.build_plan(angle_step_zone_axis_deg=3.0, verbose=False, progress_bar=False)
        om.match_orientations(progress_bar=False)
        oms[x.name] = om
    om_b = oms["b"]
    om_b.match_residual(oms["a"], progress_bar=False)
    assert om_b.quats.shape[2] == 2
    for name in ("corr", "corr_residual", "mirror"):
        assert getattr(om_b, name).shape == (1, N, 2), name
    # with alpha's peaks removed, beta is found by one of its two matches
    err = misorientation_angle_deg(q_b, om_b.quats[0], ti_b.sym_quats).amin(dim=-1)
    assert float(err.max()) < 2.0, err
    assert float(om_b.corr[0, :, 1].min()) > 0
    assert "match_residual" in om_b.metadata

    # nothing left once every peak is deleted: no error, an empty second match
    om_a = oms["a"]
    om_a.match_residual(om_a, delete_radius=10.0, progress_bar=False)
    assert om_a.quats.shape[2] == 2
    assert float(om_a.corr[..., 1].abs().max()) == 0.0
    assert om_a.corr_residual.shape == om_a.corr.shape


def test_plot_pattern_matches_defaults_with_one_match():
    import matplotlib

    matplotlib.use("Agg")
    from types import SimpleNamespace

    from quantem.diffraction.orientation_visualization import plot_pattern_matches

    torch.manual_seed(0)
    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True), verbose=False)
    xtl.calculate_structure_factors(k_max=1.5)
    peaks = _make_peaks(xtl, qnormalize(torch.randn(2, 4, dtype=torch.float64)))
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(angle_step_zone_axis_deg=3.0, verbose=False, progress_bar=False)
    om.match_orientations(progress_bar=False)
    # default matches=(0, 1) with a single match: one panel, no IndexError
    fig, axs = plot_pattern_matches(om, [(0, 0), (0, 1)])
    assert axs.shape == (2, 1)
    with pytest.raises(ValueError, match="none of matches"):
        plot_pattern_matches(om, [(0, 0)], matches=(3,))
    img = np.ones((1, 2, 16, 16))
    with pytest.raises(ValueError, match="pixel_size"):
        plot_pattern_matches(om, [(0, 0)], dataset=SimpleNamespace(array=img, shape=img.shape))
    matplotlib.pyplot.close("all")


def test_misorientation_map_and_cluster_plots():
    import matplotlib

    matplotlib.use("Agg")
    from quantem.diffraction.orientation_visualization import _pole_family

    xtl = Crystal.from_ase(bulk("Ti", "hcp", a=2.95, c=4.686), verbose=False)
    xtl.calculate_structure_factors(k_max=1.5)
    q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    q1 = _quat_deg((1.0, 0.0, 0.0), 25.0)
    R, C = 4, 6
    q = torch.where((torch.arange(C) < 3)[None, :, None], q0, q1).expand(R, C, 4).clone()
    peaks = _make_peaks(xtl, q0[None])
    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.quats = q[..., None, :]
    om.corr = torch.ones((R, C, 1), dtype=torch.float64)

    mis = om.misorientation_map()
    assert mis.shape == (R, C)
    assert float(mis[:, :3].max()) < 1e-6 and np.allclose(mis[:, 3:].numpy(), 25.0, atol=1e-6)
    assert np.allclose(om.misorientation_map(reference=q1)[:, 3:].numpy(), 0.0, atol=1e-6)

    clusters = om.cluster_orientations(min_cluster_size=2)
    assert clusters["sizes"].tolist() == [12, 12]
    fig, ax = om.plot_cluster_map(clusters)
    assert len(ax.get_legend().get_texts()) == 2
    fig, ax = om.plot_cluster_pole_figure(clusters, pole=(0, 0, 0, 1), pole_label="[0001]")
    assert len(ax.collections) == 2
    # a mask selects positions above one half
    half = np.zeros((R, C))
    half[:, :3] = 0.9
    half[:, 3:] = 0.4
    assert om.cluster_orientations(mask=half, min_cluster_size=2)["sizes"].tolist() == [12]
    matplotlib.pyplot.close("all")

    # poles are Miller indices: hexagonal [110] is 60 degrees from [100],
    # not the 45 degrees of the Cartesian (1, 1, 0)
    fam = _pole_family(xtl, (1, 1, 0))
    d = xtl.direction_vector((1, 1, 0))
    assert float((fam @ d).max()) > 1 - 1e-5
    assert torch.allclose(_pole_family(xtl, (0, 0, 1)), _pole_family(xtl, (0, 0, 0, 1)))
    a = xtl.direction_vector((1, 0, 0))
    assert np.isclose(float(a @ d), 0.5)
