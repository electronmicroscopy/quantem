"""End-to-end two-phase mapping on a synthetic alpha/beta titanium scan.

Left region: bcc beta in a fixed orientation. Right region: hcp alpha in a
Burgers-related orientation ((110)beta || (0001)alpha). A two-column band in
the middle contains both patterns superimposed, as at a lath boundary.
"""

import numpy as np
import torch
from ase.build import bulk

from quantem.core.datastructures.vector import Vector
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.orientation import OrientationMap
from quantem.diffraction.phase import PhaseMap
from quantem.diffraction.rotations import (
    misorientation_angle_deg,
    qmult,
    quat_from_axis_angle,
)


def _pattern(xtl, q, rng):
    p = xtl.generate_pattern(q, energy_ev=200e3, sigma_excitation=0.02)
    arr = np.column_stack([p["qx"].numpy(), p["qy"].numpy(), p["intensity"].numpy()])
    arr[:, :2] += rng.normal(0, 0.003, (arr.shape[0], 2))
    arr[:, 2] *= rng.lognormal(0, 0.3, arr.shape[0])
    return arr


def test_two_phase_map():
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    ti_a = Crystal.from_ase(
        bulk("Ti", "hcp", a=2.9505, c=4.6855), name="Ti alpha", verbose=False
    ).calculate_structure_factors(k_max=1.5)
    ti_b = Crystal.from_ase(
        bulk("Ti", "bcc", a=3.26, cubic=True), name="Ti beta", verbose=False
    ).calculate_structure_factors(k_max=1.5)

    # beta along [111] zone; alpha along [0001]: the Burgers-related pair
    # shares the hexagonal net, the hard case for phase mapping
    q_beta = quat_from_axis_angle(
        torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64) / np.sqrt(2),
        torch.tensor(np.arccos(1 / np.sqrt(3)), dtype=torch.float64),
    )
    q_alpha = qmult(
        quat_from_axis_angle(
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            torch.tensor(np.deg2rad(14.0), dtype=torch.float64),
        ),
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
    )

    R, C = 6, 11
    band = (5, 6)  # columns with both phases
    cells = []
    truth = np.zeros((R, C), dtype=int)  # 0 alpha, 1 beta
    for r in range(R):
        row = []
        for c in range(C):
            if c < band[0]:
                arr = _pattern(ti_b, q_beta, rng)
                truth[r, c] = 1
            elif c > band[1]:
                arr = _pattern(ti_a, q_alpha, rng)
                truth[r, c] = 0
            else:
                a = _pattern(ti_a, q_alpha, rng)
                b = _pattern(ti_b, q_beta, rng)
                a[:, 2] *= 0.5
                b[:, 2] *= 0.5
                arr = np.concatenate([a, b])
                truth[r, c] = 2
            row.append(arr)
        cells.append(row)
    peaks = Vector.from_data(cells, fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t")

    oms = []
    for xtl in (ti_a, ti_b):
        om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
        om.build_plan()
        om.match_orientations(num_matches=2, progress_bar=False)
        om.refine_orientations(progress_bar=False)
        oms.append(om)

    # orientation recovery in the pure regions
    err_a = misorientation_angle_deg(
        q_alpha, oms[0].quats[:, band[1] + 1 :, 0].reshape(-1, 4), ti_a.sym_quats
    ).numpy()
    # along [111], beta and its 60 degree twin about [111] give identical
    # kinematical patterns, so either is a correct match; which one wins is
    # decided by round-off and differs between platforms
    q_beta_twin = qmult(
        q_beta,
        quat_from_axis_angle(
            torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64) / np.sqrt(3),
            torch.tensor(np.pi / 3, dtype=torch.float64),
        ),
    )
    q_found = oms[1].quats[:, : band[0], 0].reshape(-1, 4)
    err_b = np.minimum(
        misorientation_angle_deg(q_beta, q_found, ti_b.sym_quats).numpy(),
        misorientation_angle_deg(q_beta_twin, q_found, ti_b.sym_quats).numpy(),
    )
    assert np.median(err_a) < 1.0
    assert np.median(err_b) < 1.0

    pm = PhaseMap.from_orientation_maps(oms)
    pm.fit(max_patterns=2, progress_bar=False)
    pi = pm.phase_index.numpy()

    pure_a = pi[:, band[1] + 1 :]
    pure_b = pi[:, : band[0]]
    assert (pure_a == 0).mean() > 0.85
    assert (pure_b == 1).mean() > 0.85

    # overlap band: every position must be assigned one of the two true
    # phases with a valid orientation (either is acceptable)
    band_pi = pi[:, band[0] : band[1] + 1]
    assert np.isin(band_pi, [0, 1]).all()


def test_crystal_map_sets_k_max_once():
    # two phases simulated to different ranges are not compared fairly, and
    # setting k_max per crystal invites exactly that mistake
    import numpy as np
    import pytest
    from ase.build import bulk

    from quantem.core.datastructures import Vector
    from quantem.diffraction import Crystal, CrystalMap

    peaks = Vector.from_data(
        [[np.array([[0.0, 0.0, 1.0], [0.3, 0.1, 0.5], [-0.2, 0.4, 0.3]])]],
        fields=["qx", "qy", "intensity"],
        name="p",
    )
    au = Crystal.from_ase(bulk("Au", "fcc", a=4.08, cubic=True), verbose=False)
    fe = Crystal.from_ase(bulk("Fe", "bcc", a=2.87, cubic=True), verbose=False)

    with pytest.raises(ValueError, match="pass k_max"):
        CrystalMap.from_vectors(peaks, [au, fe])

    au.calculate_structure_factors(k_max=1.4)
    fe.calculate_structure_factors(k_max=2.0)
    with pytest.raises(ValueError, match="different k_max"):
        CrystalMap.from_vectors(peaks, [au, fe])

    cm = CrystalMap.from_vectors(peaks, [au, fe], k_max=1.8)
    assert cm.k_max == 1.8
    assert au.k_max == fe.k_max == 1.8
    assert float(au.g_len.max()) <= 1.8 and float(fe.g_len.max()) <= 1.8


def test_dynamical_update_is_local_and_examples_are_spread():
    from quantem.diffraction.crystal_map import CrystalMap

    rng = np.random.default_rng(4)
    ti_a = Crystal.from_ase(
        bulk("Ti", "hcp", a=2.9505, c=4.6855), name="Ti alpha", verbose=False
    ).calculate_structure_factors(k_max=1.5)
    ti_b = Crystal.from_ase(
        bulk("Ti", "bcc", a=3.26, cubic=True), name="Ti beta", verbose=False
    ).calculate_structure_factors(k_max=1.5)
    q_a = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    q_b = quat_from_axis_angle(
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor(np.deg2rad(20.0), dtype=torch.float64),
    )
    R, C = 3, 8
    cells = [
        [_pattern(ti_a, q_a, rng) if c < 4 else _pattern(ti_b, q_b, rng) for c in range(C)]
        for _ in range(R)
    ]
    peaks = Vector.from_data(cells, fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t")
    oms = []
    for xtl in (ti_a, ti_b):
        om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
        om.build_plan()
        om.match_orientations(progress_bar=False)
        oms.append(om)
    cm = CrystalMap.from_orientation_maps(oms)
    cm.fit(progress_bar=False)
    before = cm.phase_index.copy()
    assert (before[:, :4] == 0).all() and (before[:, 4:] == 1).all()

    # a dynamical result that reached one position and flipped it
    F = len(cm.phases.candidates)
    cost = torch.full((R, C, F), torch.nan, dtype=torch.float64)
    cost[0, 0] = torch.tensor([0.9, 0.1])
    cm.phases.apply_dynamical({"cost": cost, "phase_index": cost.nan_to_num(9).argmin(-1)})
    after = cm.phase_index
    assert after[0, 0] == 1
    changed = after != before
    changed[0, 0] = False
    assert not changed.any()
    assert (cm.phases.metadata["kinematical"]["phase_index"].numpy() == before).all()

    picks = cm.example_positions(phase="Ti beta", num=3, min_distance=3)
    assert all(cm.phase_index[p] == 1 for p in picks)
    assert all(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 >= 9
        for i, a in enumerate(picks)
        for b in picks[:i]
    )


def test_loaded_crystal_map_shares_orientation_maps(tmp_path):
    from quantem.core.io.serialize import load
    from quantem.diffraction.crystal_map import CrystalMap

    rng = np.random.default_rng(5)
    ti_a = Crystal.from_ase(
        bulk("Ti", "hcp", a=2.9505, c=4.6855), name="Ti alpha", verbose=False
    ).calculate_structure_factors(k_max=1.5)
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    cells = [[_pattern(ti_a, q, rng) for _ in range(3)] for _ in range(2)]
    peaks = Vector.from_data(cells, fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t")
    om = OrientationMap.from_vectors(peaks, ti_a, energy_ev=200e3)
    om.build_plan()
    om.match_orientations(progress_bar=False)
    cm = CrystalMap.from_orientation_maps([om])
    cm.fit(progress_bar=False)
    cm.save(tmp_path / "cm.zip", mode="o")
    cm2 = load(tmp_path / "cm.zip")
    # one set of maps: what a refinement writes through the phase map is
    # what the crystal map shows
    assert cm2.phases.orientation_maps[0] is cm2.orientation_maps[0]
    assert cm2.phases.orientation_maps[0].crystal is cm2[0].crystal


def test_plot_calibration_shows_every_crystal():
    import matplotlib

    matplotlib.use("Agg")
    from quantem.diffraction.crystal_map import CrystalMap

    rng = np.random.default_rng(6)
    ti_a = Crystal.from_ase(
        bulk("Ti", "hcp", a=2.9505, c=4.6855), name="Ti alpha", verbose=False
    ).calculate_structure_factors(k_max=1.5)
    ti_b = Crystal.from_ase(
        bulk("Ti", "bcc", a=3.26, cubic=True), name="Ti beta", verbose=False
    ).calculate_structure_factors(k_max=1.5)
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    cells = [[_pattern(ti_a, q, rng) for _ in range(3)] for _ in range(2)]
    peaks = Vector.from_data(cells, fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t")
    cm = CrystalMap.from_vectors(peaks, [ti_a, ti_b], energy_ev=200e3)
    fig, axs = cm.plot_calibration()
    # one column per crystal, the histogram above the azimuth panel
    assert axs.shape == (2, 2)
    assert [ax.get_title() for ax in axs[0]] == ["Ti alpha", "Ti beta"]
    assert all(len(ax.collections) > 0 for ax in axs[1])
    matplotlib.pyplot.close(fig)
    fig, ax = cm.plot_bragg_rings()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Ti alpha rings", "Ti beta rings"]
    matplotlib.pyplot.close(fig)


def test_refine_skips_crystals_out_of_the_running():
    from quantem.diffraction.crystal_map import CrystalMap

    rng = np.random.default_rng(7)
    ti_a = Crystal.from_ase(
        bulk("Ti", "hcp", a=2.9505, c=4.6855), name="Ti alpha", verbose=False
    ).calculate_structure_factors(k_max=1.5)
    ti_b = Crystal.from_ase(
        bulk("Ti", "bcc", a=3.26, cubic=True), name="Ti beta", verbose=False
    ).calculate_structure_factors(k_max=1.5)
    q_a = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    q_b = quat_from_axis_angle(
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor(np.deg2rad(20.0), dtype=torch.float64),
    )
    cells = [
        [_pattern(ti_a, q_a, rng) if c < 3 else _pattern(ti_b, q_b, rng) for c in range(6)]
        for _ in range(2)
    ]
    peaks = Vector.from_data(cells, fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t")
    fractions = []
    for margin in (None, 0.1):
        cm = CrystalMap.from_vectors(peaks, [ti_a, ti_b], energy_ev=200e3)
        cm.build_plan()
        cm.match_orientations(progress_bar=False)
        corr = np.stack([om.corr[..., 0].numpy() for om in cm])
        q_lib = [om.quats.clone() for om in cm]
        cm.refine_orientations(competitive_margin=margin, progress_bar=False)
        cm.fit(progress_bar=False)
        fractions.append(cm.phase_index.copy())
        if margin is not None:
            out = corr < corr.max(axis=0) - margin
            for i, om in enumerate(cm):
                # where a crystal is out of the running its library match stays
                assert torch.equal(om.quats[out[i]], q_lib[i][out[i]])
            assert out.any()
    assert (fractions[0] == fractions[1]).all()


def test_zone_reflections_keep_one_zone():
    from quantem.diffraction.calibration import plot_calibration, zone_reflections

    ti_a = Crystal.from_ase(
        bulk("Ti", "hcp", a=2.9505, c=4.6855), name="Ti alpha", verbose=False
    ).calculate_structure_factors(k_max=1.5)
    basal = zone_reflections(ti_a, (0, 0, 0, 1))
    # [0001] keeps exactly the hk0 reflections, and 3- and 4-index agree
    assert (basal.hkl[:, 2] == 0).all()
    assert len(basal.hkl) == int((ti_a.hkl[:, 2] == 0).sum())
    assert torch.equal(zone_reflections(ti_a, (0, 0, 1)).hkl, basal.hkl)
    assert basal.name == "Ti alpha [0001]" and ti_a.name == "Ti alpha"
    assert len(ti_a.hkl) > len(basal.hkl)
    # the plots accept it and title the column by the zone
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.default_rng(8)
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    peaks = Vector.from_data(
        [[_pattern(ti_a, q, rng)]], fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    fig, axs = plot_calibration(peaks, ti_a, zone_axis=(0, 0, 0, 1))
    assert axs[0, 0].get_title() == "Ti alpha [0001]"
    matplotlib.pyplot.close(fig)
