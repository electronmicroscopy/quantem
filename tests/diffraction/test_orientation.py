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
        peaks[0, i] = np.stack(
            [p["qx"].numpy(), p["qy"].numpy(), p["intensity"].numpy()], axis=1
        )
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
    om.build_plan(
        angle_step_zone_axis_deg=2.0, angle_step_in_plane_deg=2.0, power_intensity=0.0
    )
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
