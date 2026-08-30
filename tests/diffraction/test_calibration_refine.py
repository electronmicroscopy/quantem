"""Recovery of a global calibration residual from matched orientations."""

import numpy as np
import torch
from ase.build import bulk

from quantem.core.datastructures.vector import Vector
from quantem.diffraction import calibration
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.orientation import OrientationMap
from quantem.diffraction.rotations import qnormalize


def test_refine_calibration_recovers_distortion():
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    xtl = Crystal.from_ase(
        bulk("Ti", "hcp", a=2.9505, c=4.6855), name="Ti alpha", verbose=False
    )
    xtl.calculate_structure_factors(k_max=1.5)

    # global calibration error: 1.5% scale, 0.8% ellipticity, 0.3 deg rotation
    scale_true = 1.015
    e11_true, e12_true = 0.008, -0.004
    th = np.deg2rad(0.3)
    M_true = (
        scale_true
        * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        @ np.array([[1 + e11_true, e12_true], [e12_true, 1 - e11_true]])
    )

    R, C = 6, 10
    q_true = qnormalize(torch.randn(R * C, 4, dtype=torch.float64))
    cells = []
    for i in range(R * C):
        p = xtl.generate_pattern(q_true[i], energy_ev=200e3, sigma_excitation=0.02)
        q2 = np.stack([p["qx"].numpy(), p["qy"].numpy()], axis=1) @ M_true.T
        q2 += rng.normal(0, 0.002, q2.shape)
        cells.append(np.column_stack([q2, p["intensity"].numpy()]))
    nested = [cells[r * C : (r + 1) * C] for r in range(R)]
    peaks = Vector.from_data(
        nested, fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )

    om = OrientationMap.from_vectors(peaks, xtl, energy_ev=200e3)
    om.build_plan(power_intensity=0.0)
    om.match_orientations(progress_bar=False)
    om.refine_orientations(progress_bar=False, neighbor_rescue=False)
    sm = om.calculate_strain(progress_bar=False)

    res = calibration.refine_calibration([sm])
    assert res["num_positions"] > 30
    assert abs(res["scale"] - scale_true) < 3e-3
    # a global rotation is absorbed into the refined orientations and must
    # read as ~0 here (it is measured independently via the scan rotation)
    assert abs(res["rotation_deg"]) < 0.1
    assert abs(res["ellipse"][0] - e11_true) < 2e-3
    assert abs(res["ellipse"][1] - e12_true) < 2e-3

    # applying the correction must bring the residual to the identity
    peaks_fixed = calibration.transform_peaks(peaks, res["correction"])
    om2 = OrientationMap.from_vectors(peaks_fixed, xtl, energy_ev=200e3)
    om2.build_plan(power_intensity=0.0)
    om2.match_orientations(progress_bar=False)
    om2.refine_orientations(progress_bar=False, neighbor_rescue=False)
    sm2 = om2.calculate_strain(progress_bar=False)
    res2 = calibration.refine_calibration([sm2])
    assert abs(res2["scale"] - 1.0) < 2e-3
    assert abs(res2["ellipse"][0]) < 1.5e-3
    assert abs(res2["ellipse"][1]) < 1.5e-3
