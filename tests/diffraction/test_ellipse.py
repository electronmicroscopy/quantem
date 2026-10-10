"""Elliptic distortion recovery from radial histogram sharpness."""

import numpy as np
import torch
from ase.build import bulk

from quantem.core.datastructures.vector import Vector
from quantem.diffraction import calibration
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.rotations import qnormalize


def test_ellipse_recovery():
    torch.manual_seed(0)
    xtl = Crystal.from_ase(bulk("Ti", "hcp", a=2.9505, c=4.6855), verbose=False)
    xtl.calculate_structure_factors(k_max=1.5)

    e_true = np.array([0.012, -0.008])
    A = np.array([[1 + e_true[0], e_true[1]], [e_true[1], 1 - e_true[0]]])
    A_inv = np.linalg.inv(A)

    rng = np.random.default_rng(0)
    cells = []
    for i in range(60):
        q = qnormalize(torch.randn(4, dtype=torch.float64))
        pat = xtl.generate_pattern(q, energy_ev=200e3, sigma_excitation=0.02)
        qxy = np.stack([pat["qx"].numpy(), pat["qy"].numpy()], axis=1)
        # distort (the inverse of the correction) plus detection noise
        qxy = qxy @ A_inv.T + rng.normal(0, 0.003, qxy.shape)
        cells.append(np.column_stack([qxy, pat["intensity"].numpy()]))
    peaks = Vector.from_data(
        [cells],
        fields=["qx", "qy", "intensity"],
        units=["A^-1", "A^-1", "counts"],
        name="synthetic",
    )

    e_fit = calibration.calibrate_ellipse(peaks)
    assert np.allclose(e_fit, e_true, atol=2e-3), (e_fit, e_true)
