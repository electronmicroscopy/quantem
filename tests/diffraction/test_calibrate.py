"""calibrate(), DiffractionCalibration and the scan rotation measurement."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase.build import bulk

from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import load
from quantem.diffraction import calibration
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.rotations import qnormalize

PIXEL_SIZE = 0.0123
E_TRUE = np.array([0.012, -0.008])


@pytest.fixture(scope="module")
def ti():
    xtl = Crystal.from_ase(bulk("Ti", "hcp", a=2.9505, c=4.6855), verbose=False)
    xtl.calculate_structure_factors(k_max=1.5)
    return xtl


@pytest.fixture(scope="module")
def peaks_px(ti):
    """Ti patterns in detector pixels with a known ellipse and pixel size."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    A_inv = np.linalg.inv(calibration._ellipse_matrix(E_TRUE))
    cells = []
    for _ in range(60):
        q = qnormalize(torch.randn(4, dtype=torch.float64))
        pat = ti.generate_pattern(q, energy_ev=200e3, sigma_excitation=0.02)
        qxy = np.stack([pat["qx"].numpy(), pat["qy"].numpy()], axis=1)
        qxy = qxy @ A_inv.T + rng.normal(0, 0.002, qxy.shape)
        cells.append(np.column_stack([qxy / PIXEL_SIZE, pat["intensity"].numpy()]))
    return Vector.from_data(
        [cells[:30], cells[30:]],
        fields=["q_row", "q_col", "intensity"],
        units=["px", "px", "counts"],
        name="synthetic",
    )


@pytest.mark.parametrize("n_iter", [1, 2, 3])
def test_calibrate_stable_in_n_iter(ti, peaks_px, n_iter):
    # each round must refine the ellipse, not replace it with the residual
    cal = calibration.calibrate(peaks_px, ti, 0.011, n_iter=n_iter)
    assert abs(cal.pixel_size / PIXEL_SIZE - 1) < 2e-3
    assert np.allclose(cal.ellipse, E_TRUE, atol=1.5e-3), cal.ellipse
    assert cal.metadata["reliable"]
    assert cal.metadata["n_rings"] >= 3


def test_calibrate_returnfig(ti, peaks_px):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cal, fig, axs = calibration.calibrate(peaks_px, ti, 0.011, plot=True, returnfig=True)
    assert isinstance(cal, calibration.DiffractionCalibration)
    assert axs.shape == (2, 2)
    plt.close(fig)


def test_compose_ellipse():
    a, b = np.array([0.01, -0.004]), np.array([0.003, 0.002])
    c = calibration._compose_ellipse(a, b)
    # to first order the components add
    assert np.allclose(c, a + b, atol=1e-4)
    assert np.allclose(calibration._compose_ellipse(a, None), a)
    assert np.allclose(calibration._compose_ellipse(np.zeros(2), b), b)


def test_diffraction_calibration_apply_rebin_save(tmp_path):
    cells = [[np.array([[10.0, 0.0, 1.0], [0.0, -20.0, 2.0]]), np.zeros((0, 3))]]
    peaks_px = Vector.from_data(
        cells, fields=["q_row", "q_col", "intensity"], units=["px", "px", "counts"], name="p"
    )
    ellipse = np.array([0.01, -0.02])
    cal = calibration.DiffractionCalibration(
        0.01, ellipse, rotation_ccw_deg=90.0, metadata={"binning": 1, "n_rings": 4}
    )

    out = cal.apply(peaks_px)
    assert out.fields == ["qx", "qy", "intensity"]
    assert out.metadata["pixel_size"] == pytest.approx(0.01)
    flat = out.numpy().astype(np.float64)
    q = np.array([[0.1, 0.0], [0.0, -0.2]]) @ calibration._ellipse_matrix(ellipse).T
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert np.allclose(flat[:, :2], q @ rot.T, atol=1e-6)
    assert np.allclose(flat[:, 2], [1.0, 2.0])

    binned = cal.rebin(2)
    assert binned.pixel_size == pytest.approx(0.02)
    assert binned.metadata["binning"] == 2
    assert np.allclose(binned.ellipse, ellipse)
    assert binned.rotation_ccw_deg == cal.rotation_ccw_deg
    assert cal.metadata["binning"] == 1

    path = tmp_path / "cal.zip"
    cal.save(path, mode="o")
    cal2 = load(path)
    assert isinstance(cal2, calibration.DiffractionCalibration)
    assert cal2.pixel_size == pytest.approx(cal.pixel_size)
    assert np.allclose(cal2.ellipse, cal.ellipse)
    assert cal2.rotation_ccw_deg == pytest.approx(90.0)
    assert cal2.metadata["n_rings"] == 4
    assert np.allclose(cal2.apply(peaks_px).numpy(), out.numpy())


def test_refine_calibration_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        calibration.refine_calibration([])
    sm = SimpleNamespace(
        u_array=np.full((2, 2, 2), np.nan),
        v_array=np.full((2, 2, 2), np.nan),
        g1_array=np.full((2, 2, 2), np.nan),
        g2_array=np.full((2, 2, 2), np.nan),
    )
    with pytest.raises(ValueError, match="no positions"):
        calibration.refine_calibration([sm])


@pytest.mark.parametrize("theta_deg", [30.0, 210.0, 125.0])
def test_measure_scan_rotation_sign(theta_deg):
    """A gradient field seen on a detector rotated by -theta returns theta mod 180.

    The returned angle uses the convention of peaks_to_calibrated: rotating
    the detector field by +theta brings it back into the scan frame.
    """
    R = C = 16
    H = W = 32
    ry, rx = np.mgrid[0:R, 0:C] / R
    # CoM shift in the scan frame: gradient of an asymmetric potential
    phi = np.sin(2 * np.pi * ry) * np.cos(np.pi * rx) + 0.5 * rx**2
    g_r, g_c = np.gradient(phi)
    g = np.stack([g_r, g_c], axis=-1)
    g *= 1.5 / np.abs(g).max()
    th = np.deg2rad(theta_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    d = g @ rot  # detector frame: rot.T applied to each scan-frame vector
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    arr = np.zeros((R, C, H, W))
    for i in range(R):
        for j in range(C):
            r0, c0 = H / 2 + d[i, j, 0], W / 2 + d[i, j, 1]
            arr[i, j] = np.exp(-((rows - r0) ** 2 + (cols - c0) ** 2) / (2 * 1.5**2))
    angle = calibration.measure_scan_rotation(SimpleNamespace(array=arr))
    expected = theta_deg % 180
    diff = (angle - expected + 90) % 180 - 90
    assert abs(diff) < 1.0, (angle, expected)
