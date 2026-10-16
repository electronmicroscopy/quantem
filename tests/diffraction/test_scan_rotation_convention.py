"""One detector-to-scan rotation convention across calibration, strain and ptychography.

The convention (``q_to_r_matrix``): a detector vector (row, col) is swapped if
``q_transpose``, then rotated counter-clockwise by ``q_to_r_rotation_ccw_deg``.
It is stored on the dataset metadata, written by every measurement and read
by every workflow.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from quantem.core.datastructures.dataset4dstem import (
    Dataset4dstem,
    q_to_r_matrix,
    scan_rotation_from_metadata,
)
from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import load
from quantem.diffraction import calibration
from quantem.diffraction.bragg_vectors import BraggVectors
from quantem.diffraction.strain import _raw_vec_to_display


def _com_scan(theta_deg, transpose, R=24, C=24, H=40, W=40):
    """A probe whose CoM follows a gradient field in the scan frame, recorded on a
    detector related to the scan by (theta, transpose)."""
    ry, rx = np.mgrid[0:R, 0:C] / R
    phi = np.sin(2 * np.pi * ry) * np.cos(np.pi * rx) + 0.6 * rx**2 + 0.3 * ry * rx
    g = np.stack(np.gradient(phi), -1)
    g *= 2.0 / np.abs(g).max()
    d = g @ np.linalg.inv(q_to_r_matrix(theta_deg, transpose)).T  # scan -> detector
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    arr = np.zeros((R, C, H, W))
    for i in range(R):
        for j in range(C):
            r0, c0 = H / 2 + d[i, j, 0], W / 2 + d[i, j, 1]
            arr[i, j] = np.exp(-((rows - r0) ** 2 + (cols - c0) ** 2) / (2 * 2.0**2)) + 1e-3
    return arr


def _ptycho(dset):
    from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster

    p = PtychographyDatasetRaster.from_dataset4dstem(dset)
    p.verbose = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p.preprocess(
            com_fit_function="constant", plot_rotation=False, plot_com=False, probe_energy=200e3
        )
    return p


def _dataset(arr):
    return Dataset4dstem.from_array(
        array=arr, sampling=(1.0, 1.0, 0.05, 0.05), units=("A", "A", "A^-1", "A^-1")
    )


def test_q_to_r_matrix_matches_strain_and_ptycho_formulas():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(20, 2))
    for theta, tr in [(0.0, False), (34.0, False), (-71.0, True), (12.5, True)]:
        M = q_to_r_matrix(theta, tr)
        assert np.allclose(v @ M.T, _raw_vec_to_display(v, rotation_ccw_deg=theta, transpose=tr))
        # ptychography's rotate_com_vectors
        a = np.deg2rad(theta)
        r, c = (v[:, 1], v[:, 0]) if tr else (v[:, 0], v[:, 1])
        ptycho = np.stack([np.cos(a) * r - np.sin(a) * c, np.sin(a) * r + np.cos(a) * c], 1)
        assert np.allclose(v @ M.T, ptycho)


@pytest.mark.parametrize("theta, transpose", [(30.0, False), (-50.0, False), (30.0, True)])
def test_bragg_and_ptycho_measure_the_same_rotation(theta, transpose):
    arr = _com_scan(theta, transpose)
    angle, tr = calibration.measure_scan_rotation(
        SimpleNamespace(array=arr), return_transpose=True
    )
    assert tr == transpose
    assert abs(angle - theta) < 0.5
    p = _ptycho(_dataset(arr))
    assert p.com_transpose == transpose
    assert abs(np.rad2deg(p.com_rotation_rad) - theta) < 1.0


def test_measured_rotation_is_shared_through_the_dataset():
    theta, transpose = -37.0, True
    dset = _dataset(_com_scan(theta, transpose))
    bv = BraggVectors.from_dataset(dset)
    angle = bv.measure_scan_rotation()
    assert scan_rotation_from_metadata(dset.metadata) == (angle, True)
    # ptychography on the same dataset takes it, instead of fitting its own
    dset.metadata["q_to_r_rotation_ccw_deg"] = angle + 0.4  # distinguishable from a refit
    p = _ptycho(dset)
    assert np.rad2deg(p.com_rotation_rad) == pytest.approx(angle + 0.4)
    assert p.com_transpose is True


def test_ptycho_fit_is_recorded_on_the_dataset():
    dset = _dataset(_com_scan(22.0, False))
    assert scan_rotation_from_metadata(dset.metadata) == (None, None)
    p = _ptycho(dset)
    rot, tr = scan_rotation_from_metadata(dset.metadata)
    assert rot == pytest.approx(np.rad2deg(p.com_rotation_rad))
    assert tr is False


def test_old_metadata_name_is_read():
    md = {"r_to_q_rotation_cw_deg": 12.0}
    assert scan_rotation_from_metadata(md) == (12.0, None)


def test_calibration_transpose_apply_and_save(tmp_path):
    cells = [[np.array([[10.0, 0.0, 1.0], [0.0, -20.0, 2.0]])]]
    peaks_px = Vector.from_data(
        cells, fields=["q_row", "q_col", "intensity"], units=["px", "px", "counts"], name="p"
    )
    cal = calibration.DiffractionCalibration(0.01, None, 30.0, {"binning": 1}, transpose=True)
    out = cal.apply(peaks_px)
    expected = (np.array([[10.0, 0.0], [0.0, -20.0]]) * 0.01) @ q_to_r_matrix(30.0, True).T
    assert np.allclose(out.numpy()[:, :2].astype(float), expected, atol=1e-6)
    assert out.metadata["q_transpose"] is True
    assert "transposed" in repr(cal)
    assert cal.rebin(2).transpose
    cal.save(tmp_path / "cal.zip", mode="o")
    assert load(tmp_path / "cal.zip").transpose is True
    # a calibration without the attribute (saved before it existed) has none
    old = calibration.DiffractionCalibration(0.01)
    del old.__dict__["transpose"]
    assert old.transpose is False


def test_unreliable_rotation_warns():
    # a center of mass that is pure noise has no curl-free frame to find
    rng = np.random.default_rng(3)
    arr = rng.random((16, 16, 24, 24))
    with pytest.warns(UserWarning, match="unreliable"):
        calibration.measure_scan_rotation(SimpleNamespace(array=arr))
