"""
Tests for the DriftCorrection class in quantem.imaging.drift

Synthetic data: chevron pattern with linear drift + jitter at 0 and 90 deg scan angles.
See PR #133 for images: https://github.com/electronmicroscopy/quantem/pull/133
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.imaging.drift import DriftCorrection


def generate_standardized_synthetic_data(scale=1, seed=42):
    """Generate standardized synthetic drift data with linear drift and jitter."""
    np.random.seed(seed)

    shape = (200 * scale, 200 * scale)
    xa, ya = np.meshgrid(
        np.arange(-shape[0] / 2, shape[0] / 2),
        np.arange(-shape[0] / 2, shape[0] / 2),
        indexing="ij",
    )
    im = (np.mod(np.abs(xa) + np.abs(ya), 16 * scale) < 8 * scale).astype("float")
    im[np.logical_and(xa > 0, ya > 0)] += 0.5
    im[np.maximum(np.abs(xa), np.abs(ya)) < 20 * scale] = 2
    im = gaussian_filter(im, sigma=0.667 * scale)

    scan_size = 128 * scale
    u = np.arange(scan_size)
    x_drift = u * 0.001 * scale
    y_drift = u * 0.1 * scale
    jitter_mag = 0.5 * scale
    jitter0 = np.random.randn(2, scan_size) * jitter_mag
    jitter1 = np.random.randn(2, scan_size) * jitter_mag

    im0 = np.zeros((scan_size, scan_size))
    for a0 in range(scan_size):
        x0 = 40 * scale + a0 + x_drift[a0] + jitter0[0, a0]
        y0 = 30 * scale + 0 + y_drift[a0] + jitter0[1, a0]
        x = x0 + u * 0
        y = y0 + u * 1
        x = np.clip(x, 0, shape[0] - 2)
        y = np.clip(y, 0, shape[1] - 2)
        xf = np.floor(x).astype("int")
        yf = np.floor(y).astype("int")
        dx = x - xf
        dy = y - yf
        im0[a0, :] = (
            im[xf, yf] * (1 - dx) * (1 - dy)
            + im[xf + 1, yf] * (dx) * (1 - dy)
            + im[xf, yf + 1] * (1 - dx) * (dy)
            + im[xf + 1, yf + 1] * (dx) * (dy)
        )

    im1 = np.zeros((scan_size, scan_size))
    for a0 in range(scan_size):
        x0 = 170 * scale + 0 + x_drift[a0] + jitter1[0, a0]
        y0 = 30 * scale + a0 + y_drift[a0] + jitter1[1, a0]
        x = x0 - u * 1
        y = y0 + u * 0
        x = np.clip(x, 0, shape[0] - 2)
        y = np.clip(y, 0, shape[1] - 2)
        xf = np.floor(x).astype("int")
        yf = np.floor(y).astype("int")
        dx = x - xf
        dy = y - yf
        im1[a0, :] = (
            im[xf, yf] * (1 - dx) * (1 - dy)
            + im[xf + 1, yf] * (dx) * (1 - dy)
            + im[xf, yf + 1] * (1 - dx) * (dy)
            + im[xf + 1, yf + 1] * (dx) * (dy)
        )

    return im0, im1, im


def test_full_pipeline_deterministic():
    """Full pipeline produces correct, deterministic, low-error results."""
    im0, im1, _ = generate_standardized_synthetic_data(scale=1, seed=42)

    drift = DriftCorrection.from_data(
        images=[im0, im1],
        scan_direction_degrees=[0.0, 90.0],
    ).preprocess(
        pad_fraction=0.25,
        pad_value="median",
        kde_sigma=0.5,
        number_knots=1,
        show_merged=False,
        show_images=False,
    )
    drift.align_affine(step=0.02, num_tests=5, refine=False)
    drift.align_nonrigid(
        num_iterations=2,
        regularization_sigma_px=0.5,
        show_merged=False,
        show_images=False,
    )
    img_corr = drift.generate_corrected_image(upsample_factor=1, show_image=False)

    assert isinstance(img_corr, Dataset2d)
    assert not np.isnan(img_corr.array).any()
    assert drift.error_track[-1, 1] < 0.1

    # Determinism: second run with same seed must match exactly
    im0_2, im1_2, _ = generate_standardized_synthetic_data(scale=1, seed=42)
    drift2 = DriftCorrection.from_data(
        images=[im0_2, im1_2],
        scan_direction_degrees=[0.0, 90.0],
    ).preprocess(
        pad_fraction=0.25,
        pad_value="median",
        kde_sigma=0.5,
        number_knots=1,
        show_merged=False,
        show_images=False,
    )
    drift2.align_affine(step=0.02, num_tests=5, refine=False)
    drift2.align_nonrigid(
        num_iterations=2,
        regularization_sigma_px=0.5,
        show_merged=False,
        show_images=False,
    )
    img_corr2 = drift2.generate_corrected_image(upsample_factor=1, show_image=False)

    np.testing.assert_array_almost_equal(
        img_corr.array, img_corr2.array, decimal=10,
        err_msg="Drift correction output is not deterministic!",
    )


# Baseline values from float32 torch path, captured once and frozen.
# (scale, error, knots0_sum, knots1_sum)
AFFINE_BASELINES = [
    (1, 0.09371880441904068, 12091.138622065724, 28612.861377934278),
    (2, 0.1400984227657318, 48473.840748950926, 114854.15925104907),
    (4, 0.163963183760643, 194424.12894570845, 459911.8710542915),
]


@pytest.mark.parametrize("scale,expected_error,expected_k0,expected_k1", AFFINE_BASELINES)
def test_align_affine_deterministic(scale, expected_error, expected_k0, expected_k1):
    """Affine on synthetic data must match frozen float32 baseline."""
    im0, im1, _ = generate_standardized_synthetic_data(scale=scale, seed=42)
    drift = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(
        step=0.02, num_tests=5, refine=True,
        show_merged=False, show_images=False,
    )
    np.testing.assert_almost_equal(
        drift.error_track[-1, 1], expected_error, decimal=4)
    np.testing.assert_almost_equal(
        drift.knots[0].sum(), expected_k0, decimal=2)
    np.testing.assert_almost_equal(
        drift.knots[1].sum(), expected_k1, decimal=2)
