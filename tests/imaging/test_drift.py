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


def make_synthetic_drift_data(scale=1, seed=42):
    """Generate a chevron base image plus two scan-distorted views.

    Image 0 scans along columns; image 1 scans along rows. Both apply the
    same linear row/col drift plus per-scanline jitter so the nonrigid
    solver has a non-trivial knot field to recover.
    """
    np.random.seed(seed)
    shape = (200 * scale, 200 * scale)
    row_grid, col_grid = np.meshgrid(
        np.arange(-shape[0] / 2, shape[0] / 2),
        np.arange(-shape[0] / 2, shape[0] / 2),
        indexing="ij",
    )
    base_image = (np.mod(np.abs(row_grid) + np.abs(col_grid), 16 * scale) < 8 * scale).astype("float")
    base_image[np.logical_and(row_grid > 0, col_grid > 0)] += 0.5
    base_image[np.maximum(np.abs(row_grid), np.abs(col_grid)) < 20 * scale] = 2
    base_image = gaussian_filter(base_image, sigma=0.667 * scale)

    scan_size = 128 * scale
    scan_positions = np.arange(scan_size)
    row_drift = scan_positions * 0.001 * scale
    col_drift = scan_positions * 0.1 * scale
    jitter_mag = 0.5 * scale
    jitter0 = np.random.randn(2, scan_size) * jitter_mag
    jitter1 = np.random.randn(2, scan_size) * jitter_mag

    im0 = np.zeros((scan_size, scan_size))
    for row_idx in range(scan_size):
        start_row = 40 * scale + row_idx + row_drift[row_idx] + jitter0[0, row_idx]
        start_col = 30 * scale + 0 + col_drift[row_idx] + jitter0[1, row_idx]
        row_coords = start_row + scan_positions * 0
        col_coords = start_col + scan_positions * 1
        row_coords = np.clip(row_coords, 0, shape[0] - 2)
        col_coords = np.clip(col_coords, 0, shape[1] - 2)
        row_floor = np.floor(row_coords).astype("int")
        col_floor = np.floor(col_coords).astype("int")
        row_frac = row_coords - row_floor
        col_frac = col_coords - col_floor
        im0[row_idx, :] = (
            base_image[row_floor, col_floor] * (1 - row_frac) * (1 - col_frac)
            + base_image[row_floor + 1, col_floor] * row_frac * (1 - col_frac)
            + base_image[row_floor, col_floor + 1] * (1 - row_frac) * col_frac
            + base_image[row_floor + 1, col_floor + 1] * row_frac * col_frac
        )

    im1 = np.zeros((scan_size, scan_size))
    for row_idx in range(scan_size):
        start_row = 170 * scale + 0 + row_drift[row_idx] + jitter1[0, row_idx]
        start_col = 30 * scale + row_idx + col_drift[row_idx] + jitter1[1, row_idx]
        row_coords = start_row - scan_positions * 1
        col_coords = start_col + scan_positions * 0
        row_coords = np.clip(row_coords, 0, shape[0] - 2)
        col_coords = np.clip(col_coords, 0, shape[1] - 2)
        row_floor = np.floor(row_coords).astype("int")
        col_floor = np.floor(col_coords).astype("int")
        row_frac = row_coords - row_floor
        col_frac = col_coords - col_floor
        im1[row_idx, :] = (
            base_image[row_floor, col_floor] * (1 - row_frac) * (1 - col_frac)
            + base_image[row_floor + 1, col_floor] * row_frac * (1 - col_frac)
            + base_image[row_floor, col_floor + 1] * (1 - row_frac) * col_frac
            + base_image[row_floor + 1, col_floor + 1] * row_frac * col_frac
        )

    return im0, im1, base_image


def test_full_pipeline_deterministic():
    """Full pipeline produces correct, deterministic, low-error results."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)

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
    im0_2, im1_2, _ = make_synthetic_drift_data(scale=1, seed=42)
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
    (1, 0.09237676858901978, 12157.7373046875, 28546.2626953125),
    (2, 0.13844291865825653, 49830.908203125, 113497.091796875),
    (4, 0.163960263133049, 194685.2421875, 459650.7578125),
]


@pytest.mark.parametrize("scale,expected_error,expected_k0,expected_k1", AFFINE_BASELINES)
def test_align_affine_matches_frozen_baseline(scale, expected_error, expected_k0, expected_k1):
    """Affine on synthetic data must match frozen float32 baseline."""
    im0, im1, _ = make_synthetic_drift_data(scale=scale, seed=42)
    drift = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(
        step=0.02, num_tests=5, refine=True,
        show_merged=False, show_images=False,
    )
    np.testing.assert_almost_equal(
        drift.error_track[-1, 1], expected_error, decimal=8)
    np.testing.assert_almost_equal(
        drift.knots[0].sum(), expected_k0, decimal=6)
    np.testing.assert_almost_equal(
        drift.knots[1].sum(), expected_k1, decimal=6)


# Frozen baselines for the pytorch backend with optimizer_name="adam".
NONRIGID_ADAM_BASELINES = [
    (1, 0.0562780499458313, 12023.870644569397, 28671.6764421463),
    (2, 0.12947668135166168, 49829.36344528198, 113481.72154045105),
]


@pytest.mark.parametrize("scale,expected_error,expected_k0,expected_k1", NONRIGID_ADAM_BASELINES)
def test_align_nonrigid_adam_matches_frozen_baseline(scale, expected_error, expected_k0, expected_k1):
    """Nonrigid on synthetic data must match frozen baseline.

    Runs preprocess → affine → nonrigid (2 iterations for speed).
    If the GPU warp or translation path changes numerical output,
    these baselines catch it immediately.
    """
    im0, im1, _ = make_synthetic_drift_data(scale=scale, seed=42)
    drift = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(
        step=0.02, num_tests=5, refine=True,
        show_merged=False, show_images=False,
    )
    drift.align_nonrigid(
        backend="pytorch", num_iterations=2, adam_steps=50,
        regularization_sigma_px=16.0,
        # Pin lr to the value the baselines were captured at - the public
        # default is now auto-derived from max_image_shift, but the frozen
        # baselines must stay numerically stable across that change.
        lr=0.02,
        show_merged=False, show_images=False,
    )
    np.testing.assert_almost_equal(
        drift.error_track[-1, 1], expected_error, decimal=8)
    np.testing.assert_almost_equal(
        drift.knots[0].sum(), expected_k0, decimal=6)
    np.testing.assert_almost_equal(
        drift.knots[1].sum(), expected_k1, decimal=6)


# Frozen baselines for the pytorch backend with optimizer_name="lbfgs".
# Shares _compiled_loss_fn with the Adam path, so this catches regressions
# in either the optimizer dispatch or the shared loss.
NONRIGID_LBFGS_BASELINES = [
    (1, 0.07269975543022156, 12152.98459815979, 28536.15177345276),
    (2, 0.11153321713209152, 50293.12340545654, 113601.38675689697),
]


@pytest.mark.parametrize("scale,expected_error,expected_k0,expected_k1", NONRIGID_LBFGS_BASELINES)
def test_align_nonrigid_lbfgs_matches_frozen_baseline(scale, expected_error, expected_k0, expected_k1):
    """Nonrigid LBFGS path on synthetic data must match frozen baseline.

    The LBFGS optimizer uses a closure-based forward+backward instead of
    Adam's compiled inner loop. This test ensures both paths stay
    numerically deterministic and that LBFGS doesn't silently regress.
    """
    im0, im1, _ = make_synthetic_drift_data(scale=scale, seed=42)
    drift = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(
        step=0.02, num_tests=5, refine=True,
        show_merged=False, show_images=False,
    )
    drift.align_nonrigid(
        backend="pytorch", optimizer_name="lbfgs",
        num_iterations=2, lbfgs_max_iter=20,
        regularization_sigma_px=16.0,
        show_merged=False, show_images=False,
    )
    np.testing.assert_almost_equal(
        drift.error_track[-1, 1], expected_error, decimal=8)
    np.testing.assert_almost_equal(
        drift.knots[0].sum(), expected_k0, decimal=6)
    np.testing.assert_almost_equal(
        drift.knots[1].sum(), expected_k1, decimal=6)
