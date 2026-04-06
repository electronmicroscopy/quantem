"""Parity tests: torch drift_utils functions vs numpy/scipy equivalents.

Every torch function in drift_utils.py must produce identical output to
its numpy/scipy counterpart at float32 precision. These tests ensure that
the GPU-accelerated path doesn't silently diverge from the reference
implementation during refactoring.
"""

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from quantem.core.utils.imaging_utils import bilinear_kde
from quantem.imaging.drift_utils import (
    _parabolic_sub_pixel,
    _symmetric_pad,
    bilinear_kde_batch,
    gaussian_smooth_batch,
)


# ---------------------------------------------------------------------------
# High-level: the core warping operation that drives the grid search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1, 2])
def test_bilinear_kde_matches_numpy(scale):
    """Torch batched KDE scatter must match numpy bilinear_kde.

    This is the core warping operation: scatter source pixels onto a
    canvas with bilinear weights, smooth, and normalize. If this diverges,
    the affine grid search scores candidates differently and picks
    wrong drift vectors. Tested at two scales to catch size-dependent bugs.
    """
    rng = np.random.default_rng(42)
    num_rows_in = 32 * scale
    num_cols_in = 32 * scale
    num_rows_out = 40 * scale
    num_cols_out = 40 * scale
    kde_sigma = 0.5
    pad_value = 100.0
    values = rng.random((num_rows_in, num_cols_in)).astype(np.float32)
    # Coordinates with small offset — mimics a slight drift
    row_coords = (np.arange(num_rows_in)[:, None] + 4.0 * scale
                  + np.zeros((1, num_cols_in))).astype(np.float32)
    col_coords = (np.zeros((num_rows_in, 1))
                  + np.arange(num_cols_in)[None, :] + 4.0 * scale).astype(np.float32)
    expected = bilinear_kde(
        row_coords, col_coords, values,
        (num_rows_out, num_cols_out), kde_sigma, pad_value,
    )
    result, _ = bilinear_kde_batch(
        torch.tensor(row_coords)[None],
        torch.tensor(col_coords)[None],
        torch.tensor(values),
        (num_rows_out, num_cols_out),
        kde_sigma, pad_value,
    )
    np.testing.assert_allclose(
        result[0].numpy(), expected.astype(np.float32), atol=1e-4,
        err_msg=f"Warped image mismatch at scale={scale}",
    )


# ---------------------------------------------------------------------------
# Mid-level: smoothing and padding that the KDE depends on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [0.5, 2.0])
def test_gaussian_smooth_matches_scipy(sigma):
    """Torch separable Gaussian must match scipy.ndimage.gaussian_filter.

    The KDE normalization step (values / counts) amplifies any smoothing
    mismatch. Tested at sigma=0.5 (tight, 3-pixel kernel) and sigma=2.0
    (wide, 9-pixel kernel) to cover both regimes.
    """
    rng = np.random.default_rng(42)
    arr = rng.random((64, 64)).astype(np.float32)
    expected = gaussian_filter(arr, sigma).astype(np.float32)
    result = gaussian_smooth_batch(torch.tensor(arr)[None], sigma)[0].numpy()
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_symmetric_pad_matches_numpy():
    """Torch symmetric padding must match np.pad(mode='symmetric').

    This is critical for parity: scipy's gaussian_filter uses symmetric
    (reflect-with-edge-repeat) boundaries. If our torch padding differs,
    the smoothed KDE images diverge at canvas edges and the frozen
    baselines in test_drift.py break.
    """
    rng = np.random.default_rng(42)
    arr = rng.random((8, 10)).astype(np.float32)
    pad_rows, pad_cols = 3, 4
    expected = np.pad(arr, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode="symmetric")
    result = _symmetric_pad(torch.tensor(arr)[None, None], pad_rows, pad_cols)[0, 0].numpy()
    np.testing.assert_allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Low-level: sub-pixel math primitives
# ---------------------------------------------------------------------------


def test_parabolic_sub_pixel_exact():
    """Parabolic fit on y = -(x - offset)^2 must recover the exact offset.

    This is the sub-pixel refinement used to center the DFT upsample
    window. If the offset is wrong, the upsampled patch misses the
    true correlation peak and the shift estimate degrades.
    """
    for offset in [0.0, 0.3, -0.4, 0.49]:
        val_m1 = torch.tensor([-((-1 - offset) ** 2)])
        val_0 = torch.tensor([-(0 - offset) ** 2])
        val_p1 = torch.tensor([-(1 - offset) ** 2])
        result = _parabolic_sub_pixel(val_m1, val_0, val_p1)
        assert abs(result.item() - offset) < 1e-6, f"Failed for offset={offset}"
