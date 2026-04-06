"""Parity tests for drift_utils.py torch functions.

Tests the core building blocks against numpy/scipy equivalents.
The full pipeline is covered by frozen baselines in test_drift.py.
"""

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from quantem.core.utils.imaging_utils import bilinear_kde
from quantem.imaging.drift_utils import (
    _parabolic_peak_2d,
    _parabolic_sub_pixel,
    _symmetric_pad,
    bilinear_kde_batch,
    cross_corr_batch,
    gaussian_smooth_1d,
    gaussian_smooth_batch,
)


# ---------------------------------------------------------------------------
# High-level: cross-correlation and warping
# ---------------------------------------------------------------------------


def test_cross_corr_zero_cost_for_identical():
    """Identical images must produce near-zero MAE after alignment.

    This validates the full sub-pixel pipeline: FFT cross-correlation →
    parabolic peak → DFT upsample → Fourier shift. If any step has a
    bias (like the 0.5 px center-index bug), identical images will show
    a nonzero cost from the spurious shift.
    """
    rng = np.random.default_rng(42)
    image = rng.random((64, 64)).astype(np.float32)
    reference = torch.tensor(image)[None]
    cost = cross_corr_batch(reference, reference.clone(), upsample_factor=8)
    assert cost.item() < 1e-6


@pytest.mark.parametrize("shift_row,shift_col", [(0, 0), (3, -5), (7, 2), (2.3, -1.7)])
def test_parabolic_peak_2d_known_shift(shift_row, shift_col):
    """Parabolic peak refinement must recover known shifts from Fourier-shifted images.

    At zero shift, this catches the negative-float-rounding bug where
    (-4.5e-8) % N = N instead of 0. At nonzero shifts, it verifies
    the periodic wrapping, stencil extraction, and sub-pixel precision.
    """
    rng = np.random.default_rng(42)
    num_pixels = 64
    image = rng.random((num_pixels, num_pixels)).astype(np.float64)
    # Fourier shift creates a perfect sub-pixel-accurate shifted image
    k_row = np.fft.fftfreq(num_pixels)[:, None]
    k_col = np.fft.fftfreq(num_pixels)[None, :]
    shifted = np.real(np.fft.ifft2(
        np.fft.fft2(image) * np.exp(-2j * np.pi * (k_row * shift_row + k_col * shift_col))
    ))
    reference = torch.tensor(image, dtype=torch.float32)[None]
    moving = torch.tensor(shifted, dtype=torch.float32)[None]
    cross_corr = torch.fft.ifft2(torch.fft.fft2(reference) * torch.fft.fft2(moving).conj()).real
    peak_flat = cross_corr.flatten(1).argmax(dim=1)
    peak_row = peak_flat // num_pixels
    peak_col = peak_flat % num_pixels
    batch_idx = torch.arange(1)
    refined_row, refined_col = _parabolic_peak_2d(
        cross_corr, peak_row, peak_col, num_pixels, num_pixels, batch_idx)
    # Cross-correlation finds the negative shift, wrapped to [0, N)
    expected_row = (-shift_row) % num_pixels
    expected_col = (-shift_col) % num_pixels
    # Parabolic gives ~0.1 px precision on sub-pixel shifts, exact on integer
    tolerance = 0.15
    assert abs(refined_row.item() - expected_row) < tolerance, f"Row: expected {expected_row}, got {refined_row.item()}"
    assert abs(refined_col.item() - expected_col) < tolerance, f"Col: expected {expected_col}, got {refined_col.item()}"


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
    source_image = rng.random((num_rows_in, num_cols_in)).astype(np.float32)
    # Fractional offsets (not integer) to exercise the bilinear weight split —
    # 4.3 means each pixel lands 0.3 of the way between grid points
    row_coords = (np.arange(num_rows_in)[:, None] + 4.3 * scale
                  + np.zeros((1, num_cols_in))).astype(np.float32)
    col_coords = (np.zeros((num_rows_in, 1))
                  + np.arange(num_cols_in)[None, :] + 4.7 * scale).astype(np.float32)
    expected = bilinear_kde(
        row_coords, col_coords, source_image,
        (num_rows_out, num_cols_out), kde_sigma, pad_value,
    )
    result, _ = bilinear_kde_batch(
        torch.tensor(row_coords)[None],
        torch.tensor(col_coords)[None],
        torch.tensor(source_image),
        (num_rows_out, num_cols_out),
        kde_sigma, pad_value,
    )
    np.testing.assert_allclose(
        result[0].numpy(), expected.astype(np.float32), atol=1e-5,
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
    image = rng.random((64, 64)).astype(np.float32)
    expected = gaussian_filter(image, sigma).astype(np.float32)
    result = gaussian_smooth_batch(torch.tensor(image)[None], sigma)[0].numpy()
    np.testing.assert_allclose(result, expected, atol=1e-5)


@pytest.mark.parametrize("sigma", [0.5, 2.0, 16.0])
def test_gaussian_smooth_1d_matches_scipy(sigma):
    """Torch 1D Gaussian must match scipy.ndimage.gaussian_filter on 1D signal.

    Used in nonrigid regularization to smooth knot residuals. sigma=16 is
    the default regularization_sigma_px — tests the exact kernel size used
    in production. If this diverges, the polynomial-detrend + smooth
    regularization produces different knot positions.
    """
    rng = np.random.default_rng(42)
    signal = rng.random(128).astype(np.float32)
    expected = gaussian_filter(signal, sigma).astype(np.float32)
    result = gaussian_smooth_1d(torch.tensor(signal)[None], sigma)[0].numpy()
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_symmetric_pad_matches_numpy():
    """Torch symmetric padding must match np.pad(mode='symmetric').

    This is critical for parity: scipy's gaussian_filter uses symmetric
    (reflect-with-edge-repeat) boundaries. If our torch padding differs,
    the smoothed KDE images diverge at canvas edges and the frozen
    baselines in test_drift.py break.
    """
    rng = np.random.default_rng(42)
    image = rng.random((8, 10)).astype(np.float32)
    pad_rows, pad_cols = 3, 4
    expected = np.pad(image, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode="symmetric")
    result = _symmetric_pad(torch.tensor(image)[None, None], pad_rows, pad_cols)[0, 0].numpy()
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
