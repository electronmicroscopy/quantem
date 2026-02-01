"""
Tests for imaging utilities in quantem.core.utils.imaging_utils
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quantem.core.utils.imaging_utils import cross_correlation_shift, cross_correlation_shift_torch


@pytest.fixture
def spot_image():
    from scipy.ndimage import gaussian_filter

    im = np.zeros((64, 64), dtype=np.float64)
    im[32, 32] = 1.0
    im = gaussian_filter(im, 2.0)
    im /= np.max(im)
    return im


def _fourier_shift_numpy(im: np.ndarray, shift_rc: tuple[float, float]) -> np.ndarray:
    dr, dc = shift_rc
    kr = np.fft.fftfreq(im.shape[0])[:, None]
    kc = np.fft.fftfreq(im.shape[1])[None, :]
    F = np.fft.fft2(im)
    phase = np.exp(-2j * np.pi * (kr * dr + kc * dc))
    return np.fft.ifft2(F * phase).real


def _wrap_shift_rc(shift_rc: tuple[float, float], shape: tuple[int, int]) -> tuple[float, float]:
    dr, dc = shift_rc
    M, N = shape
    dr = ((dr + M / 2) % M) - M / 2
    dc = ((dc + N / 2) % N) - N / 2
    return float(dr), float(dc)


@pytest.mark.parametrize(
    "shift_true, upsample_factor, atol",
    [
        ((5.0, -3.0), 1000, 1e-3),
        ((-7.123, 1.789), 1000, 1e-3),
    ],
)
def test_cross_correlation_shift_numpy_matches_expected(spot_image, shift_true, upsample_factor, atol):
    im_ref = spot_image
    im = _fourier_shift_numpy(im_ref, shift_true)
    expected = _wrap_shift_rc((-shift_true[0], -shift_true[1]), im_ref.shape)

    meas = cross_correlation_shift(im_ref, im, upsample_factor=upsample_factor)
    assert meas[0] == pytest.approx(expected[0], abs=atol)
    assert meas[1] == pytest.approx(expected[1], abs=atol)


@pytest.mark.parametrize(
    "shift_true, upsample_factor, atol",
    [
        ((5.0, -3.0), 1000, 1e-3),
        ((-7.123, 1.789), 1000, 1e-3),
    ],
)
def test_cross_correlation_shift_torch_matches_expected(spot_image, shift_true, upsample_factor, atol):
    im_ref = spot_image
    im = _fourier_shift_numpy(im_ref, shift_true)
    expected = _wrap_shift_rc((-shift_true[0], -shift_true[1]), im_ref.shape)

    t_ref = torch.from_numpy(im_ref)
    t_im = torch.from_numpy(im)
    meas = cross_correlation_shift_torch(t_ref, t_im, upsample_factor=upsample_factor).cpu().numpy()

    assert float(meas[0]) == pytest.approx(expected[0], abs=atol)
    assert float(meas[1]) == pytest.approx(expected[1], abs=atol)
