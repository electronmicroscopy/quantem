"""
Tests for imaging utilities in quantem.core.utils.imaging_utils
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import pytest

torch = pytest.importorskip("torch")

from quantem.core.utils.imaging_utils import cross_correlation_shift, cross_correlation_shift_torch, weighted_cross_correlation_shift


@pytest.fixture
def spot_image():

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

import numpy as np
import pytest

from quantem.core.utils.imaging_utils import weighted_cross_correlation_shift


@pytest.fixture
def peak_grid_images():
    im_ref = np.zeros((80, 80), dtype=float)
    im = np.zeros_like(im_ref)

    r_ref = np.array([17, 27, 37, 47], dtype=int)
    r_im  = np.array([27, 37, 47, 57], dtype=int)  # shifted +10 rows
    c = np.array([17, 27, 37, 47], dtype=int)

    for rr in r_ref:
        for cc in c:
            im_ref[rr, cc] = 1.0

    for rr in r_im:
        for cc in c:
            im[rr, cc] = 1.0

    im_ref[37,27] = 3.0
    im[27,27] = 3.0

    im_ref = gaussian_filter(im_ref,1.0)
    im = gaussian_filter(im,1.0)

    # Smooth wrapped radial weight centered at 0 shift
    M, N = im_ref.shape
    fr = np.fft.fftfreq(M) * M
    fc = np.fft.fftfreq(N) * N
    dr2 = fr[:, None] ** 2 + fc[None, :] ** 2

    sigma = 3.0
    weight = np.exp(dr2 / (-2.0*sigma**2))

    return im_ref, im, weight


def test_weighted_cross_correlation_shift_unweighted_prefers_full_overlap(peak_grid_images):
    im_ref, im, weight = peak_grid_images
    shift = weighted_cross_correlation_shift(im_ref, im, upsample_factor=1000)
    assert np.allclose(shift, (-10.0, 0.0), atol=1e-3)


def test_weighted_cross_correlation_shift_weighted_prefers_near_zero(peak_grid_images):
    im_ref, im, weight = peak_grid_images
    shift = weighted_cross_correlation_shift(im_ref, im, weight_real=weight, upsample_factor=1000)
    assert np.allclose(shift, (0.0, 0.0), atol=1e-3)
