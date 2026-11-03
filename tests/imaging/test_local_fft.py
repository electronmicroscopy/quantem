"""LocalFFT: windowed Fourier transforms of an image as a pseudo-4DSTEM dataset."""

import numpy as np

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.imaging.local_fft import LocalFFT, window_mask


def test_local_fft():
    # full-radius taper sin^2(pi/2 * (1 - r/R)), on a row through the center
    r = np.hypot(0.5, np.arange(64) - 31.5)
    expected = np.sin(np.pi / 2 * np.clip(1 - r / 32, 0, 1)) ** 2
    assert np.allclose(window_mask(64)[31], expected, atol=1e-6)

    # cosine fringes with g = 0.25 1/A at 60 degrees from the row axis toward the column axis
    sampling, k, theta = 0.5, 0.25, np.deg2rad(60.0)
    rr, cc = np.mgrid[:256, :256] * sampling
    image = 100.0 + 5.0 * np.cos(2 * np.pi * k * (rr * np.cos(theta) + cc * np.sin(theta)))
    image += np.random.default_rng(0).normal(0, 1.0, image.shape)
    ds = Dataset2d.from_array(image, sampling=[0.05, 0.05], units=["nm", "nm"])
    lf = LocalFFT.from_image(ds, window_size=64, step=32, k_max=0.4)

    assert np.isclose(lf.sampling, sampling), "nm must convert to Angstroms"
    dp = lf.dataset.array[3, 3]
    h = int(lf.center[0])
    assert dp[h, h] == 0.0, "the center is removed by the normalization"
    row, col = np.unravel_index(np.argmax(dp), dp.shape)
    q = np.array([row - h, col - h]) * lf.pixel_size
    assert np.isclose(np.hypot(*q), k, atol=lf.pixel_size)
    assert np.isclose(np.rad2deg(np.arctan2(q[1], q[0])) % 180, 60.0, atol=5.0)
    assert 0.5 < np.median(dp[dp != 0] + 1.0) < 1.5, "normalized noise must be close to 1"
