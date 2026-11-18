"""Tests for the pure-torch Radon transform (``quantem.tomography.radon.radon``).

All CPU, deterministic. Cross-checked against scikit-image where a ground truth helps.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from quantem.tomography.radon.radon import (
    get_fourier_filter_torch,
    iradon_torch,
    radon_torch,
)
from quantem.tomography.utils import rot_ZXZ


def _disk(n: int, cy: int, cx: int, r: int) -> torch.Tensor:
    yy, xx = np.mgrid[0:n, 0:n]
    return torch.from_numpy((((yy - cy) ** 2 + (xx - cx) ** 2) < r**2).astype(np.float32))


class TestRadonShapes:
    def test_radon_shapes(self):
        theta = torch.linspace(0, 180, 30)
        assert radon_torch(_disk(64, 32, 32, 10), theta=theta).shape == (30, 64)
        imgs = torch.stack([_disk(48, 24, 20, 8), _disk(48, 24, 28, 8)])
        assert radon_torch(imgs, theta=theta).shape == (2, 30, 48)
        assert radon_torch(_disk(32, 16, 16, 6)).shape == (180, 32)  # default theta

    def test_iradon_shapes(self):
        theta = torch.linspace(0, 180, 25)
        sino = radon_torch(_disk(40, 20, 20, 8), theta=theta)
        assert iradon_torch(sino, theta=theta).shape == (40, 40)
        assert iradon_torch(sino, theta=theta, output_size=32).shape == (32, 32)


class TestFourierFilter:
    def test_bad_arguments_raise(self):
        with pytest.raises(ValueError):
            get_fourier_filter_torch(63, "ramp")
        with pytest.raises(ValueError):
            get_fourier_filter_torch(64, "not-a-filter")

    def test_none_filter_is_all_ones(self):
        f = get_fourier_filter_torch(64, None)
        assert torch.allclose(f, torch.ones_like(f))

    def test_named_filters_window_the_ramp(self):
        ramp = get_fourier_filter_torch(64, "ramp")
        assert ramp.shape == (1, 64)
        assert int(torch.argmax(ramp)) == 32  # the ramp peaks at Nyquist
        for name in ["shepp-logan", "cosine", "hamming", "hann"]:
            f = get_fourier_filter_torch(64, name)
            assert f.shape == (1, 64), name
            assert torch.all(f >= -1e-6) and torch.all(f <= ramp + 1e-6), name
            assert f[0, 32] < ramp[0, 32], name  # attenuated at Nyquist


class TestRadonBehaviour:
    def test_circular_mask_zeros_corners(self):
        """The forward transform masks to the inscribed circle, so corner mass is dropped."""
        img = torch.ones(32, 32)
        full = img.sum()
        sino = radon_torch(img, theta=torch.tensor([0.0]))
        # A single 0-degree projection sums columns; total equals the masked mass < full.
        assert sino.sum() < full

    def test_iradon_circle_zeros_outside(self):
        sino = radon_torch(_disk(48, 24, 24, 10), theta=torch.linspace(0, 180, 30))
        rec = iradon_torch(sino, theta=torch.linspace(0, 180, 30), circle=True)
        n = rec.shape[0]
        yy, xx = np.mgrid[0:n, 0:n]
        c = (n - 1) / 2
        outside = ((yy - c) ** 2 + (xx - c) ** 2) > (n // 2) ** 2
        assert torch.allclose(rec[outside], torch.zeros(int(outside.sum())))

    def test_roundtrip_recovers_structure(self):
        """Also with ``theta`` defaulted on both sides: radon and iradon must share an angle
        convention. iradon's default previously included the 180-degree endpoint while radon's
        did not, so a default-theta round-trip sampled mismatched angles.
        """
        theta = torch.linspace(0, 180, 60)
        for disk, kwargs in [
            (_disk(64, 32, 24, 9), {"theta": theta}),
            (_disk(64, 32, 28, 10), {}),
        ]:
            rec = iradon_torch(radon_torch(disk, **kwargs), filter_name="ramp", **kwargs)
            corr = np.corrcoef(disk.numpy().ravel(), rec.numpy().ravel())[0, 1]
            assert corr > 0.9, kwargs

    def test_iradon_circle_edge_reads_filtered_detector(self):
        """Padding the sinogram with zeros must not change the reconstruction: pixels at the
        circle edge read the filtered detector instead of extrapolating past its last bin."""
        theta = torch.linspace(-70, 70, 29)
        sino = radon_torch(_disk(48, 20, 27, 9), theta=theta)
        rec = iradon_torch(sino, theta=theta)
        rec_padded = iradon_torch(F.pad(sino, (6, 6)), theta=theta, output_size=48)
        assert torch.allclose(rec, rec_padded, atol=1e-5)

    def test_matches_rot_zxz_tilt_series(self):
        """A [X, Z, Y] object projected by radon_torch gives the tilt series of the same
        [Z, Y, X] volume under rot_ZXZ, so SIRT / FBP and the INR share one tilt convention."""
        vol = torch.zeros(32, 32, 32)  # [Z, Y, X], even N
        vol[8:16, 6:16, 10:22] = 1.0
        vol[16:24, 16:26, 5:11] = 0.6
        theta = torch.tensor([-60.0, -25.0, 0.0, 35.0])
        sino = radon_torch(vol.permute(2, 0, 1), theta=theta)  # [X, A, Y]
        series = rot_ZXZ(vol[None], 0.0, theta, 0.0)[:, 0].sum(1)  # [A, Y, X]
        assert torch.allclose(sino.permute(1, 2, 0), series, atol=1e-4)


class TestRadonVsSkimage:
    """scikit-image's theta is minus quantem's tilt; for odd N the rotation centres agree."""

    def test_matches_skimage_with_negated_theta(self):
        sk = pytest.importorskip("skimage.transform")
        n = 63
        disk = _disk(n, 28, 36, 12)
        theta = np.linspace(-70.0, 70.0, 29)
        ours = radon_torch(disk, theta=torch.from_numpy(theta)).numpy()  # (A, N)
        ref = sk.radon(disk.numpy().astype(np.float64), theta=-theta, circle=True)  # (N, A)
        np.testing.assert_allclose(ours, ref.T, atol=1e-4)
        rec = iradon_torch(torch.from_numpy(ref.T).float(), theta=torch.from_numpy(theta))
        ref_rec = sk.iradon(ref, theta=-theta, circle=True, filter_name="ramp")
        np.testing.assert_allclose(rec.numpy(), ref_rec, atol=1e-5)
