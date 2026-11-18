"""Tests for ``quantem.tomography.utils``: 1D total-variation loss and the
differentiable ZXZ rotation. All CPU."""

import pytest
import torch
import torch.nn.functional as F

from quantem.tomography.dataset_models import TomographyINRDataset
from quantem.tomography.utils import (
    rot_ZXZ,
    tv_loss_1d,
)


class TestTVLoss1D:
    def test_constant_input_is_zero(self):
        assert tv_loss_1d(torch.ones(10)) == 0.0

    def test_known_value_mean(self):
        # diffs are [1, 1, 1], abs-mean = 1.0
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        assert torch.isclose(tv_loss_1d(x, reduction="mean"), torch.tensor(1.0))

    def test_known_value_sum(self):
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        assert torch.isclose(tv_loss_1d(x, reduction="sum"), torch.tensor(3.0))

    def test_reduction_none_shape(self):
        x = torch.zeros(2, 5)
        out = tv_loss_1d(x, reduction="none")
        assert out.shape == (2, 4)

    def test_bad_reduction_raises(self):
        with pytest.raises(ValueError):
            tv_loss_1d(torch.zeros(4), reduction="median")


def _block_volume(n: int = 16) -> torch.Tensor:
    """(1, n, n, n) volume with an off-centre block so rotations are detectable."""
    vol = torch.zeros(1, n, n, n)
    vol[0, 4:12, 4:10, 5:11] = 1.0
    return vol


class TestRotations:
    def test_zero_rotation_is_identity(self):
        vol = _block_volume()
        out = rot_ZXZ(vol, 0.0, 0.0, 0.0, device="cpu")
        assert torch.max(torch.abs(out - vol)) < 1e-4

    def test_rotation_preserves_mass(self):
        vol = _block_volume()
        rotated = rot_ZXZ(vol, 0.0, 30.0, 0.0, device="cpu")
        rel_err = abs(float(rotated.sum()) - float(vol.sum())) / float(vol.sum())
        assert rel_err < 0.02

    def test_accepts_python_float_and_tensor_angle(self):
        vol = _block_volume()
        out_float = rot_ZXZ(vol, 0.0, 25.0, 0.0, device="cpu")
        out_tensor = rot_ZXZ(
            vol,
            torch.tensor(0.0),
            torch.tensor(25.0),
            torch.tensor(0.0),
            device="cpu",
        )
        assert torch.allclose(out_float, out_tensor, atol=1e-5)

    def test_rotation_changes_volume(self):
        vol = _block_volume()
        rotated = rot_ZXZ(vol, 0.0, 90.0, 0.0, device="cpu")
        assert torch.max(torch.abs(rotated - vol)) > 0.1

    def test_gradient_flows_through_rotation(self):
        vol = _block_volume().requires_grad_(True)
        z1 = torch.tensor(20.0, requires_grad=True)
        out = rot_ZXZ(vol, z1, 30.0, 0.0)
        (out * torch.linspace(0, 1, 16)).sum().backward()
        assert torch.isfinite(vol.grad).all() and vol.grad.abs().sum() > 0
        assert torch.isfinite(z1.grad) and z1.grad != 0

    def test_projection_matches_inr_rays(self):
        """rot_ZXZ and the INR rays use the same pose, including the signs of z1 and z3."""
        vol = _block_volume()
        n = vol.shape[-1]
        i, j = torch.meshgrid(torch.arange(n).float(), torch.arange(n).float(), indexing="ij")
        rays = TomographyINRDataset.create_batch_rays(i.flatten(), j.flatten(), n, n)
        for pose in [(0.0, 40.0, 0.0), (15.0, 40.0, -25.0), (-30.0, -55.0, 20.0)]:
            angles = [torch.full((n * n,), a) for a in pose]
            p = TomographyINRDataset.transform_batch_rays(
                rays, *angles, shifts=torch.zeros(n * n, 2), N=n, sampling_rate=1.0
            )
            sampled = F.grid_sample(vol[None], p[None, :, :, None], align_corners=True)
            ray_proj = sampled.view(n, n, n).sum(-1)  # [row, col]
            assert torch.allclose(rot_ZXZ(vol, *pose)[0].sum(0), ray_proj, atol=1e-4), pose

    def test_rigid_for_non_cubic_volumes(self):
        D, H, W = 12, 20, 28
        zz, yy, xx = torch.meshgrid(
            torch.arange(D), torch.arange(H), torch.arange(W), indexing="ij"
        )
        r2 = (zz - (D - 1) / 2) ** 2 + (yy - (H - 1) / 2) ** 2 + (xx - (W - 1) / 2) ** 2
        sphere = (r2 <= 5**2).float()[None]
        rotated = rot_ZXZ(sphere, 30.0, 90.0, -10.0)[0]
        extent = [
            int(e)
            for e in rotated.gt(0.5).nonzero().amax(0) - rotated.gt(0.5).nonzero().amin(0) + 1
        ]
        # a sphere of diameter 11 stays one; trilinear sampling can trim a voxel
        assert all(abs(e - 11) <= 1 for e in extent), extent

    def test_several_poses_at_once(self):
        vol = _block_volume()
        z1, x, z3 = (
            torch.tensor([0.0, 10.0]),
            torch.tensor([20.0, -40.0]),
            torch.tensor([0.0, 5.0]),
        )
        out = rot_ZXZ(vol, z1, x, z3)
        assert out.shape == (2, *vol.shape)
        for t in range(2):
            assert torch.allclose(out[t], rot_ZXZ(vol, z1[t], x[t], z3[t]), atol=1e-5), t
