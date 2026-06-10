"""Tests for ``quantem.tomography.utils``: 1D total-variation loss and the
differentiable ZXZ rotation operators. All CPU."""

import pytest
import torch

from quantem.tomography.utils import (
    differentiable_rotx_vectorized,
    differentiable_rotz_vectorized,
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
        out = differentiable_rotz_vectorized(vol, torch.tensor(20.0))
        out.sum().backward()
        assert vol.grad is not None
        assert torch.isfinite(vol.grad).all()


class TestRotZXZGradients:
    def test_grad_flows_with_mixed_float_and_tensor_angles(self):
        """Regression: a non-tensor angle made rot_ZXZ re-wrap every angle with
        torch.tensor(), detaching gradients through tensor angles."""
        vol = _block_volume()
        x = torch.tensor(20.0, requires_grad=True)
        out = rot_ZXZ(vol, 0.0, x, 0.0, device="cpu")
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad)
    @pytest.mark.parametrize(
        "rot_fn", [differentiable_rotz_vectorized, differentiable_rotx_vectorized]
    )
    def test_multi_angle_matches_per_angle_calls(self, rot_fn):
        # Regression: the per-slice vmap version raised for more than one angle
        # (affine_grid requires the matrix batch to match the slice batch of 1).
        vol = _block_volume()
        angles = torch.tensor([10.0, -25.0, 60.0])
        multi = rot_fn(vol, angles)
        per_angle = torch.cat([rot_fn(vol, a) for a in angles])
        assert multi.shape == (3, *vol.shape[1:])
        assert torch.allclose(multi, per_angle, atol=1e-6)

    @pytest.mark.parametrize(
        "rot_fn", [differentiable_rotz_vectorized, differentiable_rotx_vectorized]
    )
    def test_per_volume_angles(self, rot_fn):
        vols = torch.cat([_block_volume(), _block_volume().flip(-1), _block_volume().flip(-2)])
        angles = torch.tensor([10.0, -25.0, 60.0])
        batched = rot_fn(vols, angles)
        per_volume = torch.cat([rot_fn(vols[i : i + 1], angles[i]) for i in range(3)])
        assert torch.allclose(batched, per_volume, atol=1e-6)
