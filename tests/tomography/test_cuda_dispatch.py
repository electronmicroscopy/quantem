"""Tests for ``tv_loss_vol_sq`` and its optional quantem-cuda dispatch.

The helper must produce the same value and gradients on every path: pure
torch on CPU, pure torch on GPU (kill-switch or quantem-cuda absent), and
the fused CUDA kernel when quantem-cuda is installed. The dispatch itself is
asserted by monkeypatching the kernel entry point, so these tests are
meaningful both with and without quantem-cuda in the environment.
"""

import pytest
import torch

from quantem.core import config
from quantem.tomography.utils import tv_loss_vol_sq

from .conftest import requires_gpu

requires_quantem_cuda = pytest.mark.skipif(
    not config.get("has_quantem_cuda"), reason="requires quantem-cuda"
)


def tv_vol_sq_ref(obj: torch.Tensor) -> torch.Tensor:
    tv_d = torch.pow(obj[..., 1:, :, :] - obj[..., :-1, :, :], 2).sum()
    tv_h = torch.pow(obj[..., :, 1:, :] - obj[..., :, :-1, :], 2).sum()
    tv_w = torch.pow(obj[..., :, :, 1:] - obj[..., :, :, :-1], 2).sum()
    return tv_d + tv_h + tv_w


@pytest.mark.parametrize("shape", [(7, 6, 5), (2, 7, 6, 5)])
def test_cpu_matches_reference(shape):
    obj = torch.rand(shape, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(tv_loss_vol_sq(obj), tv_vol_sq_ref(obj))


def test_constant_volume_is_zero():
    assert tv_loss_vol_sq(torch.ones(8, 8, 8)).item() == 0.0


@requires_gpu
@pytest.mark.parametrize("shape", [(7, 6, 5), (2, 7, 6, 5)])
def test_gpu_matches_cpu_reference(shape):
    obj = torch.rand(shape, generator=torch.Generator().manual_seed(1))
    expected = tv_vol_sq_ref(obj)
    actual = tv_loss_vol_sq(obj.cuda()).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@requires_gpu
def test_gpu_grad_matches_cpu_reference():
    obj = torch.rand(6, 7, 8, generator=torch.Generator().manual_seed(2))
    v_cpu = obj.clone().requires_grad_(True)
    tv_vol_sq_ref(v_cpu).backward()
    v_gpu = obj.cuda().requires_grad_(True)
    tv_loss_vol_sq(v_gpu).backward()
    torch.testing.assert_close(v_gpu.grad.cpu(), v_cpu.grad, rtol=1e-4, atol=1e-6)


@requires_gpu
@requires_quantem_cuda
def test_dispatches_to_kernel(monkeypatch):
    import quantem.cuda

    calls = []
    real = quantem.cuda.tv_loss_sq_3d

    def spy(volume):
        calls.append(volume.shape)
        return real(volume)

    monkeypatch.setattr(quantem.cuda, "tv_loss_sq_3d", spy)
    tv_loss_vol_sq(torch.rand(4, 4, 4, device="cuda"))
    assert len(calls) == 1


@requires_gpu
@requires_quantem_cuda
def test_kill_switch_forces_torch_path(monkeypatch):
    import quantem.cuda

    def boom(volume):
        raise AssertionError("kernel should not be called with use_cuda_kernels=False")

    monkeypatch.setattr(quantem.cuda, "tv_loss_sq_3d", boom)
    obj = torch.rand(4, 4, 4, device="cuda")
    # config.set lacks __exit__, so restore explicitly rather than via `with`.
    config.set({"use_cuda_kernels": False})
    try:
        torch.testing.assert_close(tv_loss_vol_sq(obj), tv_vol_sq_ref(obj))
    finally:
        config.set({"use_cuda_kernels": True})
