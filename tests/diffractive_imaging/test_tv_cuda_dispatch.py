"""Tests for ``ObjectConstraints._calc_tv_loss``'s optional quantem-cuda dispatch.

The constraint must produce the same loss and gradients on every path: pure
torch on CPU, pure torch on GPU (kill-switch or quantem-cuda absent), and
the fused L1 kernel when quantem-cuda is installed. The dispatch itself is
asserted by monkeypatching the kernel entry point, so these tests are
meaningful both with and without quantem-cuda in the environment.
"""

import pytest
import torch

from quantem.core import config
from quantem.diffractive_imaging.object_models import ObjectConstraints

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
requires_quantem_cuda = pytest.mark.skipif(
    not config.get("has_quantem_cuda"), reason="requires quantem-cuda"
)


class _Host:
    """Minimal stand-in providing what _calc_tv_loss needs from self."""

    def __init__(self, device):
        self.device = device

    _get_zero_loss_tensor = ObjectConstraints._get_zero_loss_tensor
    _calc_tv_loss = ObjectConstraints._calc_tv_loss


def _phase(device, shape=(8, 24, 20), seed=0, requires_grad=False):
    gen = torch.Generator().manual_seed(seed)
    arr = torch.rand(shape, generator=gen, dtype=torch.float32)
    return arr.to(device).requires_grad_(requires_grad)


WEIGHTS = [(5.0, 0.1), (5.0, 0.0), (0.0, 0.3), (0.0, 0.0)]


@requires_gpu
@pytest.mark.parametrize("weight", WEIGHTS)
def test_gpu_matches_cpu_reference(weight):
    expected = _Host("cpu")._calc_tv_loss(_phase("cpu"), weight)
    actual = _Host("cuda")._calc_tv_loss(_phase("cuda"), weight)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-7)


@requires_gpu
@pytest.mark.parametrize("weight", WEIGHTS[:3])
def test_gpu_grads_match_cpu_reference(weight):
    a_cpu = _phase("cpu", requires_grad=True)
    _Host("cpu")._calc_tv_loss(a_cpu, weight).backward()
    a_gpu = _phase("cuda", requires_grad=True)
    _Host("cuda")._calc_tv_loss(a_gpu, weight).backward()
    torch.testing.assert_close(a_gpu.grad.cpu(), a_cpu.grad, rtol=1e-4, atol=1e-7)


@requires_gpu
@pytest.mark.parametrize("shape", [(1, 24, 20), (24, 20)])
def test_degenerate_and_2d_match_cpu(shape):
    # num_slices == 1 (weight[0] zeroed upstream, as get_tv_loss does) and
    # plain 2-D arrays keep exact parity whichever path runs.
    weight = (0.0, 0.3)
    expected = _Host("cpu")._calc_tv_loss(_phase("cpu", shape=shape), weight)
    actual = _Host("cuda")._calc_tv_loss(_phase("cuda", shape=shape), weight)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-7)


@requires_gpu
@requires_quantem_cuda
def test_dispatches_to_kernel(monkeypatch):
    import quantem.cuda.core

    calls = []
    real = quantem.cuda.core.tv_loss_l1_3d

    def spy(volume):
        calls.append(volume.shape)
        return real(volume)

    monkeypatch.setattr(quantem.cuda.core, "tv_loss_l1_3d", spy)
    _Host("cuda")._calc_tv_loss(_phase("cuda"), (5.0, 0.1))
    assert len(calls) == 1


@requires_gpu
@requires_quantem_cuda
def test_kill_switch_forces_torch_path(monkeypatch):
    import quantem.cuda.core

    def boom(volume):
        raise AssertionError("kernel should not be called with use_cuda_kernels=False")

    monkeypatch.setattr(quantem.cuda.core, "tv_loss_l1_3d", boom)
    expected = _Host("cpu")._calc_tv_loss(_phase("cpu"), (5.0, 0.1))
    # config.set lacks __exit__, so restore explicitly rather than via `with`.
    config.set({"use_cuda_kernels": False})
    try:
        actual = _Host("cuda")._calc_tv_loss(_phase("cuda"), (5.0, 0.1))
    finally:
        config.set({"use_cuda_kernels": True})
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-7)
