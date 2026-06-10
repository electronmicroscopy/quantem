"""Tests for ``interpolate_ms_features_tilted``'s optional quantem-cuda dispatch.

The function must produce the same features and gradients on every path:
pure torch on CPU, pure torch on GPU (kill-switch or quantem-cuda absent),
and the fused CUDA kernel when quantem-cuda is installed. The dispatch
itself is asserted by monkeypatching the kernel entry point, so these tests
are meaningful both with and without quantem-cuda in the environment.
"""

import pytest
import torch
from torch import nn

from quantem.core import config
from quantem.core.ml.models.kplanes import interpolate_ms_features_tilted

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
requires_quantem_cuda = pytest.mark.skipif(
    not config.get("has_quantem_cuda"), reason="requires quantem-cuda"
)


def _inputs(device, requires_grad=False, seed=0, B=64, T=3, C=5, scales=(9, 17)):
    gen = torch.Generator().manual_seed(seed)
    pts = torch.rand(B, 3, generator=gen) * 2.2 - 1.1  # some points outside [-1, 1]
    rotations = torch.rand(T, 3, 3, generator=gen) * 2 - 1
    grids = nn.ParameterList(
        nn.Parameter(torch.rand(3 * T, C, s, s, generator=gen) * 0.4 + 0.1) for s in scales
    )
    pts = pts.to(device).requires_grad_(requires_grad)
    rotations = rotations.to(device).requires_grad_(requires_grad)
    grids = grids.to(device)
    if not requires_grad:
        for g in grids:
            g.requires_grad_(False)
    return pts, rotations, grids


@requires_gpu
def test_gpu_matches_cpu_reference():
    pts_c, rot_c, grids_c = _inputs("cpu")
    expected = interpolate_ms_features_tilted(pts_c, grids_c, rot_c)
    pts_g, rot_g, grids_g = _inputs("cuda")
    actual = interpolate_ms_features_tilted(pts_g, grids_g, rot_g).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@requires_gpu
def test_gpu_grads_match_cpu_reference():
    pts_c, rot_c, grids_c = _inputs("cpu", requires_grad=True)
    interpolate_ms_features_tilted(pts_c, grids_c, rot_c).square().sum().backward()
    pts_g, rot_g, grids_g = _inputs("cuda", requires_grad=True)
    interpolate_ms_features_tilted(pts_g, grids_g, rot_g).square().sum().backward()
    torch.testing.assert_close(pts_g.grad.cpu(), pts_c.grad, rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(rot_g.grad.cpu(), rot_c.grad, rtol=1e-3, atol=1e-5)
    for gg, gc in zip(grids_g, grids_c):
        torch.testing.assert_close(gg.grad.cpu(), gc.grad, rtol=1e-3, atol=1e-5)


@requires_gpu
@requires_quantem_cuda
def test_dispatches_to_kernel_per_scale(monkeypatch):
    import quantem.cuda.core.ml

    calls = []
    real = quantem.cuda.core.ml.kplanes_tilted_fuse

    def spy(pts, rotations, plane):
        calls.append(plane.shape)
        return real(pts, rotations, plane)

    monkeypatch.setattr(quantem.cuda.core.ml, "kplanes_tilted_fuse", spy)
    pts, rot, grids = _inputs("cuda")
    interpolate_ms_features_tilted(pts, grids, rot)
    assert len(calls) == len(grids)


@requires_gpu
@requires_quantem_cuda
def test_kill_switch_forces_torch_path(monkeypatch):
    import quantem.cuda.core.ml

    def boom(pts, rotations, plane):
        raise AssertionError("kernel should not be called with use_cuda_kernels=False")

    monkeypatch.setattr(quantem.cuda.core.ml, "kplanes_tilted_fuse", boom)
    pts, rot, grids = _inputs("cuda")
    pts_c, rot_c, grids_c = _inputs("cpu")
    expected = interpolate_ms_features_tilted(pts_c, grids_c, rot_c)
    # config.set lacks __exit__, so restore explicitly rather than via `with`.
    config.set({"use_cuda_kernels": False})
    try:
        actual = interpolate_ms_features_tilted(pts, grids, rot).cpu()
    finally:
        config.set({"use_cuda_kernels": True})
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@requires_gpu
def test_non_fp32_takes_torch_path():
    pts, rot, grids = _inputs("cuda")
    pts64 = pts.double()
    rot64 = rot.double()
    grids64 = nn.ParameterList(nn.Parameter(g.double()) for g in grids)
    out = interpolate_ms_features_tilted(pts64, grids64, rot64)
    assert out.dtype == torch.float64
