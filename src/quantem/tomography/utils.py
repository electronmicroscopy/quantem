import torch
import torch.nn.functional as F

from quantem.core import config

# --- Projection Operator Utils ---


def rot_ZXZ(mags, z1, x, z3, device, mode="bilinear"):
    if not isinstance(x, torch.Tensor) or not isinstance(z1, torch.Tensor):
        z1 = torch.tensor(z1, dtype=torch.float32, device=device)
        x = torch.tensor(x, dtype=torch.float32, device=device)
        z3 = torch.tensor(z3, dtype=torch.float32, device=device)
    curr_mags = mags

    curr_mags = differentiable_rotz_vectorized(curr_mags, z1, mode)
    curr_mags = differentiable_rotx_vectorized(curr_mags, x, mode)

    curr_mags = differentiable_rotz_vectorized(curr_mags, z3, mode)

    return curr_mags


def differentiable_rotz_vectorized(mags, theta, mode="bilinear"):
    _, dimz, dimy, dimx = mags.shape

    if theta.dim() == 0:
        theta = theta.unsqueeze(0)

    theta_rad = torch.deg2rad(theta)

    cos_t, sin_t = torch.cos(theta_rad), torch.sin(theta_rad)
    affine_matrix = torch.stack(
        [cos_t, -sin_t, torch.zeros_like(theta), sin_t, cos_t, torch.zeros_like(theta)], dim=-1
    ).view(-1, 2, 3)

    mags = mags.permute(1, 0, 2, 3)

    def transform_slice(mag_slice):
        grid = F.affine_grid(affine_matrix, mag_slice.unsqueeze(0).shape, align_corners=False)
        return F.grid_sample(mag_slice.unsqueeze(0), grid, mode=mode, align_corners=False).squeeze(
            0
        )

    rotated_mags = torch.vmap(transform_slice)(mags)
    return rotated_mags.permute(1, 0, 2, 3)


def differentiable_rotx_vectorized(mags, theta, mode="bilinear"):
    _, dimz, dimy, dimx = mags.shape

    if theta.dim() == 0:
        theta = theta.unsqueeze(0)

    theta_rad = torch.deg2rad(theta)

    cos_t, sin_t = torch.cos(theta_rad), torch.sin(theta_rad)
    affine_matrix = torch.stack(
        [cos_t, -sin_t, torch.zeros_like(theta), sin_t, cos_t, torch.zeros_like(theta)], dim=-1
    ).view(-1, 2, 3)

    mags = mags.permute(3, 0, 1, 2)

    def transform_slice(mag_slice):
        grid = F.affine_grid(affine_matrix, mag_slice.unsqueeze(0).shape, align_corners=False)
        return F.grid_sample(mag_slice.unsqueeze(0), grid, mode=mode, align_corners=False).squeeze(
            0
        )

    rotated_mags = torch.vmap(transform_slice)(mags)
    return rotated_mags.permute(1, 2, 3, 0)


def tv_loss_vol_sq(obj: torch.Tensor) -> torch.Tensor:
    """Squared-anisotropic volume TV: sum of squared forward differences.

    Computes ``Σ (Δd)² + Σ (Δh)² + Σ (Δw)²`` over the three trailing
    spatial dims, leaving any leading channel/batch axes intact (they are
    included in the sum). This is the unnormalized ``tv_vol`` regularizer;
    callers apply their own ``weight / numel`` scaling.

    When the optional ``quantem-cuda`` package is installed
    (``pip install quantem[cuda]``), the tensor is on a CUDA device, and the
    ``use_cuda_kernels`` config option is true (default), this dispatches to
    the fused CUDA forward/backward kernel — identical math, one kernel
    launch instead of several large intermediates.

    Args:
        obj: Tensor of shape ``[..., D, H, W]`` (ndim >= 3).

    Returns:
        0-dim tensor on the same device as ``obj``; differentiable.
    """
    if (
        obj.is_cuda
        and obj.dtype == torch.float32
        and config.get("has_quantem_cuda")
        and config.get("use_cuda_kernels", default=True)
    ):
        from quantem.cuda.core import tv_loss_sq_3d

        return tv_loss_sq_3d(obj)

    tv_d = torch.pow(obj[..., 1:, :, :] - obj[..., :-1, :, :], 2).sum()
    tv_h = torch.pow(obj[..., :, 1:, :] - obj[..., :, :-1, :], 2).sum()
    tv_w = torch.pow(obj[..., :, :, 1:] - obj[..., :, :, :-1], 2).sum()
    return tv_d + tv_h + tv_w


def tv_loss_1d(x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    1D Total Variation Loss.

    Encourages piecewise smoothness by penalizing differences between
    adjacent elements.

    Args:
        x:         Input tensor of shape (N, C, L) or (N, L) or (L,)
        reduction: 'mean' | 'sum' | 'none'

    Returns:
        Scalar loss (or per-sample tensor if reduction='none')
    """
    # Difference between adjacent elements along the last dimension
    diff = x[..., 1:] - x[..., :-1]  # shape: (..., L-1)
    tv = diff.abs()  # L1 variant  ← most common

    if reduction == "mean":
        return tv.mean()
    elif reduction == "sum":
        return tv.sum()
    elif reduction == "none":
        return tv
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")
