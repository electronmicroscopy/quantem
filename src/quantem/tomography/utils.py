import torch
import torch.nn.functional as F

# --- Projection Operator Utils ---


def rot_ZXZ(mags, z1, x, z3, device, mode="bilinear"):
    # Convert each angle independently: re-wrapping an existing tensor with
    # torch.tensor() copies and detaches it, silently cutting gradient flow
    # through tensor angles whenever any other angle is passed as a float.
    z1, x, z3 = (
        a if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.float32, device=device)
        for a in (z1, x, z3)
    )
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
