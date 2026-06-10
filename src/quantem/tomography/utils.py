import torch
import torch.nn.functional as F

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


def _rot2d_affine_matrix(theta: torch.Tensor, batch: int) -> torch.Tensor:
    """(B, 2, 3) in-plane rotation matrices, broadcasting a single angle over the batch."""
    theta_rad = torch.deg2rad(theta)
    cos_t, sin_t = torch.cos(theta_rad), torch.sin(theta_rad)
    affine_matrix = torch.stack(
        [cos_t, -sin_t, torch.zeros_like(theta), sin_t, cos_t, torch.zeros_like(theta)], dim=-1
    ).view(-1, 2, 3)
    if affine_matrix.shape[0] == 1 and batch > 1:
        affine_matrix = affine_matrix.expand(batch, 2, 3)
    elif affine_matrix.shape[0] != batch and batch != 1:
        raise ValueError(
            f"Got {affine_matrix.shape[0]} angles for a batch of {batch} volumes; "
            "pass one angle, or one angle per volume."
        )
    return affine_matrix


def differentiable_rotz_vectorized(mags, theta, mode="bilinear"):
    B, dimz, dimy, dimx = mags.shape

    if theta.dim() == 0:
        theta = theta.unsqueeze(0)

    # A rotation about z applies the same 2-D transform to every z-slice, so the
    # z-axis rides along as grid_sample channels in a single call. (The previous
    # per-slice vmap also broke for more than one angle, because affine_grid
    # requires the matrix batch to match the slice batch of 1.)
    affine_matrix = _rot2d_affine_matrix(theta, B)
    if affine_matrix.shape[0] > B:  # one volume, many angles
        mags = mags.expand(affine_matrix.shape[0], dimz, dimy, dimx)
    grid = F.affine_grid(affine_matrix, mags.shape, align_corners=False)
    return F.grid_sample(mags, grid, mode=mode, align_corners=False)


def differentiable_rotx_vectorized(mags, theta, mode="bilinear"):
    B, dimz, dimy, dimx = mags.shape

    if theta.dim() == 0:
        theta = theta.unsqueeze(0)

    # Same trick as rotz with the x-axis as the channel dim: rotate in (z, y).
    affine_matrix = _rot2d_affine_matrix(theta, B)
    mags = mags.permute(0, 3, 1, 2)  # (B, X, Z, Y)
    if affine_matrix.shape[0] > B:  # one volume, many angles
        mags = mags.expand(affine_matrix.shape[0], dimx, dimz, dimy)
    grid = F.affine_grid(affine_matrix, mags.shape, align_corners=False)
    rotated = F.grid_sample(mags, grid, mode=mode, align_corners=False)
    return rotated.permute(0, 2, 3, 1)  # back to (B, Z, Y, X)


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
