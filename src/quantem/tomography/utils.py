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


def differentiable_shift_2d(image, shift_x, shift_y, sampling_rate):
    """
    Shifts a 2D image using grid_sample in a differentiable manner.

    Args:
        image: Tensor of shape [H, W]
        shift_x: Scalar tensor (dx) for shift in x-direction (in physical units)
        shift_y: Scalar tensor (dy) for shift in y-direction (in physical units)
        sampling_rate: Scalar value (physical units per pixel) to correctly normalize shifts

    Returns:
        Shifted image of shape [H, W]
    """
    H, W = image.shape

    # Convert physical shift to pixel shift
    shift_x_pixel = shift_x
    shift_y_pixel = shift_y

    # Normalize shift for grid_sample (assuming align_corners=True)
    normalized_shift_x = shift_x_pixel * 2 / (W - 1)
    normalized_shift_y = shift_y_pixel * 2 / (H - 1)

    # Create normalized grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=image.device),
        torch.linspace(-1, 1, W, device=image.device),
        indexing="ij",
    )

    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # [1, H, W, 2]

    # Apply shift (ensure it's differentiable)
    grid[:, :, :, 0] -= normalized_shift_x
    grid[:, :, :, 1] -= normalized_shift_y

    # Add batch and channel dimensions
    image = image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # Sample using grid_sample (fully differentiable)
    shifted_image = F.grid_sample(
        image, grid, mode="bicubic", padding_mode="zeros", align_corners=True
    )

    return shifted_image.squeeze(0).squeeze(0)  # Back to [H, W]


# --- TV loss ---


def get_TV_loss(tensor, factor=1e-3):
    tv_d = torch.pow(tensor[:, :, 1:, :, :] - tensor[:, :, :-1, :, :], 2).sum()
    tv_h = torch.pow(tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :], 2).sum()
    tv_w = torch.pow(tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1], 2).sum()
    tv_loss = tv_d + tv_h + tv_w

    return tv_loss * factor / (torch.prod(torch.tensor(tensor.shape)))

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
