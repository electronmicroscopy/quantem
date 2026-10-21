import torch
import torch.nn.functional as F

from quantem.core.utils.rotations import quat_random_uniform, quat_to_matrix


def random_quaternion(
    size: tuple, device: str | torch.device = "cpu", generator: torch.Generator | None = None
) -> torch.Tensor:
    """Uniform random rotations (Shoemake), scalar-first [w, x, y, z] unit quaternions."""
    return quat_random_uniform(size, generator=generator, dtype=torch.float32, device=device)


def quaternion_to_2d_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """Extract 2D rotation angle around Z-axis from quaternions."""
    q = F.normalize(quaternion, dim=-1)
    return 2 * torch.atan2(q[..., 3], q[..., 0])  # 2 * atan2(z, w)


def quaternion_to_rotation_matrix_2d(
    quaternion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert quaternions to 2D rotation matrix components."""
    w, x, y, z = F.normalize(quaternion, dim=-1).unbind(-1)
    return (1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * y + z * w), 1 - 2 * (x**2 + z**2))


def rasterization_2dgs(
    positions: torch.Tensor,
    sigmas: torch.Tensor,
    intensities: torch.Tensor,
    grids: tuple[torch.Tensor, torch.Tensor],
    isotropic_splats: bool = True,
    quaternions: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    Rasterize 2D Gaussian splats, supporting [sigma_z, sigma_y, sigma_x].
    For isotropic splats (sigma_y == sigma_x, sigma_z == 0), this reduces to the standard 2D Gaussian.
    For anisotropic or nonzero sigma_z, the effective 2D sigmas are used.
    The normalization is always correct for the 2D marginal (i.e., integrates to intensity).

    If quaternions are provided for anisotropic splats, rotation will be applied.
    Quaternion format: [w, x, y, z] where w is the scalar part and [x,y,z] is the vector part.
    """
    grid_y, grid_x = grids
    z, y, x = positions.T
    dx = grid_x.unsqueeze(0) - x.unsqueeze(1).unsqueeze(2)  # (N, H, W)
    dy = grid_y.unsqueeze(0) - y.unsqueeze(1).unsqueeze(2)  # (N, H, W)

    if isotropic_splats:
        sigmas = sigmas.mean(dim=1)
        # sigmas = sigmas[:, 0]
        amps = intensities * ((2 * torch.pi) ** 0.5) * sigmas
        exp_arg = (dy**2 + dx**2) / (sigmas[:, None, None] ** 2 + 1e-12)
    else:
        sigma_z, sigma_y, sigma_x = sigmas[:, 0], sigmas[:, 1], sigmas[:, 2]
        amps = intensities * ((2 * torch.pi) ** 0.5) * sigma_z

        if quaternions is not None:
            # Use quaternions to apply rotation to the Gaussian ellipse
            # Convert quaternions to 2D rotation matrix components
            r00, r01, r10, r11 = quaternion_to_rotation_matrix_2d(quaternions)

            # Apply rotation to each point
            dx_rot = r00[:, None, None] * dx + r01[:, None, None] * dy
            dy_rot = r10[:, None, None] * dx + r11[:, None, None] * dy

            # Compute the Gaussian exponent using the rotated coordinates
            exp_arg = (dx_rot**2) / (sigma_x[:, None, None] ** 2 + 1e-12) + (dy_rot**2) / (
                sigma_y[:, None, None] ** 2 + 1e-12
            )
        else:
            # Without rotation, just use the standard formula
            exp_arg = (dy**2) / (sigma_y[:, None, None] ** 2 + 1e-12) + (dx**2) / (
                sigma_x[:, None, None] ** 2 + 1e-12
            )

    gaussians = amps[:, None, None] * torch.exp(-0.5 * exp_arg)
    image = gaussians.sum(dim=0)
    return image[None,]


def rasterization_volume(
    positions: torch.Tensor,
    sigmas: torch.Tensor,
    intensities: torch.Tensor,
    volume_shape: tuple[int, int, int],  # (D, H, W)
    volume_size: tuple[float, float, float],  # Physical size in each dimension
    isotropic_splats: bool = True,
    quaternions: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    Rasterize 3D Gaussian splats to a volume.

    Args:
        positions: (N, 3) tensor of [z, y, x] positions
        sigmas: (N, 3) tensor of [sigma_z, sigma_y, sigma_x]
        intensities: (N,) tensor of intensities
        volume_shape: (D, H, W) output volume dimensions
        volume_size: (depth, height, width) physical size of volume
        isotropic_splats: If True, use mean of sigmas for all dimensions
        quaternions: (N, 4) tensor of [w, x, y, z] quaternions for rotation

    Returns:
        torch.Tensor: (D, H, W) rasterized volume
    """
    device = positions.device
    dtype = positions.dtype
    N = positions.shape[0]

    # Create coordinate grids more efficiently
    coords = [
        torch.linspace(0, size, steps, device=device, dtype=dtype)
        for size, steps in zip(volume_size, volume_shape)
    ]
    grid_z, grid_y, grid_x = torch.meshgrid(*coords, indexing="ij")

    # Vectorized distance computation using broadcasting
    pos_expanded = positions.view(N, 3, 1, 1, 1)  # (N, 3, 1, 1, 1)
    grid_coords = torch.stack([grid_z, grid_y, grid_x], dim=0).unsqueeze(0)  # (1, 3, D, H, W)
    diffs = grid_coords - pos_expanded  # (N, 3, D, H, W)

    if isotropic_splats:
        sigma_iso = sigmas.mean(dim=1)
        norm_const = intensities * (2 * torch.pi) ** 1.5 * sigma_iso.pow(3)
        dist_sq = (diffs**2).sum(dim=1)  # (N, D, H, W)
        exp_arg = dist_sq / ((sigma_iso**2).view(N, 1, 1, 1) + 1e-12)

    else:
        sigma_z, sigma_y, sigma_x = sigmas.unbind(1)
        sigma_sq_expanded = (sigmas**2).view(N, 3, 1, 1, 1)  # (N, 3, 1, 1, 1)
        norm_const = intensities * (2 * torch.pi) ** 1.5 * (sigma_z * sigma_y * sigma_x)

        if quaternions is not None:
            R = quaternion_to_rotation_matrix_3d(quaternions)  # (N, 3, 3)
            diffs_rot = torch.einsum("nij,njdhw->nidhw", R, diffs)  # (N, 3, D, H, W)
            exp_arg = ((diffs_rot**2) / (sigma_sq_expanded + 1e-12)).sum(dim=1)  # (N, D, H, W)
        else:
            # Axis-aligned ellipsoids - more efficient computation
            exp_arg = ((diffs**2) / (sigma_sq_expanded + 1e-12)).sum(dim=1)  # (N, D, H, W)

    # Compute Gaussians and sum efficiently
    gaussians = norm_const.view(N, 1, 1, 1) * torch.exp(-0.5 * exp_arg)
    volume = gaussians.sum(dim=0)

    return volume  # (D, H, W)


def quaternion_to_rotation_matrix_3d(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to 3D rotation matrices.

    Args:
        quaternions: (N, 4) tensor of [w, x, y, z] quaternions, normalized here

    Returns:
        torch.Tensor: (N, 3, 3) rotation matrices, see
        :func:`quantem.core.utils.rotations.quat_to_matrix`
    """
    return quat_to_matrix(F.normalize(quaternions, dim=-1))
