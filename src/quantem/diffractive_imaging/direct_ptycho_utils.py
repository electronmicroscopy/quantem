from collections import defaultdict
from typing import TYPE_CHECKING

from quantem.core import config

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch

import math

from tqdm.auto import tqdm

from quantem.core.utils.imaging_utils import cross_correlation_shift_torch, unwrap_phase_2d_torch
from quantem.diffractive_imaging.complex_probe import (
    spatial_frequencies,
)

# fmt: off
ABERRATION_PRESETS = {
    "defocus": ["C10"],
    "quadratic": ["C10", "C12_a", "C12_b"],
    "low_order": [
        "C10", "C12_a", "C12_b",
        "C21_a", "C21_b", "C30",
    ],
    "all": [
        "C10", "C12_a", "C12_b",
        "C21_a", "C21_b", "C23_a", "C23_b",
        "C30", "C32_a", "C32_b", "C34_a", "C34_b",
        "C41_a", "C41_b", "C43_a", "C43_b", "C45_a", "C45_b",
        "C50", "C52_a", "C52_b", "C54_a", "C54_b", "C56_a", "C56_b",
    ],
}
# fmt: on


def create_edge_window(shape, edge_blend_pixels, device="cpu"):
    """
    Create a smooth edge window that transitions from 0 at edges to 1 in center.

    Parameters
    ----------
    shape : tuple
        (height, width) of the window
    edge_blend_pixels : float
        Width of the transition region in pixels
    device : str or torch.device
        Device to create tensor on

    Returns
    -------
    window : torch.Tensor
        2D window with smooth edges, shape (height, width)
    """
    if edge_blend_pixels == 0:
        return torch.ones(shape, device=device)

    h, w = shape
    # Create 1D windows for each dimension
    x = torch.linspace(-1, 1, w, device=device)
    y = torch.linspace(-1, 1, h, device=device)

    # Distance from edge (0 at edge, increases toward center)
    dist_x = torch.clamp((1 - torch.abs(x)) * w / 2 / edge_blend_pixels, 0, 1)
    dist_y = torch.clamp((1 - torch.abs(y)) * h / 2 / edge_blend_pixels, 0, 1)

    # Smooth transition using sin^2
    wx = torch.sin(dist_x * (torch.pi / 2)) ** 2
    wy = torch.sin(dist_y * (torch.pi / 2)) ** 2

    # 2D window is product of 1D windows
    window = wy[:, None] * wx[None, :]

    return window


def _synchronize_shifts(num_nodes, rel_shifts, device):
    """
    Solve for absolute shifts t[i] given pairwise differences δ_ij = t_j - t_i.
    rel_shifts: list of (i, j, δ_ij)
    """
    N = num_nodes
    A = torch.zeros((N, N), device=device)
    b = torch.zeros((N, 2), device=device)
    for i, j, s in rel_shifts:
        A[i, i] += 1
        A[j, j] += 1
        A[i, j] -= 1
        A[j, i] -= 1
        b[i] -= s
        b[j] += s
    # Fix gauge (anchor one node)
    A[0, :] = 0
    A[:, 0] = 0
    A[0, 0] = 1
    b[0] = 0
    t = torch.linalg.solve(A, b)
    return t


def _make_periodic_pairs(
    bf_mask: torch.Tensor,
    connectivity: int = 4,
    max_pairs: int | None = None,
):
    """
    Construct periodic neighbor pairs (i1, j1, i2, j2) from a corner-centered mask.

    Parameters
    ----------
    bf_mask : torch.BoolTensor
        (Q, R) mask of valid positions (corner-centered grid)
    connectivity : int
        4 or 8 for neighbor connectivity
    max_pairs: int
        optional max_pairs limit for speed (random subset of edges)

    Returns
    -------
    pairs : LongTensor, shape (M, 2)
        indices (in flattened valid-index order) of neighbor pairs
    """
    Q, R = bf_mask.shape
    device = bf_mask.device
    inds_i, inds_j = torch.where(bf_mask)
    N = inds_i.numel()

    linear = -torch.ones((Q, R), dtype=torch.long, device=device)
    linear[inds_i, inds_j] = torch.arange(N, device=device)

    if connectivity == 4:
        offsets = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], device=device)
    elif connectivity == 8:
        offsets = torch.tensor(
            [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]], device=device
        )
    else:
        raise ValueError("connectivity must be 4 or 8")

    pairs = []
    for di, dj in offsets:
        # periodic wrapping
        ni = (inds_i + di) % Q
        nj = (inds_j + dj) % R
        neighbor_idx = linear[ni, nj]
        valid = neighbor_idx >= 0
        src = torch.arange(N, device=device)[valid]
        dst = neighbor_idx[valid]
        pairs.append(torch.stack([src, dst], dim=1))

    pairs = torch.sort(torch.cat(pairs, dim=0), dim=1)[0]
    pairs = torch.unique(pairs.cpu(), dim=0).to(device=device)

    if max_pairs is not None and len(pairs) > max_pairs:
        # random subsampling
        pairs = pairs[torch.randperm(len(pairs))[:max_pairs]]

    return pairs


def _compute_pairwise_shifts(
    vbf_stack: torch.Tensor,
    pairs: torch.Tensor,
    upsample_factor: int = 4,
) -> list[tuple[int, int, torch.Tensor]]:
    """
    Compute relative shifts between pairs of virtual BF images.

    Parameters
    ----------
    vbf_stack : torch.Tensor
        (N, H, W) stack of virtual BF images
    pairs : torch.Tensor
        (M, 2) pairs of indices to correlate
    upsample_factor : int
        Upsampling factor for subpixel accuracy

    Returns
    -------
    rel_shifts : list of (i, j, shift_ij)
        Relative shifts between each pair
    """
    rel_shifts = []
    for i, j in pairs:
        s_ij = cross_correlation_shift_torch(
            vbf_stack[i],
            vbf_stack[j],
            upsample_factor=upsample_factor,
        )
        rel_shifts.append((i.item(), j.item(), s_ij))
    return rel_shifts


def _compute_reference_shifts(
    vbf_stack: torch.Tensor,
    reference: torch.Tensor,
    upsample_factor: int = 4,
) -> torch.Tensor:
    """
    Compute shifts to align each image in the stack to a reference image.

    Parameters
    ----------
    vbf_stack : torch.Tensor
        (N, H, W) stack of virtual BF images
    reference : torch.Tensor
        (H, W) reference image to align to
    upsample_factor : int
        Upsampling factor for subpixel accuracy

    Returns
    -------
    shifts : torch.Tensor
        (N, 2) shifts for each image
    """
    N = len(vbf_stack)
    device = vbf_stack.device
    shifts = torch.zeros((N, 2), device=device)

    for i in range(N):
        shift = cross_correlation_shift_torch(
            reference,
            vbf_stack[i],
            upsample_factor=upsample_factor,
        )
        shifts[i] = shift

    return shifts


def _bin_mask_and_stack_centered(
    bf_mask: torch.Tensor,
    inds_i: torch.Tensor,
    inds_j: torch.Tensor,
    vbf_stack: torch.Tensor,
    bin_factor: int,
):
    """
    Centered binning for corner-centered masks.

    Each bin is centered around its binned coordinate. For bin_factor=3, bin 0
    contains original indices {-1, 0, 1}, bin 1 contains {2, 3, 4}, etc.

    Parameters
    ----------
    bf_mask : torch.BoolTensor
        (Q, R) corner-centered mask of valid positions
    inds_i, inds_j : torch.Tensor
        Corner-centered coordinates for each vBF
    vbf_stack : torch.Tensor
        (N, P, Qpix) stack of virtual BF images
    bin_factor : int
        Binning factor (1 = no binning)

    Returns
    -------
    bf_mask_b : torch.BoolTensor
        (Qb, Rb) binned mask
    inds_ib, inds_jb : torch.Tensor
        Binned coordinates for each bin (corner-centered)
    vbf_binned : torch.Tensor
        (Nb, P, Qpix) binned vBF stack
    mapping : torch.LongTensor
        (N,) mapping from original index to binned index
    """
    device = bf_mask.device
    Q, R = bf_mask.shape
    N_orig = inds_i.numel()

    if bin_factor == 1:
        bf_mask_b = bf_mask
        inds_ib = inds_i.clone()
        inds_jb = inds_j.clone()
        vbf_binned = vbf_stack.clone()
        mapping = torch.arange(N_orig, device=device, dtype=torch.long)
        return bf_mask_b, inds_ib, inds_jb, vbf_binned, mapping

    # Convert corner-centered indices to center-centered
    center_i = (inds_i + Q // 2) % Q
    center_j = (inds_j + R // 2) % R

    # Binned grid size
    Qb = math.ceil(Q / bin_factor)
    Rb = math.ceil(R / bin_factor)

    # For centered bins: bin_idx = floor((center_coord + bin_factor//2) / bin_factor)
    # This makes bin 0 contain center coords {-bin_factor//2, ..., bin_factor//2}
    offset = bin_factor // 2
    qb_center = torch.div(center_i + offset, bin_factor, rounding_mode="floor") % Qb
    rb_center = torch.div(center_j + offset, bin_factor, rounding_mode="floor") % Rb

    # Convert back to corner-centered coordinates for the binned grid
    qb = (qb_center - Qb // 2) % Qb
    rb = (rb_center - Rb // 2) % Rb

    # Encode as single coordinate for unique operation
    coords = qb * Rb + rb
    unique_coords, inverse = torch.unique(coords.cpu(), return_inverse=True, sorted=True)
    unique_coords = unique_coords.to(device=device)
    Nb = unique_coords.numel()
    mapping = inverse.to(dtype=torch.long, device=device)

    # Recover binned indices (corner-centered)
    inds_ib = (unique_coords // Rb).to(torch.long)
    inds_jb = (unique_coords % Rb).to(torch.long)

    # Accumulate vbf_stack into bins
    dtype = vbf_stack.dtype
    Ppix, Qpix = vbf_stack.shape[1], vbf_stack.shape[2]
    vbf_binned = torch.zeros((Nb, Ppix, Qpix), device=device, dtype=dtype)
    vbf_binned = vbf_binned.index_add(0, mapping, vbf_stack)

    # Form binned boolean mask
    bf_mask_b = torch.zeros((Qb, Rb), dtype=torch.bool, device=device)
    bf_mask_b[inds_ib, inds_jb] = True

    return bf_mask_b, inds_ib, inds_jb, vbf_binned, mapping


def _fourier_shift_stack(images: torch.Tensor, shifts: torch.Tensor):
    """
    Apply subpixel shifts to a stack of images using Fourier phase ramps.

    Parameters
    ----------
    images : torch.Tensor
        (N, H, W) stack of images
    shifts : torch.Tensor
        (N, 2) shifts in pixels, (shift_i, shift_j) for each image

    Returns
    -------
    shifted : torch.Tensor
        (N, H, W) shifted images
    """
    N, H, W = images.shape
    device = images.device
    dtype = images.dtype

    # FFT of images
    img_fft = torch.fft.fft2(images, dim=(-2, -1))

    # Create frequency grids (corner-centered, then convert to actual frequencies)
    freq_i = torch.fft.fftfreq(H, d=1.0, device=device)
    freq_j = torch.fft.fftfreq(W, d=1.0, device=device)
    grid_i, grid_j = torch.meshgrid(freq_i, freq_j, indexing="ij")

    # Compute phase ramps for each image
    # shift in real space = phase ramp exp(-2πi * freq * shift) in Fourier space
    shift_i = shifts[:, 0].view(-1, 1, 1)  # (N, 1, 1)
    shift_j = shifts[:, 1].view(-1, 1, 1)  # (N, 1, 1)

    phase_ramp = torch.exp(-2j * torch.pi * (grid_i * shift_i + grid_j * shift_j))

    # Apply phase ramp and inverse FFT
    shifted_fft = img_fft * phase_ramp
    shifted = torch.fft.ifft2(shifted_fft, dim=(-2, -1)).real

    return shifted.to(dtype)


def align_vbf_stack_multiscale(
    vbf_stack: torch.Tensor,
    bf_mask: torch.Tensor,
    inds_i: torch.Tensor,
    inds_j: torch.Tensor,
    bin_factors: tuple[int, ...],
    pair_connectivity: int = 4,
    upsample_factor: int = 4,
    reference: torch.Tensor | None = None,
    initial_shifts: torch.Tensor | None = None,
    running_average: bool = False,
    basis: torch.Tensor | None = None,
    verbose: int | bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align virtual BF stack using multi-scale coarse-to-fine approach.

    Parameters
    ----------
    vbf_stack : torch.Tensor
        (N, H, W) stack of virtual BF images to align. If initial_shifts provided,
        this should be the already-shifted stack.
    bf_mask : torch.BoolTensor
        (Q, R) corner-centered mask of valid BF positions
    inds_i, inds_j : torch.Tensor
        Corner-centered coordinates for each vBF
    bin_factors : tuple of int
        Sequence of binning factors from coarse to fine (e.g., (7, 6, 5, 4, 3, 2, 1))
    pair_connectivity : int
        Number of neighbors for pairwise alignment (4 or 8). Ignored if reference is provided.
    upsample_factor : int
        Upsampling factor for subpixel accuracy
    reference : torch.Tensor, optional
        (H, W) reference image to align all images to. If None, uses pairwise alignment.
        Should have same shape as each image in vbf_stack (no binning needed).
    initial_shifts : torch.Tensor, optional
        (N, 2) initial shifts already applied to vbf_stack. New shifts will be
        added to these. If None, starts from zero.
    running_average : bool
        If True and using reference mode, updates reference as a running average of
        aligned images at each bin level. Helps stabilize alignment with noisy data.
    verbose : bool
        Show progress bar


    Returns
    -------
    global_shifts : torch.Tensor
        (N, 2) computed shifts in pixels for each vBF
    aligned_stack : torch.Tensor
        (N, H, W) aligned virtual BF stack

    Notes
    -----
    Two alignment modes:
    - **Pairwise** (reference=None): Uses graph synchronization on neighbor pairs.
      More robust to outliers but slower.
    - **Reference-based** (reference provided): Aligns each image directly to reference.
      Faster and often more accurate when good reference is available.
    """

    device = vbf_stack.device
    N = len(vbf_stack)

    if initial_shifts is None:
        global_shifts = torch.zeros((N, 2), device=device)
    else:
        global_shifts = initial_shifts.clone().to(device)

    mode = "reference" if reference is not None else "pairwise"
    desc = f"Aligning ({mode})"

    iteration = 0
    current_reference = reference.clone() if reference is not None else None

    pbar = tqdm(bin_factors, desc=desc, disable=not verbose)
    for bin_factor in pbar:
        iteration += 1
        # Bin the mask and stack
        bf_mask_binned, inds_ib, inds_jb, vbf_binned, mapping = _bin_mask_and_stack_centered(
            bf_mask, inds_i, inds_j, vbf_stack, bin_factor=bin_factor
        )

        if current_reference is not None:
            # Reference-based alignment: bin the reference too
            shifts = _compute_reference_shifts(
                vbf_binned, current_reference, upsample_factor=upsample_factor
            )
        else:
            # Pairwise alignment with synchronization
            pairs = _make_periodic_pairs(bf_mask_binned, connectivity=pair_connectivity)
            rel_shifts = _compute_pairwise_shifts(
                vbf_binned, pairs, upsample_factor=upsample_factor
            )
            shifts = _synchronize_shifts(len(vbf_binned), rel_shifts, device)

        # Accumulate shifts and apply to full-resolution stack
        incremental_shifts = shifts[mapping]

        if basis is not None:
            # constrain coefficients
            global_shifts_new = global_shifts + incremental_shifts
            coeffs = torch.linalg.lstsq(basis.cpu(), global_shifts_new.cpu(), rcond=None)[0].to(
                basis.device
            )
            projected_shifts = basis @ coeffs

            incremental_shifts = projected_shifts - global_shifts
            global_shifts = projected_shifts
        else:
            global_shifts += incremental_shifts

        vbf_stack = _fourier_shift_stack(vbf_stack, incremental_shifts)

        if current_reference is not None:
            new_mean = vbf_stack.mean(0)
            if running_average:
                alpha = iteration / (iteration + 1)
                current_reference = current_reference * alpha + new_mean * (1 - alpha)
            else:
                current_reference = new_mean

    pbar.close()
    return global_shifts, vbf_stack


def fit_aberrations_from_shifts(
    shifts_ang: torch.Tensor,
    bf_mask: torch.Tensor,
    wavelength: float,
    gpts: tuple[int, int],
    sampling: tuple[float, float],
) -> dict[str, float]:
    """ """
    device = shifts_ang.device

    # Get spatial frequencies at BF positions
    kxa, kya = spatial_frequencies(gpts, sampling, device=device)
    kvec = torch.dstack((kxa[bf_mask], kya[bf_mask])).view((-1, 2))
    basis = kvec * wavelength

    # Least-squares fit: shifts = basis @ M
    M = torch.linalg.lstsq(basis.cpu(), shifts_ang.cpu(), rcond=None)[0]
    # Decompose M = R @ A (rotation × aberration)
    M_rotation, M_aberration = _torch_polar(M)

    # Extract rotation angle
    rotation_rad = -torch.arctan2(M_rotation[1, 0], M_rotation[0, 0])

    # Handle angle wrapping and sign conventions
    if 2 * torch.abs(torch.remainder(rotation_rad + math.pi, 2 * math.pi) - math.pi) > math.pi:
        rotation_rad = torch.remainder(rotation_rad, 2 * math.pi) - math.pi
        M_aberration = -M_aberration

    # Extract aberration coefficients from symmetric matrix
    a = M_aberration[0, 0]
    b = (M_aberration[1, 0] + M_aberration[0, 1]) / 2  # Symmetrize
    c = M_aberration[1, 1]

    # Defocus (isotropic component)
    C10 = (a + c) / 2

    # 2-fold astigmatism (anisotropic component)
    C12a = (a - c) / 2
    C12b = b
    C12 = torch.sqrt(C12a**2 + C12b**2)
    phi12 = torch.arctan2(C12b, C12a) / 2

    return {
        "C10": C10.item(),
        "C12": C12.item(),
        "phi12": phi12.item(),
        "rotation_angle": rotation_rad.item(),
    }


def _torch_polar(m: torch.Tensor):
    U, S, Vh = torch.linalg.svd(m)
    u = U @ Vh
    p = Vh.T.conj() @ S.diag().to(dtype=m.dtype) @ Vh
    return u, p


def unwrap_bf_overlap_phase_torch(
    complex_data_bf,  # (N_k,)
    mask_bf,  # (N_k,)
    bf_mask,  # (N_kx, N_ky)
    *,
    method="reliability-sorting",
    two_pass=True,
    **unwrap_kwargs,
):
    phase_bf = torch.angle(complex_data_bf)
    phase_grid = torch.zeros_like(bf_mask, dtype=torch.float32)
    mask_grid = torch.zeros_like(bf_mask, dtype=torch.bool)

    phase_grid[bf_mask] = phase_bf
    mask_grid[bf_mask] = mask_bf

    if mask_grid.any():
        if phase_grid.max() - phase_grid.min() > math.pi:
            phase_grid = unwrap_phase_2d_torch(
                phase_grid * mask_grid,
                method=method,
                mask=mask_grid,
                **unwrap_kwargs,
            )
            phase_grid = phase_grid * mask_grid

            if two_pass:
                phase_grid = unwrap_phase_2d_torch(
                    phase_grid,
                    method=method,
                    mask=mask_grid,
                    **unwrap_kwargs,
                )
                phase_grid = phase_grid * mask_grid

    return phase_grid[bf_mask]


def group_basis_by_method(
    basis_list: list[str],
    fit_method: str,
) -> list[list[str]]:
    """
    Group basis functions according to fit method.

    Args:
        basis_list: Flat list of basis function names
        fit_method: "global", "recursive", or "sequential"

    Returns:
        List of basis groups for iterative fitting
    """
    if fit_method == "global":
        return [basis_list]

    radial_groups = defaultdict(list)

    for basis_name in basis_list:
        if basis_name.startswith("C"):
            radial_order = int(basis_name[1])  # First digit after 'C'
            radial_groups[radial_order].append(basis_name)
        else:
            raise ValueError()

    # Sort by radial order
    sorted_orders = sorted(radial_groups.keys())

    if fit_method == "recursive":
        groups = []
        accumulated = []
        for order in sorted_orders:
            accumulated.extend(radial_groups[order])
            groups.append(accumulated.copy())
        return groups

    elif fit_method == "sequential":
        return [radial_groups[order] for order in sorted_orders]

    else:
        raise ValueError(f"Unknown fit_method: {fit_method}")


def _crop_corner_centered_mask(mask: torch.Tensor, bf_mask_padding_px: int):
    mask_c = torch.fft.fftshift(mask)
    ys, xs = torch.where(mask_c)

    px = bf_mask_padding_px
    y0, y1 = ys.min() - px, ys.max() + px + 1
    x0, x1 = xs.min() - px, xs.max() + px + 1
    return torch.fft.ifftshift(mask_c[y0:y1, x0:x1])
