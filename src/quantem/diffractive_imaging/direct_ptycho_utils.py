from collections import defaultdict
from typing import TYPE_CHECKING, Literal

from quantem.core import config

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch

import math
import warnings

import numpy as np
from tqdm.auto import tqdm

from quantem.core.utils.imaging_utils import cross_correlation_shift_torch, unwrap_phase_2d_torch
from quantem.core.utils.validators import validate_tensor
from quantem.diffractive_imaging.complex_probe import (
    spatial_frequencies,
)

# bilinear corner offsets: (row offset, col offset)
_BILINEAR_CORNERS = ((0, 0), (1, 0), (0, 1), (1, 1))

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


def _rotation_degrees_to_radians(rotation_angle: float | None) -> float | None:
    if rotation_angle is None:
        return None
    return math.radians(float(rotation_angle))


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
    """Fit low-order aberrations from lateral shifts.

    Returns ``rotation_angle`` in degrees.
    """
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
        "rotation_angle": torch.rad2deg(rotation_rad).item(),
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


def preferred_float_dtype(device) -> torch.dtype:
    """Widest float the device supports: float64 everywhere except MPS, which has none.

    Used for splat accumulators and for scan coordinates. On MPS the float32 fallback
    resolves canvas positions to roughly 1e-3 pixels at a 10k-pixel canvas, well below the
    sub-pixel detail any of this is trying to preserve.
    """
    return torch.float32 if torch.device(device).type == "mps" else torch.float64


def allocate_splat_buffers(
    canvas_shape: tuple[int, int],
    device,
    dtype: torch.dtype | None = None,
    accumulate_squares: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Flat, zeroed ``(sum_w, sum_wv, sum_wv2)`` accumulators for :func:`scatter_add_splat`."""
    if dtype is None:
        dtype = preferred_float_dtype(device)
    numel = canvas_shape[0] * canvas_shape[1]

    def _zeros():
        return torch.zeros(numel, device=device, dtype=dtype)

    return _zeros(), _zeros(), _zeros() if accumulate_squares else None


def _deposition_corners(coords: torch.Tensor, interpolation: str):
    """``(base, frac, corners)`` for a sub-pixel deposition scheme.

    ``frac`` is ``None`` for nearest-neighbour, where every corner weight is one.
    """
    if interpolation == "bilinear":
        base = torch.floor(coords)
        return base, coords - base, _BILINEAR_CORNERS
    if interpolation == "nearest":
        return torch.round(coords), None, ((0, 0),)
    raise ValueError(f"`interpolation` must be 'bilinear' or 'nearest', got {interpolation!r}")


def _resolve_indices(row, col, weights, n_rows: int, n_cols: int, boundary: str):
    """Apply the boundary rule and flatten to ``(flat_indices, weights)``."""
    if boundary == "wrap":
        row = row % n_rows
        col = col % n_cols
    elif boundary == "pad":
        # clamp the indices and zero their weights instead of masking, which would
        # need a `nonzero()` and hence a device->host synchronization
        valid = (row >= 0) & (row < n_rows) & (col >= 0) & (col < n_cols)
        row = row.clamp(0, n_rows - 1)
        col = col.clamp(0, n_cols - 1)
        weights = weights * valid
    else:
        raise ValueError(f"`boundary` must be 'wrap' or 'pad', got {boundary!r}")

    return (row * n_cols + col).reshape(-1), weights


def scatter_add_splat(
    values: torch.Tensor,
    coords: torch.Tensor,
    canvas_shape: tuple[int, int],
    *,
    boundary: Literal["wrap", "pad"] = "wrap",
    interpolation: Literal["bilinear", "nearest"] = "bilinear",
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Batched sub-pixel scatter-add of values onto a 2D canvas.

    This is the torch counterpart of :func:`quantem.core.utils.imaging_utils.bilinear_kde`'s
    accumulation stage: it splits each point across the four surrounding pixels with bilinear
    weights and accumulates ``w``, ``w * v`` and ``w * v**2`` via ``index_add_``. Unlike that
    function it takes a leading batch axis, runs on any torch device, offers a drop
    (``"pad"``) as well as a wrap boundary, and does no smoothing or normalization.

    Parameters
    ----------
    values : torch.Tensor
        ``(..., T)`` values to deposit.
    coords : torch.Tensor
        ``(..., T, 2)`` canvas coordinates in pixels, ordered ``(row, col)``. Broadcast
        against ``values``.
    canvas_shape : tuple of int
        ``(n_rows, n_cols)`` of the output canvas.
    boundary : {"wrap", "pad"}
        ``"wrap"`` wraps coordinates periodically; ``"pad"`` drops out-of-bounds points.
    interpolation : {"bilinear", "nearest"}
        Sub-pixel deposition scheme.
    out : tuple of torch.Tensor, optional
        Pre-allocated flat buffers ``(sum_w, sum_wv, sum_wv2)`` to accumulate into, as
        returned by :func:`allocate_splat_buffers`. ``sum_wv2`` may be ``None`` to skip
        the sum-of-squares. If omitted, fresh buffers are allocated.

    Returns
    -------
    sum_w, sum_wv, sum_wv2 : torch.Tensor
        Flat ``(n_rows * n_cols,)`` accumulators. ``sum_wv2`` is ``None`` if not requested.

    Notes
    -----
    ``index_add_`` uses atomics on CUDA and is therefore not bit-reproducible there;
    compare results with a tolerance rather than for exact equality.
    """
    n_rows, n_cols = int(canvas_shape[0]), int(canvas_shape[1])

    if out is None:
        out = allocate_splat_buffers(canvas_shape, coords.device)
    sum_w, sum_wv, sum_wv2 = out
    dtype = sum_w.dtype

    values = values.to(dtype)
    coords = coords.to(dtype)

    base, frac, corners = _deposition_corners(coords, interpolation)
    base_row = base[..., 0].to(torch.int64)
    base_col = base[..., 1].to(torch.int64)

    for d_row, d_col in corners:
        if frac is None:
            weights = torch.ones_like(values)
        else:
            w_row = frac[..., 0] if d_row else 1 - frac[..., 0]
            w_col = frac[..., 1] if d_col else 1 - frac[..., 1]
            weights = w_row * w_col
            weights = weights.expand_as(values) if weights.shape != values.shape else weights

        flat_indices, weights = _resolve_indices(
            base_row + d_row, base_col + d_col, weights, n_rows, n_cols, boundary
        )
        flat_weights = weights.reshape(-1)
        flat_values = values.reshape(-1)

        sum_w.index_add_(0, flat_indices, flat_weights)
        sum_wv.index_add_(0, flat_indices, flat_weights * flat_values)
        if sum_wv2 is not None:
            sum_wv2.index_add_(0, flat_indices, flat_weights * flat_values * flat_values)

    return sum_w, sum_wv, sum_wv2


def splat_stack(
    values: torch.Tensor,
    coords: torch.Tensor,
    canvas_shape: tuple[int, int],
    *,
    boundary: Literal["wrap", "pad"] = "wrap",
    interpolation: Literal["bilinear", "nearest"] = "nearest",
) -> torch.Tensor:
    """
    Splat each row of a batch onto its own canvas: ``(B, T)`` -> ``(B, n_rows, n_cols)``.

    :func:`scatter_add_splat` accumulates a whole batch into one shared canvas, which is what
    the parallax kernel wants. A convolution kernel needs each bright-field image separately,
    so that it can be convolved with that image's own kernel before the sum -- see
    :func:`splat_and_convolve` and :func:`convolve_stack_fourier`.

    Deposition matches :func:`scatter_add_splat` exactly, so splatting and then convolving is
    the same operator as :func:`scatter_add_convolve`, only reorganized.
    """
    n_rows, n_cols = int(canvas_shape[0]), int(canvas_shape[1])
    batch = int(values.shape[0])
    dtype = values.dtype if values.is_floating_point() else torch.float32

    flat = torch.zeros(batch * n_rows * n_cols, device=values.device, dtype=dtype)
    values = values.to(dtype)
    coords = coords.to(dtype)

    base, frac, corners = _deposition_corners(coords, interpolation)
    base_row = base[..., 0].to(torch.int64)
    base_col = base[..., 1].to(torch.int64)
    # offset each batch element into its own slab of the flat buffer
    slab = (
        torch.arange(batch, device=values.device, dtype=torch.int64).view(-1, 1) * n_rows * n_cols
    )

    for d_row, d_col in corners:
        if frac is None:
            weights = torch.ones_like(values)
        else:
            w_row = frac[..., 0] if d_row else 1 - frac[..., 0]
            w_col = frac[..., 1] if d_col else 1 - frac[..., 1]
            weights = w_row * w_col
            weights = weights.expand_as(values) if weights.shape != values.shape else weights

        indices, weights = _resolve_indices(
            base_row + d_row, base_col + d_col, weights, n_rows, n_cols, boundary
        )
        indices = (indices.view(batch, -1) + slab).reshape(-1)
        flat.index_add_(0, indices, (weights * values).reshape(-1))

    return flat.view(batch, n_rows, n_cols)


def splat_and_convolve(
    values: torch.Tensor,
    coords: torch.Tensor,
    canvas_shape: tuple[int, int],
    stencil_weights: torch.Tensor,
    radius: int,
    *,
    boundary: Literal["wrap", "pad"] = "wrap",
    interpolation: Literal["bilinear", "nearest"] = "nearest",
) -> torch.Tensor:
    """
    Splat each bright-field image, then convolve it with its own square kernel.

    The same operator as :func:`scatter_add_convolve`, reorganized so the convolution is a
    grouped ``conv2d`` rather than a loop over taps. Measured on MPS with 167k bright-field
    pixels and a 180x140 canvas, a radius-8 stencil takes 5.4 s here against 78 s there.

    ``stencil_weights`` is ``(B, (2 * radius + 1) ** 2)``, ordered as the ``"ij"`` meshgrid
    :meth:`DirectPtychographyMontage._return_kernel_stencil` builds.

    Returns ``(B, n_rows, n_cols)``; sum over the batch to accumulate.
    """
    n_rows, n_cols = int(canvas_shape[0]), int(canvas_shape[1])
    size = 2 * radius + 1

    if boundary == "wrap":
        stack = splat_stack(
            values, coords, canvas_shape, boundary="wrap", interpolation=interpolation
        )
        stack = torch.nn.functional.pad(stack.unsqueeze(0), (radius,) * 4, mode="circular")
    else:
        # `scatter_add_convolve` tests the boundary at the *deposit* position, so a point
        # just outside still contributes inward through the kernel. Growing the canvas by
        # the radius keeps those points; the unpadded conv2d below crops back.
        grown = (n_rows + 2 * radius, n_cols + 2 * radius)
        stack = splat_stack(
            values, coords + radius, grown, boundary="pad", interpolation=interpolation
        ).unsqueeze(0)

    batch = int(values.shape[0])
    # torch conv2d correlates rather than convolves, so flip the kernel
    weight = torch.flip(stencil_weights.reshape(batch, 1, size, size), dims=(-2, -1))
    convolved = torch.nn.functional.conv2d(stack.to(weight.dtype), weight, groups=batch)
    return convolved.view(batch, n_rows, n_cols)


def convolve_stack_fourier(
    stack: torch.Tensor,
    kernel_fourier: torch.Tensor,
) -> torch.Tensor:
    """
    Multiply each canvas's transform by its own Fourier kernel, and return the sum in ``q``.

    Exact, where a stencil is truncated -- which matters because the SSB, OBF and
    matched-filter kernels are never compact in real space (their transforms have ``r**-1.5``
    tails). It is also asymptotically cheaper for a kernel that spans the canvas: one FFT per
    bright-field image against ``(2 * radius + 1) ** 2`` taps per point.

    The convolution is circular, as it is for any Fourier method -- ``DirectPtychography``
    included. Zero-pad the canvas beforehand to get a linear one.
    """
    return (torch.fft.fft2(stack) * kernel_fourier).sum(0)


def scatter_add_convolve(
    values: torch.Tensor,
    coords: torch.Tensor,
    canvas_shape: tuple[int, int],
    stencil_offsets: torch.Tensor,
    stencil_weights: torch.Tensor,
    *,
    boundary: Literal["wrap", "pad"] = "wrap",
    interpolation: Literal["bilinear", "nearest"] = "bilinear",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scatter-add each value onto a canvas spread over a complex convolution stencil.

    Where :func:`scatter_add_splat` deposits a value at a point, this deposits
    ``value * stencil_weights[b, s]`` at ``coords + stencil_offsets[s]`` for every tap ``s``
    -- the real-space form of multiplying a bright-field image by a Fourier kernel. The
    parallax kernel is the special case of a single tap of weight one, which is why the
    montage is cheap; SSB and OBF kernels need hundreds of taps.

    Kept separate from :func:`scatter_add_splat` rather than folded into it: the accumulator
    here is complex, and the weight and sum-of-squares buffers that drive
    ``variance_loss`` have no meaning when the taps are kernel weights rather than a
    partition of unity.

    Parameters
    ----------
    values : torch.Tensor
        ``(B, T)`` values to deposit.
    coords : torch.Tensor
        ``(B, T, 2)`` canvas coordinates in pixels, ordered ``(row, col)``.
    canvas_shape : tuple of int
        ``(n_rows, n_cols)`` of the output canvas.
    stencil_offsets : torch.Tensor
        ``(S, 2)`` integer pixel offsets of the stencil taps.
    stencil_weights : torch.Tensor
        ``(B, S)`` complex weight of each tap, per batch element.
    out : torch.Tensor, optional
        Flat complex ``(n_rows * n_cols,)`` accumulator to add into.

    Returns
    -------
    torch.Tensor
        The flat complex accumulator.

    Notes
    -----
    Taps are looped over rather than broadcast, so peak memory stays ``O(B * T)`` however
    large the stencil is. ``index_add_`` uses atomics on CUDA and is not bit-reproducible
    there.
    """
    n_rows, n_cols = int(canvas_shape[0]), int(canvas_shape[1])

    if out is None:
        out = torch.zeros(n_rows * n_cols, device=values.device, dtype=torch.complex64)
    values = values.to(out.dtype)
    stencil_weights = stencil_weights.to(out.dtype)

    base, frac, corners = _deposition_corners(coords, interpolation)
    base_row = base[..., 0].to(torch.int64)
    base_col = base[..., 1].to(torch.int64)

    for d_row, d_col in corners:
        if frac is None:
            corner_weight = None
        else:
            w_row = frac[..., 0] if d_row else 1 - frac[..., 0]
            w_col = frac[..., 1] if d_col else 1 - frac[..., 1]
            corner_weight = (w_row * w_col).to(out.dtype)

        for tap, (s_row, s_col) in enumerate(stencil_offsets.tolist()):
            contribution = values * stencil_weights[:, tap : tap + 1]
            if corner_weight is not None:
                contribution = contribution * corner_weight

            flat_indices, contribution = _resolve_indices(
                base_row + d_row + int(s_row),
                base_col + d_col + int(s_col),
                contribution,
                n_rows,
                n_cols,
                boundary,
            )
            out.index_add_(0, flat_indices, contribution.reshape(-1))

    return out


def validate_probe_positions(positions):
    """``(N, 2)`` float64 probe positions in Angstrom, from an array, tensor or Dataset2d."""
    from quantem.core.datastructures import Dataset2d

    if isinstance(positions, Dataset2d):
        if str(positions.units[0]) != "A":
            raise ValueError(f"`positions` must be given in 'A', got {tuple(positions.units)!r}")
        positions = positions.array

    positions = np.asarray(
        positions.detach().cpu().numpy() if hasattr(positions, "detach") else positions,
        dtype=np.float64,
    )
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"`positions` must have shape (N, 2), got {positions.shape}")
    return positions


def infer_scan_sampling(positions_ang, max_points: int = 4096):
    """Median nearest-neighbour spacing, isotropic, from a subsample of positions."""
    points = positions_ang
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points).astype(int)]

    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    spacing = float(np.median(distances.min(axis=1)))
    return (spacing, spacing)


def build_vbf_stack_from_dataset3d(
    dataset,
    positions,
    scan_sampling,
    device: str | int = "cpu",
    max_batch_size: int | None = None,
    fit_method: str = "plane",
    mode: str = "bilinear",
    force_measured_origin=None,
    force_fitted_origin=None,
    rotation_angle: float | None = None,
    intensity_threshold: float = 0.5,
    normalization_order: int = 0,
    bf_mask=None,
):
    """
    Origin-correct and mask an ungridded diffraction stack into a flat vBF stack.

    The ungridded counterpart of :func:`build_vbf_stack_from_dataset4d`, shared by
    ``DirectPtychographyMontage.from_dataset3d`` and ``DirectPtychography.from_dataset3d``
    so the two entry points cannot drift apart.

    Returns
    -------
    vbf_stack : torch.Tensor
        ``(N_bf, N)`` bright-field intensities, flattened over scan positions.
    positions_px : ndarray
        ``(N, 2)`` positions in scan pixels, anchored at the bounding-box corner.
    bf_mask_dataset : Dataset2d
    scan_gpts : tuple of int
        Grid that just covers the positions.
    scan_sampling : tuple of float
        Resolved pixel size, with ``"auto"`` replaced by the inferred value.
    rotation_angle : float
    scan_origin : tuple of float
        The bounding-box corner the positions were anchored at, in Angstrom. Keeping it
        lets ``scan_origin + positions_px * scan_sampling`` recover the input positions,
        so several acquisitions of the same region stay in one coordinate frame.
    """
    from quantem.core.datastructures import Dataset2d

    positions_ang = validate_probe_positions(positions)
    if positions_ang.shape[0] != dataset.shape[0]:
        raise ValueError(
            f"`positions` has {positions_ang.shape[0]} rows but `dataset` has "
            f"{dataset.shape[0]} diffraction patterns."
        )

    if isinstance(scan_sampling, str):
        if scan_sampling != "auto":
            raise ValueError(f"`scan_sampling` must be a pair or 'auto', got {scan_sampling!r}")
        scan_sampling = infer_scan_sampling(positions_ang)
        warnings.warn(
            f"Inferred scan_sampling={scan_sampling} Angstrom from the median "
            "nearest-neighbour position spacing.",
            stacklevel=3,
        )
    scan_sampling = tuple(float(s) for s in scan_sampling)

    if normalization_order != 0:
        raise ValueError(
            "`normalization_order=1` fits a 2D linear background per bright-field image "
            "and needs a scan grid, which an ungridded scan does not have; use "
            "`normalization_order=0`."
        )

    shifted_tensor, rotation_angle = fit_and_shift_diffraction_origin(
        dataset,
        device=device,
        max_batch_size=max_batch_size,
        fit_method=fit_method,
        mode=mode,
        force_measured_origin=force_measured_origin,
        force_fitted_origin=force_fitted_origin,
        rotation_angle=rotation_angle,
        probe_positions=positions_ang,
    )

    if bf_mask is None:
        bf_mask = bf_mask_from_mean_pattern(shifted_tensor, intensity_threshold)
    else:
        bf_mask = validate_tensor(bf_mask, "bf_mask", dtype=torch.bool).to(shifted_tensor.device)
        if tuple(bf_mask.shape) != tuple(shifted_tensor.shape[-2:]):
            raise ValueError(
                f"`bf_mask` has shape {tuple(bf_mask.shape)} but the detector is "
                f"{tuple(shifted_tensor.shape[-2:])}."
            )
    bf_mask_dataset = Dataset2d.from_array(
        bf_mask.cpu().numpy(),
        name="BF mask",
        units=dataset.units[-2:],
        sampling=dataset.sampling[-2:],
    )

    vbf_stack = shifted_tensor[..., bf_mask].cpu()  # (N, N_bf)
    vbf_stack = normalize_vbf_stack(vbf_stack, normalization_order, vbf_stack.shape[:1])
    vbf_stack = vbf_stack.T.contiguous()  # (N_bf, N)

    # anchor to the position bounding box, then convert to canvas pixels
    scan_origin = positions_ang.min(axis=0)
    positions_px = (positions_ang - scan_origin) / np.asarray(scan_sampling)
    scan_gpts = tuple(int(math.ceil(v)) + 1 for v in positions_px.max(axis=0))

    return (
        vbf_stack,
        positions_px,
        bf_mask_dataset,
        scan_gpts,
        scan_sampling,
        rotation_angle,
        tuple(float(v) for v in scan_origin),
    )


def regrid_vbf_stack(
    vbf_stack,
    positions_px,
    scan_gpts,
    interpolation: Literal["nearest", "bilinear"] = "nearest",
    hole_fill: Literal["mean", "zero"] = "mean",
):
    """
    Resample a flat ``(N_bf, N)`` vBF stack onto a regular scan grid.

    Splats each bright-field image at its probe positions with no aberration shift and
    divides by the accumulated weight, which lets the scan-Fourier formulation accept an
    ungridded acquisition.

    A grid finer than the scan is a supported case: the empty pixels between the probe
    positions are then the sparse comb that ``upsampling_factor`` would build by Fourier
    tiling, except that the teeth land on the real probe positions rather than a lattice,
    and the deconvolution unfolds them from the bright-field shifts.

    Parameters
    ----------
    interpolation : {"nearest", "bilinear"}
        Deposition scheme, matching ``DirectPtychographyMontage.reconstruct``. ``"nearest"``
        by default: it keeps each measurement on one pixel, where bilinear smears it over
        four and blurs away the sub-pixel detail a finer grid exists to capture (radial CTF
        correlation 0.997 versus 0.986 on a scattered scan).
    hole_fill : {"mean", "zero"}
        What to put in grid pixels no probe position reached.

        ``"mean"`` (default) uses each bright-field image's mean over the visited pixels.
        ``DirectPtychography._preprocess`` zeroes the DC bin, which subtracts the mean over
        the whole grid, holes included -- so zero-filled holes sit at ``-mean``, a hard-edged
        step the deconvolution then smears across the reconstruction. Filling with the
        occupied mean puts them at the zero level and the step disappears. Measured on a
        masked disk-shaped scan with 20% holes, correlation with ground truth goes from 0.25
        (zero-filled) to 0.69 (mean-filled), against 0.69 for the montage on the same data.

        The same choice serves a finer-than-scan grid: filling the comb's gaps with the
        occupied mean and then zeroing the DC bin leaves them at zero, which is what the
        unfolding needs.

        ``"zero"`` leaves them at zero without that centering, and inverts the
        reconstruction. Kept for comparison.

    Returns
    -------
    gridded : torch.Tensor
        ``(N_bf, *scan_gpts)``.
    hole_fraction : float
    occupied : torch.Tensor
        Boolean ``scan_gpts`` map of which pixels a probe position reached.
    """
    if hole_fill not in ("mean", "zero"):
        raise ValueError(f"`hole_fill` must be 'mean' or 'zero', got {hole_fill!r}")

    num_bf = int(vbf_stack.shape[0])
    device = vbf_stack.device
    scan_gpts = (int(scan_gpts[0]), int(scan_gpts[1]))
    coords = torch.as_tensor(positions_px, device=device, dtype=preferred_float_dtype(device))[
        None
    ]

    # one image at a time: `scatter_add_splat` accumulates its whole batch into a single
    # canvas, which is what the montage wants but would sum the stack away here
    gridded = torch.empty((num_bf, *scan_gpts), device=device, dtype=torch.float32)
    weights = None
    for index in range(num_bf):
        buffers = allocate_splat_buffers(scan_gpts, device, accumulate_squares=False)
        sum_w, sum_wv, _ = scatter_add_splat(
            vbf_stack[index : index + 1],
            coords,
            scan_gpts,
            boundary="pad",
            interpolation=interpolation,
            out=buffers,
        )
        if weights is None:
            weights = sum_w
        normalized = sum_wv / sum_w.clamp_min(torch.finfo(sum_w.dtype).tiny)
        gridded[index] = normalized.reshape(scan_gpts).to(torch.float32)

    occupied = (weights > 0).reshape(scan_gpts)
    hole_fraction = float((~occupied).sum()) / occupied.numel()

    if hole_fill == "mean" and bool(occupied.any()):
        occupied_mean = gridded[:, occupied].mean(dim=1)
        gridded[:, ~occupied] = occupied_mean[:, None]

    # Empty pixels only matter when there were enough positions to fill the grid and they
    # still did not: on a deliberately finer grid the gaps are the point, and `nearest`
    # deposition leaves ~30% empty even at the same size -- a configuration that measures
    # *better* than a gapless bilinear one, so a low threshold would mislead.
    positions_per_pixel = len(positions_px) / (scan_gpts[0] * scan_gpts[1])
    if positions_per_pixel >= 1.0 and hole_fraction > 0.5:
        warnings.warn(
            f"{hole_fraction:.1%} of the {scan_gpts[0]}x{scan_gpts[1]} scan grid received no "
            f"probe position, despite {len(positions_px)} positions being available to cover "
            f"{scan_gpts[0] * scan_gpts[1]} pixels, so the positions are clustered rather "
            f"than merely sparse. Those pixels were filled with `hole_fill={hole_fill!r}`. "
            "Use a coarser `scan_sampling`, or DirectPtychographyMontage, which needs no "
            "grid at all.",
            stacklevel=3,
        )

    return gridded, hole_fraction, occupied


def fit_and_shift_diffraction_origin(
    dataset,
    device: str | int = "cpu",
    max_batch_size: int | None = None,
    fit_method: str = "plane",
    mode: str = "bilinear",
    force_measured_origin=None,
    force_fitted_origin=None,
    rotation_angle: float | None = None,
    probe_positions=None,
):
    """
    Measure, fit and remove the diffraction origin, returning a corner-centered stack.

    Works for both 4D ``(Rx, Ry, Qx, Qy)`` and 3D ``(N, Qx, Qy)`` datasets. For 3D input
    ``probe_positions`` (``(N, 2)``) must be supplied for the background fit, and
    ``rotation_angle`` must be given -- rotation estimation needs the 2D scan grid.

    Returns
    -------
    shifted_tensor : torch.Tensor
        Same shape as the input, with the diffraction origin moved to ``(0, 0)``.
    rotation_angle : float
        The supplied angle, or the estimated one when ``rotation_angle`` was ``None``.
    """
    from quantem.diffractive_imaging.origin_models import CenterOfMassOriginModel

    origin = CenterOfMassOriginModel.from_dataset(dataset, device=device)

    # measure and fit origin
    if force_fitted_origin is None:
        if force_measured_origin is None:
            origin.calculate_origin(max_batch_size)
        else:
            origin.origin_measured = force_measured_origin
        if probe_positions is None:
            origin.fit_origin_background(fit_method=fit_method)
        else:
            origin.fit_origin_background(probe_positions=probe_positions, fit_method=fit_method)
    else:
        origin.origin_fitted = force_fitted_origin

    if rotation_angle is None:
        if dataset.ndim != 4:
            raise ValueError(
                "`rotation_angle` must be given for non-raster scans: detector rotation is "
                "estimated from the curl of the center-of-mass over a 2D scan grid, which "
                "requires 4D data."
            )
        origin.estimate_detector_rotation()
        rotation_angle = origin.detector_rotation_deg

    # shift to origin
    origin.shift_origin_to(
        max_batch_size=max_batch_size,
        mode=mode,
    )

    return origin.shifted_tensor, rotation_angle


def estimate_frame_drift(
    reconstructions,
    upsample_factor: int = 16,
    num_iterations: int = 3,
    verbose: bool = True,
):
    """
    Rigid drift of each frame of a multi-frame acquisition, in Angstrom.

    A long acquisition is often split into several interleaved frames -- successive passes
    of a self-filling hexagonal grid, say -- so that specimen drift shows up as a shift
    *between* frames rather than as a smear within one. Reconstruct each frame on its own,
    pass the reconstructions here, and subtract the returned drift from the probe positions
    before reconstructing them all together::

        drift = estimate_frame_drift(montages)
        combined_positions = np.concatenate([p - d for p, d in zip(positions, drift)])

    Every reconstruction must cover the *same* window of the specimen, since the estimate
    is a plain cross-correlation between them; pin it with ``reconstruct``'s ``obj_origin``
    and ``obj_fov``, which is checked here.

    Parameters
    ----------
    reconstructions : sequence of DirectPtychographyBase
        Reconstructed frames, in acquisition order. Each must have been reconstructed.
    upsample_factor : int
        Sub-pixel refinement of the correlation peak.
    num_iterations : int
        Leave-one-out refinement passes. Each frame is aligned against the mean of the
        others, so no single frame is privileged as the reference; one pass is usually
        enough, and the estimate converges within two or three.
    verbose : bool
        Report the drift per frame, and the largest update of the final pass.

    Returns
    -------
    drift : ndarray
        ``(n_frames, 2)`` drift in Angstrom, ordered ``(row, col)`` and referred to the mean
        over frames, so it sums to zero rather than pinning frame 0.

    Notes
    -----
    The drift is *rigid per frame*: it cannot represent drift accumulating within a frame,
    which is what interleaving the frames is meant to avoid in the first place. For drift
    that varies along the scan, see
    :class:`~quantem.imaging.drift.DriftCorrection`, which warps individual scanlines.
    """
    frames = list(reconstructions)
    if len(frames) < 2:
        raise ValueError("`estimate_frame_drift` needs at least two reconstructions.")

    images, samplings = [], []
    for i, frame in enumerate(frames):
        obj = frame.obj
        if obj is None:
            raise ValueError(f"Frame {i} has not been reconstructed yet; call `.reconstruct()`.")
        images.append(np.asarray(obj, dtype=np.float64))
        samplings.append(np.asarray(frame._obj_sampling, dtype=np.float64))

    shapes = {img.shape for img in images}
    if len(shapes) != 1:
        raise ValueError(
            f"Frames must share a canvas to be correlated, got shapes {sorted(shapes)}. "
            "Reconstruct them with the same `obj_origin` and `obj_fov`."
        )

    # a canvas of the right shape in the wrong place is the subtler failure: the correlation
    # would then measure the canvas offset rather than the drift, and silently succeed
    origins = np.array([frame.obj_origin for frame in frames], dtype=np.float64)
    sampling = samplings[0]
    if not np.allclose(np.abs(origins - origins[0]).max(), 0.0, atol=1e-3 * sampling.min()):
        raise ValueError(
            "Frames share a canvas shape but not a canvas origin, so a correlation between "
            f"them would measure that offset rather than the drift: {origins.tolist()}. "
            "Reconstruct them with the same `obj_origin`."
        )
    if not all(np.allclose(s, sampling) for s in samplings):
        raise ValueError(f"Frames must share a sampling, got {[s.tolist() for s in samplings]}.")

    stack = torch.as_tensor(np.array(images), dtype=torch.float64)
    spectra = torch.fft.fft2(stack)
    kx = torch.fft.fftfreq(stack.shape[-2], dtype=torch.float64)[:, None]
    ky = torch.fft.fftfreq(stack.shape[-1], dtype=torch.float64)[None, :]

    shifts = torch.zeros((len(frames), 2), dtype=torch.float64)
    for _ in range(max(1, int(num_iterations))):
        ramp = torch.exp(
            -2j * np.pi * (kx * shifts[:, 0, None, None] + ky * shifts[:, 1, None, None])
        )
        aligned = torch.fft.ifft2(spectra * ramp).real

        total = aligned.sum(dim=0)
        updates = torch.zeros_like(shifts)
        for i in range(len(frames)):
            # leave-one-out reference, so no frame is privileged as "the" reference
            reference = (total - aligned[i]) / (len(frames) - 1)
            updates[i] = cross_correlation_shift_torch(
                reference, aligned[i], upsample_factor=upsample_factor
            )
        shifts = shifts + updates
        shifts = shifts - shifts.mean(dim=0, keepdim=True)

    # `cross_correlation_shift_torch` returns the shift that *undoes* the displacement, so
    # the drift itself -- how far the frame moved -- is its negation
    drift = -shifts.numpy() * sampling

    if verbose:
        residual = float(updates.abs().max()) * float(sampling.max())
        print(f"Frame drift (Angstrom), final pass moved at most {residual:.2f} A:")
        for i, d in enumerate(drift):
            print(f"  frame {i}: ({d[0]:+8.2f}, {d[1]:+8.2f})")

    return drift


def bf_mask_from_mean_pattern(shifted_tensor, intensity_threshold: float = 0.5):
    """Bright-field mask from the mean diffraction pattern of a corner-centered stack."""
    scan_dims = tuple(range(shifted_tensor.ndim - 2))
    mean_dp = shifted_tensor.mean(dim=scan_dims)
    return mean_dp > mean_dp.max() * intensity_threshold


def normalize_vbf_stack(vbf_stack, normalization_order: int, gpts: tuple[int, int]):
    """
    Normalize a ``(*scan_gpts, N_bf)`` virtual bright-field stack.

    ``normalization_order=0`` scales each BF image to unity mean over the scan;
    ``normalization_order=1`` divides out a least-squares linear background instead.
    """
    if normalization_order == 0:
        scan_dims = tuple(range(vbf_stack.ndim - 1))
        vbf_stack = vbf_stack / vbf_stack.mean(scan_dims)  # unity mean, important

    elif normalization_order == 1:
        # Fit linear background to each BF image
        x = torch.linspace(-0.5, 0.5, gpts[0])
        y = torch.linspace(-0.5, 0.5, gpts[1])
        ya, xa = torch.meshgrid(y, x, indexing="ij")

        # Basis for linear fit: [1, x, y]
        basis = torch.stack(
            [torch.ones_like(xa.ravel()), xa.ravel(), ya.ravel()], dim=1
        )  # shape: [N_pixels, 3]

        # Fit each BF image
        for k in range(vbf_stack.shape[-1]):
            intensities = vbf_stack[..., k].ravel()

            # Least squares
            coefs = torch.linalg.lstsq(basis, intensities).solution

            # Normalize
            background = (basis @ coefs).reshape(gpts)
            vbf_stack[..., k] /= background
    else:
        raise ValueError(f"`normalization_order` must be 0 or 1, got {normalization_order!r}")

    return vbf_stack


def build_vbf_stack_from_dataset4d(
    dataset,
    device: str | int = "cpu",
    max_batch_size: int | None = None,
    fit_method: str = "plane",
    mode: str = "bilinear",
    force_measured_origin=None,
    force_fitted_origin=None,
    rotation_angle: float | None = None,
    intensity_threshold: float = 0.5,
    normalization_order: int = 0,
    edge_blend_pixels: int = 0,
):
    """
    Turn a 4D-STEM dataset into the virtual bright-field stack the direct-ptychography
    classes consume.

    Returns
    -------
    vbf_dataset : Dataset3d
        ``(N_bf, Rx, Ry)`` stack of virtual bright-field images.
    bf_mask_dataset : Dataset2d
        Corner-centered bright-field mask on the detector grid.
    rotation_angle : float
        The supplied angle, or the estimated one when ``rotation_angle`` was ``None``.
    """
    from quantem.core.datastructures import Dataset2d, Dataset3d

    shifted_tensor, rotation_angle = fit_and_shift_diffraction_origin(
        dataset,
        device=device,
        max_batch_size=max_batch_size,
        fit_method=fit_method,
        mode=mode,
        force_measured_origin=force_measured_origin,
        force_fitted_origin=force_fitted_origin,
        rotation_angle=rotation_angle,
    )

    bf_mask = bf_mask_from_mean_pattern(shifted_tensor, intensity_threshold)
    bf_mask_dataset = Dataset2d.from_array(
        bf_mask.cpu().numpy(),
        name="BF mask",
        units=dataset.units[-2:],
        sampling=dataset.sampling[-2:],
    )

    # vbf_stack
    vbf_stack = shifted_tensor[..., bf_mask].cpu()
    gpts = vbf_stack.shape[:2]
    vbf_stack = normalize_vbf_stack(vbf_stack, normalization_order, gpts)

    # smooth window
    window_edge = create_edge_window(shape=gpts, edge_blend_pixels=edge_blend_pixels, device="cpu")
    vbf_stack = (1 - window_edge[..., None]) + window_edge[..., None] * vbf_stack

    vbf_stack = torch.moveaxis(vbf_stack, (0, 1, 2), (1, 2, 0))
    vbf_dataset = Dataset3d.from_array(
        vbf_stack.numpy(),
        name="vBF stack",
        units=("index",) + tuple(dataset.units[:2]),
        sampling=(1,) + tuple(dataset.sampling[:2]),
    )

    return vbf_dataset, bf_mask_dataset, rotation_angle
