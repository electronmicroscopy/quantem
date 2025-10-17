from typing import TYPE_CHECKING

from quantem.core import config

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch

import math

from tqdm.auto import tqdm

from quantem.core.utils.imaging_utils import cross_correlation_shift_torch
from quantem.diffractive_imaging.complex_probe import spatial_frequencies


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

    pairs = torch.cat(pairs, dim=0)
    pairs = torch.unique(torch.sort(pairs, dim=1)[0], dim=0)

    if max_pairs is not None and len(pairs) > max_pairs:
        # random subsampling
        pairs = pairs[torch.randperm(len(pairs))[:max_pairs]]

    return pairs


def _compute_pairwise_shifts(
    vbf_stack: torch.Tensor,
    pairs: torch.Tensor,
    upsample_factor: int = 1,
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
        s_ij, _ = cross_correlation_shift_torch(
            vbf_stack[i],
            vbf_stack[j],
            upsample_factor=upsample_factor,
        )
        rel_shifts.append((i.item(), j.item(), s_ij))
    return rel_shifts


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
    unique_coords, inverse = torch.unique(coords, return_inverse=True, sorted=True)
    Nb = unique_coords.numel()
    mapping = inverse.to(torch.long)

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
    upsample_factor: int = 1,
    verbose: int | bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align virtual BF stack using multi-scale coarse-to-fine approach.

    Parameters
    ----------
    vbf_stack : torch.Tensor
        (N, H, W) stack of virtual BF images to align
    bf_mask : torch.BoolTensor
        (Q, R) corner-centered mask of valid BF positions
    inds_i, inds_j : torch.Tensor
        Corner-centered coordinates for each vBF
    bin_factors : tuple of int
        Sequence of binning factors from coarse to fine (e.g., (7, 6, 5, 4, 3, 2, 1))
    pair_connectivity : int
        Number of neighbors for pairwise alignment (4 or 8)
    device : torch.device
        Device to use for computation

    Returns
    -------
    global_shifts : torch.Tensor
        (N, 2) computed shifts in pixels for each vBF
    aligned_stack : torch.Tensor
        (N, H, W) aligned virtual BF stack
    """

    device = vbf_stack.device
    N = len(vbf_stack)
    global_shifts = torch.zeros((N, 2), device=device)

    pbar = tqdm(range(len(bin_factors)), disable=not verbose)
    for bin_factor in bin_factors:
        # Bin the mask and stack
        bf_mask_binned, inds_ib, inds_jb, vbf_binned, mapping = _bin_mask_and_stack_centered(
            bf_mask, inds_i, inds_j, vbf_stack, bin_factor=bin_factor
        )

        # Create neighbor pairs for correlation
        pairs = _make_periodic_pairs(bf_mask_binned, connectivity=pair_connectivity)

        # Compute pairwise shifts
        rel_shifts = _compute_pairwise_shifts(vbf_binned, pairs, upsample_factor=upsample_factor)

        # Solve for global shifts via synchronization
        shifts = _synchronize_shifts(len(vbf_binned), rel_shifts, device)

        # Accumulate shifts and apply to full-resolution stack
        global_shifts += shifts[mapping]
        vbf_stack = _fourier_shift_stack(vbf_stack, shifts[mapping])
        pbar.update(n=1)
    pbar.close()

    return global_shifts, vbf_stack


def fit_aberrations_from_shifts(
    shifts_px: torch.Tensor,
    bf_mask: torch.BoolTensor,
    wavelength: float,
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    scan_sampling: tuple[float, float],
) -> dict[str, float]:
    """ """
    device = shifts_px.device

    # Get spatial frequencies at BF positions
    kxa, kya = spatial_frequencies(gpts, sampling, device=device)
    kvec = torch.dstack((kxa[bf_mask], kya[bf_mask])).view((-1, 2))
    basis = kvec * wavelength
    scan_s = torch.as_tensor(scan_sampling, device=device)

    # Convert shifts to physical units (Angstroms)
    shifts_ang = (shifts_px * scan_s).to(dtype=basis.dtype, device=device)

    # Least-squares fit: shifts = basis @ M
    M = torch.linalg.lstsq(basis, shifts_ang, rcond=None)[0]

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
