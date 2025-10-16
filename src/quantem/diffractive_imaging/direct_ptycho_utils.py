from typing import TYPE_CHECKING

from quantem.core import config

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch

import math


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


def _torch_polar(m: torch.Tensor):
    U, S, Vh = torch.linalg.svd(m)
    u = U @ Vh
    p = Vh.T.conj() @ S.diag().to(dtype=m.dtype) @ Vh
    return u, p
