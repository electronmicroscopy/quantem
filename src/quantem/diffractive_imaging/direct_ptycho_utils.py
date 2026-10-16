from dataclasses import dataclass
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

from quantem.core.utils.imaging_utils import cross_correlation_shift_torch
from quantem.core.utils.validators import validate_tensor
from quantem.diffractive_imaging.complex_probe import (
    aberration_surface_cartesian_basis,
    cartesian_to_polar_aberrations,
    overlap_terms,
    partial_coherence_envelope,
    polar_coordinates,
    spatial_frequencies,
)

# bilinear corner offsets: (row offset, col offset)
_BILINEAR_CORNERS = ((0, 0), (1, 0), (0, 1), (1, 1))

#: rows x taps to hold at once when broadcasting an event list over a stencil
_MAX_TAP_ELEMENTS = 4_194_304

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


def _rotation_degrees_to_radians(rotation_angle):
    """Degrees to radians, keeping a tensor that requires grad as a tensor."""
    if rotation_angle is None:
        return None
    if isinstance(rotation_angle, torch.Tensor) and rotation_angle.requires_grad:
        return torch.deg2rad(rotation_angle)
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
    reference_rotation: float | None = None,
) -> dict[str, float]:
    """Fit low-order aberrations from lateral shifts.

    Returns ``rotation_angle`` in degrees. The shifts are ``R(theta) A k``, which the twin
    ``R(theta + 180) (-A) k`` reproduces exactly, so they determine the solution only up to that
    pair. With ``reference_rotation`` (degrees) the member whose rotation is nearer it is
    returned; without one, the member with ``|rotation| <= 90``.
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

    # choose between the two members of the twin pair
    if reference_rotation is None:
        reference_rad = torch.zeros_like(rotation_rad)
    else:
        reference_rad = torch.as_tensor(
            math.radians(float(reference_rotation)), dtype=rotation_rad.dtype
        )
    offset = torch.remainder(rotation_rad - reference_rad + math.pi, 2 * math.pi) - math.pi
    if torch.abs(offset) > math.pi / 2:
        rotation_rad = torch.remainder(rotation_rad, 2 * math.pi) - math.pi
        M_aberration = -M_aberration
    rotation_rad = (
        reference_rad
        + torch.remainder(rotation_rad - reference_rad + math.pi, 2 * math.pi)
        - math.pi
    )

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


def fit_aberrations_from_probe(
    fourier_probe: torch.Tensor,
    bf_mask: torch.Tensor,
    wavelength: float,
    sampling: tuple[float, float],
    cartesian_basis: str | list[str] = "low_order",
    rotation_angle: float | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit aberrations to a measured ``psi(k)``, with a goodness-of-fit.

    A measured probe carries whatever phase the optic actually has. The parallax kernel needs
    that phase to be an aberration surface ``chi(k)``, so that each bright-field pixel sees the
    object displaced by ``grad chi / 2pi``. Whether it is one is a property of the optic rather
    than something to assume. A Fresnel zone plate is dominated by defocus and astigmatism,
    while an optic built to produce structured illumination has no smooth surface at all. The
    returned diagnostics say which case a probe is in.

    The fit runs on nearest-neighbour phase *differences* rather than on the unwrapped phase.
    Differences need no branch cuts, and they tolerate a pupil that a central stop or a
    bad-pixel mask has split into disconnected pieces.

    Parameters
    ----------
    fourier_probe : torch.Tensor
        ``(Qx, Qy)`` complex probe in the detector plane, corner-centered.
    bf_mask : torch.Tensor
        ``(Qx, Qy)`` boolean, the pixels to fit over.
    wavelength : float
        Angstrom.
    sampling : tuple of float
        Real-space sampling per axis in Angstrom, as :func:`spatial_frequencies` and
        :func:`fit_aberrations_from_shifts` take it. The detector step is
        ``1 / (gpts * sampling)``.
    cartesian_basis : str or list of str
        An :data:`ABERRATION_PRESETS` key or an explicit list of cartesian labels.
    rotation_angle : float, optional
        Degrees, applied to the k-grid the same way :func:`spatial_frequencies` applies it.

    Returns
    -------
    aberration_coefs : dict
        Polar coefficients, ready to pass as ``aberration_coefs``.
    diagnostics : dict
        ``phase_rms_rad_px`` and ``residual_rms_rad_px``, the weighted phase step per detector
        pixel before and after the fit; ``variance_explained``; and ``shift_error_ang``, the
        residual converted into the parallax shift error it causes. Parallax is worth using
        when that last one comes out below the canvas pixel size.
    """
    if isinstance(cartesian_basis, str):
        if cartesian_basis not in ABERRATION_PRESETS:
            raise ValueError(
                f"Unknown basis preset {cartesian_basis!r}. "
                f"Choose one of {sorted(ABERRATION_PRESETS)} or pass an explicit label list."
            )
        labels = list(ABERRATION_PRESETS[cartesian_basis])
    else:
        labels = list(cartesian_basis)

    # a linear ramp across the pupil only translates the object, so it is fitted and discarded
    nuisance = ["C01_a", "C01_b"]
    fit_labels = labels + nuisance

    probe = torch.as_tensor(fourier_probe)
    device = probe.device
    mask = torch.as_tensor(bf_mask, device=device).bool()
    gpts = (int(probe.shape[-2]), int(probe.shape[-1]))

    kxa, kya = spatial_frequencies(gpts, sampling, rotation_angle=rotation_angle, device=device)

    rows, obs, weights, scales = [], [], [], []
    # torch.roll keeps the pairing consistent with the corner-centered grid; pairs that would
    # straddle the Nyquist wrap are dropped with the mask, which never reaches that far
    for axis, axis_sampling in enumerate(sampling):
        shifted = [torch.roll(t, -1, dims=axis) for t in (probe, mask, kxa, kya)]
        probe_j, mask_j, kx_j, ky_j = shifted
        pair = mask & mask_j
        if not bool(pair.any()):
            continue

        k_i, phi_i = polar_coordinates(kxa[pair], kya[pair])
        k_j, phi_j = polar_coordinates(kx_j[pair], ky_j[pair])
        basis_i = aberration_surface_cartesian_basis(
            k_i * wavelength, phi_i, wavelength, fit_labels
        )
        basis_j = aberration_surface_cartesian_basis(
            k_j * wavelength, phi_j, wavelength, fit_labels
        )
        rows.append(basis_j - basis_i)

        # the probe is `aperture * exp(-1j * chi)`, so the phase step is minus the chi step
        obs.append(-torch.angle(probe_j[pair] * probe[pair].conj()))
        weights.append((probe[pair].abs() * probe_j[pair].abs()).sqrt())
        # radians per pixel along this axis -> Angstrom of parallax shift, using 1/dq
        scales.append(torch.full_like(weights[-1], gpts[axis] * axis_sampling / (2 * math.pi)))

    if not rows:
        raise ValueError("`bf_mask` contains no adjacent pairs to fit.")

    design = torch.cat(rows).cpu().to(torch.float64)
    target = torch.cat(obs).cpu().to(torch.float64)
    weight = torch.cat(weights).cpu().to(torch.float64)
    scale = torch.cat(scales).cpu().to(torch.float64)

    solution = torch.linalg.lstsq(
        design * weight[:, None], (target * weight)[:, None], rcond=None
    ).solution[:, 0]
    residual = target - design @ solution

    def _weighted_rms(values):
        return float(torch.sqrt((weight.square() * values.square()).sum() / weight.square().sum()))

    phase_rms = _weighted_rms(target)
    residual_rms = _weighted_rms(residual)
    diagnostics = {
        "phase_rms_rad_px": phase_rms,
        "residual_rms_rad_px": residual_rms,
        "variance_explained": 1.0 - (residual_rms / phase_rms) ** 2 if phase_rms > 0 else 0.0,
        "shift_error_ang": _weighted_rms(residual * scale),
    }

    cartesian = {
        label: solution[i].to(torch.float32)
        for i, label in enumerate(fit_labels)
        if label not in nuisance
    }
    polar = cartesian_to_polar_aberrations(cartesian)
    aberration_coefs = {k: float(v) for k, v in polar.items() if float(v) != 0.0}

    return aberration_coefs, diagnostics


def _torch_polar(m: torch.Tensor):
    U, S, Vh = torch.linalg.svd(m)
    u = U @ Vh
    p = Vh.T.conj() @ S.diag().to(dtype=m.dtype) @ Vh
    return u, p


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

    Exact, where a stencil is truncated -- which matters because the SSB and OBF kernels are
    never compact in real space (their transforms have ``r**-1.5`` tails). It is also asymptotically cheaper for a kernel that spans the canvas: one FFT per
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
    stencil_index: torch.Tensor | None = None,
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
    ``residual_loss`` have no meaning when the taps are kernel weights rather than a
    partition of unity.

    Parameters
    ----------
    values : torch.Tensor
        ``(B, T)`` values to deposit, or ``(E,)`` when ``stencil_index`` is given.
    coords : torch.Tensor
        ``(B, T, 2)`` canvas coordinates in pixels, ordered ``(row, col)``, or ``(E, 2)``
        when ``stencil_index`` is given.
    canvas_shape : tuple of int
        ``(n_rows, n_cols)`` of the output canvas.
    stencil_offsets : torch.Tensor
        ``(S, 2)`` integer pixel offsets of the stencil taps.
    stencil_weights : torch.Tensor
        ``(B, S)`` complex weight of each tap, per batch element.
    stencil_index : torch.Tensor, optional
        ``(E,)`` row of ``stencil_weights`` to use for each value, which turns the dense
        ``(B, T)`` grid into a flat list of ``E`` deposits. Sparse frames arrive in this form,
        with one entry per occupied (detector pixel, scan position) cell rather than one per
        cell of the full grid.
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

        if stencil_index is None:
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
            continue

        # A flat event list is short enough to broadcast the taps rather than loop them. Each
        # tap is one `index_add_` over comparatively few rows, so a loop over a few hundred of
        # them is dominated by launch overhead rather than by arithmetic.
        taps = int(stencil_offsets.shape[0])
        chunk = max(1, _MAX_TAP_ELEMENTS // max(taps, 1))
        for lo in range(0, int(values.shape[0]), chunk):
            window = slice(lo, lo + chunk)
            contribution = values[window, None] * stencil_weights[stencil_index[window]]
            if corner_weight is not None:
                contribution = contribution * corner_weight[window, None]

            flat_indices, contribution = _resolve_indices(
                base_row[window, None] + d_row + stencil_offsets[None, :, 0],
                base_col[window, None] + d_col + stencil_offsets[None, :, 1],
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
    intensity_threshold: float | None = None,
    normalization_order: int = 0,
    bf_mask=None,
    aperture=None,
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
        bf_mask = _bright_field_mask_of(
            shifted_tensor, dataset, intensity_threshold, aperture, rotation_angle
        )
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


@dataclass
class EventStack:
    """
    Detected electrons as a flat table, sorted by bright-field pixel.

    The sparse counterpart of the ``(N_bf, N_pos)`` vBF stack. That array holds one entry per
    (bright-field pixel, scan position) pair whatever was detected there, so its size follows
    the acquisition geometry rather than the dose. This holds one row per occupied cell
    instead, which is smaller whenever most cells are empty.

    Rows are sorted by ``bf_index`` so that a contiguous batch of bright-field pixels is a
    contiguous slice, which is the order ``reconstruct`` iterates the detector in.

    Attributes
    ----------
    position_index : torch.Tensor
        ``(E,)`` row of ``positions_px`` each electron was detected at.
    bf_index : torch.Tensor
        ``(E,)`` row of the vBF stack, equivalently which bright-field pixel it landed on.
    weights : torch.Tensor
        ``(E,)`` measured intensity of each cell. Ones unless a pixel was struck more than
        once, the ``Vector`` carried an intensity field, or a bilinear origin shift split rows
        across neighbouring detector pixels.
    bf_offsets : torch.Tensor
        ``(N_bf + 1,)`` start of each bright-field pixel's slice of the sorted table.
    mean_per_bf : torch.Tensor
        ``(N_bf,)`` scan-mean of each bright-field image, the quantity ``_preprocess``
        subtracts from a dense stack.
    counts_per_position : torch.Tensor
        ``(N_pos,)`` total intensity at each scan position, for ``subtract_frame_mean``.
    """

    position_index: "torch.Tensor"
    bf_index: "torch.Tensor"
    weights: "torch.Tensor"
    bf_offsets: "torch.Tensor"
    mean_per_bf: "torch.Tensor"
    counts_per_position: "torch.Tensor"
    num_bf: int
    num_positions: int

    def slice_for(self, bf_lo: int, bf_hi: int):
        """``(position_index, bf_index, weights)`` for bright-field pixels ``[bf_lo, bf_hi)``."""
        start = int(self.bf_offsets[bf_lo])
        stop = int(self.bf_offsets[bf_hi])
        window = slice(start, stop)
        return self.position_index[window], self.bf_index[window], self.weights[window]

    def to(self, device):
        """Move every table to ``device``, leaving the counts alone."""
        return EventStack(
            position_index=self.position_index.to(device),
            bf_index=self.bf_index.to(device),
            weights=self.weights.to(device),
            bf_offsets=self.bf_offsets.to(device),
            mean_per_bf=self.mean_per_bf.to(device),
            counts_per_position=self.counts_per_position.to(device),
            num_bf=self.num_bf,
            num_positions=self.num_positions,
        )


def vector_from_frames(frames, fields=("kx", "ky"), name="events", aggregate=True):
    """
    A ragged ``Vector`` of detected electrons from a stack of counted diffraction patterns.

    The inverse of histogramming an event list, for turning a simulated or already-counted
    acquisition into the form :meth:`DirectPtychographyMontage.from_vector` takes. A raster
    ``(Rx, Ry, Qx, Qy)`` stack keeps its scan grid, which pairs with
    :meth:`Dataset4dstem.probe_positions`. A flat ``(N, Qx, Qy)`` one gives ``(N,)`` cells.

    Parameters
    ----------
    frames : ndarray, torch.Tensor or Dataset
        ``(..., Qx, Qy)`` counts. Values need not be integers, but they are intensities of a
        detector pixel rather than a continuous signal, and zeros are dropped.
    fields : sequence of str
        Names for the two detector-coordinate columns, and optionally a third for the count.
        With two names each detected electron becomes its own row. With three the count rides
        in the third column instead, which is smaller wherever a pixel caught more than one.
    aggregate : bool
        Ignored unless three ``fields`` are given, where ``False`` would expand the counts
        anyway. Kept so the two forms can be compared.

    Returns
    -------
    Vector
        Fixed grid matching ``frames.shape[:-2]``, one cell per pattern.
    """
    from quantem.core.datastructures import Dataset, Vector

    if isinstance(frames, Dataset):
        frames = frames.array
    if hasattr(frames, "detach"):
        frames = frames.detach().cpu().numpy()
    frames = np.asarray(frames)
    if frames.ndim < 3:
        raise ValueError(f"`frames` must be (..., Qx, Qy), got shape {frames.shape}.")

    fields = list(fields)
    if len(fields) not in (2, 3):
        raise ValueError(f"`fields` must name two or three columns, got {fields}.")
    weighted = len(fields) == 3 and aggregate

    grid = frames.shape[:-2]
    flat = frames.reshape(-1, *frames.shape[-2:])

    cells = []
    for frame in flat:
        kx, ky = np.nonzero(frame)
        counts = frame[kx, ky]
        pixels = np.stack((kx, ky), axis=-1).astype(np.float64)
        if weighted:
            cells.append(np.concatenate((pixels, counts[:, None].astype(np.float64)), axis=-1))
        else:
            repeats = np.rint(counts).astype(np.int64)
            cells.append(np.repeat(pixels, repeats, axis=0))

    # `Vector.from_data` reads the fixed grid off the outer nesting, so rebuild it
    nested = cells
    for size in reversed(grid[1:]):
        nested = [nested[i : i + size] for i in range(0, len(nested), size)]

    return Vector.from_data(
        nested,
        fields=fields,
        units=["px", "px", "counts"][: len(fields)],
        name=name,
    )


def _event_table_from_vector(vector, weight_field: str | None):
    """``(position_index, coords, weights)`` for every row of a ragged Vector.

    A fixed grid of more than one dimension is flattened row-major, which is the order
    :meth:`Dataset4dstem.probe_positions` lists a raster scan in.
    """
    if tuple(vector.shape) == ():
        raise ValueError("`vector` must have a fixed grid of diffraction patterns, not a cell.")

    missing = [f for f in ("kx", "ky") if f not in vector.fields]
    if missing:
        raise ValueError(
            f"`vector` must carry 'kx' and 'ky' fields of detector pixel coordinates, missing "
            f"{missing}. Its fields are {vector.fields}."
        )

    coords = np.asarray(vector.select_fields("kx", "ky").flatten(), dtype=np.float64)
    row_counts = np.asarray(vector.row_counts(), dtype=np.int64)
    position_index = np.repeat(np.arange(row_counts.size, dtype=np.int64), row_counts)

    if weight_field is None:
        weights = np.ones(coords.shape[0], dtype=np.float64)
    else:
        if weight_field not in vector.fields:
            raise ValueError(
                f"`weight_field={weight_field!r}` is not a field of `vector`, whose fields are "
                f"{vector.fields}."
            )
        weights = np.asarray(
            vector.select_fields(weight_field).flatten(), dtype=np.float64
        ).reshape(-1)

    return position_index, coords, weights


def _measure_event_origins(position_index, coords, weights, num_positions, device):
    """Per-frame intensity-weighted center of mass, in detector pixels.

    The same quantity :meth:`CenterOfMassOriginModel.calculate_origin` takes as a moment of
    each diffraction pattern, computed here as a weighted mean over that frame's rows. The two
    agree exactly for events at integer pixel coordinates, and this one never builds the
    patterns.
    """
    totals = np.bincount(position_index, weights=weights, minlength=num_positions)
    # a frame with no counts has no measurable origin; leave it at the global one so the
    # plane fit sees a finite value rather than a nan that would poison the covariance
    safe = np.where(totals > 0, totals, 1.0)
    measured = np.stack(
        [
            np.bincount(position_index, weights=weights * coords[:, axis], minlength=num_positions)
            / safe
            for axis in (0, 1)
        ],
        axis=-1,
    )
    global_com = (weights[:, None] * coords).sum(0) / max(weights.sum(), 1e-12)
    measured[totals == 0] = global_com

    return torch.as_tensor(measured, dtype=torch.float, device=device)


def build_event_stack_from_vector(
    vector,
    positions,
    scan_sampling,
    gpts,
    reciprocal_sampling,
    detector_units=("A^-1", "A^-1"),
    device: str | int = "cpu",
    fit_method: str = "plane",
    mode: str = "nearest",
    force_measured_origin=None,
    force_fitted_origin=None,
    rotation_angle: float | None = None,
    intensity_threshold: float | None = None,
    weight_field: str | None = None,
    normalization_order: int = 0,
    bf_mask=None,
    aperture=None,
):
    """
    Origin-correct and mask a ragged event list into an :class:`EventStack`.

    The sparse counterpart of :func:`build_vbf_stack_from_dataset3d`, returning the same tuple
    with the vBF stack replaced by an event table. Position bookkeeping,
    ``scan_sampling="auto"`` inference and the bright-field threshold are shared with that
    function, so the two entry points cannot drift apart.

    Parameters
    ----------
    vector : Vector
        ``(N,)`` ragged cells, one per diffraction pattern, with fields ``"kx"`` and ``"ky"``
        holding detector pixel coordinates.
    positions : Dataset2d, torch.Tensor or ndarray
        ``(N, 2)`` probe positions in Angstrom.
    gpts : tuple of int
        ``(Qx, Qy)`` detector shape, which a ``Vector`` does not carry.
    reciprocal_sampling : tuple of float
        Detector pixel size, also absent from a ``Vector``. Units are ``detector_units``.
    detector_units : tuple of str
        ``("A^-1", "A^-1")`` or ``("mrad", "mrad")``.
    mode : {"nearest", "bilinear"}
        How the fitted origin shift is applied. ``"nearest"`` rounds each corrected coordinate
        onto the detector grid, leaving one integer-weighted row per pixel that was actually
        struck. ``"bilinear"`` instead splits each row across its four neighbours with the
        weights the dense resample uses, which reproduces that path to float precision.

        ``"nearest"`` is the default because a detection is a discrete event. Splitting it
        fractionally invents counts that were never measured and correlates the shot noise of
        neighbouring detector pixels, and it costs the sparsity that the event list exists for:
        on apoferritin at 4 mrad the occupancy goes from 41% to 87%. Use ``"bilinear"`` when
        the point is to reproduce ``from_dataset3d`` exactly on a sub-pixel origin.

    Returns
    -------
    events, positions_px, bf_mask_dataset, scan_gpts, scan_sampling, rotation_angle, scan_origin
    """
    from quantem.core.datastructures import Dataset2d
    from quantem.diffractive_imaging.origin_models import fit_origin_from_measured

    gpts = tuple(int(n) for n in gpts)
    if len(gpts) != 2:
        raise ValueError(f"`gpts` must be a (Qx, Qy) pair, got {gpts}.")
    if rotation_angle is None:
        raise ValueError(
            "`rotation_angle` must be given for an event list, in degrees: rotation is "
            "otherwise estimated from the curl of the center of mass over a 2D scan grid, "
            "which an ungridded acquisition lacks."
        )
    if mode not in ("nearest", "bilinear"):
        raise ValueError(f"`mode` must be 'nearest' or 'bilinear', got {mode!r}")
    if normalization_order != 0:
        raise ValueError(
            "`normalization_order=1` fits a 2D linear background per bright-field image "
            "and needs a scan grid, which an event list does not have; use "
            "`normalization_order=0`."
        )

    positions_ang = validate_probe_positions(positions)
    position_index, coords, weights = _event_table_from_vector(vector, weight_field)
    num_positions = int(positions_ang.shape[0])
    if int(np.prod(vector.shape)) != num_positions:
        raise ValueError(
            f"`positions` has {num_positions} rows but `vector` has "
            f"{int(np.prod(vector.shape))} diffraction patterns, over a fixed grid of "
            f"{tuple(vector.shape)}."
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

    # measure and fit the origin, mirroring `fit_and_shift_diffraction_origin`
    if force_fitted_origin is not None:
        origin_fitted = validate_tensor(
            force_fitted_origin, "fitted origin", dtype=torch.float
        ).view((-1, 2))
    else:
        if force_measured_origin is None:
            origin_measured = _measure_event_origins(
                position_index, coords, weights, num_positions, device
            )
        else:
            origin_measured = (
                validate_tensor(force_measured_origin, "measured origin", dtype=torch.float)
                .to(device)
                .view((-1, 2))
                .expand((num_positions, 2))
            )
        origin_fitted = fit_origin_from_measured(
            origin_measured,
            torch.as_tensor(positions_ang, dtype=torch.float, device=device),
            fit_method=fit_method,
            device=device,
        )
    origin_fitted = np.asarray(origin_fitted.detach().cpu().numpy(), dtype=np.float64).reshape(
        -1, 2
    )
    if origin_fitted.shape[0] == 1:
        origin_fitted = np.broadcast_to(origin_fitted, (num_positions, 2))

    # shift by moving the coordinates rather than resampling an image
    shifted = coords - origin_fitted[position_index]
    if mode == "nearest":
        detector_index = np.rint(shifted).astype(np.int64)
    else:
        base = np.floor(shifted)
        frac = shifted - base
        base = base.astype(np.int64)
        detector_index = np.concatenate(
            [base + np.array(corner, dtype=np.int64) for corner in _BILINEAR_CORNERS]
        )
        corner_weights = np.concatenate(
            [
                (frac[:, 0] if d_row else 1 - frac[:, 0])
                * (frac[:, 1] if d_col else 1 - frac[:, 1])
                for d_row, d_col in _BILINEAR_CORNERS
            ]
        )
        weights = np.tile(weights, len(_BILINEAR_CORNERS)) * corner_weights
        position_index = np.tile(position_index, len(_BILINEAR_CORNERS))

    # corner-center the detector, matching `shift_origin_to`
    detector_index = np.stack(
        [detector_index[:, 0] % gpts[0], detector_index[:, 1] % gpts[1]], axis=-1
    )

    mean_pattern = np.bincount(
        detector_index[:, 0] * gpts[1] + detector_index[:, 1],
        weights=weights,
        minlength=gpts[0] * gpts[1],
    ).reshape(gpts) / max(num_positions, 1)
    mean_pattern_t = torch.as_tensor(mean_pattern, dtype=torch.float, device=device)

    if bf_mask is None:
        amplitude = (
            None
            if aperture is None
            else aperture(gpts, reciprocal_sampling, detector_units, rotation_angle)
        )
        bf_mask = bright_field_mask(mean_pattern_t, intensity_threshold, amplitude)
    else:
        bf_mask = validate_tensor(bf_mask, "bf_mask", dtype=torch.bool).to(device)
        if tuple(bf_mask.shape) != gpts:
            raise ValueError(
                f"`bf_mask` has shape {tuple(bf_mask.shape)} but the detector is {gpts}."
            )

    # row-major enumeration of the true pixels, which is the order `torch.nonzero` gives and
    # therefore the order `vbf_index_mapping` indexes the stack in
    mask_flat = bf_mask.reshape(-1).cpu().numpy()
    lookup = np.full(mask_flat.size, -1, dtype=np.int64)
    lookup[mask_flat] = np.arange(int(mask_flat.sum()), dtype=np.int64)
    num_bf = int(mask_flat.sum())

    bf_index = lookup[detector_index[:, 0] * gpts[1] + detector_index[:, 1]]
    keep = bf_index >= 0
    bf_index = bf_index[keep]
    position_index = position_index[keep]
    weights = weights[keep]

    # Collapse rows that share a (bright-field pixel, position) cell, so a weight holds that
    # cell's measured intensity rather than one electron's contribution to it. Two electrons in
    # one cell are not the same as one electron of twice the weight wherever the value is
    # squared, which is what `residual_loss` accumulates. Sorting the key also sorts by
    # bright-field pixel, so a contiguous batch of the detector stays a contiguous slice.
    cell = bf_index * num_positions + position_index
    occupied, inverse = np.unique(cell, return_inverse=True)
    weights = np.bincount(inverse, weights=weights, minlength=occupied.size)
    bf_index = occupied // num_positions
    position_index = occupied % num_positions

    bf_offsets = np.concatenate(([0], np.cumsum(np.bincount(bf_index, minlength=num_bf)))).astype(
        np.int64
    )

    mean_per_bf = np.bincount(bf_index, weights=weights, minlength=num_bf) / max(num_positions, 1)
    if normalization_order == 0:
        # `normalize_vbf_stack` scales each bright-field image to unity mean over the scan,
        # which for an event list is a per-row division by that pixel's mean. A pixel inside
        # the mask that caught nothing has no scale, so its empty rows are left alone.
        scale = np.where(mean_per_bf > 0, mean_per_bf, 1.0)
        weights = weights / scale[bf_index]
        mean_per_bf = np.where(mean_per_bf > 0, 1.0, 0.0)
    elif normalization_order != 1:
        raise ValueError(f"`normalization_order` must be 0 or 1, got {normalization_order!r}")

    counts_per_position = np.bincount(position_index, weights=weights, minlength=num_positions)

    def _tensor(array, dtype):
        return torch.as_tensor(array, dtype=dtype, device=device)

    events = EventStack(
        position_index=_tensor(position_index, torch.int64),
        bf_index=_tensor(bf_index, torch.int64),
        weights=_tensor(weights, torch.float),
        bf_offsets=_tensor(bf_offsets, torch.int64),
        mean_per_bf=_tensor(mean_per_bf, torch.float),
        counts_per_position=_tensor(counts_per_position, torch.float),
        num_bf=num_bf,
        num_positions=num_positions,
    )

    bf_mask_dataset = Dataset2d.from_array(
        bf_mask.cpu().numpy(),
        name="BF mask",
        units=tuple(str(u) for u in detector_units),
        sampling=tuple(float(s) for s in reciprocal_sampling),
    )

    # anchor to the position bounding box, then convert to canvas pixels
    scan_origin = positions_ang.min(axis=0)
    positions_px = (positions_ang - scan_origin) / np.asarray(scan_sampling)
    scan_gpts = tuple(int(math.ceil(v)) + 1 for v in positions_px.max(axis=0))

    return (
        events,
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


#: The mean pattern's fraction of its peak below which ``bright_field_mask`` takes the mean
#: pattern's view instead of the model's, when no model aperture is available.
DEFAULT_INTENSITY_THRESHOLD = 0.5

#: A model-aperture pixel whose measured mean intensity is below this fraction of the mean
#: pattern's peak holds no data to normalize by, so the mask leaves it out. It guards against a
#: semiangle or origin calibrated slightly too large on experimental data; on simulated data
#: it removes only the outermost corner of the soft edge.
EMPTY_PIXEL_FRACTION = 1e-3


def model_aperture(
    gpts,
    reciprocal_sampling,
    detector_units,
    wavelength: float,
    semiangle_cutoff: float | None = None,
    fourier_probe=None,
    rotation_angle: float = 0.0,
    device="cpu",
):
    """``|psi(k)|`` of the probe model on a corner-centred detector grid, or ``None`` without one.

    The analytic probe's soft aperture when ``semiangle_cutoff`` is given, otherwise the
    ``fourier_probe``'s amplitude (summed over modes), read in the scan frame at the grid
    rotated by ``rotation_angle`` degrees, as the reconstructions read it.
    """
    if fourier_probe is None and semiangle_cutoff is None:
        return None
    dk = np.asarray(reciprocal_sampling, dtype=np.float64)
    if tuple(detector_units)[0] == "mrad":
        dk = dk * 1e-3 / wavelength
    sampling = tuple(1 / (n * d) for n, d in zip(gpts, dk))
    kx, ky = spatial_frequencies(
        tuple(int(n) for n in gpts),
        sampling,
        rotation_angle=_rotation_degrees_to_radians(rotation_angle),
        device=device,
    )
    if fourier_probe is not None:
        return fourier_probe.to(device).read(kx, ky).abs()
    from quantem.diffractive_imaging.complex_probe import evaluate_probe

    k, phi = polar_coordinates(kx, ky)
    angular = tuple(float(d) * wavelength * 1e3 for d in dk)
    return evaluate_probe(k * wavelength, phi, semiangle_cutoff, angular, wavelength).abs()


def bright_field_mask(mean_pattern, intensity_threshold: float | None = None, aperture=None):
    """The bright-field pixels: those the model aperture reaches, or above a threshold.

    With ``intensity_threshold=None`` and a model ``aperture`` (``|psi(k)|``, as
    :func:`model_aperture` returns it), the mask is every pixel the aperture reaches, its soft
    edge included, less any that measured nothing (:data:`EMPTY_PIXEL_FRACTION`). That holds
    the whole disk whatever the specimen scatters. A threshold on the mean pattern cannot: at
    half its peak it cuts the soft edge, which WDD's convolution model needs and the double
    overlaps that separate amplitude from phase sit on, and at 0.05 it admitted 350-380
    dark-field pixels around thick SrTiO3 and gold nanoparticles, reaching 1.4 and 1.2 alpha.

    With ``intensity_threshold`` given, or no model to read, the mask is the mean pattern above
    that fraction of its peak (:data:`DEFAULT_INTENSITY_THRESHOLD` without a model).
    """
    if intensity_threshold is None and aperture is None:
        intensity_threshold = DEFAULT_INTENSITY_THRESHOLD
    if intensity_threshold is not None:
        return mean_pattern > mean_pattern.max() * intensity_threshold
    aperture = aperture.to(mean_pattern.device)
    reached = aperture > OVERLAP_FLOOR * aperture.max()
    return reached & (mean_pattern > EMPTY_PIXEL_FRACTION * mean_pattern.max())


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


def _bright_field_mask_of(shifted_tensor, dataset, intensity_threshold, aperture, rotation_angle):
    """:func:`bright_field_mask` of an origin-corrected dense stack, on ``dataset``'s detector."""
    scan_dims = tuple(range(shifted_tensor.ndim - 2))
    mean_pattern = shifted_tensor.mean(dim=scan_dims)
    amplitude = None
    if aperture is not None:
        amplitude = aperture(
            tuple(int(n) for n in shifted_tensor.shape[-2:]),
            tuple(float(s) for s in dataset.sampling[-2:]),
            tuple(dataset.units[-2:]),
            rotation_angle,
        )
    return bright_field_mask(mean_pattern, intensity_threshold, amplitude)


def build_vbf_stack_from_dataset4d(
    dataset,
    device: str | int = "cpu",
    max_batch_size: int | None = None,
    fit_method: str = "plane",
    mode: str = "bilinear",
    force_measured_origin=None,
    force_fitted_origin=None,
    rotation_angle: float | None = None,
    intensity_threshold: float | None = None,
    normalization_order: int = 0,
    edge_blend_pixels: int = 0,
    aperture=None,
):
    """
    Turn a 4D-STEM dataset into the virtual bright-field stack the direct-ptychography
    classes consume.

    The bright-field mask is every pixel the probe model reaches when ``aperture`` is given
    and ``intensity_threshold`` is not, and the mean pattern above ``intensity_threshold`` of
    its peak otherwise; see :func:`bright_field_mask`. ``aperture`` is a callable
    ``(gpts, reciprocal_sampling, detector_units, rotation_angle) -> |psi(k)|``, such as a
    partial of :func:`model_aperture`, evaluated once the rotation is known.

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

    bf_mask = _bright_field_mask_of(
        shifted_tensor, dataset, intensity_threshold, aperture, rotation_angle
    )
    bf_mask_dataset = Dataset2d.from_array(
        bf_mask.cpu().numpy(),
        name="BF mask",
        units=dataset.units[-2:],
        sampling=dataset.sampling[-2:],
    )

    # vbf_stack
    vbf_stack = shifted_tensor[..., bf_mask].cpu()
    gpts = vbf_stack.shape[:2]
    means = vbf_stack.mean((0, 1))
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
    if normalization_order == 0:
        # what the normalization divided out, which WDD needs back: it deconvolves the
        # intensities themselves, whose mean pattern a strong object reshapes
        vbf_dataset.metadata["bright_field_means"] = means.numpy()

    return vbf_dataset, bf_mask_dataset, rotation_angle


# ---------------------------------------------------------------------------
# per-frequency normal equations: complex-valued and multislice reconstructions
# ---------------------------------------------------------------------------

#: values of ``obj_type`` the direct reconstructions accept
OBJ_TYPES = ("pure_phase", "complex")


def slice_depths(num_slices: int, slice_thicknesses=None, device="cpu") -> "torch.Tensor":
    """Depths of ``num_slices`` thin slices in Angstrom, downstream positive, centred on C10.

    ``slice_thicknesses`` are the ``num_slices - 1`` gaps between consecutive slices, as in the
    multislice object models. A single value is used for every gap. The stack is centred on
    the plane the aberration coefficients refer to, so one slice sits exactly there and a
    ``C10`` fitted with one slice carries over to several.

    A tensor that requires grad keeps it, which is how a gradient fit reaches the thicknesses.
    """
    num_slices = int(num_slices)
    if num_slices < 1:
        raise ValueError(f"`num_slices` must be at least 1, got {num_slices}")
    if num_slices == 1:
        if slice_thicknesses is not None and torch.as_tensor(slice_thicknesses).numel() > 1:
            raise ValueError("`slice_thicknesses` has no meaning for a single slice")
        return torch.zeros(1, dtype=torch.float32, device=device)
    if slice_thicknesses is None:
        raise ValueError(f"num_slices = {num_slices}, so `slice_thicknesses` cannot be None")

    gaps = torch.as_tensor(slice_thicknesses, dtype=torch.float32, device=device).reshape(-1)
    if gaps.numel() == 1:
        gaps = gaps.expand(num_slices - 1)
    if gaps.numel() != num_slices - 1:
        raise ValueError(
            f"`slice_thicknesses` must hold one value or num_slices - 1 = {num_slices - 1}, "
            f"got {gaps.numel()}"
        )
    if bool((gaps.detach() <= 0).any()):
        raise ValueError(f"`slice_thicknesses` must be positive, got {gaps.detach().tolist()}")

    positions = torch.cat([gaps.new_zeros(1), torch.cumsum(gaps, 0)])
    return positions - positions.mean()


def depth_phase_factors(qxa, qya, kx, ky, wavelength: float, depths):
    """``(e_minus, e_plus)``, each ``(N, B, H, W)``, or ``None`` when every depth is zero.

    A thin slice ``z`` downstream of the reference plane sees the probe at ``C10 + z``, which
    multiplies ``t_minus`` by ``exp(i pi lambda z (|k|^2 - |q - k|^2))`` and ``t_plus`` by
    ``exp(i pi lambda z (|q + k|^2 - |k|^2))``. Applying the free-space propagator this way
    rather than through the aberration surface works for an empirical probe too.
    """
    depths = torch.as_tensor(depths, device=qxa.device)
    if not depths.requires_grad and bool((depths == 0).all()):
        return None
    qx, qy = qxa.unsqueeze(0), qya.unsqueeze(0)
    k2 = kx.square() + ky.square()
    d_minus = k2 - ((qx - kx).square() + (qy - ky).square())
    d_plus = ((qx + kx).square() + (qy + ky).square()) - k2
    z = depths.to(d_minus.dtype).view(-1, 1, 1, 1) * (math.pi * wavelength)
    return torch.exp(1j * z * d_minus.unsqueeze(0)), torch.exp(1j * z * d_plus.unsqueeze(0))


def alias_tiles(array, factor: int):
    """``(..., U H, U W, K)`` on an upsampled grid to ``(..., H, W, U^2 K)`` on the scan's.

    A scan of ``H x W`` positions samples the object's spectrum only modulo its own grid, so
    each of its frequencies holds the sum of ``U^2`` frequencies of the upsampled grid, its
    alias tiles. Column ``tile * K + channel`` of the result is that tile's ``channel``, with
    tiles in row-major order of the upsampled grid's blocks. :func:`from_alias_tiles` inverts it.
    """
    *lead, rows, cols, k = array.shape
    h, w = rows // factor, cols // factor
    n = len(lead)
    tiled = array.reshape(*lead, factor, h, factor, w, k)
    order = list(range(n)) + [n + 1, n + 3, n, n + 2, n + 4]
    return tiled.permute(order).reshape(*lead, h, w, factor * factor * k)


def from_alias_tiles(array, factor: int, channels: int):
    """``(..., H, W, U^2 K)`` alias-tile columns back to ``(..., U H, U W, K)``."""
    *lead, h, w, _ = array.shape
    n = len(lead)
    tiled = array.reshape(*lead, h, w, factor, factor, channels)
    order = list(range(n)) + [n + 2, n, n + 3, n + 1, n + 4]
    return tiled.permute(order).reshape(*lead, factor * h, factor * w, channels)


def woa_columns(t_minus, t_plus, depth_factors=None, complex_obj: bool = False):
    """``(B, H, W, K)`` weak-object columns: ``Gamma-`` per slice, then ``Gamma+`` per slice.

    The data model is ``G = sum_n Gamma+_n Psi_a,n + Gamma-_n i Psi_phi,n``, so ``K`` is the
    number of slices for a phase object and twice that for a complex one.
    """
    if depth_factors is None:
        minus, plus = t_minus.unsqueeze(0), t_plus.unsqueeze(0)
    else:
        e_minus, e_plus = depth_factors
        minus, plus = t_minus.unsqueeze(0) * e_minus, t_plus.unsqueeze(0) * e_plus
    columns = [minus - plus]
    if complex_obj:
        columns.append(minus + plus)
    return torch.cat(columns, 0).permute(1, 2, 3, 0)


def accumulate_normal_equations(columns, vbf_fourier, intensity=None):
    """``(gram, numerator)`` summed over the bright-field batch: ``(H, W, K, K)`` and ``(H, W, K)``.

    Each bright-field image is normalized by its mean, which is ``|psi(k)|^2`` for a weak
    object, so the data at pixel ``k`` are ``c_k x / |psi(k)|^2`` rather than ``c_k x``, and
    their Poisson variance goes as ``1 / |psi(k)|^2``. Passing that ``intensity`` (``(B, 1, 1)``)
    weights the Gram matrix by ``1 / |psi(k)|^2``, which makes the estimate unbiased and its
    noise normalization exact. The numerator needs no weight. Where the aperture is flat the
    weight is one; it matters at the soft edge, which is where the double overlaps that
    separate amplitude from phase at low ``q`` sit, so without it the two channels leak about
    1% into each other there.
    """
    numerator = torch.einsum("bhwi,bhw->hwi", columns.conj(), vbf_fourier)
    return weighted_gram(columns, intensity), numerator


def weighted_gram(columns, intensity=None):
    """``sum_k c_k^H c_k / |psi(k)|^2`` over the batch, ``(H, W, K, K)``.

    Without ``intensity`` it is the unweighted ``sum_k c_k^H c_k``.
    """
    if intensity is not None:
        nonzero = intensity > 0
        weight = torch.where(
            nonzero, torch.rsqrt(torch.where(nonzero, intensity, torch.ones_like(intensity))), 0
        )
        columns = columns * weight[..., None]
    return torch.einsum("bhwi,bhwj->hwij", columns.conj(), columns)


def double_overlap_mask(psi_k, psi_minus, psi_plus, floor: float):
    """Pixels inside the aperture that overlap exactly one of its two displaced copies.

    On these double-overlap (sideband) pixels ``Gamma+ = +-Gamma-``, so amplitude and phase
    separate exactly for any aberration.
    """
    inside = psi_k.abs() > floor
    return inside & ((psi_minus.abs() > floor) ^ (psi_plus.abs() > floor))


def _solver_device(tensor) -> "torch.device":
    """Where the small per-frequency solves run: float64 needs the CPU on MPS, which has none."""
    return torch.device("cpu") if tensor.device.type == "mps" else tensor.device


def _to(tensor, device, dtype):
    """``tensor.to(device, dtype)`` in the order MPS needs.

    MPS has no float64 or complex128, and moving a complex64 tensor off it and widening it in
    one call returns NaN for some elements (torch 2.13). A tensor leaving MPS therefore moves
    first and changes precision second, and any other tensor changes precision first.
    """
    device = torch.device(device)
    if tensor.device.type == "mps" and device.type != "mps":
        return tensor.to(device).to(dtype)
    return tensor.to(dtype).to(device)


def _pseudo_inverse(gram, eigenvalue_floor: float):
    """Per-frequency regularized inverse of a Hermitian ``(..., K, K)`` Gram matrix.

    The matrix is first scaled to unit diagonal, so its eigenvalues measure how parallel the
    columns are rather than how large: at low ``q`` the phase column is orders of magnitude
    weaker than the amplitude one, and a regularizer on the raw matrix would suppress it for
    that alone. A ridge of ``eigenvalue_floor`` is then added before inverting. Directions
    whose eigenvalue falls well below it, columns too parallel to tell apart in float32 data,
    are suppressed rather than amplified, and everywhere else the inverse is exact to that
    order. Unlike a hard eigenvalue cut this stays differentiable when eigenvalues coincide,
    which they do wherever a frequency carries no signal at all.
    """
    if gram.shape[-1] == 1:
        g = gram[..., 0, 0].real
        nonzero = g > 0
        safe = torch.where(nonzero, g, torch.ones_like(g))
        return torch.where(nonzero, 1 / safe, torch.zeros_like(g))[..., None, None].to(gram.dtype)
    diagonal = torch.diagonal(gram, dim1=-2, dim2=-1).real
    nonzero = diagonal > 0
    scale = torch.where(
        nonzero, torch.rsqrt(torch.where(nonzero, diagonal, torch.ones_like(diagonal))), 0
    )
    outer = (scale[..., :, None] * scale[..., None, :]).to(gram.dtype)
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    inverse = torch.linalg.inv(gram * outer + eigenvalue_floor * eye)
    return inverse * outer


#: An overlap below this fraction of the probe's peak intensity, or a column power below its
#: square times the column's peak power, is round-off rather than signal. An analytic aperture
#: is exactly zero outside itself; a sampled probe leaves values near 1e-7 there, which a
#: normalization would otherwise divide by. Upsampling past the band is where that happens.
OVERLAP_FLOOR = 1e-6


def unit_overlap(gamma, peak_intensity: float = 1.0):
    """``gamma* / |gamma|``, and zero where ``|gamma|`` is round-off (see :data:`OVERLAP_FLOOR`)."""
    magnitude = gamma.abs()
    live = magnitude > OVERLAP_FLOOR * peak_intensity
    safe = torch.where(live, magnitude, torch.ones_like(magnitude))
    return torch.where(live, gamma.conj() / safe, torch.zeros_like(gamma))


#: ridge added to the unit-diagonal Gram matrix before inverting. Float32 rounding sits below
#: about 1e-6; two slices 3.35 A apart at 300 kV and 20 mrad still separate with 1e-4 and fail
#: at 1e-3, so this keeps a decade on either side.
EIGENVALUE_FLOOR = 1e-5


def normal_equation_operator(
    gram,
    bf_weights,
    num_phase: int,
    eigenvalue_floor: float = EIGENVALUE_FLOOR,
    amplitude_factor: float = 0.5,
    channels=None,
):
    """``A``, ``(..., K, K)``, with which every channel's Fourier coefficients are ``A @ b``.

    ``b = sum_k c_k^* G_k`` is the numerator of the normal equations, so the reconstruction is
    linear in the data, and a per-pixel kernel ``sum_m A_nm c_{k,m}^*`` gives the same result
    pixel by pixel. That is what lets the montage truncate it to a real-space stencil. See
    :func:`solve_normal_equations` for the gains.
    """
    gain, factors, pinv = _channel_scales(
        gram, bf_weights, num_phase, eigenvalue_floor, amplitude_factor, channels
    )
    return (gain * factors)[..., :, None] * pinv


def _channel_scales(
    gram,
    bf_weights,
    num_phase: int,
    eigenvalue_floor: float,
    amplitude_factor: float,
    channels=None,
):
    """``(gain, factors, M^+)``: each channel's OBF gain ``(..., K)``, its constant factor
    ``(K,)`` (``-i`` on the phase channels) and the regularized inverse, in complex128."""
    solver = _solver_device(gram)
    gram = _to(gram, solver, torch.complex128)
    pinv = _pseudo_inverse(gram, eigenvalue_floor)
    d = torch.diagonal(pinv, dim1=-2, dim2=-1).real.clamp_min(0)
    if gram.shape[-1] == 1:
        # one column: the inverse is exact, so the transfer is one
        transfer = torch.ones_like(d)
    else:
        # with a ridge, x = M^+ b has transfer diag(M^+ M)
        transfer = torch.diagonal(pinv @ gram, dim1=-2, dim2=-1).real.clamp_min(0)

    # Whitening with the exact variance would undo the ridge along every direction it
    # shrinks, amplifying again the combinations float32 data cannot tell apart. (M^+)_nn
    # keeps them suppressed; one scale per channel then undoes the ridge's overall
    # shrinkage, so that a large ridge tends to the OBF sweep rather than a dimmed copy.
    safe = torch.where(d > 0, d, torch.ones_like(d))
    weight = float(torch.as_tensor(bf_weights).detach())  # |psi| does not depend on chi
    level = transfer.reshape(-1, transfer.shape[-1]).amax(0)
    level = torch.where(level > 0, level, torch.ones_like(level))
    gain = torch.where(d > 0, torch.rsqrt(safe * weight * level), torch.zeros_like(d))
    # frequencies whose columns hold only round-off carry no signal, whatever the gain says
    power = torch.diagonal(gram, dim1=-2, dim2=-1).real
    peak = power.reshape(-1, power.shape[-1]).amax(0)
    gain = torch.where(power > OVERLAP_FLOOR**2 * peak, gain, torch.zeros_like(gain))

    factors = column_factors(gram.shape[-1], num_phase, amplitude_factor, gram.device, channels)
    return gain, factors, pinv


def column_factors(
    num_columns: int, num_phase: int, amplitude_factor=1.0, device="cpu", channels=None
):
    """``(K,)``: ``-i`` on the phase columns, whose unknown carries a factor ``i``, and
    ``amplitude_factor`` on the amplitude ones. With ``channels`` per alias tile (see
    :func:`alias_tiles`) the pattern repeats tile by tile."""
    index = torch.arange(num_columns, device=device) % (channels or num_columns)
    phase = torch.full((num_columns,), -1j, dtype=torch.complex128, device=device)
    return torch.where(index < num_phase, phase, torch.full_like(phase, amplitude_factor))


def solve_normal_equations(
    gram,
    numerator,
    bf_weights,
    num_phase: int,
    eigenvalue_floor: float = EIGENVALUE_FLOOR,
    amplitude_factor: float = 0.5,
    prior=None,
    noise_variance=None,
    weighted_gram=None,
    channels=None,
):
    """Fourier coefficients ``(H, W, K)`` of every channel, from the per-frequency normal equations.

    The least-squares estimate ``x = M^+ b`` separates the channels exactly wherever the
    columns are independent, with ``M^+`` the inverse of :func:`_pseudo_inverse` after a ridge
    of ``eigenvalue_floor`` on the unit-diagonal Gram matrix. It is the only estimate here with
    a transfer of one, values in radians and nepers, but its noise grows without bound where a
    column's power vanishes. Each channel is therefore scaled to optimum bright field: divided
    by its noise standard deviation, ``x_n / sqrt((M^+)_nn W)``, so the noise is white at the
    default ridge and the transfer follows the signal-to-noise ratio. For one column this is
    ``b / sqrt(M W)``, the classic optimum bright field.

    The ridge enters the projection: it shrinks the combinations of columns the data barely
    tell apart, trading their separation for noise. As it grows, every channel tends to its
    own column's projection alone, a sweep of independent planes for several slices. One scale
    per channel, the largest transfer ``(M^+ M)_nn`` over frequencies, undoes its overall
    shrinkage, so the gain tends to that sweep's OBF rather than a dimmed copy.

    The first ``num_phase`` channels are phase columns, whose unknown carries a factor ``i``,
    so they are multiplied by ``-i``. The amplitude channels are halved, since ``Gamma+``
    reaches ``2 |psi|^2`` as ``q -> 0`` where ``Gamma-`` reaches ``|psi|^2`` on a sideband,
    which gives both channels the same transfer there.

    With ``prior="ard"`` the least-squares estimate is replaced by :func:`ard_estimate`'s
    posterior mean, from ``weighted_gram`` (the inverse-variance Gram matrix, which for one
    column differs from ``gram`` where the aperture is not flat) and ``noise_variance``. It then
    takes the same gain as least squares, so the images compare directly.

    ``channels`` is the number of channels per alias tile when the columns hold several tiles
    side by side (:func:`alias_tiles`); ``num_phase`` then counts the phase channels per tile.
    """
    device, out_dtype = numerator.device, numerator.dtype
    if prior is None:
        operator = normal_equation_operator(
            gram, bf_weights, num_phase, eigenvalue_floor, amplitude_factor, channels
        )
        numerator = _to(numerator, operator.device, torch.complex128)
        return _to(torch.einsum("...ij,...j->...i", operator, numerator), device, out_dtype)
    if prior != "ard":
        raise ValueError(f"`prior` must be None or 'ard', got {prior!r}")
    gain, factors, _ = _channel_scales(
        gram, bf_weights, num_phase, eigenvalue_floor, amplitude_factor, channels
    )
    estimate = ard_estimate(
        gram if weighted_gram is None else weighted_gram,
        numerator,
        noise_variance,
        eigenvalue_floor=eigenvalue_floor,
    ).to(gain.device)
    return _to(gain * factors * estimate, device, out_dtype)


#: Expectation-maximization steps of :func:`ard_estimate`. On two twisted layers at 1e4 and 1e5
#: electrons per pattern the slices' correlation with their layers changed by less than 0.001
#: between 10 and 100.
ARD_ITERATIONS = 30


def ard_estimate(
    gram,
    numerator,
    noise_variance: float,
    iterations: int = ARD_ITERATIONS,
    eigenvalue_floor: float = EIGENVALUE_FLOOR,
):
    """Posterior mean ``(..., K)`` of the normal equations under automatic relevance determination.

    Every coefficient, each channel at each frequency, gets a zero-mean Gaussian prior with its
    own variance ``gamma``, and the variances are re-estimated from the data by
    expectation-maximization (sparse Bayesian learning):

        ``Sigma = (M / sigma^2 + diag(1 / gamma))^-1``,  ``mu = Sigma b / sigma^2``,
        ``gamma <- |mu|^2 + Sigma_nn``.

    A coefficient the data do not support has its variance driven to zero and drops out, so a
    frequency where only one of several slices carries signal is fitted with that slice's
    column alone, at the noise of a single plane rather than the ``1 / (1 - rho^2)`` of
    separating near-parallel columns. Where several coefficients carry signal their variances
    stay large and the estimate is least squares. With one column it keeps the frequencies whose
    signal stands above the noise and suppresses the rest, a data-driven Fourier filter.

    It starts from the least-squares estimate and its variance. ``noise_variance`` is the
    variance of the normalized data per unit weight, for which ``b`` has covariance
    ``sigma^2 M``; :func:`noise_variance_from_residual` estimates it. The result is nonlinear
    in the data, so it has no per-pixel kernel, and it is computed without gradients.
    """
    with torch.no_grad():
        if gram.dim() == numerator.dim():
            gram = gram[..., None, None]
            numerator = numerator[..., None]
        solver = _solver_device(gram)
        gram = _to(gram.detach(), solver, torch.complex128)
        numerator = _to(numerator.detach(), solver, torch.complex128)
        sigma2 = float(noise_variance)
        if not sigma2 > 0:
            raise ValueError(f"`noise_variance` must be positive, got {noise_variance!r}")

        pinv = _pseudo_inverse(gram, eigenvalue_floor)
        mean = torch.einsum("...ij,...j->...i", pinv, numerator)
        gamma = mean.abs().square() + sigma2 * torch.diagonal(pinv, dim1=-2, dim2=-1).real
        floor = 1e-30 * float(gamma.max()) if gamma.numel() and float(gamma.max()) > 0 else 1e-300
        for _ in range(int(iterations)):
            mean, covariance = _gaussian_posterior(gram, numerator, sigma2, gamma, floor)
            gamma = mean.abs().square() + torch.diagonal(covariance, dim1=-2, dim2=-1).real
        return mean


def _gaussian_posterior(gram, numerator, sigma2: float, gamma, floor: float):
    """``(mean, covariance)`` of ``x`` given ``b ~ N(M x, sigma^2 M)`` and ``x ~ N(0, diag(gamma))``.

    A prior variance at or below ``floor`` is a coefficient fixed at zero.
    """
    prior_precision = torch.diag_embed((1 / gamma.clamp_min(floor)).to(gram.dtype))
    covariance = _hermitian_inverse(gram / sigma2 + prior_precision)
    return torch.einsum("...ij,...j->...i", covariance, numerator / sigma2), covariance


#: How far a ring's power must stand above the noise, in standard errors of the noise power, to
#: enter :func:`wiener_estimate`'s prior. With some sixty rings per channel, 3 let one through by
#: chance often enough to matter where a finite source is divided out: the noise there is large
#: in the object's units, so a chance ring shows as ringing. On one WSe2 layer with a 0.5 A
#: source at 1e4 electrons per pattern, before the rings were pooled, four noise seeds gave
#: correlations of 0.62-0.90 at 3 and 0.895-0.898 at 4; across twelve other cases 4 cost at
#: most 0.003.
WIENER_SIGNIFICANCE = 4.0


def frequency_rings(qxa, qya):
    """``(H, W)`` ring index ``round(|q| / dq)``, ``dq`` the finer of the grid's two steps."""
    step = min(float(qxa[1, 0].abs()), float(qya[0, 1].abs()))
    return torch.round(torch.hypot(qxa, qya) / step).long()


def wiener_estimate(
    gram,
    numerator,
    noise_variance: float,
    rings,
    significance: float = WIENER_SIGNIFICANCE,
    eigenvalue_floor: float = EIGENVALUE_FLOOR,
    channels=None,
):
    """Posterior mean ``(..., K)`` of the normal equations under the data's own radial spectrum.

    Each channel gets a zero-mean Gaussian prior whose variance depends on ``|q|`` alone, the
    power the data hold in that ring beyond the noise: the ring's mean ``|x_n|^2`` of the
    least-squares estimate less its mean variance ``sigma^2 (M^+)_nn``. A ring enters only if
    that excess exceeds ``significance`` standard errors of the noise power; elsewhere the
    channel is fixed at zero. The posterior mean is then

        ``(M / sigma^2 + diag(1 / P(|q|)))^-1 b / sigma^2``,

    the Wiener filter of the least-squares estimate for one column, ``x P / (P + sigma^2 / M)``,
    and its multichannel form otherwise. Unlike the OBF gain it estimates the object itself: its
    transfer is ``P / (P + noise)``, near one wherever the signal stands well above the noise,
    and a factor in the columns that depends on ``q`` alone, such as a finite source, is divided
    back out. It has no free parameter but the gate.

    ``rings`` is ``(...)`` integer ring indices, from :func:`frequency_rings`, or ``(..., K)``
    with one per column, and frequencies in ring 0 are left out. With ``channels`` per alias
    tile (:func:`alias_tiles`), the columns of one channel in every tile share its spectrum.
    ``noise_variance`` is as for :func:`ard_estimate`. Computed without gradients.
    """
    with torch.no_grad():
        if gram.dim() == numerator.dim():
            gram = gram[..., None, None]
            numerator = numerator[..., None]
        solver = _solver_device(gram)
        gram = _to(gram.detach(), solver, torch.complex128)
        numerator = _to(numerator.detach(), solver, torch.complex128)
        rings = rings.to(solver)
        sigma2 = float(noise_variance)
        if not sigma2 > 0:
            raise ValueError(f"`noise_variance` must be positive, got {noise_variance!r}")

        pinv = _pseudo_inverse(gram, eigenvalue_floor)
        estimate = torch.einsum("...ij,...j->...i", pinv, numerator)
        variance = sigma2 * torch.diagonal(pinv, dim1=-2, dim2=-1).real
        num_columns = gram.shape[-1]
        if rings.dim() == gram.dim() - 2:
            rings = rings[..., None].expand(*rings.shape, num_columns)
        power = torch.diagonal(gram, dim1=-2, dim2=-1).real
        peak = power.reshape(-1, num_columns).amax(0)
        live = (power > OVERLAP_FLOOR**2 * peak) & (rings > 0)
        channel = torch.arange(num_columns, device=solver) % (channels or num_columns)

        num_rings = int(rings.max()) + 1
        prior = torch.zeros_like(variance)
        for n in range(channels or num_columns):
            selected = live & (channel == n)
            ring_prior = _ring_signal_power(
                estimate[selected].abs().square(),
                variance[selected],
                rings[selected],
                num_rings,
                significance,
            )
            prior = torch.where(selected, ring_prior[rings], prior)

        floor = 1e-30 * float(prior.max()) if float(prior.max()) > 0 else 1e-300
        mean, _ = _gaussian_posterior(gram, numerator, sigma2, prior, floor)
        return torch.where(prior > floor, mean, torch.zeros_like(mean))


#: Rings are pooled from low ``|q|`` outwards until each bin holds at least this many
#: frequencies, before :func:`wiener_estimate` estimates their power.
WIENER_MINIMUM_COUNT = 64

#: Standard errors of the estimated ring power that :func:`wiener_estimate` subtracts from it.
WIENER_MARGIN = 2.0


def _ring_signal_power(
    power,
    variance,
    index,
    num_rings: int,
    significance: float,
    iterations=20,
    minimum_count=WIENER_MINIMUM_COUNT,
    margin=WIENER_MARGIN,
):
    """``(num_rings,)`` signal power per ring from noisy coefficients, zero where not significant.

    Each coefficient's power is ``P + v`` in expectation, ``v`` its own noise variance. Within a
    ring ``v`` can span orders of magnitude, near the band edge where a column's power falls to
    zero, so the coefficients are weighted by ``1 / (P + v)^2``, iterated to the maximum-likelihood
    ``P``, rather than averaged: a plain mean lets the noisiest few decide. A ring is kept when
    its excess at ``P = 0`` exceeds ``significance`` standard errors, ``1 / sqrt(sum 1 / v^2)``,
    which for equal variances is the plain mean's ``v / sqrt(n)``.

    The lowest rings hold a handful of frequencies each, where the noise of a weak column is
    largest, so a power estimated from them alone is itself very noisy, and an overestimate
    passes that noise into the image as a slow variation across the field. The rings are
    therefore pooled from low ``|q|`` until each bin holds ``minimum_count`` frequencies, and the
    power used is the estimate less ``margin`` of its standard errors. On two random-atom layers 7
    A apart at 1e4 electrons per pattern this raised the slices' correlation with their layers,
    counting every frequency, from 0.61 to 0.83, where OBF reached 0.76; one layer was unchanged.
    """
    count = torch.bincount(index, minlength=num_rings)
    bins = torch.zeros(num_rings, dtype=torch.long)
    label = filled = 0
    for ring, n in enumerate(count.tolist()):
        bins[ring] = label
        filled += n
        if filled >= minimum_count:
            label, filled = label + 1, 0
    bins = bins.to(index.device)
    pooled = bins[index]
    num_bins = int(bins.max()) + 1
    pooled_count = torch.bincount(pooled, minlength=num_bins)

    def weighted_excess(weights):
        total = torch.bincount(pooled, weights, minlength=num_bins)
        excess = torch.bincount(pooled, weights * (power - variance), minlength=num_bins)
        return excess / total.clamp_min(torch.finfo(total.dtype).tiny), total

    null_excess, information = weighted_excess(variance.square().reciprocal())
    keep = (pooled_count >= 4) & (null_excess * information.sqrt() > significance)
    signal = null_excess.clamp_min(0)
    for _ in range(iterations):
        signal, _ = weighted_excess((signal[pooled] + variance).square().reciprocal())
        signal = signal.clamp_min(0)
    _, information = weighted_excess((signal[pooled] + variance).square().reciprocal())
    signal = signal - margin * information.clamp_min(torch.finfo(information.dtype).tiny).rsqrt()
    signal = torch.where(keep, signal.clamp_min(0), torch.zeros_like(signal))
    return signal[bins]


def _hermitian_inverse(matrix):
    """Inverse of Hermitian positive-definite ``(..., K, K)`` matrices, scaled to unit diagonal
    first so that a pruned coefficient's huge prior precision does not swamp the others."""
    diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1).real
    scale = torch.rsqrt(diagonal.clamp_min(torch.finfo(diagonal.dtype).tiny))
    outer = (scale[..., :, None] * scale[..., None, :]).to(matrix.dtype)
    return torch.linalg.inv(matrix * outer) * outer


def residual_per_degree_of_freedom(
    gram, numerator, weighted_energy, num_bf: int, eigenvalue_floor=EIGENVALUE_FLOOR
):
    """``(...)``: each frequency's least-squares residual over its ``num_bf - K`` degrees of
    freedom, which is the noise variance per unit weight wherever the model holds."""
    with torch.no_grad():
        if gram.dim() == numerator.dim():
            gram = gram[..., None, None]
            numerator = numerator[..., None]
        solver = _solver_device(gram)
        pinv = _pseudo_inverse(_to(gram.detach(), solver, torch.complex128), eigenvalue_floor)
        numerator = _to(numerator.detach(), solver, torch.complex128)
        explained = torch.einsum("...i,...ij,...j->...", numerator.conj(), pinv, numerator).real
        residual = _to(weighted_energy.detach(), solver, torch.float64) - explained
        return residual / max(int(num_bf) - gram.shape[-1], 1)


def noise_variance_from_residual(
    gram, numerator, weighted_energy, num_bf: int, mask=None, eigenvalue_floor=EIGENVALUE_FLOOR
):
    """The noise variance of the normalized bright-field data per unit weight, from the residual.

    With inverse-variance weights the residual at one frequency,
    ``sum_k w_k |G_k|^2 - b^H M^+ b``, has expectation ``(num_bf - K) sigma^2`` wherever the
    model holds. The median over the frequencies in ``mask`` discounts the few where model
    error inflates it, such as a strong scatterer's Bragg reflections. Model error spread over
    every frequency still raises it, which makes :func:`ard_estimate` shrink more, not less.
    """
    with torch.no_grad():
        if gram.dim() == numerator.dim():
            gram = gram[..., None, None]
            numerator = numerator[..., None]
        solver = _solver_device(gram)
        pinv = _pseudo_inverse(_to(gram.detach(), solver, torch.complex128), eigenvalue_floor)
        numerator = _to(numerator.detach(), solver, torch.complex128)
        explained = torch.einsum("...i,...ij,...j->...", numerator.conj(), pinv, numerator).real
        residual = _to(weighted_energy.detach(), solver, torch.float64) - explained
        if mask is not None:
            residual = residual[mask.to(solver) > 0]
        dof = max(int(num_bf) - gram.shape[-1], 1)
        return float(residual.median()) / dof


def explained_energy(gram, numerator, mask=None, eigenvalue_floor: float = EIGENVALUE_FLOOR):
    """``sum_q Re(b^H M^+ b)``, the data energy a set of columns explains, by Parseval.

    Subtracted from ``sum_q sum_k |G|^2`` it is the least-squares residual, so the ratio of
    the two is what :meth:`residual_loss` reports.
    """
    if gram.dim() == numerator.dim():
        gram = gram[..., None, None]
        numerator = numerator[..., None]
    solver = _solver_device(numerator)
    pinv = _pseudo_inverse(_to(gram, solver, torch.complex128), eigenvalue_floor)
    numerator = _to(numerator, solver, torch.complex128)
    energy = torch.einsum("...i,...ij,...j->...", numerator.conj(), pinv, numerator).real
    if mask is not None:
        energy = energy * mask.to(solver)
    return energy.sum()


def woa_kernel_terms(
    kernel: str,
    vbf_fourier,
    probe,
    cmplx_probe_at_k,
    kx,
    ky,
    qxa,
    qya,
    wavelength: float,
    depths,
    complex_obj: bool,
    aperture_floor: float,
    peak_intensity: float = 1.0,
    alias_factor: int = 1,
    slice_phases=None,
) -> dict:
    """One batch's contribution to the sums of the weak-object kernels: ``ssb``, ``obf``,
    ``wiener`` and ``wdd``.

    Shared by both direct-ptychography classes, which differ only in where ``vbf_fourier``
    (``(B, Ny, Nx)``, the bright-field spectra on the reconstruction's q-grid) comes from: one
    scan FFT, or a splat and an FFT per bright-field image.

    Returns ``gram`` and ``numerator`` for the normal equations, which ``obf`` and ``wiener``
    solve and every kernel's residual reads, and for ``ssb`` its own per-pixel sums ``phase`` and,
    for a complex object, ``amplitude``.

    ``peak_intensity`` is the largest ``|psi(k)|^2`` over the aperture. The Gram weights are
    taken relative to it, so they are one where the aperture is flat whatever the probe's
    normalization. The estimate does not depend on it, but the explained energy, and so the
    residual, would otherwise scale with it.

    The residual always uses the weighted model, whatever the number of columns, so it is a
    least-squares residual for a measured probe, whose ``|psi(k)|^2`` can vary severalfold
    across the aperture, and comparable between one slice and several. Its sums are
    ``residual_gram``, for one column only (with several it is ``gram``), and
    ``weighted_energy``, ``sum_k |psi(k)|^2 |G_k|^2`` relative to the peak.

    With ``alias_factor = U > 1``, ``qxa, qya`` are a grid ``U`` times finer than the scan and
    ``vbf_fourier`` the scan's spectra tiled onto it. The normal equations are then accumulated
    on the scan's own grid, with the columns of all ``U^2`` alias tiles of each frequency side
    by side (:func:`alias_tiles`), so that the solve separates the tiles jointly rather than
    treating the others as noise. SSB's own sums stay on the fine grid.

    ``slice_phases``, ``(N, H, W)``, multiplies every column of slice ``n`` by a phase that
    depends on ``q`` alone: a lateral displacement of that slice, such as a specimen tilt gives
    each depth in a tilt series.
    """
    t_minus, t_plus, psi_minus, psi_plus = overlap_terms(
        probe, cmplx_probe_at_k, kx, ky, qxa, qya, partial_coherence=False
    )
    coherent = (t_minus, t_plus)
    if probe.is_partially_coherent:
        env_minus, env_plus = partial_coherence_envelope(
            kx, ky, qxa, qya, probe.wavelength, probe.focal_spread, probe.source_size
        )
        t_minus = t_minus * env_minus.to(t_minus.real.dtype)
        t_plus = t_plus * env_plus.to(t_plus.real.dtype)
    factors = depth_phase_factors(qxa, qya, kx, ky, wavelength, depths)
    columns = woa_columns(t_minus, t_plus, factors, complex_obj)
    if slice_phases is not None:
        # the columns are the slices' phase columns, then their amplitude columns
        repeat = columns.shape[-1] // slice_phases.shape[0]
        columns = columns * slice_phases.repeat(repeat, 1, 1).permute(1, 2, 0).to(columns.dtype)
    data = vbf_fourier
    if alias_factor > 1:
        columns = alias_tiles(columns, alias_factor)
        data = vbf_fourier[:, : columns.shape[1], : columns.shape[2]]
    # one column only rescales each frequency, so the single-plane phase keeps the unweighted
    # Gram it has always had; several columns need the exact weights
    intensity = cmplx_probe_at_k.abs().square() / peak_intensity
    several = columns.shape[-1] > 1
    terms = {}
    terms["gram"], terms["numerator"] = accumulate_normal_equations(
        columns, data, intensity if several else None
    )
    if not several:
        terms["residual_gram"] = weighted_gram(columns, intensity)
    terms["weighted_energy"] = (intensity * data.abs().square()).sum(0)

    if kernel == "ssb":
        # unit-modulus weights from the coherent overlap: the partial-coherence envelope would
        # give the in-focus triple overlap, where the coherent Gamma- vanishes, a phase of its
        # own and admit pixels that carry no signal
        t_minus, t_plus = coherent
        unit = unit_overlap(t_minus - t_plus, peak_intensity)
        if not complex_obj:
            terms["phase"] = (-1.0j * vbf_fourier * unit).sum(0)
        else:
            # strategy D: on the two double-overlap sidebands Gamma+ = +-Gamma-, so their
            # phase-compensated sum and difference separate the two channels
            double = double_overlap_mask(cmplx_probe_at_k, psi_minus, psi_plus, aperture_floor)
            terms["phase"] = (double * vbf_fourier * unit).sum(0)
            terms["amplitude"] = (
                double * vbf_fourier * unit_overlap(t_minus + t_plus, peak_intensity)
            ).sum(0)
    return terms


# ---------------------------------------------------------------------------
# Wigner distribution deconvolution
# ---------------------------------------------------------------------------

#: Default relative Wiener constant of WDD. On simulated data it was best near 1e-8 without
#: noise, 1e-3 at 1e5 and 1e-2 at 1e4 electrons per Angstrom^2.
WDD_RIDGE = 1e-3

#: WDD's convolution over the detector is linear, so its transforms run on a grid this many
#: times the detector's, with the data zero outside it.
WDD_PADDING = 2


def wdd_upsilon(
    probe,
    kx,
    ky,
    support,
    rows,
    cols,
    vbf_fourier,
    intensity,
    dc: float,
    qxa,
    qya,
    ridge: float = WDD_RIDGE,
    batch_size: int | None = None,
    verbose: bool = False,
    return_transfer: bool = False,
):
    """``Upsilon(Q, 0) = T~(0) T~*(-Q)`` on the ``(H, W)`` grid ``qxa, qya``, by bright-field WDD.

    For a multiplicative object ``T`` of any strength, the scan transform of the intensity at
    detector frequency ``k`` is the convolution over the detector

        ``G(Q, k) = sum_kappa Gamma(Q, k - kappa) Upsilon(Q, kappa)``,

    with ``Gamma(Q, k) = psi(k) psi*(k + Q)``, the ``t_plus`` of :func:`overlap_terms` (summed
    over modes and with the partial-coherence envelope, for a partially coherent probe), and
    ``Upsilon(Q, kappa) = T~(kappa) T~*(kappa - Q)``. Under the weak-object approximation it
    reduces to the overlap-function model the other kernels use. Transforming over ``k`` turns
    the convolution into a product, divided here by the Wiener filter
    ``Gamma^* / (|Gamma^|^2 + ridge max |Gamma^|^2)``; the ``kappa = 0`` term of the result is
    its mean over the conjugate variable. By Parseval that is ``sum_k w(Q, k) G(Q, k)`` with
    ``w`` the transform of the filter, one weight per bright-field pixel and frequency.

    ``kx, ky`` are the scan-frame frequencies of a detector grid :data:`WDD_PADDING` times the
    detector's at the same spacing, ``support`` the pixels of it the aperture reaches, outside
    which ``Gamma`` vanishes and is not evaluated, and ``rows, cols`` the bright-field pixels'
    indices in it.
    ``vbf_fourier`` is ``(num_bf, Ny, Nx)``, the scan transforms of the mean-normalized
    bright-field images with the DC removed, which ``intensity`` (``(num_bf,)``, the means the
    normalization divided out) turns back into intensities and ``dc`` restores at ``Q = 0``. A
    grid ``qxa, qya`` finer than the scan reads the data tiled, as the other kernels do.
    Computed without gradients.

    The filter keeps only part of the ``kappa = 0`` term: its weight there is the mean over the
    conjugate variable of ``|Gamma^|^2 / (|Gamma^|^2 + ridge max |Gamma^|^2)``, below one wherever
    the probe's ambiguity function ``Gamma^`` is weak. ``return_transfer`` also returns that
    weight, ``(H, W)``, which is the transfer at each ``Q`` for a weak object.
    """
    with torch.no_grad():
        real_dtype = kx.dtype
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
        support_kx, support_ky = kx[support][None], ky[support][None]
        psi = probe.read(support_kx, support_ky).to(complex_dtype)
        scale = float(psi.abs().square().sum()) ** 2
        num_points = kx.numel()
        _, num_rows, num_cols = vbf_fourier.shape
        height, width = qxa.shape
        device = kx.device
        qx_all = qxa.reshape(-1).to(device, real_dtype)
        qy_all = qya.reshape(-1).to(device, real_dtype)
        q_rows = torch.arange(height, device=device).repeat_interleave(width)
        q_cols = torch.arange(width, device=device).repeat(height)
        intensity = intensity.to(device, complex_dtype)[:, None]
        if batch_size is None:
            batch_size = max(1, 2**22 // num_points)

        upsilon = torch.zeros(height * width, dtype=complex_dtype, device=device)
        transfer = torch.zeros(height * width, dtype=real_dtype, device=device)
        starts = range(0, height * width, batch_size)
        for start in tqdm(starts, disable=not verbose):
            batch = slice(start, start + batch_size)
            qx = qx_all[batch].view(-1, 1, 1)
            qy = qy_all[batch].view(-1, 1, 1)
            _, values, _, _ = overlap_terms(
                probe, psi, support_kx, support_ky, qx[..., 0], qy[..., 0]
            )
            gamma = torch.zeros((values.shape[0], *kx.shape), dtype=complex_dtype, device=device)
            gamma[:, support] = values.to(complex_dtype)
            gamma_hat = torch.fft.fft2(gamma)
            power = gamma_hat.abs().square()
            wiener = gamma_hat.conj() / (power + ridge * scale)
            transfer[batch] = (power / (power + ridge * scale)).mean((-2, -1))
            weights = torch.fft.fft2(wiener)[:, rows, cols] / num_points  # (batch, num_bf)

            data_rows, data_cols = q_rows[batch] % num_rows, q_cols[batch] % num_cols
            data = vbf_fourier[:, data_rows, data_cols].to(complex_dtype) * intensity
            # the DC was removed from the scan transform; restore it at Q = 0 alone, since on
            # a finer grid its tiled copies stand for aliases the data never held
            if start == 0:
                data[:, 0] = dc * intensity[:, 0]
            upsilon[batch] = (weights * data.T).sum(-1)
        if return_transfer:
            return upsilon.reshape(height, width), transfer.reshape(height, width)
        return upsilon.reshape(height, width)
