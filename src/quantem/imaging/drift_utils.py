"""Torch helper functions for drift correction."""

import math

import torch
from torch.fft import fft2, fftfreq, ifft2, ifftshift


# ---------------------------------------------------------------------------
# Public API — called by DriftCorrection in drift.py
# ---------------------------------------------------------------------------


def bilinear_kde_batch(
    row_coords: torch.Tensor,
    col_coords: torch.Tensor,
    values: torch.Tensor,
    output_shape: tuple[int, int],
    kde_sigma: float,
    pad_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched bilinear KDE for warping images under candidate drift vectors.

    Each candidate drift vector maps source pixels to different canvas
    positions. This function scatters all candidates onto the canvas
    in one pass, producing N warped images simultaneously. Without
    batched warping, each candidate would require a separate Python
    loop iteration — the single largest bottleneck before GPU acceleration.

    Each pixel scatters its value to its 4 nearest grid neighbors with
    weights ``w = (1-dr)·(1-dc)``, ``dr·(1-dc)``, ``(1-dr)·dc``, ``dr·dc``
    where ``dr, dc`` are fractional row/col distances. Accumulated counts
    and values are Gaussian-smoothed, then normalized:
    ``output = pad_value·(1-coverage) + coverage·(values/counts)``.

    Parameters
    ----------
    row_coords : torch.Tensor
        Row coordinates of input pixels, shape ``(N, rows, cols)``.
    col_coords : torch.Tensor
        Column coordinates of input pixels, shape ``(N, rows, cols)``.
    values : torch.Tensor
        Pixel values to scatter, shape ``(rows, cols)``.
        Same image is used for all N candidates.
    output_shape : tuple[int, int]
        Canvas size ``(num_rows, num_cols)`` for the output images.
    kde_sigma : float
        Gaussian smoothing sigma in pixels.
    pad_value : float
        Fill value where pixel coverage is below threshold.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(images, pix_count)`` — warped images and smoothed pixel coverage,
        both shape ``(N, num_rows, num_cols)``.

    Examples
    --------
    >>> rows = torch.rand(5, 32, 32, dtype=torch.float64) * 38 + 0.5
    >>> cols = torch.rand(5, 32, 32, dtype=torch.float64) * 38 + 0.5
    >>> vals = torch.randn(32, 32, dtype=torch.float64)
    >>> out, weights = bilinear_kde_batch(rows, cols, vals, (40, 40), 0.5, 0.0)
    >>> out.shape
    torch.Size([5, 40, 40])
    """
    num_candidates = row_coords.shape[0]
    num_rows, num_cols = output_shape
    coverage_threshold = 1e-3
    # int32 uses half the memory of int64, which matters because scatter_add_
    # allocates index buffers for all N candidates × all pixels simultaneously
    row_floor = row_coords.reshape(num_candidates, -1).floor().int()
    col_floor = col_coords.reshape(num_candidates, -1).floor().int()
    frac_row = row_coords.reshape(num_candidates, -1).float() - row_floor.float()
    frac_col = col_coords.reshape(num_candidates, -1).float() - col_floor.float()
    values_flat = values.reshape(-1).float().expand(num_candidates, -1).reshape(-1)
    # All N candidates scatter into one flat buffer — offset separates them
    candidate_offset = (
        torch.arange(num_candidates, device=row_coords.device, dtype=torch.int32)
        * num_rows * num_cols
    )[:, None]
    pix_count = torch.zeros(
        num_candidates * num_rows * num_cols, dtype=torch.float32, device=row_coords.device
    )
    pix_output = torch.zeros_like(pix_count)
    # Periodic wrapping so pixels near edges scatter to the opposite side
    row_wrapped = row_floor % num_rows
    col_wrapped = col_floor % num_cols
    row_next = (row_wrapped + 1) % num_rows
    col_next = (col_wrapped + 1) % num_cols
    # Each pixel distributes its value to the 4 nearest grid neighbors
    # weighted by bilinear distance: (1-dr)(1-dc), dr(1-dc), (1-dr)dc, dr·dc
    for corner_row, corner_col, corner_weight in [
        (row_wrapped, col_wrapped, ((1 - frac_row) * (1 - frac_col)).reshape(-1)),
        (row_next, col_wrapped, (frac_row * (1 - frac_col)).reshape(-1)),
        (row_wrapped, col_next, ((1 - frac_row) * frac_col).reshape(-1)),
        (row_next, col_next, (frac_row * frac_col).reshape(-1)),
    ]:
        flat_idx = (corner_row * num_cols + corner_col + candidate_offset).reshape(-1)
        pix_count.scatter_add_(0, flat_idx, corner_weight)
        pix_output.scatter_add_(0, flat_idx, corner_weight * values_flat)
    pix_count = pix_count.reshape(num_candidates, num_rows, num_cols)
    pix_output = pix_output.reshape(num_candidates, num_rows, num_cols)
    # Smooth the scattered counts and values to fill gaps between pixels
    pix_count = gaussian_smooth_batch(pix_count, kde_sigma)
    pix_output = gaussian_smooth_batch(pix_output, kde_sigma)
    # Blend between pad_value (uncovered) and normalized values (covered),
    # ramping linearly with coverage to avoid hard edges at the boundary
    coverage_weight = torch.clamp(pix_count / coverage_threshold, max=1.0)
    images = pad_value * (1 - coverage_weight) + coverage_weight * (
        pix_output / torch.clamp(pix_count, min=1e-8)
    )
    return images, pix_count


def cross_corr_batch(
    images_ref: torch.Tensor,
    images_moving: torch.Tensor,
    upsample_factor: int,
    max_shift_mask: torch.Tensor | None = None,
    freq_grids: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Score candidate drift vectors by cross-correlation alignment cost.

    Core cost function of the affine grid search. For each candidate,
    measures how well the warped image pairs align after sub-pixel
    translation correction. Without this, the grid search has no way
    to rank candidates — it is the signal that drives drift estimation.

    Pipeline: FFT cross-correlation → parabolic coarse peak →
    DFT upsample for sub-pixel refinement → Fourier-domain shift →
    MAE between reference and aligned image.

    Parameters
    ----------
    images_ref : torch.Tensor
        Reference images, shape ``(N, num_rows, num_cols)``.
    images_moving : torch.Tensor
        Images to align, shape ``(N, num_rows, num_cols)``.
    upsample_factor : int
        Sub-pixel upsampling factor for DFT refinement.
    max_shift_mask : torch.Tensor or None
        Precomputed boolean mask, shape ``(num_rows, num_cols)``.
        True where correlation peaks should be zeroed (beyond max shift).
    freq_grids : tuple[torch.Tensor, torch.Tensor] or None, optional
        Precomputed ``(k_row, k_col)`` from ``torch.fft.fftfreq``,
        shapes ``(num_rows, 1)`` and ``(1, num_cols)``. Avoids
        recomputing the same grids each call. Default is None.

    Returns
    -------
    torch.Tensor
        MAE cost per pair, shape ``(N,)``.

    Examples
    --------
    >>> ref = torch.randn(5, 64, 64, dtype=torch.float64)
    >>> mov = torch.randn(5, 64, 64, dtype=torch.float64)
    >>> cost = cross_corr_batch(ref, mov, 8, 32.0)
    >>> cost.shape
    torch.Size([5])
    """
    num_candidates, num_rows, num_cols = images_ref.shape
    dtype = images_ref.dtype
    fft_moving = fft2(images_moving)
    cc_fft = fft2(images_ref) * fft_moving.conj()
    cc_real = ifft2(cc_fft).real
    # Reject correlation peaks beyond max shift to avoid locking onto
    # periodic lattice repeats or noise peaks far from the true shift
    if max_shift_mask is not None:
        cc_real[:, max_shift_mask] = 0.0
    # Find best-matching shift: integer peak → parabola (~0.1 px) → DFT (~0.01 px)
    peak_flat = cc_real.reshape(num_candidates, -1).argmax(dim=1)
    peak_row = peak_flat // num_cols
    peak_col = peak_flat % num_cols
    batch_idx = torch.arange(num_candidates, device=images_ref.device)
    refined_row, refined_col = _parabolic_peak_2d(cc_real, peak_row, peak_col, num_rows, num_cols, batch_idx)
    shifts = _dft_refine_shifts(cc_fft, refined_row, refined_col, upsample_factor, batch_idx)
    # Wrap from [0, N) to [-N/2, N/2) so shifts represent actual displacement
    shifts[:, 0] = ((shifts[:, 0] + num_rows / 2) % num_rows) - num_rows / 2
    shifts[:, 1] = ((shifts[:, 1] + num_cols / 2) % num_cols) - num_cols / 2
    if freq_grids is not None:
        k_row, k_col = freq_grids
    else:
        k_row = fftfreq(num_rows, device=images_ref.device, dtype=dtype)[:, None]
        k_col = fftfreq(num_cols, device=images_ref.device, dtype=dtype)[None, :]
    phase = -2j * math.pi * (k_row[None] * shifts[:, 0, None, None] + k_col[None] * shifts[:, 1, None, None])
    return torch.mean(torch.abs(images_ref - ifft2(fft_moving * torch.exp(phase)).real), dim=(1, 2))


def translate_align(
    warped_t: torch.Tensor,
    upsample_factor: int,
    max_image_shift: float | None,
) -> torch.Tensor:
    """Pairwise translation alignment of warped images via cross-correlation.

    Called by ``_warp_and_translate_torch`` after each affine warp to
    remove residual translational misalignment between the image pair.
    Without this step, the merged image would be blurred by the remaining
    translation offset even after the affine drift is corrected.

    Starting from image 0 as reference, sequentially aligns each image
    using FFT cross-correlation with parabolic + DFT sub-pixel refinement.
    The reference is updated as a running Fourier-domain average.

    Parameters
    ----------
    warped_t : torch.Tensor
        Warped images, shape ``(num_images, num_rows, num_cols)``.
    upsample_factor : int
        Sub-pixel precision (1/N pixel) for DFT refinement.
    max_image_shift : float or None
        Maximum allowed shift in pixels. Peaks beyond this radius are masked.

    Returns
    -------
    torch.Tensor
        Zero-mean shifts, shape ``(num_images, 2)`` in (row, col) order.
    """
    num_images, num_rows, num_cols = warped_t.shape
    dtype = warped_t.dtype
    device = warped_t.device
    shifts_t = torch.zeros(num_images, 2, dtype=dtype, device=device)
    fft_ref = fft2(warped_t[0])
    # Reject spurious correlation peaks from noise or periodicity
    # by zeroing everything beyond max_image_shift pixels from origin
    shift_mask = None
    if max_image_shift is not None:
        dist_row = fftfreq(num_rows, 1.0 / num_rows, device=device, dtype=dtype)
        dist_col = fftfreq(num_cols, 1.0 / num_cols, device=device, dtype=dtype)
        shift_mask = dist_row[:, None] ** 2 + dist_col[None, :] ** 2 >= max_image_shift ** 2
    k_row = fftfreq(num_rows, device=device, dtype=dtype)[:, None]
    k_col = fftfreq(num_cols, device=device, dtype=dtype)[None, :]
    for img_idx in range(1, num_images):
        fft_current = fft2(warped_t[img_idx])
        cc_fft = fft_ref * fft_current.conj()
        cc_real = ifft2(cc_fft).real
        if shift_mask is not None:
            cc_real[shift_mask] = 0.0
        # Find the integer peak, refine with parabola to ~0.1 px
        peak_flat = cc_real.reshape(-1).argmax()
        peak_row = peak_flat[None] // num_cols
        peak_col = peak_flat[None] % num_cols
        idx = torch.zeros(1, dtype=torch.long, device=device)
        refined_row, refined_col = _parabolic_peak_2d(cc_real[None], peak_row, peak_col, num_rows, num_cols, idx)
        # Zoom into peak neighborhood with DFT to get ~0.01 px precision,
        # then wrap from [0, N) to centered [-N/2, N/2) convention
        shift = _dft_refine_shifts(cc_fft[None], refined_row, refined_col, upsample_factor, idx)
        shifts_t[img_idx, 0] = ((shift[0, 0] + num_rows / 2) % num_rows) - num_rows / 2
        shifts_t[img_idx, 1] = ((shift[0, 1] + num_cols / 2) % num_cols) - num_cols / 2
        # Apply the recovered shift to current image via Fourier shift theorem,
        # then blend into running average so later images align to the cumulative mean
        phase = torch.exp(-2j * math.pi * (k_row * shifts_t[img_idx, 0] + k_col * shifts_t[img_idx, 1]))
        fft_ref = fft_ref * img_idx / (img_idx + 1) + fft_current * phase / (img_idx + 1)
    # Remove mean so shifts are relative (no absolute reference frame)
    shifts_t -= shifts_t.mean(dim=0)
    return shifts_t

def transform_coordinates(
    knots: torch.Tensor,
    scan_fast: torch.Tensor,
    input_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute canvas-space coordinates from single Bezier knot.

    Called by ``preprocess``, ``_affine_grid_search_batch``, and
    ``_warp_and_translate_torch`` to map source image pixels onto the
    padded output canvas. Without this, the warped images would have
    no spatial mapping and the grid search couldn't score candidates.

    Each input row maps to a line on the canvas:
    ``row = knot_row + fraction * scan_fast[0] * (num_rows - 1)``
    ``col = knot_col + fraction * scan_fast[1] * (num_cols - 1)``
    where row and col dimensions scale independently for non-square images.

    Parameters
    ----------
    knots : torch.Tensor
        Knot positions, shape ``(2, num_rows, 1)``. First dim is (row, col).
    scan_fast : torch.Tensor
        Fast scan direction vector, shape ``(2,)``.
    input_shape : tuple[int, int]
        Original image shape ``(num_rows, num_cols)``.

    Returns
    -------
    row_coords : torch.Tensor
        Row coordinates on canvas, shape ``(num_rows, num_cols)``.
    col_coords : torch.Tensor
        Column coordinates on canvas, shape ``(num_rows, num_cols)``.

    Examples
    --------
    >>> knots = torch.zeros(2, 64, 1)
    >>> scan_fast = torch.tensor([0.0, 1.0])
    >>> r, c = transform_coordinates(knots, scan_fast, (64, 64))
    >>> r.shape
    torch.Size([64, 64])
    """
    num_rows, num_cols = input_shape
    fast_fraction = torch.linspace(0, 1, num_cols, dtype=knots.dtype, device=knots.device)
    row_coords = knots[0, :, 0:1] + fast_fraction[None, :] * scan_fast[0] * (num_rows - 1)
    col_coords = knots[1, :, 0:1] + fast_fraction[None, :] * scan_fast[1] * (num_cols - 1)
    return row_coords, col_coords


def gaussian_smooth_batch(
    images: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Batched 2D Gaussian smoothing matching ``scipy.ndimage.gaussian_filter``.

    Used by ``bilinear_kde_batch`` to smooth scattered counts and
    values before normalization. Without smoothing, the warped images
    have salt-and-pepper artifacts from the scatter step.

    Parameters
    ----------
    images : torch.Tensor
        Input tensor of shape ``(N, num_rows, num_cols)``.
    sigma : float
        Standard deviation of the Gaussian kernel in pixels.

    Returns
    -------
    torch.Tensor
        Smoothed tensor of shape ``(N, num_rows, num_cols)``.

    """
    kernel, radius = _gaussian_kernel_1d(sigma, images.dtype, images.device)
    # Reshape for conv2d: (1,1,1,K) for horizontal pass, (1,1,K,1) for vertical
    kernel_col = kernel[None, None, None, :]
    kernel_row = kernel[None, None, :, None]
    images = images[:, None]
    images = torch.nn.functional.conv2d(_symmetric_pad(images, 0, radius), kernel_col)
    images = torch.nn.functional.conv2d(_symmetric_pad(images, radius, 0), kernel_row)
    return images[:, 0]

# ---------------------------------------------------------------------------
# Building blocks — used internally by the public API functions above
# ---------------------------------------------------------------------------


def _dft_refine_shifts(cc_fft, refined_row, refined_col, upsample_factor, batch_idx=None):
    """Refine coarse sub-pixel shifts using DFT upsampling + parabolic fit.

    After ``_parabolic_peak_2d`` gives a coarse sub-pixel position, this
    function zooms into a small neighborhood via the matrix-multiply DFT
    and applies a second parabolic refinement on the upsampled patch.
    The result is sub-pixel shifts with ``1 / upsample_factor`` precision.

    Without this step, shifts would have only ~0.1 px precision from
    parabolic fitting alone. With ``upsample_factor=8``, precision
    improves to ~0.01 px.

    Parameters
    ----------
    cc_fft : torch.Tensor
        Complex cross-correlation in Fourier domain, ``(N, num_rows, num_cols)``.
    refined_row, refined_col : torch.Tensor
        Coarse sub-pixel peak positions in [0, N) from ``_parabolic_peak_2d``.
    upsample_factor : int
        Sub-pixel precision factor.
    batch_idx : torch.Tensor
        Batch indices, ``torch.arange(N)``.

    Returns
    -------
    shifts : torch.Tensor
        Sub-pixel shifts in [0, N) coordinates, shape ``(N, 2)``.
    """
    num_candidates = cc_fft.shape[0]
    dtype = refined_row.dtype
    # Evaluate the correlation surface at 1/upsample_factor pixel spacing
    # in a small window around each coarse peak — gives actual values,
    # not the parabolic approximation from step 1
    upsampled_cc = _dft_upsample_batch(
        cc_fft, upsample_factor, torch.stack([refined_row, refined_col], dim=1))
    upsample_size = upsampled_cc.shape[1]
    peak_flat = upsampled_cc.reshape(num_candidates, -1).argmax(dim=1)
    local_row = peak_flat // upsample_size
    local_col = peak_flat % upsample_size
    # Final parabolic fit on the dense grid for last fraction of precision.
    # Peaks at the edge of the upsampled window can't use the 3-point stencil
    # (no neighbor on one side), so those are masked and kept at integer position
    can_refine = ((local_row >= 1) & (local_row < upsample_size - 1)
                  & (local_col >= 1) & (local_col < upsample_size - 1))
    peak_val = upsampled_cc[batch_idx, local_row, local_col]
    d_row_fine = _parabolic_sub_pixel(
        upsampled_cc[batch_idx, (local_row - 1).clamp(min=0), local_col], peak_val,
        upsampled_cc[batch_idx, (local_row + 1).clamp(max=upsample_size - 1), local_col],
        mask=can_refine)
    d_col_fine = _parabolic_sub_pixel(
        upsampled_cc[batch_idx, local_row, (local_col - 1).clamp(min=0)], peak_val,
        upsampled_cc[batch_idx, local_row, (local_col + 1).clamp(max=upsample_size - 1)],
        mask=can_refine)
    # Convert upsampled-grid position back to image-pixel coordinates:
    # local_row is in upsampled grid units, upsample_factor is the center index,
    # so (local_row - upsample_factor) / upsample_factor = offset from coarse peak
    shifts = torch.zeros(num_candidates, 2, dtype=dtype, device=cc_fft.device)
    shifts[:, 0] = refined_row + (local_row.to(dtype) - upsample_factor + d_row_fine) / upsample_factor
    shifts[:, 1] = refined_col + (local_col.to(dtype) - upsample_factor + d_col_fine) / upsample_factor
    return shifts


def _dft_upsample_batch(
    cross_corr_fft: torch.Tensor,
    upsample_factor: int,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Sub-pixel peak refinement for all drift candidates in one pass.

    After the coarse FFT cross-correlation finds integer-pixel peaks,
    zooms into a small neighborhood using the Guizar-Sicairos
    matrix-multiply DFT. Without DFT upsampling, shift precision is
    limited to ~0.1 px from parabolic fitting alone. With
    ``upsample_factor=8``, precision improves to ~0.01 px.

    Parameters
    ----------
    cross_corr_fft : torch.Tensor
        Complex 2D cross-correlation in Fourier domain, shape ``(N, num_rows, num_cols)``.
    upsample_factor : int
        Upsampling factor (typically 8).
    shifts : torch.Tensor
        Coarse peak locations ``(row, col)`` per candidate, shape ``(N, 2)``.

    Returns
    -------
    torch.Tensor
        Real-valued upsampled correlation neighborhoods, shape ``(N, P, P)``
        where ``P = 2 * ceil(1.5 * upsample_factor) + 1``.

    """
    num_candidates, num_rows, num_cols = cross_corr_fft.shape
    real_dtype = torch.float32 if cross_corr_fft.dtype == torch.complex64 else torch.float64
    # 1.5x radius ensures the patch captures the true peak after parabolic shift
    patch_radius = math.ceil(1.5 * upsample_factor)
    # Upsampled grid positions centered at zero: [-radius, ..., 0, ..., +radius]
    upsample_grid = torch.arange(-patch_radius, patch_radius + 1, dtype=real_dtype, device=cross_corr_fft.device)
    # ifftshift reorders [0,1,...,N-1] to match FFT output ordering,
    # then subtract N//2 to center at zero
    freq_row_base = ifftshift(
        torch.arange(num_rows, dtype=real_dtype, device=cross_corr_fft.device)
    ) - num_rows // 2
    freq_col_base = ifftshift(
        torch.arange(num_cols, dtype=real_dtype, device=cross_corr_fft.device)
    ) - num_cols // 2
    freq_row = freq_row_base[None, :] + (shifts[:, 0] - num_rows // 2)[:, None]
    freq_col = freq_col_base[None, :] + (shifts[:, 1] - num_cols // 2)[:, None]
    # Guizar-Sicairos matrix-multiply DFT: K_row @ CC @ K_col
    kern_row = torch.exp(
        -2j * math.pi / (num_rows * upsample_factor)
        * upsample_grid[None, :, None] * freq_row[:, None, :]
    ).to(cross_corr_fft.dtype)
    kern_col = torch.exp(
        -2j * math.pi / (num_cols * upsample_factor)
        * freq_col[:, :, None] * upsample_grid[None, None, :]
    ).to(cross_corr_fft.dtype)
    # (N,P,M) @ (N,M,K) @ (N,K,P) -> (N,P,P)
    return (kern_row @ cross_corr_fft @ kern_col).real

# ---------------------------------------------------------------------------
# Primitives — lowest-level operations
# ---------------------------------------------------------------------------


def _parabolic_peak_2d(cc, peak_row, peak_col, num_rows, num_cols, batch_idx):
    """Refine an integer cross-correlation peak to sub-pixel precision.

    Extracts the 3-point stencil along each axis and fits a parabola.
    Without this, the DFT upsample window would be centered on the
    integer peak which may be up to 0.5 px away from the true peak,
    causing the upsampled patch to miss the true maximum.

    Parameters
    ----------
    cc : torch.Tensor
        Batched correlation map, shape ``(N, num_rows, num_cols)``.
    peak_row, peak_col : torch.Tensor
        Integer peak positions, shape ``(N,)``.
    num_rows, num_cols : int
        Dimensions for periodic wrapping.
    batch_idx : torch.Tensor
        Batch indices, ``torch.arange(N)``.

    Returns
    -------
    refined_row, refined_col : torch.Tensor
        Sub-pixel peak positions in [0, N) coordinates.
    """
    dtype = cc.dtype
    val_center = cc[batch_idx, peak_row, peak_col]
    val_row_m1 = cc[batch_idx, (peak_row - 1) % num_rows, peak_col]
    val_row_p1 = cc[batch_idx, (peak_row + 1) % num_rows, peak_col]
    val_col_m1 = cc[batch_idx, peak_row, (peak_col - 1) % num_cols]
    val_col_p1 = cc[batch_idx, peak_row, (peak_col + 1) % num_cols]
    refined_row = (peak_row.to(dtype) + _parabolic_sub_pixel(val_row_m1, val_center, val_row_p1)) % num_rows
    refined_col = (peak_col.to(dtype) + _parabolic_sub_pixel(val_col_m1, val_center, val_col_p1)) % num_cols
    return refined_row, refined_col


def _parabolic_sub_pixel(val_m1, val_0, val_p1, mask=None):
    """Sub-pixel offset from a 3-point stencil via parabolic interpolation.

    Cross-correlation peaks fall on integer pixel positions, but the true
    shift is usually between pixels. Fitting a parabola through the peak
    and its two neighbors gives ~0.1 px precision cheaply:
    ``offset = (val_p1 - val_m1) / (4·val_0 - 2·val_p1 - 2·val_m1)``.
    Without this, the DFT upsample window may be centered on the wrong
    pixel and miss the true peak.
    """
    denom = 4 * val_0 - 2 * val_p1 - 2 * val_m1
    valid = denom != 0
    if mask is not None:
        valid = valid & mask
    return torch.where(valid, (val_p1 - val_m1) / denom, torch.zeros_like(denom))


def _symmetric_pad(
    images: torch.Tensor,
    pad_rows: int,
    pad_cols: int,
) -> torch.Tensor:
    """Symmetric padding matching scipy's reflect mode for parity.

    Scipy's ``mode='reflect'`` repeats the edge pixel
    (``[1,2,3]`` → ``[2,1,1,2,3,3,2]``), but PyTorch's
    ``F.pad(mode='reflect')`` does not (``[1,2,3]`` → ``[3,2,1,2,3,2,1]``).
    Without this, the torch and numpy Gaussian smoothing paths produce
    different results near canvas edges, breaking numerical parity.

    Parameters
    ----------
    images : torch.Tensor
        Input tensor of shape ``(N, C, num_rows, num_cols)``.
    pad_rows : int
        Number of rows to pad on top and bottom.
    pad_cols : int
        Number of columns to pad on left and right.

    Returns
    -------
    torch.Tensor
        Padded tensor.

    Examples
    --------
    >>> t = torch.tensor([[[[1., 2., 3.]]]])
    >>> _symmetric_pad(t, 0, 2)
    tensor([[[[2., 1., 1., 2., 3., 3., 2.]]]])
    """
    if pad_cols > 0:
        left = images[:, :, :, :pad_cols].flip(-1)
        right = images[:, :, :, -pad_cols:].flip(-1)
        images = torch.cat([left, images, right], dim=-1)
    if pad_rows > 0:
        top = images[:, :, :pad_rows, :].flip(-2)
        bottom = images[:, :, -pad_rows:, :].flip(-2)
        images = torch.cat([top, images, bottom], dim=-2)
    return images


def _gaussian_kernel_1d(sigma, dtype, device, _cache={}):
    """Normalized 1D Gaussian ``exp(-0.5*(x/sigma)^2)``, radius ``4*sigma``.

    Cached via mutable default arg — the grid search calls this ~800 times
    with the same sigma, saving ~44ms of redundant kernel construction.
    """
    key = (sigma, dtype, device)
    if key not in _cache:
        radius = int(4 * sigma + 0.5)
        offsets = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
        kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        _cache[key] = (kernel / kernel.sum(), radius)
    return _cache[key]

