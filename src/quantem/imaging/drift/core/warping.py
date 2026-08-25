"""Cross-correlation and translation refinement for drift correction."""

import math

import torch
from torch.fft import fft2, fftfreq, ifft2, ifftshift


def cross_corr_batch(
    ref_images: torch.Tensor,
    mov_images: torch.Tensor,
    upsample_factor: int,
    max_shift_mask: torch.Tensor | None = None,
    freq_grids: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Score test drift vectors by cross-correlation alignment cost.

    Core cost function of the affine grid search. For each test drift,
    measures how well the warped image pairs align after sub-pixel
    translation correction. Without this, the grid search has no way
    to rank test drifts - it is the signal that drives drift estimation.

    Pipeline: FFT cross-correlation → parabolic coarse peak →
    DFT upsample for sub-pixel refinement → Fourier-domain shift →
    MAE between reference and aligned image.

    Parameters
    ----------
    ref_images : torch.Tensor
        Reference images, shape ``(N, num_rows, num_cols)``.
    mov_images : torch.Tensor
        Images to align, shape ``(N, num_rows, num_cols)``.
    upsample_factor : int
        Sub-pixel upsampling factor for DFT refinement.
    max_shift_mask : torch.Tensor or None
        Precomputed boolean mask, shape ``(num_rows, num_cols)``.
        True where correlation peaks should be zeroed (beyond max shift).
    freq_grids : tuple[torch.Tensor, torch.Tensor] or None, optional
        Precomputed ``(freq_row, freq_col)`` from ``torch.fft.fftfreq``,
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
    num_test_drifts, num_rows, num_cols = ref_images.shape
    dtype = ref_images.dtype
    mov_fft = fft2(mov_images)
    cross_corr_fft = fft2(ref_images) * mov_fft.conj()
    cross_corr = ifft2(cross_corr_fft).real
    # Reject correlation peaks beyond max shift to avoid locking onto
    # periodic lattice repeats or noise peaks far from the true shift
    if max_shift_mask is not None:
        cross_corr.masked_fill_(max_shift_mask[None], 0.0)
    # Find best-matching shift: integer peak → parabola (~0.1 px) → DFT (~0.01 px)
    peak_flat_idx = cross_corr.flatten(1).argmax(dim=1)
    peak_row = peak_flat_idx // num_cols
    peak_col = peak_flat_idx % num_cols
    batch_idx = torch.arange(num_test_drifts, device=ref_images.device)
    refined_row, refined_col = _parabolic_peak_2d(
        cross_corr, peak_row, peak_col, num_rows, num_cols, batch_idx
    )
    image_shifts = _dft_refine_shifts(
        cross_corr_fft, refined_row, refined_col, upsample_factor
    )
    # Wrap from [0, N) to [-N/2, N/2) so shifts represent actual displacement
    image_shifts[:, 0] = ((image_shifts[:, 0] + num_rows / 2) % num_rows) - num_rows / 2
    image_shifts[:, 1] = ((image_shifts[:, 1] + num_cols / 2) % num_cols) - num_cols / 2
    if freq_grids is not None:
        freq_row, freq_col = freq_grids
    else:
        freq_row = fftfreq(num_rows, device=ref_images.device, dtype=dtype)[:, None]
        freq_col = fftfreq(num_cols, device=ref_images.device, dtype=dtype)[None, :]
    phase = -2j * math.pi * (
        freq_row[None] * image_shifts[:, 0, None, None]
        + freq_col[None] * image_shifts[:, 1, None, None]
    )
    aligned_images = ifft2(mov_fft * torch.exp(phase)).real
    return torch.mean(torch.abs(ref_images - aligned_images), dim=(1, 2))


def translate_align(
    warped_images: torch.Tensor,
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
    warped_images : torch.Tensor
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
    num_images, num_rows, num_cols = warped_images.shape
    dtype = warped_images.dtype
    device = warped_images.device
    image_shifts = torch.zeros(num_images, 2, dtype=dtype, device=device)
    ref_fft = fft2(warped_images[0])
    # Reject bad correlation peaks from noise or periodicity
    # by zeroing everything beyond max_image_shift pixels from origin
    shift_mask = None
    if max_image_shift is not None:
        dist_row = fftfreq(num_rows, 1.0 / num_rows, device=device, dtype=dtype)
        dist_col = fftfreq(num_cols, 1.0 / num_cols, device=device, dtype=dtype)
        shift_mask = dist_row[:, None] ** 2 + dist_col[None, :] ** 2 >= max_image_shift ** 2
    freq_row = fftfreq(num_rows, device=device, dtype=dtype)[:, None]
    freq_col = fftfreq(num_cols, device=device, dtype=dtype)[None, :]
    for img_idx in range(1, num_images):
        mov_fft = fft2(warped_images[img_idx])
        cross_corr_fft = ref_fft * mov_fft.conj()
        cross_corr = ifft2(cross_corr_fft).real
        if shift_mask is not None:
            cross_corr.masked_fill_(shift_mask, 0.0)
        # Find the integer peak, refine with parabola to ~0.1 px
        peak_flat_idx = cross_corr.flatten().argmax()
        peak_row = peak_flat_idx[None] // num_cols
        peak_col = peak_flat_idx[None] % num_cols
        batch_idx = torch.zeros(1, dtype=torch.long, device=device)
        refined_row, refined_col = _parabolic_peak_2d(
            cross_corr[None], peak_row, peak_col, num_rows, num_cols, batch_idx
        )
        # Zoom into peak neighborhood with DFT to get ~0.01 px precision,
        # then wrap from [0, N) to centered [-N/2, N/2) convention
        refined_shift = _dft_refine_shifts(
            cross_corr_fft[None], refined_row, refined_col, upsample_factor
        )
        image_shifts[img_idx, 0] = ((refined_shift[0, 0] + num_rows / 2) % num_rows) - num_rows / 2
        image_shifts[img_idx, 1] = ((refined_shift[0, 1] + num_cols / 2) % num_cols) - num_cols / 2
        # Apply the recovered shift to current image via Fourier shift theorem,
        # then blend into running average so later images align to the cumulative mean
        phase = torch.exp(
            -2j * math.pi * (
                freq_row * image_shifts[img_idx, 0] + freq_col * image_shifts[img_idx, 1]
            )
        )
        ref_fft = ref_fft * img_idx / (img_idx + 1) + mov_fft * phase / (img_idx + 1)
    # Remove mean so shifts are relative (no absolute reference frame)
    image_shifts -= image_shifts.mean(dim=0)
    return image_shifts


def _dft_refine_shifts(
    cross_corr_fft,
    peak_row,
    peak_col,
    upsample_factor,
):
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
    cross_corr_fft : torch.Tensor
        Complex cross-correlation in Fourier domain, ``(N, num_rows, num_cols)``.
    peak_row, peak_col : torch.Tensor
        Coarse sub-pixel peak positions in [0, N) from ``_parabolic_peak_2d``.
    upsample_factor : int
        Sub-pixel precision factor.

    Returns
    -------
    image_shifts : torch.Tensor
        Sub-pixel shifts in [0, N) coordinates, shape ``(N, 2)``.
    """
    num_test_drifts = cross_corr_fft.shape[0]
    dtype = peak_row.dtype
    batch_idx = torch.arange(num_test_drifts, device=cross_corr_fft.device)
    # Evaluate the correlation surface at 1/upsample_factor pixel spacing
    # in a small window around each coarse peak - gives actual values,
    # not the parabolic approximation from step 1
    upsampled_corr = _dft_upsample_batch(
        cross_corr_fft, upsample_factor, torch.stack([peak_row, peak_col], dim=1)
    )
    upsample_size = upsampled_corr.shape[1]
    peak_flat_idx = upsampled_corr.flatten(1).argmax(dim=1)
    local_row = peak_flat_idx // upsample_size
    local_col = peak_flat_idx % upsample_size
    # Final parabolic fit on the dense grid for last fraction of precision.
    # Peaks at the edge of the upsampled window can't use the 3-point stencil
    # (no neighbor on one side), so those are masked and kept at integer position
    can_refine = (
        (local_row >= 1)
        & (local_row < upsample_size - 1)
        & (local_col >= 1)
        & (local_col < upsample_size - 1)
    )
    peak_val = upsampled_corr[batch_idx, local_row, local_col]
    d_row_fine = _parabolic_sub_pixel(
        upsampled_corr[batch_idx, (local_row - 1).clamp(min=0), local_col],
        peak_val,
        upsampled_corr[batch_idx, (local_row + 1).clamp(max=upsample_size - 1), local_col],
        mask=can_refine,
    )
    d_col_fine = _parabolic_sub_pixel(
        upsampled_corr[batch_idx, local_row, (local_col - 1).clamp(min=0)],
        peak_val,
        upsampled_corr[batch_idx, local_row, (local_col + 1).clamp(max=upsample_size - 1)],
        mask=can_refine,
    )
    # Convert upsampled-grid position back to image-pixel coordinates:
    # patch center is at index patch_radius in the upsampled grid,
    # so (local_row - patch_radius) / upsample_factor = offset from coarse peak
    patch_radius = math.ceil(1.5 * upsample_factor)
    image_shifts = torch.zeros(num_test_drifts, 2, dtype=dtype, device=cross_corr_fft.device)
    # local_row/col are int from argmax - cast to float for sub-pixel arithmetic
    image_shifts[:, 0] = peak_row + (local_row.to(dtype) - patch_radius + d_row_fine) / upsample_factor
    image_shifts[:, 1] = peak_col + (local_col.to(dtype) - patch_radius + d_col_fine) / upsample_factor
    return image_shifts


def _dft_upsample_batch(
    cross_corr_fft: torch.Tensor,
    upsample_factor: int,
    peak_positions: torch.Tensor,
) -> torch.Tensor:
    """Sub-pixel peak refinement for all test drifts in one pass.

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
    peak_positions : torch.Tensor
        Coarse peak locations ``(row, col)`` per test drift, shape ``(N, 2)``.

    Returns
    -------
    torch.Tensor
        Real-valued upsampled correlation neighborhoods, shape ``(N, P, P)``
        where ``P = 2 * ceil(1.5 * upsample_factor) + 1``.

    """
    num_test_drifts, num_rows, num_cols = cross_corr_fft.shape
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
    freq_row = freq_row_base[None, :] + (peak_positions[:, 0] - num_rows // 2)[:, None]
    freq_col = freq_col_base[None, :] + (peak_positions[:, 1] - num_cols // 2)[:, None]
    # Guizar-Sicairos matrix-multiply DFT: K_row @ CC @ K_col
    kern_row = torch.exp(
        -2j * math.pi / (num_rows * upsample_factor)
        * upsample_grid[None, :, None] * freq_row[:, None, :]
    ).to(cross_corr_fft.dtype)  # real → complex for matrix multiply
    kern_col = torch.exp(
        -2j * math.pi / (num_cols * upsample_factor)
        * freq_col[:, :, None] * upsample_grid[None, None, :]
    ).to(cross_corr_fft.dtype)  # real → complex for matrix multiply
    # (N,P,M) @ (N,M,K) @ (N,K,P) -> (N,P,P)
    return (kern_row @ cross_corr_fft @ kern_col).real


def _parabolic_peak_2d(cross_corr, peak_row, peak_col, num_rows, num_cols, batch_idx):
    """Refine an integer cross-correlation peak to sub-pixel precision.

    Extracts the 3-point stencil along each axis and fits a parabola.
    Without this, the DFT upsample window would be centered on the
    integer peak which may be up to 0.5 px away from the true peak,
    causing the upsampled patch to miss the true maximum.

    Parameters
    ----------
    cross_corr : torch.Tensor
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
    dtype = cross_corr.dtype
    val_center = cross_corr[batch_idx, peak_row, peak_col]
    val_row_m1 = cross_corr[batch_idx, (peak_row - 1) % num_rows, peak_col]
    val_row_p1 = cross_corr[batch_idx, (peak_row + 1) % num_rows, peak_col]
    val_col_m1 = cross_corr[batch_idx, peak_row, (peak_col - 1) % num_cols]
    val_col_p1 = cross_corr[batch_idx, peak_row, (peak_col + 1) % num_cols]
    # peak_row/col are int from argmax - cast to float for sub-pixel addition.
    # Double modulo handles tiny negative offsets from float32 rounding
    # that would otherwise wrap to N instead of 0 (e.g. -4e-8 % 64 = 64.0)
    refined_row = ((peak_row.to(dtype) + _parabolic_sub_pixel(val_row_m1, val_center, val_row_p1)) % num_rows) % num_rows
    refined_col = ((peak_col.to(dtype) + _parabolic_sub_pixel(val_col_m1, val_center, val_col_p1)) % num_cols) % num_cols
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
