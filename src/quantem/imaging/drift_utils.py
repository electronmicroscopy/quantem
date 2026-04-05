"""Torch helper functions for GPU-accelerated drift correction.

Default dtype is float32 for CUDA/MPS/CPU portability.
Float64 is supported when the caller passes float64 tensors.
"""

import math

import numpy as np
import torch
from torch.fft import fft2, fftfreq, ifft2, ifftshift


def transform_coordinates_torch(
    knots: torch.Tensor,
    scan_fast: torch.Tensor,
    input_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute canvas-space coordinates from Bezier knots (single-knot only)

    For single-knot drift correction, each input row maps to a line on the
    canvas defined by ``knot_position + u * scan_fast * (width - 1)``.
    This is the torch equivalent of ``DriftInterpolator.transform_coordinates``
    for the ``number_knots=1`` case.

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
    >>> r, c = transform_coordinates_torch(knots, scan_fast, (64, 64))
    >>> r.shape
    torch.Size([64, 64])
    """
    num_rows, num_cols = input_shape
    u = torch.linspace(0, 1, num_cols, dtype=knots.dtype, device=knots.device)
    row_coords = knots[0, :, 0:1] + u[None, :] * scan_fast[0] * (num_rows - 1)
    col_coords = knots[1, :, 0:1] + u[None, :] * scan_fast[1] * (num_cols - 1)
    return row_coords, col_coords


def symmetric_pad(
    images: torch.Tensor,
    pad_rows: int,
    pad_cols: int,
) -> torch.Tensor:
    """Symmetric padding to match scipy's boundary handling in drift correction.

    The bilinear KDE scatter step can place pixels near canvas edges, and
    Gaussian smoothing of the scattered counts/values must handle boundaries
    identically to the NumPy path (``scipy.ndimage.gaussian_filter``). Scipy
    uses ``mode='reflect'`` which repeats the edge pixel (``[1,2,3]`` →
    ``[2,1,1,2,3,3,2]``), but PyTorch's ``F.pad(mode='reflect')`` does NOT
    repeat the edge (``[1,2,3]`` → ``[3,2,1,2,3,2,1]``). This function
    reproduces scipy's behavior so the torch and numpy paths match.

    Parameters
    ----------
    images : torch.Tensor
        Input tensor of shape ``(N, C, H, W)``.
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
    >>> symmetric_pad(t, 0, 2)
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


def gaussian_smooth_batch(
    images: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Batched 2D Gaussian smoothing matching ``scipy.ndimage.gaussian_filter``.

    Standard separable Gaussian filter with symmetric (scipy ``reflect``)
    boundary padding. Used internally by ``bilinear_kde_batch_torch`` to
    smooth scattered counts and values before normalization.

    Parameters
    ----------
    images : torch.Tensor
        Input tensor of shape ``(N, H, W)``.
    sigma : float
        Standard deviation of the Gaussian kernel in pixels.

    Returns
    -------
    torch.Tensor
        Smoothed tensor of shape ``(N, H, W)``.

    Examples
    --------
    >>> imgs = torch.randn(2, 32, 32, dtype=torch.float64, device="cpu")
    >>> out = gaussian_smooth_batch(imgs, sigma=0.5)
    >>> out.shape
    torch.Size([2, 32, 32])
    """
    # BSBL: +0.5 rounds to nearest int, matching scipy's truncate=4.0 kernel size
    radius = int(4 * sigma + 0.5)
    t = torch.arange(-radius, radius + 1, dtype=images.dtype, device=images.device)
    kernel_1d = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    images = images[:, None]  # (N, 1, H, W)

    # Horizontal pass
    images = symmetric_pad(images, 0, radius)
    k_col = kernel_1d[None, None, None, :]
    images = torch.nn.functional.conv2d(images, k_col)

    # Vertical pass
    images = symmetric_pad(images, radius, 0)
    k_row = kernel_1d[None, None, :, None]
    images = torch.nn.functional.conv2d(images, k_row)

    return images[:, 0]


def dft_upsample_torch(
    cross_corr_fft: torch.Tensor,
    upsample_factor: int,
    shift: tuple[float, float],
) -> torch.Tensor:
    """Upsampled DFT around a correlation peak for sub-pixel shift estimation.

    Used in the per-image translation alignment step (``_warp_and_translate_torch``)
    where images are processed sequentially, not batched. The coarse FFT
    cross-correlation gives integer-pixel peaks; this function evaluates
    the DFT at arbitrary sub-pixel positions to refine the shift.

    Guizar-Sicairos matrix-multiply DFT: ``K_row @ CC @ K_col`` where
    ``K[i,j] = exp(-2πi·coord_i·freq_j / (N·upsample_factor))``.
    Produces a small ``(2*du+1, 2*du+1)`` real-valued patch.

    Parameters
    ----------
    cross_corr_fft : torch.Tensor
        Complex 2D cross-correlation in Fourier domain, shape ``(M, N)``.
    upsample_factor : int
        Upsampling factor (typically 8).
    shift : tuple[float, float]
        Coarse peak location ``(row, col)`` in pixel units.

    Returns
    -------
    torch.Tensor
        Real-valued upsampled correlation patch.

    Examples
    --------
    >>> F = fft2(torch.randn(64, 64, dtype=torch.float64))
    >>> cc = F * F.conj()
    >>> patch = dft_upsample_torch(cc, 8, (0.0, 0.0))
    >>> patch.shape
    torch.Size([25, 25])
    """
    num_rows, num_cols = cross_corr_fft.shape
    # BSBL: infer real dtype from input — complex64→float32, complex128→float64
    real_dtype = torch.float32 if cross_corr_fft.dtype == torch.complex64 else torch.float64
    # BSBL: 1.5x radius ensures the upsampled patch extends far enough to
    # capture the true peak even after parabolic refinement shifts it off-center
    du = int(np.ceil(1.5 * upsample_factor))

    row_coords = torch.arange(-du, du + 1, dtype=real_dtype, device=cross_corr_fft.device)
    col_coords = torch.arange(-du, du + 1, dtype=real_dtype, device=cross_corr_fft.device)

    # BSBL: center the DFT window on the correlation peak by shifting
    # from pixel coordinates to centered-FFT coordinates
    row_shift = shift[0] - num_rows // 2
    col_shift = shift[1] - num_cols // 2

    freq_row = (
        ifftshift(
            torch.arange(num_rows, dtype=real_dtype, device=cross_corr_fft.device)
        )
        - num_rows // 2
        + row_shift
    )
    freq_col = (
        ifftshift(
            torch.arange(num_cols, dtype=real_dtype, device=cross_corr_fft.device)
        )
        - num_cols // 2
        + col_shift
    )

    kern_row = torch.exp(
        -2j
        * math.pi
        / (num_rows * upsample_factor)
        * row_coords[:, None]
        * freq_row[None, :]
    ).to(cross_corr_fft.dtype)
    kern_col = torch.exp(
        -2j
        * math.pi
        / (num_cols * upsample_factor)
        * freq_col[:, None]
        * col_coords[None, :]
    ).to(cross_corr_fft.dtype)

    return (kern_row @ cross_corr_fft @ kern_col).real


def dft_upsample_batch_torch(
    cross_corr_fft: torch.Tensor,
    upsample_factor: int,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Sub-pixel peak refinement for all drift candidates in one pass.

    After the coarse FFT cross-correlation identifies integer-pixel peaks,
    this function zooms into a small neighborhood around each peak using
    the Guizar-Sicairos matrix-multiply DFT. The grid search evaluates
    tens of candidates simultaneously, so this batched version avoids
    a Python loop over candidates.

    Vectorized ``dft_upsample_torch``: all N candidates share the base
    frequency axes and differ only in per-candidate peak shifts.

    Parameters
    ----------
    cross_corr_fft : torch.Tensor
        Complex 2D cross-correlation in Fourier domain, shape ``(N, M, K)``.
    upsample_factor : int
        Upsampling factor (typically 8).
    shifts : torch.Tensor
        Coarse peak locations ``(row, col)`` per candidate, shape ``(N, 2)``.

    Returns
    -------
    torch.Tensor
        Real-valued upsampled correlation patches, shape ``(N, P, P)``
        where ``P = 2 * ceil(1.5 * upsample_factor) + 1``.

    Examples
    --------
    >>> F = fft2(torch.randn(5, 64, 64, dtype=torch.float64))
    >>> cc = F * F.conj()
    >>> shifts = torch.zeros(5, 2, dtype=torch.float64)
    >>> patches = dft_upsample_batch_torch(cc, 8, shifts)
    >>> patches.shape
    torch.Size([5, 25, 25])
    """
    num_candidates, num_rows, num_cols = cross_corr_fft.shape
    real_dtype = torch.float32 if cross_corr_fft.dtype == torch.complex64 else torch.float64
    du = int(np.ceil(1.5 * upsample_factor))
    patch_coords = torch.arange(-du, du + 1, dtype=real_dtype, device=cross_corr_fft.device)

    # Base frequency axes (shared across candidates)
    freq_row_base = (
        ifftshift(
            torch.arange(num_rows, dtype=real_dtype, device=cross_corr_fft.device)
        )
        - num_rows // 2
    )
    freq_col_base = (
        ifftshift(
            torch.arange(num_cols, dtype=real_dtype, device=cross_corr_fft.device)
        )
        - num_cols // 2
    )

    # Per-candidate shifted frequencies: (N, num_rows) and (N, num_cols)
    row_shifts = shifts[:, 0] - num_rows // 2  # (N,)
    col_shifts = shifts[:, 1] - num_cols // 2  # (N,)
    freq_row = freq_row_base[None, :] + row_shifts[:, None]  # (N, num_rows)
    freq_col = freq_col_base[None, :] + col_shifts[:, None]  # (N, num_cols)

    # Build batched kernels
    # kern_row: (N, P, num_rows)
    kern_row = torch.exp(
        -2j * math.pi / (num_rows * upsample_factor)
        * patch_coords[None, :, None] * freq_row[:, None, :]
    ).to(cross_corr_fft.dtype)
    # kern_col: (N, num_cols, P)
    kern_col = torch.exp(
        -2j * math.pi / (num_cols * upsample_factor)
        * freq_col[:, :, None] * patch_coords[None, None, :]
    ).to(cross_corr_fft.dtype)

    # Batched matmul: (N,P,M) @ (N,M,K) @ (N,K,P) -> (N,P,P)
    return (kern_row @ cross_corr_fft @ kern_col).real


def bilinear_kde_batch_torch(
    row_coords: torch.Tensor,
    col_coords: torch.Tensor,
    values: torch.Tensor,
    output_shape: tuple[int, int],
    kde_sigma: float,
    pad_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched bilinear KDE for warping images under candidate drift vectors.

    In the affine grid search, each candidate drift vector produces a
    different set of pixel coordinates. This function scatters all source
    pixels onto the output canvas simultaneously for N candidates,
    producing N warped images in one pass. The cross-correlation cost
    is then computed between these warped images to score each candidate.

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
    >>> out, weights = bilinear_kde_batch_torch(rows, cols, vals, (40, 40), 0.5, 0.0)
    >>> out.shape
    torch.Size([5, 40, 40])
    """
    num_candidates = row_coords.shape[0]
    num_rows, num_cols = output_shape
    # BSBL: minimum accumulated bilinear weight below which a pixel is
    # considered uncovered and filled with pad_value instead
    threshold = 1e-3

    # int32 indices halve memory bandwidth for index arithmetic (1.4x faster,
    # 3.2 GB less VRAM at chunk=48). Verified: scatter_add_ with int32 produces
    # identical candidate ranking and same winner as int64 on Amy's 2048x2048 data.
    # Safe: num_candidates * num_rows * num_cols < 2^31 for any realistic canvas.
    row_floor = row_coords.reshape(num_candidates, -1).floor().int()
    col_floor = col_coords.reshape(num_candidates, -1).floor().int()
    d_row = row_coords.reshape(num_candidates, -1).float() - row_floor.float()
    d_col = col_coords.reshape(num_candidates, -1).float() - col_floor.float()
    pv_flat = values.reshape(-1).float().expand(num_candidates, -1).reshape(-1)

    candidate_offset = (
        torch.arange(num_candidates, device=row_coords.device, dtype=torch.int32)
        * num_rows * num_cols
    )[:, None]

    pix_count = torch.zeros(
        num_candidates * num_rows * num_cols, dtype=torch.float32, device=row_coords.device
    )
    pix_output = torch.zeros(
        num_candidates * num_rows * num_cols, dtype=torch.float32, device=row_coords.device
    )

    # Wrapped row/col for periodic boundary, compute each corner inline
    rw = row_floor % num_rows
    cw = col_floor % num_cols
    rn = (rw + 1) % num_rows
    cn = (cw + 1) % num_cols

    # Scatter all 4 bilinear corners — index computed and consumed inline
    for ri, ci, weights in [
        (rw, cw, ((1 - d_row) * (1 - d_col)).reshape(-1)),
        (rn, cw, (d_row * (1 - d_col)).reshape(-1)),
        (rw, cn, ((1 - d_row) * d_col).reshape(-1)),
        (rn, cn, (d_row * d_col).reshape(-1)),
    ]:
        inds = (ri * num_cols + ci + candidate_offset).reshape(-1)
        pix_count.scatter_add_(0, inds, weights)
        pix_output.scatter_add_(0, inds, weights * pv_flat)

    pix_count = pix_count.reshape(num_candidates, num_rows, num_cols)
    pix_output = pix_output.reshape(num_candidates, num_rows, num_cols)

    pix_count = gaussian_smooth_batch(pix_count, kde_sigma)
    pix_output = gaussian_smooth_batch(pix_output, kde_sigma)

    # BSBL: soft blend — linearly ramp from 0 to 1 as coverage grows
    # from 0 to threshold, then saturate. Avoids hard edges at the
    # boundary between covered and uncovered regions.
    coverage_weight = torch.clamp(pix_count / threshold, max=1.0)
    images = pad_value * (1 - coverage_weight) + coverage_weight * (
        pix_output / torch.clamp(pix_count, min=1e-8)
    )

    return images, pix_count


def cross_corr_batch_torch(
    images_ref: torch.Tensor,
    images_moving: torch.Tensor,
    upsample_factor: int,
    max_shift: float | None,
    max_shift_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score candidate drift vectors by cross-correlation alignment cost.

    This is the core cost function of the affine grid search: for each
    candidate drift vector, the caller warps both images via
    ``bilinear_kde_batch_torch``, then this function measures how well
    they align. The candidate with the lowest MAE cost wins.

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
    max_shift : float or None
        Maximum allowed shift in pixels. Correlation peaks beyond
        this distance from the origin are masked.
    max_shift_mask : torch.Tensor or None
        Precomputed boolean mask, shape ``(num_rows, num_cols)``.
        True where pixels should be zeroed. If provided, skips
        recomputation from ``max_shift``.

    Returns
    -------
    torch.Tensor
        MAE cost per pair, shape ``(N,)``.

    Examples
    --------
    >>> ref = torch.randn(5, 64, 64, dtype=torch.float64)
    >>> mov = torch.randn(5, 64, 64, dtype=torch.float64)
    >>> cost = cross_corr_batch_torch(ref, mov, 8, 32.0)
    >>> cost.shape
    torch.Size([5])
    """
    num_candidates, num_rows, num_cols = images_ref.shape
    dtype = images_ref.dtype

    # BSBL: fft_ref is inlined — no reason to store it since it's
    # only used once to compute cross_corr
    fft_moving = fft2(images_moving)
    cross_corr = fft2(images_ref) * fft_moving.conj()
    cross_corr_real = ifft2(cross_corr).real

    # BSBL: passing d=1/num_rows makes fftfreq return values in pixel
    # units (0, 1, 2, ...) instead of normalized frequencies (0, 1/N, ...)
    # so dist_sq is directly comparable to max_shift in pixels
    if max_shift_mask is not None:
        cross_corr_real[:, max_shift_mask] = 0.0
    elif max_shift is not None:
        freq_row = fftfreq(
            num_rows, 1.0 / num_rows, device=images_ref.device, dtype=dtype
        )
        freq_col = fftfreq(
            num_cols, 1.0 / num_cols, device=images_ref.device, dtype=dtype
        )
        dist_sq = freq_row[:, None] ** 2 + freq_col[None, :] ** 2
        cross_corr_real[:, dist_sq >= max_shift**2] = 0.0

    # Vectorized coarse peak
    peak_flat_idx = cross_corr_real.reshape(num_candidates, -1).argmax(dim=1)
    peak_row = peak_flat_idx // num_cols
    peak_col = peak_flat_idx % num_cols
    batch_idx = torch.arange(num_candidates, device=images_ref.device)

    # BSBL: parabolic refinement first gives a sub-pixel coarse estimate
    # so the DFT upsample window is centered closer to the true peak,
    # avoiding edge-of-patch artifacts in the upsampled region
    val_row_m1 = cross_corr_real[batch_idx, (peak_row - 1) % num_rows, peak_col]
    val_row_0 = cross_corr_real[batch_idx, peak_row, peak_col]
    val_row_p1 = cross_corr_real[batch_idx, (peak_row + 1) % num_rows, peak_col]
    val_col_m1 = cross_corr_real[batch_idx, peak_row, (peak_col - 1) % num_cols]
    val_col_0 = val_row_0
    val_col_p1 = cross_corr_real[batch_idx, peak_row, (peak_col + 1) % num_cols]

    denom_row = 4 * val_row_0 - 2 * val_row_p1 - 2 * val_row_m1
    denom_col = 4 * val_col_0 - 2 * val_col_p1 - 2 * val_col_m1
    sub_row = torch.where(
        denom_row != 0,
        (val_row_p1 - val_row_m1) / denom_row,
        torch.zeros_like(denom_row),
    )
    sub_col = torch.where(
        denom_col != 0,
        (val_col_p1 - val_col_m1) / denom_col,
        torch.zeros_like(denom_col),
    )

    refined_row = (peak_row.to(dtype) + sub_row) % num_rows
    refined_col = (peak_col.to(dtype) + sub_col) % num_cols

    # BSBL: cross_corr_real is only needed for coarse peak + parabolic
    # refinement — free it to reclaim 8 bytes/pixel per candidate
    del cross_corr_real

    # DFT upsample — batched across all candidates
    shifts = torch.zeros(num_candidates, 2, dtype=dtype, device=images_ref.device)
    # BSBL: upsample_factor=1 means skip DFT refinement — only use
    # the parabolic estimate (integer + fractional pixel precision)
    if upsample_factor > 1:
        locals_batch = dft_upsample_batch_torch(
            cross_corr, upsample_factor,
            torch.stack([refined_row, refined_col], dim=1),
        )
        # BSBL: cross_corr (complex128) is only needed for DFT upsample —
        # free it to reclaim 16 bytes/pixel per candidate before Fourier shift
        del cross_corr
        patch_size = locals_batch.shape[1]

        # Vectorized peak on upsampled patches
        pk_flat = locals_batch.reshape(num_candidates, -1).argmax(dim=1)
        local_row = pk_flat // patch_size
        local_col = pk_flat % patch_size

        # BSBL: parabolic refinement reads 3-point stencil (peak-1, peak, peak+1),
        # so peaks on the patch boundary cannot be refined — fall back to
        # integer-upsampled position to avoid out-of-bounds indexing
        can_refine = (
            (local_row >= 1) & (local_row < patch_size - 1)
            & (local_col >= 1) & (local_col < patch_size - 1)
        )

        # BSBL: clamp neighbor indices to valid range before indexing —
        # can_refine zeroes out edge-peak results, but the indexing itself
        # must not go OOB or CUDA triggers a device-side assert
        lr_m1 = torch.clamp(local_row - 1, min=0)
        lr_p1 = torch.clamp(local_row + 1, max=patch_size - 1)
        lc_m1 = torch.clamp(local_col - 1, min=0)
        lc_p1 = torch.clamp(local_col + 1, max=patch_size - 1)

        val_r_m1 = locals_batch[batch_idx, lr_m1, local_col]
        val_r_0 = locals_batch[batch_idx, local_row, local_col]
        val_r_p1 = locals_batch[batch_idx, lr_p1, local_col]
        val_c_m1 = locals_batch[batch_idx, local_row, lc_m1]
        val_c_0 = val_r_0
        val_c_p1 = locals_batch[batch_idx, local_row, lc_p1]

        dr_denom = 4 * val_r_0 - 2 * val_r_p1 - 2 * val_r_m1
        dc_denom = 4 * val_c_0 - 2 * val_c_p1 - 2 * val_c_m1
        d_row_fine = torch.where(
            can_refine & (dr_denom != 0),
            (val_r_p1 - val_r_m1) / dr_denom,
            torch.zeros_like(dr_denom),
        )
        d_col_fine = torch.where(
            can_refine & (dc_denom != 0),
            (val_c_p1 - val_c_m1) / dc_denom,
            torch.zeros_like(dc_denom),
        )

        # BSBL: convert from upsampled-patch coordinates back to image pixels:
        # refined_row = coarse position in pixels
        # (local_row - upsample_factor) / upsample_factor = offset from patch center
        # d_row_fine / upsample_factor = sub-pixel correction within upsampled grid
        shifts[:, 0] = (
            refined_row
            + (local_row.to(dtype) - upsample_factor) / upsample_factor
            + d_row_fine / upsample_factor
        )
        shifts[:, 1] = (
            refined_col
            + (local_col.to(dtype) - upsample_factor) / upsample_factor
            + d_col_fine / upsample_factor
        )
        # Wrap shifts into [-N/2, N/2) range (centered convention)
        shifts[:, 0] = ((shifts[:, 0] + num_rows / 2) % num_rows) - num_rows / 2
        shifts[:, 1] = ((shifts[:, 1] + num_cols / 2) % num_cols) - num_cols / 2
    else:
        del cross_corr
        shifts[:, 0] = ((refined_row + num_rows / 2) % num_rows) - num_rows / 2
        shifts[:, 1] = ((refined_col + num_cols / 2) % num_cols) - num_cols / 2

    # Apply sub-pixel shift in Fourier domain (exact, no interpolation artifacts)
    freq_row = fftfreq(
        num_rows, device=images_ref.device, dtype=dtype
    )[:, None]
    freq_col = fftfreq(
        num_cols, device=images_ref.device, dtype=dtype
    )[None, :]
    phase = -2j * torch.pi * (
        freq_row[None] * shifts[:, 0, None, None]
        + freq_col[None] * shifts[:, 1, None, None]
    )
    images_aligned = ifft2(fft_moving * torch.exp(phase)).real

    cost = torch.mean(torch.abs(images_ref - images_aligned), dim=(1, 2))
    return cost
