"""Warping primitives: cross-correlation translation + backward resampling.

These are the fused torch operations used after the knot's forward warp has
put scan images on the same canvas. Forward warping itself lives in
``core.knots``; these functions run once the warped images need alignment and the learned drift
needs to be applied to raw data.

Contains:

* :func:`cross_corr_batch` + :func:`translate_align` — sub-pixel translation
  alignment from FFT cross-correlation peaks (and the candidate scoring
  signal for ``correct_affine``'s grid search).
* :func:`backward_warp` + :func:`backward_warp_grid_search` — bicubic
  backward resampling that applies the learned drift to raw data, and
  the affine candidate scoring loop that bypasses canvas KDE.
* Private DFT-upsample + parabolic-peak helpers used by the above.
"""

import math

import numpy as np
import torch
from torch.fft import fft2, fftfreq, ifft2, ifftshift
from tqdm.auto import tqdm

from quantem.imaging.drift.core import knots as drift_knots


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
    >>> cost = cross_corr_batch(ref, mov, 8)
    >>> cost.shape
    torch.Size([5])
    """
    _, num_rows, num_cols = ref_images.shape
    dtype = ref_images.dtype
    mov_fft = fft2(mov_images)
    cross_corr_fft = fft2(ref_images) * mov_fft.conj()
    image_shifts = _translation_from_cross_correlation(
        cross_corr_fft, upsample_factor, max_shift_mask
    )
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


def fixed_overlap_ncc(
    ref_images: torch.Tensor,
    mov_images: torch.Tensor,
    scan_shape: tuple[int, int],
    max_image_shift: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank translations without letting padded pixels choose the lattice branch.

    A centered reference crop is compared with equally sized windows from the
    moving scan. Every shift therefore uses the same number of measured pixels,
    and normalized correlation removes intensity-scale differences.
    """
    scan_rows, scan_cols = scan_shape
    canvas_rows, canvas_cols = ref_images.shape[-2:]
    row_start = (canvas_rows - scan_rows) // 2
    col_start = (canvas_cols - scan_cols) // 2
    ref = ref_images[
        :, row_start : row_start + scan_rows, col_start : col_start + scan_cols
    ]
    mov = mov_images[
        :, row_start : row_start + scan_rows, col_start : col_start + scan_cols
    ]

    shift_limit = (
        min(scan_shape) / 4 if max_image_shift is None else float(max_image_shift)
    )
    margin = min(
        max(1, int(math.ceil(shift_limit))),
        (min(scan_shape) - 2) // 2,
    )
    template = ref[:, margin:-margin, margin:-margin]
    template = template - template.mean(dim=(-2, -1), keepdim=True)
    template_rows, template_cols = template.shape[-2:]

    fft_rows = 1 << (scan_rows + template_rows - 2).bit_length()
    fft_cols = 1 << (scan_cols + template_cols - 2).bit_length()
    numerator = torch.fft.irfft2(
        torch.fft.rfft2(mov, s=(fft_rows, fft_cols))
        * torch.fft.rfft2(
            template.flip((-2, -1)), s=(fft_rows, fft_cols)
        ),
        s=(fft_rows, fft_cols),
    )
    numerator = numerator[
        :,
        template_rows - 1 : scan_rows,
        template_cols - 1 : scan_cols,
    ]

    integral = torch.nn.functional.pad(mov, (1, 0, 1, 0))
    integral = integral.cumsum(-2).cumsum(-1)
    integral_sq = torch.nn.functional.pad(mov.square(), (1, 0, 1, 0))
    integral_sq = integral_sq.cumsum(-2).cumsum(-1)

    def window_sum(table):
        return (
            table[:, template_rows:, template_cols:]
            - table[:, :-template_rows, template_cols:]
            - table[:, template_rows:, :-template_cols]
            + table[:, :-template_rows, :-template_cols]
        )

    moving_sum = window_sum(integral)
    moving_sum_sq = window_sum(integral_sq)
    pixels = float(template_rows * template_cols)
    moving_norm = torch.sqrt(
        (moving_sum_sq - moving_sum.square() / pixels).clamp_min(0.0)
    )
    template_norm = template.norm(dim=(-2, -1), keepdim=True)
    ncc = numerator / (template_norm * moving_norm).clamp_min(1e-12)

    shifts = torch.arange(
        -margin,
        margin + 1,
        device=ncc.device,
        dtype=ncc.dtype,
    )
    allowed = shifts[:, None].square() + shifts[None, :].square() <= shift_limit**2
    ncc.masked_fill_(~allowed[None], -torch.inf)
    flat_index = ncc.flatten(1).argmax(dim=1)
    peak_row = flat_index // ncc.shape[-1]
    peak_col = flat_index % ncc.shape[-1]
    batch = torch.arange(ncc.shape[0], device=ncc.device)
    best_ncc = ncc[batch, peak_row, peak_col]
    image_shifts = -torch.stack(
        (peak_row - margin, peak_col - margin), dim=1
    ).to(ncc.dtype)
    return 1.0 - best_ncc, image_shifts, best_ncc - ncc[:, margin, margin]


def translate_align(
    warped_images: torch.Tensor,
    upsample_factor: int,
    max_image_shift: float | None,
) -> torch.Tensor:
    """Pairwise translation alignment of warped images via cross-correlation.

    Called by :func:`warp_and_translate` after each canvas warp to
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
        image_shifts[img_idx] = _translation_from_cross_correlation(
            cross_corr_fft[None], upsample_factor, shift_mask
        )[0]
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


def translate_align_pair_batch(
    warped_pairs: torch.Tensor,
    upsample_factor: int,
    max_image_shift: float | None,
) -> torch.Tensor:
    """Solve translations for a batch of independent two-image pairs.

    This is the candidate-batched equivalent of :func:`translate_align` for
    the common two-scan case. It preserves the same FFT, parabolic peak, DFT
    refinement, shift bounds, and zero-mean convention while avoiding a
    Python-level full-canvas solve for every affine validation candidate.

    Parameters
    ----------
    warped_pairs : torch.Tensor
        Candidate image pairs, shape ``(N, 2, H, W)``.
    upsample_factor : int
        Sub-pixel precision (1/N pixel).
    max_image_shift : float or None
        Maximum allowed translational shift in pixels.

    Returns
    -------
    torch.Tensor
        Zero-mean shifts with shape ``(N, 2, 2)`` in ``(row, col)`` order.
    """
    if warped_pairs.ndim != 4 or warped_pairs.shape[1] != 2:
        raise ValueError(
            "warped_pairs must have shape (N, 2, H, W); got "
            f"{tuple(warped_pairs.shape)}."
        )
    num_pairs, _, num_rows, num_cols = warped_pairs.shape
    dtype = warped_pairs.dtype
    device = warped_pairs.device
    ref_fft = fft2(warped_pairs[:, 0])
    mov_fft = fft2(warped_pairs[:, 1])
    cross_corr_fft = ref_fft * mov_fft.conj()
    shift_mask = None
    if max_image_shift is not None:
        dist_row = fftfreq(
            num_rows, 1.0 / num_rows, device=device, dtype=dtype,
        )
        dist_col = fftfreq(
            num_cols, 1.0 / num_cols, device=device, dtype=dtype,
        )
        shift_mask = (
            dist_row[:, None] ** 2 + dist_col[None, :] ** 2
            >= max_image_shift**2
        )
    pair_shift = _translation_from_cross_correlation(
        cross_corr_fft, upsample_factor, shift_mask
    )
    shifts = torch.zeros(
        num_pairs, 2, 2, dtype=dtype, device=device,
    )
    shifts[:, 1] = pair_shift
    shifts -= shifts.mean(dim=1, keepdim=True)
    return shifts


@torch.inference_mode()
def warp_and_translate(
    correction,
    max_image_shift: float | None,
    upsample_factor: int = 8,
    knots_batch: torch.Tensor | None = None,
    solve_translation: bool = True,
    fixed_indices: frozenset[int] | None = None,
    imgs_t_override: list[torch.Tensor] | None = None,
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Warp scans to their shared canvas and remove residual translation.

    Affine, strip, and non-rigid stages all need the same scientific operation:
    render the current scan geometry, estimate the remaining rigid offset, move
    the knots, and render once more. Keeping that operation here prevents the
    affine search from owning general correction mechanics.
    """
    device = correction._device
    dtype = correction._dtype
    num_images = correction.shape[0]
    canvas_shape = (correction.shape[1], correction.shape[2])
    fixed_set = fixed_indices if fixed_indices else frozenset()
    imgs_t = imgs_t_override if imgs_t_override is not None else correction.imgs_t
    # AutoSerialize restores tensors on the host so one archive can be opened
    # on CUDA, MPS, or CPU. Move the small alignment images at the shared warp
    # boundary; callers should not need to repair a reloaded correction before
    # requesting a report, coverage mask, or figure.
    imgs_t = [image.to(device=device, dtype=dtype) for image in imgs_t]
    if imgs_t_override is None:
        correction.imgs_t = imgs_t
    if knots_batch is None:
        correction.knots = [
            knots.to(device=device, dtype=dtype) for knots in correction.knots
        ]
    else:
        knots_batch = knots_batch.to(device=device, dtype=dtype)

    def render(warped_t, weights_t):
        for img_idx in range(num_images):
            knots_img = (
                knots_batch[img_idx].detach() if knots_batch is not None else None
            )
            warped, image_weights = drift_knots.interpolator(
                correction, img_idx, knots_img
            ).warp_to_canvas(
                imgs_t[img_idx],
                canvas_shape,
                correction.kde_sigma,
                correction.pad_value[img_idx],
            )
            warped_t[img_idx] = warped
            weights_t[img_idx] = image_weights

    warped_t = torch.zeros(num_images, *canvas_shape, dtype=dtype, device=device)
    weights_t = torch.zeros_like(warped_t)
    render(warped_t, weights_t)
    if not solve_translation:
        if knots_batch is None:
            correction.imgs_warped.array[:] = warped_t.cpu().numpy()
        return (warped_t, weights_t) if return_weights else warped_t

    shifts_t = translate_align(warped_t, upsample_factor, max_image_shift)
    if fixed_set:
        fixed_idx_list = sorted(fixed_set)
        anchor = shifts_t[fixed_idx_list].mean(0)
        shifts_t -= anchor
        for idx in fixed_set:
            shifts_t[idx] = 0.0

    if knots_batch is not None:
        knots_batch[:, 0] += shifts_t[:, 0, None, None]
        knots_batch[:, 1] += shifts_t[:, 1, None, None]
    else:
        for img_idx in range(num_images):
            correction.knots[img_idx][0] += shifts_t[img_idx, 0]
            correction.knots[img_idx][1] += shifts_t[img_idx, 1]

    render(warped_t, weights_t)
    if knots_batch is None:
        correction.imgs_warped.array[:] = warped_t.cpu().numpy()
    return (warped_t, weights_t) if return_weights else warped_t


def backward_warp(
    images: torch.Tensor,
    drift: tuple[float, float] | torch.Tensor,
    rigid_shift: tuple[float, float] = (0.0, 0.0),
    mode: str = "bilinear",
) -> torch.Tensor:
    """Apply drift correction via backward interpolation (``grid_sample``).

    Builds a per-scanline sampling grid that undoes the estimated drift
    and optional rigid translation, then resamples with the chosen
    interpolation kernel.

    Parameters
    ----------
    images : torch.Tensor
        Images to correct, shape ``(N, H, W)`` or ``(H, W)``.
        For 4D-STEM, pass detector-pixel slices in chunks.
    drift : tuple[float, float] | torch.Tensor
        **Affine mode** — ``(row_slope, col_slope)`` scalar drift rate
        in pixels per scan line, as returned by ``correct_affine``.

        **Tensor mode** — per-row ``(2, H)`` for K=1 (one shift per
        scanline, broadcast across columns) or per-pixel ``(2, H, W)``
        for K>=2 (drift varies along the fast axis).
        :class:`DriftKnot` returns the right shape per K, so
        callers don't materialize the per-pixel tensor when the per-row
        form suffices.  *rigid_shift* is ignored in tensor mode.
    rigid_shift : tuple[float, float], default (0.0, 0.0)
        Global ``(row, col)`` translation to apply (affine mode only).
    mode : str, default "bilinear"
        Interpolation kernel passed to ``grid_sample``.

    Returns
    -------
    torch.Tensor
        Corrected images, same shape as *images*.
    """
    squeeze = images.dim() == 2
    if squeeze:
        images = images[None]
    n, h, w = images.shape
    device, dtype = images.device, images.dtype

    if isinstance(drift, torch.Tensor):
        drift = drift.to(device=device, dtype=dtype)
        if drift.shape == (2, h):
            # Per-row (K=1): broadcast a single shift across columns.
            row_shift = drift[0][:, None]
            col_shift = drift[1][:, None]
        elif drift.shape == (2, h, w):
            # Per-pixel (K>=2): drift varies along fast axis.
            row_shift = drift[0]
            col_shift = drift[1]
        else:
            raise ValueError(
                f"Drift must be (2, {h}) for K=1 or (2, {h}, {w}) for K>=2; "
                f"got {tuple(drift.shape)}")
        sample_r = torch.arange(h, device=device, dtype=dtype)[:, None].expand(-1, w) - row_shift
        sample_c = torch.arange(w, device=device, dtype=dtype)[None, :].expand(h, -1) - col_shift
    else:
        # Affine mode: drift is (row_slope, col_slope)
        offset = torch.arange(h, device=device, dtype=dtype) - (h - 1) / 2
        drift_row, drift_col = drift
        shift_row, shift_col = rigid_shift
        sample_r = (
            torch.arange(h, device=device, dtype=dtype)[:, None].expand(-1, w)
            - drift_row * offset[:, None]
            - shift_row
        )
        sample_c = (
            torch.arange(w, device=device, dtype=dtype)[None, :].expand(h, -1)
            - drift_col * offset[:, None]
            - shift_col
        )

    grid_row = 2.0 * sample_r / (h - 1) - 1.0
    grid_col = 2.0 * sample_c / (w - 1) - 1.0
    # Pass as (1, N, H, W) with a single (1, H, W, 2) grid so grid_sample applies
    # one grid to all N channels in one kernel call. The alternative (N, 1, H, W)
    # with (N, H, W, 2) materialises N identical grids — 4096× more memory for EDS.
    grid = torch.stack([grid_col, grid_row], dim=-1)[None]  # (1, H, W, 2) — col first per grid_sample convention

    out = torch.nn.functional.grid_sample(
        images[None], grid, mode=mode,
        align_corners=True, padding_mode="border",
    )[0]  # (N, H, W)
    return out[0] if squeeze else out


def canvas_center_to_scan(correction, canvas_image) -> np.ndarray:
    """Crop a solver canvas to the acquired scan field of view."""
    array = (
        canvas_image.detach().cpu().numpy()
        if isinstance(canvas_image, torch.Tensor)
        else np.asarray(canvas_image)
    )
    scan_rows, scan_columns = correction.imgs[0].shape[:2]
    row = (array.shape[0] - scan_rows) // 2
    column = (array.shape[1] - scan_columns) // 2
    return np.ascontiguousarray(
        array[row : row + scan_rows, column : column + scan_columns],
        dtype=np.float32,
    )


def co_registered_scan_stack(
    correction,
    *,
    fixed_set: frozenset[int],
    max_image_shift: float | None = 32.0,
    solve_translation: bool = True,
    knots: list[torch.Tensor] | None = None,
) -> list[np.ndarray]:
    """Warp scans into one scan-sized frame for residual measurements."""
    knots_batch = torch.stack(
        [knot.detach() for knot in knots or correction.knots]
    ).to(device=correction._device, dtype=correction._dtype)
    warped = warp_and_translate(
        correction,
        max_image_shift=max_image_shift if solve_translation else None,
        upsample_factor=8,
        knots_batch=knots_batch,
        solve_translation=solve_translation,
        fixed_indices=fixed_set if fixed_set else None,
    )
    if solve_translation:
        with torch.no_grad():
            for index in range(len(correction.knots)):
                correction.knots[index][...] = knots_batch[index]
    return [canvas_center_to_scan(correction, image) for image in warped]


def reference_scan_stack(
    correction,
    knots: list[torch.Tensor] | None = None,
) -> list[np.ndarray]:
    """Warp a moving scan while preserving its external reference frame."""
    reference = np.asarray(correction.imgs[0].array, dtype=np.float32)
    moving = backward_warp(
        correction.imgs_t[1],
        drift=drift_knots.interpolator(
            correction, 1, (knots or correction.knots)[1]
        ).drift_raw(correction._initial_knots[1]),
        mode="bilinear",
    ).detach().cpu().numpy()
    return [
        np.ascontiguousarray(reference),
        np.ascontiguousarray(moving, dtype=np.float32),
    ]


def knot_fingerprint(correction):
    """Summarize current knots for validating the cached warped images."""
    parts = []
    for knot in correction.knots:
        array = (
            knot.detach().cpu().numpy()
            if hasattr(knot, "detach")
            else np.asarray(knot)
        )
        parts.append(
            (array.shape, float(array.sum()), float(np.abs(array).max()))
        )
    return tuple(parts)


def ensure_warped_images(correction):
    """Keep the displayed warped stack synchronized with the solved knots."""
    fingerprint = knot_fingerprint(correction)
    if (
        not getattr(correction, "_images_warped_stale", True)
        and getattr(correction, "_warped_fingerprint", None) == fingerprint
    ):
        return

    if getattr(correction, "_reference_mode", False):
        scans = np.stack(reference_scan_stack(correction)).astype(
            np.float32, copy=False
        )
        canvas_rows, canvas_columns = correction.shape[-2:]
        scan_rows, scan_columns = scans.shape[-2:]
        row = (canvas_rows - scan_rows) // 2
        column = (canvas_columns - scan_columns) // 2
        for index in range(scans.shape[0]):
            correction.imgs_warped.array[index].fill(
                float(correction.pad_value[index])
            )
            correction.imgs_warped.array[
                index,
                row : row + scan_rows,
                column : column + scan_columns,
            ] = scans[index]
    else:
        warp_and_translate(
            correction,
            getattr(correction, "_max_image_shift_cached", None),
            upsample_factor=8,
            solve_translation=False,
        )
    correction._images_warped_stale = False
    correction._warped_fingerprint = knot_fingerprint(correction)




@torch.inference_mode()
def backward_warp_grid_search(
    ref_image: torch.Tensor,
    mov_image: torch.Tensor,
    drift_vectors: torch.Tensor,
    upsample_factor: int,
    max_image_shift: float | None,
    chunk_size: int | None = None,
    progress_desc: str | None = None,
) -> tuple[int, torch.Tensor]:
    """Score drift candidates by backward-warping the moving image.

    For each candidate ``(dr, dc)``, builds a sampling grid that undoes the
    drift::

        sample_row[i, j] = i - dr * (i - center)
        sample_col[i, j] = j - dc * (i - center)

    then backward-warps the moving image with ``grid_sample`` (bicubic) and
    scores alignment with the reference via ``cross_corr_batch``.

    This avoids forward-scatter KDE artifacts that bias the cost when only
    one image's geometry changes (``fixed_indices`` mode).  The periodic
    wrapping in ``bilinear_kde_batch`` creates geometry-dependent seam
    artifacts that differ between the fixed reference and the drift-shifted
    moving image, making the MAE minimum diverge from the true drift.
    Backward-warp scoring eliminates this by working at original resolution
    without any canvas or KDE.

    Parameters
    ----------
    ref_image : torch.Tensor
        Reference image, shape ``(H, W)``.
    mov_image : torch.Tensor
        Moving image to correct, shape ``(H, W)``.
    drift_vectors : torch.Tensor
        Candidate drift rates, shape ``(N, 2)`` — columns ``(row_rate, col_rate)``.
    upsample_factor : int
        Sub-pixel precision for cross-correlation refinement.
    max_image_shift : float or None
        Maximum allowed translational shift in pixels.
    chunk_size : int or None
        Candidates per GPU pass.  ``None`` auto-selects based on free memory.
    progress_desc : str or None
        Description for a progress bar shown when the search needs multiple
        chunks. ``None`` disables progress reporting.

    Returns
    -------
    tuple[int, torch.Tensor]
        Index of the best candidate and full cost tensor of shape ``(N,)``.
    """
    device = ref_image.device
    dtype = ref_image.dtype
    h, w = ref_image.shape
    num_candidates = drift_vectors.shape[0]
    center = (h - 1) / 2.0

    rows = torch.arange(h, device=device, dtype=dtype)
    cols = torch.arange(w, device=device, dtype=dtype)
    offset = rows - center  # (H,)

    shift_mask = None
    if max_image_shift is not None:
        dist_r = fftfreq(h, 1.0 / h, device=device, dtype=dtype)
        dist_c = fftfreq(w, 1.0 / w, device=device, dtype=dtype)
        shift_mask = dist_r[:, None] ** 2 + dist_c[None, :] ** 2 >= max_image_shift ** 2
    freq_grids = (
        fftfreq(h, device=device, dtype=dtype)[:, None],
        fftfreq(w, device=device, dtype=dtype)[None, :],
    )

    if chunk_size is None:
        if device.type == "cuda":
            bytes_per_element = torch.finfo(dtype).bits // 8
            # grid (H*W*2) + warped (H*W) + FFT buffers (H*W*8*2)
            per_cand_bytes = h * w * bytes_per_element * (2 + 1 + 16)
            free_bytes, _ = torch.cuda.mem_get_info(device)
            chunk_size = max(1, int(free_bytes * 0.4 / per_cand_bytes))
            chunk_size = min(chunk_size, num_candidates)
        else:
            chunk_size = num_candidates

    all_costs = []
    chunk_starts = range(0, num_candidates, chunk_size)
    show_progress = progress_desc is not None and len(chunk_starts) > 1
    pbar = tqdm(
        total=num_candidates,
        desc=progress_desc,
        unit="candidate",
        disable=not show_progress,
    )
    try:
        for chunk_start in chunk_starts:
            chunk_end = min(chunk_start + chunk_size, num_candidates)
            n_chunk = chunk_end - chunk_start
            drift_chunk = drift_vectors[chunk_start:chunk_end]

            drift_row_rate = drift_chunk[:, 0]  # (n_chunk,)
            drift_col_rate = drift_chunk[:, 1]  # (n_chunk,)

            row_shift = drift_row_rate[:, None] * offset[None, :]  # (n_chunk, H)
            col_shift = drift_col_rate[:, None] * offset[None, :]  # (n_chunk, H)

            # A candidate that moves most source pixels outside the detector
            # footprint can look deceptively good because ``padding_mode``
            # repeats a nearly constant border.  Reject rates with less than
            # half-field geometric overlap before ranking their image cost.
            sample_row_1d = rows[None] - row_shift
            valid_rows = (
                (sample_row_1d >= 0.0) & (sample_row_1d <= h - 1)
            ).to(dtype)
            lower_col = torch.maximum(
                torch.zeros_like(col_shift), col_shift,
            )
            upper_col = torch.minimum(
                torch.full_like(col_shift, w - 1),
                (w - 1) + col_shift,
            )
            valid_columns = torch.clamp(
                upper_col - lower_col + 1.0, min=0.0, max=float(w),
            )
            overlap_fraction = torch.sum(
                valid_rows * valid_columns, dim=1,
            ) / float(h * w)

            sample_rows = rows[None, :, None].expand(n_chunk, h, w) - row_shift[:, :, None]
            sample_cols = cols[None, None, :].expand(n_chunk, h, w) - col_shift[:, :, None]

            grid_row = 2.0 * sample_rows / (h - 1) - 1.0
            grid_col = 2.0 * sample_cols / (w - 1) - 1.0
            grid = torch.stack(
                [grid_col, grid_row], dim=-1
            )  # col first per grid_sample convention

            warped = torch.nn.functional.grid_sample(
                mov_image[None, None].expand(n_chunk, 1, h, w), grid,
                mode="bicubic", align_corners=True, padding_mode="border",
            )[:, 0]

            ref_batch = ref_image[None].expand(n_chunk, -1, -1)
            costs = cross_corr_batch(
                ref_batch, warped, upsample_factor,
                max_shift_mask=shift_mask, freq_grids=freq_grids,
            )
            costs.masked_fill_(overlap_fraction < 0.5, torch.inf)
            all_costs.append(costs)
            pbar.update(n_chunk)
    finally:
        pbar.close()

    all_costs = torch.cat(all_costs)
    return torch.argmin(all_costs).item(), all_costs


# ---------------------------------------------------------------------------
# Building blocks - used internally by the public API functions above
# ---------------------------------------------------------------------------


def _translation_from_cross_correlation(
    cross_corr_fft: torch.Tensor,
    upsample_factor: int,
    max_shift_mask: torch.Tensor | None,
) -> torch.Tensor:
    # Keep every solver on one integer → parabolic → DFT refinement and the
    # same centered (row, col) shift convention.
    num_images, num_rows, num_cols = cross_corr_fft.shape
    cross_corr = ifft2(cross_corr_fft).real
    if max_shift_mask is not None:
        cross_corr.masked_fill_(max_shift_mask[None], 0.0)
    peak_flat_idx = cross_corr.flatten(1).argmax(dim=1)
    peak_row = peak_flat_idx // num_cols
    peak_col = peak_flat_idx % num_cols
    batch_idx = torch.arange(num_images, device=cross_corr_fft.device)
    refined_row, refined_col = _parabolic_peak_2d(
        cross_corr, peak_row, peak_col, num_rows, num_cols, batch_idx
    )
    shifts = _dft_refine_shifts(
        cross_corr_fft, refined_row, refined_col, upsample_factor
    )
    shifts[:, 0] = ((shifts[:, 0] + num_rows / 2) % num_rows) - num_rows / 2
    shifts[:, 1] = ((shifts[:, 1] + num_cols / 2) % num_cols) - num_cols / 2
    return shifts


def _dft_refine_shifts(
    cross_corr_fft: torch.Tensor,
    peak_row: torch.Tensor,
    peak_col: torch.Tensor,
    upsample_factor: int,
) -> torch.Tensor:
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

# ---------------------------------------------------------------------------
# Primitives - lowest-level operations
# ---------------------------------------------------------------------------


def _parabolic_peak_2d(
    cross_corr: torch.Tensor,
    peak_row: torch.Tensor,
    peak_col: torch.Tensor,
    num_rows: int,
    num_cols: int,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _parabolic_sub_pixel(
    val_m1: torch.Tensor,
    val_0: torch.Tensor,
    val_p1: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
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
