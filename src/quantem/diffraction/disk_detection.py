from __future__ import annotations

import numpy as np
import torch

SUBPIXEL_MODES = ("none", "parabolic", "upsample")


def make_template(
    probe: torch.Tensor,
    center: tuple[float, float] | None = None,
    subtract_mean: bool = False,
) -> torch.Tensor:
    """Build a cross-correlation template from a (vacuum) probe image.

    The probe is normalized to unit sum and rolled so its center sits at the array
    origin ``[0, 0]`` (FFT corner), so correlation peaks land at absolute disk
    positions.

    Parameters
    ----------
    probe : torch.Tensor
        ``(H, W)`` probe / vacuum disk image.
    center : tuple of float, optional
        ``(row, col)`` probe center rolled to the origin; defaults to the geometric
        center ``(H // 2, W // 2)``.
    subtract_mean : bool, default=False
        If ``True``, make the template zero-sum — a band-pass kernel that suppresses
        the uniform background in the correlation.

    Returns
    -------
    torch.Tensor
        ``(H, W)`` template, corner-centered (and zero-sum when ``subtract_mean``).
    """
    probe = torch.as_tensor(probe)
    total = probe.sum()
    if total != 0:
        probe = probe / total

    H, W = probe.shape
    if center is None:
        cr, cc = H // 2, W // 2
    else:
        cr, cc = int(round(float(center[0]))), int(round(float(center[1])))

    template = torch.roll(probe, shifts=(-cr, -cc), dims=(0, 1))
    if subtract_mean:
        template = template - template.mean()
    return template


def synthetic_probe(
    shape: tuple[int, int],
    radius: float,
    edge: float = 1.0,
    center: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Soft-edged disk for a synthetic correlation template.

    Returns ``0.5 - 0.5*tanh((r - radius)/edge)``: a disk of the given ``radius``
    (pixels) with a ``tanh`` falloff over ``edge`` pixels.

    Parameters
    ----------
    shape : tuple of int
        ``(H, W)`` output shape in pixels.
    radius : float
        Disk radius in pixels.
    edge : float, default=1.0
        Width in pixels of the ``tanh`` edge falloff.
    center : tuple of float, optional
        ``(row, col)`` disk center; defaults to the geometric center
        ``((H - 1) / 2, (W - 1) / 2)``.

    Returns
    -------
    torch.Tensor
        ``(H, W)`` soft-edged disk image.
    """
    H, W = int(shape[0]), int(shape[1])
    if center is None:
        cr, cc = (H - 1) / 2.0, (W - 1) / 2.0
    else:
        cr, cc = float(center[0]), float(center[1])
    rows = torch.arange(H, dtype=torch.float).view(H, 1)
    cols = torch.arange(W, dtype=torch.float).view(1, W)
    rr = torch.sqrt((rows - cr) ** 2 + (cols - cc) ** 2)
    edge = max(float(edge), 1e-6)
    return 0.5 - 0.5 * torch.tanh((rr - float(radius)) / edge)


def _central_blob(image: torch.Tensor, threshold: float) -> tuple[np.ndarray | None, np.ndarray]:
    """Mask of the connected bright region containing the brightest pixel.

    Thresholds ``image`` at ``threshold * max`` and keeps only the connected
    component holding the brightest pixel — normally the central (unscattered)
    disk of a mean diffraction pattern or vacuum probe — so other diffracted disks
    are excluded.

    Parameters
    ----------
    image : torch.Tensor
        ``(H, W)`` image, e.g. a mean diffraction pattern or vacuum probe.
    threshold : float
        Fraction of the peak intensity (after min-subtraction) used to threshold
        the image before connected-component labeling.

    Returns
    -------
    blob_mask : np.ndarray or None
        Boolean ``(H, W)`` mask of the central component, or ``None`` for an empty
        / flat image.
    img : np.ndarray
        The min-subtracted image.
    """
    from scipy import ndimage

    img = np.asarray(torch.as_tensor(image, dtype=torch.float).detach().cpu())
    img = img - img.min()
    peak = float(img.max())
    if peak <= 0:
        return None, img
    labels, n = ndimage.label(img >= threshold * peak)
    if n == 0:
        return None, img
    peak_label = int(labels[np.unravel_index(int(np.argmax(img)), img.shape)])
    return labels == peak_label, img


def estimate_central_beam(
    image: torch.Tensor,
    threshold: float = 0.5,
    plot_result: bool = False,
    **kwargs,
) -> tuple[tuple[float, float], float]:
    """Center ``(row, col)`` and radius (pixels) of the central (direct) beam.

    Locates the connected bright region containing the brightest pixel (see
    :func:`_central_blob`) — the unscattered / direct beam of a mean diffraction
    pattern — and returns its intensity-weighted center together with an
    area-equivalent radius (``A = pi r^2``). Other diffracted disks are excluded, so
    the estimate holds whether the pattern shows one disk or many. Falls back to the
    geometric center and unit radius for an empty / flat image.

    Parameters
    ----------
    image : torch.Tensor or Dataset2d
        ``(H, W)`` image — a raw array / tensor or a :class:`Dataset2d` (e.g.
        ``dataset.dp_mean``).
    threshold : float, default=0.5
        Fraction of the peak intensity used to threshold the image when isolating
        the central beam (see :func:`_central_blob`).
    plot_result : bool, default=False
        If ``True``, show the image in greyscale with the fitted beam drawn as a red
        circle.
    **kwargs
        Extra keyword arguments (e.g. ``norm``, ``cbar``, ``scalebar``) forwarded to
        :func:`~quantem.core.visualization.show_2d` when ``plot_result=True``.

    Returns
    -------
    center : tuple of float
        ``(row, col)`` intensity-weighted center of the central beam.
    radius : float
        Area-equivalent radius in pixels (``A = pi r^2``).
    """
    arr = image.array if hasattr(image, "array") else image
    blob, img = _central_blob(arr, threshold)
    if blob is None:
        center = (img.shape[0] / 2.0, img.shape[1] / 2.0)
        radius = 1.0
    else:
        radius = float(np.sqrt(max(float(blob.sum()), 1.0) / np.pi))
        w = img * blob
        total = float(w.sum())
        if total <= 0:
            rr, cc = np.nonzero(blob)
            center = (float(rr.mean()), float(cc.mean()))
        else:
            rows = np.arange(img.shape[0])[:, None]
            cols = np.arange(img.shape[1])[None, :]
            center = (float((w * rows).sum() / total), float((w * cols).sum() / total))

    if plot_result:
        from matplotlib.patches import Circle

        from quantem.core.visualization import show_2d

        show_kwargs = {"cmap": "gray", "title": "central beam", **kwargs}
        _fig, ax = show_2d(arr, **show_kwargs)
        ax.add_patch(
            Circle((center[1], center[0]), radius, fill=False, edgecolor="red", linewidth=1.5)
        )

    return center, radius


def probe_centroid(probe: torch.Tensor) -> tuple[float, float]:
    """Intensity-weighted ``(row, col)`` centroid of a probe image.

    Parameters
    ----------
    probe : torch.Tensor
        ``(H, W)`` probe image. Negative values are clamped to zero before
        weighting.

    Returns
    -------
    tuple of float
        ``(row, col)`` intensity-weighted centroid; the geometric center for a
        non-positive image.
    """
    p = torch.clamp(torch.as_tensor(probe, dtype=torch.float), min=0.0)
    total = p.sum()
    if total <= 0:
        return (p.shape[0] / 2.0, p.shape[1] / 2.0)
    rows = torch.arange(p.shape[0], dtype=torch.float, device=p.device).view(-1, 1)
    cols = torch.arange(p.shape[1], dtype=torch.float, device=p.device).view(1, -1)
    return (float((p * rows).sum() / total), float((p * cols).sum() / total))


def template_fourier(template: torch.Tensor) -> torch.Tensor:
    """Pre-compute the conjugate FT of a template for repeated correlation.

    Parameters
    ----------
    template : torch.Tensor
        ``(H, W)`` corner-centered correlation template.

    Returns
    -------
    torch.Tensor
        ``(H, W)`` complex ``conj(fft2(template))``, ready to multiply against
        ``fft2(dp)``.
    """
    return torch.conj(torch.fft.fft2(template))


def cross_correlation(
    dp: torch.Tensor,
    template_ft: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-correlate a diffraction pattern with a template.

    Parameters
    ----------
    dp : torch.Tensor
        ``(H, W)`` diffraction pattern.
    template_ft : torch.Tensor
        ``(H, W)`` pre-computed template FT from :func:`template_fourier`.

    Returns
    -------
    corr_map : torch.Tensor
        ``(H, W)`` real-space correlation map ``relu(real(ifft2(m)))`` (used for
        peak finding).
    m : torch.Tensor
        ``(H, W)`` Fourier-domain product ``fft2(dp) * template_ft`` (used for DFT
        subpixel refinement).
    """
    dp = torch.as_tensor(dp)
    m = torch.fft.fft2(dp) * template_ft
    corr_map = torch.clamp(torch.fft.ifft2(m).real, min=0.0)
    return corr_map, m


def cross_correlation_batch(
    dps: torch.Tensor,
    template_ft: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-correlate a stack of diffraction patterns with one template.

    Batched form of :func:`cross_correlation`. ``fft2`` acts on the trailing two
    axes and the ``(H, W)`` ``template_ft`` broadcasts over the batch, so the result
    for each pattern is bit-identical to :func:`cross_correlation`.

    Parameters
    ----------
    dps : torch.Tensor
        ``(B, H, W)`` stack of diffraction patterns.
    template_ft : torch.Tensor
        ``(H, W)`` pre-computed template FT from :func:`template_fourier`.

    Returns
    -------
    corr_map : torch.Tensor
        ``(B, H, W)`` real-space correlation maps.
    m : torch.Tensor
        ``(B, H, W)`` Fourier-domain products.
    """
    dps = torch.as_tensor(dps)
    m = torch.fft.fft2(dps) * template_ft
    corr_map = torch.clamp(torch.fft.ifft2(m).real, min=0.0)
    return corr_map, m


def _corr_map_rfft(dps: torch.Tensor, template_ft: torch.Tensor) -> torch.Tensor:
    """Real-FFT correlation map(s), used when no Fourier product is needed downstream.

    For real ``dps`` and a real template the Fourier product is conjugate-symmetric,
    so ``relu(real(ifft2(fft2(dps) * template_ft)))`` is reproduced exactly by an
    ``rfft2`` / ``irfft2`` pair at roughly half the FFT cost. Only the correlation map
    is returned — the complex product (needed solely for DFT upsampling) is skipped.

    Parameters
    ----------
    dps : torch.Tensor
        ``(H, W)`` or ``(B, H, W)`` diffraction pattern(s).
    template_ft : torch.Tensor
        ``(H, W)`` pre-computed template FT from :func:`template_fourier`.

    Returns
    -------
    torch.Tensor
        ``(H, W)`` or ``(B, H, W)`` real-space correlation map(s), ``relu``-clamped.
    """
    dps = torch.as_tensor(dps)
    H, W = dps.shape[-2], dps.shape[-1]
    prod = torch.fft.rfft2(dps) * template_ft[..., : W // 2 + 1]
    corr_map = torch.fft.irfft2(prod, s=(H, W))
    return torch.clamp(corr_map, min=0.0)


def detect_disks(
    dp: torch.Tensor,
    template_ft: torch.Tensor,
    *,
    min_abs_intensity: float = 0.0,
    min_spacing: float = 0.0,
    edge_boundary: int = 1,
    subpixel: str = "upsample",
    upsample_factor: int = 16,
    max_num_peaks: int = 1000,
) -> np.ndarray:
    """Detect Bragg disks in one diffraction pattern by template matching.

    Parameters
    ----------
    dp : torch.Tensor
        ``(H, W)`` diffraction pattern.
    template_ft : torch.Tensor
        ``(H, W)`` pre-computed template FT from :func:`template_fourier`.
    min_abs_intensity : float, default=0.0
        Drop correlation peaks below this absolute intensity.
    min_spacing : float, default=0.0
        Minimum spacing in pixels between kept peaks; closer / dimmer peaks are
        suppressed.
    edge_boundary : int, default=1
        Width in pixels of the border in which peaks are ignored.
    subpixel : {"none", "parabolic", "upsample"}, default="upsample"
        ``"none"`` returns pixel-resolution peaks; ``"parabolic"`` adds a 3-point
        quadratic refinement; ``"upsample"`` further refines each peak by
        Guizar-Sicairos DFT upsampling.
    upsample_factor : int, default=16
        Upsampling factor for the ``"upsample"`` subpixel refinement.
    max_num_peaks : int, default=1000
        Maximum number of peaks to keep (after intensity sorting).

    Returns
    -------
    np.ndarray
        ``(M, 3)`` array of ``[q_row, q_col, intensity]`` rows, sorted by descending
        intensity.
    """
    if subpixel not in SUBPIXEL_MODES:
        raise ValueError(f"subpixel must be in {SUBPIXEL_MODES}, got {subpixel!r}")

    corr_map, m = cross_correlation(dp, template_ft)

    peaks = _local_maxima(corr_map, edge_boundary)
    peaks = _filter_maxima(peaks, min_abs_intensity, min_spacing, max_num_peaks)

    if peaks.shape[0] == 0 or subpixel == "none":
        return _to_numpy(peaks)

    peaks = _refine_parabolic(corr_map, peaks)

    if subpixel == "parabolic":
        return _to_numpy(peaks)

    peaks = _refine_dft(m, peaks, upsample_factor)
    return _to_numpy(peaks)


def detect_disks_batch(
    dps: torch.Tensor,
    template_ft: torch.Tensor,
    *,
    min_abs_intensity: float = 0.0,
    min_spacing: float = 0.0,
    edge_boundary: int = 1,
    subpixel: str = "upsample",
    upsample_factor: int = 16,
    max_num_peaks: int = 1000,
) -> list[np.ndarray]:
    """Detect Bragg disks across a stack of diffraction patterns (batched).

    Batched equivalent of :func:`detect_disks`: cross-correlation, peak extraction,
    ``min_spacing`` suppression, and subpixel refinement are all batched across
    patterns. The local-maxima search and greedy ``min_spacing`` suppression are
    vectorized over the whole stack (no per-pattern Python loop) but reproduce the
    per-pattern greedy result bit-for-bit, so each output matches :func:`detect_disks`
    for that pattern. When ``subpixel`` is not ``"upsample"`` the correlation maps are
    formed with a real FFT (``rfft2``), skipping the complex Fourier product that only
    DFT upsampling needs.

    Parameters
    ----------
    dps : torch.Tensor
        ``(B, H, W)`` stack of diffraction patterns.
    template_ft : torch.Tensor
        ``(H, W)`` pre-computed template FT from :func:`template_fourier`.
    min_abs_intensity : float, default=0.0
        Drop correlation peaks below this absolute intensity.
    min_spacing : float, default=0.0
        Minimum spacing in pixels between kept peaks; closer / dimmer peaks are
        suppressed.
    edge_boundary : int, default=1
        Width in pixels of the border in which peaks are ignored.
    subpixel : {"none", "parabolic", "upsample"}, default="upsample"
        ``"none"`` returns pixel-resolution peaks; ``"parabolic"`` adds a 3-point
        quadratic refinement; ``"upsample"`` further refines each peak by
        Guizar-Sicairos DFT upsampling.
    upsample_factor : int, default=16
        Upsampling factor for the ``"upsample"`` subpixel refinement.
    max_num_peaks : int, default=1000
        Maximum number of peaks to keep per pattern (after intensity sorting).

    Returns
    -------
    list of np.ndarray
        Length-``B`` list of ``(M, 3)`` arrays of ``[q_row, q_col, intensity]``
        rows, each sorted by descending intensity.
    """
    if subpixel not in SUBPIXEL_MODES:
        raise ValueError(f"subpixel must be in {SUBPIXEL_MODES}, got {subpixel!r}")

    if subpixel == "upsample":
        corr_map, m = cross_correlation_batch(dps, template_ft)
    else:
        corr_map = _corr_map_rfft(dps, template_ft)
        m = None

    peaks_all, bidx, counts = _detect_peaks_batched(
        corr_map, edge_boundary, min_abs_intensity, min_spacing, max_num_peaks
    )

    if subpixel == "none" or peaks_all.shape[0] == 0:
        return _split_by_counts(peaks_all, counts)

    peaks_all = _refine_parabolic_batched(corr_map, peaks_all, bidx)
    if subpixel == "upsample":
        peaks_all = _refine_dft_batched(m, peaks_all, bidx, upsample_factor)

    return _split_by_counts(peaks_all, counts)


# ---- helpers ----


def _local_maxima(corr_map: torch.Tensor, edge_boundary: int) -> torch.Tensor:
    """Find 8-neighbor local maxima, sorted by descending intensity.

    Parameters
    ----------
    corr_map : torch.Tensor
        ``(H, W)`` correlation map.
    edge_boundary : int
        Width in pixels of the border in which maxima are ignored.

    Returns
    -------
    torch.Tensor
        ``(K, 3)`` tensor of ``[row, col, intensity]`` maxima, sorted by descending
        intensity.
    """
    is_max = _local_maxima_mask(corr_map, edge_boundary)
    return _extract_maxima(corr_map, is_max)


def _local_maxima_mask(a: torch.Tensor, edge_boundary: int) -> torch.Tensor:
    """Boolean 8-neighbor local-maxima mask of ``a``.

    Works on a single ``(H, W)`` map or a batch ``(B, H, W)`` — the neighbor
    comparisons and edge masking use the trailing two axes — so the same code drives
    the single-pattern and batched detection paths bit-identically.

    Parameters
    ----------
    a : torch.Tensor
        ``(H, W)`` or ``(B, H, W)`` correlation map(s).
    edge_boundary : int
        Width in pixels of the border (clamped to at least 1) set to ``False``.

    Returns
    -------
    torch.Tensor
        Boolean mask the same shape as ``a``, ``True`` at 8-neighbor local maxima.
    """
    is_max = (
        (a >= torch.roll(a, (-1, 0), dims=(-2, -1)))
        & (a > torch.roll(a, (1, 0), dims=(-2, -1)))
        & (a >= torch.roll(a, (0, -1), dims=(-2, -1)))
        & (a > torch.roll(a, (0, 1), dims=(-2, -1)))
        & (a >= torch.roll(a, (-1, -1), dims=(-2, -1)))
        & (a > torch.roll(a, (-1, 1), dims=(-2, -1)))
        & (a >= torch.roll(a, (1, -1), dims=(-2, -1)))
        & (a > torch.roll(a, (1, 1), dims=(-2, -1)))
    )

    eb = max(1, int(edge_boundary))
    is_max[..., :eb, :] = False
    is_max[..., -eb:, :] = False
    is_max[..., :, :eb] = False
    is_max[..., :, -eb:] = False
    return is_max


def _extract_maxima(a: torch.Tensor, is_max: torch.Tensor) -> torch.Tensor:
    """Gather masked maxima of one ``(H, W)`` map into a descending-sorted ``(K, 3)``.

    Rows are taken in row-major ``nonzero`` order, then stably sorted by descending
    intensity — matching the original single-pattern behaviour.

    Parameters
    ----------
    a : torch.Tensor
        ``(H, W)`` correlation map.
    is_max : torch.Tensor
        Boolean ``(H, W)`` local-maxima mask from :func:`_local_maxima_mask`.

    Returns
    -------
    torch.Tensor
        ``(K, 3)`` tensor of ``[row, col, intensity]`` maxima, sorted by descending
        intensity.
    """
    rows, cols = torch.nonzero(is_max, as_tuple=True)
    intensity = a[rows, cols]
    order = torch.argsort(intensity, descending=True)
    return torch.stack((rows[order].to(a.dtype), cols[order].to(a.dtype), intensity[order]), dim=1)


def _filter_maxima(
    peaks: torch.Tensor,
    min_abs_intensity: float,
    min_spacing: float,
    max_num_peaks: int,
) -> torch.Tensor:
    """Drop dim peaks, suppress peaks closer than ``min_spacing``, cap the count.

    Parameters
    ----------
    peaks : torch.Tensor
        ``(K, 3)`` ``[row, col, intensity]`` peaks, sorted by descending intensity.
    min_abs_intensity : float
        Drop peaks below this absolute intensity (ignored when ``<= 0``).
    min_spacing : float
        Minimum spacing in pixels; for each kept peak, dimmer peaks within this
        distance are suppressed (ignored when ``<= 0``).
    max_num_peaks : int
        Maximum number of peaks to keep; the brightest are retained.

    Returns
    -------
    torch.Tensor
        ``(M, 3)`` filtered peaks.
    """
    if peaks.shape[0] == 0:
        return peaks

    if min_abs_intensity > 0:
        peaks = peaks[peaks[:, 2] >= min_abs_intensity]

    if min_spacing > 0 and peaks.shape[0] > 1:
        keep = torch.ones(peaks.shape[0], dtype=torch.bool, device=peaks.device)
        rc = peaks[:, :2]
        for i in range(peaks.shape[0]):
            if not keep[i]:
                continue
            d2 = ((rc - rc[i]) ** 2).sum(dim=1)
            too_close = d2 < min_spacing**2
            too_close[: i + 1] = False
            keep[too_close] = False
        peaks = peaks[keep]

    if max_num_peaks is not None and peaks.shape[0] > max_num_peaks:
        peaks = peaks[:max_num_peaks]

    return peaks


def _detect_peaks_batched(
    corr: torch.Tensor,
    edge_boundary: int,
    min_abs_intensity: float,
    min_spacing: float,
    max_num_peaks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract, suppress, and cap peaks across a whole ``(B, H, W)`` stack at once.

    Vectorized replacement for the per-pattern ``_extract_maxima`` + ``_filter_maxima``
    loop. Local maxima are found and intensity-thresholded for the whole stack, then
    the greedy ``min_spacing`` suppression runs as a single loop over the *rank* axis
    (brightest-first) that is batched over every pattern. Suppressing peak ``k``'s
    fainter neighbours only when ``k`` is still kept reproduces the per-pattern greedy
    result of :func:`_filter_maxima` exactly, with no Python loop over patterns.

    Parameters
    ----------
    corr : torch.Tensor
        ``(B, H, W)`` correlation maps.
    edge_boundary : int
        Width in pixels of the border in which maxima are ignored.
    min_abs_intensity : float
        Drop peaks below this absolute intensity (ignored when ``<= 0``).
    min_spacing : float
        Minimum spacing in pixels; for each kept peak, fainter peaks within this
        distance are suppressed (ignored when ``<= 0``).
    max_num_peaks : int
        Maximum number of peaks kept per pattern (the brightest are retained).

    Returns
    -------
    peaks : torch.Tensor
        ``(T, 3)`` ``[row, col, intensity]`` peaks pooled over all patterns, grouped
        by ascending pattern index and sorted by descending intensity within a pattern.
    bidx : torch.Tensor
        ``(T,)`` pattern index of each peak.
    counts : torch.Tensor
        ``(B,)`` number of peaks kept per pattern.
    """
    B = corr.shape[0]
    device = corr.device
    dtype = corr.dtype

    mask = _local_maxima_mask(corr, edge_boundary)
    if min_abs_intensity > 0:
        mask = mask & (corr >= min_abs_intensity)

    idx = torch.nonzero(mask)  # (T0, 3): [b, row, col]
    counts = torch.zeros(B, dtype=torch.long, device=device)
    if idx.shape[0] == 0:
        return corr.new_zeros((0, 3)), torch.zeros(0, dtype=torch.long, device=device), counts

    bcand, rcand, ccand = idx[:, 0], idx[:, 1], idx[:, 2]
    inten = corr[bcand, rcand, ccand]

    # Group candidates by pattern, descending intensity within each pattern. A global
    # descending-intensity sort followed by a stable sort on the pattern index keeps the
    # within-pattern order identical to the per-pattern argsort in _extract_maxima.
    o1 = torch.argsort(inten, descending=True)
    order = o1[torch.argsort(bcand[o1], stable=True)]
    bcand, rcand, ccand, inten = bcand[order], rcand[order], ccand[order], inten[order]

    counts = torch.bincount(bcand, minlength=B)
    kmax = int(counts.max())

    # Pack candidates into a dense (B, kmax) grid indexed by [pattern, brightness rank].
    starts = torch.zeros(B, dtype=torch.long, device=device)
    starts[1:] = torch.cumsum(counts, 0)[:-1]
    rank = torch.arange(bcand.shape[0], device=device) - starts[bcand]
    flat = bcand * kmax + rank

    rows = torch.zeros(B * kmax, dtype=dtype, device=device)
    cols = torch.zeros(B * kmax, dtype=dtype, device=device)
    vals = torch.zeros(B * kmax, dtype=dtype, device=device)
    valid = torch.zeros(B * kmax, dtype=torch.bool, device=device)
    rows[flat] = rcand.to(dtype)
    cols[flat] = ccand.to(dtype)
    vals[flat] = inten
    valid[flat] = True
    rows, cols, valid = rows.view(B, kmax), cols.view(B, kmax), valid.view(B, kmax)

    keep = valid.clone()
    if min_spacing > 0 and kmax > 1:
        s2 = float(min_spacing) ** 2
        for k in range(kmax - 1):
            active = keep[:, k] & valid[:, k]  # (B,) brightest unsuppressed peak at rank k
            dr = rows[:, k + 1 :] - rows[:, k : k + 1]
            dc = cols[:, k + 1 :] - cols[:, k : k + 1]
            suppress = ((dr * dr + dc * dc) < s2) & active[:, None] & valid[:, k + 1 :]
            keep[:, k + 1 :] &= ~suppress

    if max_num_peaks is not None:
        kept_rank = torch.cumsum(keep.to(torch.long), dim=1) - 1
        keep &= kept_rank < max_num_peaks

    final = (keep & valid).view(-1)
    sel = torch.nonzero(final, as_tuple=True)[0]  # row-major: pattern-major, rank-minor
    peaks = torch.stack((rows.view(-1)[sel], cols.view(-1)[sel], vals[sel]), dim=1)
    bidx = torch.div(sel, kmax, rounding_mode="floor")
    counts = torch.bincount(bidx, minlength=B)
    return peaks, bidx, counts


def _split_by_counts(peaks: torch.Tensor, counts: torch.Tensor) -> list[np.ndarray]:
    """Split a pattern-grouped ``(T, 3)`` peak stack into one ``(M, 3)`` array per pattern.

    Parameters
    ----------
    peaks : torch.Tensor
        ``(T, 3)`` peaks ordered by ascending pattern index (contiguous per pattern).
    counts : torch.Tensor
        ``(B,)`` number of peaks belonging to each pattern, in order.

    Returns
    -------
    list of np.ndarray
        Length-``B`` list of ``(M, 3)`` arrays.
    """
    out = []
    start = 0
    for cnt in counts.tolist():
        out.append(_to_numpy(peaks[start : start + cnt]))
        start += cnt
    return out


def _refine_parabolic(corr_map: torch.Tensor, peaks: torch.Tensor) -> torch.Tensor:
    """3-point quadratic subpixel refinement of every peak (vectorized over peaks).

    Parameters
    ----------
    corr_map : torch.Tensor
        ``(H, W)`` correlation map the peaks were found in.
    peaks : torch.Tensor
        ``(M, 3)`` ``[row, col, intensity]`` peaks to refine.

    Returns
    -------
    torch.Tensor
        ``(M, 3)`` peaks with subpixel ``[row, col]`` and bilinearly interpolated
        intensity.
    """
    if peaks.shape[0] == 0:
        return peaks
    a = corr_map
    H, W = a.shape
    out = peaks.clone()
    zero = torch.zeros(out.shape[0], device=a.device, dtype=a.dtype)

    r = out[:, 0].round().long().clamp(0, H - 1)
    c = out[:, 1].round().long().clamp(0, W - 1)

    r_in = (r > 0) & (r < H - 1)
    ix0, ix1, ix2 = a[(r - 1).clamp(0, H - 1), c], a[r, c], a[(r + 1).clamp(0, H - 1), c]
    denom_r = 4.0 * ix1 - 2.0 * ix2 - 2.0 * ix0
    dr = torch.where(r_in & (denom_r != 0), (ix2 - ix0) / denom_r, zero)

    c_in = (c > 0) & (c < W - 1)
    iy0, iy1, iy2 = a[r, (c - 1).clamp(0, W - 1)], a[r, c], a[r, (c + 1).clamp(0, W - 1)]
    denom_c = 4.0 * iy1 - 2.0 * iy2 - 2.0 * iy0
    dc = torch.where(c_in & (denom_c != 0), (iy2 - iy0) / denom_c, zero)

    r_sub = r.to(a.dtype) + dr
    c_sub = c.to(a.dtype) + dc
    out[:, 0] = r_sub
    out[:, 1] = c_sub
    out[:, 2] = _bilinear(a, r_sub, c_sub)
    return out


def _refine_parabolic_batched(
    corr: torch.Tensor, peaks: torch.Tensor, bidx: torch.Tensor
) -> torch.Tensor:
    """3-point quadratic refinement of peaks pooled across a batch of correlation maps.

    Batched form of :func:`_refine_parabolic`. Reading ``corr[bidx, r, c]`` gathers
    exactly the values the single-pattern path would read, so each peak refines
    identically.

    Parameters
    ----------
    corr : torch.Tensor
        ``(B, H, W)`` correlation maps.
    peaks : torch.Tensor
        ``(T, 3)`` ``[row, col, intensity]`` peaks pooled over all patterns.
    bidx : torch.Tensor
        ``(T,)`` batch index selecting each peak's correlation map.

    Returns
    -------
    torch.Tensor
        ``(T, 3)`` peaks with subpixel ``[row, col]`` and bilinearly interpolated
        intensity.
    """
    if peaks.shape[0] == 0:
        return peaks
    H, W = corr.shape[-2], corr.shape[-1]
    dtype = corr.dtype
    out = peaks.clone()
    zero = torch.zeros(out.shape[0], device=corr.device, dtype=dtype)

    r = out[:, 0].round().long().clamp(0, H - 1)
    c = out[:, 1].round().long().clamp(0, W - 1)

    r_in = (r > 0) & (r < H - 1)
    ix0 = corr[bidx, (r - 1).clamp(0, H - 1), c]
    ix1 = corr[bidx, r, c]
    ix2 = corr[bidx, (r + 1).clamp(0, H - 1), c]
    denom_r = 4.0 * ix1 - 2.0 * ix2 - 2.0 * ix0
    dr = torch.where(r_in & (denom_r != 0), (ix2 - ix0) / denom_r, zero)

    c_in = (c > 0) & (c < W - 1)
    iy0 = corr[bidx, r, (c - 1).clamp(0, W - 1)]
    iy1 = corr[bidx, r, c]
    iy2 = corr[bidx, r, (c + 1).clamp(0, W - 1)]
    denom_c = 4.0 * iy1 - 2.0 * iy2 - 2.0 * iy0
    dc = torch.where(c_in & (denom_c != 0), (iy2 - iy0) / denom_c, zero)

    r_sub = r.to(dtype) + dr
    c_sub = c.to(dtype) + dc
    out[:, 0] = r_sub
    out[:, 1] = c_sub
    out[:, 2] = _bilinear_batched(corr, bidx, r_sub, c_sub)
    return out


def _refine_dft(m: torch.Tensor, peaks: torch.Tensor, upsample_factor: int) -> torch.Tensor:
    """Guizar-Sicairos DFT upsampling refinement of every peak (vectorized over peaks).

    Each peak is rounded to half-pixel precision (matching py4DSTEM multicorr) before
    upsampling, then all peaks are refined together with one batched DFT upsampling.

    Parameters
    ----------
    m : torch.Tensor
        ``(H, W)`` Fourier-domain correlation product from :func:`cross_correlation`.
    peaks : torch.Tensor
        ``(M, 3)`` ``[row, col, intensity]`` peaks to refine.
    upsample_factor : int
        DFT upsampling factor.

    Returns
    -------
    torch.Tensor
        ``(M, 3)`` peaks with DFT-refined ``[row, col]`` (intensity unchanged).
    """
    if peaks.shape[0] == 0:
        return peaks
    out = peaks.clone()
    xy = torch.round(out[:, :2] * 2.0) / 2.0
    refined = _upsampled_correlation_batch(m, int(upsample_factor), xy)
    out[:, :2] = refined
    return out


def _refine_dft_batched(
    m: torch.Tensor, peaks: torch.Tensor, bidx: torch.Tensor, upsample_factor: int
) -> torch.Tensor:
    """DFT-upsampling refinement of peaks pooled across a batch of correlation products.

    Batched form of :func:`_refine_dft`. Peaks are processed in sub-chunks so the
    gathered per-peak product ``m[bidx]`` (``(chunk, H, W)`` complex) stays within a
    fixed memory budget regardless of how many peaks were found.

    Parameters
    ----------
    m : torch.Tensor
        ``(B, H, W)`` Fourier-domain correlation products.
    peaks : torch.Tensor
        ``(T, 3)`` ``[row, col, intensity]`` peaks pooled over all patterns.
    bidx : torch.Tensor
        ``(T,)`` batch index selecting each peak's correlation product.
    upsample_factor : int
        DFT upsampling factor.

    Returns
    -------
    torch.Tensor
        ``(T, 3)`` peaks with DFT-refined ``[row, col]`` (intensity unchanged).
    """
    if peaks.shape[0] == 0:
        return peaks
    out = peaks.clone()
    M, N = m.shape[-2], m.shape[-1]
    xy = torch.round(out[:, :2] * 2.0) / 2.0
    cap = max(1, 8_000_000 // (M * N))
    total = peaks.shape[0]
    for start in range(0, total, cap):
        stop = min(start + cap, total)
        b = bidx[start:stop]
        out[start:stop, :2] = _upsampled_correlation_batch(
            m[b], int(upsample_factor), xy[start:stop]
        )
    return out


def _upsampled_correlation_batch(
    m: torch.Tensor, upsample_factor: int, xy: torch.Tensor
) -> torch.Tensor:
    """Batched DFT upsampling of the correlation peak for many shifts at once.

    Vectorizes :func:`~quantem.core.utils.imaging_utils.upsampled_correlation_torch`
    over the shifts ``xy`` with batched matmuls. Two exact speedups are applied: (1) ``m``
    is conjugate-symmetric (the diffraction pattern and template are real), so only the
    non-negative half of the row frequencies is contracted — halving the dominant matmul —
    with a rank-1 correction on the Nyquist column; (2) the per-peak DFT kernels are
    factored into shared base kernels times per-peak phases, so ``torch.exp`` runs on far
    fewer elements. Both are algebraically identical to the full-spectrum DFT upsample.

    Parameters
    ----------
    m : torch.Tensor
        ``(M, N)`` or ``(K, M, N)`` Fourier-domain correlation product(s). A single
        ``(M, N)`` product broadcasts over all ``K`` shifts.
    upsample_factor : int
        DFT upsampling factor.
    xy : torch.Tensor
        ``(K, 2)`` ``[row, col]`` peak shifts at half-pixel precision.

    Returns
    -------
    torch.Tensor
        ``(K, 2)`` DFT-refined ``[row, col]`` positions.
    """
    import math

    device = m.device
    dtype = torch.get_default_dtype()
    uf = float(upsample_factor)
    M, N = m.shape[-2], m.shape[-1]

    xy = torch.round(xy * uf) / uf
    global_shift = math.floor(math.ceil(uf * 1.5) / 2.0)
    upsample_center = global_shift - uf * xy  # (K, 2): [row, col]

    num = int(math.ceil(1.5 * uf))
    half = M // 2 + 1
    col_freq = (torch.fft.ifftshift(torch.arange(N, device=device)) - math.floor(N / 2)).to(dtype)
    row_freq = (torch.fft.ifftshift(torch.arange(M, device=device)) - math.floor(M / 2)).to(dtype)

    # ``m = F(dp) * conj(F(template))`` is 2D conjugate-symmetric for real ``dp`` and
    # template, so the upper-half row frequencies are redundant. Contract only the
    # non-negative half (rows ``u = 0 .. M // 2``) and fold the conjugate upper rows
    # back in with a weight of 2 — 1 for the self-paired DC row and, when ``M`` is even,
    # the Nyquist row. This halves the dominant matmul over the row axis.
    row_freq = row_freq[:half]
    row_weight = torch.full((half,), 2.0, device=device, dtype=dtype)
    row_weight[0] = 1.0
    if M % 2 == 0:
        row_weight[-1] = 1.0

    base = torch.arange(num, device=device, dtype=dtype)
    factor_col = -2j * math.pi / (N * uf)
    factor_row = -2j * math.pi / (M * uf)

    # Factor each kernel ``exp(f · freq · (base - center))`` into a peak-independent base
    # kernel ``exp(f · freq · base)`` times a per-peak phase ``exp(-f · freq · center)``.
    # The base kernels are tiny and built once; the phases fold into ``m`` and the
    # intermediate product, so ``torch.exp`` runs on ~20x fewer elements.
    base_col = torch.exp(factor_col * (col_freq[:, None] * base[None, :]))  # (N, num)
    base_row = torch.exp(factor_row * (base[:, None] * row_freq[None, :]))[None]  # (1, num, half)
    phase_col = torch.exp(-factor_col * (col_freq[None, :] * upsample_center[:, 1:2]))  # (K, N)
    phase_row = torch.exp(-factor_row * (upsample_center[:, 0:1] * row_freq[None, :]))  # (K, half)

    mc_half = m.conj()[..., :half, :]  # (K, half, N) — non-negative row frequencies only
    # Fold the row phase and conjugate-symmetry weight into ``m`` so the shared base row
    # kernel is reused across peaks; then contract rows, apply the column phase, and
    # contract columns against the shared base column kernel.
    prod = torch.matmul(base_row, mc_half * (row_weight * phase_row)[:, :, None])  # (K, num, N)
    up = torch.matmul(prod * phase_col[:, None, :], base_col)  # (K, num, num)

    if N % 2 == 0:
        # Row-only folding is exact for every column except the Nyquist column ``v = N/2``,
        # whose conjugate partner stays in the same column. Correct it with an exact rank-1
        # update ``pr_diff ⊗ col_kern[:, N/2]``, where ``pr_diff = -2i Im(S)`` and ``S`` sums
        # only the kept interior rows.
        nyq = N // 2
        interior = row_weight - 1.0  # 1 on interior rows, 0 on the DC / Nyquist rows
        s = torch.matmul(base_row, (mc_half[..., nyq] * (interior * phase_row)).unsqueeze(-1))
        pr_diff = -2j * s[..., 0].imag  # (K, num)
        col_nyq = base_col[nyq, :][None, :] * phase_col[:, nyq : nyq + 1]  # (K, num)
        up = up + pr_diff[:, :, None] * col_nyq[:, None, :]

    image_up = up.real

    K = xy.shape[0]
    kidx = torch.arange(K, device=device)
    idx = torch.argmax(image_up.reshape(K, -1), dim=1)
    sub_r = torch.div(idx, num, rounding_mode="floor")
    sub_c = idx % num

    # 3-point parabolic refinement around the upsampled maximum (interior only)
    interior = (sub_r > 0) & (sub_r < num - 1) & (sub_c > 0) & (sub_c < num - 1)
    rr = sub_r.clamp(1, num - 2)
    cc = sub_c.clamp(1, num - 2)
    c11 = image_up[kidx, rr, cc]
    c21, c01 = image_up[kidx, rr + 1, cc], image_up[kidx, rr - 1, cc]
    c12, c10 = image_up[kidx, rr, cc + 1], image_up[kidx, rr, cc - 1]
    zero = torch.zeros(K, device=device, dtype=dtype)
    denom_x = 4.0 * c11 - 2.0 * c21 - 2.0 * c01
    denom_y = 4.0 * c11 - 2.0 * c12 - 2.0 * c10
    dx = torch.where(interior & (denom_x != 0), (c21 - c01) / denom_x, zero)
    dy = torch.where(interior & (denom_y != 0), (c12 - c10) / denom_y, zero)

    sub = torch.stack([sub_r.to(dtype), sub_c.to(dtype)], dim=1) - global_shift
    return xy + (sub + torch.stack([dx, dy], dim=1)) / uf


def _bilinear(a: torch.Tensor, r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Bilinear interpolation of ``a`` at fractional ``(r, c)`` (vectorized over peaks).

    Parameters
    ----------
    a : torch.Tensor
        ``(H, W)`` map to sample.
    r : torch.Tensor
        ``(K,)`` fractional row coordinates.
    c : torch.Tensor
        ``(K,)`` fractional column coordinates.

    Returns
    -------
    torch.Tensor
        ``(K,)`` interpolated values.
    """
    H, W = a.shape
    r0 = torch.floor(r).long()
    c0 = torch.floor(c).long()
    r1 = (r0 + 1).clamp(max=H - 1)
    c1 = (c0 + 1).clamp(max=W - 1)
    r0 = r0.clamp(0, H - 1)
    c0 = c0.clamp(0, W - 1)
    dr = r - r0.to(a.dtype)
    dc = c - c0.to(a.dtype)
    return (
        (1 - dr) * (1 - dc) * a[r0, c0]
        + (1 - dr) * dc * a[r0, c1]
        + dr * (1 - dc) * a[r1, c0]
        + dr * dc * a[r1, c1]
    )


def _bilinear_batched(
    corr: torch.Tensor, bidx: torch.Tensor, r: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Bilinear interpolation of a batch ``corr`` ``(B, H, W)`` at per-peak ``(r, c)``.

    Batched form of :func:`_bilinear`. The four corner samples are gathered as
    ``corr[bidx, r0, c0]`` etc., matching the single-pattern interpolation
    value-for-value.

    Parameters
    ----------
    corr : torch.Tensor
        ``(B, H, W)`` maps to sample.
    bidx : torch.Tensor
        ``(T,)`` batch index selecting each peak's map.
    r : torch.Tensor
        ``(T,)`` fractional row coordinates.
    c : torch.Tensor
        ``(T,)`` fractional column coordinates.

    Returns
    -------
    torch.Tensor
        ``(T,)`` interpolated values.
    """
    H, W = corr.shape[-2], corr.shape[-1]
    dtype = corr.dtype
    r0 = torch.floor(r).long()
    c0 = torch.floor(c).long()
    r1 = (r0 + 1).clamp(max=H - 1)
    c1 = (c0 + 1).clamp(max=W - 1)
    r0 = r0.clamp(0, H - 1)
    c0 = c0.clamp(0, W - 1)
    dr = r - r0.to(dtype)
    dc = c - c0.to(dtype)
    return (
        (1 - dr) * (1 - dc) * corr[bidx, r0, c0]
        + (1 - dr) * dc * corr[bidx, r0, c1]
        + dr * (1 - dc) * corr[bidx, r1, c0]
        + dr * dc * corr[bidx, r1, c1]
    )


def _to_numpy(peaks: torch.Tensor) -> np.ndarray:
    """Convert a peaks tensor to a contiguous ``(M, 3)`` float64 numpy array.

    Parameters
    ----------
    peaks : torch.Tensor
        ``(M, 3)`` (or flat) peaks tensor on any device.

    Returns
    -------
    np.ndarray
        ``(M, 3)`` ``float64`` array of ``[q_row, q_col, intensity]`` rows.
    """
    return peaks.detach().cpu().numpy().astype(np.float64).reshape(-1, 3)
