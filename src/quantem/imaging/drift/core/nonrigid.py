"""Per-scanline non-rigid drift optimization."""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import quantem.imaging.drift.plot as drift_plot
from quantem.imaging.drift.core import knots as drift_knots
from quantem.imaging.drift.core.warping import warp_and_translate


def _grid_sample_mse(
    grid_row: torch.Tensor,
    grid_col: torch.Tensor,
    ref_t: torch.Tensor,
    target_batch: torch.Tensor,
) -> torch.Tensor:
    """Shared tail: stack the (col, row) grid, sample, and return the MSE.

    Inlined into both compiled kernels (`@torch.compile` follows the call).
    Pulling the grid_sample + MSE out of the two K-paths means the only
    difference between K=1 and K>=2 kernels is grid construction.

    The MSE is averaged over both the batch (N images) and the spatial dims,
    so each image's gradient is scaled by 1/N relative to a per-image solve.
    Adam's adaptive step size absorbs the constant rescale; LBFGS line search
    rescales itself.
    """
    grid = torch.stack([grid_col, grid_row], dim=-1)
    warped = F.grid_sample(
        ref_t, grid, mode='bilinear', align_corners=True, padding_mode='border')[:, 0]
    return ((warped - target_batch) ** 2).mean()


def _grid_sample_ncc(
    grid_row: torch.Tensor,
    grid_col: torch.Tensor,
    ref_t: torch.Tensor,
    target_batch: torch.Tensor,
    ref_coverage_t: torch.Tensor,
) -> torch.Tensor:
    """Warp ``ref_t`` and return coverage-weighted ``1 - mean NCC``.

    Reference coverage is sampled through the same grid and detached before
    reduction. This excludes canvas fill without rewarding knot motion merely
    for changing the valid area. Per-image weighted zero-mean correlation is
    then averaged over the batch.
    """
    grid = torch.stack([grid_col, grid_row], dim=-1)
    warped = F.grid_sample(
        ref_t, grid, mode='bilinear', align_corners=True, padding_mode='border')[:, 0]
    coverage = F.grid_sample(
        ref_coverage_t[:, None],
        grid,
        mode='bilinear',
        align_corners=True,
        padding_mode='zeros',
    )[:, 0].detach().clamp(0.0, 1.0)
    # (N, H, W) → (N, H*W), with weighted means/norms on measured pixels.
    w = warped.reshape(warped.shape[0], -1)
    t = target_batch.reshape(target_batch.shape[0], -1)
    mask = coverage.reshape(coverage.shape[0], -1)
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    w0 = w - (w * mask).sum(dim=-1, keepdim=True) / count
    t0 = t - (t * mask).sum(dim=-1, keepdim=True) / count
    num = (mask * w0 * t0).sum(dim=-1)
    den = (
        (mask * w0.square()).sum(dim=-1).sqrt()
        * (mask * t0.square()).sum(dim=-1).sqrt()
    ).clamp_min(1e-12)
    ncc = (num / den).mean()
    return 1.0 - ncc


@torch.compile(mode="reduce-overhead", dynamic=False)
def _compiled_loss_fn_single(
    knots_batch: torch.Tensor,
    ref_t: torch.Tensor,
    target_batch: torch.Tensor,
    row_scan_offsets: torch.Tensor,
    col_scan_offsets: torch.Tensor,
    row_scale: float,
    col_scale: float,
) -> torch.Tensor:
    """Fused K=1 forward pass: knot anchor + scan_fast walk → MSE.

    ``knots_batch`` shape ``(N, 2, num_rows, 1)``.  Each scanline has one
    anchor knot and the per-pixel canvas position is filled in by adding
    the precomputed ``scan_fast`` walk.  ``correct_nonrigid`` selects this
    kernel when ``K == 1`` and :func:`_compiled_loss_fn_multi` for ``K > 1``.
    """
    grid_row = (knots_batch[:, 0, :, :] + row_scan_offsets[:, None, :]) * row_scale - 1.0
    grid_col = (knots_batch[:, 1, :, :] + col_scan_offsets[:, None, :]) * col_scale - 1.0
    return _grid_sample_mse(grid_row, grid_col, ref_t, target_batch)


@torch.compile(mode="reduce-overhead", dynamic=False)
def _compiled_loss_fn_multi(
    knots_batch: torch.Tensor,
    ref_t: torch.Tensor,
    target_batch: torch.Tensor,
    seg_idx: torch.Tensor,
    seg_frac: torch.Tensor,
    row_scale: float,
    col_scale: float,
) -> torch.Tensor:
    """Fused K-knot forward pass with linear knot interpolation along scanline.

    ``knots_batch`` shape ``(N, 2, num_rows, K)`` with ``K >= 2``.
    ``seg_idx`` (long, shape ``(num_cols,)``) and ``seg_frac`` (shape
    ``(num_cols,)``) precompute, per output column, which adjacent knot
    pair to interpolate and the local fraction.  Both are constant for
    the lifetime of the optimization, so we lift them out of the loop.
    """
    knot_lo = knots_batch[:, :, :, seg_idx]
    knot_hi = knots_batch[:, :, :, seg_idx + 1]
    interp = knot_lo + (knot_hi - knot_lo) * seg_frac[None, None, None, :]
    grid_row = interp[:, 0] * row_scale - 1.0
    grid_col = interp[:, 1] * col_scale - 1.0
    return _grid_sample_mse(grid_row, grid_col, ref_t, target_batch)


@torch.compile(mode="reduce-overhead", dynamic=False)
def _compiled_loss_fn_single_ncc(
    knots_batch: torch.Tensor,
    ref_t: torch.Tensor,
    target_batch: torch.Tensor,
    ref_coverage_t: torch.Tensor,
    row_scan_offsets: torch.Tensor,
    col_scan_offsets: torch.Tensor,
    row_scale: float,
    col_scale: float,
) -> torch.Tensor:
    """K=1 path with ``1 - NCC`` loss (brightness-invariant)."""
    grid_row = (knots_batch[:, 0, :, :] + row_scan_offsets[:, None, :]) * row_scale - 1.0
    grid_col = (knots_batch[:, 1, :, :] + col_scan_offsets[:, None, :]) * col_scale - 1.0
    return _grid_sample_ncc(
        grid_row, grid_col, ref_t, target_batch, ref_coverage_t
    )


@torch.compile(mode="reduce-overhead", dynamic=False)
def _compiled_loss_fn_multi_ncc(
    knots_batch: torch.Tensor,
    ref_t: torch.Tensor,
    target_batch: torch.Tensor,
    ref_coverage_t: torch.Tensor,
    seg_idx: torch.Tensor,
    seg_frac: torch.Tensor,
    row_scale: float,
    col_scale: float,
) -> torch.Tensor:
    """K>=2 path with ``1 - NCC`` loss."""
    knot_lo = knots_batch[:, :, :, seg_idx]
    knot_hi = knots_batch[:, :, :, seg_idx + 1]
    interp = knot_lo + (knot_hi - knot_lo) * seg_frac[None, None, None, :]
    grid_row = interp[:, 0] * row_scale - 1.0
    grid_col = interp[:, 1] * col_scale - 1.0
    return _grid_sample_ncc(
        grid_row, grid_col, ref_t, target_batch, ref_coverage_t
    )


def _optimize_knots_adam(
    ref_batch, target_batch, knots_batch,
    loss_fn, loss_args,
    optimizer, steps, grad_mask=None,
):
    """Run ``steps`` of Adam on a batched knot tensor against ``loss_fn``."""
    ref_t = ref_batch[:, None]
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(knots_batch, ref_t, target_batch, *loss_args)
        loss.backward()
        if grad_mask is not None:
            knots_batch.grad.mul_(grad_mask)
        optimizer.step()


def _optimize_knots_lbfgs(
    ref_batch, target_batch, knots_batch,
    loss_fn, loss_args,
    optimizer, grad_mask=None,
):
    """Run one LBFGS outer step (line search re-evaluates the closure several times)."""
    ref_t = ref_batch[:, None]
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(knots_batch, ref_t, target_batch, *loss_args)
        loss.backward()
        if grad_mask is not None:
            knots_batch.grad.mul_(grad_mask)
        return loss
    optimizer.step(closure)


def sobel_gradient_magnitude(
    images: torch.Tensor,
    pre_smooth: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute per-image Sobel gradient magnitude with optional Gaussian pre-smooth.

    Returns ``(N, H, W)`` z-score-normalized per image so each output has
    zero mean and unit variance — removes gain/offset sensitivity for
    cross-detector loss comparisons.
    """
    img = images[:, None]  # (N, 1, H, W) for conv2d
    if pre_smooth > 0:
        ks = max(3, int(6 * pre_smooth) | 1)  # odd kernel size
        x = torch.arange(ks, dtype=dtype, device=device) - ks // 2
        g = torch.exp(-0.5 * (x / max(pre_smooth, 1e-6)) ** 2)
        g = g / g.sum()
        pad_h = ks // 2
        img = F.pad(img, (pad_h, pad_h, 0, 0), mode='reflect')
        img = F.conv2d(img, g.reshape(1, 1, 1, -1))
        img = F.pad(img, (0, 0, pad_h, pad_h), mode='reflect')
        img = F.conv2d(img, g.reshape(1, 1, -1, 1))
    sobel_column = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=dtype, device=device).reshape(1, 1, 3, 3)
    sobel_row = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=dtype, device=device).reshape(1, 1, 3, 3)
    img_pad = F.pad(img, (1, 1, 1, 1), mode='reflect')
    gradient_column = F.conv2d(img_pad, sobel_column)
    gradient_row = F.conv2d(img_pad, sobel_row)
    grad_mag = (gradient_row**2 + gradient_column**2).sqrt()[:, 0]
    mean = grad_mag.mean(dim=(-2, -1), keepdim=True)
    std = grad_mag.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return (grad_mag - mean) / std


def _regularize_knots(
    knots_batch: torch.Tensor,
    knots_prev: torch.Tensor,
    vander: torch.Tensor | None,
    max_shift_px: float | None,
    sigma_px: float | None,
    step_size: float | None,
) -> None:
    """Apply per-iteration knot regularization (in place on ``knots_batch``).

    Three independent stages, each gated by its parameter being non-None:
        1. Per-knot shift cap: clamp ``|new - prev|`` to ``max_shift_px``
           so the optimizer can't move any knot too far in one outer iter.
        2. Polynomial detrend + Gaussian smooth: keep low-order trends,
           smooth the residual along the scan-line dimension. Removes
           high-frequency optimizer wobble while preserving the drift signal.
        3. Step-size blend: ``new = prev + step_size · (new - prev)``,
           under-relaxes the update for stability across outer iterations.
    """
    # Knots are 4D ``(N, 2, R, K)`` everywhere — K=1 just has trailing 1.
    num_images, _, num_rows_knot, K = knots_batch.shape
    with torch.no_grad():
        if max_shift_px is not None:
            shift = knots_batch - knots_prev
            dist = torch.norm(shift, dim=1, keepdim=True)
            scale_factor = torch.clamp(max_shift_px / dist.clamp(min=1e-8), max=1.0)
            knots_batch.copy_(knots_prev + shift * scale_factor)
        if sigma_px is not None and sigma_px > 0 and vander is not None:
            # Smooth/detrend along the row axis.  Treat each (axis, intra-row
            # knot) slot as an independent series along rows by moving the row
            # dim last and flattening the leading channels.
            knots_flat = knots_batch.permute(0, 1, 3, 2).reshape(-1, num_rows_knot).T
            if vander.device.type == "mps":
                # MPS does not implement lstsq. The normalized polynomial basis
                # has at most four columns, so its full-rank system is small.
                if vander.shape[0] < vander.shape[1]:
                    coefficients = torch.linalg.lstsq(
                        vander.cpu(), knots_flat.cpu()
                    ).solution.to(vander.device)
                else:
                    normal_matrix = vander.T @ vander
                    coefficients = torch.linalg.solve(
                        normal_matrix, vander.T @ knots_flat
                    )
            else:
                coefficients = torch.linalg.lstsq(vander, knots_flat).solution
            trend = (vander @ coefficients).T
            residual = knots_flat.T - trend
            smoothed = drift_knots.gaussian_smooth_1d(residual, sigma_px)
            knots_batch.copy_(
                (smoothed + trend)
                .reshape(num_images, 2, K, num_rows_knot)
                .permute(0, 1, 3, 2))
        if step_size is not None:
            knots_batch.copy_(knots_prev + (knots_batch - knots_prev) * step_size)


def setup_loss_kernel(
    self,
    K: int,
    canvas_shape: tuple[int, int],
    loss: str = "mse",
):
    """Pick the K-aware compiled loss kernel and precompute its constants.

    Returns ``(loss_fn, loss_args)`` ready to be passed to
    ``_optimize_knots_adam`` / ``_optimize_knots_lbfgs``.

    K=1 path: scan_fast walk offsets per image (``row_scan_offsets`` /
    ``col_scan_offsets``).  K>=2 path: per-output-column segment indices
    + local fractions for the linear knot interpolation.  Constants are
    lifted out of the inner Adam / LBFGS loop so the compiled kernel
    sees them as static.

    ``loss`` selects the scalar: ``"mse"`` / ``"gradient_mse"`` use MSE
    kernels; ``"ncc"`` uses ``1 - mean NCC`` (brightness-invariant).
    """
    device, dtype = self._device, self._dtype
    num_images = self.shape[0]
    row_scale = 2.0 / (canvas_shape[0] - 1)
    col_scale = 2.0 / (canvas_shape[1] - 1)
    use_ncc = loss == "ncc"
    if K > 1:
        # Identical to _transform_coordinates_multi_knot's geometry so
        # apply_correction downstream inverts it cleanly.
        num_cols = self.imgs[0].shape[1]
        t = torch.linspace(0, 1, num_cols, dtype=dtype, device=device) * (K - 1)
        seg_idx = torch.clamp(t.long(), max=K - 2)
        seg_frac = t - seg_idx.to(dtype)
        fn = (
            _compiled_loss_fn_multi_ncc
            if use_ncc
            else _compiled_loss_fn_multi
        )
        return (fn, (seg_idx, seg_frac, row_scale, col_scale))
    # K=1: same scan-position vector projects onto row/col via scan_fast.
    u_t = [
        torch.as_tensor(self.u_per_image[i], dtype=dtype, device=device)
        for i in range(num_images)
    ]
    row_scan_offsets = torch.stack(
        [
            u_t[i] * (self.scan_fast[i][0] * (self.imgs[i].shape[0] - 1))
            for i in range(num_images)
        ]
    )
    col_scan_offsets = torch.stack(
        [
            u_t[i] * (self.scan_fast[i][1] * (self.imgs[i].shape[1] - 1))
            for i in range(num_images)
        ]
    )
    fn = (
        _compiled_loss_fn_single_ncc
        if use_ncc
        else _compiled_loss_fn_single
    )
    return (fn, (row_scan_offsets, col_scan_offsets, row_scale, col_scale))


def correct_nonrigid(
    self,
    *,
    num_knots: int | None = None,
    optimizer: str = "adam",
    num_refine_cycles: int = 16,
    knot_smoothing_sigma: float = 8.0,
    update_fraction: float | None = 0.8,
    trend_order: int = 1,
    max_image_shift: float = 32.0,
    optimizer_steps: int | None = 30,
    learning_rate: float | None = None,
    max_knot_step: float | None = None,
    fixed_scans: list[int] | None = None,
    loss: str = "auto",
    edge_smoothing_sigma: float = 1.0,
    early_stop_patience: int = 3,
    early_stop_rtol: float = 1e-4,
    min_iterations: int = 4,
    show_combined: bool = True,
    show_scans: bool = False,
    show_knots: bool = True,
    show_knot_plot: bool = False,
    show_report: bool = False,
    verbose: bool = True,
):
    """Remove residual scanline drift after affine or strip correction.

    The optimizer updates scanline knots against the other corrected scans or
    a fixed reference. Smooth knot regularization suppresses scanline jitter
    while retaining drift that changes gradually across the acquisition.

    Prefer :meth:`correct_strip` first when residual after affine is a
    smooth function of slow-scan position (deceleration / bend), especially
    for reference-mode EDS on pure lattices - free nonrigid with a large
    ``max_image_shift`` can lattice-alias. On atomic lattices keep
    ``max_image_shift`` small (e.g. 2).

    Calls are cumulative. On an atomic lattice, begin with
    ``max_image_shift=1`` or ``2`` and strong regularization (8-16 rows), then
    accept the stage only when common-mask NCC and the RGB overlay improve.
    In reference mode, the affine/strip solution already establishes the
    external reference frame, so nonrigid iterations preserve that frame
    instead of re-solving a potentially lattice-aliased global translation.

    Parameters
    ----------
    num_knots : int or None, default None
        Knots along every fast-scan line. ``None`` keeps the current layout;
        use more than one when residual drift changes within a scanline. The
        existing affine or strip field is preserved before non-rigid
        optimization begins.
    optimizer : str, default "adam"
        ``"adam"`` (first-order momentum) or ``"lbfgs"`` (quasi-Newton
        with strong-Wolfe line search). Adam is the fastest default for
        ≤1024 px images. LBFGS auto-scales the step size and is preferred
        for ≥2048 px or when the drift magnitude is unknown. Don't normalize
        inputs to [0, 1] when using LBFGS - strong-Wolfe needs absolute
        gradient magnitude and silently returns step=0 on unit-variance images.
    num_refine_cycles : int, default 16
        Outer iterations for alternating reference build + knot update.
    knot_smoothing_sigma : float, default 8.0
        Gaussian smoothing sigma for knot regularization. 4-12 typical
        for STEM data; smaller = finer per-row correction.
    update_fraction : float, default 0.8
        Step size for knot updates (0-1, lower = more conservative).
    trend_order : int, default 1
        Polynomial order for trend removal in knot regularization.
    max_image_shift : float, default 32.0
        Maximum shift for translation alignment between iterations.
        Adam's auto-``learning_rate`` derives from this - set close to your expected
        drift bound, otherwise Adam silently under-converges.
    optimizer_steps : int or None, default 30
        Inner optimizer steps per refinement cycle. ``None`` selects 30
        for Adam and 20 for LBFGS.
    learning_rate : float or None, default None
        Adam learning rate. When ``None``, it is derived as
        ``max_image_shift / (num_refine_cycles * optimizer_steps * 4)``.
        Adam's ``m/sqrt(v)`` update self-normalizes the gradient, so each
        step moves a knot by ~``learning_rate`` pixels regardless of intensity scale.
        Override only when you know the actual drift magnitude.
    fixed_scans : list[int] or None, default None
        Indices of images whose knots are frozen (their mean becomes the
        target for every moving image). Use ``[0]`` for single-sided
        alignment with a HAADF reference. Auto-set to ``[0]`` in
        reference mode (when the constructor was called with a 2-D ref +
        ≥3-D drifted dataset).
    loss : str, default "auto"
        Most drift workflows compare **similar detectors** (0°/90° HAADF,
        survey HAADF vs EDS-session HAADF, etc.). ``"auto"`` follows that
        typical case and resolves to ``"ncc"`` (coverage-masked normalized
        cross-correlation), so the train loss matches regional NCC reports.
        Explicit choices:

        - ``"ncc"`` - coverage-masked ``1 - mean NCC``; brightness / offset
          invariant; excludes unmeasured canvas fill. Default via ``"auto"``
          for similar-detector pairs.
        - ``"mse"`` - raw-intensity MSE when both images share the same
          detector and nearly identical gain (no intensity bias).
        - ``"gradient_mse"`` - Sobel-gradient MSE after pre-smooth +
          per-image z-score; only needed for **dissimilar / cross-detector**
          pairs (e.g. HAADF + virtual dark field).

    edge_smoothing_sigma : float, default 1.0
        Gaussian sigma applied before Sobel when ``loss="gradient_mse"``.
        Set to 0 to disable. Ignored for ``"mse"`` / ``"ncc"``.
    early_stop_patience : int, default 3
        Stop when ``patience`` consecutive iterations show no improvement.
        Set to ``num_refine_cycles`` to disable.
    early_stop_rtol : float, default 1e-4
        Minimum relative improvement to count as progress.
    min_iterations : int, default 4
        Floor before early stopping can trigger.
    show_combined, show_scans : bool
        Display knobs forwarded to the plot helpers.
    show_report : bool, default False
        Print all completed regional NCC checkpoints, including the
        nonrigid result.
    verbose : bool, default True
        Show refinement-cycle progress with the current and best mean
        absolute alignment error. Set to ``False`` for compact notebooks.

    Returns
    -------
    Self
        For method chaining.

    Examples
    --------
    Similar-detector HAADF pair (``loss="auto"`` → NCC):

    >>> dc = DriftCorrection(im0, im1, scan_direction_degrees=[0, 90])
    >>> dc.correct_affine(show_combined=False)
    >>> dc.correct_nonrigid()  # auto → ncc

    Atomic lattice: keep the shift tiny so nonrigid cannot lattice-alias:

    >>> dc.correct_nonrigid(max_image_shift=2)

    Let residual motion vary along the fast-scan direction:

    >>> dc.correct_nonrigid(num_knots=6, max_image_shift=2)

    Dissimilar / cross-detector only (HAADF + VDF):

    >>> dc.correct_nonrigid(loss="gradient_mse", knot_smoothing_sigma=8.0)

    Notes
    -----
    Plotting refreshes the cached warped images lazily. ``corrected()`` builds
    its result directly from the fitted knots.
    """
    if not hasattr(self, "knots"):
        raise RuntimeError("No knots found. Call .preprocess() before running alignment.")
    # Reloaded tensors start on the host; move solve state to the selected
    # device before optimization.
    self.imgs_t = [t.to(self._device) for t in self.imgs_t]
    self.knots = [k.to(self._device) for k in self.knots]
    for attr in ("_knots_after_affine", "_knots_after_strip", "_initial_knots"):
        snapshot = getattr(self, attr, None)
        if snapshot is not None:
            setattr(self, attr, [k.to(self._device) for k in snapshot])
    if num_knots is not None:
        drift_knots.resize_scanline_knots(self, num_knots)
    if loss == "auto":
        loss = "ncc"
    valid_losses = ("mse", "gradient_mse", "ncc")
    if loss not in valid_losses:
        raise ValueError(f"loss must be one of {valid_losses!r} or 'auto', got {loss!r}")
    if optimizer == "lbfgs" and self._normalized:
        import warnings

        warnings.warn(
            "normalize=True + LBFGS can cause silent convergence failure. "
            "Wolfe line search may return step=0 on unit-variance images. "
            "Consider using optimizer='adam' or normalize=False.",
            UserWarning,
            stacklevel=2,
        )
    # Reference-mode auto-anchors the reference image (index 0).
    if fixed_scans is None and self._reference_mode:
        fixed_scans = [0]
    fixed_set = frozenset(fixed_scans) if fixed_scans is not None else frozenset()
    moving_indices = [i for i in range(self.shape[0]) if i not in fixed_set]
    if fixed_set and not moving_indices:
        raise ValueError(
            f"All {self.shape[0]} images are in fixed_scans - nothing left to "
            "optimize. fixed_scans must leave at least one moving image."
        )
    device = self._device
    dtype = self._dtype
    num_images = self.shape[0]
    canvas_shape = (self.shape[1], self.shape[2])
    K_per_image = {self.knots[i].shape[2] for i in range(num_images)}
    if len(K_per_image) > 1:
        raise ValueError(f"All images must use the same number of knots, got {K_per_image}")
    K = K_per_image.pop()
    # Knots are 4D throughout the optimizer - K=1 just keeps the trailing 1
    # so downstream code (regularizer, warp, sync) doesn't branch on shape.
    knots_batch = (
        torch.stack([self.knots[i] for i in range(num_images)]).detach().requires_grad_(True)
    )
    num_rows_knot = knots_batch.shape[2]
    # a reloaded (AutoSerialize) object leaves imgs_t on CPU; alignment
    # tensors must live on the solve device or Sobel/warp kernels mismatch
    target_batch = torch.stack(self.imgs_t).to(self._device)
    loss_fn, loss_args = setup_loss_kernel(self, K, canvas_shape, loss=loss)
    optimizer_steps = (
        optimizer_steps if optimizer_steps is not None else (30 if optimizer == "adam" else 20)
    )
    if optimizer == "adam":
        # Auto-derive lr so the total movement budget covers a quarter
        # of max_image_shift. The safety factor of 4 (not 2) prevents
        # over-shooting at small image sizes where actual drift is well
        # below max_image_shift; at large sizes the same factor still
        # converges because the loss surface is smoother. See the `lr`
        # parameter docstring for the full rationale.
        adam_lr = (
            learning_rate
            if learning_rate is not None
            else max_image_shift / (num_refine_cycles * optimizer_steps * 4)
        )
        torch_optimizer = torch.optim.Adam([knots_batch], lr=adam_lr, fused=True)
    elif optimizer == "lbfgs":
        torch_optimizer = torch.optim.LBFGS(
            [knots_batch], lr=1.0, max_iter=optimizer_steps, line_search_fn="strong_wolfe"
        )
    else:
        raise ValueError(f"optimizer must be 'adam' or 'lbfgs', got {optimizer!r}")
    if knot_smoothing_sigma is not None and knot_smoothing_sigma > 0:
        x_knot = torch.arange(num_rows_knot, dtype=dtype, device=device)
        x_norm = (x_knot - x_knot.mean()) / x_knot.std()
        vander = torch.stack([x_norm**p for p in range(trend_order + 1)], dim=1)
    else:
        vander = None
    # For gradient_mse, warp the edge-filtered images instead of the
    # raw ones. Knots are spatial transforms independent of image
    # content, so optimizing in gradient space yields the same drift
    # field while being robust to intensity/contrast differences.
    # imgs_t_override threads Sobel images through warp_and_translate
    # without mutating self.imgs_t (which would silently corrupt
    # apply_correction, visualization, and error metrics afterwards).
    if loss == "gradient_mse":
        sobel_batch = sobel_gradient_magnitude(
            target_batch, edge_smoothing_sigma, device, dtype
        )
        imgs_t_override = [sobel_batch[i] for i in range(num_images)]
        target_batch = sobel_batch
    else:
        imgs_t_override = None
    # Affine (and an optional preceding strip stage) already establishes
    # the absolute frame in reference mode. Re-solving a global shift here
    # can hop by one lattice period after an otherwise good correction.
    # Mutual multi-scan mode still solves translation every outer step.
    solve_global_translation = not self._reference_mode
    warp_result = warp_and_translate(
        self,
        max_image_shift,
        upsample_factor=8,
        knots_batch=knots_batch,
        solve_translation=solve_global_translation,
        fixed_indices=fixed_set,
        imgs_t_override=imgs_t_override,
        return_weights=loss == "ncc",
    )
    if loss == "ncc":
        warped_t, coverage_weights_t = warp_result
    else:
        warped_t = warp_result
        coverage_weights_t = None
    # Build a boolean mask on device to zero fixed gradients efficiently.
    # knots_batch is always 4D ``(N, 2, R, K)``; broadcast over (2, R, K).
    if fixed_set:
        grad_mask = torch.ones(num_images, 1, 1, 1, dtype=dtype, device=device)
        for idx in fixed_set:
            grad_mask[idx] = 0.0
    error_buffer = []
    best_error = float("inf")
    patience_counter = 0
    pbar = tqdm(
        range(num_refine_cycles),
        desc=f"Solving nonrigid drift ({optimizer})",
        disable=not verbose,
    )
    for iter_idx in pbar:
        # Build the reference under no_grad: arithmetic on warped_t (an
        # inference tensor) would otherwise return an autograd-tracked
        # leaf, and the optimizer would build a graph through it.
        with torch.no_grad():
            if fixed_set:
                # Fixed images define the reference for all moving images.
                fixed_mean = warped_t[sorted(fixed_set)].mean(0)
                ref_batch = fixed_mean[None].expand(num_images, -1, -1)
            else:
                warped_sum = warped_t.sum(0)
                ref_batch = (warped_sum[None] - warped_t) / (num_images - 1)
            knots_prev = knots_batch.detach().clone()
            if loss == "ncc":
                # ``bilinear_kde_batch`` considers weights >= 1e-3 covered.
                # Reuse that exact ramp so the optimizer and renderer agree
                # about which reference pixels are measured rather than fill.
                coverage_batch = (coverage_weights_t / 1e-3).clamp(0.0, 1.0)
                if fixed_set:
                    fixed_coverage = coverage_batch[sorted(fixed_set)].amin(0)
                    ref_coverage_batch = fixed_coverage[None].expand(
                        num_images, -1, -1
                    )
                else:
                    ref_coverage_batch = torch.stack(
                        [
                            coverage_batch[
                                [j for j in range(num_images) if j != i]
                            ].amin(0)
                            for i in range(num_images)
                        ]
                    )
                cycle_loss_args = (ref_coverage_batch, *loss_args)
            else:
                cycle_loss_args = loss_args
        # Regularization alters the loss surface between outer iters, so
        # stale momentum / curvature history would push knots the wrong way.
        torch_optimizer.state.clear()
        if optimizer == "adam":
            _optimize_knots_adam(
                ref_batch,
                target_batch,
                knots_batch,
                loss_fn,
                cycle_loss_args,
                torch_optimizer,
                optimizer_steps,
                grad_mask=grad_mask if fixed_set else None,
            )
        else:
            _optimize_knots_lbfgs(
                ref_batch,
                target_batch,
                knots_batch,
                loss_fn,
                cycle_loss_args,
                torch_optimizer,
                grad_mask=grad_mask if fixed_set else None,
            )
        _regularize_knots(
            knots_batch,
            knots_prev,
            vander,
            max_knot_step,
            knot_smoothing_sigma,
            update_fraction,
        )
        # Restore fixed knots - regularization is a global smooth that
        # would subtly shift them via polynomial detrend + Gaussian blur.
        if fixed_set:
            with torch.no_grad():
                for idx in fixed_set:
                    knots_batch[idx] = knots_prev[idx]
        warp_result = warp_and_translate(
            self,
            max_image_shift,
            upsample_factor=8,
            knots_batch=knots_batch,
            solve_translation=solve_global_translation,
            fixed_indices=fixed_set,
            imgs_t_override=imgs_t_override,
            return_weights=loss == "ncc",
        )
        if loss == "ncc":
            warped_t, coverage_weights_t = warp_result
        else:
            warped_t = warp_result
        # Per-iter error stays on GPU; sync once after the loop
        images_mean = warped_t.mean(dim=0)
        iter_error = torch.mean(torch.abs(warped_t - images_mean[None]), dim=(1, 2))
        error_buffer.append(iter_error)
        # Early stopping: monitor post-iteration alignment quality
        current_error = float(iter_error.mean())
        if current_error < best_error * (1 - early_stop_rtol):
            best_error = current_error
            patience_counter = 0
        else:
            patience_counter += 1
        pbar.set_postfix_str(
            f"error={current_error:.4g}, best={best_error:.4g}",
            refresh=verbose,
        )
        if iter_idx >= min_iterations - 1 and patience_counter >= early_stop_patience:
            pbar.set_postfix_str(
                f"converged at cycle {iter_idx + 1}, "
                f"error={current_error:.4g}, best={best_error:.4g}",
                refresh=verbose,
            )
            break
    # Sync knots back; leave imgs_warped lazy so callers
    # that never plot avoid the GPU→CPU transfer of the warped stack.
    knots_final = knots_batch.detach()
    for img_idx in range(num_images):
        self.knots[img_idx][...] = knots_final[img_idx]
    self._images_warped_stale = True
    self._max_image_shift_cached = max_image_shift
    if error_buffer:
        # Transfer and append the full convergence history once.
        errors_np = torch.stack(error_buffer).cpu().numpy()  # (num_iterations, num_images)
        mode_col = np.full((len(errors_np), 1), 2.0)
        mean_col = errors_np.mean(axis=1, keepdims=True)
        new_rows = np.hstack((mode_col, mean_col, errors_np))
        if not hasattr(self, "error_track"):
            self.error_track = new_rows
        else:
            self.error_track = np.vstack((self.error_track, new_rows))

    drift_plot.show_after_step(
        self,
        "non-rigid",
        show_combined=show_combined,
        show_scans=show_scans,
        show_knots=show_knots,
    )
    if show_knot_plot:
        self.plot_knots()
    if show_report:
        print(self.report().to_string())
    return self
