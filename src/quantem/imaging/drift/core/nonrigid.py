
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from tqdm import tqdm

from quantem.imaging.drift import diagnostics
from quantem.imaging.drift.core.knots import gaussian_smooth_1d


def align_nonrigid(
    self,
    backend: str = "pytorch",
    optimizer_name: str = "adam",
    num_iterations: int = 8,
    regularization_sigma_px: float = 16.0,
    regularization_update_step_size: float | None = 0.8,
    regularization_poly_order: int = 1,
    max_image_shift: float | None = 32.0,
    adam_steps: int = 30,
    lr: float | None = None,
    lbfgs_max_iter: int = 20,
    max_optimize_iterations: int = 10,
    regularization_max_image_shift_px: float | None = None,
    solve_individual_rows: bool = True,
    show_merged: bool = True,
    show_images: bool = False,
    show_knots: bool = True,
    **kwargs,
):
    """
    Non-rigid drift correction using PyTorch (default) or SciPy backend.

    Parameters
    ----------
    backend : str, default "pytorch"
        Optimization backend.
          - "pytorch": GPU-accelerated batched optimization. Single-knot only.
          - "scipy": CPU L-BFGS row-by-row. Use when you need multi-knot
            mode (``number_knots > 1``), which the pytorch path does not
            yet support.
    optimizer_name : str, default "adam"
        PyTorch optimizer (ignored if backend="scipy").

        **"adam"** - first-order momentum optimizer. Default. Best when:
          - You want the fastest possible runtime, especially at image
            sizes ≤512 px where the per-step grid_sample is small and
            Adam's tight inner loop wins on launch overhead.
          - You're confident ``max_image_shift`` reflects the true drift
            bound (Adam's auto-lr derives from it; if it's too small,
            Adam silently under-converges).
          - You want bit-reproducible results across runs (LBFGS line
            search has subtle non-determinism from Wolfe condition checks).

          **Provisional override guidance** (validated on one real-data
          pair - Bob's gold-nanoparticle HAADF on a spectra background -
          and the synthetic chevron test; needs broader testing on
          diverse datasets before being treated as authoritative). If
          you override ``lr`` manually, the rough formula is
          ``expected_drift_px / (num_iterations * adam_steps)``.
          Indicative starting values for the default ``num_iterations=8,
          adam_steps=30`` (240 total steps):
            * ~5 px drift (synthetic chevron, small drift): ``lr≈0.02``
            * ~50-100 px drift (gold-nanoparticle HAADF, real STEM): ``lr≈0.5``
            * larger / unknown drift: prefer ``optimizer_name="lbfgs"``
              which auto-scales via line search and doesn't need this
              per-dataset tuning.

        **"lbfgs"** - quasi-Newton optimizer with strong-Wolfe line search.
        Best when:
          - The image is ≥512 px and you don't mind paying Python closure
            overhead for fewer total steps (typically 2-3× faster than
            Adam at 2048+ px because it converges in ~30 steps not 240).
          - You're unsure about the drift magnitude or don't want to think
            about ``lr`` tuning - LBFGS line search auto-scales the step
            without any hand-tuning.
          - You want quality over speed.

        **Failure modes to avoid:**
          - **Don't normalize inputs to [0, 1] when using LBFGS** -
            strong-Wolfe's curvature condition needs absolute gradient
            magnitude above a threshold; with normalized intensities the
            gradient is ~1e-4 and the line search returns step=0,
            producing zero correction silently. Adam is unaffected.
          - **Don't set ``max_image_shift`` smaller than your actual drift
            if using Adam with default ``lr=None``** - Adam's auto-derived
            lr scales with max_image_shift, so a too-small bound silently
            clamps how much drift Adam can recover. LBFGS is unaffected.

        If unsure, start with Adam (the default) for ≤1024 px images and
        switch to LBFGS for ≥2048 px or for unknown-drift exploratory work.

    Shared Parameters
    -----------------
    num_iterations : int, default 8
        Number of outer iterations for alternating optimization.
    regularization_sigma_px : float, default 16.0
        Gaussian smoothing sigma for knot regularization.
    regularization_update_step_size : float, default 0.8
        Step size for knot updates (0-1, lower = more conservative).
    regularization_poly_order : int, default 1
        Polynomial order for trend removal in knot regularization
        (used by both pytorch and scipy backends).
    max_image_shift : float, default 32.0
        Maximum shift for translation alignment between iterations.

    PyTorch Parameters (ignored if backend="scipy")
    -----------------------------------------------
    adam_steps : int, default 30
        Number of Adam optimization steps per outer iteration.
    lr : float or None, default None
        Learning rate for Adam. When None (default), auto-derived as
        ``max_image_shift / (num_iterations * adam_steps * 4)``.

        **Why auto-derive?** Adam's ``m/sqrt(v)`` update self-normalizes
        the gradient, so each step moves a knot by ~``lr`` pixels
        regardless of image intensity scale. The total movement budget
        is ``lr × num_iterations × adam_steps`` and is hard-bounded:
        Adam cannot find drift larger than that budget no matter how
        many iterations you give it. This means ``lr`` must be matched
        to the EXPECTED DRIFT MAGNITUDE IN PIXELS, not to gradient
        magnitude - the same default value that works on small synthetic
        drift will silently under-converge on real data with larger drift.

        The auto-derived formula reserves half the total step budget for
        search (covering up to ``max_image_shift / 2`` of nonlinear drift)
        and the other half for refinement near the minimum.

        Override with an explicit float when you know the actual drift
        magnitude - e.g. ``lr=2.0`` for very-large-drift in-situ data,
        or ``lr=0.005`` for atomic-resolution stable samples.
    lbfgs_max_iter : int, default 20
        Maximum LBFGS iterations per outer iteration (line search probes
        within each iter happen automatically). Only used when
        optimizer="lbfgs".

    SciPy Parameters (ignored if backend="pytorch")
    -----------------------------------------------
    max_optimize_iterations : int, default 10
        Maximum L-BFGS iterations per row.
    regularization_max_image_shift_px : float, optional
        Maximum allowed shift per iteration.
    solve_individual_rows : bool, default True
        If True, optimize each row independently.

    Display Parameters
    ------------------
    show_merged : bool, default True
        Show merged image after alignment.
    show_images : bool, default False
        Show individual aligned images.
    show_knots : bool, default True
        Overlay knot positions on visualizations.

    Notes
    -----
    With backend="pytorch", ``self.images_warped`` is left STALE after
    the loop and refreshed lazily on first access via plot methods or
    ``calculate_error()``. Code that reads ``self.images_warped.array``
    directly should call ``self._ensure_warped_images()`` first, or
    use ``generate_corrected_image()`` which builds its own warps from
    ``self.knots``.
    """
    if not hasattr(self, "knots"):
        raise RuntimeError(
            "No knots found. Call .preprocess() before running alignment."
        )
    if backend == "pytorch":
        device = self._device
        dtype = self._dtype
        num_images = self.shape[0]
        canvas_shape = (self.shape[1], self.shape[2])
        if any(self.knots[i].shape[2] != 1 for i in range(num_images)):
            raise NotImplementedError(
                "PyTorch backend only supports single knot. "
                "Use backend='scipy' for multiple knots.")
        knots_batch = torch.tensor(
            np.stack([self.knots[i][:, :, 0] for i in range(num_images)]),
            dtype=dtype, device=device, requires_grad=True)
        num_rows_knot = knots_batch.shape[2]
        target_batch = torch.stack(self.images_t)
        # Build u tensors once and reuse - same scan-position vector projects
        # onto row and col offsets via the per-image scan_fast components.
        u_t = [
            torch.as_tensor(self.interpolator[i].u, dtype=dtype, device=device)
            for i in range(num_images)
        ]
        row_scan_offsets = torch.stack([
            u_t[i] * (self.interpolator[i].scan_fast[0] * (self.images[i].shape[0] - 1))
            for i in range(num_images)
        ])
        col_scan_offsets = torch.stack([
            u_t[i] * (self.interpolator[i].scan_fast[1] * (self.images[i].shape[1] - 1))
            for i in range(num_images)
        ])
        row_scale = 2.0 / (canvas_shape[0] - 1)
        col_scale = 2.0 / (canvas_shape[1] - 1)
        if optimizer_name == "adam":
            # Auto-derive lr so the total movement budget covers a quarter
            # of max_image_shift. The safety factor of 4 (not 2) prevents
            # over-shooting at small image sizes where actual drift is well
            # below max_image_shift; at large sizes the same factor still
            # converges because the loss surface is smoother. See the `lr`
            # parameter docstring for the full rationale.
            adam_lr = lr if lr is not None else max_image_shift / (num_iterations * adam_steps * 4)
            optimizer = torch.optim.Adam([knots_batch], lr=adam_lr, fused=True)
        elif optimizer_name == "lbfgs":
            optimizer = torch.optim.LBFGS(
                [knots_batch], lr=1.0, max_iter=lbfgs_max_iter,
                line_search_fn="strong_wolfe")
        else:
            raise ValueError(f"optimizer_name must be 'adam' or 'lbfgs', got {optimizer_name!r}")
        if regularization_sigma_px is not None and regularization_sigma_px > 0:
            x_knot = torch.arange(num_rows_knot, dtype=dtype, device=device)
            x_norm = (x_knot - x_knot.mean()) / x_knot.std()
            vander = torch.stack([x_norm ** p for p in range(regularization_poly_order + 1)], dim=1)
        warped_t = self._warp_and_translate_torch(
            max_image_shift, upsample_factor=8, knots_batch=knots_batch)
        error_buffer = []
        for _ in tqdm(range(num_iterations), desc=f"Solving nonrigid drift ({optimizer_name})"):
            # Build the reference under no_grad: arithmetic on warped_t (an
            # inference tensor) would otherwise return an autograd-tracked
            # leaf, and the optimizer would build a graph through it.
            with torch.no_grad():
                warped_sum = warped_t.sum(0)
                ref_batch = (warped_sum[None] - warped_t) / (num_images - 1)
                knots_prev = knots_batch.detach().clone()
            # Regularization alters the loss surface between outer iters, so
            # stale momentum / curvature history would push knots the wrong way.
            optimizer.state.clear()
            if optimizer_name == "adam":
                self._optimize_knots_adam(
                    ref_batch, target_batch, knots_batch,
                    row_scan_offsets, col_scan_offsets, row_scale, col_scale,
                    optimizer, adam_steps)
            else:
                self._optimize_knots_lbfgs(
                    ref_batch, target_batch, knots_batch,
                    row_scan_offsets, col_scan_offsets, row_scale, col_scale,
                    optimizer)
            self._regularize_knots(
                knots_batch, knots_prev, vander,
                regularization_max_image_shift_px,
                regularization_sigma_px,
                regularization_update_step_size)
            warped_t = self._warp_and_translate_torch(
                max_image_shift, upsample_factor=8, knots_batch=knots_batch)
            # Per-iter error stays on GPU; sync once after the loop
            images_mean = warped_t.mean(dim=0)
            error_buffer.append(torch.mean(torch.abs(warped_t - images_mean[None]), dim=(1, 2)))
        # Sync knots back to numpy; leave images_warped lazy so callers
        # that never plot avoid the GPU→CPU transfer of the warped stack.
        knots_final = knots_batch.detach().cpu().numpy()
        for img_idx in range(num_images):
            self.knots[img_idx][:, :, 0] = knots_final[img_idx]
        self._images_warped_stale = True
        self._max_image_shift_cached = max_image_shift
        if error_buffer:
            # Build all error rows in one DtoH transfer + one vstack, instead of
            # the quadratic vstack-per-iteration pattern used by calculate_error.
            errors_np = torch.stack(error_buffer).cpu().numpy()  # (num_iterations, num_images)
            mode_col = np.full((len(errors_np), 1), 2.0)
            mean_col = errors_np.mean(axis=1, keepdims=True)
            new_rows = np.hstack((mode_col, mean_col, errors_np))
            if not hasattr(self, "error_track"):
                self.error_track = new_rows
            else:
                self.error_track = np.vstack((self.error_track, new_rows))
    else:
        for _ in tqdm(range(num_iterations), desc="Solving nonrigid drift (scipy)"):
            for ind in range(self.shape[0]):
                image_ref = np.delete(self.images_warped.array, ind, axis=0).mean(axis=0)
                knots_updated = self._optimize_knots_scipy(
                    ind, image_ref, self.knots[ind],
                    max_optimize_iterations=max_optimize_iterations,
                    solve_individual_rows=solve_individual_rows)
                if regularization_max_image_shift_px is not None:
                    knots_shift = knots_updated - self.knots[ind]
                    knots_dist = np.sqrt(np.sum(knots_shift**2, axis=0))
                    sub = knots_dist > regularization_max_image_shift_px
                    knots_updated[0][sub] = (self.knots[ind][0][sub]
                        + knots_shift[0][sub] * regularization_max_image_shift_px / knots_dist[sub])
                    knots_updated[1][sub] = (self.knots[ind][1][sub]
                        + knots_shift[1][sub] * regularization_max_image_shift_px / knots_dist[sub])
                if regularization_sigma_px is not None and regularization_sigma_px > 0:
                    knots_smoothed = knots_updated.copy()
                    for dim in range(2):
                        x = np.arange(knots_updated.shape[1])
                        for knot_ind in range(knots_updated.shape[2]):
                            y = knots_updated[dim, :, knot_ind]
                            coefs = np.polyfit(x, y, deg=regularization_poly_order)
                            trend = np.polyval(coefs, x)
                            residual = y - trend
                            residual_smooth = gaussian_filter(residual, sigma=regularization_sigma_px)
                            knots_smoothed[dim, :, knot_ind] = residual_smooth + trend
                    knots_updated = knots_smoothed
                if regularization_update_step_size is not None:
                    knots_updated = (self.knots[ind]
                        + (knots_updated - self.knots[ind]) * regularization_update_step_size)
                self.knots[ind] = knots_updated
            warped_t = self._warp_and_translate_torch(max_image_shift, upsample_factor=8)
            self.calculate_error(2, _warped_t=warped_t)

    diagnostics._record_stage(self, "nonrigid")

    if show_merged:
        self.plot_merged_images(
            show_knots=show_knots,
            title="Merged: non-rigid",
            **kwargs,
        )

    if show_images:
        self.plot_transformed_images(
            show_knots=show_knots,
            title=[f"Image {i}: non-rigid" for i in range(self.shape[0])],
            **kwargs,
        )

    return self


def _optimize_knots_adam(
    self, ref_batch, target_batch, knots_batch,
    row_scan_offsets, col_scan_offsets, row_scale, col_scale,
    optimizer, adam_steps,
):
    """Run ``adam_steps`` of Adam on a batched knot tensor against ``_compiled_loss_fn``."""
    ref_t = ref_batch[:, None]
    for _ in range(adam_steps):
        optimizer.zero_grad()
        loss = self._compiled_loss_fn(
            knots_batch, ref_t, target_batch,
            row_scan_offsets, col_scan_offsets, row_scale, col_scale)
        loss.backward()
        optimizer.step()


@torch.compile(mode="reduce-overhead", dynamic=False)
def _compiled_loss_fn(
    knots_batch, ref_t, target_batch,
    row_scan_offsets, col_scan_offsets, row_scale, col_scale,
):
    """Fused forward pass: knot offsets → grid → grid_sample → MSE loss.

    The MSE is averaged over both the batch (N images) and the spatial
    dims, so each image's gradient is scaled by 1/N relative to a
    per-image solve. Adam's adaptive step size absorbs the constant
    rescale; LBFGS line search rescales itself.
    """
    grid_row = (knots_batch[:, 0, :, None] + row_scan_offsets[:, None, :]) * row_scale - 1.0
    grid_col = (knots_batch[:, 1, :, None] + col_scan_offsets[:, None, :]) * col_scale - 1.0
    grid = torch.stack([grid_col, grid_row], dim=-1)
    warped = torch.nn.functional.grid_sample(
        ref_t, grid, mode='bilinear', align_corners=True, padding_mode='border')[:, 0]
    return ((warped - target_batch) ** 2).mean()


def _optimize_knots_lbfgs(
    self, ref_batch, target_batch, knots_batch,
    row_scan_offsets, col_scan_offsets, row_scale, col_scale,
    optimizer,
):
    """Run one LBFGS outer step (line search re-evaluates the closure several times)."""
    ref_t = ref_batch[:, None]
    def closure():
        optimizer.zero_grad()
        loss = self._compiled_loss_fn(
            knots_batch, ref_t, target_batch,
            row_scan_offsets, col_scan_offsets, row_scale, col_scale)
        loss.backward()
        return loss
    optimizer.step(closure)


def _regularize_knots(
    self, knots_batch, knots_prev, vander,
    max_shift_px, sigma_px, step_size,
):
    """Apply per-iteration knot regularization (in-place on ``knots_batch``).

    Three independent stages, each gated by its parameter being non-None:
        1. Per-knot shift cap: clamp ``|new - prev|`` to ``max_shift_px``
           so the optimizer can't move any knot too far in one outer iter.
        2. Polynomial detrend + Gaussian smooth: keep low-order trends,
           smooth the residual along the scan-line dimension. Removes
           high-frequency optimizer wobble while preserving the drift signal.
        3. Step-size blend: ``new = prev + step_size · (new - prev)``,
           under-relaxes the update for stability across outer iterations.
    """
    num_images, _, num_rows_knot = knots_batch.shape
    with torch.no_grad():
        if max_shift_px is not None:
            shift = knots_batch - knots_prev
            dist = torch.norm(shift, dim=1, keepdim=True)
            scale_factor = torch.clamp(max_shift_px / dist.clamp(min=1e-8), max=1.0)
            knots_batch.copy_(knots_prev + shift * scale_factor)
        if sigma_px is not None and sigma_px > 0:
            # Detrend + smooth all (N*2, num_rows) knots in one batched lstsq + smooth
            knots_flat = knots_batch.reshape(-1, num_rows_knot).T  # (num_rows, N*2)
            coefs, _, _, _ = torch.linalg.lstsq(vander, knots_flat)
            trend = (vander @ coefs).T  # (N*2, num_rows)
            residual = knots_batch.reshape(-1, num_rows_knot) - trend
            smoothed = gaussian_smooth_1d(residual, sigma_px)
            knots_batch.copy_((smoothed + trend).reshape(num_images, 2, num_rows_knot))
        if step_size is not None:
            knots_batch.copy_(knots_prev + (knots_batch - knots_prev) * step_size)


def _optimize_knots_scipy(
    self, idx: int, image_ref: np.ndarray, knots_init: np.ndarray,
    max_optimize_iterations: int = 10, solve_individual_rows: bool = True,
) -> np.ndarray:
    """SciPy L-BFGS optimization for one image."""
    shape_knots = knots_init.shape
    options = {"maxiter": max_optimize_iterations} if max_optimize_iterations else {}
    if solve_individual_rows:
        knots_updated = np.zeros_like(knots_init)
        for row_ind in range(knots_init.shape[1]):
            x0 = knots_init[:, row_ind, :].ravel()
            def cost_function(x):
                knots_row = x.reshape(shape_knots[0], shape_knots[2])
                xa, ya = self.interpolator[idx].transform_rows(knots_row)
                xf = np.clip(np.floor(xa).astype(int), 0, self.shape[1] - 2)
                yf = np.clip(np.floor(ya).astype(int), 0, self.shape[2] - 2)
                dx, dy = xa - xf, ya - yf
                warped = (image_ref[xf, yf] * (1 - dx) * (1 - dy)
                          + image_ref[xf + 1, yf] * dx * (1 - dy)
                          + image_ref[xf, yf + 1] * (1 - dx) * dy
                          + image_ref[xf + 1, yf + 1] * dx * dy)
                return np.sum((warped - self.images[idx].array[row_ind, :]) ** 2)
            result = minimize(cost_function, x0, method="L-BFGS-B", options=options)
            knots_updated[:, row_ind, :] = result.x.reshape((2, -1))
    else:
        x0 = knots_init.ravel()
        def cost_function(x):
            knots = x.reshape(shape_knots)
            xa, ya = self.interpolator[idx].transform_coordinates(knots)
            xf = np.clip(np.floor(xa).astype(int), 0, self.shape[1] - 2)
            yf = np.clip(np.floor(ya).astype(int), 0, self.shape[2] - 2)
            dx, dy = xa - xf, ya - yf
            warped = (image_ref[xf, yf] * (1 - dx) * (1 - dy)
                      + image_ref[xf + 1, yf] * dx * (1 - dy)
                      + image_ref[xf, yf + 1] * (1 - dx) * dy
                      + image_ref[xf + 1, yf + 1] * dx * dy)
            return np.sum((warped - self.images[idx].array) ** 2)
        result = minimize(cost_function, x0, method="L-BFGS-B", options=options)
        knots_updated = result.x.reshape(shape_knots)
    return knots_updated
