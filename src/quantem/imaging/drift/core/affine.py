
import numpy as np
import torch
from torch.fft import fftfreq

from quantem.imaging.drift_utils import (
    bilinear_kde_batch,
    cross_corr_batch,
    transform_coordinates_single_knot,
    translate_align,
)


def align_affine(
    self,
    step: float = 0.01,
    num_tests: int = 9,
    refine: bool = True,
    upsample_factor: int = 8,
    max_image_shift: float | None = 32,
    chunk_size: int | None = None,
    show_merged: bool = True,
    show_images: bool = False,
    show_knots: bool = True,
    verbose: bool = False,
    **kwargs,
):
    """Correct affine drift between scan pairs using a batched grid search.

    Builds a grid of candidate linear-drift vectors, warps both images
    for each candidate, and picks the one with the lowest cross-correlation
    cost. An optional refinement pass subdivides the winning cell for
    sub-step accuracy. Without affine correction, per-scanline drift
    causes shear distortion that translation alignment alone cannot fix.

    Parameters
    ----------
    step : float
        Search resolution in pixels per scan line. The grid search
        tests drift rates from ``-step * num_tests/2`` to
        ``+step * num_tests/2`` px/line. For example, ``step=0.02``
        with ``num_tests=11`` searches drifts from -0.10 to +0.10
        px/line. Smaller values detect subtler drift but test more
        candidates.
    num_tests : int
        Number of drift rates to test along each axis. Must be odd
        so the grid is centered on zero drift. Total candidates
        ≈ ``π/4 * num_tests²``: ``num_tests=5`` → 21,
        ``num_tests=9`` → 61, ``num_tests=11`` → 97.
    refine : bool
        If True, run a second pass at ``step / (num_tests - 1)``
        resolution, centered on the coarse winner.
    upsample_factor : int
        Sub-pixel precision for measuring the translational shift
        between warped image pairs. 8 means 1/8-pixel precision.
        Higher values are more accurate but slower.
    max_image_shift : float or None
        Maximum allowed translational shift in pixels. Cross-correlation
        peaks beyond this radius are masked to reject spurious matches
        from noise or periodic artifacts. Set to None to allow any shift.
    chunk_size : int or None
        Number of candidates per pass. If None, all candidates at once.
        Set to a smaller value if you run out of memory.
    show_merged : bool
        Display the merged (averaged) image after alignment.
    show_images : bool
        Display each individual warped image after alignment.
    show_knots : bool
        Overlay knot positions on the displayed images.
    verbose : bool
        If True, print the top 5 candidate drift vectors with their
        cost and direction after the grid search. Useful for
        diagnosing ambiguous alignments or verifying the winning
        candidate has a clear margin over runner-ups.
    **kwargs
        Additional keyword arguments passed to the plotting functions.

    Returns
    -------
    DriftCorrection
        Self, for method chaining.

    Examples
    --------
    >>> drift = DriftCorrection.from_data(
    ...     images=[im0, im1], scan_direction_degrees=[0, 90])
    >>> drift.preprocess().align_affine(step=0.02, num_tests=11)
    """
    if self.shape[0] < 2:
        raise ValueError(
            f"align_affine requires at least 2 images (got {self.shape[0]}). "
            f"Provide image pairs with different scan directions."
        )
    if num_tests % 2 == 0:
        raise ValueError(
            f"num_tests must be odd (got {num_tests}). Try {num_tests + 1}."
        )
    # Build candidate grid with circular mask (~21% fewer than square)
    grid_axis = np.arange(-(num_tests - 1) / 2, (num_tests + 1) / 2)
    row_grid, col_grid = np.meshgrid(grid_axis, grid_axis, indexing="ij")
    circular_mask = row_grid**2 + col_grid**2 <= (num_tests / 2) ** 2
    drift_vectors = np.vstack((row_grid[circular_mask], col_grid[circular_mask])).T * step

    def _print_top_candidates(label, candidates, costs_tensor):
        costs_np = costs_tensor.cpu().numpy()
        ranked = np.argsort(costs_np)
        best_cost = costs_np[ranked[0]]
        print(f"  {label} - top 5 candidates:")
        for rank in range(min(5, len(ranked))):
            idx = ranked[rank]
            drift_row, drift_col = candidates[idx]
            magnitude = np.sqrt(drift_row**2 + drift_col**2)
            gap = (costs_np[idx] - best_cost) / best_cost * 100 if rank > 0 else 0
            print(f"    drift=({drift_row:+.4f}, {drift_col:+.4f}) px/line "
                  f"({magnitude:.4f} magnitude), cost={costs_np[idx]:.4f}"
                  f"{f' (+{gap:.1f}%)' if rank > 0 else ' (best)'}")

    def _apply_drift(drift_vec):
        for img_idx in range(self.shape[0]):
            scanline_offset = np.arange(self.knots[img_idx].shape[1]) - (self.knots[img_idx].shape[1] - 1) / 2
            self.knots[img_idx][0] += drift_vec[0] * scanline_offset[:, None]
            self.knots[img_idx][1] += drift_vec[1] * scanline_offset[:, None]

    def _search_and_apply(candidates, label):
        best_idx, costs = self._affine_grid_search_batch(candidates, upsample_factor, max_image_shift, chunk_size)
        _apply_drift(candidates[best_idx])
        if verbose:
            _print_top_candidates(label, candidates, costs)
        warped_t = self._warp_and_translate_torch(max_image_shift, upsample_factor)
        self.calculate_error(1, _warped_t=warped_t)
        return candidates[best_idx]

    drift_total = _search_and_apply(drift_vectors, "Coarse search")
    if refine:
        drift_fine = drift_vectors / (num_tests - 1)
        drift_total = drift_total + _search_and_apply(drift_fine, "Refine search")
    if verbose:
        num_rows = self.images[0].shape[0]
        drift_rate = np.sqrt(drift_total[0] ** 2 + drift_total[1] ** 2)
        total_shift = drift_rate * num_rows
        angle_deg = np.degrees(np.arctan2(drift_total[1], drift_total[0]))
        print(f"align_affine: step={step}, num_tests={num_tests} "
              f"({len(drift_vectors)} candidates), refine={refine}, "
              f"max_image_shift={max_image_shift}")
        msg = (f"Drift: ({drift_total[0]:+.4f}, {drift_total[1]:+.4f}) px/line, "
               f"{drift_rate:.4f} magnitude, {angle_deg:.1f} deg, "
               f"{total_shift:.1f} px total over {num_rows} lines")
        if self.images[0].sampling is not None:
            px_size = self.images[0].sampling[0]
            unit = self.images[0].units[0] if self.images[0].units else "px"
            msg += f" = {total_shift * px_size:.2f} {unit}"
        print(msg)
        err = self.error_track
        print(f"Error: {err[0, 1]:.2f} -> {err[-1, 1]:.2f} "
              f"({(err[0, 1] - err[-1, 1]) / err[0, 1] * 100:+.1f}%)")

    # Plots
    kwargs.pop("title", None)
    if show_merged:
        self.plot_merged_images(
            show_knots=show_knots,
            title="Merged: affine",
            **kwargs,
        )
    if show_images:
        self.plot_transformed_images(
            show_knots=show_knots,
            title=[f"Image {i}: affine" for i in range(self.shape[0])],
            **kwargs,
        )

    return self


@torch.inference_mode()
def _affine_grid_search_batch(self, drift_vectors, upsample_factor, max_image_shift, chunk_size=None):
    """Evaluate all candidate drift vectors in parallel.

    Warps both images for each candidate using ``bilinear_kde_batch``
    and scores alignment quality via ``cross_corr_batch``. Without
    batching, each candidate would be a separate Python iteration - this
    is the key operation that enables the 300x speedup.

    Parameters
    ----------
    drift_vectors : ndarray, shape (N, 2)
        Candidate drift vectors to test, columns are (row, col).
    upsample_factor : int
        Subpixel cross-correlation upsampling factor.
    max_image_shift : float or None
        Maximum allowed shift for cross-correlation peak search.
    chunk_size : int or None
        Number of candidates per pass. If None, all at once.

    Returns
    -------
    tuple[int, torch.Tensor]
        Index of the best candidate in ``drift_vectors``, and the full
        cost tensor of shape ``(N,)`` for all candidates (used by
        verbose mode to rank runner-ups).
    """
    device = self._device
    dtype = self._dtype
    num_candidates = drift_vectors.shape[0]
    drift_vectors_t = torch.tensor(drift_vectors, dtype=dtype, device=device)
    canvas_shape = (self.shape[1], self.shape[2])
    # Base coordinates shared across all candidates
    base_data = []
    for img_idx in range(2):
        knots_t = torch.tensor(self.knots[img_idx], dtype=dtype, device=device)
        row_base, col_base = transform_coordinates_single_knot(
            knots_t, self.scan_fast_t[img_idx], self.images[img_idx].shape)
        num_rows = self.knots[img_idx].shape[1]
        scanline_offset = (torch.arange(num_rows, dtype=dtype, device=device)
                           - (num_rows - 1) / 2)
        base_data.append((self.images_t[img_idx], row_base, col_base, scanline_offset))
    # Precompute shift mask and frequency grids (shared across chunks)
    shift_mask = None
    if max_image_shift is not None:
        canvas_rows, canvas_cols = canvas_shape
        freq_row = fftfreq(canvas_rows, 1.0 / canvas_rows, device=device, dtype=dtype)
        freq_col = fftfreq(canvas_cols, 1.0 / canvas_cols, device=device, dtype=dtype)
        shift_mask = freq_row[:, None] ** 2 + freq_col[None, :] ** 2 >= max_image_shift ** 2
    freq_grids = (
        fftfreq(canvas_shape[0], device=device, dtype=dtype)[:, None],
        fftfreq(canvas_shape[1], device=device, dtype=dtype)[None, :],
    )
    if chunk_size is None:
        chunk_size = self._auto_chunk_size(num_candidates, canvas_shape, dtype, device)
    on_cuda = torch.device(device).type == "cuda"
    chunked = on_cuda and chunk_size < num_candidates
    all_costs = []
    chunk_start = 0
    chunk_idx = 0
    while chunk_start < num_candidates:
        chunk_end = min(chunk_start + chunk_size, num_candidates)
        drift_chunk = drift_vectors_t[chunk_start:chunk_end]
        if chunk_idx == 0 and chunked:
            torch.cuda.reset_peak_memory_stats(device)
        warped_pair = []
        for img_idx in range(2):
            image_t, row_base, col_base, scanline_offset = base_data[img_idx]
            row_candidates = row_base[None] + drift_chunk[:, 0, None, None] * scanline_offset[None, :, None]
            col_candidates = col_base[None] + drift_chunk[:, 1, None, None] * scanline_offset[None, :, None]
            warped, _ = bilinear_kde_batch(
                row_candidates, col_candidates, image_t,
                canvas_shape, self.kde_sigma,
                self.pad_value[img_idx])
            warped_pair.append(warped)
        all_costs.append(cross_corr_batch(
            warped_pair[0], warped_pair[1],
            upsample_factor,
            max_shift_mask=shift_mask,
            freq_grids=freq_grids))
        # After chunk 0, replace the conservative static estimate with the
        # actual measured per-candidate cost and print one summary line so
        # the user can see how the chunking adapted to their GPU state.
        if chunk_idx == 0 and chunked:
            per_candidate_actual = torch.cuda.max_memory_allocated(device) / chunk_size
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            tuned_chunk_size = max(1, int(free_bytes * 0.5 / per_candidate_actual))
            tuned_chunk_size = min(tuned_chunk_size, num_candidates)
            if tuned_chunk_size > chunk_size:
                chunk_size = tuned_chunk_size
            num_chunks_final = 1 + (num_candidates - chunk_end + chunk_size - 1) // chunk_size
            print(
                f"  affine grid: {num_candidates} cand × {canvas_shape[0]}×{canvas_shape[1]}, "
                f"{per_candidate_actual / 1e9:.2f} GB/cand → {chunk_size}/chunk × {num_chunks_final} passes "
                f"({free_bytes / 1e9:.0f}/{total_bytes / 1e9:.0f} GB free)"
            )
        chunk_start = chunk_end
        chunk_idx += 1
    all_costs = torch.cat(all_costs)
    return torch.argmin(all_costs).item(), all_costs


def _auto_chunk_size(num_candidates, canvas_shape, dtype, device):
    """Pick a candidate-batch size that fits in current free GPU memory.

    Empirical per-candidate peak (measured at 4096×4096): bilinear KDE
    scatter buffers, gaussian smoothing temporaries, then cross-correlation
    FFT pairs (complex64) - together about ``32 × canvas_pixels``
    ``× dtype_bytes`` at peak. We sample free memory at call time, divide
    by that estimate with a 0.4 safety factor, and cap the result at
    ``num_candidates`` (no point splitting if it all fits).
    On CPU we just process all candidates at once - no VRAM constraint.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return num_candidates
    bytes_per_element = torch.finfo(dtype).bits // 8
    per_candidate_bytes = canvas_shape[0] * canvas_shape[1] * bytes_per_element * 32
    free_bytes, _ = torch.cuda.mem_get_info(device)
    chunk_size = max(1, int(free_bytes * 0.4 / per_candidate_bytes))
    return min(chunk_size, num_candidates)


@torch.inference_mode()
def _warp_and_translate_torch(
    self,
    max_image_shift: float | None,
    upsample_factor: int = 8,
    knots_batch: torch.Tensor | None = None,
    solve_translation: bool = True,
) -> torch.Tensor:
    """Regenerate warped images and solve translation on GPU.

    Three phases: warp → solve translation → re-warp. When ``knots_batch``
    is provided, reads/writes a single batched torch tensor (zero numpy
    crossings). Without it, falls back to ``self.knots`` (numpy) for
    compatibility with ``align_affine``.

    Set ``solve_translation=False`` to only warp and sync without
    re-solving translation - used after the nonrigid loop to populate
    ``self.images_warped`` from final knots.

    Parameters
    ----------
    max_image_shift : float or None
        Maximum allowed translational shift in pixels.
    upsample_factor : int
        Sub-pixel precision for cross-correlation (1/N pixel).
    knots_batch : torch.Tensor or None
        If provided, batched ``(N, 2, num_rows)`` torch tensor on GPU.
        Translation shifts are applied in-place. Skips numpy sync.
    solve_translation : bool
        If False, skip translation alignment (Phase 2+3). Only warp
        once using current knots and sync to CPU.

    Returns
    -------
    torch.Tensor
        Warped images on GPU, shape ``(num_images, H, W)``.
    """
    device = self._device
    dtype = self._dtype
    num_images = self.shape[0]
    canvas_shape = (self.shape[1], self.shape[2])

    def _warp_all(warped_t, weights_t):
        """Warp all images onto the canvas using current knots."""
        for img_idx in range(num_images):
            if knots_batch is not None:
                # transform_coordinates_single_knot expects (2, N, 1)
                knots_img = knots_batch[img_idx].detach()[:, :, None]
            else:
                knots_img = torch.as_tensor(self.knots[img_idx], dtype=dtype, device=device)
            row_t, col_t = transform_coordinates_single_knot(
                knots_img, self.scan_fast_t[img_idx], self.images[img_idx].shape)
            warped, weights = bilinear_kde_batch(
                row_t[None], col_t[None], self.images_t[img_idx], canvas_shape,
                self.kde_sigma, self.pad_value[img_idx])
            warped_t[img_idx] = warped[0]
            weights_t[img_idx] = weights[0]

    warped_t = torch.zeros(num_images, *canvas_shape, dtype=dtype, device=device)
    weights_t = torch.zeros_like(warped_t)
    _warp_all(warped_t, weights_t)
    if not solve_translation:
        self.images_warped.array[:] = warped_t.cpu().numpy()
        self.weights_warped.array[:] = weights_t.cpu().numpy()
        return warped_t
    # Solve translation shifts and apply to knots
    shifts_t = translate_align(warped_t, upsample_factor, max_image_shift)
    if knots_batch is not None:
        knots_batch[:, 0] += shifts_t[:, 0:1]
        knots_batch[:, 1] += shifts_t[:, 1:2]
    else:
        shifts_np = shifts_t.cpu().numpy()
        for img_idx in range(num_images):
            self.knots[img_idx][0] += shifts_np[img_idx, 0]
            self.knots[img_idx][1] += shifts_np[img_idx, 1]
    # Re-warp with corrected knots
    _warp_all(warped_t, weights_t)
    if knots_batch is None:
        self.images_warped.array[:] = warped_t.cpu().numpy()
        self.weights_warped.array[:] = weights_t.cpu().numpy()
    return warped_t
