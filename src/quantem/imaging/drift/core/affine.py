"""Affine drift-rate search and scanline-knot updates."""

import time

import numpy as np
import torch
from torch.fft import fftfreq
from tqdm import tqdm

import quantem.imaging.drift.apply as drift_apply
import quantem.imaging.drift.plot as drift_plot
import quantem.imaging.drift.preprocess as preprocessing
import quantem.imaging.drift.report as report
from quantem.imaging.drift.core import knots as drift_knots
from quantem.imaging.drift.core.warping import (
    backward_warp_grid_search,
    cross_corr_batch,
    fixed_overlap_ncc,
    translate_align_pair_batch,
    warp_and_translate,
)


def drift_rate(correction) -> tuple[float, float]:
    """Return affine drift rate ``(row, col)`` in pixels per scanline.

    The affine checkpoint is used when later strip or non-rigid refinement has
    modified the live knots, keeping this measurement specific to the linear
    drift model.
    """
    if not hasattr(correction, "_initial_knots"):
        raise RuntimeError("Call preprocess() then correct_affine() first.")
    index = len(correction.knots) - 1
    if hasattr(correction, "_knots_after_affine"):
        delta = (
            correction._knots_after_affine[index]
            - correction._initial_knots[index]
        )
    else:
        delta = drift_knots.knot_delta_canvas(correction, index)
    lines = delta.shape[1]
    row = float(
        (delta[0, -1, 0] - delta[0, 0, 0]) / max(lines - 1, 1)
    )
    column = float(
        (delta[1, -1, 0] - delta[1, 0, 0]) / max(lines - 1, 1)
    )
    return row, column


def validate_num_rates(num_rates: int) -> int:
    """Validate the number of candidate rates sampled along each drift axis."""
    width = int(num_rates)
    if width < 3 or width % 2 == 0:
        raise ValueError(
            "num_rates must be an odd integer >= 3 so the search includes "
            f"zero drift; got {num_rates!r}."
        )
    return width


def candidate_grid(
    center: np.ndarray,
    radius: float,
    width: int,
    *,
    circular: bool,
) -> np.ndarray:
    """Build a small 2-D drift-rate grid around ``center``."""
    axis = np.linspace(-radius, radius, width, dtype=np.float64)
    row, col = np.meshgrid(axis, axis, indexing="ij")
    keep = row**2 + col**2 <= (radius * 1.001) ** 2 if circular else np.ones_like(row, dtype=bool)
    return center[None, :] + np.column_stack((row[keep], col[keep]))


def _apply_affine_rate(correction, rate, fixed_set: frozenset[int]) -> None:
    """Apply one ``(row, col)`` drift rate to every free scan."""
    rate = torch.as_tensor(
        rate,
        dtype=correction.knots[0].dtype,
        device=correction.knots[0].device,
    )
    for image_index in range(correction.shape[0]):
        if image_index not in fixed_set:
            drift_knots.interpolator(correction, image_index).apply_affine_shift(rate)


def _scan_disagreement(warped: torch.Tensor, scan_axis: int) -> torch.Tensor:
    """Return absolute differences from the mean corrected scan."""
    mean = warped.mean(dim=scan_axis, keepdim=True)
    return torch.abs(warped - mean)


def _downsampled_correction(correction, factor):
    """Build the same correction problem on average-pooled scan images."""
    images = [
        preprocessing.average_downsample_2d(np.asarray(image.array), factor)
        for image in correction.imgs
    ]
    if correction._reference_mode:
        pyramid = type(correction).from_reference(
            images[0],
            images[1],
            scan_direction_degrees=float(correction.scan_direction_degrees[1]),
        )
    else:
        pyramid = type(correction).from_images(
            *images,
            scan_direction_degrees=tuple(correction.scan_direction_degrees),
        )
    pyramid.preprocess(
        padding_fraction=correction.pad_fraction,
        padding_value="median",
        smoothing_sigma=correction.kde_sigma,
        num_knots=1,
        normalize=False,
        verbose=False,
    )
    return pyramid


def _delivered_candidate(
    correction,
    rate,
    starting_knots,
    fixed_set,
    max_image_shift,
    upsample_factor,
    bridge=None,
):
    """Score one affine rate after the translation applied to the final image."""
    correction.knots = [knot.clone() for knot in starting_knots]
    if bridge is not None:
        _apply_affine_rate(correction, bridge, fixed_set)
        warp_and_translate(
            correction,
            max_image_shift,
            upsample_factor,
            fixed_indices=fixed_set,
        )
        rate = rate - bridge
    _apply_affine_rate(correction, rate, fixed_set)
    warped = warp_and_translate(
        correction,
        max_image_shift,
        upsample_factor,
        fixed_indices=fixed_set,
    )
    return (
        float(_scan_disagreement(warped, 0).mean().cpu()),
        [knot.clone() for knot in correction.knots],
        warped.clone(),
        None,
    )


def _delivered_candidates(
    correction,
    rates,
    starting_knots,
    fixed_set,
    max_image_shift,
    upsample_factor,
):
    """Score a two-image affine neighborhood together when memory permits."""
    def sequential():
        return [
            _delivered_candidate(
                correction,
                rate,
                starting_knots,
                fixed_set,
                max_image_shift,
                upsample_factor,
            )
            for rate in rates
        ]

    if correction.shape[0] != 2 or torch.device(correction._device).type != "cuda":
        return sequential(), set()

    free_bytes, _ = torch.cuda.mem_get_info(correction._device)
    bytes_per_element = torch.finfo(correction._dtype).bits // 8
    estimated_bytes = (
        len(rates)
        * correction.shape[1]
        * correction.shape[2]
        * bytes_per_element
        * 64
    )
    if estimated_bytes > free_bytes * 0.4:
        return sequential(), {"memory"}

    correction.knots = [knot.clone() for knot in starting_knots]
    rates_t = torch.as_tensor(
        rates,
        dtype=correction._dtype,
        device=correction._device,
    )
    canvas_shape = (correction.shape[1], correction.shape[2])
    first_warps = []
    candidate_coordinates = []
    for image_index in range(correction.shape[0]):
        row_base, col_base, scanline_offset = drift_knots.interpolator(
            correction, image_index
        ).affine_candidate_base()
        row_candidates = (
            row_base[None]
            + rates_t[:, 0, None, None] * scanline_offset[None, :, None]
        )
        col_candidates = (
            col_base[None]
            + rates_t[:, 1, None, None] * scanline_offset[None, :, None]
        )
        candidate_coordinates.append((row_candidates, col_candidates))
        candidate_warped, _ = drift_knots.bilinear_kde_batch(
            row_candidates,
            col_candidates,
            correction.imgs_t[image_index],
            canvas_shape,
            correction.kde_sigma,
            correction.pad_value[image_index],
        )
        first_warps.append(candidate_warped)
    shifts = translate_align_pair_batch(
        torch.stack(first_warps, dim=1),
        upsample_factor,
        max_image_shift,
    )
    final_warps = []
    for image_index, (row_candidates, col_candidates) in enumerate(
        candidate_coordinates
    ):
        candidate_warped, _ = drift_knots.bilinear_kde_batch(
            row_candidates + shifts[:, image_index, 0, None, None],
            col_candidates + shifts[:, image_index, 1, None, None],
            correction.imgs_t[image_index],
            canvas_shape,
            correction.kde_sigma,
            correction.pad_value[image_index],
        )
        final_warps.append(candidate_warped)
    warped_batch = torch.stack(final_warps, dim=1)
    costs = _scan_disagreement(warped_batch, 1).mean(dim=(1, 2, 3))
    ranked = torch.sort(costs).values
    if len(rates) > 1 and float(ranked[1] - ranked[0]) <= max(
        1e-7, abs(float(ranked[0])) * 1e-6
    ):
        return sequential(), {"batch", "tie"}
    return (
        [
            (
                float(costs[index].cpu()),
                None,
                warped_batch[index].clone(),
                shifts[index].clone(),
            )
            for index in range(len(rates))
        ],
        {"batch"},
    )


def correct_affine(
    self,
    *,
    max_drift_rate: float | None = None,
    num_rates: int | None = None,
    refine: bool = True,
    max_image_shift: float | None | str = "auto",
    fixed_scans: list[int] | None = None,
    region: str | tuple[int, int, int, int] | None = None,
    region_smoothing_sigma: float = 4.0,
    show_combined: bool = True,
    show_scans: bool = False,
    show_knots: bool = True,
    show_knot_plot: bool = False,
    show_report: bool = False,
    verbose: bool = True,
    downsample: int | str = "auto",
    chunk_size: int | None = None,
):
    """Automatically correct the dominant linear drift between scans.

    With no search parameters, QuantEM prepares the scanline-knot canvas,
    chooses the minimum safe padding and translation bound, selects a
    broad-search downsampling factor, expands the drift-rate range when
    needed, and refines the winning candidate at full coordinate scale.
    It then updates the scanline knots; the corrected output itself is not
    downsampled.

    Advanced users may set ``max_drift_rate`` and ``num_rates`` together
    to reproduce an explicit candidate grid. Numerical controls such as
    ``downsample`` and ``chunk_size`` normally remain automatic. Affine
    correction is needed because translation alone cannot remove the
    shear caused by a changing position from one scanline to the next.

    Parameters
    ----------
    max_drift_rate : float or None, default None
        Search scale for the row and column drift-rate components, in
        pixels per scanline. QuantEM first samples each component from
        ``-max_drift_rate`` to ``+max_drift_rate``, then removes the corner
        combinations outside a circular 2-D search region. Increase this
        when the best candidate reaches the search boundary. Leave this
        and ``num_rates`` unset for the automatic image-pyramid search.
    num_rates : int or None, default None
        Number of candidate drift rates sampled along each of the row and
        column axes, including both bounds and zero. Must be odd. This is
        a per-axis count, not the final number of 2-D candidates: QuantEM
        first forms all ``num_rates ** 2`` ``(row_rate, column_rate)``
        combinations, then removes the square grid's corners with a
        circular mask. The mask avoids evaluating diagonal candidates
        whose combined drift magnitude is much larger than the requested
        per-axis search scale. For example, ``num_rates=11`` forms
        ``11 * 11 = 121`` combinations; the circular mask removes 24
        corner vectors, leaving 97 candidates to evaluate. Leave this and
        ``max_drift_rate`` unset for automatic range expansion and local
        refinement.
    refine : bool
        If True, run a second, finer search centered on the coarse winner.
    max_image_shift : float, None, or "auto", default "auto"
        Maximum allowed translational shift in pixels. Cross-correlation
        peaks beyond this radius are masked to reject spurious matches
        from noise or periodic artifacts. ``"auto"`` derives the bound
        from image geometry. Set to None to allow any shift.
    chunk_size : int or None
        Number of candidates per pass. If None, all candidates at once.
        Set to a smaller value if you run out of memory.
    downsample : int or "auto", default "auto"
        Average-pooling factor used only for the automatic broad affine
        search. ``"auto"`` selects the largest exact divisor up to 8.
        Set ``4``, ``2``, or ``1`` to retain more native detail when
        validating a difficult or highly periodic specimen. This does not
        downsample the corrected output or change the knot coordinates.
    fixed_scans : list[int] or None
        Indices of images whose knots should never be modified.
        Use ``fixed_scans=[0]`` for single-sided alignment where
        image 0 is a fixed reference (e.g. a merged HAADF) and only
        the remaining images are optimized. When ``None`` (default),
        all images receive the affine drift correction - the standard
        behavior for 0°/90° scan pairs.
    region : str, tuple of int, or None
        Region used to estimate the affine drift. The fitted affine model is
        still applied to the complete scans. Leave as None for the standard
        whole-image search. For periodic lattices, use ``diagnose_affine()``
        to identify a region containing distinctive defects, then pass its
        name (``"top_left"``, ``"top_right"``, ``"bottom_left"``, or
        ``"bottom_right"``) or
        ``(row_start, row_stop, column_start, column_stop)`` bounds.
    region_smoothing_sigma : float, default 4.0
        Gaussian smoothing used only when ``region`` is set. Smoothing helps
        the regional search follow distinctive defect structure instead of
        selecting a neighboring periodic lattice peak.
    show_combined : bool
        Display the combined RGB comparison after alignment.
    show_scans : bool
        Display each individual warped image after alignment.
    show_knots : bool, default True
        Show knot positions on top of the combined/per-scan plots
        (cheap, useful diagnostic).
    show_knot_plot : bool, default False
        Render the standalone 2-panel knot trajectory + per-row delta
        chart via ``dc.plot_knots()`` after this step.
    show_report : bool, default False
        Print a screenshot-friendly common/top/middle/bottom NCC table
        comparing the before and affine checkpoints.
    verbose : bool
        If True, show candidate progress when the search requires
        multiple batches, then print the top 5 drift vectors with their
        cost and direction. Useful for diagnosing ambiguous alignments or
        verifying the winning candidate has a clear margin over runner-ups.
    Returns
    -------
    DriftCorrection
        Self, for method chaining.

    Examples
    --------
    >>> drift = DriftCorrection(
    ...     im0, im1, scan_direction_degrees=[0, 90])
    >>> drift.correct_affine()

    Exact reproduction of a historical explicit grid:

    >>> drift.correct_affine(max_drift_rate=0.10, num_rates=11)

    Diagnose a periodic lattice, then anchor the affine fit to a region with
    distinctive structure:

    >>> figure, regions = drift.diagnose_affine(stage="initial")
    >>> drift.correct_affine(region="top_left")

    Single-sided alignment (4D-STEM VDF against a fixed HAADF reference):

    >>> drift = DriftCorrection(
    ...     haadf_ref, vdf, scan_direction_degrees=[0, 0])
    >>> drift.correct_affine(fixed_scans=[0])
    """
    # Reference-mode auto-anchors the reference image (index 0) so the
    # user doesn't repeat what they declared via the constructor reference mode.
    if fixed_scans is None and self._reference_mode:
        fixed_scans = [0]
    fixed_set = frozenset(fixed_scans) if fixed_scans is not None else frozenset()
    automatic = max_drift_rate is None and num_rates is None
    if (max_drift_rate is None) != (num_rates is None):
        raise ValueError(
            "Set both max_drift_rate and num_rates for an explicit grid, "
            "or leave both unset for automatic affine alignment."
        )
    if not automatic and downsample != "auto":
        raise ValueError(
            "downsample only applies when max_drift_rate and "
            "num_rates are left unset for automatic affine alignment."
        )
    if region is None and region_smoothing_sigma != 4.0:
        raise ValueError(
            "region_smoothing_sigma only applies when region is set. "
            "Pass region='top_left' or custom row/column bounds, or leave "
            "region_smoothing_sigma at its default."
        )
    if region is not None:
        _correct_affine_region(
            self,
            region=region,
            smoothing_sigma=region_smoothing_sigma,
            max_drift_rate=max_drift_rate,
            num_rates=num_rates,
            refine=refine,
            max_image_shift=max_image_shift,
            fixed_scans=fixed_scans,
            verbose=verbose,
            downsample=downsample,
            chunk_size=chunk_size,
        )
        drift_plot.show_after_step(
            self,
            "affine",
            show_combined=show_combined,
            show_scans=show_scans,
            show_knots=show_knots,
        )
        if show_knot_plot:
            self.plot_knots()
        if show_report:
            print(self.report().to_string())
        return self
    if not hasattr(self, "_initial_knots"):
        preparation_start = time.perf_counter()
        planned_rate = 0.25 if automatic else abs(float(max_drift_rate))
        translation_margin = (
            min(self.imgs[0].shape[:2]) * 0.125
            if self._built_from_datasets and not self._reference_mode
            else 0.0
        )
        if self._reference_mode:
            normalize = True
            normalization_reason = "reference_mode"
        elif self._built_from_datasets:
            normalize = False
            normalization_reason = "4dstem_collection"
        else:
            normalize, normalization_reason = preprocessing.automatic_alignment_normalization(
                self.imgs
            )
        padding = preprocessing.minimum_affine_padding_fraction(
            tuple(int(value) for value in self.imgs[0].shape[:2]),
            self.scan_direction_degrees,
            planned_rate,
            translation_margin,
        )
        self.preprocess(
            padding_fraction=padding,
            normalize=normalize,
            verbose=False,
            show_combined=False,
            show_scans=False,
            show_knots=False,
        )
        self._implicit_preprocess_seconds = time.perf_counter() - preparation_start
        self.preprocess_info.update(
            {
                "padding_mode": "implicit_auto",
                "planned_max_drift_rate": planned_rate,
                "translation_margin": translation_margin,
                "normalization_mode": "implicit_auto",
                "normalization_reason": normalization_reason,
                "seconds": self._implicit_preprocess_seconds,
            }
        )
        if verbose:
            print(
                "correct_affine: prepared scanline knots automatically; "
                f"padding={padding:.4g}, canvas={self.shape[1:]}, "
                f"{self._implicit_preprocess_seconds:.2f} s"
            )
    else:
        self._implicit_preprocess_seconds = 0.0
    if self.shape[0] < 2:
        raise ValueError(
            f"correct_affine requires at least 2 images (got {self.shape[0]}). "
            f"Provide image pairs with different scan directions."
        )
    if automatic:
        if not refine:
            raise ValueError("refine=False is only available for an explicit affine grid.")
        return automatic_affine_search(
            self,
            fixed_set=fixed_set,
            max_image_shift=max_image_shift,
            show_combined=show_combined,
            show_scans=show_scans,
            show_knots=show_knots,
            show_knot_plot=show_knot_plot,
            show_report=show_report,
            verbose=verbose,
            upsample_factor=8,
            chunk_size=chunk_size,
            pyramid_downsample=downsample,
        )
    # Translation-peak refinement is an internal numerical detail. Eight
    # is accurate (~0.01 px after the final parabolic fit) and is not a
    # meaningful microscope control, so the public API does not expose it.
    upsample_factor = 8
    num_tests = validate_num_rates(num_rates)
    if max_image_shift == "auto":
        max_image_shift = 256.0
    # Build candidate grid with circular mask (~21% fewer than square)
    grid_axis = np.arange(-(num_tests - 1) / 2, (num_tests + 1) / 2)
    row_grid, col_grid = np.meshgrid(grid_axis, grid_axis, indexing="ij")
    circular_mask = row_grid**2 + col_grid**2 <= (num_tests / 2) ** 2
    drift_rate_step = max_drift_rate / ((num_tests - 1) / 2)
    drift_vectors = (
        np.vstack((row_grid[circular_mask], col_grid[circular_mask])).T * drift_rate_step
    )

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
            print(
                f"    drift=({drift_row:+.4f}, {drift_col:+.4f}) px/line "
                f"({magnitude:.4f} magnitude), cost={costs_np[idx]:.4f}"
                f"{f' (+{gap:.1f}%)' if rank > 0 else ' (best)'}"
            )

    def _search_and_apply(candidates, label, accumulated_drift=None):
        # When fixed_indices is set, backward_warp_grid_search scores
        # absolute drift rates on the raw images (not canvas-warped).
        # After the coarse pass, _apply_drift bakes the coarse drift into
        # the knots, but the raw images are unchanged - so the refine
        # candidates (small deltas) must be offset by the accumulated
        # drift so that backward_warp_grid_search tests the correct total
        # drift rates.
        search_candidates = candidates
        if fixed_set and accumulated_drift is not None:
            search_candidates = candidates + accumulated_drift[None, :]
        best_idx, costs = grid_search_batch(
            self,
            search_candidates,
            upsample_factor,
            max_image_shift,
            chunk_size,
            fixed_indices=fixed_set,
            progress_desc=f"Affine {label.lower()}" if verbose else None,
        )
        _apply_affine_rate(self, candidates[best_idx], fixed_set)
        if verbose:
            _print_top_candidates(label, candidates, costs)
        warped_t = warp_and_translate(
            self,
            max_image_shift, upsample_factor, fixed_indices=fixed_set
        )
        report.record_error(self, 1, warped_t)

        # Confidence: cost gap between best and runner-up (%)
        costs_np = costs.cpu().numpy()
        ranked = np.argsort(costs_np)
        best_cost = costs_np[ranked[0]]
        runner_up = costs_np[ranked[1]] if len(ranked) > 1 else best_cost
        margin = (runner_up - best_cost) / (best_cost + 1e-12) * 100
        return candidates[best_idx], margin

    drift_total, coarse_margin = _search_and_apply(drift_vectors, "Coarse search")
    if refine:
        drift_fine = drift_vectors / (num_tests - 1)
        dt, refine_margin = _search_and_apply(
            drift_fine, "Refine search", accumulated_drift=drift_total
        )
        drift_total = drift_total + dt
        self.affine_confidence_margin = refine_margin
    else:
        self.affine_confidence_margin = coarse_margin
    if verbose:
        num_rows = self.imgs[0].shape[0]
        drift_rate = np.sqrt(drift_total[0] ** 2 + drift_total[1] ** 2)
        total_shift = drift_rate * num_rows
        angle_deg = np.degrees(np.arctan2(drift_total[1], drift_total[0]))
        print(
            f"correct_affine: max_drift_rate={max_drift_rate:g}, "
            f"num_rates={num_tests} per axis; "
            f"{len(drift_vectors)} drift vectors evaluated; refine={refine}, "
            f"max_image_shift={max_image_shift}"
        )
        msg = (
            f"Drift: ({drift_total[0]:+.4f}, {drift_total[1]:+.4f}) px/line, "
            f"{drift_rate:.4f} magnitude, {angle_deg:.1f}°, "
            f"{total_shift:.1f} px total over {num_rows} lines"
        )
        if self.imgs[0].sampling is not None:
            px_size = self.imgs[0].sampling[0]
            unit = self.imgs[0].units[0] if self.imgs[0].units else "px"
            msg += f" = {total_shift * px_size:.2f} {unit}"
        print(msg)
        err = self.error_track
        print(
            f"Error: {err[0, 1]:.2f} -> {err[-1, 1]:.2f} "
            f"({(err[0, 1] - err[-1, 1]) / err[0, 1] * 100:+.1f}%)"
        )
        margin = self.affine_confidence_margin
        confidence = "high" if margin > 5 else "low" if margin < 2 else "moderate"
        print(f"Confidence: {margin:.1f}% cost margin to runner-up ({confidence})")

    drift_plot.show_after_step(
        self,
        "affine",
        show_combined=show_combined,
        show_scans=show_scans,
        show_knots=show_knots,
    )
    if show_knot_plot:
        self.plot_knots()
    self._knots_after_affine = [k.clone() for k in self.knots]
    # Knots moved, so the cached warped stack no longer matches them.
    # correct_strip and correct_nonrigid mark this too; without it every
    # display after an affine-only solve draws the pre-alignment state.
    self._images_warped_stale = True
    if show_report:
        print(self.report().to_string())
    return self


def _correct_affine_region(
    self,
    *,
    region: str | tuple[int, int, int, int],
    smoothing_sigma: float = 4.0,
    max_drift_rate: float | None = None,
    num_rates: int | None = None,
    refine: bool = True,
    max_image_shift: float | None | str = "auto",
    fixed_scans: list[int] | None = None,
    verbose: bool = True,
    downsample: int | str = "auto",
    chunk_size: int | None = None,
):
    """Fit one affine model from a trusted region and apply it to both scans."""
    image_rows, image_columns = self.imgs[0].shape[:2]
    middle_row = image_rows // 2
    middle_column = image_columns // 2
    quadrants = {
        "top_left": (0, middle_row, 0, middle_column),
        "top_right": (0, middle_row, middle_column, image_columns),
        "bottom_left": (middle_row, image_rows, 0, middle_column),
        "bottom_right": (
            middle_row,
            image_rows,
            middle_column,
            image_columns,
        ),
    }
    if isinstance(region, str):
        if region not in quadrants:
            raise ValueError(
                f"You entered region={region!r}. Choose from "
                f"{sorted(quadrants)} or provide four pixel bounds."
            )
        bounds = quadrants[region]
        region_name = region
    else:
        bounds = tuple(int(value) for value in region)
        if len(bounds) != 4:
            raise ValueError(
                "region needs four bounds: "
                "(row_start, row_stop, column_start, column_stop)."
            )
        region_name = "custom"
    row_start, row_stop, column_start, column_stop = bounds
    if not (
        0 <= row_start < row_stop <= image_rows
        and 0 <= column_start < column_stop <= image_columns
    ):
        raise ValueError(
            f"region bounds {bounds} are outside the image shape "
            f"{(image_rows, image_columns)}."
        )
    region_slice = (
        slice(row_start, row_stop),
        slice(column_start, column_stop),
    )

    self.preprocess(
        padding_fraction=0.25,
        smoothing_sigma=smoothing_sigma,
        num_knots=1,
        normalize=True,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    panels = drift_apply.comparison_panels(self, stage="initial")
    angles = np.asarray(self.scan_direction_degrees, dtype=float)
    relative_angles = (angles - angles[0] + 180.0) % 360.0 - 180.0
    quarter_turns = np.rint(-relative_angles / 90.0).astype(int)
    regional_images = [
        np.ascontiguousarray(
            np.rot90(image[region_slice], -quarter_turns[index])
        )
        for index, image in enumerate(panels["raw_scans"])
    ]
    regional = type(self).from_images(
        *regional_images,
        scan_direction_degrees=tuple(relative_angles),
        device=self.device,
    )
    regional.preprocess(
        padding_fraction=0.25,
        smoothing_sigma=smoothing_sigma,
        num_knots=1,
        normalize=False,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    regional.correct_affine(
        max_drift_rate=max_drift_rate,
        num_rates=num_rates,
        refine=refine,
        max_image_shift=max_image_shift,
        fixed_scans=fixed_scans,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=verbose,
        downsample=downsample,
        chunk_size=chunk_size,
    )

    solved_knots = [knots.clone() for knots in regional.knots]
    regional.knots = [knots.clone() for knots in regional._initial_knots]
    _apply_affine_rate(regional, regional.drift_rate, frozenset())
    translations = [
        (solved - rate_only).mean(dim=(1, 2))
        for solved, rate_only in zip(
            solved_knots,
            regional.knots,
            strict=True,
        )
    ]
    fixed_set = frozenset(fixed_scans) if fixed_scans is not None else frozenset()
    _apply_affine_rate(self, regional.drift_rate, fixed_set)
    for index, (knots, translation) in enumerate(
        zip(self.knots, translations, strict=True)
    ):
        if index in fixed_set:
            continue
        knots[0] += translation[0]
        knots[1] += translation[1]
    self._knots_after_affine = [knots.clone() for knots in self.knots]
    self._images_warped_stale = True
    self.affine_search_info = {
        **regional.affine_search_info,
        "strategy": "trusted_region",
        "trusted_region": region_name,
        "trusted_region_bounds_row_column": list(bounds),
        "full_image_num_knots": 1,
    }
    return self


@torch.inference_mode()
def automatic_affine_search(
    self,
    *,
    fixed_set: frozenset[int],
    max_image_shift: float | None | str,
    show_combined: bool,
    show_scans: bool,
    show_knots: bool,
    show_knot_plot: bool,
    show_report: bool,
    verbose: bool,
    upsample_factor: int,
    chunk_size: int | None,
    pyramid_downsample: int | str,
):
    """Find the affine drift basin cheaply, then verify the delivered result.

    Broad rates are searched on an exact average-pooled pyramid. Small local
    grids recover native-coordinate precision, and final candidates are ranked
    after rigid translation because that is the corrected image users receive.
    """
    # The four phases stay together because each one narrows or verifies the
    # result of the phase before it. Read from top to bottom, the solver moves
    # from a cheap broad search to the final image delivered to the scientist.
    # Splitting this sequence across a one-use class would hide that numerical
    # story behind temporary object state.

    # Preserve the untouched state so every fallback starts from the same scan
    # geometry instead of accumulating shifts from a rejected candidate.
    start_time = time.perf_counter()
    starting_knots = [knot.clone() for knot in self.knots]
    initial_error_row = np.asarray(self.error_track[-1], dtype=np.float64).copy()
    image_shape = tuple(int(value) for value in self.imgs[0].shape[:2])
    factor = preprocessing.resolve_downsample(
        pyramid_downsample,
        image_shape,
    )

    coarse = _downsampled_correction(self, factor)

    # Pooling both scans preserves a mutual alignment objective. A fixed
    # reference is more sensitive to pooling bias, so it refines natively.
    reference_search = bool(fixed_set)
    if reference_search or any(size % 2 for size in image_shape):
        refine_factor = 1
    elif max(image_shape) <= 256:
        refine_factor = 2
    else:
        refine_factor = 4
    refine_object = (
        self if refine_factor == 1 else _downsampled_correction(self, refine_factor)
    )
    if max_image_shift == "auto":
        native_shift = max(
            16.0,
            min(image_shape) * (0.0625 if reference_search else 0.25),
        )
    else:
        native_shift = max_image_shift
    coarse_shift = None if native_shift is None else max(2.0, float(native_shift) / factor)
    refine_shift = (
        None if native_shift is None else max(2.0, float(native_shift) / refine_factor)
    )

    # -------------------------------------------------------------------------
    # Phase 1 of 4 — Find the broad drift-rate basin cheaply
    # Search nested circular grids on the pooled scans. Stop expanding when the
    # winner is safely inside the tested physical drift-rate radius.
    # -------------------------------------------------------------------------
    center = np.zeros(2, dtype=np.float64)
    best_center = center.copy()
    best_cost = np.inf
    coarse_step = 0.0
    evaluations = 0
    history = []
    coarse_cost_cache: dict[tuple[float, float], float] = {}
    broad_candidate_requests = 0
    if reference_search:
        radii = (0.10, 0.20, 0.40, 0.80, 1.60)
    elif self.shape[0] == 2 and max(image_shape) <= 256:
        # On small mutual scans, kernel-launch overhead dominates and the
        # local 2x grids supply the fine resolution. Start broad; evaluate
        # 0.25 only when the 0.20 winner approaches the boundary. Figure 5
        # stays interior and drops from 203 to 105 total evaluations.
        radii = (0.20, 0.25)
    else:
        radii = (0.05, 0.10, 0.20, 0.25)
    for radius in radii:
        candidates = candidate_grid(
            np.zeros(2),
            radius,
            9,
            circular=True,
        )
        broad_candidate_requests += len(candidates)
        # Absolute broad grids are nested: the 0.10 grid repeats 13
        # vectors from 0.05, and each doubled radius does the same. Cache
        # only identical coordinates; stage ordering and costs stay exact.
        candidate_keys = [tuple(np.round(candidate, 12)) for candidate in candidates]
        new_indices = [
            index for index, key in enumerate(candidate_keys) if key not in coarse_cost_cache
        ]
        if new_indices:
            _, new_costs = grid_search_batch(
                coarse,
                candidates[new_indices],
                upsample_factor=max(2, upsample_factor // 2),
                max_image_shift=coarse_shift,
                chunk_size=chunk_size,
                fixed_indices=fixed_set,
                # A second top-1 score only guards the broad basin. Once the
                # basin is chosen, the established objective owns refinement.
                fixed_overlap_check=True,
            )
            evaluations += len(new_indices)
            for index, cost in zip(
                new_indices,
                new_costs.cpu().tolist(),
                strict=True,
            ):
                coarse_cost_cache[candidate_keys[index]] = float(cost)
        costs = np.asarray(
            [coarse_cost_cache[key] for key in candidate_keys],
            dtype=np.float64,
        )
        best_index = int(np.argmin(costs))
        stage_cost = float(costs[best_index])
        stage_center = candidates[int(best_index)]
        previous_best = best_cost
        if stage_cost < best_cost:
            best_cost = stage_cost
            best_center = stage_center.copy()
            coarse_step = 2.0 * radius / 8.0
        boundary_fraction = float(np.linalg.norm(stage_center) / radius)
        history.append(
            {
                "stage": "pyramid",
                "radius": float(radius),
                "best_rate": stage_center.tolist(),
                "cost": stage_cost,
            }
        )
        if not reference_search and boundary_fraction < 0.72:
            break
        if not reference_search and stage_cost > previous_best * 1.001:
            break

    # -------------------------------------------------------------------------
    # Phase 2 of 4 — Refine the broad winner in native coordinates
    # Repeatedly halve a local 3x3 grid. The images may remain pooled for speed,
    # but the drift rate and radius always use native pixels per scanline.
    # -------------------------------------------------------------------------
    center = best_center
    radius = max(coarse_step * 1.5, 0.002)
    bridge_center = None
    native_levels = 8 if reference_search else 5
    final_costs = None
    final_index = 0
    refinement_evaluations = 0
    native_evaluations = 0
    level = 0
    while level < native_levels:
        previous_center = center.copy()
        candidates = candidate_grid(
            center,
            radius,
            3,
            circular=False,
        )
        final_index, final_costs = grid_search_batch(
            refine_object,
            candidates,
            upsample_factor=upsample_factor,
            max_image_shift=refine_shift,
            chunk_size=chunk_size,
            fixed_indices=fixed_set,
        )
        evaluations += len(candidates)
        refinement_evaluations += len(candidates)
        if refine_factor == 1:
            native_evaluations += len(candidates)
        center = candidates[int(final_index)]
        if (
            level == 0
            and refine_factor == 4
            and np.max(np.abs(center - previous_center)) >= radius * 0.999
        ):
            # A boundary winner on the first 4x local grid says the
            # coarser pyramid changed the correlation basin.  Retry at
            # 2x automatically; periodic WS2 needs this, while the other
            # real workflows retain the faster 4x refinement.
            history.append(
                {
                    "stage": "refine_4x_probe_rejected",
                    "radius": float(radius),
                    "best_rate": center.tolist(),
                    "cost": float(final_costs[int(final_index)].cpu()),
                }
            )
            refine_factor = 2
            refine_object = _downsampled_correction(self, refine_factor)
            refine_shift = (
                None if native_shift is None else max(2.0, float(native_shift) / refine_factor)
            )
            center = best_center.copy()
            radius = max(coarse_step * 1.5, 0.002)
            continue
        history.append(
            {
                "stage": (
                    f"native_{level + 1}"
                    if refine_factor == 1
                    else f"refine_{refine_factor}x_{level + 1}"
                ),
                "radius": float(radius),
                "best_rate": center.tolist(),
                "cost": float(final_costs[int(final_index)].cpu()),
            }
        )
        if bridge_center is None and radius * image_shape[0] <= 40.0:
            bridge_center = center.copy()
        radius *= 0.5
        level += 1

    # -------------------------------------------------------------------------
    # Phase 3 of 4 — Remove the small drift-rate bias from pooling
    # Test one native-resolution sentinel grid before accepting the pyramid
    # result, while retaining nearly all of the speed from downsampling.
    # -------------------------------------------------------------------------
    if not reference_search and refine_factor > 1:
        candidates = candidate_grid(
            center,
            radius * 2.0,
            3,
            circular=True,
        )
        final_index, final_costs = grid_search_batch(
            self,
            candidates,
            upsample_factor=upsample_factor,
            max_image_shift=native_shift,
            chunk_size=chunk_size,
            fixed_indices=fixed_set,
        )
        evaluations += len(candidates)
        native_evaluations += len(candidates)
        center = candidates[int(final_index)]
        history.append(
            {
                "stage": "native_sentinel",
                "radius": float(radius * 2.0),
                "best_rate": center.tolist(),
                "cost": float(final_costs[int(final_index)].cpu()),
            }
        )

    # -------------------------------------------------------------------------
    # Phase 4 of 4 — Verify the image that will actually be delivered
    # Rank the final neighborhood after translation correction. This prevents
    # periodic specimens from selecting an adjacent, translation-equivalent
    # basin that scores well during the cheaper pyramid search.
    # -------------------------------------------------------------------------
    validation_costs = None
    validation_evaluations = 0
    validation_status = set()
    translation_verification_shift = np.zeros(2, dtype=np.float64)
    translation_verification_gain = 0.0
    if not reference_search:
        # The batched forward-scatter score is excellent for locating the
        # drift-rate basin, but the public result is judged *after* the
        # translation solve.  Polish that delivered objective with five
        # geometry-scaled trials.  This also avoids committing to a
        # neighboring lattice-translation basin on periodic specimens.
        validation_radius = 2.0 / image_shape[0]
        cache = {}

        row_rates = np.stack(
            [
                center + np.asarray((row_delta, 0.0))
                for row_delta in (
                    -validation_radius,
                    0.0,
                    validation_radius,
                )
            ]
        )
        row_results, status = _delivered_candidates(
            self,
            row_rates,
            starting_knots,
            fixed_set,
            native_shift,
            upsample_factor,
        )
        validation_status.update(status)
        row_trials = []
        for rate, result in zip(row_rates, row_results, strict=True):
            cache[tuple(rate)] = result
            row_trials.append((result[0], rate))
        _, row_center = min(row_trials, key=lambda item: item[0])
        col_trials = []
        new_col_rates = []
        for col_delta in (
            -validation_radius,
            0.0,
            validation_radius,
        ):
            rate = row_center + np.asarray((0.0, col_delta))
            key = tuple(rate)
            if key not in cache:
                new_col_rates.append(rate)
        if new_col_rates:
            new_col_rates_array = np.stack(new_col_rates)
            new_col_results, status = _delivered_candidates(
                self,
                new_col_rates_array,
                starting_knots,
                fixed_set,
                native_shift,
                upsample_factor,
            )
            validation_status.update(status)
            for rate, result in zip(
                new_col_rates_array,
                new_col_results,
                strict=True,
            ):
                cache[tuple(rate)] = result
        for col_delta in (
            -validation_radius,
            0.0,
            validation_radius,
        ):
            rate = row_center + np.asarray((0.0, col_delta))
            key = tuple(rate)
            col_trials.append((cache[key][0], rate))
        _, polished_center = min(col_trials, key=lambda item: item[0])

        # One extra two-stage candidate protects periodic images where a
        # translation-only solve before the rate correction establishes
        # the physically correct correlation basin.
        zero_bridge = np.zeros(2, dtype=np.float64)
        zero_result = _delivered_candidate(
            self,
            polished_center,
            starting_knots,
            fixed_set,
            native_shift,
            upsample_factor,
            zero_bridge,
        )
        cache[("zero_bridge",)] = zero_result
        best_key, best_result = min(cache.items(), key=lambda item: item[1][0])
        if best_key == ("zero_bridge",):
            center = polished_center
            bridge_center = zero_bridge
        else:
            center = np.asarray(best_key, dtype=np.float64)
            bridge_center = center.copy()
        if best_result[1] is None:
            self.knots = [knot.clone() for knot in starting_knots]
            _apply_affine_rate(self, center, fixed_set)
            for image_index in range(self.shape[0]):
                self.knots[image_index][0] += best_result[3][image_index, 0]
                self.knots[image_index][1] += best_result[3][image_index, 1]
        else:
            self.knots = [knot.clone() for knot in best_result[1]]
        warped = best_result[2]
        residual_limit = min(float(native_shift or 64.0), 64.0)
        _, residual, residual_gain = fixed_overlap_ncc(
            warped[:1],
            warped[1:],
            image_shape,
            residual_limit,
        )
        if residual_gain[0] >= 0.01:
            residual = residual[0]
            translation_verification_shift = residual.cpu().numpy()
            translation_verification_gain = float(residual_gain[0].cpu())
            pair_shifts = torch.stack((-residual / 2, residual / 2))
            for image_index in range(2):
                self.knots[image_index][0] += pair_shifts[image_index, 0]
                self.knots[image_index][1] += pair_shifts[image_index, 1]
            warped = warp_and_translate(
                self,
                native_shift,
                solve_translation=False,
            )
        self.imgs_warped.array[:] = warped.cpu().numpy()
        validation_costs = sorted(result[0] for result in cache.values())
        validation_evaluations = len(cache)
        evaluations += validation_evaluations
        history.append(
            {
                "stage": "delivered_objective",
                "radius": float(validation_radius),
                "best_rate": center.tolist(),
                "translation_bridge": bridge_center.tolist(),
                "cost": float(best_result[0]),
                "evaluations": validation_evaluations,
            }
        )
    else:
        if bridge_center is None:
            bridge_center = center.copy()
        _apply_affine_rate(self, bridge_center, fixed_set)
        warped = warp_and_translate(
            self,
            native_shift,
            upsample_factor,
            fixed_indices=fixed_set,
        )
        final_delta = center - bridge_center
        if np.any(final_delta != 0):
            _apply_affine_rate(self, final_delta, fixed_set)
            warped = warp_and_translate(
                self,
                native_shift,
                upsample_factor,
                fixed_indices=fixed_set,
            )
    report.record_error(self, 1, warped)

    fallback_reason = None
    suspicious_reference_basin = (
        reference_search
        and np.linalg.norm(center) > 0.25
        and (
            self.error_track[-1, 1] > initial_error_row[1] * 0.8
            or best_cost < float(history[0]["cost"]) * 0.75
        )
    )
    if reference_search and (
        self.error_track[-1, 1] > initial_error_row[1] or suspicious_reference_basin
    ):
        # Backward-warp screening can still become border-dominated for
        # extreme rates.  When the delivered result is weak, worse than
        # the input, or implausibly better in a wide-rate basin, switch
        # to a slower multi-start pattern search that scores the actual
        # post-translation output. Ordinary reference workflows
        # (including XEDS) do not pay this fallback cost.
        fallback_reason = (
            "screened solution entered a suspicious wide-rate basin"
            if suspicious_reference_basin
            else "screened solution worsened delivered error"
        )
        attempted_center = center.copy()
        best_cost = float(initial_error_row[1])
        best_rate = np.zeros(2, dtype=np.float64)
        best_knots = [knot.clone() for knot in starting_knots]
        best_warped = None
        fallback_costs = [best_cost]
        fallback_evaluations = 0

        # Rank the winners from every broad-search radius with the
        # delivered post-translation objective.  A border-dominated
        # backward-warp minimum at the widest radius must not erase a
        # physically correct seed found one level earlier.
        seed_records = [
            (np.zeros(2, dtype=np.float64), 4.0 / image_shape[0]),
            (attempted_center, coarse_step * 1.5),
        ]
        seed_records.extend(
            (
                np.asarray(item["best_rate"], dtype=np.float64),
                1.5 * (2.0 * float(item["radius"]) / 8.0),
            )
            for item in history
            if item["stage"] == "pyramid"
        )
        unique_seeds = {}
        for seed_rate, seed_radius in seed_records:
            unique_seeds.setdefault(
                tuple(seed_rate),
                (seed_rate, seed_radius),
            )
        seed_results = []
        for seed_rate, seed_radius in unique_seeds.values():
            result = _delivered_candidate(
                self,
                seed_rate,
                starting_knots,
                fixed_set,
                native_shift,
                upsample_factor,
            )
            seed_results.append(
                (
                    result[0],
                    seed_rate.copy(),
                    seed_radius,
                    result[1],
                    result[2],
                )
            )
        fallback_evaluations += len(seed_results)
        fallback_costs.extend(result[0] for result in seed_results)
        seed_best = min(seed_results, key=lambda result: result[0])
        if seed_best[0] < best_cost:
            (
                best_cost,
                best_rate,
                _,
                best_knots,
                best_warped,
            ) = seed_best

        # Refine the three best delivered seeds independently.  The
        # single-point seed ranking can still favor a nearby alias before
        # either basin reaches its optimum; multi-start keeps the robust
        # basin without returning to a dense global native grid.
        selected_seeds = sorted(
            seed_results,
            key=lambda result: result[0],
        )[:3]
        for required_rate in (attempted_center, best_center):
            required = next(
                result for result in seed_results if np.array_equal(result[1], required_rate)
            )
            if not any(np.array_equal(result[1], required[1]) for result in selected_seeds):
                selected_seeds.append(required)
        for seed_result in selected_seeds:
            fallback_center = seed_result[1].copy()
            fallback_radius = max(
                float(seed_result[2]),
                4.0 / image_shape[0],
            )
            while True:
                candidates = candidate_grid(
                    fallback_center,
                    fallback_radius,
                    3,
                    circular=False,
                )
                local_results = []
                for rate in candidates:
                    result = _delivered_candidate(
                        self,
                        rate,
                        starting_knots,
                        fixed_set,
                        native_shift,
                        upsample_factor,
                    )
                    local_results.append(
                        (result[0], rate.copy(), result[1], result[2])
                    )
                fallback_evaluations += len(candidates)
                local_best = min(local_results, key=lambda result: result[0])
                fallback_costs.extend(result[0] for result in local_results)
                fallback_center = local_best[1].copy()
                if local_best[0] < best_cost:
                    (
                        best_cost,
                        best_rate,
                        best_knots,
                        best_warped,
                    ) = local_best
                if fallback_radius * image_shape[0] <= 0.5:
                    break
                fallback_radius *= 0.5

        evaluations += fallback_evaluations
        validation_evaluations += fallback_evaluations
        validation_costs = sorted(fallback_costs)
        center = best_rate.copy()
        bridge_center = center.copy()
        self.knots = [knot.clone() for knot in best_knots]
        if best_warped is None:
            warped = warp_and_translate(
                self,
                native_shift,
                upsample_factor,
                solve_translation=False,
                fixed_indices=fixed_set,
            )
            final_error_row = initial_error_row.copy()
            final_error_row[0] = 1.0
        else:
            warped = best_warped
            self.imgs_warped.array[:] = warped.cpu().numpy()
            per_image = (
                _scan_disagreement(warped, 0).mean(dim=(1, 2))
                .cpu()
                .numpy()
            )
            final_error_row = np.hstack((1.0, np.mean(per_image), per_image))
        self.error_track[-1] = final_error_row
        history.append(
            {
                "stage": "delivered_reference_fallback",
                "attempted_rate": attempted_center.tolist(),
                "best_rate": center.tolist(),
                "cost": float(self.error_track[-1, 1]),
                "evaluations": fallback_evaluations,
            }
        )

    if self.error_track[-1, 1] > initial_error_row[1]:
        # Automatic correction may find no improvement, but it must never
        # return a worse alignment than the untouched input.
        fallback_reason = fallback_reason or "delivered error worsened"
        self.knots = [knot.clone() for knot in starting_knots]
        warped = warp_and_translate(
            self,
            native_shift,
            upsample_factor,
            solve_translation=False,
            fixed_indices=fixed_set,
        )
        final_error_row = initial_error_row.copy()
        final_error_row[0] = 1.0
        self.error_track[-1] = final_error_row
        center = np.zeros(2, dtype=np.float64)
        bridge_center = center.copy()
        history.append(
            {
                "stage": "safe_noop_fallback",
                "best_rate": center.tolist(),
                "cost": float(initial_error_row[1]),
            }
        )

    if validation_costs is not None and len(validation_costs) > 1:
        self.affine_confidence_margin = (
            (validation_costs[1] - validation_costs[0]) / (validation_costs[0] + 1e-12) * 100
        )
    elif final_costs is not None and len(final_costs) > 1:
        ranked = torch.sort(final_costs).values
        self.affine_confidence_margin = float(
            ((ranked[1] - ranked[0]) / (ranked[0] + 1e-12) * 100).cpu()
        )
    else:
        self.affine_confidence_margin = 0.0
    elapsed = time.perf_counter() - start_time
    self.affine_search_info = {
        "strategy": "automatic_pyramid",
        "downsample_factor": factor,
        "refine_downsample_factor": refine_factor,
        "candidate_evaluations": evaluations,
        "broad_candidate_evaluations": len(coarse_cost_cache),
        "broad_candidate_reuses": (broad_candidate_requests - len(coarse_cost_cache)),
        "refinement_candidate_evaluations": refinement_evaluations,
        "native_candidate_evaluations": native_evaluations,
        "delivered_objective_evaluations": validation_evaluations,
        "delivered_objective_batched": "batch" in validation_status,
        "delivered_objective_tie_fallback": "tie" in validation_status,
        "delivered_objective_memory_fallback": "memory" in validation_status,
        "drift_rate_row_col": center.tolist(),
        "translation_bridge_row_col": bridge_center.tolist(),
        "translation_verification_shift_row_col": (
            translation_verification_shift.tolist()
        ),
        "translation_verification_ncc_gain": translation_verification_gain,
        "max_image_shift": native_shift,
        "fallback_reason": fallback_reason,
        "seconds": elapsed,
        "preprocess_seconds": self._implicit_preprocess_seconds,
        "total_seconds": elapsed + self._implicit_preprocess_seconds,
        "preprocess": dict(self.preprocess_info),
        "history": history,
    }
    if verbose:
        print(
            "correct_affine: automatic pyramid "
            f"{factor}x, {native_evaluations} native candidates, "
            f"{elapsed:.2f} s"
        )
        print(
            "Drift: "
            f"({center[0]:+.5f}, {center[1]:+.5f}) px/line; "
            f"translation bound {native_shift} px"
        )

    # Fallback candidates are created inside this inference-mode method.
    # Materialize ordinary tensors before handing knots to subsequent
    # affine/strip/nonrigid calls, which legitimately update them in
    # place outside inference mode.
    with torch.inference_mode(False):
        self.knots = [knot.detach().clone() for knot in self.knots]
    drift_plot.show_after_step(
        self,
        "affine",
        show_combined=show_combined,
        show_scans=show_scans,
        show_knots=show_knots,
    )
    if show_knot_plot:
        self.plot_knots()
    with torch.inference_mode(False):
        self._knots_after_affine = [knot.detach().clone() for knot in self.knots]
    self._images_warped_stale = True
    if show_report:
        print(self.report().to_string())
    return self


@torch.inference_mode()
def grid_search_batch(
    self,
    drift_vectors,
    upsample_factor,
    max_image_shift,
    chunk_size=None,
    fixed_indices=None,
    progress_desc=None,
    fixed_overlap_check=False,
):
    """Evaluate all candidate drift vectors in parallel.

    Warps both images for each candidate using ``bilinear_kde_batch``
    and scores alignment quality via ``cross_corr_batch``. Without
    batching, each candidate would be a separate Python iteration - this
    is the key operation that enables the 300x speedup.

    When ``fixed_indices`` is provided, images at those indices are
    warped once with their current knots (no candidate drift) and
    reused across all chunks. Only non-fixed images receive the
    candidate drift offsets.

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
    fixed_indices : frozenset[int] or None
        Indices of images whose knots should not receive candidate
        drift. These images are warped once and reused.
    progress_desc : str or None
        Description for a progress bar shown only when candidate
        evaluation requires multiple chunks. ``None`` disables it.
    fixed_overlap_check : bool
        Guard the automatic search's broad basin against a translation peak
        selected mostly by padding. Refinement keeps the established cost.

    Returns
    -------
    tuple[int, torch.Tensor]
        Index of the best candidate in ``drift_vectors``, and the full
        cost tensor of shape ``(N,)`` for all candidates (used by
        verbose mode to rank runner-ups).
    """
    device = self._device
    dtype = self._dtype
    fixed_set = fixed_indices if fixed_indices else frozenset()
    num_candidates = drift_vectors.shape[0]
    drift_vectors_t = torch.tensor(drift_vectors, dtype=dtype, device=device)

    # When fixed_indices is set, use backward-warp scoring to avoid
    # KDE forward-scatter bias.  The periodic wrapping in
    # bilinear_kde_batch creates geometry-dependent seam artifacts
    # that differ between the fixed reference and drift-shifted
    # moving image, making the MAE minimum diverge from the true
    # drift.  Backward-warp scoring works at original resolution
    # with grid_sample (no canvas, no KDE).
    if fixed_set:
        fixed_idx = sorted(fixed_set)[0]
        moving_indices = [i for i in range(len(self.imgs_t)) if i not in fixed_set]
        if not moving_indices:
            raise ValueError("All images are fixed - nothing to optimize.")
        total_costs = None
        for mov_idx in moving_indices:
            desc = progress_desc
            if desc is not None and len(moving_indices) > 1:
                desc = f"{desc} (scan {mov_idx})"
            _, costs = backward_warp_grid_search(
                self.imgs_t[fixed_idx],
                self.imgs_t[mov_idx],
                drift_vectors_t,
                upsample_factor,
                max_image_shift,
                chunk_size,
                progress_desc=desc,
            )
            total_costs = costs if total_costs is None else total_costs + costs
        return torch.argmin(total_costs).item(), total_costs

    canvas_shape = (self.shape[1], self.shape[2])
    n_images = len(self.imgs_t)
    # Base coordinates shared across all candidates
    base_data = []
    for img_idx in range(n_images):
        row_base, col_base, scanline_offset = drift_knots.interpolator(
            self, img_idx
        ).affine_candidate_base()
        base_data.append((self.imgs_t[img_idx], row_base, col_base, scanline_offset))
    # Precompute shift mask and frequency grids (shared across chunks)
    shift_mask = None
    if max_image_shift is not None:
        canvas_rows, canvas_cols = canvas_shape
        freq_row = fftfreq(canvas_rows, 1.0 / canvas_rows, device=device, dtype=dtype)
        freq_col = fftfreq(canvas_cols, 1.0 / canvas_cols, device=device, dtype=dtype)
        shift_mask = freq_row[:, None] ** 2 + freq_col[None, :] ** 2 >= max_image_shift**2
    freq_grids = (
        fftfreq(canvas_shape[0], device=device, dtype=dtype)[:, None],
        fftfreq(canvas_shape[1], device=device, dtype=dtype)[None, :],
    )
    device_type = torch.device(device).type
    if chunk_size is None:
        chunk_size = automatic_chunk_size(num_candidates, canvas_shape, dtype, device)
    chunked = chunk_size < num_candidates
    all_costs = []
    all_overlap_costs = []
    chunk_start = 0
    chunk_idx = 0
    pbar = tqdm(
        total=num_candidates,
        desc=progress_desc,
        unit="candidate",
        disable=progress_desc is None or not chunked,
    )
    try:
        while chunk_start < num_candidates:
            chunk_end = min(chunk_start + chunk_size, num_candidates)
            drift_chunk = drift_vectors_t[chunk_start:chunk_end]
            if chunk_idx == 0 and chunked and device_type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            # Warp each image (fixed → expand once, moving → drift-shifted)
            warped_images = []
            for img_idx in range(n_images):
                image_t, row_base, col_base, scanline_offset = base_data[img_idx]
                row_candidates = (
                    row_base[None]
                    + drift_chunk[:, 0, None, None] * scanline_offset[None, :, None]
                )
                col_candidates = (
                    col_base[None]
                    + drift_chunk[:, 1, None, None] * scanline_offset[None, :, None]
                )
                warped, _ = drift_knots.bilinear_kde_batch(
                    row_candidates,
                    col_candidates,
                    image_t,
                    canvas_shape,
                    self.kde_sigma,
                    self.pad_value[img_idx],
                )
                warped_images.append(warped)
            # Score all unique pairs and sum costs
            chunk_cost = torch.zeros(chunk_end - chunk_start, dtype=dtype, device=device)
            overlap_cost = torch.zeros_like(chunk_cost)
            for i in range(n_images):
                for j in range(i + 1, n_images):
                    chunk_cost += cross_corr_batch(
                        warped_images[i],
                        warped_images[j],
                        upsample_factor,
                        max_shift_mask=shift_mask,
                        freq_grids=freq_grids,
                    )
                    if fixed_overlap_check:
                        pair_cost, _, _ = fixed_overlap_ncc(
                            warped_images[i],
                            warped_images[j],
                            tuple(int(value) for value in self.imgs[i].shape[:2]),
                            max_image_shift,
                        )
                        overlap_cost += pair_cost
            all_costs.append(chunk_cost)
            if fixed_overlap_check:
                all_overlap_costs.append(overlap_cost)
            # After chunk 0, replace the conservative static estimate with the
            # actual measured per-candidate cost and print one summary line so
            # the user can see how the chunking adapted to their GPU state.
            if chunk_idx == 0 and chunked and device_type in {"cuda", "mps"}:
                if device_type == "cuda":
                    per_candidate_actual = torch.cuda.max_memory_allocated(device) / chunk_size
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                    tuned_chunk_size = max(1, int(free_bytes * 0.5 / per_candidate_actual))
                    tuned_chunk_size = min(tuned_chunk_size, num_candidates)
                    if tuned_chunk_size > chunk_size:
                        chunk_size = tuned_chunk_size
                    memory_text = f"{free_bytes / 1e9:.0f}/{total_bytes / 1e9:.0f} GB free"
                else:
                    total_bytes = torch.mps.recommended_max_memory()
                    live_bytes = max(
                        torch.mps.current_allocated_memory(),
                        torch.mps.driver_allocated_memory(),
                    )
                    free_bytes = max(total_bytes - live_bytes, 0)
                    memory_text = (
                        f"{free_bytes / 1e9:.0f}/{total_bytes / 1e9:.0f} GB MPS headroom"
                    )
                num_chunks_final = (
                    1 + (num_candidates - chunk_end + chunk_size - 1) // chunk_size
                )
                if progress_desc is not None:
                    pbar.write(
                        f"  affine grid: {num_candidates} drift vectors × "
                        f"{canvas_shape[0]}×{canvas_shape[1]}, "
                        f"auto chunk {chunk_size}/chunk × "
                        f"{num_chunks_final} passes ({memory_text})"
                    )
            pbar.update(chunk_end - chunk_start)
            chunk_start = chunk_end
            chunk_idx += 1
    finally:
        pbar.close()
    all_costs = torch.cat(all_costs)
    if not fixed_overlap_check:
        return torch.argmin(all_costs).item(), all_costs
    overlap_costs = torch.cat(all_overlap_costs)
    legacy_index = torch.argmin(all_costs)
    overlap_index = torch.argmin(overlap_costs)
    overlap_gain = float(overlap_costs[legacy_index] - overlap_costs[overlap_index])
    if overlap_gain >= 0.20:
        return overlap_index.item(), overlap_costs
    return legacy_index.item(), all_costs


def automatic_chunk_size(num_candidates, canvas_shape, dtype, device):
    """Pick a candidate-batch size that fits in current free GPU memory.

    Empirical per-candidate peak (measured at 4096×4096): bilinear KDE
    scatter buffers, gaussian smoothing temporaries, then cross-correlation
    FFT pairs (complex64) - together about ``32 × canvas_pixels``
    ``× dtype_bytes`` at peak. CUDA uses a 0.4 safety factor and then
    retunes after the first chunk from measured peak memory.

    MPS (Apple unified memory) needs a more conservative static factor:
    Metal shares memory with the OS and the bilinear KDE + smoothing
    kernels can become memory-pressure limited well before the reported
    recommended maximum.  Use the larger of current and driver allocation
    as live memory, then spend only a small fraction of the remaining
    headroom.  On CPU we process all candidates at once - no separate
    device pool to overflow.
    """
    device = torch.device(device)
    bytes_per_element = torch.finfo(dtype).bits // 8
    per_candidate_bytes = canvas_shape[0] * canvas_shape[1] * bytes_per_element * 32
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        safety_factor = 0.4
    elif device.type == "mps":
        # recommended_max is Metal's working-set ceiling; subtract the
        # larger live allocation estimate to get practical headroom.
        live_bytes = max(
            torch.mps.current_allocated_memory(),
            torch.mps.driver_allocated_memory(),
        )
        free_bytes = max(torch.mps.recommended_max_memory() - live_bytes, 0)
        safety_factor = 0.075
    else:
        return num_candidates
    chunk_size = max(1, int(free_bytes * safety_factor / per_candidate_bytes))
    return min(chunk_size, num_candidates)
