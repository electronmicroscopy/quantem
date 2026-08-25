"""Scientific diagnostics for affine and non-rigid drift correction."""

import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from tqdm import tqdm

import quantem.imaging.drift.apply as drift_apply
from quantem.imaging.drift.core import strip
from quantem.imaging.drift.core.warping import canvas_center_to_scan
from quantem.imaging.drift.plot import overlay_pair


def _diagnostic_grid(num_rows: int):
    """Keep square diagnostic panels close enough for direct comparison."""
    figure, axes = plt.subplots(
        num_rows,
        4,
        figsize=(13.1, 3.8 * num_rows),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.02,
        top=0.91,
        wspace=0.015,
        hspace=0.18,
    )
    return figure, axes


def _copy_for_diagnosis(self):
    """Isolate exploratory solves and rendering from the live correction."""
    memo = {}
    if getattr(self, "_datasets", None) is not None:
        # Diagnostics use the resident 2-D alignment images, not the potentially
        # multi-gigabyte EDS or 4D-STEM source payload.
        memo[id(self._datasets)] = None
    return copy.deepcopy(self, memo)


def diagnose_affine(
    self,
    *,
    stage: str | None = "affine",
    grid_shape: tuple[int, int] = (2, 2),
    smoothing_sigma: float | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """Show where one regional affine model leaves non-rigid disagreement

    Periodic atomic images can achieve a high global score even when defects
    do not correspond everywhere. This view divides the corrected field into
    regions so scientists can verify the trusted affine fit and see where
    spatially varying drift remains for non-rigid correction.

    The correction is never modified. Each row represents one physical image
    region. Four columns show corrected scan 0, corrected scan 1, their actual
    RGB registration, and the absolute scan difference. No additional shift
    is searched or applied.

    Parameters
    ----------
    stage : {"initial", "affine", "strip", "nonrigid", None}, optional
        Correction stage to diagnose. Default is ``"affine"``. ``None`` uses
        the current knots.
    grid_shape : tuple of int, optional
        Number of regional rows and columns. Default is ``(2, 2)``.
    smoothing_sigma : float or None, optional
        Display and measurement smoothing in pixels. ``None`` uses the current
        correction value. Use 0.5 to validate near-native atomic detail after
        searching with stronger smoothing.

    Returns
    -------
    matplotlib.figure.Figure
        Four-column regional comparison of both scans, their actual RGB
        registration, and their absolute difference.
    pandas.DataFrame
        Regional role and bounds, current NCC, yellow agreement, and mean
        absolute difference.

    Examples
    --------
    >>> drift.correct_affine(show_combined=False)
    >>> figure, regions = drift.diagnose_affine()
    >>> regions[["region", "current_ncc", "current_yellow"]]
    """
    grid_rows, grid_columns = (int(value) for value in grid_shape)
    if grid_rows < 1 or grid_columns < 1:
        raise ValueError(f"grid_shape must contain positive counts, got {grid_shape}")
    candidate = _copy_for_diagnosis(self)
    if smoothing_sigma is not None:
        candidate.kde_sigma = float(smoothing_sigma)
        candidate._images_warped_stale = True
    stack = drift_apply.warped_stack(candidate, stage)
    if stack.shape[0] != 2:
        raise ValueError(
            "diagnose_affine() currently compares one image pair; "
            f"this correction contains {stack.shape[0]} images"
        )
    reference, moving = [canvas_center_to_scan(candidate, image) for image in stack]

    # -----------------------------------------------------------------------
    # Divide the actual corrected field into physical regions
    # -----------------------------------------------------------------------
    tile_rows = reference.shape[0] // grid_rows
    tile_columns = reference.shape[1] // grid_columns
    reference_tiles = []
    moving_tiles = []
    regions = []
    search = getattr(self, "affine_search_info", {})
    trusted_bounds = search.get("trusted_region_bounds_row_column")
    vertical = ("top", "bottom") if grid_rows == 2 else ()
    horizontal = ("left", "right") if grid_columns == 2 else ()
    for row in range(grid_rows):
        for column in range(grid_columns):
            row_slice = slice(row * tile_rows, (row + 1) * tile_rows)
            column_slice = slice(
                column * tile_columns,
                (column + 1) * tile_columns,
            )
            reference_tiles.append(reference[row_slice, column_slice])
            moving_tiles.append(moving[row_slice, column_slice])
            name = (
                f"{vertical[row]} {horizontal[column]}"
                if vertical and horizontal
                else f"row {row}, column {column}"
            )
            trusted = (
                search.get("strategy") == "trusted_region"
                and trusted_bounds is not None
                and row_slice.start >= trusted_bounds[0]
                and row_slice.stop <= trusted_bounds[1]
                and column_slice.start >= trusted_bounds[2]
                and column_slice.stop <= trusted_bounds[3]
            )
            regions.append(
                (
                    name,
                    row,
                    column,
                    row_slice,
                    column_slice,
                    "trusted affine fit" if trusted else "validation region",
                )
            )

    # -----------------------------------------------------------------------
    # Plot only the delivered result and summarize the visible agreement
    # -----------------------------------------------------------------------
    figure, axes = _diagnostic_grid(len(regions))
    records = []
    stage_label = "current final" if stage is None else str(stage)
    directions = np.asarray(self.scan_direction_degrees, dtype=float)
    for index, (name, row, column, row_slice, column_slice, role) in enumerate(
        regions
    ):
        reference_tile = reference_tiles[index]
        moving_tile = moving_tiles[index]
        reference_display = reference_tile
        moving_display = moving_tile
        current_overlay = overlay_pair(reference_display, moving_display)

        red, green = current_overlay[..., 0], current_overlay[..., 1]
        current_yellow = float(
            np.minimum(red, green).sum()
            / max(float(np.maximum(red, green).sum()), 1e-12)
        )
        finite = np.isfinite(reference_display) & np.isfinite(moving_display)
        first = reference_display[finite].astype(np.float64)
        second = moving_display[finite].astype(np.float64)
        first -= first.mean()
        second -= second.mean()
        current_ncc = float(
            first @ second
            / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-12)
        )
        records.append(
            {
                "region": name,
                "region_role": role,
                "row_start": row_slice.start,
                "row_stop": row_slice.stop,
                "column_start": column_slice.start,
                "column_stop": column_slice.stop,
                "grid_row": row,
                "grid_column": column,
                "current_ncc": current_ncc,
                "current_yellow": current_yellow,
                "mean_absolute_difference": float(
                    np.abs(reference_display[finite] - moving_display[finite]).mean()
                ),
            }
        )
        contrast = np.concatenate(
            (reference_display.ravel()[::16], moving_display.ravel()[::16])
        )
        low, high = np.percentile(contrast[np.isfinite(contrast)], (1, 99))
        scale = max(float(high - low), np.finfo(np.float32).eps)
        difference = np.abs(
            np.clip((reference_display - low) / scale, 0, 1)
            - np.clip((moving_display - low) / scale, 0, 1)
        )
        axes[index, 0].imshow(
            reference_display,
            cmap="gray",
            vmin=low,
            vmax=high,
        )
        axes[index, 0].set_title(
            f"Scan 0 ({directions[0]:g}°)\nactual {stage_label} image"
        )
        axes[index, 1].imshow(
            moving_display,
            cmap="gray",
            vmin=low,
            vmax=high,
        )
        axes[index, 1].set_title(
            f"Scan 1 ({directions[1]:g}°)\nactual {stage_label} image"
        )
        axes[index, 2].imshow(current_overlay)
        axes[index, 2].set_title(
            f"Actual {stage_label} RGB\n"
            f"NCC {current_ncc:.3f}, yellow {current_yellow:.1%}"
        )
        axes[index, 3].imshow(difference, cmap="magma", vmin=0, vmax=1)
        axes[index, 3].set_title("Residual difference |scan 0 - scan 1|")
        axes[index, 0].set_ylabel(
            f"{name}\nrows {row_slice.start}:{row_slice.stop}\n"
            f"columns {column_slice.start}:{column_slice.stop}\n{role}",
            fontsize=11,
        )
        for axis in axes[index]:
            axis.set_xticks([])
            axis.set_yticks([])

    drift_rate = search.get(
        "drift_rate_row_col",
        getattr(self, "drift_rate", None),
    )
    rate_text = (
        "unknown"
        if drift_rate is None
        else f"[{float(drift_rate[0]):+.5f}, {float(drift_rate[1]):+.5f}]"
    )
    trusted_name = search.get("trusted_region", "whole image")
    trusted_text = (
        trusted_name.replace("_", " ")
        if trusted_bounds is None
        else f"{trusted_name.replace('_', ' ')} {tuple(trusted_bounds)}"
    )
    figure.suptitle(
        f"Affine model: one drift vector {rate_text} px/scanline applied to "
        f"the complete image\nfit region: {trusted_text}; "
        f"displayed stage: {stage_label}",
        fontsize=16,
    )
    return figure, pd.DataFrame.from_records(records)


def diagnose_nonrigid(
    self,
    *,
    num_knots: tuple[int, ...] = (1, 4, 6),
    verbose: bool = True,
    **nonrigid_options,
) -> tuple[Figure, pd.DataFrame]:
    """Compare non-rigid knot counts without changing the correction

    Multiple knots let the displacement vary along each fast-scan line. This
    diagnostic holds the affine or strip starting field and every optimizer
    setting fixed, then compares knot counts using one common measured mask.
    Use it when a single whole-field score cannot distinguish a physically
    smooth correction from a periodic-lattice hop.

    The returned table pairs registration metrics with displacement-field
    smoothness. A higher NCC alone is not sufficient evidence on an atomic
    lattice; inspect the RGB rows and prefer the smallest knot count that
    aligns distinctive features without introducing fast-direction roughness.
    The supplied correction is unchanged.

    Parameters
    ----------
    num_knots : tuple of int, optional
        Knot counts to compare along every fast-scan line. Default is
        ``(1, 4, 6)``.
    verbose : bool, optional
        Show one progress bar across knot-count candidates. Default is True.
    **nonrigid_options
        Options forwarded to :meth:`correct_nonrigid`, such as
        ``num_refine_cycles``, ``knot_smoothing_sigma``, ``loss``, and
        ``max_image_shift``. Diagnostic plotting is disabled inside each run.

    Returns
    -------
    matplotlib.figure.Figure
        One four-column row per knot count: both corrected scans, their RGB
        agreement, and their absolute difference.
    pandas.DataFrame
        Common-mask NCC, yellow agreement, coverage, runtime, and residual
        displacement smoothness for every knot count. ``fast_roughness_px``
        is the root-mean-square difference between neighboring knot
        displacements along each fast-scan line. It is zero for one knot,
        which has no neighbor, and does not measure image noise.

    Examples
    --------
    >>> drift.correct_affine(show_combined=False)
    >>> figure, metrics = drift.diagnose_nonrigid(
    ...     num_knots=(1, 4, 6),
    ...     num_refine_cycles=128,
    ...     knot_smoothing_sigma=8,
    ... )
    >>> metrics[["num_knots", "common_ncc", "fast_roughness_px"]]
    """
    if not hasattr(self, "_knots_after_affine"):
        raise RuntimeError(
            "diagnose_nonrigid() requires an affine correction. "
            "Run correct_affine() first."
        )
    counts = tuple(dict.fromkeys(int(value) for value in num_knots))
    if not counts or min(counts) < 1:
        raise ValueError(
            f"num_knots must contain positive integers, got {num_knots!r}."
        )

    solve_options = {
        **nonrigid_options,
        "show_combined": False,
        "show_scans": False,
        "show_knots": False,
        "show_knot_plot": False,
        "show_report": False,
        "verbose": False,
    }
    candidates = []
    progress = tqdm(
        counts,
        desc="Diagnosing non-rigid knot counts",
        unit="candidate",
        disable=not verbose or len(counts) == 1,
    )
    for count in progress:
        candidate = _copy_for_diagnosis(self)
        start = (
            candidate._knots_after_strip
            if hasattr(candidate, "_knots_after_strip")
            else candidate._knots_after_affine
        )
        candidate.knots = [value.clone() for value in start]
        candidate._images_warped_stale = True
        started = time.perf_counter()
        candidate.correct_nonrigid(num_knots=count, **solve_options)
        elapsed = time.perf_counter() - started

        stack = drift_apply.warped_stack(candidate, stage=None)
        reference, moving = [
            canvas_center_to_scan(candidate, image) for image in stack
        ]
        baseline = (
            candidate._knots_after_strip
            if hasattr(candidate, "_knots_after_strip")
            else candidate._knots_after_affine
        )
        residual = torch.stack(
            [
                current - baseline
                for current, baseline in zip(
                    candidate.knots,
                    baseline,
                    strict=True,
                )
            ]
        ).detach().cpu().numpy()
        candidates.append(
            {
                "num_knots": count,
                "reference": reference,
                "moving": moving,
                "mask": candidate.coverage_mask(),
                "residual": residual,
                "seconds": elapsed,
            }
        )

    common_mask = np.logical_and.reduce(
        [np.asarray(candidate["mask"], dtype=bool) for candidate in candidates]
    )
    figure, axes = _diagnostic_grid(len(candidates))
    records = []
    for row, candidate in enumerate(candidates):
        reference = candidate["reference"]
        moving = candidate["moving"]
        overlay = overlay_pair(reference, moving)
        scores = strip.region_ncc(
            reference,
            moving,
            common_mask,
            device=self._device,
        )
        red = overlay[..., 0][common_mask]
        green = overlay[..., 1][common_mask]
        yellow = float(
            np.minimum(red, green).sum()
            / max(float(np.maximum(red, green).sum()), 1e-12)
        )
        residual = candidate["residual"]
        fast_first = (
            float(np.sqrt(np.mean(np.diff(residual, axis=3) ** 2)))
            if residual.shape[3] > 1
            else 0.0
        )
        fast_second = (
            float(np.sqrt(np.mean(np.diff(residual, n=2, axis=3) ** 2)))
            if residual.shape[3] > 2
            else 0.0
        )
        records.append(
            {
                "num_knots": candidate["num_knots"],
                "common_ncc": scores["common"],
                "top_ncc": scores["top"],
                "middle_ncc": scores["middle"],
                "bottom_ncc": scores["bottom"],
                "yellow": yellow,
                "coverage": float(common_mask.mean()),
                "residual_rms_px": float(np.sqrt(np.mean(residual**2))),
                "residual_max_px": float(np.max(np.abs(residual))),
                "slow_roughness_px": float(
                    np.sqrt(np.mean(np.diff(residual, axis=2) ** 2))
                ),
                "fast_roughness_px": fast_first,
                "fast_curvature_px": fast_second,
                "seconds": candidate["seconds"],
            }
        )

        contrast = np.concatenate((reference.ravel()[::16], moving.ravel()[::16]))
        low, high = np.percentile(contrast[np.isfinite(contrast)], (1, 99))
        scale = max(float(high - low), np.finfo(np.float32).eps)
        difference = np.abs(
            np.clip((reference - low) / scale, 0, 1)
            - np.clip((moving - low) / scale, 0, 1)
        )
        axes[row, 0].imshow(reference, cmap="gray", vmin=low, vmax=high)
        axes[row, 1].imshow(moving, cmap="gray", vmin=low, vmax=high)
        axes[row, 2].imshow(overlay)
        axes[row, 3].imshow(difference, cmap="magma", vmin=0, vmax=1)
        count = candidate["num_knots"]
        noun = "knot" if count == 1 else "knots"
        axes[row, 0].set_ylabel(f"{count} {noun} per scanline", fontsize=11)
        axes[row, 0].set_title("Corrected scan 0")
        axes[row, 1].set_title("Corrected scan 1")
        axes[row, 2].set_title(
            f"RGB agreement\nNCC {scores['common']:.3f}, yellow {yellow:.1%}"
        )
        axes[row, 3].set_title("Residual difference |scan 0 - scan 1|")
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(
        "Non-rigid knot-count diagnosis\n"
        "one affine/strip starting field and one common measured mask",
        fontsize=16,
    )
    return figure, pd.DataFrame.from_records(records)
