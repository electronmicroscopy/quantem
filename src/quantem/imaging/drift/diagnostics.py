"""Pure numerical diagnostics for drift-correction results.

This module measures existing correction checkpoints. It does not run an
optimizer, update knots, or replace the current warped-image cache. Public API
decisions are intentionally deferred to the final API integration change.
"""

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

_COVERAGE_THRESHOLD = 1e-3
_MIN_CORRELATION_PIXELS = 64


def _copy_knots(knots: Sequence[NDArray | torch.Tensor]) -> tuple[NDArray, ...]:
    """Copy knot arrays to CPU without retaining mutable solver storage."""
    copied = []
    for knot in knots:
        if isinstance(knot, torch.Tensor):
            knot = knot.detach().cpu().numpy()
        copied.append(np.asarray(knot).copy())
    return tuple(copied)


def _record_stage(correction, stage: str) -> None:
    """Record one lightweight knot checkpoint after a completed stage."""
    if not hasattr(correction, "knots"):
        raise RuntimeError(
            "A diagnostic checkpoint requires prepared knots. Run preprocess() first."
        )
    if not hasattr(correction, "_diagnostic_knots"):
        correction._diagnostic_knots = {}
    stage = str(stage)
    recorded = list(correction._diagnostic_knots)
    if stage in recorded:
        for stale_stage in recorded[recorded.index(stage) :]:
            del correction._diagnostic_knots[stale_stage]
    correction._diagnostic_knots[stage] = _copy_knots(correction.knots)


def _available_stages(correction) -> tuple[str, ...]:
    """Return recorded correction stages in completion order."""
    snapshots = getattr(correction, "_diagnostic_knots", {})
    return tuple(snapshots)


def _select_stages(
    correction,
    stages: Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate a requested stage sequence against recorded checkpoints."""
    available = _available_stages(correction)
    if not available:
        raise RuntimeError(
            "Drift diagnostics require a prepared correction. Run preprocess() first."
        )
    if stages is None:
        return available
    requested = tuple(dict.fromkeys(str(stage) for stage in stages))
    missing = [stage for stage in requested if stage not in available]
    if missing:
        raise ValueError(
            f"Diagnostic stages are not available: {missing}. Available stages: {list(available)}."
        )
    return requested


def _stage_knots(correction, stage: str) -> tuple[NDArray, ...]:
    """Return isolated knot arrays for one recorded correction stage."""
    selected = _select_stages(correction, (stage,))
    return _copy_knots(correction._diagnostic_knots[selected[0]])


def _warp_stage(
    correction,
    stage: str,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Render a checkpoint without changing the live correction object."""
    knots = _stage_knots(correction, stage)
    images = []
    weights = []
    for image_index, stage_knots in enumerate(knots):
        warped, coverage = correction.interpolator[image_index].warp_image(
            np.asarray(correction.images[image_index].array),
            stage_knots,
        )
        images.append(np.asarray(warped, dtype=np.float32))
        weights.append(np.asarray(coverage, dtype=np.float32))
    return np.stack(images), np.stack(weights)


def _masked_ncc(
    reference: NDArray,
    moving: NDArray,
    mask: NDArray[np.bool_],
) -> float:
    """Calculate normalized cross-correlation on explicitly measured pixels."""
    if int(mask.sum()) < _MIN_CORRELATION_PIXELS:
        return float("nan")
    first = np.asarray(reference, dtype=np.float64)[mask]
    second = np.asarray(moving, dtype=np.float64)[mask]
    first -= first.mean()
    second -= second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= np.finfo(np.float64).eps:
        return float("nan")
    return float(first @ second / denominator)


def _masked_errors(
    reference: NDArray,
    moving: NDArray,
    mask: NDArray[np.bool_],
) -> tuple[float, float]:
    """Return mean absolute and root-mean-square disagreement."""
    if not np.any(mask):
        return float("nan"), float("nan")
    difference = (
        np.asarray(reference, dtype=np.float64)[mask] - np.asarray(moving, dtype=np.float64)[mask]
    )
    return float(np.mean(np.abs(difference))), float(np.sqrt(np.mean(difference**2)))


def _scan_comparisons(stack: NDArray) -> list[tuple[NDArray, NDArray]]:
    """Build pair or leave-one-out comparisons for a registered scan stack."""
    if stack.shape[0] < 2:
        raise ValueError(
            f"Registration diagnostics require at least two scans, got {stack.shape[0]}."
        )
    if stack.shape[0] == 2:
        return [(stack[0], stack[1])]
    return [
        (
            np.mean(np.delete(stack, image_index, axis=0), axis=0),
            stack[image_index],
        )
        for image_index in range(stack.shape[0])
    ]


def _pair_metrics(
    reference: NDArray,
    moving: NDArray,
    mask: NDArray[np.bool_],
) -> dict[str, float]:
    """Measure one registered pair over common and regional coverage."""
    num_rows = mask.shape[0]
    metrics = {"common_ncc": _masked_ncc(reference, moving, mask)}
    for name, (row_start, row_stop) in zip(
        ("top_ncc", "middle_ncc", "bottom_ncc"),
        (
            (0, num_rows // 3),
            (num_rows // 3, 2 * num_rows // 3),
            (2 * num_rows // 3, num_rows),
        ),
        strict=True,
    ):
        regional_mask = np.zeros_like(mask)
        regional_mask[row_start:row_stop] = mask[row_start:row_stop]
        metrics[name] = _masked_ncc(reference, moving, regional_mask)
    mean_absolute, root_mean_square = _masked_errors(reference, moving, mask)
    metrics["mean_absolute_difference"] = mean_absolute
    metrics["root_mean_square_difference"] = root_mean_square
    return metrics


def _finite_mean(values: Sequence[float]) -> float:
    """Average finite measurements without emitting all-NaN warnings."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def _registration_data(
    correction,
    stages: Sequence[str] | None = None,
) -> tuple[
    tuple[str, ...],
    dict[str, NDArray[np.float32]],
    NDArray[np.bool_],
    list[dict[str, float | str]],
]:
    """Build common-mask registration evidence for completed stages."""
    selected = _select_stages(correction, stages)
    stacks = {}
    coverage_by_stage = {}
    for stage in selected:
        stack, weights = _warp_stage(correction, stage)
        stacks[stage] = stack
        coverage_by_stage[stage] = np.all(weights >= _COVERAGE_THRESHOLD, axis=0)
    common_mask = np.logical_and.reduce([coverage_by_stage[stage] for stage in selected])

    rows: list[dict[str, float | str]] = []
    for stage in selected:
        pair_rows = [
            _pair_metrics(reference, moving, common_mask)
            for reference, moving in _scan_comparisons(stacks[stage])
        ]
        row: dict[str, float | str] = {"stage": stage}
        for key in pair_rows[0]:
            row[key] = _finite_mean([pair[key] for pair in pair_rows])
        row["coverage"] = float(common_mask.mean())
        rows.append(row)
    return selected, stacks, common_mask, rows


def _displacement_fields(
    correction,
    stage: str,
) -> NDArray[np.float64]:
    """Return row/column knot displacement from the initial checkpoint."""
    initial = _stage_knots(correction, "initial")
    selected = _stage_knots(correction, stage)
    if any(
        current.shape != baseline.shape
        for current, baseline in zip(selected, initial, strict=True)
    ):
        raise ValueError("Displacement diagnostics require matching knot topology across stages.")
    return np.stack(
        [
            np.asarray(current, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
            for current, baseline in zip(selected, initial, strict=True)
        ]
    )


def _displacement_rows(
    correction,
    stages: Sequence[str] | None = None,
) -> list[dict[str, float | int | str]]:
    """Summarize scan-line-origin displacement for each image and stage.

    Adjacent-change values are component-wise RMS measurements. Fast-knot
    differences are not normalized by knot separation, so they are comparable
    only when the fast-direction knot topology and spacing are unchanged.
    """
    selected = _select_stages(correction, stages)
    rows: list[dict[str, float | int | str]] = []
    for stage in selected:
        fields = _displacement_fields(correction, stage)
        for image_index, field in enumerate(fields):
            magnitude = np.linalg.norm(field, axis=0)
            endpoint_vector = np.mean(field[:, -1] - field[:, 0], axis=1)
            adjacent_line_change = np.diff(field, axis=1)
            adjacent_fast_knot_change = np.diff(field, axis=2)
            fast_knot_second_difference = np.diff(field, n=2, axis=2)
            rows.append(
                {
                    "stage": stage,
                    "image": image_index,
                    "endpoint_row_displacement_px": float(endpoint_vector[0]),
                    "endpoint_column_displacement_px": float(endpoint_vector[1]),
                    "endpoint_displacement_px": float(np.linalg.norm(endpoint_vector)),
                    "rms_displacement_px": float(np.sqrt(np.mean(magnitude**2))),
                    "max_displacement_px": float(np.max(magnitude)),
                    "component_rms_adjacent_line_change_px": (
                        float(np.sqrt(np.mean(adjacent_line_change**2)))
                        if adjacent_line_change.size
                        else 0.0
                    ),
                    "component_rms_adjacent_fast_knot_change_px": (
                        float(np.sqrt(np.mean(adjacent_fast_knot_change**2)))
                        if adjacent_fast_knot_change.size
                        else 0.0
                    ),
                    "component_rms_fast_knot_second_difference_px": (
                        float(np.sqrt(np.mean(fast_knot_second_difference**2)))
                        if fast_knot_second_difference.size
                        else 0.0
                    ),
                }
            )
    return rows
