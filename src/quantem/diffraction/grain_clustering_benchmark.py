"""Metrics for benchmarking signal-level grain clustering on synthetic truth.

The benchmark deliberately reports two different questions.  Signal-association
metrics score labels only where a diffraction signal was observed.  Footprint metrics
compare those assigned signal positions with the dense latent grain masks; Stage A is
not expected to fill unobserved pixels, so a low footprint recall is diagnostic rather
than hidden inside the association score.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.ndimage import distance_transform_edt, label as connected_components
from scipy.optimize import linear_sum_assignment

__all__ = [
    "adjusted_rand_index", "bcubed_precision_recall", "variation_of_information",
    "categorical_grain_rgb", "evaluate_clustering",
]


def categorical_grain_rgb(label_map):
    """Render grain ids as unordered colors, with distinct colors for 8-neighbors.

    Numeric grain ids never control brightness or hue ordering.  A small graph-coloring
    pass makes labels that touch at an edge or corner use different high-contrast
    palette entries.  Unassigned pixels (negative labels) are black.
    """
    labels = np.asarray(label_map, dtype=np.int64)
    if labels.ndim != 2:
        raise ValueError("label_map must be two-dimensional")
    palette = np.asarray(
        [
            (0.894, 0.102, 0.110), (0.216, 0.494, 0.722),
            (0.302, 0.686, 0.290), (0.596, 0.306, 0.639),
            (1.000, 0.498, 0.000), (1.000, 1.000, 0.200),
            (0.651, 0.337, 0.157), (0.969, 0.506, 0.749),
            (0.090, 0.745, 0.812), (0.600, 0.600, 0.600),
            (0.000, 0.450, 0.300), (0.350, 0.200, 0.050),
        ]
    )
    grain_ids = [int(value) for value in np.unique(labels[labels >= 0])]
    neighbors = {grain_id: set() for grain_id in grain_ids}
    for dx, dy in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a_rows = slice(max(0, dx), labels.shape[0] + min(0, dx))
        a_cols = slice(max(0, dy), labels.shape[1] + min(0, dy))
        b_rows = slice(max(0, -dx), labels.shape[0] + min(0, -dx))
        b_cols = slice(max(0, -dy), labels.shape[1] + min(0, -dy))
        left = labels[a_rows, a_cols].ravel()
        right = labels[b_rows, b_cols].ravel()
        boundary = (left >= 0) & (right >= 0) & (left != right)
        for first, second in zip(left[boundary], right[boundary]):
            neighbors[int(first)].add(int(second))
            neighbors[int(second)].add(int(first))
    color_index = {}
    for grain_id in sorted(grain_ids, key=lambda value: (-len(neighbors[value]), value)):
        forbidden = {color_index[value] for value in neighbors[grain_id] if value in color_index}
        preferred = (grain_id * 7) % len(palette)
        choices = [(preferred + offset) % len(palette) for offset in range(len(palette))]
        color_index[grain_id] = next((value for value in choices if value not in forbidden), preferred)
    rgb = np.zeros((*labels.shape, 3), dtype=float)
    for grain_id, index in color_index.items():
        rgb[labels == grain_id] = palette[index]
    return rgb


def _choose2(values):
    values = np.asarray(values, dtype=np.int64)
    return values * (values - 1) // 2


def adjusted_rand_index(truth, predicted) -> float:
    """Adjusted Rand index without requiring scikit-learn."""
    truth = np.asarray(truth).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    if truth.size != predicted.size:
        raise ValueError("truth and predicted must have equal length")
    if truth.size < 2:
        return 1.0
    _, ti = np.unique(truth, return_inverse=True)
    _, pi = np.unique(predicted, return_inverse=True)
    contingency = np.zeros((ti.max() + 1, pi.max() + 1), dtype=np.int64)
    np.add.at(contingency, (ti, pi), 1)
    pairs = float(_choose2(contingency).sum())
    truth_pairs = float(_choose2(contingency.sum(axis=1)).sum())
    pred_pairs = float(_choose2(contingency.sum(axis=0)).sum())
    total_pairs = float(_choose2(np.array([truth.size]))[0])
    expected = truth_pairs * pred_pairs / total_pairs if total_pairs else 0.0
    maximum = 0.5 * (truth_pairs + pred_pairs)
    denominator = maximum - expected
    return 1.0 if denominator == 0 else float((pairs - expected) / denominator)


def _contingency(truth, predicted):
    truth = np.asarray(truth).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    if truth.size != predicted.size:
        raise ValueError("truth and predicted must have equal length")
    if truth.size == 0:
        return np.zeros((0, 0), dtype=np.int64)
    _, ti = np.unique(truth, return_inverse=True)
    _, pi = np.unique(predicted, return_inverse=True)
    table = np.zeros((ti.max() + 1, pi.max() + 1), dtype=np.int64)
    np.add.at(table, (ti, pi), 1)
    return table


def bcubed_precision_recall(truth, predicted):
    """Return element-weighted B-cubed precision and recall."""
    table = _contingency(truth, predicted)
    n = int(table.sum())
    if n == 0:
        return 1.0, 1.0
    pred_size = table.sum(axis=0)
    truth_size = table.sum(axis=1)
    precision = np.sum((table.astype(float) ** 2) / np.where(pred_size > 0, pred_size, 1)[None, :]) / n
    recall = np.sum((table.astype(float) ** 2) / np.where(truth_size > 0, truth_size, 1)[:, None]) / n
    return float(precision), float(recall)


def variation_of_information(truth, predicted):
    """Return total VI and its split/merge conditional-entropy components (nats)."""
    table = _contingency(truth, predicted).astype(float)
    n = table.sum()
    if n == 0:
        return 0.0, 0.0, 0.0
    joint = table / n
    truth_prob = joint.sum(axis=1)
    pred_prob = joint.sum(axis=0)
    nz = joint > 0
    split = -float(np.sum(joint[nz] * np.log(
        joint[nz] / np.broadcast_to(truth_prob[:, None], joint.shape)[nz]
    )))  # H(predicted | truth): fragmentation
    merge = -float(np.sum(joint[nz] * np.log(
        joint[nz] / np.broadcast_to(pred_prob[None, :], joint.shape)[nz]
    )))  # H(truth | predicted): false merging
    return split + merge, split, merge


def _label_matching(truth, predicted):
    """Return one-to-one predicted-to-truth matching based on observed signals."""
    truth_ids = np.unique(truth[truth >= 0])
    pred_ids = np.unique(predicted[predicted >= 0])
    overlap = np.zeros((len(truth_ids), len(pred_ids)), dtype=np.int64)
    if overlap.size:
        truth_index = {int(value): i for i, value in enumerate(truth_ids)}
        pred_index = {int(value): i for i, value in enumerate(pred_ids)}
        for t, p in zip(truth, predicted):
            if t >= 0 and p >= 0:
                overlap[truth_index[int(t)], pred_index[int(p)]] += 1
        rows, cols = linear_sum_assignment(-overlap)
        matches = {
            int(pred_ids[col]): int(truth_ids[row])
            for row, col in zip(rows, cols)
            if overlap[row, col] > 0
        }
    else:
        matches = {}
    truth_to_pred = {truth_id: pred_id for pred_id, truth_id in matches.items()}
    return matches, truth_to_pred


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision, recall):
    return _safe_ratio(2.0 * precision * recall, precision + recall)


def _mask_shape_metrics(mask):
    n_pixels = int(mask.sum())
    labels, n_components = connected_components(mask, structure=np.ones((3, 3), int))
    sizes = np.bincount(labels.ravel())[1:]
    largest_fraction = _safe_ratio(int(sizes.max()), n_pixels) if sizes.size else 0.0
    positions = np.argwhere(mask)
    if positions.size:
        span = positions.max(axis=0) - positions.min(axis=0) + 1
        bbox_area = int(np.prod(span))
    else:
        bbox_area = 0
    return int(n_components), largest_fraction, _safe_ratio(n_pixels, bbox_area)


def evaluate_clustering(sample, result, *, config_name: str, runtime_seconds: float = 0.0):
    """Evaluate a :class:`GrainResult` against a ``SyntheticGrainSample``.

    Returns ``(sample_row, grain_rows)``.  Predicted grain ids are aligned to truth by
    a maximum-overlap one-to-one assignment before membership and per-grain scoring.
    """
    truth = np.asarray(sample.ground_truth, dtype=np.int64)
    predicted = np.asarray(result.labels, dtype=np.int64)
    if truth.shape != predicted.shape:
        raise ValueError("result labels do not align with the synthetic signals")

    pred_to_truth, truth_to_pred = _label_matching(truth, predicted)
    true_signal = truth >= 0
    noise_signal = ~true_signal
    mapped = np.full(predicted.shape, -2, dtype=np.int64)
    mapped[predicted < 0] = -1
    for pred_id, truth_id in pred_to_truth.items():
        mapped[predicted == pred_id] = truth_id

    n_true_signals = int(true_signal.sum())
    n_noise_signals = int(noise_signal.sum())
    correct_true = int(np.count_nonzero(true_signal & (mapped == truth)))
    assigned_true = int(np.count_nonzero(true_signal & (predicted >= 0)))
    rejected_noise = int(np.count_nonzero(noise_signal & (predicted < 0)))
    assigned_noise = int(np.count_nonzero(noise_signal & (predicted >= 0)))
    signal_precision = _safe_ratio(correct_true, int(np.count_nonzero(predicted >= 0)))
    signal_recall = _safe_ratio(correct_true, n_true_signals)

    # Multi-membership is scored per probe/window.  Sets preserve overlap without
    # flattening truth to a single primary raster label.
    width = int(sample.signals.map_shape[1])
    truth_sets = {}
    pred_sets = {}
    for i, (rx, ry) in enumerate(sample.signals.pos):
        key = (int(sample.signals.window[i]), int(rx) * width + int(ry))
        if truth[i] >= 0:
            truth_sets.setdefault(key, set()).add(int(truth[i]))
        if predicted[i] >= 0:
            pred_sets.setdefault(key, set()).add(int(mapped[i]))
    overlap_tp = overlap_fp = overlap_fn = overlap_probes = exact_overlap = 0
    for key, true_members in truth_sets.items():
        if len(true_members) < 2:
            continue
        overlap_probes += 1
        pred_members = pred_sets.get(key, set())
        overlap_tp += len(true_members & pred_members)
        overlap_fp += len(pred_members - true_members)
        overlap_fn += len(true_members - pred_members)
        exact_overlap += int(pred_members == true_members)
    overlap_precision = _safe_ratio(overlap_tp, overlap_tp + overlap_fp)
    overlap_recall = _safe_ratio(overlap_tp, overlap_tp + overlap_fn)

    true_grain_ids = [int(value) for value in sample.grain_ids]
    predicted_ids = [int(value) for value in np.unique(predicted[predicted >= 0])]
    truth_fragment_counts = [
        len(np.unique(predicted[truth == grain_id][predicted[truth == grain_id] >= 0]))
        for grain_id in true_grain_ids
    ]
    pred_truth_counts = [
        len(np.unique(truth[predicted == pred_id][truth[predicted == pred_id] >= 0]))
        for pred_id in predicted_ids
    ]
    split_count = sum(max(0, count - 1) for count in truth_fragment_counts)
    merge_count = sum(max(0, count - 1) for count in pred_truth_counts)
    bc_precision, bc_recall = bcubed_precision_recall(truth[true_signal], predicted[true_signal])
    vi, vi_split, vi_merge = variation_of_information(truth[true_signal], predicted[true_signal])

    grain_rows = []
    for grain_id in true_grain_ids:
        truth_signal_mask = truth == grain_id
        truth_extent = sample.grain_mask(grain_id)
        pred_id = truth_to_pred.get(grain_id)
        pred_signal_mask = predicted == pred_id if pred_id is not None else np.zeros_like(predicted, bool)
        pred_extent = np.zeros(sample.signals.map_shape, dtype=bool)
        if np.any(pred_signal_mask):
            pred_extent[tuple(sample.signals.pos[pred_signal_mask].T)] = True

        tp = int(np.count_nonzero(truth_signal_mask & pred_signal_mask))
        fp = int(np.count_nonzero(~truth_signal_mask & pred_signal_mask))
        fn = int(np.count_nonzero(truth_signal_mask & ~pred_signal_mask))
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        footprint_tp = int(np.count_nonzero(pred_extent & truth_extent))
        footprint_fp = int(np.count_nonzero(pred_extent & ~truth_extent))
        footprint_fn = int(np.count_nonzero(~pred_extent & truth_extent))
        footprint_precision = _safe_ratio(footprint_tp, footprint_tp + footprint_fp)
        footprint_recall = _safe_ratio(footprint_tp, footprint_tp + footprint_fn)
        footprint_iou = _safe_ratio(footprint_tp, footprint_tp + footprint_fp + footprint_fn)
        distance = distance_transform_edt(truth_extent)
        core = truth_extent & (distance > 2)
        edge = truth_extent & ~core
        components, largest_component_fraction, bbox_occupancy = _mask_shape_metrics(pred_extent)
        grain_rows.append(
            {
                "scenario": sample.scenario,
                "config": config_name,
                "truth_grain_id": grain_id,
                "predicted_grain_id": pred_id if pred_id is not None else -1,
                "window": int(sample.grain_windows[np.flatnonzero(sample.grain_mask_ids == grain_id)[0]]),
                "truth_area_pixels": int(truth_extent.sum()),
                "truth_observed_signals": int(truth_signal_mask.sum()),
                "truth_observed_coverage": _safe_ratio(
                    int(truth_signal_mask.sum()), int(truth_extent.sum())
                ),
                "grain_size_regime": (
                    "small" if int(truth_extent.sum()) < 100
                    else "medium" if int(truth_extent.sum()) < 500 else "large"
                ),
                "predicted_signals": int(pred_signal_mask.sum()),
                "signal_precision": precision,
                "signal_recall": recall,
                "signal_f1": _f1(precision, recall),
                "footprint_precision": footprint_precision,
                "footprint_recall": footprint_recall,
                "footprint_iou": footprint_iou,
                "core_recall": _safe_ratio(np.count_nonzero(pred_extent & core), np.count_nonzero(core)),
                "edge_recall": _safe_ratio(np.count_nonzero(pred_extent & edge), np.count_nonzero(edge)),
                "predicted_components_8": components,
                "largest_predicted_component_fraction": largest_component_fraction,
                "predicted_bbox_occupancy": bbox_occupancy,
            }
        )

    tags = dict(sample.metadata)
    for row in grain_rows:
        for key in ("morphology", "scale_regime", "overlap_regime",
                    "orientation_heterogeneity", "detection_regime"):
            if key in tags:
                row[key] = tags[key]
    sample_row = {
        "scenario": sample.scenario,
        "config": config_name,
        "runtime_seconds": float(runtime_seconds),
        "n_signals": int(len(truth)),
        "n_true_signals": n_true_signals,
        "n_noise_signals": n_noise_signals,
        "n_true_grains": len(true_grain_ids),
        "n_predicted_grains": len(predicted_ids),
        "grain_count_error": len(predicted_ids) - len(true_grain_ids),
        "grain_count_abs_error": abs(len(predicted_ids) - len(true_grain_ids)),
        "matched_grains": len(truth_to_pred),
        "split_excess": int(split_count),
        "merge_excess": int(merge_count),
        "split_grain_rate": _safe_ratio(sum(count > 1 for count in truth_fragment_counts), len(true_grain_ids)),
        "merged_prediction_rate": _safe_ratio(sum(count > 1 for count in pred_truth_counts), len(predicted_ids)),
        "signal_ari": adjusted_rand_index(truth[true_signal], predicted[true_signal]),
        "bcubed_precision": bc_precision,
        "bcubed_recall": bc_recall,
        "bcubed_f1": _f1(bc_precision, bc_recall),
        "variation_of_information": vi,
        "vi_split": vi_split,
        "vi_merge": vi_merge,
        "signal_accuracy": _safe_ratio(correct_true, n_true_signals),
        "signal_precision": signal_precision,
        "signal_recall": signal_recall,
        "signal_f1": _f1(signal_precision, signal_recall),
        "true_signal_assignment_rate": _safe_ratio(assigned_true, n_true_signals),
        "noise_rejection_rate": _safe_ratio(rejected_noise, n_noise_signals) if n_noise_signals else 1.0,
        "noise_false_assignment_rate": _safe_ratio(assigned_noise, n_noise_signals) if n_noise_signals else 0.0,
        "observed_overlap_probes": overlap_probes,
        "overlap_membership_precision": overlap_precision if overlap_probes else 1.0,
        "overlap_membership_recall": overlap_recall if overlap_probes else 1.0,
        "overlap_membership_f1": _f1(overlap_precision, overlap_recall) if overlap_probes else 1.0,
        "overlap_exact_probe_rate": _safe_ratio(exact_overlap, overlap_probes) if overlap_probes else 1.0,
    }
    for key in ("kind", "index", "morphology", "scale_regime", "overlap_regime",
                "orientation_heterogeneity", "detection_regime", "n_windows"):
        if key in tags:
            sample_row[key] = tags[key]
    return sample_row, grain_rows
