"""Tests for synthetic grain-clustering benchmark metrics."""

from __future__ import annotations

import os
import sys

import numpy as np


_DIFF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "quantem", "diffraction",
)
sys.path.insert(0, _DIFF_DIR)

from grain_clustering import GrainResult, cluster_signals_into_grains  # noqa: E402
from grain_clustering_benchmark import (  # noqa: E402
    adjusted_rand_index,
    bcubed_precision_recall,
    categorical_grain_rgb,
    evaluate_clustering,
    variation_of_information,
)
from grain_clustering_synthetic import generate_canonical_sample  # noqa: E402


def test_adjusted_rand_index_is_label_permutation_invariant():
    assert adjusted_rand_index([0, 0, 1, 1], [7, 7, 3, 3]) == 1.0
    assert adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) < 0.0


def test_extended_partition_metrics_distinguish_splits_from_merges():
    truth = np.array([0, 0, 1, 1])
    perfect = np.array([7, 7, 3, 3])
    precision, recall = bcubed_precision_recall(truth, perfect)
    assert precision == recall == 1.0
    assert variation_of_information(truth, perfect) == (0.0, 0.0, 0.0)

    split = np.array([0, 1, 2, 2])
    _, split_recall = bcubed_precision_recall(truth, split)
    _, vi_split, vi_merge = variation_of_information(truth, split)
    assert split_recall < 1.0
    assert vi_split > 0.0 and abs(vi_merge) < 1e-12

    merged = np.zeros(4, dtype=int)
    merged_precision, _ = bcubed_precision_recall(truth, merged)
    _, vi_split, vi_merge = variation_of_information(truth, merged)
    assert merged_precision < 1.0
    assert abs(vi_split) < 1e-12 and vi_merge > 0.0


def test_categorical_colors_separate_touching_grains_and_black_background():
    labels = np.array([[-1, 1, 1], [2, 2, 3], [2, 3, 3]])
    rgb = categorical_grain_rgb(labels)
    assert np.array_equal(rgb[0, 0], np.zeros(3))
    assert not np.array_equal(rgb[0, 1], rgb[1, 0])
    assert not np.array_equal(rgb[1, 0], rgb[1, 2])


def test_clean_overlap_scores_perfect_signal_association():
    sample = generate_canonical_sample("partial_overlap", map_shape=(32, 32), seed=2)
    result = cluster_signals_into_grains(sample.signals, theta_tol_deg=10, area_min=1)
    sample_row, grain_rows = evaluate_clustering(sample, result, config_name="test")
    assert sample_row["signal_ari"] == 1.0
    assert sample_row["signal_f1"] == 1.0
    assert sample_row["bcubed_f1"] == 1.0
    assert sample_row["variation_of_information"] == 0.0
    assert sample_row["grain_count_abs_error"] == 0
    assert sample_row["overlap_membership_f1"] == 1.0
    assert sample_row["overlap_exact_probe_rate"] == 1.0
    assert all(row["signal_f1"] == 1.0 for row in grain_rows)
    assert all(row["footprint_recall"] == 1.0 for row in grain_rows)


def test_sparse_observations_expose_footprint_limitation():
    sample = generate_canonical_sample("sparse_smooth_percolation", map_shape=(64, 64), seed=5)
    # An oracle signal partition still cannot reconstruct pixels that were not observed.
    labels = sample.ground_truth.copy()
    result = GrainResult(labels, 1, [], np.zeros((1, 64, 64), dtype=np.int64))
    sample_row, grain_rows = evaluate_clustering(sample, result, config_name="oracle")
    assert sample_row["signal_ari"] == 1.0
    assert grain_rows[0]["signal_f1"] == 1.0
    assert grain_rows[0]["footprint_recall"] < 0.25
    assert grain_rows[0]["predicted_components_8"] > 20
    assert grain_rows[0]["largest_predicted_component_fraction"] < 0.5


def test_outliers_are_reported_separately_from_true_signals():
    sample = generate_canonical_sample("false_positive_outliers", map_shape=(32, 32), seed=7)
    labels = sample.ground_truth.copy()
    labels[labels >= 0] = 0
    result = GrainResult(labels, 1, [], np.zeros((1, 32, 32), dtype=np.int64))
    sample_row, _ = evaluate_clustering(sample, result, config_name="oracle")
    assert sample_row["noise_rejection_rate"] == 1.0
    assert sample_row["noise_false_assignment_rate"] == 0.0
    assert sample_row["true_signal_assignment_rate"] == 1.0
