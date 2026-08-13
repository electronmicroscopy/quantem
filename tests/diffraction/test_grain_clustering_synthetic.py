"""Tests for the signal-level synthetic grain-clustering dataset generator."""

from __future__ import annotations

import json
import os
import sys

import numpy as np


_DIFF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
    "quantem",
    "diffraction",
)
sys.path.insert(0, _DIFF_DIR)

from grain_clustering import cluster_signals_into_grains  # noqa: E402
from grain_clustering_synthetic import (  # noqa: E402
    CANONICAL_SCENARIOS,
    generate_canonical_sample,
    generate_canonical_suite,
    generate_dataset,
    generate_random_sample,
    load_sample,
    save_sample,
)


def _partition_ari(a, b):
    from math import comb

    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    shared = sum(comb(int(value), 2) for value in table.ravel() if value >= 2)
    aa = sum(comb(int(value), 2) for value in table.sum(1) if value >= 2)
    bb = sum(comb(int(value), 2) for value in table.sum(0) if value >= 2)
    total = comb(len(a), 2)
    expected = aa * bb / total if total else 0
    denom = 0.5 * (aa + bb) - expected
    return 1.0 if denom == 0 else (shared - expected) / denom


def test_canonical_suite_complete_and_valid():
    suite = generate_canonical_suite(map_shape=(32, 32), seed=10)
    assert [sample.scenario for sample in suite] == list(CANONICAL_SCENARIOS)
    for sample in suite:
        sample.validate()
        assert len(sample.signals) > 0
        assert len(sample.grain_ids) > 0


def test_overlap_ground_truth_is_multimembership():
    sample = generate_canonical_sample("triple_overlap", map_shape=(48, 48), seed=1)
    counts = sample.membership_count_map()
    assert counts.max() == 3
    assert np.count_nonzero(counts >= 2) > 0
    assert sample.primary_label_map().shape == (48, 48)
    for grain_id in sample.grain_ids:
        assert np.any(sample.grain_mask(grain_id))


def test_latent_dense_extent_is_separate_from_observed_dropout():
    sample = generate_canonical_sample("dense_interior_noisy_edges", map_shape=(64, 64), seed=3)
    latent = sample.membership_count_map() > 0
    observed = sample.observed_membership_count_map() > 0
    assert np.count_nonzero(latent) > np.count_nonzero(observed)
    distance = __import__("scipy.ndimage", fromlist=["distance_transform_edt"]).distance_transform_edt(latent)
    interior = distance >= 5
    edge = latent & (distance <= 2)
    assert observed[interior].mean() > observed[edge].mean()


def test_latent_dense_extent_can_contain_observation_holes():
    sample = generate_canonical_sample("dense_interior_with_holes", map_shape=(64, 64), seed=4)
    latent = sample.membership_count_map() > 0
    observed = sample.observed_membership_count_map() > 0
    assert np.count_nonzero(latent & ~observed) > 20


def test_sparse_percolation_records_fragmentation_metrics():
    sample = generate_canonical_sample("sparse_smooth_percolation", map_shape=(64, 64), seed=5)
    row = sample.grain_table()[0]
    assert row["observed_fraction"] < 0.25
    assert row["observed_components_8"] > 20
    assert row["largest_observed_component_fraction"] < 0.5


def test_radius2_bridge_has_two_latent_grains_and_noise_chain():
    sample = generate_canonical_sample("radius2_false_bridge", map_shape=(64, 64), seed=6)
    assert len(sample.grain_ids) == 2
    assert np.count_nonzero(sample.ground_truth < 0) == 3
    assert sample.membership_count_map().max() == 1


def test_random_generation_is_order_independent_and_deterministic():
    a = generate_random_sample(17, map_shape=(32, 32), seed=42)
    generate_random_sample(3, map_shape=(32, 32), seed=42)
    b = generate_random_sample(17, map_shape=(32, 32), seed=42)
    assert a.scenario == b.scenario and a.seed == b.seed
    for name in ("pos", "theta", "r", "intensity", "window"):
        assert np.array_equal(getattr(a.signals, name), getattr(b.signals, name))
    assert np.array_equal(a.ground_truth, b.ground_truth)


def test_random_strata_and_invariants():
    samples = [generate_random_sample(index, map_shape=(32, 32), seed=5) for index in range(96)]
    assert len({sample.metadata["morphology"] for sample in samples}) == 8
    assert len({sample.metadata["scale_regime"] for sample in samples}) == 3
    assert len({sample.metadata["overlap_regime"] for sample in samples}) == 4
    for sample in samples:
        sample.validate()


def test_save_load_round_trip(tmp_path):
    original = generate_canonical_sample("crossing_interpenetration", map_shape=(32, 32), seed=9)
    path = save_sample(original, tmp_path / "sample.npz")
    loaded = load_sample(path)
    assert loaded.scenario == original.scenario
    assert loaded.seed == original.seed
    assert loaded.metadata == original.metadata
    assert np.array_equal(loaded.ground_truth, original.ground_truth)
    assert np.array_equal(loaded.signals.pos, original.signals.pos)
    assert np.array_equal(loaded.signals.theta, original.signals.theta)
    assert np.array_equal(loaded.grain_mask_ids, original.grain_mask_ids)
    assert np.array_equal(loaded.grain_masks_truth, original.grain_masks_truth)
    assert np.array_equal(loaded.grain_windows, original.grain_windows)


def test_clean_canonical_stage_a_partitions():
    settings = {
        "uniform_large": (1, dict(theta_tol_deg=10.0)),
        "bicrystal": (2, dict(theta_tol_deg=10.0)),
        "disconnected_equal_orientation": (2, dict(theta_tol_deg=10.0, neighbor_dist=2)),
        "continuous_bend": (1, dict(theta_tol_deg=10.0)),
        "partial_overlap": (2, dict(theta_tol_deg=10.0)),
        "equal_orientation_distinct_radius": (2, dict(theta_tol_deg=10.0, r_tol_rel=0.10)),
    }
    for scenario, (expected, params) in settings.items():
        sample = generate_canonical_sample(scenario, map_shape=(24, 24), seed=2)
        result = cluster_signals_into_grains(sample.signals, area_min=1, **params)
        assert result.n_grains == expected, scenario
        assert _partition_ari(result.labels, sample.ground_truth) == 1.0, scenario


def test_generate_dataset_smoke(tmp_path):
    output = tmp_path / "dataset"
    manifest = generate_dataset(
        output, num_random=10, map_shape=(24, 24), seed=7, quicklooks="none"
    )
    assert manifest["num_canonical"] == len(CANONICAL_SCENARIOS)
    assert manifest["num_random"] == 10
    assert sum(manifest["splits"].values()) == len(CANONICAL_SCENARIOS) + 10
    on_disk = json.loads((output / "manifest.json").read_text())
    assert on_disk["splits"] == manifest["splits"]
    for row in manifest["samples"]:
        assert (output / row["path"]).exists()


def test_generate_dataset_resume_reuses_and_repairs_missing_sample(tmp_path):
    output = tmp_path / "dataset"
    first = generate_dataset(
        output, num_random=8, map_shape=(20, 20), seed=11, quicklooks="none"
    )
    missing_row = next(row for row in first["samples"] if row["split"] != "canonical")
    missing_path = output / missing_row["path"]
    original = load_sample(missing_path)
    missing_path.unlink()

    resumed = generate_dataset(
        output,
        num_random=8,
        map_shape=(20, 20),
        seed=11,
        quicklooks="none",
        resume=True,
    )
    repaired = load_sample(missing_path)
    assert resumed["last_run"] == {"reused": len(first["samples"]) - 1, "generated": 1}
    assert np.array_equal(repaired.signals.pos, original.signals.pos)
    assert np.array_equal(repaired.ground_truth, original.ground_truth)


def test_generate_dataset_resume_rejects_config_mismatch(tmp_path):
    output = tmp_path / "dataset"
    generate_dataset(output, num_random=2, map_shape=(16, 16), seed=3, quicklooks="none")
    try:
        generate_dataset(
            output,
            num_random=3,
            map_shape=(16, 16),
            seed=3,
            quicklooks="none",
            resume=True,
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("resume should reject a mismatched corpus configuration")
