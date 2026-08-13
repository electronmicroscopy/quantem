#!/usr/bin/env python3
"""Benchmark Stage-A grain clustering against the synthetic corpus."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from quantem.diffraction.grain_clustering import (
    FragmentMergeConfig,
    GapBridgeConfig,
    cluster_signals_into_grains,
    gap_candidate_table,
    merge_grain_fragments,
)
from quantem.diffraction.grain_clustering_benchmark import categorical_grain_rgb, evaluate_clustering
from quantem.diffraction.grain_clustering_synthetic import load_sample


CONFIGS = {
    "default_n1": {
        "theta_tol_deg": 10.0,
        "r_tol_rel": 0.10,
        "neighbor_dist": 1,
        "area_min": 3,
        "enforce_one_per_probe": True,
    },
    "bridge_n2": {
        "theta_tol_deg": 10.0,
        "r_tol_rel": 0.10,
        "neighbor_dist": 2,
        "area_min": 3,
        "enforce_one_per_probe": True,
    },
    "experimental_v2": {
        "theta_tol_deg": 15.0,
        "r_tol_rel": 0.15,
        "neighbor_dist": 2,
        "area_min": 10,
        "enforce_one_per_probe": True,
    },
    "conditional_bridge": {
        "theta_tol_deg": 10.0, "r_tol_rel": 0.10, "neighbor_dist": 1,
        "area_min": 3, "enforce_one_per_probe": True,
        "gap_bridge": {},
    },
    "fragment_merge": {
        "theta_tol_deg": 10.0, "r_tol_rel": 0.10, "neighbor_dist": 1,
        "area_min": 3, "enforce_one_per_probe": True,
        "fragment_merge": {},
    },
    "bridge_and_merge": {
        "theta_tol_deg": 10.0, "r_tol_rel": 0.10, "neighbor_dist": 1,
        "area_min": 3, "enforce_one_per_probe": True,
        "gap_bridge": {}, "fragment_merge": {},
    },
}


def _run_configuration(signals, params):
    params = dict(params)
    bridge_params = params.pop("gap_bridge", None)
    merge_params = params.pop("fragment_merge", None)
    bridge = GapBridgeConfig(**bridge_params) if bridge_params is not None else None
    result = cluster_signals_into_grains(signals, gap_bridge_config=bridge, **params)
    if merge_params is not None:
        merge_bridge = bridge if bridge is not None else GapBridgeConfig()
        result = merge_grain_fragments(
            signals, result,
            FragmentMergeConfig(bridge=merge_bridge, **merge_params),
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", default=("canonical", "calibration", "validation"),
        choices=("canonical", "calibration", "validation", "test"),
    )
    parser.add_argument(
        "--configs", nargs="+",
        default=("default_n1", "conditional_bridge", "fragment_merge", "bridge_and_merge"),
        choices=tuple(CONFIGS),
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Deterministic limit per split")
    parser.add_argument("--worst-per-config", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _numeric_summary(rows, metrics):
    result = {"n": len(rows)}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue
        result[metric] = {
            "mean": float(finite.mean()),
            "median": float(np.median(finite)),
            "p10": float(np.percentile(finite, 10)),
            "p90": float(np.percentile(finite, 90)),
            "min": float(finite.min()),
            "max": float(finite.max()),
        }
    return result


def _summarize(sample_rows, grain_rows, configurations, selected_splits):
    sample_metrics = (
        "signal_ari", "signal_f1", "grain_count_abs_error", "split_excess", "merge_excess",
        "bcubed_precision", "bcubed_recall", "bcubed_f1", "variation_of_information",
        "vi_split", "vi_merge", "split_grain_rate", "merged_prediction_rate",
        "noise_rejection_rate", "overlap_membership_f1", "overlap_exact_probe_rate",
        "runtime_seconds",
    )
    grain_metrics = (
        "signal_f1", "footprint_precision", "footprint_recall", "footprint_iou",
        "core_recall", "edge_recall", "predicted_components_8",
        "largest_predicted_component_fraction", "predicted_bbox_occupancy",
    )
    summary = {
        "benchmark_version": 2,
        "splits": list(selected_splits),
        "configurations": configurations,
        "configs": {},
    }
    for config in configurations:
        sr = [row for row in sample_rows if row["config"] == config]
        gr = [row for row in grain_rows if row["config"] == config]
        entry = {
            "samples": _numeric_summary(sr, sample_metrics),
            "grains": _numeric_summary(gr, grain_metrics),
            "by_split": {},
            "by_scenario_factor": {},
        }
        for split in selected_splits:
            split_rows = [row for row in sr if row["split"] == split]
            if split_rows:
                entry["by_split"][split] = _numeric_summary(split_rows, sample_metrics)
        for factor in ("morphology", "scale_regime", "overlap_regime",
                       "orientation_heterogeneity", "detection_regime"):
            values = sorted({str(row[factor]) for row in sr if factor in row})
            if values:
                entry["by_scenario_factor"][factor] = {
                    value: _numeric_summary(
                        [row for row in sr if str(row.get(factor)) == value], sample_metrics
                    )
                    for value in values
                }
        summary["configs"][config] = entry
    return summary


def _make_summary_figure(sample_rows, grain_rows, output):
    import matplotlib.pyplot as plt

    configs = list(dict.fromkeys(row["config"] for row in sample_rows))
    sample_metrics = ["signal_ari", "overlap_membership_f1", "noise_rejection_rate"]
    grain_metrics = ["footprint_recall", "core_recall", "predicted_bbox_occupancy"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, metric in zip(axes[0], sample_metrics):
        def applicable(row):
            if metric == "overlap_membership_f1":
                return int(row["observed_overlap_probes"]) > 0
            if metric == "noise_rejection_rate":
                return int(row["n_noise_signals"]) > 0
            return True

        values = [
            [float(row[metric]) for row in sample_rows if row["config"] == config and applicable(row)]
            for config in configs
        ]
        ax.boxplot(values, tick_labels=configs, showfliers=False)
        suffix = " (applicable samples)" if metric != "signal_ari" else ""
        ax.set_title(metric.replace("_", " ") + suffix)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.2)
    for ax, metric in zip(axes[1], grain_metrics):
        values = [[float(row[metric]) for row in grain_rows if row["config"] == config]
                  for config in configs]
        ax.boxplot(values, tick_labels=configs, showfliers=False)
        ax.set_title(metric.replace("_", " "))
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.2)
    fig.suptitle("Synthetic grain-clustering benchmark: association vs dense footprint")
    fig.tight_layout()
    fig.savefig(output / "benchmark_summary.png", dpi=180)
    plt.close(fig)


def _primary_map(sample, labels):
    raster = np.full(sample.signals.map_shape, -1, dtype=np.int64)
    best = np.full(sample.signals.map_shape, -np.inf)
    for index, (rx, ry) in enumerate(sample.signals.pos):
        if labels[index] >= 0 and sample.signals.intensity[index] >= best[rx, ry]:
            raster[rx, ry] = labels[index]
            best[rx, ry] = sample.signals.intensity[index]
    return raster


def _save_worst_cases(selected, dataset, output, configurations, n_worst):
    import matplotlib.pyplot as plt

    case_dir = output / "worst_cases"
    case_dir.mkdir(exist_ok=True)
    for config_name, params in configurations.items():
        candidates = [row for row in selected if row["config"] == config_name]
        candidates.sort(key=lambda row: (float(row["signal_ari"]), -float(row["grain_count_abs_error"])))
        for rank, row in enumerate(candidates[:n_worst], start=1):
            sample = load_sample(dataset / row["sample_path"])
            result = _run_configuration(sample.signals, params)
            truth_primary = sample.primary_label_map()
            predicted_primary = _primary_map(sample, result.labels)
            panels = (
                (sample.membership_count_map(), "latent memberships", "viridis"),
                (sample.observed_membership_count_map(), "observed memberships", "viridis"),
                (categorical_grain_rgb(truth_primary), "truth primary grain (categorical)", None),
                (categorical_grain_rgb(predicted_primary), "Stage-A primary grain (categorical)", None),
            )
            fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
            for ax, (image, title, cmap) in zip(axes, panels):
                ax.imshow(image, cmap=cmap, origin="upper", interpolation="nearest")
                ax.set_title(title, fontsize=9)
                ax.set_axis_off()
            fig.suptitle(
                f"{config_name} worst #{rank}: {sample.scenario} | ARI={float(row['signal_ari']):.3f}, "
                f"count error={int(row['grain_count_error']):+d}", fontsize=10
            )
            fig.tight_layout()
            safe = sample.scenario.replace("/", "_")
            fig.savefig(case_dir / f"{config_name}_{rank:02d}_{safe}.png", dpi=170)
            plt.close(fig)


def main():
    args = parse_args()
    manifest_path = args.dataset / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(f"output is non-empty (pass --overwrite to replace result files): {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    with manifest_path.open() as stream:
        manifest = json.load(stream)

    entries = []
    for split in args.splits:
        split_entries = [entry for entry in manifest["samples"] if entry["split"] == split]
        if args.max_samples is not None:
            split_entries = split_entries[: args.max_samples]
        entries.extend(split_entries)
    configurations = {name: CONFIGS[name] for name in args.configs}
    print(f"selected {len(entries)} samples across {list(args.splits)}", flush=True)
    print(f"configurations: {list(configurations)}", flush=True)

    sample_rows = []
    grain_rows = []
    gap_rows = []
    total = len(entries) * len(configurations)
    completed = 0
    started = time.perf_counter()
    for entry in entries:
        sample = load_sample(args.dataset / entry["path"])
        if entry["split"] in ("canonical", "calibration"):
            for row in gap_candidate_table(sample.signals, GapBridgeConfig(), sample.ground_truth):
                row["split"] = entry["split"]
                row["sample_path"] = entry["path"]
                row["scenario"] = sample.scenario
                gap_rows.append(row)
        for config_name, params in configurations.items():
            tick = time.perf_counter()
            result = _run_configuration(sample.signals, params)
            runtime = time.perf_counter() - tick
            sample_row, rows = evaluate_clustering(
                sample, result, config_name=config_name, runtime_seconds=runtime
            )
            sample_row["split"] = entry["split"]
            sample_row["sample_path"] = entry["path"]
            for row in rows:
                row["split"] = entry["split"]
                row["sample_path"] = entry["path"]
            sample_rows.append(sample_row)
            grain_rows.extend(rows)
            completed += 1
            if completed % 10 == 0 or completed == total:
                elapsed = time.perf_counter() - started
                print(f"completed {completed}/{total} runs in {elapsed:.1f} s", flush=True)

    _write_csv(args.output / "per_sample.csv", sample_rows)
    _write_csv(args.output / "per_grain.csv", grain_rows)
    _write_csv(args.output / "gap_candidates.csv", gap_rows)
    summary = _summarize(sample_rows, grain_rows, configurations, args.splits)
    summary["dataset"] = str(args.dataset.resolve())
    summary["n_unique_samples"] = len(entries)
    summary["n_clustering_runs"] = len(sample_rows)
    summary["elapsed_seconds"] = float(time.perf_counter() - started)
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    _make_summary_figure(sample_rows, grain_rows, args.output)
    _save_worst_cases(sample_rows, args.dataset, args.output, configurations, args.worst_per_config)
    print(f"output: {args.output.resolve()}", flush=True)
    print(f"elapsed: {summary['elapsed_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
