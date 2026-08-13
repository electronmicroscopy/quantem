#!/usr/bin/env python3
"""Build the review notebook for the completed synthetic clustering benchmark."""

from pathlib import Path

import nbformat as nbf


OUTPUT = Path(
    "/global/u2/n/njmarch/DP_peak_detection/2026_05_model_V2/inspection/"
    "grain_clustering_synthetic_evaluation_milestone1_2026_07_17.ipynb"
)


def main():
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "quantem_dev", "language": "python", "name": "python3"
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Synthetic grain-clustering evaluation\n\n"
            "This notebook reviews Stage A before further experimental-data work. It keeps two "
            "questions separate: **signal association** (are detected signals grouped correctly?) and "
            "**latent footprint recovery** (are dense physical grain bodies represented?). Grain-ID "
            "maps use unordered, adjacency-aware categorical colors; black means unassigned.\n\n"
            "Dependencies are limited to the `quantem_dev` environment: the Python standard library, "
            "NumPy, Matplotlib, IPython, and QuantEM. Pandas is not required."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "from collections import Counter\n"
            "import csv\n"
            "import html\n"
            "import json\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import HTML, Image, display\n\n"
            "from quantem.diffraction.grain_clustering import (FragmentMergeConfig, GapBridgeConfig, cluster_signals_into_grains, merge_grain_fragments)\n"
            "from quantem.diffraction.grain_clustering_benchmark import categorical_grain_rgb, evaluate_clustering\n"
            "from quantem.diffraction.grain_clustering_synthetic import load_sample\n\n"
            "DATASET = Path('/pscratch/sd/n/njmarch/grain_clustering_synthetic_v1')\n"
            "RESULTS = Path('/pscratch/sd/n/njmarch/grain_clustering_m1_test_frozen_v2_20260717')\n"
            "VALIDATION = Path('/pscratch/sd/n/njmarch/grain_clustering_m1_validation_20260717')\n"
            "CANONICAL = Path('/pscratch/sd/n/njmarch/grain_clustering_m1_canonical_20260717_v2')\n"
            "CONFIGS = {\n"
            "    'default_n1': dict(theta_tol_deg=10., r_tol_rel=.10, neighbor_dist=1, area_min=3, enforce_one_per_probe=True),\n"
            "    'bridge_and_merge': dict(theta_tol_deg=10., r_tol_rel=.10, neighbor_dist=1, area_min=3, enforce_one_per_probe=True),\n"
            "}\n"
            "def run_config(signals, name):\n"
            "    result = cluster_signals_into_grains(signals, gap_bridge_config=GapBridgeConfig() if name == 'bridge_and_merge' else None, **CONFIGS[name])\n"
            "    return merge_grain_fragments(signals, result, FragmentMergeConfig()) if name == 'bridge_and_merge' else result\n"
            "assert all(path.exists() for path in (DATASET, RESULTS, VALIDATION, CANONICAL))"
        ),
        nbf.v4.new_code_cell(
            "def _parse_csv_value(value):\n"
            "    if value == '':\n"
            "        return None\n"
            "    if value in {'True', 'False'}:\n"
            "        return value == 'True'\n"
            "    try:\n"
            "        return int(value)\n"
            "    except ValueError:\n"
            "        try:\n"
            "            return float(value)\n"
            "        except ValueError:\n"
            "            return value\n\n"
            "def read_csv_rows(path):\n"
            "    with path.open(newline='') as stream:\n"
            "        return [{key: _parse_csv_value(value) for key, value in row.items()}\n"
            "                for row in csv.DictReader(stream)]\n\n"
            "def unique_values(rows, key):\n"
            "    return list(dict.fromkeys(row[key] for row in rows))\n\n"
            "def select_rows(rows, **criteria):\n"
            "    return [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]\n\n"
            "def mean_of(rows, key, predicate=None):\n"
            "    values = [row[key] for row in rows if row.get(key) is not None and (predicate is None or predicate(row))]\n"
            "    return float(np.mean(values)) if values else np.nan\n\n"
            "def _format_cell(value, spec=None):\n"
            "    if value is None:\n"
            "        return '—'\n"
            "    if spec is not None and isinstance(value, (int, float, np.number)):\n"
            "        return format(value, spec)\n"
            "    if isinstance(value, (float, np.floating)):\n"
            "        return f'{value:.4g}'\n"
            "    return str(value)\n\n"
            "def show_table(rows, columns=None, formats=None, max_rows=None):\n"
            "    rows = list(rows)\n"
            "    if columns is None:\n"
            "        columns = list(rows[0]) if rows else []\n"
            "    formats = formats or {}\n"
            "    shown = rows if max_rows is None else rows[:max_rows]\n"
            "    parts = ['<div style=\"overflow-x:auto\"><table style=\"border-collapse:collapse\">', '<thead><tr>']\n"
            "    parts.extend(f'<th style=\"padding:4px 8px;border:1px solid #bbb\">{html.escape(str(column))}</th>' for column in columns)\n"
            "    parts.append('</tr></thead><tbody>')\n"
            "    for row in shown:\n"
            "        parts.append('<tr>')\n"
            "        for column in columns:\n"
            "            value = _format_cell(row.get(column), formats.get(column))\n"
            "            parts.append(f'<td style=\"padding:4px 8px;border:1px solid #ddd\">{html.escape(value)}</td>')\n"
            "        parts.append('</tr>')\n"
            "    parts.append('</tbody></table></div>')\n"
            "    display(HTML(''.join(parts)))\n"
            "    if max_rows is not None and len(rows) > max_rows:\n"
            "        print(f'Showing {max_rows} of {len(rows)} rows.')\n\n"
            "def show_mapping(mapping):\n"
            "    show_table([{'metric': key, 'value': value} for key, value in mapping.items()], ['metric', 'value'])"
        ),
        nbf.v4.new_code_cell(
            "samples = read_csv_rows(RESULTS / 'per_sample.csv')\n"
            "grains = read_csv_rows(RESULTS / 'per_grain.csv')\n"
            "summary = json.loads((RESULTS / 'summary.json').read_text())\n"
            "print(f\"{summary['n_unique_samples']} samples, {summary['n_clustering_runs']} clustering runs, {summary['elapsed_seconds']:.1f} s\")\n"
            "display(Image(filename=str(RESULTS / 'benchmark_summary.png')))"
        ),
        nbf.v4.new_markdown_cell("## Held-out test summary"),
        nbf.v4.new_code_cell(
            "test = select_rows(samples, split='test')\n"
            "rows = []\n"
            "for config in unique_values(test, 'config'):\n"
            "    frame = select_rows(test, config=config)\n"
            "    ari = np.asarray([row['signal_ari'] for row in frame])\n"
            "    overlap = [row for row in frame if row['observed_overlap_probes'] > 0]\n"
            "    noise = [row for row in frame if row['n_noise_signals'] > 0]\n"
            "    rows.append({\n"
            "        'config': config,\n"
            "        'ARI median': float(np.median(ari)),\n"
            "        'ARI mean': float(np.mean(ari)),\n"
            "        'fraction ARI >= 0.9': float(np.mean(ari >= .9)),\n"
            "        'fraction ARI < 0.5': float(np.mean(ari < .5)),\n"
            "        'exact grain count': float(np.mean([row['grain_count_abs_error'] == 0 for row in frame])),\n"
            "        'grain-count MAE': mean_of(frame, 'grain_count_abs_error'),\n"
            "        'overlap F1 (applicable)': mean_of(overlap, 'overlap_membership_f1'),\n"
            "        'noise rejection (applicable)': mean_of(noise, 'noise_rejection_rate'),\n"
            "    })\n"
            "show_table(rows, formats={key: '.3f' for key in rows[0] if key != 'config'})"
        ),
        nbf.v4.new_markdown_cell("## Validation acceptance gates"),
        nbf.v4.new_code_cell(
            "validation = read_csv_rows(VALIDATION / 'per_sample.csv')\n"
            "gate_rows = []\n"
            "for config in unique_values(validation, 'config'):\n"
            "    frame = select_rows(validation, config=config)\n"
            "    overlap = [row for row in frame if row['observed_overlap_probes'] > 0]\n"
            "    noise = [row for row in frame if row['n_noise_signals'] > 0]\n"
            "    gate_rows.append({'config': config, 'mean ARI': mean_of(frame, 'signal_ari'), 'overlap F1': mean_of(overlap, 'overlap_membership_f1'), 'noise rejection': mean_of(noise, 'noise_rejection_rate'), 'low-ARI count': sum(row['signal_ari'] < .5 for row in frame), 'count MAE': mean_of(frame, 'grain_count_abs_error'), 'split excess': mean_of(frame, 'split_excess'), 'merge excess': mean_of(frame, 'merge_excess')})\n"
            "show_table(gate_rows, formats={key: '.4f' for key in gate_rows[0] if key not in {'config', 'low-ARI count'}})"
        ),
        nbf.v4.new_markdown_cell("## Truth-labelled canonical gap candidates"),
        nbf.v4.new_code_cell(
            "gaps = read_csv_rows(CANONICAL / 'gap_candidates.csv')\n"
            "counts = Counter((row['accepted'], row['same_true_grain']) for row in gaps)\n"
            "show_table([{'accepted': key[0], 'same_true_grain': key[1], 'count': count} for key, count in sorted(counts.items())])\n"
            "accepted = sorted((row for row in gaps if row['accepted']), key=lambda row: (row['same_true_grain'], row['left_support'], row['right_support']))\n"
            "show_table(accepted, max_rows=30)"
        ),
        nbf.v4.new_markdown_cell(
            "The median is near-perfect because many cases are easy, so readiness should not be "
            "judged from the median alone. The failure rate, exact grain-count rate, and scenario "
            "breakdowns below expose fragmentation and merging."
        ),
        nbf.v4.new_code_cell(
            "factors = ['morphology', 'scale_regime', 'overlap_regime', 'orientation_heterogeneity', 'detection_regime']\n"
            "configs = unique_values(test, 'config')\n"
            "for factor in factors:\n"
            "    levels = sorted(unique_values(test, factor), key=str)\n"
            "    matrix = np.full((len(levels), len(configs)), np.nan)\n"
            "    for i, level in enumerate(levels):\n"
            "        for j, config in enumerate(configs):\n"
            "            matrix[i, j] = mean_of([row for row in test if row[factor] == level and row['config'] == config], 'signal_ari')\n"
            "    fig, ax = plt.subplots(figsize=(max(5, 2.2 * len(configs)), max(2.5, .48 * len(levels))))\n"
            "    image = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')\n"
            "    ax.set_xticks(range(len(configs)), configs)\n"
            "    ax.set_yticks(range(len(levels)), levels)\n"
            "    for i in range(len(levels)):\n"
            "        for j in range(len(configs)):\n"
            "            ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', color='black' if .25 < matrix[i, j] < .85 else 'white')\n"
            "    ax.set_title(f'Mean signal ARI by {factor}')\n"
            "    fig.colorbar(image, ax=ax, label='mean signal ARI')\n"
            "    fig.tight_layout()\n"
            "    plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "The heatmaps above encode a score, not grain IDs, so a sequential scale is appropriate "
            "there. Grain maps below are strictly categorical."
        ),
        nbf.v4.new_markdown_cell("## Inspect one sample"),
        nbf.v4.new_code_cell(
            "# Change these and rerun this cell plus the next two.\n"
            "SCENARIO = test[0]['scenario']\n"
            "CONFIG_NAME = 'bridge_and_merge'\n\n"
            "record = next(row for row in samples if row['scenario'] == SCENARIO and row['config'] == CONFIG_NAME)\n"
            "sample = load_sample(DATASET / record['sample_path'])\n"
            "result = run_config(sample.signals, CONFIG_NAME)\n"
            "sample_metrics, grain_metrics = evaluate_clustering(sample, result, config_name=CONFIG_NAME)\n"
            "show_mapping(sample_metrics)"
        ),
        nbf.v4.new_code_cell(
            "def primary_map(sample, labels):\n"
            "    raster = np.full(sample.signals.map_shape, -1, dtype=np.int64)\n"
            "    best = np.full(sample.signals.map_shape, -np.inf)\n"
            "    for i, (rx, ry) in enumerate(sample.signals.pos):\n"
            "        if labels[i] >= 0 and sample.signals.intensity[i] >= best[rx, ry]:\n"
            "            raster[rx, ry], best[rx, ry] = labels[i], sample.signals.intensity[i]\n"
            "    return raster\n\n"
            "truth_primary = sample.primary_label_map()\n"
            "pred_primary = primary_map(sample, result.labels)\n"
            "fig, axes = plt.subplots(2, 3, figsize=(12, 8))\n"
            "panels = [\n"
            "    (sample.membership_count_map(), 'latent memberships/probe', 'viridis'),\n"
            "    (sample.observed_membership_count_map(), 'observed memberships/probe', 'viridis'),\n"
            "    (sample.membership_count_map() - sample.observed_membership_count_map(), 'missing latent memberships', 'magma'),\n"
            "    (categorical_grain_rgb(truth_primary), 'truth primary grain (categorical)', None),\n"
            "    (categorical_grain_rgb(pred_primary), 'Stage-A primary grain (categorical)', None),\n"
            "    ((pred_primary >= 0).astype(int), 'assigned footprint (binary)', 'gray'),\n"
            "]\n"
            "for ax, (image, title, cmap) in zip(axes.ravel(), panels):\n"
            "    ax.imshow(image, origin='upper', interpolation='nearest', cmap=cmap)\n"
            "    ax.set_title(title)\n"
            "    ax.set_axis_off()\n"
            "fig.suptitle(f'{SCENARIO} — {CONFIG_NAME}')\n"
            "fig.tight_layout()"
        ),
        nbf.v4.new_code_cell(
            "grain_columns = ['truth_grain_id', 'predicted_grain_id', 'signal_f1', 'footprint_recall', 'core_recall', 'edge_recall', 'predicted_components_8', 'predicted_bbox_occupancy']\n"
            "show_table(grain_metrics, grain_columns, formats={key: '.3f' for key in grain_columns if key not in {'truth_grain_id', 'predicted_grain_id', 'predicted_components_8'}})"
        ),
        nbf.v4.new_markdown_cell("## Lowest-ARI held-out cases"),
        nbf.v4.new_code_cell(
            "worst = []\n"
            "for config in unique_values(test, 'config'):\n"
            "    worst.extend(sorted(select_rows(test, config=config), key=lambda row: row['signal_ari'])[:8])\n"
            "worst_columns = ['config', 'scenario', 'signal_ari', 'n_true_grains', 'n_predicted_grains', 'morphology', 'orientation_heterogeneity', 'detection_regime']\n"
            "show_table(worst, worst_columns, formats={'signal_ari': '.3f'})\n"
            "print('Pre-rendered categorical panels:', RESULTS / 'worst_cases')"
        ),
        nbf.v4.new_markdown_cell(
            "## Current conclusion\n\n"
            "The conservative combined candidate passes the predeclared validation gates and improves "
            "held-out grain-count MAE and split excess, with unchanged noise rejection and fewer severe "
            "failures. Held-out mean ARI is slightly below `default_n1`, so the candidate remains "
            "review-ready rather than silently replacing production. Unconditional radius-2 linking "
            "remains rejected.\n\n"
            "A separate multi-label footprint reconstruction stage should be evaluated against the "
            "latent masks before treating experimental maps as filled grain bodies. Do not enable the "
            "current Stage B on experimental data merely to make maps look filled: its previous label "
            "changes were too destructive and it has not passed this synthetic benchmark."
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
