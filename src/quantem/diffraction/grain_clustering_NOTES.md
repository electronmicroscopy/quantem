# grain_clustering — deferred work & suggestions

Status of follow-ups suggested during design/implementation. Implemented pieces:
`extract_signals`, `cluster_signals_into_grains` (Stage A), `refine_grains_crf` (Stage B),
visualization overlays. Tests: `quantem/tests/diffraction/test_grain_clustering.py`.

## Real-data validation (2026-07-10)
- Validated on the saved real PB2T/TEG golden-model peak vectors (60x60 scan grid), using a
  20x20 crop and the source workflow's estimated/manually adjusted radial windows and -6.8 degree
  orientation offset. Notebook: `notebooks/diffraction/grain_clustering_real_data_validation_PB2T_TEG.ipynb`.
- Fixed the ingestion defaults to match the real `BraggPeaksPolymer` interface: `r_invA`, `theta`,
  and `intensities`. The interface-faithful synthetic fixture and all 27 tests pass.
- Crop result (735 signals, nine populated windows): baseline Stage A produced 53 grains and 169
  outliers (23.0%) in 0.08 s total. Stage B produced 49 grains and 43 outliers (5.9%) in 0.89 s,
  but changed some partitions substantially (notably window 7 ARI=0.12 and +68 boundary edges),
  so Stage B remains optional pending tuning.
- Stored-theta convention matches `make_orientation_histogram` at the median (1.03 degree circular
  difference); the 95th percentile is 43.9 degrees at multi-signal/smoothed probes and needs further
  interpretation before treating histogram mode as a per-signal ground truth.

## Deferred (not yet built)
1. **Auto-tuning helpers for thresholds.** Estimate `theta_tol` from the bimodal valley of
   nearest-neighbour orientation differences; estimate `r_tol_rel` from the radial gap between
   quantized peaks within a window (note `BraggPeaksPolymer.estimate_peak_windows` already finds the
   windows themselves). Also data-driven defaults for Stage B `theta_sigma_deg` / `r_sigma_rel` /
   `outlier_energy` / `lam`.
2. **Stage B optimizer alternatives.** α-expansion graph-cut backend (stronger global optimum than
   ICM) if a maxflow dep is acceptable; and vectorise the ICM Python loops for large N.
3. **Bayesian nonparametric ddCRP.** Full posterior over partitions + unknown grain count — the
   principled generative sibling of Stage A (uncertainty over the clustering itself).
4. **Expose K-best / m-best labels per ambiguous signal** (the "beam"). Currently only per-signal
   `confidence` + `margin` are surfaced.
5. **Downstream-compat output.** Optionally emit a clustered `OrientationMap` analogous to
   py4DSTEM `cluster_orientation_map`.

## Evaluated and intentionally NOT planned
- **Cross-radial-bin linking** — removed by design: radial windows are immutable signal classes
  (backbone / lamellar / pi-pi); grains must never span windows. Do not re-add.
- **Min-cost flow / multi-hypothesis tracking / literal beam search** — re-introduce raster-order
  dependence in a 2D field; the CRF/graph formulation is the correct global generalization.
- **Spectral clustering** — needs a preset cluster count and scales poorly; rejected as primary.
