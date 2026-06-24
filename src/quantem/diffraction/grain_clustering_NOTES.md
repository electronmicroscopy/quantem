# grain_clustering — deferred work & suggestions

Status of follow-ups suggested during design/implementation. Implemented pieces:
`extract_signals`, `cluster_signals_into_grains` (Stage A), `refine_grains_crf` (Stage B),
visualization overlays. Tests: `quantem/tests/diffraction/test_grain_clustering.py`.

## Deferred (not yet built)
1. **Real-data validation of `extract_signals`.** It is unit-tested only against a fake polar
   object. Exercise its duck-typed interface (`bp.polar_peaks[field][rx,ry]`, `.shape`,
   `.peak_intensities[field][rx,ry]`) on a real `BraggPeaksPolymer` before trusting on real data.
2. **Auto-tuning helpers for thresholds.** Estimate `theta_tol` from the bimodal valley of
   nearest-neighbour orientation differences; estimate `r_tol_rel` from the radial gap between
   quantized peaks within a window (note `BraggPeaksPolymer.estimate_peak_windows` already finds the
   windows themselves). Also data-driven defaults for Stage B `theta_sigma_deg` / `r_sigma_rel` /
   `outlier_energy` / `lam`.
3. **Stage B optimizer alternatives.** α-expansion graph-cut backend (stronger global optimum than
   ICM) if a maxflow dep is acceptable; and vectorise the ICM Python loops for large N.
4. **Bayesian nonparametric ddCRP.** Full posterior over partitions + unknown grain count — the
   principled generative sibling of Stage A (uncertainty over the clustering itself).
5. **Expose K-best / m-best labels per ambiguous signal** (the "beam"). Currently only per-signal
   `confidence` + `margin` are surfaced.
6. **Downstream-compat output.** Optionally emit a clustered `OrientationMap` analogous to
   py4DSTEM `cluster_orientation_map`.

## Evaluated and intentionally NOT planned
- **Cross-radial-bin linking** — removed by design: radial windows are immutable signal classes
  (backbone / lamellar / pi-pi); grains must never span windows. Do not re-add.
- **Min-cost flow / multi-hypothesis tracking / literal beam search** — re-introduce raster-order
  dependence in a 2D field; the CRF/graph formulation is the correct global generalization.
- **Spectral clustering** — needs a preset cluster count and scales poorly; rejected as primary.
