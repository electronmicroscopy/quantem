"""Phase mapping from matched crystal orientations.

PhaseMap compares candidate crystal patterns (each carrying orientations
matched and refined by OrientationMap) against the measured Bragg peaks at
every probe position. Every subset of candidates up to `max_patterns` is
scored as a joint model: the candidate pattern weights are solved by
non-negative least squares on the paired peak intensities, and the model cost
extends Diebold et al., Microsc. Microanal. 31, ozaf019 (2025):

    c(S) = sum_exp_peaks |I_m - sum_f w_f I_pred,f| + sum_f w_f I_unpaired,f
         + penalty * (|S| - 1)

normalized by the total measured intensity. Unpaired experimental intensity
appears in the first term (its prediction is zero); unpaired simulated
intensity is charged in full. The best subset answers orientation/phase
ambiguity directly: one orientation, two orientations of one phase, or two
phases, whichever explains the pattern best. Reliability is the cost gap
between the best models with and without the winning phase.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from tqdm import tqdm

from quantem.core.io.serialize import AutoSerialize
from quantem.diffraction.defaults import (
    MIN_NUMBER_PEAKS,
    MIN_SIM_INTENSITY_REL,
    PAIR_DISTANCE,
    POWER_INTENSITY,
    resolve,
)
from quantem.diffraction.orientation import OrientationMap, position_mask

# Diffracted intensity is strongly skewed, so a linear brightness leaves most
# indexed positions nearly black. This exponent is applied wherever a map is
# shaded by signal, so the phase map and the orientation maps agree.
SHADE_GAMMA = 0.5


def _majority_filter(phase: np.ndarray, radius: int) -> np.ndarray:
    """Replace each position by the most common phase around it.

    Unindexed positions (-1) take part, so an isolated crystal pixel in
    vacuum is removed rather than spreading.

    Parameters
    ----------
    phase : np.ndarray
        ``(scan_row, scan_col)`` phase indices, -1 where unindexed.
    radius : int
        Half-width of the square neighbourhood in probe positions.

    Returns
    -------
    np.ndarray
        Filtered phase indices, same shape and dtype.
    """
    from scipy.ndimage import uniform_filter

    labels = np.unique(phase)
    votes = np.stack(
        [
            uniform_filter((phase == v).astype(float), size=2 * radius + 1, mode="nearest")
            for v in labels
        ]
    )
    return labels[votes.argmax(axis=0)]


class PhaseMap(AutoSerialize):
    """Assign best-fit phases to every probe position.

    Workflow::

        pm = PhaseMap.from_orientation_maps([om_alpha, om_beta])
        pm.fit()
        pm.plot_phase()

    Candidates are all (crystal, match) pairs of the input OrientationMaps,
    so two matched orientations of one crystal compete on equal footing with
    one orientation of each of two crystals.
    """

    _token = object()

    def __init__(self, orientation_maps: list[OrientationMap], _token=None):
        """Private constructor; use :meth:`from_orientation_maps`.

        Enumerates the candidate list, one entry per (orientation map, match)
        pair, so that two matched orientations of one crystal compete on equal
        footing with one orientation of each of two crystals. The fit results
        are left as None until :meth:`fit` runs.

        Parameters
        ----------
        orientation_maps : list of OrientationMap
            Per-crystal maps sharing one set of peaks.
        _token : object
            Guard against direct construction.

        Raises
        ------
        RuntimeError
            If called without the class token.
        """
        if _token is not self._token:
            raise RuntimeError("Use PhaseMap.from_orientation_maps().")
        self.orientation_maps = orientation_maps
        self.names = [om.crystal.name for om in orientation_maps]
        # candidate list: (map index, match index)
        self.candidates: list[tuple[int, int]] = []
        for i, om in enumerate(orientation_maps):
            assert om.quats is not None
            for m in range(om.quats.shape[2]):
                self.candidates.append((i, m))

        # hyperparameters inherited from the maps and recorded per stage
        self.metadata: dict = {"orientation_maps": [dict(om.metadata) for om in orientation_maps]}
        # fit results, (R, C, ...) over the scan; see fit()
        self.phase_weights: torch.Tensor | None = None  # (R, C, F) per candidate
        self.crystal_weights: torch.Tensor | None = None  # (R, C, n_crystals)
        self.cost_best: torch.Tensor | None = None
        self.phase_index: torch.Tensor | None = None
        self.reliability: torch.Tensor | None = None
        self.diffracted_intensity: torch.Tensor | None = None
        self.num_diffracted: torch.Tensor | None = None

    @classmethod
    def from_orientation_maps(cls, orientation_maps: list[OrientationMap]) -> "PhaseMap":
        """Create from OrientationMaps that share the same peaks.

        Parameters
        ----------
        orientation_maps : list of OrientationMap
            Matched maps, one per candidate crystal.

        Returns
        -------
        PhaseMap

        Raises
        ------
        ValueError
            If the maps have different scan shapes.
        RuntimeError
            If a map has not been matched.
        """
        p0 = orientation_maps[0].peaks
        for om in orientation_maps:
            if om.peaks.shape != p0.shape:
                raise ValueError("All OrientationMaps must share the same scan shape.")
            if om.quats is None:
                raise RuntimeError(
                    f"OrientationMap for {om.crystal.name}: run match_orientations() first."
                )
        return cls(orientation_maps, _token=cls._token)

    def fit(
        self,
        positions=None,
        pair_distance: float | None = None,
        power_intensity: float | None = None,
        max_patterns: int = 2,
        complexity_penalty: float = 0.02,
        weight_unmatched_sim: float = 0.5,
        weight_overprediction: float = 1.0,
        min_sim_intensity_rel: float | None = None,
        k_max: float | None = None,
        min_number_peaks: int | None = None,
        min_diffracted_peaks: int = 2,
        null_k_min: float = 0.05,
        progress_bar: bool = True,
    ) -> "PhaseMap":
        """Score all candidate subsets at every probe position.

        Parameters left as None inherit the values the orientation
        matching used (its plan kernel, intensity power and peak minimum),
        so the phase decision compares the same peaks the same way; the
        resolved values are recorded in `metadata['fit']`.

        Parameters
        ----------
        positions : list[tuple[int, int]] | np.ndarray | None
            Scan positions to fit: a list of (row, col) or an (R, C)
            boolean mask. None (default) fits every position matched by
            all of the orientation maps, so a staged test run on a few
            positions carries through without repeating the list.
        pair_distance : float | None
            Pairing distance delta (1/Angstroms) between simulated and
            measured peaks; inherits the plan's correlation kernel.
        power_intensity : float | None
            Intensities are raised to this power before comparison;
            inherits the plan's value.
        max_patterns : int, default=2
            Maximum number of candidate patterns fit simultaneously.
        complexity_penalty : float, default=0.02
            Added cost per extra pattern in a model; sets how much better a
            two-pattern fit must be to beat a single-pattern fit.
        weight_unmatched_sim : float, default=0.5
            Cost weight of simulated intensity with no experimental partner.
        weight_overprediction : float, default=1.0
            Cost weight of predicted intensity in excess of the measured
            value on paired peaks. Unexplained measured intensity always
            costs in full (coverage), but the symmetric intensity-mismatch
            term assumes kinematical intensities are trustworthy; on
            non-precession data with strong dynamical scattering, lowering
            this weight makes the decision coverage-driven and removes the
            bias toward sparse templates.
        min_sim_intensity_rel : float | None
            Simulated reflections weaker than this fraction of the pattern
            maximum are dropped before comparison: kinematically weak spots
            are frequently unobservable and should not penalize a phase whose
            structure factors happen to include many of them. None takes
            MIN_SIM_INTENSITY_REL (0.02).
        k_max : float | None
            Restrict the comparison below this scattering vector
            (1/Angstroms).
        min_number_peaks : int | None
            Positions with fewer measured peaks (direct beam included) are
            not fit and stay unindexed; inherits the matching's value.
        min_diffracted_peaks : int, default=2
            Null hypothesis: a position needs at least this many measured
            peaks beyond `null_k_min` before any phase is assigned. Vacuum
            and amorphous support carry the direct beam and little else, and
            a crystal fit to that is noise; those positions are left
            unindexed (`phase_index` of -1) and plot black.
        null_k_min : float, default=0.05
            Scattering vector (1/Angstroms) above which a measured peak
            counts as diffracted. The default excludes the direct beam,
            which sits at the origin after `correct_peak_origins`.
        progress_bar : bool, default=True
            Show a progress bar over the positions.

        Returns
        -------
        PhaseMap
            Self, with these (R, C, ...) results:

            - `phase_index`: winning crystal, -1 where unindexed (not fit,
              rejected by the null hypothesis, or no candidate weight).
            - `phase_weights`: (R, C, F) non-negative weight of every
              candidate in the best model, F = len(`candidates`).
            - `crystal_weights`: (R, C, n_crystals) those weights summed per
              crystal and normalized to sum to one, zero where unindexed.
            - `cost_best`: cost of the best model, NaN where not fit.
            - `reliability`: cost gap to the best model without the winning
              crystal, NaN where not fit or no such model exists.
            - `diffracted_intensity`, `num_diffracted`: measured intensity
              and number of peaks beyond `null_k_min`.
        """
        from scipy.optimize import nnls

        oms = self.orientation_maps
        plan_md = oms[0].metadata.get("plan")
        match_md = oms[0].metadata.get("match")
        pair_distance = resolve(pair_distance, "pair_distance", plan_md, default=PAIR_DISTANCE)
        power_intensity = resolve(
            power_intensity, "power_intensity", plan_md, default=POWER_INTENSITY
        )
        min_sim_intensity_rel = resolve(
            min_sim_intensity_rel, "min_sim_intensity_rel", default=MIN_SIM_INTENSITY_REL
        )
        min_number_peaks = resolve(
            min_number_peaks, "min_number_peaks", match_md, default=MIN_NUMBER_PEAKS
        )
        self.metadata["fit"] = dict(
            positions=None if positions is None else "subset",
            pair_distance=float(pair_distance),
            power_intensity=float(power_intensity),
            max_patterns=int(max_patterns),
            complexity_penalty=float(complexity_penalty),
            weight_unmatched_sim=float(weight_unmatched_sim),
            weight_overprediction=float(weight_overprediction),
            min_sim_intensity_rel=float(min_sim_intensity_rel),
            k_max=k_max,
            min_number_peaks=int(min_number_peaks),
            min_diffracted_peaks=int(min_diffracted_peaks),
            null_k_min=float(null_k_min),
        )
        peaks = oms[0].peaks
        R, C = peaks.shape[0], peaks.shape[1]
        cands = self.candidates
        F = len(cands)
        delta = pair_distance

        fields = peaks.fields
        ix = [fields.index(f) for f in ("qx", "qy", "intensity")]

        subsets = [s for n in range(1, max_patterns + 1) for s in combinations(range(F), n)]

        cost_best = torch.full((R, C), torch.nan, dtype=torch.float64)
        weights_out = torch.zeros((R, C, F), dtype=torch.float64)
        reliability = torch.full((R, C), torch.nan, dtype=torch.float64)
        diffracted = torch.zeros((R, C), dtype=torch.float64)
        num_diffracted = torch.zeros((R, C), dtype=torch.long)

        active = position_mask(positions, (R, C))
        for om in oms:
            if om.computed is not None:
                active = active & om.computed
        iterator = [(rx, ry) for rx, ry in np.ndindex(R, C) if active[rx, ry]]
        if progress_bar:
            iterator = tqdm(iterator, desc="phase mapping")
        for rx, ry in iterator:
            data = peaks[rx, ry].numpy().astype(np.float64)
            if data.shape[0] < min_number_peaks:
                continue
            # null hypothesis: no diffracted signal, so no phase to decide.
            # Vacuum and amorphous support carry the direct beam and nothing
            # else, and the measured signal beyond it is the evidence that
            # any crystal is present at all.
            qr_meas = np.hypot(data[:, ix[0]], data[:, ix[1]])
            beyond = qr_meas > null_k_min
            diffracted[rx, ry] = float(data[beyond, ix[2]].clip(min=0).sum())
            num_diffracted[rx, ry] = int(beyond.sum())
            if int(beyond.sum()) < min_diffracted_peaks:
                continue
            qxy = torch.as_tensor(data[:, ix[:2]], dtype=torch.float64)
            im = torch.as_tensor(data[:, ix[2]], dtype=torch.float64).clamp_min(0)
            if k_max is not None:
                keep = torch.linalg.norm(qxy, dim=1) <= k_max
                qxy, im = qxy[keep], im[keep]
            im = im**power_intensity
            n_exp = im.shape[0]
            int_total = float(im.sum())

            # per-candidate predicted intensity on each experimental peak,
            # and unpaired simulated intensity
            pred = np.zeros((n_exp, F))
            unpaired_sim = np.zeros(F)
            for f, (i_om, m) in enumerate(cands):
                om = oms[i_om]
                if om.corr[rx, ry, m] <= 0:
                    continue
                sim = om.generate_pattern(rx, ry, match=m, k_max=k_max)
                sq = torch.stack((sim["qx"], sim["qy"]), dim=1)
                s_raw = sim["intensity"]
                if sq.shape[0] == 0:
                    continue
                vis = s_raw > min_sim_intensity_rel * s_raw.max()
                sq, s_raw = sq[vis], s_raw[vis]
                si = s_raw**power_intensity
                d = torch.cdist(sq, qxy)
                d_min, j_min = d.min(dim=1)
                pair = d_min < delta
                frac = (d_min[pair] / delta).clamp(0, 1)
                np.add.at(
                    pred[:, f],
                    j_min[pair].numpy(),
                    (si[pair] * (1 - frac)).numpy(),
                )
                unpaired_sim[f] = weight_unmatched_sim * (
                    float(si[~pair].sum()) + float((si[pair] * frac).sum())
                )

            im_np = im.numpy()
            results = []
            for s in subsets:
                cols = [f for f in s if pred[:, f].any() or unpaired_sim[f] > 0]
                if len(cols) == 0:
                    continue
                B = pred[:, cols]
                w, _ = nnls(B, im_np)
                model = B @ w
                under = np.maximum(im_np - model, 0).sum()  # unexplained measured
                over = np.maximum(model - im_np, 0).sum()  # overpredicted paired
                cost = (under + weight_overprediction * over + (w * unpaired_sim[cols]).sum()) / (
                    int_total + 1e-12
                ) + complexity_penalty * (len(cols) - 1)
                results.append((cost, s, cols, w))
            if not results:
                continue
            results.sort(key=lambda r: r[0])
            c_best, _, cols_best, w_best = results[0]
            cost_best[rx, ry] = c_best
            for f, w in zip(cols_best, w_best):
                weights_out[rx, ry, f] = w

            # reliability: cost gap to the best model containing NO candidate
            # of the dominant crystal (candidates of one crystal can be
            # near-duplicates, e.g. after residual re-matching)
            f_dom = cols_best[int(np.argmax(w_best))]
            i_dom = cands[f_dom][0]
            others = [c for c, s, _, _ in results if all(cands[f][0] != i_dom for f in s)]
            reliability[rx, ry] = (min(others) - c_best) if others else torch.nan

        self.cost_best = cost_best
        self.diffracted_intensity = diffracted
        self.num_diffracted = num_diffracted
        self.phase_weights = weights_out
        self.reliability = reliability

        # dominant phase: candidate weights summed per crystal
        n_maps = len(oms)
        w_phase = torch.zeros((R, C, n_maps), dtype=torch.float64)
        for f, (i_om, _) in enumerate(cands):
            w_phase[..., i_om] += weights_out[..., f]
        # argmax over all-zero weights returns 0, which would label every
        # position that was never fit, or whose best model has no weight
        # (NNLS returns all zeros when no candidate overlaps the peaks), as
        # the first phase; mark them instead
        w_sum = w_phase.sum(dim=-1)
        self.phase_index = w_phase.argmax(dim=-1)
        unindexed = torch.isnan(cost_best) | (w_sum <= 0)
        self.phase_index[unindexed] = -1
        self.reliability[unindexed] = torch.nan
        self.crystal_weights = w_phase / w_sum[..., None].clamp_min(1e-12)
        return self

    def apply_dynamical(self, result: dict) -> "PhaseMap":
        """Take the phase decision from a dynamical refinement.

        A phase map can be built from the orientation maps at any stage:
        after matching, after refine_orientations, or after
        bloch.refine_dynamical, whose per-candidate intensity costs decide
        the phase here. The reliability becomes the cost gap between the
        best candidates of the winning crystal and of the runner-up
        crystal, and the kinematical result is kept under
        `metadata['kinematical']`. Positions the refinement did not reach
        (outside its `mask`) keep their current decision, so a refinement
        of one region, or several in stages, updates only that region.

        Parameters
        ----------
        result : dict
            Output of :func:`~quantem.diffraction.bloch.refine_dynamical`:
            "cost" (R, C, F) per-candidate cost, NaN where not refined,
            "phase_index" (R, C) the winning crystal, and optionally
            "metadata".

        Returns
        -------
        PhaseMap
            Self, with `phase_index`, `reliability` and `cost_best` updated
            at the refined positions. With one crystal there is no
            runner-up and the reliability is NaN, as in :meth:`fit`.
        """
        cost = torch.nan_to_num(result["cost"], nan=torch.inf)
        n_maps = len(self.orientation_maps)
        R, C = cost.shape[:2]
        cost_phase = torch.full((R, C, n_maps), torch.inf, dtype=cost.dtype)
        for f, (i_om, _) in enumerate(self.candidates):
            cost_phase[..., i_om] = torch.minimum(cost_phase[..., i_om], cost[..., f])
        order = cost_phase.sort(dim=-1).values
        reliability = torch.where(
            torch.isfinite(order[..., 0]),
            (order[..., 1] - order[..., 0]).clamp_min(0)
            if n_maps > 1
            else torch.full_like(order[..., 0], torch.nan),
            torch.full_like(order[..., 0], torch.nan),
        )
        done = torch.isfinite(order[..., 0])
        if self.phase_index is None:
            self.phase_index = torch.full((R, C), -1, dtype=torch.long)
            self.reliability = torch.full((R, C), torch.nan, dtype=cost.dtype)
            self.cost_best = torch.full((R, C), torch.nan, dtype=cost.dtype)
        # keep the kinematical decision from before the first dynamical pass
        self.metadata.setdefault(
            "kinematical",
            {
                "phase_index": self.phase_index.clone(),
                "reliability": self.reliability.clone(),
            },
        )
        self.phase_index = torch.where(done, result["phase_index"], self.phase_index)
        self.reliability = torch.where(done, reliability, self.reliability)
        self.cost_best = torch.where(done, order[..., 0], self.cost_best)
        self.metadata["dynamical_applied"] = dict(result.get("metadata", {}))
        return self

    def signal_confidence(
        self,
        signal_range: tuple[float, float] | str = "auto",
        gamma: float = 1.0,
    ) -> np.ndarray:
        """Confidence in [0, 1] that a crystal is present, from the data alone.

        The measured intensity beyond the direct beam, scaled to [0, 1]. This
        is the null-hypothesis test made visible: vacuum and amorphous support
        diffract nothing, so they score zero however well some orientation
        happens to correlate. Positions the null hypothesis left unindexed in
        :meth:`fit` are forced to zero.

        Use it to fade the phase map and the orientation maps together::

            conf = pm.signal_confidence()
            mask_a = (pm.phase_index.numpy() == 0) * conf

        Parameters
        ----------
        signal_range : tuple | "auto", default="auto"
            Diffracted intensity mapped to 0 ... 1. "auto" spans zero to half
            the median over the indexed positions, so a crystal shows at full
            strength and only weakly diffracting positions fade.
        gamma : float, default=1.0
            Exponent applied to the result. The default returns the raw
            confidence, which is what a threshold should be taken on;
            :attr:`SHADE_GAMMA` is the value used when shading a display, and
            zero stays zero either way.

        Returns
        -------
        np.ndarray
            ``(scan_row, scan_col)`` confidence in [0, 1].
        """
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if self.diffracted_intensity is None:
            raise ValueError("Run fit() before signal_confidence().")
        sig = np.nan_to_num(self.diffracted_intensity.numpy())
        indexed = self.phase_index.numpy() >= 0
        if isinstance(signal_range, str):
            # full brightness from half the median signal of the indexed
            # positions: crystals show at full strength and only positions
            # that diffract well below typical fade
            vals = sig[indexed]
            hi = 0.5 * float(np.median(vals)) if vals.size else 1.0
            lo, hi = 0.0, max(hi, 1e-12)
        else:
            lo, hi = signal_range
        conf = ((sig - lo) / max(hi - lo, 1e-12)).clip(0, 1) * indexed
        return conf if gamma == 1.0 else np.power(conf, gamma)

    def plot_phase(
        self,
        phase_colors: np.ndarray | None = None,
        shade_by: str = "signal",
        shade_range: tuple[float, float] | str = "auto",
        shade_gamma: float = SHADE_GAMMA,
        majority_filter: int = 0,
        reliability_range: tuple[float, float] | None = None,
        scalebar: dict | str | None = "auto",
        figax=None,
    ):
        """Dominant-phase map: color gives the crystal, brightness the evidence.

        By default the brightness is the diffracted signal, so vacuum and
        unindexed positions are black; see `shade_by`.

        Parameters
        ----------
        phase_colors : np.ndarray | None
            One RGB color per phase, cycled when there are more phases;
            defaults to `DEFAULT_PHASE_COLORS`, the palette shared with the
            pattern overlay plots (gold, cyan, green, purple).
        shade_by : {"signal", "reliability", "none"}, default="signal"
            What the brightness means. "signal" fades each position by the
            measured diffracted intensity (see :meth:`signal_confidence`), so
            vacuum and amorphous support go black and the map shows where
            crystals actually are. "reliability" uses the cost gap to the best
            model without the winning crystal, which answers a different
            question -- which phase, given that there is one -- and carries no
            information about whether anything is there. "none" draws every
            indexed position at full color.
        shade_range : tuple | "auto", default="auto"
            Values mapped to black ... full color. "auto" takes a high
            percentile over the indexed positions, since the absolute scale
            depends on the data.
        shade_gamma : float, default=:attr:`SHADE_GAMMA` (0.5)
            Exponent applied to the brightness, ``alpha ** shade_gamma``.
            Diffracted intensity is strongly skewed, so a linear scale leaves
            most indexed positions dark and only the brightest grains
            readable. Values below 1 lift the faint ones: 0.5 is the default
            and 1.0 restores the linear scale. Zero brightness is a fixed
            point, so vacuum and unindexed positions stay black however low
            this is set, and the colorbars carry the same curve.
        majority_filter : int, default=0
            Radius in probe positions of a majority filter applied to the
            phase decision for display only; the stored decision is
            untouched. 1 replaces each position by the most common phase in
            its 3x3 neighbourhood, which removes isolated single-pixel
            phases without moving a real boundary. Orientation smoothing
            does not do this: it averages orientations within one phase and
            leaves the phase assignment alone.
        reliability_range : tuple | None
            Backwards-compatible shortcut: setting it selects
            ``shade_by="reliability"`` with this range.
        scalebar : dict | "auto" | None
            Real-space scale bar. "auto" (the default) takes the scan step
            and units carried from the dataset by the orientation maps; a
            dict such as {"sampling": 30, "units": "A"} overrides it, and
            None draws no bar.
        figax : (fig, ax) | None
            Existing axes to draw into.

        Returns
        -------
        tuple
            ``(fig, ax)``.

        Raises
        ------
        ValueError
            If `shade_by` is unknown or `shade_gamma` is not positive.
        """
        if isinstance(scalebar, str):
            scalebar = self.orientation_maps[0].scan_scalebar if scalebar == "auto" else None
        import matplotlib.pyplot as plt

        from quantem.core.visualization.visualization_utils import add_scalebar_to_ax
        from quantem.diffraction.orientation_visualization import phase_color_cycle

        assert self.phase_index is not None and self.reliability is not None
        phase_colors = phase_color_cycle(len(self.names), phase_colors)
        if reliability_range is not None:
            shade_by, shade_range = "reliability", reliability_range
        phase = self.phase_index.numpy()
        indexed = phase >= 0
        if shade_by == "signal":
            alpha = self.signal_confidence(shade_range)
            lo, hi = 0.0, 1.0
            cbar_label = "diffracted signal"
        elif shade_by == "reliability":
            rel = np.nan_to_num(self.reliability.numpy(), nan=0.0)
            if isinstance(shade_range, str):
                vals = rel[indexed]
                hi = float(np.percentile(vals, 98)) if vals.size else 1.0
                lo, hi = 0.0, max(hi, 1e-12)
            else:
                lo, hi = shade_range
            alpha = ((rel - lo) / max(hi - lo, 1e-12)).clip(0, 1) * indexed
            cbar_label = "reliability"
        elif shade_by == "none":
            alpha = indexed.astype(float)
            lo, hi = 0.0, 1.0
            cbar_label = "indexed"
        else:
            raise ValueError(
                f"shade_by must be 'signal', 'reliability' or 'none', got {shade_by!r}"
            )
        if shade_gamma <= 0:
            raise ValueError(f"shade_gamma must be positive, got {shade_gamma}")
        # zero maps to zero under any positive exponent, so unindexed positions
        # stay black and only the faint indexed ones are lifted
        alpha = np.power(alpha, shade_gamma)
        if majority_filter > 0:
            # the filter can turn an indexed position unindexed (-1) and the
            # reverse, so the colors and the black mask follow the filtered
            # decision; a position it newly indexes has no brightness of its
            # own and stays black
            phase = _majority_filter(phase, int(majority_filter))
            alpha = alpha * (phase >= 0)
        rgb = phase_colors[np.where(phase >= 0, phase, 0)] * alpha[..., None]

        if figax is None:
            fig, ax = plt.subplots(figsize=(9, 4.5))
        else:
            fig, ax = figax
        ax.imshow(rgb, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if scalebar is not None:
            add_scalebar_to_ax(
                ax,
                array_size=rgb.shape[1],
                sampling=scalebar.get("sampling", 1.0),
                length_units=scalebar.get("length", None),
                units=scalebar.get("units", "pixels"),
                width_px=rgb.shape[0] / 40,
                pad_px=rgb.shape[0] / 80,
                color=scalebar.get("color", "white"),
                loc="lower right",
            )
        handles = [
            plt.Line2D([0], [0], marker="s", ls="", color=c, label=n)
            for c, n in zip(phase_colors, self.names)
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=8)
        # stacked reliability colorbars, black -> phase color
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap, Normalize

        n_ph = len(phase_colors)
        for k, color in enumerate(phase_colors):
            cmap_k = LinearSegmentedColormap.from_list(
                f"rel{k}", [(0, 0, 0), tuple(color)], gamma=shade_gamma
            )
            cax = ax.inset_axes([1.02 + 0.025 * k, 0.05, 0.025, 0.9])
            cb = fig.colorbar(ScalarMappable(norm=Normalize(lo, hi), cmap=cmap_k), cax=cax)
            if k < n_ph - 1:
                cb.set_ticks([])
            else:
                cb.set_ticks([lo, hi])
                cb.set_label(cbar_label, fontsize=9)
        return fig, ax
