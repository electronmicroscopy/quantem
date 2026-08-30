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
from quantem.diffraction.orientation import OrientationMap


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

        self.phase_weights: torch.Tensor | None = None
        self.costs_single: torch.Tensor | None = None
        self.cost_best: torch.Tensor | None = None
        self.phase_index: torch.Tensor | None = None
        self.reliability: torch.Tensor | None = None

    @classmethod
    def from_orientation_maps(cls, orientation_maps: list[OrientationMap]) -> "PhaseMap":
        """Create from OrientationMaps that share the same peaks."""
        p0 = orientation_maps[0].peaks
        for om in orientation_maps:
            if om.peaks.shape != p0.shape:
                raise ValueError("All OrientationMaps must share the same scan shape.")
            if om.quats is None:
                raise RuntimeError(f"OrientationMap for {om.crystal.name}: run match() first.")
        return cls(orientation_maps, _token=cls._token)

    def fit(
        self,
        pair_distance: float = 0.05,
        power_intensity: float = 0.25,
        max_patterns: int = 2,
        complexity_penalty: float = 0.02,
        weight_unmatched_sim: float = 0.5,
        weight_overprediction: float = 1.0,
        min_sim_intensity_rel: float = 0.02,
        k_max: float | None = None,
        min_number_peaks: int = 3,
        progress_bar: bool = True,
    ) -> "PhaseMap":
        """Score all candidate subsets at every probe position.

        Parameters
        ----------
        pair_distance : float, default=0.05
            Pairing distance delta (1/Angstroms) between simulated and
            measured peaks.
        power_intensity : float, default=0.25
            Intensities are raised to this power before comparison.
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
        min_sim_intensity_rel : float, default=0.02
            Simulated reflections weaker than this fraction of the pattern
            maximum are dropped before comparison: kinematically weak spots
            are frequently unobservable and should not penalize a phase whose
            structure factors happen to include many of them.
        k_max : float | None
            Restrict the comparison below this scattering vector.
        """
        from scipy.optimize import nnls

        oms = self.orientation_maps
        peaks = oms[0].peaks
        R, C = peaks.shape[0], peaks.shape[1]
        cands = self.candidates
        F = len(cands)
        delta = pair_distance

        fields = peaks.fields
        ix = [fields.index(f) for f in ("qx", "qy", "intensity")]

        subsets = [s for n in range(1, max_patterns + 1) for s in combinations(range(F), n)]

        costs_single = torch.full((R, C, F), torch.nan, dtype=torch.float64)
        cost_best = torch.full((R, C), torch.nan, dtype=torch.float64)
        weights_out = torch.zeros((R, C, F), dtype=torch.float64)
        reliability = torch.zeros((R, C), dtype=torch.float64)
        best_subset = torch.full((R, C), -1, dtype=torch.long)

        iterator = list(np.ndindex(R, C))
        if progress_bar:
            iterator = tqdm(iterator, desc="phase mapping")
        for rx, ry in iterator:
            data = peaks[rx, ry].array
            if data.shape[0] < min_number_peaks:
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
                cost = (
                    under + weight_overprediction * over + (w * unpaired_sim[cols]).sum()
                ) / (int_total + 1e-12) + complexity_penalty * (len(cols) - 1)
                results.append((cost, s, cols, w))
            if not results:
                continue
            results.sort(key=lambda r: r[0])
            c_best, s_best, cols_best, w_best = results[0]
            cost_best[rx, ry] = c_best
            best_subset[rx, ry] = subsets.index(s_best)
            for f, w in zip(cols_best, w_best):
                weights_out[rx, ry, f] = w
            for cost, s, _, _ in results:
                if len(s) == 1:
                    costs_single[rx, ry, s[0]] = cost

            # reliability: cost gap to the best model containing NO candidate
            # of the dominant crystal (candidates of one crystal can be
            # near-duplicates, e.g. after residual re-matching)
            f_dom = cols_best[int(np.argmax(w_best))]
            i_dom = cands[f_dom][0]
            others = [
                c
                for c, s, _, _ in results
                if all(cands[f][0] != i_dom for f in s)
            ]
            reliability[rx, ry] = (min(others) - c_best) if others else torch.nan

        self.costs_single = costs_single
        self.cost_best = cost_best
        self.phase_weights = weights_out
        self.reliability = reliability
        self.best_subset = best_subset

        # dominant phase: candidate weights summed per crystal
        n_maps = len(oms)
        w_phase = torch.zeros((R, C, n_maps), dtype=torch.float64)
        for f, (i_om, _) in enumerate(cands):
            w_phase[..., i_om] += weights_out[..., f]
        self.phase_index = w_phase.argmax(dim=-1)
        self.phase_fractions = w_phase / w_phase.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return self

    def plot_phase(
        self,
        phase_colors: np.ndarray | None = None,
        reliability_range: tuple[float, float] = (0.0, 0.1),
        scalebar: dict | None = None,
        figax=None,
    ):
        """Dominant-phase map, colored by phase and shaded by reliability.

        Parameters
        ----------
        phase_colors : np.ndarray | None
            One RGB color per phase; defaults to the shared palette used by
            the pattern overlay plots (gold, light blue, ...).
        reliability_range : tuple, default=(0.0, 0.1)
            Reliability values mapped to black ... full color.
        scalebar : dict | None
            Real-space scale bar, e.g. {"sampling": 30, "units": "A"}.
        """
        import matplotlib.pyplot as plt

        from quantem.core.visualization.visualization_utils import add_scalebar_to_ax
        from quantem.diffraction.orientation_visualization import DEFAULT_PHASE_COLORS

        assert self.phase_index is not None and self.reliability is not None
        if phase_colors is None:
            phase_colors = DEFAULT_PHASE_COLORS[: len(self.names)]
        lo, hi = reliability_range
        rel = np.nan_to_num(self.reliability.numpy(), nan=0.0)
        alpha = ((rel - lo) / (hi - lo)).clip(0, 1)
        rgb = phase_colors[self.phase_index.numpy()] * alpha[..., None]

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
        ax.legend(handles=handles, loc="upper right", fontsize=8)
        # stacked reliability colorbars, black -> phase color
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap, Normalize

        n_ph = len(phase_colors)
        for k, color in enumerate(phase_colors):
            cmap_k = LinearSegmentedColormap.from_list(
                f"rel{k}", [(0, 0, 0), tuple(color)]
            )
            cax = ax.inset_axes([1.02 + 0.025 * k, 0.05, 0.025, 0.9])
            cb = fig.colorbar(
                ScalarMappable(norm=Normalize(lo, hi), cmap=cmap_k), cax=cax
            )
            if k < n_ph - 1:
                cb.set_ticks([])
            else:
                cb.set_ticks([lo, hi])
                cb.set_label("reliability", fontsize=9)
        return fig, ax
