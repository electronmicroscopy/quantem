"""Phase and in-plane orientation from single systematic rows of reflections."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.visualization.visualization_utils import add_scalebar_to_ax, array_to_rgba
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.orientation import scan_scalebar
from quantem.diffraction.orientation_visualization import phase_color_cycle


class SystematicRowMap(AutoSerialize):
    """Phase and in-plane orientation mapping from systematic rows.

    A pattern containing a single systematic row of reflections, such as the
    local Fourier transform of an HRTEM image showing one set of lattice
    fringes, determines the length and the in-plane direction of the
    reciprocal lattice vector g, but not the rotation of the crystal about g.
    The orientation is then known only up to a fiber about g, and matching
    against full zone axis patterns has no unique answer. This class indexes
    each row on its own: Friedel pairs of peaks (+g and -g) are combined into
    one row, the length of g is assigned to the nearest ring of the candidate
    crystals, and the phase, the reflection family and the in-plane angle of
    g are stored for up to `num_rows` rows at every position. A position can
    hold several rows, from overlapping grains or from a grain showing two
    fringe sets.

    The in-plane angle of g is ``atan2(g_col, g_row)``, measured from the scan
    row axis toward the column axis, the azimuth convention of the quantEM
    calibration and orientation matching, and wrapped to [0, 180) degrees
    since g and -g give the same fringes. The lattice fringes run
    perpendicular to g.

    Use :meth:`from_vectors` to construct an instance, then :meth:`index`.

    Parameters
    ----------
    peaks : Vector
        Calibrated peaks with fields (qx, qy, intensity) in 1/Angstroms,
        origin at zero.
    crystals : list of Crystal
        Candidate phases with structure factors calculated.
    """

    _token = object()

    def __init__(
        self,
        peaks,
        crystals: list[Crystal],
        k_min: float,
        k_max: float,
        min_ring_intensity: float,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use SystematicRowMap.from_vectors() to instantiate this class.")
        super().__init__()
        self.peaks = peaks
        self.crystals = list(crystals)
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.min_ring_intensity = float(min_ring_intensity)

        # ring table, one entry per reflection family of every crystal
        self.ring_k, self.ring_phase, self.ring_label = _ring_table(
            self.crystals, self.k_min, self.k_max, self.min_ring_intensity
        )

        # per-row results, (scan_row, scan_col, num_rows), filled by index()
        self.k: NDArray | None = None
        self.angle_deg: NDArray | None = None
        self.intensity: NDArray | None = None
        self.phase: NDArray | None = None
        self.ring: NDArray | None = None
        self.ambiguous: NDArray | None = None
        self.metadata: dict[str, Any] = {}

    @classmethod
    def from_vectors(
        cls,
        peaks,
        crystals: Crystal | list[Crystal],
        k_min: float = 0.1,
        k_max: float | None = None,
        min_ring_intensity: float = 0.01,
    ) -> "SystematicRowMap":
        """Bind calibrated peaks to the candidate crystals.

        Parameters
        ----------
        peaks : Vector
            Calibrated peaks (qx, qy, intensity) in 1/Angstroms, for example
            from :meth:`BraggVectors.apply_calibration`.
        crystals : Crystal or list of Crystal
            Candidate phases, structure factors calculated.
        k_min : float, default=0.1
            Peaks closer than this to the origin (1/Angstroms) are ignored,
            which removes the direct beam and low-frequency contrast.
        k_max : float, optional
            Largest ring radius considered (1/Angstroms). None (default) uses
            the largest measured peak radius.
        min_ring_intensity : float, default=0.01
            A reflection family is a candidate when its structure factor
            intensity reaches this fraction of the strongest reflection of its
            crystal.

        Returns
        -------
        SystematicRowMap
        """
        if isinstance(crystals, Crystal):
            crystals = [crystals]
        if k_max is None:
            flat = peaks.select_fields("qx", "qy").numpy()
            k_max = float(np.hypot(flat[:, 0], flat[:, 1]).max()) if flat.size else 1.0
        return cls(
            peaks=peaks,
            crystals=crystals,
            k_min=k_min,
            k_max=k_max,
            min_ring_intensity=min_ring_intensity,
            _token=cls._token,
        )

    @property
    def names(self) -> list[str]:
        """Names of the candidate crystals."""
        return [str(getattr(c, "name", f"phase {i}")) for i, c in enumerate(self.crystals)]

    @property
    def shape(self) -> tuple[int, int]:
        """Scan shape."""
        return (int(self.peaks.shape[0]), int(self.peaks.shape[1]))

    def rings(self, phase=None) -> list[tuple[str, str, float]]:
        """Candidate rings as (phase name, reflection family, radius in 1/Angstroms)."""
        idx = range(len(self.ring_k)) if phase is None else self._ring_indices(phase)
        return [
            (self.names[self.ring_phase[i]], self.ring_label[i], float(self.ring_k[i]))
            for i in idx
        ]

    def index(
        self,
        k_tol: float = 0.01,
        pair_tol: float | None = None,
        min_intensity: float = 0.0,
        num_rows: int = 4,
        require_pair: bool = True,
    ) -> "SystematicRowMap":
        """Pair the peaks into rows and index every row against the rings.

        Each peak is paired with the brightest unpaired peak within
        `pair_tol` of its inversion through the origin, and the pair becomes
        one row with ``g = (q_1 - q_2) / 2`` and the summed intensity. The row
        is assigned to the nearest candidate ring within `k_tol` of ``|g|``; a
        row with a second ring inside `k_tol` is marked ambiguous, and a row
        with no ring inside `k_tol` keeps phase -1. Rows are sorted by
        intensity and the strongest `num_rows` are kept.

        Parameters
        ----------
        k_tol : float, default=0.01
            Radial tolerance for the ring assignment, 1/Angstroms. Rings of
            different phases closer than about twice this value cannot be
            told apart.
        pair_tol : float, optional
            Largest distance between a peak and the inverted position of its
            Friedel mate, 1/Angstroms. None (default) uses `k_tol`.
        min_intensity : float, default=0.0
            Rows with a smaller summed intensity are dropped.
        num_rows : int, default=4
            Maximum number of rows kept per position.
        require_pair : bool, default=True
            Keep only rows seen at both +g and -g. The local Fourier transform
            of a real image always has both; for 4DSTEM data, where one side
            of a row can be weak, set False to keep single peaks as rows.

        Returns
        -------
        SystematicRowMap
            Self, with the per-row arrays :attr:`k`, :attr:`angle_deg`,
            :attr:`intensity`, :attr:`phase`, :attr:`ring` and
            :attr:`ambiguous` of shape ``(scan_row, scan_col, num_rows)``.
            Empty slots have phase -1, ring -1 and intensity 0.
        """
        pair_tol = float(k_tol if pair_tol is None else pair_tol)
        R, C = self.shape
        M = int(num_rows)
        k = np.full((R, C, M), np.nan)
        ang = np.full((R, C, M), np.nan)
        inten = np.zeros((R, C, M))
        phase = np.full((R, C, M), -1, dtype=np.int16)
        ring = np.full((R, C, M), -1, dtype=np.int16)
        ambiguous = np.zeros((R, C, M), dtype=bool)

        flat = self.peaks.select_fields("qx", "qy", "intensity").numpy().astype(np.float64)
        counts = np.asarray(self.peaks.row_counts(), dtype=int)
        cells = np.split(flat, np.cumsum(counts)[:-1])
        for n, data in enumerate(cells):
            if data.shape[0] == 0:
                continue
            rows = _pair_rows(data, self.k_min, pair_tol, require_pair)
            if rows.shape[0] == 0:
                continue
            rows = rows[rows[:, 2] >= min_intensity]
            rows = rows[np.argsort(-rows[:, 2])][:M]
            r, c = divmod(n, C)
            m = rows.shape[0]
            kk = np.hypot(rows[:, 0], rows[:, 1])
            k[r, c, :m] = kk
            ang[r, c, :m] = np.rad2deg(np.arctan2(rows[:, 1], rows[:, 0])) % 180.0
            inten[r, c, :m] = rows[:, 2]
            if self.ring_k.size:
                dist = np.abs(kk[:, None] - self.ring_k[None, :])
                order = np.argsort(dist, axis=1)
                best = order[:, 0]
                hit = dist[np.arange(m), best] <= k_tol
                ring[r, c, :m] = np.where(hit, best, -1)
                phase[r, c, :m] = np.where(hit, self.ring_phase[best], -1)
                if self.ring_k.size > 1:
                    second = dist[np.arange(m), order[:, 1]]
                    ambiguous[r, c, :m] = hit & (second <= k_tol)

        self.k, self.angle_deg, self.intensity = k, ang, inten
        self.phase, self.ring, self.ambiguous = phase, ring, ambiguous
        self.metadata["index"] = dict(
            k_tol=float(k_tol),
            pair_tol=pair_tol,
            min_intensity=float(min_intensity),
            num_rows=M,
            require_pair=bool(require_pair),
        )
        return self

    def _require_index(self) -> None:
        if self.phase is None:
            raise RuntimeError("Run index() first.")

    def _phase_index(self, phase) -> int:
        if isinstance(phase, str):
            return self.names.index(phase)
        return int(phase)

    def _ring_indices(self, phase) -> list[int]:
        p = self._phase_index(phase)
        return [i for i in range(len(self.ring_k)) if self.ring_phase[i] == p]

    def _ring_index(self, phase, ring) -> int:
        """Ring table index from a family label such as "(200)" or an index."""
        if isinstance(ring, str):
            label = ring if ring.startswith("(") else f"({ring})"
            for i in self._ring_indices(phase):
                if self.ring_label[i] == label:
                    return i
            raise ValueError(f"No ring {ring} for {self.names[self._phase_index(phase)]}.")
        return int(ring)

    def _panels(self, phase, ring) -> list[tuple[int, int | None, str]]:
        """(phase index, ring index or None, title) of every panel of a plot."""
        if phase is None:
            phases = list(range(len(self.crystals)))
        elif isinstance(phase, (str, int)):
            phases = [self._phase_index(phase)]
        else:
            phases = [self._phase_index(p) for p in phase]
        panels = []
        for p in phases:
            name = self.names[p]
            if ring is None:
                panels.append((p, None, f"{name}, all reflections"))
                continue
            if isinstance(ring, str) and ring in ("strongest", "all"):
                sig = {
                    g: float((self.intensity * self._row_mask(p, g)).sum())
                    for g in self._ring_indices(p)
                }
                found = [g for g in sig if sig[g] > 0]
                if not found:
                    panels.append((p, None, f"{name}, not found"))
                    continue
                rings = [max(found, key=sig.get)] if ring == "strongest" else found
            else:
                rings = [self._ring_index(p, ring)]
            panels += [(p, g, f"{name} {self.ring_label[g]}") for g in rings]
        return panels

    def _row_mask(self, phase=None, ring=None, include_ambiguous: bool = False) -> NDArray:
        """Rows of a phase (and ring), as a (R, C, M) boolean mask."""
        self._require_index()
        if phase is None:
            m = self.phase >= 0
        else:
            m = self.phase == self._phase_index(phase)
        if ring is not None:
            m &= self.ring == self._ring_index(phase, ring)
        if not include_ambiguous:
            m &= ~self.ambiguous
        return m

    def phase_signal(self, include_ambiguous: bool = False) -> NDArray:
        """``(scan_row, scan_col, num_phases)`` summed row intensity of every phase.

        Parameters
        ----------
        include_ambiguous : bool, default=False
            Count rows whose length matches rings of more than one phase.
        """
        self._require_index()
        out = np.zeros(self.shape + (len(self.crystals),))
        for p in range(len(self.crystals)):
            m = self._row_mask(p, include_ambiguous=include_ambiguous)
            out[..., p] = (self.intensity * m).sum(axis=-1)
        return out

    def phase_index(self, include_ambiguous: bool = False) -> NDArray:
        """``(scan_row, scan_col)`` phase with the largest row intensity, -1 for none."""
        sig = self.phase_signal(include_ambiguous)
        out = np.argmax(sig, axis=-1).astype(int)
        out[sig.max(axis=-1) <= 0] = -1
        return out

    def phase_fractions(self) -> dict[str, float]:
        """Fraction of the indexed positions assigned to each phase."""
        idx = self.phase_index()
        n = max(int((idx >= 0).sum()), 1)
        return {name: float((idx == p).sum()) / n for p, name in enumerate(self.names)}

    def in_plane_angle(self, phase, ring=None) -> tuple[NDArray, NDArray]:
        """In-plane angle of g of the strongest row of a phase at every position.

        Parameters
        ----------
        phase : str or int
            Crystal name or index.
        ring : str or int, optional
            Restrict to one reflection family, e.g. "200". None (default)
            takes the strongest row of any family.

        Returns
        -------
        angle_deg : np.ndarray
            ``(scan_row, scan_col)`` angle in [0, 180) degrees, NaN where the
            phase has no row.
        intensity : np.ndarray
            ``(scan_row, scan_col)`` intensity of that row, 0 where none.
        """
        m = self._row_mask(phase, ring)
        w = np.where(m, self.intensity, -np.inf)
        j = np.argmax(w, axis=-1)
        has = np.isfinite(np.take_along_axis(w, j[..., None], axis=-1)[..., 0])
        angle = np.take_along_axis(self.angle_deg, j[..., None], axis=-1)[..., 0]
        inten = np.take_along_axis(self.intensity, j[..., None], axis=-1)[..., 0]
        return np.where(has, angle, np.nan), np.where(has, inten, 0.0)

    def num_directions(self, min_angle_deg: float = 15.0) -> NDArray:
        """``(scan_row, scan_col)`` number of distinct row directions at each position.

        Rows within `min_angle_deg` of a stronger row count as the same
        direction. Positions with two or more directions carry enough
        information to determine a full orientation.
        """
        self._require_index()
        out = np.zeros(self.shape, dtype=int)
        for r, c in zip(*np.nonzero(self.intensity[..., 0] > 0)):
            dirs: list[float] = []
            for a in self.angle_deg[r, c][self.intensity[r, c] > 0]:
                if all(_angle_diff(a, b) > min_angle_deg for b in dirs):
                    dirs.append(a)
            out[r, c] = len(dirs)
        return out

    @property
    def scan_scalebar(self) -> dict | None:
        """Real-space scale bar of the scan, carried by the peaks."""
        return scan_scalebar(getattr(self.peaks, "metadata", {}) or {})

    def _shade(self, values: NDArray, shade_range, gamma: float) -> NDArray:
        """Brightness in [0, 1] from row intensities."""
        v = np.asarray(values, dtype=float)
        if isinstance(shade_range, str):
            pos = v[v > 0]
            # the median signal is full brightness, so half the positions
            # saturate and weak fringes fade
            hi = float(np.median(pos)) if pos.size else 1.0
            shade_range = (0.0, hi)
        lo, hi = shade_range
        return np.clip((v - lo) / max(hi - lo, 1e-12), 0.0, 1.0) ** gamma

    def _finish_map(self, ax, rgb, title, scalebar) -> None:
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(title)
        ax.set_axis_off()
        scalebar = self.scan_scalebar if scalebar == "auto" else scalebar
        if scalebar:
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

    def plot_phase(
        self,
        shade_range="auto",
        shade_gamma: float = 1.0,
        phase_colors=None,
        scalebar="auto",
        figax=None,
        figsize: tuple[float, float] = (7, 6),
    ):
        """Map of the phase with the largest row intensity at each position.

        Color gives the phase, brightness the summed row intensity of that
        phase. Positions without an indexed row are black.

        Parameters
        ----------
        shade_range : tuple or "auto", default="auto"
            Intensities mapped to black ... full color; "auto" uses 0 to the
            median over the positions with indexed fringes.
        shade_gamma : float, default=1.0
            Exponent applied to the brightness.
        phase_colors : sequence, optional
            One color per crystal; defaults to the quantEM phase palette.
        scalebar : dict, "auto" or None, default="auto"
            Real-space scale bar; "auto" takes the scan step carried by the
            peaks.
        figax : tuple, optional
            Existing (fig, ax).
        figsize : tuple, default=(7, 6)
            Figure size in inches.

        Returns
        -------
        fig, ax
        """
        colors = phase_color_cycle(len(self.crystals), phase_colors)
        sig = self.phase_signal()
        idx = self.phase_index()
        shade = self._shade(sig.max(axis=-1), shade_range, shade_gamma)
        rgb = np.where(idx[..., None] >= 0, colors[np.maximum(idx, 0)], 0.0) * shade[..., None]

        fig, ax = plt.subplots(figsize=figsize) if figax is None else figax
        self._finish_map(ax, rgb, "phase", scalebar)
        frac = self.phase_fractions()
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=colors[p], label=f"{n} ({100 * frac[n]:.0f}%)")
            for p, n in enumerate(self.names)
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(handles),
            frameon=False,
        )
        return fig, ax

    def plot_orientation(
        self,
        phase=None,
        ring="strongest",
        angle_range: tuple[float, float] | None = None,
        shade_range="auto",
        shade_gamma: float = 1.0,
        scalebar="auto",
        legend: bool = True,
        axsize: tuple[float, float] = (6, 6),
    ):
        """Maps of the in-plane angle of g, one per phase and reflection family.

        Hue gives the angle of the strongest row of the phase and family,
        brightness its intensity. Each title names the phase and the
        reflection family. By default the hue runs once around the color wheel over
        [0, 180) degrees, so every angle has its own color; for a textured
        sample, `angle_range` spreads the colors over a narrower window.

        Parameters
        ----------
        phase : str, int or list, optional
            Phase(s) to show. None (default) shows every phase.
        ring : str, int or None, default="strongest"
            Reflection family of each map. "strongest" takes the family with
            the largest summed row intensity of each phase, "all" draws one
            map per family with indexed rows, a label such as "200" or a ring
            index selects that family, and None takes the strongest row of
            any family.
        angle_range : tuple, optional
            ``(low, high)`` angles in degrees mapped to the ends of the
            "turbo" colormap. Angles are wrapped modulo 180 into the half
            turn centered on the window, and values outside it are clipped.
            None (default) uses the cyclic color wheel over [0, 180).
        shade_range : tuple or "auto", default="auto"
            Intensities mapped to black ... full color, per map; "auto" uses 0
            to the median over the positions with indexed fringes.
        shade_gamma : float, default=1.0
            Exponent applied to the brightness.
        scalebar : dict, "auto" or None, default="auto"
            Real-space scale bar.
        legend : bool, default=True
            Draw the color wheel of the angle.
        axsize : tuple, default=(6, 6)
            Size of each map in inches.

        Returns
        -------
        fig, axs
        """
        panels = self._panels(phase, ring)
        fig, axs = plt.subplots(
            1, len(panels), figsize=(axsize[0] * len(panels), axsize[1]), squeeze=False
        )
        for ax, (p, g, title) in zip(axs[0], panels):
            angle, inten = self.in_plane_angle(p, g)
            rgb = _angle_rgb(angle, angle_range, self._shade(inten, shade_range, shade_gamma))
            self._finish_map(ax, rgb, title, scalebar)
        if legend:
            fig.subplots_adjust(right=0.88)
            if angle_range is None:
                _angle_legend(fig.add_axes((0.89, 0.7, 0.1, 0.2)))
            else:
                cax = fig.add_axes((0.9, 0.25, 0.015, 0.5))
                lo, hi = angle_range
                t = np.linspace(lo, hi, 256)
                cax.imshow(
                    _angle_rgb(t, angle_range)[::-1, None], aspect="auto", extent=(0, 1, lo, hi)
                )
                cax.set_xticks([])
                cax.yaxis.tick_right()
                cax.set_title("g angle", fontsize=9)
        return fig, axs

    def plot_angle_distribution(
        self,
        phase=None,
        ring="strongest",
        angle_range: tuple[float, float] | None = None,
        bin_deg: float = 1.0,
        weight: str = "intensity",
        axsize: tuple[float, float] = (5, 3),
    ):
        """Histograms of the in-plane angle of g, one per phase and reflection family.

        The bars take the colors of :meth:`plot_orientation` with the same
        `angle_range`, so each histogram also serves as the color legend of
        the matching map. A narrow distribution is a fiber texture: the
        crystals share the direction of g in the image plane, and their
        rotation about g is not measured.

        Parameters
        ----------
        phase : str, int or list, optional
            Phase(s) to show. None (default) shows every phase.
        ring : str, int or None, default="strongest"
            Reflection family of each histogram, as in
            :meth:`plot_orientation`; None pools every family of a phase.
        angle_range : tuple, optional
            ``(low, high)`` window of the angle axis in degrees, with angles
            wrapped modulo 180 into the half turn centered on it. None
            (default) shows [0, 180).
        bin_deg : float, default=1.0
            Histogram bin width in degrees.
        weight : {"intensity", "count"}, default="intensity"
            Weight each row by its intensity, or count rows alike.
        axsize : tuple, default=(5, 3)
            Size of each panel in inches.

        Returns
        -------
        fig, axs
        """
        panels = self._panels(phase, ring)
        fig, axs = plt.subplots(
            1, len(panels), figsize=(axsize[0] * len(panels), axsize[1]), squeeze=False
        )
        lo, hi = (0.0, 180.0) if angle_range is None else (float(v) for v in angle_range)
        mid = 0.5 * (lo + hi)
        edges = np.arange(lo, hi + 0.5 * bin_deg, bin_deg)
        centers = 0.5 * (edges[:-1] + edges[1:])
        for ax, (p, g, title) in zip(axs[0], panels):
            m = self._row_mask(p, g)
            a = self.angle_deg[m]
            if angle_range is not None:
                a = (a - mid + 90.0) % 180.0 - 90.0 + mid
            w = self.intensity[m] if weight == "intensity" else np.ones(a.size)
            h, _ = np.histogram(a, edges, weights=w)
            h = h / max(w.sum(), 1e-12)
            ax.bar(centers, h, width=bin_deg, color=_angle_rgb(centers, angle_range), lw=0)
            ax.set_xlim(lo, hi)
            ax.set_xlabel("g angle (degrees)")
            ax.set_ylabel("fraction of count" if weight == "count" else "fraction of intensity")
            ax.set_title(title)
        fig.tight_layout()
        return fig, axs

    def plot_rings(
        self,
        bin_size: float = 0.002,
        figsize: tuple[float, float] = (10, 4),
    ):
        """Histogram of the row lengths under the candidate rings.

        Checks the calibration and the ring assignment: every peak of the
        histogram should sit on a ring, and rings of different phases closer
        than twice `k_tol` are ambiguous.

        Parameters
        ----------
        bin_size : float, default=0.002
            Histogram bin width, 1/Angstroms.
        figsize : tuple, default=(10, 4)
            Figure size in inches.

        Returns
        -------
        fig, ax
        """
        self._require_index()
        colors = phase_color_cycle(len(self.crystals))
        valid = self.intensity > 0
        k = self.k[valid]
        edges = np.arange(self.k_min, self.k_max + bin_size, bin_size)
        fig, ax = plt.subplots(figsize=figsize)
        h_all, _ = np.histogram(k, edges)
        ax.stairs(h_all, edges, color="0.6", fill=True, label="unassigned")
        bottom = np.zeros_like(h_all, dtype=float)
        for p, name in enumerate(self.names):
            h, _ = np.histogram(self.k[valid & (self.phase == p) & ~self.ambiguous], edges)
            ax.stairs(bottom + h, edges, baseline=bottom, color=colors[p], fill=True, label=name)
            bottom = bottom + h
        if self.ambiguous.any():
            h, _ = np.histogram(self.k[valid & self.ambiguous], edges)
            ax.stairs(bottom + h, edges, baseline=bottom, color="k", fill=True, label="ambiguous")
        k_tol = self.metadata["index"]["k_tol"]
        top = max(h_all.max(), 1)
        for i, kr in enumerate(self.ring_k):
            c = colors[self.ring_phase[i]]
            ax.axvspan(kr - k_tol, kr + k_tol, color=c, alpha=0.15, lw=0)
            ax.axvline(kr, color=c, lw=1)
            ax.text(
                kr,
                1.12 * top,
                f"{self.names[self.ring_phase[i]]} {self.ring_label[i]}",
                color="k",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
                clip_on=False,
            )
        ax.set_xlim(self.k_min, self.k_max)
        ax.set_ylim(0, 1.1 * top)
        ax.set_xlabel("|g| (1/Å)")
        ax.set_ylabel("count")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        fig.tight_layout()
        return fig, ax

    def __repr__(self) -> str:
        head = f"SystematicRowMap({self.shape[0]} x {self.shape[1]}, phases {self.names}"
        if self.phase is None:
            return head + ")"
        n = int((self.intensity > 0).sum())
        assigned = int((self.phase >= 0).sum())
        amb = int(self.ambiguous.sum())
        return head + f", {n} rows, {assigned} indexed, {amb} ambiguous)"


def _ring_table(crystals, k_min: float, k_max: float, min_ring_intensity: float):
    """Ring radius, phase index and family label of every candidate ring."""
    ks, phases, labels = [], [], []
    for p, xtl in enumerate(crystals):
        if getattr(xtl, "g_len", None) is None:
            raise ValueError("Calculate the structure factors of every crystal first.")
        g = xtl.g_len.numpy()
        f = xtl.struct_factors_int.numpy()
        hkl = xtl.hkl.numpy()
        nz = g > 1e-6
        keep = nz & (g >= k_min) & (g <= k_max) & (f >= min_ring_intensity * f[nz].max())
        radii = np.round(g[keep], 4)
        for kr in np.unique(radii):
            members = np.nonzero(keep)[0][radii == kr]
            # label each family by its largest member with non-negative
            # indices, e.g. (200) rather than (0-20)
            best = max(members, key=lambda i: (bool((hkl[i] >= 0).all()), tuple(hkl[i])))
            ks.append(float(kr))
            phases.append(p)
            labels.append("(" + "".join(str(int(v)) for v in hkl[best]) + ")")
    order = np.argsort(ks)
    return (
        np.asarray(ks)[order],
        np.asarray(phases, dtype=int)[order],
        [labels[i] for i in order],
    )


def _pair_rows(data: NDArray, k_min: float, pair_tol: float, require_pair: bool) -> NDArray:
    """Friedel-paired rows (g_row, g_col, intensity) of one pattern."""
    q = data[:, :2]
    inten = data[:, 2]
    keep = np.hypot(q[:, 0], q[:, 1]) >= k_min
    q, inten = q[keep], inten[keep]
    order = np.argsort(-inten)
    used = np.zeros(q.shape[0], dtype=bool)
    rows = []
    for i in order:
        if used[i]:
            continue
        used[i] = True
        d = np.hypot(*(q + q[i]).T)
        d[used] = np.inf
        j = int(np.argmin(d)) if d.size else -1
        if j >= 0 and d[j] <= pair_tol:
            used[j] = True
            g = 0.5 * (q[i] - q[j])
            rows.append((g[0], g[1], inten[i] + inten[j]))
        elif not require_pair:
            rows.append((q[i, 0], q[i, 1], inten[i]))
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def _angle_diff(a: float, b: float) -> float:
    """Distance between two angles modulo 180 degrees."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _angle_rgb(
    angle_deg: NDArray,
    angle_range: tuple[float, float] | None = None,
    brightness: NDArray | None = None,
) -> NDArray:
    """Perceptually uniform colors of in-plane angles; NaN is black.

    The colors come from :func:`array_to_rgba`, which works in the CIECAM02
    JCh space: the angle sets the hue at constant lightness, and `brightness`
    sets the lightness, so every angle is equally bright and shading darkens
    every hue alike. Without a range the hue runs once around the color
    circle over [0, 180) degrees. With a range, angles are wrapped into the
    half turn centered on it and mapped linearly onto 300 degrees of the hue
    circle, so the two ends keep different colors, clipped beyond the ends.
    """
    angle_deg = np.asarray(angle_deg, dtype=float)
    finite = np.isfinite(angle_deg)
    a = np.nan_to_num(angle_deg)
    if angle_range is None:
        phase = np.deg2rad(2.0 * a)
    else:
        lo, hi = (float(v) for v in angle_range)
        mid = 0.5 * (lo + hi)
        a = (a - mid + 90.0) % 180.0 - 90.0 + mid
        t = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
        phase = np.deg2rad(300.0 * t - 150.0)
    amp = np.ones_like(a) if brightness is None else np.asarray(brightness, dtype=float)
    amp = np.where(finite, np.clip(amp, 0.0, 1.0), 0.0)
    # array_to_rgba stacks 2D images; run any shape through it as a column
    rgba = array_to_rgba(amp.reshape(-1, 1), phase.reshape(-1, 1))
    return rgba[..., :3].reshape(a.shape + (3,))


def _angle_legend(ax) -> None:
    """Half-wheel legend of the in-plane angle hue, in the image frame."""
    n = 181
    t = np.linspace(0, 2 * np.pi, 2 * n)
    r = np.linspace(0.4, 1.0, 24)
    T, Rr = np.meshgrid(t, r)
    # image frame: angle from the row axis (down) toward the column axis (right)
    x, y = Rr * np.sin(T), -Rr * np.cos(T)
    color = _angle_rgb(np.rad2deg(T))
    ax.scatter(x.ravel(), y.ravel(), c=color.reshape(-1, 3), s=2, marker="s", lw=0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("g angle", fontsize=9)
