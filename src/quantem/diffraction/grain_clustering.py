"""Signal-level grain clustering for 4D-STEM polymer orientation data.

This module assigns individual diffraction *signals* (Bragg arcs) to *grains* by a
global, order-independent procedure, as an alternative to greedy seed-and-grow
region growing.

Design (see the full proposal for rationale):

* The clustered unit is a **signal**, not a probe/pixel: a single detected peak at a
  probe position, carrying ``(pos, theta, r, intensity, window)``.  Because one signal
  carries exactly one label, "each signal belongs to at most one grain" holds by
  construction, and several grains may coexist at one probe (their signals get
  different labels).

* A **window** (radial range) is a hard, immutable *signal class* (backbone, lamellar,
  pi-pi, ...).  Clustering happens strictly within a window; ``r`` never drifts across
  a window boundary.

* Within a window, both **orientation** (circular, 180-deg period for 2-fold polymer
  texture) and **radius** are "smooth within a grain, a discontinuity splits grains".
  Two signals at the same orientation but distinct *quantized* radii are different
  grains.  A merge across a probe boundary is allowed only if *both* the orientation
  jump and the (relative) radius jump are below tolerance.

* Grains are spatially coherent because adjacency exists **only between neighbouring
  probes** -- never in pure feature space.  Hence identical orientation in
  *disconnected* regions stays separate, and a missing detection can be bridged by a
  larger ``neighbor_dist``.

Core algorithm (Stage A): build a region-adjacency graph over signals (nodes = signals,
edges = signal pairs at neighbouring probes), then agglomeratively merge the adjacent
region pair with the smallest *boundary discontinuity* (mean orientation/radius jump
across the shared boundary), stopping when no boundary is within tolerance.  This is
average-linkage on the spatial graph with a discontinuity stop: order-independent
(driven by the global minimum cost, not by traversal), chaining-resistant (a boundary
statistic, not a single lucky edge), tolerant of gentle orientation gradients (bent
grains), and it separates quantized-radius grains.

Only numpy + scipy + the standard library are required.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "SignalTable",
    "GrainInfo",
    "GrainResult",
    "extract_signals",
    "cluster_signals_into_grains",
    "refine_grains_crf",
    "circular_distance_deg",
    "orientation_to_rgb",
    "grain_rgb_overlay",
    "orientation_legend_image",
    "plot_grain_map",
]


# --------------------------------------------------------------------------------------
# data structures
# --------------------------------------------------------------------------------------
@dataclass
class SignalTable:
    """A flat table of detected signals across the probe grid.

    Attributes
    ----------
    pos : (N, 2) int array
        Probe position ``(rx, ry)`` of each signal.
    theta : (N,) float array
        Orientation angle in degrees, folded to ``[0, 180)``.
    r : (N,) float array
        Scattering-vector magnitude ``|q|`` of each signal.
    intensity : (N,) float array
        Peak intensity.
    window : (N,) int array
        Radial-window / signal-class id (immutable class label).
    map_shape : (Rx, Ry)
        Probe-grid shape.
    """

    pos: np.ndarray
    theta: np.ndarray
    r: np.ndarray
    intensity: np.ndarray
    window: np.ndarray
    map_shape: tuple

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=np.int64).reshape(-1, 2)
        self.theta = np.asarray(self.theta, dtype=np.float64).reshape(-1)
        self.r = np.asarray(self.r, dtype=np.float64).reshape(-1)
        self.intensity = np.asarray(self.intensity, dtype=np.float64).reshape(-1)
        self.window = np.asarray(self.window, dtype=np.int64).reshape(-1)
        n = self.pos.shape[0]
        if not (len(self.theta) == len(self.r) == len(self.intensity) == len(self.window) == n):
            raise ValueError("SignalTable field lengths are inconsistent")

    def __len__(self) -> int:
        return self.pos.shape[0]


@dataclass
class GrainInfo:
    label: int
    window: int
    signal_ids: np.ndarray
    n_signals: int
    theta_mean: float
    theta_std: float
    r_mean: float
    intensity_median: float
    centroid: tuple


@dataclass
class GrainResult:
    """Result of :func:`cluster_signals_into_grains`.

    ``labels`` (length N, -1 = outlier) is the authoritative output.  ``label_map`` is a
    convenience raster ``(num_windows, Rx, Ry)`` for visualisation; where a probe holds
    several same-window signals in different grains it keeps the highest-intensity one.
    """

    labels: np.ndarray
    n_grains: int
    grains: list
    label_map: np.ndarray
    params: dict = field(default_factory=dict)
    confidence: Optional[np.ndarray] = None  # (N,) max posterior, filled by Stage B
    margin: Optional[np.ndarray] = None      # (N,) energy gap top1-top2, filled by Stage B


# --------------------------------------------------------------------------------------
# orientation geometry
# --------------------------------------------------------------------------------------
def circular_distance_deg(a, b, period: float = 180.0):
    """Circular distance between angles (degrees), default 180-deg period (2-fold)."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % period
    return np.minimum(d, period - d)


# --------------------------------------------------------------------------------------
# ingestion adapter (matches make_orientation_histogram conventions)
# --------------------------------------------------------------------------------------
def extract_signals(
    bragg_peaks,
    radial_ranges,
    *,
    r_field: Optional[str] = None,
    theta_field: str = "theta",
    intensity_field: Optional[str] = None,
    flip_sign: bool = False,
    offset_deg: float = 0.0,
) -> SignalTable:
    """Build a :class:`SignalTable` from a ``BraggPeaksPolymer``-style object.

    Replicates the orientation convention of
    ``BraggPeaksPolymer.make_orientation_histogram`` (Karen's polar transform): the
    **stored polar** ``theta`` (radians) is used directly -- optionally sign-flipped
    (``flip_sign`` <-> ``orientation_flip_sign``), offset (``offset_deg`` <->
    ``orientation_offset_degrees``), then folded ``mod pi`` to ``[0, 180)`` degrees.
    Reading the stored ``theta`` inherits Karen's sign by construction rather than
    re-deriving it from ``qx``/``qy``.  ``r`` is the stored polar magnitude; radial
    windows gate on ``r**2`` against ``radial_ranges`` (each row ``[r_min, r_max]``).

    Two ingestion APIs are supported and auto-detected:

    * A real quantem ``Vector`` (``bp.polar_peaks`` / ``bp.peak_intensities``): per-cell
      1-D field arrays are read with ``vec.select_fields(field)[rx, ry].array[:, 0]``
      -- exactly the access ``make_orientation_histogram`` uses.  Default fields are then
      ``r_invA`` / ``theta`` / ``intensities``.
    * A legacy field-indexable container (``polar[field][rx, ry]`` -> 1-D array), used by
      the unit tests.  Default fields are then ``r`` / ``theta`` / ``intensity``.

    ``r_field`` / ``intensity_field`` default to ``None`` and resolve per API above;
    pass explicit names to override.

    Parameters
    ----------
    bragg_peaks : object
        Provides ``.polar_peaks`` (a ``Vector`` or field-indexable container with
        ``.shape == (Rx, Ry)``) and ``.peak_intensities``; or is itself such a
        container (then intensities are read from it too).
    """
    polar = getattr(bragg_peaks, "polar_peaks", bragg_peaks)
    inten_src = getattr(bragg_peaks, "peak_intensities", polar)

    # quantem Vector rejects string field indexing (needs select_fields); auto-detect it.
    is_vector = hasattr(polar, "select_fields")
    if r_field is None:
        r_field = "r_invA" if is_vector else "r"
    if intensity_field is None:
        intensity_field = "intensities" if is_vector else "intensity"

    radial_ranges = np.atleast_2d(np.asarray(radial_ranges, dtype=float))
    rr2 = radial_ranges ** 2
    Rx, Ry = polar.shape
    offset_rad = np.deg2rad(offset_deg)

    if is_vector:
        R_v = polar.select_fields(r_field)
        TH_v = polar.select_fields(theta_field)
        II_v = inten_src.select_fields(intensity_field)

        def r_cell(i, j):
            return np.asarray(R_v[i, j].array[:, 0], dtype=float)

        def th_cell(i, j):
            return np.asarray(TH_v[i, j].array[:, 0], dtype=float)

        def i_cell(i, j):
            return np.asarray(II_v[i, j].array[:, 0], dtype=float)
    else:
        R, TH, II = polar[r_field], polar[theta_field], inten_src[intensity_field]

        def _legacy(grid, i, j):
            a = grid[i, j]
            return np.empty(0) if a is None else np.asarray(a, dtype=float)

        def r_cell(i, j):
            return _legacy(R, i, j)

        def th_cell(i, j):
            return _legacy(TH, i, j)

        def i_cell(i, j):
            return _legacy(II, i, j)

    pos_l, th_l, r_l, i_l, w_l = [], [], [], [], []
    for rx in range(Rx):
        for ry in range(Ry):
            p_r = r_cell(rx, ry)
            if len(p_r) == 0:
                continue
            p_th = th_cell(rx, ry)
            inten = i_cell(rx, ry)
            r2 = p_r ** 2
            ang = -p_th if flip_sign else p_th            # do not mutate the source array
            ang = np.degrees(np.mod(ang + offset_rad, np.pi))  # -> [0, 180)
            for w, (lo2, hi2) in enumerate(rr2):
                sub = (r2 >= lo2) & (r2 < hi2)
                if not np.any(sub):
                    continue
                n = int(sub.sum())
                pos_l.append(np.column_stack([np.full(n, rx), np.full(n, ry)]))
                th_l.append(ang[sub])
                r_l.append(p_r[sub])
                i_l.append(inten[sub])
                w_l.append(np.full(n, w))

    if not pos_l:
        empty_i = np.zeros((0, 2), dtype=np.int64)
        empty_f = np.zeros((0,), dtype=float)
        return SignalTable(empty_i, empty_f, empty_f, empty_f,
                           empty_f.astype(np.int64), (Rx, Ry))

    return SignalTable(
        np.concatenate(pos_l, axis=0),
        np.concatenate(th_l),
        np.concatenate(r_l),
        np.concatenate(i_l),
        np.concatenate(w_l),
        (Rx, Ry),
    )


# --------------------------------------------------------------------------------------
# adjacency
# --------------------------------------------------------------------------------------
def _signal_edges(pos: np.ndarray, neighbor_dist: int):
    """All signal pairs whose probes are within Chebyshev ``neighbor_dist`` (excluding
    same-probe pairs).  Returns local index arrays ``(ii, jj)`` with ``ii < jj``.
    """
    n = pos.shape[0]
    if n < 2:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    tree = cKDTree(pos.astype(float))
    pairs = tree.query_pairs(r=neighbor_dist, p=np.inf, output_type="ndarray")
    if pairs.shape[0] == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    ii, jj = pairs[:, 0], pairs[:, 1]
    # drop same-probe pairs (different signals at the identical probe are not neighbours)
    diff = np.any(pos[ii] != pos[jj], axis=1)
    return ii[diff], jj[diff]


# --------------------------------------------------------------------------------------
# agglomerative boundary merge (Stage A core)
# --------------------------------------------------------------------------------------
def _key(a: int, b: int):
    return (a, b) if a < b else (b, a)


def _agglomerative_merge(
    n: int,
    ii: np.ndarray,
    jj: np.ndarray,
    d_theta: np.ndarray,
    d_r_rel: np.ndarray,
    d_i_rel: np.ndarray,
    theta_tol: float,
    r_tol_rel: float,
    intensity_tol_rel: float,
    probe_lin: np.ndarray,
    enforce_one_per_probe: bool,
):
    """Average-linkage agglomeration on the spatial signal graph with a discontinuity
    stop.  Returns a (n,) array of root ids (a flat clustering)."""
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    inv_theta = (1.0 / theta_tol) if np.isfinite(theta_tol) else 0.0
    inv_r = (1.0 / r_tol_rel) if np.isfinite(r_tol_rel) else 0.0
    inv_i = (1.0 / intensity_tol_rel) if np.isfinite(intensity_tol_rel) else 0.0

    def cost_of(stats) -> float:
        s_th, s_r, s_i, c = stats
        return max((s_th / c) * inv_theta, (s_r / c) * inv_r, (s_i / c) * inv_i)

    # region-adjacency graph: edge stats keyed by current root pair, neighbour sets
    edge_stats: dict = {}
    nbrs: list = [set() for _ in range(n)]
    for a, b, t, rr, di in zip(ii.tolist(), jj.tolist(), d_theta, d_r_rel, d_i_rel):
        k = _key(a, b)
        st = edge_stats.get(k)
        if st is None:
            edge_stats[k] = [float(t), float(rr), float(di), 1]
            nbrs[a].add(b)
            nbrs[b].add(a)
        else:  # parallel edges between the same singleton pair shouldn't happen, but be safe
            st[0] += float(t); st[1] += float(rr); st[2] += float(di); st[3] += 1

    probes = [{int(probe_lin[k])} for k in range(n)] if enforce_one_per_probe else None

    heap = []
    for (a, b), st in edge_stats.items():
        heapq.heappush(heap, (cost_of(st), a, b))

    while heap:
        cost, a, b = heapq.heappop(heap)
        if cost >= 1.0:
            break
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        k = _key(ra, rb)
        st = edge_stats.get(k)
        if st is None:
            continue  # no longer adjacent
        cur = cost_of(st)
        if cur > cost + 1e-12:  # stale: re-push with the up-to-date cost
            heapq.heappush(heap, (cur, ra, rb))
            continue
        if cur >= 1.0:
            continue

        # keep the region with the larger probe set as the survivor (small-to-large)
        if enforce_one_per_probe:
            if len(probes[rb]) > len(probes[ra]):
                ra, rb = rb, ra
                k = _key(ra, rb)
            small, large = (probes[rb], probes[ra]) if len(probes[rb]) <= len(probes[ra]) else (probes[ra], probes[rb])
            if not small.isdisjoint(large):
                # merging would put two signals of one probe in a single grain: forbid
                edge_stats.pop(k, None)
                nbrs[ra].discard(rb)
                nbrs[rb].discard(ra)
                continue

        # merge rb -> ra
        parent[rb] = ra
        edge_stats.pop(k, None)
        nbrs[ra].discard(rb)
        nbrs[rb].discard(ra)
        if enforce_one_per_probe:
            probes[ra] |= probes[rb]
            probes[rb] = None

        for c in list(nbrs[rb]):
            kbc = _key(rb, c)
            stbc = edge_stats.pop(kbc)
            nbrs[c].discard(rb)
            if c == ra:
                continue
            kac = _key(ra, c)
            stac = edge_stats.get(kac)
            if stac is None:
                edge_stats[kac] = stbc
                nbrs[ra].add(c)
                nbrs[c].add(ra)
            else:
                stac[0] += stbc[0]; stac[1] += stbc[1]; stac[2] += stbc[2]; stac[3] += stbc[3]
            heapq.heappush(heap, (cost_of(edge_stats[kac]), ra, c))
        nbrs[rb] = set()

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    return roots


# --------------------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------------------
def cluster_signals_into_grains(
    signals: SignalTable,
    *,
    theta_tol_deg: float = 10.0,
    r_tol_rel: float = 0.10,
    intensity_tol_rel: float = np.inf,
    neighbor_dist: int = 1,
    area_min: int = 3,
    enforce_one_per_probe: bool = True,
) -> GrainResult:
    """Cluster signals into grains, strictly within each radial window.

    Parameters
    ----------
    signals : SignalTable
    theta_tol_deg : float
        Max orientation discontinuity (deg, circular) across a within-grain boundary.
        Tie to the histogram angular resolution (a few x ``sigma_theta``).  ``inf``
        disables orientation gating.
    r_tol_rel : float
        Max *relative* radius discontinuity ``|dr| / r_mean`` across a within-grain
        boundary.  Set below the inter-peak (quantized) radial gap so distinct radii in
        the same window separate into different grains.  ``inf`` disables radius gating.
    intensity_tol_rel : float
        Optional relative intensity discontinuity tolerance.  Default ``inf`` (off):
        intensity varies within real grains, so it is not gated.
    neighbor_dist : int
        Chebyshev probe radius for adjacency.  1 = 8-connectivity; 2 bridges single
        missing detections so a dropout does not fragment a grain.
    area_min : int
        Grains with fewer than this many signals become outliers (label -1).
    enforce_one_per_probe : bool
        Forbid a grain from containing two signals at the same probe.

    Returns
    -------
    GrainResult
    """
    N = len(signals)
    Rx, Ry = signals.map_shape
    labels = np.full(N, -1, dtype=np.int64)
    theta_tol = float(theta_tol_deg)
    next_label = 0
    windows = np.unique(signals.window)

    for w in windows:
        idx = np.nonzero(signals.window == w)[0]
        if idx.size == 0:
            continue
        pos = signals.pos[idx]
        theta = signals.theta[idx]
        r = signals.r[idx]
        inten = signals.intensity[idx]
        probe_lin = pos[:, 0].astype(np.int64) * Ry + pos[:, 1].astype(np.int64)

        ii, jj = _signal_edges(pos, neighbor_dist)
        if ii.size:
            d_theta = circular_distance_deg(theta[ii], theta[jj])
            rbar = 0.5 * (r[ii] + r[jj])
            d_r_rel = np.abs(r[ii] - r[jj]) / np.where(rbar > 0, rbar, 1.0)
            ibar = 0.5 * (inten[ii] + inten[jj])
            d_i_rel = np.abs(inten[ii] - inten[jj]) / np.where(ibar > 0, ibar, 1.0)
        else:
            d_theta = d_r_rel = d_i_rel = np.empty(0)

        roots = _agglomerative_merge(
            idx.size, ii, jj, d_theta, d_r_rel, d_i_rel,
            theta_tol, float(r_tol_rel), float(intensity_tol_rel),
            probe_lin, enforce_one_per_probe,
        )

        # area filter + contiguous relabelling (per window), offset into the global space
        uniq, counts = np.unique(roots, return_counts=True)
        keep = {root: counts[i] >= area_min for i, root in enumerate(uniq)}
        remap = {}
        for root in uniq:
            if keep[root]:
                remap[root] = next_label
                next_label += 1
        for local_i, root in enumerate(roots):
            if keep[root]:
                labels[idx[local_i]] = remap[root]

    n_grains = next_label
    grains = _summarize(signals, labels, n_grains)
    label_map = _rasterize(signals, labels, len(windows))

    return GrainResult(
        labels=labels,
        n_grains=n_grains,
        grains=grains,
        label_map=label_map,
        params=dict(
            theta_tol_deg=theta_tol_deg,
            r_tol_rel=r_tol_rel,
            intensity_tol_rel=intensity_tol_rel,
            neighbor_dist=neighbor_dist,
            area_min=area_min,
            enforce_one_per_probe=enforce_one_per_probe,
        ),
    )


# --------------------------------------------------------------------------------------
# postprocessing helpers
# --------------------------------------------------------------------------------------
def _summarize(signals: SignalTable, labels: np.ndarray, n_grains: int) -> list:
    grains = []
    for g in range(n_grains):
        sids = np.nonzero(labels == g)[0]
        if sids.size == 0:
            continue
        th = signals.theta[sids]
        # circular mean / std on the doubled angle (2-fold)
        ang2 = np.deg2rad(2.0 * th)
        c, s = np.cos(ang2).mean(), np.sin(ang2).mean()
        theta_mean = (np.rad2deg(np.arctan2(s, c)) / 2.0) % 180.0
        R = np.hypot(c, s)
        theta_std = np.rad2deg(np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-12))))) / 2.0
        pos = signals.pos[sids]
        grains.append(
            GrainInfo(
                label=g,
                window=int(signals.window[sids[0]]),
                signal_ids=sids,
                n_signals=int(sids.size),
                theta_mean=float(theta_mean),
                theta_std=float(theta_std),
                r_mean=float(signals.r[sids].mean()),
                intensity_median=float(np.median(signals.intensity[sids])),
                centroid=(float(pos[:, 0].mean()), float(pos[:, 1].mean())),
            )
        )
    return grains


def _rasterize(signals: SignalTable, labels: np.ndarray, num_windows: int) -> np.ndarray:
    Rx, Ry = signals.map_shape
    label_map = np.full((num_windows, Rx, Ry), -1, dtype=np.int64)
    order = np.argsort(signals.intensity, kind="stable")  # higher intensity overwrites
    for i in order:
        g = labels[i]
        if g < 0:
            continue
        w = int(signals.window[i])
        rx, ry = signals.pos[i]
        label_map[w, rx, ry] = g
    return label_map


def _apply_area_min(labels: np.ndarray, area_min: int) -> np.ndarray:
    """Dissolve grains with < area_min signals to -1 and relabel 0..K-1 contiguously."""
    out = np.full_like(labels, -1)
    valid = labels >= 0
    if not np.any(valid):
        return out
    uniq, counts = np.unique(labels[valid], return_counts=True)
    keep = uniq[counts >= area_min]
    for new, old in enumerate(np.sort(keep)):
        out[labels == old] = new
    return out


# --------------------------------------------------------------------------------------
# Stage B: CRF + EM refinement (boundary precision, outliers, soft confidence)
# --------------------------------------------------------------------------------------
def _circular_mean_deg(theta_deg: np.ndarray, weights: np.ndarray) -> float:
    """Weighted circular mean on the doubled angle (2-fold), returned in [0, 180)."""
    a = np.deg2rad(2.0 * np.asarray(theta_deg, float))
    w = np.asarray(weights, float)
    c = float(np.sum(w * np.cos(a)))
    s = float(np.sum(w * np.sin(a)))
    return (np.rad2deg(np.arctan2(s, c)) / 2.0) % 180.0


def refine_grains_crf(
    signals: SignalTable,
    init,
    *,
    theta_sigma_deg: float = 8.0,
    r_sigma_rel: float = 0.06,
    lam: float = 0.5,
    model_radius: float = 3.0,
    neighbor_dist: int = 1,
    outlier_energy: float = 6.0,
    max_iter: int = 8,
    enforce_one_per_probe: bool = True,
    area_min: int = 1,
) -> GrainResult:
    """Refine a Stage-A clustering by minimising a contrast-sensitive CRF energy with
    ICM, EM-style: each grain's orientation model is a *local* weighted circular mean of
    its nearby members (recomputed from current labels), so the energy is

        E(x) = sum_s U(x_s) + lam * sum_{(i,j) in nbrs} w_ij * [x_i != x_j]
        U(s, g)       = dtheta(theta_s, theta_g_local(p_s))^2 / (2 sigma_theta^2)
                        + (drel r)^2 / (2 sigma_r^2)
        U(s, outlier) = outlier_energy
        w_ij          = exp(-dtheta_ij^2/2sigma_theta^2 - drel_ij^2/2sigma_r^2)

    Because the grain model is evaluated *at the signal's location* (not a global mean),
    continuously-flowing / bent grains are preserved exactly as in Stage A.  A signal's
    candidate labels are only grains present within ``model_radius`` probes, so it can
    never jump to a spatially-distant grain; ``enforce_one_per_probe`` is upheld.

    Returns a :class:`GrainResult` with ``confidence`` (max posterior) and ``margin``
    (top1-top2 energy gap) populated.  ``init`` may be a Stage-A ``GrainResult`` or an
    ``(N,)`` label array; for best results run Stage A with ``area_min=1`` so no grains
    are dissolved before refinement, then let this apply the final ``area_min``.
    """
    init_labels = init.labels if isinstance(init, GrainResult) else np.asarray(init)
    labels = init_labels.astype(np.int64).copy()
    N = len(signals)
    Rx, Ry = signals.map_shape
    confidence = np.zeros(N)
    margin = np.full(N, np.inf)

    two_sig_th2 = 2.0 * theta_sigma_deg ** 2
    two_sig_r2 = 2.0 * r_sigma_rel ** 2
    sig_d = max(model_radius / 2.0, 1e-6)

    for w in np.unique(signals.window):
        idx = np.nonzero(signals.window == w)[0]
        if idx.size == 0:
            continue
        pos = signals.pos[idx].astype(float)
        theta = signals.theta[idx]
        r = signals.r[idx]
        probe_lin = (signals.pos[idx, 0] * Ry + signals.pos[idx, 1]).astype(np.int64)
        lab = labels[idx].copy()
        n = idx.size

        # contrast-sensitive pairwise adjacency
        ii, jj = _signal_edges(signals.pos[idx], neighbor_dist)
        adj = [[] for _ in range(n)]
        if ii.size:
            dth = circular_distance_deg(theta[ii], theta[jj])
            rbar = 0.5 * (r[ii] + r[jj])
            drr = np.abs(r[ii] - r[jj]) / np.where(rbar > 0, rbar, 1.0)
            wij = np.exp(-(dth ** 2) / two_sig_th2 - (drr ** 2) / two_sig_r2)
            for a, b, wv in zip(ii.tolist(), jj.tolist(), wij.tolist()):
                adj[a].append((b, wv))
                adj[b].append((a, wv))

        # local-model neighbours within model_radius (Chebyshev) + Gaussian weights
        tree = cKDTree(pos)
        ball = tree.query_ball_point(pos, r=model_radius, p=np.inf)
        model_nbr, model_w = [], []
        for i in range(n):
            nb = np.array([k for k in ball[i] if k != i], dtype=np.int64)
            model_nbr.append(nb)
            if nb.size:
                d = np.linalg.norm(pos[nb] - pos[i], axis=1)
                model_w.append(np.exp(-(d ** 2) / (2.0 * sig_d ** 2)))
            else:
                model_w.append(np.zeros(0))

        probe_members = {}
        for i in range(n):
            probe_members.setdefault(int(probe_lin[i]), []).append(i)

        def pair_cost(i, g):
            c = 0.0
            for (j, wv) in adj[i]:
                if lab[j] != g:
                    c += wv
            return lam * c

        def unary(i, g, nb, nbl, nbw):
            mask = nbl == g
            if not np.any(mask):
                return None
            th_g = _circular_mean_deg(theta[nb[mask]], nbw[mask])
            wsum = float(nbw[mask].sum())
            r_g = float(np.sum(r[nb[mask]] * nbw[mask]) / wsum) if wsum > 0 else float(r[nb[mask]].mean())
            dth = float(circular_distance_deg(theta[i], th_g))
            drr = abs(r[i] - r_g) / r_g if r_g > 0 else 0.0
            return (dth ** 2) / two_sig_th2 + (drr ** 2) / two_sig_r2

        # ICM sweeps to convergence (EM model recomputed implicitly each evaluation)
        for _ in range(max_iter):
            changed = 0
            for i in range(n):
                nb = model_nbr[i]
                if nb.size == 0:
                    new = -1
                else:
                    nbl = lab[nb]
                    nbw = model_w[i]
                    cands = {int(x) for x in nbl if x >= 0}
                    forbidden = set()
                    if enforce_one_per_probe:
                        for m in probe_members[int(probe_lin[i])]:
                            if m != i and lab[m] >= 0:
                                forbidden.add(int(lab[m]))
                    best_lab, best_E = -1, outlier_energy + pair_cost(i, -1)
                    for g in cands:
                        if g in forbidden:
                            continue
                        U = unary(i, g, nb, nbl, nbw)
                        if U is None:
                            continue
                        E = U + pair_cost(i, g)
                        if E < best_E:
                            best_E, best_lab = E, g
                    new = best_lab
                if new != lab[i]:
                    lab[i] = new
                    changed += 1
            if changed == 0:
                break

        # posteriors from final neighbour labels
        for i in range(n):
            nb = model_nbr[i]
            energies = [outlier_energy + pair_cost(i, -1)]
            if nb.size:
                nbl = lab[nb]
                nbw = model_w[i]
                for g in {int(x) for x in nbl if x >= 0}:
                    U = unary(i, g, nb, nbl, nbw)
                    if U is not None:
                        energies.append(U + pair_cost(i, g))
            e = np.sort(np.array(energies))
            p = np.exp(-(e - e[0]))
            p /= p.sum()
            confidence[idx[i]] = float(p[0])
            margin[idx[i]] = float(e[1] - e[0]) if e.size > 1 else np.inf

        labels[idx] = lab

    labels = _apply_area_min(labels, area_min)
    n_grains = int(labels.max()) + 1 if labels.max() >= 0 else 0
    grains = _summarize(signals, labels, n_grains)
    label_map = _rasterize(signals, labels, int(np.unique(signals.window).size))
    return GrainResult(
        labels=labels,
        n_grains=n_grains,
        grains=grains,
        label_map=label_map,
        confidence=confidence,
        margin=margin,
        params=dict(
            stage="B",
            theta_sigma_deg=theta_sigma_deg,
            r_sigma_rel=r_sigma_rel,
            lam=lam,
            model_radius=model_radius,
            neighbor_dist=neighbor_dist,
            outlier_energy=outlier_energy,
            max_iter=max_iter,
            area_min=area_min,
        ),
    )


# --------------------------------------------------------------------------------------
# visualization overlays
# --------------------------------------------------------------------------------------
# flowline colour basis (matches make_flowline_rainbow_image so hues are consistent)
_FLOWLINE_C0 = np.array([1.0, 0.0, 0.0])
_FLOWLINE_C1 = np.array([0.0, 0.7, 0.0])
_FLOWLINE_C2 = np.array([0.0, 0.3, 1.0])


def orientation_to_rgb(theta_deg, sym_rotation_order: int = 2, theta_offset: float = 0.0):
    """Map orientation angle(s) (degrees) to RGB using the *flowline* colour basis.

    Reproduces ``make_flowline_rainbow_image``: ``theta_color = theta_offset +
    sym_rotation_order * theta`` projected onto three colour vectors peaked at 0, 2pi/3,
    4pi/3, so grain hues match existing flowline plots.  Output adds a trailing length-3
    axis.  For ``sym_rotation_order=2`` (polymer 2-fold) angles theta and theta+180 map to
    the same colour.
    """
    th = np.deg2rad(np.asarray(theta_deg, dtype=float))
    tc = theta_offset + sym_rotation_order * th
    denom = (np.pi * 2.0 / 3.0) ** 2

    def proj(shift):
        return np.maximum(1.0 - np.abs(np.mod(tc - shift + np.pi, 2 * np.pi) - np.pi) ** 2 / denom, 0.0)

    b0, b1, b2 = proj(0.0), proj(np.pi * 2.0 / 3.0), proj(np.pi * 4.0 / 3.0)
    rgb = b0[..., None] * _FLOWLINE_C0 + b1[..., None] * _FLOWLINE_C1 + b2[..., None] * _FLOWLINE_C2
    return np.clip(rgb, 0.0, 1.0)


def _hsv_to_rgb(h, s, v):
    h = np.asarray(h, float); s = np.asarray(s, float); v = np.asarray(v, float)
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def _qualitative_palette(n: int, seed: int = 0):
    """n maximally-spaced distinct colours (golden-ratio hue spacing)."""
    if n <= 0:
        return np.zeros((0, 3))
    k = np.arange(n)
    h = (0.61803398875 * (k + 1) + seed * 0.137) % 1.0
    return _hsv_to_rgb(h, np.full(n, 0.62), np.full(n, 0.97))


def _rasterize_window(signals: SignalTable, result: GrainResult, window: int):
    """Per-window maps (highest-intensity signal wins each probe): label, theta, confidence,
    and a 'filled' mask (a signal of this window present regardless of label)."""
    Rx, Ry = signals.map_shape
    lab = np.full((Rx, Ry), -1, dtype=np.int64)
    th = np.zeros((Rx, Ry))
    conf = np.zeros((Rx, Ry)) if result.confidence is not None else None
    filled = np.zeros((Rx, Ry), dtype=bool)
    idx = np.nonzero(signals.window == window)[0]
    order = idx[np.argsort(signals.intensity[idx], kind="stable")]
    for i in order:
        rx, ry = signals.pos[i]
        filled[rx, ry] = True
        th[rx, ry] = signals.theta[i]
        lab[rx, ry] = result.labels[i]
        if conf is not None:
            conf[rx, ry] = result.confidence[i]
    return lab, th, conf, filled


def _boundary_mask(lab: np.ndarray, outline_background: bool = False) -> np.ndarray:
    """Boundary pixels between two *distinct grains* (both labels >= 0).

    Grain<->outlier / grain<->empty transitions are NOT marked, so scattered outliers do
    not leave black halos.  With ``outline_background=True`` the assigned side of a
    grain<->background edge is also outlined.
    """
    b = np.zeros(lab.shape, dtype=bool)
    up, dn = lab[:-1, :], lab[1:, :]
    le, ri = lab[:, :-1], lab[:, 1:]
    dv = (up != dn) & (up >= 0) & (dn >= 0)
    dh = (le != ri) & (le >= 0) & (ri >= 0)
    b[:-1, :] |= dv; b[1:, :] |= dv
    b[:, :-1] |= dh; b[:, 1:] |= dh
    if outline_background:
        b[:-1, :] |= (up >= 0) & (dn < 0)
        b[1:, :] |= (dn >= 0) & (up < 0)
        b[:, :-1] |= (le >= 0) & (ri < 0)
        b[:, 1:] |= (ri >= 0) & (le < 0)
    return b


def grain_rgb_overlay(
    signals: SignalTable,
    result: GrainResult,
    *,
    window: int = 0,
    mode: str = "orientation",
    overlap: str = "dominant",
    stripe_width: int = 2,
    boundary: bool = True,
    boundary_color=(0.0, 0.0, 0.0),
    outline_background: bool = False,
    background=(0.12, 0.12, 0.12),
    outlier_color=None,
    confidence_shading: bool = False,
    sym_rotation_order: int = 2,
    theta_offset: float = 0.0,
    qualitative_seed: int = 0,
    upsample: int = 1,
) -> np.ndarray:
    """Build an RGB image of the grain clustering for one radial window.

    Visually distinct from flowlines: a *filled segmentation with hard grain boundaries*
    rather than streamlines.  Modes:

    * ``"orientation"``      -- each probe coloured by its own signal orientation (flowline
      hue), so within-grain orientation gradients stay visible *and* grains are outlined
      (the candidate to supersede flowlines: same orientation field + grain structure).
    * ``"mean_orientation"`` -- each grain a flat colour = its circular-mean orientation.
    * ``"grain"``            -- a distinct qualitative colour per grain id (partition only).

    ``overlap="stripe"`` renders probes carrying several grains as diagonally striped tiles
    (one stripe colour per grain, ordered by intensity), so overlapping grains are visible in
    one image; there ``upsample`` sets the tile size (auto-bumped to 8 if < 4) and
    ``stripe_width`` the stripe period.  ``overlap="dominant"`` (default) keeps the
    highest-intensity grain per probe.

    ``confidence_shading`` (Stage B) dims low-confidence signals; ``outlier_color`` fills
    rejected signals; ``upsample`` does nearest-neighbour zoom.  Returns ``(Rx*u, Ry*u, 3)``
    in [0, 1].
    """
    Rx, Ry = signals.map_shape
    if overlap == "stripe":
        tile = upsample if upsample >= 4 else 8
        return _striped_overlay(
            signals, result, window, mode=mode, tile=tile, stripe_width=stripe_width,
            background=background, boundary=boundary, boundary_color=boundary_color,
            outline_background=outline_background, confidence_shading=confidence_shading,
            sym_rotation_order=sym_rotation_order, theta_offset=theta_offset,
            qualitative_seed=qualitative_seed,
        )
    if overlap != "dominant":
        raise ValueError(f"unknown overlap {overlap!r}")
    lab, th, conf, filled = _rasterize_window(signals, result, window)
    assigned = lab >= 0

    rgb = np.zeros((Rx, Ry, 3), float)
    if background is not None:
        rgb[:] = np.asarray(background, float)

    if mode == "orientation":
        rgb[assigned] = orientation_to_rgb(th[assigned], sym_rotation_order, theta_offset)
    elif mode == "mean_orientation":
        mean_th = {int(g.label): g.theta_mean for g in result.grains}
        gm = np.array([mean_th.get(int(l), 0.0) for l in lab[assigned]])
        rgb[assigned] = orientation_to_rgb(gm, sym_rotation_order, theta_offset)
    elif mode == "grain":
        palette = _qualitative_palette(max(result.n_grains, 1), qualitative_seed)
        rgb[assigned] = palette[lab[assigned]]
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if outlier_color is not None:
        rgb[filled & ~assigned] = np.asarray(outlier_color, float)

    if confidence_shading and conf is not None:
        factor = np.ones((Rx, Ry))
        factor[assigned] = np.clip(conf[assigned], 0.0, 1.0)
        rgb = rgb * factor[..., None]

    if boundary:
        rgb[_boundary_mask(lab, outline_background)] = np.asarray(boundary_color, float)

    if upsample > 1:
        rgb = np.kron(rgb, np.ones((upsample, upsample, 1)))
    return rgb


def _probe_signal_stacks(signals: SignalTable, result: GrainResult, window: int):
    """Per probe, the list of assigned signals ``(label, theta, intensity, confidence)``,
    de-duplicated by grain and sorted by descending intensity (the 'stack' at that probe)."""
    idx = np.nonzero(signals.window == window)[0]
    conf_arr = result.confidence
    stacks: dict = {}
    for i in idx:
        lab = int(result.labels[i])
        if lab < 0:
            continue
        key = (int(signals.pos[i, 0]), int(signals.pos[i, 1]))
        c = float(conf_arr[i]) if conf_arr is not None else 1.0
        stacks.setdefault(key, []).append((lab, float(signals.theta[i]), float(signals.intensity[i]), c))
    out = {}
    for key, lst in stacks.items():
        lst.sort(key=lambda t: -t[2])
        seen, uniq = set(), []
        for t in lst:
            if t[0] in seen:
                continue
            seen.add(t[0])
            uniq.append(t)
        out[key] = uniq
    return out


def _striped_overlay(
    signals, result, window, *, mode, tile, stripe_width, background, boundary,
    boundary_color, outline_background, confidence_shading, sym_rotation_order,
    theta_offset, qualitative_seed,
):
    """Render multi-grain probes as diagonally striped tiles (see ``grain_rgb_overlay``)."""
    Rx, Ry = signals.map_shape
    palette = _qualitative_palette(max(result.n_grains, 1), qualitative_seed)
    mean_th = {int(g.label): g.theta_mean for g in result.grains}

    def color_for(label, theta):
        if mode == "orientation":
            return np.asarray(orientation_to_rgb(theta, sym_rotation_order, theta_offset), float)
        if mode == "mean_orientation":
            return np.asarray(orientation_to_rgb(mean_th.get(int(label), 0.0), sym_rotation_order, theta_offset), float)
        if mode == "grain":
            return np.asarray(palette[int(label)], float)
        raise ValueError(f"unknown mode {mode!r}")

    stacks = _probe_signal_stacks(signals, result, window)
    img = np.zeros((Rx * tile, Ry * tile, 3), float)
    if background is not None:
        img[:] = np.asarray(background, float)
    dom = np.full((Rx, Ry), -1, dtype=np.int64)
    iu, ju = np.mgrid[0:tile, 0:tile]
    base = (iu + ju) // max(int(stripe_width), 1)

    for (rx, ry), stack in stacks.items():
        dom[rx, ry] = stack[0][0]
        colors = []
        for (lab, theta, _inten, conf) in stack:
            c = color_for(lab, theta)
            if confidence_shading:
                c = c * float(np.clip(conf, 0.0, 1.0))
            colors.append(c)
        sub = img[rx * tile:(rx + 1) * tile, ry * tile:(ry + 1) * tile]
        if len(colors) == 1:
            sub[:] = colors[0]
        else:
            sidx = base % len(colors)
            for k, c in enumerate(colors):
                sub[sidx == k] = c

    if boundary:
        bt = max(1, tile // 6)
        bc = np.asarray(boundary_color, float)
        for rx, ry in zip(*np.nonzero((dom[:-1, :] >= 0) & (dom[1:, :] >= 0) & (dom[:-1, :] != dom[1:, :]))):
            y = (int(rx) + 1) * tile
            img[max(0, y - bt):y + bt, int(ry) * tile:(int(ry) + 1) * tile] = bc
        for rx, ry in zip(*np.nonzero((dom[:, :-1] >= 0) & (dom[:, 1:] >= 0) & (dom[:, :-1] != dom[:, 1:]))):
            x = (int(ry) + 1) * tile
            img[int(rx) * tile:(int(rx) + 1) * tile, max(0, x - bt):x + bt] = bc
    return img


def orientation_legend_image(size: int = 128, sym_rotation_order: int = 2, theta_offset: float = 0.0):
    """RGBA colour-wheel legend (orientation -> flowline hue), transparent outside a ring."""
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    c = (size - 1) / 2.0
    dx, dy = xx - c, -(yy - c)
    rad = np.hypot(dx, dy) / (size / 2.0)
    ang = np.degrees(np.arctan2(dy, dx)) % 180.0
    rgb = orientation_to_rgb(ang, sym_rotation_order, theta_offset)
    alpha = ((rad <= 1.0) & (rad >= 0.32)).astype(float)
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


def plot_grain_map(
    signals: SignalTable,
    result: GrainResult,
    *,
    window: int = 0,
    mode: str = "orientation",
    ax=None,
    title=None,
    legend: bool = True,
    **overlay_kw,
):
    """Plot a grain overlay (lazy matplotlib import).  Returns the matplotlib Axes."""
    import matplotlib.pyplot as plt

    rgb = grain_rgb_overlay(signals, result, window=window, mode=mode, **overlay_kw)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb, origin="upper", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or f"grains (window {window}, mode={mode}, n={result.n_grains})")
    if legend and mode in ("orientation", "mean_orientation"):
        leg = ax.inset_axes([0.80, 0.80, 0.18, 0.18])
        leg.imshow(
            orientation_legend_image(
                sym_rotation_order=overlay_kw.get("sym_rotation_order", 2),
                theta_offset=overlay_kw.get("theta_offset", 0.0),
            ),
            origin="upper",
            interpolation="bilinear",
        )
        leg.set_xticks([]); leg.set_yticks([])
        leg.patch.set_alpha(0.0)
    return ax
