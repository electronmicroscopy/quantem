"""Synthetic acceptance + stability tests for grain_clustering.

Runnable two ways:
    uv run --with numpy --with scipy python tests/diffraction/test_grain_clustering.py
    uv run --with pytest --with numpy --with scipy python -m pytest tests/diffraction/test_grain_clustering.py

The module under test depends only on numpy + scipy, so we import it directly (bypassing
the quantem package __init__, which would pull torch) by adding its directory to sys.path.
"""

from __future__ import annotations

import os
import sys
from math import comb

import numpy as np

_DIFF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "quantem", "diffraction",
)
sys.path.insert(0, _DIFF_DIR)

from grain_clustering import (  # noqa: E402
    SignalTable,
    cluster_signals_into_grains,
    circular_distance_deg,
    extract_signals,
    refine_grains_crf,
    orientation_to_rgb,
    grain_rgb_overlay,
    orientation_legend_image,
)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def adjusted_rand_index(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    ua = {v: i for i, v in enumerate(np.unique(a))}
    ub = {v: i for i, v in enumerate(np.unique(b))}
    C = np.zeros((len(ua), len(ub)), dtype=np.int64)
    for x, y in zip(a, b):
        C[ua[x], ub[y]] += 1
    sum_c = sum(comb(int(v), 2) for v in C.ravel() if v >= 2)
    a_c = sum(comb(int(v), 2) for v in C.sum(axis=1) if v >= 2)
    b_c = sum(comb(int(v), 2) for v in C.sum(axis=0) if v >= 2)
    total = comb(int(C.sum()), 2)
    if total == 0:
        return 1.0
    expected = a_c * b_c / total
    max_index = 0.5 * (a_c + b_c)
    if max_index == expected:
        return 1.0
    return (sum_c - expected) / (max_index - expected)


class _SB:
    """Tiny signal builder accumulating (pos, theta, r, intensity, window, gt_label)."""

    def __init__(self, map_shape):
        self.map_shape = map_shape
        self.rows = []  # (rx, ry, theta, r, I, window, gt)

    def add(self, rx, ry, theta, r=1.0, I=1.0, window=0, gt=0):
        self.rows.append((rx, ry, theta % 180.0, r, I, window, gt))

    def add_rect(self, x0, x1, y0, y1, theta_fn, r=1.0, I=1.0, window=0, gt=0, keep=None, rng=None):
        for rx in range(x0, x1):
            for ry in range(y0, y1):
                if keep is not None and rng is not None and rng.random() > keep:
                    continue
                th = theta_fn(rx, ry) if callable(theta_fn) else theta_fn
                ii = I(rx, ry) if callable(I) else I
                self.add(rx, ry, th, r, ii, window, gt)

    def build(self):
        rows = self.rows
        pos = np.array([(r[0], r[1]) for r in rows], dtype=np.int64)
        theta = np.array([r[2] for r in rows], float)
        rad = np.array([r[3] for r in rows], float)
        inten = np.array([r[4] for r in rows], float)
        win = np.array([r[5] for r in rows], np.int64)
        gt = np.array([r[6] for r in rows], np.int64)
        return SignalTable(pos, theta, rad, inten, win, self.map_shape), gt


def _grain_count(result):
    return result.n_grains


def _no_double_probe(result, signals):
    """Assert no grain holds two signals from the same probe."""
    Rx, Ry = signals.map_shape
    lin = signals.pos[:, 0] * Ry + signals.pos[:, 1]
    for g in range(result.n_grains):
        sids = np.nonzero(result.labels == g)[0]
        assert len(np.unique(lin[sids])) == len(sids), f"grain {g} has two signals at one probe"


# --------------------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------------------
def test_bicrystal_splits():
    b = _SB((20, 20))
    b.add_rect(0, 10, 0, 20, 20.0, gt=0)
    b.add_rect(10, 20, 0, 20, 70.0, gt=1)
    sig, gt = b.build()
    res = cluster_signals_into_grains(sig, theta_tol_deg=10.0)
    assert _grain_count(res) == 2, res.n_grains
    assert adjusted_rand_index(res.labels, gt) == 1.0
    _no_double_probe(res, sig)


def test_bent_grain_stays_one():
    b = _SB((30, 20))
    # orientation ramps 1 deg/probe along x: total spread 29 deg, local step << tol
    b.add_rect(0, 30, 0, 20, theta_fn=lambda rx, ry: float(rx), gt=0)
    sig, gt = b.build()
    res = cluster_signals_into_grains(sig, theta_tol_deg=10.0)
    assert _grain_count(res) == 1, res.n_grains


def test_quantized_radius_same_theta_two_grains():
    b = _SB((15, 15))
    # two co-located signal classes in the SAME window: same theta, r=1.0 vs r=1.5
    b.add_rect(0, 15, 0, 15, 30.0, r=1.0, gt=0)
    b.add_rect(0, 15, 0, 15, 30.0, r=1.5, gt=1)
    sig, gt = b.build()
    res = cluster_signals_into_grains(sig, theta_tol_deg=10.0, r_tol_rel=0.10)
    assert _grain_count(res) == 2, res.n_grains
    assert adjusted_rand_index(res.labels, gt) == 1.0
    _no_double_probe(res, sig)


def test_disconnected_same_theta_two_grains():
    b = _SB((10, 30))
    b.add_rect(0, 10, 0, 10, 45.0, gt=0)
    b.add_rect(0, 10, 20, 30, 45.0, gt=1)  # gap of 10 columns
    sig, gt = b.build()
    res = cluster_signals_into_grains(sig, theta_tol_deg=10.0, neighbor_dist=2)
    assert _grain_count(res) == 2, res.n_grains
    assert adjusted_rand_index(res.labels, gt) == 1.0


def test_dropout_does_not_fragment():
    rng = np.random.default_rng(0)
    b = _SB((20, 20))
    b.add_rect(0, 20, 0, 20, 30.0, gt=0, keep=0.7, rng=rng)  # 30% dropout
    sig, gt = b.build()
    res = cluster_signals_into_grains(sig, theta_tol_deg=10.0, neighbor_dist=2)
    assert _grain_count(res) == 1, res.n_grains


def test_overlapping_grains_coexist():
    b = _SB((20, 20))
    # grain A cols 0..12 (theta 10); grain B cols 8..20 (theta 80); overlap 8..12
    b.add_rect(0, 13, 0, 20, 10.0, gt=0)
    b.add_rect(8, 20, 0, 20, 80.0, gt=1)
    sig, gt = b.build()
    res = cluster_signals_into_grains(sig, theta_tol_deg=10.0)
    assert _grain_count(res) == 2, res.n_grains
    assert adjusted_rand_index(res.labels, gt) == 1.0
    _no_double_probe(res, sig)


def test_noise_becomes_outliers():
    rng = np.random.default_rng(1)
    b = _SB((40, 40))
    b.add_rect(0, 20, 0, 20, 30.0, gt=0)  # one real grain
    n_noise = 15
    used = set()
    placed = 0
    while placed < n_noise:  # isolated, well-separated, in the empty half
        rx = int(rng.integers(25, 40))
        ry = int(rng.integers(25, 40))
        if (rx // 3, ry // 3) in used:
            continue
        used.add((rx // 3, ry // 3))
        b.add(rx, ry, float(rng.uniform(0, 180)), gt=-(placed + 1))
        placed += 1
    sig, gt = b.build()
    res = cluster_signals_into_grains(sig, theta_tol_deg=10.0, area_min=3)
    assert _grain_count(res) == 1, res.n_grains
    noise_mask = gt < 0
    assert np.all(res.labels[noise_mask] == -1)


def test_intensity_ramp_does_not_fragment():
    b = _SB((20, 20))
    b.add_rect(0, 20, 0, 20, 30.0, I=lambda rx, ry: 1.0 + rx, gt=0)
    sig, gt = b.build()
    res = cluster_signals_into_grains(sig, theta_tol_deg=10.0)  # intensity not gated
    assert _grain_count(res) == 1, res.n_grains


def test_order_independent_stability():
    b = _SB((20, 20))
    b.add_rect(0, 10, 0, 20, 20.0, gt=0)
    b.add_rect(10, 20, 0, 20, 70.0, gt=1)
    sig, gt = b.build()
    res1 = cluster_signals_into_grains(sig, theta_tol_deg=10.0)

    rng = np.random.default_rng(7)
    perm = rng.permutation(len(sig))
    sig2 = SignalTable(sig.pos[perm], sig.theta[perm], sig.r[perm],
                       sig.intensity[perm], sig.window[perm], sig.map_shape)
    res2 = cluster_signals_into_grains(sig2, theta_tol_deg=10.0)

    # invert the permutation to compare partitions on the original ordering
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    assert adjusted_rand_index(res1.labels, res2.labels[inv]) == 1.0


def test_distinct_windows_are_independent():
    # same orientation across two windows must NEVER merge (windows are class gates)
    b = _SB((15, 15))
    b.add_rect(0, 15, 0, 15, 30.0, r=1.0, window=0, gt=0)
    b.add_rect(0, 15, 0, 15, 30.0, r=3.0, window=1, gt=1)
    sig, gt = b.build()
    res = cluster_signals_into_grains(sig, theta_tol_deg=10.0, r_tol_rel=np.inf)
    assert _grain_count(res) == 2, res.n_grains
    assert adjusted_rand_index(res.labels, gt) == 1.0


def test_circular_distance():
    assert abs(circular_distance_deg(10.0, 170.0)) == 20.0
    assert abs(circular_distance_deg(0.0, 90.0)) == 90.0
    assert abs(circular_distance_deg(5.0, 175.0)) == 10.0


# --------------------------------------------------------------------------------------
# extract_signals: Karen / BraggPeaksPolymer convention
# --------------------------------------------------------------------------------------
class _PolarField:
    """Minimal stand-in for BraggPeaksPolymer.polar_peaks / .peak_intensities."""

    def __init__(self, arrays, fields):
        self._a = arrays
        self.fields = set(fields)
        self.shape = next(iter(arrays.values())).shape

    def __getitem__(self, f):
        return self._a[f]


class _FakeBP:
    def __init__(self, polar, inten):
        self.polar_peaks = polar
        self.peak_intensities = inten


def _make_fake_bp():
    Rx, Ry = 2, 1
    R = np.empty((Rx, Ry), dtype=object)
    TH = np.empty((Rx, Ry), dtype=object)
    II = np.empty((Rx, Ry), dtype=object)
    R[0, 0] = np.array([1.0, 2.0]); TH[0, 0] = np.array([0.3, 1.2]); II[0, 0] = np.array([5.0, 7.0])
    R[1, 0] = np.array([]); TH[1, 0] = np.array([]); II[1, 0] = np.array([])
    polar = _PolarField({"r": R, "theta": TH}, ["r", "theta"])
    inten = _PolarField({"intensity": II}, ["intensity"])
    return _FakeBP(polar, inten)


def test_extract_signals_karen_convention():
    bp = _make_fake_bp()
    rr = np.array([[0.5, 1.5], [1.5, 2.5]])  # two windows gate the two radii
    sig = extract_signals(bp, rr)
    assert len(sig) == 2
    order = np.argsort(sig.window)
    assert list(sig.window[order]) == [0, 1]
    assert np.allclose(sig.r[order], [1.0, 2.0])
    assert np.allclose(sig.intensity[order], [5.0, 7.0])
    # theta uses the stored polar angle directly: degrees(mod(theta_rad, pi))
    assert np.allclose(sig.theta[order], np.degrees([0.3, 1.2]))
    assert np.all((sig.theta >= 0) & (sig.theta < 180))


def test_extract_signals_flip_sign():
    bp = _make_fake_bp()
    rr = np.array([[0.5, 1.5], [1.5, 2.5]])
    sig = extract_signals(bp, rr, flip_sign=True)
    order = np.argsort(sig.window)
    assert np.allclose(sig.theta[order], np.degrees(np.mod([-0.3, -1.2], np.pi)))


# --------------------------------------------------------------------------------------
# "grains flowing into one another" (continuous chain == one grain)
# --------------------------------------------------------------------------------------
def _focal_flow_theta(frow, fcol):
    """Orientation of strands converging toward a focal point (frow, fcol)."""
    def fn(rx, ry):
        return np.degrees(np.arctan2(frow - rx, fcol - ry)) % 180.0
    return fn


def test_flowing_grains_merge_into_one():
    # Strands converging to a focal point above the top-centre (the sketch): the
    # orientation field is continuous everywhere, so it is ONE grain even though the
    # local orientation sweeps a wide angular range.
    H, W = 30, 30
    b = _SB((H, W))
    b.add_rect(0, H, 0, W, theta_fn=_focal_flow_theta(frow=-30, fcol=W / 2), gt=0)
    sig, _ = b.build()
    res = cluster_signals_into_grains(sig, theta_tol_deg=20.0, neighbor_dist=1)
    assert _grain_count(res) == 1, res.n_grains


def test_flow_with_cusp_splits():
    # Same flow but with a 40-deg discontinuity along the central seam: a genuine
    # orientation jump -> TWO grains.  Continuity (not similarity) is the criterion.
    H, W = 30, 30
    fn = _focal_flow_theta(frow=-30, fcol=W / 2)
    b = _SB((H, W))
    for rx in range(H):
        for ry in range(W):
            jump = 40.0 if ry >= W // 2 else 0.0
            b.add(rx, ry, fn(rx, ry) + jump, gt=0 if ry < W // 2 else 1)
    sig, gt = b.build()
    res = cluster_signals_into_grains(sig, theta_tol_deg=20.0, neighbor_dist=1)
    assert _grain_count(res) == 2, res.n_grains
    assert adjusted_rand_index(res.labels, gt) == 1.0


# --------------------------------------------------------------------------------------
# Stage B: CRF + EM refinement
# --------------------------------------------------------------------------------------
def _bicrystal():
    b = _SB((20, 20))
    b.add_rect(0, 10, 0, 20, 20.0, gt=0)
    b.add_rect(10, 20, 0, 20, 70.0, gt=1)
    return b.build()


def test_refine_preserves_clean_partition():
    sig, gt = _bicrystal()
    a = cluster_signals_into_grains(sig, theta_tol_deg=10.0, area_min=1)
    rb = refine_grains_crf(sig, a)
    assert rb.n_grains == 2, rb.n_grains
    assert adjusted_rand_index(rb.labels, gt) == 1.0
    assert rb.confidence is not None
    assert np.all((rb.confidence >= 0) & (rb.confidence <= 1.0 + 1e-9))
    assert np.all(rb.confidence > 0.5)  # high-contrast bicrystal -> confident everywhere
    _no_double_probe(rb, sig)


def test_refine_recovers_corrupted_boundary():
    sig, gt = _bicrystal()
    a = cluster_signals_into_grains(sig, theta_tol_deg=10.0, area_min=1)
    a_label = int(a.labels[np.nonzero(sig.pos[:, 0] == 0)[0][0]])   # a grain-A signal
    b_label = int(a.labels[np.nonzero(sig.pos[:, 0] == 19)[0][0]])  # a grain-B signal
    corrupt = a.labels.copy()
    strip = np.nonzero(sig.pos[:, 0] == 9)[0]  # column 9 belongs to grain A (cols 0..9)
    corrupt[strip] = b_label                   # mislabel it as grain B
    ari_corrupt = adjusted_rand_index(corrupt, gt)
    rb = refine_grains_crf(sig, corrupt, theta_sigma_deg=8.0)
    ari_refined = adjusted_rand_index(rb.labels, gt)
    assert ari_refined > ari_corrupt
    assert ari_refined > 0.95
    assert a_label != b_label


def test_refine_does_not_split_flow():
    # Stage B must not undo Stage A's flowing-grain behaviour (local orientation model).
    H, W = 30, 30
    b = _SB((H, W))
    b.add_rect(0, H, 0, W, theta_fn=_focal_flow_theta(frow=-30, fcol=W / 2), gt=0)
    sig, _ = b.build()
    a = cluster_signals_into_grains(sig, theta_tol_deg=20.0, area_min=1)
    rb = refine_grains_crf(sig, a, theta_sigma_deg=12.0, model_radius=3.0)
    assert rb.n_grains == 1, rb.n_grains


def test_refine_rejects_outlier():
    b = _SB((15, 15))
    b.add_rect(0, 15, 0, 15, 30.0, gt=0)
    sig, _ = b.build()
    a = cluster_signals_into_grains(sig, theta_tol_deg=10.0, area_min=1)
    off = int(np.nonzero((sig.pos[:, 0] == 7) & (sig.pos[:, 1] == 7))[0][0])
    sig.theta[off] = 120.0                  # orientation far from its surroundings
    init = a.labels.copy()
    init[off] = int(a.labels[np.nonzero(sig.pos[:, 0] == 0)[0][0]])  # force it into the grain
    rb = refine_grains_crf(sig, init, theta_sigma_deg=8.0, outlier_energy=6.0)
    assert rb.labels[off] == -1


def test_refine_confidence_ranges():
    sig, _ = _bicrystal()
    a = cluster_signals_into_grains(sig, theta_tol_deg=10.0, area_min=1)
    rb = refine_grains_crf(sig, a)
    assert np.all(np.isfinite(rb.confidence))
    assert np.all((rb.confidence >= 0) & (rb.confidence <= 1.0 + 1e-9))
    assert np.all(rb.margin >= 0)


# --------------------------------------------------------------------------------------
# visualization overlays
# --------------------------------------------------------------------------------------
def test_orientation_to_rgb_basic():
    rgb = orientation_to_rgb(np.array([0.0, 60.0, 120.0]))
    assert rgb.shape == (3, 3)
    assert np.all((rgb >= 0) & (rgb <= 1))
    # distinct colours for distinct orientations
    assert not np.allclose(rgb[0], rgb[1])
    assert not np.allclose(rgb[1], rgb[2])
    # 2-fold periodicity: theta and theta+180 map to the same colour
    assert np.allclose(orientation_to_rgb(30.0), orientation_to_rgb(210.0))


def test_grain_rgb_overlay_orientation():
    sig, _ = _bicrystal()  # grain A theta=20, grain B theta=70
    a = cluster_signals_into_grains(sig, theta_tol_deg=10.0)
    rgb = grain_rgb_overlay(sig, a, mode="orientation")
    assert rgb.shape == (20, 20, 3)
    # the two grains get different colours
    assert not np.allclose(rgb[0, 0], rgb[19, 0])
    # hard black boundary present along the seam (orientation colours are never pure black)
    assert np.any(np.all(rgb == 0.0, axis=-1))


def test_grain_rgb_overlay_grain_mode_distinct():
    sig, _ = _bicrystal()
    a = cluster_signals_into_grains(sig, theta_tol_deg=10.0)
    rgb = grain_rgb_overlay(sig, a, mode="grain", boundary=False)
    colors = np.unique(rgb.reshape(-1, 3), axis=0)
    assert colors.shape[0] >= a.n_grains  # at least one distinct colour per grain


def test_grain_rgb_overlay_confidence_shading():
    sig, _ = _bicrystal()
    a = cluster_signals_into_grains(sig, theta_tol_deg=10.0)
    a.confidence = np.full(len(sig), 0.5)  # controlled confidence
    base = grain_rgb_overlay(sig, a, mode="orientation", boundary=False)
    shaded = grain_rgb_overlay(sig, a, mode="orientation", boundary=False, confidence_shading=True)
    assert np.allclose(shaded, base * 0.5)


def test_grain_rgb_overlay_stripe_multigrain():
    # overlap band: grain A (cols 0..12, theta 10) + grain B (cols 8..20, theta 80)
    b = _SB((20, 20))
    b.add_rect(0, 13, 0, 20, 10.0, gt=0)
    b.add_rect(8, 20, 0, 20, 80.0, gt=1)
    sig, _ = b.build()
    a = cluster_signals_into_grains(sig, theta_tol_deg=10.0)
    assert a.n_grains == 2
    tile = 8
    rgb = grain_rgb_overlay(sig, a, mode="orientation", overlap="stripe",
                            upsample=tile, boundary=False, background=None)
    assert rgb.shape == (20 * tile, 20 * tile, 3)
    # overlap probe (rx in 8..12) -> striped tile with both grain colours
    t_over = rgb[10 * tile:11 * tile, 5 * tile:6 * tile].reshape(-1, 3)
    assert np.unique(t_over, axis=0).shape[0] >= 2
    # single-grain probe (rx < 8) -> solid tile
    t_one = rgb[0:tile, 5 * tile:6 * tile].reshape(-1, 3)
    assert np.unique(t_one, axis=0).shape[0] == 1


def test_orientation_legend_image():
    leg = orientation_legend_image(size=64)
    assert leg.shape == (64, 64, 4)
    assert np.any(leg[..., 3] > 0) and np.any(leg[..., 3] == 0)  # ring + transparent


def test_plot_grain_map_smoke():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("  (matplotlib unavailable -- skipping plot smoke test)")
        return
    from grain_clustering import plot_grain_map

    sig, _ = _bicrystal()
    a = cluster_signals_into_grains(sig, theta_tol_deg=10.0)
    plot_grain_map(sig, a, mode="orientation")
    plot_grain_map(sig, a, mode="grain", legend=False)
    rb = refine_grains_crf(sig, a)
    plot_grain_map(sig, rb, mode="orientation", confidence_shading=True)
    plt.close("all")


# --------------------------------------------------------------------------------------
# script runner
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
