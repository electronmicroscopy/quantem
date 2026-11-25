import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pytest

from quantem.core.datastructures import Vector
from quantem.diffraction.calibration import _ellipse_matrix, ellipse_from_axes
from quantem.diffraction.polymer_ice import (
    IceFlaggerParams,
    detect_ice,
    measure_peak_widths,
    orientation_histograms,
    plot_q_intensity_density,
    selection_box,
)

FIELDS = ["q", "theta", "intensity"]


def _peaks(cells):
    """Vector from a nested list of [q, theta_deg, intensity] rows per scan position."""
    cells = [[np.array(c, dtype=np.float64).reshape(-1, 3) for c in row] for row in cells]
    for c in (c for row in cells for c in row):
        c[:, 1] = np.deg2rad(c[:, 1] % 360.0)
    return Vector.from_data(cells, fields=FIELDS)


def _lattice(phi, jitter=np.zeros(6)):
    """Six ice peaks at phi + 60 k + jitter[k] degrees."""
    return [[1.61, phi + 60 * k + j, 1.0] for k, j in enumerate(jitter)]


def _flags(result):
    return result.mask.numpy()[:, 0] > 0


def test_detect_ice_flags_six_fold_sets_and_filter_removes_them():
    rng = np.random.default_rng(0)

    def polymer(angle):
        # A split arc and its Friedel partner: one folded arm, never six-fold.
        return [[1.61, angle + d, 0.8] for d in (0, 3, 180, 183)]

    def background(low, high):
        return [[0.6, a, i] for a, i in zip(rng.uniform(0, 360, 10), rng.uniform(low, high, 10))]

    # Near 0 and 60 degrees the lattice wraps on the 0-60 wedge. The old period-360 fit split
    # it into two 3-peak halves that lost to the 4-peak polymer cluster, so no ice was found.
    split = 2.0 * (-1.0) ** np.arange(6)
    phis = [0.0, 59.0, rng.uniform(0, 60), None, 30.0, rng.uniform(0, 60)]
    cells, truth, weak = [], [], []
    for index, phi in enumerate(phis):
        ice = [] if phi is None else _lattice(phi, split if index < 2 else rng.uniform(-2, 2, 6))
        rest = polymer(rng.uniform(0, 360) if phi is None else phi + 30)
        rest += background(0.0, 0.05) if index == 4 else background(0.1, 0.3)
        sweep = [[1.61, phi + 120, 0.05]] if index == 2 else []
        cells.append(ice + rest + sweep)
        truth.append([index != 4] * len(ice) + [False] * (len(rest) + len(sweep)))
        weak += [False] * (len(ice) + len(rest)) + [True] * len(sweep)
    peaks = _peaks([cells[:3], cells[3:]])
    scan_mask = np.array([[True, True, True], [True, False, True]])

    result = detect_ice(peaks, IceFlaggerParams(intensity_percentile_global=30.0), scan_mask)
    analysed = np.repeat(scan_mask.ravel(), peaks.row_counts())
    expected = np.percentile(peaks.select_fields("intensity").numpy()[analysed, 0], 30.0)
    assert result.intensity_threshold == pytest.approx(expected)
    is_ice = np.concatenate(truth)
    np.testing.assert_array_equal(_flags(result), is_ice)
    assert result.matched_arm_count_map.tolist() == [[3, 3, 3], [0, 0, 3]]

    kept = result.filter(peaks)
    np.testing.assert_array_equal(kept.numpy(), peaks.numpy()[~is_ice])
    assert kept.row_counts() == [len(c) - sum(t) for c, t in zip(cells, truth)]

    params = IceFlaggerParams(intensity_percentile_global=30.0, conservative=False)
    np.testing.assert_array_equal(_flags(detect_ice(peaks, params, scan_mask)), is_ice | weak)
    with pytest.raises(ValueError, match="min_matches"):
        IceFlaggerParams(min_matches=4)


def test_measure_peak_widths_and_sharpness_gate():
    origin, pixel_size, radius = (47.3, 48.6), 0.04, 1.61 / 0.04
    rows, cols = np.mgrid[:96, :96].astype(float)
    sigma, sigma_arc = 3.0 / 2.3548, np.deg2rad(40.0) / 2.3548  # FWHM 3 px and 40 deg
    angles = np.deg2rad([30.0, 90.0, 270.0])
    unit = np.column_stack([np.sin(angles), np.cos(angles)])
    params = IceFlaggerParams(sharpness_window_q_invA=0.2)
    for ellipse in (None, ellipse_from_axes(1.05, 0.95, 25.0)):
        to_circle = np.eye(2) if ellipse is None else _ellipse_matrix(ellipse)
        d = np.stack([rows - origin[0], cols - origin[1]], -1) @ to_circle.T
        dr, dc = d[..., 0], d[..., 1]
        spot = np.exp(
            -((dr - radius * unit[0, 0]) ** 2 + (dc - radius * unit[0, 1]) ** 2) / (2 * sigma**2)
        )
        dt = (np.arctan2(dr, dc) - angles[1] + np.pi) % (2 * np.pi) - np.pi
        arc = np.exp(
            -((np.hypot(dr, dc) - radius) ** 2) / (2 * sigma**2) - dt**2 / (2 * sigma_arc**2)
        )
        dp = spot + arc + np.random.default_rng(0).normal(0, 0.03, spot.shape)
        peaks_rc = np.array(origin) + (radius * unit) @ np.linalg.inv(to_circle).T

        width_q, width_theta = measure_peak_widths(
            dp, peaks_rc, origin, pixel_size, params, ellipse=ellipse
        )
        assert width_q == pytest.approx([0.12] * 3, abs=0.015), ellipse
        expected_theta = [np.rad2deg(3.0 / radius), 40.0, 40.0]
        assert np.rad2deg(width_theta) == pytest.approx(expected_theta, rel=0.2), ellipse

    width_q, width_theta = width_q[:2], width_theta[:2]
    angles = angles[:2]
    cell = np.column_stack([[1.61, 1.61], angles, [1.0, 1.0], width_q, width_theta])
    peaks = Vector.from_data([[cell]], fields=FIELDS + ["width_q", "width_theta"])
    for ceiling, expected in ((None, [True, True]), (15.0, [True, False])):
        p = IceFlaggerParams(intensity_cutoff=0.5, min_matches=1, max_width_theta_deg=ceiling)
        assert _flags(detect_ice(peaks, p)).tolist() == expected, ceiling
    with pytest.raises(ValueError, match="width_theta"):
        detect_ice(peaks.select_fields(FIELDS), p)


def test_multiple_crystallites_are_peeled():
    for label, phis, n, separation, found in [
        ("two lattices", (10.0, 35.0), 2, None, [10.0, 35.0]),
        ("one pass", (10.0, 35.0), 1, None, [10.0]),
        ("within min_phi_separation", (10.0, 18.0), 2, 15.0, [10.0]),
    ]:
        peaks = _peaks([[_lattice(phis[0]) + _lattice(phis[1])]])
        params = IceFlaggerParams(
            intensity_cutoff=0.5,
            dtheta_deg=3.0,
            max_crystallites=n,
            min_phi_separation_deg=separation,
        )
        result = detect_ice(peaks, params, return_debug=True)
        assert result.debug[(0, 0)]["phi_deg"] == pytest.approx(found), label
        np.testing.assert_array_equal(_flags(result), np.arange(12) < 6 * len(found), label)


def test_require_friedel_pair_rejects_close_peaks():
    # Arms at 20 and 80 degrees hold opposed pairs; the arm at 140 holds two peaks 3 degrees apart.
    peaks = _peaks([[[[1.61, a, 1.0] for a in (20, 200, 80, 260, 140, 143)]]])
    for pair, expected in ((False, [True] * 6), (True, [True] * 4 + [False] * 2)):
        p = IceFlaggerParams(intensity_cutoff=0.5, min_peaks_per_arm=2, require_friedel_pair=pair)
        assert _flags(detect_ice(peaks, p)).tolist() == expected, pair


def test_ice_figures(tmp_path):
    # rows of [q, theta_deg, intensity, width_q, width_theta_deg]
    lattice = [[1.61, 10 + 60 * k, 1.0, 0.01, 3.0] for k in range(6)]
    arc = [[1.61, a, 0.8, 0.01, 30.0] for a in (40, 220)]
    other = [[0.6, 15.0, 0.2, np.nan, np.nan]]
    cells = [[lattice + other, arc + other], [[], []]]
    cells = [[np.array(c, dtype=np.float64).reshape(-1, 5) for c in row] for row in cells]
    for c in (c for row in cells for c in row):
        c[:, [1, 4]] = np.deg2rad(c[:, [1, 4]])
    peaks = Vector.from_data(cells, fields=FIELDS + ["width_q", "width_theta"])
    params = IceFlaggerParams(intensity_cutoff=0.5, min_matches=2)

    # the plotted threshold is the one detect_ice applies
    percentile = dataclasses.replace(params, intensity_cutoff=None, intensity_percentile_global=50)
    for p in (params, percentile):
        _, ax = plot_q_intensity_density(peaks, p)
        threshold = detect_ice(peaks, p).intensity_threshold
        assert f"{threshold:.3g}" in ax.texts[0].get_text()

    tuned = selection_box(peaks, params, max_width_theta_deg=10.0, fig_path=tmp_path / "box.png")
    assert (tuned.max_width_theta_deg, tuned.max_width_q_invA) == (10.0, None)
    assert plt.gcf()._suptitle.get_text().startswith("6 of 8")
    assert (tmp_path / "box.png").exists()
    with pytest.raises(ValueError, match="width_q"):
        selection_box(peaks.select_fields(FIELDS), params)

    result = detect_ice(peaks, tuned)
    hists = orientation_histograms(peaks, result, tuned)
    assert np.allclose(hists["all"], hists["kept"] + hists["removed"])
    assert np.isclose(hists["removed"].sum(), 6.0)
    assert np.isclose(hists["kept"].sum(), 1.6)
    plt.close("all")
