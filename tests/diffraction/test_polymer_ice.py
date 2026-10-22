from __future__ import annotations

import re

import numpy as np
import pytest

from quantem.core.datastructures import Vector
from quantem.diffraction import IceFlaggerParams, detect_ice, measure_peak_widths, sharpness_mask

FOLD = {"theta_period_deg": 180.0}


def _vectors(shape=(1, 1), unfolded=False):
    fields, units = ["r_invA", "theta"], ["1/A", "rad"]
    if unfolded:
        fields, units = fields + ["theta_unfolded"], units + ["rad"]
    polar = Vector.from_shape(shape=shape, fields=fields, units=units)
    intensity = Vector.from_shape(shape=shape, fields=["intensities"], units=["normalized"])
    return polar, intensity


def _cell(angles_deg, unfolded_deg=None, q=1.61):
    """One scan position with peaks at q, angles_deg (and theta_unfolded if given)."""
    polar, intensity = _vectors(unfolded=unfolded_deg is not None)
    columns = [np.full(len(angles_deg), q), np.deg2rad(angles_deg)]
    if unfolded_deg is not None:
        columns.append(np.deg2rad(unfolded_deg))
    polar[0, 0] = np.column_stack(columns)
    intensity[0, 0] = np.full((len(angles_deg), 1), 0.9)
    return polar, intensity


def _lattices(*phis):
    """Six-fold lattices at the given orientations, overlaid in one pattern."""
    return _cell([phi + 60.0 * k for phi in phis for k in range(6)])


def _raises(label, match, call):
    try:
        call()
    except ValueError as error:
        assert re.search(match, str(error)), f"{label}: {error}"
    else:
        pytest.fail(f"{label}: no ValueError raised")


def test_detect_ice_and_filter_does_not_mutate_source():
    polar, intensity = _vectors((1, 2))
    polar[0, 0] = np.column_stack([np.full(3, 1.61), np.deg2rad([5, 65, 140])])
    intensity[0, 0] = np.array([[0.9], [0.8], [0.7]])
    polar[0, 1] = np.empty((0, 2))
    intensity[0, 1] = np.empty((0, 1))
    original = polar[0, 0].numpy().copy()

    result = detect_ice(
        polar, intensity,
        params=IceFlaggerParams(intensity_cutoff=0.5, min_matches=2, dtheta_deg=6),
        return_debug=True,
    )
    assert result.threshold == 0.5
    assert result.flagged_peaks_count_map.tolist() == [[2, 0]]
    filtered = result.filter(polar)
    np.testing.assert_array_equal(polar[0, 0].numpy(), original)
    assert len(filtered[0, 0].numpy()) == 1
    assert (0, 1) in result.debug_records


OBSERVED = {
    "flagged": lambda r: r.flagged_peaks_count_map[0, 0],
    "matched": lambda r: r.matched_bins_count_map[0, 0],
    "n_phi": lambda r: len(r.debug_records[(0, 0)].phi_deg),
    "phi": lambda r: sorted(r.debug_records[(0, 0)].phi_deg),
    "bins": lambda r: list(r.debug_records[(0, 0)].matched_bins),
    "n_bins": lambda r: len(r.debug_records[(0, 0)].matched_bins),
}


def test_flagging_counts():
    """Each case: (label, vectors, IceFlaggerParams extras, detect_ice kwargs, expected)."""
    mm3 = dict(min_matches=3)
    arm2 = dict(min_matches=1, min_peaks_per_arm=2)
    friedel = dict(arm2, require_friedel_pair=True)
    near_pair = ([5.0, 8.0], [5.0, 8.0])  # same side, 3 deg apart: same arm, not a pair
    cases = [
        ("masked cell is not analyzed", _cell([0.0, 60.0]), dict(intensity_cutoff=0.0),
         dict(scan_mask=np.zeros((1, 1), dtype=bool)), dict(flagged=0)),
        # 12 peaks present, only the strongest lattice's 6 claimed.
        ("one crystallite by default", _lattices(3.0, 31.0), mm3, {}, dict(flagged=6, n_phi=1)),
        ("max_crystallites claims both; bins accumulate", _lattices(3.0, 31.0),
         dict(mm3, max_crystallites=3), {}, dict(flagged=12, matched=12, phi=[3.0, 31.0])),
        # A single lattice must not be re-found as a near-duplicate crystallite.
        ("peel does not split one lattice", _lattices(5.0),
         dict(mm3, max_crystallites=5), {}, dict(flagged=6, n_phi=1)),
        # 8 deg apart: separable at dtheta_deg=3, not with min_phi_separation_deg=15.
        ("close lattices separable", _lattices(3.0, 11.0),
         dict(mm3, dtheta_deg=3.0, max_crystallites=3), {}, dict(n_phi=2)),
        ("min_phi_separation merges close lattices", _lattices(3.0, 11.0),
         dict(mm3, dtheta_deg=3.0, max_crystallites=3, min_phi_separation_deg=15.0), {},
         dict(n_phi=1)),
        # theta and theta+180 both fold to 5 deg; before the period was honoured the
        # pair counted as one bin and was rejected by min_matches=2.
        ("folded friedel pair", _cell([5.0, 5.0]), dict(min_matches=1), FOLD, dict(flagged=2)),
        ("three arms reachable on 180 deg", _cell([5.0, 65.0, 125.0]), mm3, FOLD,
         dict(flagged=3, n_bins=3)),
        # Params insist on the full circle, so min_matches=5 is reachable: no error.
        ("params theta_period overrides caller", _cell([5.0, 5.0]),
         dict(min_matches=5, theta_period_deg=360.0), FOLD, dict(flagged=0)),
        ("lone peak, default min_peaks_per_arm", _cell([5.0]), dict(min_matches=1), FOLD,
         dict(flagged=1)),
        ("lone peak below min_peaks_per_arm", _cell([5.0]), arm2, FOLD, dict(flagged=0)),
        # Also: vectors predating theta_unfolded still work when it is not required.
        ("pair meets min_peaks_per_arm", _cell([5.0, 5.0]), arm2, FOLD, dict(flagged=2)),
        # An under-populated arm must not contribute its peaks to the flagged set.
        ("under-populated arm dropped", _cell([5.0, 5.0, 65.0]), arm2, FOLD,
         dict(flagged=2, bins=[0])),
        ("two populated arms meet min_matches", _cell([5.0, 5.0, 65.0, 65.0]),
         dict(arm2, min_matches=2), FOLD, dict(flagged=4)),
        ("three populated arms required", _cell([5.0, 5.0, 65.0, 65.0]),
         dict(arm2, min_matches=3), FOLD, dict(flagged=0)),
        # min_peaks_per_arm cannot tell a true pair from two near peaks; theta_unfolded can.
        ("true friedel pair", _cell([5.0, 5.0], [5.0, 185.0]), friedel, FOLD, dict(flagged=2)),
        ("near peaks are not a friedel pair", _cell(*near_pair), friedel, FOLD, dict(flagged=0)),
        ("near peaks pass without require_friedel_pair", _cell(*near_pair), arm2, FOLD,
         dict(flagged=2)),
    ]
    for label, (polar, intensity), extra, kwargs, want in cases:
        params = IceFlaggerParams(**{"intensity_cutoff": 0.5, "dtheta_deg": 6.0, **extra})
        debug = bool(want.keys() - {"flagged"})
        result = detect_ice(polar, intensity, params=params, return_debug=debug, **kwargs)
        for key, value in want.items():
            assert OBSERVED[key](result) == pytest.approx(value, abs=0.5), f"{label}: {key}"


def test_bad_inputs_fail_clearly():
    ragged = _cell([0.0, 0.0])
    ragged[1][0, 0] = np.zeros((1, 1))
    cases = [
        ("ragged", ragged, dict(intensity_cutoff=0.0), {}, "Row count mismatch"),
        ("ceiling without polar_data", _cell([0.0]),
         dict(intensity_cutoff=0.5, max_width_r_invA=0.1), {}, "requires polar_data"),
        ("unreachable min_matches when folded", _cell([5.0, 65.0]),
         dict(intensity_cutoff=0.5, min_matches=5), FOLD, "can never be reached"),
        ("friedel without theta_unfolded", _cell([5.0, 5.0]),
         dict(intensity_cutoff=0.5, require_friedel_pair=True), FOLD, "theta_unfolded"),
    ]
    for label, (polar, intensity), params, kwargs, match in cases:
        _raises(label, match, lambda: detect_ice(
            polar, intensity, params=IceFlaggerParams(**params), **kwargs))


def test_invalid_params_are_rejected_at_construction():
    """A bad value here would otherwise surface as 'nothing was flagged'."""
    for kwargs, match in [
        ({"dq_invA": 0}, "dq_invA must be positive"),
        ({"dtheta_deg": -1}, "dtheta_deg must be positive"),
        ({"min_matches": 0}, "min_matches must be at least 1"),
        ({"min_peaks_per_arm": 0}, "min_peaks_per_arm must be at least 1"),
        ({"max_crystallites": 0}, "max_crystallites must be at least 1"),
        ({"sharpness_crossing_persistence": 0}, "sharpness_crossing_persistence"),
        ({"max_width_theta_deg": 0}, "max_width_theta_deg must be positive when set"),
        ({"sharpness_baseline_quantile": 1.0}, "must be in \\[0, 1\\)"),
        ({"sharpness_smooth_sigma_bins": -1}, "must be non-negative"),
        ({"intensity_cutoff_mode": "nope"}, "must be 'absolute' or 'percentile'"),
    ]:
        _raises(str(kwargs), match, lambda: IceFlaggerParams(**kwargs))
    # Edge values are valid.
    IceFlaggerParams(
        max_width_r_invA=None, max_width_theta_deg=None,
        sharpness_baseline_quantile=0.0, sharpness_smooth_sigma_bins=0.0,
        min_matches=1, min_peaks_per_arm=1, max_crystallites=1,
    )


def _polar_volume(peaks, *, shape=(1, 1), n_r=600, n_theta=180, r_max=3.0):
    """Polar volume with a Gaussian blob per (q, theta_deg, fwhm_q, fwhm_deg)."""
    r_axis = np.linspace(0.0, r_max, n_r)
    theta_axis = np.linspace(0.0, np.pi, n_theta, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r_axis, theta_axis, indexing="ij")
    intensity = np.zeros(shape + (n_r, n_theta))
    for q, theta_deg, width_q, width_deg in peaks:
        sigma_q = width_q / (2 * np.sqrt(2 * np.log(2)))
        sigma_theta = np.deg2rad(width_deg) / (2 * np.sqrt(2 * np.log(2)))
        delta = np.abs(theta_grid - np.deg2rad(theta_deg))
        delta = np.minimum(delta, np.pi - delta)
        intensity += np.exp(-0.5 * (((r_grid - q) / sigma_q) ** 2 + (delta / sigma_theta) ** 2))
    return {"intensity": intensity, "r_invA": r_grid, "theta": theta_grid}


def _widths(peak, polar_data, params):
    return measure_peak_widths(
        [peak[0]], [np.deg2rad(peak[1])], polar_data["intensity"][0, 0],
        polar_data["r_invA"][:, 0], polar_data["theta"][0, :], params=params,
    )


def test_measured_widths_recover_the_input_blob_widths():
    blob = (1.61, 30.0, 0.10, 8.0)
    width_r, width_theta = _widths(
        blob, _polar_volume([blob]), IceFlaggerParams(sharpness_baseline_quantile=0.0)
    )
    assert width_r[0] == pytest.approx(0.10, abs=0.03)
    assert width_theta[0] == pytest.approx(8.0, abs=2.0)


def test_sharpness_gate_keeps_sharp_ice_and_spares_broad_peaks():
    """Two aligned six-fold peaks: one sharp (ice), one broad (polymer)."""
    sharp = (1.61, 5.0, 0.04, 5.0)
    broad = (1.61, 65.0, 0.30, 40.0)
    polar, intensity = _cell([sharp[1], broad[1]])
    base = dict(intensity_cutoff=0.5, min_matches=2, dtheta_deg=6.0, q_target_invA=1.61)
    # Without the gate both peaks are aligned six-fold candidates and both go.
    ungated = detect_ice(polar, intensity, params=IceFlaggerParams(**base))
    assert ungated.flagged_peaks_count_map[0, 0] == 2
    # With it only the sharp one is a candidate, below min_matches=2: nothing flagged.
    gated = detect_ice(
        polar, intensity,
        params=IceFlaggerParams(**base, max_width_r_invA=0.10, max_width_theta_deg=15.0),
        polar_data=_polar_volume([sharp, broad]), return_debug=True,
    )
    assert gated.debug_records[(0, 0)].n_candidates_sharp == 1
    assert gated.flagged_peaks_count_map[0, 0] == 0


def test_annular_only_gate_keeps_dots_and_radial_streaks():
    """Ice is annularly sharp whether a dot or a radial streak (narrow theta, long q).

    A radial ceiling would reject the streak; the annular width alone keeps both and
    still rejects the annularly broad polymer arc at the same q.
    """
    params = IceFlaggerParams(max_width_theta_deg=12.0, sharpness_window_r_invA=0.30)
    for label, peak, expected in (
        ("dot", (1.61, 5.0, 0.02, 5.0), True),
        ("streak", (1.61, 5.0, 0.12, 5.0), True),
        ("arc", (1.61, 5.0, 0.02, 30.0), False),
    ):
        width_r, width_theta = _widths(peak, _polar_volume([peak]), params)
        assert bool(sharpness_mask(width_r, width_theta, params)[0]) is expected, label


def _annular_arc(fwhm_deg, *, amplitude=1.0, ring=0.0, n_r=300):
    """Arc at q=0.27, 45 deg on a 2 deg theta grid, optionally over an amorphous ring."""
    theta_axis = np.deg2rad(np.arange(0, 180, 2.0))
    r_axis = np.linspace(0.0, 1.0, n_r)
    grid_r, grid_t = np.meshgrid(r_axis, theta_axis, indexing="ij")
    fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
    delta = np.abs(grid_t - np.deg2rad(45.0))
    delta = np.minimum(delta, np.pi - delta)
    image = amplitude * np.exp(-0.5 * (((grid_r - 0.27) / (0.02 * fwhm_to_sigma)) ** 2
                                       + (delta / (np.deg2rad(fwhm_deg) * fwhm_to_sigma)) ** 2))
    image += ring * np.exp(-0.5 * ((grid_r - 0.27) / (0.05 * fwhm_to_sigma)) ** 2)
    return image, r_axis, theta_axis


def _theta_width(image, r_axis, theta_axis, params):
    return measure_peak_widths([0.27], [np.deg2rad(45.0)], image, r_axis, theta_axis,
                               params=params)[1][0]


def test_annular_width_tracks_broad_arcs_and_clamps_window():
    """The default window must not saturate: a broad arc has to read broad.

    With a 40 deg window and a 0.25 baseline quantile, 60 and 100 deg arcs measured
    ~37 and ~39 deg. A window wider than the annulus must be clamped, not repeat bins.
    """
    cases = [(f"fwhm {fwhm}", fwhm, IceFlaggerParams(), 0.15) for fwhm in (6.0, 30.0, 60.0, 100.0)]
    cases.append(("huge window", 6.0, IceFlaggerParams(sharpness_window_theta_deg=10_000.0), 0.2))
    for label, fwhm, params, rel in cases:
        image, r_axis, theta_axis = _annular_arc(fwhm, n_r=200)
        image += np.random.default_rng(0).normal(0, 0.02, image.shape)
        assert _theta_width(image, r_axis, theta_axis, params) == pytest.approx(fwhm, rel=rel), label


def test_annular_width_survives_noise():
    """A weak, diffuse arc must not measure as narrow as a sharp peak.

    A half-maximum walk stopping at the first sample below half ends early on one
    downward fluctuation; before smoothing + persistent crossings a 40 deg arc at low
    SNR measured ~10-30 deg.
    """
    for label, amplitude, noise, fwhm in [
        ("sharp and clean", 1.0, 0.01, 5.0),
        ("broad and clean", 1.0, 0.01, 40.0),
        ("broad and weak (used to read as sharp)", 0.3, 0.05, 40.0),
    ]:
        image, r_axis, theta_axis = _annular_arc(fwhm, amplitude=amplitude, ring=0.3)
        measured = [
            _theta_width(image + np.random.default_rng(seed).normal(0, noise, image.shape),
                         r_axis, theta_axis, IceFlaggerParams())
            for seed in range(8)
        ]
        assert np.median(measured) == pytest.approx(fwhm, rel=0.25), label


def test_smoothing_is_deconvolved_so_sharp_peaks_stay_unbiased():
    """The Gaussian kernel is removed in quadrature, exactly for Gaussian peaks."""
    image, r_axis, theta_axis = _annular_arc(6.0)
    smoothed = _theta_width(image, r_axis, theta_axis, IceFlaggerParams())
    raw = _theta_width(image, r_axis, theta_axis, IceFlaggerParams(
        sharpness_smooth_sigma_bins=0.0, sharpness_crossing_persistence=1))
    # Deconvolution keeps the smoothed estimate within a bin of the raw one.
    assert abs(smoothed - raw) < 2.0
