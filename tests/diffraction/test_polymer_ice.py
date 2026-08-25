from __future__ import annotations

import numpy as np
import pytest

from quantem.core.datastructures import Vector
from quantem.diffraction import (IceFlaggerParams, detect_ice,
                                 measure_peak_widths, sharpness_mask)


def _vectors(shape=(1, 2)):
    polar = Vector.from_shape(
        shape=shape, fields=["r_invA", "theta"], units=["1/A", "rad"]
    )
    intensity = Vector.from_shape(
        shape=shape, fields=["intensities"], units=["normalized"]
    )
    return polar, intensity


def test_detect_ice_and_filter_does_not_mutate_source():
    polar, intensity = _vectors()
    polar[0, 0] = np.column_stack(
        [np.full(3, 1.61), np.deg2rad([5, 65, 140])]
    )
    intensity[0, 0] = np.array([[0.9], [0.8], [0.7]])
    polar[0, 1] = np.empty((0, 2))
    intensity[0, 1] = np.empty((0, 1))
    original = polar[0, 0].array.copy()

    result = detect_ice(
        polar,
        intensity,
        params=IceFlaggerParams(
            intensity_cutoff=0.5, min_matches=2, dtheta_deg=6
        ),
        return_debug=True,
    )
    assert result.threshold == 0.5
    assert result.flagged_peaks_count_map.tolist() == [[2, 0]]
    filtered = result.filter(polar)
    np.testing.assert_array_equal(polar[0, 0].array, original)
    assert len(filtered[0, 0].array) == 1
    assert (0, 1) in result.debug_records


def test_masked_cells_are_not_analyzed():
    polar, intensity = _vectors((1, 1))
    polar[0, 0] = np.column_stack(
        [np.full(2, 1.61), np.deg2rad([0, 60])]
    )
    intensity[0, 0] = np.ones((2, 1))
    result = detect_ice(
        polar,
        intensity,
        params=IceFlaggerParams(intensity_cutoff=0.0),
        scan_mask=np.zeros((1, 1), dtype=bool),
    )
    assert result.flagged_peaks_count_map[0, 0] == 0


def test_misaligned_ragged_vectors_fail_clearly():
    polar, intensity = _vectors((1, 1))
    polar[0, 0] = np.zeros((2, 2))
    intensity[0, 0] = np.zeros((1, 1))
    with pytest.raises(ValueError, match="Row count mismatch"):
        detect_ice(
            polar,
            intensity,
            params=IceFlaggerParams(intensity_cutoff=0.0),
        )


def _polar_volume(peaks, *, shape=(1, 1), n_r=600, n_theta=180, r_max=3.0):
    """Polar volume with a Gaussian blob per (q, theta_deg, width_q, width_deg)."""
    r_axis = np.linspace(0.0, r_max, n_r)
    theta_axis = np.linspace(0.0, np.pi, n_theta, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r_axis, theta_axis, indexing="ij")
    intensity = np.zeros(shape + (n_r, n_theta))
    for q, theta_deg, width_q, width_deg in peaks:
        # width_* are FWHM; convert to the Gaussian sigma that produces them.
        sigma_q = width_q / (2 * np.sqrt(2 * np.log(2)))
        sigma_theta = np.deg2rad(width_deg) / (2 * np.sqrt(2 * np.log(2)))
        delta = np.abs(theta_grid - np.deg2rad(theta_deg))
        delta = np.minimum(delta, np.pi - delta)
        intensity += np.exp(
            -0.5 * (((r_grid - q) / sigma_q) ** 2 + (delta / sigma_theta) ** 2)
        )
    return {"intensity": intensity, "r_invA": r_grid, "theta": theta_grid}


def test_measured_widths_recover_the_input_blob_widths():
    from quantem.diffraction.polymer_ice import measure_peak_widths

    polar_data = _polar_volume([(1.61, 30.0, 0.10, 8.0)])
    width_r, width_theta = measure_peak_widths(
        [1.61],
        [np.deg2rad(30.0)],
        polar_data["intensity"][0, 0],
        polar_data["r_invA"][:, 0],
        polar_data["theta"][0, :],
        params=IceFlaggerParams(sharpness_baseline_quantile=0.0),
    )
    assert width_r[0] == pytest.approx(0.10, abs=0.03)
    assert width_theta[0] == pytest.approx(8.0, abs=2.0)


def test_sharpness_gate_keeps_sharp_ice_and_spares_broad_peaks():
    """Two aligned six-fold peaks: one sharp (ice), one broad (polymer)."""
    sharp = (1.61, 5.0, 0.04, 5.0)
    broad = (1.61, 65.0, 0.30, 40.0)
    polar_data = _polar_volume([sharp, broad])
    polar, intensity = _vectors((1, 1))
    polar[0, 0] = np.column_stack([[sharp[0], broad[0]], np.deg2rad([sharp[1], broad[1]])])
    intensity[0, 0] = np.array([[0.9], [0.9]])

    base = dict(intensity_cutoff=0.5, min_matches=2, dtheta_deg=6.0, q_target_invA=1.61)
    # Without the gate both peaks are aligned six-fold candidates and both go.
    ungated = detect_ice(polar, intensity, params=IceFlaggerParams(**base))
    assert ungated.flagged_peaks_count_map[0, 0] == 2

    # With the gate only the sharp one survives as a candidate, and a single
    # candidate no longer reaches min_matches=2, so nothing is flagged.
    gated = detect_ice(
        polar,
        intensity,
        params=IceFlaggerParams(
            **base, max_width_r_invA=0.10, max_width_theta_deg=15.0
        ),
        polar_data=polar_data,
        return_debug=True,
    )
    assert gated.debug_records[(0, 0)].n_candidates_sharp == 1
    assert gated.flagged_peaks_count_map[0, 0] == 0


def test_annular_only_gate_keeps_dots_and_radial_streaks():
    """Ice is annularly sharp whether it is a dot or a radial streak.

    A radial streak is narrow in theta and extended in q, so a radial ceiling
    would reject it; gating on the annular width alone keeps both ice shapes and
    still rejects the annularly broad polymer arc at the same q.
    """
    dot = (1.61, 5.0, 0.02, 5.0)
    streak = (1.61, 5.0, 0.12, 5.0)
    arc = (1.61, 5.0, 0.02, 30.0)
    params = IceFlaggerParams(max_width_theta_deg=12.0, sharpness_window_r_invA=0.30)
    for label, (q, theta_deg, width_q, width_deg), expected in (
        ("dot", dot, True), ("streak", streak, True), ("arc", arc, False)
    ):
        polar_data = _polar_volume([(q, theta_deg, width_q, width_deg)])
        width_r, width_theta = measure_peak_widths(
            [q], [np.deg2rad(theta_deg)], polar_data["intensity"][0, 0],
            polar_data["r_invA"][:, 0], polar_data["theta"][0, :], params=params,
        )
        assert bool(sharpness_mask(width_r, width_theta, params)[0]) is expected, label


def test_sharpness_ceiling_without_polar_data_fails_clearly():
    polar, intensity = _vectors((1, 1))
    polar[0, 0] = np.column_stack([[1.61], [0.0]])
    intensity[0, 0] = np.array([[0.9]])
    with pytest.raises(ValueError, match="requires polar_data"):
        detect_ice(
            polar,
            intensity,
            params=IceFlaggerParams(intensity_cutoff=0.5, max_width_r_invA=0.1),
        )


def _two_lattice_pattern(phi_a=3.0, phi_b=31.0, q=1.61):
    """Two six-fold lattices at unrelated orientations, overlaid in one pattern."""
    angles = [phi_a + 60.0 * k for k in range(6)] + [phi_b + 60.0 * k for k in range(6)]
    polar, intensity = _vectors((1, 1))
    polar[0, 0] = np.column_stack([np.full(len(angles), q), np.deg2rad(angles)])
    intensity[0, 0] = np.full((len(angles), 1), 0.9)
    return polar, intensity


def test_single_crystallite_default_finds_only_the_strongest_lattice():
    polar, intensity = _two_lattice_pattern()
    result = detect_ice(
        polar,
        intensity,
        params=IceFlaggerParams(intensity_cutoff=0.5, min_matches=3, dtheta_deg=6.0),
        return_debug=True,
    )
    # 12 peaks present, only one lattice's 6 claimed.
    assert result.flagged_peaks_count_map[0, 0] == 6
    assert len(result.debug_records[(0, 0)].phi_deg) == 1


def test_max_crystallites_claims_both_lattices():
    polar, intensity = _two_lattice_pattern(phi_a=3.0, phi_b=31.0)
    result = detect_ice(
        polar,
        intensity,
        params=IceFlaggerParams(
            intensity_cutoff=0.5, min_matches=3, dtheta_deg=6.0, max_crystallites=3
        ),
        return_debug=True,
    )
    assert result.flagged_peaks_count_map[0, 0] == 12
    found = sorted(result.debug_records[(0, 0)].phi_deg)
    assert len(found) == 2
    assert found[0] == pytest.approx(3.0, abs=0.5)
    assert found[1] == pytest.approx(31.0, abs=0.5)


def test_peel_stops_instead_of_splitting_one_lattice_in_two():
    """A single lattice must not be re-found as a near-duplicate crystallite."""
    polar, intensity = _vectors((1, 1))
    angles = [5.0 + 60.0 * k for k in range(6)]
    polar[0, 0] = np.column_stack([np.full(6, 1.61), np.deg2rad(angles)])
    intensity[0, 0] = np.full((6, 1), 0.9)
    result = detect_ice(
        polar,
        intensity,
        params=IceFlaggerParams(
            intensity_cutoff=0.5, min_matches=3, dtheta_deg=6.0, max_crystallites=5
        ),
        return_debug=True,
    )
    assert result.flagged_peaks_count_map[0, 0] == 6
    assert len(result.debug_records[(0, 0)].phi_deg) == 1


def test_min_phi_separation_rejects_a_too_close_second_lattice():
    # 8 degrees apart: separable at the default (dtheta_deg=6), not at 15.
    polar, intensity = _two_lattice_pattern(phi_a=3.0, phi_b=11.0)
    common = dict(intensity_cutoff=0.5, min_matches=3, dtheta_deg=3.0, max_crystallites=3)
    both = detect_ice(polar, intensity, params=IceFlaggerParams(**common), return_debug=True)
    merged = detect_ice(
        polar,
        intensity,
        params=IceFlaggerParams(**common, min_phi_separation_deg=15.0),
        return_debug=True,
    )
    assert len(both.debug_records[(0, 0)].phi_deg) == 2
    assert len(merged.debug_records[(0, 0)].phi_deg) == 1


def test_matched_bins_accumulate_across_crystallites():
    polar, intensity = _two_lattice_pattern()
    result = detect_ice(
        polar,
        intensity,
        params=IceFlaggerParams(
            intensity_cutoff=0.5, min_matches=3, dtheta_deg=6.0, max_crystallites=3
        ),
        return_debug=True,
    )
    # Six bins per lattice, two lattices.
    assert result.matched_bins_count_map[0, 0] == 12


def test_folded_theta_matches_a_friedel_pair_as_one_lattice():
    """With two-fold folding, theta and theta+180 are the same angle.

    The pair must still be flagged; before the period was honoured it counted as a
    single bin and was rejected by min_matches=2.
    """
    polar, intensity = _vectors((1, 1))
    # As process_polar(two_fold_symmetry=True) would deliver them: both folded to 5 deg.
    polar[0, 0] = np.column_stack([np.full(2, 1.61), np.deg2rad([5.0, 5.0])])
    intensity[0, 0] = np.full((2, 1), 0.9)
    result = detect_ice(
        polar,
        intensity,
        params=IceFlaggerParams(intensity_cutoff=0.5, min_matches=1, dtheta_deg=6.0),
        theta_period_deg=180.0,
    )
    assert result.flagged_peaks_count_map[0, 0] == 2


def test_folded_theta_reaches_all_three_arms():
    """Three arms 60 deg apart are all reachable on a 180 deg period."""
    polar, intensity = _vectors((1, 1))
    polar[0, 0] = np.column_stack([np.full(3, 1.61), np.deg2rad([5.0, 65.0, 125.0])])
    intensity[0, 0] = np.full((3, 1), 0.9)
    params = IceFlaggerParams(intensity_cutoff=0.5, min_matches=3, dtheta_deg=6.0)
    folded = detect_ice(polar, intensity, params=params, theta_period_deg=180.0,
                        return_debug=True)
    assert folded.flagged_peaks_count_map[0, 0] == 3
    assert len(folded.debug_records[(0, 0)].matched_bins) == 3


def test_unsatisfiable_min_matches_on_folded_theta_is_rejected():
    polar, intensity = _vectors((1, 1))
    polar[0, 0] = np.column_stack([np.full(2, 1.61), np.deg2rad([5.0, 65.0])])
    intensity[0, 0] = np.full((2, 1), 0.9)
    with pytest.raises(ValueError, match="can never be reached"):
        detect_ice(
            polar,
            intensity,
            params=IceFlaggerParams(intensity_cutoff=0.5, min_matches=5),
            theta_period_deg=180.0,
        )


def test_params_theta_period_overrides_the_caller():
    polar, intensity = _vectors((1, 1))
    polar[0, 0] = np.column_stack([np.full(2, 1.61), np.deg2rad([5.0, 5.0])])
    intensity[0, 0] = np.full((2, 1), 0.9)
    # Caller says folded, params insist on the full circle: params win, so min_matches=5
    # becomes reachable in principle and no error is raised.
    result = detect_ice(
        polar,
        intensity,
        params=IceFlaggerParams(intensity_cutoff=0.5, min_matches=5, theta_period_deg=360.0),
        theta_period_deg=180.0,
    )
    assert result.flagged_peaks_count_map[0, 0] == 0


def _folded(angles_deg, intensity=0.9):
    polar, intensity_vec = _vectors((1, 1))
    polar[0, 0] = np.column_stack(
        [np.full(len(angles_deg), 1.61), np.deg2rad(angles_deg)]
    )
    intensity_vec[0, 0] = np.full((len(angles_deg), 1), intensity)
    return polar, intensity_vec


def test_min_peaks_per_arm_rejects_a_lone_peak_but_keeps_a_friedel_pair():
    """On a folded axis a Friedel pair is two peaks on one arm; a lone peak is one."""
    common = dict(intensity_cutoff=0.5, dtheta_deg=6.0, min_matches=1, min_peaks_per_arm=2)

    lone_polar, lone_int = _folded([5.0])
    lone = detect_ice(lone_polar, lone_int, params=IceFlaggerParams(**common),
                      theta_period_deg=180.0)
    assert lone.flagged_peaks_count_map[0, 0] == 0

    # theta and theta+180 both fold to 5 deg.
    pair_polar, pair_int = _folded([5.0, 5.0])
    pair = detect_ice(pair_polar, pair_int, params=IceFlaggerParams(**common),
                      theta_period_deg=180.0)
    assert pair.flagged_peaks_count_map[0, 0] == 2


def test_min_peaks_per_arm_default_is_unchanged_behaviour():
    polar, intensity = _folded([5.0])
    result = detect_ice(
        polar, intensity,
        params=IceFlaggerParams(intensity_cutoff=0.5, dtheta_deg=6.0, min_matches=1),
        theta_period_deg=180.0)
    assert result.flagged_peaks_count_map[0, 0] == 1


def test_arms_below_the_peak_floor_are_dropped_not_just_uncounted():
    """An under-populated arm must not contribute its peaks to the flagged set."""
    # Arm A (5 deg) has a pair, arm B (65 deg) has a single peak.
    polar, intensity = _folded([5.0, 5.0, 65.0])
    result = detect_ice(
        polar, intensity,
        params=IceFlaggerParams(intensity_cutoff=0.5, dtheta_deg=6.0,
                                min_matches=1, min_peaks_per_arm=2),
        theta_period_deg=180.0, return_debug=True)
    assert result.flagged_peaks_count_map[0, 0] == 2          # the pair only
    assert result.debug_records[(0, 0)].matched_bins == [0]   # arm B dropped


def test_min_peaks_per_arm_combines_with_min_matches():
    # Two arms, each a Friedel pair -> 2 arms of 2 peaks.
    polar, intensity = _folded([5.0, 5.0, 65.0, 65.0])
    ok = detect_ice(
        polar, intensity,
        params=IceFlaggerParams(intensity_cutoff=0.5, dtheta_deg=6.0,
                                min_matches=2, min_peaks_per_arm=2),
        theta_period_deg=180.0)
    assert ok.flagged_peaks_count_map[0, 0] == 4
    # Same peaks, but demanding three populated arms: nothing qualifies.
    strict = detect_ice(
        polar, intensity,
        params=IceFlaggerParams(intensity_cutoff=0.5, dtheta_deg=6.0,
                                min_matches=3, min_peaks_per_arm=2),
        theta_period_deg=180.0)
    assert strict.flagged_peaks_count_map[0, 0] == 0


def _folded_with_unfolded(pairs):
    """pairs: list of (folded_deg, unfolded_deg) as polar_transform_peaks records them."""
    polar = Vector.from_shape(
        shape=(1, 1),
        fields=["r_invA", "theta", "theta_unfolded"],
        units=["1/A", "rad", "rad"],
    )
    intensity = Vector.from_shape(shape=(1, 1), fields=["intensities"], units=["normalized"])
    polar[0, 0] = np.column_stack([
        np.full(len(pairs), 1.61),
        np.deg2rad([f for f, _ in pairs]),
        np.deg2rad([u for _, u in pairs]),
    ])
    intensity[0, 0] = np.full((len(pairs), 1), 0.9)
    return polar, intensity


def test_require_friedel_pair_distinguishes_a_true_pair_from_two_near_peaks():
    """min_peaks_per_arm cannot tell these apart; the unfolded angle can."""
    common = dict(intensity_cutoff=0.5, dtheta_deg=6.0, min_matches=1,
                  min_peaks_per_arm=2, require_friedel_pair=True)

    # Genuinely opposed: 5 and 185 deg, both folding to 5.
    true_pair, ints = _folded_with_unfolded([(5.0, 5.0), (5.0, 185.0)])
    assert detect_ice(true_pair, ints, params=IceFlaggerParams(**common),
                      theta_period_deg=180.0).flagged_peaks_count_map[0, 0] == 2

    # Two peaks on the same side, 3 deg apart: same arm, same folded angle, not a pair.
    near, ints2 = _folded_with_unfolded([(5.0, 5.0), (8.0, 8.0)])
    assert detect_ice(near, ints2, params=IceFlaggerParams(**common),
                      theta_period_deg=180.0).flagged_peaks_count_map[0, 0] == 0

    # Without the strict test, min_peaks_per_arm=2 accepts the near pair.
    loose = dict(common, require_friedel_pair=False)
    assert detect_ice(near, ints2, params=IceFlaggerParams(**loose),
                      theta_period_deg=180.0).flagged_peaks_count_map[0, 0] == 2


def test_require_friedel_pair_without_the_field_fails_clearly():
    polar, intensity = _folded([5.0, 5.0])           # no theta_unfolded column
    with pytest.raises(ValueError, match="theta_unfolded"):
        detect_ice(
            polar, intensity,
            params=IceFlaggerParams(intensity_cutoff=0.5, require_friedel_pair=True),
            theta_period_deg=180.0)


def test_unfolded_field_is_optional_when_not_required():
    """Vectors predating theta_unfolded still work for everything else."""
    polar, intensity = _folded([5.0, 5.0])
    result = detect_ice(
        polar, intensity,
        params=IceFlaggerParams(intensity_cutoff=0.5, dtheta_deg=6.0,
                                min_matches=1, min_peaks_per_arm=2),
        theta_period_deg=180.0)
    assert result.flagged_peaks_count_map[0, 0] == 2


def _annular_arc(fwhm_deg, q=0.27, theta_deg=45.0, n_theta=90, n_r=200, seed=0):
    theta_axis = np.deg2rad(np.arange(0, 180, 180 / n_theta))
    r_axis = np.linspace(0.0, 1.0, n_r)
    grid_r, grid_t = np.meshgrid(r_axis, theta_axis, indexing="ij")
    sigma_q = 0.02 / (2 * np.sqrt(2 * np.log(2)))
    sigma_t = np.deg2rad(fwhm_deg) / (2 * np.sqrt(2 * np.log(2)))
    delta = np.abs(grid_t - np.deg2rad(theta_deg))
    delta = np.minimum(delta, np.pi - delta)
    image = np.exp(-0.5 * (((grid_r - q) / sigma_q) ** 2 + (delta / sigma_t) ** 2))
    image += np.random.default_rng(seed).normal(0, 0.02, image.shape)
    return image, r_axis, theta_axis, q, theta_deg


@pytest.mark.parametrize("true_fwhm", [6.0, 30.0, 60.0, 100.0])
def test_annular_width_tracks_broad_arcs_not_just_sharp_ones(true_fwhm):
    """The default window must not saturate: a broad arc has to read broad.

    With a 40 degree window and a 0.25 baseline quantile, a 60 degree arc measured
    ~37 degrees and a 100 degree arc ~39 -- both look sharper than a real threshold.
    """
    from quantem.diffraction.polymer_ice import measure_peak_widths

    image, r_axis, theta_axis, q, theta_deg = _annular_arc(true_fwhm)
    _, width_theta = measure_peak_widths(
        [q], [np.deg2rad(theta_deg)], image, r_axis, theta_axis,
        params=IceFlaggerParams(),
    )
    assert width_theta[0] == pytest.approx(true_fwhm, rel=0.15)


def test_annular_window_cannot_wrap_onto_itself():
    """A window wider than the annulus must be clamped, not allowed to repeat bins."""
    from quantem.diffraction.polymer_ice import measure_peak_widths

    image, r_axis, theta_axis, q, theta_deg = _annular_arc(6.0)
    _, width_theta = measure_peak_widths(
        [q], [np.deg2rad(theta_deg)], image, r_axis, theta_axis,
        params=IceFlaggerParams(sharpness_window_theta_deg=10_000.0),
    )
    assert width_theta[0] == pytest.approx(6.0, rel=0.2)


@pytest.mark.parametrize("amplitude,noise,true_fwhm", [
    (1.0, 0.01, 5.0),    # sharp and clean
    (1.0, 0.01, 40.0),   # broad and clean
    (0.3, 0.05, 40.0),   # broad and weak -- the case that used to read as sharp
])
def test_annular_width_survives_noise(amplitude, noise, true_fwhm):
    """A weak, diffuse arc must not measure as narrow as a sharp peak.

    A half-maximum walk that stops at the first sample below half is ended early by
    one downward fluctuation, so before smoothing + persistent crossings a true 40
    degree arc at low SNR measured ~10-30 degrees.
    """
    from quantem.diffraction.polymer_ice import measure_peak_widths

    n_theta = 90
    theta_axis = np.deg2rad(np.arange(0, 180, 2.0))
    r_axis = np.linspace(0.0, 1.0, 300)
    grid_r, grid_t = np.meshgrid(r_axis, theta_axis, indexing="ij")
    delta = np.abs(grid_t - np.deg2rad(45.0))
    delta = np.minimum(delta, np.pi - delta)
    sigma_t = np.deg2rad(true_fwhm) / 2.3548
    image = amplitude * np.exp(
        -0.5 * (((grid_r - 0.27) / (0.02 / 2.3548)) ** 2 + (delta / sigma_t) ** 2)
    )
    image += 0.3 * np.exp(-0.5 * ((grid_r - 0.27) / (0.05 / 2.3548)) ** 2)  # amorphous ring

    measured = []
    for seed in range(8):
        noisy = image + np.random.default_rng(seed).normal(0, noise, image.shape)
        _, width_theta = measure_peak_widths(
            [0.27], [np.deg2rad(45.0)], noisy, r_axis, theta_axis,
            params=IceFlaggerParams(),
        )
        measured.append(width_theta[0])
    assert np.median(measured) == pytest.approx(true_fwhm, rel=0.25)


def test_smoothing_is_deconvolved_so_sharp_peaks_stay_unbiased():
    """The Gaussian kernel is removed in quadrature, exactly for Gaussian peaks."""
    from quantem.diffraction.polymer_ice import measure_peak_widths

    theta_axis = np.deg2rad(np.arange(0, 180, 2.0))
    r_axis = np.linspace(0.0, 1.0, 300)
    grid_r, grid_t = np.meshgrid(r_axis, theta_axis, indexing="ij")
    delta = np.abs(grid_t - np.deg2rad(45.0))
    delta = np.minimum(delta, np.pi - delta)
    image = np.exp(-0.5 * (((grid_r - 0.27) / (0.02 / 2.3548)) ** 2
                           + (delta / (np.deg2rad(6.0) / 2.3548)) ** 2))
    smoothed, raw = (
        measure_peak_widths([0.27], [np.deg2rad(45.0)], image, r_axis, theta_axis,
                            params=p)[1][0]
        for p in (IceFlaggerParams(),
                  IceFlaggerParams(sharpness_smooth_sigma_bins=0.0,
                                   sharpness_crossing_persistence=1))
    )
    # Deconvolution keeps the smoothed estimate within a bin of the raw one.
    assert abs(smoothed - raw) < 2.0


@pytest.mark.parametrize("kwargs,match", [
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
])
def test_invalid_params_are_rejected_at_construction(kwargs, match):
    """A bad value here would otherwise surface as 'nothing was flagged'."""
    with pytest.raises(ValueError, match=match):
        IceFlaggerParams(**kwargs)


def test_valid_edge_values_are_accepted():
    IceFlaggerParams(
        max_width_r_invA=None, max_width_theta_deg=None,
        sharpness_baseline_quantile=0.0, sharpness_smooth_sigma_bins=0.0,
        min_matches=1, min_peaks_per_arm=1, max_crystallites=1,
    )
