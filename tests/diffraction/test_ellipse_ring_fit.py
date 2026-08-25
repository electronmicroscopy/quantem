import numpy as np
import pytest

from quantem.diffraction import BraggPeaksPolymer
from quantem.diffraction.ellipse_fitting import (
    fit_ellipse_from_ridge,
    fit_ellipse_from_ring,
)


def _elliptical_ring(
    shape=(96, 96),
    *,
    radius=27.0,
    ratio_b_over_a=0.9,
    theta_deg=35.0,
    sigma=1.8,
    center_offset=(0.0, 0.0),
):
    yy, xx = np.indices(shape, dtype=float)
    cy, cx = (np.asarray(shape) - 1) / 2
    cy += center_offset[0]
    cx += center_offset[1]
    theta = np.deg2rad(theta_deg)
    dx, dy = xx - cx, yy - cy
    major = dx * np.cos(theta) + dy * np.sin(theta)
    minor = -dx * np.sin(theta) + dy * np.cos(theta)
    elliptical_radius = np.sqrt(
        (major * ratio_b_over_a) ** 2 + minor**2
    )
    return (
        2.0
        * np.exp(-0.5 * ((elliptical_radius - radius) / sigma) ** 2)
        + 0.02
    )


def _center(pattern):
    return ((pattern.shape[0] - 1) / 2, (pattern.shape[1] - 1) / 2)


def _fit(pattern, **kwargs):
    return fit_ellipse_from_ring(
        pattern,
        _center(pattern),
        n_ratio=7,
        n_theta=12,
        max_ring_candidates=3,
        **kwargs,
    )


def _fit_ridge(pattern, **kwargs):
    return fit_ellipse_from_ridge(
        pattern,
        _center(pattern),
        num_annular_bins=120,
        **kwargs,
    )


def test_ring_fit_recovers_synthetic_ellipse():
    fit = _fit(
        _elliptical_ring(ratio_b_over_a=0.9, theta_deg=35.0)
    )

    assert fit.accepted is True
    assert fit.ratio == pytest.approx(1 / 0.9, abs=0.025)
    assert fit.theta_deg == pytest.approx(35.0, abs=3.0)
    assert fit.ring_band[0] < 27 < fit.ring_band[1]


def test_sparse_outer_bragg_spots_do_not_select_the_calibration_band():
    pattern = _elliptical_ring(
        radius=26.0, ratio_b_over_a=0.94, theta_deg=118.0
    )
    cy, cx = (np.asarray(pattern.shape) - 1) / 2
    for angle in np.deg2rad([5, 42, 91, 147, 221, 305]):
        row = int(round(cy + 39 * np.sin(angle)))
        column = int(round(cx + 39 * np.cos(angle)))
        pattern[row - 1 : row + 2, column - 1 : column + 2] += 50.0

    fit = _fit(pattern)

    assert fit.accepted is True
    assert fit.details["selected"]["r0"] < 32
    assert fit.ring_band[0] < 26 < fit.ring_band[1]


def test_boundary_solution_is_rejected_and_refinement_is_clipped():
    pattern = _elliptical_ring(ratio_b_over_a=0.7)

    with pytest.warns(RuntimeWarning, match="ratio search boundary"):
        fit = _fit(pattern)

    selected = fit.details["selected"]
    assert fit.accepted is False
    assert selected["boundary_limited"] is True
    assert 0.85 <= selected["ratio_b_over_a"] <= 1.18
    assert (fit.ratio, fit.theta_deg) == pytest.approx((1.0, 0.0))


def test_low_information_pattern_falls_back_to_circle():
    with pytest.warns(RuntimeWarning, match="using a circular correction"):
        fit = _fit(np.ones((96, 96)))

    assert fit.accepted is False
    assert (fit.ratio, fit.theta_deg) == pytest.approx((1.0, 0.0))


def test_ridge_fit_jointly_recovers_center_and_ellipse():
    offset = (1.2, -0.8)
    fit = _fit_ridge(
        _elliptical_ring(
            ratio_b_over_a=0.9,
            theta_deg=35.0,
            center_offset=offset,
        )
    )

    expected_center = np.asarray((47.5, 47.5)) + offset
    assert fit.accepted is True
    assert fit.center_refined == pytest.approx(
        expected_center, abs=0.25
    )
    assert fit.ratio == pytest.approx(1 / 0.9, abs=0.03)
    assert fit.theta_deg == pytest.approx(35.0, abs=3.0)
    assert fit.ring_band[0] < 27 < fit.ring_band[1]


def test_ridge_fit_rejects_out_of_range_ellipse():
    with pytest.warns(RuntimeWarning, match="ratio search boundary"):
        fit = _fit_ridge(
            _elliptical_ring(ratio_b_over_a=0.7)
        )

    assert fit.accepted is False
    assert (fit.ratio, fit.theta_deg) == pytest.approx((1.0, 0.0))


def test_ridge_fit_ignores_sparse_outer_bragg_spots():
    pattern = _elliptical_ring(
        radius=26.0, ratio_b_over_a=0.94, theta_deg=118.0
    )
    cy, cx = (np.asarray(pattern.shape) - 1) / 2
    for angle in np.deg2rad([5, 42, 91, 147, 221, 305]):
        row = int(round(cy + 39 * np.sin(angle)))
        column = int(round(cx + 39 * np.cos(angle)))
        pattern[row - 1 : row + 2, column - 1 : column + 2] += 50.0

    fit = _fit_ridge(pattern)

    assert fit.accepted is True
    assert fit.details["selected"]["r0"] < 32
    assert fit.ring_band[0] < 26 < fit.ring_band[1]


def test_low_information_ridge_falls_back_to_circle():
    with pytest.warns(RuntimeWarning, match="using a circular correction"):
        fit = _fit_ridge(
            np.ones((96, 96), dtype=float)
        )

    assert fit.method == "ridge"
    assert fit.accepted is False
    assert (fit.ratio, fit.theta_deg) == pytest.approx((1.0, 0.0))


def test_preprocess_rejects_unknown_ellipse_fit_method_before_data_access():
    detector = object.__new__(BraggPeaksPolymer)

    with pytest.raises(ValueError, match="ellipse_fit_method"):
        detector.preprocess(ellipse_fit_method="not-a-fit-method")


def test_fit_exposes_the_params_triple_and_legacy_diagnostics_mapping():
    """``preprocess`` and downstream notebooks depend on both of these shapes.

    ``params`` is the (a, b, theta_deg) triple the polar transforms consume, and
    ``diagnostics`` reproduces the flat mapping that ``BraggPeaksPolymer`` has
    always published as ``ellipse_fit_diagnostics``.
    """
    fit = _fit(_elliptical_ring(ratio_b_over_a=0.9, theta_deg=35.0))

    assert fit.params == (fit.a, fit.b, fit.theta_deg)
    assert fit.ratio == pytest.approx(fit.a / fit.b)

    diagnostics = fit.diagnostics
    assert set(diagnostics) == {
        "method", "accepted", "selected", "candidates", "explicit_band",
        "center_initial", "center_refined", "rejection_reasons",
        "quality_thresholds",
    }
    assert diagnostics["method"] == "angular_variance"
    assert diagnostics["accepted"] is fit.accepted
    assert diagnostics["center_refined"] == fit.center_refined


def test_ridge_diagnostics_mapping_matches_the_ridge_method():
    fit = _fit_ridge(_elliptical_ring(ratio_b_over_a=0.9, theta_deg=35.0))

    assert set(fit.diagnostics) == {
        "method", "accepted", "selected", "candidates", "explicit_band",
        "center_initial", "center_refined", "rejection_reasons",
    }
    assert fit.diagnostics["method"] == "ridge"
