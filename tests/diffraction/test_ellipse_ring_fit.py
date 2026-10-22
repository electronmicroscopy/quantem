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


FITS = {"angular_variance": _fit, "ridge": _fit_ridge}
DIAGNOSTIC_KEYS = {
    "method", "accepted", "selected", "candidates", "explicit_band",
    "center_initial", "center_refined", "rejection_reasons",
}


def test_fit_recovers_synthetic_ellipse_and_exposes_diagnostics():
    """Ridge fitting also refines the centre, so its pattern is offset.

    ``params`` is the (a, b, theta_deg) triple the polar transforms consume, and
    ``diagnostics`` is the flat mapping ``BraggPeaksPolymer`` publishes as
    ``ellipse_fit_diagnostics``; ``preprocess`` and notebooks depend on both.
    """
    offset = (1.2, -0.8)
    cases = {
        "angular_variance": ((0.0, 0.0), 0.025, DIAGNOSTIC_KEYS | {"quality_thresholds"}),
        "ridge": (offset, 0.03, DIAGNOSTIC_KEYS),
    }
    for method, (center_offset, ratio_tol, keys) in cases.items():
        fit = FITS[method](
            _elliptical_ring(ratio_b_over_a=0.9, theta_deg=35.0, center_offset=center_offset)
        )

        assert fit.accepted is True, method
        assert fit.ratio == pytest.approx(1 / 0.9, abs=ratio_tol), method
        assert fit.theta_deg == pytest.approx(35.0, abs=3.0), method
        assert fit.ring_band[0] < 27 < fit.ring_band[1], method
        assert fit.params == (fit.a, fit.b, fit.theta_deg), method
        assert fit.ratio == pytest.approx(fit.a / fit.b), method
        diagnostics = fit.diagnostics
        assert set(diagnostics) == keys, method
        assert diagnostics["method"] == method
        assert diagnostics["accepted"] is fit.accepted, method
        assert diagnostics["center_refined"] == fit.center_refined, method
    assert fit.center_refined == pytest.approx(np.asarray((47.5, 47.5)) + offset, abs=0.25)


def test_sparse_outer_bragg_spots_do_not_select_the_calibration_band():
    pattern = _elliptical_ring(radius=26.0, ratio_b_over_a=0.94, theta_deg=118.0)
    cy, cx = (np.asarray(pattern.shape) - 1) / 2
    for angle in np.deg2rad([5, 42, 91, 147, 221, 305]):
        row = int(round(cy + 39 * np.sin(angle)))
        column = int(round(cx + 39 * np.cos(angle)))
        pattern[row - 1 : row + 2, column - 1 : column + 2] += 50.0

    for method, fitter in FITS.items():
        fit = fitter(pattern)
        assert fit.accepted is True, method
        assert fit.details["selected"]["r0"] < 32, method
        assert fit.ring_band[0] < 26 < fit.ring_band[1], method


def test_boundary_solution_is_rejected_and_refinement_is_clipped():
    fits = {}
    for method, fitter in FITS.items():
        with pytest.warns(RuntimeWarning, match="ratio search boundary"):
            fits[method] = fit = fitter(_elliptical_ring(ratio_b_over_a=0.7))

        assert fit.accepted is False, method
        assert (fit.ratio, fit.theta_deg) == pytest.approx((1.0, 0.0)), method
    selected = fits["angular_variance"].details["selected"]
    assert selected["boundary_limited"] is True
    assert 0.85 <= selected["ratio_b_over_a"] <= 1.18


def test_low_information_pattern_falls_back_to_circle():
    for method, fitter in FITS.items():
        with pytest.warns(RuntimeWarning, match="using a circular correction"):
            fit = fitter(np.ones((96, 96)))

        assert fit.method == method
        assert fit.accepted is False, method
        assert (fit.ratio, fit.theta_deg) == pytest.approx((1.0, 0.0)), method


def test_preprocess_rejects_unknown_ellipse_fit_method_before_data_access():
    detector = object.__new__(BraggPeaksPolymer)

    with pytest.raises(ValueError, match="ellipse_fit_method"):
        detector.preprocess(ellipse_fit_method="not-a-fit-method")
