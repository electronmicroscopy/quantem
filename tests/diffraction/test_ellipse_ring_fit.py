import numpy as np
import pytest

from quantem.diffraction import BraggPeaksPolymer


def _elliptical_ring(
    shape=(96, 96),
    *,
    radius=27.0,
    ratio_b_over_a=0.9,
    theta_deg=35.0,
    sigma=1.8,
):
    yy, xx = np.indices(shape, dtype=float)
    cy, cx = (np.asarray(shape) - 1) / 2
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


def _fit(pattern, **kwargs):
    detector = object.__new__(BraggPeaksPolymer)
    result = detector._fit_ellipse_from_ring(
        pattern,
        ((pattern.shape[0] - 1) / 2, (pattern.shape[1] - 1) / 2),
        n_ratio=7,
        n_theta=12,
        max_ring_candidates=3,
        **kwargs,
    )
    return detector, result


def test_ring_fit_recovers_synthetic_ellipse():
    detector, (a_axis, b_axis, theta, band) = _fit(
        _elliptical_ring(ratio_b_over_a=0.9, theta_deg=35.0)
    )

    assert detector.ellipse_fit_diagnostics["accepted"] is True
    assert a_axis / b_axis == pytest.approx(1 / 0.9, abs=0.025)
    assert theta == pytest.approx(35.0, abs=3.0)
    assert band[0] < 27 < band[1]


def test_sparse_outer_bragg_spots_do_not_select_the_calibration_band():
    pattern = _elliptical_ring(
        radius=26.0, ratio_b_over_a=0.94, theta_deg=118.0
    )
    cy, cx = (np.asarray(pattern.shape) - 1) / 2
    for angle in np.deg2rad([5, 42, 91, 147, 221, 305]):
        row = int(round(cy + 39 * np.sin(angle)))
        column = int(round(cx + 39 * np.cos(angle)))
        pattern[row - 1 : row + 2, column - 1 : column + 2] += 50.0

    detector, (_, _, _, band) = _fit(pattern)

    assert detector.ellipse_fit_diagnostics["accepted"] is True
    assert detector.ellipse_fit_diagnostics["selected"]["r0"] < 32
    assert band[0] < 26 < band[1]


def test_boundary_solution_is_rejected_and_refinement_is_clipped():
    pattern = _elliptical_ring(ratio_b_over_a=0.7)

    with pytest.warns(RuntimeWarning, match="ratio search boundary"):
        detector, (a_axis, b_axis, theta, _) = _fit(pattern)

    selected = detector.ellipse_fit_diagnostics["selected"]
    assert detector.ellipse_fit_diagnostics["accepted"] is False
    assert selected["boundary_limited"] is True
    assert 0.85 <= selected["ratio_b_over_a"] <= 1.18
    assert (a_axis / b_axis, theta) == pytest.approx((1.0, 0.0))


def test_low_information_pattern_falls_back_to_circle():
    with pytest.warns(RuntimeWarning, match="using a circular correction"):
        detector, (a_axis, b_axis, theta, _) = _fit(np.ones((96, 96)))

    assert detector.ellipse_fit_diagnostics["accepted"] is False
    assert (a_axis / b_axis, theta) == pytest.approx((1.0, 0.0))
