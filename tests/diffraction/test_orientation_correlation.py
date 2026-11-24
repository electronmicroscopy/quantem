import numpy as np
import pytest

from quantem.diffraction.orientation_correlation import (
    calculate_orientation_correlation,
    orientation_correlation_metrics,
)


def _direct_correlation(hist, first, second, radius_max):
    """Direct sum over displacements and angle lags, linear radial binning, normalized."""
    _, size_x, size_y, num_theta = hist.shape
    radial = np.zeros((num_theta, radius_max + 1))
    for dx in range(1 - size_x, size_x):
        for dy in range(1 - size_y, size_y):
            r = np.sqrt(dx**2 + dy**2)
            if r > radius_max:
                continue
            a = hist[first, max(dx, 0) : size_x + min(dx, 0), max(dy, 0) : size_y + min(dy, 0)]
            b = hist[
                second, max(-dx, 0) : size_x + min(-dx, 0), max(-dy, 0) : size_y + min(-dy, 0)
            ]
            lags = np.array([np.sum(a * np.roll(b, lag, axis=-1)) for lag in range(num_theta)])
            low = int(r)
            radial[:, low] += (1 - (r - low)) * lags
            if r <= radius_max - 1:
                radial[:, low + 1] += (r - low) * lags
    return radial[: num_theta // 2 + 1] / (radial.sum(0) / num_theta)


def test_correlation_matches_direct_sum():
    hist = np.random.default_rng(0).random((2, 6, 5, 8))
    corr, pairs = calculate_orientation_correlation(
        hist, radius_max=4, mode_batch_size=2, dtype="float64", progress_bar=False
    )
    np.testing.assert_array_equal(pairs, [[0, 0], [0, 1], [1, 1]])
    for index, (first, second) in enumerate(pairs):
        expected = _direct_correlation(hist, first, second, radius_max=4)
        np.testing.assert_allclose(corr[index], expected, rtol=1e-10, err_msg=f"{first, second}")


def test_zero_signal_policy():
    empty = np.zeros((1, 4, 5, 8))
    nan, zero = (
        calculate_orientation_correlation(empty, zero_policy=policy, progress_bar=False)[0]
        for policy in ("nan", "zero")
    )
    assert np.isnan(nan).all()
    np.testing.assert_array_equal(zero, 0)
    with pytest.raises(ZeroDivisionError):
        calculate_orientation_correlation(empty, zero_policy="raise", progress_bar=False)


def _linear_panel(slope, num_distances=21):
    """1 + (60 + slope * d - theta) / 20: 4 at the origin, 50% level 2.5 at theta = 30."""
    theta = np.linspace(0, 90, 37)[:, None]
    return 1 + (60 + slope * np.arange(num_distances) - theta) / 20


def test_metrics_recover_intercepts_and_boundary_slope():
    """Distances are in pixels of 0.5 units; the correlation = 1 boundary is 60 + slope * d."""
    d = np.arange(101)
    turning = np.where(d <= 30, 40 + 0.5 * d, 55 - 0.8 * (d - 30))
    turning = 1 + (turning - np.linspace(0, 90, 91)[:, None]) / 40
    # The boundary reaches 0 degrees at 15 pixels; weights decay over 10% of that span.
    weights = np.exp(-np.arange(16) / 1.5)
    cases = {
        "lobe at 0": (
            _linear_panel(-4.0),
            {},
            {
                "lobe_angle_degrees": 0.0,
                "half_probability": 2.5,
                "radial_distance": 3.75,
                "annular_distance_degrees": 30.0,
                "slope_degrees_per_unit": -8.0,
                "slope_fit_r_squared": 1.0,
                "slope_fit_effective_point_count": weights.sum() ** 2 / np.sum(weights**2),
            },
        ),
        "lobe at 90": (
            _linear_panel(-4.0)[::-1],
            {},
            {
                "lobe_angle_degrees": 90.0,
                "radial_distance": 3.75,
                "annular_distance_degrees": 30.0,
                "slope_degrees_per_unit": 8.0,
            },
        ),
        "censored, unweighted": (
            _linear_panel(-0.5),
            {"slope_weight_scale": np.inf},
            {
                "radial_distance_censored": True,
                "radial_distance_lower_bound": 10.0,
                "slope_degrees_per_unit": -1.0,
                "slope_fit_effective_point_count": 21.0,
            },
        ),
        "boundary turns back": (
            turning,
            {},
            {"slope_degrees_per_unit": 1.0, "slope_fit_stop_distance": 15.0},
        ),
    }
    for name, (panel, kwargs, expected) in cases.items():
        metrics = orientation_correlation_metrics(panel[None], pixel_size=0.5, **kwargs)[0]
        for key, value in expected.items():
            assert metrics[key] == pytest.approx(value), f"{name}: {key}"
