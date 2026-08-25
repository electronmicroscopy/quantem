import numpy as np
import pytest

from quantem.diffraction import BraggPeaksPolymer
from quantem.diffraction.orientation_correlation import (
    calculate_orientation_correlation,
)


def _direct_correlation_reference(orient_hist, radius_max):
    """Small full-volume implementation used only as a correctness oracle."""
    num_radii, size_x, size_y, num_theta = orient_hist.shape
    padded_x = max(2 * size_x, 2 * radius_max)
    padded_y = max(2 * size_y, 2 * radius_max)

    x = np.mod(np.arange(padded_x) + padded_x / 2, padded_x) - padded_x / 2
    y = np.mod(np.arange(padded_y) + padded_y / 2, padded_y) - padded_y / 2
    yy, xx = np.meshgrid(y, x)
    radius = np.sqrt(xx**2 + yy**2)
    lower_mask = radius <= radius_max
    upper_mask = radius <= radius_max - 1
    lower_floor = np.floor(radius[lower_mask]).astype(int)
    upper_floor = np.floor(radius[upper_mask]).astype(int)
    bins = np.concatenate((lower_floor, upper_floor + 1))
    weights = np.concatenate(
        (
            1 - (radius[lower_mask] - lower_floor),
            radius[upper_mask] - upper_floor,
        )
    )

    spectrum = np.fft.fftn(
        orient_hist,
        s=(padded_x, padded_y, num_theta),
        axes=(1, 2, 3),
    )
    pairs = [
        (first, second)
        for first in range(num_radii)
        for second in range(first, num_radii)
    ]
    output = []
    for first, second in pairs:
        spatial_angular = np.fft.ifftn(
            spectrum[first] * np.conj(spectrum[second]),
            axes=(0, 1, 2),
        ).real
        radial = np.stack(
            [
                np.bincount(
                    bins,
                    weights=weights
                    * np.concatenate(
                        (
                            spatial_angular[:, :, theta][lower_mask],
                            spatial_angular[:, :, theta][upper_mask],
                        )
                    ),
                    minlength=radius_max + 1,
                )[: radius_max + 1]
                for theta in range(num_theta)
            ]
        )
        denominator = radial.sum(axis=0) / num_theta
        output.append(
            radial[: num_theta // 2 + 1] / denominator[None, :]
        )
    return np.stack(output), np.asarray(pairs)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_streamed_correlation_matches_full_volume_reference(backend):
    histogram = np.random.default_rng(7).random(
        (3, 7, 6, 12), dtype=np.float32
    )
    expected, expected_pairs = _direct_correlation_reference(
        histogram, radius_max=4
    )

    actual, actual_pairs = calculate_orientation_correlation(
        histogram,
        radius_max=4,
        backend=backend,
        device="cpu",
        mode_batch_size=3,
        pair_batch_size=2,
        progress_bar=False,
    )

    np.testing.assert_array_equal(actual_pairs, expected_pairs)
    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=5e-6)


def test_three_dimensional_input_and_autocorrelation_pairs():
    histogram = np.random.default_rng(8).random(
        (2, 5, 4, 9), dtype=np.float32
    )

    single, single_pairs = calculate_orientation_correlation(
        histogram[0],
        backend="numpy",
        progress_bar=False,
    )
    diagonal, diagonal_pairs = calculate_orientation_correlation(
        histogram,
        pairs="autocorrelation",
        backend="numpy",
        progress_bar=False,
    )

    assert single.shape == (1, 5, 3)
    np.testing.assert_array_equal(single_pairs, [[0, 0]])
    assert diagonal.shape == (2, 5, 3)
    np.testing.assert_array_equal(diagonal_pairs, [[0, 0], [1, 1]])


@pytest.mark.parametrize(
    ("zero_policy", "expected"),
    [("nan", "nan"), ("zero", "zero")],
)
def test_empty_histogram_zero_policy(zero_policy, expected):
    output, _ = calculate_orientation_correlation(
        np.zeros((1, 4, 5, 8), dtype=np.float32),
        backend="numpy",
        zero_policy=zero_policy,
        progress_bar=False,
    )

    if expected == "nan":
        assert np.isnan(output).all()
    else:
        np.testing.assert_array_equal(output, 0)


def test_empty_histogram_raise_policy():
    with pytest.raises(ZeroDivisionError):
        calculate_orientation_correlation(
            np.zeros((1, 4, 5, 8), dtype=np.float32),
            backend="numpy",
            zero_policy="raise",
            progress_bar=False,
        )


def test_bragg_peaks_polymer_native_correlation_plot():
    detector = object.__new__(BraggPeaksPolymer)
    detector.orient_corr = np.ones((3, 5, 4), dtype=np.float32)
    detector.orient_corr_pairs = np.array([[0, 0], [0, 1], [1, 1]])

    figure, axes, metrics = detector.plot_orientation_correlation(
        pixel_size=0.25,
        pixel_units="scan pixels",
        return_metrics=True,
    )

    assert axes.shape == (1, 3)
    assert axes[0, 1].get_title() == "Correlation of Rings 0 and 1"
    assert metrics[0]["title"] == "Autocorrelation of Ring 0"
    figure.canvas.draw()


def test_correlation_plot_reports_half_probability_intercepts():
    detector = object.__new__(BraggPeaksPolymer)
    distances = np.arange(11, dtype=float)
    angles = np.linspace(0, 90, 37)
    boundary = 20 - 0.5 * distances
    panel = 1 + 9 * np.exp(-distances[None, :] / 4) * (
        boundary[None, :] - angles[:, None]
    ) / 20
    detector.orient_corr = panel[None]
    detector.orient_corr_pairs = np.array([[0, 0]])

    figure, _, metrics = detector.plot_orientation_correlation(
        pixel_size=1.0,
        pixel_units="nm",
        return_metrics=True,
    )

    assert np.isfinite(metrics[0]["radial_distance"])
    assert np.isfinite(metrics[0]["annular_distance_degrees"])
    assert metrics[0]["slope_degrees_per_unit"] == pytest.approx(-0.5, abs=0.05)
    assert metrics[0]["slope_fit_r_squared"] == pytest.approx(1.0)
    assert metrics[0]["slope_fit_point_count"] >= 2
    assert metrics[0]["slope_contour_probability"] == 1.0
    # Only the two intercept markers remain; neither the 50% contour nor the
    # correlation=1 boundary is drawn. The single line is the signed fit.
    assert len(figure.axes[0].collections) == 2
    assert len(figure.axes[0].lines) == 1
    assert len(figure.axes[0].texts) == 1


def test_correlation_plot_resolves_below_baseline_feature():
    detector = object.__new__(BraggPeaksPolymer)
    distances = np.arange(11, dtype=float)
    angles = np.linspace(0, 90, 37)
    boundary = 20 - 0.5 * distances
    panel = 1 - 0.5 * np.exp(-distances[None, :] / 4) * (
        boundary[None, :] - angles[:, None]
    ) / 20
    detector.orient_corr = panel[None]
    detector.orient_corr_pairs = np.array([[0, 1]])

    _, axes, metrics = detector.plot_orientation_correlation(
        pixel_size=1.0,
        pixel_units="nm",
        return_metrics=True,
    )

    assert np.isfinite(metrics[0]["radial_distance"])
    assert np.isfinite(metrics[0]["annular_distance_degrees"])
    assert metrics[0]["slope_degrees_per_unit"] == pytest.approx(-0.5, abs=0.05)
    assert axes[0, 0].get_legend() is not None


def test_correlation_slope_stops_before_connected_boundary_turns_back():
    detector = object.__new__(BraggPeaksPolymer)
    distances = np.arange(101, dtype=float)
    angles = np.linspace(0, 90, 91)
    boundary = np.where(
        distances <= 30,
        40 + 0.5 * distances,
        55 - 0.8 * (distances - 30),
    )
    panel = 1 + (boundary[None, :] - angles[:, None]) / 40
    detector.orient_corr = panel[None]
    detector.orient_corr_pairs = np.array([[0, 0]])

    _, _, metrics = detector.plot_orientation_correlation(
        pixel_size=1.0,
        pixel_units="nm",
        show_metrics=False,
        return_metrics=True,
    )

    assert metrics[0]["slope_degrees_per_unit"] == pytest.approx(0.5, abs=0.05)
    assert metrics[0]["slope_fit_r_squared"] > 0.95
    assert metrics[0]["slope_fit_point_count"] < len(distances) // 2


def _bimodal_panel(num_angles=91, num_distances=101):
    """Zero separation runs linearly from 0.5 at one end to 4.0 at the other.

    Built from bin indices, not degree labels, so it pins behaviour rather
    than the axis convention.
    """
    distances = np.arange(num_distances, dtype=float)
    decay = np.exp(-distances / 20)
    lobe = 1 + 3 * decay  # lobe end: 4 -> 1
    plateau = 0.5 + 0.22 * (1 - decay)  # far end: 0.5 -> 0.72, never reaches 1
    fraction = np.linspace(0.0, 1.0, num_angles)[:, None]
    return plateau[None, :] * (1 - fraction) + lobe[None, :] * fraction


def test_relative_angle_axis_spans_ninety_degrees():
    """Orientations are pi-periodic, so the retained lags stop at 90 degrees.

    Ground truth: two families offset by a known angle. The cross-correlation
    is the autocorrelation shifted by that offset, which fixes the degrees per
    lag and therefore the span of the stored axis.
    """
    size, num_theta = 32, 180
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    orientation = (
        60 * np.sin(2 * np.pi * grid_x / size)
        + 40 * np.cos(2 * np.pi * grid_y / size)
    ) % 180
    bins = np.arange(num_theta)

    def histogram(centre):
        angle = (bins[None, None, :] - centre[..., None]) * np.pi / num_theta
        weights = np.exp(8.0 * (np.cos(2 * angle) - 1.0))
        return weights / weights.sum(-1, keepdims=True)

    offset = 30.0
    stack = np.stack(
        [histogram(orientation), histogram((orientation + offset) % 180)]
    )
    correlation, pairs = calculate_orientation_correlation(
        stack, radius_max=8, backend="numpy", dtype="float64",
        progress_bar=False,
    )
    lookup = [tuple(pair) for pair in np.asarray(pairs)]
    auto = correlation[lookup.index((0, 0))][:, 0]
    cross = correlation[lookup.index((0, 1))][:, 0]

    num_lags = auto.size
    degrees_per_lag = 180.0 / num_theta
    lags = np.arange(num_lags) * degrees_per_lag
    assert lags[-1] == pytest.approx(90.0)

    def extended(degrees):
        folded = np.mod(degrees, 180.0)
        folded = np.where(folded > 90.0, 180.0 - folded, folded)
        return np.interp(folded, lags, auto)

    # The cross-correlation is the autocorrelation shifted by the true offset.
    assert cross == pytest.approx(extended(lags + offset), abs=1e-9)


def test_correlation_intercepts_follow_the_lobe_for_orthogonal_families():
    """Orthogonal families correlate at 90 degrees, not at zero.

    Their parallel cut is a shallow anticorrelated plateau that never reaches
    the half level, so measuring there returned a bare NaN radial distance
    and an annular intercept referenced to the wrong end of the axis.
    """
    detector = object.__new__(BraggPeaksPolymer)
    panel = _bimodal_panel()
    detector.orient_corr = panel[None]
    detector.orient_corr_pairs = np.array([[0, 1]])

    # The parallel cut on its own cannot resolve a half-probability crossing.
    assert panel[0, :].max() < 1 + 0.5 * (panel[0, 0] - 1)

    _, _, metrics = detector.plot_orientation_correlation(
        pixel_size=1.0,
        pixel_units="nm",
        return_metrics=True,
    )

    assert metrics[0]["lobe_angle_degrees"] == pytest.approx(90.0)
    assert metrics[0]["max_relative_angle_degrees"] == pytest.approx(90.0)
    assert metrics[0]["half_probability"] == pytest.approx(2.5)
    # Decay along the lobe: 1 + 3 exp(-d/20) reaches 2.5 at 20 ln 2.
    assert metrics[0]["radial_distance"] == pytest.approx(
        20 * np.log(2), abs=0.01
    )
    assert not metrics[0]["radial_distance_censored"]
    # Angular offset away from the lobe, not an absolute orientation.
    assert metrics[0]["annular_distance_degrees"] == pytest.approx(
        90 - 90 * (2.5 - 0.5) / 3.5, abs=0.01
    )


def test_correlation_intercepts_unchanged_when_the_lobe_is_at_zero():
    """Aligned families and autocorrelations keep the parallel reference."""
    detector = object.__new__(BraggPeaksPolymer)
    detector.orient_corr = _bimodal_panel()[::-1][None]  # lobe now at zero
    detector.orient_corr_pairs = np.array([[0, 0]])

    _, _, metrics = detector.plot_orientation_correlation(
        pixel_size=1.0,
        pixel_units="nm",
        return_metrics=True,
    )

    assert metrics[0]["lobe_angle_degrees"] == pytest.approx(0.0)
    assert metrics[0]["radial_distance"] == pytest.approx(
        20 * np.log(2), abs=0.01
    )
    assert metrics[0]["annular_distance_degrees"] == pytest.approx(
        90 - 90 * (2.5 - 0.5) / 3.5, abs=0.01
    )


def test_interior_maximum_is_not_mistaken_for_a_lobe():
    """A weakly correlated pair peaking mid-axis is noise, not a lobe.

    Selecting it produced decay lengths that swung by an order of magnitude
    between neighbouring pairs of the same scan.
    """
    detector = object.__new__(BraggPeaksPolymer)
    panel = _bimodal_panel()[::-1]  # lobe at zero: 4.0 down to 0.5
    panel[45, :] += 6.0  # spurious spike above both ends
    detector.orient_corr = panel[None]
    detector.orient_corr_pairs = np.array([[0, 1]])

    _, _, metrics = detector.plot_orientation_correlation(
        pixel_size=1.0,
        pixel_units="nm",
        show_metrics=False,
        return_metrics=True,
    )

    assert panel[:, 0].argmax() == 45  # the spike really is the maximum
    assert metrics[0]["lobe_angle_degrees"] == pytest.approx(0.0)
    assert metrics[0]["radial_distance"] == pytest.approx(
        20 * np.log(2), abs=0.01
    )


def test_unresolved_intercept_is_censored_not_silently_missing():
    """A correlation still decaying at the scan edge is bounded, not dropped."""
    detector = object.__new__(BraggPeaksPolymer)
    distances = np.arange(101, dtype=float)
    angles = np.linspace(0.0, 1.0, 91)[:, None]
    # Decays far too slowly to reach the half level inside the window.
    lobe = 1 + 3 * np.exp(-distances / 5000)
    panel = lobe[None, :] * (1 - angles) + np.ones_like(lobe)[None, :] * angles
    detector.orient_corr = panel[None]
    detector.orient_corr_pairs = np.array([[0, 0]])

    _, _, metrics = detector.plot_orientation_correlation(
        pixel_size=2.0,
        pixel_units="nm",
        return_metrics=True,
    )

    assert not np.isfinite(metrics[0]["radial_distance"])
    assert metrics[0]["radial_distance_censored"]
    assert metrics[0]["radial_distance_lower_bound"] == pytest.approx(200.0)
