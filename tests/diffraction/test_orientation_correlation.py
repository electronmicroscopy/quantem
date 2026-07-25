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
    angles = np.linspace(0, 180, 37)
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
    angles = np.linspace(0, 180, 37)
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
    angles = np.linspace(0, 180, 91)
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
