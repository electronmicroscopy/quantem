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
    pairs = [(first, second) for first in range(num_radii) for second in range(first, num_radii)]
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
        output.append(radial[: num_theta // 2 + 1] / denominator[None, :])
    return np.stack(output), np.asarray(pairs)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_streamed_correlation_matches_full_volume_reference(backend):
    histogram = np.random.default_rng(7).random((3, 7, 6, 12), dtype=np.float32)
    expected, expected_pairs = _direct_correlation_reference(histogram, radius_max=4)

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
    histogram = np.random.default_rng(8).random((2, 5, 4, 9), dtype=np.float32)

    single, single_pairs = calculate_orientation_correlation(
        histogram[0], backend="numpy", progress_bar=False
    )
    diagonal, diagonal_pairs = calculate_orientation_correlation(
        histogram, pairs="autocorrelation", backend="numpy", progress_bar=False
    )

    assert single.shape == (1, 5, 3)
    np.testing.assert_array_equal(single_pairs, [[0, 0]])
    assert diagonal.shape == (2, 5, 3)
    np.testing.assert_array_equal(diagonal_pairs, [[0, 0], [1, 1]])


def test_empty_histogram_zero_policy():
    def correlate(zero_policy):
        return calculate_orientation_correlation(
            np.zeros((1, 4, 5, 8), dtype=np.float32),
            backend="numpy",
            zero_policy=zero_policy,
            progress_bar=False,
        )[0]

    assert np.isnan(correlate("nan")).all()
    np.testing.assert_array_equal(correlate("zero"), 0)
    with pytest.raises(ZeroDivisionError):
        correlate("raise")


def _plot(panels, pairs, pixel_size=1.0, **kwargs):
    """Plot a hand-built correlation stack; returns (figure, axes, metrics)."""
    detector = object.__new__(BraggPeaksPolymer)
    detector.orient_corr = np.asarray(panels)
    detector.orient_corr_pairs = np.array(pairs)
    return detector.plot_orientation_correlation(
        pixel_size=pixel_size, pixel_units="nm", return_metrics=True, **kwargs
    )


def test_bragg_peaks_polymer_native_correlation_plot():
    figure, axes, metrics = _plot(
        np.ones((3, 5, 4), dtype=np.float32),
        [[0, 0], [0, 1], [1, 1]],
        pixel_size=0.25,
    )

    assert axes.shape == (1, 3)
    assert axes[0, 1].get_title() == "Correlation of Rings 0 and 1"
    assert metrics[0]["title"] == "Autocorrelation of Ring 0"
    figure.canvas.draw()


def test_correlation_plot_reports_half_probability_intercepts():
    """A linear 50% boundary of slope -0.5 deg/nm, above and below baseline."""
    distances = np.arange(11, dtype=float)
    angles = np.linspace(0, 90, 37)
    shape = (
        np.exp(-distances[None, :] / 4) * (20 - 0.5 * distances[None, :] - angles[:, None]) / 20
    )
    results = {
        "above": _plot((1 + 9 * shape)[None], [[0, 0]]),
        "below": _plot((1 - 0.5 * shape)[None], [[0, 1]]),
    }
    for name, (_, axes, metrics) in results.items():
        m = metrics[0]
        assert np.isfinite(m["radial_distance"]), name
        assert np.isfinite(m["annular_distance_degrees"]), name
        assert m["slope_degrees_per_unit"] == pytest.approx(-0.5, abs=0.05), name
        assert axes[0, 0].get_legend() is not None, name

    figure, _, metrics = results["above"]
    m = metrics[0]
    assert m["slope_fit_r_squared"] == pytest.approx(1.0)
    assert m["slope_fit_point_count"] >= 2
    assert m["slope_contour_probability"] == 1.0
    # Only the two intercept markers remain; neither the 50% contour nor the
    # correlation=1 boundary is drawn. The single line is the signed fit.
    assert len(figure.axes[0].collections) == 2
    assert len(figure.axes[0].lines) == 1
    assert len(figure.axes[0].texts) == 1


def test_correlation_slope_stops_before_connected_boundary_turns_back():
    distances = np.arange(101, dtype=float)
    angles = np.linspace(0, 90, 91)
    boundary = np.where(distances <= 30, 40 + 0.5 * distances, 55 - 0.8 * (distances - 30))
    panel = 1 + (boundary[None, :] - angles[:, None]) / 40

    _, _, metrics = _plot(panel[None], [[0, 0]], show_metrics=False)

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

    Two families offset by a known angle: the cross-correlation is the
    autocorrelation shifted by that offset, which fixes the degrees per lag
    and therefore the span of the stored axis.
    """
    size, num_theta = 32, 180
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    orientation = (
        60 * np.sin(2 * np.pi * grid_x / size) + 40 * np.cos(2 * np.pi * grid_y / size)
    ) % 180
    bins = np.arange(num_theta)

    def histogram(centre):
        angle = (bins[None, None, :] - centre[..., None]) * np.pi / num_theta
        weights = np.exp(8.0 * (np.cos(2 * angle) - 1.0))
        return weights / weights.sum(-1, keepdims=True)

    offset = 30.0
    stack = np.stack([histogram(orientation), histogram((orientation + offset) % 180)])
    correlation, pairs = calculate_orientation_correlation(
        stack, radius_max=8, backend="numpy", dtype="float64", progress_bar=False
    )
    lookup = [tuple(pair) for pair in np.asarray(pairs)]
    auto = correlation[lookup.index((0, 0))][:, 0]
    cross = correlation[lookup.index((0, 1))][:, 0]

    lags = np.arange(auto.size) * 180.0 / num_theta
    assert lags[-1] == pytest.approx(90.0)

    # The cross-correlation is the autocorrelation shifted by the true offset.
    folded = np.mod(lags + offset, 180.0)
    folded = np.where(folded > 90.0, 180.0 - folded, folded)
    assert cross == pytest.approx(np.interp(folded, lags, auto), abs=1e-9)


def test_correlation_intercepts_follow_the_lobe():
    """The half-probability intercepts are measured along the correlated lobe.

    - orthogonal: families correlate at 90 degrees; the parallel cut is a
      shallow plateau that never reaches the half level, which used to give a
      bare NaN radial distance and an annular intercept from the wrong end.
    - aligned: lobe at zero keeps the parallel reference.
    - spike: a mid-axis maximum is noise, not a lobe; selecting it made decay
      lengths swing by an order of magnitude between neighbouring pairs.
    """
    radial = pytest.approx(20 * np.log(2), abs=0.01)  # 1 + 3 exp(-d/20) = 2.5
    # Angular offset away from the lobe, not an absolute orientation.
    annular = pytest.approx(90 - 90 * (2.5 - 0.5) / 3.5, abs=0.01)
    orthogonal = _bimodal_panel()
    spike = _bimodal_panel()[::-1].copy()
    spike[45, :] += 6.0  # spurious spike above both ends
    cases = {
        "orthogonal": (
            orthogonal,
            [[0, 1]],
            {},
            {
                "lobe_angle_degrees": pytest.approx(90.0),
                "max_relative_angle_degrees": pytest.approx(90.0),
                "half_probability": pytest.approx(2.5),
                "radial_distance": radial,
                "radial_distance_censored": False,
                "annular_distance_degrees": annular,
            },
        ),
        "aligned": (
            _bimodal_panel()[::-1],
            [[0, 0]],
            {},
            {
                "lobe_angle_degrees": pytest.approx(0.0),
                "radial_distance": radial,
                "annular_distance_degrees": annular,
            },
        ),
        "spike": (
            spike,
            [[0, 1]],
            {"show_metrics": False},
            {
                "lobe_angle_degrees": pytest.approx(0.0),
                "radial_distance": radial,
            },
        ),
    }
    # Preconditions: the orthogonal parallel cut cannot resolve a crossing,
    # and the spike really is the maximum.
    assert orthogonal[0, :].max() < 1 + 0.5 * (orthogonal[0, 0] - 1)
    assert spike[:, 0].argmax() == 45

    for name, (panel, pairs, kwargs, expected) in cases.items():
        metrics = _plot(panel[None], pairs, **kwargs)[2][0]
        for key, value in expected.items():
            assert metrics[key] == value, f"{name}: {key}"


def test_unresolved_intercept_is_censored_not_silently_missing():
    """A correlation still decaying at the scan edge is bounded, not dropped."""
    distances = np.arange(101, dtype=float)
    angles = np.linspace(0.0, 1.0, 91)[:, None]
    # Decays far too slowly to reach the half level inside the window.
    lobe = 1 + 3 * np.exp(-distances / 5000)
    panel = lobe[None, :] * (1 - angles) + angles

    metrics = _plot(panel[None], [[0, 0]], pixel_size=2.0)[2][0]

    assert not np.isfinite(metrics["radial_distance"])
    assert metrics["radial_distance_censored"]
    assert metrics["radial_distance_lower_bound"] == pytest.approx(200.0)
