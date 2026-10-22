"""Smoke coverage for the extracted peak-plotting functions.

Each is called both as a free function and through the ``BraggPeaksPolymer``
forwarding method: notebooks call the latter (three appear in 51 notebooks),
and the former is where the code now lives.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantem.core.datastructures import Vector
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.diffraction import BraggPeaksPolymer
from quantem.diffraction import peak_visualization as pv


@pytest.fixture
def bp():
    array = np.arange(2 * 2 * 8 * 8, dtype=np.float32).reshape(2, 2, 8, 8)
    dataset = Dataset4dstem.from_array(
        array=array, name="viz_test", origin=(0, 0, 0, 0),
        sampling=(1.0, 1.0, 1.0, 1.0),
        units=["pixels"] * 4, signal_units="counts",
    )
    analysis = BraggPeaksPolymer.from_data(
        dataset, device="cpu",
        compute_parameters=lambda x, **kwargs: (0.0, 1.0),
        normalize_data=lambda x, lo, hi: x,
    )
    peaks = Vector.from_shape(shape=(2, 2), fields=["y_pixels", "x_pixels"],
                             units=["pixels"] * 2, name="cartesian_peaks")
    polar = Vector.from_shape(shape=(2, 2), fields=["r_invA", "theta"],
                              units=["1/Å", "radians"], name="polar_peaks")
    intensities = Vector.from_shape(shape=(2, 2), fields=["intensities"],
                                    units=["counts"], name="peak_intensities")
    for i in range(2):
        for j in range(2):
            peaks[i, j] = np.array([[4.0, 4.0], [2.0, 6.0], [6.0, 2.0]])
            polar[i, j] = np.array([[0.1, 0.0], [0.4, np.pi / 2], [0.7, np.pi]])
            intensities[i, j] = np.array([[10.0], [20.0], [30.0]])
    analysis.peak_coordinates_cartesian = peaks
    analysis.polar_peaks = polar
    analysis.peak_intensities = intensities
    analysis.polar_data = {"intensity": np.ones((2, 2, 5, 6), dtype=np.float32)}
    yield analysis
    plt.close("all")


def test_peak_plots_agree_with_their_forwarding_methods(bp):
    direct = pv.peak_radial_intensity_plot(
        bp.polar_peaks, bp.peak_intensities, num_bins=8, plot=False, return_data=True)
    through = bp.peak_radial_intensity_plot(num_bins=8, plot=False, return_data=True)
    np.testing.assert_allclose(np.asarray(direct[0]), np.asarray(through[0]))

    direct = pv.peak_radial_count_plot(bp.polar_peaks, num_bins=8, plot=False, return_data=True)
    through = bp.peak_radial_count_plot(num_bins=8, plot=False, return_data=True)
    np.testing.assert_allclose(np.asarray(direct[0]), np.asarray(through[0]))

    # The notebooks call this as ``bp.plot_peak_count_map(peak_windows, ...)``.
    windows = [[0.0, 0.5], [0.5, 1.0]]
    _, _, counts = bp.plot_peak_count_map(windows, return_values=True)
    assert len(counts) == len(windows)
    assert all(np.asarray(c).shape == (2, 2) for c in counts)
    _, _, direct_counts = pv.plot_peak_count_map(
        bp.peak_coordinates_cartesian, bp.polar_peaks, windows, return_values=True)
    np.testing.assert_allclose(np.asarray(counts), np.asarray(direct_counts))

    bp.plot_peak_histogram_map(intensity_threshold=15.0, return_values=True)
    pv.plot_peak_histogram_map(
        bp.peak_coordinates_cartesian, bp.peak_intensities,
        intensity_threshold=15.0, return_values=True)
    bp.visualize_selected_patterns([(0, 0), (1, 1)])
    pv.visualize_selected_patterns(bp.dataset_cartesian, [(0, 0)])


def test_visualize_peak_detection_builds_the_qa_grid(bp):
    """Five columns per pattern: polar, polar+peaks, cartesian+peaks, original, normalized.

    This is the only view that puts the original and normalized patterns beside
    the detections, which is what reveals a normalization problem on a new dataset.
    """
    bp.image_centers = np.zeros((2, 2, 2), dtype=float)
    bp.image_centers[:, :, :] = np.array([4.0, 4.0])[:, None, None]
    bp.num_annular_bins = 3

    _, axes = bp.visualize_peak_detection(indices=[(0, 0), (1, 1)], images_per_row=2)
    # images_per_row patterns wide, five visualisations each.
    assert np.asarray(axes).shape == (1, 10)
    titles = [a.get_title() for a in np.asarray(axes).ravel() if a.get_title()]
    assert any("Polar" in t for t in titles)

    # Without indices, n_images random positions are drawn.
    _, axes = bp.visualize_peak_detection(n_images=2, images_per_row=1)
    assert np.asarray(axes).shape == (2, 5)
