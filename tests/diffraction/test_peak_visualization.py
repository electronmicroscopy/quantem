"""Smoke coverage for the extracted peak-plotting functions.

Each is checked twice: as a free function and through the ``BraggPeaksPolymer``
forwarding method, since notebooks call the latter (three of these appear in 51
notebooks) and only the former is where the code now lives.
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


def test_peak_radial_intensity_plot(bp):
    direct = pv.peak_radial_intensity_plot(
        bp.polar_peaks, bp.peak_intensities, num_bins=8, plot=False, return_data=True)
    through = bp.peak_radial_intensity_plot(num_bins=8, plot=False, return_data=True)
    assert direct is not None and through is not None
    np.testing.assert_allclose(np.asarray(direct[0]), np.asarray(through[0]))


def test_peak_radial_count_plot(bp):
    direct = pv.peak_radial_count_plot(bp.polar_peaks, num_bins=8, plot=False, return_data=True)
    through = bp.peak_radial_count_plot(num_bins=8, plot=False, return_data=True)
    assert direct is not None and through is not None
    np.testing.assert_allclose(np.asarray(direct[0]), np.asarray(through[0]))


def test_plot_peak_count_map_positionally(bp):
    """The notebooks call this as ``bp.plot_peak_count_map(peak_windows, ...)``."""
    windows = [[0.0, 0.5], [0.5, 1.0]]
    fig, axes, counts = bp.plot_peak_count_map(windows, return_values=True)
    assert len(counts) == len(windows)
    assert all(np.asarray(c).shape == (2, 2) for c in counts)

    fig2, axes2, counts2 = pv.plot_peak_count_map(
        bp.peak_coordinates_cartesian, bp.polar_peaks, windows, return_values=True)
    np.testing.assert_allclose(np.asarray(counts), np.asarray(counts2))


def test_plot_peak_histogram_map(bp):
    result = bp.plot_peak_histogram_map(intensity_threshold=15.0, return_values=True)
    assert result is not None
    direct = pv.plot_peak_histogram_map(
        bp.peak_coordinates_cartesian, bp.peak_intensities,
        intensity_threshold=15.0, return_values=True)
    assert direct is not None


def test_visualize_selected_patterns(bp):
    bp.visualize_selected_patterns([(0, 0), (1, 1)])
    pv.visualize_selected_patterns(bp.dataset_cartesian, [(0, 0)])
