"""Baseline coverage for radial peak-window estimation.

Written before extracting it. ``estimate_peak_windows`` appears in 51 notebooks and
had no tests, and it decides the q-windows every count map, radial profile and
flowline family downstream is computed over -- so a silent change here would shift
every result without failing anything.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from quantem.core.datastructures import Vector
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.diffraction import BraggPeaksPolymer

# Three well-separated radial families, as a polymer scan would show.
FAMILIES = (0.08, 0.20, 0.35)


@pytest.fixture
def bp():
    dataset = Dataset4dstem.from_array(
        array=np.zeros((4, 4, 8, 8), dtype=np.float32), name="windows_test",
        origin=(0, 0, 0, 0), sampling=(1.0, 1.0, 1.0, 1.0),
        units=["pixels"] * 4, signal_units="counts",
    )
    analysis = BraggPeaksPolymer.from_data(
        dataset, device="cpu",
        compute_parameters=lambda x, **kwargs: (0.0, 1.0),
        normalize_data=lambda x, lo, hi: x,
    )
    polar = Vector.from_shape(shape=(4, 4), fields=["r_invA", "theta"],
                             units=["1/A", "rad"], name="polar_peaks")
    intensities = Vector.from_shape(shape=(4, 4), fields=["intensities"],
                                    units=["counts"], name="peak_intensities")
    rng = np.random.default_rng(0)
    for i in range(4):
        for j in range(4):
            q = np.concatenate([rng.normal(c, 0.006, 40) for c in FAMILIES])
            polar[i, j] = np.column_stack([q, rng.uniform(0, np.pi, q.size)])
            intensities[i, j] = np.full((q.size, 1), 1.0)
    analysis.polar_peaks = polar
    analysis.peak_intensities = intensities
    return analysis


def test_recovers_the_injected_families(bp):
    centers, windows, info = bp.estimate_peak_windows(
        num_bins=100, q_min=0.0, q_max=0.5, n_peaks=5,
        prominence_factor=0.05, smoothing_sigma=1.0)

    assert len(centers) == len(FAMILIES)
    # Ordered by q, and each within a bin width of the injected centre.
    assert np.all(np.diff(centers) > 0)
    for found, injected in zip(centers, FAMILIES):
        assert found == pytest.approx(injected, abs=0.01)


def test_windows_bracket_their_centres_and_stay_in_range(bp):
    centers, windows, _ = bp.estimate_peak_windows(
        num_bins=100, q_min=0.0, q_max=0.5, prominence_factor=0.05, smoothing_sigma=1.0)

    assert windows.shape == (len(centers), 2)
    assert np.all(windows[:, 0] < centers) and np.all(centers < windows[:, 1])
    assert np.all(windows[:, 0] >= 0.0) and np.all(windows[:, 1] <= 0.5)
    # min_width floors the half-width, so no window is narrower than min_width.
    assert np.all((windows[:, 1] - windows[:, 0]) >= 0.05 - 1e-9)


def test_count_mode_agrees_with_intensity_mode_for_uniform_intensities(bp):
    """With every intensity equal to 1, the two profiles differ only by scale."""
    by_intensity, _, _ = bp.estimate_peak_windows(
        num_bins=100, q_min=0.0, q_max=0.5, mode="intensity",
        prominence_factor=0.05, smoothing_sigma=1.0)
    by_count, _, _ = bp.estimate_peak_windows(
        num_bins=100, q_min=0.0, q_max=0.5, mode="count",
        prominence_factor=0.05, smoothing_sigma=1.0)

    np.testing.assert_allclose(by_intensity, by_count)


def test_peak_info_keys_and_the_duplicated_profile_alias(bp):
    """`profile` and `intensity_profile` are the same array -- a back-compat alias."""
    _, _, info = bp.estimate_peak_windows(
        num_bins=100, q_min=0.0, q_max=0.5, prominence_factor=0.05, smoothing_sigma=1.0)

    assert set(info) == {"heights", "prominences", "widths_fwhm", "intensity_profile",
                         "profile", "r_centers", "mode", "log_scale"}
    np.testing.assert_array_equal(info["profile"], info["intensity_profile"])
    assert info["r_centers"].shape == info["profile"].shape == (100,)
    assert info["mode"] == "intensity" and info["log_scale"] is False


def test_log_scale_compresses_the_profile(bp):
    _, _, linear = bp.estimate_peak_windows(
        num_bins=100, q_min=0.0, q_max=0.5, prominence_factor=0.05, smoothing_sigma=1.0)
    _, _, logged = bp.estimate_peak_windows(
        num_bins=100, q_min=0.0, q_max=0.5, prominence_factor=0.05,
        smoothing_sigma=1.0, log_scale=True)

    assert logged["log_scale"] is True
    assert logged["profile"].max() < linear["profile"].max()


def test_no_peaks_returns_empty_arrays_of_the_right_shape(bp):
    """A window containing no peaks must still return usable shapes."""
    centers, windows, info = bp.estimate_peak_windows(
        num_bins=20, q_min=0.9, q_max=1.0, prominence_factor=0.5)

    assert len(centers) == 0
    assert windows.shape == (0, 2)
    assert info == {}


def test_rejects_an_unknown_mode(bp):
    with pytest.raises(ValueError, match="mode must be"):
        bp.estimate_peak_windows(mode="not-a-mode")
