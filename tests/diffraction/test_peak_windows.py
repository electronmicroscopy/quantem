"""Baseline coverage for radial peak-window estimation.

Written before extracting it. ``estimate_peak_windows`` appears in 51 notebooks and
decides the q-windows every count map, radial profile and flowline family downstream
is computed over, so a silent change here would shift every result.
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
KWARGS = dict(num_bins=100, q_min=0.0, q_max=0.5, prominence_factor=0.05, smoothing_sigma=1.0)


@pytest.fixture(scope="module")
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


def test_recovers_the_injected_families_and_windows(bp):
    centers, windows, info = bp.estimate_peak_windows(**KWARGS)

    # Ordered by q, and each within a bin width of the injected centre.
    assert len(centers) == len(FAMILIES)
    assert np.all(np.diff(centers) > 0)
    for found, injected in zip(centers, FAMILIES):
        assert found == pytest.approx(injected, abs=0.01)

    assert windows.shape == (len(centers), 2)
    assert np.all(windows[:, 0] < centers) and np.all(centers < windows[:, 1])
    assert np.all(windows[:, 0] >= 0.0) and np.all(windows[:, 1] <= 0.5)
    # min_width floors the half-width, so no window is narrower than min_width.
    assert np.all((windows[:, 1] - windows[:, 0]) >= 0.05 - 1e-9)

    # `profile` and `intensity_profile` are the same array -- a back-compat alias.
    assert set(info) == {"heights", "prominences", "widths_fwhm", "intensity_profile",
                         "profile", "r_centers", "mode", "log_scale"}
    np.testing.assert_array_equal(info["profile"], info["intensity_profile"])
    assert info["r_centers"].shape == info["profile"].shape == (100,)
    assert info["mode"] == "intensity" and info["log_scale"] is False

    # With every intensity equal to 1, the count profile differs only by scale.
    by_count, _, _ = bp.estimate_peak_windows(mode="count", **KWARGS)
    np.testing.assert_allclose(centers, by_count)

    _, _, logged = bp.estimate_peak_windows(log_scale=True, **KWARGS)
    assert logged["log_scale"] is True
    assert logged["profile"].max() < info["profile"].max()


def test_no_peaks_and_bad_mode(bp):
    """A window containing no peaks must still return usable shapes."""
    centers, windows, info = bp.estimate_peak_windows(
        num_bins=20, q_min=0.9, q_max=1.0, prominence_factor=0.5)

    assert len(centers) == 0
    assert windows.shape == (0, 2)
    assert info == {}
    with pytest.raises(ValueError, match="mode must be"):
        bp.estimate_peak_windows(mode="not-a-mode")
