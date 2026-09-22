"""Tests for ``detect_peaks_whole_range()`` (quantem.spectroscopy)."""

import numpy as np

from quantem.spectroscopy.dataset3deels import Dataset3deels
from quantem.spectroscopy.spectroscopy_visualzitions import detect_peaks_whole_range

E = 250.0 + 0.03 * np.arange(2400)


def _ds(seed=0):
    rng = np.random.default_rng(seed)
    g = lambda c, a, w: a * np.exp(-0.5 * ((E - c) / w) ** 2)  # noqa: E731
    mean = 3.0 * (E / 250.0) ** -3 + g(285.5, 0.6, 1.0) + g(293.0, 0.5, 2.0) + g(300.0, 0.3, 3.0)
    cube = mean + rng.normal(0, 0.05, (30, 30, E.size))
    return Dataset3deels.from_array(
        cube, sampling=[1, 1, 0.03], origin=[0, 0, 250.0], units=["px", "px", "eV"]
    )


def test_finds_the_peaks_over_the_whole_range():
    peaks, _ = detect_peaks_whole_range(_ds(), smoothing_window_eV=2.0, show=False)
    energies = [p["energy_eV"] for p in peaks]
    for want in (285.5, 293.0):
        assert min(abs(x - want) for x in energies) < 1.0
    assert all(p["fwhm_eV"] > 0 and p["snr"] >= 5 for p in peaks)


def test_excluded_window_is_not_searched():
    peaks, _ = detect_peaks_whole_range(
        _ds(), exclude_windows=((283.0, 288.0),), smoothing_window_eV=2.0, show=False
    )
    assert all(not (282.0 <= p["energy_eV"] <= 289.0) for p in peaks)
    assert any(abs(p["energy_eV"] - 293.0) < 1.0 for p in peaks)


def test_no_peaks_in_pure_background():
    rng = np.random.default_rng(1)
    cube = 3.0 * (E / 250.0) ** -3 + rng.normal(0, 0.05, (30, 30, E.size))
    ds = Dataset3deels.from_array(
        cube, sampling=[1, 1, 0.03], origin=[0, 0, 250.0], units=["px", "px", "eV"]
    )
    peaks, _ = detect_peaks_whole_range(ds, smoothing_window_eV=2.0, show=False)
    assert peaks == []
