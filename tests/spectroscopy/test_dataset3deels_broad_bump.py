"""Tests for ``detect_broad_bump_via_slope_change()``
(quantem.spectroscopy).

Builds a synthetic spectrum from a decaying Lorentzian "ZLP tail" plus a
small injected Gaussian "hidden bump" whose amplitude is deliberately too
small to ever create an actual local maximum in the combined spectrum --
the whole point being that plain peak-finding (including
``detect_peaks_in_range``'s "shoulder" fallback, which still needs a real
extremum of the 1st derivative) has nothing to find there, while a
2nd-derivative/curvature dip does.
"""

import numpy as np
import pytest
from scipy.signal import find_peaks

from quantem.spectroscopy.dataset3deels import Dataset3deels
from quantem.spectroscopy.spectroscopy_visualzitions import detect_broad_bump_via_slope_change

ZLP_AMPLITUDE = 1000.0
ZLP_GAMMA = 0.4  # Lorentzian HWHM-like scale

BUMP_AMPLITUDE = 10.0
BUMP_CENTER = 2.0
BUMP_SIGMA = 0.3

NOISE_STD = 0.5

ENERGY_LO = 0.5
ENERGY_HI = 4.0
N_ENERGY = 1400
TEST_RANGE = (ENERGY_LO, ENERGY_HI)
SMOOTHING_WINDOW_EV = 0.4

# Chosen so that even at the bump's steepest (uphill) flank the Lorentzian
# tail's own slope still dominates -- see the derivation in the PR/commit
# notes; verified numerically (np.diff(spectrum) < 0 everywhere) below
# rather than trusted blindly.
BUMP_CENTER_TOLERANCE_EV = 0.1

# Below the noise-only spectrum's own strongest spurious curvature dip
# (empirically ~27 at this noise level) but well below the real bump's
# curvature-dip prominence (empirically ~116) -- see
# test_min_prominence_filters_pure_noise_spectrum.
NOISE_FILTER_MIN_PROMINENCE = 50.0


def _energy_axis() -> np.ndarray:
    return np.linspace(ENERGY_LO, ENERGY_HI, N_ENERGY)


def _lorentzian_tail(energy: np.ndarray) -> np.ndarray:
    return ZLP_AMPLITUDE * (ZLP_GAMMA**2) / (energy**2 + ZLP_GAMMA**2)


def _hidden_bump(energy: np.ndarray) -> np.ndarray:
    return BUMP_AMPLITUDE * np.exp(-0.5 * ((energy - BUMP_CENTER) / BUMP_SIGMA) ** 2)


def _dataset_from_spectrum(spectrum: np.ndarray, energy: np.ndarray) -> Dataset3deels:
    array = spectrum.reshape(1, 1, -1)
    return Dataset3deels.from_array(
        array=array,
        sampling=[1, 1, (energy[-1] - energy[0]) / (energy.size - 1)],
        origin=[0, 0, energy[0]],
        units=["px", "px", "eV"],
    )


class TestDetectBroadBumpViaSlopeChange:
    def test_recovers_hidden_bump_missed_by_plain_peak_finding(self):
        energy = _energy_axis()
        spectrum = _lorentzian_tail(energy) + _hidden_bump(energy)

        # The premise this test proves: no local maximum exists anywhere in
        # TEST_RANGE, so a plain peak finder is structurally incapable of
        # locating the hidden bump.
        assert np.all(np.diff(spectrum) < 0), (
            "synthetic spectrum must be strictly decreasing everywhere in the test "
            "range -- otherwise this test doesn't actually demonstrate 'no local max'"
        )
        raw_peak_idx, _ = find_peaks(spectrum)
        assert len(raw_peak_idx) == 0, (
            "plain scipy.signal.find_peaks must find nothing on a monotonic decay"
        )

        ds = _dataset_from_spectrum(spectrum, energy)
        result = detect_broad_bump_via_slope_change(
            ds,
            energy_range=TEST_RANGE,
            smoothing_window_eV=SMOOTHING_WINDOW_EV,
            plot=False,
        )

        assert len(result["candidates"]) == 1
        recovered_center = result["candidates"][0]["center_eV"]
        assert abs(recovered_center - BUMP_CENTER) < BUMP_CENTER_TOLERANCE_EV
        assert result["candidates"][0]["dip_prominence"] > 0
        assert result["candidates"][0]["slope_change_amplitude"] > 0

    def test_min_prominence_filters_pure_noise_spectrum(self):
        energy = _energy_axis()
        rng = np.random.default_rng(1)
        spectrum = _lorentzian_tail(energy) + rng.normal(0, NOISE_STD, energy.size)
        ds = _dataset_from_spectrum(spectrum, energy)

        result = detect_broad_bump_via_slope_change(
            ds,
            energy_range=TEST_RANGE,
            smoothing_window_eV=SMOOTHING_WINDOW_EV,
            min_prominence=NOISE_FILTER_MIN_PROMINENCE,
            plot=False,
        )
        assert result["candidates"] == []

    def test_smoothing_window_wider_than_energy_range_raises(self):
        energy = _energy_axis()
        spectrum = _lorentzian_tail(energy) + _hidden_bump(energy)
        ds = _dataset_from_spectrum(spectrum, energy)

        with pytest.raises(ValueError, match="does not fit inside energy_range"):
            detect_broad_bump_via_slope_change(
                ds,
                energy_range=TEST_RANGE,
                smoothing_window_eV=10.0,
                plot=False,
            )
