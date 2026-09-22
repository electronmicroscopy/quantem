"""Tests for the split-half reproducibility check added to ``detect_peaks_whole_range()`` and for
``crop_energy_range()`` (quantem.spectroscopy)."""

import numpy as np
import pytest

from quantem.spectroscopy.dataset3deels import Dataset3deels
from quantem.spectroscopy.spectroscopy_visualzitions import detect_peaks_whole_range
from quantem.spectroscopy.utils import crop_energy_range

E = 250.0 + 0.03 * np.arange(2400)


def _ds(cube):
    return Dataset3deels.from_array(
        cube, sampling=[1, 1, 0.03], origin=[0, 0, 250.0], units=["px", "px", "eV"]
    )


def test_peak_present_everywhere_is_reproducible():
    rng = np.random.default_rng(0)
    g = lambda c, a, w: a * np.exp(-0.5 * ((E - c) / w) ** 2)  # noqa: E731
    bg = 3.0 * (E / 250.0) ** -3
    cube = bg + g(293.0, 0.6, 2.0) + rng.normal(0, 0.03, (20, 20, E.size))
    peaks, _ = detect_peaks_whole_range(_ds(cube), smoothing_window_eV=2.0, show=False)
    assert peaks and all(p["reproducible"] for p in peaks)


def test_peak_confined_to_one_checkerboard_half_is_flagged_not_reproducible():
    # A feature present in EXACTLY one checkerboard parity (not a handful of pixels, so it's still large
    # and clearly detected in the full mean) is the case the checkerboard split is built to catch: fully
    # present in one half, fully absent in the other.
    rng = np.random.default_rng(1)
    bg = 3.0 * (E / 250.0) ** -3
    cube = np.broadcast_to(bg, (20, 20, E.size)).copy() + rng.normal(0, 0.03, (20, 20, E.size))
    checker = np.add.outer(np.arange(20), np.arange(20)) % 2 == 0
    bump = 3.0 * ((E > 292.5) & (E < 293.5)).astype(float)
    add = np.zeros_like(cube)
    add[checker] = bump
    cube = cube + add
    peaks, _ = detect_peaks_whole_range(_ds(cube), smoothing_window_eV=2.0, show=False)
    assert peaks and any(not p["reproducible"] for p in peaks)


def test_check_reproducibility_false_leaves_field_none():
    rng = np.random.default_rng(2)
    bg = 3.0 * (E / 250.0) ** -3
    g = lambda c, a, w: a * np.exp(-0.5 * ((E - c) / w) ** 2)  # noqa: E731
    cube = bg + g(293.0, 0.6, 2.0) + rng.normal(0, 0.03, (20, 20, E.size))
    peaks, _ = detect_peaks_whole_range(
        _ds(cube), smoothing_window_eV=2.0, check_reproducibility=False, show=False
    )
    assert peaks and all(p["reproducible"] is None for p in peaks)


def test_crop_energy_range_keeps_only_requested_channels():
    cube = np.tile(E, (3, 4, 1))
    ds = _ds(cube)
    cropped = crop_energy_range(ds, 260.0, 270.0)
    axis = np.asarray(cropped.energy_axis)
    assert axis[0] >= 260.0 - 1e-9 and axis[-1] <= 270.0 + 1e-9
    np.testing.assert_allclose(cropped.array[0, 0, :], axis, atol=1e-6)
    assert cropped.shape[2] == np.count_nonzero((E >= 260.0) & (E <= 270.0))


def test_crop_energy_range_input_untouched():
    cube = np.tile(E, (2, 2, 1))
    ds = _ds(cube)
    before = ds.shape
    crop_energy_range(ds, 260.0, 270.0)
    assert ds.shape == before


def test_crop_energy_range_no_overlap_raises():
    cube = np.tile(E, (2, 2, 1))
    ds = _ds(cube)
    with pytest.raises(ValueError, match="does not overlap"):
        crop_energy_range(ds, 1000.0, 1010.0)


def test_crop_energy_range_invalid_order_raises():
    cube = np.tile(E, (2, 2, 1))
    ds = _ds(cube)
    with pytest.raises(ValueError, match="lo_eV must be"):
        crop_energy_range(ds, 270.0, 260.0)
