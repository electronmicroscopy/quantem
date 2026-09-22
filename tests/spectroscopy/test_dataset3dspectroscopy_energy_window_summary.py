"""Tests for ``summarize_energy_windows()`` (quantem.spectroscopy)."""

import numpy as np

from quantem.spectroscopy.dataset3deels import Dataset3deels
from quantem.spectroscopy.utils import summarize_energy_windows


def test_signal_window_is_significant_and_noise_window_is_not():
    rng = np.random.default_rng(0)
    e = 0.5 + 0.03 * np.arange(200)
    cube = rng.normal(0, 1.0, (30, 30, e.size))
    cube[..., (e > 2.0) & (e < 2.5)] += 0.8  # a real feature in 2.0-2.5 eV
    ds = Dataset3deels.from_array(
        cube, sampling=[1, 1, 0.03], origin=[0, 0, 0.5], units=["px", "px", "eV"]
    )
    sig, noise = summarize_energy_windows(ds, ((2.0, 2.5), (3.5, 4.0)))
    assert sig["t"] > 10 and sig["frac_positive"] > 0.8
    assert abs(noise["t"]) < 4
    assert 2.0 <= sig["peak_eV"] <= 2.5
