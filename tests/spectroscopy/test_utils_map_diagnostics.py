"""Tests for ``spatial_coherence()``, ``correlate_with_reference()`` and the map-diagnostics summary/plot (quantem.spectroscopy)."""

import numpy as np
import pytest

from quantem.spectroscopy.spectroscopy_visualzitions import plot_map_diagnostics
from quantem.spectroscopy.utils import (
    correlate_with_reference,
    spatial_coherence,
    summarize_map_diagnostics,
)


def test_coherent_patches_get_high_significant_moran_i():
    yy, xx = np.mgrid[0:30, 0:30]
    smooth = np.sin(yy / 4.0) * np.cos(xx / 4.0)  # slowly-varying -> strongly coherent
    r = spatial_coherence(smooth, n_perm=200, seed=0)
    assert r["moran_i"] > 0.5
    assert r["p"] < 0.01
    assert r["n_valid"] == 900


def test_pure_noise_gets_low_moran_i_and_large_p():
    rng = np.random.default_rng(1)
    noise = rng.normal(size=(30, 30))
    r = spatial_coherence(noise, n_perm=200, seed=1)
    assert abs(r["moran_i"]) < 0.15
    assert r["p"] > 0.05


def test_checkerboard_gives_strongly_negative_moran_i():
    checker = (np.add.outer(np.arange(20), np.arange(20)) % 2).astype(float)
    r = spatial_coherence(checker, n_perm=100, seed=2)
    assert r["moran_i"] < -0.8


def test_nan_pixels_are_excluded():
    yy, xx = np.mgrid[0:20, 0:20]
    smooth = np.sin(yy / 3.0) * np.cos(xx / 3.0)
    smooth_with_nan = smooth.copy()
    smooth_with_nan[0:3, 0:3] = np.nan
    r = spatial_coherence(smooth_with_nan, n_perm=100, seed=0)
    assert r["n_valid"] == 400 - 9
    assert np.isfinite(r["moran_i"])


def test_too_few_valid_pixels_returns_nan_not_error():
    x = np.full((5, 5), np.nan)
    x[0, 0] = 1.0
    r = spatial_coherence(x)
    assert r["n_valid"] == 1
    assert np.isnan(r["moran_i"]) and np.isnan(r["p"])


def test_correlate_with_reference_identical_map_gives_r_one():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 20))
    r = correlate_with_reference(a, a)
    assert r["r"] == pytest.approx(1.0, abs=1e-9)
    assert r["p"] < 1e-6
    assert r["n"] == 400


def test_correlate_with_reference_independent_maps_near_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 40))
    b = rng.normal(size=(40, 40))
    r = correlate_with_reference(a, b)
    assert abs(r["r"]) < 0.2


def test_correlate_with_reference_follows_a_real_relationship():
    rng = np.random.default_rng(0)
    thickness = rng.uniform(0.5, 2.0, size=(30, 30))
    signal = 3.0 * thickness + rng.normal(0, 0.05, size=(30, 30))  # follows thickness closely
    r = correlate_with_reference(signal, thickness)
    assert r["r"] > 0.9 and r["p"] < 1e-6


def test_correlate_with_reference_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        correlate_with_reference(np.zeros((10, 10)), np.zeros((5, 5)))


def test_correlate_with_reference_nan_pixels_excluded_pairwise():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 20))
    b = a.copy()
    a[0, 0] = np.nan
    b[1, 1] = np.nan
    r = correlate_with_reference(a, b)
    assert r["n"] == 398
    assert r["r"] > 0.99


def test_map_diagnostics_combines_coherence_and_correlation():
    yy, xx = np.mgrid[0:25, 0:25]
    thickness = 1.0 + 0.02 * xx  # smooth gradient
    coherent = np.sin(yy / 4.0) * np.cos(xx / 4.0)
    rng = np.random.default_rng(0)
    noisy = rng.normal(size=(25, 25))
    follows_thickness = 2.0 * thickness + rng.normal(0, 0.02, size=(25, 25))

    maps = {(0.8, 1.2): noisy, (1.2, 1.8): coherent, (1.8, 2.2): follows_thickness}
    rows, fig = plot_map_diagnostics(maps, thickness_map=thickness, n_perm=150, seed=0, show=False)

    by_window = {r["window"]: r for r in rows}
    assert by_window[(1.2, 1.8)]["z"] > by_window[(0.8, 1.2)]["z"]  # coherent >> noise
    assert by_window[(1.8, 2.2)]["corr_thickness"]["r"] > 0.9  # follows thickness closely
    assert (
        by_window[(1.2, 1.8)]["corr_thickness"]["r"] < 0.5
    )  # coherent pattern is unrelated to thickness
    assert fig is None
    assert set(by_window) == set(maps)
    # the plot function reports exactly what the compute function returns
    assert rows == summarize_map_diagnostics(maps, thickness_map=thickness, n_perm=150, seed=0)
