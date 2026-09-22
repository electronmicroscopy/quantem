"""Tests for the per-map ``map_normalization`` option of ``show_energy_window_map()``.

Two spatially identical features are placed in two energy windows whose
intensities differ by ~100x. Unnormalized, the dim window's map is nearly
invisible next to the bright one; normalized, both maps must land on the
same scale so their spatial contrast can be compared.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from quantem.spectroscopy.dataset3deels import Dataset3deels

N_ROW = N_COL = 16
BRIGHT_WINDOW = (1.0, 1.5)
DIM_WINDOW = (2.0, 2.5)
BRIGHT_SCALE = 1000.0
DIM_SCALE = 10.0


@pytest.fixture(scope="module")
def dataset():
    energy = np.linspace(0.5, 3.0, 251)
    yy, xx = np.mgrid[0:N_ROW, 0:N_COL]
    pattern = np.exp(-(((yy - 8) ** 2 + (xx - 8) ** 2) / 20.0))  # same blob for both
    array = np.zeros((N_ROW, N_COL, energy.size))
    for (lo, hi), scale in ((BRIGHT_WINDOW, BRIGHT_SCALE), (DIM_WINDOW, DIM_SCALE)):
        in_win = (energy >= lo) & (energy <= hi)
        array[:, :, in_win] = (1.0 + pattern)[:, :, None] * scale
    return Dataset3deels.from_array(
        array=array,
        sampling=[1, 1, energy[1] - energy[0]],
        origin=[0, 0, energy[0]],
        units=["px", "px", "eV"],
    )


def _map(dataset, window, **kw):
    fig, _, emap = dataset.show_energy_window_map(energy_window=list(window), show=False, **kw)
    plt.close(fig)
    return emap


def test_raw_maps_differ_by_orders_of_magnitude(dataset):
    bright = _map(dataset, BRIGHT_WINDOW, map_normalization=None)
    dim = _map(dataset, DIM_WINDOW, map_normalization=None)
    assert bright.max() / dim.max() == pytest.approx(BRIGHT_SCALE / DIM_SCALE, rel=1e-6)


@pytest.mark.parametrize("mode", ["percentile", "minmax"])
def test_zero_one_modes_put_both_windows_on_same_scale(dataset, mode):
    bright = _map(dataset, BRIGHT_WINDOW, map_normalization=mode)
    dim = _map(dataset, DIM_WINDOW, map_normalization=mode)
    for m in (bright, dim):
        assert m.min() == pytest.approx(0.0)
        assert m.max() == pytest.approx(1.0)
    # identical spatial pattern -> identical normalized maps
    np.testing.assert_allclose(bright, dim, atol=1e-9)


def test_zscore_and_relative(dataset):
    z = _map(dataset, DIM_WINDOW, map_normalization="zscore")
    assert z.mean() == pytest.approx(0.0, abs=1e-9)
    assert z.std() == pytest.approx(1.0)
    rel = _map(dataset, DIM_WINDOW, map_normalization="relative")
    assert rel.mean() == pytest.approx(1.0)


def test_default_normalizes(dataset):
    default = _map(dataset, DIM_WINDOW)
    assert default.max() <= 1.0 + 1e-12 and default.min() >= 0.0


def test_percentile_is_robust_to_hot_pixel(dataset):
    arr = np.array(dataset.array, copy=True)
    hot = Dataset3deels.from_array(
        array=arr,
        sampling=list(dataset.sampling),
        origin=list(dataset.origin),
        units=list(dataset.units),
    )
    hot.array[0, 0, :] = 1e6  # one saturated pixel
    p = _map(hot, DIM_WINDOW, map_normalization="percentile")
    mm = _map(hot, DIM_WINDOW, map_normalization="minmax")
    # min-max is crushed by the hot pixel; percentile keeps the blob's contrast
    assert np.median(p) > 10 * np.median(mm)


def test_flat_map_does_not_divide_by_zero(dataset):
    flat = Dataset3deels.from_array(
        array=np.ones((4, 4, 50)),
        sampling=[1, 1, 0.1],
        origin=[0, 0, 0.0],
        units=["px", "px", "eV"],
    )
    fig, _, emap = flat.show_energy_window_map(energy_window=[1.0, 2.0], show=False)
    plt.close(fig)
    assert np.all(emap == 0)


def test_invalid_mode_raises(dataset):
    with pytest.raises(ValueError, match="map_normalization"):
        _map(dataset, DIM_WINDOW, map_normalization="bogus")


def test_robust_z_not_inflated_by_hot_pixel(dataset):
    arr = np.array(dataset.array, copy=True)
    hot = Dataset3deels.from_array(
        array=arr,
        sampling=list(dataset.sampling),
        origin=list(dataset.origin),
        units=list(dataset.units),
    )
    hot.array[0, 0, :] = 1e4 * DIM_SCALE  # one saturated pixel
    z = _map(hot, DIM_WINDOW, map_normalization="zscore")
    rz = _map(hot, DIM_WINDOW, map_normalization="robust_z")
    # plain z-score: the hot pixel inflates std, so it can never exceed ~sqrt(N)
    assert z.max() < np.sqrt(N_ROW * N_COL) + 1e-6
    # robust z: median/MAD ignore it, so it stands out enormously
    assert rz.max() > 10 * z.max()
    assert np.median(rz) == pytest.approx(0.0, abs=1e-9)


def test_robust_z_sparse_map_falls_back_to_std():
    arr = np.zeros((8, 8, 50))
    arr[3, 3, :] = 5.0  # >half the pixels identical -> MAD == 0
    ds = Dataset3deels.from_array(
        array=arr, sampling=[1, 1, 0.1], origin=[0, 0, 0.0], units=["px", "px", "eV"]
    )
    fig, _, emap = ds.show_energy_window_map(
        energy_window=[1.0, 2.0], show=False, map_normalization="robust_z"
    )
    plt.close(fig)
    assert np.all(np.isfinite(emap)) and emap[3, 3] > 0
