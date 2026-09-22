"""Tests for ``Dataset3deels.despike()`` and ``Dataset3dspectroscopy.bin_spatial()``."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from quantem.spectroscopy.dataset3deels import Dataset3deels  # noqa: E402

E = 0.3 + 0.03 * np.arange(400)  # 0.3 .. 12.27 eV


def _ds(cube):
    return Dataset3deels.from_array(
        cube, sampling=[30.0, 30.0, 0.03], origin=[0, 0, E[0]], units=["nm", "nm", "eV"]
    )


def _cube(ny=12, nx=10, seed=0):
    rng = np.random.default_rng(seed)
    return 5.0 * E**-2.2 + 0.3 + rng.normal(0, 0.05, (ny, nx, E.size))


def teardown_function():
    plt.close("all")


# ---------------------------------------------------------------- despike


def test_despike_interpolates_only_inside_the_range():
    cube = _cube()
    i0, i1 = 150, 153
    cube[:, :, i0 : i1 + 1] += 50.0  # a 4-channel spike in every pixel
    ds = _ds(cube)
    out = ds.despike([(E[i0] - 0.001, E[i1] + 0.001)], show=False)
    a = np.asarray(out.array)
    # outside the range: untouched
    np.testing.assert_array_equal(a[:, :, :i0], cube[:, :, :i0])
    np.testing.assert_array_equal(a[:, :, i1 + 1 :], cube[:, :, i1 + 1 :])
    # inside: the straight line between the neighbouring clean channels
    frac = (E[i0 : i1 + 1] - E[i0 - 1]) / (E[i1 + 1] - E[i0 - 1])
    expected = cube[:, :, [i0 - 1]] + frac * (cube[:, :, [i1 + 1]] - cube[:, :, [i0 - 1]])
    np.testing.assert_allclose(a[:, :, i0 : i1 + 1], expected)
    np.testing.assert_array_equal(np.asarray(ds.array), cube)  # input not modified


def test_despike_empty_ranges_returns_unchanged_copy():
    ds = _ds(_cube())
    out = ds.despike([], show=False)
    assert out is not ds
    np.testing.assert_array_equal(np.asarray(out.array), np.asarray(ds.array))


@pytest.mark.parametrize(
    "bad",
    [
        (2.0, 1.0),  # lo >= hi
        (0.0, 1.0),  # below the axis
        (float(E[0]), float(E[2])),  # touches the first channel
        (float(E[-3]), float(E[-1])),  # touches the last channel
        (2.0001, 2.0002),  # between two channels: no channel inside
    ],
)
def test_despike_rejects_bad_ranges(bad):
    with pytest.raises(ValueError):
        _ds(_cube()).despike([bad], show=False)


def test_despike_preview_plot():
    out = _ds(_cube()).despike([(3.0, 3.1)], show=True, display_energy_range=(2.0, 4.0))
    assert out.shape == (12, 10, E.size)


# ---------------------------------------------------------------- bin_spatial


def test_bin_spatial():
    cube = _cube()
    ds = _ds(cube)
    b = ds.bin_spatial(2)
    assert b.shape == (6, 5, E.size)
    assert np.allclose(b.sampling[:2], 60.0)
    np.testing.assert_allclose(b.array[0, 0], cube[:2, :2].mean(axis=(0, 1)))
    assert ds.bin_spatial(1).shape == ds.shape


def test_bin_spatial_keeps_incomplete_edge_blocks():
    cube = _cube(ny=5, nx=7)
    b = _ds(cube).bin_spatial(2)
    assert b.shape == (3, 4, E.size)  # Dataset.bin() would give (2, 3)
    np.testing.assert_allclose(b.array[2, 3], cube[4, 6])  # 1x1 corner block
    np.testing.assert_allclose(b.array[2, 0], cube[4, 0:2].mean(axis=0))  # 1x2 edge block


def test_bin_spatial_rejects_bad_factor():
    with pytest.raises(ValueError):
        _ds(_cube()).bin_spatial(0)
