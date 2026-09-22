"""Tests for ``subtract_background_two_sided()`` / ``fit_windows_inside_axis()``
(quantem.spectroscopy)."""

import numpy as np
import pytest

from quantem.spectroscopy.dataset3deels import Dataset3deels
from quantem.spectroscopy.utils import (
    fit_windows_inside_axis,
    subtract_background_two_sided,
)

E = 0.3 + 0.03 * np.arange(400)  # 0.3 .. 12.27 eV
BUMP_AT, BUMP_AMP = 2.0, 0.6


def _cube(ny=12, nx=10, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    scale = (
        0.8 + 0.6 * (yy / ny) + 0.4 * (xx / nx)
    )  # ZLP-tail intensity varies smoothly across the scan
    tail = 5.0 * E**-2.2 + 0.3
    bump = BUMP_AMP * np.exp(-0.5 * ((E - BUMP_AT) / 0.15) ** 2)
    cube = scale[..., None] * tail + bump + rng.normal(0, noise, (ny, nx, E.size))
    return cube, bump


def _ds(cube, e0=0.3):
    return Dataset3deels.from_array(
        cube, sampling=[30.0, 30.0, 0.03], origin=[0, 0, e0], units=["nm", "nm", "eV"]
    )


WINS = ((0.5, 0.9), (3.0, 4.0))


def test_recovers_bump_and_flat_baseline():
    cube, bump = _cube()
    out = subtract_background_two_sided(_ds(cube), WINS, form="powerlaw_const")
    mean = np.asarray(out.array).mean(axis=(0, 1))
    i = int(np.argmin(abs(E - BUMP_AT)))
    assert mean[i] == pytest.approx(bump[i], abs=0.12)
    far = (E > 5.0) & (E < 10.0)
    assert abs(mean[far].mean()) < 0.03


def test_binning_reduces_pixel_noise_and_keeps_shape():
    cube, _ = _cube(noise=0.3, seed=1)
    truth = np.broadcast_to(_cube(noise=0.0)[1], cube.shape)
    plain = np.asarray(subtract_background_two_sided(_ds(cube), WINS, bin_factor=1).array)
    binned = np.asarray(subtract_background_two_sided(_ds(cube), WINS, bin_factor=2).array)
    assert binned.shape == cube.shape
    # per-pixel error of the recovered bump region: binned fit (less noisy background) is at least as good
    sel = (E > 1.5) & (E < 2.5)
    assert np.std((binned - truth)[..., sel]) <= np.std((plain - truth)[..., sel]) * 1.02


def test_forms_run_and_negative_values_kept_unless_clipped():
    cube, _ = _cube()
    ds = _ds(cube)
    for form in ("powerlaw", "powerlaw_const", "powerlaw_curved", "polynomial"):
        out = subtract_background_two_sided(ds, WINS, form=form, polynomial_degree=3)
        assert np.asarray(out.array).shape == cube.shape
    raw = np.asarray(subtract_background_two_sided(ds, WINS).array)
    clipped = np.asarray(subtract_background_two_sided(ds, WINS, clip_negative=True).array)
    assert raw.min() < 0 <= clipped.min()


def test_input_untouched_and_details():
    cube, _ = _cube()
    ds = _ds(cube)
    before = np.array(ds.array, copy=True)
    out, det = subtract_background_two_sided(ds, WINS, bin_factor=3, return_details=True)
    np.testing.assert_array_equal(ds.array, before)
    assert det["r"] == pytest.approx(2.2, abs=0.4)
    assert (
        det["background"].shape == cube.shape and det["n_blocks"] == 4 * 4
    )  # ceil(12/3) x ceil(10/3)


def test_window_validation():
    cube, _ = _cube()
    ds = _ds(cube)
    with pytest.raises(ValueError, match="outside the energy axis"):
        subtract_background_two_sided(ds, ((0.1, 0.5), (3.0, 4.0)))
    with pytest.raises(ValueError, match="lo < hi"):
        subtract_background_two_sided(ds, ((1.0, 0.5),))


def test_fit_windows_inside_axis():
    ds = _ds(_cube()[0], e0=0.3)
    assert fit_windows_inside_axis(ds, ((0.5, 0.8), (2.4, 3.2))) == [(0.5, 0.8), (2.4, 3.2)]
    ds2 = Dataset3deels.from_array(
        _cube()[0], sampling=[30.0, 30.0, 0.03], origin=[0, 0, 1.215], units=["nm", "nm", "eV"]
    )
    w = fit_windows_inside_axis(ds2, ((0.5, 0.8), (2.4, 3.2)))
    assert (
        w[0][0] > 1.2 and w[0][1] - w[0][0] == pytest.approx(0.3, abs=1e-6) and w[1] == (2.4, 3.2)
    )


def test_curved_form_handles_bending_tail_better():
    rng = np.random.default_rng(3)
    tail = (
        4.0 * E**-2.0 * np.exp(-0.5 * np.log(E) ** 2) + 0.2
    )  # power law whose exponent drifts with energy
    bump = 0.5 * np.exp(-0.5 * ((E - 2.0) / 0.2) ** 2)
    cube = np.broadcast_to(tail + bump, (8, 8, E.size)) + rng.normal(0, 0.02, (8, 8, E.size))
    wins = ((0.5, 0.8), (3.5, 5.0))
    flat_region = (E > 1.0) & (E < 1.5)  # between the windows, away from the bump
    err = {}
    for form in ("powerlaw_const", "powerlaw_curved"):
        out = np.asarray(subtract_background_two_sided(_ds(cube), wins, form=form).array).mean(
            axis=(0, 1)
        )
        err[form] = float(np.abs(out[flat_region]).mean())
    assert err["powerlaw_curved"] < 0.5 * err["powerlaw_const"]


def test_exclude_windows_are_never_fitted():
    cube, _ = _cube()
    # a huge feature sitting inside a fit window wrecks the fit unless it is excluded
    spike_at = (E > 3.4) & (E < 3.7)
    dirty = cube.copy()
    dirty[..., spike_at] += 5.0
    wins = ((0.5, 0.9), (3.0, 4.0))
    ds = _ds(dirty)
    bad = np.asarray(subtract_background_two_sided(ds, wins).array).mean(axis=(0, 1))
    good, det = subtract_background_two_sided(
        ds, wins, exclude_windows=((3.3, 3.8),), return_details=True
    )
    good = np.asarray(good.array).mean(axis=(0, 1))
    assert not det["fit_mask"][(E > 3.3) & (E < 3.8)].any()
    far = (E > 6.0) & (E < 10.0)
    assert abs(good[far].mean()) < abs(bad[far].mean()) / 3


def test_exclusion_leaving_too_few_channels_raises():
    cube, _ = _cube()
    with pytest.raises(ValueError, match="after exclude_windows"):
        subtract_background_two_sided(_ds(cube), ((0.5, 0.9),), exclude_windows=((0.4, 1.0),))
