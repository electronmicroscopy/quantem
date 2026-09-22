"""Tests for the Wave 4 background / energy-window diagnostics on Dataset3deels:
``compare_background_methods``, ``plot_background_fit_ranges``,
``show_energy_windows_with_peaks`` and ``plot_energy_windows_summary``."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from quantem.spectroscopy.dataset3deels import Dataset3deels  # noqa: E402

E = 250.0 + 0.1 * np.arange(800)  # 250 .. 329.9 eV
EDGE = 284.0
NY, NX = 6, 7


def _hl(seed=0):
    """Power-law background + a step edge at EDGE + a peak at 290 eV, mild noise."""
    rng = np.random.default_rng(seed)
    bg = 1e7 * E**-2.5
    edge = np.where(E > EDGE, 0.15 * bg[np.argmin(np.abs(E - EDGE))], 0.0)
    peak = 3.0 * np.exp(-0.5 * ((E - 290.0) / 0.8) ** 2)
    arr = np.broadcast_to(bg + edge + peak, (NY, NX, E.size)).copy()
    arr += rng.normal(0, 0.05, arr.shape)
    return Dataset3deels.from_array(
        arr, sampling=[1.0, 1.0, 0.1], origin=[0, 0, E[0]], units=["nm", "nm", "eV"], name="hl"
    )


def teardown_function():
    plt.close("all")


def test_compare_background_methods_table_and_backgrounds():
    ds = _hl()
    out = ds.compare_background_methods(
        target_edge=EDGE,
        pre_edge_range=(255.0, 276.0),
        windows=[(284.0, 287.5), (289.0, 300.0)],
        two_sided_windows=[(255.0, 276.0), (315.0, 328.0)],
        show=False,
    )
    labels = [r["method"] for r in out["table"]]
    assert labels[:4] == ["powerlaw", "linear", "polynomial deg 2", "polynomial deg 3"]
    assert len(labels) == 6  # + the two two-sided forms
    for r in out["table"][:4]:
        assert np.isfinite(r["fit_rms"]) and 0.0 <= r["worst_zero_pixel_fraction"] <= 1.0
    assert set(out["background"]) >= {"powerlaw", "linear"}
    assert out["background"]["powerlaw"].shape == E.shape
    assert out["figure"] is None


def test_compare_background_methods_window_reaching_edge_is_reported_not_raised():
    ds = _hl()
    out = ds.compare_background_methods(
        target_edge=EDGE, pre_edge_range=(255.0, EDGE + 2.0), windows=[(289.0, 300.0)], show=False
    )
    for r in out["table"]:
        assert "subtraction failed: ValueError" in r["note"]
        assert np.isnan(r["worst_zero_pixel_fraction"])


def test_plot_background_fit_ranges_recovers_one_sided_curve():
    ds = _hl()
    curve = 1e7 * E**-2.5
    after = Dataset3deels.from_array(
        np.asarray(ds.array) - curve, sampling=ds.sampling, origin=ds.origin, units=ds.units
    )
    figs = ds.plot_background_fit_ranges(
        after, None, fit_windows=[(255.0, 276.0)], ranges=((None,), (270.0, 300.0)), n_pixels=2
    )
    assert len(figs) == 2
    # top-left panel's second line is the mean fitted background, recovered from before - after
    recovered = np.asarray(figs[0].axes[0].get_lines()[1].get_ydata(), float)
    np.testing.assert_allclose(recovered, curve, rtol=1e-6)
    assert (
        ds.plot_background_fit_ranges(after, None, fit_windows=[(255.0, 276.0)], show=False) == []
    )


def test_show_energy_windows_with_peaks_finds_the_peak():
    ds = _hl()
    wins = [(286.0, 294.0), (300.0, 310.0)]
    sub = ds.subtract_background_two_sided([(255.0, 276.0), (315.0, 328.0)], form="powerlaw")
    detected = sub.show_energy_windows_with_peaks(wins, target_edge=EDGE)
    assert set(detected) == set(wins)
    assert any(abs(p - 290.0) < 0.5 for p in detected[(286.0, 294.0)])
    assert sub.show_energy_windows_with_peaks(wins, detect_peaks=False) == {w: [] for w in wins}


def test_plot_energy_windows_summary_rows_match_summary():
    ds = _hl()
    wins = [(286.0, 294.0), (300.0, 310.0)]
    rows, fig = ds.plot_energy_windows_summary(wins, title="t")
    assert rows == ds.summarize_energy_windows(wins)
    assert fig is not None
    rows2, fig2 = ds.plot_energy_windows_summary(wins, show=False)
    assert fig2 is None and rows2 == rows
