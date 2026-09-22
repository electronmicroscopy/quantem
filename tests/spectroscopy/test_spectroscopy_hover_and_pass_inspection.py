"""Tests for the hover / coordinate-reading plots and ``inspect_single_pass``
(quantem.spectroscopy.spectroscopy_visualzitions)."""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.backend_bases import MouseEvent  # noqa: E402
from matplotlib.text import Annotation  # noqa: E402

import quantem.core.io.file_readers as file_readers  # noqa: E402
from quantem.spectroscopy.dataset3deels import Dataset3deels  # noqa: E402
from quantem.spectroscopy.spectroscopy_visualzitions import (  # noqa: E402
    attach_hover_to_axes,
    compare_eels_spectra_with_hover,
    inspect_single_pass,
)

E = -2.0 + 0.02 * np.arange(500)  # -2 .. 7.98 eV


def _ds(seed=0):
    rng = np.random.default_rng(seed)
    spec = (
        1000.0 * np.exp(-0.5 * (E / 0.1) ** 2) + 5.0 + 2.0 * np.exp(-0.5 * ((E - 4.0) / 0.5) ** 2)
    )
    arr = spec + rng.normal(0, 0.1, (5, 6, E.size))
    return Dataset3deels.from_array(
        arr, sampling=[1.0, 1.0, 0.02], origin=[0, 0, E[0]], units=["nm", "nm", "eV"], name="ll"
    )


def _hover_text(fig, ax):
    """Move the mouse to the centre of `ax`; return the visible annotation texts."""
    fig.canvas.draw()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    px, py = ax.transData.transform((0.5 * (x0 + x1), 0.5 * (y0 + y1)))
    fig.canvas.callbacks.process(
        "motion_notify_event", MouseEvent("motion_notify_event", fig.canvas, px, py)
    )
    return [
        c.get_text()
        for c in ax.get_children()
        if isinstance(c, Annotation) and c.get_visible() and c.get_text()
    ]


def teardown_function():
    plt.close("all")


def _new_figs(before):
    return [plt.figure(n) for n in plt.get_fignums() if n not in before]


def test_attach_hover_to_axes_shows_energy_coordinates():
    fig, ax = plt.subplots()
    ax.plot(E, np.exp(-(E**2)))
    attach_hover_to_axes(fig, ax)
    texts = _hover_text(fig, ax)
    assert texts and texts[0].startswith("Energy:") and "eV" in texts[0]


def test_show_eels_spectrum_with_hover():
    before = set(plt.get_fignums())
    _ds().show_eels_spectrum_with_hover(title="t", highlight_range=(3.5, 4.5))
    (fig,) = _new_figs(before)
    assert any(t.startswith("Energy:") for t in _hover_text(fig, fig.axes[0]))


def test_compare_eels_spectra_with_hover_plots_every_dataset():
    before = set(plt.get_fignums())
    compare_eels_spectra_with_hover({"a": _ds(0), "b": _ds(1)}, title="t")
    (fig,) = _new_figs(before)
    assert len(fig.axes[0].get_lines()) >= 2
    assert _hover_text(fig, fig.axes[0])


def test_show_low_loss_zlp_cutoff_inspection():
    before = set(plt.get_fignums())
    _ds().show_low_loss_zlp_cutoff_inspection(title="t", current_cutoff_eV=0.5)
    figs = _new_figs(before)
    assert figs and any(_hover_text(f, ax) for f in figs for ax in f.axes)


def test_despike_preview_has_hover_on_every_panel():
    before = set(plt.get_fignums())
    _ds().despike([(4.0, 4.1)], show=True)
    (fig,) = _new_figs(before)
    assert all(_hover_text(fig, ax) for ax in fig.axes)


# ---------------------------------------------------------------- inspect_single_pass


def _fake_raw(n_passes=4, n_energy=50, ny=3, nx=4):
    rng = np.random.default_rng(0)
    # (n_frames, n_energy, ny, nx), energy on axis 1 as in MultipassRawStacks
    ll = rng.random((n_passes, n_energy, ny, nx)) + np.arange(n_passes)[:, None, None, None]
    hl = rng.random((n_passes, n_energy, ny, nx))
    adf = rng.random((n_passes, ny, nx))
    return SimpleNamespace(
        n_passes=n_passes,
        ll_stack=ll,
        hl_stack=hl,
        adf_stack=adf,
        ll_energy_axis=np.linspace(-1, 5, n_energy),
        hl_energy_axis=np.linspace(280, 300, n_energy),
    )


def test_inspect_single_pass_values(monkeypatch):
    raw = _fake_raw()
    monkeypatch.setattr(file_readers, "load_multipass_raw_stacks", lambda folder: raw)
    out = inspect_single_pass("any-folder", pass_number=3, good_passes="1-2", show=False)
    np.testing.assert_allclose(out["ll_spectrum_i"], raw.ll_stack[2].mean(axis=(1, 2)))
    np.testing.assert_allclose(out["ll_spectrum_1"], raw.ll_stack[0].mean(axis=(1, 2)))
    np.testing.assert_allclose(out["hl_mean_overall"], raw.hl_stack[[0, 1]].mean(axis=(0, 2, 3)))
    np.testing.assert_array_equal(out["adf_image_i"], raw.adf_stack[2])
    assert out["pass_number"] == 3 and out["good_passes_used"] == [1, 2]
    # default pass_number is the last pass
    assert inspect_single_pass("f", show=False)["pass_number"] == raw.n_passes


def test_inspect_single_pass_rejects_out_of_range_pass(monkeypatch):
    monkeypatch.setattr(file_readers, "load_multipass_raw_stacks", lambda folder: _fake_raw())
    with pytest.raises(ValueError, match="out of range"):
        inspect_single_pass("f", pass_number=9, show=False)


def test_inspect_single_pass_plots_with_hover(monkeypatch):
    monkeypatch.setattr(file_readers, "load_multipass_raw_stacks", lambda folder: _fake_raw())
    before = set(plt.get_fignums())
    inspect_single_pass("f", pass_number=2, show=True)
    figs = _new_figs(before)
    spectra_fig = figs[-1]
    assert all(_hover_text(spectra_fig, ax) for ax in spectra_fig.axes)
