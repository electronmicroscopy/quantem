import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LinearSegmentedColormap, Normalize

from quantem.core import config
from quantem.core.visualization.custom_normalizations import CustomNormalization
from quantem.core.visualization.visualization_utils import (
    ScalebarConfig,
    _resolve_scalebar,
    add_arg_cbar_to_ax,
    add_cbar_to_ax,
    add_scalebar_to_ax,
    array_to_rgba,
    bilinear_histogram_2d,
    combine_arrays_to_rgba,
    estimate_scalebar_length,
    turbo_black,
)


@pytest.fixture
def mock_fig_ax():
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


class TestArrayToRGBA:
    def test_array_to_rgba_without_angle_applies_cmap(self):
        amplitude = np.linspace(0.0, 1.0, 9).reshape(3, 3)
        for cmap in ("gray", "viridis"):
            rgba = array_to_rgba(amplitude, cmap=cmap)
            assert np.allclose(rgba, mpl.colormaps[cmap](amplitude)), cmap

    def test_array_to_rgba_with_angle(self):
        amplitude = np.array([[0.0, 0.5, 1.0], [0.5, 0.5, 0.5]])
        angle = np.array([[0.0, 0.0, 0.0], [0.0, np.pi / 2, -np.pi / 2]])
        rgba = array_to_rgba(amplitude, angle)
        boosted = array_to_rgba(amplitude, angle, chroma_boost=2.0)

        assert rgba.shape == (2, 3, 4)
        assert np.all(rgba[..., 3] == 1)  # alpha channel should be 1
        assert np.allclose(rgba[0, 0, :3], 0.0)  # zero amplitude is black
        # lightness grows with amplitude at a fixed angle
        assert rgba[0, 0, :3].sum() < rgba[0, 1, :3].sum() < rgba[0, 2, :3].sum()
        # equal amplitude, different angles -> different hues
        assert not np.allclose(rgba[1, 0, :3], rgba[1, 1, :3])
        assert not np.allclose(rgba[1, 1, :3], rgba[1, 2, :3])
        # a chroma boost spreads the channels further apart
        assert np.all(np.ptp(boosted[1, :, :3], axis=-1) > np.ptp(rgba[1, :, :3], axis=-1))

    def test_array_to_rgba_shape_mismatch(self):
        with pytest.raises(ValueError):
            array_to_rgba(np.ones((3, 3)), np.ones((2, 2)))


class TestCombineArraysToRGBA:
    def test_combine_arrays_to_rgba(self):
        norm = CustomNormalization(interval_type="manual", vmin=0, vmax=1)
        first = np.array([[1.0, 0.0], [0.0, 0.0]])
        second = np.array([[0.0, 1.0], [0.0, 0.0]])
        rgba = combine_arrays_to_rgba([first, second], norm=norm)
        boosted = combine_arrays_to_rgba([first, second], norm=norm, chroma_boost=2.0)

        assert rgba.shape == (2, 2, 4)
        assert np.all(rgba[..., 3] == 1)  # alpha channel should be 1
        assert np.allclose(rgba[1, :, :3], 0.0)  # no signal is black
        assert not np.allclose(rgba[0, 0, :3], rgba[0, 1, :3])  # each array gets its own hue
        assert not np.allclose(boosted[0, :, :3], rgba[0, :, :3])


class TestScalebar:
    def test_resolve_scalebar(self):
        assert _resolve_scalebar(None) is None
        assert _resolve_scalebar(False) is None
        assert _resolve_scalebar(True) == ScalebarConfig()
        assert _resolve_scalebar({"sampling": 2.0, "units": "nm"}) == ScalebarConfig(
            sampling=2.0, units="nm"
        )
        assert _resolve_scalebar(None, sampling=2.0) == ScalebarConfig(
            sampling=2.0, units=config.get("viz.real_space_units")
        )
        original = ScalebarConfig(sampling=2.0, units="nm")
        assert _resolve_scalebar(original) is original
        with pytest.raises(TypeError):
            _resolve_scalebar(123)  # type: ignore

    def test_estimate_scalebar_length(self):
        # a "nice" length near a quarter of the field of view, and its size in pixels
        assert estimate_scalebar_length(100, 1.0) == (20.0, 20.0)
        assert estimate_scalebar_length(100, 2.0) == (50.0, 25.0)
        assert estimate_scalebar_length(100, (2.0, 2.0)) == (50.0, 25.0)
        with pytest.raises(ValueError):
            estimate_scalebar_length(100, (1.0, 2.0))

    def test_add_scalebar_to_ax(self, mock_fig_ax):
        fig, ax = mock_fig_ax
        add_scalebar_to_ax(
            ax,
            array_size=100,
            sampling=1.0,
            length_units=10.0,
            units="nm",
            width_px=1,
            pad_px=0.5,
            color="white",
            loc="lower right",
        )
        # Check if scalebar was added (this is a bit tricky to test directly)
        assert len(ax.artists) > 0


class TestColorbars:
    def test_add_cbar_to_ax(self, mock_fig_ax):
        fig, ax = mock_fig_ax
        norm = Normalize(vmin=0.0, vmax=1.0)
        cmap = LinearSegmentedColormap.from_list("test", ["blue", "red"])
        cax = fig.add_axes((0.85, 0.15, 0.05, 0.7))  # [left, bottom, width, height]
        cbar = add_cbar_to_ax(fig, cax=cax, norm=norm, cmap=cmap)
        assert isinstance(cbar, Colorbar)
        assert cbar.norm is norm

    def test_add_arg_cbar_to_ax(self, mock_fig_ax):
        fig, ax = mock_fig_ax
        cbar = add_arg_cbar_to_ax(fig, cax=fig.add_axes((0.85, 0.15, 0.05, 0.7)))
        boosted = add_arg_cbar_to_ax(fig, ax, chroma_boost=2.0)
        assert isinstance(cbar, Colorbar)
        assert np.allclose(cbar.get_ticks(), [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        assert not np.allclose(boosted.cmap.colors, cbar.cmap.colors)


class TestTurboBlack:
    def test_turbo_black(self):
        for num_colors, fade_len in [(256, None), (128, None), (256, 32)]:
            cmap = turbo_black(num_colors=num_colors, fade_len=fade_len)
            fade = num_colors // 8 if fade_len is None else fade_len
            turbo = mpl.colormaps["turbo"].resampled(num_colors)(np.linspace(0, 1, num_colors))
            colors = np.asarray(cmap.colors)
            assert cmap.N == num_colors, (num_colors, fade_len)
            assert np.allclose(colors[0, :3], 0.0), (num_colors, fade_len)  # starts at black
            assert np.allclose(colors[fade:], turbo[fade:]), (num_colors, fade_len)
            assert np.allclose(colors[fade - 1], turbo[fade - 1]), (num_colors, fade_len)


class TestBilinearHistogram2D:
    def test_bins_weights(self):
        x = np.array([0.5, 0.5, 2.5])
        y = np.array([0.5, 0.5, 3.5])
        weight = np.array([1.0, 2.0, 4.0])
        for kwargs, expected in [
            ({}, {(0, 0): 3.0, (2, 3): 4.0}),
            ({"statistic": "mean"}, {(0, 0): 1.5, (2, 3): 4.0}),
            # the point at y = 3.5 falls outside [-1, 3) and is dropped
            ({"origin": (-1.0, -1.0)}, {(1, 1): 3.0}),
            ({"sampling": (2.0, 2.0)}, {(0, 0): 3.0, (1, 1): 4.0}),
        ]:
            hist = bilinear_histogram_2d((4, 4), x, y, weight, **kwargs)
            assert hist.shape == (4, 4), kwargs
            empty = 0.0 if kwargs.get("statistic", "sum") == "sum" else np.nan
            want = np.full((4, 4), empty)
            for index, value in expected.items():
                want[index] = value
            np.testing.assert_allclose(hist, want, err_msg=str(kwargs))

    def test_accepts_column_vectors(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[1.5], [2.5], [3.5]])
        weight = np.array([[2.0], [3.0], [4.0]])
        hist = bilinear_histogram_2d((8, 8), x, y, weight)
        assert np.array_equal(hist, bilinear_histogram_2d((8, 8), x[:, 0], y[:, 0], weight[:, 0]))
        assert hist.sum() == 9.0
        with pytest.raises(ValueError):
            bilinear_histogram_2d((8, 8), x, y[:2], weight)
