import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LinearSegmentedColormap, Normalize

from quantem.core.visualization.visualization_utils import (
    ScalebarConfig,
    _resolve_scalebar,
    add_arg_cbar_to_ax,
    add_cbar_to_ax,
    add_scalebar_to_ax,
    array_to_rgba,
    bilinear_histogram_2d,
    estimate_scalebar_length,
    list_of_arrays_to_rgba,
    turbo_black,
)


@pytest.fixture
def sample_array():
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)


@pytest.fixture
def sample_angle_array():
    return np.array(
        [
            [0.0, np.pi / 4, np.pi / 2],
            [-np.pi / 2, 0.0, np.pi / 4],
            [-np.pi, -3 * np.pi / 4, -np.pi / 2],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def sample_complex_array():
    return np.array(
        [[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j, 9 + 9j]],
        dtype=np.complex128,
    )


@pytest.fixture
def sample_arrays():
    return [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
    ]


@pytest.fixture
def mock_fig_ax():
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


class TestArrayToRGBA:
    def test_array_to_rgba_without_angle(self, sample_array):
        rgba = array_to_rgba(sample_array)
        assert rgba.shape == (*sample_array.shape, 4)
        assert np.all(rgba[..., 3] == 1)  # alpha channel should be 1

    def test_array_to_rgba_with_angle(self, sample_array, sample_angle_array):
        rgba = array_to_rgba(sample_array, sample_angle_array)
        assert rgba.shape == (*sample_array.shape, 4)
        assert np.all(rgba[..., 3] == 1)  # alpha channel should be 1

    def test_array_to_rgba_with_custom_cmap(self, sample_array):
        rgba = array_to_rgba(sample_array, cmap="viridis")
        assert rgba.shape == (*sample_array.shape, 4)

    def test_array_to_rgba_with_chroma_boost(self, sample_array, sample_angle_array):
        rgba = array_to_rgba(sample_array, sample_angle_array, chroma_boost=2.0)
        assert rgba.shape == (*sample_array.shape, 4)

    def test_array_to_rgba_shape_mismatch(self, sample_array):
        wrong_shape = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            array_to_rgba(sample_array, wrong_shape)


class TestListOfArraysToRGBA:
    def test_list_of_arrays_to_rgba(self, sample_arrays):
        rgba = list_of_arrays_to_rgba(sample_arrays)
        assert rgba.shape == (*sample_arrays[0].shape, 4)
        assert np.all(rgba[..., 3] == 1)  # alpha channel should be 1

    def test_list_of_arrays_to_rgba_with_chroma_boost(self, sample_arrays):
        rgba = list_of_arrays_to_rgba(sample_arrays, chroma_boost=2.0)
        assert rgba.shape == (*sample_arrays[0].shape, 4)


class TestScalebarConfig:
    def test_scalebar_config_default(self):
        config = ScalebarConfig()
        assert config.sampling == 1.0
        assert config.units == "pixels"
        assert config.length is None
        assert config.width_px == 1
        assert config.pad_px == 0.5
        assert config.color == "white"
        assert config.loc == "lower right"

    def test_scalebar_config_custom(self):
        config = ScalebarConfig(
            sampling=2.0,
            units="nm",
            length=10.0,
            width_px=2,
            pad_px=1.0,
            color="black",
            loc="upper left",
        )
        assert config.sampling == 2.0
        assert config.units == "nm"
        assert config.length == 10.0
        assert config.width_px == 2
        assert config.pad_px == 1.0
        assert config.color == "black"
        assert config.loc == "upper left"


class TestResolveScalebar:
    def test_resolve_scalebar_none(self):
        config = _resolve_scalebar(None)
        assert config is None

    def test_resolve_scalebar_false(self):
        config = _resolve_scalebar(False)
        assert config is None

    def test_resolve_scalebar_true(self):
        config = _resolve_scalebar(True)
        assert isinstance(config, ScalebarConfig)

    def test_resolve_scalebar_dict(self):
        config = _resolve_scalebar({"sampling": 2.0, "units": "nm"})
        assert isinstance(config, ScalebarConfig)
        assert config.sampling == 2.0
        assert config.units == "nm"

    def test_resolve_scalebar_config(self):
        original = ScalebarConfig(sampling=2.0, units="nm")
        config = _resolve_scalebar(original)
        assert config is original

    def test_resolve_scalebar_invalid_type(self):
        with pytest.raises(TypeError):
            _resolve_scalebar(123)  # type: ignore


class TestEstimateScalebarLength:
    def test_estimate_scalebar_length(self):
        length, length_px = estimate_scalebar_length(100, 1.0)
        assert isinstance(length, float)
        assert isinstance(length_px, float)

    def test_estimate_scalebar_length_with_sampling(self):
        length, length_px = estimate_scalebar_length(100, 2.0)
        assert length > 0
        assert length_px > 0


class TestAddScalebarToAx:
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


class TestAddCbarToAx:
    def test_add_cbar_to_ax(self, mock_fig_ax):
        fig, ax = mock_fig_ax
        data = np.random.rand(10, 10)
        norm = Normalize(vmin=data.min(), vmax=data.max())
        cmap = LinearSegmentedColormap.from_list("test", ["blue", "red"])
        cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # [left, bottom, width, height]
        cbar = add_cbar_to_ax(fig, cax=cax, norm=norm, cmap=cmap)
        assert isinstance(cbar, Colorbar)


class TestAddArgCbarToAx:
    def test_add_arg_cbar_to_ax(self, mock_fig_ax):
        fig, ax = mock_fig_ax
        # Create sample complex data
        data = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
        cmap = LinearSegmentedColormap.from_list("test", ["blue", "red"])
        norm = Normalize(vmin=0, vmax=2 * np.pi)
        ax.imshow(np.angle(data), cmap=cmap, norm=norm)
        cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # [left, bottom, width, height]
        cbar = add_arg_cbar_to_ax(fig, cax=cax)
        assert isinstance(cbar, Colorbar)

    def test_add_arg_cbar_to_ax_with_chroma_boost(self, mock_fig_ax):
        fig, ax = mock_fig_ax
        cbar = add_arg_cbar_to_ax(fig, ax, chroma_boost=2.0)
        assert cbar is not None
        assert hasattr(cbar, "ax")
        assert isinstance(cbar.ax, Axes)


class TestTurboBlack:
    def test_turbo_black_default(self):
        cmap = turbo_black()
        assert isinstance(cmap, colors.ListedColormap)
        assert cmap.N == 256

    def test_turbo_black_custom_num_colors(self):
        cmap = turbo_black(num_colors=128)
        assert isinstance(cmap, colors.ListedColormap)
        assert cmap.N == 128

    def test_turbo_black_custom_fade_len(self):
        cmap = turbo_black(fade_len=32)
        assert isinstance(cmap, colors.ListedColormap)
        assert cmap.N == 256


class TestBilinearHistogram2D:
    def test_basic_histogram(self):
        x = np.random.rand(100)
        y = np.random.rand(100)
        weight = np.ones_like(x)
        shape = (10, 10)
        hist = bilinear_histogram_2d(x=x, y=y, weight=weight, shape=shape)
        assert hist.shape == shape

    def test_with_custom_weights(self):
        x = np.random.rand(100)
        y = np.random.rand(100)
        weight = np.random.rand(100)
        shape = (10, 10)
        hist = bilinear_histogram_2d(x=x, y=y, weight=weight, shape=shape)
        assert hist.shape == shape

    def test_bilinear_histogram_2d(self, sample_array):
        # Create a grid of points that matches the sample array size
        x = np.linspace(0, 1, 3, dtype=np.float64)  # 3 points to match sample_array shape
        y = np.linspace(0, 1, 3, dtype=np.float64)
        X, Y = np.meshgrid(x, y)
        Z = sample_array.flatten()
        weight = np.ones_like(Z)

        hist = bilinear_histogram_2d(
            (10, 10),
            X.flatten(),
            Y.flatten(),
            weight,
            origin=(0.0, 0.0),
            sampling=(1.0, 1.0),
            statistic="sum",
        )
        assert hist.shape == (10, 10)
        assert not np.any(np.isnan(hist))
        assert np.all(hist >= 0)

    def test_bilinear_histogram_2d_with_custom_origin(self):
        shape = (10, 10)
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        weight = np.array([1, 1, 1])
        hist = bilinear_histogram_2d(shape, x, y, weight, origin=(0.5, 0.5))
        assert hist.shape == shape
        assert np.all(~np.isnan(hist[hist > 0]))  # Only check non-zero values for NaN

    def test_bilinear_histogram_2d_with_custom_sampling(self):
        shape = (10, 10)
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        weight = np.array([1, 1, 1])
        hist = bilinear_histogram_2d(shape, x, y, weight, sampling=(2.0, 2.0))
        assert hist.shape == shape
        assert np.all(~np.isnan(hist[hist > 0]))  # Only check non-zero values for NaN

    def test_bilinear_histogram_2d_with_custom_statistic(self):
        shape = (10, 10)
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        weight = np.array([1, 1, 1])
        hist = bilinear_histogram_2d(shape, x, y, weight, statistic="mean")
        assert hist.shape == shape
        assert np.all(~np.isnan(hist[hist > 0]))  # Only check non-zero values for NaN
