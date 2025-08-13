import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from quantem.core.visualization.custom_normalizations import NormalizationConfig
from quantem.core.visualization.visualization import (
    _normalize_show_input_to_grid,
    _show_2d_array,
    _show_2d_combined,
    show_2d,
)
from quantem.core.visualization.visualization_utils import ScalebarConfig


@pytest.fixture
def sample_array():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture
def sample_complex_array():
    return np.array([[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j, 9 + 9j]])


@pytest.fixture
def sample_arrays():
    return [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
        np.array([[9, 10], [11, 12]]),
    ]


@pytest.fixture
def mock_fig_ax():
    fig = Figure()
    ax = fig.add_subplot(111)
    return fig, ax


class TestShow2DFunction:
    def test_show_2d_basic(self, sample_array, mock_fig_ax):
        fig, ax = _show_2d_array(sample_array, figax=mock_fig_ax)
        assert isinstance(fig, Figure)
        assert len(ax.images) == 1

    def test_show_2d_with_complex(self, sample_complex_array, mock_fig_ax):
        fig, ax = _show_2d_array(sample_complex_array, figax=mock_fig_ax)
        assert isinstance(fig, Figure)
        assert len(ax.images) == 1

    def test_show_2d_with_norm(self, sample_array, mock_fig_ax):
        norm = NormalizationConfig(interval_type="manual", vmin=1, vmax=9)
        fig, ax = _show_2d_array(sample_array, norm=norm, figax=mock_fig_ax)
        assert isinstance(fig, Figure)
        assert len(ax.images) == 1

    def test_show_2d_with_scalebar(self, sample_array, mock_fig_ax):
        scalebar = ScalebarConfig(sampling=1.0, units="nm")
        fig, ax = _show_2d_array(sample_array, scalebar=scalebar, figax=mock_fig_ax)
        assert isinstance(fig, Figure)
        assert len(ax.images) == 1
        assert len(ax.artists) > 0  # scalebar should be added

    def test_show_2d_with_cbar(self, sample_array, mock_fig_ax):
        fig, ax = _show_2d_array(sample_array, cbar=True, figax=mock_fig_ax)
        assert isinstance(fig, Figure)
        assert len(ax.images) == 1
        assert len(fig.axes) > 1  # colorbar should be added

    def test_show_2d_with_custom_cmap(self, sample_array, mock_fig_ax):
        fig, ax = _show_2d_array(sample_array, cmap="viridis", figax=mock_fig_ax)
        assert isinstance(fig, Figure)
        assert len(ax.images) == 1

    def test_show_2d_with_chroma_boost(self, sample_complex_array, mock_fig_ax):
        fig, ax = _show_2d_array(sample_complex_array, chroma_boost=2.0, figax=mock_fig_ax)
        assert isinstance(fig, Figure)
        assert len(ax.images) == 1

    def test_show_2d_with_title(self, sample_array, mock_fig_ax):
        fig, ax = _show_2d_array(sample_array, title="Test Title", figax=mock_fig_ax)
        assert isinstance(fig, Figure)
        assert len(ax.images) == 1
        assert ax.get_title() == "Test Title"


class TestShow2DCombined:
    def test_show_2d_combined_basic(self, sample_arrays, mock_fig_ax):
        fig, ax = _show_2d_combined(sample_arrays, figax=mock_fig_ax)
        assert isinstance(fig, Figure)
        assert len(ax.images) == 1

    def test_show_2d_combined_with_norm(self, sample_arrays, mock_fig_ax):
        # Convert arrays to float type before normalization
        sample_arrays = [arr.astype(float) for arr in sample_arrays]
        norm = NormalizationConfig(interval_type="manual", vmin=1, vmax=12)
        fig, ax = _show_2d_combined(sample_arrays, norm=norm, figax=mock_fig_ax)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_show_2d_combined_with_scalebar(self, sample_arrays, mock_fig_ax):
        scalebar = ScalebarConfig(sampling=1.0, units="nm")
        fig, ax = _show_2d_combined(sample_arrays, scalebar=scalebar, figax=mock_fig_ax)
        assert isinstance(fig, Figure)
        assert len(ax.images) == 1
        assert len(ax.artists) > 0  # scalebar should be added

    def test_show_2d_combined_with_custom_cmap(self, sample_arrays, mock_fig_ax):
        fig, ax = _show_2d_combined(sample_arrays, cmap="viridis", figax=mock_fig_ax)
        assert isinstance(fig, Figure)
        assert len(ax.images) == 1

    def test_show_2d_combined_with_chroma_boost(self, sample_arrays, mock_fig_ax):
        fig, ax = _show_2d_combined(sample_arrays, chroma_boost=2.0, figax=mock_fig_ax)
        assert isinstance(fig, Figure)
        assert len(ax.images) == 1


class TestNormalizeShowInputToGrid:
    def test_normalize_show_input_to_grid_single_array(self, sample_array):
        grid = _normalize_show_input_to_grid(sample_array)
        assert len(grid) == 1
        assert len(grid[0]) == 1
        assert np.array_equal(grid[0][0], sample_array)

    def test_normalize_show_input_to_grid_list(self, sample_arrays):
        grid = _normalize_show_input_to_grid(sample_arrays)
        assert len(grid) == 1
        assert len(grid[0]) == len(sample_arrays)
        for arr1, arr2 in zip(grid[0], sample_arrays):
            assert np.array_equal(arr1, arr2)

    def test_normalize_show_input_to_grid_nested_list(self, sample_arrays):
        nested = [sample_arrays[:2], sample_arrays[1:]]
        grid = _normalize_show_input_to_grid(nested)
        assert len(grid) == 2
        assert len(grid[0]) == 2
        assert len(grid[1]) == 2


class TestShow2DMain:
    # def test_show_2d_single_array(self, sample_array):
    #     fig, axs = show_2d(sample_array)
    #     assert isinstance(fig, Figure)
    #     assert len(axs.shape) == 2
    #     assert axs.shape == (1, 1)

    # def test_show_2d_list(self, sample_arrays):
    #     fig, axs = show_2d(sample_arrays)
    #     assert isinstance(fig, Figure)
    #     assert len(axs.shape) == 2
    #     assert axs.shape == (1, len(sample_arrays))

    def test_show_2d_nested_list(self, sample_arrays):
        nested = [sample_arrays[:2], sample_arrays[1:]]
        fig, axs = show_2d(nested)
        assert isinstance(fig, Figure)
        assert len(axs.shape) == 2
        assert axs.shape == (2, 2)

    def test_show_2d_with_combine_images(self, sample_arrays, mock_fig_ax):
        # Convert arrays to float type and ensure proper shape handling
        sample_arrays = [arr.astype(float) for arr in sample_arrays]
        fig, ax = show_2d(sample_arrays, combine_images=True, figax=mock_fig_ax)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert len(ax.images) == 1  # Combined images should result in a single image

    def test_show_2d_with_invalid_combine_images(self, sample_arrays):
        nested = [sample_arrays[:2], sample_arrays[1:]]
        with pytest.raises(ValueError):
            show_2d(nested, combine_images=True)

    # def test_show_2d_with_figax(self, sample_array, mock_fig_ax):
    #     fig, axs = show_2d(sample_array, figax=mock_fig_ax)
    #     assert isinstance(fig, Figure)
    #     assert len(axs.shape) == 2
    #     assert axs.shape == (1, 1)

    def test_show_2d_with_invalid_figax_shape(self, sample_arrays, mock_fig_ax):
        with pytest.raises(ValueError):
            show_2d(sample_arrays, figax=mock_fig_ax)
