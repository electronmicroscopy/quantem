"""
Tests for the Dataset4dstem class in quantem.core.datastructures.dataset4dstem
"""

import numpy as np
import pytest

from quantem.core.datastructures.dataset4dstem import Dataset4dstem


@pytest.fixture
def sample_4d_array():
    """Create a sample 4D array for testing."""
    return np.random.rand(5, 5, 10, 10)  # (scan_y, scan_x, dp_y, dp_x)


@pytest.fixture
def sample_dataset4dstem(sample_4d_array):
    """Create a sample 4D-STEM dataset for testing."""
    return Dataset4dstem.from_array(
        array=sample_4d_array,
        name="test_4dstem_dataset",
        origin=(0, 0, 0, 0),
        sampling=(1, 1, 0.1, 0.1),
        units=["nm", "nm", "1/nm", "1/nm"],
        signal_units="counts",
    )


class TestDataset4dstemInitialization:
    """Test Dataset4dstem initialization and factory methods."""

    def test_from_array(self, sample_4d_array):
        """Test creating a Dataset4dstem from a 4D array."""
        dataset = Dataset4dstem.from_array(
            array=sample_4d_array,
            name="test_dataset",
            origin=(0, 0, 0, 0),
            sampling=(1, 1, 0.1, 0.1),
            units=["nm", "nm", "1/nm", "1/nm"],
        )

        assert dataset.name == "test_dataset"
        assert dataset.ndim == 4
        assert dataset.shape == (5, 5, 10, 10)
        assert np.array_equal(dataset.origin, np.array([0, 0, 0, 0]))
        assert np.array_equal(dataset.sampling, np.array([1, 1, 0.1, 0.1]))
        assert dataset.units == ["nm", "nm", "1/nm", "1/nm"]
        assert dataset.signal_units == "arb. units"  # Default value

    def test_from_array_defaults(self, sample_4d_array):
        """Test creating a Dataset4dstem with default parameters."""
        dataset = Dataset4dstem.from_array(array=sample_4d_array)

        assert dataset.name == "4D-STEM dataset"  # Default name
        assert dataset.ndim == 4
        assert np.array_equal(dataset.origin, np.array([0, 0, 0, 0]))  # Default origin
        assert np.array_equal(dataset.sampling, np.array([1, 1, 1, 1]))  # Default sampling
        assert dataset.units == [
            "pixels",
            "pixels",
            "pixels",
            "pixels",
        ]  # Default units
        assert dataset.signal_units == "arb. units"  # Default signal units


class TestDataset4dstemGetItem:
    """Test Dataset4dstem __getitem__ method."""

    def test_getitem_2d(self, sample_dataset4dstem):
        """Test __getitem__ method to get a 2D slice (single diffraction pattern)."""
        # Get a 2D slice (single diffraction pattern)
        dp = sample_dataset4dstem[0, 0]
        assert dp.ndim == 2
        assert dp.shape == (10, 10)

        # Check that the data is correct
        assert np.array_equal(dp.array, sample_dataset4dstem.array[0, 0])

    def test_getitem_3d(self, sample_dataset4dstem):
        """Test __getitem__ method to get a 3D slice (line of diffraction patterns)."""
        # Get a 3D slice (line of diffraction patterns)
        line = sample_dataset4dstem[0, :]
        assert line.ndim == 3
        assert line.shape == (5, 10, 10)

        # Check that the data is correct
        assert np.array_equal(line.array, sample_dataset4dstem.array[0, :])


class TestDataset4dstemDpMethods:
    """Test Dataset4dstem diffraction pattern methods."""

    def test_dp_mean_property(self, sample_dataset4dstem):
        """Test dp_mean property."""
        # Initially, dp_mean should not be attached
        assert not hasattr(sample_dataset4dstem, "_dp_mean")

        # Accessing dp_mean should trigger calculation but not attach
        dp_mean = sample_dataset4dstem.dp_mean
        assert not hasattr(sample_dataset4dstem, "_dp_mean")
        assert dp_mean.ndim == 2
        assert dp_mean.shape == (10, 10)

        # Accessing again should recalculate
        dp_mean2 = sample_dataset4dstem.dp_mean
        assert dp_mean2 is not dp_mean  # Should be a new calculation each time

    def test_get_dp_mean(self, sample_dataset4dstem):
        """Test get_dp_mean method."""
        # Get dp_mean without attaching
        dp_mean = sample_dataset4dstem.get_dp_mean(attach=False)
        assert not hasattr(sample_dataset4dstem, "_dp_mean")
        assert dp_mean.ndim == 2
        assert dp_mean.shape == (10, 10)

        # Get dp_mean with attaching
        dp_mean2 = sample_dataset4dstem.get_dp_mean(attach=True)
        assert hasattr(sample_dataset4dstem, "_dp_mean")
        assert dp_mean2 is sample_dataset4dstem._dp_mean

    def test_dp_max_property(self, sample_dataset4dstem):
        """Test dp_max property."""
        # Initially, dp_max should not be attached
        assert not hasattr(sample_dataset4dstem, "_dp_max")

        # Accessing dp_max should trigger calculation but not attach
        dp_max = sample_dataset4dstem.dp_max
        assert not hasattr(sample_dataset4dstem, "_dp_max")
        assert dp_max.ndim == 2
        assert dp_max.shape == (10, 10)

        # Accessing again should recalculate
        dp_max2 = sample_dataset4dstem.dp_max
        assert dp_max2 is not dp_max  # Should be a new calculation each time

    def test_get_dp_max(self, sample_dataset4dstem):
        """Test get_dp_max method."""
        # Get dp_max without attaching
        dp_max = sample_dataset4dstem.get_dp_max(attach=False)
        assert not hasattr(sample_dataset4dstem, "_dp_max")
        assert dp_max.ndim == 2
        assert dp_max.shape == (10, 10)

        # Get dp_max with attaching
        dp_max2 = sample_dataset4dstem.get_dp_max(attach=True)
        assert hasattr(sample_dataset4dstem, "_dp_max")
        assert dp_max2 is sample_dataset4dstem._dp_max

    def test_dp_median_property(self, sample_dataset4dstem):
        """Test dp_median property."""
        # Initially, dp_median should not be attached
        assert not hasattr(sample_dataset4dstem, "_dp_median")

        # Accessing dp_median should trigger calculation but not attach
        dp_median = sample_dataset4dstem.dp_median
        assert not hasattr(sample_dataset4dstem, "_dp_median")
        assert dp_median.ndim == 2
        assert dp_median.shape == (10, 10)

        # Accessing again should recalculate
        dp_median2 = sample_dataset4dstem.dp_median
        assert dp_median2 is not dp_median  # Should be a new calculation each time

    def test_get_dp_median(self, sample_dataset4dstem):
        """Test get_dp_median method."""
        # Get dp_median without attaching
        dp_median = sample_dataset4dstem.get_dp_median(attach=False)
        assert not hasattr(sample_dataset4dstem, "_dp_median")
        assert dp_median.ndim == 2
        assert dp_median.shape == (10, 10)

        # Get dp_median with attaching
        dp_median2 = sample_dataset4dstem.get_dp_median(attach=True)
        assert hasattr(sample_dataset4dstem, "_dp_median")
        assert dp_median2 is sample_dataset4dstem._dp_median

    def test_get_virtual_image(self, sample_dataset4dstem):
        """Test get_virtual_image method."""
        # Create a simple mask
        mask = np.zeros((10, 10))
        mask[4:6, 4:6] = 1  # A 2x2 square in the center

        # Get virtual image without attaching
        virtual_image = sample_dataset4dstem.get_virtual_image(
            mask=mask, name="test_virtual_image", attach=False
        )
        assert virtual_image.ndim == 2
        assert virtual_image.shape == (5, 5)
        assert "test_virtual_image" not in sample_dataset4dstem.virtual_images

        # Get virtual image with attaching
        virtual_image2 = sample_dataset4dstem.get_virtual_image(
            mask=mask, name="test_virtual_image", attach=True
        )
        assert "test_virtual_image" in sample_dataset4dstem.virtual_images
        assert virtual_image2 is sample_dataset4dstem.virtual_images["test_virtual_image"]


class TestDataset4dstemShow:
    """Test Dataset4dstem show method."""

    def test_show(self, sample_dataset4dstem, monkeypatch):
        """Test show method."""

        # Mock the show_2d function to avoid actual plotting
        def mock_show_2d(*args, **kwargs):
            return None

        # Apply the mock
        monkeypatch.setattr("quantem.core.visualization.visualization.show_2d", mock_show_2d)

        # Call show method
        result = sample_dataset4dstem.show()

        # Check that the method returns a tuple of (fig, axs)
        assert isinstance(result, tuple)
        assert len(result) == 2
