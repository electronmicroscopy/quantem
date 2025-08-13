"""
Tests for the Dataset class in quantem.core.datastructures.dataset
"""

import numpy as np
import pytest

from quantem.core.datastructures.dataset import Dataset


@pytest.fixture
def sample_2d_array():
    """Create a sample 2D array for testing."""
    return np.random.rand(10, 10)


@pytest.fixture
def sample_3d_array():
    """Create a sample 3D array for testing."""
    return np.random.rand(5, 10, 15)


@pytest.fixture
def sample_dataset_2d(sample_2d_array):
    """Create a sample 2D dataset for testing."""
    return Dataset.from_array(
        array=sample_2d_array,
        name="test_2d_dataset",
        origin=(0, 0),
        sampling=(1, 1),
        units=["nm", "nm"],
        signal_units="counts",
    )


@pytest.fixture
def sample_dataset_3d(sample_3d_array):
    """Create a sample 3D dataset for testing."""
    return Dataset.from_array(
        array=sample_3d_array,
        name="test_3d_dataset",
        origin=(0, 0, 0),
        sampling=(1, 1, 1),
        units=["nm", "nm", "nm"],
        signal_units="counts",
    )


class TestDatasetInitialization:
    """Test Dataset initialization and factory methods."""

    def test_from_array_2d(self, sample_2d_array):
        """Test creating a Dataset from a 2D array."""
        dataset = Dataset.from_array(
            array=sample_2d_array,
            name="test_dataset",
            origin=(0, 0),
            sampling=(1, 1),
            units=["nm", "nm"],
        )

        assert dataset.name == "test_dataset"
        assert dataset.ndim == 2
        assert dataset.shape == (10, 10)
        assert np.array_equal(dataset.origin, np.array([0, 0]))
        assert np.array_equal(dataset.sampling, np.array([1, 1]))
        assert dataset.units == ["nm", "nm"]
        assert dataset.signal_units == "arb. units"  # Default value

    def test_from_array_defaults(self, sample_2d_array):
        """Test creating a Dataset with default parameters."""
        dataset = Dataset.from_array(array=sample_2d_array)

        assert dataset.name == "2d dataset"  # Default name based on dimensions
        assert dataset.ndim == 2
        assert np.array_equal(dataset.origin, np.array([0, 0]))  # Default origin
        assert np.array_equal(dataset.sampling, np.array([1, 1]))  # Default sampling
        assert dataset.units == ["pixels", "pixels"]  # Default units
        assert dataset.signal_units == "arb. units"  # Default signal units

    def test_direct_initialization_error(self, sample_2d_array):
        """Test that direct initialization raises an error."""
        with pytest.raises(RuntimeError):
            Dataset(
                array=sample_2d_array,
                name="test_dataset",
                origin=(0, 0),
                sampling=(1, 1),
                units=["nm", "nm"],
            )


class TestDatasetProperties:
    """Test Dataset properties and setters."""

    def test_array_property(self, sample_dataset_2d, sample_2d_array):
        """Test array property getter and setter."""
        # Test getter
        assert np.array_equal(sample_dataset_2d.array, sample_2d_array)

        # Test setter
        new_array = np.ones((10, 10))
        sample_dataset_2d.array = new_array
        assert np.array_equal(sample_dataset_2d.array, new_array)

    def test_name_property(self, sample_dataset_2d):
        """Test name property getter and setter."""
        # Test getter
        assert sample_dataset_2d.name == "test_2d_dataset"

        # Test setter
        sample_dataset_2d.name = "new_name"
        assert sample_dataset_2d.name == "new_name"

    def test_origin_property(self, sample_dataset_2d):
        """Test origin property getter and setter."""
        # Test getter
        assert np.array_equal(sample_dataset_2d.origin, np.array([0, 0]))

        # Test setter
        sample_dataset_2d.origin = (1, 1)
        assert np.array_equal(sample_dataset_2d.origin, np.array([1, 1]))

        # Test validation
        with pytest.raises(ValueError):
            sample_dataset_2d.origin = (1, 1, 1)  # Wrong number of dimensions

    def test_sampling_property(self, sample_dataset_2d):
        """Test sampling property getter and setter."""
        # Test getter
        assert np.array_equal(sample_dataset_2d.sampling, np.array([1, 1]))

        # Test setter
        sample_dataset_2d.sampling = (2, 2)
        assert np.array_equal(sample_dataset_2d.sampling, np.array([2, 2]))

        # Test validation
        with pytest.raises(ValueError):
            sample_dataset_2d.sampling = (2, 2, 2)  # Wrong number of dimensions

    def test_units_property(self, sample_dataset_2d):
        """Test units property getter and setter."""
        # Test getter
        assert sample_dataset_2d.units == ["nm", "nm"]

        # Test setter
        sample_dataset_2d.units = ["Å", "Å"]
        assert sample_dataset_2d.units == ["Å", "Å"]

        # Test validation
        with pytest.raises(ValueError):
            sample_dataset_2d.units = ["Å", "Å", "Å"]  # Wrong number of dimensions

    def test_signal_units_property(self, sample_dataset_2d):
        """Test signal_units property getter and setter."""
        # Test getter
        assert sample_dataset_2d.signal_units == "counts"

        # Test setter
        sample_dataset_2d.signal_units = "electrons"
        assert sample_dataset_2d.signal_units == "electrons"


class TestDatasetMethods:
    """Test Dataset methods."""

    def test_copy(self, sample_dataset_2d):
        """Test copying a Dataset."""
        copied_dataset = sample_dataset_2d.copy()

        # Check that it's a different object
        assert copied_dataset is not sample_dataset_2d

        # Check that the data is the same
        assert np.array_equal(copied_dataset.array, sample_dataset_2d.array)
        assert copied_dataset.name == sample_dataset_2d.name
        assert np.array_equal(copied_dataset.origin, sample_dataset_2d.origin)
        assert np.array_equal(copied_dataset.sampling, sample_dataset_2d.sampling)
        assert copied_dataset.units == sample_dataset_2d.units
        assert copied_dataset.signal_units == sample_dataset_2d.signal_units

        # Check that modifying the copy doesn't affect the original
        copied_dataset.name = "modified_copy"
        assert sample_dataset_2d.name == "test_2d_dataset"

    def test_mean(self, sample_dataset_2d):
        """Test mean method."""
        # Mean of all elements
        mean_value = sample_dataset_2d.mean()
        assert isinstance(mean_value, (float, np.float64))

        # Mean along axis 0
        mean_along_axis = sample_dataset_2d.mean(axes=(0,))
        assert mean_along_axis.shape == (10,)

        # Mean along both axes (should be the same as mean of all elements)
        mean_along_both = sample_dataset_2d.mean(axes=(0, 1))
        assert isinstance(mean_along_both, (float, np.float64))
        assert np.isclose(mean_value, mean_along_both)

    def test_max(self, sample_dataset_2d):
        """Test max method."""
        # Max of all elements
        max_value = sample_dataset_2d.max()
        assert isinstance(max_value, (float, np.float64))

        # Max along axis 0
        max_along_axis = sample_dataset_2d.max(axes=(0,))
        assert max_along_axis.shape == (10,)

    def test_min(self, sample_dataset_2d):
        """Test min method."""
        # Min of all elements
        min_value = sample_dataset_2d.min()
        assert isinstance(min_value, (float, np.float64))

        # Min along axis 0
        min_along_axis = sample_dataset_2d.min(axes=(0,))
        assert min_along_axis.shape == (10,)

    def test_pad(self, sample_dataset_2d):
        """Test pad method."""
        # Pad with zeros
        padded_dataset = sample_dataset_2d.pad(pad_width=1)

        # Check shape
        assert padded_dataset.shape == (12, 12)  # Original (10, 10) + 1 on each side

        # Check that the original dataset is unchanged
        assert sample_dataset_2d.shape == (10, 10)

        # Test modify_in_place
        original_name = sample_dataset_2d.name
        sample_dataset_2d.pad(pad_width=1, modify_in_place=True)
        assert sample_dataset_2d.shape == (12, 12)
        assert sample_dataset_2d.name == original_name

    def test_crop(self, sample_dataset_2d):
        """Test crop method."""
        # Crop 1 pixel from each side
        cropped_dataset = sample_dataset_2d.crop(crop_widths=((1, 1), (1, 1)))

        # Check shape
        assert cropped_dataset.shape == (8, 8)  # Original (10, 10) - 1 from each side

        # Check that the original dataset is unchanged
        assert sample_dataset_2d.shape == (10, 10)

        # Test modify_in_place
        sample_dataset_2d.crop(crop_widths=((1, 1), (1, 1)), modify_in_place=True)
        assert sample_dataset_2d.shape == (8, 8)

    def test_bin(self, sample_dataset_2d):
        """Test bin method."""
        # Bin by factor of 2
        binned_dataset = sample_dataset_2d.bin(bin_factors=2)

        # Check shape
        assert binned_dataset.shape == (5, 5)  # Original (10, 10) / 2

        # Check that the original dataset is unchanged
        assert sample_dataset_2d.shape == (10, 10)

        # Test modify_in_place
        sample_dataset_2d.bin(bin_factors=2, modify_in_place=True)
        assert sample_dataset_2d.shape == (5, 5)


class TestDatasetRepresentation:
    """Test Dataset string representation."""

    def test_repr(self, sample_dataset_2d):
        """Test __repr__ method."""
        repr_str = repr(sample_dataset_2d)
        assert "Dataset" in repr_str
        assert "shape=(10, 10)" in repr_str
        assert "name='test_2d_dataset'" in repr_str

    def test_str(self, sample_dataset_2d):
        """Test __str__ method."""
        str_str = str(sample_dataset_2d)
        assert "quantem Dataset" in str_str
        assert "shape: (10, 10)" in str_str
        assert "name: 'test_2d_dataset'" in str_str or "named 'test_2d_dataset'" in str_str
