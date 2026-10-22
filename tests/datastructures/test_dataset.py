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


def center_world(d):
    return d.origin + (np.array(d.shape) - 1) / 2.0 * d.sampling


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
        assert hasattr(dataset, "metadata")

    def test_from_array_defaults(self, sample_2d_array):
        """Test creating a Dataset with default parameters."""
        dataset = Dataset.from_array(array=sample_2d_array)

        assert dataset.name == "2d dataset"  # Default name based on dimensions
        assert dataset.ndim == 2
        assert np.array_equal(dataset.origin, np.array([0, 0]))  # Default origin
        assert np.array_equal(dataset.sampling, np.array([1, 1]))  # Default sampling
        assert dataset.units == ["pixels", "pixels"]  # Default units
        assert dataset.signal_units == "arb. units"  # Default signal units
        assert hasattr(dataset, "metadata")

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
    def test_axis_metadata_setters_validate_length(self, sample_dataset_2d):
        """origin, sampling and units accept one value per axis and reject any other length."""
        for name, valid, expected, wrong in [
            ("origin", (1, 1), np.array([1, 1]), (1, 1, 1)),
            ("sampling", (2, 2), np.array([2, 2]), (2, 2, 2)),
            ("units", ["Å", "Å"], ["Å", "Å"], ["Å", "Å", "Å"]),
        ]:
            setattr(sample_dataset_2d, name, valid)
            assert np.array_equal(getattr(sample_dataset_2d, name), expected), name
            with pytest.raises(ValueError):
                setattr(sample_dataset_2d, name, wrong)


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

    def test_reductions_match_numpy(self, sample_dataset_2d, sample_2d_array):
        for name, reference in [("mean", np.mean), ("max", np.max), ("min", np.min)]:
            reduce = getattr(sample_dataset_2d, name)
            assert np.isclose(reduce(), reference(sample_2d_array)), name
            assert np.isclose(reduce(axes=(0, 1)), reference(sample_2d_array)), name
            assert np.allclose(reduce(axes=(0,)), reference(sample_2d_array, axis=0)), name

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
        cropped_dataset = sample_dataset_2d.crop(crop_widths=((1, 9), (1, 9)))
        # Check shape
        assert cropped_dataset.shape == (8, 8)  # Original (10, 10) - 1 from each side
        assert np.array_equal(cropped_dataset.origin, np.array([1, 1]))
        # Check that the original dataset is unchanged
        assert sample_dataset_2d.shape == (10, 10)
        assert np.array_equal(sample_dataset_2d.origin, np.array([0, 0]))
        # Test modify_in_place
        sample_dataset_2d.crop(crop_widths=((1, 9), (1, 9)), modify_in_place=True)
        assert sample_dataset_2d.shape == (8, 8)
        assert np.array_equal(sample_dataset_2d.origin, np.array([1, 1]))

    def test_crop_selected_axes(self):
        """Cropping a subset of axes, including stop=0 (keep the rest) and a single int axis."""
        cases = [
            # shape, origin, sampling, crop_widths, axes, expected shape, expected origin
            ((8, 8, 96, 96), (0, 0, 0, 0), (1, 1, 1, 1), ((4, 92), (4, 92)), (2, 3),
             (8, 8, 88, 88), (0, 0, 4, 4)),
            ((8, 8, 96, 96), (0, 0, 0, 0), (1, 1, 1, 1), ((10, 0), (10, 0)), (2, 3),
             (8, 8, 86, 86), (0, 0, 10, 10)),
            ((16, 16, 32, 32), (10, 20, 30, 40), (0.5, 0.25, 2, 3), ((4, 12), (4, 12)), (0, 1),
             (8, 8, 32, 32), (12, 21, 30, 40)),
            ((2048, 64), (5, 7), (0.5, 2), ((80, 2000),), 0,
             (1920, 64), (45, 7)),
        ]  # fmt: skip
        for shape, origin, sampling, crop_widths, axes, out_shape, out_origin in cases:
            dset = Dataset.from_array(np.random.rand(*shape), origin=origin, sampling=sampling)
            cropped = dset.crop(crop_widths=crop_widths, axes=axes)
            assert cropped.shape == out_shape, (shape, crop_widths, axes)
            assert np.allclose(cropped.origin, out_origin), (shape, crop_widths, axes)
            assert dset.shape == shape, (shape, crop_widths, axes)

            dset.crop(crop_widths=crop_widths, axes=axes, modify_in_place=True)
            assert dset.shape == out_shape, (shape, crop_widths, axes)
            assert np.allclose(dset.origin, out_origin), (shape, crop_widths, axes)

    def test_getitem_slice_updates_origin_like_crop(self):
        """Test slicing keeps origin aligned with crop semantics."""
        dset = Dataset.from_array(
            np.random.rand(16, 8),
            origin=(10, 20),
            sampling=(0.5, 2.0),
            units=["nm", "nm"],
        )

        sliced = dset[4:12]
        cropped = dset.crop(crop_widths=((4, 12),), axes=0)

        assert sliced.shape == (8, 8)
        assert np.array_equal(sliced.origin, np.array([12, 20]))
        assert np.array_equal(sliced.origin, cropped.origin)
        assert np.array_equal(sliced.sampling, cropped.sampling)

    def test_getitem_slice_reduced_rank_updates_origin_and_sampling(self):
        """Test slicing before integer indexing updates remaining metadata."""
        dset = Dataset.from_array(
            np.random.rand(10, 6, 4),
            origin=(1.5, 10, -2),
            sampling=(0.25, 2.0, 5.0),
            units=["nm", "nm", "1/nm"],
        )

        sliced = dset[2:8:2, 3]

        assert sliced.shape == (3, 4)
        assert np.allclose(sliced.origin, np.array([2.0, -2.0]))
        assert np.allclose(sliced.sampling, np.array([0.5, 5.0]))
        assert sliced.units == ["nm", "1/nm"]

    def test_bin(self, sample_dataset_2d):
        """Test bin method."""
        # Original metadata
        assert np.array_equal(sample_dataset_2d.origin, np.array([0.0, 0.0]))
        assert np.array_equal(sample_dataset_2d.sampling, np.array([1.0, 1.0]))

        # Bin by factor of 2
        binned = sample_dataset_2d.bin(bin_factors=2)
        assert binned.shape == (5, 5)  # Original (10, 10) / 2
        # Sampling doubled
        assert np.allclose(binned.sampling, np.array([2.0, 2.0]))
        # Origin shifted to center of first 2x2 block: +0.5 * (2-1) * old_sampling
        assert np.allclose(binned.origin, np.array([0.5, 0.5]))
        # Check that the original dataset is unchanged
        assert sample_dataset_2d.shape == (10, 10)

        # Test modify_in_place
        sample_dataset_2d.bin(bin_factors=2, modify_in_place=True)
        assert sample_dataset_2d.shape == (5, 5)


class TestFourierResample:
    def test_resample_preserves_mean_and_center(self):
        cases = [
            # shape, origin, sampling, kwargs, expected shape, expected sampling, center tol
            ((10, 10), (0, 0), (1.0, 1.0), {"factors": 0.5}, (5, 5), (2.0, 2.0), 1e-12),
            ((5, 6), (1.5, -2.0), (0.2, 0.5), {"factors": 2.0}, (10, 12), (0.1, 0.25), 1e-10),
        ]
        for shape, origin, sampling, kwargs, out_shape, out_sampling, tol in cases:
            ds = Dataset.from_array(np.ones(shape), origin=origin, sampling=sampling)
            ds2 = ds.fourier_resample(**kwargs)
            assert ds2.shape == out_shape, kwargs
            assert np.isclose(ds2.array.mean(), 1.0, rtol=1e-6, atol=1e-6), kwargs
            assert np.allclose(ds2.sampling, np.array(out_sampling)), kwargs
            assert np.allclose(center_world(ds), center_world(ds2), rtol=tol, atol=tol), kwargs

    def test_odd_even_center_preservation(self):
        # Odd→even sizes; check center preservation rather than sampling integers
        arr = np.random.rand(9, 7)
        ds = Dataset.from_array(arr, origin=(0.0, 0.0), sampling=(1.0, 1.0), units=["px", "px"])

        # Target out_shape (5, 4) explicitly
        ds2 = ds.fourier_resample(out_shape=(5, 4))
        assert ds2.shape == (5, 4)
        assert np.allclose(center_world(ds), center_world(ds2), rtol=1e-12, atol=1e-12)

    def test_api_errors(self, sample_dataset_2d):
        # Both specified
        with pytest.raises(ValueError):
            sample_dataset_2d.fourier_resample(out_shape=(5, 5), factors=0.5)

        # Neither specified
        with pytest.raises(ValueError):
            sample_dataset_2d.fourier_resample()


class TestDatasetTorch:
    """Tests for torch-backed Dataset (from_tensor path)."""

    def test_numpy_copy_is_readonly(self):
        """``.numpy()`` on a tensor-backed dataset returns a read-only CPU copy
        so writes raise instead of silently updating only the detached copy.
        """
        import torch

        from quantem.core.datastructures.dataset4dstem import Dataset4dstem

        ds = Dataset4dstem.from_tensor(torch.zeros(2, 2, 2, 2))
        arr = ds.numpy()
        assert arr.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            arr[0, 0, 0, 0] = 99.0
