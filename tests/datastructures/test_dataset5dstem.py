"""Tests for Dataset5dstem class."""

import numpy as np
import pytest

from quantem.core.datastructures.dataset5dstem import Dataset5dstem
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.dataset3d import Dataset3d


@pytest.fixture
def sample_dataset():
    """Create a sample 5D-STEM dataset."""
    array = np.random.rand(3, 5, 5, 10, 10)
    return Dataset5dstem.from_array(array=array, stack_type="time")


class TestDataset5dstem:
    """Core Dataset5dstem tests."""

    def test_from_array(self):
        """Test creating Dataset5dstem from array."""
        array = np.random.rand(3, 5, 5, 10, 10)
        data = Dataset5dstem.from_array(array=array)

        assert data.shape == (3, 5, 5, 10, 10)
        assert data.stack_type == "generic"
        assert len(data) == 3

    def test_from_4dstem(self):
        """Test creating Dataset5dstem from list of Dataset4dstem."""
        datasets = [
            Dataset4dstem.from_array(array=np.random.rand(5, 5, 10, 10))
            for _ in range(3)
        ]
        data = Dataset5dstem.from_4dstem(datasets, stack_type="tilt")

        assert data.shape == (3, 5, 5, 10, 10)
        assert data.stack_type == "tilt"

    def test_indexing(self, sample_dataset):
        """Test data[i] returns Dataset4dstem."""
        frame = sample_dataset[0]  # -> Dataset4dstem

        assert isinstance(frame, Dataset4dstem)
        assert frame.shape == (5, 5, 10, 10)
        assert np.array_equal(frame.array, sample_dataset.array[0])

    def test_iteration(self, sample_dataset):
        """Test iteration over frames."""
        frames = list(sample_dataset)

        assert len(frames) == 3
        assert all(isinstance(f, Dataset4dstem) for f in frames)

    def test_stack_mean(self, sample_dataset):
        """Test stack_mean returns Dataset4dstem."""
        mean = sample_dataset.stack_mean()

        assert isinstance(mean, Dataset4dstem)
        assert mean.shape == (5, 5, 10, 10)
        assert np.allclose(mean.array, np.mean(sample_dataset.array, axis=0))

    def test_stack_min(self, sample_dataset):
        """Test stack_min returns Dataset4dstem."""
        minimum = sample_dataset.stack_min()

        assert isinstance(minimum, Dataset4dstem)
        assert minimum.shape == (5, 5, 10, 10)
        assert np.allclose(minimum.array, np.min(sample_dataset.array, axis=0))

    def test_stack_sum(self, sample_dataset):
        """Test stack_sum returns Dataset4dstem."""
        total = sample_dataset.stack_sum()

        assert isinstance(total, Dataset4dstem)
        assert total.shape == (5, 5, 10, 10)

    def test_stack_max(self, sample_dataset):
        """Test stack_max returns Dataset4dstem."""
        maximum = sample_dataset.stack_max()

        assert isinstance(maximum, Dataset4dstem)
        assert maximum.shape == (5, 5, 10, 10)

    def test_stack_std(self, sample_dataset):
        """Test stack_std returns Dataset4dstem."""
        std = sample_dataset.stack_std()

        assert isinstance(std, Dataset4dstem)
        assert std.shape == (5, 5, 10, 10)
        assert np.allclose(std.array, np.std(sample_dataset.array, axis=0))

    def test_slicing(self, sample_dataset):
        """Test data[1:3] returns Dataset5dstem with correct data."""
        sliced = sample_dataset[1:3]  # -> Dataset5dstem
        assert isinstance(sliced, Dataset5dstem)
        assert sliced.stack_type == "time"
        assert np.array_equal(sliced.array, sample_dataset.array[1:3])

    def test_slicing_ellipsis(self, sample_dataset):
        """Test data[1:3, ...] returns Dataset5dstem with correct data."""
        sliced = sample_dataset[1:3, ...]  # -> Dataset5dstem
        assert isinstance(sliced, Dataset5dstem)
        assert np.array_equal(sliced.array, sample_dataset.array[1:3, ...])

    def test_slicing_scan_position(self, sample_dataset):
        """Test data[:, 2, 2] returns Dataset3d with correct data."""
        sliced = sample_dataset[:, 2, 2]  # -> Dataset3d
        assert isinstance(sliced, Dataset3d)
        assert np.array_equal(sliced.array, sample_dataset.array[:, 2, 2])

    def test_slicing_k_roi(self, sample_dataset):
        """Test data[:, :, :, 2:8, 2:8] returns Dataset5dstem with correct data."""
        sliced = sample_dataset[:, :, :, 2:8, 2:8]  # -> Dataset5dstem
        assert isinstance(sliced, Dataset5dstem)
        assert np.array_equal(sliced.array, sample_dataset.array[:, :, :, 2:8, 2:8])

    def test_slicing_frame_ellipsis(self, sample_dataset):
        """Test data[0, ...] same as data[0] with correct data."""
        sliced = sample_dataset[0, ...]  # -> Dataset4dstem
        assert isinstance(sliced, Dataset4dstem)
        assert np.array_equal(sliced.array, sample_dataset.array[0, ...])

    def test_slicing_last_axis(self, sample_dataset):
        """Test data[..., 0] slices last axis with correct data."""
        sliced = sample_dataset[..., 0]  # -> Dataset4d
        assert np.array_equal(sliced.array, sample_dataset.array[..., 0])

    def test_slicing_scan_k_roi(self, sample_dataset):
        """Test data[:, 2, 2, 2:8, 2:8] with correct data."""
        sliced = sample_dataset[:, 2, 2, 2:8, 2:8]  # -> Dataset3d
        assert np.array_equal(sliced.array, sample_dataset.array[:, 2, 2, 2:8, 2:8])

    def test_slicing_substack_k_roi(self, sample_dataset):
        """Test data[1:3, :, :, 2:8, 2:8] with correct data."""
        sliced = sample_dataset[1:3, :, :, 2:8, 2:8]  # -> Dataset5dstem
        assert isinstance(sliced, Dataset5dstem)
        assert np.array_equal(sliced.array, sample_dataset.array[1:3, :, :, 2:8, 2:8])

    def test_bin_all(self):
        """Test bin(2) bins all dimensions including stack."""
        array = np.random.rand(4, 6, 6, 10, 10)
        data = Dataset5dstem.from_array(array=array)

        binned = data.bin(2)

        assert binned.shape == (2, 3, 3, 5, 5)  # all dims halved

    def test_bin_preserve_stack(self):
        """Test bin with axes preserves stack."""
        array = np.random.rand(4, 6, 6, 10, 10)
        data = Dataset5dstem.from_array(array=array)

        binned = data.bin(2, axes=(1, 2, 3, 4))

        assert binned.shape == (4, 3, 3, 5, 5)  # stack preserved

    def test_get_virtual_image(self, sample_dataset):
        """Test virtual image creation."""
        vi = sample_dataset.get_virtual_image(  # -> Dataset3d
            mode="circle",
            geometry=((5, 5), 3),
            name="bf",
        )

        assert isinstance(vi, Dataset3d)
        assert vi.shape == (3, 5, 5)
        assert "bf" in sample_dataset.virtual_images

    def test_repr(self, sample_dataset):
        """Test __repr__ shows Dataset5dstem."""
        r = repr(sample_dataset)

        assert "Dataset5dstem" in r
        assert "stack_type='time'" in r

    def test_str(self, sample_dataset):
        """Test __str__ shows formatted output."""
        s = str(sample_dataset)

        assert "Dataset5dstem" in s
        assert "3 frames" in s
        assert "stack_type: 'time'" in s

    def test_stack_values_validation(self):
        """Test stack_values length must match number of frames."""
        array = np.random.rand(3, 5, 5, 10, 10)

        with pytest.raises(ValueError, match="stack_values length"):
            Dataset5dstem.from_array(array=array, stack_values=np.array([1, 2]))  # wrong length

    def test_from_4dstem_validation(self):
        """Test from_4dstem validates consistent shapes."""
        ds1 = Dataset4dstem.from_array(array=np.random.rand(5, 5, 10, 10))
        ds2 = Dataset4dstem.from_array(array=np.random.rand(6, 6, 10, 10))  # different shape

        with pytest.raises(ValueError, match="shape"):
            Dataset5dstem.from_4dstem([ds1, ds2])
