"""Tests for Dataset5dstem class."""
import numpy as np
import pytest
from quantem.core.datastructures.dataset5dstem import Dataset5dstem
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.dataset3d import Dataset3d


@pytest.fixture
def sample_dataset():
    """Create a sample 5D-STEM dataset with distinct sizes for clarity."""
    array = np.random.rand(4, 6, 7, 3, 5)  # (frames, scan_row, scan_col, k_row, k_col)
    return Dataset5dstem.from_array(array=array, stack_type="time")


class TestDataset5dstem:
    """Core Dataset5dstem tests."""

    def test_from_array(self):
        """Test creating Dataset5dstem from array."""
        array = np.random.rand(4, 6, 7, 3, 5)
        data = Dataset5dstem.from_array(array=array, stack_type="tilt")
        assert data.shape == (4, 6, 7, 3, 5)
        assert data.stack_type == "tilt"
        assert len(data) == 4

    def test_from_4dstem(self):
        """Test creating Dataset5dstem from list of Dataset4dstem."""
        datasets = [Dataset4dstem.from_array(np.random.rand(6, 7, 3, 5)) for _ in range(4)]
        data = Dataset5dstem.from_4dstem(datasets, stack_type="tilt")
        assert data.shape == (4, 6, 7, 3, 5)
        assert data.stack_type == "tilt"

    def test_indexing_and_iteration(self, sample_dataset):
        """Test data[i] and iteration return Dataset4dstem frames."""
        frame = sample_dataset[0]  # -> Dataset4dstem
        assert isinstance(frame, Dataset4dstem)
        assert frame.shape == (6, 7, 3, 5)
        frames = list(sample_dataset)
        assert len(frames) == 4
        assert all(isinstance(f, Dataset4dstem) for f in frames)

    def test_stack_reductions(self, sample_dataset):
        """Test stack reduction methods return Dataset4dstem with correct values."""
        mean = sample_dataset.stack_mean()  # -> Dataset4dstem
        assert isinstance(mean, Dataset4dstem)
        assert mean.shape == (6, 7, 3, 5)
        arr = sample_dataset.array
        assert np.allclose(sample_dataset.stack_mean().array, np.mean(arr, axis=0))
        assert np.allclose(sample_dataset.stack_sum().array, np.sum(arr, axis=0))
        assert np.allclose(sample_dataset.stack_max().array, np.max(arr, axis=0))
        assert np.allclose(sample_dataset.stack_min().array, np.min(arr, axis=0))
        assert np.allclose(sample_dataset.stack_std().array, np.std(arr, axis=0))

    def test_slicing(self, sample_dataset):
        """Test common slicing patterns."""
        substack = sample_dataset[1:3]  # -> Dataset5dstem (2, 6, 7, 3, 5)
        assert isinstance(substack, Dataset5dstem)
        assert substack.shape == (2, 6, 7, 3, 5)
        assert substack.stack_type == "time"  # metadata preserved
        position = sample_dataset[:, 2, 3]  # -> Dataset3d (4, 3, 5)
        assert isinstance(position, Dataset3d)
        assert position.shape == (4, 3, 5)
        cropped = sample_dataset[:, :, :, 1:3, 1:4]  # -> Dataset5dstem (4, 6, 7, 2, 3)
        assert isinstance(cropped, Dataset5dstem)
        assert cropped.shape == (4, 6, 7, 2, 3)

    def test_get_virtual_image(self, sample_dataset):
        """Test virtual image creation returns Dataset3d stack."""
        vi = sample_dataset.get_virtual_image(mode="circle", geometry=((1, 2), 1), name="bf")
        assert isinstance(vi, Dataset3d)
        assert vi.shape == (4, 6, 7)
        assert "bf" in sample_dataset.virtual_images

    def test_validation(self):
        """Test validation errors."""
        array = np.random.rand(4, 6, 7, 3, 5)
        with pytest.raises(ValueError, match="stack_values length"):
            Dataset5dstem.from_array(array=array, stack_values=np.array([1, 2]))
        ds1 = Dataset4dstem.from_array(np.random.rand(6, 7, 3, 5))
        ds2 = Dataset4dstem.from_array(np.random.rand(8, 9, 3, 5))
        with pytest.raises(ValueError, match="shape"):
            Dataset5dstem.from_4dstem([ds1, ds2])
