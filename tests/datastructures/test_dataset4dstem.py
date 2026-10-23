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

    def test_getitem(self, sample_dataset4dstem):
        """A single diffraction pattern and a line of diffraction patterns."""
        dp = sample_dataset4dstem[0, 0]
        assert dp.ndim == 2
        assert dp.shape == (10, 10)
        assert np.array_equal(dp.array, sample_dataset4dstem.array[0, 0])

        line = sample_dataset4dstem[0, :]
        assert line.ndim == 3
        assert line.shape == (5, 10, 10)
        assert np.array_equal(line.array, sample_dataset4dstem.array[0, :])


class TestDataset4dstemDpMethods:
    """Test Dataset4dstem diffraction pattern methods."""

    def test_dp_reductions_and_attach(self, sample_dataset4dstem):
        """dp_* computes without attaching; get_dp_*(attach=True) caches what dp_* then returns."""
        array = sample_dataset4dstem.array
        for name, reference in [("mean", np.mean), ("max", np.max), ("median", np.median)]:
            attr = f"_dp_{name}"
            expected = reference(array, axis=(0, 1))

            dp = getattr(sample_dataset4dstem, f"dp_{name}")
            assert not hasattr(sample_dataset4dstem, attr), name
            assert np.allclose(dp.array, expected), name
            assert np.allclose(dp.sampling, [0.1, 0.1]), name
            assert getattr(sample_dataset4dstem, f"dp_{name}") is not dp, name

            get = getattr(sample_dataset4dstem, f"get_dp_{name}")
            assert np.allclose(get(attach=False).array, expected), name
            assert not hasattr(sample_dataset4dstem, attr), name

            attached = get(attach=True)
            assert getattr(sample_dataset4dstem, attr) is attached, name
            assert getattr(sample_dataset4dstem, f"dp_{name}") is attached, name

    def test_get_virtual_image(self, sample_dataset4dstem):
        """Test get_virtual_image method."""
        # Create a simple mask
        mask = np.zeros((10, 10))
        mask[4:6, 4:6] = 1  # A 2x2 square in the center

        # Get virtual image without attaching
        virtual_image = sample_dataset4dstem.get_virtual_image(
            mask=mask, name="test_virtual_image", attach=False
        )
        assert virtual_image.shape == (5, 5)
        assert np.allclose(
            virtual_image.array, sample_dataset4dstem.array[:, :, 4:6, 4:6].sum(axis=(-2, -1))
        )
        assert "test_virtual_image" not in sample_dataset4dstem.virtual_images

        # Get virtual image with attaching
        virtual_image2 = sample_dataset4dstem.get_virtual_image(
            mask=mask, name="test_virtual_image", attach=True
        )
        assert "test_virtual_image" in sample_dataset4dstem.virtual_images
        assert virtual_image2 is sample_dataset4dstem.virtual_images["test_virtual_image"]


class TestProbePositions:
    """`probe_positions` hands the implicit raster grid to consumers that need it explicitly."""

    @staticmethod
    def _dataset(**kwargs):
        return Dataset4dstem.from_array(
            np.zeros((3, 4, 5, 5)),
            name="scan",
            **kwargs,
        )

    def test_origin_and_sampling_set_the_grid(self):
        dataset = self._dataset(
            sampling=(2.0, 3.0, 0.1, 0.1),
            origin=(1.0, -4.0, 0.0, 0.0),
            units=["A", "A", "A^-1", "A^-1"],
        )
        positions = dataset.probe_positions()

        assert positions.shape == (12, 2)
        assert list(positions.units) == ["A", "A"]
        assert np.allclose(positions.array[0], [1.0, -4.0])
        # row-major, so reshaping recovers the scan grid
        assert np.allclose(positions.array.reshape(3, 4, 2)[2, 3], [1.0 + 2 * 2.0, -4.0 + 3 * 3.0])

    def test_ordering_matches_a_reshape_of_the_patterns(self):
        dataset = self._dataset(sampling=(2.0, 3.0, 0.1, 0.1), units=["A", "A", "A^-1", "A^-1"])
        rows, cols = dataset.shape[:2]
        expected = np.stack(
            np.meshgrid(np.arange(rows) * 2.0, np.arange(cols) * 3.0, indexing="ij"), axis=-1
        ).reshape(-1, 2)
        assert np.allclose(dataset.probe_positions().array, expected)

    def test_rejects_mismatched_scan_units(self):
        dataset = self._dataset(sampling=(2.0, 3.0, 0.1, 0.1), units=["A", "nm", "A^-1", "A^-1"])
        with pytest.raises(ValueError, match="must share a unit"):
            dataset.probe_positions()
