import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.io.serialize import load
from quantem.imaging.lattice import Lattice


class TestLatticeInit:
    """Test Lattice initialization and from_data."""

    def test_init_and_constructor(self):
        """Test that direct init is blocked and from_data works."""
        image = np.random.randn(100, 100)
        ds2d = Dataset2d.from_array(image)

        with pytest.raises(RuntimeError, match="Use Lattice.from_data"):
            Lattice(ds2d)

        lattice_img = Lattice.from_data(image)
        lattice_dset = Lattice.from_data(ds2d)
        assert isinstance(lattice_img, Lattice)
        assert lattice_img.image is not None
        assert isinstance(lattice_dset, Lattice)
        assert lattice_dset.image is not None

    def test_normalization(self):
        """Test min/max normalization."""
        image = np.random.randn(100, 100) * 1000.0
        image[0, 0] = -10.0
        image[99, 99] = 1000.0

        # Both normalizations
        lattice = Lattice.from_data(image)
        assert lattice.image.array.min() == 0
        assert lattice.image.array.max() == 1

        # No normalization
        lattice = Lattice.from_data(image, normalize_min=False, normalize_max=False)
        assert_array_almost_equal(lattice.image.array, image)

        # Min normalization
        lattice = Lattice.from_data(image, normalize_min=True, normalize_max=False)
        assert lattice.image.array.min() == 0

        # Max normalization
        lattice = Lattice.from_data(image, normalize_min=False, normalize_max=True)
        assert lattice.image.array.max() == 1

    def test_edge_cases(self):
        """Test NaN handling."""
        nan_arr = np.array([[1, np.nan], [3, 4]], dtype=float)
        lattice = Lattice.from_data(nan_arr)
        assert isinstance(lattice, Lattice)

    def test_invalid_inputs(self):
        """Test that invalid inputs raise errors."""
        with pytest.raises(ValueError, match="must be a 2D array"):
            Lattice.from_data(np.array([1, 2, 3]))

        with pytest.raises(ValueError, match="must be a 2D array"):
            Lattice.from_data(np.ones((2, 2, 2)))

        with pytest.raises(ValueError, match="must not be empty"):
            Lattice.from_data(np.array([[]]))


class TestLatticeImage:
    """Test image property getter and setter."""

    def test_image_property(self):
        """Test getting and setting image."""
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)

        # Get
        assert isinstance(lattice.image, Dataset2d)

        # Set with new array
        new_image = np.random.randn(50, 50)
        lattice.image = new_image
        assert lattice.image.array.shape == (50, 50)

        # Invalid set
        with pytest.raises(ValueError, match="must be a 2D array"):
            lattice.image = np.array([1, 2, 3])


class TestDefineLatticeVectors:
    """Test define_lattice_vectors method."""

    def test_basic_define(self):
        """Test basic lattice definition."""
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)

        result = lattice.define_lattice_vectors(
            origin=[50, 50], u=[5, 0], v=[0, 5], refine_lattice=False
        )

        assert result is lattice
        assert hasattr(lattice, "_lat")
        assert lattice._lat.shape == (3, 2)

    def test_refinement_options(self):
        """Test lattice refinement and block_size."""
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)

        # With refinement
        lattice.define_lattice_vectors(
            origin=[50, 50],
            u=[5, 0],
            v=[0, 5],
            refine_lattice=True,
            refine_maxiter=5,
        )
        assert lattice._lat.shape == (3, 2)

        # With block_size
        lattice.define_lattice_vectors(
            origin=[50, 50],
            u=[5, 0],
            v=[0, 5],
            refine_lattice=True,
            refine_maxiter=5,
            block_size=5,
        )
        assert lattice._lat.shape == (3, 2)

    def test_invalid_lattice_params(self):
        """Test invalid lattice parameters."""
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)

        # Wrong shape
        with pytest.raises(ValueError):
            lattice.define_lattice_vectors(origin=[1, 2, 3], u=[5, 0], v=[0, 5])

        # Negative block_size
        with pytest.raises(ValueError):
            lattice.define_lattice_vectors(origin=[50, 50], u=[5, 0], v=[0, 5], block_size=-1)

        # Origin out of bounds
        with pytest.raises(ValueError):
            lattice.define_lattice_vectors(origin=[10, 105], u=[5, 0], v=[0, 5])

        # Non-ivertible lattice vectors
        with pytest.raises(ValueError):
            lattice.define_lattice_vectors(origin=[50, 50], u=[5, 0], v=[10, 0])


class TestLatticeSerialize:
    """Test Lattice Autoserialize implementation."""

    @pytest.mark.parametrize("store", ["zip", "dir"])
    def test_lattice_save_load(self, tmp_path, store):
        """Test save/load of lattice."""
        # Create lattice with image and defined lattice
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)
        lattice.define_lattice_vectors(origin=[50, 50], u=[5, 0], v=[0, 5], refine_lattice=False)

        # Save
        filepath = tmp_path / ("lattice.zip" if store == "zip" else "lattice_dir")
        lattice.save(str(filepath), mode="w", store=store)

        # Load
        loaded = load(str(filepath))

        # Verify
        assert isinstance(loaded, Lattice)
        assert isinstance(loaded.image, Dataset2d)
        assert loaded.image.array.shape == lattice.image.array.shape
        assert np.allclose(loaded.image.array, lattice.image.array)
        assert hasattr(loaded, "_lat")
        assert loaded._lat.shape == (3, 2)
        assert np.allclose(loaded._lat, lattice._lat)
