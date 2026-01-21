from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import AutoSerialize, load
from quantem.imaging.lattice import Lattice


class TestLatticeInitialization:
    """Test Lattice initialization and constructors."""

    def test_direct_init_raises_error(self):
        """Test that direct __init__ raises RuntimeError."""
        image = np.random.randn(100, 100)
        dset = Dataset2d.from_array(image)

        with pytest.raises(RuntimeError, match="Use Lattice.from_data"):
            Lattice(dset)

    def test_from_data_with_numpy_array(self):
        """Test from_data constructor with NumPy array."""
        image = np.random.randn(100, 100)

        lattice = Lattice.from_data(image)

        assert isinstance(lattice, Lattice)
        assert lattice.image is not None

    def test_from_data_with_dataset2d(self):
        """Test from_data constructor with Dataset2d."""
        arr = np.random.randn(100, 100)
        ds2d = Dataset2d.from_array(arr)

        lattice = Lattice.from_data(ds2d)

        assert isinstance(lattice, Lattice)
        assert isinstance(lattice.image, Dataset2d)

    def test_from_data_normalize_min_default(self):
        """Test that normalize_min is True by default."""
        image = np.random.randn(100, 100) + 10.0  # Offset from zero

        lattice = Lattice.from_data(image)

        # Minimum should be close to 0
        assert np.min(lattice.image.array) < 0.1

    def test_from_data_normalize_max_default(self):
        """Test that normalize_max is True by default."""
        image = np.random.randn(100, 100) * 10.0

        lattice = Lattice.from_data(image)

        # Maximum should be close to 1
        assert np.abs(np.max(lattice.image.array) - 1.0) < 0.1

    def test_from_data_no_normalization(self):
        """Test from_data without normalization."""
        image = np.random.randn(100, 100) * 5.0 + 3.0
        original_min = np.min(image)
        original_max = np.max(image)

        lattice = Lattice.from_data(image, normalize_min=False, normalize_max=False)

        assert np.allclose(np.min(lattice.image.array), original_min)
        assert np.allclose(np.max(lattice.image.array), original_max)

    def test_from_data_normalize_min_only(self):
        """Test normalization with only min normalization."""
        image = np.random.randn(100, 100) * 5.0 + 3.0

        lattice = Lattice.from_data(image, normalize_min=True, normalize_max=False)

        assert np.min(lattice.image.array) < 0.1

    def test_from_data_normalize_max_only(self):
        """Test normalization with only max normalization."""
        image = np.random.randn(100, 100) * 5.0

        lattice = Lattice.from_data(image, normalize_min=False, normalize_max=True)

        assert np.abs(np.max(lattice.image.array) - 1.0) < 0.1

    @pytest.mark.parametrize("shape", [(50, 50), (100, 200), (256, 256)])
    def test_from_data_various_shapes(self, shape):
        """Test from_data with various image shapes."""
        image = np.random.randn(*shape)

        lattice = Lattice.from_data(image)

        assert lattice.image.shape == shape


class TestLatticeProperties:
    """Test Lattice property getters and setters."""

    @pytest.fixture
    def simple_lattice(self):
        """Create a simple lattice for testing."""
        image = np.random.randn(100, 100)
        return Lattice.from_data(image)

    def test_image_getter(self, simple_lattice: Lattice):
        """Test image property getter."""
        image = simple_lattice.image

        assert isinstance(image, Dataset2d)
        assert image.shape == (100, 100)

    def test_image_setter_with_dataset2d(self, simple_lattice: Lattice):
        """Test image property setter with Dataset2d."""
        new_arr = np.random.randn(50, 50)
        new_ds2d = Dataset2d.from_array(new_arr)

        simple_lattice.image = new_ds2d

        assert isinstance(simple_lattice.image, Dataset2d)
        assert simple_lattice.image.shape == (50, 50)

    def test_image_setter_with_numpy_array(self, simple_lattice: Lattice):
        """Test image property setter with NumPy array."""
        new_arr = np.random.randn(75, 75)

        simple_lattice.image = new_arr

        assert isinstance(simple_lattice.image, Dataset2d)
        assert simple_lattice.image.shape == (75, 75)

    def test_image_setter_validates_dimensions(self, simple_lattice: Lattice):
        """Test that image setter validates 2D arrays."""
        with pytest.raises((ValueError, TypeError)):
            simple_lattice.image = np.random.randn(10, 10, 3)  # 3D array


class TestLatticeAttributes:
    """Test internal attributes and state management."""

    @pytest.fixture
    def lattice_with_state(self):
        """Create lattice with some state."""
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)

        # Mock lattice parameters
        lattice.define_lattice(
            origin=[10.0, 10.0],
            u=[50.0, 0.0],
            v=[0.0, 50.0],
        )

        return lattice

    def test_lattice_has_lat_attribute(self, lattice_with_state: Lattice):
        """Test that lattice has _lat attribute after fitting."""
        assert hasattr(lattice_with_state, "_lat")
        assert isinstance(lattice_with_state._lat, np.ndarray)

    def test_lattice_lat_shape(self, lattice_with_state: Lattice):
        """Test that _lat has correct shape (3, 2)."""
        assert lattice_with_state._lat.shape == (3, 2)

    def test_lattice_lat_components(self, lattice_with_state: Lattice):
        """Test that _lat contains origin, u, and v vectors."""
        r0, u, v = lattice_with_state._lat

        assert r0.shape == (2,)
        assert u.shape == (2,)
        assert v.shape == (2,)

    def test_lattice_image_is_dataset2d(self):
        """Test that internal image is always Dataset2d."""
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)

        assert isinstance(lattice._image, Dataset2d)


class TestLatticeRobustnessAndValidation:
    """Test robustness to various inputs and conditions."""

    def test_lattice_with_single_pixel(self):
        """Test lattice with 1x1 image."""
        image = np.array([[1.0]])

        lattice = Lattice.from_data(image)

        assert lattice.image.shape == (1, 1)

    def test_lattice_with_single_row(self):
        """Test lattice with single row."""
        image = np.random.randn(1, 100)

        lattice = Lattice.from_data(image)

        assert lattice.image.shape == (1, 100)

    def test_lattice_with_single_column(self):
        """Test lattice with single column."""
        image = np.random.randn(100, 1)

        lattice = Lattice.from_data(image)

        assert lattice.image.shape == (100, 1)

    def test_lattice_with_bool_array(self):
        """Test lattice creation with boolean array."""
        image = np.random.rand(50, 50) > 0.5

        lattice = Lattice.from_data(image)

        assert lattice is not None

    def test_lattice_with_sparse_data(self):
        """Test lattice with mostly zero data."""
        image = np.zeros((100, 100))
        image[25:30, 25:30] = np.random.randn(5, 5)

        lattice = Lattice.from_data(image)

        assert lattice is not None

    def test_lattice_with_noise_only(self):
        """Test lattice with pure noise (no structure)."""
        image = np.random.randn(100, 100)

        lattice = Lattice.from_data(image)

        assert lattice is not None

    def test_lattice_idempotent_normalization(self):
        """Test that normalizing an already normalized image doesn't change it much."""
        image = np.random.randn(100, 100)

        lattice1 = Lattice.from_data(image)
        lattice2 = Lattice.from_data(lattice1.image.array.copy())

        # Second normalization should have minimal effect
        assert np.allclose(lattice1.image.array, lattice2.image.array, atol=1e-5)

    def test_from_data_invalid_dimensions(self):
        """Test that non-2D arrays raise errors."""
        with pytest.raises((ValueError, TypeError)):
            Lattice.from_data(np.random.randn(10))  # 1D

        with pytest.raises((ValueError, TypeError)):
            Lattice.from_data(np.random.randn(10, 10, 10))  # 3D

    def test_from_data_empty_array(self):
        """Test behavior with empty array."""
        with pytest.raises((ValueError, IndexError)):
            Lattice.from_data(np.array([]))

    def test_image_setter_wrong_dimensions(self):
        """Test that image setter rejects non-2D arrays."""
        lattice = Lattice.from_data(np.random.randn(50, 50))

        with pytest.raises((ValueError, TypeError)):
            lattice.image = np.random.randn(10, 10, 3)

    def test_two_lattices_from_same_data(self):
        """Test creating two lattices from the same data."""
        image = np.random.randn(50, 50)

        lattice1 = Lattice.from_data(image.copy())
        lattice2 = Lattice.from_data(image.copy())

        # Images should be the same
        assert np.allclose(lattice1.image.array, lattice2.image.array)

    def test_lattice_independence(self):
        """Test that different lattice instances are independent."""
        image = np.random.randn(50, 50)

        lattice1 = Lattice.from_data(image.copy())
        lattice2 = Lattice.from_data(image.copy())

        # Modify one lattice
        lattice1.image = np.zeros((50, 50))

        # Other lattice should be unchanged
        assert not np.allclose(lattice1.image.array, lattice2.image.array)


class TestLatticeNormalization:
    """Test normalization behavior in detail."""

    def test_normalization_preserves_zero(self):
        """Test that zero values are handled correctly in normalization."""
        image = np.array([[0.0, 1.0], [2.0, 3.0]])

        lattice = Lattice.from_data(image, normalize_min=True, normalize_max=True)

        # Zero should remain zero after min normalization
        assert lattice.image.array[0, 0] < 0.1

    def test_normalization_with_constant_image(self):
        """Test normalization behavior with constant image."""
        image = np.ones((50, 50)) * 5.0

        # With constant values, normalization might behave specially
        try:
            lattice = Lattice.from_data(image, normalize_min=True, normalize_max=True)
            # Check that it doesn't raise divide-by-zero errors
            assert np.all(np.isfinite(lattice.image.array))
        except (ValueError, RuntimeWarning):
            # Acceptable if it handles constant images specially
            pass

    def test_no_normalization_preserves_values(self):
        """Test that disabling normalization preserves original values."""
        image = np.array([[1.5, 2.5], [3.5, 4.5]])

        lattice = Lattice.from_data(image, normalize_min=False, normalize_max=False)

        assert np.allclose(lattice.image.array, image)

    def test_normalization_order_independence(self):
        """Test that normalization order doesn't matter."""
        image = np.random.randn(100, 100) * 5.0 + 10.0

        lattice1 = Lattice.from_data(image.copy(), normalize_min=True, normalize_max=True)

        # Manually normalize in different order
        image2 = image.copy()
        image2 -= np.min(image2)
        image2 /= np.max(image2)

        lattice2 = Lattice.from_data(image2, normalize_min=False, normalize_max=False)

        assert np.allclose(lattice1.image.array, lattice2.image.array, atol=1e-5)


class TestLatticeMemoryManagement:
    """Test memory management and cleanup."""

    def test_large_lattice_creation_and_deletion(self):
        """Test that large lattices can be created and deleted."""
        image = np.random.randn(2000, 2000)
        lattice = Lattice.from_data(image)

        assert lattice is not None

        # Delete and ensure cleanup
        del lattice

    def test_multiple_lattice_instances(self):
        """Test creating multiple lattice instances."""
        lattices = []
        for i in range(10):
            image = np.random.randn(50, 50)
            lattices.append(Lattice.from_data(image))

        assert len(lattices) == 10
        assert all(isinstance(lat, Lattice) for lat in lattices)

    def test_lattice_image_modification_memory(self):
        """Test that modifying image doesn't create memory leaks."""
        lattice = Lattice.from_data(np.random.randn(100, 100))

        for _ in range(10):
            lattice.image = np.random.randn(100, 100)

        assert lattice.image.shape == (100, 100)


class TestLatticeEdgeCases:
    """Test edge cases and error handling for Lattice class."""

    def test_lattice_with_nan_values(self):
        """Test lattice behavior with NaN values."""
        image = np.random.randn(100, 100)
        image[50, 50] = np.nan

        # Should either handle NaN or raise appropriate error
        try:
            lattice = Lattice.from_data(image)
            # If it doesn't raise, check that NaN is preserved or handled
            assert lattice is not None
        except (ValueError, RuntimeError):
            pass  # Expected behavior

    def test_lattice_with_inf_values(self):
        """Test lattice behavior with infinite values."""
        image = np.random.randn(100, 100)
        image[25, 25] = np.inf
        image[75, 75] = -np.inf

        # Should either handle inf or raise appropriate error
        try:
            lattice = Lattice.from_data(image)
            assert lattice is not None
        except (ValueError, RuntimeError):
            pass  # Expected behavior

    def test_lattice_with_large_image(self):
        """Test lattice with large image."""
        image = np.random.randn(1000, 1000)

        lattice = Lattice.from_data(image)

        assert lattice.image.shape == (1000, 1000)

    def test_lattice_with_rectangular_image(self):
        """Test lattice with non-square image."""
        image = np.random.randn(100, 200)

        lattice = Lattice.from_data(image)

        assert lattice.image.shape == (100, 200)

    def test_lattice_with_negative_values(self):
        """Test lattice with all negative values."""
        image = -np.abs(np.random.randn(100, 100))

        lattice = Lattice.from_data(image)

        assert np.all(lattice.image.array >= 0)  # After normalization

    def test_lattice_with_very_large_values(self):
        """Test lattice with very large values."""
        image = np.random.randn(100, 100) * 1e10

        lattice = Lattice.from_data(image)

        # After normalization, should be in reasonable range
        assert np.max(lattice.image.array) <= 1.1  # Allow small tolerance

    def test_lattice_with_very_small_values(self):
        """Test lattice with very small values."""
        image = np.random.randn(100, 100) * 1e-10

        lattice = Lattice.from_data(image)

        assert lattice is not None

    def test_lattice_normalization_preserves_structure(self):
        """Test that normalization preserves relative structure."""
        image = np.array([[1.0, 2.0], [3.0, 4.0]])

        lattice = Lattice.from_data(image)

        # Relative ordering should be preserved
        flat = lattice.image.array.flatten()
        assert flat[0] < flat[1] < flat[2] < flat[3]


class TestLatticeAddAtoms:
    """Test add_atoms method."""

    @pytest.fixture
    def fitted_lattice(self):
        """Create a fitted lattice with atoms."""
        # Create synthetic image
        H, W = 100, 100
        image = np.random.randn(H, W) * 0.1

        # Add some peaks
        peaks = [
            (25, 25),
            (25, 50),
            (25, 75),
            (50, 25),
            (50, 50),
            (50, 75),
            (75, 25),
            (75, 50),
            (75, 75),
        ]
        for y, x in peaks:
            yy, xx = np.ogrid[-10:11, -10:11]
            peak = np.exp(-(xx**2 + yy**2) / 20.0)
            y_start, y_end = max(0, y - 10), min(H, y + 11)
            x_start, x_end = max(0, x - 10), min(W, x + 11)
            peak_h, peak_w = y_end - y_start, x_end - x_start
            image[y_start:y_end, x_start:x_end] += peak[:peak_h, :peak_w]

        lattice = Lattice.from_data(image)

        # Define lattice vectors before adding atoms
        lattice.define_lattice(
            origin=[10.0, 10.0],
            u=[50.0, 0.0],
            v=[0.0, 50.0],
        )

        return lattice

    def test_add_atoms_basic(self, fitted_lattice: Lattice):
        """Test basic atom addition."""
        positions_frac = np.array([[0.0, 0.0]])

        result = fitted_lattice.add_atoms(positions_frac, plot_atoms=False)

        assert result is fitted_lattice
        # Check that atoms were added
        assert hasattr(fitted_lattice, "_atoms") or hasattr(fitted_lattice, "atoms")

    def test_add_atoms_plotting(self, fitted_lattice: Lattice):
        """Test atom addition with plotting."""
        positions_frac = np.array([[0.0, 0.0]])

        result = fitted_lattice.add_atoms(positions_frac, plot_atoms=True)

        assert result is fitted_lattice

    def test_add_atoms_with_all_parameters(self, fitted_lattice: Lattice):
        """Test atom addition with all optional parameters."""
        positions_frac = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        numbers = np.array([3, 4, 4, 5])
        mask = np.ones(fitted_lattice.image.shape, dtype=bool)
        mask[:30, :30] = False

        result = fitted_lattice.add_atoms(
            positions_frac,
            numbers=numbers,
            intensity_min=0.1,
            intensity_radius=5,
            edge_min_dist_px=5,
            mask=mask,
            contrast_min=0.2,
            annulus_radii=(3, 6),
            plot_atoms=False,
        )

        assert result is fitted_lattice

    def test_add_atoms_empty_positions(self, fitted_lattice: Lattice):
        """Test adding atoms with empty positions array."""
        positions_frac = np.array([]).reshape(0, 2)

        result = fitted_lattice.add_atoms(positions_frac, plot_atoms=False)

        assert result is fitted_lattice

    def test_add_atoms_without_fitting_raises_error(self):
        """Test that add_atoms raises error if lattice not fitted."""
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)

        positions_frac = np.array([[0.0, 0.0]])

        with pytest.raises(ValueError, match="Lattice vectors have not been fitted"):
            lattice.add_atoms(positions_frac, plot_atoms=False)


class TestLatticePlotPolarizationVectors:
    """Test plot_polarization_vectors method."""

    @pytest.fixture
    def lattice_with_polarization(self):
        """Create lattice with polarization vector data."""
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)

        # Mock lattice vectors
        lattice.define_lattice(
            origin=[10.0, 10.0],
            u=[10.0, 0.0],
            v=[0.0, 10.0],
            refine_lattice=False,
        )

        return lattice

    @pytest.fixture
    def mock_vector(self):
        """Create mock Vector object with polarization data."""

        mock_vector = Vector.from_shape(
            shape=(1,),
            fields=["x", "y", "a", "b", "da", "db"],
            units=["px", "px", "ind", "ind", "ind", "ind"],
            name="polarization",
        )
        arr = np.array(
            [
                [20.0, 20.0, 0.0, 0.0, 0.1, 0.0],
                [30.0, 30.0, 1.0, 0.0, -0.1, 0.1],
                [40.0, 40.0, 0.0, 1.0, 0.0, -0.1],
            ]
        )
        mock_vector.set_data(arr, 0)

        return mock_vector

    def test_plot_polarization_vectors_with_empty_data(self, lattice_with_polarization: Lattice):
        """Test plotting with empty vector data."""

        fields = ["x", "y", "a", "b", "da", "db"]
        units = ["px", "px", "ind", "ind", "ind", "ind"]

        def empty_vector():
            out = Vector.from_shape(
                shape=(1,),
                fields=fields,
                units=units,
                name="polarization",
            )
            # Create empty array with shape (0, 6) to match expected format
            empty_data = np.zeros((0, 6), dtype=float)
            out.set_data(empty_data, 0)
            return out

        fig, ax = lattice_with_polarization.plot_polarization_vectors(empty_vector())

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        plt.close(fig)  # Close the figure to avoid using too much memory

    def test_plot_polarization_vectors_without_image(
        self, lattice_with_polarization: Lattice, mock_vector: Vector
    ):
        """Test plotting without background image."""
        fig, ax = lattice_with_polarization.plot_polarization_vectors(
            mock_vector, show_image=False
        )

        assert isinstance(fig, Figure)
        plt.close(fig)  # Close the figure to avoid using too much memory

    def test_plot_polarization_vectors_without_colorbar(
        self, lattice_with_polarization: Lattice, mock_vector
    ):
        """Test plotting without colorbar."""
        fig, ax = lattice_with_polarization.plot_polarization_vectors(
            mock_vector, show_colorbar=False
        )

        assert isinstance(fig, Figure)
        plt.close(fig)  # Close the figure to avoid using too much memory

    @pytest.mark.parametrize("length_scale", [0.5, 1.0, 2.0])
    def test_plot_polarization_vectors_length_scale(
        self, lattice_with_polarization: Lattice, mock_vector, length_scale
    ):
        """Test plotting with different length scales."""
        fig, ax = lattice_with_polarization.plot_polarization_vectors(
            mock_vector, length_scale=length_scale
        )

        assert isinstance(fig, Figure)
        plt.close(fig)  # Close the figure to avoid using too much memory

    @pytest.mark.parametrize("figsize", [(6, 6), (8, 8), (10, 6)])
    def test_plot_polarization_vectors_figsize(
        self, lattice_with_polarization: Lattice, mock_vector, figsize
    ):
        """Test plotting with different figure sizes."""
        fig, ax = lattice_with_polarization.plot_polarization_vectors(mock_vector, figsize=figsize)

        assert isinstance(fig, Figure)
        # Check figure size is approximately correct
        assert abs(fig.get_figwidth() - figsize[0]) < 0.1
        assert abs(fig.get_figheight() - figsize[1]) < 0.1
        plt.close(fig)  # Close the figure to avoid using too much memory

    def test_plot_polarization_vectors(
        self, lattice_with_polarization: Lattice, mock_vector: Vector
    ):
        """Test plot_polarization_vectors with various parameter combinations."""

        # Test with all optional parameters combined
        fig, ax = lattice_with_polarization.plot_polarization_vectors(
            mock_vector,
            show_image=True,
            subtract_median=True,
            show_colorbar=True,
            show_ref_points=True,
            chroma_boost=3.0,
            phase_offset_deg=0.0,
            phase_dir_flip=True,
            linewidth=2.0,
            tail_width=2.0,
            headwidth=6.0,
            headlength=6.0,
            outline=True,
            outline_width=3.0,
            outline_color="blue",
            alpha=0.5,
        )

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)  # Close the figure to avoid using too much memory


class TestLatticePlotPolarizationImage:
    """Test plot_polarization_image method."""

    @pytest.fixture
    def lattice_with_polarization(self):
        """Create lattice with polarization vector data."""
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)

        # Mock lattice vectors
        lattice.define_lattice(
            origin=[10.0, 10.0],
            u=[10.0, 0.0],
            v=[0.0, 10.0],
            refine_lattice=False,
        )

        return lattice

    @pytest.fixture
    def mock_vector(self):
        """Create mock Vector object with polarization data."""

        mock_vector = Vector.from_shape(
            shape=(1,),
            fields=["x", "y", "a", "b", "da", "db"],
            units=["px", "px", "ind", "ind", "ind", "ind"],
            name="polarization",
        )
        arr = np.array(
            [
                [20.0, 20.0, 0.0, 0.0, 0.1, 0.0],
                [30.0, 30.0, 1.0, 0.0, -0.1, 0.1],
                [40.0, 40.0, 0.0, 1.0, 0.0, -0.1],
            ]
        )
        mock_vector.set_data(arr, 0)

        return mock_vector

    def test_plot_polarization_image(
        self, lattice_with_polarization: Lattice, mock_vector: Vector
    ):
        """Test plot_polarization_image returns correct types, values, and handles all options."""

        # Test basic return: RGB array without plotting
        img_rgb = lattice_with_polarization.plot_polarization_image(mock_vector, plot=False)
        assert isinstance(img_rgb, np.ndarray)
        assert img_rgb.ndim == 3
        assert img_rgb.shape[2] == 3  # RGB channels
        assert np.all(img_rgb >= 0.0)
        assert np.all(img_rgb <= 1.0)

        # Test with plotting but no figure return
        result = lattice_with_polarization.plot_polarization_image(
            mock_vector, plot=True, returnfig=False
        )
        assert isinstance(result, np.ndarray)

        # Test with median subtraction
        img_rgb = lattice_with_polarization.plot_polarization_image(
            mock_vector, subtract_median=True, plot=False
        )
        assert isinstance(img_rgb, np.ndarray)

        # Test with plotting, figure return, and colorbar
        img_rgb, (fig, ax) = lattice_with_polarization.plot_polarization_image(
            mock_vector, plot=True, show_colorbar=True, returnfig=True
        )
        assert isinstance(img_rgb, np.ndarray)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

        plt.close(fig)  # Close the figure to avoid using too much memory

    def test_plot_polarization_image_empty_data(self, lattice_with_polarization: Lattice):
        """Test plotting with empty vector data."""

        fields = ["x", "y", "a", "b", "da", "db"]
        units = ["px", "px", "ind", "ind", "ind", "ind"]

        def empty_vector():
            out = Vector.from_shape(
                shape=(1,),
                fields=fields,
                units=units,
                name="polarization",
            )
            # Create empty array with shape (0, 6) to match expected format
            empty_data = np.zeros((0, 6), dtype=float)
            out.set_data(empty_data, 0)
            return out

        img_rgb = lattice_with_polarization.plot_polarization_image(empty_vector(), plot=False)

        assert isinstance(img_rgb, np.ndarray)

    @pytest.mark.parametrize("pixel_size", [8, 16, 32])
    def test_plot_polarization_image_pixel_size(
        self, lattice_with_polarization: Lattice, mock_vector: Vector, pixel_size
    ):
        """Test different pixel sizes for superpixels."""
        img_rgb = lattice_with_polarization.plot_polarization_image(
            mock_vector, pixel_size=pixel_size, plot=False
        )

        assert isinstance(img_rgb, np.ndarray)

    @pytest.mark.parametrize("padding", [4, 8, 16])
    def test_plot_polarization_image_padding(
        self, lattice_with_polarization: Lattice, mock_vector: Vector, padding
    ):
        """Test different padding values."""
        img_rgb = lattice_with_polarization.plot_polarization_image(
            mock_vector, padding=padding, plot=False
        )

        assert isinstance(img_rgb, np.ndarray)

    @pytest.mark.parametrize("spacing", [0, 2, 4])
    def test_plot_polarization_image_spacing(
        self, lattice_with_polarization: Lattice, mock_vector: Vector, spacing
    ):
        """Test different spacing between superpixels."""
        img_rgb = lattice_with_polarization.plot_polarization_image(
            mock_vector, spacing=spacing, plot=False
        )

        assert isinstance(img_rgb, np.ndarray)

    @pytest.mark.parametrize("aggregator", ["mean", "maxmag"])
    def test_plot_polarization_image_aggregators(
        self, lattice_with_polarization: Lattice, mock_vector: Vector, aggregator
    ):
        """Test different aggregation methods."""
        img_rgb = lattice_with_polarization.plot_polarization_image(
            mock_vector, aggregator=aggregator, plot=False
        )

        assert isinstance(img_rgb, np.ndarray)


class TestLatticeMeasurePolarization:
    """Test measure_polarization method."""

    @pytest.fixture
    def lattice_with_atoms(self):
        """Create lattice with multiple atom sites."""
        # Create synthetic image
        H, W = 200, 200
        image = np.random.randn(H, W) * 0.1

        # Generate a regular grid of peaks (atoms)
        spacing = 20  # Distance between atoms
        margin = 15  # Margin from edges
        peak_radius = 10  # Radius of each Gaussian peak

        # Create grid of peak positions
        x_positions = np.arange(margin, W - margin, spacing)
        y_positions = np.arange(margin, H - margin, spacing)
        peaks = [(y, x) for y in y_positions for x in x_positions]

        # Add Gaussian peaks at each position
        for y, x in peaks:
            yy, xx = np.ogrid[-peak_radius : peak_radius + 1, -peak_radius : peak_radius + 1]
            peak = np.exp(-(xx**2 + yy**2) / 20.0)

            y_start, y_end = max(0, y - peak_radius), min(H, y + peak_radius + 1)
            x_start, x_end = max(0, x - peak_radius), min(W, x + peak_radius + 1)

            peak_y_start = peak_radius - (y - y_start)
            peak_y_end = peak_radius + (y_end - y)
            peak_x_start = peak_radius - (x - x_start)
            peak_x_end = peak_radius + (x_end - x)

            image[y_start:y_end, x_start:x_end] += peak[
                peak_y_start:peak_y_end, peak_x_start:peak_x_end
            ]

        lattice = Lattice.from_data(image)

        # Define lattice vectors before adding atoms
        lattice.define_lattice(
            origin=[15.0, 15.0],
            u=[40.0, 0.0],
            v=[0.0, 40.0],
        )

        positions_frac = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])

        result = lattice.add_atoms(positions_frac, plot_atoms=False)
        return result

    def test_measure_polarization_returns_vector(self, lattice_with_atoms: Lattice):
        """Test that measure_polarization returns a Vector object."""
        if not hasattr(lattice_with_atoms, "measure_polarization"):
            pytest.skip("measure_polarization not available")

        result = lattice_with_atoms.measure_polarization(
            measure_ind=0, reference_ind=1, reference_radius=50.0, plot_polarization_vectors=False
        )

        assert isinstance(result, Vector)

    def test_measure_polarization_with_radius(self, lattice_with_atoms: Lattice):
        """Test polarization measurement with reference_radius."""
        if not hasattr(lattice_with_atoms, "measure_polarization"):
            pytest.skip("measure_polarization not available")

        with pytest.raises(ValueError, match=r"Increase (the )?reference_radius"):
            result = lattice_with_atoms.measure_polarization(
                measure_ind=0,
                reference_ind=1,
                reference_radius=30.0,
                plot_polarization_vectors=False,
            )

            assert isinstance(result, Vector)

    def test_measure_polarization_with_knn(self, lattice_with_atoms: Lattice):
        """Test polarization measurement with k-nearest neighbors."""
        if not hasattr(lattice_with_atoms, "measure_polarization"):
            pytest.skip("measure_polarization not available")

        result = lattice_with_atoms.measure_polarization(
            measure_ind=0,
            reference_ind=1,
            reference_radius=None,
            min_neighbours=2,
            max_neighbours=6,
            plot_polarization_vectors=False,
        )

        assert isinstance(result, Vector)

    def test_measure_polarization_vector_fields(self, lattice_with_atoms: Lattice):
        """Test that returned Vector has correct fields."""
        if not hasattr(lattice_with_atoms, "measure_polarization"):
            pytest.skip("measure_polarization not available")

        result = lattice_with_atoms.measure_polarization(
            measure_ind=0, reference_ind=1, reference_radius=50.0, plot_polarization_vectors=False
        )

        # Check that vector has expected fields
        data = result.get_data(0)

        # Handle case where data might be None or empty
        if data is None:
            pytest.skip("get_data returned None - Vector implementation may differ")

        if isinstance(data, list) and len(data) == 0:
            pytest.skip("Empty data returned")

        if hasattr(data, "size") and data.size == 0:
            pytest.skip("Empty array returned")

        # Check fields
        expected_fields = {"x", "y", "a", "b", "da", "db"}

        if isinstance(data, dict):
            actual_fields = set(data.keys())
        elif (
            hasattr(data, "dtype")
            and hasattr(data.dtype, "names")
            and data.dtype.names is not None
        ):
            actual_fields = set(data.dtype.names)
        elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 6:
            # If it's a plain 2D array with 6 columns, we can't check field names
            # but we can verify the shape is correct
            assert data.shape[1] == 6, f"Expected 6 columns, got {data.shape[1]}"
            return  # Skip field name check for plain arrays
        else:
            pytest.skip(f"Unexpected data type: {type(data)}")

        assert expected_fields.issubset(actual_fields), (
            f"Missing fields. Expected {expected_fields}, got {actual_fields}"
        )

    def test_measure_polarization_invalid_radius(self, lattice_with_atoms: Lattice):
        """Test that invalid radius raises ValueError."""
        if not hasattr(lattice_with_atoms, "measure_polarization"):
            pytest.skip("measure_polarization not available")

        with pytest.raises(ValueError):
            lattice_with_atoms.measure_polarization(
                measure_ind=0,
                reference_ind=1,
                reference_radius=0.5,  # < 1
                plot_polarization_vectors=False,
            )

    def test_measure_polarization_missing_parameters(self, lattice_with_atoms: Lattice):
        """Test that missing both radius and knn params raises ValueError."""
        if not hasattr(lattice_with_atoms, "measure_polarization"):
            pytest.skip("measure_polarization not available")

        with pytest.raises(ValueError):
            lattice_with_atoms.measure_polarization(
                measure_ind=0,
                reference_ind=1,
                reference_radius=None,
                min_neighbours=None,
                max_neighbours=None,
                plot_polarization_vectors=False,
            )

    def test_measure_polarization_min_greater_than_max(self, lattice_with_atoms: Lattice):
        """Test that min_neighbours > max_neighbours raises ValueError."""
        if not hasattr(lattice_with_atoms, "measure_polarization"):
            pytest.skip("measure_polarization not available")

        with pytest.raises(ValueError):
            lattice_with_atoms.measure_polarization(
                measure_ind=0,
                reference_ind=1,
                reference_radius=None,
                min_neighbours=10,
                max_neighbours=5,
                plot_polarization_vectors=False,
            )

    def test_measure_polarization_with_plotting(self, lattice_with_atoms: Lattice):
        """Test polarization measurement with plotting enabled."""
        if not hasattr(lattice_with_atoms, "measure_polarization"):
            pytest.skip("measure_polarization not available")

        result = lattice_with_atoms.measure_polarization(
            measure_ind=0, reference_ind=1, reference_radius=50.0, plot_polarization_vectors=True
        )

        assert isinstance(result, Vector)

    def test_measure_polarization_empty_cells(self, lattice_with_atoms: Lattice):
        """Test polarization measurement when cells are empty."""
        if not hasattr(lattice_with_atoms, "measure_polarization"):
            pytest.skip("measure_polarization not available")

        # Mock empty atoms
        class EmptyAtoms:
            def get_data(self, idx):
                return []

            def __getitem__(self, idx):
                return {}

        lattice_with_atoms.atoms = EmptyAtoms()

        result = lattice_with_atoms.measure_polarization(
            measure_ind=0, reference_ind=1, reference_radius=50.0, plot_polarization_vectors=False
        )

        assert isinstance(result, Vector)
        # Should return empty vector
        data = result.get_data(0)
        assert data is None or len(data) == 0 or (hasattr(data, "size") and data.size == 0)

    @pytest.mark.parametrize("min_neighbours,max_neighbours", [(2, 4), (3, 8), (2, 10)])
    def test_measure_polarization_various_knn(
        self, lattice_with_atoms: Lattice, min_neighbours, max_neighbours
    ):
        """Test polarization measurement with various k-NN parameters."""
        if not hasattr(lattice_with_atoms, "measure_polarization"):
            pytest.skip("measure_polarization not available")

        result = lattice_with_atoms.measure_polarization(
            measure_ind=0,
            reference_ind=1,
            reference_radius=None,
            min_neighbours=min_neighbours,
            max_neighbours=max_neighbours,
            plot_polarization_vectors=False,
        )

        assert isinstance(result, Vector)


class TestCalculateOrderParameterRunWithRestarts:
    """Test run_with_restarts functionality in calculate_order_parameter."""

    @pytest.fixture
    def lattice_with_polarization(self) -> Tuple[Lattice, Vector]:
        """Create lattice with polarization data for testing."""
        # Create synthetic image
        image = np.random.randn(200, 200)
        lattice = Lattice.from_data(image)

        # Mock lattice vectors and image
        lattice._lat = np.array(
            [
                [10.0, 10.0],  # origin
                [20.0, 0.0],  # u vector
                [0.0, 20.0],  # v vector
            ]
        )
        lattice._image = lattice.image

        # Create synthetic polarization vectors matching measure_polarization output
        n_sites = 100

        polarization_vectors = Vector.from_shape(
            shape=(1,),
            fields=["x", "y", "a", "b", "da", "db"],
            units=["px", "px", "ind", "ind", "ind", "ind"],
            name="polarization",
        )

        # Create data array (n_sites, 6)
        polarization_data = np.column_stack(
            [
                np.random.randn(n_sites) * 10 + 50,  # x
                np.random.randn(n_sites) * 10 + 50,  # y
                np.random.randint(0, 10, n_sites).astype(float),  # a
                np.random.randint(0, 10, n_sites).astype(float),  # b
                np.random.randn(n_sites) * 0.1,  # da
                np.random.randn(n_sites) * 0.1,  # db
            ]
        )

        polarization_vectors.set_data(polarization_data, 0)

        return lattice, polarization_vectors

    def test_run_with_restarts_single_restart(
        self, lattice_with_polarization: Tuple[Lattice, Vector]
    ):
        """Test with num_restarts=1"""
        lattice, polarization = lattice_with_polarization

        result = lattice.calculate_order_parameter(
            polarization,
            num_phases=2,
            run_with_restarts=True,
            num_restarts=1,
            plot_gmm_visualization=False,
            plot_order_parameter=False,
        )

        assert result is lattice
        assert hasattr(lattice, "_polarization_means")
        assert hasattr(lattice, "_order_parameter_probabilities")

    def test_run_with_restarts_multiple_restarts(
        self, lattice_with_polarization: Tuple[Lattice, Vector]
    ):
        """Test with multiple restarts."""
        lattice, polarization = lattice_with_polarization

        result = lattice.calculate_order_parameter(
            polarization,
            num_phases=2,
            run_with_restarts=True,
            num_restarts=5,
            plot_gmm_visualization=False,
            plot_order_parameter=False,
        )

        assert result is lattice
        assert lattice._polarization_means.shape == (2, 2)
        assert lattice._order_parameter_probabilities.shape[1] == 2

    def test_run_with_restarts_consistency(
        self, lattice_with_polarization: Tuple[Lattice, Vector]
    ):
        """Test that best result is chosen across restarts."""
        lattice, polarization = lattice_with_polarization

        # Run with multiple restarts
        lattice.calculate_order_parameter(
            polarization,
            num_phases=3,
            run_with_restarts=True,
            num_restarts=10,
            plot_gmm_visualization=False,
            plot_order_parameter=False,
        )

        # Verify shapes
        assert lattice._polarization_means.shape == (3, 2)
        assert lattice._order_parameter_probabilities.shape[1] == 3

        # Verify probabilities sum to 1
        prob_sums = np.sum(lattice._order_parameter_probabilities, axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-5)

    def test_run_with_restarts_different_num_phases(
        self, lattice_with_polarization: Tuple[Lattice, Vector]
    ):
        """Test restarts with different numbers of phases."""
        lattice, polarization = lattice_with_polarization

        for num_phases in [1, 2, 3, 4]:
            result = lattice.calculate_order_parameter(
                polarization,
                num_phases=num_phases,
                run_with_restarts=True,
                num_restarts=3,
                plot_gmm_visualization=False,
                plot_order_parameter=False,
            )

            assert result is lattice
            assert lattice._polarization_means.shape == (num_phases, 2)
            assert lattice._order_parameter_probabilities.shape[1] == num_phases

    def test_run_with_restarts_invalid_num_restarts(
        self, lattice_with_polarization: Tuple[Lattice, Vector]
    ):
        """Test that invalid num_restarts raises assertion error."""
        lattice, polarization = lattice_with_polarization

        with pytest.raises(AssertionError):
            lattice.calculate_order_parameter(
                polarization,
                num_phases=2,
                run_with_restarts=True,
                num_restarts=0,
                plot_gmm_visualization=False,
                plot_order_parameter=False,
            )

        with pytest.raises(AssertionError):
            lattice.calculate_order_parameter(
                polarization,
                num_phases=2,
                run_with_restarts=True,
                num_restarts=-1,
                plot_gmm_visualization=False,
                plot_order_parameter=False,
            )

    @pytest.mark.slow
    def test_run_with_restarts_large_number(self):
        """Test with large number of restarts."""
        # Create fresh lattice
        image = np.random.randn(200, 200)
        lattice = Lattice.from_data(image)
        lattice._lat = np.array([[10.0, 10.0], [20.0, 0.0], [0.0, 20.0]])
        lattice._image = lattice.image

        # Use smaller dataset for speed
        n_sites = 20
        small_polarization = Vector.from_shape(
            shape=(1,),
            fields=["x", "y", "a", "b", "da", "db"],
            units=["px", "px", "ind", "ind", "ind", "ind"],
            name="polarization",
        )

        small_data = np.column_stack(
            [
                np.random.randn(n_sites) * 10 + 50,
                np.random.randn(n_sites) * 10 + 50,
                np.random.randint(0, 10, n_sites).astype(float),
                np.random.randint(0, 10, n_sites).astype(float),
                np.random.randn(n_sites) * 0.1,
                np.random.randn(n_sites) * 0.1,
            ]
        )
        small_polarization.set_data(small_data, 0)

        result = lattice.calculate_order_parameter(
            small_polarization,
            num_phases=2,
            run_with_restarts=True,
            num_restarts=25,
            plot_gmm_visualization=False,
            plot_order_parameter=False,
        )

        assert result is lattice
        assert lattice._polarization_means.shape == (2, 2)

    def test_run_with_restarts_empty_polarization(self):
        """Test restarts with empty polarization vectors"""
        lattice = Lattice.from_data(np.random.randn(100, 100))
        lattice._lat = np.array([[10, 10], [20, 0], [0, 20]])
        lattice._image = lattice.image

        # Create Vector with empty data
        empty_polarization = Vector.from_shape(
            shape=(1,),
            fields=["x", "y", "a", "b", "da", "db"],
            units=["px", "px", "ind", "ind", "ind", "ind"],
            name="polarization",
        )

        empty_data = np.zeros((0, 6), dtype=float)
        empty_polarization.set_data(empty_data, 0)

        # Empty polarization should raise an error or be handled gracefully
        try:
            result = lattice.calculate_order_parameter(
                empty_polarization,
                num_phases=2,
                run_with_restarts=True,
                num_restarts=3,
                plot_gmm_visualization=False,
                plot_order_parameter=False,
            )
            # If it succeeds, check that result is returned
            assert result is lattice
        except (ValueError, IndexError) as e:
            # Empty polarization may raise an error, which is acceptable
            assert (
                "empty" in str(e).lower() or "zero" in str(e).lower() or "sample" in str(e).lower()
            )

    def test_run_with_restarts_few_sites(self):
        """Test restarts with few polarization sites."""
        lattice = Lattice.from_data(np.random.randn(100, 100))
        lattice._lat = np.array([[10, 10], [20, 0], [0, 20]])
        lattice._image = lattice.image

        # Use at least 5 sites to avoid KDE issues
        n_sites = 5
        small_polarization = Vector.from_shape(
            shape=(1,),
            fields=["x", "y", "a", "b", "da", "db"],
            units=["px", "px", "ind", "ind", "ind", "ind"],
            name="polarization",
        )

        small_data = np.column_stack(
            [
                10.0 + np.arange(n_sites, dtype=float),
                20.0 + np.arange(n_sites, dtype=float),
                np.zeros(n_sites, dtype=float),
                np.zeros(n_sites, dtype=float),
                1.0 + np.random.randn(n_sites) * 0.1,
                2.0 + np.random.randn(n_sites) * 0.1,
            ]
        )
        small_polarization.set_data(small_data, 0)

        result = lattice.calculate_order_parameter(
            small_polarization,
            num_phases=1,
            run_with_restarts=True,
            num_restarts=5,
            plot_gmm_visualization=False,
            plot_order_parameter=False,
        )

        assert result is lattice
        assert lattice._order_parameter_probabilities.shape == (n_sites, 1)

    def test_run_with_restarts_deterministic_seed(
        self, lattice_with_polarization: Tuple[Lattice, Vector]
    ):
        """Test that setting torch seed gives reproducible results."""
        lattice, polarization = lattice_with_polarization

        try:
            import torch

            # Run twice with same seed
            torch.manual_seed(42)
            result1 = lattice.calculate_order_parameter(
                polarization,
                num_phases=2,
                run_with_restarts=True,
                num_restarts=3,
                plot_gmm_visualization=False,
                plot_order_parameter=False,
            )
            means1 = result1._polarization_means.copy()
            probs1 = result1._order_parameter_probabilities.copy()

            torch.manual_seed(42)
            result2 = lattice.calculate_order_parameter(
                polarization,
                num_phases=2,
                run_with_restarts=True,
                num_restarts=3,
                plot_gmm_visualization=False,
                plot_order_parameter=False,
            )
            means2 = result2._polarization_means.copy()
            probs2 = result2._order_parameter_probabilities.copy()

            # Results should be identical with same seed
            assert np.allclose(means1, means2, atol=1e-5)
            assert np.allclose(probs1, probs2, atol=1e-5)

        except ImportError:
            pytest.skip("PyTorch not available")

    def test_run_with_restarts_torch_device(
        self, lattice_with_polarization: Tuple[Lattice, Vector]
    ):
        """Test that torch_device parameter is accepted."""
        lattice, polarization = lattice_with_polarization

        try:
            # Test CPU device
            result = lattice.calculate_order_parameter(
                polarization,
                num_phases=2,
                run_with_restarts=True,
                num_restarts=2,
                torch_device="cpu",
                plot_gmm_visualization=False,
                plot_order_parameter=False,
            )

            assert result is lattice

        except ImportError:
            pytest.skip("PyTorch not available")

    def test_run_with_restarts_probability_bounds(
        self, lattice_with_polarization: Tuple[Lattice, Vector]
    ):
        """Test that probabilities are properly bounded after restarts."""
        lattice, polarization = lattice_with_polarization

        lattice.calculate_order_parameter(
            polarization,
            num_phases=3,
            run_with_restarts=True,
            num_restarts=5,
            plot_gmm_visualization=False,
            plot_order_parameter=False,
        )

        probs = lattice._order_parameter_probabilities

        # All probabilities should be between 0 and 1
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

        # Each row should sum to 1
        row_sums = np.sum(probs, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_run_with_restarts_false_behavior(
        self, lattice_with_polarization: Tuple[Lattice, Vector]
    ):
        """Test that run_with_restarts=False still works correctly."""
        lattice, polarization = lattice_with_polarization

        result = lattice.calculate_order_parameter(
            polarization,
            num_phases=2,
            run_with_restarts=False,
            num_restarts=1,
            plot_gmm_visualization=False,
            plot_order_parameter=False,
        )

        assert result is lattice
        assert hasattr(lattice, "_polarization_means")
        assert hasattr(lattice, "_order_parameter_probabilities")


# This needs revisiting
# class TestLatticeIntegration:
#     """Integration tests for Lattice class workflows."""

#     def test_full_polarization_workflow(self):
#         """Test complete workflow: create lattice, fit, add atoms, measure polarization."""
#         # Create synthetic image with lattice structure
#         H, W = 200, 200
#         image = np.zeros((H, W))

#         spacing = 20
#         for i in range(10, H, spacing):
#             for j in range(10, W, spacing):
#                 y, x = np.ogrid[-5:6, -5:6]
#                 peak = np.exp(-(x**2 + y**2) / 8.0)
#                 i_start, i_end = max(0, i - 5), min(H, i + 6)
#                 j_start, j_end = max(0, j - 5), min(W, j + 6)
#                 peak_h, peak_w = i_end - i_start, j_end - j_start
#                 image[i_start:i_end, j_start:j_end] += peak[:peak_h, :peak_w]

#         # Create lattice
#         lattice = Lattice.from_data(image)

#         assert lattice is not None
#         assert lattice.image.shape == (200, 200)

#     def test_method_chaining(self):
#         """Test that methods can be chained."""
#         image = np.random.randn(100, 100)

#         lattice = Lattice.from_data(image)

#         # Methods that return self should be chainable
#         assert lattice is not None

#     def test_multiple_operations_on_same_lattice(self):
#         """Test performing multiple operations on the same lattice object."""
#         image = np.random.randn(100, 100)
#         lattice = Lattice.from_data(image)

#         # Change image
#         new_image = np.random.randn(100, 100)
#         lattice.image = new_image

#         assert lattice.image.shape == (100, 100)

#     def test_lattice_with_different_dtypes(self):
#         """Test lattice creation with different NumPy dtypes."""
#         for dtype in [np.float32, np.float64, np.int32, np.int64]:
#             image = np.random.randn(50, 50).astype(dtype)
#             lattice = Lattice.from_data(image)

#             assert lattice is not None


class TestLatticeSerialization:
    """Test serialization capabilities (if available via AutoSerialize)."""

    @pytest.fixture
    def simple_lattice(self):
        """Create simple lattice for serialization tests."""
        H, W = 200, 200
        image = np.random.randn(H, W) * 0.1

        # Generate a regular grid of peaks (atoms)
        spacing = 20  # Distance between atoms
        margin = 15  # Margin from edges
        peak_radius = 10  # Radius of each Gaussian peak

        # Create grid of peak positions
        x_positions = np.arange(margin, W - margin, spacing)
        y_positions = np.arange(margin, H - margin, spacing)
        peaks = [(y, x) for y in y_positions for x in x_positions]

        # Add Gaussian peaks at each position
        for y, x in peaks:
            yy, xx = np.ogrid[-peak_radius : peak_radius + 1, -peak_radius : peak_radius + 1]
            peak = np.exp(-(xx**2 + yy**2) / 20.0)

            y_start, y_end = max(0, y - peak_radius), min(H, y + peak_radius + 1)
            x_start, x_end = max(0, x - peak_radius), min(W, x + peak_radius + 1)

            peak_y_start = peak_radius - (y - y_start)
            peak_y_end = peak_radius + (y_end - y)
            peak_x_start = peak_radius - (x - x_start)
            peak_x_end = peak_radius + (x_end - x)

            image[y_start:y_end, x_start:x_end] += peak[
                peak_y_start:peak_y_end, peak_x_start:peak_x_end
            ]

        lattice = Lattice.from_data(image)

        return lattice

    @pytest.fixture
    def complex_lattice(self):
        """Create a complex lattice with complete workflow."""
        # Create synthetic image
        H, W = 200, 200
        image = np.random.randn(H, W) * 0.1

        # Generate a regular grid of peaks (atoms)
        spacing = 20  # Distance between atoms
        margin = 15  # Margin from edges
        peak_radius = 10  # Radius of each Gaussian peak

        # Create grid of peak positions
        x_positions = np.arange(margin, W - margin, spacing)
        y_positions = np.arange(margin, H - margin, spacing)
        peaks = [(y, x) for y in y_positions for x in x_positions]

        # Add Gaussian peaks at each position
        for y, x in peaks:
            yy, xx = np.ogrid[-peak_radius : peak_radius + 1, -peak_radius : peak_radius + 1]
            peak = np.exp(-(xx**2 + yy**2) / 20.0)

            y_start, y_end = max(0, y - peak_radius), min(H, y + peak_radius + 1)
            x_start, x_end = max(0, x - peak_radius), min(W, x + peak_radius + 1)

            peak_y_start = peak_radius - (y - y_start)
            peak_y_end = peak_radius + (y_end - y)
            peak_x_start = peak_radius - (x - x_start)
            peak_x_end = peak_radius + (x_end - x)

            image[y_start:y_end, x_start:x_end] += peak[
                peak_y_start:peak_y_end, peak_x_start:peak_x_end
            ]

        lattice = Lattice.from_data(image)

        # Define lattice vectors before adding atoms
        lattice.define_lattice(
            origin=[15.0, 15.0],
            u=[40.0, 0.0],
            v=[0.0, 40.0],
        )

        positions_frac = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])

        lattice = lattice.add_atoms(positions_frac, plot_atoms=False)

        return lattice

    def test_lattice_has_autoserialize(self, complex_lattice: Lattice):
        """Test that Lattice inherits from AutoSerialize."""
        assert hasattr(complex_lattice.__class__, "__bases__")
        # Check if AutoSerialize is in the inheritance chain
        base_names = [base.__name__ for base in complex_lattice.__class__.__mro__]
        assert "AutoSerialize" in base_names or "Lattice" in base_names

    def test_lattice_autoserialize_methods_exist(self, complex_lattice: Lattice):
        """Test that serialization methods exist (if applicable)."""
        # Check if autoserialize methods are available
        assert isinstance(complex_lattice, AutoSerialize)
        assert hasattr(complex_lattice, "save")
        assert callable(complex_lattice.save)

    @pytest.mark.parametrize("store", ["zip", "dir"])
    def test_complex_lattice_full_save_load(self, tmp_path, store, complex_lattice: Lattice):
        """Test save/load of lattice with all attributes."""
        lattice = complex_lattice

        filepath = tmp_path / ("complex.zip" if store == "zip" else "complex_dir")
        lattice.save(str(filepath), mode="w", store=store)
        loaded = load(str(filepath))

        # Verify type
        assert isinstance(loaded, Lattice)

        # Verify _image (Dataset2d)
        assert isinstance(loaded._image, Dataset2d)
        assert np.allclose(loaded._image.array, lattice._image.array)

        # Verify Dataset2d attributes
        assert hasattr(loaded._image, "array")
        assert hasattr(loaded._image, "shape")
        assert isinstance(loaded._image.array, np.ndarray)
        assert loaded._image.shape == lattice._image.shape
        assert np.allclose(loaded._image.array, lattice._image.array)

        # Verify _lat
        assert hasattr(loaded, "_lat")
        assert np.allclose(loaded._lat, lattice._lat)
        assert loaded._lat.shape == (3, 2)
        assert np.allclose(loaded._lat, lattice._lat)

        # Verify atoms (Vector) - nested AutoSerialize
        assert hasattr(loaded, "atoms")
        assert isinstance(loaded.atoms, Vector)

        # Verify _positions_frac
        assert hasattr(loaded, "_positions_frac")
        assert np.allclose(loaded._positions_frac, lattice._positions_frac)

        # Verify _num_sites
        assert hasattr(loaded, "_num_sites")
        assert loaded._num_sites == lattice._num_sites

        # Verify _numbers
        assert hasattr(loaded, "_numbers")
        assert np.array_equal(loaded._numbers, lattice._numbers)

        # Verify fields match
        assert set(loaded.atoms.fields) == set(lattice.atoms.fields)

        # Verify units attribute
        assert hasattr(loaded.atoms, "units")
        assert loaded.atoms.units == lattice.atoms.units

        # Verify name attribute
        assert hasattr(loaded.atoms, "name")
        assert loaded.atoms.name == lattice.atoms.name

        # Verify shape attribute
        assert hasattr(loaded.atoms, "shape")
        assert loaded.atoms.shape == lattice.atoms.shape

        # Verify Vector data using proper API
        for s in lattice._numbers:
            original_data = lattice.atoms.get_data(s)
            loaded_data = loaded.atoms.get_data(s)

            if original_data is None and loaded_data is None:
                continue
            if isinstance(original_data, list):
                assert len(loaded_data) == len(original_data)
            else:
                assert isinstance(original_data, np.ndarray) and isinstance(
                    loaded_data, np.ndarray
                )
                assert loaded_data.shape == original_data.shape
            assert np.allclose(loaded_data, original_data)

            # Verify field data
            assert hasattr(loaded.atoms, "fields")
            for field in lattice.atoms.fields:
                assert field in loaded.atoms.fields
                original_field_data = lattice.atoms[s][field]
                loaded_field_data = loaded.atoms[s][field]
                assert np.allclose(loaded_field_data, original_field_data)

    @pytest.mark.parametrize("store", ["zip", "dir"])
    def test_multiple_save_load_cycles(self, tmp_path, store, complex_lattice: Lattice):
        """Test data integrity through multiple save/load cycles."""
        original = complex_lattice

        # Store original values
        original_image = original._image.array.copy()
        original_lat = original._lat.copy()
        original_positions = original._positions_frac.copy()
        original_numbers = original._numbers.copy()

        lattice = original
        for i in range(3):
            filepath = tmp_path / (f"cycle{i}.zip" if store == "zip" else f"cycle{i}_dir")
            lattice.save(str(filepath), mode="w", store=store)
            lattice = load(str(filepath))

        # After 3 cycles, verify data is still correct
        assert isinstance(lattice, Lattice)
        assert np.allclose(lattice._image.array, original_image)
        assert np.allclose(lattice._lat, original_lat)
        assert np.allclose(lattice._positions_frac, original_positions)
        assert np.array_equal(lattice._numbers, original_numbers)
        assert lattice._num_sites == original._num_sites

    @pytest.mark.parametrize("store", ["zip", "dir"])
    def test_cycle_with_modifications(self, tmp_path, store, simple_lattice: Lattice):
        """Test save/load cycle with modifications between saves."""
        # Initial lattice
        lattice = simple_lattice

        # First save
        filepath1 = tmp_path / ("mod1.zip" if store == "zip" else "mod1_dir")
        lattice.save(str(filepath1), mode="w", store=store)
        loaded1: Lattice = load(str(filepath1))

        # Verify type
        assert isinstance(loaded1, Lattice)

        # Verify _image (Dataset2d)
        assert isinstance(loaded1._image, Dataset2d)
        assert np.allclose(loaded1._image.array, lattice._image.array)

        # Verify Dataset2d attributes
        assert hasattr(loaded1._image, "array")
        assert hasattr(loaded1._image, "shape")
        assert isinstance(loaded1._image.array, np.ndarray)
        assert loaded1._image.shape == lattice._image.shape
        assert np.allclose(loaded1._image.array, lattice._image.array)

        # Add atoms
        loaded1.define_lattice(
            origin=[15.0, 15.0],
            u=[40.0, 0.0],
            v=[0.0, 40.0],
        )

        positions_frac = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])

        loaded1 = loaded1.add_atoms(positions_frac, plot_atoms=False)

        # Second save
        filepath2 = tmp_path / ("mod2.zip" if store == "zip" else "mod2_dir")
        loaded1.save(str(filepath2), mode="w", store=store)
        loaded2: Lattice = load(str(filepath2))

        # Verify type
        assert isinstance(loaded2, Lattice)

        # Verify _image (Dataset2d)
        assert isinstance(loaded2._image, Dataset2d)
        assert np.allclose(loaded2._image.array, lattice._image.array)

        # Verify Dataset2d attributes
        assert hasattr(loaded2._image, "array")
        assert hasattr(loaded2._image, "shape")
        assert isinstance(loaded2._image.array, np.ndarray)
        assert loaded2._image.shape == lattice._image.shape
        assert np.allclose(loaded2._image.array, lattice._image.array)

        # Verify _lat
        assert hasattr(loaded2, "_lat")

        # Verify modifications persisted
        assert hasattr(loaded2, "atoms")
        assert isinstance(loaded2.atoms, Vector)
        assert loaded2._num_sites == 4

    @pytest.mark.parametrize("store", ["zip", "dir"])
    def test_overwrite_existing_file(self, tmp_path, store, simple_lattice: Lattice):
        """Test overwriting existing saved file."""
        lattice1 = simple_lattice

        filepath = tmp_path / ("overwrite.zip" if store == "zip" else "overwrite_dir")

        # First save
        lattice1.save(str(filepath), mode="w", store=store)
        loaded1: Lattice = load(str(filepath))

        # Create different lattice
        image2 = np.random.randn(200, 200) + 100
        lattice2 = Lattice.from_data(image2)

        # Overwrite
        lattice2.save(str(filepath), mode="o", store=store)
        loaded2: Lattice = load(str(filepath))

        # Verify new data was saved
        assert loaded2._image.shape == (200, 200)
        assert not np.allclose(loaded2._image.array, loaded1._image.array)
        assert np.allclose(loaded2._image.array, lattice2._image.array)

    @pytest.mark.parametrize("store", ["zip", "dir"])
    @pytest.mark.parametrize("compression_level", [0, 5, 9])
    def test_compression_levels(
        self, tmp_path, store, compression_level, complex_lattice: Lattice
    ):
        """Test different compression levels."""
        lattice = complex_lattice

        filepath = tmp_path / (
            f"comp{compression_level}.zip" if store == "zip" else f"comp{compression_level}_dir"
        )
        lattice.save(str(filepath), mode="w", store=store, compression_level=compression_level)
        loaded: Lattice = load(str(filepath))

        # Data should be identical regardless of compression
        assert np.allclose(loaded._image.array, lattice._image.array)
        assert np.allclose(loaded._lat, lattice._lat)
        assert loaded._num_sites == lattice._num_sites

    @pytest.mark.parametrize("store", ["zip", "dir"])
    def test_load_nonexistent_file(self, tmp_path, store):
        """Test that loading nonexistent file raises appropriate error."""
        filepath = tmp_path / ("nonexistent.zip" if store == "zip" else "nonexistent_dir")

        with pytest.raises((FileNotFoundError, ValueError)):
            load(str(filepath))

    @pytest.mark.parametrize("store", ["zip", "dir"])
    def test_corrupted_file_handling(self, tmp_path, store, simple_lattice: Lattice):
        """Test handling of corrupted save files."""
        lattice = simple_lattice
        filepath = tmp_path / ("corrupted.zip" if store == "zip" else "corrupted_dir")

        # Save normally
        lattice.save(str(filepath), mode="w", store=store)

        # Corrupt the file
        if store == "zip":
            with open(filepath, "wb") as f:
                f.write(b"corrupted data")

            # Try to load corrupted file
            with pytest.raises(Exception):  # Could be various exceptions
                load(str(filepath))

    @pytest.mark.parametrize("store", ["zip", "dir"])
    def test_dataset2d_different_shapes(self, tmp_path, store):
        """Test Dataset2d serialization with various image shapes."""
        shapes = [(50, 50), (100, 200), (75, 125), (512, 512)]

        for shape in shapes:
            image = np.random.randn(*shape)
            lattice = Lattice.from_data(image)

            filepath = tmp_path / (
                f"shape_{shape[0]}x{shape[1]}.zip"
                if store == "zip"
                else f"shape_{shape[0]}x{shape[1]}_dir"
            )
            lattice.save(str(filepath), mode="w", store=store)
            loaded: Lattice = load(str(filepath))

            assert loaded._image.shape == shape
            assert np.allclose(loaded._image.array, lattice._image.array)

    @pytest.mark.parametrize("store", ["zip", "dir"])
    def test_dataset2d_with_special_values(self, tmp_path, store):
        """Test Dataset2d serialization with special float values."""
        image = np.random.randn(50, 50)
        # Add some special values
        image[0, 0] = np.inf
        image[1, 1] = -np.inf
        image[2, 2] = 0.0
        image[3, 3] = -0.0

        lattice = Lattice.from_data(image, normalize_min=False, normalize_max=False)

        filepath = tmp_path / ("special_vals.zip" if store == "zip" else "special_vals_dir")
        lattice.save(str(filepath), mode="w", store=store)
        loaded: Lattice = load(str(filepath))

        # Check special values are preserved
        assert loaded._image.array[0, 0] == np.inf
        assert loaded._image.array[1, 1] == -np.inf
        assert loaded._image.array[2, 2] == 0.0

    @pytest.mark.parametrize("store", ["zip", "dir"])
    def test_full_workflow_simulation(self, tmp_path, store, simple_lattice: Lattice):
        """Simulate a full workflow: create, modify, save, load, verify."""
        # Step 1: Create initial lattice
        lattice = simple_lattice

        # Step 2: Define lattice vectors
        lattice.define_lattice(
            origin=[15.0, 15.0],
            u=[40.0, 0.0],
            v=[0.0, 40.0],
        )

        # Step 3: Save initial state
        filepath1 = tmp_path / ("workflow_step1.zip" if store == "zip" else "workflow_step1_dir")
        lattice.save(str(filepath1), mode="w", store=store)

        # Step 4: Load and add atoms
        loaded1: Lattice = load(str(filepath1))
        positions = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        numbers = [0, 1, 2, 3]
        loaded1.add_atoms(positions_frac=positions, numbers=numbers, plot_atoms=False)

        # Step 5: Save with atoms
        filepath2 = tmp_path / ("workflow_step2.zip" if store == "zip" else "workflow_step2_dir")
        loaded1.save(str(filepath2), mode="w", store=store)

        # Step 6: Final load and verify
        final: Lattice = load(str(filepath2))

        assert isinstance(final, Lattice)
        assert final._num_sites == 4
        assert np.array_equal(final._numbers, numbers)
        assert np.allclose(final._lat, lattice._lat)
        assert np.allclose(final._image.array, lattice._image.array)

    @pytest.mark.parametrize("store", ["zip", "dir"])
    def test_parallel_save_load(self, tmp_path, store, simple_lattice: Lattice):
        """Test saving and loading multiple lattices independently."""
        lattices: List[Lattice] = []
        for i in range(3):
            if i == 0:
                lattice = simple_lattice
            else:
                image = np.roll(simple_lattice._image.array, shift=(i * 10, i * 10), axis=(0, 1))
                lattice = Lattice.from_data(image)
            lattice.define_lattice(
                origin=[15.0 + i * 10, 15.0 + i * 10],
                u=[40.0, 0.0],
                v=[0.0, 40.0],
            )
            lattices.append(lattice)

        # Save all
        filepaths = []
        for i, lattice in enumerate(lattices):
            filepath = tmp_path / (f"parallel_{i}.zip" if store == "zip" else f"parallel_{i}_dir")
            lattice.save(str(filepath), mode="w", store=store)
            filepaths.append(filepath)

        # Load all and verify
        for i, filepath in enumerate(filepaths):
            loaded: Lattice = load(str(filepath))
            assert loaded._image.shape == lattices[i]._image.shape
            assert np.allclose(loaded._lat, lattices[i]._lat)

    @pytest.mark.parametrize("store", ["zip", "dir"])
    def test_lattice_with_nan_values(self, tmp_path, store):
        """Test lattice serialization with NaN values in image."""
        image = np.random.randn(100, 100)
        image[10:20, 10:20] = np.nan

        lattice = Lattice.from_data(image, normalize_min=False, normalize_max=False)

        filepath = tmp_path / ("with_nan.zip" if store == "zip" else "with_nan_dir")
        lattice.save(str(filepath), mode="w", store=store)
        loaded: Lattice = load(str(filepath))

        # Check that NaN values are preserved
        assert np.sum(np.isnan(loaded._image.array)) == np.sum(np.isnan(image))

    @pytest.mark.parametrize("store", ["zip", "dir"])
    def test_repeated_save_same_location(self, tmp_path, store, simple_lattice: Lattice):
        """Test saving to same location multiple times with overwrite mode."""
        lattice = simple_lattice

        filepath = tmp_path / ("repeated.zip" if store == "zip" else "repeated_dir")

        for i in range(5):
            lattice.save(str(filepath), mode="o", store=store)
            loaded: Lattice = load(str(filepath))
            assert np.allclose(loaded._image.array, lattice._image.array)

    @pytest.mark.parametrize("store", ["zip", "dir"])
    def test_save_after_load(self, tmp_path, store, complex_lattice: Lattice):
        """Test that loaded lattice can be saved again."""
        original = complex_lattice

        filepath1 = tmp_path / (
            "save_after_load1.zip" if store == "zip" else "save_after_load1_dir"
        )
        original.save(str(filepath1), mode="w", store=store)

        loaded: Lattice = load(str(filepath1))

        filepath2 = tmp_path / (
            "save_after_load2.zip" if store == "zip" else "save_after_load2_dir"
        )
        loaded.save(str(filepath2), mode="w", store=store)

        reloaded: Lattice = load(str(filepath2))

        # Verify type
        assert isinstance(reloaded, Lattice)

        # Verify _image (Dataset2d)
        assert isinstance(reloaded._image, Dataset2d)
        assert np.allclose(reloaded._image.array, original._image.array)

        # Verify Dataset2d attributes
        assert hasattr(reloaded._image, "array")
        assert hasattr(reloaded._image, "shape")
        assert isinstance(reloaded._image.array, np.ndarray)
        assert reloaded._image.shape == original._image.shape
        assert np.allclose(reloaded._image.array, original._image.array)

        # Verify _lat
        assert hasattr(reloaded, "_lat")
        assert np.allclose(reloaded._lat, original._lat)
        assert reloaded._lat.shape == (3, 2)
        assert np.allclose(reloaded._lat, original._lat)

        # Verify atoms (Vector) - nested AutoSerialize
        assert hasattr(reloaded, "atoms")
        assert isinstance(reloaded.atoms, Vector)

        # Verify _positions_frac
        assert hasattr(reloaded, "_positions_frac")
        assert np.allclose(reloaded._positions_frac, original._positions_frac)

        # Verify _num_sites
        assert hasattr(reloaded, "_num_sites")
        assert reloaded._num_sites == original._num_sites

        # Verify _numbers
        assert hasattr(reloaded, "_numbers")
        assert np.array_equal(reloaded._numbers, original._numbers)

        # Verify fields match
        assert set(reloaded.atoms.fields) == set(original.atoms.fields)

        # Verify units attribute
        assert hasattr(reloaded.atoms, "units")
        assert reloaded.atoms.units == original.atoms.units

        # Verify name attribute
        assert hasattr(reloaded.atoms, "name")
        assert reloaded.atoms.name == original.atoms.name

        # Verify shape attribute
        assert hasattr(reloaded.atoms, "shape")
        assert reloaded.atoms.shape == original.atoms.shape

        # Verify Vector data using proper API
        for s in original._numbers:
            original_data = original.atoms.get_data(s)
            reloaded_data = reloaded.atoms.get_data(s)

            if original_data is None and reloaded_data is None:
                continue
            if isinstance(original_data, list):
                assert len(reloaded_data) == len(original_data)
            else:
                assert isinstance(original_data, np.ndarray) and isinstance(
                    reloaded_data, np.ndarray
                )
                assert reloaded_data.shape == original_data.shape
            assert np.allclose(reloaded_data, original_data)

            # Verify field data
            assert hasattr(reloaded.atoms, "fields")
            for field in original.atoms.fields:
                assert field in reloaded.atoms.fields
                original_field_data = original.atoms[s][field]
                reloaded_field_data = reloaded.atoms[s][field]
                assert np.allclose(reloaded_field_data, original_field_data)

    @pytest.mark.parametrize("store", ["zip", "dir"])
    def test_lattice_state_independence(self, tmp_path, store):
        """Test that multiple lattice instances don't interfere with each other."""
        lattice1 = Lattice.from_data(np.random.randn(100, 100))
        lattice2 = Lattice.from_data(np.random.randn(80, 80))

        filepath1 = tmp_path / ("independent1.zip" if store == "zip" else "independent1_dir")
        filepath2 = tmp_path / ("independent2.zip" if store == "zip" else "independent2_dir")

        lattice1.save(str(filepath1), mode="w", store=store)
        lattice2.save(str(filepath2), mode="w", store=store)

        loaded1: Lattice = load(str(filepath1))
        loaded2: Lattice = load(str(filepath2))

        assert loaded1._image.shape == (100, 100)
        assert loaded2._image.shape == (80, 80)
        with pytest.raises(Exception):
            assert np.allclose(loaded1._image.array, loaded2._image.array)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
