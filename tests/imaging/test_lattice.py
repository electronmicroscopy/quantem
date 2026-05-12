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


class TestLatticeAtoms:
    """Test add_atoms method."""

    @pytest.fixture
    def simple_lattice(self):
        """Create a fitted lattice with atoms without defined lattice vectors."""
        # Create synthetic image
        H, W = 100, 100
        image = np.random.randn(H, W) * 0.1

        # Add some peaks with random shifts
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
            # Add random shift up to 3 pixels in each direction
            y_shift = np.random.randint(-3, 4)
            x_shift = np.random.randint(-3, 4)
            y_shifted = y + y_shift
            x_shifted = x + x_shift

            yy, xx = np.ogrid[-10:11, -10:11]
            peak = np.exp(-(xx**2 + yy**2) / 20.0)
            y_start, y_end = max(0, y_shifted - 10), min(H, y_shifted + 11)
            x_start, x_end = max(0, x_shifted - 10), min(W, x_shifted + 11)
            peak_h, peak_w = y_end - y_start, x_end - x_start
            image[y_start:y_end, x_start:x_end] += peak[:peak_h, :peak_w]

        lattice = Lattice.from_data(image)

        return lattice

    @pytest.fixture
    def fitted_lattice(self):
        """Create a fitted lattice with atoms."""
        # Create synthetic image
        H, W = 100, 100
        image = np.random.randn(H, W) * 0.1

        # Add some peaks with random shifts
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
            # Add random shift up to 2 pixels in each direction
            y_shift = np.random.randint(-2, 3)
            x_shift = np.random.randint(-2, 3)
            y_shifted = y + y_shift
            x_shifted = x + x_shift

            yy, xx = np.ogrid[-10:11, -10:11]
            peak = np.exp(-(xx**2 + yy**2) / 20.0)
            y_start, y_end = max(0, y_shifted - 10), min(H, y_shifted + 11)
            x_start, x_end = max(0, x_shifted - 10), min(W, x_shifted + 11)
            peak_h, peak_w = y_end - y_start, x_end - x_start
            image[y_start:y_end, x_start:x_end] += peak[:peak_h, :peak_w]

        lattice = Lattice.from_data(image)

        # Define lattice vectors before adding atoms
        lattice.define_lattice_vectors(
            origin=[23.0, 24.0],
            u=[27.0, 0.0],
            v=[0.0, 25.0],
        )

        return lattice

    def test_add_atoms_basic(self, fitted_lattice: Lattice):
        """Test basic add_atoms."""
        positions_frac = np.array([[0.0, 0.0]])

        result = fitted_lattice.add_atoms(positions_frac)

        assert result is fitted_lattice
        # Check that atoms were added
        assert hasattr(fitted_lattice, "_atoms") or hasattr(fitted_lattice, "atoms")

    def test_add_atoms_plotting(self, fitted_lattice: Lattice):
        """Test add_atoms sets defualt_plot to atoms."""
        positions_frac = np.array([[0.0, 0.0]])

        fitted_lattice.add_atoms(positions_frac)

        assert fitted_lattice.default_plot == "atoms"

    def test_add_atoms_with_all_parameters(self, simple_lattice: Lattice):
        """Test add_atoms with all optional parameters."""
        fitted_lattice = simple_lattice.define_lattice_vectors(
            origin=[25.0, 25.0], u=[50.0, 0.0], v=[0.0, 50.0]
        )
        positions_frac = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        numbers = np.array([0, 1, 1, 2])
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
        )

        assert result is simple_lattice

    def test_add_atoms_empty_positions(self, fitted_lattice: Lattice):
        """Test add_atoms with empty positions array."""
        positions_frac = np.array([]).reshape(0, 2)

        result = fitted_lattice.add_atoms(positions_frac)

        assert result is fitted_lattice

    def test_add_atoms_without_fitting_raises_error(self):
        """Test that add_atoms raises error if lattice not fitted."""
        image = np.random.randn(100, 100)
        lattice = Lattice.from_data(image)

        positions_frac = np.array([[0.0, 0.0]])

        with pytest.raises(ValueError, match="Lattice vectors have not been fitted"):
            lattice.add_atoms(positions_frac)

    def test_refine_atoms_chaining_and_plotting(self, fitted_lattice: Lattice):
        """Test refine_atoms can be called by chaining functions and default_plot is set to atoms."""
        positions_frac = np.array([[0.0, 0.0]])

        result = fitted_lattice.add_atoms(positions_frac).refine_atoms()

        assert result is fitted_lattice
        # Check that atoms were added
        assert hasattr(fitted_lattice, "_atoms") or hasattr(fitted_lattice, "atoms")

        assert fitted_lattice.default_plot == "atoms"

    def test_refine_atoms_without_add_atoms_raises_error(self, fitted_lattice: Lattice):
        """Test that refine_atoms raises error if add_atoms hasn't been called."""
        with pytest.raises(ValueError, match=r"No atoms to refine\. Call add_atoms\(\) first\."):
            fitted_lattice.refine_atoms()

    def test_refine_atoms_internal_call(self, fitted_lattice: Lattice):
        """Test refine_atoms = True parameter in add_atoms"""
        positions_frac = np.array([[0.0, 0.0]])

        result = fitted_lattice.add_atoms(positions_frac, refine_atoms=True)

        assert result is fitted_lattice
        # Check that atoms were added
        assert hasattr(fitted_lattice, "_atoms") or hasattr(fitted_lattice, "atoms")
        assert fitted_lattice.default_plot == "atoms"

    def test_refine_atoms_with_all_parameters(self, simple_lattice: Lattice):
        """Test refine_atoms with all optional parameters."""
        fitted_lattice = simple_lattice.define_lattice_vectors(
            origin=[25.0, 25.0], u=[50.0, 0.0], v=[0.0, 50.0]
        )
        positions_frac = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        numbers = np.array([0, 1, 1, 2])
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
        ).refine_atoms(
            fit_radius=15.0,
            max_nfev=250,
            max_move_px=5.0,
        )

        assert result is simple_lattice

    def test_refine_atoms_internal_call_with_all_parameters(self, simple_lattice: Lattice):
        """Test refine_atoms with all optional parameters."""
        fitted_lattice = simple_lattice.define_lattice_vectors(
            origin=[25.0, 25.0], u=[50.0, 0.0], v=[0.0, 50.0]
        )
        positions_frac = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        numbers = np.array([0, 1, 1, 2])
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
            refine_atoms=True,
            fit_radius=15.0,
            max_nfev=250,
            max_move_px=5.0,
        )

        assert result is simple_lattice


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
