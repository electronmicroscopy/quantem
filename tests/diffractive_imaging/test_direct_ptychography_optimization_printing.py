"""Tests for optimization parameter printing in DirectPtychography."""

import numpy as np
import pytest

from quantem.diffractive_imaging.direct_ptychography import DirectPtychography
from quantem.core.datastructures import Dataset4d


@pytest.fixture
def mock_dataset4d():
    """Create a minimal mock Dataset4d for testing."""
    data = np.ones((4, 4, 8, 8))
    for i in range(4):
        for j in range(4):
            data[i, j] *= (i + j + 1)
    return Dataset4d.from_array(
            array=data,
            name="mock_dataset",
            origin=(0, 0, 0, 0),
            sampling=(1, 1, 1, 1),
            units=("A", "A", "mrad", "mrad"), 
        )

@pytest.fixture
def direct_ptycho(mock_dataset4d):
    """Create a DirectPtychography instance with standard test parameters.
    
    Return a factory function that can create instances with different rotation angles.
    """
    def _create(*, rotation_angle_rad=None, rotation_angle_deg=None):
        return DirectPtychography.from_dataset4d(
            mock_dataset4d,
            energy=300e3,
            semiangle_cutoff=20,
            rotation_angle_rad=rotation_angle_rad,
            rotation_angle_deg=rotation_angle_deg,
            aberration_coefs={"C10": 0, "C12": 0, "phi12": 0},
        )
    return _create

class TestOptimizationParameterPrinting:
    """Test that optimization results are printed in the same units used for input."""

    def test_optimize_hyperparameters_prints_degrees(self, direct_ptycho, capsys):
        """Test that results are printed in degrees when optimizing in degrees."""
        ptycho = direct_ptycho(rotation_angle_deg=90)
        ptycho.optimize_hyperparameters(
            aberration_coefs={"C10": (-10, 10)},
            rotation_angle_deg=(-180, -160),
            n_trials=1,  # Minimal trials for test
        )
        captured = capsys.readouterr()
        assert "rotation_angle_deg" in captured.out
        assert "rotation_angle_rad" not in captured.out

    def test_optimize_hyperparameters_prints_radians(self, direct_ptycho, capsys):
        """Test that results are printed in radians when optimizing in radians."""
        ptycho = direct_ptycho(rotation_angle_rad=np.pi/2)
        ptycho.optimize_hyperparameters(
            aberration_coefs={"C10": (-10, 10)},
            rotation_angle_rad=(-np.pi, -np.pi/2),
            n_trials=1,  # Minimal trials for test
        )
        captured = capsys.readouterr()
        assert "rotation_angle_rad" in captured.out
        assert "rotation_angle_deg" not in captured.out

    def test_grid_search_prints_degrees(self, direct_ptycho, capsys):
        """Test that grid search results are printed in degrees when using degrees."""
        ptycho = direct_ptycho(rotation_angle_deg=90)
        ptycho.verbose = True
        ptycho.grid_search_hyperparameters(
            aberration_coefs={"C10": (-10, 10, 2)},
            rotation_angle_deg=(-180, -160, 2)
        )
        captured = capsys.readouterr()
        assert "rotation_angle_deg" in captured.out
        assert "rotation_angle_rad" not in captured.out

    def test_grid_search_prints_radians(self, direct_ptycho, capsys):
        """Test that grid search results are printed in radians when using radians."""
        ptycho = direct_ptycho(rotation_angle_rad=np.pi/2)
        ptycho.verbose = True
        ptycho.grid_search_hyperparameters(
            aberration_coefs={"C10": (-10, 10, 2)},
            rotation_angle_rad=(-np.pi, -np.pi/2, 2)
        )
        captured = capsys.readouterr()
        assert "rotation_angle_rad" in captured.out
        assert "rotation_angle_deg" not in captured.out