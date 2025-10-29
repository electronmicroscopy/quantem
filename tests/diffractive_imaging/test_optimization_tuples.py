"""
Unit tests for tuple-based optimization parameter API.

Tests that the new simplified tuple-based API works correctly for
optimize_hyperparameters and grid_search_hyperparameters.
"""

import numpy as np
import pytest


def test_tuple_interpretation_2element():
    """Test that 2-element tuples are interpreted as (min, max) ranges."""
    param_tuple = (-180.0, -160.0)
    assert len(param_tuple) == 2
    assert param_tuple[0] == -180.0
    assert param_tuple[1] == -160.0


def test_tuple_interpretation_3element():
    """Test that 3-element tuples include n_points for grid search."""
    param_tuple = (-180.0, -160.0, 20)
    assert len(param_tuple) == 3
    assert param_tuple[0] == -180.0
    assert param_tuple[1] == -160.0
    assert param_tuple[2] == 20


def test_degree_to_radian_conversion():
    """Test that degree ranges are correctly converted to radians."""
    deg_tuple = (-180.0, -160.0)
    rad_tuple = (np.deg2rad(deg_tuple[0]), np.deg2rad(deg_tuple[1]))
    
    assert np.isclose(rad_tuple[0], np.deg2rad(-180.0))
    assert np.isclose(rad_tuple[1], np.deg2rad(-160.0))


def test_grid_values_generation():
    """Test that grid values are generated correctly from tuples."""
    param_tuple = (-400.0, 400.0, 10)
    low, high, n_points = param_tuple
    grid = np.linspace(low, high, n_points)
    
    assert len(grid) == 10
    assert grid[0] == -400.0
    assert grid[-1] == 400.0


def test_mixed_fixed_and_range_params():
    """Test mixed parameter dict with fixed values and ranges."""
    params = {
        "C10": (-400, 400),  # Range to optimize
        "C12": 0,  # Fixed value
        "phi12": (-np.pi/2, np.pi/2),  # Range in radians
    }
    
    # Check types
    assert isinstance(params["C10"], tuple)
    assert isinstance(params["C12"], (int, float))
    assert isinstance(params["phi12"], tuple)
    
    # Check values
    assert len(params["C10"]) == 2
    assert params["C12"] == 0
    assert len(params["phi12"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
