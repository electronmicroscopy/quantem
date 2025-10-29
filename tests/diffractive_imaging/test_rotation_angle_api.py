"""
Test the unified rotation_angle_rad/rotation_angle_deg API across direct and indirect ptychography.
"""

import numpy as np
import pytest


def test_process_angle_parameters_in_dataset():
    """Test that rotation angle parameters work in dataset preprocessing."""
    from quantem.diffractive_imaging.direct_ptycho_utils import process_angle_parameters
    
    # Test radians
    angle_rad = np.pi / 4
    result = process_angle_parameters(rotation_angle_rad=angle_rad)
    assert np.isclose(result, angle_rad)
    
    # Test degrees
    angle_deg = 45.0
    expected_rad = np.deg2rad(45.0)
    result = process_angle_parameters(rotation_angle_deg=angle_deg)
    assert np.isclose(result, expected_rad)
    
    # Test mutual exclusivity
    with pytest.raises(ValueError, match="Cannot specify both"):
        process_angle_parameters(rotation_angle_rad=np.pi/4, rotation_angle_deg=45.0)
    
    # Test neither returns None
    result = process_angle_parameters()
    assert result is None


def test_negative_angles():
    """Test that negative angles work correctly."""
    from quantem.diffractive_imaging.direct_ptycho_utils import process_angle_parameters
    
    angle_deg = -169.0
    expected_rad = np.deg2rad(-169.0)
    result = process_angle_parameters(rotation_angle_deg=angle_deg)
    assert np.isclose(result, expected_rad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
