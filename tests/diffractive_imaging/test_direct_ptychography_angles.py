"""
Unit tests for DirectPtychography angle parameter handling.

Tests the new API that allows users to specify rotation angles in either
degrees or radians for improved usability.
"""

import numpy as np
import pytest

from quantem.diffractive_imaging.direct_ptycho_utils import process_angle_parameters


class TestAngleParameters:
    """Test rotation angle parameter handling in DirectPtychography."""

    def test_process_angle_parameters_rad_only(self):
        """Test that radians input is returned unchanged."""
        angle_rad = np.pi / 4  # 45 degrees
        result = process_angle_parameters(rotation_angle_rad=angle_rad)
        assert result == angle_rad

    def test_process_angle_parameters_deg_only(self):
        """Test that degrees input is converted to radians."""
        angle_deg = 45.0
        expected_rad = np.deg2rad(45.0)
        result = process_angle_parameters(rotation_angle_deg=angle_deg)
        assert np.isclose(result, expected_rad)

    def test_process_angle_parameters_both_raises_error(self):
        """Test that providing both rad and deg raises ValueError."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            process_angle_parameters(
                rotation_angle_rad=np.pi / 4, rotation_angle_deg=45.0
            )

    def test_process_angle_parameters_neither_returns_none(self):
        """Test that providing neither returns None."""
        result = process_angle_parameters()
        assert result is None

    def test_negative_angles(self):
        """Test that negative angles work correctly (common in experiments)."""
        angle_deg = -169.0
        expected_rad = np.deg2rad(-169.0)
        result = process_angle_parameters(rotation_angle_deg=angle_deg)
        assert np.isclose(result, expected_rad)

    def test_conversion_accuracy(self):
        """Test various angle conversions for accuracy."""
        test_angles = [0, 30, 45, 90, 180, -45, -90, -180, 360]
        for angle_deg in test_angles:
            result = process_angle_parameters(rotation_angle_deg=angle_deg)
            expected = np.deg2rad(angle_deg)
            assert np.isclose(result, expected), f"Failed for {angle_deg} degrees"

    def test_zero_angle(self):
        """Test that zero angle works in both formats."""
        result_deg = process_angle_parameters(rotation_angle_deg=0.0)
        result_rad = process_angle_parameters(rotation_angle_rad=0.0)
        assert result_deg == 0.0
        assert result_rad == 0.0

    def test_full_rotation_equivalence(self):
        """Test that 360 degrees equals 2*pi radians."""
        result_deg = process_angle_parameters(rotation_angle_deg=360.0)
        result_rad = process_angle_parameters(rotation_angle_rad=2 * np.pi)
        assert np.isclose(result_deg, result_rad)

    def test_optimization_parameter_passthrough(self):
        """Test that OptimizationParameter objects are passed through unchanged."""
        # Create a mock OptimizationParameter-like object
        class MockOptParam:
            def __init__(self, low, high):
                self.low = low
                self.high = high
        
        opt_param_deg = MockOptParam(-180, -160)
        result = process_angle_parameters(rotation_angle_deg=opt_param_deg)
        
        # Should return the same object, not try to convert it
        assert result is opt_param_deg
        assert result.low == -180
        assert result.high == -160


# Note: To test the DirectPtychography.rotation_angle_rad and rotation_angle_deg
# properties, we would need to create a full DirectPtychography instance with
# actual data, which is beyond the scope of these unit tests. Those properties
# are tested implicitly through integration tests with real datasets.
