"""Tests for read_5dstem with mocked Nion Swift metadata."""
import json
import numpy as np
import pytest
import h5py
from quantem.core.io import read_5dstem
from quantem.core.datastructures import Dataset5dstem


class TestRead5dstemNionSwift:
    """Tests for reading Nion Swift 5D-STEM data."""

    def test_read_nion_5dstem(self, mock_nion_5dstem_file):
        """Test reading Nion Swift file extracts shape, type, and calibrations."""
        data = read_5dstem(mock_nion_5dstem_file)
        assert isinstance(data, Dataset5dstem)
        assert data.shape == (4, 6, 7, 3, 5)
        assert data.stack_type == "time"  # is_sequence=True
        assert np.allclose(data.sampling, [1.0, 0.5, 0.5, 0.006, 0.006])
        assert np.allclose(data.origin[:3], [0.0, -1.0, -1.5])
        assert data.units[1:3] == ["nm", "nm"]
        assert data.units[3:] == ["rad", "rad"]
        assert data.signal_units == "counts"

    def test_read_override_stack_type(self, mock_nion_5dstem_file):
        """Test stack_type can be overridden."""
        data = read_5dstem(mock_nion_5dstem_file, stack_type="tilt")
        assert data.stack_type == "tilt"

    def test_read_no_sequence(self, tmp_path):
        """Test is_sequence=False -> stack_type='generic', empty units -> 'arb. units'."""
        path = tmp_path / "no_seq.h5"
        properties = {
            "is_sequence": False,
            "dimensional_calibrations": [{"offset": 0, "scale": 1, "units": ""}] * 5,
            "intensity_calibration": {"units": ""},
            "metadata": {},
        }
        with h5py.File(path, "w") as f:
            dset = f.create_dataset("data", data=np.zeros((2, 3, 3, 4, 4), dtype=np.float32))
            dset.attrs["properties"] = json.dumps(properties)
        data = read_5dstem(path)
        assert data.stack_type == "generic"
        assert data.signal_units == "arb. units"
