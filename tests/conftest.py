import json
from pathlib import Path

import h5py
import numpy as np
import pytest


def create_mock_nion_swift_h5(path: Path, shape=(4, 6, 7, 3, 5), is_sequence=True):
    """Create a mock Nion Swift h5 file with realistic metadata structure.

    Parameters
    ----------
    path : Path
        Output file path.
    shape : tuple
        Data shape (frames, scan_row, scan_col, k_row, k_col).
    is_sequence : bool
        Whether to set is_sequence=True in properties.
    """
    properties = {
        "type": "data-item",
        "uuid": "test-uuid",
        "is_sequence": is_sequence,
        "intensity_calibration": {"offset": 0.0, "scale": 1.0, "units": "counts" if is_sequence else ""},
        "dimensional_calibrations": [
            {"offset": 0.0, "scale": 1.0, "units": ""},  # frames
            {"offset": -1.0, "scale": 0.5, "units": "nm"},  # scan_row
            {"offset": -1.5, "scale": 0.5, "units": "nm"},  # scan_col
            {"offset": -0.036, "scale": 0.006, "units": "rad"},  # k_row
            {"offset": -0.036, "scale": 0.006, "units": "rad"},  # k_col
        ],
        "collection_dimension_count": 2,
        "datum_dimension_count": 2,
        "metadata": {
            "instrument": {
                "high_tension": 60000.0,  # 60 keV
                "defocus": 0.0,
                "ImageScanned": {
                    "probe_ha": 0.035,  # 35 mrad half-angle
                    "C10": 0.0,
                    "C30": 0.0,
                },
            },
            "scan": {
                "scan_size": [shape[1], shape[2]],
            },
        },
    }

    with h5py.File(path, "w") as f:
        data = np.random.rand(*shape).astype(np.float32)
        dset = f.create_dataset("data", data=data)
        dset.attrs["properties"] = json.dumps(properties)


@pytest.fixture
def mock_nion_5dstem_file(tmp_path):
    """Create a temporary mock Nion Swift 5D-STEM h5 file."""
    path = tmp_path / "mock_nion_5dstem.h5"
    create_mock_nion_swift_h5(path, shape=(4, 6, 7, 3, 5))
    return path


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
