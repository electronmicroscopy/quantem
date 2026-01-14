"""Tests for quantem.hpc.io module."""

import numpy as np
import h5py
import hdf5plugin
import pytest


def _gpu_available():
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False

pytestmark = pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU not available")


@pytest.fixture
def mock_h5_file(tmp_path):
    """Create a temporary HDF5 file with bitshuffle+LZ4 compressed data."""
    filepath = tmp_path / "test_data.h5"
    n_frames, height, width = 100, 192, 192
    rng = np.random.default_rng(42)
    data = rng.integers(0, 1000, size=(n_frames, height, width), dtype=np.uint32)
    with h5py.File(filepath, "w") as f:
        f.create_dataset(
            "entry/data/data",
            data=data,
            chunks=(1, height, width),
            **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
    return filepath, data


def test_load_and_bin(mock_h5_file):
    """Test GPU load and binning with sum/mean reductions."""
    import cupy as cp
    from quantem.hpc.io import load, bin
    filepath, original = mock_h5_file
    data = load(str(filepath))
    # Verify load
    assert data.shape == original.shape
    assert np.array_equal(cp.asnumpy(data), original)
    # Verify bin sum (factor=2)
    binned_sum = bin(data, factor=2, reduction="sum")
    assert binned_sum.shape == (100, 96, 96)
    assert np.isclose(float(data.sum()), float(binned_sum.sum()), rtol=1e-5)
    # Verify bin mean (factor=2)
    binned_mean = bin(data, factor=2, reduction="mean")
    assert binned_mean.shape == (100, 96, 96)
    assert np.isclose(float(data.mean()), float(binned_mean.mean()), rtol=1e-5)
