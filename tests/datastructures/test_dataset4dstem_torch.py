"""
Tests for the Dataset4dstemTorch class in quantem.core.datastructures.dataset4dstem_torch
"""

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset4dstemTorch


@pytest.fixture
def sample_4d_tensor():
    return torch.rand(5, 5, 10, 10)


class TestDataset4dstemTorch:
    def test_from_array(self, sample_4d_tensor):
        ds = Dataset4dstemTorch.from_array(
            sample_4d_tensor,
            name="lamella",
            sampling=(0.5, 0.5, 0.46, 0.46),
            units=["A", "A", "mrad", "mrad"],
            signal_units="counts",
        )
        assert ds.array is sample_4d_tensor
        assert ds.name == "lamella"
        assert ds.units == ["A", "A", "mrad", "mrad"]
        assert ds.signal_units == "counts"
        np.testing.assert_allclose(ds.sampling, [0.5, 0.5, 0.46, 0.46])

    def test_direct_init_blocked(self, sample_4d_tensor):
        with pytest.raises(RuntimeError, match="from_array"):
            Dataset4dstemTorch(sample_4d_tensor)
