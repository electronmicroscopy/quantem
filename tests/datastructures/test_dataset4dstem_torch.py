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
    def test_construction(self, sample_4d_tensor):
        ds = Dataset4dstemTorch(sample_4d_tensor)
        assert ds.array is sample_4d_tensor  # zero-copy wrap
        assert ds.name == "4dstem (torch)"
        assert ds.units == ["pixels"] * 4
        assert ds.signal_units == "arb. units"
        np.testing.assert_array_equal(ds.sampling, np.ones(4))
        np.testing.assert_array_equal(ds.origin, np.zeros(4))

    def test_metadata_overrides(self, sample_4d_tensor):
        ds = Dataset4dstemTorch(
            sample_4d_tensor,
            name="lamella",
            sampling=(0.5, 0.5, 0.46, 0.46),
            units=["A", "A", "mrad", "mrad"],
            signal_units="counts",
        )
        assert ds.name == "lamella"
        assert ds.units == ["A", "A", "mrad", "mrad"]
        assert ds.signal_units == "counts"
        np.testing.assert_allclose(ds.sampling, [0.5, 0.5, 0.46, 0.46])
