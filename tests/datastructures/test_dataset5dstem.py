"""Tests for Dataset5dstem (quantem.core.datastructures.dataset5dstem)."""

import numpy as np
import torch

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.dataset5dstem import Dataset5dstem


def test_from_tensor():
    ds = Dataset5dstem.from_tensor(
        tensor=torch.rand(3, 5, 5, 8, 8),
        name="t",
        sampling=(0.5, 0.5, 0.1, 0.1),
        units=["nm", "nm", "1/nm", "1/nm"],
        series_type="time",
        series=[0.0, 2.0, 4.0],
    )
    assert ds.shape == (3, 5, 5, 8, 8)
    assert np.array_equal(ds.sampling, np.array([0.5, 0.5, 0.1, 0.1]))
    assert ds.units == ["nm", "nm", "1/nm", "1/nm"]
    assert ds.series_type == "time"
    assert np.array_equal(ds.series, np.array([0.0, 2.0, 4.0]))
    assert isinstance(ds[0], Dataset4dstem)


def test_slice():
    """A scientist slices a sub-stack - gets a smaller Dataset5dstem with series sliced."""
    ds = Dataset5dstem.from_tensor(
        tensor=torch.rand(5, 5, 5, 8, 8),
        series_type="time", series=[0.0, 1.0, 2.0, 3.0, 4.0],
    )
    sub = ds[1:4]
    assert isinstance(sub, Dataset5dstem)
    assert sub.shape == (3, 5, 5, 8, 8)
    assert np.array_equal(sub.series, np.array([1.0, 2.0, 3.0]))


def test_for_loop():
    """A scientist loops frame-by-frame - each yield is a Dataset4dstem."""
    ds = Dataset5dstem.from_tensor(tensor=torch.rand(3, 5, 5, 8, 8))
    seen = [f for f in ds]
    assert len(seen) == 3
    assert all(isinstance(f, Dataset4dstem) and f.shape == (5, 5, 8, 8) for f in seen)


def test_from_4dstem():
    d4_list = [
        Dataset4dstem.from_tensor(
            torch.rand(5, 5, 8, 8),
            sampling=(0.5, 0.5, 0.1, 0.1),
            units=("nm", "nm", "1/nm", "1/nm"),
            name=f"f{i}",
        )
        for i in range(3)
    ]
    ds = Dataset5dstem.from_4dstem(d4_list, series_type="tilt", series=[-30, 0, 30])
    assert ds.shape == (3, 5, 5, 8, 8)
    assert np.array_equal(ds.sampling, np.array([0.5, 0.5, 0.1, 0.1]))
    assert ds.units == ["nm", "nm", "1/nm", "1/nm"]
    assert ds.series_type == "tilt"
    assert np.array_equal(ds.series, np.array([-30.0, 0.0, 30.0]))
