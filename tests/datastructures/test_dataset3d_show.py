"""Tests for Dataset3d.show() frame selection logic."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from quantem.core.datastructures import Dataset3d


@pytest.fixture
def dataset_with_10_frames():
    """Create a Dataset3d with 10 frames for testing."""
    return Dataset3d.from_array(np.random.rand(10, 8, 8))


@pytest.fixture
def dataset_with_25_frames():
    """Create a Dataset3d with more frames than the default max of 20."""
    return Dataset3d.from_array(np.random.rand(25, 8, 8))


def extract_frame_indices_from_figure(fig):
    """Extract frame indices from subplot titles like 'Frame 0', 'Frame 1', etc."""
    return [
        int(ax.get_title().split()[-1])
        for ax in fig.get_axes()
        if ax.get_title().startswith("Frame ")
    ]


def assert_shows_frames(dataset, cases):
    for kwargs, expected_indices in cases:
        fig, _ = dataset.show(returnfig=True, **kwargs)
        assert extract_frame_indices_from_figure(fig) == expected_indices, kwargs
        plt.close(fig)


class TestShowInputValidation:
    """Test that invalid inputs raise clear errors."""

    def test_raises_value_error(self, dataset_with_10_frames):
        for kwargs, match in [
            ({"step": 0}, "cannot be zero"),
            ({"start": 100}, "out of bounds"),
            ({"start": -100}, "out of bounds"),
            ({"start": 5, "end": 5}, "No frames to display"),
            ({"ncols": 0}, "ncols must be >= 1"),
            ({"max": 0}, "max must be >= 1"),
        ]:
            with pytest.raises(ValueError, match=match):
                dataset_with_10_frames.show(**kwargs)


class TestShowFrameSelection:
    """Test frame selection with start, end, step, max combinations."""

    def test_large_dataset(self, dataset_with_25_frames):
        assert_shows_frames(
            dataset_with_25_frames,
            [
                ({}, list(range(20))),
                ({"max": 5}, [0, 1, 2, 3, 4]),
                ({"max": None}, list(range(25))),
                ({"start": 20}, list(range(20, 25))),
                ({"start": 20, "max": 3}, [20, 21, 22]),
            ],
        )

    def test_small_dataset(self, dataset_with_10_frames):
        assert_shows_frames(
            dataset_with_10_frames,
            [
                # Default shows all frames (< max)
                ({}, list(range(10))),
                # Start and end
                ({"start": 5}, [5, 6, 7, 8, 9]),
                ({"end": 5}, [0, 1, 2, 3, 4]),
                # Step
                ({"step": 3}, [0, 3, 6, 9]),
                ({"start": 2, "end": 8, "step": 2}, [2, 4, 6]),
                # Negative start index
                ({"start": -1, "max": 1}, [9]),
                ({"start": -3, "max": 2}, [7, 8]),
                # Negative step (reverse order)
                ({"start": 9, "step": -1}, [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
                ({"start": 9, "end": 4, "step": -1}, [9, 8, 7, 6, 5]),
                ({"start": 9, "step": -2}, [9, 7, 5, 3, 1]),
                ({"start": 9, "step": -1, "max": 3}, [9, 8, 7]),
            ],
        )
