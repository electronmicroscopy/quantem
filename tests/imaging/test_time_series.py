import numpy as np
import pytest
from scipy.ndimage import shift

from quantem.imaging.time_series import TimeSeries


@pytest.fixture
def shifted_stack():
    """
    Create a stack where each frame is shifted by (dy, dx) pixels.
    """
    t, r, c = 6, 64, 64
    stack = np.zeros((t, r, c))

    base = np.zeros((r, c))
    base[20:40, 20:40] = 1.0

    return stack, base


def test_from_array_creates_timeseries(shifted_stack):
    stack, base = shifted_stack
    true_shifts = [(0, 0), (1, 2), (2, 4), (3, 6), (4, 8), (5, 10)]
    pad = (3, 5)
    pad_val = 'mean'
    t, r, c = stack.shape

    for i, (dr, dc) in enumerate(true_shifts):
        stack[i] = shift(base, shift=(dr,dc), order=3)
    
    ts = TimeSeries.from_array(
        data = stack,
        pad_shape = pad,
        pad_val = pad_val,
        blend_shape = 2,
    )

    assert isinstance(ts, TimeSeries)
    assert isinstance(ts.data, object)
    assert isinstance(ts.array,np.ndarray)
    assert ts.array.ndim == 3
    assert ts.shape == ts.array.shape
    assert ts.n_frames == ts.shape[0]
    assert ts.orig_shape == stack.shape
    assert ts.shape == (t, r + 2*pad[0], c + 2*pad[1])    
    assert ts.pad_val == pytest.approx(np.mean(stack))


def test_align_stack_pixel_shift(shifted_stack):
    stack, base = shifted_stack
    true_shifts = [(0, 0), (1, 2), (2, 4), (3, 6), (4, 8), (5, 10)]

    for i, (dr, dc) in enumerate(true_shifts):
        stack[i] = shift(base, shift=(dr,dc), order=3)

    ts = TimeSeries.from_array(
        data = stack,
        pad_shape = 8,
        pad_val = 0.0,
        blend_shape = 4,
    )

    ts.align_stack(
        running_average_frames = 10,
        correlation_power = 1.0,
        sigma_edge = 1,
        sf_val = 1,
    )

    assert hasattr(ts, "_align_coords")
    assert hasattr(ts, "_align_im")

    recovered = ts.align_coords
    error = recovered[1:] - -1*np.array(true_shifts, dtype=float)[1:]

    assert np.all(np.abs(error) < 0.5)


def test_align_stack_recovers_subpixel_shift(shifted_stack):
    stack, base = shifted_stack
    true_shifts = [(0, 0), (1.1, 2.3), (2, 4.5), (3.4, 6.4), (4.2, 8), (5.7, 10.9)]

    for i, (dr, dc) in enumerate(true_shifts):
        stack[i] = shift(base, shift=(dr,dc), order=3)

    ts = TimeSeries.from_array(
        data = stack,
        pad_shape = 8,
        pad_val = 0.0,
        blend_shape = 4,
    )

    ts.align_stack(
        running_average_frames = 10,
        correlation_power = 1.0,
        sigma_edge = 1,
        sf_val = 1,
    )

    assert hasattr(ts, "_align_coords")
    assert hasattr(ts, "_align_im")

    recovered = ts.align_coords
    error = np.array(recovered, dtype = float)[1:] - -1*np.array(true_shifts, dtype = float)[1:]

    assert np.all(np.abs(error) < 0.05)
