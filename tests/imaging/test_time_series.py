# Tests:
# pad_val with mean, median, max, min
# pad_width as int and tuple
# edge_blend as int and tuple
import numpy as np
import pytest
from scipy.ndimage import shift as ndi_shift

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.imaging.time_series import TimeSeries


@pytest.fixture
def sample_static_stack():
    static = np.random.randint(0, 255, (3, 64, 64))
    frames = Dataset3d.from_array(static)
    return frames

@pytest.fixture
def padded_static_stack(sample_static_stack):
    padded_frames = TimeSeries.pad_and_blend(sample_static_stack, pad_width=(3,10))
    return padded_frames

def shifted_stack(im,shift):
    stack = []
    for s in range(len(shift)):
        print(shift[s])
        stack.append(ndi_shift(im, shift=shift[s]))
        print(stack)
    return np.stack(stack)


class TestPadAndBlendFunction:
    def test_pad_width_int(self, sample_static_stack):
        pad_width = 2
        padded = TimeSeries.pad_and_blend(sample_static_stack, pad_width=pad_width)
        assert padded.shape[1] == sample_static_stack.array.shape[1] + (pad_width * 2)
        assert padded.shape[2] == sample_static_stack.array.shape[2] + (pad_width * 2)

    def test_pad_width_tuple(self, sample_static_stack):
        pad_width = (2,3)
        padded = TimeSeries.pad_and_blend(sample_static_stack, pad_width=pad_width)
        assert padded.shape[1] == sample_static_stack.array.shape[1] + (pad_width[0] * 2)
        assert padded.shape[2] == sample_static_stack.array.shape[2] + (pad_width[1] * 2) 

    def test_pad_width_none(self, sample_static_stack):
        padded = TimeSeries.pad_and_blend(sample_static_stack,pad_width=None)
        assert padded.shape[1] == sample_static_stack.array.shape[1]
        assert padded.shape[2] == sample_static_stack.array.shape[2]


class TestAlignImage:
    def test_pixel_shift(self, padded_static_stack):
        expected_shift = [(0,0),(1,-3)]
        stack = shifted_stack(padded_static_stack[0], expected_shift)
        aligned, xy = TimeSeries.align_image(stack[0],stack[1])
        expected = np.array(expected_shift[1])
        detected = -1*xy
        assert np.allclose(detected, expected, atol=1e-3)

    def test_subpixel_shift(self, padded_static_stack):
        expected_shift = [(0,0),(1.5,-3.5)]
        stack = shifted_stack(padded_static_stack[0], expected_shift)
        aligned, xy = TimeSeries.align_image(stack[0],stack[1])
        expected = np.array(expected_shift[1])
        detected = -1*xy
        assert np.allclose(detected, expected, atol=0.5)


class TestAlignStack:
    def test_pixel_shift(self, padded_static_stack):
        expected_shift = [(0,0),(1,-3),(-2,-2)]
        stack = shifted_stack(padded_static_stack[0], expected_shift)
        aligned, xy = TimeSeries.align_stack(stack)
        expected = np.array(expected_shift)
        detected = -1*xy   
        assert np.allclose(xy[0], 0)
        assert np.allclose(detected, expected, atol=1e-3)

    def test_subpixel_shift(self, padded_static_stack):
        expected_shift = [(0,0),(1.5,-3.5),(-2.5,-4.5)]
        stack = shifted_stack(padded_static_stack[0], expected_shift)
        aligned, xy = TimeSeries.align_stack(stack)
        expected = np.array(expected_shift)
        detected = -1 * xy   
        assert np.allclose(xy[0], 0)
        assert np.allclose(detected, expected, atol=0.5)


    
        
        


