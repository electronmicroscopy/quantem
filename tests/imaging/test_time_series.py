# Tests:
# pad_val with mean, median, max, min
# pad_width as int and tuple
# edge_blend as int and tuple
import numpy as np
import pytest
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.imaging.time_series import TimeSeries

@pytest.fixture
def sample_image():
    im_ref = np.zeros((10,10))
    im = np.zeros((10,10))
    im_ref[3,3] = 1
    im[2:4,5:7] = 1
    return im_ref, im


@pytest.fixture
def sample_static_stack():
    static = np.random.randint(0, 255, (10, 64, 64))
    frames = Dataset3d.from_array(static)
    return frames


@pytest.fixture
def sample_pixel_stack():
    im0 = np.zeros((10,10))
    im1 = np.zeros((10,10))
    im2 = np.zeros((10,10))
    im0[3,3] = 1
    im1[2:4,5:7] = 1
    im2[5,4] = 1

    im_stack = np.zeros((3,10,10))
    im_stack[0][:][:] = im0
    im_stack[1][:][:] = im1
    im_stack[2][:][:] = im2

    return im_stack


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

#     def test_pad_val_mean:
    

class TestAlignImage:
    def test_coord_shift(self, sample_pixel_stack):
        aligned, xy = TimeSeries.align_stack(sample_pixel_stack)
        expected_shift = [(0,0),(0.5,-2.5),(-2,-1)]
        assert np.allclose(xy[0], 0)
        assert np.allclose(xy, expected_shift, atol = 0.5)


# class TestAlignStack: