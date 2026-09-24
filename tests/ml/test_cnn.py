"""Tests for the CNN architectures."""

import pytest
import torch

from quantem.core.ml import CNN2d, CNN3d


@pytest.mark.parametrize("shape", [(1, 16, 32, 32), (1, 15, 30, 33), (2, 1, 7, 9, 12)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_cnn3d_shape_not_divisible_by_pooling(shape, dtype):
    in_channels = shape[-4]
    net = CNN3d(in_channels=in_channels, dtype=dtype, start_filters=4)
    out = net(torch.randn(shape, dtype=dtype))
    assert out.shape == shape


@pytest.mark.parametrize("shape", [(1, 3, 32, 32), (1, 15, 30, 33), (2, 1, 9, 12)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_cnn2d_shape_not_divisible_by_pooling(shape, dtype):
    net = CNN2d(in_channels=shape[1], dtype=dtype, start_filters=4)
    out = net(torch.randn(shape, dtype=dtype))
    assert out.shape == shape
