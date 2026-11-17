"""Tests for the CNN architectures."""

import torch

from quantem.core.ml import CNN2d, CNN3d


def test_cnn2d_output_shape_matches_input():
    cases = {
        "divisible, real": ((1, 3, 32, 32), torch.float32),
        "padded, complex": ((2, 1, 9, 12), torch.complex64),
    }
    for name, (shape, dtype) in cases.items():
        net = CNN2d(in_channels=shape[1], dtype=dtype, start_filters=4)
        out = net(torch.randn(shape, dtype=dtype))
        assert out.shape == shape, name


def test_cnn3d_output_shape_matches_input():
    cases = {
        "unbatched, divisible, real": ((1, 16, 32, 32), torch.float32),
        "batched, padded, complex": ((2, 1, 7, 9, 12), torch.complex64),
    }
    for name, (shape, dtype) in cases.items():
        net = CNN3d(in_channels=shape[-4], dtype=dtype, start_filters=4)
        out = net(torch.randn(shape, dtype=dtype))
        assert out.shape == shape, name
