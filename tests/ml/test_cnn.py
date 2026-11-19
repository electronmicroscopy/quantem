"""CNN2d conv_kernel_size sets every conv kernel and keeps the image size."""

import pytest
import torch

from quantem.core.ml.cnn import CNN2d


def test_conv_kernel_size_sets_kernels_and_keeps_image_size():
    for k in (3, 5):
        model = CNN2d(in_channels=1, num_layers=1, start_filters=4, conv_kernel_size=k)
        out = model(torch.rand(1, 1, 16, 16))
        assert out.shape == (1, 1, 16, 16), f"k={k}: output shape {tuple(out.shape)}"
        first = model.down_conv_blocks[0].block[0].weight.shape[-2:]
        final = model.final_conv.block[0].weight.shape[-2:]
        assert first == (k, k), f"k={k}: first conv kernel {tuple(first)}"
        assert final == (k, k), f"k={k}: output conv kernel {tuple(final)}"

    for bad in (4, 0, -1, 3.0, "3"):
        with pytest.raises(ValueError, match="positive and odd"):
            CNN2d(in_channels=1, num_layers=1, start_filters=4, conv_kernel_size=bad)
