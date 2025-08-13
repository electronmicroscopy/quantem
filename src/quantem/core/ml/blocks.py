from typing import TYPE_CHECKING, Callable

import numpy as np

from quantem.core import config

from .activation_functions import Complex_ReLU

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:
    if config.get("has_torch"):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F


def complex_pool(z, m, **kwargs):
    return m(z.real) + 1.0j * m(z.imag)


def passfunc(z, m):
    return m(z)


def reset_weights(m: nn.Module) -> None:
    """
    Reset the weights of a given module.

    Args:
        m: The neural network module to reset.
    """
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        reset_parameters()


class Conv2dBlock(nn.Module):
    """
    Creates block(s) consisting of convolutional
    layer, leaky relu and (optionally) dropout and
    batch normalization
    """

    def __init__(
        self,
        nb_layers,
        input_channels: int | list | tuple,
        output_channels: int | list | tuple,
        kernel_size=3,
        stride=1,
        padding=1,
        use_batchnorm=False,
        dropout: float = 0.0,
        dtype=torch.float32,
        activation: Callable | None = None,
    ):
        """Initializes module nn.Parameters"""
        super().__init__()

        if not isinstance(input_channels, (int, float)):
            assert isinstance(input_channels, (list, tuple, np.ndarray, torch.Tensor))
            assert isinstance(output_channels, (list, tuple, np.ndarray, torch.Tensor))
            assert len(input_channels) == len(output_channels) == nb_layers
            input_channels_list = input_channels
            output_channels_list = output_channels
            for a0 in range(len(input_channels_list) - 1):
                assert output_channels_list[a0] == input_channels_list[a0 + 1]
        else:
            assert isinstance(output_channels, (int, float)), f"output channels: {output_channels}"
            input_channels_list = [int(input_channels)] + (nb_layers - 1) * [int(output_channels)]
            output_channels_list = nb_layers * [int(output_channels)]

        self.dtype = dtype
        if dtype.is_complex:
            self.bn = ComplexBatchNorm2D
        else:
            self.bn = nn.BatchNorm2d

        if activation is None:  # defaults to ReLU
            if dtype == torch.complex64:
                activation = Complex_ReLU()
            else:
                activation = nn.ReLU()

        block = []
        for idx in range(nb_layers):
            block.append(
                nn.Conv2d(
                    input_channels_list[idx],
                    output_channels_list[idx],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dtype=dtype,
                    padding_mode="circular",
                )
            )
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            block.append(activation)
            if use_batchnorm:
                block.append(self.bn(output_channels_list[idx]))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        """Forward path"""
        output = self.block(x)
        return output


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        linear = nn.Linear(in_features, out_features, dtype=torch.cfloat)
        self.weight = nn.Parameter(torch.view_as_real(linear.weight))
        self.bias = nn.Parameter(torch.view_as_real(linear.bias))

    def forward(self, x):
        weight = torch.view_as_complex(self.weight)
        bias = torch.view_as_complex(self.bias)
        return F.linear(x, weight, bias)


class Upsample2dBlock(nn.Module):
    """
    Upsampling block using interpolation followed by a convolution.

    Args:
        input_channels: int, number of input channels.
        output_channels: int, number of output channels.
        scale_factor: int, factor by which to scale the input.
        mode: str, interpolation mode, either "bilinear" or "nearest".
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        use_batchnorm: bool = False,
        dtype: "torch.dtype" = torch.float32,
        scale_factor: int = 2,
        mode: str = "bilinear",
    ):
        super().__init__()
        assert mode in ["bilinear", "nearest"], "Mode must be 'bilinear' or 'nearest'."
        self.scale_factor = scale_factor
        self.mode = mode
        self.use_batchnorm = use_batchnorm
        self.dtype = dtype
        self.upsample2x = nn.ConvTranspose2d(
            input_channels,
            input_channels,
            kernel_size=3,
            stride=2,
            padding=(1, 1),
            output_padding=(1, 1),
            dtype=self.dtype,
        )
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=self.dtype,
            padding_mode="circular",
        )
        if self.dtype.is_complex:
            self.bn = ComplexBatchNorm2D(output_channels)
        else:
            self.bn = nn.BatchNorm2d(output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the upsample_block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after upsampling and convolution.
        """
        if self.scale_factor == 2:
            x = self.upsample2x(x)
        else:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        return x


class ComplexBatchNorm2D(nn.Module):
    """
    Batch normalization for complex inputs (real and imaginary parts separately).
    """

    def __init__(self, num_features):
        super(ComplexBatchNorm2D, self).__init__()
        self.real_bn = nn.BatchNorm2d(num_features)
        self.imag_bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        real = x.real
        imag = x.imag
        real_out = self.real_bn(real)
        imag_out = self.imag_bn(imag)
        return torch.complex(real_out, imag_out)


class ComplexNormalize(nn.Module):
    def __init__(self, mean, std, inplace=True):
        """
        Mean shape (C, 2) (real, complex)
        std shape (C, 2) (real, complex)
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.inplace:
            ### raises warning, but this is copied directly from the torchvision documentation
            tensor = tensor.clone()

        view = torch.view_as_real(tensor)

        view[..., 0] = (
            view[..., 0].sub(self.mean[:, None, None, 0]).div_(self.std[:, None, None, 0])
        )
        view[..., 1] = (
            view[..., 1].sub(self.mean[:, None, None, 1]).div_(self.std[:, None, None, 1])
        )

        return torch.view_as_complex(view)

    def to_device(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def __repr__(self):
        return f"Mean (C,2) (R,C):\n{self.mean}\nStd:\n{self.std}"


class Conv3dBlock(nn.Module):
    def __init__(
        self,
        nb_layers,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        use_batchnorm=False,
        dropout=0.0,
        dtype=torch.float32,
        activation: None | Callable = None,
    ):
        super().__init__()
        self.dtype = dtype
        self.bn = ComplexBatchNorm3D if dtype.is_complex else nn.BatchNorm3d
        activation_func = activation

        layers = []
        for _ in range(nb_layers):
            layers.append(
                nn.Conv3d(
                    input_channels,
                    output_channels,
                    kernel_size,
                    stride,
                    padding,
                    dtype=dtype,
                    padding_mode="circular",
                )
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(activation_func)
            if use_batchnorm:
                layers.append(self.bn(output_channels))
            input_channels = output_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Upsample3dBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        use_batchnorm=False,
        dtype=torch.float32,
        scale_factor=2,
        mode="trilinear",
    ):
        super().__init__()
        self.dtype = dtype
        self.use_batchnorm = use_batchnorm
        self.upsample = nn.ConvTranspose3d(
            input_channels,
            input_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            dtype=dtype,
        )
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1, dtype=dtype)
        self.bn = (
            ComplexBatchNorm3D(output_channels)
            if dtype.is_complex
            else nn.BatchNorm3d(output_channels)
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        return x


class ComplexBatchNorm3D(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.real_bn = nn.BatchNorm3d(num_features)
        self.imag_bn = nn.BatchNorm3d(num_features)

    def forward(self, x):
        return torch.complex(self.real_bn(x.real), self.imag_bn(x.imag))
