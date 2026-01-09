from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation_functions import get_activation_function

# region ---- Utility Functions ----


def passfunc(z: torch.Tensor, m: Callable) -> torch.Tensor:
    return m(z)


# endregion ---- Utility Functions ----

# region ---- Convolutional Layers ----


def complex_pool(z: torch.Tensor, pooler: Callable) -> torch.Tensor:
    """Apply pooling operation to complex tensor by applying pooler separately to real and imaginary parts.

    Parameters
    ----------
    z : torch.Tensor
        Complex input tensor.
    pooler : callable
        Pooling function to apply.

    Returns
    -------
    torch.Tensor
        Complex tensor with pooled real and imaginary parts.
    """
    return pooler(z.real) + 1.0j * pooler(z.imag)


def reset_weights(m: nn.Module) -> None:
    """Reset the weights of a given module.

    Parameters
    ----------
    m : nn.Module
        The neural network module to reset.
    """
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        reset_parameters()


class Conv2dBlock(nn.Module):
    """Block of convolutional layers with activation, optional batch normalization and dropout."""

    def __init__(
        self,
        nb_layers: int,
        input_channels: int | list | tuple,
        output_channels: int | list | tuple,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batchnorm: bool = False,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
        activation: str | Callable | None = None,
    ) -> None:
        """Initialize Conv2dBlock.

        Parameters
        ----------
        nb_layers : int
            Number of convolutional layers in the block.
        input_channels : int or list or tuple
            Number of input channels. If list/tuple, must match nb_layers.
        output_channels : int or list or tuple
            Number of output channels. If list/tuple, must match nb_layers.
        kernel_size : int, optional
            Size of the convolutional kernel, by default 3
        stride : int, optional
            Stride of the convolution, by default 1
        padding : int, optional
            Padding added to input, by default 1
        use_batchnorm : bool, optional
            Whether to include batch normalization, by default False
        dropout : float, optional
            Dropout probability, by default 0.0
        dtype : torch.dtype, optional
            Data type for the layers, by default torch.float32
        activation : str or Callable or None, optional
            Activation function to use (string or callable). If None, defaults to "relu", by default None
        """
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

        self._activation = activation if activation is not None else "relu"

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
            # Get new activation instance for each conv layer
            block.append(get_activation_function(self._activation, dtype))
            if use_batchnorm:
                block.append(self.bn(output_channels_list[idx]))
            if dropout > 0:
                block.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block(x)
        return output


class Upsample2dBlock(nn.Module):
    """Upsampling block using transposed convolution or interpolation followed by convolution."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        use_batchnorm: bool = False,
        dtype: "torch.dtype" = torch.float32,
        scale_factor: int = 2,
        mode: str = "bilinear",
    ):
        """Initialize Upsample2dBlock.

        Parameters
        ----------
        input_channels : int
            Number of input channels.
        output_channels : int
            Number of output channels.
        use_batchnorm : bool, optional
            Whether to include batch normalization, by default False
        dtype : torch.dtype, optional
            Data type for the layers, by default torch.float32
        scale_factor : int, optional
            Factor by which to scale the input, by default 2
        mode : str, optional
            Interpolation mode, either "bilinear" or "nearest", by default "bilinear"
        """
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
        if self.scale_factor == 2:
            x = self.upsample2x(x)
        else:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        return x


class ComplexBatchNorm1D(nn.Module):
    """Batch normalization for complex inputs (real and imaginary parts separately)."""

    def __init__(self, num_features: int) -> None:
        """Initialize ComplexBatchNorm1D.

        Parameters
        ----------
        num_features : int
            Number of features/channels.
        """
        super(ComplexBatchNorm1D, self).__init__()
        self.real_bn = nn.BatchNorm1d(num_features)
        self.imag_bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.real_bn(x.real), self.imag_bn(x.imag))


class ComplexBatchNorm2D(nn.Module):
    """Batch normalization for complex inputs (real and imaginary parts separately)."""

    def __init__(self, num_features: int) -> None:
        """Initialize ComplexBatchNorm2D.

        Parameters
        ----------
        num_features : int
            Number of features/channels.
        """
        super(ComplexBatchNorm2D, self).__init__()
        self.real_bn = nn.BatchNorm2d(num_features)
        self.imag_bn = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.real_bn(x.real), self.imag_bn(x.imag))


class ComplexBatchNorm3D(nn.Module):
    """Batch normalization for complex inputs (real and imaginary parts separately)."""

    def __init__(self, num_features: int) -> None:
        """Initialize ComplexBatchNorm3D.

        Parameters
        ----------
        num_features : int
            Number of features/channels.
        """
        super(ComplexBatchNorm3D, self).__init__()
        self.real_bn = nn.BatchNorm3d(num_features)
        self.imag_bn = nn.BatchNorm3d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.real_bn(x.real), self.imag_bn(x.imag))


class ComplexNormalize(nn.Module):
    """Normalization layer for complex tensors using separate mean and std for real and imaginary parts."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, inplace: bool = True) -> None:
        """Initialize ComplexNormalize.

        Parameters
        ----------
        mean : torch.Tensor
            Mean values with shape (C, 2) for (real, imaginary) parts.
        std : torch.Tensor
            Standard deviation values with shape (C, 2) for (real, imaginary) parts.
        inplace : bool, optional
            Whether to perform normalization in-place, by default True
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

    def to_device(self, device: torch.device | str) -> "ComplexNormalize":
        """Move mean and std tensors to the specified device.

        Parameters
        ----------
        device : torch.device or str
            Target device.

        Returns
        -------
        ComplexNormalize
            Self for method chaining.
        """
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def __repr__(self):
        return f"Mean (C,2) (R,C):\n{self.mean}\nStd:\n{self.std}"


class Conv3dBlock(nn.Module):
    """3D convolutional block with optional batch normalization, dropout, and activation."""

    def __init__(
        self,
        nb_layers: int,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batchnorm: bool = False,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
        activation: str | Callable | None = None,
    ) -> None:
        """Initialize Conv3dBlock.

        Parameters
        ----------
        nb_layers : int
            Number of convolutional layers in the block.
        input_channels : int
            Number of input channels.
        output_channels : int
            Number of output channels.
        kernel_size : int, optional
            Size of the convolutional kernel, by default 3
        stride : int, optional
            Stride of the convolution, by default 1
        padding : int, optional
            Padding added to input, by default 1
        use_batchnorm : bool, optional
            Whether to include batch normalization, by default False
        dropout : float, optional
            Dropout probability, by default 0.0
        dtype : torch.dtype, optional
            Data type for the layers, by default torch.float32
        activation : str or Callable or None, optional
            Activation function to use (string or callable). If None, defaults to "relu", by default None
        """
        super().__init__()
        self.dtype = dtype
        self.bn = ComplexBatchNorm3D if dtype.is_complex else nn.BatchNorm3d
        self._activation = activation if activation is not None else "relu"

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
            # Get new activation instance for each conv layer
            from .activation_functions import get_activation_function

            layers.append(get_activation_function(self._activation, dtype))
            if use_batchnorm:
                layers.append(self.bn(output_channels))
            input_channels = output_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Upsample3dBlock(nn.Module):
    """3D upsampling block using transposed convolution followed by 1x1x1 convolution."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        use_batchnorm: bool = False,
        dtype: torch.dtype = torch.float32,
        scale_factor: int = 2,
        mode: str = "trilinear",
    ) -> None:
        """Initialize Upsample3dBlock.

        Parameters
        ----------
        input_channels : int
            Number of input channels.
        output_channels : int
            Number of output channels.
        use_batchnorm : bool, optional
            Whether to include batch normalization, by default False
        dtype : torch.dtype, optional
            Data type for the layers, by default torch.float32
        scale_factor : int, optional
            Factor by which to scale the input, by default 2
        mode : str, optional
            Interpolation mode, by default "trilinear"
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        return x


# endregion ---- Convolutional Layers ----


# region ---- Siren Family of Layers ----


class SineLayer(nn.Module):
    """Sine layer for H-Siren and SIREN implementations.

    Note: H-Siren uses the hyperbolic sine function only for the first layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30,
        hsiren: bool = False,
        alpha: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize SineLayer.

        Parameters
        ----------
        in_features : int
            Dimensionality of input coordinates (e.g., 3 for 3D: z, y, x).
        out_features : int
            Dimensionality of output (e.g., 1 for scalar field).
        bias : bool, optional
            Whether to use a bias term in the linear layer, by default True
        is_first : bool, optional
            Whether this is the first layer (affects weight initialization and activation), by default False
        omega_0 : float, optional
            Scaling factor for the activation function, by default 30
        hsiren : bool, optional
            Whether this is an H-Siren layer (uses hyperbolic sine for first layer if True), by default False
        alpha : float, optional
            Scaling factor for weight initialization, by default 1.0
        dtype : torch.dtype, optional
            Data type for the layer, by default torch.float32
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.hsiren = hsiren
        self.in_features = in_features
        self.alpha = alpha
        self.dtype = dtype
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                # Scale the first layer initialization by alpha
                self.linear.weight.uniform_(
                    -self.alpha / self.in_features, self.alpha / self.in_features
                )
            else:
                # Scale the hidden layer initialization by alpha
                self.linear.weight.uniform_(
                    -self.alpha * np.sqrt(6 / self.in_features) / self.omega_0,
                    self.alpha * np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.is_first and self.hsiren:
            out = torch.sin(self.omega_0 * torch.sinh(2 * self.linear(input)))
        else:
            out = torch.sin(self.omega_0 * self.linear(input))
        return out


# endregion ---- Siren Family of Layers ----
