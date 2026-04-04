from typing import Callable

import torch
import torch.nn as nn

from .activation_functions import get_activation_function
from .blocks import (
    Conv2dBlock,
    Conv3dBlock,
    Upsample2dBlock,
    Upsample3dBlock,
    complex_pool,
    passfunc,
)


class CNN2d(nn.Module):
    """UNet-like CNN for 2D images. Can be used for classification, regression, or segmentation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        start_filters: int = 16,
        num_layers: int = 3,
        num_per_layer: int = 2,
        use_skip_connections: bool = True,
        dtype: torch.dtype = torch.float32,
        dropout: float = 0,
        activation: str | Callable = "relu",
        final_activation: str | Callable = nn.Identity(),
        use_batchnorm: bool = True,
    ):
        """Initialize CNN2d.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int or None, optional
            Number of output channels. If None, defaults to in_channels, by default None
        start_filters : int, optional
            Starting number of filters for CNN, by default 16
        num_layers : int, optional
            Number of CNN encoder layers, by default 3
        num_per_layer : int, optional
            Number of conv blocks per CNN layer, by default 2
        use_skip_connections : bool, optional
            Whether to use skip connections, by default True
        dtype : torch.dtype, optional
            Data type for the network, by default torch.float32
        dropout : float, optional
            Dropout probability, by default 0
        activation : str or Callable, optional
            Activation function for hidden layers, by default "relu"
        final_activation : str or Callable, optional
            Activation function for output layer, by default nn.Identity()
        use_batchnorm : bool, optional
            Whether to use batch normalization, by default True

        Raises
        ------
        ValueError
            If use_skip_connections is True and num_per_layer < 2.
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels) if out_channels is not None else int(in_channels)
        self.start_filters = start_filters
        self.num_layers = num_layers
        self._num_per_layer = num_per_layer
        if use_skip_connections and num_per_layer < 2:
            raise ValueError(
                "If using skip connections, num_per_layer must be at least 2 to allow for "
                + "channel concatenation."
            )
        self.use_skip_connections = use_skip_connections
        self.dtype = dtype
        self.dropout = dropout
        self._use_batchnorm = use_batchnorm

        if self.dtype.is_complex:
            self.pool = complex_pool
        else:
            self.pool = passfunc
        self._pooler = nn.MaxPool2d(kernel_size=2, stride=2)

        self.concat = torch.cat
        self.flatten = nn.Flatten()

        self.activation = activation
        self.final_activation = final_activation

        self._build()

    @property
    def activation(self) -> Callable:
        """Create a new activation instance each time this is accessed."""
        return get_activation_function(self._activation, self.dtype)

    @activation.setter
    def activation(self, act: str | Callable):
        self._activation = act

    @property
    def final_activation(self) -> Callable:
        return self._final_activation

    @final_activation.setter
    def final_activation(self, act: str | Callable):
        self._final_activation = get_activation_function(act, self.dtype)

    def _build(self) -> None:
        self.down_conv_blocks = nn.ModuleList()
        self.up_conv_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        in_channels = self.in_channels
        out_channels = self.start_filters
        for a0 in range(self.num_layers):
            if a0 != 0:
                out_channels = in_channels * 2
            self.down_conv_blocks.append(
                Conv2dBlock(
                    nb_layers=self._num_per_layer,
                    input_channels=in_channels,
                    output_channels=out_channels,
                    use_batchnorm=self._use_batchnorm,
                    dropout=self.dropout,
                    dtype=self.dtype,
                    activation=self._activation,  # Pass activation config, not instance
                )
            )
            in_channels = out_channels

        out_channels = in_channels * 2
        self.bottleneck = Conv2dBlock(
            nb_layers=self._num_per_layer,
            input_channels=in_channels,
            output_channels=out_channels,
            use_batchnorm=self._use_batchnorm,
            dropout=self.dropout,
            dtype=self.dtype,
            activation=self._activation,
        )
        in_channels = out_channels

        for a0 in range(self.num_layers):
            out_channels = self.start_filters if a0 == self.num_layers - 1 else in_channels // 2

            in_channels2 = in_channels if self.use_skip_connections else out_channels

            self.upsample_blocks.append(
                Upsample2dBlock(
                    in_channels, out_channels, use_batchnorm=self._use_batchnorm, dtype=self.dtype
                )
            )

            self.up_conv_blocks.append(
                Conv2dBlock(
                    nb_layers=self._num_per_layer,
                    input_channels=in_channels2,
                    output_channels=out_channels,
                    use_batchnorm=self._use_batchnorm,
                    dropout=self.dropout,
                    dtype=self.dtype,
                    activation=self._activation,
                )
            )

            in_channels = out_channels

        self.final_conv = Conv2dBlock(
            nb_layers=1,
            input_channels=self.start_filters,
            output_channels=self.out_channels,
            use_batchnorm=False,
            dropout=self.dropout,
            dtype=self.dtype,
            activation=self.final_activation,
        )
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down_block in self.down_conv_blocks:
            x = down_block(x)
            if self.use_skip_connections:
                skips.append(x)
            x = self.pool(x, self._pooler)

        x = self.bottleneck(x)
        for upsample_block, up_conv_block in zip(self.upsample_blocks, self.up_conv_blocks):
            x = upsample_block(x)
            if self.use_skip_connections:
                skip = skips.pop()
                x = torch.cat((x, skip), dim=1)
            x = up_conv_block(x)

        y = self.final_conv(x)

        return y

    def reset_weights(self):
        """Reset all weights in the network."""

        def _reset(m: nn.Module) -> None:
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

        self.apply(_reset)


class CNN3d(nn.Module):
    """UNet-like 3D CNN for processing volumetric data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        start_filters: int = 16,
        num_layers: int = 3,
        num_per_layer: int = 2,
        use_skip_connections: bool = True,
        dtype: torch.dtype = torch.complex64,
        dropout: float = 0,
        activation: str | Callable = "relu",
        final_activation: str | Callable = nn.Identity(),
        use_batchnorm: bool = True,
        mode: str = "complex",
    ):
        """Initialize CNN3d.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int or None, optional
            Number of output channels. If None, defaults to in_channels, by default None
        start_filters : int, optional
            Starting number of filters for CNN, by default 16
        num_layers : int, optional
            Number of CNN encoder layers, by default 3
        num_per_layer : int, optional
            Number of conv blocks per CNN layer, by default 2
        use_skip_connections : bool, optional
            Whether to use skip connections, by default True
        dtype : torch.dtype, optional
            Data type for the network, by default torch.complex64
        dropout : float, optional
            Dropout probability, by default 0
        activation : str or Callable, optional
            Activation function for hidden layers, by default "relu"
        final_activation : str or Callable, optional
            Activation function for output layer, by default nn.Identity()
        use_batchnorm : bool, optional
            Whether to use batch normalization, by default True
        mode : str, optional
            Mode for the network, by default "complex"

        Raises
        ------
        ValueError
            If use_skip_connections is True and num_per_layer < 2.
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels) if out_channels is not None else int(in_channels)
        self.start_filters = start_filters
        self.num_layers = num_layers
        self._num_per_layer = num_per_layer
        if use_skip_connections and num_per_layer < 2:
            raise ValueError(
                "If using skip connections, num_per_layer must be at least 2 to allow for channel concatenation."
            )
        self.use_skip_connections = use_skip_connections
        self.dtype = dtype
        self.dropout = dropout
        self._use_batchnorm = use_batchnorm
        self.mode = mode

        self.pool = complex_pool if dtype.is_complex else passfunc
        self._pooler = nn.MaxPool3d(kernel_size=2, stride=2)

        self.concat = torch.cat
        self.flatten = nn.Flatten()

        self.activation = activation
        self.final_activation = final_activation

        self._build()

    @property
    def activation(self) -> Callable:
        """Create a new activation instance each time this is accessed."""
        return get_activation_function(self._activation, self.dtype)

    @activation.setter
    def activation(self, act: str | Callable):
        self._activation = act

    @property
    def final_activation(self) -> Callable:
        return self._final_activation

    @final_activation.setter
    def final_activation(self, act: str | Callable):
        self._final_activation = get_activation_function(act, self.dtype)

    def _build(self) -> None:
        self.down_conv_blocks = nn.ModuleList()
        self.up_conv_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        in_channels = self.in_channels
        out_channels = self.start_filters
        for a0 in range(self.num_layers):
            if a0 != 0:
                out_channels = in_channels * 2
            self.down_conv_blocks.append(
                Conv3dBlock(
                    nb_layers=self._num_per_layer,
                    input_channels=in_channels,
                    output_channels=out_channels,
                    use_batchnorm=self._use_batchnorm,
                    dropout=self.dropout,
                    dtype=self.dtype,
                    activation=self._activation,
                )
            )
            in_channels = out_channels

        out_channels = in_channels * 2
        self.bottleneck = Conv3dBlock(
            nb_layers=self._num_per_layer,
            input_channels=in_channels,
            output_channels=out_channels,
            use_batchnorm=self._use_batchnorm,
            dropout=self.dropout,
            dtype=self.dtype,
            activation=self._activation,
        )
        in_channels = out_channels

        for a0 in range(self.num_layers):
            if a0 == self.num_layers - 1:
                out_channels = self.start_filters
            else:
                out_channels = in_channels // 2

            if self.use_skip_connections:
                in_channels2 = in_channels
            else:
                in_channels2 = out_channels

            self.upsample_blocks.append(
                Upsample3dBlock(
                    in_channels, out_channels, use_batchnorm=self._use_batchnorm, dtype=self.dtype
                )
            )

            self.up_conv_blocks.append(
                Conv3dBlock(
                    nb_layers=self._num_per_layer,
                    input_channels=in_channels2,
                    output_channels=out_channels,
                    use_batchnorm=self._use_batchnorm,
                    dropout=self.dropout,
                    dtype=self.dtype,
                    activation=self._activation,
                )
            )

            in_channels = out_channels

        self.final_conv = Conv3dBlock(
            nb_layers=1,
            input_channels=self.start_filters,
            output_channels=self.out_channels,
            use_batchnorm=False,
            dropout=self.dropout,
            dtype=self.dtype,
            activation=self.final_activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_0 = False
        if x.dim() != 5:
            if x.dim() == 4:
                x = x[None]
                squeeze_0 = True
            else:
                raise ValueError(
                    f"Input tensor must have 4 or 5 dimensions, got {x.dim()} dimensions."
                )
        skips = []
        for down_block in self.down_conv_blocks:
            x = down_block(x)
            if self.use_skip_connections:
                skips.append(x)
            x = self.pool(x, self._pooler)

        x = self.bottleneck(x)
        for upsample_block, up_conv_block in zip(self.upsample_blocks, self.up_conv_blocks):
            x = upsample_block(x)
            if self.use_skip_connections:
                x = torch.cat((x, skips.pop()), dim=1)
            x = up_conv_block(x)

        y = self.final_conv(x)
        if squeeze_0:
            y = y.squeeze(0)
        return y

    def reset_weights(self):
        """Reset all weights in the network."""

        def _reset(m: nn.Module) -> None:
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

        self.apply(_reset)
