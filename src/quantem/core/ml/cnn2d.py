from typing import TYPE_CHECKING, Callable

from quantem.core import config
from math import floor

from .activation_functions import get_activation_function
from .blocks import Conv2dBlock, Upsample2dBlock, complex_pool, passfunc

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
else:
    if config.get("has_torch"):
        import torch
        import torch.nn as nn


class CNN2d(nn.Module):
    """ """

    def __init__(
        self,
        in_channels: int,  # input channels (C_in, H, W)
        out_channels: int | None = None,  # output channels (C_out, H, W)
        start_filters: int = 16,
        num_layers: int = 3,  # num_layers
        num_per_layer: int = 2,  # number conv per layer
        use_skip_connections: bool = False,
        dtype: torch.dtype = torch.float32,
        dropout: float = 0,
        activation: str | Callable = "relu",
        final_activation: str | Callable = nn.Identity(),
        use_batchnorm: bool = True,
        conv_kernel_size: int = 3,
    ):
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

        if callable(activation):
            self._activation = activation
        else:
            self._activation = get_activation_function(activation, self.dtype)
        if callable(final_activation):
            self._final_activation = final_activation
        else:
            self._final_activation = get_activation_function(final_activation, self.dtype)
        if conv_kernel_size <=0:
            raise ValueError(f"Convolutional kernel size must be greater than 0. Got value {conv_kernel_size}")
        if conv_kernel_size % 2 == 0:
            raise ValueError(f"Convolutional kernel size must be an odd number. Got value {conv_kernel_size}")
        self._conv_kernel_size = int(conv_kernel_size)

        self._build()

    @property
    def activation(self) -> Callable:
        return self._activation

    @property
    def final_activation(self) -> Callable:
        return self._final_activation

    @property
    def conv_kernel_size(self) -> int:
        return self._conv_kernel_size

    def _build(self):
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
                    activation=self.activation,
                    kernel_size=self.conv_kernel_size,
                    padding=int(floor(self.conv_kernel_size/2)),
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
            activation=self.activation,
            kernel_size=self.conv_kernel_size,
            padding=int(floor(self.conv_kernel_size/2)),
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
                    activation=self.activation,
                    kernel_size=self.conv_kernel_size,
                    padding=int(floor(self.conv_kernel_size/2)),
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
        """
        Reset all weights.
        """

        def _reset(m: nn.Module) -> None:
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

        self.apply(_reset)


class MultiChannelCNN2d(CNN2d):
    def __init__(
        self,
        in_channels=1,
        out_channels: int = 2,
        final_activations: list = ["sigmoid", "sigmoid"],
        **kwargs
    ):
        # Always use identity activation in base CNN, handle activations here
        super().__init__(in_channels=in_channels, out_channels=out_channels, final_activation="identity", **kwargs)
        self.final_activations = final_activations

    @property
    def final_activations(self): 
        return self._final_activations

    @final_activations.setter
    def final_activations(self, value):
        if not isinstance(value, (list, tuple)) or len(value) != self.out_channels:
            raise ValueError(f"final_activations must be a list of length {self.out_channels}")
        self._final_activations = [get_activation_function(act, self.dtype) for act in value]

    def forward(self, x):
        out = super().forward(x) # B,C,H,W
        # Apply per-channel activation
        outs = []
        for i, fn in enumerate(self.final_activations):
            outs.append(fn(out[:, i:i+1]))
        return torch.cat(outs, dim=1)