from typing import TYPE_CHECKING, Callable

from quantem.core import config

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
        num_channels: int,  # (C, H, W) input shape same as output shape
        start_filters: int = 16,
        num_layers: int = 3,  # num_layers
        num_per_layer: int = 2,  # number conv per layer
        use_skip_connections: bool = False,
        dtype: torch.dtype = torch.complex64,
        dropout: float = 0,
        activation: str | Callable = "relu",
        final_activation: str | Callable = nn.Identity(),
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.in_channels = self.out_channels = int(num_channels)
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
        return self._activation

    @activation.setter
    def activation(self, act: str | Callable):
        if callable(act):
            self._activation = act
        else:
            self._activation = get_activation_function(act, self.dtype)

    @property
    def final_activation(self) -> Callable:
        return self._final_activation

    @final_activation.setter
    def final_activation(self, act: str | Callable):
        if callable(act):
            self._final_activation = act
        else:
            self._final_activation = get_activation_function(act, self.dtype)

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
