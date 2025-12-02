from typing import Callable

import torch
import torch.nn as nn

from .activation_functions import get_activation_function
from .blocks import Conv2dBlock, complex_pool, passfunc
from .dense_nn import DenseNN


class CNNDense(nn.Module):
    """CNN encoder followed by dense layers for classification or regression."""

    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        image_size: tuple[int, int],
        start_filters: int = 16,
        cnn_num_layers: int = 3,
        cnn_num_per_layer: int = 2,
        dense_num_layers: int = 2,
        dense_hidden_size: int = 128,
        dense_hidden_dims: list[int] | None = None,
        dtype: torch.dtype = torch.float32,
        dropout: float = 0,
        activation: str | Callable = "relu",
        final_activation: str | Callable = nn.Identity(),
        use_batchnorm: bool = True,
    ):
        """
        Initialize CNNDense.

        Args:
            in_channels: Input channels (C_in, H, W)
            output_dim: Output dimension
            image_size: Input image size (H, W)
            start_filters: Starting number of filters for CNN
            cnn_num_layers: Number of CNN encoder layers
            cnn_num_per_layer: Number of conv blocks per CNN layer
            dense_num_layers: Number of dense layers
            dense_hidden_size: Size of dense hidden layers
            dense_hidden_dims: List of hidden layer dimensions for dense part (overrides num_dense_layers/hidden_size)
            dense_hidden_size: Size of dense hidden layers
            dense_hidden_dims: List of hidden layer dimensions for dense part (overrides num_dense_layers/hidden_size)
            dtype: Data type for the network
            dropout: Dropout probability
            activation: Activation function for hidden layers
            final_activation: Activation function for output layer
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.output_dim = int(output_dim)
        self.image_size = image_size
        self.start_filters = start_filters
        self.num_cnn_layers = cnn_num_layers
        self._num_per_layer = cnn_num_per_layer
        self.dtype = dtype
        self.dropout = dropout
        self._use_batchnorm = use_batchnorm

        if dense_hidden_dims is not None:
            self.hidden_dims = [int(d) for d in dense_hidden_dims]
        else:
            self.hidden_dims = [int(dense_hidden_size)] * dense_num_layers

        if self.dtype.is_complex:
            self.pool = complex_pool
        else:
            self.pool = passfunc
        self._pooler = nn.MaxPool2d(kernel_size=2, stride=2)

        self.activation = activation
        self.final_activation = final_activation

        self.flatten = nn.Flatten()

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
        self.cnn_blocks = nn.ModuleList()

        in_channels = self.in_channels
        out_channels = self.start_filters
        for a0 in range(self.num_cnn_layers):
            if a0 != 0:
                out_channels = in_channels * 2
            self.cnn_blocks.append(
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

        h, w = self.image_size
        for _ in range(self.num_cnn_layers):
            h = h // 2
            w = w // 2
        self.flattened_dim = out_channels * h * w

        self.dense_net = DenseNN(
            input_dim=self.flattened_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            dtype=self.dtype,
            dropout=self.dropout,
            activation=self.activation,
            final_activation=self.final_activation,
            use_batchnorm=self._use_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for cnn_block in self.cnn_blocks:
            x = cnn_block(x)
            x = self.pool(x, self._pooler)

        x = self.flatten(x)
        y = self.dense_net(x)

        return y

    def reset_weights(self):
        """Reset all weights."""

        def _reset(m: nn.Module) -> None:
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

        self.apply(_reset)
