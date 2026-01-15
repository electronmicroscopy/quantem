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
        image_shape: tuple[int, int],
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
        """Initialize CNNDense.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        output_dim : int
            Output dimension.
        image_shape : tuple[int, int]
            2D input image shape (H, W).
        start_filters : int, optional
            Starting number of filters for CNN, by default 16
        cnn_num_layers : int, optional
            Number of CNN encoder layers, by default 3
        cnn_num_per_layer : int, optional
            Number of conv blocks per CNN layer, by default 2
        dense_num_layers : int, optional
            Number of dense layers, by default 2
        dense_hidden_size : int, optional
            Size of dense hidden layers, by default 128
        dense_hidden_dims : list[int] or None, optional
            List of hidden layer dimensions for dense part (overrides dense_num_layers/dense_hidden_size), by default None
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
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.output_dim = int(output_dim)
        self.image_shape = image_shape
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
                    activation=self._activation,
                )
            )
            in_channels = out_channels

        h, w = self.image_shape
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
            activation=self._activation,
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
        """Reset all weights in the network."""

        def _reset(m: nn.Module) -> None:
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

        self.apply(_reset)
