from typing import Callable

import torch
import torch.nn as nn

from quantem.core.ml.blocks import ComplexBatchNorm1D

from .activation_functions import get_activation_function


class DenseNN(nn.Module):
    """Fully connected neural network with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        hidden_dims: list[int] | None = None,
        num_layers: int = 3,
        hidden_size: int = 128,
        dtype: torch.dtype = torch.float32,
        dropout: float = 0,
        activation: str | Callable = "relu",
        final_activation: str | Callable = nn.Identity(),
        use_batchnorm: bool = False,
    ):
        """
        Initialize DenseNN.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension (defaults to input_dim if None)
            hidden_dims: List of hidden layer dimensions (overrides num_layers/hidden_size)
            num_layers: Number of hidden layers (used if hidden_dims not provided)
            hidden_size: Size of hidden layers (used if hidden_dims not provided)
            dtype: Data type for the network
            dropout: Dropout probability
            activation: Activation function for hidden layers
            final_activation: Activation function for output layer
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim) if output_dim is not None else int(input_dim)
        self.dtype = dtype
        self.dropout = dropout
        self._use_batchnorm = use_batchnorm

        if hidden_dims is not None:
            self.hidden_dims = [int(d) for d in hidden_dims]
        else:
            self.hidden_dims = [int(hidden_size)] * num_layers

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
        self.layers = nn.ModuleList()

        dims = [self.input_dim] + self.hidden_dims

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            block = []
            block.append(nn.Linear(in_dim, out_dim, dtype=self.dtype))

            block.append(self.activation)

            if self._use_batchnorm:
                if self.dtype.is_complex:
                    block.append(ComplexBatchNorm1D(out_dim))
                else:
                    block.append(nn.BatchNorm1d(out_dim, dtype=self.dtype))

            if self.dropout > 0:
                block.append(nn.Dropout(self.dropout))

            self.layers.append(nn.Sequential(*block))

        self.layers.append(
            nn.Linear(
                self.hidden_dims[-1] if self.hidden_dims else self.input_dim,
                self.output_dim,
                dtype=self.dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x = x.reshape(-1, self.input_dim)

        for layer in self.layers:
            x = layer(x)

        y = self.final_activation(x)

        if len(original_shape) > 1:
            y = y.reshape(*original_shape, self.output_dim)

        return y

    def reset_weights(self):
        """Reset all weights."""

        def _reset(m: nn.Module) -> None:
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

        self.apply(_reset)
