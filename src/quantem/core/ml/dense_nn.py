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
    ) -> None:
        """Initialize DenseNN.

        Parameters
        ----------
        input_dim : int
            Input dimension.
        output_dim : int or None, optional
            Output dimension. If None, defaults to input_dim, by default None
        hidden_dims : list[int] or None, optional
            List of hidden layer dimensions (overrides num_layers/hidden_size), by default None
        num_layers : int, optional
            Number of hidden layers (used if hidden_dims not provided), by default 3
        hidden_size : int, optional
            Size of hidden layers (used if hidden_dims not provided), by default 128
        dtype : torch.dtype, optional
            Data type for the network, by default torch.float32
        dropout : float, optional
            Dropout probability, by default 0
        activation : str or Callable, optional
            Activation function for hidden layers, by default "relu"
        final_activation : str or Callable, optional
            Activation function for output layer, by default nn.Identity()
        use_batchnorm : bool, optional
            Whether to use batch normalization, by default False
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
        """Create a new activation instance each time this is accessed.
        This avoids caching the activation function and helps with summary()"""
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
        self.layers = nn.ModuleList()

        dims = [self.input_dim] + self.hidden_dims

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            layer = []
            layer.append(nn.Linear(in_dim, out_dim, dtype=self.dtype))
            layer.append(self.activation)

            if self._use_batchnorm:
                if self.dtype.is_complex:
                    layer.append(ComplexBatchNorm1D(out_dim))
                else:
                    layer.append(nn.BatchNorm1d(out_dim, dtype=self.dtype))

            if self.dropout > 0:
                layer.append(nn.Dropout(self.dropout))

            self.layers.append(nn.Sequential(*layer))

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

    def reset_weights(self) -> None:
        """Reset all weights in the network."""

        def _reset(m: nn.Module) -> None:
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

        self.apply(_reset)
