from typing import Callable

import numpy as np
import torch
from torch import nn

from .activation_functions import get_activation_function
from .blocks import SineLayer


class Siren(nn.Module):
    """Original SIREN implementation."""

    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1,
        hidden_layers: int = 3,
        hidden_features: int = 256,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        alpha: float = 1.0,
        hsiren: bool = False,
        dtype: torch.dtype = torch.float32,
        final_activation: str | Callable = "identity",
    ) -> None:
        """Initialize Siren.

        Parameters
        ----------
        in_features : int, optional
            Dimensionality of input coordinates (3 for 3D: z, y, x), by default 3
        out_features : int, optional
            Dimensionality of output (1 for scalar field), by default 1
        hidden_layers : int, optional
            Number of hidden layers, by default 3
        hidden_features : int, optional
            Number of features in each hidden layer, by default 256
        first_omega_0 : float, optional
            Activation function scaling factor for the first layer, by default 30.0
        hidden_omega_0 : float, optional
            Activation function scaling factor for the hidden layers, by default 30.0
        alpha : float, optional
            Weight initialization scaling factor, by default 1.0
        hsiren : bool, optional
            Whether to use the H-Siren activation function, by default False
        dtype : torch.dtype, optional
            Data type for the network, by default torch.float32
        final_activation : str or Callable, optional
            Final activation function, by default "identity"
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.alpha = alpha
        self.hsiren = hsiren
        self.dtype = dtype

        self.final_activation = final_activation

        self._build()

    @property
    def final_activation(self) -> Callable:
        return self._final_activation

    @final_activation.setter
    def final_activation(self, act: str | Callable):
        self._final_activation = get_activation_function(act, dtype=self.dtype)

    def _build(self) -> None:
        net_list = []
        net_list.append(
            SineLayer(
                self.in_features,
                self.hidden_features,
                is_first=True,
                omega_0=self.first_omega_0,
                hsiren=self.hsiren,
                alpha=self.alpha,
                dtype=self.dtype,
            )
        )

        for i in range(self.hidden_layers):
            net_list.append(
                SineLayer(
                    self.hidden_features,
                    self.hidden_features,
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                    alpha=self.alpha,
                    dtype=self.dtype,
                )
            )

        final_linear = nn.Linear(self.hidden_features, self.out_features, dtype=self.dtype)
        with torch.no_grad():
            # Final layer keeps original initialization (no alpha scaling)
            final_linear.weight.uniform_(
                -np.sqrt(6 / self.hidden_features) / self.hidden_omega_0,
                np.sqrt(6 / self.hidden_features) / self.hidden_omega_0,
            )
        net_list.append(final_linear)
        net_list.append(self._final_activation)
        self.net = nn.Sequential(*net_list)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        output = self.net(coords)
        return output

    def reset_weights(self) -> None:
        """Reset all weights in the network."""
        self._build()

    def make_equispaced_grid(
        self, bounds: tuple[tuple[float, float], ...], sampling: tuple[float, ...]
    ) -> torch.Tensor:
        """Create an equispaced coordinate grid for the implicit neural representation.

        Parameters
        ----------
        bounds : tuple of tuples
            Bounds for each dimension as ((min_0, max_0), (min_1, max_1), ...).
            Length must match in_features.
        sampling : tuple of float
            Sampling interval for each dimension (spacing_0, spacing_1, ...).
            Length must match in_features.

        Returns
        -------
        torch.Tensor
            Flattened coordinate grid of shape (N, in_features), where N is the
            total number of grid points.

        Raises
        ------
        ValueError
            If bounds or sampling length does not match in_features.

        Examples
        --------
        For a model with in_features=2:
        >>> bounds = ((0, 1), (0, 1))
        >>> sampling = (0.1, 0.1)
        >>> coords = siren.make_equispaced_grid(bounds, sampling)
        """
        if len(bounds) != self.in_features:
            raise ValueError(
                f"Bounds length ({len(bounds)}) must match in_features ({self.in_features})"
            )
        if len(sampling) != self.in_features:
            raise ValueError(
                f"Sampling length ({len(sampling)}) must match in_features ({self.in_features})"
            )

        grids = []
        for (bound_min, bound_max), sample in zip(bounds, sampling):
            num_points = int((bound_max - bound_min) / sample) + 1
            grids.append(torch.linspace(bound_min, bound_max, num_points))

        coords = torch.meshgrid(*grids, indexing="ij")
        coords = torch.stack(coords, dim=-1).to(self.dtype)
        return coords.reshape(-1, self.in_features)


class HSiren(Siren):
    """H-Siren implementation, the first layer uses sinh instead of sine activation function."""

    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1,
        hidden_layers: int = 3,
        hidden_features: int = 256,
        first_omega_0: float = 30,
        hidden_omega_0: float = 30,
        alpha: float = 1.0,
        dtype: torch.dtype = torch.float32,
        final_activation: str | Callable = "identity",
    ) -> None:
        """Initialize HSiren.

        Parameters
        ----------
        in_features : int, optional
            Dimensionality of input coordinates (3 for 3D: z, y, x), by default 3
        out_features : int, optional
            Dimensionality of output (1 for scalar field), by default 1
        hidden_layers : int, optional
            Number of hidden layers, by default 3
        hidden_features : int, optional
            Number of features in each hidden layer, by default 256
        first_omega_0 : float, optional
            Activation function scaling factor for the first layer, by default 30
        hidden_omega_0 : float, optional
            Activation function scaling factor for the hidden layers, by default 30
        alpha : float, optional
            Weight initialization scaling factor, by default 1.0
        dtype : torch.dtype, optional
            Data type for the network, by default torch.float32
        final_activation : str or Callable, optional
            Final activation function, by default "identity"
        """
        super().__init__(
            in_features,
            out_features,
            hidden_layers,
            hidden_features,
            first_omega_0,
            hidden_omega_0,
            alpha,
            hsiren=True,
            dtype=dtype,
            final_activation=final_activation,
        )
