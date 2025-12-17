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
        final_activation : str or Callable, optional
            Final activation function, by default "identity"
        """
        super().__init__()
        self.net_list = []
        self.net_list.append(
            SineLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
                hsiren=True,
                alpha=alpha,
            )
        )

        for i in range(hidden_layers):
            self.net_list.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    alpha=alpha,
                )
            )

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            # Final layer keeps original initialization (no alpha scaling)
            final_linear.weight.uniform_(
                -np.sqrt(6 / hidden_features) / hidden_omega_0,
                np.sqrt(6 / hidden_features) / hidden_omega_0,
            )
        self.net_list.append(final_linear)
        self.net_list.append(get_activation_function(final_activation, dtype=torch.float32))
        self.net = nn.Sequential(*self.net_list)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        output = self.net(coords)
        return output


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
            final_activation=final_activation,
        )
