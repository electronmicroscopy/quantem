import numpy as np
import torch
from torch import nn

from .blocks import FinerLayer, SineLayer

""""
All the INR implementations are used for coordinate inputs (x, y, z) to an intensity to that coordinate (I(x, y, z)).
Hence, we use 3 as the number of input features, and 1 output feature as a default.
"""


class Siren(nn.Module):
    """
    Original SIREN implementation.
    """

    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1,
        hidden_layers: int = 3,
        hidden_features: int = 256,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.net_list = []
        self.net_list.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0, alpha=alpha
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
        self.net_list.append(nn.Softplus())
        self.net = nn.Sequential(*self.net_list)

    def forward(self, coords):
        output = self.net(coords)
        return output


class HSiren(nn.Module):
    """
    H-Siren implementation, the first layer is a sinh instead of a sine activation function.
    """

    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1,
        hidden_layers: int = 3,
        hidden_features: int = 256,
        first_omega_0: float = 30,
        hidden_omega_0: float = 30,
        alpha: float = 1.0,
    ):
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
        self.net_list.append(nn.Softplus())
        self.net = nn.Sequential(*self.net_list)

    def forward(self, coords):
        output = self.net(coords)
        return output


class Finer(nn.Module):
    """
    Finer implementation.
    """

    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1,
        hidden_layers: int = 3,
        hidden_features: int = 256,
        first_omega: float = 30,
        hidden_omega: float = 30,
        init_method: str = "sine",
        init_gain: float = 1,
        fbs=None,  # Need to check what FBS/HBS/alphaType/alphaReqGrad are
        hbs=None,
        alphaType=None,
        alphaReqGrad=False,
    ):
        super().__init__()
        self.net_list = []
        self.net_list.append(
            FinerLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega=first_omega,
                init_method=init_method,
                init_gain=init_gain,
                fbs=fbs,
                alphaType=alphaType,
                alphaReqGrad=alphaReqGrad,
            )
        )

        for i in range(hidden_layers):
            self.net_list.append(
                FinerLayer(
                    hidden_features,
                    hidden_features,
                    omega=hidden_omega,
                    init_method=init_method,
                    init_gain=init_gain,
                    hbs=hbs,
                    alphaType=alphaType,
                    alphaReqGrad=alphaReqGrad,
                )
            )

        self.net_list.append(
            FinerLayer(
                hidden_features,
                out_features,
                is_last=True,
                omega=hidden_omega,
                init_method=init_method,
                init_gain=init_gain,
                hbs=hbs,
            )
        )  # omega: For weight init
        self.net = nn.Sequential(*self.net_list)

    def forward(self, coords):
        return self.net(coords)
