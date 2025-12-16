import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30,
        hsiren=False,
        alpha=1.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.hsiren = hsiren
        self.in_features = in_features
        self.alpha = alpha
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # Scale the first layer initialization by alpha
                self.linear.weight.uniform_(
                    -self.alpha / self.in_features, self.alpha / self.in_features
                )
            else:
                # Scale the hidden layer initialization by alpha
                self.linear.weight.uniform_(
                    -self.alpha * np.sqrt(6 / self.in_features) / self.omega_0,
                    self.alpha * np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        if self.is_first and self.hsiren:
            out = torch.sin(self.omega_0 * torch.sinh(2 * self.linear(input)))
        else:
            out = torch.sin(self.omega_0 * self.linear(input))
        return out


class HSiren(nn.Module):
    def __init__(
        self,
        in_features=2,
        out_features=3,
        hidden_layers=3,
        hidden_features=256,
        first_omega_0=30,
        hidden_omega_0=30,
        alpha=1.0,
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
