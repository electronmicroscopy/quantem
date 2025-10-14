from .blocks import SineLayer, FinerLayer
from torch import nn
import torch
import numpy as np

class Siren(nn.Module):
    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 3,
        hidden_layers=3,
        hidden_features: int = 256,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.net_list = []
        self.net_list.append(SineLayer(in_features, hidden_features, is_first=True,
                                  omega_0=first_omega_0, alpha=alpha))

        for i in range(hidden_layers):
            self.net_list.append(SineLayer(hidden_features, hidden_features, is_first=False,
                                     omega_0=hidden_omega_0, alpha=alpha))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            # Final layer keeps original initialization (no alpha scaling)
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                          np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net_list.append(final_linear)

        self.net = nn.Sequential(*self.net_list)

    def forward(self, coords):
        output = self.net(coords)
        return output

class HSiren(nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_layers=3, hidden_features=256,
                 first_omega_0=30, hidden_omega_0=30, alpha=1.0):
        super().__init__()
        self.net_list = []
        self.net_list.append(SineLayer(in_features, hidden_features, is_first=True,
                                  omega_0=first_omega_0, hsiren=True, alpha=alpha))

        for i in range(hidden_layers):
            self.net_list.append(SineLayer(hidden_features, hidden_features, is_first=False,
                                     omega_0=hidden_omega_0, alpha=alpha))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            # Final layer keeps original initialization (no alpha scaling)
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                          np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net_list.append(final_linear)

        self.net = nn.Sequential(*self.net_list)

    def forward(self, coords):
        output = self.net(coords)
        return output
    
class Finer(nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_layers=3, hidden_features=256,
                 first_omega=30, hidden_omega=30,
                 init_method='sine', init_gain=1, fbs=None, hbs=None,
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.net_list = []
        self.net_list.append(FinerLayer(in_features, hidden_features, is_first=True,
                                   omega=first_omega,
                                   init_method=init_method, init_gain=init_gain, fbs=fbs,
                                   alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        for i in range(hidden_layers):
            self.net_list.append(FinerLayer(hidden_features, hidden_features,
                                       omega=hidden_omega,
                                       init_method=init_method, init_gain=init_gain, hbs=hbs,
                                       alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        self.net_list.append(FinerLayer(hidden_features, out_features, is_last=True,
                                   omega=hidden_omega,
                                   init_method=init_method, init_gain=init_gain, hbs=hbs)) # omega: For weight init
        self.net = nn.Sequential(*self.net_list)

    def forward(self, coords):
        return self.net(coords)