from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    import torch


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        shape: tuple[int, int, int],  # (C, H, W) shape of output image
        in_features: int = 2,  # number of spatial dimensions of input
        hidden_features: int | None = None,
        num_layers: int = 3,
        outermost_linear: bool = False,
        hidden_omega_0: float = 30.0,
        first_omega_0: float | None = None,
    ):
        super().__init__()

        if len(shape) != 3:
            raise ValueError(f"shape should be of len 3, (C, H, W), got shape: {shape}")
        elif shape[1] != shape[2]:
            raise NotImplementedError
        self.outshape = shape

        self.in_features = in_features
        if hidden_features is None:
            hidden_features = self.outshape[1] * self.outshape[2]
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.outermost_linear = outermost_linear
        self.hidden_omega_0 = hidden_omega_0
        if first_omega_0 is None:
            first_omega_0 = hidden_omega_0
        self.first_omega_0 = first_omega_0

        self._build()

    def _build(self):
        layers = []
        layers.append(
            SineLayer(
                self.in_features, self.hidden_features, is_first=True, omega_0=self.first_omega_0
            )
        )

        for i in range(self.num_layers):
            layers.append(
                SineLayer(
                    self.hidden_features,
                    self.hidden_features,
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                )
            )

        if self.outermost_linear:
            final_linear = nn.Linear(self.hidden_features, self.outshape[0])

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / self.hidden_features) / self.hidden_omega_0,
                    np.sqrt(6 / self.hidden_features) / self.hidden_omega_0,
                )

            layers.append(final_linear)
        else:
            layers.append(
                SineLayer(
                    self.hidden_features,
                    self.outshape[0],
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        """Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!"""
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations["input"] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations["_".join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations["_".join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

    @staticmethod
    def generate_coords(sidelen, dim=2):
        """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
        sidelen: int
        dim: int

        for now square only
        """
        tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(-1, dim)
        return mgrid


# class Siren(nn.Module):
#     def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
#                  first_omega_0=30, hidden_omega_0=30.):
#         super().__init__()

#         layers = []
#         layers.append(SineLayer(in_features, hidden_features,
#                                   is_first=True, omega_0=first_omega_0))

#         for i in range(hidden_layers):
#             layers.append(SineLayer(hidden_features, hidden_features,
#                                       is_first=False, omega_0=hidden_omega_0))

#         if outermost_linear:
#             final_linear = nn.Linear(hidden_features, out_features)

#             with torch.no_grad():
#                 final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
#                                               np.sqrt(6 / hidden_features) / hidden_omega_0)

#             layers.append(final_linear)
#         else:
#             layers.append(SineLayer(hidden_features, out_features,
#                                       is_first=False, omega_0=hidden_omega_0))

#         self.net = nn.Sequential(*layers)

#     def forward(self, coords):
#         coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
#         output = self.net(coords)
#         return output, coords

#     def forward_with_activations(self, coords, retain_grad=False):
#         '''Returns not only model output, but also intermediate activations.
#         Only used for visualizing activations later!'''
#         activations = OrderedDict()

#         activation_count = 0
#         x = coords.clone().detach().requires_grad_(True)
#         activations['input'] = x
#         for i, layer in enumerate(self.net):
#             if isinstance(layer, SineLayer):
#                 x, intermed = layer.forward_with_intermediate(x)

#                 if retain_grad:
#                     x.retain_grad()
#                     intermed.retain_grad()

#                 activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
#                 activation_count += 1
#             else:
#                 x = layer(x)

#                 if retain_grad:
#                     x.retain_grad()

#             activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
#             activation_count += 1

#         return activations
