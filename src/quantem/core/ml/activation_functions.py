from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModReLU(nn.Module):
    """Modulated ReLU activation for complex-valued inputs.

    Applies ReLU to the absolute value plus a learnable bias, then multiplies
    by the complex exponential of the phase.
    """

    def __init__(self) -> None:
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(torch.abs(x) + self.b) * torch.exp(1.0j * torch.angle(x))


class Complex_ReLU(nn.Module):
    """Complex ReLU activation that applies ReLU separately to real and imaginary parts.

    For a complex input z = a + bi, returns ReLU(a) + ReLU(b)i.
    """

    def __init__(self) -> None:
        super().__init__()

    def _complex_relu(self, z: torch.Tensor) -> torch.Tensor:
        return F.relu(z.real) + 1.0j * F.relu(z.imag)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self._complex_relu(z)


class Complex_Phase_ReLU(nn.Module):
    """Complex Phase ReLU activation that modulates activation based on complex phase.

    Applies activation based on the magnitude and phase of complex inputs,
    with configurable phase fraction and activation function type.
    """

    def __init__(self, phase_frac: float = 0.5, sigmoid: bool = True) -> None:
        """Initialize Complex_Phase_ReLU.

        Parameters
        ----------
        phase_frac : float, optional
            Fraction of the complex phase range that gets activated (0-1), by default 0.5
        sigmoid : bool, optional
            Whether to use sigmoid-like activation (True) or linear (False), by default True
        """
        super().__init__()
        self.phase_frac = phase_frac
        self.sigmoid = sigmoid

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.PhaseReLU(z, self.phase_frac, self.sigmoid)

    def PhaseReLU(
        self,
        z: torch.Tensor,
        phase_frac: float = 0.5,
        sigmoid: bool = True,
    ) -> torch.Tensor:
        """Apply phase-based ReLU activation to complex tensor.

        Parameters
        ----------
        z : torch.Tensor
            Complex input tensor.
        phase_frac : float, optional
            Fraction of complex phase range that gets activated, by default 0.5
        sigmoid : bool, optional
            If True, use sigmoid function along phase; if False, use linear, by default True

        Returns
        -------
        torch.Tensor
            Activated complex tensor with same shape as input.
        """

        # complex inputs
        a = torch.abs(z)
        p = torch.abs(torch.angle(z))

        # positive real outputs
        if sigmoid:
            f = (
                a
                * torch.cos(torch.minimum(p / (2 * phase_frac), torch.ones_like(p) * torch.pi / 2))
                ** 2
            )
        else:
            f = a * (1.0 - torch.minimum(p / (torch.pi * phase_frac), torch.ones_like(p)))

        return f.type(torch.complex64)


def get_activation_function(
    activation_type: str | Callable,
    dtype: "torch.dtype",
    activation_phase_frac: float = 0.5,
    activation_sigmoid: bool = True,
) -> nn.Module:
    """Get an activation function module.

    Parameters
    ----------
    activation_type : str or Callable
        String name of activation (e.g., 'relu', 'phase_relu', 'identity') or
        a callable activation function.
    dtype : torch.dtype
        Data type (used to determine complex vs real activations).
    activation_phase_frac : float, optional
        Fraction for phase relu (complex only), by default 0.5
    activation_sigmoid : bool, optional
        Whether to use sigmoid for phase relu (complex only), by default True

    Returns
    -------
    nn.Module
        Activation function module.

    Raises
    ------
    ValueError
        If activation type is unknown or not supported for the given dtype.
    """
    # If it's already a callable/module, check if it's a module
    if callable(activation_type):
        if isinstance(activation_type, nn.Module):
            return activation_type
        else:
            # Wrap callable in a lambda module
            class CallableWrapper(nn.Module):
                def __init__(self, func: Callable) -> None:
                    super().__init__()
                    self.func = func

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.func(x)

            return CallableWrapper(activation_type)

    activation_type = activation_type.lower()

    if activation_type in ["identity", "eye", "ident"]:
        activation = nn.Identity()
    elif dtype.is_complex:
        if activation_type in ["complexrelu", "complex_relu", "relu"]:
            activation = Complex_ReLU()
        elif activation_type in ["modrelu", "mod_relu"]:
            activation = ModReLU()
        elif activation_type in ["phaserelu", "phase_relu"]:
            activation = Complex_Phase_ReLU(
                phase_frac=activation_phase_frac, sigmoid=activation_sigmoid
            )
        else:
            raise ValueError(
                f"Unknown activation for complex, {activation_type}. "
                + "Should be 'complexrelu', 'modrelu', or 'phaserelu'"
            )
    else:
        if activation_type in ["relu"]:
            activation = nn.ReLU()
        elif activation_type in ["leaky_relu", "leakyrelu", "lrelu"]:
            activation = nn.LeakyReLU()
        elif activation_type in ["elu"]:
            activation = nn.ELU()
        elif activation_type in ["tanh"]:
            activation = nn.Tanh()
        elif activation_type in ["sigmoid"]:
            activation = nn.Sigmoid()
        elif activation_type in ["softplus"]:
            activation = nn.Softplus()
        elif activation_type in ["finer", "siren", "hsiren"]:
            raise ValueError(
                "Siren type layers are not supported for activation functions. "
                + "Use the Siren class instead."
            )
        else:
            raise ValueError(f"Unknown activation type {activation_type}")

    return activation
