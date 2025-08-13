from typing import TYPE_CHECKING, Callable

from quantem.core import config

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:
    if config.get("has_torch"):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F


class ModReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return F.relu(torch.abs(x) + self.b) * torch.exp(1.0j * torch.angle(x))


class Complex_ReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _complex_relu(self, z: "torch.Tensor"):
        return F.relu(z.real) + 1.0j * F.relu(z.imag)

    def forward(self, z):
        return self._complex_relu(z)


class Complex_Phase_ReLU(nn.Module):
    def __init__(self, phase_frac=0.5, sigmoid=True) -> None:
        super().__init__()
        self.phase_frac = phase_frac
        self.sigmoid = sigmoid

    def forward(self, z):
        return self.PhaseReLU(z, self.phase_frac, self.sigmoid)

    def PhaseReLU(
        self,
        z,
        phase_frac=0.5,
        sigmoid=True,
    ):
        """
        This function returns the activation for an array of complex numbers z.

        Parameters
        ----------
        z: np.array
            array-like inputs
        phase_frac: float
            Fraction of complex number phase range which are activated.
        sigmoid: bool
            Set to True to use a sigmoid function along phase, or False to use a linear function.

        Returns
        -------
        f: np.array
            output array with the same dimensions as z.

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
    activation_type: str,
    dtype: "torch.dtype",
    activation_phase_frac: float = 0.5,
    activation_sigmoid: bool = True,
) -> Callable:
    """
    Allowed activation_types: relu, phase_relu

    """
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
        else:
            raise ValueError(f"Unknown activation type {activation_type}")

    return activation
