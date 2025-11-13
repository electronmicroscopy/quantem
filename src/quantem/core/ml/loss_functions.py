from typing import TYPE_CHECKING, Callable

from quantem.core import config

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


def get_loss_function(name: str | Callable, dtype: torch.dtype) -> Callable:
    if isinstance(name, Callable):
        return name
    else:
        name = name.lower()
    if dtype.is_complex:
        if name in ["l2", "complex_l2"]:
            return complex_l2
        elif name in ["complex_cartesian_l2"]:
            return complex_cartesian_l2
        elif name in ["amp_phase_l2"]:
            return amp_phase_l2
        elif name in ["combined_l2"]:
            return combined_l2
        else:
            raise ValueError(f"Unknown loss function for complex dtype: {name}")
    else:
        if name in ["l2"]:
            return torch.nn.functional.mse_loss
        elif name in ["l1"]:
            return torch.nn.functional.l1_loss
        else:
            raise ValueError(f"Unknown loss function for real dtype: {name}")


def complex_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    real_l2 = torch.mean((pred.real - target.real) ** 2)
    imag_l2 = torch.mean((pred.imag - target.imag) ** 2)
    return (real_l2 + imag_l2) / 2


def complex_cartesian_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    real_dif = pred.real - target.real
    imag_dif = pred.imag - target.imag
    loss = torch.mean(real_dif**2 + imag_dif**2)
    return loss


def amp_phase_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    amp_l2 = ((target.abs() - pred.abs()) ** 2).mean()
    phase_dif = torch.abs(target.angle() - pred.angle())
    phase_dif = torch.min(phase_dif, 2 * torch.pi - phase_dif)  # phase wrapping
    phase_l2 = torch.mean(phase_dif**2)
    return amp_l2 + phase_l2


def combined_l2(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    """
    alpha * amp_phase_l2 + (1 - alpha) * complex_l2
    so larger alpha -> more weight on amp/phase and smaller alpha -> more weight on real/imag

    funnily enough alpha = 0.7 is stable, 0.8 isnt and 0.6
    """
    comp_l2 = complex_l2(pred, target)
    amp_ph_l2 = amp_phase_l2(pred, target)
    return alpha * amp_ph_l2 + (1 - alpha) * comp_l2
