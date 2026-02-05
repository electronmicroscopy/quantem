from typing import TYPE_CHECKING, Callable

import torch.nn as nn

from quantem.core import config

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


def get_loss_function(name: str | Callable, dtype: torch.dtype) -> Callable:
    """Get a loss function by name or return callable if provided.

    Parameters
    ----------
    name : str or Callable
        Loss function name or callable function.
    dtype : torch.dtype
        Data type (used to determine complex vs real loss functions).

    Returns
    -------
    Callable
        Loss function.

    Raises
    ------
    ValueError
        If loss function name is unknown for the given dtype.
    """
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
    """Compute L2 loss for complex tensors (separate real and imaginary parts).

    Parameters
    ----------
    pred : torch.Tensor
        Predicted complex tensor.
    target : torch.Tensor
        Target complex tensor.

    Returns
    -------
    torch.Tensor
        L2 loss value.
    """
    real_l2 = torch.mean((pred.real - target.real) ** 2)
    imag_l2 = torch.mean((pred.imag - target.imag) ** 2)
    return (real_l2 + imag_l2) / 2


def complex_cartesian_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute L2 loss for complex tensors in Cartesian coordinates.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted complex tensor.
    target : torch.Tensor
        Target complex tensor.

    Returns
    -------
    torch.Tensor
        L2 loss value.
    """
    real_dif = pred.real - target.real
    imag_dif = pred.imag - target.imag
    loss = torch.mean(real_dif**2 + imag_dif**2)
    return loss


def amp_phase_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute L2 loss for complex tensors in amplitude-phase representation.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted complex tensor.
    target : torch.Tensor
        Target complex tensor.

    Returns
    -------
    torch.Tensor
        L2 loss value (amplitude + phase).
    """
    amp_l2 = ((target.abs() - pred.abs()) ** 2).mean()
    phase_dif = torch.abs(target.angle() - pred.angle())
    phase_dif = torch.min(phase_dif, 2 * torch.pi - phase_dif)  # phase wrapping
    phase_l2 = torch.mean(phase_dif**2)
    return amp_l2 + phase_l2


def combined_l2(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    """Combined L2 loss: weighted sum of amplitude-phase and complex L2 losses.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted complex tensor.
    target : torch.Tensor
        Target complex tensor.
    alpha : float, optional
        Weight for amplitude-phase loss. Larger alpha gives more weight to
        amp/phase, smaller alpha gives more weight to real/imag, by default 0.7

    Returns
    -------
    torch.Tensor
        Combined L2 loss value.

    different alpha values can affect stability of training.
    """
    comp_l2 = complex_l2(pred, target)
    amp_ph_l2 = amp_phase_l2(pred, target)
    return alpha * amp_ph_l2 + (1 - alpha) * comp_l2


# TODO: Better loss function implementation? More torch-like.


class L1Loss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
    ):
        super(L1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.l1_loss(pred, target, reduction=self.reduction)


class MSELoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
    ):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(pred, target, reduction=self.reduction)


class MSELogMSELoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super(MSELogMSELoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = (pred - target) ** 2
        log_mse = -mse * torch.log(mse + self.eps)
        if self.reduction == "mean":
            return log_mse.mean()
        elif self.reduction == "sum":
            return log_mse.sum()
        return log_mse


class LLMSELoss(nn.Module):
    """
    Logarithmic Linear Mean Squared Error (LLMSE) loss:
        L = -log(1 - |y - y_hat| / max(|y - y_hat|))
    """

    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Absolute residual
        abs_diff = torch.abs(pred - target)

        # Normalization by max error in batch (avoid div-by-zero)
        max_diff = torch.max(abs_diff.detach()) + self.eps
        norm_diff = abs_diff / max_diff

        # Apply -log(1 - normalized_error)
        loss = -torch.log(1.0 - norm_diff + self.eps)

        # Reduce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-12, reduction="mean"):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, prediction, target):
        diff = prediction - target
        loss = torch.sqrt(diff * diff + self.epsilon**2)

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:  # 'none'
            return loss
