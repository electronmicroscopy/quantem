from typing import TYPE_CHECKING, Callable

import torch.nn as nn

from quantem.core import config

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


def get_loss_module(name: str | nn.Module | Callable, dtype: torch.dtype, **kwargs) -> nn.Module:
    """Return a loss *module* by name, or wrap/return what was provided."""
    if isinstance(name, nn.Module):
        return name

    if callable(name) and not isinstance(name, str):
        # Wrap a bare callable into an nn.Module
        class _CallableLoss(nn.Module):
            def __init__(self, fn: Callable):
                super().__init__()
                self.fn = fn

            def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return self.fn(pred, target)

        return _CallableLoss(name)

    loss_name = str(name).lower()

    if dtype.is_complex:
        if loss_name in {"l2", "complex_l2"}:
            return ComplexL2Loss(**kwargs)
        if loss_name in {"complex_cartesian_l2"}:
            return ComplexCartesianL2Loss(**kwargs)
        if loss_name in {"amp_phase_l2"}:
            return AmpPhaseL2Loss(**kwargs)
        if loss_name in {"combined_l2"}:
            return CombinedL2Loss(**kwargs)
        raise ValueError(f"Unknown loss module for complex dtype: {loss_name}")

    # real dtype
    if loss_name in {"l2"}:
        return nn.MSELoss(**kwargs)
    if loss_name in {"l1"}:
        return nn.L1Loss(**kwargs)
    if loss_name in {"smooth_l1"}:
        return nn.SmoothL1Loss(**kwargs)
    if loss_name in {"charbonnier"}:
        return CharbonnierLoss(**kwargs)
    if loss_name in {"llmse"}:
        return LLMSELoss(**kwargs)
    if loss_name in {"mse_log_mse"}:
        return MSELogMSELoss(**kwargs)

    raise ValueError(f"Unknown loss module for real dtype: {loss_name}")


class ComplexL2Loss(nn.Module):
    """L2 loss for complex tensors (separate real/imag, then average)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        real_l2 = torch.mean((pred.real - target.real) ** 2)
        imag_l2 = torch.mean((pred.imag - target.imag) ** 2)
        return (real_l2 + imag_l2) / 2


class ComplexCartesianL2Loss(nn.Module):
    """L2 loss for complex tensors in Cartesian form: E[(Δre^2 + Δim^2)]."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        real_dif = pred.real - target.real
        imag_dif = pred.imag - target.imag
        return torch.mean(real_dif**2 + imag_dif**2)


class AmpPhaseL2Loss(nn.Module):
    """L2 loss on amplitude + wrapped phase."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        amp_l2 = ((target.abs() - pred.abs()) ** 2).mean()

        phase_dif = torch.abs(target.angle() - pred.angle())
        phase_dif = torch.min(phase_dif, 2 * torch.pi - phase_dif)  # wrap to [0, pi]
        phase_l2 = torch.mean(phase_dif**2)

        return amp_l2 + phase_l2


class CombinedL2Loss(nn.Module):
    """Weighted sum of AmpPhaseL2 and ComplexL2.

    loss = alpha * amp_phase + (1 - alpha) * complex_l2
    """

    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = float(alpha)
        self.complex_l2 = ComplexL2Loss()
        self.amp_phase_l2 = AmpPhaseL2Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        comp_l2 = self.complex_l2(pred, target)
        amp_ph_l2 = self.amp_phase_l2(pred, target)
        return self.alpha * amp_ph_l2 + (1 - self.alpha) * comp_l2


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
