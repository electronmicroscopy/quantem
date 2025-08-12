from abc import ABC, abstractmethod
from typing import Any

import torch

from quantem.core import config


class BaseConstraints(ABC):
    """Base class for constraint management with common functionality."""

    # Subclasses should define their own DEFAULT_CONSTRAINTS
    DEFAULT_CONSTRAINTS: dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._soft_constraint_loss = {}
        self._constraints = self.DEFAULT_CONSTRAINTS.copy()
        self._epoch_constraint_losses = {}

    @property
    def constraints(self) -> dict[str, Any]:
        return self._constraints

    @constraints.setter
    def constraints(self, c: dict[str, Any]):
        allowed_keys = self.DEFAULT_CONSTRAINTS.keys()
        constraint_type = self.__class__.__name__.lower().replace("constraints", "")

        for key, value in c.items():
            if key not in allowed_keys:
                raise KeyError(
                    f"Invalid {constraint_type} constraint key '{key}', allowed keys are {list(allowed_keys)}"
                )
            self._constraints[key] = value

    @property
    def soft_constraint_loss(self) -> dict[str, torch.Tensor | float]:
        return self._soft_constraint_loss

    def add_constraint(self, key: str, value: Any):
        allowed_keys = self.DEFAULT_CONSTRAINTS.keys()
        constraint_type = self.__class__.__name__.lower().replace("constraints", "")

        if key not in allowed_keys:
            raise KeyError(
                f"Invalid {constraint_type} constraint key '{key}', allowed keys are {list(allowed_keys)}"
            )
        self._constraints[key] = value

    @abstractmethod
    def apply_soft_constraints(self, *args, **kwargs) -> torch.Tensor:
        """Apply soft constraints and return total constraint loss."""
        pass

    def _get_zero_loss_tensor(self) -> torch.Tensor:
        """Helper method to create a zero loss tensor with proper device and dtype."""
        device = getattr(self, "device", "cpu")
        return torch.tensor(0, device=device, dtype=getattr(torch, config.get("dtype_real")))

    # --- helpers for consistent loss logging ---
    def reset_soft_constraint_losses(self) -> None:
        self._soft_constraint_loss = {}

    def add_soft_constraint_loss(self, name: str, value: torch.Tensor | float) -> None:
        """Record a single soft-constraint loss for logging without holding the graph."""
        if isinstance(value, torch.Tensor):
            val = value.detach()
            if val.ndim != 0:
                val = val.mean()
            self._soft_constraint_loss[name] = val
        else:
            self._soft_constraint_loss[name] = float(value)

    def accumulate_constraint_losses(
        self, batch_constraint_losses: dict[str, torch.Tensor | float] | None = None
    ) -> None:
        """Accumulate constraint losses across batches."""
        if batch_constraint_losses is None:
            batch_constraint_losses = self.soft_constraint_loss

        for loss_name, loss_value in batch_constraint_losses.items():
            if isinstance(loss_value, torch.Tensor):
                try:
                    v = loss_value.item()
                except Exception:
                    print("loss value not singular: ", loss_value)  # TODO remove
                    v = loss_value.detach().mean().item()
            else:
                v = float(loss_value)
            self._epoch_constraint_losses[loss_name] = (
                self._epoch_constraint_losses.get(loss_name, 0.0) + v
            )

    def get_epoch_constraint_losses(self) -> dict[str, float]:
        return getattr(self, "_epoch_constraint_losses", {})  # TODO clean this up

    def reset_epoch_constraint_losses(self) -> None:
        self._epoch_constraint_losses = {}
