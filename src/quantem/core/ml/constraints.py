from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Self, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core import config


@dataclass
class BaseContext(ABC):
    """
    Context object bundling the data a constraint needs to be applied.

    Tomography's ``ReconstructionContext`` subclasses this and is passed to its
    ``apply_soft_constraints(ctx)`` overrides. Ptychography models instead pass
    their tensors positionally, so the base ``apply_soft_constraints`` signature
    stays ``*args, **kwargs`` to accommodate both domains.
    """

    pass


@dataclass(slots=False)
class Constraints(ABC):
    """
    Any model that inherits from BaseConstraints will contain a Constraints instance that contains soft and hard constraints.
    """

    soft_constraint_keys = []
    hard_constraint_keys = []

    @property
    def allowed_keys(self) -> list[str]:
        """
        List of all allowed keys.
        """
        return self.hard_constraint_keys + self.soft_constraint_keys

    def copy(self) -> Self:
        """
        Copy the constraints.
        """
        return deepcopy(self)

    def __str__(self) -> str:
        hard = "\n".join(f"{key}: {getattr(self, key)}" for key in self.hard_constraint_keys)
        soft = "\n".join(f"{key}: {getattr(self, key)}" for key in self.soft_constraint_keys)

        # Fix: Move the replace operations outside the f-string or assign to variables
        hard_indented = hard.replace("\n", "\n    ")
        soft_indented = soft.replace("\n", "\n    ")

        return (
            "Constraints:\n"
            "  Hard constraints:\n"
            f"    {hard_indented}\n"
            "  Soft constraints:\n"
            f"    {soft_indented}"
        )


def parse_constraint_dict(
    namespace: type,
    d: dict,
    *,
    kind: str = "constraint",
) -> Constraints:
    """Dispatch a config dict to one of ``namespace``'s nested ``Constraints`` variants.

    ``namespace`` is a class with one or more nested ``@dataclass``\\ -decorated
    ``Constraints`` subclasses. The dict must contain a ``"name"`` or ``"type"`` key
    whose value (case-insensitive) matches one variant's ``_name`` field; the
    remaining keys are forwarded as constructor kwargs to that variant.

    ``kind`` is a short human-readable label ("object", "probe", "dataset", ...)
    used only in error messages.
    """
    d = dict(d)
    name = d.pop("name", None) or d.pop("type", None)
    if name is None:
        raise ValueError(f"Must provide either 'name' or 'type' key for {kind} constraints")
    if isinstance(name, type):
        name = name.__name__.lower()
    elif isinstance(name, str):
        name = name.lower()
    else:
        raise ValueError(f"Unknown {kind} constraint type: {name!r}")

    variants: dict[str, type[Constraints]] = {}
    for attr in vars(namespace).values():
        if isinstance(attr, type) and issubclass(attr, Constraints) and attr is not Constraints:
            variant_name = getattr(attr, "_name", None)
            if isinstance(variant_name, str):
                variants[variant_name.lower()] = attr

    if name not in variants:
        raise ValueError(
            f"Unknown {kind} constraint type: {name!r}; expected one of {sorted(variants)}"
        )
    return variants[name](**d)


C = TypeVar("C", bound=Constraints)


class BaseConstraints(ABC, Generic[C]):
    """
    Base class for constraints.

    Generic over a concrete ``Constraints`` subclass so that subclasses (and the
    type checker) can see the specific fields available on ``self.constraints``.
    Subclasses parameterize like ``BaseConstraints[MyConstraintsType]`` and set
    ``DEFAULT_CONSTRAINTS`` to an instance of that type.
    """

    DEFAULT_CONSTRAINTS: C
    _constraints: C

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._soft_constraint_losses = []
        self._soft_constraint_loss: dict[str, torch.Tensor | float] = {}
        self._iter_constraint_losses: dict[str, float] = {}
        self.constraints = self.DEFAULT_CONSTRAINTS.copy()

    @property
    def soft_constraint_losses(self) -> NDArray[np.float32]:
        return np.array(self._soft_constraint_losses, dtype=np.float32)

    @property
    def soft_constraint_loss(self) -> dict[str, torch.Tensor | float]:
        return self._soft_constraint_loss

    @property
    def constraints(self) -> C:
        """
        Constraints for the model.
        """
        return self._constraints

    @constraints.setter
    def constraints(self, constraints: C | dict[str, Any]):
        """
        Setter for constraints class, can be a Constraints instance or a dictionary.
        Dict keys are validated against the active Constraints dataclass's allowed_keys.
        """
        if isinstance(constraints, Constraints):
            self._constraints = constraints
        elif isinstance(constraints, dict):
            allowed = self._constraints.allowed_keys
            for key, value in constraints.items():
                if key not in allowed:
                    raise KeyError(
                        f"Invalid constraint key '{key}' for {type(self._constraints).__name__}, "
                        f"allowed keys are {allowed}"
                    )
                setattr(self._constraints, key, value)
        else:
            raise ValueError(f"Invalid constraints type: {type(constraints)}")

    def add_constraint(self, key: str, value: Any) -> None:
        """
        Set a single constraint field by name, with validation against allowed_keys.
        """
        allowed = self._constraints.allowed_keys
        if key not in allowed:
            raise KeyError(
                f"Invalid constraint key '{key}' for {type(self._constraints).__name__}, "
                f"allowed keys are {allowed}"
            )
        setattr(self._constraints, key, value)

    # --- helpers for consistent loss logging ---
    def _get_zero_loss_tensor(self) -> torch.Tensor:
        """Helper method to create a zero loss tensor with proper device and dtype."""
        device = getattr(self, "device", "cpu")
        return torch.tensor(0, device=device, dtype=getattr(torch, config.get("dtype_real")))

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
                    v = loss_value.detach().mean().item()
            else:
                v = float(loss_value)
            self._iter_constraint_losses[loss_name] = (
                self._iter_constraint_losses.get(loss_name, 0.0) + v
            )

    def get_iter_constraint_losses(self) -> dict[str, float]:
        return self._iter_constraint_losses

    def reset_iter_constraint_losses(self) -> None:
        self._iter_constraint_losses = {}

    # --- Required methods that need to be implemented in subclasses ---
    @abstractmethod
    def apply_hard_constraints(self, *args, **kwargs) -> torch.Tensor | None:
        """
        Apply hard constraints to the model.

        May return a projected tensor (most models) or ``None`` when the
        implementation mutates state in place (e.g. ``DatasetConstraints``).
        """
        raise NotImplementedError

    @abstractmethod
    def apply_soft_constraints(self, *args, **kwargs) -> torch.Tensor:
        """
        Apply soft constraints to the model.

        Signature is intentionally permissive: ptychography models override with
        positional tensors (e.g. ``(obj, mask)``), while tomography models
        override with a ``ReconstructionContext`` (``(ctx)``).
        """
        raise NotImplementedError
