from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import torch
from numpy.typing import NDArray


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


class BaseConstraints(ABC):
    """
    Base class for constraints.
    """

    # Default constraints are the dataclasses themselves.
    DEFAULT_CONSTRAINTS = Constraints()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._soft_constraint_losses = []
        self.constraints = self.DEFAULT_CONSTRAINTS.copy()

    @property
    def soft_constraint_losses(self) -> NDArray[np.float32]:
        return np.array(self._soft_constraint_losses, dtype=np.float32)

    @property
    def constraints(self) -> Constraints:
        """
        Constraints for the model.
        """
        return self._constraints

    @constraints.setter
    def constraints(self, constraints: Constraints | dict[str, Any]):
        """
        Setter for constraints class, can be a Constraints instance or a dictionary.
        """
        if isinstance(constraints, Constraints):
            self._constraints = constraints
        elif isinstance(constraints, dict):
            for key, value in constraints.items():
                setattr(self._constraints, key, value)
        else:
            raise ValueError(f"Invalid constraints type: {type(constraints)}")

    # --- Required methods tha tneeds to implemented in subclasses ---
    @abstractmethod
    def apply_hard_constraints(self, *args, **kwargs) -> torch.Tensor:
        """
        Apply hard constraints to the model.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_soft_constraints(self, *args, **kwargs) -> torch.Tensor:
        """
        Apply soft constraints to the model.
        """
        raise NotImplementedError
