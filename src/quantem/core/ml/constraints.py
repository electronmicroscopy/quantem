from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Constraints(ABC):
    """
    Needs to be implemented in all object models that inherit from BaseConstraints.
    """

    pass


class BaseConstraints(ABC):
    """
    Base class for constraints.
    """

    # Default constraints are the dataclasses themselves.
    DEFAULT_CONSTRAINTS = Constraints()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._soft_constraint_loss = {}
        self._epoch_constraint_losses = {}
        self.constraints = self.DEFAULT_CONSTRAINTS.copy()

    @property
    def constraints(self) -> Constraints:
        """
        Constraints for the object model.
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
                if key not in self._constraints.allowed_keys:
                    raise ValueError(f"Invalid constraint key: {key}")
                setattr(self._constraints, key, value)
        else:
            raise ValueError(f"Invalid constraints type: {type(constraints)}")

    def add_constraint(self, key: str, value: Any):
        """
        Add a constraint to the constraints class.
        Note the allowed keys should be implemented for each constraint subclass.
        """
        if key not in self._constraints.allowed_keys:
            raise ValueError(f"Invalid constraint key: {key}")
        setattr(self._constraints, key, value)

    # --- Required methods tha tneeds to implemented in subclasses ---
    @abstractmethod
    def apply_hard_constraints(self, *args, **kwargs) -> torch.Tensor:
        """
        Apply hard constraints to the object model.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_soft_constraints(self, *args, **kwargs) -> torch.Tensor:
        """
        Apply soft constraints to the object model.
        """
        raise NotImplementedError
