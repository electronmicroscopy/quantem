from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
import torch.nn as nn

from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.constraints import BaseConstraints, Constraints
from quantem.core.ml.optimizer_mixin import OptimizerMixin
from quantem.core.utils.rng import RNGMixin


@dataclass
class ConstraintsTomography(Constraints):
    """
    Data class for all constraints that can be applied to the object model.
    """

    # Hard Constraints
    positivity: bool = True
    shrinkage: float = 0.0
    circular_mask: bool = False
    fourier_filter: str | None = None  # Hamming, etc...

    # Soft Constraints
    tv_vol: float = 0.0

    @property
    def hard_constraint_keys(self) -> list[str]:
        """
        List of hard constraint keys.
        """
        return ["positivity", "shrinkage", "circular_mask", "fourier_filter"]

    @property
    def soft_constraint_keys(self) -> list[str]:
        """
        List of soft constraint keys.
        """
        return ["tv_vol"]

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
        return f"""Constraints:
        Positivity: {self.positivity}
        Shrinkage: {self.shrinkage}
        Circular Mask: {self.circular_mask}
        Fourier Filter: {self.fourier_filter}
        TV Volume: {self.tv_vol}
        """


class ObjectBase(AutoSerialize, nn.Module, RNGMixin, OptimizerMixin):
    DEFAULT_LRS = {
        "object": 8e-6,
    }
    _token = object()
    """
    Base class for all ObjectModels to inherit from.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use a factory method to instantiate this class.")

        self._shape = shape

        # Initialize dependencies
        nn.Module.__init__(self)
        RNGMixin.__init__(self, rng=rng, device=device)
        OptimizerMixin.__init__(self)

        # Initialize a torch.zeros volume with the given shape
        self._obj = torch.zeros(self._shape, device=device, dtype=torch.float32)

        # --- Properties ---
        @property
        def shape(self) -> tuple[int, int, int]:
            """
            Shape of the object (x, y, z).
            """
            return self._shape

        @shape.setter
        def shape(self, shape: tuple[int, int, int]):
            self._shape = shape

        @property
        def obj(self) -> torch.Tensor:
            """
            Returns the object, should be implemented in subclasses.
            """
            raise NotImplementedError

        @abstractmethod
        def forward(self, *args, **kwargs) -> torch.Tensor:
            """
            Forward pass, should be implemented in subclasses. Note for any nn.Module this is
            a required method.
            """
            raise NotImplementedError

        @abstractmethod
        def reset(self) -> None:
            """
            Reset the object, should be implemented in subclasses.
            """
            raise NotImplementedError

        # --- Helper Functions ---
        def get_optimization_parameters(self) -> list[nn.Parameter]:
            """
            Get the parameters that should be optimized for this model.

            TODO: I have no idea what this does.
            """
            return list[nn.Parameter](self.parameters())

        @abstractmethod  # Each subclass should implement this.
        def to(self, *args, **kwargs):
            """
            Move the object to a device
            """

            raise NotImplementedError


class ObjectConstraints(ObjectBase, BaseConstraints):
    DEFAULT_CONSTRAINTS = ConstraintsTomography()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraints = self.DEFAULT_CONSTRAINTS.copy()

    def apply_hard_constraints(
        self,
        obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply hard constraints to the object model.

        Only hard constraint here is the positivity and shrinkage. TODO: Add the other hard constraints.
        """
        obj2 = obj.clone()
        if self.constraints.positivity:
            obj2 = torch.clamp(obj2, min=0.0, max=None)
        if self.constraints.shrinkage:
            obj2 = torch.max(obj2 - self.constraints.shrinkage, torch.zeros_like(obj2))

        # TODO: Need to implement the other hard constraints: Fourier Filter and Circular Mask.
        return obj2

    def apply_soft_constraints(
        self,
        obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply soft constraints to the object model.

        Only soft constraint here is the TV loss.
        """

        soft_loss = torch.tensor(0.0, device=obj.device, dtype=obj.dtype, requires_grad=True)
        if self.constraints.tv_vol > 0:
            tv_loss = self.get_tv_loss(
                obj.unsqueeze(0).unsqueeze(0), factor=self.constraints.tv_vol
            )
            soft_loss += tv_loss
        return soft_loss

    @abstractmethod
    def get_tv_loss(self, obj: torch.Tensor, tv_weight: float = 0.0) -> torch.Tensor:
        """
        Get the TV loss for the object model. Must be implemented in each subclass.
        """
        raise NotImplementedError


class ObjectPixelated(ObjectConstraints):
    """
    Object model for pixelated objects.

    Supports: Conventional algorithms (SIRT, FBP), and AD-based reconstructions.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        super().__init__(
            shape=shape,
            device=device,
            rng=rng,
            _token=self._token,
        )

    # --- Properties ----
    @property
    def obj(self) -> torch.Tensor:
        return self.apply_hard_constraints(self._obj)

    @obj.setter
    def obj(self, obj: torch.Tensor):
        self._obj = obj

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @shape.setter
    def shape(self, shape: tuple[int, int, int]):
        self._shape = shape

    @property
    def soft_loss(self) -> torch.Tensor:
        return self.apply_soft_constraints(self._obj)

    @property
    def name(self) -> str:
        return "ObjectPixelated"

    @property
    def obj_type(self) -> str:
        return "pixelated"

    # --- Forward method ---
    def forward(self, dummy_input=None) -> torch.Tensor:
        return self.obj

    # --- Defining the TV loss ---
    def get_tv_loss(self, obj: torch.Tensor, tv_weight: float = 1e-3) -> torch.Tensor:
        tv_d = torch.pow(obj[:, :, 1:, :, :] - obj[:, :, :-1, :, :], 2).sum()
        tv_h = torch.pow(obj[:, :, :, 1:, :] - obj[:, :, :, :-1, :], 2).sum()
        tv_w = torch.pow(obj[:, :, :, :, 1:] - obj[:, :, :, :, :-1], 2).sum()
        tv_loss = tv_d + tv_h + tv_w

        return tv_loss * tv_weight / (torch.prod(torch.tensor(obj.shape)))

    # --- Helper Functions ---
    def to(self, device: str):
        self._obj = self._obj.to(device)


class ObjectINR(ObjectConstraints):
    pass


ObjectModelType = ObjectPixelated | ObjectINR
