from abc import abstractmethod
from copy import deepcopy
from typing import Any, Callable, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.blocks import reset_weights
from quantem.core.utils.validators import validate_gt, validate_tensor
from quantem.tomography.utils import get_TV_loss


class ObjectBase(AutoSerialize):
    """
    Base class for all ObjectModels to inherit from.
    """

    def __init__(
        self,
        volume_shape: tuple[int, int, int],
        device: str,
        offset_obj: float = 1e-5,
    ):
        self._shape = volume_shape

        self._obj = torch.zeros(self._shape, device=device, dtype=torch.float32) + offset_obj
        self._offset_obj = offset_obj
        self._device = device
        self._hard_constraints = {}
        self._soft_constraints = {}  # One big dicitonary

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @shape.setter
    def shape(self, shape: tuple[int, int, int]):
        self._shape = shape

    @property
    def offset_obj(self) -> float:
        return self._offset_obj

    @offset_obj.setter
    def offset_obj(self, offset_obj: float):
        self._offset_obj = offset_obj

    @property
    def obj(self) -> torch.Tensor:
        pass

    @obj.setter
    def obj(self, obj: torch.Tensor):
        self._obj = obj

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device: str):
        self._device = device

    @abstractmethod
    def forward(
        self, z1: torch.Tensor, z3: torch.Tensor, shift_x: torch.Tensor, shift_y: torch.Tensor
    ):
        pass

    @abstractmethod
    def obj(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def to(self, device: str):
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def params(self) -> torch.Tensor:
        pass


class ObjectConstraints(ObjectBase):
    DEFAULT_HARD_CONSTRAINTS = {
        "fourier_filter": False,
        "positivity": False,
        "shrinkage": False,
        "circular_mask": False,
    }

    DEFAULT_SOFT_CONSTRAINTS = {
        "tv_vol": 0,
    }

    @property
    def hard_constraints(self) -> dict[str, Any]:
        return self._hard_constraints

    @hard_constraints.setter
    def hard_constraints(self, hard_constraints: dict[str, Any]):
        gkeys = self.DEFAULT_HARD_CONSTRAINTS.keys()
        for key, value in hard_constraints.items():
            if key not in gkeys:  # This might be redundant since add_constraint is checking.
                raise KeyError(f"Invalid object constraint key '{key}', allowed keys are {gkeys}")
            self._hard_constraints[key] = value

    @property
    def soft_constraints(self) -> dict[str, Any]:
        return self._soft_constraints

    @soft_constraints.setter
    def soft_constraints(self, soft_constraints: dict[str, Any]):
        gkeys = self.DEFAULT_SOFT_CONSTRAINTS.keys()
        for key, value in soft_constraints.items():
            if key not in gkeys:
                raise KeyError(f"Invalid object constraint key '{key}', allowed keys are {gkeys}")
            self._soft_constraints[key] = value

    def add_hard_constraint(self, constraint: str, value: Any):
        """Add constraints to the object model."""
        gkeys = self.DEFAULT_HARD_CONSTRAINTS.keys()
        if constraint not in gkeys:
            raise KeyError(
                f"Invalid object constraint key '{constraint}', allowed keys are {gkeys}"
            )
        self._hard_constraints[constraint] = value

    def add_soft_constraint(self, constraint: str, value: Any):
        """Add constraints to the object model."""
        gkeys = self.DEFAULT_SOFT_CONSTRAINTS.keys()
        if constraint not in gkeys:
            raise KeyError(
                f"Invalid object constraint key '{constraint}', allowed keys are {gkeys}"
            )
        self._soft_constraints[constraint] = value

    def apply_hard_constraints(
        self,
        obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply constraints to the object model.
        """
        obj2 = obj.clone()
        if self.hard_constraints["positivity"]:
            obj2 = torch.clamp(obj, min=0.0, max=None)
        if self.hard_constraints["shrinkage"]:
            obj2 = torch.max(obj2 - self.hard_constraints["shrinkage"], torch.zeros_like(obj2))

        return obj2

    def apply_soft_constraints(
        self,
        obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        'Applies' soft constraints to the object model. This will return additional loss terms.
        """
        soft_loss = torch.tensor(0.0, device=obj.device, dtype=obj.dtype, requires_grad=True)
        if self.soft_constraints["tv_vol"] > 0:
            tv_loss = get_TV_loss(
                obj.unsqueeze(0).unsqueeze(0), factor=self.soft_constraints["tv_vol"]
            )

            soft_loss += tv_loss

        return soft_loss


class ObjectVoxelwise(ObjectConstraints):
    """
    Object model for voxelwise objects.
    """

    def __init__(
        self,
        volume_shape: tuple[int, int, int],
        device: str,
        initial_volume: torch.Tensor | None = None,
    ):
        super().__init__(
            volume_shape=volume_shape,
            device=device,
        )
        self.hard_constraints = self.DEFAULT_HARD_CONSTRAINTS.copy()
        self.soft_constraints = self.DEFAULT_SOFT_CONSTRAINTS.copy()

        if initial_volume is not None:
            self._initial_obj = initial_volume
        else:
            self.initial_obj = (
                torch.zeros(self._shape, device=self._device, dtype=torch.float32)
                + self.offset_obj
            )

    @property
    def obj(self):
        return self.apply_hard_constraints(self._obj)

    @obj.setter
    def obj(self, obj: torch.Tensor):
        self._obj = obj

    @property
    def initial_obj(self):
        return self._initial_obj

    @initial_obj.setter
    def initial_obj(self, initial_obj: torch.Tensor):
        if not isinstance(initial_obj, torch.Tensor):
            raise ValueError("initial_obj must be a torch.Tensor")

        self._initial_obj = initial_obj

    def forward(self):
        return self.obj

    def reset(self):
        self._obj = (
            torch.zeros(self._shape, device=self._device, dtype=torch.float32) + self.offset_obj
        )

    def to(self, device: str):
        self._device = device
        self._obj = self._obj.to(self._device)

    @property
    def name(self) -> str:
        return "ObjectVoxelwise"

    @property
    def params(self) -> torch.Tensor:
        return self._obj

    @property
    def soft_loss(self) -> torch.Tensor:
        return self.apply_soft_constraints(self._obj)


class ObjectDIP(ObjectConstraints):
    """
    Object model for DIP objects.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        volume_shape: tuple[int, int, int],
        model_input: torch.Tensor
        | None = None,  # Determines output size, model input pretraining target
        input_noise_std: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__(
            volume_shape=volume_shape,
            device=device,
        )
        self.hard_constraints = self.DEFAULT_HARD_CONSTRAINTS.copy()
        self.soft_constraints = self.DEFAULT_SOFT_CONSTRAINTS.copy()

        if model_input is None:
            self.model_input = torch.randn(1, 1, volume_shape[0], volume_shape[1], volume_shape[2])
        else:
            self.model_input = model_input.clone().detach()

        self.pretrain_target = model_input.clone().detach()

        self._model = model
        self._optimizer = None
        self._scheduler = None
        self._pretrain_losses = []
        self._pretrain_lrs = []
        self._model_input_noise_std = input_noise_std

    @property
    def name(self) -> str:
        return "ObjectDIP"

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Model must be a torch.nn.Module, got {type(model)}")
        self._model = model.to(self._device)
        self.set_pretrained_weights(self._model)

    @property
    def pretrained_weights(self) -> dict[str, torch.Tensor]:
        return self._pretrained_weights

    def set_pretrained_weights(self, model: torch.nn.Module):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Pretrained model must be a torch.nn.Module, got {type(model)}")
        self._pretrained_weights = deepcopy(model.state_dict())

    @property
    def model_input(self) -> torch.Tensor:
        return self._model_input

    @model_input.setter
    def model_input(self, input_tensor: torch.Tensor):
        inp = validate_tensor(
            input_tensor,
            name="model_input",
            dtype=torch.float32,
            ndim=5,
            expand_dims=True,
        )
        self._model_input = inp.to(self._device)

    @property
    def pretrain_target(self) -> torch.Tensor:
        return self._pretrain_target

    @pretrain_target.setter
    def pretrain_target(self, target: torch.Tensor):
        if target.ndim == 5:
            target = target.squeeze(0).squeeze(0)

        target = validate_tensor(
            target,
            name="pretrain_target",
            ndim=3,
            dtype=torch.float32,
            expand_dims=True,
        )
        if target.shape[-3:] != self.model_input.shape[-3:]:
            raise ValueError(
                f"Pretrain target shape {target.shape} does not match model input shape {self.model_input.shape}"
            )
        self._pretrain_target = target.to(self._device)

    @property
    def _model_input_noise_std(self) -> float:
        """standard deviation of the gaussian noise added to the model input each forward call"""
        return self._input_noise_std

    @_model_input_noise_std.setter
    def _model_input_noise_std(self, std: float):
        validate_gt(std, 0.0, "input_noise_std", geq=True)
        self._input_noise_std = std

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """get the optimizer for the DIP model"""
        if self._optimizer is None:
            raise ValueError("Optimizer is not set. Use set_optimizer() to set it.")
        return self._optimizer

    def set_optimizer(self, opt_params: dict):
        opt_type = opt_params.pop("type")
        if isinstance(opt_type, torch.optim.Optimizer):
            self._optimizer = opt_type
        elif isinstance(opt_type, type):
            self._optimizer = opt_type(self.model.parameters(), **opt_params)
        elif opt_type == "adam":
            self._optimizer = torch.optim.Adam(self.model.parameters(), **opt_params)
        elif opt_type == "adamw":
            self._optimizer = torch.optim.AdamW(self.model.parameters(), **opt_params)
        elif opt_type == "sgd":
            self._optimizer = torch.optim.SGD(self.model.parameters(), **opt_params)
        else:
            raise NotImplementedError(f"Unknown optimizer type: {opt_params['type']}")

    @property
    def scheduler(
        self,
    ) -> (
        torch.optim.lr_scheduler._LRScheduler
        | torch.optim.lr_scheduler.CyclicLR
        | torch.optim.lr_scheduler.ReduceLROnPlateau
        | torch.optim.lr_scheduler.ExponentialLR
        | None
    ):
        return self._scheduler

    def set_scheduler(self, params: dict, num_iter: int | None = None) -> None:
        sched_type: str = params["type"].lower()
        optimizer = self.optimizer
        base_LR = optimizer.param_groups[0]["lr"]
        if sched_type == "none":
            scheduler = None
        elif sched_type == "cyclic":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=params.get("base_lr", base_LR / 4),
                max_lr=params.get("max_lr", base_LR * 4),
                step_size_up=params.get("step_size_up", 100),
                mode=params.get("mode", "triangular2"),
                cycle_momentum=params.get("momentum", False),
            )
        elif sched_type.startswith(("plat", "reducelronplat")):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=params.get("factor", 0.5),
                patience=params.get("patience", 10),
                threshold=params.get("threshold", 1e-3),
                min_lr=params.get("min_lr", base_LR / 20),
                cooldown=params.get("cooldown", 20),
            )
        elif sched_type in ["exp", "gamma", "exponential"]:
            if "gamma" in params.keys():
                gamma = params["gamma"]
            elif num_iter is not None:
                fac = params.get("factor", 0.01)
                gamma = fac ** (1.0 / num_iter)
            else:
                gamma = 0.999
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")
        self._scheduler = scheduler

    @property
    def pretrain_losses(self) -> np.ndarray:
        return np.array(self._pretrain_losses)

    @property
    def pretrain_lrs(self) -> np.ndarray:
        return np.array(self._pretrain_lrs)

    @property
    def obj(self):
        obj = self.model(self._model_input)[0]
        return self.apply_hard_constraints(obj)

    def forward(self):
        return self.model(self._model_input)

    def to(self, device: str):
        self.device = device
        self._model = self._model.to(self.device)
        self._model_input = self._model_input.to(self.device)
        self._pretrain_target = self._pretrain_target.to(self.device)

    @property
    def params(self):
        return self._model.parameters()

    def reset(self):
        self.model.load_state_dict(self.pretrained_weights.copy())

    def pretrain(
        self,
        model_input: torch.Tensor,
        pretrain_target: torch.Tensor,
        reset: bool = True,
        num_epochs: int = 100,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        loss_fn: Callable | str = "l2",
        apply_constraints: bool = False,
        show: bool = True,
    ):
        model_input.to(self.device)
        pretrain_target.to(self.device)

        if optimizer_params is not None:
            self.set_optimizer(optimizer_params)

        if scheduler_params is not None:
            self.set_scheduler(scheduler_params, num_epochs)

        if reset:
            self._model.apply(reset_weights)
            self._pretrain_losses = []
            self._pretrain_lrs = []

        if model_input is not None:
            self.model_input = model_input

        if pretrain_target.shape[-3:] != self.model_input.shape[-3:]:
            raise ValueError(
                f"Pretrain target shape {pretrain_target.shape} does not match model input shape {self.model_input.shape}"
            )
        self.pretrain_target = pretrain_target.clone().detach().to(self.device)

        loss_fn = torch.nn.functional.mse_loss

        self._pretrain(
            num_epochs=num_epochs,
            loss_fn=loss_fn,
            apply_constraints=apply_constraints,
            show=show,
        )
        self.set_pretrained_weights(self.model)

    def _pretrain(
        self,
        num_epochs: int,
        loss_fn: Callable,
        apply_constraints: bool = False,
        show: bool = False,
    ):
        if not hasattr(self, "pretrain_target"):
            raise ValueError("Pretrain target is not set. Use pretrain_target to set it.")

        self.model.train()
        optimizer = self.optimizer
        sch = self.scheduler
        pbar = tqdm(range(num_epochs))
        output = self.obj

        for a0 in pbar:
            if apply_constraints:
                output = self.obj
            else:
                output = self.model(self.model_input).squeeze(0).squeeze(0)

            loss = loss_fn(output, self.pretrain_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if sch is not None:
                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sch.step(loss.item())
                else:
                    sch.step()

            self._pretrain_losses.append(loss.item())
            self._pretrain_lrs.append(optimizer.param_groups[0]["lr"])
            pbar.set_description(f"Epoch {a0 + 1}/{num_epochs}, Loss: {loss.item():.4f}, ")


ObjectModelType = Union[ObjectVoxelwise]  # | ObjectDIP | ObjectImplicit (ObjectFFN?)
