from collections.abc import Mapping

import torch

from quantem.core.ml.optimizer_mixin import (
    OptimizerParams,
    OptimizerParamsType,
    SchedulerParamsType,
)
from quantem.tomography.tomography_base import TomographyBase


class TomographyOpt(TomographyBase):
    """
    Class for handling all the optimizers and schedulers for the tomography reconstruction.
    """

    OPTIMIZABLE_VALS = ["object", "pose"]
    DEFAULT_OPTIMIZER_TYPE: OptimizerParamsType = OptimizerParams.Adam()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_default_lr(self, key: str) -> float:
        """Get default learning rate for a given optimization key."""
        if key == "object":
            return self.obj_model.DEFAULT_LRS.get("object", 1e-5)
        elif key == "pose":
            return self.dset.DEFAULT_LRS.get("pose", 5e-2)
        else:
            raise ValueError(f"Unknown optimization key: {key}")

    @property
    def optimizer_params(self) -> dict[str, OptimizerParamsType | dict[str, OptimizerParamsType]]:
        return {
            key: params
            for key, params in [
                ("object", self.obj_model.optimizer_params),
                ("pose", self.dset.optimizer_params),
            ]
            if params
        }

    @optimizer_params.setter
    def optimizer_params(self, d: dict[str, OptimizerParamsType] | dict[str, dict]):
        """Set the optimizer parameters."""
        if isinstance(d, (tuple, list)):
            d = {k: {} for k in d}

        targets = {
            "object": self.obj_model,
            "pose": self.dset,
        }

        for k, v in d.items():
            if k not in targets:
                raise ValueError(f"Unknown optimization key: {k}")

            # if not isinstance(v, OptimizerParamsType):
            #     v = OptimizerParams.parse_dict(v)

            targets[k].optimizer_params = v

    @property
    def optimizers(self) -> dict[str, torch.optim.Optimizer]:
        optimizers = {}

        if self.obj_model.optimizer is not None:
            optimizers["object"] = self.obj_model.optimizer
        if self.dset.optimizer is not None:
            optimizers["pose"] = self.dset.optimizer

        return optimizers

    def set_optimizers(self):
        for key, params in self.optimizer_params.items():
            if key == "object":
                self.obj_model.set_optimizer(params)
            elif key == "pose":
                self.dset.set_optimizer(params)
            else:
                raise ValueError(f"Unknown optimization key: {key}")

    def get_current_lrs(self) -> dict[str, float]:
        if self.obj_model.has_optimizer():
            obj_lr = self.obj_model.get_current_lr()
        else:
            obj_lr = 0.0
        if self.dset.has_optimizer():
            pose_lr = self.dset.get_current_lr()
        else:
            pose_lr = 0.0
        return {
            "object": obj_lr,
            "pose": pose_lr,
        }

    def remove_optimizer(self, key: str):
        if key == "object":
            self.obj_model.remove_optimizer()
        elif key == "pose":
            self.dset.remove_optimizer()
        else:
            raise ValueError(f"Unknown optimization key: {key}")

    @property
    def scheduler_params(self) -> dict[str, SchedulerParamsType]:
        """Returns the parameters used to set the schedulers."""
        return {
            "object": self.obj_model.scheduler_params,
            "pose": self.dset.scheduler_params,
        }

    @scheduler_params.setter
    def scheduler_params(self, d: dict):
        """Set the scheduler parameters."""
        d = dict(d) if d else {}
        self._scheduler_params = d.copy()

        for key in self.OPTIMIZABLE_VALS:
            if key not in d:
                d[key] = {}

        for k, v in d.items():
            if k == "object":
                self.obj_model.scheduler_params = v
            elif k == "pose":
                self.dset.scheduler_params = v
            else:
                raise ValueError(f"Unknown optimization key: {k}")

    @property
    def schedulers(self) -> dict[str, torch.optim.lr_scheduler._LRScheduler]:
        schedulers = {}

        if self.obj_model.scheduler is not None:
            schedulers["object"] = self.obj_model.scheduler
        if self.dset.scheduler is not None:
            schedulers["pose"] = self.dset.scheduler

        return schedulers

    def set_schedulers(
        self, params: Mapping[str, SchedulerParamsType | dict], num_iter: int | None = None
    ):
        for key, scheduler_params in params.items():
            if key == "object":
                self.obj_model.set_scheduler(scheduler_params, num_iter=num_iter)
            elif key == "pose":
                self.dset.set_scheduler(scheduler_params, num_iter=num_iter)
            else:
                raise ValueError(f"Unknown optimization key: {key}")

    def step_optimizers(self):
        for key in self.optimizer_params.keys():
            if key not in self.OPTIMIZABLE_VALS:
                raise ValueError(f"Unknown optimization key: {key}")
            if key == "object" and self.obj_model.has_optimizer():
                self.obj_model.step_optimizer()
            elif key == "pose" and self.dset.has_optimizer():
                self.dset.step_optimizer()

    def zero_grad_all(self):
        for key in self.optimizer_params.keys():
            if key not in self.OPTIMIZABLE_VALS:
                raise ValueError(f"Unknown optimization key: {key}")
            if key == "object" and self.obj_model.has_optimizer():
                self.obj_model.zero_optimizer_grad()
            elif key == "pose" and self.dset.has_optimizer():
                self.dset.zero_optimizer_grad()

    def step_schedulers(self, loss: float | None = None):
        for key in self.scheduler_params.keys():
            if self.obj_model.scheduler is not None and key == "object":
                self.obj_model.step_scheduler(loss)
            if self.dset.scheduler is not None and key == "pose":
                self.dset.step_scheduler(loss)
            if key not in self.OPTIMIZABLE_VALS:
                raise ValueError(f"Unknown optimization key: {key}")
