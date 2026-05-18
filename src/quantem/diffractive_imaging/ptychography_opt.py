from dataclasses import replace
from typing import TYPE_CHECKING

from quantem.core import config
from quantem.core.ml.optimizer_mixin import (
    OptimizerParams,
    OptimizerType,
    SchedulerParams,
    SchedulerType,
)
from quantem.diffractive_imaging.ptychography_base import PtychographyBase

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


class PtychographyOpt(PtychographyBase):
    """
    A class for performing phase retrieval using the Ptychography algorithm.
    """

    OPTIMIZABLE_VALS = ["object", "probe", "dataset"]
    DEFAULT_OPTIMIZER_TYPE: OptimizerType = OptimizerParams.Adam()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_default_lr(self, key: str) -> float:
        """Get default learning rate for a given optimization key."""
        if key == "object":
            return self.obj_model.DEFAULT_LRS.get("object", 5e-3)
        elif key == "probe":
            return self.probe_model.DEFAULT_LRS.get("probe", 1e-3)
        elif key == "dataset":
            return 1e-3  # Dataset model uses different keys, so use fallback
        else:
            raise ValueError(f"Unknown optimization key: {key}")

    # region --- explicit properties and setters ---

    @property
    def optimizer_params(self) -> dict[str, OptimizerType]:
        return {
            key: params
            for key, params in [
                ("object", self.obj_model.optimizer_params),
                ("probe", self.probe_model.optimizer_params),
                ("dataset", self.dset.optimizer_params),
            ]
            if not isinstance(params, OptimizerParams.NoneOptimizer)
        }

    @optimizer_params.setter
    def optimizer_params(self, d: dict) -> None:
        """
        Takes a dictionary mapping optimizable keys to either an ``OptimizerType``
        dataclass or a plain dict (with optional ``"name"``/``"type"`` and ``"lr"``
        keys).  Missing ``"name"`` / ``"lr"`` are filled from ``DEFAULT_OPTIMIZER_TYPE``
        and ``_get_default_lr`` respectively.

        Examples
        --------
        >>> ptycho.optimizer_params = {"object": OptimizerParams.Adam(lr=5e-3)}
        >>> ptycho.optimizer_params = {"object": {"name": "adam", "lr": 5e-3}}
        >>> ptycho.optimizer_params = ["object", "probe"]  # use all defaults
        """
        if isinstance(d, (tuple, list)):
            d = {k: {} for k in d}

        for k, v in d.items():
            if isinstance(v, OptimizerType):
                pass  # already a dataclass, pass through
            elif isinstance(v, dict):
                if not v:
                    v = replace(self.DEFAULT_OPTIMIZER_TYPE, lr=self._get_default_lr(k))
                else:
                    if "name" not in v and "type" not in v:
                        v["name"] = self.DEFAULT_OPTIMIZER_TYPE._name
                    if "lr" not in v:
                        v["lr"] = self._get_default_lr(k)
            else:
                raise TypeError(f"Expected OptimizerType or dict for key '{k}', got {type(v)}")

            if k == "object":
                self.obj_model.optimizer_params = v
            elif k == "probe":
                self.probe_model.optimizer_params = v
            elif k == "dataset":
                self.dset.optimizer_params = v
            else:
                raise ValueError(
                    f"key to be optimized, {k}, not in allowed keys: {self.OPTIMIZABLE_VALS}"
                )

    @property
    def optimizers(self) -> dict[str, "torch.optim.Optimizer"]:
        """Get optimizers from all models."""
        optimizers = {}
        if self.obj_model.has_optimizer():
            optimizers["object"] = self.obj_model.optimizer
        if self.probe_model.has_optimizer():
            optimizers["probe"] = self.probe_model.optimizer
        if self.dset.has_optimizer():
            optimizers["dataset"] = self.dset.optimizer
        return optimizers

    def set_optimizers(self):
        """Set optimizers for each model."""
        for key, params in self.optimizer_params.items():
            if key == "object":
                self.obj_model.set_optimizer(params)
            elif key == "probe":
                self.probe_model.set_optimizer(params)
            elif key == "dataset":
                self.dset.set_optimizer(params)
            else:
                raise ValueError(
                    f"key to be optimized, {key}, not in allowed keys: {self.OPTIMIZABLE_VALS}"
                )

    def remove_optimizer(self, key: str) -> None:
        """Remove optimizer from a specific model."""
        if key == "object":
            self.obj_model.remove_optimizer()
        elif key == "probe":
            self.probe_model.remove_optimizer()
        elif key == "dataset":
            self.dset.remove_optimizer()

    @property
    def scheduler_params(self) -> dict[str, SchedulerType]:
        """Returns the parameters used to set the schedulers."""
        return {
            "object": self.obj_model.scheduler_params,
            "probe": self.probe_model.scheduler_params,
            "dataset": self.dset.scheduler_params,
        }

    @scheduler_params.setter
    def scheduler_params(self, d: dict) -> None:
        """
        Takes a dictionary mapping optimizable keys to either a ``SchedulerType``
        dataclass or a plain dict.  Keys not present in ``d`` are set to
        ``SchedulerParams.NoneScheduler()`` (disables scheduling for that model).

        Examples
        --------
        >>> ptycho.scheduler_params = {"object": SchedulerParams.Plateau(factor=0.5)}
        >>> ptycho.scheduler_params = {"object": {"name": "plateau", "factor": 0.5}}
        """
        for key in self.OPTIMIZABLE_VALS:
            if key not in d:
                d[key] = SchedulerParams.NoneScheduler()
        for k, v in d.items():
            if k == "object":
                self.obj_model.scheduler_params = v
            elif k == "probe":
                self.probe_model.scheduler_params = v
            elif k == "dataset":
                self.dset.scheduler_params = v
            else:
                raise ValueError(
                    f"key to be optimized, {k}, not in allowed keys: {self.OPTIMIZABLE_VALS}"
                )

    @property
    def schedulers(self) -> dict[str, "torch.optim.lr_scheduler._LRScheduler"]:
        """Get schedulers from all models."""
        schedulers = {}
        if self.obj_model.scheduler is not None:
            schedulers["object"] = self.obj_model.scheduler
        if self.probe_model.scheduler is not None:
            schedulers["probe"] = self.probe_model.scheduler
        if self.dset.scheduler is not None:
            schedulers["dataset"] = self.dset.scheduler
        return schedulers

    def set_schedulers(self, params: dict[str, SchedulerType], num_iter: int | None = None):
        """Set schedulers for each model."""
        for key, scheduler_params in params.items():
            if key not in self.OPTIMIZABLE_VALS:
                raise ValueError(
                    f"key to be optimized, {key}, not in allowed keys: {self.OPTIMIZABLE_VALS}"
                )

            if key == "object":
                self.obj_model.set_scheduler(scheduler_params, num_iter)
            elif key == "probe":
                self.probe_model.set_scheduler(scheduler_params, num_iter)
            elif key == "dataset":
                self.dset.set_scheduler(scheduler_params, num_iter)

    def step_optimizers(self):
        """Step all active optimizers."""
        for key in self.optimizer_params.keys():
            if key == "object" and self.obj_model.has_optimizer():
                self.obj_model.step_optimizer()
            elif key == "probe" and self.probe_model.has_optimizer():
                self.probe_model.step_optimizer()
            elif key == "dataset" and self.dset.has_optimizer():
                self.dset.step_optimizer()

    def zero_grad_all(self):
        """Zero gradients for all active optimizers."""
        for key in self.optimizer_params.keys():
            if key == "object" and self.obj_model.has_optimizer():
                self.obj_model.zero_optimizer_grad()
            elif key == "probe" and self.probe_model.has_optimizer():
                self.probe_model.zero_optimizer_grad()
            elif key == "dataset" and self.dset.has_optimizer():
                self.dset.zero_optimizer_grad()

    def step_schedulers(self, loss: float | None = None):
        """Step all active schedulers."""
        for key in self.scheduler_params.keys():
            if key == "object" and self.obj_model.scheduler is not None:
                self.obj_model.step_scheduler(loss)
            elif key == "probe" and self.probe_model.scheduler is not None:
                self.probe_model.step_scheduler(loss)
            elif key == "dataset" and self.dset.scheduler is not None:
                self.dset.step_scheduler(loss)

    # endregion --- explicit properties and setters ---
