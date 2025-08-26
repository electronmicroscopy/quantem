from typing import TYPE_CHECKING

from quantem.core import config
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
    DEFAULT_OPTIMIZER_TYPE = "adam"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._optimizer_params = {}
        # self._scheduler_params = {}

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
    def optimizer_params(self) -> dict[str, dict]:
        return {
            key: params
            for key, params in [
                ("object", self.obj_model.optimizer_params),
                ("probe", self.probe_model.optimizer_params),
                ("dataset", self.dset.optimizer_params),
            ]
            if params
        }

    @optimizer_params.setter
    def optimizer_params(self, d: dict) -> None:
        """
        Takes a dictionary:
        {
            "object": {
                "type": "adam",
                "lr": 0.001,
                },
            "probe": {
                "type": "adam",
                "lr": 0.001,
                },
            "dataset": {
                "type": "adam",
                "lr": 0.001,
                },
            ...
        }
        """
        if isinstance(d, (tuple, list)):
            d = {k: {} for k in d}

        for key in self.OPTIMIZABLE_VALS:
            # if not specified, remove the scheduler for that model
            if key not in d:
                d[key] = {"type": "none"}

        for k, v in d.items():
            if "type" not in v.keys():
                v["type"] = self.DEFAULT_OPTIMIZER_TYPE
            if "lr" not in v.keys():
                v["lr"] = self._get_default_lr(k)
            # self._optimizer_params[k] = v
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
    def scheduler_params(self) -> dict[str, dict]:
        """Returns the parameters used to set the schedulers."""
        return {
            "object": self.obj_model.scheduler_params,
            "probe": self.probe_model.scheduler_params,
            "dataset": self.dset.scheduler_params,
        }

    @scheduler_params.setter
    def scheduler_params(self, d: dict) -> None:
        """
        Takes a dictionary:
        {
            "object": {
                "type": "cyclic",
                "base_lr": 0.001,
                },
            "probe": {
                ...
                },
            ...
        }
        """
        for key in self.OPTIMIZABLE_VALS:
            # if not specified, remove the scheduler for that model
            if key not in d:
                d[key] = {}
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

    def set_schedulers(
        self,
        params: dict[str, dict],
        num_iter: int | None = None,
    ):
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
