import torch

from quantem.tomography.tomography_base import TomographyBase


class TomographyOpt(TomographyBase):
    """
    Class for handling all the optimizers and schedulers for the tomography reconstruction.
    """

    OPTIMIZABLE_VALS = ["object", "pose"]
    DEFAULT_OPTIMIZER_TYPE = "adam"

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
    def optimizer_params(self) -> dict[str, dict]:
        return {
            key: params
            for key, params in [
                ("object", self.obj_model.optimizer_params),
                ("pose", self.dset.optimizer_params),
            ]
            if params
        }

    @optimizer_params.setter
    def optimizer_params(self, d: dict):
        """Set the optimizer parameters."""
        if isinstance(d, (tuple, list)):
            d = {k: {} for k in d}

        for k, v in d.items():
            if "type" not in v.keys():
                v["type"] = self.DEFAULT_OPTIMIZER_TYPE
            if "lr" not in v.keys():
                v["lr"] = self._get_default_lr(k)
            if k == "object":
                self.obj_model.optimizer_params = v
            elif k == "pose":
                self.dset.optimizer_params = v
            else:
                raise ValueError(f"Unknown optimization key: {k}")

    @property
    def optimizers(self) -> dict[str, torch.optim.Optimizer]:
        return {
            "object": self.obj_model.optimizer,
            "pose": self.dset.optimizer,
        }

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
    def scheduler_params(self) -> dict[str, dict]:
        """Returns the parameters used to set the schedulers."""
        return {
            "object": self.obj_model.scheduler_params,
            "pose": self.dset.scheduler_params,
        }

    @scheduler_params.setter
    def scheduler_params(self, d: dict):
        """Set the scheduler parameters."""
        self._scheduler_params = d.copy() if d else {}

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

    def set_schedulers(self, params: dict[str, dict], num_iter: int | None = None):
        for key, scheduler_params in params.items():
            if key == "object":
                self.obj_model.set_scheduler(scheduler_params, num_iter)
            elif key == "pose":
                self.dset.set_scheduler(scheduler_params, num_iter)
            else:
                raise ValueError(f"Unknown optimization key: {key}")

    def step_optimizers(self):
        for key in self.optimizer_params.keys():
            if key == "object" and self.obj_model.has_optimizer():
                self.obj_model.step_optimizer()
            elif key == "pose" and self.dset.has_optimizer():
                self.dset.step_optimizer()
            else:
                raise ValueError(f"Unknown optimization key: {key}")

    def zero_grad_all(self):
        for key in self.optimizer_params.keys():
            if key == "object" and self.obj_model.has_optimizer():
                self.obj_model.zero_optimizer_grad()
            elif key == "pose" and self.dset.has_optimizer():
                self.dset.zero_optimizer_grad()
            else:
                raise ValueError(f"Unknown optimization key: {key}")

    def step_schedulers(self, loss: float | None = None):
        for key in self.scheduler_params.keys():
            if key == "object" and self.obj_model.scheduler is not None:
                self.obj_model.step_scheduler(loss)
            elif self.dset.scheduler is not None and key == "pose":
                self.dset.step_scheduler(loss)
