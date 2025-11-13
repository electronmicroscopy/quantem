from typing import Any, Generator, Iterator, Sequence

import torch

from quantem.tomography.tomography_base import TomographyBase


class TomographyML(TomographyBase):
    """
    Class for handling conventional reconstruction methods of tomography data.
    """

    OPTIMIZABLE_VALS = ["volume", "z1", "x", "z3", "shifts"]
    DEFAULT_LRS = {
        "volume": 1e-2,
        "z1": 1e-1,
        "x": 1e-1,
        "z3": 1e-1,
        "shifts": 1e-1,
        "tv_weight_vol": 0,
        "tv_weight_z1": 0,
        "tv_weight_x": 0,
        "tv_weight_z3": 0,
    }
    DEFAULT_OPTIMIZER_TYPE = "adam"

    # --- Properties ---

    @property
    def optimizer_params(self) -> dict[str, dict]:
        """Returns the parameters used to set the optimizers."""
        return self._optimizer_params

    @optimizer_params.setter
    def optimizer_params(self, d: dict) -> None:
        """
        # Takes a dictionary {key: torch.optim.Adam(params=[blah], lr=[blah]), ...}
        Takes a dictionary:
        {
            "key1": {
                "type": "adam",
                "lr": 0.001,
                },
            "key2": {
                ...
                },
            ...
        }
        """
        # resets _optimizers as well
        self._optimizers = {}
        self._optimizer_params = {}
        if isinstance(d, (tuple, list)):
            d = {k: {} for k in d}

        for k, v in d.items():
            if k not in self.OPTIMIZABLE_VALS:
                raise ValueError(
                    f"key to be optimized, {k}, not in allowed keys: {self.OPTIMIZABLE_VALS}"
                )
            if "type" not in v.keys():
                v["type"] = self.DEFAULT_OPTIMIZER_TYPE
            if "lr" not in v.keys():
                v["lr"] = self.DEFAULT_LRS[k]
            self._optimizer_params[k] = v

    @property
    def optimizers(self) -> dict[str, torch.optim.Adam | torch.optim.AdamW]:
        return self._optimizers

    def set_optimizers(self):
        """Reset all optimizers and set them according to the optimizer_params."""
        for key, _ in self._optimizer_params.items():
            if key == "volume":
                self._add_optimizer(key, self.volume_obj.params, self._optimizer_params[key])
            elif key == "shifts":
                self._add_optimizer(key, self.dataset.shifts, self._optimizer_params[key])
            elif key == "z1":
                self._add_optimizer(key, self.dataset.z1_angles, self._optimizer_params[key])
            elif key == "x":
                self._add_optimizer(key, self.dataset.tilt_angles, self._optimizer_params[key])
            elif key == "z3":
                self._add_optimizer(key, self.dataset.z3_angles, self._optimizer_params[key])
            else:
                raise ValueError(
                    f"key to be optimized, {key}, not in allowed keys: {self.OPTIMIZABLE_VALS}"
                )

    def remove_optimizer(self, key: str) -> None:
        self._optimizers.pop(key, None)
        self._optimizer_params.pop(key, None)
        return

    def _add_optimizer(
        self,
        key: str,
        params: "torch.Tensor|Sequence[torch.Tensor]|Iterator[torch.Tensor]",
        opt_params: dict,
    ) -> None:
        """Can be used to add an optimizer without resetting the other optimizers."""

        if key not in self.OPTIMIZABLE_VALS:
            raise ValueError(
                f"key to be optimized, {key}, not in allowed keys: {self.OPTIMIZABLE_VALS}"
            )
        if isinstance(params, torch.Tensor):
            params = [params]
        elif isinstance(params, Generator):
            params = list(params)
        [p.requires_grad_(True) for p in params]
        self.optimizer_params[key] = opt_params
        opt_params = opt_params.copy()
        opt_type = opt_params.pop("type")
        if isinstance(opt_type, type):
            opt = opt_type(params, **opt_params)
        elif opt_type == "adam":
            opt = torch.optim.Adam(params, **opt_params)
        elif opt_type == "adamw":
            opt = torch.optim.AdamW(params, **opt_params)  # TODO pass all other kwargs
        else:
            raise NotImplementedError(f"Unknown optimizer type: {opt_params['type']}")
        # if key in self.optimizers.keys():
        #     self.vprint(f"Key {key} is already in optimizers, overwriting.")
        self._optimizers[key] = opt

    @property
    def scheduler_params(self) -> dict[str, dict]:
        """Returns the parameters used to set the schedulers."""
        return self._scheduler_params

    @scheduler_params.setter
    def scheduler_params(self, d: dict) -> None:
        """
        Takes a dictionary:
        {
            "key1": {
                "type": "cyclic",
                "base_lr": 0.001,
                },
            "key2": {
                ...
                },
            ...
        }
        """
        self._schedulers = {}
        for k, v in d.items():
            if not any(v):
                continue
            if k not in self.OPTIMIZABLE_VALS:
                raise ValueError(
                    f"key to be optimized, {k}, not in allowed keys: {self.OPTIMIZABLE_VALS}"
                )
            if v["type"] not in ["cyclic", "plateau", "exp", "gamma", "none"]:
                raise ValueError(
                    f"Unknown scheduler type: {v['type']}, expected one of ['cyclic', 'plateau', 'exp', 'gamma', 'none']"
                )
        self._scheduler_params = d

    @property
    def schedulers(
        self,
    ) -> dict[
        str,
        (
            torch.optim.lr_scheduler.CyclicLR
            | torch.optim.lr_scheduler.ReduceLROnPlateau
            | torch.optim.lr_scheduler.ExponentialLR
            | None
        ),
    ]:
        return self._schedulers

    def set_schedulers(
        self,
        params: dict[str, dict],
        num_iter: int | None = None,
    ):
        """
        TODO allow for new schedulers to be passed in when adding new optimizers without
        removing the old schedulers or overwrtiting them. Not entirely sure what usecases there
        will be for this.

        Sets the schedulers for the optimizer from a dictionary. Expects a dictionary of the form:
        {
            "optimizable_key1": {
                "type": "scheduler_type",
                "scheduler_kwarg": scheduler_kwarg_value,
                ...
            },
            "optimizable_key2": {
                "type": "scheduler_type",
                "scheduler_kwarg": scheduler_kwarg_value,
                ...
            },
            ...
        }
        where the keys are the same as the keys in self.OPTIMIZABLE_VALS.

        The scheduler type can be one of the following:
        - "cyclic"
        - "plateau" or "reducelronplateau"
        - "exponential"
        - None

        The num_iter kwarg is only used for exponential schedulers and if a "factor" is given
        as a scheduler_kwarg instead of gamma. In that case, the gamma is calculated from num_iter
        and the factor.

        TODO could update this to allow passing key:optimizer directly, would likely need to
        rewrite get_schedulers to check the tpye
        """
        if not any(self.optimizers):
            raise NameError("self.optimizers have not yet been set.")
        self._schedulers = self._get_schedulers(
            params=params,
            optimizers=self.optimizers,
            num_iter=num_iter,
        )

    def _get_schedulers(
        self,
        params: dict[str, dict],
        optimizers: dict,
        num_iter: int | None = None,
    ) -> dict[
        str,
        (
            torch.optim.lr_scheduler.CyclicLR
            | torch.optim.lr_scheduler.ReduceLROnPlateau
            | torch.optim.lr_scheduler.ExponentialLR
            | None
        ),
    ]:
        """
        return schedulers for a given set of optimizers. Kept seperate from schedulers.setter so
        that it can be called for pre-training
        """
        schedulers = {}
        for opt_key, p in params.items():
            if not any(p):
                continue
            elif opt_key not in self.OPTIMIZABLE_VALS:
                raise KeyError(
                    f"Scheduler got bad key {opt_key}, schedulers can only be attached to one of {self.OPTIMIZABLE_VALS}"
                )
            elif opt_key not in optimizers.keys():
                raise KeyError(f"optimizers does not have an optimizer for: {opt_key}")
            else:
                schedulers[opt_key] = self._get_scheduler(
                    optimizer=optimizers[opt_key], params=p, num_iter=num_iter
                )
        return schedulers

    def _get_scheduler(
        self,
        optimizer: torch.optim.Adam,
        params: dict[str, Any] | torch.optim.lr_scheduler._LRScheduler,
        num_iter: int | None = None,
    ) -> (
        torch.optim.lr_scheduler._LRScheduler
        | torch.optim.lr_scheduler.CyclicLR
        | torch.optim.lr_scheduler.ReduceLROnPlateau
        | torch.optim.lr_scheduler.ExponentialLR
        | None
    ):
        if isinstance(params, torch.optim.lr_scheduler._LRScheduler):
            return params

        sched_type: str = params["type"].lower()
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
                cooldown=params.get("cooldown", 50),
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
        return scheduler
