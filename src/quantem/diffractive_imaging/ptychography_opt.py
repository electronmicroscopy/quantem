from dataclasses import replace
from typing import TYPE_CHECKING, Any

from quantem.core import config
from quantem.core.ml.optimizer_mixin import (
    OptimizerMixin,
    OptimizerParams,
    OptimizerParamsType,
    SchedulerParams,
    SchedulerParamsType,
)
from quantem.diffractive_imaging.ptychography_base import PtychographyBase

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


class PtychographyOpt(PtychographyBase):
    """
    Optimizer/scheduler dispatch layer for `Ptychography`.

    Each optimizable component (`object`, `probe`, `dataset`) lives on its own
    `OptimizerMixin`-equipped model. The methods here are thin façades that fan a
    single dict (`{key: params}`) out to the three models via the `_models` dict.
    """

    OPTIMIZABLE_VALS = ["object", "probe", "dataset"]
    DEFAULT_OPTIMIZER_TYPE: OptimizerParamsType = OptimizerParams.Adam()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _models(self) -> dict[str, OptimizerMixin]:
        """Maps each optimization key to the model that owns its parameters.

        Not cached: `obj_model`, `probe_model`, and `dset` can be reassigned
        (e.g. by `from_ptychography`), and we must always see the current binding.
        """
        return {
            "object": self.obj_model,
            "probe": self.probe_model,
            "dataset": self.dset,
        }

    def _check_key(self, key: str) -> None:
        if key not in self.OPTIMIZABLE_VALS:
            raise ValueError(
                f"key to be optimized, {key}, not in allowed keys: {self.OPTIMIZABLE_VALS}"
            )

    @staticmethod
    def _is_pplr_dict(v: dict) -> bool:
        """Detect a nested per-parameter-group (PPLR) optimizer spec.

        A PPLR dict maps parameter-group keys (e.g. ``"grids"``/``"sigma_net"`` for
        ``ObjectTensorDecomp``) to per-group ``OptimizerParamsType`` or dict specs — as opposed
        to a single-optimizer shorthand like ``{"name": "adam", "lr": 1e-3}``. Such dicts are
        passed through to the owning model untouched so its ``_normalize_optimizer_params`` can
        validate the keys against ``param_keys``.
        """
        return (
            len(v) > 0
            and not OptimizerMixin._is_single_optimizer_dict(v)
            and all(isinstance(val, (OptimizerParamsType, dict)) for val in v.values())
        )

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

    # region --- optimizer params ---

    @property
    def optimizer_params(self) -> dict[str, OptimizerParamsType | dict[str, OptimizerParamsType]]:
        return {
            key: model.optimizer_params
            for key, model in self._models.items()
            if not isinstance(model.optimizer_params, OptimizerParams.NoneOptimizer)
        }

    @optimizer_params.setter
    def optimizer_params(self, d: dict[str, Any] | list[str] | tuple[str, ...]) -> None:
        """
        Takes a dictionary mapping optimizable keys to either an ``OptimizerParamsType``
        dataclass or a plain dict (with optional ``"name"``/``"type"`` and ``"lr"``
        keys). Missing ``"name"`` / ``"lr"`` are filled from ``DEFAULT_OPTIMIZER_TYPE``
        and ``_get_default_lr`` respectively.

        Examples
        --------
        >>> ptycho.optimizer_params = {"object": OptimizerParams.Adam(lr=5e-3)}
        >>> ptycho.optimizer_params = {"object": {"name": "adam", "lr": 5e-3}}
        >>> ptycho.optimizer_params = ["object", "probe"]  # use all defaults
        """
        _d: dict[str, Any] = {k: {} for k in d} if isinstance(d, (list, tuple)) else d

        for k, v in _d.items():
            if isinstance(v, OptimizerParamsType):
                pass  # already a dataclass, pass through
            elif isinstance(v, dict) and self._is_pplr_dict(v):
                pass  # nested per-parameter-group (PPLR) spec -> pass through untouched
            elif isinstance(v, dict):
                if not v:
                    v = replace(self.DEFAULT_OPTIMIZER_TYPE, lr=self._get_default_lr(k))
                else:
                    if "name" not in v and "type" not in v:
                        v["name"] = self.DEFAULT_OPTIMIZER_TYPE._name
                    if "lr" not in v:
                        v["lr"] = self._get_default_lr(k)
            else:
                raise TypeError(
                    f"Expected OptimizerParamsType or dict for key '{k}', got {type(v)}"
                )

            self._models[k].optimizer_params = v  # type: ignore[assignment]

    # endregion --- optimizer params ---

    # region --- optimizers ---

    @property
    def optimizers(self) -> dict[str, "torch.optim.Optimizer"]:
        """Active optimizers, keyed by optimization key."""
        return {
            key: model.optimizer  # type: ignore[reportIncompatibleMethodOverride]
            for key, model in self._models.items()
            if model.has_optimizer()
        }

    def set_optimizers(self) -> None:
        """(Re)create an optimizer on each model whose params are not `NoneOptimizer`."""
        for key, params in self.optimizer_params.items():
            self._models[key].set_optimizer(params)

    def remove_optimizer(self, key: str) -> None:
        """Tear down the optimizer on the model for `key`."""
        self._check_key(key)
        self._models[key].remove_optimizer()

    def step_optimizers(self) -> None:
        for model in self._models.values():
            if model.has_optimizer():
                model.step_optimizer()

    def zero_grad_all(self) -> None:
        for model in self._models.values():
            if model.has_optimizer():
                model.zero_optimizer_grad()

    # endregion --- optimizers ---

    # region --- schedulers ---

    @property
    def scheduler_params(self) -> dict[str, SchedulerParamsType]:
        """Returns the parameters used to set the schedulers."""
        return {
            "object": self.obj_model.scheduler_params,
            "probe": self.probe_model.scheduler_params,
            "dataset": self.dset.scheduler_params,
        }

    @scheduler_params.setter
    def scheduler_params(self, d: dict[str, Any] | list[str] | tuple[str, ...]) -> None:
        """
        Takes a dictionary mapping optimizable keys to either a ``SchedulerParamsType``
        dataclass or a plain dict.  Keys not present in ``d`` are set to
        ``SchedulerParams.NoneScheduler()`` (disables scheduling for that model).

        Examples
        --------
        >>> ptycho.scheduler_params = {"object": SchedulerParams.Plateau(factor=0.5)}
        >>> ptycho.scheduler_params = {"object": {"name": "plateau", "factor": 0.5}}
        """
        _d: dict[str, Any] = {k: {} for k in d} if isinstance(d, (list, tuple)) else dict(d)
        for key in self.OPTIMIZABLE_VALS:
            _d.setdefault(key, SchedulerParams.NoneScheduler())
        for k, v in _d.items():
            self._check_key(k)
            self._models[k].scheduler_params = v

    @property
    def schedulers(self) -> dict[str, "torch.optim.lr_scheduler.LRScheduler"]:
        return {
            key: model.scheduler
            for key, model in self._models.items()
            if model.scheduler is not None
        }

    def set_schedulers(self, params: dict[str, SchedulerParamsType], num_iter: int | None = None):
        """Set schedulers for each model."""
        for key, scheduler_params in params.items():
            self._check_key(key)
            self._models[key].set_scheduler(scheduler_params, num_iter)

    def step_schedulers(self, loss: float | None = None) -> None:
        for model in self._models.values():
            if model.scheduler is not None:
                model.step_scheduler(loss)

    # endregion --- schedulers ---
