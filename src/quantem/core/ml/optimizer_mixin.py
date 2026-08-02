import textwrap
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from quantem.core import config

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


class OptimizerParams:
    """
    Container for optimizer parameter dataclasses.

    Each nested class configures a specific PyTorch optimizer and can be passed
    directly to ``OptimizerMixin.set_optimizer``, or constructed from a dict via
    ``OptimizerParams.parse_dict``.

    Supported optimizers
    --------------------
    Adam
        ``torch.optim.Adam`` — adaptive moment estimation.
    AdamW
        ``torch.optim.AdamW`` — Adam with decoupled weight decay.
    SGD
        ``torch.optim.SGD`` — stochastic gradient descent with optional momentum and Nesterov.
    NoneOptimizer
        Sentinel that disables / removes the optimizer.

    Examples
    --------
    >>> obj.set_optimizer(OptimizerParams.Adam(lr=1e-4))
    >>> obj.set_optimizer({"name": "adam", "lr": 1e-4})  # equivalent dict form
    """

    @dataclass
    class Adam:
        """
        Adam optimizer (``torch.optim.Adam``).

        Parameters
        ----------
        lr : float
            Learning rate. Default: 1e-3.
        betas : tuple[float, float]
            Coefficients for computing running averages of the gradient and its square.
            Default: (0.9, 0.999).
        eps : float
            Term added to the denominator for numerical stability. Default: 1e-8.
        weight_decay : float
            L2 regularisation penalty. Default: 0.
        """

        lr: float = 1e-3
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8
        weight_decay: float = 0
        _name: str = "adam"

        def params(self) -> dict:
            return {
                "lr": self.lr,
                "betas": self.betas,
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            }

        def __str__(self) -> str:
            return textwrap.dedent(f"""
                OptimizerParams.Adam(
                    lr = {self.lr},
                    betas = {self.betas},
                    eps = {self.eps},
                    weight_decay = {self.weight_decay},
                )
            """).strip()

    @dataclass
    class AdamW:
        """
        AdamW optimizer (``torch.optim.AdamW``).

        Identical to Adam but applies weight decay directly to the parameters
        rather than folding it into the gradient update (decoupled weight decay).

        Parameters
        ----------
        lr : float
            Learning rate. Default: 1e-3.
        betas : tuple[float, float]
            Coefficients for computing running averages of the gradient and its square.
            Default: (0.9, 0.999).
        eps : float
            Term added to the denominator for numerical stability. Default: 1e-8.
        weight_decay : float
            Decoupled L2 regularisation penalty. Default: 0.
        """

        lr: float = 1e-3
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8
        weight_decay: float = 0
        _name: str = "adamw"

        def params(self) -> dict:
            return {
                "lr": self.lr,
                "betas": self.betas,
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            }

        def __str__(self) -> str:
            return textwrap.dedent(f"""
                OptimizerParams.AdamW(
                    lr = {self.lr},
                    betas = {self.betas},
                    eps = {self.eps},
                    weight_decay = {self.weight_decay},
                )
            """).strip()

    @dataclass
    class SGD:
        """
        SGD optimizer (``torch.optim.SGD``).

        Parameters
        ----------
        lr : float
            Learning rate. Default: 1e-3.
        momentum : float
            Momentum factor. Default: 0.
        dampening : float
            Dampening for momentum. Default: 0.
        weight_decay : float
            L2 regularisation penalty. Default: 0.
        nesterov : bool
            Enables Nesterov momentum. Default: False.
        """

        lr: float = 1e-3
        momentum: float = 0
        dampening: float = 0
        weight_decay: float = 0
        nesterov: bool = False
        _name: str = "sgd"

        def params(self) -> dict:
            return {
                "lr": self.lr,
                "momentum": self.momentum,
                "dampening": self.dampening,
                "weight_decay": self.weight_decay,
                "nesterov": self.nesterov,
            }

        def __str__(self) -> str:
            return textwrap.dedent(f"""
                OptimizerParams.SGD(
                    lr = {self.lr},
                    momentum = {self.momentum},
                    dampening = {self.dampening},
                    weight_decay = {self.weight_decay},
                    nesterov = {self.nesterov},
                )
            """).strip()

    @dataclass
    class NoneOptimizer:
        """
        Sentinel optimizer that disables optimization.

        Passing this to ``set_optimizer`` will call ``remove_optimizer``,
        clearing both the optimizer and scheduler.
        """

        _name: str = "none"

        def params(self) -> dict:
            return {}

        def __str__(self) -> str:
            return textwrap.dedent("""
                OptimizerParams.NoneOptimizer()
            """).strip()

    @classmethod
    def parse_dict(cls, d: dict):
        """
        Parse dictionary to a optimizer params object.
        Accepts either ``"name"`` or ``"type"`` as the optimizer key.
        """
        d = dict(d)  # avoid mutating caller's dict
        name = d.pop("name", None)
        type_ = d.pop("type", None)
        name = name or type_
        if name is None:
            raise ValueError("Must provide either 'name' or 'type' key")
        if isinstance(name, type):
            name = name.__name__.lower()
        elif isinstance(name, str):
            name = name.lower()
        else:
            raise ValueError(f"Unknown optimizer type: {name}")
        if name == "adam":
            return OptimizerParams.Adam(**d)
        elif name == "adamw":
            return OptimizerParams.AdamW(**d)
        elif name == "sgd":
            return OptimizerParams.SGD(**d)
        elif name == "none":
            return OptimizerParams.NoneOptimizer()
        else:
            raise ValueError(f"Unknown optimizer type: {name.lower()}")


OptimizerParamsType = (
    OptimizerParams.Adam
    | OptimizerParams.AdamW
    | OptimizerParams.SGD
    | OptimizerParams.NoneOptimizer
)


class SchedulerParams:
    """
    Container for learning-rate scheduler parameter dataclasses.

    Each nested class configures a specific PyTorch LR scheduler and can be passed
    directly to ``OptimizerMixin.set_scheduler``, or constructed from a dict via
    ``SchedulerParams.parse_dict``.

    Supported schedulers
    --------------------
    Plateau
        ``torch.optim.lr_scheduler.ReduceLROnPlateau`` — reduce LR when a metric stops improving.
    Exponential
        ``torch.optim.lr_scheduler.ExponentialLR`` — multiply LR by ``gamma`` each step.
    Cyclic
        ``torch.optim.lr_scheduler.CyclicLR`` — cycle LR between ``base_lr`` and ``max_lr``.
    Linear
        ``torch.optim.lr_scheduler.LinearLR`` — linearly interpolate LR over ``total_iters`` steps.
    CosineAnnealing
        ``torch.optim.lr_scheduler.CosineAnnealingLR`` — cosine-annealing LR schedule.
    NoneScheduler
        Sentinel that disables / removes the scheduler.

    Examples
    --------
    >>> obj.set_scheduler(SchedulerParams.Plateau(factor=0.5, patience=10, cooldown=20))
    >>> obj.set_scheduler({"name": "plateau", "factor": 0.5})  # equivalent dict form
    """

    @dataclass
    class Plateau:
        """
        ReduceLROnPlateau scheduler (``torch.optim.lr_scheduler.ReduceLROnPlateau``).

        Reduces the learning rate when a monitored metric stops improving.

        Parameters
        ----------
        mode : {'min', 'max'}
            Whether the monitored metric should be minimised or maximised. Default: 'min'.
        min_lr_factor : float
            Sets ``min_lr = min_lr_factor * base_lr`` when ``min_lr`` is not provided.
            Default: 1/20.
        min_lr : float or None
            Absolute lower bound on the learning rate. Overrides ``min_lr_factor`` when set.
            Default: None.
        factor :     float
            Factor by which the LR is reduced: ``new_lr = lr * factor``. Default: 0.5.
        patience : int
            Number of epochs with no improvement before reducing LR. Default: 10.
        threshold : float
            Minimum change in the monitored metric to qualify as an improvement. Default: 1e-5.
        cooldown : int
            Number of epochs to wait after a LR reduction before resuming normal operation.
            Default: 50.
        """

        mode: Literal["min", "max"] = "min"
        min_lr_factor: float = 1 / 20
        min_lr: float | None = None
        factor: float = 0.5
        patience: int = 10
        threshold: float = 1e-5
        cooldown: int = 50
        _name: str = "plateau"

        def params(self, base_LR: float, num_iter: int | None = None) -> dict:
            if self.min_lr is None:
                self.min_lr = self.min_lr_factor * base_LR
            return {
                "mode": self.mode,
                "factor": self.factor,
                "patience": self.patience,
                "threshold": self.threshold,
                "min_lr": self.min_lr,
                "cooldown": self.cooldown,
            }

    @dataclass
    class Exponential:
        """
        ExponentialLR scheduler (``torch.optim.lr_scheduler.ExponentialLR``).

        Multiplies the learning rate by ``gamma`` after each step.

        Parameters
        ----------
        gamma : float
            Multiplicative decay factor applied each step. Default: 0.9.
        factor : float or None
            Reserved for future use. Default: 0.5.
        num_iter : int or None
            Reserved for future use. Default: None.
        """

        gamma: float = 0.9
        factor: float | None = None
        num_iter: int | None = None
        _name: str = "exponential"

        def params(self, base_LR: float, num_iter: int | None = None) -> dict:
            effective_num_iter = self.num_iter if self.num_iter is not None else num_iter
            if effective_num_iter is None:
                raise ValueError("num_iter must be set if num_iter is not provided")

            self.num_iter = effective_num_iter

            if self.factor is not None:
                self.gamma = self.factor ** (1.0 / effective_num_iter)

            return {
                "gamma": self.gamma,
            }

    @dataclass
    class Cyclic:
        """
        CyclicLR scheduler (``torch.optim.lr_scheduler.CyclicLR``).

        Cycles the learning rate between a lower bound (``base_lr``) and an upper
        bound (``max_lr``). Bounds can be set directly or derived from the optimizer's
        base LR via the factor parameters.

        Parameters
        ----------
        base_lr_factor : float
            Sets ``base_lr = base_lr_factor * optimizer_lr`` when ``base_lr`` is not provided.
            Default: 1/4.
        max_lr_factor : float
            Sets ``max_lr = max_lr_factor * optimizer_lr`` when ``max_lr`` is not provided.
            Default: 4.
        base_lr : float or None
            Absolute lower bound of the LR cycle. Overrides ``base_lr_factor`` when set.
            Default: None.
        max_lr : float or None
            Absolute upper bound of the LR cycle. Overrides ``max_lr_factor`` when set.
            Default: None.
        step_size_up : int
            Number of steps in the increasing half of each cycle. Default: 100.
        step_size_down : int
            Number of steps in the decreasing half of each cycle. Default: 100.
        mode : {'triangular2', 'triangular', 'exp_range'}
            Cycling policy. Default: 'triangular2'.
        cycle_momentum : bool
            If True, cycles momentum inversely to the learning rate. Default: False.
        """

        base_lr_factor: float = 1 / 4
        max_lr_factor: float = 4
        base_lr: float | None = None
        max_lr: float | None = None
        step_size_up: int = 100
        step_size_down: int = 100
        mode: Literal["triangular2", "triangular", "exp_range"] = "triangular2"
        cycle_momentum: bool = False
        _name: str = "cyclic"

        def params(self, base_LR: float, num_iter: int | None = None) -> dict:
            if self.base_lr is None:
                self.base_lr = self.base_lr_factor * base_LR
            if self.max_lr is None:
                self.max_lr = self.max_lr_factor * base_LR
            return {
                "base_lr": self.base_lr,
                "max_lr": self.max_lr,
                "step_size_up": self.step_size_up,
                "step_size_down": self.step_size_down,
                "mode": self.mode,
                "cycle_momentum": self.cycle_momentum,
            }

    @dataclass
    class Linear:
        """
        LinearLR scheduler (``torch.optim.lr_scheduler.LinearLR``).

        Linearly interpolates the learning rate from ``start_factor * base_lr`` to
        ``end_factor * base_lr`` over ``total_iters`` steps.

        Parameters
        ----------
        total_iters : int
            Number of steps over which to interpolate the LR. Required.
        start_factor : float
            Multiplicative factor applied to the LR at the first step. Default: 0.1.
        end_factor : float
            Multiplicative factor applied to the LR at the final step. Default: 1.0.
        """

        total_iters: int | None = None
        start_factor: float = 0.1
        end_factor: float = 1.0
        _name: str = "linear"

        def params(self, base_LR: float, num_iter: int | None = None) -> dict:
            if num_iter is None and self.total_iters is None:
                raise ValueError(
                    "total_iters must be set if num_iter is not provided"
                )  # Should never be reached
            if self.total_iters is None:
                self.total_iters = num_iter
            return {
                "start_factor": self.start_factor,
                "end_factor": self.end_factor,
                "total_iters": self.total_iters,
            }

    @dataclass
    class CosineAnnealing:
        """
        CosineAnnealingLR scheduler (``torch.optim.lr_scheduler.CosineAnnealingLR``).

        Anneals the learning rate following a cosine curve from the base LR down to
        ``eta_min`` over ``T_max`` steps, then restarts.

        Parameters
        ----------
        T_max : int
            Maximum number of iterations (half-period of the cosine cycle). Required.
        eta_min : float
            Minimum learning rate at the trough of the cosine curve. Default: 1e-7.
        """

        eta_min: float = 1e-7
        T_max: int | None = None
        _name: str = "cosine_annealing"

        def params(self, base_LR: float, num_iter: int | None = None) -> dict:
            if num_iter is None and self.T_max is None:
                raise ValueError(
                    "T_max must be set if num_iter is not provided"
                )  # Should never be reached
            if self.T_max is None:
                self.T_max = num_iter
            return {
                "T_max": self.T_max,
                "eta_min": self.eta_min,
            }

    @dataclass
    class NoneScheduler:
        """
        Sentinel scheduler that disables LR scheduling.

        Passing this to ``set_scheduler`` clears the active scheduler without
        affecting the optimizer.
        """

        _name: str = "none"

        def params(self, base_LR: float, num_iter: int | None = None) -> dict:
            return {}

    @classmethod
    def parse_dict(cls, d: dict):
        """
        Parse dictionary to a scheduler params object.
        Accepts either ``"name"`` or ``"type"`` as the scheduler key.
        """
        d = dict(d)  # avoid mutating caller's dict
        name = d.pop("name", None)
        type_ = d.pop("type", None)
        name = name or type_
        if name is None:
            name = "none"
        if isinstance(name, type):
            name = name.__name__.lower()
        elif isinstance(name, str):
            name = name.lower()
        else:
            raise ValueError(f"Unknown scheduler type: {name}")
        if name == "plateau":
            return SchedulerParams.Plateau(**d)
        elif name == "exponential":
            return SchedulerParams.Exponential(**d)
        elif name == "cyclic":
            return SchedulerParams.Cyclic(**d)
        elif name == "linear":
            return SchedulerParams.Linear(**d)
        elif name == "cosine_annealing":
            return SchedulerParams.CosineAnnealing(**d)
        elif name == "none":
            return SchedulerParams.NoneScheduler()
        else:
            raise ValueError(f"Unknown scheduler type: {name}")


SchedulerParamsType = (
    SchedulerParams.Plateau
    | SchedulerParams.Exponential
    | SchedulerParams.Cyclic
    | SchedulerParams.Linear
    | SchedulerParams.CosineAnnealing
    | SchedulerParams.NoneScheduler
)


class OptimizerMixin:
    """
    Mixin class for handling optimizer and scheduler management.
    Each model (object, probe, dataset) can inherit from this to manage its own optimizers.
    """

    DEFAULT_OPTIMIZER_TYPE = "adamw"
    DEFAULT_OPTIMIZER_KEY = "default"

    def __init__(self):
        """Initialize the optimizer mixin."""
        self._optimizer = None
        self._scheduler = None
        self._optimizer_params: dict[str, OptimizerParamsType] = {
            self.DEFAULT_OPTIMIZER_KEY: OptimizerParams.NoneOptimizer()
        }
        self._scheduler_params: SchedulerParamsType = SchedulerParams.NoneScheduler()
        # Don't call super().__init__() in mixin classes to avoid MRO issues

    @property
    def optimizer(self) -> "torch.optim.Optimizer | None":
        """Get the optimizer for this model."""
        return self._optimizer

    @property
    def scheduler(self) -> "torch.optim.lr_scheduler.LRScheduler | None":
        """Get the scheduler for this model."""
        return self._scheduler

    @property
    def optimizer_params(self) -> dict[str, OptimizerParamsType]:
        """Get the optimizer parameters."""
        return self._optimizer_params

    @optimizer_params.setter
    def optimizer_params(
        self, params: OptimizerParamsType | dict[str, OptimizerParamsType] | dict[str, Any]
    ) -> None:
        self._optimizer_params = self._normalize_optimizer_params(params)

    def _normalize_optimizer_params(
        self, params: OptimizerParamsType | dict[str, Any]
    ) -> dict[str, OptimizerParamsType]:
        """Normalize input to dict[str, OptimizerParamsType]. Subclasses can override to validate keys."""
        # Single optimizer, already an OptimizerParamsType
        if isinstance(params, OptimizerParamsType):
            return {self.DEFAULT_OPTIMIZER_KEY: params}
        if not isinstance(params, dict):
            raise TypeError(f"optimizer_params must be OptimizerParamsType or dict, got {type(params)}")
        # Single optimizer as dict shorthand, e.g. {"name": "adam", "lr": 1e-3}
        if self._is_single_optimizer_dict(params):
            return {self.DEFAULT_OPTIMIZER_KEY: OptimizerParams.parse_dict(d=params)}
        # dict-of-OptimizerParamsType form (PPLR)
        return {
            k: v if isinstance(v, OptimizerParamsType) else OptimizerParams.parse_dict(d=v)
            for k, v in params.items()
        }

    @staticmethod
    def _is_single_optimizer_dict(d: dict) -> bool:
        return "type" in d or "name" in d

    @property
    def scheduler_params(self) -> SchedulerParamsType:
        """Get the scheduler parameters."""
        return self._scheduler_params

    @scheduler_params.setter
    def scheduler_params(self, params: SchedulerParamsType | dict):
        """Set the scheduler parameters."""
        if isinstance(params, dict):
            params = SchedulerParams.parse_dict(d=params)
        if not isinstance(params, SchedulerParamsType):
            raise TypeError(f"scheduler parameters must be a SchedulerParamsType, got {type(params)}")
        self._scheduler_params = params

    @abstractmethod
    def get_optimization_parameters(
        self,
    ) -> "dict[str, list[torch.Tensor]]":
        """
        Get the parameters that should be optimized for this model, grouped by name.

        Returns a mapping ``{group_key: [tensors]}``. The group keys MUST match the keys of
        ``optimizer_params`` (the common single-group case uses ``DEFAULT_OPTIMIZER_KEY``).
        ``set_optimizer`` joins each group to its optimizer spec by key and bakes the per-group
        hyperparameters (``spec.params()``) into the torch param group — implementations return
        only the tensors, NOT pre-baked hyperparameters. Return ``{}`` when there is nothing
        to optimize.
        """
        raise NotImplementedError("Subclasses must implement get_optimization_parameters")

    def set_optimizer(self, opt_params: OptimizerParamsType | dict | None = None) -> None:
        """
        Set the optimizer for this model, supporting per-parameter-group learning rates (PPLR).

        ``optimizer_params`` is a ``dict[str, OptimizerParamsType]`` keyed by parameter group. Each
        group's spec is joined by key to the tensors returned by ``get_optimization_parameters()``
        and its hyperparameters are baked into the corresponding torch param group here. All
        groups must use the same optimizer class. If every group is a ``NoneOptimizer`` (or there
        are no groups), the optimizer is removed.
        """
        if opt_params is not None:
            self.optimizer_params = opt_params

        # Single canonical "disable" path: drop NoneOptimizer sentinels and, if nothing is left,
        # remove the optimizer. Done BEFORE get_optimization_parameters() because some models
        # (e.g. the dataset model) raise / return nothing when there is nothing to optimize.
        specs = {
            key: spec
            for key, spec in self.optimizer_params.items()
            if not isinstance(spec, OptimizerParams.NoneOptimizer)
        }
        if not specs:
            self.remove_optimizer()
            return

        # All groups must agree on the optimizer class.
        spec_list = list(specs.values())
        for spec in spec_list[1:]:
            if type(spec) is not type(spec_list[0]):
                raise ValueError(
                    f"All parameter groups must use the same optimizer type, "
                    f"got {type(spec_list[0]).__name__} and {type(spec).__name__}"
                )

        # Join specs to param groups by key; bake each group's hyperparameters here.
        groups = self.get_optimization_parameters()  # dict[str, list[tensor]]
        if set(groups) != set(specs):
            raise ValueError(
                f"optimizer_params keys {set(specs)} do not match parameter group keys "
                f"{set(groups)} from {type(self).__name__}.get_optimization_parameters()"
            )

        param_groups = []
        for key, tensors in groups.items():
            for p in tensors:
                p.requires_grad_(True)
            param_groups.append({"params": tensors, **specs[key].params()})
        self._optimizer = self._build_optimizer(spec_list[0], param_groups)

    def _build_optimizer(self, opt_params, param_groups) -> "torch.optim.Optimizer":
        """Construct the torch optimizer for ``opt_params`` over pre-baked ``param_groups``.

        ``param_groups`` already carry their per-group hyperparameters (see ``set_optimizer``),
        so each group's ``lr`` etc. overrides the optimizer-level default. ``NoneOptimizer`` must
        have been filtered out by the caller.
        """
        match opt_params:
            case OptimizerParams.Adam():
                return torch.optim.Adam(param_groups)
            case OptimizerParams.AdamW():
                return torch.optim.AdamW(param_groups)
            case OptimizerParams.SGD():
                return torch.optim.SGD(param_groups)
            case OptimizerParams.NoneOptimizer():
                raise ValueError(
                    "NoneOptimizer must be filtered out before _build_optimizer; "
                    "set_optimizer should have short-circuited to remove_optimizer()."
                )
            case _:
                raise NotImplementedError(f"Unknown optimizer type: {opt_params}")

    def set_scheduler(
        self, scheduler_params: SchedulerParamsType | dict | None = None, num_iter: int | None = None
    ) -> None:
        """Set the scheduler for this model."""
        if scheduler_params is not None:
            self.scheduler_params = scheduler_params

        if not self._scheduler_params or self._optimizer is None:
            self._scheduler = None
            return

        optimizer = self._optimizer
        # Schedulers scale every torch param group proportionally off its own initial_lr; this
        # scalar only seeds scheduler config (e.g. min_lr, cyclic bounds). Use the max group LR
        # as the representative (collapses to group 0 in the single-group case).
        base_LR = max(pg["lr"] for pg in optimizer.param_groups)
        params = self._scheduler_params.params(base_LR, num_iter=num_iter)
        match self.scheduler_params:
            case SchedulerParams.NoneScheduler():
                self._scheduler = None
            case SchedulerParams.Cyclic():
                self._scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    **params,
                )
            case SchedulerParams.Plateau():
                self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    **params,
                )
            case SchedulerParams.Exponential():
                self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    **params,
                )
            case SchedulerParams.Linear():
                self._scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    **params,
                )
            case SchedulerParams.CosineAnnealing():
                self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    **params,
                )
            case _:
                raise ValueError(f"Unknown scheduler type: {self.scheduler_params}")

    def step_optimizer(self) -> None:
        """Step the optimizer if it exists."""
        if self._optimizer is not None:
            self._optimizer.step()

    def zero_optimizer_grad(self) -> None:
        """Zero gradients if optimizer exists."""
        if self._optimizer is not None:
            self._optimizer.zero_grad()

    def step_scheduler(self, loss: float | None = None) -> None:
        """Step the scheduler if it exists."""
        if self._scheduler is not None:
            if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if loss is not None:
                    self._scheduler.step(loss)
            else:
                self._scheduler.step()

    def has_optimizer(self) -> bool:
        """Check if this model has an active optimizer."""
        return self._optimizer is not None

    def get_current_lr(self) -> float:
        """Get the current learning rate."""
        if self._optimizer is not None:
            return self._optimizer.param_groups[0]["lr"]
        return 0.0

    def remove_optimizer(self) -> None:
        """Remove the optimizer and scheduler."""
        self._optimizer = None
        self._optimizer_params = {self.DEFAULT_OPTIMIZER_KEY: OptimizerParams.NoneOptimizer()}
        self._scheduler = None
        self._scheduler_params = SchedulerParams.NoneScheduler()

    def reset_optimizer(self) -> None:
        """Reset the optimizer and scheduler."""
        self.set_optimizer(self._optimizer_params)
        self.set_scheduler(self._scheduler_params)

    def reconnect_optimizer_to_parameters(self) -> None:
        """
        Reconnect optimizer to parameters after device changes.
        This is needed because AutoSerialize loads to CPU, but optimizers
        need to reference tensors on the current device.
        """
        if self._optimizer is None:
            return

        new_groups = self.get_optimization_parameters()
        if not new_groups:
            print(f"No optimizable parameters for {type(self).__name__}, removing optimizer")
            self.remove_optimizer()
            return

        # Ensure leaf params with grad
        for tensors in new_groups.values():
            for p in tensors:
                if not p.is_leaf:
                    raise ValueError("Non-leaf tensor in param group; build groups from leaves")
                p.requires_grad_(True)

        old_state = dict(self._optimizer.state)
        old_hyperparams = [
            {k: v for k, v in pg.items() if k != "params"} for pg in self._optimizer.param_groups
        ]

        self._optimizer.param_groups.clear()
        for tensors in new_groups.values():
            self._optimizer.add_param_group({"params": tensors})

        # Restore per-group hyperparameters by index
        for new_pg, old_pg in zip(self._optimizer.param_groups, old_hyperparams):
            new_pg.update(old_pg)

        # Remap state for tensors that survived
        new_state = {}
        for new_pg in self._optimizer.param_groups:
            for new_param in new_pg["params"]:
                if new_param in old_state:
                    new_state[new_param] = {
                        k: (v.to(new_param.device) if isinstance(v, torch.Tensor) else v)
                        for k, v in old_state[new_param].items()
                    }

        self._optimizer.state.clear()
        self._optimizer.state.update(new_state)

        if self._scheduler is not None:
            self._scheduler.optimizer = self._optimizer
