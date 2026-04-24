from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Self, Sequence, cast

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from quantem.core.ml.optimizer_mixin import OptimizerMixin


def parse_bounded_init(
    value: float | int | Sequence[float | int | None], *, name: str
) -> tuple[float, float | None, float | None]:
    """
    Parse a scalar or bounded initializer specification.

    Parameters
    ----------
    value : float | int | Sequence[float | int | None]
        Accepted forms:
        - ``x`` -> init ``x`` with no bounds.
        - ``(x0, delta)`` -> init ``x0`` with bounds ``[x0-|delta|, x0+|delta|]``.
        - ``(x0, lo, hi)`` -> init ``x0`` with explicit bounds.
    name : str
        Parameter name used in error messages.

    Returns
    -------
    tuple[float, float | None, float | None]
        Parsed ``(init, lo, hi)``.

    Raises
    ------
    ValueError
        If the sequence form is invalid, contains required ``None`` entries,
        has invalid ordering, or ``init`` lies outside explicit bounds.
    """
    if not isinstance(value, (list, tuple, np.ndarray)):
        x = float(cast(float | int, value))
        return x, None, None

    seq = list(value)
    if len(seq) == 0:
        raise ValueError(f"{name} cannot be empty.")
    if seq[0] is None:
        raise ValueError(f"{name} initial value cannot be None.")
    x0 = float(cast(float | int, seq[0]))

    if len(seq) == 1:
        return x0, None, None
    if len(seq) == 2:
        if seq[1] is None:
            raise ValueError(f"{name} delta cannot be None.")
        delta = abs(float(cast(float | int, seq[1])))
        return x0, x0 - delta, x0 + delta
    if len(seq) == 3:
        if seq[1] is None or seq[2] is None:
            raise ValueError(f"{name} bounds cannot contain None.")
        lo = float(cast(float | int, seq[1]))
        hi = float(cast(float | int, seq[2]))
        if lo > hi:
            raise ValueError(f"{name} has invalid bounds: lo ({lo}) > hi ({hi}).")
        if x0 < lo or x0 > hi:
            raise ValueError(f"{name} initial value {x0} is outside bounds [{lo}, {hi}].")
        return x0, lo, hi

    raise ValueError(f"{name} must be scalar, (x0, delta), or (x0, lo, hi).")


@dataclass
class RenderContext:
    shape: tuple[int, ...]
    device: torch.device
    dtype: torch.dtype
    mask: torch.Tensor | None = None
    fields: dict[str, Any] = field(default_factory=dict)


class OriginND(nn.Module):
    def __init__(self, *, ndim: int, init: Sequence[float]):
        super().__init__()
        if int(ndim) <= 0:
            raise ValueError("ndim must be >= 1.")
        if len(init) != int(ndim):
            raise ValueError("init length must match ndim.")
        self.ndim = int(ndim)
        self.coords = nn.Parameter(torch.as_tensor(init, dtype=torch.float32).reshape(self.ndim))


class RenderComponent(OptimizerMixin,nn.Module):
    DEFAULT_HARD_CONSTRAINTS: dict[str, Any] = {}
    DEFAULT_SOFT_CONSTRAINTS: dict[str, Any] = {}
    DEFAULT_CONSTRAINT_CONFIG: dict[str, Any] = {}

    DEFAULT_OPTIMIZER: str = "adam"
    DEFAULT_LR: float = 1e-2
    DEFAULT_SCHEDULER_TYPE: str = "none"

    def __init__(self) -> None:
        nn.Module.__init__(self)
        
        self._optimizer = None
        self._scheduler = None
        self._optimizer_params = {}
        self._scheduler_params = {}
        self.hard_constraints: dict[str, Any] = dict(self.DEFAULT_HARD_CONSTRAINTS)
        self.soft_constraints: dict[str, Any] = dict(self.DEFAULT_SOFT_CONSTRAINTS)
        self.constraint_config: dict[str, Any] = dict(self.DEFAULT_CONSTRAINT_CONFIG)
        self.parameter_bounds: dict[str, tuple[float | None, float | None]] = {}

        optimizer_params = {
                "type": self.DEFAULT_OPTIMIZER,
                "lr": self.DEFAULT_LR
            }
        self.optimizer_params = optimizer_params

        scheduler_params = {
                "type": self.DEFAULT_SCHEDULER_TYPE
            }
        self.scheduler_params = scheduler_params

    @staticmethod
    def parse_bounded_init(
        value: float | int | Sequence[float | int | None], *, name: str
    ) -> tuple[float, float | None, float | None]:
        """
        Parse bounded initializer forms into ``(init, lo, hi)``.

        Parameters
        ----------
        value : float | int | Sequence[float | int | None]
            Scalar, ``(x0, delta)``, or ``(x0, lo, hi)``.
        name : str
            Parameter name used in error messages.

        Returns
        -------
        tuple[float, float | None, float | None]
            Parsed ``(init, lo, hi)``.
        """
        return parse_bounded_init(value, name=name)

    def register_parameter_bounds(
        self, parameter_name: str, lo: float | None, hi: float | None
    ) -> None:
        """
        Register hard bounds for a trainable parameter.

        Parameters
        ----------
        parameter_name : str
            Name of an ``nn.Parameter`` attribute on this component.
        lo : float | None
            Lower bound, or ``None`` for unbounded lower side.
        hi : float | None
            Upper bound, or ``None`` for unbounded upper side.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``lo > hi``.
        """
        if lo is not None and hi is not None and float(lo) > float(hi):
            raise ValueError(f"Invalid bounds for {parameter_name}: lo ({lo}) > hi ({hi}).")
        self.parameter_bounds[str(parameter_name)] = (
            None if lo is None else float(lo),
            None if hi is None else float(hi),
        )

    def _enforce_parameter_bounds(self) -> None:
        """
        Clamp registered parameters in-place to configured bounds.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If a registered parameter attribute is missing.
        TypeError
            If a registered attribute is not an ``nn.Parameter``.
        """
        if not self.parameter_bounds:
            return
        with torch.no_grad():
            for param_name, (lo, hi) in self.parameter_bounds.items():
                if not hasattr(self, param_name):
                    raise AttributeError(
                        f"Parameter '{param_name}' is not an attribute of {self.__class__.__name__}."
                    )
                param = getattr(self, param_name)
                if not isinstance(param, nn.Parameter):
                    raise TypeError(
                        f"Attribute '{param_name}' on {self.__class__.__name__} is not an nn.Parameter."
                    )
                if lo is None and hi is None:
                    continue
                if lo is None:
                    assert hi is not None
                    param.clamp_(max=float(hi))
                elif hi is None:
                    assert lo is not None
                    param.clamp_(min=float(lo))
                else:
                    param.clamp_(min=float(lo), max=float(hi))

    def _set_constraints(
        self,
        current: dict[str, Any],
        defaults: dict[str, Any],
        constraints: dict[str, Any],
        *,
        strict: bool,
    ) -> None:
        if strict:
            unknown = [k for k in constraints if k not in defaults]
            if unknown:
                keys = ", ".join(str(k) for k in unknown)
                raise KeyError(f"Unknown constraint keys: {keys}")
        current.update(constraints)

    def set_hard_constraints(self, constraints: dict[str, Any], strict: bool = True) -> None:
        self._set_constraints(
            self.hard_constraints, self.DEFAULT_HARD_CONSTRAINTS, constraints, strict=strict
        )

    def set_soft_constraints(self, constraints: dict[str, Any], strict: bool = True) -> None:
        self._set_constraints(
            self.soft_constraints, self.DEFAULT_SOFT_CONSTRAINTS, constraints, strict=strict
        )

    def apply_constraint_params(self, params: dict[str, Any], strict: bool = True) -> None:
        if not isinstance(params, dict):
            raise TypeError("constraint params must be a dict.")
        if "hard" in params or "soft" in params:
            hard = params.get("hard")
            soft = params.get("soft")
            if hard is not None:
                if not isinstance(hard, dict):
                    raise TypeError("constraint params 'hard' value must be a dict.")
                self.set_hard_constraints(hard, strict=strict)
            if soft is not None:
                if not isinstance(soft, dict):
                    raise TypeError("constraint params 'soft' value must be a dict.")
                self.set_soft_constraints(soft, strict=strict)
            return

        hard_updates: dict[str, Any] = {}
        soft_updates: dict[str, Any] = {}
        unknown: dict[str, Any] = {}
        for k, v in params.items():
            if k in self.DEFAULT_HARD_CONSTRAINTS:
                hard_updates[k] = v
            elif k in self.DEFAULT_SOFT_CONSTRAINTS:
                soft_updates[k] = v
            else:
                unknown[k] = v

        if unknown and strict:
            keys = ", ".join(str(k) for k in unknown.keys())
            raise KeyError(f"Unknown constraint keys for {self.__class__.__name__}: {keys}")
        if unknown:
            soft_updates.update(unknown)
        if hard_updates:
            self.set_hard_constraints(hard_updates, strict=strict)
        if soft_updates:
            self.set_soft_constraints(soft_updates, strict=strict)

    def effective_soft_constraints(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        effective = dict(self.soft_constraints)
        if isinstance(params, dict):
            effective.update(params)
        return effective

    def enforce_hard_constraints(self, ctx: RenderContext) -> None:
        self._enforce_parameter_bounds()

    def forward(self, ctx: RenderContext) -> torch.Tensor:
        raise NotImplementedError

    def constraint_loss(
        self, ctx: RenderContext, params: dict[str, Any] | None = None
    ) -> torch.Tensor:
        return torch.zeros((), device=ctx.device, dtype=ctx.dtype)

    def get_optimization_parameters(self) -> Any: 
        return [p for p in self.parameters() if p.requires_grad]
    
    def initialize_optimizer(self, 
        optimizer_params: dict[str, Any] | None = None,
        scheduler_params: dict[str, Any] | None = None,
        num_iter: int | None = None,
        ) -> None:
        trainable_params = list(self.get_optimization_parameters())
        if not trainable_params:
            self._optimizer = None
            self._scheduler = None
            return
        if optimizer_params is not None:
            self.optimizer_params = optimizer_params
        else:
            self.optimizer_params = {
                "type": self.DEFAULT_OPTIMIZER,
                "lr": self.DEFAULT_LR
            }
        self.set_optimizer(self.optimizer_params)

        if scheduler_params is not None:
            self.scheduler_params = scheduler_params
        else:
            self.scheduler_params = {
                "type": self.DEFAULT_SCHEDULER_TYPE
            }
        self.set_scheduler(self.scheduler_params, num_iter = num_iter)
        return
    
    
    def _infer_optimizer_rebuild_params(self) -> dict[str, Any]:
        if self.optimizer_params:
            return dict(self.optimizer_params)
        if self.optimizer is not None:
            opt_type: str | type[torch.optim.Optimizer]
            if isinstance(self.optimizer, torch.optim.AdamW):
                opt_type = "adamw"
            elif isinstance(self.optimizer, torch.optim.Adam):
                opt_type = "adam"
            elif isinstance(self.optimizer, torch.optim.SGD):
                opt_type = "sgd"
            else:
                opt_type = type(self.optimizer)
            lr = float(
                self.optimizer.param_groups[0].get(
                    "lr", getattr(self, "DEFAULT_LR", self.DEFAULT_LR)
                )
            )
            return {"type": opt_type, "lr": lr}
        return {
            "type": getattr(self, "DEFAULT_OPTIMIZER_TYPE", self.DEFAULT_OPTIMIZER),
            "lr": float(getattr(self, "DEFAULT_LR", self.DEFAULT_LR)),
        }
    
    def _infer_scheduler_rebuild_params(self) -> dict[str, Any]:
        if self.scheduler_params:
            return dict(self.scheduler_params)
        return {
            "type": self.DEFAULT_SCHEDULER_TYPE,
        }
    
    def _rebuild_optimizer_after_trainability_change(self) -> None:
        trainable_params = list(self.get_optimization_parameters())
        if not trainable_params:
            self._optimizer = None
            self._scheduler = None
            return
        rebuild_params = self._infer_optimizer_rebuild_params()
        rebuild_params_scheduler = self._infer_scheduler_rebuild_params()
        self.set_optimizer(rebuild_params)
        self.set_scheduler(rebuild_params_scheduler)
    
    def initialize_constraint_config(self, config: dict[str, Any], strict: bool = True) -> None:
        if not hasattr(self, 'constraint_config'):
            if strict:
                raise AttributeError(
                    f"{self.__class__.__name__} does not have constraint_config attribute"
                )
            return
        if not isinstance(config, dict):
            raise TypeError("constraint config must be a dict.")
        
        unknown: dict[str, Any] = {}
        for k, v in config.items():
            if k in self.DEFAULT_CONSTRAINT_CONFIG:
                self.constraint_config[k] = v
            else:
                unknown[k] = v

        if unknown and strict:
            keys = ", ".join(str(k) for k in unknown.keys())
            raise KeyError(f"Unknown constraint keys for {self.__class__.__name__}: {keys}")
        return


class AdditiveRenderModel(nn.Module): # step all otpimzers
    def __init__(self, *, origin: nn.Module, components: list[RenderComponent]):
        super().__init__()
        self.origin = origin
        self.components = nn.ModuleList(components)

    def forward(self, ctx: RenderContext) -> torch.Tensor:
        if len(self.components) == 0:
            return torch.zeros(ctx.shape, device=ctx.device, dtype=ctx.dtype)
        out = self.components[0](ctx)
        for component in self.components[1:]:
            out = out + component(ctx)
        return out

    def _component_constraint_name(self, component: RenderComponent, idx: int) -> str:
        name = getattr(component, "name", None)
        if isinstance(name, str) and name:
            return name
        class_name = component.__class__.__name__
        if class_name:
            return class_name
        return f"component_{idx}"

    def apply_constraint_params(
        self, constraint_params: dict[str, Any], strict: bool = True
    ) -> None:
        if not isinstance(constraint_params, dict):
            raise TypeError("constraint_params must be a dict.")
        source = constraint_params.get("components")
        component_map = source if isinstance(source, dict) else constraint_params
        for target, params in component_map.items():
            if not isinstance(params, dict):
                if strict:
                    raise TypeError(f"Constraint params for '{target}' must be a dict.")
                continue
            target_str = str(target)
            name_matches: list[RenderComponent] = []
            class_matches: list[RenderComponent] = []
            for idx, module in enumerate(self.components):
                component = cast(RenderComponent, module)
                if self._component_constraint_name(component, idx) == target_str:
                    name_matches.append(component)
                if component.__class__.__name__ == target_str:
                    class_matches.append(component)
            targets = name_matches if name_matches else class_matches
            if not targets:
                if strict:
                    raise KeyError(f"No matching component for constraint target '{target_str}'.")
                continue
            for component in targets:
                component.apply_constraint_params(params, strict=strict)

    def apply_hard_constraints(self, ctx: RenderContext) -> None:
        for module in self.components:
            component = cast(RenderComponent, module)
            component.enforce_hard_constraints(ctx)

    def total_constraint_loss(self, ctx: RenderContext) -> torch.Tensor:
        loss = torch.zeros((), device=ctx.device, dtype=ctx.dtype)
        for module in self.components:
            component = cast(RenderComponent, module)
            loss = loss + component.constraint_loss(ctx)
        return loss

    def initilize_independant_optimizers(self, 
        individual_optimizers: dict[str, dict[str, Any]] | None = None,
        individual_schedulers: dict[str, dict[str, Any]] | None = None,
        num_iter: int | None = None,
        ) -> None:
        for idx, module in enumerate(self.components):
            component = cast(RenderComponent, module)
            component_name = self._component_constraint_name(component, idx)

            component_optimizer_params = None
            component_scheduler_params = None
            if individual_optimizers is not None:
                if component_name in individual_optimizers:
                    component_optimizer_params = individual_optimizers[component_name]
                elif component.__class__.__name__ in individual_optimizers :
                    component_optimizer_params = individual_optimizers[component.__class__.__name__]

            if individual_schedulers is not None:
                if component_name in individual_schedulers:
                    component_scheduler_params = individual_schedulers[component_name]
                elif component.__class__.__name__ in individual_schedulers:
                    component_scheduler_params = individual_schedulers[component.__class__.__name__]
            
            component.initialize_optimizer(component_optimizer_params, component_scheduler_params, num_iter=num_iter)
    
    def set_independant_optimizer_params(self, 
        individual_optimizer_params: dict[str, dict[str, Any]],
        individual_scheduler_params: dict[str, dict[str, Any]] | None = None,
        num_iter: int | None = None,
        ) -> None:
        for component_name, param in individual_optimizer_params.items():
            scheduler_params = None
            if individual_scheduler_params is not None and component_name in individual_scheduler_params:
                scheduler_params = individual_scheduler_params[component_name]
        
            component = self._resolve_component_by_name(str(component_name))
            component.initialize_optimizer(param,scheduler_params,num_iter)
    
    def rebuild_independant_optimizers(self) -> None:
        for module in self.components:
            component = cast(RenderComponent, module)
            component.initialize_optimizer()

    def step_optimizers(self) -> None:
        for module in self.components:
            component = cast(RenderComponent, module)
            component.step_optimizer()
    
    def step_schedulers(self, loss: float | None = None) -> None:
        for module in self.components:
            component = cast(RenderComponent, module)
            if hasattr(component, 'step_scheduler'):
                try:
                    component.step_scheduler(loss)
                except (AttributeError, TypeError):
                    pass
    
    def zero_grad_optimizers(self) -> None:
        for module in self.components:
            component = cast(RenderComponent, module)
            component.zero_optimizer_grad()
    
    def _iter_named_components(self) -> list[tuple[str, RenderComponent]]:
        """
        Return canonical component names paired with components.

        Returns
        -------
        list[tuple[str, RenderComponent]]
            ``(name, component)`` entries using the model's canonical naming
            rule. Names fall back to class-name/index behavior when ``.name`` is
            missing.

        Raises
        ------
        RuntimeError
            If the model is not defined.
        """
        entries: list[tuple[str, RenderComponent]] = []
        for idx, module in enumerate(self.components):
            component = cast(RenderComponent, module)
            name = self._component_constraint_name(component, idx)
            entries.append((name, component))
        return entries
    
    def _resolve_component_by_name(self, component_name: str) -> RenderComponent:
        target = str(component_name)
        for resolved_name, component in self._iter_named_components():
            if resolved_name == target:
                return component
        
        for resolved_name, component in self._iter_named_components():
            if component.__class__.__name__ == target:
                return component
        
        known = ", ".join(self.get_component_names())
        raise KeyError(f"Component not found: {target}. Known components: {known}")
    
    def get_component_names(self) -> list[str]:
        """
        Return canonical component names.

        Returns
        -------
        list[str]
            Canonical component names.
        """
        return [name for name, _ in self._iter_named_components()]
    
    def get_component_by_name(self, component_name: str) -> RenderComponent:
        return self._resolve_component_by_name(component_name) 
    

    def set_component_trainable(
        self, 
        component_name: str, 
        enabled: bool, 
        rebuild_optimizer: bool = True,  
        num_iter: int | None = None,
    ) -> None:
        """
        Enable or disable optimization for all parameters in one component.

        Parameters
        ----------
        component_name : str
            Resolved component name.
        enabled : bool
            If ``True``, mark component parameters trainable.
        rebuild_optimizer : bool, optional
            If ``True``, rebuild optimizer param groups after toggling.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the model is not defined.
        KeyError
            If ``component_name`` is unknown.

        Notes
        -----
        When rebuilding, the optimizer is reconstructed from stored optimizer
        parameters if available, otherwise inferred from the current optimizer
        type and learning rate, else defaults. Scheduler state is cleared
        predictably by setting scheduler type to ``"none"``.
        """
        component = self._resolve_component_by_name(component_name)
        for _, param in component.named_parameters(recurse=True):
            param.requires_grad_(bool(enabled))
        if rebuild_optimizer:
            component._rebuild_optimizer_after_trainability_change()

    def set_parameter_trainable(
        self,
        component_name: str,
        parameter_name: str,
        enabled: bool,
        rebuild_optimizer: bool = True,
        num_iter: int | None = None,
    ) -> None:
        """
        Enable or disable optimization for one component parameter.

        Parameters
        ----------
        component_name : str
            Resolved component name.
        parameter_name : str
            Parameter name from ``component.named_parameters()``.
        enabled : bool
            If ``True``, mark parameter trainable.
        rebuild_optimizer : bool, optional
            If ``True``, rebuild optimizer param groups after toggling.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the model is not defined.
        KeyError
            If ``component_name`` or ``parameter_name`` is unknown.

        Notes
        -----
        When rebuilding, scheduler state is cleared by setting scheduler type
        to ``"none"``.
        """
        component = self._resolve_component_by_name(component_name)
        params = dict(component.named_parameters(recurse=True))
        if parameter_name not in params:
            known = ", ".join(sorted(params.keys()))
            raise KeyError(
                f"Parameter '{parameter_name}' not found in component '{component_name}'. "
                f"Known parameters: {known}"
            )
        params[parameter_name].requires_grad_(bool(enabled))
        if rebuild_optimizer:
            component._rebuild_optimizer_after_trainability_change()

    def set_parameters_trainable(
        self,
        component_name: str,
        parameter_names: list[str],
        enabled: bool,
        rebuild_optimizer: bool = True,
        num_iter: int | None = None,
    ) -> None:
        """
        Enable or disable optimization for multiple component parameters.

        Parameters
        ----------
        component_name : str
            Resolved component name.
        parameter_names : list[str]
            Parameter names from ``component.named_parameters()``.
        enabled : bool
            If ``True``, mark parameters trainable.
        rebuild_optimizer : bool, optional
            If ``True``, rebuild optimizer param groups after toggling.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the model is not defined.
        KeyError
            If any parameter name is unknown.
        """
        component = self._resolve_component_by_name(component_name)
        params = dict(component.named_parameters(recurse=True))
        missing = [name for name in parameter_names if name not in params]
        if missing:
            known = ", ".join(sorted(params.keys()))
            raise KeyError(
                f"Unknown parameters for component '{component_name}': {', '.join(missing)}. "
                f"Known parameters: {known}"
            )
        for name in parameter_names:
            params[name].requires_grad_(bool(enabled))
        if rebuild_optimizer:
            component._rebuild_optimizer_after_trainability_change()

    def get_component_trainable(self, component_name: str) -> dict[str, bool]:
        """
        Return trainability flags for one component's parameters.

        Parameters
        ----------
        component_name : str
            Resolved component name.

        Returns
        -------
        dict[str, bool]
            Mapping of parameter name to ``requires_grad``.

        Raises
        ------
        RuntimeError
            If the model is not defined.
        KeyError
            If ``component_name`` is unknown.
        """
        component = self._resolve_component_by_name(component_name)
        return {name: bool(param.requires_grad) for name, param in component.named_parameters()}
    
    def apply_constraint_configs(
        self, constraint_configs: dict[str, Any], strict: bool = True
    ) -> None:
        for component_name, param in constraint_configs.items():
            component = self._resolve_component_by_name(str(component_name))
            component.initialize_constraint_config(param, strict=strict)

           

    
    



@dataclass
class FitResult:
    losses: list[float]
    lrs: list[float]
    final_loss: float
    num_steps: int
    metrics: dict[str, list[float]] = field(default_factory=dict)

class BCEMSELoss(nn.Module):

    def __init__(
        self,
        gamma: float = 1,  
        percentile: float = 0.95,
    ):
        super().__init__()
        self.gamma = gamma
        self.percentile = percentile
    
    def forward(self, pred, target):
        mse_loss = torch.mean((pred - target) ** 2)
        
        threshold_pred = torch.quantile(pred, self.percentile)
        threshold_target = torch.quantile(target, self.percentile)
        
        # Create binary masks
        target_binary = (target > threshold_target).float()
        pred_binary = (pred > threshold_pred).float()
        
        epsilon = 1e-7
        pred_binary_clamped = torch.clamp(pred_binary, epsilon, 1.0 - epsilon)
        bce_loss = -torch.mean(
            target_binary * torch.log(pred_binary_clamped) + 
            (1 - target_binary) * torch.log(1 - pred_binary_clamped)
        )
        
        # Combined loss
        total_loss = mse_loss + self.gamma ** bce_loss
        
        return total_loss

class SqrtMSELoss(nn.Module):
    def __init__(
        self,
        gamma: float = 0.25,  
    ):
        super().__init__()
        self.gamma = gamma
        self.mse_fn = torch.nn.MSELoss(reduction="mean")
    
    def forward(self, pred, target):
        eps = 1
        pred_modified = (pred-torch.min(pred)+eps)**self.gamma
        # pred_modified = pred_modified / torch.linalg.norm(pred_modified)

        target_modified = (target-torch.min(target)+eps)**self.gamma
        # target_modified = target_modified / torch.linalg.norm(target_modified)

        loss = self.mse_fn(pred_modified, target_modified)
        return loss

class LogMSELoss(nn.Module):
    def __init__(
        self,  
    ):
        super().__init__()
        self.mse_fn = torch.nn.MSELoss(reduction="mean")
    
    def forward(self, pred, target):
        return self.mse_fn(torch.log(1+pred), torch.log(1+target))

class FitBase(OptimizerMixin):

    def __init__(self):
        super().__init__()
        # Core wiring
        # self.loss_fn = torch.nn.L1Loss(reduction="mean")
        # self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.loss_fn = SqrtMSELoss()
        self.model: AdditiveRenderModel | None = None
        self.ctx: RenderContext | None = None

        # State/checkpoints
        self.state_initialized: dict[str, torch.Tensor] | None = None

        # Histories/results
        self.fit_history: dict[str, FitResult] = {}

        # self.multi_optimizers: dict[str, torch.optim.Optimizer] | None = None
        # self.use_mutiple_optimizers: bool = False

    def get_optimization_parameters(self) -> Any:
        if self.model is None:
            return []
        
        return [p for p in self.model.parameters() if p.requires_grad]

    @property
    def state_current(self) -> dict[str, torch.Tensor] | None:
        if self.model is None:
            return None
        return self._get_model_state_dict_copy()

    @property
    def render_initialized(self) -> np.ndarray:
        if self.state_initialized is None:
            raise RuntimeError("initialized state is unavailable. Call .define_model(...) first.")
        return self._render_state_array(self.state_initialized)

    @property
    def render_current(self) -> np.ndarray:
        if self.model is None or self.ctx is None:
            raise RuntimeError("Call .define_model(...) first.")
        return self.model(self.ctx).detach().cpu().numpy()
    
    def set_loss(self, loss_fn: str = "sqrtmse", gamma: float = 0.25):
        mode_in = loss_fn.strip().lower()
        if mode_in in {"mse"}:
            self.loss_fn = torch.nn.MSELoss(reduction="mean")
        elif mode_in in {"sqrtmse"}:
            self.loss_fn = SqrtMSELoss(gamma = gamma)
        elif mode_in in {"logmse"}:
            self.loss_fn = LogMSELoss()
        elif mode_in in {"l1"}:
            self.loss_fn = torch.nn.L1Loss(reduction="mean")
        else:
            raise ValueError(
                "loss function must be mse, sqrtmse, logmse or l1"
            )

    def reset(
        self,
        reset_to: Literal["initialized"] = "initialized",
    ) -> Self:
        if reset_to != "initialized":
            raise ValueError("FitBase.reset only supports reset_to='initialized'.")
        if self.state_initialized is None:
            raise RuntimeError("initialized state is unavailable. Call .define_model(...) first.")
        self._load_model_state_dict_copy(self.state_initialized)
        self._clear_fit_history_all()
        return self

    def fit_render(
        self,
        *,
        target: torch.Tensor,
        n_steps: int,
        constraint_weight: float = 1.0,
        constraint_params: dict[str, Any] | None = None,
        constraint_config_params: dict[str, Any] | None = None,
        optimizer_params: dict[str, dict[str, Any]] | None = None,
        scheduler_params: dict[str, dict[str, Any]] | None = None,
        progress: bool = False,
        run_key: str = "default",
        **kwargs: Any,
    ) -> FitResult:
        """
        Fit model parameters to a target render.

        Parameters
        ----------
        target : torch.Tensor
            Target tensor to fit.
        n_steps : int
            Number of optimization steps.
        constraint_weight : float, optional
            Multiplier applied to the summed soft-constraint loss.
        constraint_params : dict[str, Any] | None, optional
            Optional constraint updates applied once to matching components before
            optimization starts. If ``None``, existing component constraints are reused.
        optimizer_params : dict | None, optional
            Optimizer configuration override for this call.
        scheduler_params : dict | None, optional
            Scheduler configuration override for this call.
        progress : bool, optional
            If ``True``, display a progress bar.
        run_key : str, optional
            History key used to store/append fit metrics.
        **kwargs : Any
            Forwarded to internal forward/loss hooks.

        Returns
        -------
        FitResult
            Fit history and final loss metadata for this run key.

        Raises
        ------
        RuntimeError
            If model/context are undefined.

        Notes
        -----
        Hard constraints are applied after each optimizer step.
        """
        if self.model is None or self.ctx is None:
            raise RuntimeError("Model and context are not defined for fitting.")
        if constraint_params is not None:
            self.model.apply_constraint_params(constraint_params, strict=True)
        if constraint_config_params is not None:
            self.model.apply_constraint_configs(constraint_config_params, strict=True)

        n_steps = int(n_steps)
        self.model.initilize_independant_optimizers(
            optimizer_params,
            scheduler_params,
            num_iter=n_steps
        )

        pbar = tqdm(range(n_steps), desc="Fit render", disable=not progress)

        losses: list[float] = []
        lrs: list[float] = []
        for _ in pbar:
            self.model.zero_grad_optimizers()
            pred = self._forward_for_fit(target=target, **kwargs)
            data_loss = self._fidelity_loss(pred, target, **kwargs)
            constraint_loss = self._constraint_loss(pred, target, **kwargs)
            total_loss = data_loss + constraint_weight * constraint_loss
            total_loss.backward()
            self.model.step_optimizers()
            if self.model is None or self.ctx is None:
                raise RuntimeError("Model and context are not defined for fitting.")
            self.model.apply_hard_constraints(self.ctx)
            total_loss_value = float(total_loss.detach().cpu())
            self.model.step_schedulers(total_loss_value)
            losses.append(total_loss_value)

            first_lr = 0.0
            if len(self.model.components) > 0:
                first_comp = cast(RenderComponent, self.model.components[0])
                if hasattr(first_comp, 'get_current_lr'):
                    try:
                        first_lr = float(first_comp.get_current_lr())
                    except (AttributeError, TypeError):
                        first_lr = 0.0
            lrs.append(first_lr)

        key = str(run_key)
        if key in self.fit_history:
            prev = self.fit_history[key]
            prev.losses.extend(losses)
            prev.lrs.extend(lrs)
            prev.final_loss = prev.losses[-1] if prev.losses else float("nan")
            prev.num_steps = len(prev.losses)
            result = prev
        else:
            result = FitResult(
                losses=losses,
                lrs=lrs,
                final_loss=(losses[-1] if losses else float("nan")),
                num_steps=n_steps,
            )
            self.fit_history[key] = result
        return result


    def _clone_state_dict(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in state.items()}

    def _get_model_state_dict_copy(self) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        return self._clone_state_dict(self.model.state_dict())

    def _load_model_state_dict_copy(self, state: dict[str, torch.Tensor]) -> None:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        self.model.load_state_dict(self._clone_state_dict(state), strict=True)

    def _clear_fit_history_all(self) -> None:
        self.fit_history.clear()

    def _clear_fit_history_run(self, run_key: str) -> None:
        self.fit_history.pop(str(run_key), None)

    def _render_state_array(self, state: dict[str, torch.Tensor]) -> np.ndarray:
        if self.model is None or self.ctx is None:
            raise RuntimeError("Call .define_model(...) first.")
        live = self._get_model_state_dict_copy()
        try:
            self._load_model_state_dict_copy(state)
            arr = self.model(self.ctx).detach().cpu().numpy()
        finally:
            self._load_model_state_dict_copy(live)
        return arr

    def _forward_for_fit(self, *, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self.model is None or self.ctx is None:
            raise RuntimeError("Model and context are not defined for fitting.")
        return self.model(self.ctx)

    def _fidelity_loss(
        self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        if self.ctx is not None and self.ctx.mask is not None:
            # TODO -- use loss modules (currently implemented in tomo branch)
            # and update them to allow for masking at module level
            diff = (pred - target) * self.ctx.mask
            denom = torch.clamp(torch.sum(self.ctx.mask), min=1.0)
            return torch.sum(diff * diff) / denom
        return self.loss_fn(pred, target)

    def _constraint_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.model is None or self.ctx is None:
            raise RuntimeError("Model and context are not defined for fitting.")
        return self.model.total_constraint_loss(self.ctx)

    def set_component_trainable(
        self, 
        component_name: str, 
        enabled: bool, 
        rebuild_optimizer: bool = True,
        num_iter: int | None = None
    ) -> None:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        self.model.set_component_trainable(component_name, enabled, rebuild_optimizer, num_iter)

    def set_parameter_trainable(
        self,
        component_name: str,
        parameter_name: str,
        enabled: bool,
        rebuild_optimizer: bool = True,
        num_iter: int | None = None
    ) -> None:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        self.model.set_parameter_trainable(component_name, parameter_name, enabled, rebuild_optimizer, num_iter)

    def set_parameters_trainable(
        self,
        component_name: str,
        parameter_names: list[str],
        enabled: bool,
        rebuild_optimizer: bool = True,
        num_iter: int | None = None
    ) -> None:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        self.model.set_parameters_trainable(component_name, parameter_names, enabled, rebuild_optimizer, num_iter)

    def get_component_trainable(self, component_name: str) -> dict[str, bool]:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        return self.model.get_component_trainable(component_name)

    def get_component_names(self) -> list[str]:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        return self.model.get_component_names()

    def _iter_named_components(self) -> list[tuple[str, RenderComponent]]:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        return self.model._iter_named_components()
    
    def _resolve_component_by_name(self, component_name: str) -> RenderComponent:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        return self.model._resolve_component_by_name(component_name)
    
    def get_component_by_name(self, component_name: str) -> RenderComponent:
        return self._resolve_component_by_name(component_name)
    
    def _rebuild_optimizer_after_trainability_change(self) -> None:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        self.model.rebuild_independant_optimizers()

    def apply_constraint_config_params(
        self, constraint_configs: dict[str, Any], strict: bool = True
    ) -> None:
        if self.model is None:
            raise RuntimeError("Call .define_model(...) first.")
        self.model.apply_constraint_configs(constraint_configs, strict=strict)
        

Component = RenderComponent
ModelContext = RenderContext
Model = AdditiveRenderModel
Parameter = nn.Parameter
