from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Self, Sequence, cast

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from quantem.core.ml.optimizer_mixin import OptimizerMixin


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


class RenderComponent(nn.Module):
    DEFAULT_HARD_CONSTRAINTS: dict[str, Any] = {}
    DEFAULT_SOFT_CONSTRAINTS: dict[str, Any] = {}

    def __init__(self) -> None:
        super().__init__()
        self.hard_constraints: dict[str, Any] = dict(self.DEFAULT_HARD_CONSTRAINTS)
        self.soft_constraints: dict[str, Any] = dict(self.DEFAULT_SOFT_CONSTRAINTS)

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
        return None

    def forward(self, ctx: RenderContext) -> torch.Tensor:
        raise NotImplementedError

    def constraint_loss(
        self, ctx: RenderContext, params: dict[str, Any] | None = None
    ) -> torch.Tensor:
        return torch.zeros((), device=ctx.device, dtype=ctx.dtype)


class AdditiveRenderModel(nn.Module):
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


@dataclass
class FitResult:
    losses: list[float]
    lrs: list[float]
    final_loss: float
    num_steps: int
    metrics: dict[str, list[float]] = field(default_factory=dict)


class FitBase(OptimizerMixin):
    DEFAULT_LR = 1e-2
    DEFAULT_OPTIMIZER_TYPE = "adam"

    def __init__(self):
        super().__init__()
        # Core wiring
        self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.model: AdditiveRenderModel | None = None
        self.ctx: RenderContext | None = None

        # State/checkpoints
        self.state_initialized: dict[str, torch.Tensor] | None = None

        # Histories/results
        self.fit_history: dict[str, FitResult] = {}

    def get_optimization_parameters(self) -> Any:
        if self.model is None:
            return []
        return self.model.parameters()

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
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
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

        optimizer_rebuilt = False
        if optimizer_params is not None:
            self.set_optimizer(optimizer_params)
            optimizer_rebuilt = True
        elif self.optimizer is None:
            if self.optimizer_params:
                self.set_optimizer(self.optimizer_params)
            else:
                self.set_optimizer(
                    {
                        "type": getattr(self, "DEFAULT_OPTIMIZER_TYPE", "adamw"),
                        "lr": float(getattr(self, "DEFAULT_LR", self.DEFAULT_LR)),
                    }
                )
            optimizer_rebuilt = True

        n_steps = int(n_steps)
        if scheduler_params is not None:
            self.set_scheduler(scheduler_params, num_iter=n_steps)
        elif self.scheduler is None and self.scheduler_params:
            self.set_scheduler(self.scheduler_params, num_iter=n_steps)
        elif optimizer_rebuilt and self.scheduler is not None and self.optimizer is not None:
            self.scheduler.optimizer = self.optimizer

        pbar = tqdm(range(n_steps), desc="Fit render", disable=not progress)

        losses: list[float] = []
        lrs: list[float] = []
        for _ in pbar:
            self.zero_optimizer_grad()
            pred = self._forward_for_fit(target=target, **kwargs)
            data_loss = self._fidelity_loss(pred, target, **kwargs)
            constraint_loss = self._constraint_loss(pred, target, **kwargs)
            total_loss = data_loss + constraint_weight * constraint_loss
            total_loss.backward()
            self.step_optimizer()
            if self.model is None or self.ctx is None:
                raise RuntimeError("Model and context are not defined for fitting.")
            self.model.apply_hard_constraints(self.ctx)
            total_loss_value = float(total_loss.detach().cpu())
            self.step_scheduler(total_loss_value)
            losses.append(total_loss_value)
            lrs.append(float(self.get_current_lr()))

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


Component = RenderComponent
ModelContext = RenderContext
Model = AdditiveRenderModel
Parameter = nn.Parameter
