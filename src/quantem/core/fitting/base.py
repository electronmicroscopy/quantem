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
    def forward(self, ctx: RenderContext) -> torch.Tensor:
        raise NotImplementedError

    def constraint_loss(self, ctx: RenderContext) -> torch.Tensor:
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
        self.model: AdditiveRenderModel | None = None
        self.ctx: RenderContext | None = None
        self.fit_history: dict[str, FitResult] = {}
        self.state_initialized: dict[str, torch.Tensor] | None = None
        self.loss_fn = torch.nn.MSELoss(reduction="mean")

    def get_optimization_parameters(self) -> Any:
        if self.model is None:
            return []
        return self.model.parameters()

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

    @property
    def state_current(self) -> dict[str, torch.Tensor] | None:
        if self.model is None:
            return None
        return self._get_model_state_dict_copy()

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
        self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        if self.model is None or self.ctx is None:
            raise RuntimeError("Model and context are not defined for fitting.")
        return self.model.total_constraint_loss(self.ctx)

    def fit_render(
        self,
        *,
        target: torch.Tensor,
        n_steps: int,
        constraint_weight: float = 1.0,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        progress: bool = False,
        run_key: str = "default",
        **kwargs: Any,
    ) -> FitResult:
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


Component = RenderComponent
ModelContext = RenderContext
Model = AdditiveRenderModel
Parameter = nn.Parameter
