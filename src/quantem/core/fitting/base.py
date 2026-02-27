from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, cast

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
        first = cast(RenderComponent, self.components[0])
        out = first(ctx)
        for module in self.components[1:]:
            component = cast(RenderComponent, module)
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
        self.fit_history_by_run: dict[str, FitResult] = {}

    @abstractmethod
    def _forward_for_fit(self, *, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _fidelity_loss(
        self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _soft_losses(
        self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        return {}

    def fit_render(
        self,
        *,
        target: torch.Tensor,
        n_steps: int,
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

        if scheduler_params is not None:
            self.set_scheduler(scheduler_params, num_iter=int(n_steps))
        elif self.scheduler is None and self.scheduler_params:
            self.set_scheduler(self.scheduler_params, num_iter=int(n_steps))
        elif optimizer_rebuilt and self.scheduler is not None and self.optimizer is not None:
            self.scheduler.optimizer = self.optimizer

        pbar = tqdm(range(int(n_steps)), desc="Fit render", disable=not progress)

        losses: list[float] = []
        lrs: list[float] = []
        metrics: dict[str, list[float]] = {}
        for _ in pbar:
            self.zero_optimizer_grad()
            pred = self._forward_for_fit(target=target, **kwargs)
            fidelity_loss = self._fidelity_loss(pred, target, **kwargs)
            soft_losses = self._soft_losses(pred, target, **kwargs)
            total_loss = fidelity_loss
            for k, v in soft_losses.items():
                metrics.setdefault(k, []).append(float(v.detach().cpu()))
                total_loss = total_loss + v
            total_loss.backward()
            self.step_optimizer()
            total_loss_value = float(total_loss.detach().cpu())
            self.step_scheduler(total_loss_value)
            losses.append(total_loss_value)
            lrs.append(float(self.get_current_lr()))

        result = FitResult(
            losses=losses,
            lrs=lrs,
            final_loss=(losses[-1] if losses else float("nan")),
            num_steps=int(n_steps),
            metrics=metrics,
        )
        self.fit_history_by_run[str(run_key)] = result
        return result


Component = RenderComponent
ModelContext = RenderContext
Model = AdditiveRenderModel
Parameter = nn.Parameter
