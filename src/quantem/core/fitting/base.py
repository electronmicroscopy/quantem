from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import nn


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


Component = RenderComponent
ModelContext = RenderContext
Model = AdditiveRenderModel
Parameter = nn.Parameter
