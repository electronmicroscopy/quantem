from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from quantem.core.models.base import Component, ModelContext, Overlay, Parameter

__all__ = ["DCBackground", "GaussianBackground"]


class DCBackground(Component):
    def __init__(
        self,
        *,
        intensity: float | tuple[float, float] | tuple[float, float, float | None],
        name: str = "DCBackground",
    ):
        super().__init__(name=name)
        self.p_intensity = Parameter(intensity, tags={"role": "background_dc", "name": name})

    def prepare(self, ctx: ModelContext) -> Any:
        pI = self.p_intensity

        @dataclass
        class PreparedDCBackground:
            name: str

            def render(self, out: torch.Tensor, x: torch.Tensor, ctx: ModelContext) -> None:
                out += pI.value(x, device=ctx.device, dtype=ctx.dtype)

            def overlays(self) -> list[Overlay]:
                return []

        return PreparedDCBackground(name=self.name)


class GaussianBackground(Component):
    def __init__(
        self,
        *,
        sigma: float | tuple[float, float] | tuple[float, float, float | None],
        intensity: float | tuple[float, float] | tuple[float, float, float | None],
        origin_key: str = "origin",
        name: str = "GaussianBackground",
    ):
        super().__init__(name=name)
        self.origin_key = str(origin_key)
        self.p_sigma = Parameter(sigma, tags={"role": "background_gaussian_sigma", "name": name})
        self.p_intensity = Parameter(intensity, tags={"role": "background_gaussian_intensity", "name": name})

    def prepare(self, ctx: ModelContext) -> Any:
        ok = self.origin_key
        pS = self.p_sigma
        pI = self.p_intensity

        rr = torch.arange(ctx.H, device=ctx.device, dtype=ctx.dtype)[:, None]
        cc = torch.arange(ctx.W, device=ctx.device, dtype=ctx.dtype)[None, :]

        @dataclass
        class PreparedGaussianBackground:
            origin_key: str

            def render(self, out: torch.Tensor, x: torch.Tensor, ctx: ModelContext) -> None:
                if self.origin_key not in ctx.origins:
                    return
                r0, c0 = ctx.origins[self.origin_key]
                sig = torch.clamp(pS.value(x, device=ctx.device, dtype=ctx.dtype), min=1e-6)
                inten = pI.value(x, device=ctx.device, dtype=ctx.dtype)
                dr = rr - r0
                dc = cc - c0
                out += inten * torch.exp(-0.5 * (dr * dr + dc * dc) / (sig * sig))

            def overlays(self) -> list[Overlay]:
                return []

        return PreparedGaussianBackground(origin_key=ok)
