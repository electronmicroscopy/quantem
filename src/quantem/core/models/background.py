from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from quantem.core.models.base import Component, ModelContext, Overlay, Parameter
from quantem.core.models.diffraction import _origin_indices


class DCBackground(Component):
    def __init__(
        self,
        *,
        intensity: float | tuple[float, float] | tuple[float, float, float | None] = 0.0,
        name: str = "dc_background",
    ):
        super().__init__(name=name)
        self.p_intensity = Parameter(intensity, lower_bound=0.0, tags={"role": "dc_bg"})

    def parameters(self) -> list[Parameter]:
        return [self.p_intensity]

    def prepare(self, ctx: ModelContext) -> Any:
        idx = self.p_intensity.index

        @dataclass
        class Prepared:
            def render(self, out: torch.Tensor, x: torch.Tensor, ctx: ModelContext) -> None:
                out.add_(x[idx])

            def overlays(self, x: torch.Tensor, ctx: ModelContext) -> Iterable[Overlay]:
                return []

        return Prepared()


class GaussianBackground(Component):
    def __init__(
        self,
        *,
        sigma: float | tuple[float, float] | tuple[float, float, float | None] = (40.0, 5.0, None),
        intensity: float | tuple[float, float] | tuple[float, float, float | None] = 0.0,
        origin_key: str = "origin",
        name: str = "gaussian_background",
    ):
        super().__init__(name=name)
        self.origin_key = str(origin_key)
        self.p_sigma = Parameter(sigma, lower_bound=1e-6, upper_bound=None, tags={"role": "gauss_sigma"})
        self.p_intensity = Parameter(intensity, lower_bound=0.0, tags={"role": "gauss_int"})

    def parameters(self) -> list[Parameter]:
        return [self.p_sigma, self.p_intensity]

    def prepare(self, ctx: ModelContext) -> Any:
        i_sig = self.p_sigma.index
        i_int = self.p_intensity.index
        r_idx, c_idx = _origin_indices(ctx, self.origin_key)

        rr = torch.arange(ctx.H, device=ctx.device, dtype=ctx.dtype)[:, None]
        cc = torch.arange(ctx.W, device=ctx.device, dtype=ctx.dtype)[None, :]

        @dataclass
        class Prepared:
            def render(self, out: torch.Tensor, x: torch.Tensor, ctx: ModelContext) -> None:
                sig = torch.clamp(x[i_sig], min=1e-6)
                inten = x[i_int]
                r0 = x[r_idx]
                c0 = x[c_idx]
                dr = rr - r0
                dc = cc - c0
                r2 = dr * dr + dc * dc
                out.add_(inten * torch.exp(-0.5 * r2 / (sig * sig)))

            def overlays(self, x: torch.Tensor, ctx: ModelContext) -> Iterable[Overlay]:
                return []

        return Prepared()
