from __future__ import annotations

from typing import Sequence, cast

import numpy as np
import torch
from torch import nn

from quantem.core.fitting.base import RenderComponent, RenderContext
from quantem.core.fitting.diffraction import OriginND


def _parse_init(value: float | int | Sequence[float | int | None], *, name: str) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            raise ValueError(f"{name} cannot be empty.")
        if value[0] is None:
            raise ValueError(f"{name} initial value cannot be None.")
        return float(value[0])
    return float(cast(float | int, value))


def _softplus_pos(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.nn.functional.softplus(x) + eps


def _relu_pos(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.nn.functional.relu(x) + eps


class DCBackground(RenderComponent):
    def __init__(
        self,
        *,
        intensity: float | int | Sequence[float | int | None] = 0.0,
        name: str = "dc_background",
    ):
        super().__init__()
        self.name = str(name)
        self.intensity_raw = nn.Parameter(
            torch.tensor(_parse_init(intensity, name="intensity"), dtype=torch.float32)
        )

    def forward(self, ctx: RenderContext) -> torch.Tensor:
        inten = _relu_pos(self.intensity_raw.to(device=ctx.device, dtype=ctx.dtype))
        # inten = _softplus_pos(self.intensity_raw.to(device=ctx.device, dtype=ctx.dtype))
        return torch.ones(ctx.shape, device=ctx.device, dtype=ctx.dtype) * inten


class GaussianBackground(RenderComponent):
    def __init__(
        self,
        *,
        sigma: float | int | Sequence[float | int | None] = (40.0, 5.0, None),
        intensity: float | int | Sequence[float | int | None] = 0.0,
        origin: OriginND | None = None,
        origin_key: str = "origin",
        name: str = "gaussian_background",
    ):
        super().__init__()
        self.name = str(name)
        self.origin = origin
        self.origin_key = str(origin_key)
        self.sigma_raw = nn.Parameter(
            torch.tensor(_parse_init(sigma, name="sigma"), dtype=torch.float32)
        )
        self.intensity_raw = nn.Parameter(
            torch.tensor(_parse_init(intensity, name="intensity"), dtype=torch.float32)
        )

    def set_origin(self, origin: OriginND) -> None:
        self.origin = origin

    def forward(self, ctx: RenderContext) -> torch.Tensor:
        if self.origin is None:
            raise RuntimeError("GaussianBackground requires an OriginND instance.")

        rr = torch.arange(ctx.shape[0], device=ctx.device, dtype=ctx.dtype)[:, None]
        cc = torch.arange(ctx.shape[1], device=ctx.device, dtype=ctx.dtype)[None, :]
        r0, c0 = self.origin.coords[0], self.origin.coords[1]

        sigma = _relu_pos(self.sigma_raw.to(device=ctx.device, dtype=ctx.dtype), eps=1e-6)
        # sigma = _softplus_pos(self.sigma_raw.to(device=ctx.device, dtype=ctx.dtype), eps=1e-6)
        inten = _relu_pos(self.intensity_raw.to(device=ctx.device, dtype=ctx.dtype))
        # inten = _softplus_pos(self.intensity_raw.to(device=ctx.device, dtype=ctx.dtype))
        r2 = (rr - r0) ** 2 + (cc - c0) ** 2
        return inten * torch.exp(-0.5 * r2 / (sigma * sigma))
