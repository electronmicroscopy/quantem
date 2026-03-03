from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from quantem.core.fitting.base import OriginND, RenderComponent, RenderContext


class DCBackground(RenderComponent):
    def __init__(
        self,
        *,
        intensity: float | int | Sequence[float | int | None] = 0.0,
        name: str = "dc_background",
        constraint_params: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.name = str(name)
        intensity_init, intensity_lo, intensity_hi = self.parse_bounded_init(
            intensity, name="intensity"
        )
        self.intensity_raw = nn.Parameter(torch.tensor(intensity_init, dtype=torch.float32))
        if intensity_lo is not None or intensity_hi is not None:
            self.register_parameter_bounds("intensity_raw", intensity_lo, intensity_hi)
        if constraint_params is not None:
            self.apply_constraint_params(constraint_params, strict=True)

    def forward(self, ctx: RenderContext) -> torch.Tensor:
        inten = torch.clamp(self.intensity_raw.to(device=ctx.device, dtype=ctx.dtype), min=0.0)
        return torch.ones(ctx.shape, device=ctx.device, dtype=ctx.dtype) * inten


class GaussianBackground(RenderComponent):  # TODO this should be N dimensional by default
    def __init__(
        self,
        *,
        sigma: float | int | Sequence[float | int | None] = (40.0, 5.0, None),
        intensity: float | int | Sequence[float | int | None] = 0.0,
        origin: OriginND | None = None,
        origin_key: str = "origin",
        name: str = "gaussian_background",
        constraint_params: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.name = str(name)
        self.origin = origin
        self.origin_key = str(origin_key)
        sigma_init, sigma_lo, sigma_hi = self.parse_bounded_init(sigma, name="sigma")
        intensity_init, intensity_lo, intensity_hi = self.parse_bounded_init(
            intensity, name="intensity"
        )
        self.sigma_raw = nn.Parameter(torch.tensor(sigma_init, dtype=torch.float32))
        if sigma_lo is not None or sigma_hi is not None:
            self.register_parameter_bounds("sigma_raw", sigma_lo, sigma_hi)
        self.intensity_raw = nn.Parameter(torch.tensor(intensity_init, dtype=torch.float32))
        if intensity_lo is not None or intensity_hi is not None:
            self.register_parameter_bounds("intensity_raw", intensity_lo, intensity_hi)
        if constraint_params is not None:
            self.apply_constraint_params(constraint_params, strict=True)

    def set_origin(self, origin: OriginND) -> None:
        self.origin = origin

    def forward(self, ctx: RenderContext) -> torch.Tensor:
        if self.origin is None:
            raise RuntimeError("GaussianBackground requires an OriginND instance.")

        rr = torch.arange(ctx.shape[0], device=ctx.device, dtype=ctx.dtype)[:, None]
        cc = torch.arange(ctx.shape[1], device=ctx.device, dtype=ctx.dtype)[None, :]
        r0, c0 = self.origin.coords[0], self.origin.coords[1]

        sigma = torch.clamp(self.sigma_raw.to(device=ctx.device, dtype=ctx.dtype), min=1e-6)
        inten = torch.clamp(self.intensity_raw.to(device=ctx.device, dtype=ctx.dtype), min=0.0)
        r2 = (rr - r0) ** 2 + (cc - c0) ** 2
        return inten * torch.exp(-0.5 * r2 / (sigma * sigma))
