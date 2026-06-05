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
        """
        Build a constant background component.

        Notes
        -----
        Validity is enforced via hard constraints/parameter bounds. Forward
        intentionally avoids hard clamps for gradient flow.
        """
        super().__init__()
        self.name = str(name)
        intensity_init, intensity_lo, intensity_hi = self.parse_bounded_init(
            intensity, name="intensity"
        )
        self.intensity_raw = nn.Parameter(torch.tensor(intensity_init, dtype=torch.float32))
        bounded_lo = 0.0 if intensity_lo is None else max(float(intensity_lo), 0.0)
        self.register_parameter_bounds("intensity_raw", bounded_lo, intensity_hi)
        if constraint_params is not None:
            self.apply_constraint_params(constraint_params, strict=True)
        self._enforce_parameter_bounds()

    def forward(self, ctx: RenderContext) -> torch.Tensor:
        """
        Render constant background from raw trainable intensity.

        Notes
        -----
        Validity is enforced via hard constraints/parameter bounds, not via
        forward-time hard clamps.
        """
        inten = self.intensity_raw.to(device=ctx.device, dtype=ctx.dtype)
        return torch.ones(ctx.shape, device=ctx.device, dtype=ctx.dtype) * inten

    def forward_batched(
        self,
        ctx: RenderContext,
        *,
        intensity_raw_b: torch.Tensor,
    ) -> torch.Tensor:
        B = intensity_raw_b.shape[0]
        return intensity_raw_b.view(B, 1, 1).expand(B, ctx.shape[0], ctx.shape[1]).to(
            device=ctx.device, dtype=ctx.dtype
        )


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
        """
        Build a Gaussian background component centered at origin.

        Notes
        -----
        ``sigma_raw`` and ``intensity_raw`` validity is enforced via hard
        constraints/parameter bounds. Forward intentionally avoids hard clamps
        for gradient flow.
        """
        super().__init__()
        self.name = str(name)
        self.origin = origin
        self.origin_key = str(origin_key)
        sigma_init, sigma_lo, sigma_hi = self.parse_bounded_init(sigma, name="sigma")
        intensity_init, intensity_lo, intensity_hi = self.parse_bounded_init(
            intensity, name="intensity"
        )
        self.sigma_raw = nn.Parameter(torch.tensor(sigma_init, dtype=torch.float32))
        sigma_bounded_lo = 1e-6 if sigma_lo is None else max(float(sigma_lo), 1e-6)
        self.register_parameter_bounds("sigma_raw", sigma_bounded_lo, sigma_hi)
        self.intensity_raw = nn.Parameter(torch.tensor(intensity_init, dtype=torch.float32))
        intensity_bounded_lo = 0.0 if intensity_lo is None else max(float(intensity_lo), 0.0)
        self.register_parameter_bounds("intensity_raw", intensity_bounded_lo, intensity_hi)
        if constraint_params is not None:
            self.apply_constraint_params(constraint_params, strict=True)
        self._enforce_parameter_bounds()

    def set_origin(self, origin: OriginND) -> None:
        self.origin = origin

    def forward(self, ctx: RenderContext) -> torch.Tensor:
        """
        Render Gaussian background from raw trainable parameters.

        Notes
        -----
        Validity is enforced via hard constraints/parameter bounds, not via
        forward-time hard clamps.
        """
        if self.origin is None:
            raise RuntimeError("GaussianBackground requires an OriginND instance.")

        rr = torch.arange(ctx.shape[0], device=ctx.device, dtype=ctx.dtype)[:, None]
        cc = torch.arange(ctx.shape[1], device=ctx.device, dtype=ctx.dtype)[None, :]
        r0, c0 = self.origin.coords[0], self.origin.coords[1]

        sigma = self.sigma_raw.to(device=ctx.device, dtype=ctx.dtype)
        inten = self.intensity_raw.to(device=ctx.device, dtype=ctx.dtype)
        r2 = (rr - r0) ** 2 + (cc - c0) ** 2
        return inten * torch.exp(-0.5 * r2 / (sigma * sigma))

    def forward_batched(
        self,
        ctx: RenderContext,
        *,
        sigma_raw_b: torch.Tensor,
        intensity_raw_b: torch.Tensor,
        origin_coords_b: torch.Tensor,
    ) -> torch.Tensor:
        B = sigma_raw_b.shape[0]
        rr = torch.arange(ctx.shape[0], device=ctx.device, dtype=ctx.dtype).view(1, ctx.shape[0], 1)
        cc = torch.arange(ctx.shape[1], device=ctx.device, dtype=ctx.dtype).view(1, 1, ctx.shape[1])
        r0 = origin_coords_b[:, 0].view(B, 1, 1)
        c0 = origin_coords_b[:, 1].view(B, 1, 1)
        sigma = sigma_raw_b.view(B, 1, 1)
        inten = intensity_raw_b.view(B, 1, 1)
        r2 = (rr - r0) ** 2 + (cc - c0) ** 2
        return inten * torch.exp(-0.5 * r2 / (sigma * sigma))