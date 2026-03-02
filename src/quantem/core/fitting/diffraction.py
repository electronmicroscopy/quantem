from __future__ import annotations

from typing import Any, Iterable, Sequence, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from quantem.core.fitting.base import OriginND, RenderComponent, RenderContext


def _parse_init(value: float | int | Sequence[float | int | None], *, name: str) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            raise ValueError(f"{name} cannot be empty.")
        if value[0] is None:
            raise ValueError(f"{name} initial value cannot be None.")
        return float(value[0])
    return float(cast(float | int, value))


def _splat_patch(
    out: torch.Tensor,
    *,
    r0: torch.Tensor,
    c0: torch.Tensor,
    patch_vals: torch.Tensor,
    dr: torch.Tensor,
    dc: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    h, w = out.shape
    r = r0 + dr
    c = c0 + dc

    r_base = torch.floor(r)
    c_base = torch.floor(c)
    fr = r - r_base
    fc = c - c_base
    r0i = r_base.to(torch.long)
    c0i = c_base.to(torch.long)

    w00 = (1.0 - fr) * (1.0 - fc)
    w01 = (1.0 - fr) * fc
    w10 = fr * (1.0 - fc)
    w11 = fr * fc
    v = patch_vals * scale

    def put(rr: torch.Tensor, cc: torch.Tensor, ww: torch.Tensor) -> None:
        keep = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        if torch.any(keep):
            out.index_put_((rr[keep], cc[keep]), v[keep] * ww[keep], accumulate=True)

    put(r0i, c0i, w00)
    put(r0i, c0i + 1, w01)
    put(r0i + 1, c0i, w10)
    put(r0i + 1, c0i + 1, w11)


class DiskTemplate(RenderComponent):
    DEFAULT_HARD_CONSTRAINTS: dict[str, bool] = {
        "force_center": False,
        "force_positive": True,
    }
    DEFAULT_SOFT_CONSTRAINTS: dict[str, float] = {"tv_weight": 0.0}

    def __init__(
        self,
        *,
        name: str,
        array: np.ndarray,
        refine_all_pixels: bool = False,
        normalize: str = "none",
        origin: OriginND | None = None,
        origin_key: str = "origin",
        intensity: float | Sequence[float] = 1.0,
        constraint_params: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.name = str(name)
        self.refine_all_pixels = bool(refine_all_pixels)
        self.origin = origin
        self.origin_key = str(origin_key)

        a = np.asarray(array, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError("DiskTemplate.array must be 2D.")
        if normalize == "max":
            s = float(np.max(a))
            if s > 0.0:
                a = a / s
        elif normalize == "mean":
            s = float(np.mean(a))
            if s != 0.0:
                a = a / s
        elif normalize != "none":
            raise ValueError("normalize must be one of: 'none', 'max', 'mean'.")

        template = torch.as_tensor(a, dtype=torch.float32)
        self.template_raw = nn.Parameter(template.clone(), requires_grad=self.refine_all_pixels)

        ht, wt = int(template.shape[0]), int(template.shape[1])
        rr, cc = np.mgrid[0:ht, 0:wt]
        rr = rr.astype(np.float32) - (ht - 1) * 0.5
        cc = cc.astype(np.float32) - (wt - 1) * 0.5
        self.register_buffer("dr", torch.as_tensor(rr.ravel(), dtype=torch.float32))
        self.register_buffer("dc", torch.as_tensor(cc.ravel(), dtype=torch.float32))
        if constraint_params is not None:
            self.apply_constraint_params(constraint_params, strict=True)
        if bool(self.hard_constraints.get("force_positive", False)):
            self._enforce_positivity()

    @classmethod
    def from_array(
        cls,
        *,
        name: str,
        array: np.ndarray,
        refine_all_pixels: bool = False,
        normalize: str = "none",
        origin: OriginND | None = None,
        origin_key: str = "origin",
        intensity: float | Sequence[float] = 1.0,
        constraint_params: dict[str, Any] | None = None,
    ) -> "DiskTemplate":
        return cls(
            name=name,
            array=array,
            refine_all_pixels=refine_all_pixels,
            normalize=normalize,
            origin=origin,
            origin_key=origin_key,
            intensity=intensity,
            constraint_params=constraint_params,
        )

    def set_origin(self, origin: OriginND) -> None:
        self.origin = origin

    def patch_values(self) -> torch.Tensor:
        return self.template_raw.reshape(-1)

    def patch_offsets(self) -> tuple[torch.Tensor, torch.Tensor]:
        return cast(torch.Tensor, self.dr), cast(torch.Tensor, self.dc)

    def add_patch(
        self, out: torch.Tensor, *, r0: torch.Tensor, c0: torch.Tensor, scale: torch.Tensor
    ) -> None:
        vals = self.patch_values().to(device=out.device, dtype=out.dtype)
        dr = cast(torch.Tensor, self.dr).to(device=out.device, dtype=out.dtype)
        dc = cast(torch.Tensor, self.dc).to(device=out.device, dtype=out.dtype)
        _splat_patch(out, r0=r0, c0=c0, patch_vals=vals, dr=dr, dc=dc, scale=scale)

    def forward(self, ctx: RenderContext) -> torch.Tensor:
        out = torch.zeros(ctx.shape, device=ctx.device, dtype=ctx.dtype)
        if self.origin is None:
            raise RuntimeError("DiskTemplate.forward() requires an OriginND instance.")
        r0, c0 = self.origin.coords[0], self.origin.coords[1]
        self.add_patch(out, r0=r0, c0=c0, scale=torch.tensor(1.0))  # scale learned directly
        return out

    def _center_disk(self) -> None:
        with torch.no_grad():
            template = self.template_raw
            h, w = int(template.shape[0]), int(template.shape[1])
            weights = torch.clamp(template, min=0.0)
            mass = torch.sum(weights)
            if float(mass.detach().cpu()) <= 1e-12:
                return
            rr = torch.arange(h, device=template.device, dtype=template.dtype)[:, None]
            cc = torch.arange(w, device=template.device, dtype=template.dtype)[None, :]
            com_r = torch.sum(weights * rr) / mass
            com_c = torch.sum(weights * cc) / mass
            target_r = torch.as_tensor((h - 1) * 0.5, device=template.device, dtype=template.dtype)
            target_c = torch.as_tensor((w - 1) * 0.5, device=template.device, dtype=template.dtype)
            shift_r = target_r - com_r
            shift_c = target_c - com_c
            denom_h = max(h - 1, 1)
            denom_w = max(w - 1, 1)
            ty = -2.0 * shift_r / float(denom_h)
            tx = -2.0 * shift_c / float(denom_w)
            theta = torch.as_tensor(
                [[1.0, 0.0, tx], [0.0, 1.0, ty]],
                device=template.device,
                dtype=template.dtype,
            )[None, ...]
            src = template[None, None, :, :]
            grid = F.affine_grid(theta, [1, 1, h, w], align_corners=True)
            shifted = F.grid_sample(
                src,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[0, 0]
            self.template_raw.copy_(shifted)

    def _enforce_positivity(self) -> None:
        with torch.no_grad():
            self.template_raw.clamp_(min=0.0)

    def enforce_hard_constraints(self, ctx: RenderContext) -> None:
        if bool(self.hard_constraints.get("force_center", False)):
            self._center_disk()
        if bool(self.hard_constraints.get("force_positive", False)):
            self._enforce_positivity()

    def constraint_loss(
        self, ctx: RenderContext, params: dict[str, object] | None = None
    ) -> torch.Tensor:
        cfg = self.effective_soft_constraints(cast(dict[str, object] | None, params))
        tv_weight = float(cfg.get("tv_weight", 0.0))
        if tv_weight <= 0.0:
            return torch.zeros((), device=ctx.device, dtype=ctx.dtype)
        template = self.template_raw.to(device=ctx.device, dtype=ctx.dtype)
        tv_r = (
            torch.mean(torch.abs(template[1:, :] - template[:-1, :]))
            if template.shape[0] > 1
            else torch.zeros((), device=ctx.device, dtype=ctx.dtype)
        )
        tv_c = (
            torch.mean(torch.abs(template[:, 1:] - template[:, :-1]))
            if template.shape[1] > 1
            else torch.zeros((), device=ctx.device, dtype=ctx.dtype)
        )
        return torch.as_tensor(tv_weight, device=ctx.device, dtype=ctx.dtype) * (tv_r + tv_c)


class SyntheticDiskLattice(RenderComponent):
    def __init__(
        self,
        *,
        name: str,
        disk: DiskTemplate,
        u_row: float | Sequence[float],
        u_col: float | Sequence[float],
        v_row: float | Sequence[float],
        v_col: float | Sequence[float],
        u_max: int = 0,
        v_max: int = 0,
        intensity_0: float | Sequence[float] = 0.0,
        intensity_row: float | Sequence[float] = 0.0,
        intensity_col: float | Sequence[float] = 0.0,
        intensity_row_row: float | Sequence[float] = 0.0,
        intensity_col_col: float | Sequence[float] = 0.0,
        intensity_row_col: float | Sequence[float] = 0.0,
        per_disk_intensity: bool = False,
        per_disk_slopes: bool = True,
        max_intensity_order: int | None = None,
        default_pattern_intensity_order: int | None = None,
        center_intensity_0: float | Sequence[float] | None = None,
        exclude_indices: Iterable[tuple[int, int]] | None = None,
        boundary_px: float = 0.0,
        origin: OriginND | None = None,
        origin_key: str = "origin",
        constraint_params: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.name = str(name)
        self.disk = disk
        self.origin = origin
        self.origin_key = str(origin_key)
        self.per_disk_intensity = bool(per_disk_intensity)
        self.u_max = int(u_max)
        self.v_max = int(v_max)
        self.boundary_px = float(boundary_px)

        if max_intensity_order is None:
            max_intensity_order = 1 if bool(per_disk_slopes) else 0
        self.max_intensity_order = int(max_intensity_order)
        if self.max_intensity_order < 0 or self.max_intensity_order > 2:
            raise ValueError("max_intensity_order must be 0, 1, or 2.")

        if default_pattern_intensity_order is None:
            default_pattern_intensity_order = self.max_intensity_order
        self.default_pattern_intensity_order = int(default_pattern_intensity_order)

        self.u_row = nn.Parameter(
            torch.tensor(_parse_init(u_row, name="u_row"), dtype=torch.float32)
        )
        self.u_col = nn.Parameter(
            torch.tensor(_parse_init(u_col, name="u_col"), dtype=torch.float32)
        )
        self.v_row = nn.Parameter(
            torch.tensor(_parse_init(v_row, name="v_row"), dtype=torch.float32)
        )
        self.v_col = nn.Parameter(
            torch.tensor(_parse_init(v_col, name="v_col"), dtype=torch.float32)
        )

        if exclude_indices is None:
            exclude = set()
        else:
            exclude = set(exclude_indices)
        uv: list[tuple[int, int]] = []
        for u in range(-self.u_max, self.u_max + 1):
            for v in range(-self.v_max, self.v_max + 1):
                if (u, v) not in exclude:
                    uv.append((u, v))
        uv_t = (
            torch.as_tensor(uv, dtype=torch.long) if uv else torch.zeros((0, 2), dtype=torch.long)
        )
        self.register_buffer("uv_indices", uv_t)

        n_uv = int(uv_t.shape[0])
        i0_init = _parse_init(intensity_0, name="intensity_0")
        i0_center = (
            i0_init
            if center_intensity_0 is None
            else _parse_init(center_intensity_0, name="center_intensity_0")
        )
        ir_init = _parse_init(intensity_row, name="intensity_row")
        ic_init = _parse_init(intensity_col, name="intensity_col")
        irr_init = _parse_init(intensity_row_row, name="intensity_row_row")
        icc_init = _parse_init(intensity_col_col, name="intensity_col_col")
        irc_init = _parse_init(intensity_row_col, name="intensity_row_col")

        if self.per_disk_intensity:
            i0_values = torch.full((n_uv,), float(i0_init), dtype=torch.float32)
            if n_uv > 0:
                center_mask = (uv_t[:, 0] == 0) & (uv_t[:, 1] == 0)
                i0_values[center_mask] = float(i0_center)
            self.i0_raw = nn.Parameter(i0_values)
            if self.max_intensity_order >= 1:
                self.ir = nn.Parameter(torch.full((n_uv,), float(ir_init), dtype=torch.float32))
                self.ic = nn.Parameter(torch.full((n_uv,), float(ic_init), dtype=torch.float32))
            else:
                self.ir = None
                self.ic = None
            if self.max_intensity_order >= 2:
                self.irr = nn.Parameter(torch.full((n_uv,), float(irr_init), dtype=torch.float32))
                self.icc = nn.Parameter(torch.full((n_uv,), float(icc_init), dtype=torch.float32))
                self.irc = nn.Parameter(torch.full((n_uv,), float(irc_init), dtype=torch.float32))
            else:
                self.irr = None
                self.icc = None
                self.irc = None
        else:
            self.i0_raw = nn.Parameter(torch.tensor(i0_init, dtype=torch.float32))
            self.ir = nn.Parameter(torch.tensor(ir_init, dtype=torch.float32))
            self.ic = nn.Parameter(torch.tensor(ic_init, dtype=torch.float32))
            self.irr = nn.Parameter(torch.tensor(irr_init, dtype=torch.float32))
            self.icc = nn.Parameter(torch.tensor(icc_init, dtype=torch.float32))
            self.irc = nn.Parameter(torch.tensor(irc_init, dtype=torch.float32))
        if constraint_params is not None:
            self.apply_constraint_params(constraint_params, strict=True)

    def set_origin(self, origin: OriginND) -> None:
        self.origin = origin

    def forward(self, ctx: RenderContext) -> torch.Tensor:
        if self.origin is None:
            raise RuntimeError("SyntheticDiskLattice requires an OriginND instance.")

        out = torch.zeros(ctx.shape, device=ctx.device, dtype=ctx.dtype)
        uv_indices = cast(torch.Tensor, self.uv_indices)
        if torch.numel(uv_indices) == 0:
            return out

        uv = torch.as_tensor(uv_indices, device=ctx.device)
        u = uv[:, 0].to(dtype=ctx.dtype)
        v = uv[:, 1].to(dtype=ctx.dtype)
        r0, c0 = self.origin.coords[0], self.origin.coords[1]
        centers_r = r0 + u * self.u_row + v * self.v_row
        centers_c = c0 + u * self.u_col + v * self.v_col

        b = torch.as_tensor(self.boundary_px, device=ctx.device, dtype=ctx.dtype)
        keep = (centers_r >= b) & (centers_r <= (ctx.shape[0] - 1) - b)
        keep = keep & (centers_c >= b) & (centers_c <= (ctx.shape[1] - 1) - b)
        keep_idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
        if keep_idx.numel() == 0:
            return out

        active_order = int(
            ctx.fields.get(
                "lattice_intensity_order_override", self.default_pattern_intensity_order
            )
        )
        active_order = max(0, min(active_order, self.max_intensity_order))

        dr, dc = self.disk.patch_offsets()
        dr = dr.to(device=ctx.device, dtype=ctx.dtype)
        dc = dc.to(device=ctx.device, dtype=ctx.dtype)
        dr2 = dr * dr
        dc2 = dc * dc
        drdc = dr * dc

        for j in keep_idx:
            rr0 = centers_r[j]
            cc0 = centers_c[j]

            if self.per_disk_intensity:
                inten = torch.clamp(self.i0_raw[j], min=0.0)
                if active_order >= 1 and self.ir is not None and self.ic is not None:
                    inten = inten + self.ir[j] * dr + self.ic[j] * dc
                if (
                    active_order >= 2
                    and self.irr is not None
                    and self.icc is not None
                    and self.irc is not None
                ):
                    inten = inten + self.irr[j] * dr2 + self.icc[j] * dc2 + self.irc[j] * drdc
                inten = torch.clamp(inten, min=0.0)
            else:
                inten = torch.clamp(self.i0_raw, min=0.0)
                if active_order >= 1:
                    assert self.ir is not None and self.ic is not None
                    inten = inten + self.ir * rr0 + self.ic * cc0
                if active_order >= 2:
                    assert self.irr is not None and self.icc is not None and self.irc is not None
                    inten = (
                        inten + self.irr * rr0 * rr0 + self.icc * cc0 * cc0 + self.irc * rr0 * cc0
                    )
                inten = torch.clamp(inten, min=0.0)

            self.disk.add_patch(out, r0=rr0, c0=cc0, scale=inten)

        return out
