from __future__ import annotations

"""Diffraction-specific model components and low-level rendering helpers."""

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from quantem.core.models.base import Component, ModelContext, Overlay, Parameter


def _as_t(x: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert values to tensors on the requested device/dtype.

    Parameters
    ----------
    x
        Input scalar/array/tensor.
    device, dtype
        Target torch device and dtype.
    """
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


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
    """
    Render a shifted patch into ``out`` with bilinear interpolation.

    Parameters
    ----------
    out
        Destination image `(H, W)` updated in-place.
    r0, c0
        Patch center coordinates in output-image row/column space.
    patch_vals
        Flattened patch intensities.
    dr, dc
        Flattened local row/column offsets for each patch sample.
    scale
        Scalar or vector multiplier applied to `patch_vals`.
    """
    H = out.shape[0]
    W = out.shape[1]

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
        m = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        if torch.any(m):
            out.index_put_((rr[m], cc[m]), (v[m] * ww[m]), accumulate=True)

    put(r0i, c0i, w00)
    put(r0i, c0i + 1, w01)
    put(r0i + 1, c0i, w10)
    put(r0i + 1, c0i + 1, w11)


def _origin_indices(ctx: ModelContext, origin_key: str) -> tuple[int, int]:
    """
    Resolve origin parameter indices from ``ModelContext``.

    Parameters
    ----------
    ctx
        Model context with an `origins` registry in `ctx.fields`.
    origin_key
        Origin name to resolve.
    """
    origins: dict[str, Any] = ctx.fields.get("origins", {})
    o = origins.get(origin_key)
    if o is None or "row_param" not in o or "col_param" not in o:
        raise RuntimeError(f"Origin '{origin_key}' not defined. Origin2D must be included first.")
    return int(o["row_param"].index), int(o["col_param"].index)


class Origin2D(Component):
    """
    Named 2D origin component that exposes row/column fit parameters.

    Parameters
    ----------
    name
        Component name.
    origin_key
        Key used to publish this origin into `ctx.fields["origins"]`.
    row, col
        Parameter specifications for origin row/column.
    """

    def __init__(
        self,
        *,
        name: str = "origin",
        origin_key: str = "origin",
        row: float | tuple[float, float] | tuple[float, float, float | None] = (0.0, 0.0, None),
        col: float | tuple[float, float] | tuple[float, float, float | None] = (0.0, 0.0, None),
    ):
        super().__init__(name=name)
        self.origin_key = str(origin_key)
        self.p_row = Parameter(row, tags={"role": "origin_row", "origin_key": self.origin_key})
        self.p_col = Parameter(col, tags={"role": "origin_col", "origin_key": self.origin_key})

    def parameters(self) -> list[Parameter]:
        return [self.p_row, self.p_col]

    def prepare(self, ctx: ModelContext) -> Any:
        origins = ctx.fields.setdefault("origins", {})
        origins[self.origin_key] = {"row_param": self.p_row, "col_param": self.p_col}

        i_r = self.p_row.index
        i_c = self.p_col.index

        @dataclass
        class Prepared:
            def render(self, out: torch.Tensor, x: torch.Tensor, ctx: ModelContext) -> None:
                return

            def overlays(self, x: torch.Tensor, ctx: ModelContext) -> Iterable[Overlay]:
                return [
                    Overlay(
                        kind="points_rc",
                        data={
                            "r": x[i_r].detach(),
                            "c": x[i_c].detach(),
                            "marker": "+",
                            "s": 100.0,
                            "color": "dodgerblue",
                        },
                    )
                ]

        return Prepared()


class DiskTemplate(Component):
    """
    Template patch component used for central disks and lattice motifs.

    Parameters
    ----------
    name
        Template name used for registry lookup by dependent components.
    array
        2D template image.
    refine_all_pixels
        If True, every template pixel is exposed as a fit parameter.
    place_at_origin
        If True, render this template once at the named origin with a
        separate nonnegative intensity parameter.
    normalize
        Optional array normalization mode: `"none"`, `"max"`, `"mean"`.
    origin_key
        Origin key used when `place_at_origin=True`.
    intensity
        Intensity parameter specification used when `place_at_origin=True`.

    Notes
    -----
    The template is also published into `ctx.fields["disk_templates"]` so
    lattice components can reuse its prepared sampling offsets.
    """

    def __init__(
        self,
        *,
        name: str,
        array: np.ndarray,
        refine_all_pixels: bool = False,
        place_at_origin: bool = False,
        normalize: str = "none",
        origin_key: str = "origin",
        intensity: float | tuple[float, float] | tuple[float, float, float | None] = 1.0,
    ):
        super().__init__(name=name)

        a = np.asarray(array, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError("DiskTemplate.array must be 2D.")

        if normalize == "max":
            mx = float(np.max(a))
            if mx > 0:
                a = a / mx
        elif normalize == "mean":
            m = float(np.mean(a))
            if m != 0:
                a = a / m
        elif normalize == "none":
            pass
        else:
            raise ValueError("normalize must be one of: 'none', 'max', 'mean'.")

        self.array = a
        self.refine_all_pixels = bool(refine_all_pixels)
        self.place_at_origin = bool(place_at_origin)
        self.origin_key = str(origin_key)

        self.p_pixels: list[Parameter] = []
        if self.refine_all_pixels:
            flat = self.array.ravel()
            for i, v in enumerate(flat):
                self.p_pixels.append(
                    Parameter(
                        float(v),
                        lower_bound=0.0,
                        tags={"role": "disk_pixel", "disk": self.name, "i": int(i)},
                    )
                )

        self.p_intensity: Parameter | None = None
        if self.place_at_origin:
            self.p_intensity = Parameter(intensity, lower_bound=0.0, tags={"role": "disk_intensity", "disk": self.name})

    @classmethod
    def from_array(
        cls,
        *,
        name: str,
        array: np.ndarray,
        refine_all_pixels: bool = False,
        place_at_origin: bool = False,
        normalize: str = "none",
        origin_key: str = "origin",
        intensity: float | tuple[float, float] | tuple[float, float, float | None] = 1.0,
    ) -> "DiskTemplate":
        return cls(
            name=name,
            array=array,
            refine_all_pixels=refine_all_pixels,
            place_at_origin=place_at_origin,
            normalize=normalize,
            origin_key=origin_key,
            intensity=intensity,
        )

    def parameters(self) -> list[Parameter]:
        out = list(self.p_pixels)
        if self.p_intensity is not None:
            out.append(self.p_intensity)
        return out

    def prepare(self, ctx: ModelContext) -> Any:
        device = ctx.device
        dtype = ctx.dtype

        Ht, Wt = self.array.shape
        rr, cc = np.mgrid[0:Ht, 0:Wt]
        rr = rr.astype(np.float32) - (Ht - 1) * 0.5
        cc = cc.astype(np.float32) - (Wt - 1) * 0.5

        dr = _as_t(rr.ravel(), device=device, dtype=dtype)
        dc = _as_t(cc.ravel(), device=device, dtype=dtype)

        if self.refine_all_pixels:
            pix_idx = torch.as_tensor([p.index for p in self.p_pixels], device=device, dtype=torch.long)
            base_vals = None
        else:
            pix_idx = None
            base_vals = _as_t(self.array.ravel(), device=device, dtype=dtype)

        ctx.fields.setdefault("disk_templates", {})[self.name] = {
            "dr": dr,
            "dc": dc,
            "pix_idx": pix_idx,
            "base": base_vals,
        }

        i_int = None if self.p_intensity is None else self.p_intensity.index
        if i_int is None:
            r_idx = c_idx = None
        else:
            r_idx, c_idx = _origin_indices(ctx, self.origin_key)

        @dataclass
        class Prepared:
            def patch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if pix_idx is None:
                    vals = base_vals
                else:
                    vals = x[pix_idx]
                return vals, dr, dc

            def render(self, out: torch.Tensor, x: torch.Tensor, ctx: ModelContext) -> None:
                if i_int is None:
                    return
                vals, drl, dcl = self.patch(x)
                r0 = x[r_idx]
                c0 = x[c_idx]
                _splat_patch(out, r0=r0, c0=c0, patch_vals=vals, dr=drl, dc=dcl, scale=x[i_int])

            def overlays(self, x: torch.Tensor, ctx: ModelContext) -> Iterable[Overlay]:
                return []

        return Prepared()


class SyntheticDiskLattice(Component):
    """
    Lattice of shifted ``DiskTemplate`` patches around a shared origin.

    Parameters
    ----------
    name
        Component name.
    disk
        `DiskTemplate` used as the motif for each lattice spot.
    u_row, u_col, v_row, v_col
        Basis vector parameter specifications in row/column coordinates.
    u_max, v_max
        Inclusive lattice index extents.
    intensity_0, intensity_row, intensity_col
        Intensity parameter specifications. Interpretation depends on
        `per_disk_intensity` and `per_disk_slopes`.
    per_disk_intensity
        If True, allocate independent base intensities per lattice disk.
    per_disk_slopes
        If True and `per_disk_intensity=True`, also allocate per-disk local
        slope parameters for template-local coordinates.
    center_intensity_0
        Optional override for the `(u, v) == (0, 0)` disk base intensity.
    exclude_indices
        Optional set/list of lattice indices to skip.
    boundary_px
        Keep only lattice centers within `[boundary_px, size-1-boundary_px]`.
    origin_key
        Origin key used for center placement.

    Notes
    -----
    Intensity models:
    - shared mode (`per_disk_intensity=False`):
      `max(i0 + ir*row_center + ic*col_center, 0)`
    - per-disk scalar mode (`per_disk_intensity=True`, `per_disk_slopes=False`):
      `max(i0_i, 0)`
    - per-disk local affine mode (`per_disk_intensity=True`, `per_disk_slopes=True`):
      `max(i0_i + ir_i*dr + ic_i*dc, 0)` where `dr/dc` are template-local offsets.
    """

    def __init__(
        self,
        *,
        name: str,
        disk: DiskTemplate,
        u_row: float | tuple[float, float] | tuple[float, float, float | None],
        u_col: float | tuple[float, float] | tuple[float, float, float | None],
        v_row: float | tuple[float, float] | tuple[float, float, float | None],
        v_col: float | tuple[float, float] | tuple[float, float, float | None],
        u_max: int = 0,
        v_max: int = 0,
        intensity_0: float | tuple[float, float] | tuple[float, float, float | None] = 0.0,
        intensity_row: float | tuple[float, float] | tuple[float, float, float | None] = 0.0,
        intensity_col: float | tuple[float, float] | tuple[float, float, float | None] = 0.0,
        per_disk_intensity: bool = False,
        per_disk_slopes: bool = True,
        center_intensity_0: float | tuple[float, float] | tuple[float, float, float | None] | None = None,
        exclude_indices: Iterable[tuple[int, int]] | None = None,
        boundary_px: float = 0.0,
        origin_key: str = "origin",
    ):
        super().__init__(name=name)
        self.disk = disk
        self.u_max = int(u_max)
        self.v_max = int(v_max)
        self.boundary_px = float(boundary_px)
        self.origin_key = str(origin_key)
        self.per_disk_intensity = bool(per_disk_intensity)
        self.per_disk_slopes = bool(per_disk_slopes)

        if exclude_indices is None:
            self.exclude_indices = {(0, 0)} if bool(getattr(disk, "place_at_origin", False)) else set()
        else:
            self.exclude_indices = set(exclude_indices)

        uv: list[tuple[int, int]] = []
        for u in range(-self.u_max, self.u_max + 1):
            for v in range(-self.v_max, self.v_max + 1):
                if (u, v) in self.exclude_indices:
                    continue
                uv.append((u, v))
        self.uv_indices = uv

        self.p_u_row = Parameter(u_row, tags={"role": "lat_u_row"})
        self.p_u_col = Parameter(u_col, tags={"role": "lat_u_col"})
        self.p_v_row = Parameter(v_row, tags={"role": "lat_v_row"})
        self.p_v_col = Parameter(v_col, tags={"role": "lat_v_col"})
        self.p_i0 = None
        self.p_ir = None
        self.p_ic = None
        self.p_i0_list: list[Parameter] = []
        self.p_ir_list: list[Parameter] = []
        self.p_ic_list: list[Parameter] = []
        if self.per_disk_intensity:
            for u, v in self.uv_indices:
                i0_val = center_intensity_0 if (center_intensity_0 is not None and (u, v) == (0, 0)) else intensity_0
                self.p_i0_list.append(Parameter(i0_val, lower_bound=0.0, tags={"role": "lat_int0", "u": u, "v": v}))
                if self.per_disk_slopes:
                    self.p_ir_list.append(Parameter(intensity_row, tags={"role": "lat_int_row", "u": u, "v": v}))
                    self.p_ic_list.append(Parameter(intensity_col, tags={"role": "lat_int_col", "u": u, "v": v}))
        else:
            self.p_i0 = Parameter(intensity_0, lower_bound=0.0, tags={"role": "lat_int0"})
            self.p_ir = Parameter(intensity_row, tags={"role": "lat_int_row"})
            self.p_ic = Parameter(intensity_col, tags={"role": "lat_int_col"})

    def parameters(self) -> list[Parameter]:
        out = [self.p_u_row, self.p_u_col, self.p_v_row, self.p_v_col]
        if self.per_disk_intensity:
            out.extend(self.p_i0_list)
            if self.per_disk_slopes:
                out.extend(self.p_ir_list)
                out.extend(self.p_ic_list)
        else:
            out.extend([self.p_i0, self.p_ir, self.p_ic])
        return out

    def prepare(self, ctx: ModelContext) -> Any:
        dt = ctx.fields.get("disk_templates", {})
        if self.disk.name not in dt:
            raise RuntimeError("DiskTemplate must be included before SyntheticDiskLattice in components.")
        d = dt[self.disk.name]
        dr = d["dr"]
        dc = d["dc"]
        pix_idx = d["pix_idx"]
        base_vals = d["base"]

        r_idx, c_idx = _origin_indices(ctx, self.origin_key)

        i_ur = self.p_u_row.index
        i_uc = self.p_u_col.index
        i_vr = self.p_v_row.index
        i_vc = self.p_v_col.index
        if self.per_disk_intensity:
            i_i0 = i_ir = i_ic = None
        else:
            i_i0 = self.p_i0.index
            i_ir = self.p_ir.index
            i_ic = self.p_ic.index

        uv = self.uv_indices
        uv_t = torch.as_tensor(uv, device=ctx.device, dtype=torch.long) if uv else None

        boundary = torch.as_tensor(self.boundary_px, device=ctx.device, dtype=ctx.dtype)

        if self.per_disk_intensity:
            i0_idx = torch.as_tensor([p.index for p in self.p_i0_list], device=ctx.device, dtype=torch.long)
            if self.per_disk_slopes:
                ir_idx = torch.as_tensor([p.index for p in self.p_ir_list], device=ctx.device, dtype=torch.long)
                ic_idx = torch.as_tensor([p.index for p in self.p_ic_list], device=ctx.device, dtype=torch.long)
            else:
                ir_idx = ic_idx = None
        else:
            i0_idx = ir_idx = ic_idx = None
        per_disk_intensity = self.per_disk_intensity
        per_disk_slopes = self.per_disk_slopes

        @dataclass
        class Prepared:
            boundary_px: float = float(self.boundary_px)

            def _patch(self, x: torch.Tensor) -> torch.Tensor:
                if pix_idx is None:
                    return base_vals
                return x[pix_idx]

            def _centers(self, x: torch.Tensor, ctx: ModelContext) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if uv_t is None:
                    z = torch.empty((0,), device=ctx.device, dtype=ctx.dtype)
                    return z, z, z

                r0 = x[r_idx]
                c0 = x[c_idx]

                ur = x[i_ur]
                uc = x[i_uc]
                vr = x[i_vr]
                vc = x[i_vc]

                u = uv_t[:, 0].to(ctx.dtype)
                v = uv_t[:, 1].to(ctx.dtype)

                centers_r = r0 + u * ur + v * vr
                centers_c = c0 + u * uc + v * vc

                keep = (centers_r >= boundary) & (centers_r <= (ctx.H - 1) - boundary)
                keep &= (centers_c >= boundary) & (centers_c <= (ctx.W - 1) - boundary)
                return centers_r[keep], centers_c[keep], keep

            def render(self, out: torch.Tensor, x: torch.Tensor, ctx: ModelContext) -> None:
                if uv_t is None:
                    return

                centers_r, centers_c, keep = self._centers(x, ctx)
                if centers_r.numel() == 0:
                    return

                vals = self._patch(x)

                if per_disk_intensity:
                    keep_idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
                    i0_keep = x[i0_idx[keep_idx]]
                    if per_disk_slopes:
                        ir_keep = x[ir_idx[keep_idx]]
                        ic_keep = x[ic_idx[keep_idx]]
                        for j, (rr0, cc0) in enumerate(zip(centers_r, centers_c)):
                            inten_local = torch.clamp(i0_keep[j] + ir_keep[j] * dr + ic_keep[j] * dc, min=0.0)
                            _splat_patch(out, r0=rr0, c0=cc0, patch_vals=vals, dr=dr, dc=dc, scale=inten_local)
                    else:
                        for j, (rr0, cc0) in enumerate(zip(centers_r, centers_c)):
                            inten_local = torch.clamp(i0_keep[j], min=0.0)
                            _splat_patch(out, r0=rr0, c0=cc0, patch_vals=vals, dr=dr, dc=dc, scale=inten_local)
                else:
                    i0 = x[i_i0]
                    ir = x[i_ir]
                    ic = x[i_ic]
                    for rr0, cc0 in zip(centers_r, centers_c):
                        inten = torch.clamp(i0 + ir * rr0 + ic * cc0, min=0.0)
                        _splat_patch(out, r0=rr0, c0=cc0, patch_vals=vals, dr=dr, dc=dc, scale=inten)

            def overlays(self, x: torch.Tensor, ctx: ModelContext) -> Iterable[Overlay]:
                if uv_t is None:
                    return []
                centers_r, centers_c, _ = self._centers(x, ctx)
                if centers_r.numel() == 0:
                    return []
                return [
                    Overlay(
                        kind="points_rc",
                        data={
                            "r": centers_r.detach(),
                            "c": centers_c.detach(),
                            "marker": "x",
                            "s": 60.0,
                            "color": "orange",
                        },
                    )
                ]

        return Prepared()
