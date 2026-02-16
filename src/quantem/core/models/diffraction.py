from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from quantem.core.models.base import Component, ModelContext, Overlay, Parameter

__all__ = ["DiskTemplate", "DiskAtOrigin", "SyntheticDiskLattice"]


def _as_tensor(x: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _keep_center(r: float, c: float, H: int, W: int, boundary_px: float) -> bool:
    b = float(boundary_px)
    return (r >= b) and (r <= (H - 1) - b) and (c >= b) and (c <= (W - 1) - b)


def _place_patch_add(out: torch.Tensor, patch: torch.Tensor, center_r: float, center_c: float) -> None:
    H, W = int(out.shape[-2]), int(out.shape[-1])
    ph, pw = int(patch.shape[-2]), int(patch.shape[-1])
    pr = ph // 2
    pc = pw // 2

    r0 = int(np.rint(center_r)) - pr
    c0 = int(np.rint(center_c)) - pc
    r1 = r0 + ph
    c1 = c0 + pw

    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(H, r1)
    cc1 = min(W, c1)
    if rr1 <= rr0 or cc1 <= cc0:
        return

    pr0 = rr0 - r0
    pc0 = cc0 - c0
    pr1 = pr0 + (rr1 - rr0)
    pc1 = pc0 + (cc1 - cc0)

    out[rr0:rr1, cc0:cc1] += patch[pr0:pr1, pc0:pc1]


class DiskTemplate(Component):
    def __init__(
        self,
        *,
        name: str,
        array: np.ndarray,
        refine_all_pixels: bool = False,
        place_at_origin: bool = False,
        origin_key: str = "origin",
        origin_intensity: float | tuple[float, float] | tuple[float, float, float | None] = 1.0,
    ):
        super().__init__(name=name)
        arr = np.asarray(array, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("DiskTemplate.array must be 2D.")
        self.shape = (int(arr.shape[0]), int(arr.shape[1]))
        self.refine_all_pixels = bool(refine_all_pixels)
        self.place_at_origin = bool(place_at_origin)
        self.origin_key = str(origin_key)

        self._array0 = arr
        self.p_origin_intensity = Parameter(origin_intensity, tags={"role": "disk_origin_intensity", "name": name})

        self.p_pixels: list[Parameter] = []
        if self.refine_all_pixels:
            flat = arr.reshape(-1)
            for i, v in enumerate(flat):
                self.p_pixels.append(Parameter(float(v), tags={"role": "disk_pixel", "name": name, "pixel_index": int(i)}))

    @classmethod
    def from_array(
        cls,
        *,
        name: str,
        array: np.ndarray,
        refine_all_pixels: bool = False,
        place_at_origin: bool = False,
        origin_key: str = "origin",
        origin_intensity: float | tuple[float, float] | tuple[float, float, float | None] = 1.0,
    ) -> "DiskTemplate":
        return cls(
            name=name,
            array=array,
            refine_all_pixels=refine_all_pixels,
            place_at_origin=place_at_origin,
            origin_key=origin_key,
            origin_intensity=origin_intensity,
        )

    def prepare(self, ctx: ModelContext) -> Any:
        disk = self
        ok = disk.origin_key

        if disk.refine_all_pixels:
            idxs = [p._index for p in disk.p_pixels]
            if any(i is None for i in idxs):
                raise RuntimeError("DiskTemplate pixel parameters not bound.")
            idx_t = torch.as_tensor([int(i) for i in idxs], device=ctx.device, dtype=torch.int64)

            def patch(x: torch.Tensor) -> torch.Tensor:
                return x.index_select(0, idx_t).to(device=ctx.device, dtype=ctx.dtype).reshape(disk.shape[0], disk.shape[1])

        else:
            patch0 = _as_tensor(disk._array0, device=ctx.device, dtype=ctx.dtype)

            def patch(x: torch.Tensor) -> torch.Tensor:
                return patch0

        @dataclass
        class PreparedDiskTemplate:
            origin_key: str
            place_at_origin: bool

            def render(self, out: torch.Tensor, x: torch.Tensor, ctx: ModelContext) -> None:
                if not self.place_at_origin:
                    return
                if self.origin_key not in ctx.origins:
                    return
                r0, c0 = ctx.origins[self.origin_key]
                r = float(r0.detach().cpu().item())
                c = float(c0.detach().cpu().item())
                inten = disk.p_origin_intensity.value(x, device=ctx.device, dtype=ctx.dtype)
                _place_patch_add(out, patch(x) * inten, r, c)

            def overlays(self) -> list[Overlay]:
                return []

        return PreparedDiskTemplate(origin_key=ok, place_at_origin=disk.place_at_origin)


class SyntheticDiskLattice(Component):
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
        exclude_indices: Iterable[tuple[int, int]] | None = [],
        boundary_px: float = 0.0,
        origin_key: str = "origin",
    ):
        super().__init__(name=name)
        self.disk = disk
        self.origin_key = str(origin_key)
        self.u_max = int(u_max)
        self.v_max = int(v_max)
        self.boundary_px = float(boundary_px)

        if exclude_indices is None:
            self.exclude_indices = {(0, 0)}
        else:
            self.exclude_indices = {(int(a), int(b)) for (a, b) in exclude_indices}

        self.p_u_row = Parameter(u_row, tags={"role": "lattice_u_row", "name": name})
        self.p_u_col = Parameter(u_col, tags={"role": "lattice_u_col", "name": name})
        self.p_v_row = Parameter(v_row, tags={"role": "lattice_v_row", "name": name})
        self.p_v_col = Parameter(v_col, tags={"role": "lattice_v_col", "name": name})

        self.p_intensity_0 = Parameter(intensity_0, tags={"role": "disk_intensity_0", "name": name})
        self.p_intensity_row = Parameter(intensity_row, tags={"role": "disk_intensity_row", "name": name})
        self.p_intensity_col = Parameter(intensity_col, tags={"role": "disk_intensity_col", "name": name})

    def _uv_list(self) -> list[tuple[int, int]]:
        uv: list[tuple[int, int]] = []
        for u in range(-self.u_max, self.u_max + 1):
            for v in range(-self.v_max, self.v_max + 1):
                if (u, v) in self.exclude_indices:
                    continue
                uv.append((u, v))
        return uv

    def prepare(self, ctx: ModelContext) -> Any:
        lat = self
        ok = lat.origin_key
        uv = lat._uv_list()

        if lat.disk.refine_all_pixels:
            didxs = [p._index for p in lat.disk.p_pixels]
            if any(i is None for i in didxs):
                raise RuntimeError("DiskTemplate pixel parameters not bound.")
            didx_t = torch.as_tensor([int(i) for i in didxs], device=ctx.device, dtype=torch.int64)

            def patch(x: torch.Tensor) -> torch.Tensor:
                return x.index_select(0, didx_t).to(device=ctx.device, dtype=ctx.dtype).reshape(lat.disk.shape[0], lat.disk.shape[1])

        else:
            patch0 = _as_tensor(lat.disk._array0, device=ctx.device, dtype=ctx.dtype)

            def patch(x: torch.Tensor) -> torch.Tensor:
                return patch0

        r0_list: list[float] = []
        c0_list: list[float] = []
        if ok in ctx.origins:
            or0, oc0 = ctx.origins[ok]
            orow0 = float(or0.detach().cpu().item())
            ocol0 = float(oc0.detach().cpu().item())
            urow0 = float(lat.p_u_row.initial_value)
            ucol0 = float(lat.p_u_col.initial_value)
            vrow0 = float(lat.p_v_row.initial_value)
            vcol0 = float(lat.p_v_col.initial_value)
            for u, v in uv:
                rr = orow0 + u * urow0 + v * vrow0
                cc = ocol0 + u * ucol0 + v * vcol0
                if _keep_center(rr, cc, ctx.H, ctx.W, lat.boundary_px):
                    r0_list.append(rr)
                    c0_list.append(cc)

        r0_np = np.asarray(r0_list, dtype=np.float32)
        c0_np = np.asarray(c0_list, dtype=np.float32)

        @dataclass
        class PreparedSyntheticDiskLattice:
            origin_key: str
            boundary_px: float
            uv: list[tuple[int, int]]
            r0: np.ndarray
            c0: np.ndarray

            def render(self, out: torch.Tensor, x: torch.Tensor, ctx: ModelContext) -> None:
                if self.origin_key not in ctx.origins:
                    return
                if not self.uv:
                    return

                r_origin, c_origin = ctx.origins[self.origin_key]
                r_origin = r_origin.reshape(())
                c_origin = c_origin.reshape(())

                urow = lat.p_u_row.value(x, device=ctx.device, dtype=ctx.dtype)
                ucol = lat.p_u_col.value(x, device=ctx.device, dtype=ctx.dtype)
                vrow = lat.p_v_row.value(x, device=ctx.device, dtype=ctx.dtype)
                vcol = lat.p_v_col.value(x, device=ctx.device, dtype=ctx.dtype)

                I0 = lat.p_intensity_0.value(x, device=ctx.device, dtype=ctx.dtype)
                Ir = lat.p_intensity_row.value(x, device=ctx.device, dtype=ctx.dtype)
                Ic = lat.p_intensity_col.value(x, device=ctx.device, dtype=ctx.dtype)

                uv_t = torch.as_tensor(self.uv, device=ctx.device, dtype=torch.int32)
                uu = uv_t[:, 0].to(device=ctx.device, dtype=ctx.dtype)
                vv = uv_t[:, 1].to(device=ctx.device, dtype=ctx.dtype)

                rr = r_origin + uu * urow + vv * vrow
                cc = c_origin + uu * ucol + vv * vcol

                p = patch(x)

                H = int(ctx.H)
                W = int(ctx.W)
                for i in range(int(rr.numel())):
                    r = float(rr[i].detach().cpu().item())
                    c = float(cc[i].detach().cpu().item())
                    if not _keep_center(r, c, H, W, self.boundary_px):
                        continue
                    dr = rr[i] - r_origin
                    dc = cc[i] - c_origin
                    inten = torch.clamp(I0 + Ir * dr + Ic * dc, min=0.0)
                    _place_patch_add(out, p * inten, r, c)

            def overlays(self) -> list[Overlay]:
                if self.r0.size == 0:
                    return []
                return [
                    Overlay(
                        kind="points_rc",
                        data={"r": self.r0, "c": self.c0, "marker": "x", "s": 60.0, "color": "orange"},
                    )
                ]

        return PreparedSyntheticDiskLattice(
            origin_key=ok,
            boundary_px=lat.boundary_px,
            uv=uv,
            r0=r0_np,
            c0=c0_np,
        )
