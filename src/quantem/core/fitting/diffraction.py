from __future__ import annotations

from typing import Any, Iterable, Sequence, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from quantem.core.fitting.base import OriginND, RenderComponent, RenderContext


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


def _splat_patch_batched(
    shape: tuple[int, int],
    *,
    r0: torch.Tensor,
    c0: torch.Tensor,
    vals: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    h, w = int(shape[0]), int(shape[1])
    B, N = r0.shape

    r_base = torch.floor(r0)
    c_base = torch.floor(c0)
    fr = r0 - r_base
    fc = c0 - c_base
    r0i = r_base.to(torch.long)
    c0i = c_base.to(torch.long)

    w00 = (1.0 - fr) * (1.0 - fc)
    w01 = (1.0 - fr) * fc
    w10 = fr * (1.0 - fc)
    w11 = fr * fc

    rr = torch.stack([r0i, r0i, r0i + 1, r0i + 1], dim=0)
    cc = torch.stack([c0i, c0i + 1, c0i, c0i + 1], dim=0)
    ww = torch.stack([w00, w01, w10, w11], dim=0)

    keep = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    weighted = ww * vals.unsqueeze(0) * keep.to(dtype)

    rr_c = rr.clamp(0, h - 1)
    cc_c = cc.clamp(0, w - 1)
    flat_idx = rr_c * w + cc_c

    flat_idx_b = flat_idx.permute(1, 0, 2).reshape(B, -1)
    weighted_b = weighted.permute(1, 0, 2).reshape(B, -1)

    out_flat = torch.zeros(B, h * w, device=device, dtype=dtype)
    out_flat.scatter_add_(1, flat_idx_b, weighted_b)
    return out_flat.reshape(B, h, w)


class DiskTemplate(RenderComponent):
    DEFAULT_HARD_CONSTRAINTS: dict[str, bool] = {
        "force_center": False,
        "force_positive": True,
        "force_norm": True, # force range [0,1]
        "force_shrinkage": False,
        "force_cutoff": False,
        "force_circular_mask": True,
    }
    DEFAULT_SOFT_CONSTRAINTS: dict[str, float] = {
        "tv_weight": 0.0,
        "cutoff_weight": 0.0,
        "circular_weight": 0.0,
    }
    DEFAULT_CONSTRAINT_CONFIG: dict[str, float] = {
        "soft_cutoff_threshold": 0.0,
        "soft_cutoff_target_ratio": 0.1,
        "hard_cutoff_threshold": 0.35,
        "shrinkage_amount": 0.25,
        "circular_mask_radius_fraction": 0.95,
        "circular_mask_sharpness": 0,
        "soft_circular_mask": False,
    }

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
        constraint_config: dict[str, Any] | None = None,
    ):
        """
        Build a disk template renderer centered at the shared origin.

        Parameters
        ----------
        intensity : float | Sequence[float], optional
            Trainable scalar amplitude applied to the rendered template.
            Accepts ``x``, ``(x0, delta)``, or ``(x0, lo, hi)``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``array`` is not 2D or if ``normalize`` is unsupported.

        Notes
        -----
        ``template_raw`` controls template shape and ``intensity_raw`` controls
        center-disk amplitude.
        """
        super().__init__()
        self.name = str(name)
        self.refine_all_pixels = bool(refine_all_pixels)
        self.origin = origin
        self.origin_key = str(origin_key)
        intensity_init, intensity_lo, intensity_hi = self.parse_bounded_init(
            intensity, name="intensity"
        )
        self.intensity_raw = nn.Parameter(torch.tensor(intensity_init, dtype=torch.float32))
        if intensity_lo is not None or intensity_hi is not None:
            self.register_parameter_bounds("intensity_raw", intensity_lo, intensity_hi)

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

        self.constraint_config = self.DEFAULT_CONSTRAINT_CONFIG.copy()
        if constraint_config is not None:
            self.constraint_config.update(constraint_config)
        
        if constraint_params is not None:
            self.apply_constraint_params(constraint_params, strict=True)
        if bool(self.hard_constraints.get("force_positive", False)):
            self._enforce_positivity()
        if bool(self.hard_constraints.get("force_shrinkage", False)) and bool(self.hard_constraints.get("force_positive", True)):
            raise RuntimeWarning("Setting shrinkage true and positivity false might cause negative values in disk template")

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
        constraint_config: dict[str, Any] | None = None,

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
            constraint_config=constraint_config,

        )

    def set_origin(self, origin: OriginND) -> None:
        self.origin = origin

    def set_intensity(self, value: float | int) -> None:
        """Assign ``intensity_raw`` in-place."""
        with torch.no_grad():
            self.intensity_raw.copy_(torch.as_tensor(float(value), dtype=self.intensity_raw.dtype))

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
        """
        Render template at origin with scalar amplitude ``intensity_raw``.

        Parameters
        ----------
        ctx : RenderContext
            Rendering context.

        Returns
        -------
        torch.Tensor
            Rendered center disk image.
        """
        out = torch.zeros(ctx.shape, device=ctx.device, dtype=ctx.dtype)
        if self.origin is None:
            raise RuntimeError("DiskTemplate.forward() requires an OriginND instance.")
        r0, c0 = self.origin.coords[0], self.origin.coords[1]
        scale = self.intensity_raw.to(device=ctx.device, dtype=ctx.dtype)
        self.add_patch(out, r0=r0, c0=c0, scale=scale)
        return out

    def forward_batched(
        self,
        ctx: RenderContext,
        *,
        template_raw_b: torch.Tensor,
        intensity_raw_b: torch.Tensor,
        origin_coords_b: torch.Tensor,
    ) -> torch.Tensor:
        B = template_raw_b.shape[0]
        N = int(cast(torch.Tensor, self.dr).numel())
        dr = cast(torch.Tensor, self.dr).to(device=ctx.device, dtype=ctx.dtype)
        dc = cast(torch.Tensor, self.dc).to(device=ctx.device, dtype=ctx.dtype)
        r0 = origin_coords_b[:, 0:1] + dr.unsqueeze(0)
        c0 = origin_coords_b[:, 1:2] + dc.unsqueeze(0)
        vals = template_raw_b.reshape(B, N) * intensity_raw_b.view(B, 1)
        return _splat_patch_batched(
            ctx.shape, r0=r0, c0=c0, vals=vals, device=ctx.device, dtype=ctx.dtype
        )

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
            self.intensity_raw.clamp_(min=0.0)
    
    def _enforce_norm(self) -> None: # pick value and cut off 5 percent of mean, or every iteration shrinkage, every iteration just subtract a valye of 0.01
        with torch.no_grad():
            self.template_raw -= self.template_raw.min()
            self.template_raw /= self.template_raw.max()
    
    def _enforce_shrinkage(self) -> None:
        with torch.no_grad():
            self.template_raw -= self.constraint_config["shrinkage_amount"]
    
    def _enforce_cutoff(self) -> None:
        with torch.no_grad():
            mean_val = torch.max(self.template_raw) * self.constraint_config["hard_cutoff_threshold"]
            self.template_raw[self.template_raw <= mean_val] = 0
    
    def _enforce_circular_mask(self) -> None:
        with torch.no_grad():
            h, w = self.template_raw.shape
            radius = (min(h, w) / 2.0) * self.constraint_config["circular_mask_radius_fraction"]
            
            r = torch.arange(-h/2, h/2, device=self.template_raw.device, dtype=self.template_raw.dtype)
            c = torch.arange(-w/2, w/2, device=self.template_raw.device, dtype=self.template_raw.dtype)
            rr, cc = torch.meshgrid(r, c, indexing='ij')
            circle_matrix = torch.sqrt(rr**2 + cc**2)
            
            if self.constraint_config["soft_circular_mask"]:
                mask = torch.sigmoid(self.constraint_config["circular_mask_sharpness"]*(radius-circle_matrix))
            else:
                mask = circle_matrix <= radius
            self.template_raw *= mask


    def enforce_hard_constraints(self, ctx: RenderContext) -> None:
        if bool(self.hard_constraints.get("force_center", False)):
            self._center_disk()
        if bool(self.hard_constraints.get("force_cutoff", False)):
            self._enforce_cutoff()
        if bool(self.hard_constraints.get("force_circular_mask", False)):
            self._enforce_circular_mask()
        if bool(self.hard_constraints.get("force_shrinkage", False)):
            self._enforce_shrinkage()
        if bool(self.hard_constraints.get("force_positive", False)):
            self._enforce_positivity()
        if bool(self.hard_constraints.get("force_norm", False)): # could be put in positivity
            self._enforce_norm()
        
        super().enforce_hard_constraints(ctx)

    def constraint_loss(
        self, ctx: RenderContext, params: dict[str, object] | None = None
    ) -> torch.Tensor:
        cfg = self.effective_soft_constraints(cast(dict[str, object] | None, params))
        tv_weight = float(cfg.get("tv_weight", 0.0))
        cutoff_weight = float(cfg.get("cutoff_weight", 0.0))
        circular_weight = float(cfg.get("circular_weight", 0.0))
        
        if tv_weight <= 0.0:
            tv_weight = 0.0
        if cutoff_weight <= 0.0:
            cutoff_weight = 0.0
        if circular_weight <= 0.0:
            circular_weight = 0.0

        # tv loss calculation
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
        tv_loss = torch.as_tensor(tv_weight, device=ctx.device, dtype=ctx.dtype) * (tv_r + tv_c)

        # Cutoff loss calculation
        num_px = torch.prod(torch.tensor(template.shape))
        px_under_threshold = torch.sum(template <= torch.mean(template)*self.constraint_config["soft_cutoff_threshold"])/num_px
        cutoff_loss = torch.as_tensor(cutoff_weight, device=ctx.device, dtype=ctx.dtype) * torch.maximum(
            px_under_threshold-self.constraint_config["soft_cutoff_target_ratio"], torch.as_tensor(0.0, device=ctx.device, dtype=ctx.dtype))
        
        # circular loss calculation
        h, w = template.shape
        radius = (min(h, w) / 2.0) * self.constraint_config["circular_mask_radius_fraction"]

        r = torch.arange(h, device=ctx.device, dtype=ctx.dtype) - h / 2.0
        c = torch.arange(w, device=ctx.device, dtype=ctx.dtype) - w / 2.0
        rr, cc = torch.meshgrid(r, c, indexing='ij')
        
        circle_mask = torch.sqrt(rr**2 + cc**2)

        dist_from_radius = torch.abs(circle_mask - radius)
        dist_from_radius = torch.relu(dist_from_radius)
        circular_err = torch.mean(dist_from_radius * template)
        
        circular_loss = torch.as_tensor(circular_weight, device=ctx.device, dtype=ctx.dtype) * circular_err

        return cutoff_loss + tv_loss + circular_loss

    def constraint_loss_batched(
        self,
        ctx: RenderContext,
        *,
        template_raw_b: torch.Tensor,
        params: dict[str, object] | None = None,
    ) -> torch.Tensor:
        """
        Per-sample analogue of ``constraint_loss`` for stacked templates.

        Parameters
        ----------
        template_raw_b : torch.Tensor
            Stacked templates with shape ``(B, H_t, W_t)``.

        Returns
        -------
        torch.Tensor
            Per-sample soft-constraint losses with shape ``(B,)``. Identical
            semantics to ``constraint_loss`` on each slice, but reductions are
            taken over ``dim=(1, 2)`` only.
        """
        cfg = self.effective_soft_constraints(cast(dict[str, object] | None, params))
        tv_weight = max(float(cfg.get("tv_weight", 0.0)), 0.0)
        cutoff_weight = max(float(cfg.get("cutoff_weight", 0.0)), 0.0)
        circular_weight = max(float(cfg.get("circular_weight", 0.0)), 0.0)

        template = template_raw_b.to(device=ctx.device, dtype=ctx.dtype)
        B, h, w = template.shape

        if h > 1:
            tv_r = torch.mean(torch.abs(template[:, 1:, :] - template[:, :-1, :]), dim=(1, 2))
        else:
            tv_r = torch.zeros(B, device=ctx.device, dtype=ctx.dtype)
        if w > 1:
            tv_c = torch.mean(torch.abs(template[:, :, 1:] - template[:, :, :-1]), dim=(1, 2))
        else:
            tv_c = torch.zeros(B, device=ctx.device, dtype=ctx.dtype)
        tv_loss = tv_weight * (tv_r + tv_c)

        # Per-sample cutoff (note: hard `<=` is non-differentiable; matches serial).
        per_sample_mean = template.mean(dim=(1, 2), keepdim=True)
        thresh = per_sample_mean * float(self.constraint_config["soft_cutoff_threshold"])
        frac_under = (template <= thresh).to(dtype=ctx.dtype).mean(dim=(1, 2))
        target_ratio = float(self.constraint_config["soft_cutoff_target_ratio"])
        cutoff_loss = cutoff_weight * torch.relu(frac_under - target_ratio)

        # Per-sample circular: mask depends only on (H, W), so build once.
        radius = (min(h, w) / 2.0) * float(self.constraint_config["circular_mask_radius_fraction"])
        r = torch.arange(h, device=ctx.device, dtype=ctx.dtype) - h / 2.0
        c = torch.arange(w, device=ctx.device, dtype=ctx.dtype) - w / 2.0
        rr, cc = torch.meshgrid(r, c, indexing="ij")
        circle_mask = torch.sqrt(rr * rr + cc * cc)
        dist_from_radius = torch.relu(torch.abs(circle_mask - radius))  # (H, W)
        circular_err = (dist_from_radius.unsqueeze(0) * template).mean(dim=(1, 2))
        circular_loss = circular_weight * circular_err

        return tv_loss + cutoff_loss + circular_loss

    def get_optimization_parameters(self) -> dict[str, list[torch.nn.Parameter]]:
        params = []
        for name, param in self.named_parameters(recurse=True):
            if not name.startswith('origin.') and param.requires_grad:
                params.append(param)
        if not params:
            return {}
        return {'default': params}

        


class SyntheticDiskLattice(RenderComponent):
    DEFAULT_HARD_CONSTRAINTS: dict[str, bool] = {
        "force_positive_intensity": True,
    }

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
        """
        Build a synthetic disk lattice renderer.

        Parameters
        ----------
        u_row, u_col, v_row, v_col : float | Sequence[float | int | None]
            Lattice basis parameters. Accept ``x``, ``(x0, delta)``, or
            ``(x0, lo, hi)``.
        intensity_0 : float | Sequence[float], optional
            Baseline intensity for included lattice disks. Accepts ``x``,
            ``(x0, delta)``, or ``(x0, lo, hi)``.
        intensity_row, intensity_col, intensity_row_row, intensity_col_col, intensity_row_col :
            float | Sequence[float | int | None]
            Intensity polynomial parameters. Accept ``x``, ``(x0, delta)``, or
            ``(x0, lo, hi)``.
        center_intensity_0 : float | Sequence[float] | None, optional
            Optional center-disk baseline. Accepts ``x``, ``(x0, delta)``, or
            ``(x0, lo, hi)`` and routes by center ownership rules.
        exclude_indices : Iterable[tuple[int, int]] | None, optional
            Lattice indices excluded from rendering. By default, ``(0, 0)`` is
            excluded. To include center explicitly, pass ``exclude_indices`` that
            does not contain ``(0, 0)``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``center_intensity_0`` is provided with center included while
            ``per_disk_intensity=False``.

        Notes
        -----
        Center-intensity routing is explicit:
        - Center excluded (default): ``center_intensity_0`` maps to
          ``disk.intensity_raw``.
        - Center included with ``per_disk_intensity=True``:
          ``center_intensity_0`` maps to lattice center ``i0_raw`` entry.
          In this case ``disk.intensity_raw`` is set to ``0`` to avoid
          duplicate center ownership when disk is rendered standalone.
        """
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

        u_row_init, u_row_lo, u_row_hi = self.parse_bounded_init(u_row, name="u_row")
        u_col_init, u_col_lo, u_col_hi = self.parse_bounded_init(u_col, name="u_col")
        v_row_init, v_row_lo, v_row_hi = self.parse_bounded_init(v_row, name="v_row")
        v_col_init, v_col_lo, v_col_hi = self.parse_bounded_init(v_col, name="v_col")
        self.u_row = nn.Parameter(torch.tensor(u_row_init, dtype=torch.float32))
        if u_row_lo is not None or u_row_hi is not None:
            self.register_parameter_bounds("u_row", u_row_lo, u_row_hi)
        self.u_col = nn.Parameter(torch.tensor(u_col_init, dtype=torch.float32))
        if u_col_lo is not None or u_col_hi is not None:
            self.register_parameter_bounds("u_col", u_col_lo, u_col_hi)
        self.v_row = nn.Parameter(torch.tensor(v_row_init, dtype=torch.float32))
        if v_row_lo is not None or v_row_hi is not None:
            self.register_parameter_bounds("v_row", v_row_lo, v_row_hi)
        self.v_col = nn.Parameter(torch.tensor(v_col_init, dtype=torch.float32))
        if v_col_lo is not None or v_col_hi is not None:
            self.register_parameter_bounds("v_col", v_col_lo, v_col_hi)

        exclude = {(0, 0)} if exclude_indices is None else set(exclude_indices)
        center_included = (0, 0) not in exclude
        if center_intensity_0 is not None and center_included and not self.per_disk_intensity:
            raise ValueError(
                "center_intensity_0 with center included requires per_disk_intensity=True, "
                "or exclude (0,0) and use DiskTemplate intensity ownership."
            )
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
        i0_init, i0_lo, i0_hi = self.parse_bounded_init(intensity_0, name="intensity_0")
        if center_intensity_0 is None:
            i0_center, i0_center_lo, i0_center_hi = i0_init, None, None
        else:
            i0_center, i0_center_lo, i0_center_hi = self.parse_bounded_init(
                center_intensity_0, name="center_intensity_0"
            )
        if center_intensity_0 is not None and not center_included:
            self.disk.set_intensity(float(i0_center))
            if i0_center_lo is not None or i0_center_hi is not None:
                self.disk.register_parameter_bounds("intensity_raw", i0_center_lo, i0_center_hi)
        ir_init, ir_lo, ir_hi = self.parse_bounded_init(intensity_row, name="intensity_row")
        ic_init, ic_lo, ic_hi = self.parse_bounded_init(intensity_col, name="intensity_col")
        irr_init, irr_lo, irr_hi = self.parse_bounded_init(
            intensity_row_row, name="intensity_row_row"
        )
        icc_init, icc_lo, icc_hi = self.parse_bounded_init(
            intensity_col_col, name="intensity_col_col"
        )
        irc_init, irc_lo, irc_hi = self.parse_bounded_init(
            intensity_row_col, name="intensity_row_col"
        )
        self._center_i0_bounds: tuple[float | None, float | None] | None = None
        self._center_i0_index: int | None = None

        if self.per_disk_intensity:
            i0_values = torch.full((n_uv,), float(i0_init), dtype=torch.float32)
            if n_uv > 0:
                center_mask = (uv_t[:, 0] == 0) & (uv_t[:, 1] == 0)
                i0_values[center_mask] = float(i0_center)
                if center_intensity_0 is not None and bool(torch.any(center_mask)):
                    self._center_i0_index = int(torch.nonzero(center_mask, as_tuple=False)[0, 0])
                    self._center_i0_bounds = (i0_center_lo, i0_center_hi)
            self.i0_raw = nn.Parameter(i0_values)
            if i0_lo is not None or i0_hi is not None:
                self.register_parameter_bounds("i0_raw", i0_lo, i0_hi)
            if center_intensity_0 is not None and center_included:
                self.disk.set_intensity(0.0)
            if self.max_intensity_order >= 1:
                self.ir = nn.Parameter(torch.full((n_uv,), float(ir_init), dtype=torch.float32))
                self.ic = nn.Parameter(torch.full((n_uv,), float(ic_init), dtype=torch.float32))
                if ir_lo is not None or ir_hi is not None:
                    self.register_parameter_bounds("ir", ir_lo, ir_hi)
                if ic_lo is not None or ic_hi is not None:
                    self.register_parameter_bounds("ic", ic_lo, ic_hi)
            else:
                self.ir = None
                self.ic = None
            if self.max_intensity_order >= 2:
                self.irr = nn.Parameter(torch.full((n_uv,), float(irr_init), dtype=torch.float32))
                self.icc = nn.Parameter(torch.full((n_uv,), float(icc_init), dtype=torch.float32))
                self.irc = nn.Parameter(torch.full((n_uv,), float(irc_init), dtype=torch.float32))
                if irr_lo is not None or irr_hi is not None:
                    self.register_parameter_bounds("irr", irr_lo, irr_hi)
                if icc_lo is not None or icc_hi is not None:
                    self.register_parameter_bounds("icc", icc_lo, icc_hi)
                if irc_lo is not None or irc_hi is not None:
                    self.register_parameter_bounds("irc", irc_lo, irc_hi)
            else:
                self.irr = None
                self.icc = None
                self.irc = None
        else:
            self.i0_raw = nn.Parameter(torch.tensor(i0_init, dtype=torch.float32))
            if i0_lo is not None or i0_hi is not None:
                self.register_parameter_bounds("i0_raw", i0_lo, i0_hi)
            self.ir = nn.Parameter(torch.tensor(ir_init, dtype=torch.float32))
            if ir_lo is not None or ir_hi is not None:
                self.register_parameter_bounds("ir", ir_lo, ir_hi)
            self.ic = nn.Parameter(torch.tensor(ic_init, dtype=torch.float32))
            if ic_lo is not None or ic_hi is not None:
                self.register_parameter_bounds("ic", ic_lo, ic_hi)
            self.irr = nn.Parameter(torch.tensor(irr_init, dtype=torch.float32))
            if irr_lo is not None or irr_hi is not None:
                self.register_parameter_bounds("irr", irr_lo, irr_hi)
            self.icc = nn.Parameter(torch.tensor(icc_init, dtype=torch.float32))
            if icc_lo is not None or icc_hi is not None:
                self.register_parameter_bounds("icc", icc_lo, icc_hi)
            self.irc = nn.Parameter(torch.tensor(irc_init, dtype=torch.float32))
            if irc_lo is not None or irc_hi is not None:
                self.register_parameter_bounds("irc", irc_lo, irc_hi)
        if constraint_params is not None:
            self.apply_constraint_params(constraint_params, strict=True)
        if bool(self.hard_constraints.get("force_positive_intensity", False)):
            self._enforce_positive_intensity_params()

    def set_origin(self, origin: OriginND) -> None:
        self.origin = origin

    def _enforce_positive_intensity_params(self) -> None:
        """
        Project base intensity parameter(s) to nonnegative values.

        Notes
        -----
        Positivity is enforced as a hard projection after optimizer steps.
        The forward path intentionally avoids clamp-based dead gradients.
        Only ``i0_raw`` is projected; slope terms remain unconstrained.
        """
        with torch.no_grad():
            self.i0_raw.clamp_(min=0.0)

    def enforce_hard_constraints(self, ctx: RenderContext) -> None:
        if bool(self.hard_constraints.get("force_positive_intensity", False)):
            self._enforce_positive_intensity_params()
        if self._center_i0_bounds is not None and self._center_i0_index is not None:
            with torch.no_grad():
                lo, hi = self._center_i0_bounds
                idx = self._center_i0_index
                if lo is not None:
                    self.i0_raw[idx].clamp_(min=float(lo))
                if hi is not None:
                    self.i0_raw[idx].clamp_(max=float(hi))
        super().enforce_hard_constraints(ctx)


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
        if not torch.any(keep):
            return out
        
        centers_r = centers_r[keep]
        centers_c = centers_c[keep]
        keep_idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
        num_disks = centers_r.shape[0]

        active_order = int(
            ctx.fields.get(
                "lattice_intensity_order_override", self.default_pattern_intensity_order
            )
        )
        active_order = max(0, min(active_order, self.max_intensity_order))

        dr = cast(torch.Tensor, self.disk.dr).to(device=ctx.device, dtype=ctx.dtype)
        dc = cast(torch.Tensor, self.disk.dc).to(device=ctx.device, dtype=ctx.dtype)
        patch_vals = self.disk.patch_values().to(device=ctx.device, dtype=ctx.dtype)
        num_pixels = patch_vals.shape[0]

        if self.per_disk_intensity:
            i0 = self.i0_raw[keep_idx][:, None]
            inten = i0.expand(-1, num_pixels)
            
            if active_order >= 1 and self.ir is not None:
                ir = self.ir[keep_idx][:, None]
                ic = self.ic[keep_idx][:, None]
                inten = inten + ir * dr[None, :] + ic * dc[None, :]
            
            if active_order >= 2 and self.irr is not None:
                irr = self.irr[keep_idx][:, None]
                icc = self.icc[keep_idx][:, None]
                irc = self.irc[keep_idx][:, None]
                inten = inten + irr * (dr*dr)[None, :] + icc * (dc*dc)[None, :] + irc * (dr*dc)[None, :]
        else:
            inten = self.i0_raw if isinstance(self.i0_raw, torch.Tensor) else self.i0_raw
            if active_order >= 1:
                inten = inten + self.ir * centers_r + self.ic * centers_c
            if active_order >= 2:
                inten = inten + self.irr * centers_r**2 + self.icc * centers_c**2 + self.irc * centers_r * centers_c
        
        inten = inten[:, None].expand(-1, num_pixels) if inten.ndim == 1 else inten.expand(num_disks, num_pixels)
        total_pixels = num_disks * num_pixels
        r0_all = centers_r[:, None].expand(-1, num_pixels).reshape(total_pixels)
        c0_all = centers_c[:, None].expand(-1, num_pixels).reshape(total_pixels)
        dr_all = dr[None, :].expand(num_disks, -1).reshape(total_pixels)
        dc_all = dc[None, :].expand(num_disks, -1).reshape(total_pixels)
        vals_all = (patch_vals[None, :] * inten).reshape(total_pixels)
        _splat_patch(
            out,
            r0=r0_all,
            c0=c0_all,
            patch_vals=vals_all,
            dr=dr_all,
            dc=dc_all,
            scale=torch.ones_like(vals_all)
        )
        return out

    def forward_batched(
        self,
        ctx: RenderContext,
        *,
        u_row_b: torch.Tensor,
        u_col_b: torch.Tensor,
        v_row_b: torch.Tensor,
        v_col_b: torch.Tensor,
        i0_raw_b: torch.Tensor,
        ir_b: torch.Tensor | None,
        ic_b: torch.Tensor | None,
        irr_b: torch.Tensor | None,
        icc_b: torch.Tensor | None,
        irc_b: torch.Tensor | None,
        template_raw_b: torch.Tensor,
        origin_coords_b: torch.Tensor,
    ) -> torch.Tensor:
        if self.origin is None:
            raise RuntimeError("SyntheticDiskLattice requires an OriginND instance.")

        B = u_row_b.shape[0]
        uv = cast(torch.Tensor, self.uv_indices).to(device=ctx.device)
        K = int(uv.shape[0])
        if K == 0:
            return torch.zeros(B, ctx.shape[0], ctx.shape[1], device=ctx.device, dtype=ctx.dtype)

        u = uv[:, 0].to(dtype=ctx.dtype)
        v = uv[:, 1].to(dtype=ctx.dtype)

        r0_kb = (
            origin_coords_b[:, 0:1]
            + u.unsqueeze(0) * u_row_b.unsqueeze(1)
            + v.unsqueeze(0) * v_row_b.unsqueeze(1)
        )
        c0_kb = (
            origin_coords_b[:, 1:2]
            + u.unsqueeze(0) * u_col_b.unsqueeze(1)
            + v.unsqueeze(0) * v_col_b.unsqueeze(1)
        )

        bb = torch.as_tensor(self.boundary_px, device=ctx.device, dtype=ctx.dtype)
        keep = (r0_kb >= bb) & (r0_kb <= (ctx.shape[0] - 1) - bb)
        keep = keep & (c0_kb >= bb) & (c0_kb <= (ctx.shape[1] - 1) - bb)
        keep_f = keep.to(dtype=ctx.dtype)

        active_order = int(
            ctx.fields.get(
                "lattice_intensity_order_override", self.default_pattern_intensity_order
            )
        )
        active_order = max(0, min(active_order, self.max_intensity_order))

        dr = cast(torch.Tensor, self.disk.dr).to(device=ctx.device, dtype=ctx.dtype)
        dc = cast(torch.Tensor, self.disk.dc).to(device=ctx.device, dtype=ctx.dtype)
        N_pix = int(dr.numel())
        patch_vals = template_raw_b.reshape(B, N_pix)

        if self.per_disk_intensity:
            inten = i0_raw_b.unsqueeze(2).expand(B, K, N_pix)
            if active_order >= 1 and ir_b is not None and ic_b is not None:
                inten = inten + ir_b.unsqueeze(2) * dr.view(1, 1, N_pix) + ic_b.unsqueeze(2) * dc.view(1, 1, N_pix)
            if active_order >= 2 and irr_b is not None and icc_b is not None and irc_b is not None:
                inten = (
                    inten
                    + irr_b.unsqueeze(2) * (dr * dr).view(1, 1, N_pix)
                    + icc_b.unsqueeze(2) * (dc * dc).view(1, 1, N_pix)
                    + irc_b.unsqueeze(2) * (dr * dc).view(1, 1, N_pix)
                )
        else:
            inten = i0_raw_b.view(B, 1, 1).expand(B, K, N_pix).clone()
            if active_order >= 1 and ir_b is not None and ic_b is not None:
                inten = inten + ir_b.view(B, 1, 1) * r0_kb.unsqueeze(2) + ic_b.view(B, 1, 1) * c0_kb.unsqueeze(2)
            if active_order >= 2 and irr_b is not None and icc_b is not None and irc_b is not None:
                inten = (
                    inten
                    + irr_b.view(B, 1, 1) * (r0_kb * r0_kb).unsqueeze(2)
                    + icc_b.view(B, 1, 1) * (c0_kb * c0_kb).unsqueeze(2)
                    + irc_b.view(B, 1, 1) * (r0_kb * c0_kb).unsqueeze(2)
                )

        inten = inten * keep_f.unsqueeze(2)
        vals = patch_vals.unsqueeze(1) * inten

        r0_full = (r0_kb.unsqueeze(2) + dr.view(1, 1, N_pix)).reshape(B, K * N_pix)
        c0_full = (c0_kb.unsqueeze(2) + dc.view(1, 1, N_pix)).reshape(B, K * N_pix)
        vals_full = vals.reshape(B, K * N_pix)

        return _splat_patch_batched(
            ctx.shape, r0=r0_full, c0=c0_full, vals=vals_full,
            device=ctx.device, dtype=ctx.dtype,
        )

    def get_optimization_parameters(self) -> dict[str, list[torch.nn.Parameter]]:
        params = []
        for name, param in self.named_parameters(recurse=True):
            if not name.startswith('disk.') and param.requires_grad:
                params.append(param)
        if not params:
            return {}
        return {'default': params}
