"""Karen Ehrhardt-derived polar transforms and angular-uniformity origin finding."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from tqdm import tqdm


OriginMethod = Literal["descent", "grid"]

__all__ = [
    "OriginMethod",
    "find_origin",
    "find_origin_angular_descent",
    "find_origin_angular_grid",
    "polar_transform",
    "polar_transform_peaks",
]


def find_origin(
    data,
    *,
    method: OriginMethod = "descent",
    ellipse_params: tuple[float, float, float] | None = None,
    radial_min: float = 4.0,
    radial_max: float | None = None,
    radial_step: float = 1.0,
    num_annular_bins: int = 180,
    n_phi: int = 120,
    two_fold_rotation_symmetry: bool = False,
    kpow: float = 0.0,
    device: str = "cpu",
    batch_size: int = 16,
    local_margin: int = 40,
) -> NDArray:
    """Estimate diffraction-pattern origins as ``(scan_y, scan_x, 2)`` row/col pixels."""
    if method == "descent":
        return find_origin_angular_descent(
            data,
            ellipse_params=ellipse_params,
            radial_min=radial_min,
            radial_max=radial_max,
            n_phi=n_phi,
            radial_step=radial_step,
            kpow=kpow,
            device=device,
        )
    if method == "grid":
        return find_origin_angular_grid(
            data,
            ellipse_params=ellipse_params,
            num_annular_bins=num_annular_bins,
            radial_min=radial_min,
            radial_max=radial_max,
            radial_step=radial_step,
            two_fold_rotation_symmetry=two_fold_rotation_symmetry,
            device=device,
            batch_size=batch_size,
            local_margin=local_margin,
        )
    raise ValueError(f"method must be 'descent' or 'grid', got {method!r}.")


def polar_transform(
    data,
    origin_array: NDArray | torch.Tensor | None = None,
    ellipse_params: tuple[float, float, float] | None = None,
    num_annular_bins: int = 180,
    radial_min: float = 0.0,
    radial_max: float | None = None,
    radial_step: float = 1.0,
    two_fold_rotation_symmetry: bool = False,
    name: str | None = None,
    signal_units: str | None = None,
    scan_pos: tuple[int, int] | None = None,
    device: str = "cpu",
    batch_size: int = 128,
    show_progress: bool = True,
):
    """Torch-native polar transform ported from Karen Ehrhardt's PDF workflow.

    The returned :class:`Polar4dstem` stores data as ``(scan_y, scan_x, phi, r)``.
    ``two_fold_rotation_symmetry=True`` follows Karen's native behavior: sample
    directly over ``0..pi``. Callers that need summed Friedel partners should
    sample the full plane and fold explicitly.
    """
    from quantem.core.datastructures.polar4dstem import Polar4dstem

    array, scan_y, scan_x, n_row, n_col = _as_4d_array(data)

    if isinstance(origin_array, torch.Tensor):
        origin_array = origin_array.detach().cpu().numpy()
    origin_array = np.asarray(origin_array, dtype=float) if origin_array is not None else None
    if origin_array is None:
        center = np.array([(n_row - 1) / 2.0, (n_col - 1) / 2.0], dtype=float)
        origins = np.broadcast_to(center, (scan_y, scan_x, 2)).copy()
    elif origin_array.shape == (2,):
        origins = np.empty((scan_y, scan_x, 2), dtype=float)
        origins[...] = origin_array
    elif origin_array.shape == (scan_y, scan_x, 2):
        origins = origin_array
    else:
        raise ValueError(
            f"origin_array must have shape None, (2,), or {(scan_y, scan_x, 2)}, "
            f"got {origin_array.shape}."
        )

    if scan_pos is not None:
        iy, ix = scan_pos
        dp = torch.as_tensor(array[iy, ix], dtype=torch.float32, device=device)
        r0 = float(origins[iy, ix, 0])
        c0 = float(origins[iy, ix, 1])
        radial_max_eff = _resolve_radial_max(
            n_row, n_col, origins[iy : iy + 1, ix : ix + 1], radial_min, radial_max, radial_step
        )
        offset_row, offset_col, _, _ = _build_polar_sampling_offsets(
            ellipse_params,
            num_annular_bins,
            radial_min,
            radial_max_eff,
            radial_step,
            two_fold_rotation_symmetry,
            device,
        )
        col_norm = 2.0 * (offset_col + c0) / (n_col - 1) - 1.0
        row_norm = 2.0 * (offset_row + r0) / (n_row - 1) - 1.0
        grid = torch.stack([col_norm, row_norm], dim=-1).unsqueeze(0)
        polar2d = F.grid_sample(
            dp[None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return polar2d.squeeze(0).squeeze(0).cpu().numpy()

    radial_max_eff = _resolve_radial_max(
        n_row, n_col, origins, radial_min, radial_max, radial_step
    )
    offset_row, offset_col, phi_bins, radial_bins = _build_polar_sampling_offsets(
        ellipse_params,
        num_annular_bins,
        radial_min,
        radial_max_eff,
        radial_step,
        two_fold_rotation_symmetry,
        device,
    )
    n_phi = phi_bins.numel()
    n_r = radial_bins.numel()

    col_norm_scale = 2.0 / (n_col - 1)
    row_norm_scale = 2.0 / (n_row - 1)
    base_col_norm = offset_col * col_norm_scale
    base_row_norm = offset_row * row_norm_scale

    n_pos = scan_y * scan_x
    dp_view = torch.as_tensor(array.reshape(n_pos, n_row, n_col), dtype=torch.float32)
    origins_t = torch.as_tensor(origins.reshape(n_pos, 2), dtype=torch.float32, device=device)
    out = torch.empty((n_pos, n_phi, n_r), dtype=torch.float32, device=device)

    for start in tqdm(
        range(0, n_pos, batch_size),
        desc="Polar transform",
        disable=(not show_progress) or n_pos < 8,
    ):
        end = min(start + batch_size, n_pos)
        row_origins = origins_t[start:end, 0]
        col_origins = origins_t[start:end, 1]
        grid_col = base_col_norm.unsqueeze(0) + (col_origins * col_norm_scale - 1.0)[:, None, None]
        grid_row = base_row_norm.unsqueeze(0) + (row_origins * row_norm_scale - 1.0)[:, None, None]
        grids = torch.stack([grid_col, grid_row], dim=-1)
        dp_batch = dp_view[start:end].to(device=device, dtype=torch.float32)
        polars = F.grid_sample(
            dp_batch.unsqueeze(1),
            grids,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        out[start:end] = polars.squeeze(1)

    out_np = out.reshape(scan_y, scan_x, n_phi, n_r).cpu().numpy()
    phi_range = np.pi if two_fold_rotation_symmetry else 2.0 * np.pi
    phi_step_deg = (phi_range / float(n_phi)) * (180.0 / np.pi)

    sampling = np.zeros(4, dtype=float)
    origin = np.zeros(4, dtype=float)
    sampling[0:2] = np.asarray(getattr(data, "sampling", np.ones(4)))[0:2]
    sampling[2] = phi_step_deg
    sampling[3] = float(np.asarray(getattr(data, "sampling", np.ones(4)))[-1]) * radial_step
    origin[0:2] = np.asarray(getattr(data, "origin", np.zeros(4)))[0:2]
    origin[2] = 0.0
    origin[3] = radial_min * float(np.asarray(getattr(data, "sampling", np.ones(4)))[-1])
    units_in = list(getattr(data, "units", ["pixels", "pixels", "pixels", "pixels"]))
    metadata = dict(getattr(data, "metadata", {}))
    metadata.update(
        {
            "polar_radial_min": float(radial_min),
            "polar_radial_max": float(radial_max_eff),
            "polar_radial_step": float(radial_step),
            "polar_num_annular_bins": int(n_phi),
            "polar_two_fold_rotation_symmetry": bool(two_fold_rotation_symmetry),
            "polar_origin_row": float(origins[0, 0, 0]),
            "polar_origin_col": float(origins[0, 0, 1]),
            "polar_ellipse_params": tuple(ellipse_params) if ellipse_params is not None else None,
        }
    )
    return Polar4dstem(
        array=out_np,
        name=name if name is not None else f"{getattr(data, 'name', 'dataset')}_polar",
        origin=origin,
        sampling=sampling,
        units=[units_in[0], units_in[1], "deg", units_in[-1]],
        signal_units=signal_units if signal_units is not None else getattr(data, "signal_units", "arb. units"),
        metadata=metadata,
        _token=Polar4dstem._token,
    )


def polar_transform_peaks(
    cartesian_vector,
    centers: NDArray,
    *,
    scan_mask: NDArray | None = None,
    x_field: str | list[str] = ["x_pixels", "x"],
    y_field: str | list[str] = ["y_pixels", "y"],
    sampling_conversion_factor: float | None = None,
    two_fold_rotation_symmetry: bool = True,
    ellipse_params: tuple[float, float, float] | None = None,
    r_unit: str = "pixels",
    theta_unit: str = "radians",
    name_suffix: str = "_polar",
    use_tqdm: bool = True,
):
    """Transform Cartesian peak coordinates with Karen's polar convention.

    Peaks remain one-to-one with the input rows. Under two-fold symmetry, partner
    peaks are folded to the same theta coordinate with ``theta % pi`` but are not
    aggregated.
    """
    from quantem.core.datastructures import Vector

    if isinstance(cartesian_vector, np.ndarray) and cartesian_vector.dtype == object:
        cartesian_vector = cartesian_vector.item()
    if not isinstance(cartesian_vector, Vector):
        raise TypeError(f"Expected Vector, got {type(cartesian_vector)}")

    def find_field(field_options, available_fields):
        fields = [field_options] if isinstance(field_options, str) else field_options
        return next((f for f in fields if f in available_fields), None)

    x_field_found = find_field(x_field, cartesian_vector.fields)
    y_field_found = find_field(y_field, cartesian_vector.fields)
    if x_field_found is None or y_field_found is None:
        raise ValueError(
            "Could not find x/y coordinate fields in Vector. "
            f"Available fields: {cartesian_vector.fields}"
        )

    n_scan_y, n_scan_x = cartesian_vector.shape
    centers = _standardize_centers(centers, n_scan_y, n_scan_x)
    if scan_mask is None:
        scan_mask = np.ones((n_scan_y, n_scan_x), dtype=bool)
    else:
        scan_mask = np.asarray(scan_mask, dtype=bool)
        if scan_mask.shape != (n_scan_y, n_scan_x):
            raise ValueError(f"scan_mask shape {scan_mask.shape} must match {(n_scan_y, n_scan_x)}")
    if sampling_conversion_factor is None:
        sampling_conversion_factor = 1.0

    x_idx = cartesian_vector.fields.index(x_field_found)
    y_idx = cartesian_vector.fields.index(y_field_found)
    extra_indices = [
        idx for idx in range(len(cartesian_vector.fields))
        if idx not in (x_idx, y_idx)
    ]
    output_fields = ["r_pixels", "theta", "r_invA"] + [
        cartesian_vector.fields[idx] for idx in extra_indices
    ]
    output_units = [r_unit, theta_unit, "1/Å"] + [
        cartesian_vector.units[idx] for idx in extra_indices
    ]
    polar_vector = Vector.from_shape(
        shape=(n_scan_y, n_scan_x),
        fields=output_fields,
        units=output_units,
        name=cartesian_vector.name + name_suffix,
    )

    theta_period = np.pi if two_fold_rotation_symmetry else 2.0 * np.pi
    iterator = tqdm(range(n_scan_y), disable=not use_tqdm, desc="Polar transform peaks")
    for i in iterator:
        for j in range(n_scan_x):
            if not scan_mask[i, j]:
                polar_vector[i, j] = np.zeros((0, len(output_fields)))
                continue

            cartesian_data = cartesian_vector[i, j].array
            if len(cartesian_data) == 0:
                polar_vector[i, j] = np.zeros((0, len(output_fields)))
                continue

            center_y, center_x = centers[i, j]
            dx = cartesian_data[:, x_idx] - center_x
            dy = cartesian_data[:, y_idx] - center_y
            r_pixels, theta = _cartesian_offsets_to_polar(dx, dy, ellipse_params)
            theta = np.mod(theta, theta_period)
            r_invA = r_pixels * sampling_conversion_factor

            polar_data = np.column_stack([r_pixels, theta, r_invA])
            if extra_indices:
                polar_data = np.column_stack([polar_data, cartesian_data[:, extra_indices]])
            polar_vector[i, j] = polar_data

    return polar_vector


def find_origin_angular_grid(
    data,
    *,
    ellipse_params: tuple[float, float, float] | None = None,
    num_annular_bins: int = 180,
    radial_min: float = 4.0,
    radial_max: float | None = None,
    radial_step: float = 2.0,
    two_fold_rotation_symmetry: bool = False,
    device: str = "cpu",
    batch_size: int = 16,
    local_margin: int = 40,
) -> NDArray:
    """Coarse-to-fine angular-variance origin finder.

    This is a surgical port of Karen Ehrhardt's PDF center finder. It first finds
    a global center on the mean diffraction pattern, then refines each scan
    position by minimizing angular intensity variation in a polar annulus.
    """
    array, scan_y, scan_x, n_row, n_col = _as_4d_array(data)
    array_t = torch.as_tensor(array, dtype=torch.float32, device=device)

    mean_dp_t = array_t.mean(dim=(0, 1))
    total_intensity = mean_dp_t.clamp(min=0).sum() + 1e-9
    row_grid_t = torch.arange(n_row, dtype=torch.float32, device=device)[:, None]
    col_grid_t = torch.arange(n_col, dtype=torch.float32, device=device)[None, :]
    com_row = int(round(float(((row_grid_t * mean_dp_t.clamp(min=0)).sum() / total_intensity).item())))
    com_col = int(round(float(((col_grid_t * mean_dp_t.clamp(min=0)).sum() / total_intensity).item())))

    com_edge_budget = min(com_row, com_col, (n_row - 1) - com_row, (n_col - 1) - com_col)
    global_margin = int(min(40, max(2, com_edge_budget // 2)))
    safe_radial_max = float(
        min(
            com_row - global_margin,
            (n_row - 1) - (com_row + global_margin),
            com_col - global_margin,
            (n_col - 1) - (com_col + global_margin),
        )
    )
    if radial_max is not None:
        safe_radial_max = min(safe_radial_max, float(radial_max))
    if safe_radial_max <= radial_min:
        safe_radial_max = radial_min + radial_step

    safe_low = int(np.ceil(safe_radial_max))
    safe_high_row = n_row - 1 - safe_low
    safe_high_col = n_col - 1 - safe_low
    search_n_phi = max(18, min(int(num_annular_bins), 60))
    local_coarse_step = 5

    offset_row, offset_col, _, radial_bins = _build_polar_sampling_offsets(
        ellipse_params,
        search_n_phi,
        radial_min,
        safe_radial_max,
        radial_step,
        two_fold_rotation_symmetry,
        device,
    )
    n_r = radial_bins.numel()
    min_r_idx = 0
    max_r_idx = max(1, int(np.ceil(0.9 * n_r)))
    col_norm_scale = 2.0 / (n_col - 1)
    row_norm_scale = 2.0 / (n_row - 1)
    base_col_norm = offset_col * col_norm_scale
    base_row_norm = offset_row * row_norm_scale

    mean_dp_batch = mean_dp_t[None, None]
    rows, cols, grids = _build_candidate_grids(
        base_col_norm,
        base_row_norm,
        com_row,
        com_col,
        global_margin,
        n_row,
        n_col,
        col_norm_scale,
        row_norm_scale,
        device,
        step=2,
    )
    scores = _angular_std_scores(mean_dp_batch, grids, min_r_idx, max_r_idx)
    valid = (
        (rows >= safe_low) & (rows <= safe_high_row) & (cols >= safe_low) & (cols <= safe_high_col)
    )
    best = int(scores.masked_fill(~valid, float("inf")).argmin().item())
    coarse_row, coarse_col = int(rows[best].item()), int(cols[best].item())

    rows, cols, grids = _build_candidate_grids(
        base_col_norm,
        base_row_norm,
        coarse_row,
        coarse_col,
        10,
        n_row,
        n_col,
        col_norm_scale,
        row_norm_scale,
        device,
        step=1,
    )
    scores = _angular_std_scores(mean_dp_batch, grids, min_r_idx, max_r_idx)
    valid = (
        (rows >= safe_low) & (rows <= safe_high_row) & (cols >= safe_low) & (cols <= safe_high_col)
    )
    best = int(scores.masked_fill(~valid, float("inf")).argmin().item())
    global_row, global_col = int(rows[best].item()), int(cols[best].item())

    coarse_rows, coarse_cols, coarse_grids = _build_candidate_grids(
        base_col_norm,
        base_row_norm,
        global_row,
        global_col,
        int(local_margin),
        n_row,
        n_col,
        col_norm_scale,
        row_norm_scale,
        device,
        step=local_coarse_step,
    )
    coarse_valid = (
        (coarse_rows >= safe_low)
        & (coarse_rows <= safe_high_row)
        & (coarse_cols >= safe_low)
        & (coarse_cols <= safe_high_col)
    )
    n_coarse = coarse_grids.shape[0]
    med_search_range = torch.arange(
        -local_coarse_step, local_coarse_step + 1, 1, dtype=torch.long, device=device
    )
    med_drow, med_dcol = (
        m.reshape(-1) for m in torch.meshgrid(med_search_range, med_search_range, indexing="ij")
    )
    fine_search_range = torch.arange(-2, 3, dtype=torch.long, device=device)
    fine_drow, fine_dcol = (
        m.reshape(-1) for m in torch.meshgrid(fine_search_range, fine_search_range, indexing="ij")
    )
    flat_dps_t = array_t.reshape(-1, n_row, n_col)
    n_pos = flat_dps_t.shape[0]
    origin_flat_t = torch.zeros(n_pos, 2, dtype=torch.float32, device=device)

    def refine(dp_batch, current_row, current_col, drow, dcol):
        n_cands = drow.numel()
        cand_rows = (current_row[:, None] + drow[None, :]).clamp(0, n_row - 1)
        cand_cols = (current_col[:, None] + dcol[None, :]).clamp(0, n_col - 1)
        g_col = (
            base_col_norm + (cand_cols.reshape(-1).float() * col_norm_scale - 1.0)[:, None, None]
        )
        g_row = (
            base_row_norm + (cand_rows.reshape(-1).float() * row_norm_scale - 1.0)[:, None, None]
        )
        grids = torch.stack([g_col, g_row], dim=-1)
        dps = dp_batch.repeat_interleave(n_cands, dim=0)
        polars = F.grid_sample(
            dps, grids, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        region = polars.view(dp_batch.shape[0], n_cands, *base_col_norm.shape)[
            ..., min_r_idx:max_r_idx
        ]
        scores = region.std(dim=2).sum(dim=2) / (region.mean(dim=2).sum(dim=2).abs() + 1e-6)
        valid = (
            (cand_rows >= safe_low)
            & (cand_rows <= safe_high_row)
            & (cand_cols >= safe_low)
            & (cand_cols <= safe_high_col)
        )
        best = scores.masked_fill(~valid, float("inf")).argmin(dim=1)
        best_row = cand_rows.gather(1, best[:, None]).squeeze(1)
        best_col = cand_cols.gather(1, best[:, None]).squeeze(1)
        return best_row, best_col, scores, valid

    n_not_converged = 0
    pbar = tqdm(total=n_pos, desc="Finding origins", disable=n_pos < 8)
    for start in range(0, n_pos, batch_size):
        end = min(start + batch_size, n_pos)
        n_dp = end - start
        dp_b = flat_dps_t[start:end].unsqueeze(1)
        polars_coarse = F.grid_sample(
            dp_b.transpose(0, 1).expand(n_coarse, n_dp, n_row, n_col),
            coarse_grids,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        region_coarse = polars_coarse[:, :, :, min_r_idx:max_r_idx]
        scores_coarse = region_coarse.std(dim=2).sum(dim=2) / (
            region_coarse.mean(dim=2).sum(dim=2).abs() + 1e-6
        )
        scores_coarse = scores_coarse.masked_fill(~coarse_valid[:, None], float("inf"))
        best_coarse = scores_coarse.argmin(dim=0)
        current_row, current_col = coarse_rows[best_coarse], coarse_cols[best_coarse]
        current_row, current_col, _, _ = refine(
            dp_b, current_row, current_col, med_drow, med_dcol
        )
        best_row, best_col, fine_scores, fine_valid = refine(
            dp_b, current_row, current_col, fine_drow, fine_dcol
        )

        side = fine_search_range.numel()
        scores_grid = fine_scores.view(n_dp, side, side)
        valid_grid = fine_valid.view(n_dp, side, side)
        flat_best = scores_grid.masked_fill(~valid_grid, float("inf")).view(n_dp, -1).argmin(dim=1)
        i_star, j_star = flat_best // side, flat_best % side
        batch_idx = torch.arange(n_dp, device=device)
        ii = torch.stack([i_star - 1, i_star, i_star + 1], dim=1).clamp(0, side - 1)
        jj = torch.stack([j_star - 1, j_star, j_star + 1], dim=1).clamp(0, side - 1)
        patch = scores_grid[batch_idx[:, None, None], ii[:, :, None], jj[:, None, :]]
        on_border = (i_star < 1) | (i_star > side - 2) | (j_star < 1) | (j_star > side - 2)
        n_not_converged += int(on_border.sum())
        offset = _quadratic_subpixel_offset(patch).to(torch.float32)
        offset = torch.where(on_border[:, None], torch.zeros_like(offset), offset)
        origin_flat_t[start:end, 0] = best_row.to(torch.float32) + offset[:, 0]
        origin_flat_t[start:end, 1] = best_col.to(torch.float32) + offset[:, 1]
        pbar.update(n_dp)
    pbar.close()

    if n_not_converged:
        warnings.warn(
            f"find_origin_angular_grid: {n_not_converged} of {n_pos} scan positions did not "
            "bracket a sub-pixel minimum. Integer-pixel origins were used there.",
            stacklevel=2,
        )
    return origin_flat_t.cpu().numpy().reshape(scan_y, scan_x, 2)


def find_origin_angular_descent(
    data,
    *,
    ellipse_params: tuple[float, float, float] | None = None,
    radial_min: float = 4.0,
    radial_max: float | None = None,
    n_phi: int = 120,
    radial_step: float = 1.0,
    kpow: float = 0.0,
    device: str = "cpu",
) -> NDArray:
    """COM-anchored local descent origin finder.

    The score is the normalized angular standard deviation in a polar annulus.
    Lower scores indicate a more radially uniform transform and therefore a
    better center. This method is fast enough to use by default in notebooks.
    """
    array, scan_y, scan_x, n_row, n_col = _as_4d_array(data)
    if radial_max is None:
        radial_max = float(min(n_row, n_col) // 2 - 2)
    if radial_max <= radial_min:
        radial_max = float(radial_min + max(radial_step, 1.0))
    n_radial = max(4, int(round((radial_max - radial_min) / radial_step)) + 1)

    array_t = torch.as_tensor(array, dtype=torch.float32, device=device)
    patterns = array_t.reshape(-1, n_row, n_col)
    n_patterns = patterns.shape[0]
    image_center = torch.tensor(
        [(n_row - 1) / 2.0, (n_col - 1) / 2.0],
        dtype=torch.float32,
        device=device,
    )
    blank_patterns = patterns.clamp(min=0).sum(dim=(1, 2)) <= 0
    if bool(blank_patterns.all().item()):
        return (
            image_center[None]
            .expand(n_patterns, 2)
            .reshape(scan_y, scan_x, 2)
            .cpu()
            .numpy()
        )
    offset_row, offset_col, ring_weights = _local_sampling(
        radial_min, radial_max, n_phi, n_radial, kpow, ellipse_params, device
    )

    mean_pattern = array_t.mean(dim=(0, 1))
    global_origin = _descend_batched(
        mean_pattern[None],
        torch.round(_com_anchor(mean_pattern))[None],
        offset_row,
        offset_col,
        ring_weights,
        n_phi,
        device,
    )[0]
    start_centers = torch.round(global_origin)[None].expand(n_patterns, 2).clone()
    origins = _descend_batched(
        patterns,
        start_centers,
        offset_row,
        offset_col,
        ring_weights,
        n_phi,
        device,
    )
    origins = torch.where(blank_patterns[:, None], image_center[None], origins)
    return origins.reshape(scan_y, scan_x, 2).cpu().numpy()


def _as_4d_array(data) -> tuple[NDArray, int, int, int, int]:
    array = np.asarray(data.array if hasattr(data, "array") else data)
    if array.ndim == 2:
        n_row, n_col = array.shape
        array = array[None, None]
        return np.ascontiguousarray(array), 1, 1, n_row, n_col
    if array.ndim == 4:
        scan_y, scan_x, n_row, n_col = array.shape
        return np.ascontiguousarray(array), scan_y, scan_x, n_row, n_col
    raise ValueError(
        f"Expected a 2D diffraction pattern or 4D-STEM array, got shape {array.shape}."
    )


def _standardize_centers(centers, scan_y: int, scan_x: int) -> NDArray:
    centers = np.asarray(centers, dtype=float)
    if centers.shape == (2,):
        out = np.empty((scan_y, scan_x, 2), dtype=float)
        out[...] = centers
        return out
    if centers.shape == (scan_y, scan_x, 2):
        return centers
    if centers.shape == (2, scan_y, scan_x):
        return np.moveaxis(centers, 0, -1)
    raise ValueError(
        f"centers must have shape (2,), {(scan_y, scan_x, 2)}, "
        f"or {(2, scan_y, scan_x)}, got {centers.shape}."
    )


def _resolve_radial_max(
    n_row: int,
    n_col: int,
    origins: NDArray,
    radial_min: float,
    radial_max: float | None,
    radial_step: float,
) -> float:
    if radial_step <= 0:
        raise ValueError(f"radial_step must be > 0, got {radial_step}.")
    if radial_max is not None:
        radial_max_eff = float(radial_max)
    else:
        origin_rows = origins[..., 0]
        origin_cols = origins[..., 1]
        radial_limits = np.minimum.reduce(
            [
                origin_rows,
                (n_row - 1) - origin_rows,
                origin_cols,
                (n_col - 1) - origin_cols,
            ]
        )
        radial_max_eff = float(np.nanmin(radial_limits))
    if not np.isfinite(radial_max_eff) or radial_max_eff <= radial_min:
        radial_max_eff = float(radial_min + radial_step)
    return radial_max_eff


def _cartesian_offsets_to_polar(
    dx: NDArray,
    dy: NDArray,
    ellipse_params: tuple[float, float, float] | None,
) -> tuple[NDArray, NDArray]:
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    if ellipse_params is None:
        return np.hypot(dx, dy), np.arctan2(dy, dx)
    if len(ellipse_params) != 3:
        raise ValueError("ellipse_params must be (a, b, theta_deg).")

    a, b, theta_deg = ellipse_params
    theta = np.deg2rad(theta_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    u = dx * cos_t + dy * sin_t
    v_prime = -dx * sin_t + dy * cos_t
    scaled_u = (b / a) * u
    r_pixels = np.hypot(scaled_u, v_prime)
    phi = np.arctan2(v_prime, scaled_u) + theta
    return r_pixels, phi


def _polar_to_cartesian_offsets(
    phi: torch.Tensor,
    r_pix: torch.Tensor,
    ellipse_params: tuple[float, float, float] | None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    if ellipse_params is None:
        offset_col = r_pix * torch.cos(phi)
        offset_row = r_pix * torch.sin(phi)
    else:
        if len(ellipse_params) != 3:
            raise ValueError("ellipse_params must be (a, b, theta_deg).")
        a, b, theta_deg = ellipse_params
        theta = torch.deg2rad(torch.tensor(theta_deg, dtype=torch.float32, device=device))
        alpha = phi - theta
        u = (a / b) * r_pix * torch.cos(alpha)
        v_prime = r_pix * torch.sin(alpha)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        offset_col = u * cos_t - v_prime * sin_t
        offset_row = u * sin_t + v_prime * cos_t
    return offset_row, offset_col


def _build_polar_sampling_offsets(
    ellipse_params: tuple[float, float, float] | None,
    num_annular_bins: int,
    radial_min: float,
    radial_max_eff: float,
    radial_step: float,
    two_fold_rotation_symmetry: bool,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if radial_step <= 0:
        raise ValueError(f"radial_step must be > 0, got {radial_step}.")
    if num_annular_bins < 1:
        raise ValueError("num_annular_bins must be >= 1.")

    radial_bins = torch.arange(
        radial_min, radial_max_eff, radial_step, dtype=torch.float32, device=device
    )
    if radial_bins.numel() == 0:
        radial_bins = torch.tensor([radial_min], dtype=torch.float32, device=device)
    phi_range = torch.pi if two_fold_rotation_symmetry else 2.0 * torch.pi
    phi_bins = torch.linspace(
        0.0, phi_range, num_annular_bins + 1, dtype=torch.float32, device=device
    )[:-1]
    phi_grid, r_pix_grid = torch.meshgrid(phi_bins, radial_bins, indexing="ij")
    offset_row, offset_col = _polar_to_cartesian_offsets(
        phi_grid, r_pix_grid, ellipse_params, device
    )
    return offset_row, offset_col, phi_bins, radial_bins


def _build_candidate_grids(
    base_col_norm: torch.Tensor,
    base_row_norm: torch.Tensor,
    center_row: int,
    center_col: int,
    margin: int,
    n_row: int,
    n_col: int,
    col_norm_scale: float,
    row_norm_scale: float,
    device: str = "cpu",
    step: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.arange(
        max(0, center_row - margin),
        min(n_row, center_row + margin + 1),
        step,
        dtype=torch.long,
        device=device,
    )
    cols = torch.arange(
        max(0, center_col - margin),
        min(n_col, center_col + margin + 1),
        step,
        dtype=torch.long,
        device=device,
    )
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
    row_flat, col_flat = row_grid.reshape(-1), col_grid.reshape(-1)
    grid_col = (
        base_col_norm.unsqueeze(0) + (col_flat.float() * col_norm_scale - 1.0)[:, None, None]
    )
    grid_row = (
        base_row_norm.unsqueeze(0) + (row_flat.float() * row_norm_scale - 1.0)[:, None, None]
    )
    grids = torch.stack([grid_col, grid_row], dim=-1)
    return row_flat, col_flat, grids


def _angular_std_scores(
    dp_batch: torch.Tensor,
    grids: torch.Tensor,
    min_r_idx: int,
    max_r_idx: int,
) -> torch.Tensor:
    n = grids.shape[0]
    polars = F.grid_sample(
        dp_batch.expand(n, -1, -1, -1),
        grids,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    region = polars.squeeze(1)[:, :, min_r_idx:max_r_idx]
    return region.std(dim=1).sum(dim=1) / (region.mean(dim=1).sum(dim=1).abs() + 1e-6)


def _quadratic_subpixel_offset(patch: torch.Tensor) -> torch.Tensor:
    device = patch.device
    scores_flat = patch.reshape(patch.shape[0], 9).to(torch.float64)
    grid = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64, device=device)
    uu, vv = torch.meshgrid(grid, grid, indexing="ij")
    u, v = uu.reshape(9), vv.reshape(9)
    basis = torch.stack([torch.ones_like(u), u, v, u * u, v * v, u * v], dim=1)
    fit_matrix = torch.linalg.pinv(basis)
    _, b, c, d, e, f = (scores_flat @ fit_matrix.T).unbind(dim=1)
    det = 4.0 * d * e - f * f
    valid = (2.0 * d > 0) & (det > 1e-12) & torch.isfinite(scores_flat).all(dim=1)
    det_safe = torch.where(valid, det, torch.ones_like(det))
    drow = torch.where(valid, (f * c - 2.0 * e * b) / det_safe, torch.zeros_like(det))
    dcol = torch.where(valid, (f * b - 2.0 * d * c) / det_safe, torch.zeros_like(det))
    return torch.stack([drow.clamp(-1.0, 1.0), dcol.clamp(-1.0, 1.0)], dim=1)


_NEIGHBOR_STEPS_8 = [
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
    [1.0, 1.0],
    [1.0, -1.0],
    [-1.0, 1.0],
    [-1.0, -1.0],
]
_PATCH_OFFSETS_3X3 = [
    [-1.0, -1.0],
    [-1.0, 0.0],
    [-1.0, 1.0],
    [0.0, -1.0],
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, -1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]


def _com_anchor(pattern: torch.Tensor) -> torch.Tensor:
    n_row, n_col = pattern.shape
    clipped = pattern.clamp(min=0)
    total_raw = clipped.sum()
    if float(total_raw.item()) <= 0:
        return torch.tensor(
            [(n_row - 1) / 2.0, (n_col - 1) / 2.0],
            dtype=torch.float32,
            device=pattern.device,
        )
    total = total_raw + 1e-9
    rows = torch.arange(n_row, device=pattern.device, dtype=torch.float32)
    cols = torch.arange(n_col, device=pattern.device, dtype=torch.float32)
    center_row = (rows[:, None] * clipped).sum() / total
    center_col = (cols[None, :] * clipped).sum() / total
    return torch.stack([center_row, center_col])


def _local_sampling(radial_min, radial_max, n_phi, n_radial, kpow, ellipse_params, device):
    phi = torch.linspace(0, 2 * np.pi, n_phi + 1, device=device)[:-1]
    radii = torch.linspace(radial_min, radial_max, n_radial, device=device)
    phi_grid, radius_grid = torch.meshgrid(phi, radii, indexing="ij")
    offset_row, offset_col = _polar_to_cartesian_offsets(
        phi_grid, radius_grid, ellipse_params, device
    )
    ring_weights = radii**kpow
    return offset_row, offset_col, ring_weights


def _local_polar_score(polar_values, valid_mask, n_phi, ring_weights, min_valid_frac):
    n_valid = valid_mask.sum(dim=-2).clamp(min=1)
    ring_mean = (polar_values * valid_mask).sum(dim=-2) / n_valid
    ring_var = (((polar_values - ring_mean.unsqueeze(-2)) ** 2) * valid_mask).sum(dim=-2) / n_valid
    ring_std = ring_var.sqrt()
    ring_usable = valid_mask.sum(dim=-2) >= (min_valid_frac * n_phi)
    weights = ring_weights * ring_usable
    usable_weight = weights.sum(dim=-1)
    score = (weights * ring_std).sum(dim=-1) / ((weights * ring_mean.abs()).sum(dim=-1) + 1e-6)
    score = torch.where(usable_weight > 0, score, torch.full_like(score, float("inf")))
    return score


def _local_score_pairs(
    patterns,
    pattern_index,
    centers,
    offset_row,
    offset_col,
    ring_weights,
    n_phi,
    device,
    min_valid_frac=0.5,
    chunk=4096,
):
    _, n_row, n_col = patterns.shape
    ones_image = torch.ones(1, 1, n_row, n_col, device=device)
    scores = torch.empty(centers.shape[0], device=device)
    for start in range(0, centers.shape[0], chunk):
        index = pattern_index[start : start + chunk]
        n_chunk = index.shape[0]
        center_row = centers[start : start + chunk, 0][:, None, None]
        center_col = centers[start : start + chunk, 1][:, None, None]
        sample_grid = torch.stack(
            [
                2.0 * (center_col + offset_col[None]) / (n_col - 1) - 1.0,
                2.0 * (center_row + offset_row[None]) / (n_row - 1) - 1.0,
            ],
            dim=-1,
        )
        polar_values = F.grid_sample(
            patterns[index][:, None],
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0]
        valid_mask = F.grid_sample(
            ones_image.expand(n_chunk, 1, n_row, n_col),
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0] > 0.999
        scores[start : start + n_chunk] = _local_polar_score(
            polar_values, valid_mask, n_phi, ring_weights, min_valid_frac
        )
    return scores


def _descend_batched(
    patterns,
    anchors,
    offset_row,
    offset_col,
    ring_weights,
    n_phi,
    device,
    schedule=(4.0, 2.0, 1.0),
    sweeps=2,
):
    n_patterns = patterns.shape[0]
    pattern_ids = torch.arange(n_patterns, device=device)
    neighbor_steps = torch.tensor(_NEIGHBOR_STEPS_8, device=device)
    patch_offsets = torch.tensor(_PATCH_OFFSETS_3X3, dtype=torch.float32, device=device)
    pattern_ids_per_neighbor = pattern_ids.repeat_interleave(8)
    pattern_ids_per_patch = pattern_ids.repeat_interleave(9)
    center = anchors.clone()

    def score_at(pattern_index, centers):
        return _local_score_pairs(
            patterns,
            pattern_index,
            centers,
            offset_row,
            offset_col,
            ring_weights,
            n_phi,
            device,
        )

    best_score = score_at(pattern_ids, center)
    for step in schedule:
        for _ in range(sweeps):
            neighbors = (center[:, None, :] + step * neighbor_steps[None]).reshape(n_patterns * 8, 2)
            neighbor_scores = score_at(pattern_ids_per_neighbor, neighbors).reshape(n_patterns, 8)
            best_neighbor = neighbor_scores.argmin(dim=1)
            best_neighbor_score = neighbor_scores.gather(1, best_neighbor[:, None]).squeeze(1)
            improved = best_neighbor_score < best_score - 1e-12
            best_neighbor_center = neighbors.reshape(n_patterns, 8, 2)[pattern_ids, best_neighbor]
            center = torch.where(improved[:, None], best_neighbor_center, center)
            best_score = torch.where(improved, best_neighbor_score, best_score)

    patch_centers = (center[:, None, :] + patch_offsets[None]).reshape(n_patterns * 9, 2)
    patch_scores = score_at(pattern_ids_per_patch, patch_centers).reshape(n_patterns, 9)
    center = patch_centers.reshape(n_patterns, 9, 2)[pattern_ids, patch_scores.argmin(dim=1)]
    patch_centers = (center[:, None, :] + patch_offsets[None]).reshape(n_patterns * 9, 2)
    score_patch = score_at(pattern_ids_per_patch, patch_centers).reshape(n_patterns, 3, 3)
    subpixel_offset = _quadratic_subpixel_offset(score_patch).to(torch.float32)
    return center + subpixel_offset
