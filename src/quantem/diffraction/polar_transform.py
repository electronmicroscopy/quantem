import warnings

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from tqdm import tqdm

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem
from quantem.core.utils.utils import to_numpy

# Standard DPs use (row, col) convention. Polar coordinates use (phi, r_pix),
# grid_sample's grid tensor requires them to be ordered (col, row)
# but is noted where the call occures


def find_origin_angular_grid(
    data: Dataset4dstem,
    *,
    ellipse_params: tuple[float, float, float] | None = None,
    num_annular_bins: int = 180,
    radial_min: float = 0.0,
    radial_max: float | None = None,
    radial_step: float = 2.0,
    two_fold_rotation_symmetry: bool = False,
    device: str = "cpu",
    batch_size: int = 16,
    local_margin: int = 40,
) -> NDArray:
    """
    Automatic diffraction center finding by minimizing angular intensity
    variation in the polar transform. A correctly centered diffraction
    pattern has uniform intensity along each ring, so the center that
    minimizes the angular standard deviation is the true beam center.

    Uses a coarse-to-fine search on the mean diffraction pattern to find
    a global center, then refines per scan position to account for descan
    across the scan.

    Parameters
    ----------
    data : Dataset4dstem
        A 4D-STEM dataset (or 2D wrapped as 4D)
    ellipse_params : tuple or None
        Ellipse parameters (a, b, theta_deg) for distortion correction
    num_annular_bins : int
        Number of angular bins for the final polar transform
    radial_min : float
        Minimum radius in pixels
    radial_max : float or None
        Maximum radius in pixels
    radial_step : float
        Radial step size in pixels for the search polar grid
    two_fold_rotation_symmetry : bool
        If True, use only 0 to pi range for angles
    device : str
        Torch device for computation
    batch_size : int
        Number of scan positions evaluated per coarse-stage kernel call.
        Larger values reduce per-iteration overhead but use more memory.
    local_margin : int
        Half-width (in pixels) of the search window used to refine each
        scan position's origin. After the global center is found on the
        mean DP, each DP's origin is searched within a
        ``(2*local_margin+1)`` square window centered on the global
        origin. Set this large enough to cover the worst-case descan
        drift across the scan.

    Returns
    -------
    origin_array : np.ndarray
        Array of shape (scan_row, scan_col, 2) containing (row, col) origin
        estimates in pixels.
    """
    if data.ndim == 2:
        n_row, n_col = data.shape
        scan_row, scan_col = 1, 1
    elif data.ndim == 4:
        scan_row, scan_col, n_row, n_col = data.shape
    else:
        raise ValueError(
            f" Got array with shape {data.shape}."
            "To use find_origin_angular_grid, pass a 2D or 4DSTEM dataset."
        )

    # Move the full dataset to the chosen device once
    dps = (
        torch.from_numpy(np.ascontiguousarray(data.array))
        if data.array is not None
        else data.tensor
    )
    array_t = dps.to(device=device, dtype=torch.float32)
    if array_t.ndim == 2:
        array_t = array_t[None, None]
    # first get COM of mean DP because it gives a robust rough center
    mean_dp_t = array_t.mean(dim=(0, 1))
    total_intensity = mean_dp_t.sum()
    row_grid_t = torch.arange(n_row, dtype=torch.float32, device=device)[:, None]
    col_grid_t = torch.arange(n_col, dtype=torch.float32, device=device)[None, :]
    com_row = int(round(float(((row_grid_t * mean_dp_t).sum() / total_intensity).item())))
    com_col = int(round(float(((col_grid_t * mean_dp_t).sum() / total_intensity).item())))
    # Radial max of the search polar grid, so dp_mean search candidates
    # (at ±global_margin from COM) stay within image bounds
    # in-image. Single pos candidates further from COM might be out of bounds
    # and are masked with [safe_low, safe_high_*] if so
    # (zero-padded samples would otherwise produce a falsely low score)
    # global_margin is the half-width of the candidate-center search around the COM.
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
    # Internal search-grid resolution. Balancing speed against robustness
    search_n_phi = 18
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
    max_r_idx = int(np.ceil(0.9 * n_r))
    # Normalize offsets to [-1, 1] because grid_sample expects normalized coordinates
    col_norm_scale = 2.0 / (n_col - 1)
    row_norm_scale = 2.0 / (n_row - 1)
    base_col_norm = offset_col * col_norm_scale
    base_row_norm = offset_row * row_norm_scale

    # Mean-DP global center search: coarse → fine, masking candidates
    # whose polar grid would extend OOB at each step.
    mean_dp_batch = mean_dp_t[None, None]
    # Coarse: step=4 over ±global_margin around the COM
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
    # Fine: step=1 over ±6 around the coarse winner
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

    # Per-scan-position refinement (coarse → medium → fine) for descan
    # medium and fine search per-DP around the previous winner
    coarse_rows, coarse_cols, coarse_grids = _build_candidate_grids(
        base_col_norm,
        base_row_norm,
        global_row,
        global_col,
        local_margin,
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
    # Per-DP relative offsets used by the medium and fine stages
    med_search_range = torch.arange(
        -local_coarse_step, local_coarse_step + 1, 1, dtype=torch.long, device=device
    )
    med_drow, med_dcol = (
        m.reshape(-1) for m in torch.meshgrid(med_search_range, med_search_range, indexing="ij")
    )
    fine_search_range = torch.arange(-2, 3, dtype=torch.long, device=device)  # [-2,-1,0,1,2] px
    fine_drow, fine_dcol = (
        m.reshape(-1) for m in torch.meshgrid(fine_search_range, fine_search_range, indexing="ij")
    )
    flat_dps_t = array_t.reshape(-1, n_row, n_col)
    n_pos = flat_dps_t.shape[0]
    origin_flat_t = torch.zeros(n_pos, 2, dtype=torch.float32, device=device)

    def refine(dp_batch, current_row, current_col, drow, dcol):
        """Score candidates per DP and return the best (row, col) per DP, plus
        the raw score grid and validity mask (used for sub-pixel refinement).
        Invalid (out of bounds) candidates are masked for the argmin only."""
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
        scores = region.std(dim=2).sum(dim=2) / (region.mean(dim=2).sum(dim=2) + 1e-6)
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

    n_not_converged = 0  # positions whose fine search did not bracket the minimum
    pbar = tqdm(total=n_pos, desc="Finding origin for each scan position")
    for start in range(0, n_pos, batch_size):
        end = min(start + batch_size, n_pos)
        n_dp = end - start  # number of diffraction patterns in this batch chunk
        dp_b = flat_dps_t[start:end].unsqueeze(1)  # (B, 1, H, W)
        # Coarse (shared grids): broadcast B DPs across n_coarse candidate
        # grids in one grid_sample call by stacking DPs in the channel dim
        # and stride-0 expanding along the candidate dim
        polars_coarse = F.grid_sample(
            dp_b.transpose(0, 1).expand(n_coarse, n_dp, n_row, n_col),
            coarse_grids,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        region_coarse = polars_coarse[:, :, :, min_r_idx:max_r_idx]
        scores_coarse = region_coarse.std(dim=2).sum(dim=2) / (
            region_coarse.mean(dim=2).sum(dim=2) + 1e-6
        )
        scores_coarse = scores_coarse.masked_fill(~coarse_valid[:, None], float("inf"))
        best_coarse = scores_coarse.argmin(dim=0)  # best candidate per DP
        current_row, current_col = coarse_rows[best_coarse], coarse_cols[best_coarse]
        # Medium: per-DP search around the coarse winner
        current_row, current_col, _, _ = refine(
            dp_b, current_row, current_col, med_drow, med_dcol
        )
        # Fine: per-DP search around the medium winner
        best_row, best_col, fine_scores, fine_valid = refine(
            dp_b, current_row, current_col, fine_drow, fine_dcol
        )
        # Sub-pixel refine: fit a 2D quadratic to the 3x3 of fine scores around
        # the integer winner and add the vertex offset. The flat (B, n) grid
        # reshapes to (B, side, side) with [:, i, j] = (fine_search_range[i], fine_search_range[j]).
        side = fine_search_range.numel()
        scores_grid = fine_scores.view(n_dp, side, side)
        valid_grid = fine_valid.view(n_dp, side, side)
        flat_best = (
            scores_grid.masked_fill(~valid_grid, float("inf")).view(n_dp, -1).argmin(dim=1)
        )
        # unravel the flat argmin (best candidate pixel) into 2D (row, col) indices in the side x side grid
        i_star, j_star = flat_best // side, flat_best % side
        # Gather the 3x3 patch around (i_star, j_star)
        batch_idx = torch.arange(n_dp, device=device)
        ii = torch.stack([i_star - 1, i_star, i_star + 1], dim=1).clamp(0, side - 1)
        jj = torch.stack([j_star - 1, j_star, j_star + 1], dim=1).clamp(0, side - 1)
        patch = scores_grid[batch_idx[:, None, None], ii[:, :, None], jj[:, None, :]]
        # A winner "on the border" of the fine grid (its 3x3 would fall off the
        # edge) means coarse+medium did not bracket the minimum -- genuine
        # non-convergence, counted and warned about below. Keep the integer
        # winner there; apply the sub-pixel offset only to interior winners.
        on_border = (i_star < 1) | (i_star > side - 2) | (j_star < 1) | (j_star > side - 2)
        n_not_converged += int(on_border.sum())
        offset = _quadratic_subpixel_offset(patch).to(torch.float32)  # (B, 2)
        offset = torch.where(on_border[:, None], torch.zeros_like(offset), offset)
        origin_flat_t[start:end, 0] = best_row.to(torch.float32) + offset[:, 0]
        origin_flat_t[start:end, 1] = best_col.to(torch.float32) + offset[:, 1]
        pbar.update(n_dp)
    pbar.close()
    if n_not_converged:
        warnings.warn(
            f"find_origin_angular_grid: {n_not_converged} of {n_pos} scan positions did not "
            "converge to a sub-pixel origin (descan may exceed local_margin). The integer-pixel origin was "
            "used at those positions.",
            stacklevel=2,
        )
    return origin_flat_t.cpu().numpy().reshape(scan_row, scan_col, 2)


def polar_transform(
    data: Dataset4dstem,
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
) -> Polar4dstem | torch.Tensor:
    if data.ndim != 4:
        raise ValueError(
            f"Found array with shape: {data.shape}. "
            "polar_transform requires a 4D-STEM dataset (ndim=4)."
        )
    scan_row, scan_col, n_row, n_col = data.shape

    # Standardize origin_array input
    if isinstance(origin_array, torch.Tensor):
        origin_array = to_numpy(origin_array)
    origin_array = np.asarray(origin_array) if origin_array is not None else None
    if origin_array is None:
        center = np.array([(n_row - 1) / 2.0, (n_col - 1) / 2.0], dtype=float)
        origins = np.broadcast_to(center, (scan_row, scan_col, 2)).copy()
    elif origin_array.shape == (2,):
        origins = np.empty((scan_row, scan_col, 2), dtype=float)
        origins[...] = origin_array
    elif origin_array.shape == (scan_row, scan_col, 2):
        origins = origin_array
    else:
        raise ValueError(
            f" Got {origin_array.shape}. "
            "origin_array must have shape None, (2,) or (scan_row, scan_col, 2)."
        )

    # If scan_pos is provided, compute polar transform only for that position
    if scan_pos is not None:
        i_row, i_col = scan_pos
        dps = (
            torch.from_numpy(np.ascontiguousarray(data.array))
            if data.array is not None
            else data.tensor
        )
        dp = dps[i_row, i_col].to(device=device, dtype=torch.float32)
        r0 = float(origins[i_row, i_col, 0])
        c0 = float(origins[i_row, i_col, 1])
        # Clamp radial range to image bounds for this origin
        if radial_max is None:
            radial_max_eff = float(min(r0, (n_row - 1) - r0, c0, (n_col - 1) - c0))
        else:
            radial_max_eff = float(radial_max)
        if radial_max_eff <= radial_min:
            radial_max_eff = radial_min + radial_step
        # Build offsets, translate to this origin, normalize for grid_sample
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
        # grid_sample requires (col, row) ordering in the last dim
        grid = torch.stack([col_norm, row_norm], dim=-1).unsqueeze(0)  # (1, n_phi, n_r, 2)
        polar2d = F.grid_sample(
            dp[None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return polar2d.squeeze(0).squeeze(0)  # (n_phi, n_r)

    # Use the global minimum safe radius across all origins so every scan
    # position maps to the same-size polar grid (required for a uniform 4D output)
    if radial_max is None:
        r_row_pos = origins[:, :, 0]
        r_row_neg = (n_row - 1) - origins[:, :, 0]
        r_col_pos = origins[:, :, 1]
        r_col_neg = (n_col - 1) - origins[:, :, 1]
        radial_max_eff_array = np.minimum.reduce([r_row_pos, r_row_neg, r_col_pos, r_col_neg])
        radial_max = float(max(radial_max_eff_array.min(), radial_min + radial_step))

    # Build origin-independent polar offsets ONCE. Only the per-origin shift
    # changes from one scan position to the next, so we can reuse these.
    offset_row, offset_col, phi_bins, radial_bins = _build_polar_sampling_offsets(
        ellipse_params,
        num_annular_bins,
        radial_min,
        float(radial_max),
        radial_step,
        two_fold_rotation_symmetry,
        device,
    )
    n_phi = phi_bins.numel()
    n_r = radial_bins.numel()
    radial_max_eff = float(radial_max)

    # Pre-normalize offsets into grid_sample's [-1, 1] coordinate convention
    col_norm_scale = 2.0 / (n_col - 1)
    row_norm_scale = 2.0 / (n_row - 1)
    base_col_norm = offset_col * col_norm_scale  # (n_phi, n_r)
    base_row_norm = offset_row * row_norm_scale  # (n_phi, n_r)

    # Flatten scan dims so we can iterate in flat batches
    n_pos = scan_row * scan_col
    dps = (
        torch.from_numpy(np.ascontiguousarray(data.array))
        if data.array is not None
        else data.tensor
    )
    dp_view = dps.reshape(n_pos, n_row, n_col)
    origins_t = torch.from_numpy(
        np.ascontiguousarray(origins.reshape(n_pos, 2), dtype=np.float32)
    ).to(device)

    out = torch.empty((n_pos, n_phi, n_r), dtype=torch.float32, device=device)
    for start in tqdm(range(0, n_pos, batch_size), desc="Polar transform"):
        end = min(start + batch_size, n_pos)
        # Translate the precomputed offsets to each origin in this batch
        row_origins = origins_t[start:end, 0]
        col_origins = origins_t[start:end, 1]
        grid_col = base_col_norm.unsqueeze(0) + (col_origins * col_norm_scale - 1.0)[:, None, None]
        grid_row = base_row_norm.unsqueeze(0) + (row_origins * row_norm_scale - 1.0)[:, None, None]
        # grid_sample requires (col, row) ordering in the last dim
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
    out = out.reshape(scan_row, scan_col, n_phi, n_r)

    # Get polar axes in physical units matching the input dataset's calibration
    phi_range = np.pi if two_fold_rotation_symmetry else 2.0 * np.pi
    phi_step_deg = (phi_range / float(n_phi)) * (180.0 / np.pi)
    sampling = np.zeros(4, dtype=float)
    origin = np.zeros(4, dtype=float)
    sampling[0:2] = np.asarray(data.sampling)[0:2]
    sampling[2] = phi_step_deg
    sampling[3] = float(np.asarray(data.sampling)[-1]) * radial_step
    origin[0:2] = np.asarray(data.origin)[0:2]
    origin[2] = 0.0
    origin[3] = radial_min * float(np.asarray(data.sampling)[-1])
    units = [
        data.units[0],
        data.units[1],
        "deg",
        data.units[-1],
    ]
    metadata = dict(data.metadata)
    metadata.update(
        {
            "polar_radial_min": float(radial_min),
            "polar_radial_max": float(radial_max_eff),
            "polar_radial_step": float(radial_step),
            "polar_num_annular_bins": int(n_phi),
            "polar_two_fold_rotation_symmetry": bool(two_fold_rotation_symmetry),
            "polar_ellipticity": tuple(ellipse_params) if ellipse_params is not None else None,
        }
    )
    return Polar4dstem(
        tensor=out,
        name=name if name is not None else f"{data.name}_polar",
        origin=origin,
        sampling=sampling,
        units=units,
        signal_units=signal_units if signal_units is not None else data.signal_units,
        metadata=metadata,
        origin_array=origins,
        _token=Polar4dstem._token,
    )


def _polar_to_cartesian_offsets(
    phi: torch.Tensor,
    r_pix: torch.Tensor,
    ellipse_params: tuple[float, float, float] | None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert polar (phi, r_pix) grids to Cartesian (row, col) pixel offsets
    from the origin, optionally correcting for elliptical distortion.

    Returns ``(offset_row, offset_col)`` where
    ``col_offset = r_pix * cos(phi)`` and ``row_offset = r_pix * sin(phi)``.
    """
    if ellipse_params is None:
        offset_col = r_pix * torch.cos(phi)
        offset_row = r_pix * torch.sin(phi)
    else:
        if len(ellipse_params) != 3:
            raise ValueError("ellipse_params must be (a, b, theta_deg).")
        a, b, theta_deg = ellipse_params
        theta = torch.deg2rad(torch.tensor(theta_deg, dtype=torch.float32, device=device))
        # Rotate into the ellipse frame, scale by a/b to undo the distortion,
        # then rotate back so sampling follows the true circular rings
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
    """Build origin-independent Cartesian (row, col) offsets for a polar
    sampling grid.

    Returns ``(offset_row, offset_col, phi_bins, radial_bins)`` where
    ``offset_row`` and ``offset_col`` have shape ``(n_phi, n_r)`` and
    represent pixel displacements from an arbitrary origin.
    """
    if radial_step <= 0:
        raise ValueError(f"Got radial_step = {radial_step}. radial_step must be > 0.")
    if num_annular_bins < 1:
        raise ValueError("num_annular_bins must be >= 1.")

    radial_bins = torch.arange(
        radial_min, radial_max_eff, radial_step, dtype=torch.float32, device=device
    )
    if radial_bins.numel() == 0:
        radial_bins = torch.tensor([radial_min], dtype=torch.float32, device=device)
    phi_range = torch.pi if two_fold_rotation_symmetry else 2.0 * torch.pi
    # Drop the last endpoint because 0 and 2pi (or pi) are the same angle
    phi_bins = torch.linspace(
        0.0, phi_range, num_annular_bins + 1, dtype=torch.float32, device=device
    )[:-1]
    phi_grid, r_pix_grid = torch.meshgrid(phi_bins, radial_bins, indexing="ij")
    # Compute offsets relative to origin (0,0) so they can be reused
    # for any candidate origin by simple translation
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
    """Build a batch of normalized sampling grids, one per candidate origin
    pixel in a search window around (center_row, center_col). Candidates are
    produced in a single batched tensor so that they can be evaluated
    simultaneously by ``_angular_std_scores``.

    Parameters
    ----------
    base_col_norm, base_row_norm : torch.Tensor of shape (n_phi, n_r)
        Polar sampling offsets, already expressed in ``grid_sample``'s
        normalized [-1, 1] coordinates, relative to origin (0, 0)
    center_row, center_col : int
        Center of the candidate search window
    margin : int
        Half-width of the search window in pixels
    n_row, n_col : int
        Diffraction-pattern image dimensions
    col_norm_scale, row_norm_scale : float
        Conversion factor from an offset in pixel units to the equivalent
        offset in ``grid_sample``'s normalized coordinates

    Returns
    -------
    row_flat, col_flat : torch.Tensor of shape (N,)
        Candidate origin positions
    grids : torch.Tensor of shape (N, n_phi, n_r, 2)
        Stacked sampling grids ready for ``F.grid_sample`` (ordered ``(col, row)`` )
    """
    # Enumerate all pixel positions in the search window, clamped to image bounds
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
    # Shift the pre-computed polar offsets to each candidate origin,
    # converting to grid_sample's [-1, 1] normalized coordinates
    grid_col = (
        base_col_norm.unsqueeze(0) + (col_flat.float() * col_norm_scale - 1.0)[:, None, None]
    )
    grid_row = (
        base_row_norm.unsqueeze(0) + (row_flat.float() * row_norm_scale - 1.0)[:, None, None]
    )
    # grid_sample requires (col, row) ordering in the last dim
    grids = torch.stack([grid_col, grid_row], dim=-1)  # (N, n_phi, n_r, 2)
    return row_flat, col_flat, grids


def _angular_std_scores(
    dp_batch: torch.Tensor,
    grids: torch.Tensor,
    min_r_idx: int,
    max_r_idx: int,
) -> torch.Tensor:
    """Score candidate origins by angular std over a mid-radius band.
    Lower scores indicate better centering."""
    n = grids.shape[0]
    # Sample the diffraction pattern at each candidate's polar grid positions
    polars = F.grid_sample(
        dp_batch.expand(n, -1, -1, -1),
        grids,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    # A correctly centered pattern has uniform intensity along each ring,
    # so the angular std is minimized at the true center. Normalize by
    # mean intensity so candidates whose polar grid samples only background
    # do not win on absolute std alone (relevant when search windows are
    # wide enough to include centers far from the direct beam).
    region = polars.squeeze(1)[:, :, min_r_idx:max_r_idx]
    return region.std(dim=1).sum(dim=1) / (region.mean(dim=1).sum(dim=1) + 1e-6)


def _quadratic_subpixel_offset(patch: torch.Tensor) -> torch.Tensor:
    """Sub-pixel location of the minimum of a 3x3 score patch.

    Fits a 2D quadratic surface
        s(u, v) = a + b u + c v + d u^2 + e v^2 + f u v
    (u = row offset, v = col offset, both over {-1, 0, 1}) to each patch by
    least squares, then returns the vertex (the point where both partial
    derivatives vanish):
        [2d  f ] [dr]   [-b]
        [f   2e] [dc] = [-c]
    The fit depends only on the fixed 3x3 offset positions, so the
    least-squares matrix (``fit_matrix``) is identical for every patch.

    Parameters
    ----------
    patch
        (N, 3, 3) score patches. ``patch[:, i, j]`` is the score at row
        offset ``i - 1`` and col offset ``j - 1``.

    Returns
    -------
    offset
        (N, 2) sub-pixel ``(drow, dcol)`` offsets in pixels, each clamped to
        [-1, 1]. Rows whose fit is not a clean minimum (Hessian not positive
        definite, or any non-finite score) get ``(0, 0)``.
    """
    device = patch.device
    scores_flat = patch.reshape(patch.shape[0], 9).to(torch.float64)  # (N, 9)
    # The 9 fixed offset positions of the 3x3 patch, flattened row-major to
    # match scores_flat.
    grid = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64, device=device)
    uu, vv = torch.meshgrid(grid, grid, indexing="ij")
    u, v = uu.reshape(9), vv.reshape(9)
    # Design matrix: each row evaluates the quadratic basis
    # [1, u, v, u^2, v^2, u v] at one offset position.
    basis_matrix = torch.stack([torch.ones_like(u), u, v, u * u, v * v, u * v], dim=1)  # (9, 6)
    # Least-squares pseudo-inverse, same for every patch: coeffs = fit_matrix @ scores.
    fit_matrix = torch.linalg.pinv(basis_matrix)  # (6, 9)
    # fit coeffs
    a, b, c, d, e, f = (scores_flat @ fit_matrix.T).unbind(dim=1)  # each (N,)
    # find minimum
    det = 4.0 * d * e - f * f  # determinant of the 2x2 curvature (Hessian) matrix
    # Accept only clean minima (Hessian positive definite) with finite scores.
    valid = (2.0 * d > 0) & (det > 1e-12) & torch.isfinite(scores_flat).all(dim=1)
    det_safe = torch.where(valid, det, torch.ones_like(det))
    drow = torch.where(valid, (f * c - 2.0 * e * b) / det_safe, torch.zeros_like(det))
    dcol = torch.where(valid, (f * b - 2.0 * d * c) / det_safe, torch.zeros_like(det))
    return torch.stack([drow.clamp(-1.0, 1.0), dcol.clamp(-1.0, 1.0)], dim=1)



def find_origin_angular_descent(
    data: Dataset4dstem,
    *,
    ellipse_params: tuple[float, float, float] | None = None,
    radial_min: float = 0.0,
    radial_max: float | None = None,
    n_phi: int = 120,
    radial_step: float = 1.0,
    kpow: float = 0.0,
    device: str = "cpu",
) -> NDArray:
    """Margin-free diffraction-center finding by COM-anchored local descent.

    Anchors every pattern at the intensity centroid and walks downhill, then 
    fits a 2D quadratic for the sub-pixel center. Robust on small detectors.

    Parameters
    ----------
    data : Dataset4dstem
        A 4D-STEM dataset (or a 2D dataset, treated as a 1x1 scan).
    ellipse_params : tuple or None
        ``(a, b, theta_deg)`` to sample along elliptical rings. Origin finding stays
        accurate without it (the center is located by the pattern's inversion
        symmetry, independent of ring shape); supplying it deepens the score minimum
        and is a mild refinement on strongly elliptical data. ``None`` = circular.
    radial_min, radial_max : float
        Inner / outer radius (px) of the scoring annulus. Set ``radial_min`` above
        the central beam (or any saturated core). ``radial_max=None`` defaults to
        ``min(n_row, n_col) // 2 - 2``.
    n_phi : int
        Number of angular samples per ring.
    radial_step : float
        Radial sampling step (px).
    kpow : float
        Up-weight each ring's contribution by ``k**kpow`` (k == radius). ``0`` is the
        plain ratio-of-sums objective; ``1``/``2`` emphasize high-k.
    device : str
        Torch device.

    Returns
    -------
    origin_array : np.ndarray
        Shape ``(scan_row, scan_col, 2)`` of ``(row, col)`` origins in pixels.

    See Also
    --------
    find_origin_angular_grid : global angular-variance grid search; more robust to a
        poor initial guess.
    """
    if data.ndim == 2:
        scan_row, scan_col, n_row, n_col = 1, 1, *data.shape
    elif data.ndim == 4:
        scan_row, scan_col, n_row, n_col = data.shape
    else:
        raise ValueError(
            f"Got array with shape {data.shape}. "
            "To use find_origin_angular_descent, pass a 2D or 4D-STEM dataset."
        )
    if radial_max is None:
        radial_max = float(min(n_row, n_col) // 2 - 2)
    n_radial = max(4, int(round((radial_max - radial_min) / radial_step)) + 1)
    data_tensor = (
        torch.from_numpy(np.ascontiguousarray(data.array))
        if data.array is not None
        else data.tensor
    )
    array_tensor = data_tensor.to(device=device, dtype=torch.float32)
    if array_tensor.ndim == 2:
        array_tensor = array_tensor[None, None]
    patterns = array_tensor.reshape(-1, n_row, n_col)
    n_patterns = patterns.shape[0]
    offset_row, offset_col, ring_weights = _local_sampling(
        radial_min, radial_max, n_phi, n_radial, kpow, ellipse_params, device
    )
    # Global anchor: descend on the high-SNR mean DP (a batch of one), seeded at its COM.
    mean_pattern = array_tensor.mean(dim=(0, 1))
    global_origin = _descend_batched(
        mean_pattern[None], torch.round(_com_anchor(mean_pattern))[None],
        offset_row, offset_col, ring_weights, n_phi, n_radial, device,
    )[0]
    # Per-position: start every DP at the integer global anchor and descend (tracks descan).
    start_centers = torch.round(global_origin)[None].expand(n_patterns, 2).clone()
    origins = _descend_batched(
        patterns, start_centers, offset_row, offset_col, ring_weights, n_phi, n_radial, device,
    )
    return origins.reshape(scan_row, scan_col, 2).cpu().numpy()


# --- helpers for find_origin_angular_descent -------------------------------------
# The 8 neighbor directions (axial + diagonal) for the pattern search.
_NEIGHBOR_STEPS_8 = [[1.0, 0], [-1, 0], [0, 1.0], [0, -1], [1, 1.0], [1, -1], [-1, 1], [-1, -1]]
# The 9 offsets of a 3x3 patch, row-major: index k = i*3 + j -> (row i-1, col j-1);
# the patch center is k = 4.
_PATCH_OFFSETS_3X3 = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]


def _com_anchor(pattern: torch.Tensor) -> torch.Tensor:
    """Rough center guess from the intensity-weighted centroid of the DP."""
    n_row, n_col = pattern.shape
    clipped = pattern.clamp(min=0)
    total = clipped.sum() + 1e-9
    rows = torch.arange(n_row, device=pattern.device, dtype=torch.float32)
    cols = torch.arange(n_col, device=pattern.device, dtype=torch.float32)
    center_row = (rows[:, None] * clipped).sum() / total
    center_col = (cols[None, :] * clipped).sum() / total
    return torch.tensor([center_row.item(), center_col.item()], device=pattern.device)


def _local_sampling(radial_min, radial_max, n_phi, n_radial, kpow, ellipse_params, device):
    """Polar-sampling template (row/col offset per (phi, radius)) + per-ring k-weights.

    With ``ellipse_params=(a, b, theta_deg)`` the template follows the elliptical rings. ``None`` samples on
    circles. Uses the same ``_polar_to_cartesian_offsets`` transform as the grid finder."""
    phi = torch.linspace(0, 2 * np.pi, n_phi + 1, device=device)[:-1]
    radii = torch.linspace(radial_min, radial_max, n_radial, device=device)
    phi_grid, radius_grid = torch.meshgrid(phi, radii, indexing="ij")   # (n_phi, n_radial)
    offset_row, offset_col = _polar_to_cartesian_offsets(
        phi_grid, radius_grid, ellipse_params, device
    )
    ring_weights = radii**kpow                                          # (n_radial,)
    return offset_row, offset_col, ring_weights


def _local_polar_score(polar_values, valid_mask, n_phi, ring_weights, min_valid_frac):
    """Angular-uniformity score from sampled polar values."""
    n_valid = valid_mask.sum(dim=-2).clamp(min=1)              # (..., n_radial)
    ring_mean = (polar_values * valid_mask).sum(dim=-2) / n_valid
    ring_var = (((polar_values - ring_mean.unsqueeze(-2)) ** 2) * valid_mask).sum(dim=-2) / n_valid
    ring_std = ring_var.sqrt()
    ring_usable = valid_mask.sum(dim=-2) >= (min_valid_frac * n_phi)
    weights = ring_weights * ring_usable
    score = (weights * ring_std).sum(dim=-1) / ((weights * ring_mean.abs()).sum(dim=-1) + 1e-6)
    return score, ring_usable


def _local_score_pairs(patterns, pattern_index, centers, offset_row, offset_col,
                       ring_weights, n_phi, device, min_valid_frac=0.5, chunk=4096):
    """Score K (pattern, center) pairs: patterns[pattern_index[k]] scored at centers[k].

    patterns:(M,H,W), pattern_index:(K,), centers:(K,2) -> (K,) angular-uniformity scores."""
    _, n_row, n_col = patterns.shape
    ones_image = torch.ones(1, 1, n_row, n_col, device=device)
    scores = torch.empty(centers.shape[0], device=device)
    for start in range(0, centers.shape[0], chunk):
        index = pattern_index[start:start + chunk]
        n_chunk = index.shape[0]
        center_row = centers[start:start + chunk, 0][:, None, None]
        center_col = centers[start:start + chunk, 1][:, None, None]
        sample_grid = torch.stack(
            [2.0 * (center_col + offset_col[None]) / (n_col - 1) - 1.0,
             2.0 * (center_row + offset_row[None]) / (n_row - 1) - 1.0], dim=-1
        )
        polar_values = F.grid_sample(
            patterns[index][:, None], sample_grid, mode="bilinear",
            padding_mode="zeros", align_corners=True
        )[:, 0]
        valid_mask = F.grid_sample(
            ones_image.expand(n_chunk, 1, n_row, n_col), sample_grid, mode="bilinear",
            padding_mode="zeros", align_corners=True
        )[:, 0] > 0.999
        scores[start:start + n_chunk], _ = _local_polar_score(
            polar_values, valid_mask, n_phi, ring_weights, min_valid_frac
        )
    return scores


def _descend_batched(patterns, anchors, offset_row, offset_col, ring_weights, n_phi,
                     n_radial, device, schedule=(4.0, 2.0, 1.0), sweeps=2):
    """Batched coarse descent (shared step schedule, to 1 px) + quadratic sub-pixel.

    Walks each pattern downhill on the angular-uniformity score from its integer
    ``anchors`` center (Hooke-Jeeves 8-neighbor search, step halving over ``schedule``),
    then fits a 2D quadratic to the local 3x3 for the sub-pixel vertex. Used for both
    the global mean-DP anchor (a batch of one) and the per-position pass.

    patterns : (M, H, W);  anchors : (M, 2) integer-valued start centers.
    Returns (M, 2) sub-pixel (row, col) origins.
    """
    n_patterns = patterns.shape[0]
    pattern_ids = torch.arange(n_patterns, device=device)
    neighbor_steps = torch.tensor(_NEIGHBOR_STEPS_8, device=device)
    patch_offsets = torch.tensor(_PATCH_OFFSETS_3X3, dtype=torch.float32, device=device)
    pattern_ids_per_neighbor = pattern_ids.repeat_interleave(8)
    pattern_ids_per_patch = pattern_ids.repeat_interleave(9)
    center = anchors.clone()

    def score_at(pattern_index, centers):
        return _local_score_pairs(patterns, pattern_index, centers, offset_row, offset_col,
                                  ring_weights, n_phi, device)

    best_score = score_at(pattern_ids, center)
    # Coarse descent: at each step size, move to the best-improving 8-neighbor; when no
    # neighbor improves, halve the step. Brackets the minimum to ~1 px.
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
    # Re-center each 3x3 patch on its integer minimum so the quadratic brackets it.
    patch_centers = (center[:, None, :] + patch_offsets[None]).reshape(n_patterns * 9, 2)
    patch_scores = score_at(pattern_ids_per_patch, patch_centers).reshape(n_patterns, 9)
    center = patch_centers.reshape(n_patterns, 9, 2)[pattern_ids, patch_scores.argmin(dim=1)]
    # Sub-pixel vertex from a 2D-quadratic fit to the final 3x3 of scores.
    patch_centers = (center[:, None, :] + patch_offsets[None]).reshape(n_patterns * 9, 2)
    score_patch = score_at(pattern_ids_per_patch, patch_centers).reshape(n_patterns, 3, 3)
    subpixel_offset = _quadratic_subpixel_offset(score_patch).to(torch.float32)
    return center + subpixel_offset
