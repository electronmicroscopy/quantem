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


def _normalize_coords_for_grid_sample(
    coords_row: torch.Tensor,
    coords_col: torch.Tensor,
    n_row: int,
    n_col: int,
) -> torch.Tensor:
    """Convert pixel (row, col) coordinates to normalized [-1, 1] coordinates
    for grid_sample."""
    col_norm = 2.0 * coords_col / (n_col - 1) - 1.0
    row_norm = 2.0 * coords_row / (n_row - 1) - 1.0
    # grid_sample requires (col, row) ordering in the last dim
    return torch.stack([col_norm, row_norm], dim=-1)


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
        dtype=torch.long,
        device=device,
    )
    cols = torch.arange(
        max(0, center_col - margin),
        min(n_col, center_col + margin + 1),
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
    # so the angular std is minimized at the true center
    region = polars.squeeze(1)[:, :, min_r_idx:max_r_idx]
    return region.std(dim=1).sum(dim=1)


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


def _compute_radial_max(
    n_row: int,
    n_col: int,
    origin_row: float,
    origin_col: float,
    radial_max: float | None,
    radial_min: float,
    radial_step: float,
) -> float:
    """Compute the effective maximum radius, clamped to image bounds."""
    # Use the shortest distance from the origin to any image edge so the
    # polar grid never samples outside the image bounds
    if radial_max is None:
        radial_max_eff = float(
            min(
                origin_row,
                (n_row - 1) - origin_row,
                origin_col,
                (n_col - 1) - origin_col,
            )
        )
    else:
        radial_max_eff = float(radial_max)
    # Guarantee at least one radial bin
    if radial_max_eff <= radial_min:
        radial_max_eff = radial_min + radial_step
    return radial_max_eff


def _precompute_polar_coords(
    n_row: int,
    n_col: int,
    origin_row: float,
    origin_col: float,
    ellipse_params: tuple[float, float, float] | None,
    num_annular_bins: int,
    radial_min: float,
    radial_max: float | None,
    radial_step: float,
    two_fold_rotation_symmetry: bool,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Build a normalized sampling grid for a single known origin."""
    origin_row = float(origin_row)
    origin_col = float(origin_col)
    # Clamp radial range so the polar grid stays within image bounds
    radial_max_eff = _compute_radial_max(
        n_row,
        n_col,
        origin_row,
        origin_col,
        radial_max,
        radial_min,
        radial_step,
    )
    # Get origin-independent polar offsets
    offset_row, offset_col, phi_bins, radial_bins = _build_polar_sampling_offsets(
        ellipse_params,
        num_annular_bins,
        radial_min,
        radial_max_eff,
        radial_step,
        two_fold_rotation_symmetry,
        device,
    )
    # Translate offsets to absolute pixel pos at this origin
    coords_row = offset_row + origin_row
    coords_col = offset_col + origin_col
    # Convert to [-1, 1] normalized coordinates expected by grid_sample
    grid = _normalize_coords_for_grid_sample(coords_row, coords_col, n_row, n_col)
    grid = grid.unsqueeze(0)  # (1, n_phi, n_r, 2)
    return grid, phi_bins, radial_bins, radial_max_eff


def auto_origin_id(
    data: Dataset4dstem,
    *,
    ellipse_params: tuple[float, float, float] | None = None,
    num_annular_bins: int = 180,
    radial_min: float = 0.0,
    radial_max: float | None = None,
    radial_step: float = 1.0,
    two_fold_rotation_symmetry: bool = False,
    device: str = "cpu",
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
        Number of angular bins for the final polar transform (not used
        during center-finding, which uses 36 bins for speed)
    radial_min : float
        Minimum radius in pixels
    radial_max : float or None
        Maximum radius in pixels
    radial_step : float
        Radial step size in pixels
    two_fold_rotation_symmetry : bool
        If True, use only 0 to pi range for angles
    device : str
        Torch device for computation

    Returns
    -------
    origin_array : np.ndarray
        Array of shape (scan_row, scan_col, 2) containing (row, col) origin
        estimates in pixels.
    """
    if len(data.array.shape) == 2:
        n_row, n_col = data.array.shape
        scan_row, scan_col = 1, 1
    elif len(data.array.shape) == 4:
        scan_row, scan_col, n_row, n_col = data.array.shape
    else:
        raise ValueError(
            f" Got array with shape {data.array.shape}."
            "To use auto_origin_id, pass a 2D or 4DSTEM dataset."
        )

    origin_array = np.zeros((scan_row, scan_col, 2), dtype=float)
    total_positions = scan_row * scan_col

    # first get COM of mean DP because it gives a robust rough center
    array_4d = data.array if data.array.ndim == 4 else data.array[None, None, :, :]
    mean_dp_np = array_4d.mean(axis=(0, 1)).astype(np.float32)
    total_intensity = mean_dp_np.sum()
    row_grid, col_grid = np.mgrid[0:n_row, 0:n_col]
    com_row = int(round(float((row_grid * mean_dp_np).sum() / total_intensity)))
    com_col = int(round(float((col_grid * mean_dp_np).sum() / total_intensity)))

    # building a fixed polar grid that is safe for all candidates
    # safe_rmax ensures no candidate's grid extends outside the image
    global_margin = 20
    safe_rmax = float(
        min(
            com_row - global_margin,
            (n_row - 1) - (com_row + global_margin),
            com_col - global_margin,
            (n_col - 1) - (com_col + global_margin),
        )
    )
    if radial_max is not None:
        safe_rmax = min(safe_rmax, float(radial_max))
    if safe_rmax <= radial_min:
        safe_rmax = radial_min + radial_step
    # use very coarse binning because asymmetry is still captured at
    # low angular resolution and is significantly faster
    search_n_phi = 36
    offset_row, offset_col, _, radial_bins = _build_polar_sampling_offsets(
        ellipse_params,
        search_n_phi,
        radial_min,
        safe_rmax,
        radial_step,
        two_fold_rotation_symmetry,
        device,
    )
    n_r = radial_bins.numel()
    min_r_idx = int(np.floor(0.1 * n_r))
    max_r_idx = int(np.ceil(0.9 * n_r))
    # Normalize offsets to [-1, 1] because grid_sample expects normalized coordinates
    col_norm_scale = 2.0 / (n_col - 1)
    row_norm_scale = 2.0 / (n_row - 1)
    base_col_norm = offset_col * col_norm_scale
    base_row_norm = offset_row * row_norm_scale

    # now find actual center
    # Coarse search over ±global_margin around COM
    coarse_step = 4
    coarse_rows = torch.arange(
        max(0, com_row - global_margin),
        min(n_row, com_row + global_margin + 1),
        coarse_step,
        dtype=torch.long,
        device=device,
    )
    coarse_cols = torch.arange(
        max(0, com_col - global_margin),
        min(n_col, com_col + global_margin + 1),
        coarse_step,
        dtype=torch.long,
        device=device,
    )
    # Create all (row, col) candidate pairs and flatten for batched evaluation
    coarse_row_grid, coarse_col_grid = torch.meshgrid(coarse_rows, coarse_cols, indexing="ij")
    coarse_row_flat, coarse_col_flat = coarse_row_grid.reshape(-1), coarse_col_grid.reshape(-1)
    # Shift polar offsets to each candidate origin in normalized coordinates
    coarse_g_col = (
        base_col_norm.unsqueeze(0)
        + (coarse_col_flat.float() * col_norm_scale - 1.0)[:, None, None]
    )
    coarse_g_row = (
        base_row_norm.unsqueeze(0)
        + (coarse_row_flat.float() * row_norm_scale - 1.0)[:, None, None]
    )
    # grid_sample requires (col, row) ordering in the last dim
    coarse_grids = torch.stack([coarse_g_col, coarse_g_row], dim=-1)
    # Score all coarse candidates on the mean DP and pick the best one
    mean_dp_batch = torch.from_numpy(mean_dp_np).to(device)[None, None]
    coarse_scores = _angular_std_scores(mean_dp_batch, coarse_grids, min_r_idx, max_r_idx)
    best_coarse_idx = coarse_scores.argmin().item()
    coarse_best_row = int(coarse_row_flat[best_coarse_idx].item())
    coarse_best_col = int(coarse_col_flat[best_coarse_idx].item())

    # Fine search (step=1) around coarse best for global center of mean DP
    fine_margin = 6
    fine_row_flat, fine_col_flat, fine_grids = _build_candidate_grids(
        base_col_norm,
        base_row_norm,
        coarse_best_row,
        coarse_best_col,
        fine_margin,
        n_row,
        n_col,
        col_norm_scale,
        row_norm_scale,
        device,
    )
    fine_scores = _angular_std_scores(mean_dp_batch, fine_grids, min_r_idx, max_r_idx)
    best_fine_idx = fine_scores.argmin().item()
    global_row = int(fine_row_flat[best_fine_idx].item())
    global_col = int(fine_col_flat[best_fine_idx].item())
    # Get center for each scan pos by fine search around global center
    # Assuming that the center doesn't shift more than 10 pixels across the scan
    local_margin = 10
    local_rf, local_cf, local_grids = _build_candidate_grids(
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
    )
    pbar = tqdm(total=total_positions, desc="Finding origin for each scan position")
    for row_pos in range(scan_row):
        row_dps = torch.from_numpy(array_4d[row_pos].astype(np.float32)).to(
            device
        )  # (scan_col, n_row, n_col)

        for col_pos in range(scan_col):
            dp_batch = row_dps[col_pos][None, None]
            scores = _angular_std_scores(dp_batch, local_grids, min_r_idx, max_r_idx)
            best_idx = scores.argmin().item()
            origin_array[row_pos, col_pos, 0] = local_rf[best_idx].item()
            origin_array[row_pos, col_pos, 1] = local_cf[best_idx].item()
            pbar.update(1)

    pbar.close()
    return origin_array


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
) -> Polar4dstem | torch.Tensor:
    if data.array.ndim != 4:
        raise ValueError(
            f"Found array with shape: {data.array.shape}. "
            "polar_transform requires a 4D-STEM dataset (ndim=4)."
        )
    scan_row, scan_col, n_row, n_col = data.array.shape

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
        dp = torch.from_numpy(data.array[i_row, i_col].astype(np.float32)).to(device)
        r0 = float(origins[i_row, i_col, 0])
        c0 = float(origins[i_row, i_col, 1])
        grid, phi_bins, radial_bins, radial_max_eff = _precompute_polar_coords(
            n_row=n_row,
            n_col=n_col,
            origin_row=r0,
            origin_col=c0,
            ellipse_params=ellipse_params,
            num_annular_bins=num_annular_bins,
            radial_min=radial_min,
            radial_max=radial_max,
            radial_step=radial_step,
            two_fold_rotation_symmetry=two_fold_rotation_symmetry,
            device=device,
        )
        dp_batch = dp[None, None]  # (1, 1, n_row, n_col)
        polar2d = F.grid_sample(
            dp_batch,
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

    # Compute grid for first position to get output shape
    grid, phi_bins, radial_bins, radial_max_eff = _precompute_polar_coords(
        n_row=n_row,
        n_col=n_col,
        origin_row=float(origins[0, 0, 0]),
        origin_col=float(origins[0, 0, 1]),
        ellipse_params=ellipse_params,
        num_annular_bins=num_annular_bins,
        radial_min=radial_min,
        radial_max=radial_max,
        radial_step=radial_step,
        two_fold_rotation_symmetry=two_fold_rotation_symmetry,
        device=device,
    )
    n_phi = phi_bins.numel()
    n_r = radial_bins.numel()
    out = np.empty((scan_row, scan_col, n_phi, n_r), dtype=np.float32)
    for i_row in range(scan_row):
        for i_col in range(scan_col):
            dp = torch.from_numpy(data.array[i_row, i_col].astype(np.float32)).to(device)
            r0 = float(origins[i_row, i_col, 0])
            c0 = float(origins[i_row, i_col, 1])
            grid, _, _, _ = _precompute_polar_coords(
                n_row=n_row,
                n_col=n_col,
                origin_row=r0,
                origin_col=c0,
                ellipse_params=ellipse_params,
                num_annular_bins=num_annular_bins,
                radial_min=radial_min,
                radial_max=radial_max,
                radial_step=radial_step,
                two_fold_rotation_symmetry=two_fold_rotation_symmetry,
                device=device,
            )
            dp_batch = dp[None, None]
            polar2d = F.grid_sample(
                dp_batch,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            out[i_row, i_col] = to_numpy(polar2d.squeeze(0).squeeze(0))

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
        array=out,
        name=name if name is not None else f"{data.name}_polar",
        origin=origin,
        sampling=sampling,
        units=units,
        signal_units=signal_units if signal_units is not None else data.signal_units,
        metadata=metadata,
        origin_array=origins,
        _token=Polar4dstem._token,
    )
