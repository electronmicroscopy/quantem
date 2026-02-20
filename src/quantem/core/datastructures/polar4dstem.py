from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .dataset4dstem import Dataset4dstem

from quantem.core.datastructures.dataset4d import Dataset4d


class Polar4dstem(Dataset4d):
    """4D-STEM dataset in polar coordinates (scan_y, scan_x, phi, r)."""

    def __init__(
        self,
        array: NDArray | Any,
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        signal_units: str = "arb. units",
        metadata: dict | None = None,
        origin_array: NDArray | None = None,
        _token: object | None = None,
    ):
        if metadata is None:
            metadata = {}
        mdata_keys_polar = [
            "polar_radial_min",
            "polar_radial_max",
            "polar_radial_step",
            "polar_num_annular_bins",
            "polar_two_fold_rotation_symmetry",
            "polar_ellipse_params",
        ]
        for k in mdata_keys_polar:
            if k not in metadata:
                metadata[k] = None
        super().__init__(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            metadata=metadata,
            _token=_token,
        )
        self._xp = np  # workaround: Dataset.coords() references _xp
        self.origin_array = origin_array

    @classmethod
    def from_array(
        cls,
        array: NDArray | Any,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
        metadata: dict | None = None,
    ) -> "Polar4dstem":
        array = np.asarray(array)
        if array.ndim != 4:
            raise ValueError("Polar4dstem.from_array expects a 4D array.")
        if origin is None:
            origin = np.zeros(4, dtype=float)
        if sampling is None:
            sampling = np.ones(4, dtype=float)
        if units is None:
            units = ["pixels", "pixels", "deg", "pixels"]
        if metadata is None:
            metadata = {}
        return cls(
            array=array,
            name=name if name is not None else "Polar 4D-STEM dataset",
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            metadata=metadata,
            _token=cls._token,
        )

    @property
    def n_phi(self) -> int:
        return int(self.array.shape[2])

    @property
    def n_r(self) -> int:
        return int(self.array.shape[3])


def _to_numpy(tensor: torch.Tensor) -> NDArray:
    """Convert torch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def _normalize_coords_for_grid_sample(
    coords_y: torch.Tensor,
    coords_x: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Convert pixel coordinates to normalized [-1, 1] coordinates for grid_sample.

    grid_sample expects x_norm = 2*x/(W-1) - 1, y_norm = 2*y/(H-1) - 1,
    stacked as (..., 2) in [x, y] order.
    """
    x_norm = 2.0 * coords_x / (width - 1) - 1.0
    y_norm = 2.0 * coords_y / (height - 1) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def _precompute_polar_coords(
    ny: int,
    nx: int,
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
    origin_row = float(origin_row)
    origin_col = float(origin_col)
    if radial_step <= 0:
        raise ValueError("radial_step must be > 0.")
    if num_annular_bins < 1:
        raise ValueError("num_annular_bins must be >= 1.")
    if radial_max is None:
        r_row_pos = origin_row
        r_row_neg = (ny - 1) - origin_row
        r_col_pos = origin_col
        r_col_neg = (nx - 1) - origin_col
        radial_max_eff = float(min(r_row_pos, r_row_neg, r_col_pos, r_col_neg))
    else:
        radial_max_eff = float(radial_max)
    if radial_max_eff <= radial_min:
        radial_max_eff = radial_min + radial_step
    radial_bins = torch.arange(
        radial_min, radial_max_eff, radial_step, dtype=torch.float32, device=device
    )
    if radial_bins.numel() == 0:
        radial_bins = torch.tensor([radial_min], dtype=torch.float32, device=device)
    phi_range = torch.pi if two_fold_rotation_symmetry else 2.0 * torch.pi
    phi_bins = torch.linspace(
        0.0, phi_range, num_annular_bins + 1, dtype=torch.float32, device=device
    )[:-1]
    phi_grid, r_grid = torch.meshgrid(phi_bins, radial_bins, indexing="ij")
    if ellipse_params is None:
        x = r_grid * torch.cos(phi_grid)
        y = r_grid * torch.sin(phi_grid)
    else:
        if len(ellipse_params) != 3:
            raise ValueError("ellipse_params must be (a, b, theta_deg).")
        a, b, theta_deg = ellipse_params
        theta = torch.deg2rad(torch.tensor(theta_deg, dtype=torch.float32, device=device))
        alpha = phi_grid - theta
        u = (a / b) * r_grid * torch.cos(alpha)
        v_prime = r_grid * torch.sin(alpha)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        x = u * cos_t - v_prime * sin_t
        y = u * sin_t + v_prime * cos_t
    coords_y = y + origin_row
    coords_x = x + origin_col
    grid = _normalize_coords_for_grid_sample(coords_y, coords_x, ny, nx)
    grid = grid.unsqueeze(0)  # (1, n_phi, n_r, 2)
    return grid, phi_bins, radial_bins, radial_max_eff


def auto_origin_id(
    data: "Dataset4dstem",
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
    Automatic diffraction center finding by minimizing the standard deviation
    along the annular direction in the polar transform.

    For each scan position, this routine:
    1) Computes a polar transform at an initial origin (image center, or
       warm-started from the previous scan position).
    2) Evaluates the sum of the standard deviation across angle (phi) over
       a mid-radius band.
    3) Performs a local search over neighboring pixel origins until the
       objective no longer improves.

    Parameters
    ----------
    data : Dataset4dstem
        A 4D-STEM dataset (or 2D diffraction pattern wrapped as 4D).
    ellipse_params : tuple or None
        Ellipse parameters (a, b, theta_deg) for distortion correction.
    num_annular_bins : int
        Number of angular bins for polar transform.
    radial_min : float
        Minimum radius in pixels.
    radial_max : float or None
        Maximum radius in pixels.
    radial_step : float
        Radial step size in pixels.
    two_fold_rotation_symmetry : bool
        If True, use only 0 to pi range for angles.
    device : str
        Torch device for computation ("cpu", "cuda", "cuda:0", etc.).

    Returns
    -------
    origin_array : np.ndarray
        Array of shape (scan_y, scan_x, 2) containing (row, col) origin
        estimates in pixels.
    """
    if len(data.array.shape) == 2:
        ny, nx = data.array.shape
        scan_y, scan_x = 1, 1
    elif len(data.array.shape) == 4:
        scan_y, scan_x, ny, nx = data.array.shape
    else:
        raise ValueError("auto_origin_id only supports 2D or 4D-STEM datasets.")

    origin_array = np.zeros((scan_y, scan_x, 2), dtype=float)
    max_steps = 1000
    # start with center but subsequent positions warm-start from the previous result
    estimated_origin_row = (ny - 1) / 2.0
    estimated_origin_col = (nx - 1) / 2.0
    for y_pos in range(scan_y):
        for x_pos in range(scan_x):
            test_origin = np.array([estimated_origin_row, estimated_origin_col], dtype=float)
            coords_cache: dict[tuple[int, int], float] = {}
            polar = data.polar_transform(
                origin_array=test_origin,
                ellipse_params=ellipse_params,
                num_annular_bins=num_annular_bins,
                radial_min=radial_min,
                radial_max=radial_max,
                radial_step=radial_step,
                two_fold_rotation_symmetry=two_fold_rotation_symmetry,
                scan_pos=(y_pos, x_pos),
                device=device,
            )
            min_r = int(np.floor(0.1 * polar.shape[1]))
            max_r = int(np.ceil(0.9 * polar.shape[1]))
            std_est_origin = polar[:, min_r:max_r].std(dim=0)
            std_est_origin_sum = std_est_origin.sum()
            origin_row = int(round(estimated_origin_row))
            origin_col = int(round(estimated_origin_col))
            coords_cache[(origin_row, origin_col)] = std_est_origin_sum

            converged = False
            best = std_est_origin_sum
            steps = 0
            while not converged and steps < max_steps:
                steps += 1
                moved = False
                neighbors = [
                    (origin_row + dr, origin_col + dc)
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if not (dr == 0 and dc == 0)
                ]
                neighbors = [(r, c) for (r, c) in neighbors if 0 <= r < ny and 0 <= c < nx]
                for origin_r, origin_c in neighbors:
                    if (origin_r, origin_c) not in coords_cache:
                        test_origin = np.array([origin_r, origin_c], dtype=float)
                        polar = data.polar_transform(
                            origin_array=test_origin,
                            ellipse_params=ellipse_params,
                            num_annular_bins=num_annular_bins,
                            radial_min=radial_min,
                            radial_max=radial_max,
                            radial_step=radial_step,
                            two_fold_rotation_symmetry=two_fold_rotation_symmetry,
                            scan_pos=(y_pos, x_pos),
                            device=device,
                        )
                        std_test = polar[:, min_r:max_r].std(dim=0)
                        coords_cache[(origin_r, origin_c)] = std_test.sum()
                    if coords_cache[(origin_r, origin_c)] < best:
                        origin_row = origin_r
                        origin_col = origin_c
                        best = coords_cache[(origin_r, origin_c)]
                        moved = True
                if not moved:
                    converged = True

            origin_array[y_pos, x_pos, 0] = origin_row
            origin_array[y_pos, x_pos, 1] = origin_col
            # start next scan position from this result
            estimated_origin_row = float(origin_row)
            estimated_origin_col = float(origin_col)

    return origin_array


def dataset4dstem_polar_transform(
    self: "Dataset4dstem",
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
    if self.array.ndim != 4:
        raise ValueError("polar_transform requires a 4D-STEM dataset (ndim=4).")
    scan_y, scan_x, ny, nx = self.array.shape

    # Standardize origin_array input
    if isinstance(origin_array, torch.Tensor):
        origin_array = _to_numpy(origin_array)
    origin_array = np.asarray(origin_array) if origin_array is not None else None
    if origin_array is None:
        center = np.array([(ny - 1) / 2.0, (nx - 1) / 2.0], dtype=float)
        origins = np.broadcast_to(center, (scan_y, scan_x, 2)).copy()
    elif origin_array.shape == (2,):
        origins = np.empty((scan_y, scan_x, 2), dtype=float)
        origins[...] = origin_array
    elif origin_array.shape == (scan_y, scan_x, 2):
        origins = origin_array
    else:
        raise ValueError(
            "origin_array must have shape None, (2,) or (scan_y, scan_x, 2)."
            f" Got {origin_array.shape}."
        )

    # If scan_pos is provided, compute polar transform only for that position
    if scan_pos is not None:
        iy, ix = scan_pos
        dp = torch.from_numpy(self.array[iy, ix].astype(np.float32)).to(device)
        r0 = float(origins[iy, ix, 0])
        c0 = float(origins[iy, ix, 1])
        grid, phi_bins, radial_bins, radial_max_eff = _precompute_polar_coords(
            ny=ny,
            nx=nx,
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
        dp_batch = dp.unsqueeze(0).unsqueeze(0)  # (1, 1, ny, nx)
        polar2d = F.grid_sample(
            dp_batch,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return polar2d.squeeze(0).squeeze(0)  # (n_phi, n_r)

    # Compute radial_max from all origins if not provided
    if radial_max is None:
        r_row_pos = origins[:, :, 0]
        r_row_neg = (ny - 1) - origins[:, :, 0]
        r_col_pos = origins[:, :, 1]
        r_col_neg = (nx - 1) - origins[:, :, 1]
        radial_max_eff_array = np.minimum.reduce([r_row_pos, r_row_neg, r_col_pos, r_col_neg])
        radial_max = float(max(radial_max_eff_array.min(), radial_min + radial_step))

    # Compute grid for first position to get output shape
    grid, phi_bins, radial_bins, radial_max_eff = _precompute_polar_coords(
        ny=ny,
        nx=nx,
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
    result_dtype = np.result_type(self.array.dtype, np.float32)
    out = np.empty((scan_y, scan_x, n_phi, n_r), dtype=result_dtype)
    for iy in range(scan_y):
        for ix in range(scan_x):
            dp = torch.from_numpy(self.array[iy, ix].astype(np.float32)).to(device)
            r0 = float(origins[iy, ix, 0])
            c0 = float(origins[iy, ix, 1])
            grid, _, _, _ = _precompute_polar_coords(
                ny=ny,
                nx=nx,
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
            dp_batch = dp.unsqueeze(0).unsqueeze(0)
            polar2d = F.grid_sample(
                dp_batch,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            out[iy, ix] = _to_numpy(polar2d.squeeze(0).squeeze(0))

    phi_range = np.pi if two_fold_rotation_symmetry else 2.0 * np.pi
    phi_step_deg = (phi_range / float(n_phi)) * (180.0 / np.pi)
    sampling = np.zeros(4, dtype=float)
    origin = np.zeros(4, dtype=float)
    sampling[0:2] = np.asarray(self.sampling)[0:2]
    sampling[2] = phi_step_deg
    sampling[3] = float(np.asarray(self.sampling)[-1]) * radial_step
    origin[0:2] = np.asarray(self.origin)[0:2]
    origin[2] = 0.0
    origin[3] = radial_min * float(np.asarray(self.sampling)[-1])
    units = [
        self.units[0],
        self.units[1],
        "deg",
        self.units[-1],
    ]
    metadata = dict(self.metadata)
    metadata.update(
        {
            "polar_radial_min": float(radial_min),
            "polar_radial_max": float(radial_max_eff),
            "polar_radial_step": float(radial_step),
            "polar_num_annular_bins": int(n_phi),
            "polar_two_fold_rotation_symmetry": bool(two_fold_rotation_symmetry),
            "polar_ellipse_params": tuple(ellipse_params) if ellipse_params is not None else None,
        }
    )
    return Polar4dstem(
        array=out,
        name=name if name is not None else f"{self.name}_polar",
        origin=origin,
        sampling=sampling,
        units=units,
        signal_units=signal_units if signal_units is not None else self.signal_units,
        metadata=metadata,
        origin_array=origins,
        _token=Polar4dstem._token,
    )
