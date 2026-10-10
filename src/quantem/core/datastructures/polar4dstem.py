from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from quantem.core.datastructures.dataset4d import Dataset4d

if TYPE_CHECKING:
    from .dataset4dstem import Dataset4dstem


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
            "polar_origin_row",
            "polar_origin_col",
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
        """
        Create a Polar4dstem from a 4D array shaped (scan_y, scan_x, phi, r).

        Parameters
        ----------
        array : NDArray
            Polar data, shape (scan_y, scan_x, n_phi, n_r).
        name : str, optional
            Dataset name, by default "Polar 4D-STEM dataset".
        origin : NDArray | tuple | list | float | int, optional
            Origin of each axis in calibrated units, by default zeros.
        sampling : NDArray | tuple | list | float | int, optional
            Sampling of each axis, by default ones.
        units : list[str] | tuple | list, optional
            Units of each axis, by default ["pixels", "pixels", "deg", "pixels"].
        signal_units : str, optional
            Units of the array values, by default "arb. units".
        metadata : dict, optional
            Metadata. Missing "polar_*" keys are set to None.

        Returns
        -------
        Polar4dstem
        """
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
        """Number of azimuthal bins (axis 2)."""
        return int(self.shape[2])

    @property
    def n_r(self) -> int:
        """Number of radial bins (axis 3)."""
        return int(self.shape[3])


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
) -> tuple[NDArray, NDArray, NDArray, float]:
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
    radial_bins = np.arange(radial_min, radial_max_eff, radial_step, dtype=np.float64)
    if radial_bins.size == 0:
        radial_bins = np.array([radial_min], dtype=np.float64)
    if two_fold_rotation_symmetry:
        phi_range = np.pi
    else:
        phi_range = 2.0 * np.pi
    phi_bins = np.linspace(0.0, phi_range, num_annular_bins, endpoint=False, dtype=np.float64)
    phi_grid, r_grid = np.meshgrid(phi_bins, radial_bins, indexing="ij")
    if ellipse_params is None:
        x = r_grid * np.cos(phi_grid)
        y = r_grid * np.sin(phi_grid)
    else:
        if len(ellipse_params) != 3:
            raise ValueError("ellipse_params must be (a, b, theta_deg).")
        a, b, theta_deg = ellipse_params
        theta = np.deg2rad(theta_deg)
        alpha = phi_grid - theta
        u = (a / b) * r_grid * np.cos(alpha)
        v_prime = r_grid * np.sin(alpha)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x = u * cos_t - v_prime * sin_t
        y = u * sin_t + v_prime * cos_t
    coords_y = y + origin_row
    coords_x = x + origin_col
    coords = np.stack((coords_y, coords_x), axis=0)
    return coords, phi_bins, radial_bins, radial_max_eff


def dataset4dstem_polar_transform(
    self: "Dataset4dstem",
    origin_row: float,
    origin_col: float,
    ellipse_params: tuple[float, float, float] | None = None,
    num_annular_bins: int = 180,
    radial_min: float = 0.0,
    radial_max: float | None = None,
    radial_step: float = 1.0,
    two_fold_rotation_symmetry: bool = False,
    name: str | None = None,
    signal_units: str | None = None,
) -> Polar4dstem:
    """
    Resample every diffraction pattern onto a polar (phi, r) grid.

    Bound to `Dataset4dstem.polar_transform`. Uses bilinear interpolation
    (`scipy.ndimage.map_coordinates`, order 1); samples outside the detector
    are set to 0.

    Parameters
    ----------
    origin_row, origin_col : float
        Center of the polar grid on the detector, in detector pixels. The same
        center is used for all scan positions.
    ellipse_params : tuple[float, float, float], optional
        Elliptical distortion (a, b, theta_deg): the radius along the
        direction theta_deg (degrees) is scaled by a / b. None for circular
        sampling.
    num_annular_bins : int, optional
        Number of azimuthal bins, by default 180.
    radial_min : float, optional
        First radial bin in detector pixels, by default 0.
    radial_max : float, optional
        Radial upper limit (exclusive) in detector pixels. If None, the
        distance from the origin to the nearest detector edge.
    radial_step : float, optional
        Radial bin width in detector pixels, by default 1.
    two_fold_rotation_symmetry : bool, optional
        If True, phi covers [0, 180) degrees instead of [0, 360).
    name : str, optional
        Name of the output, by default "<name>_polar".
    signal_units : str, optional
        Signal units of the output, by default those of this dataset.

    Returns
    -------
    Polar4dstem
        Array shaped (scan_y, scan_x, n_phi, n_r). Phi is in degrees, 0 along
        +column and increasing toward +row. The radial axis uses the sampling
        and units of the last detector axis. The polar parameters are stored
        in metadata under "polar_*" keys.

    Notes
    -----
    Tensor-backed datasets are copied to a CPU numpy array first.
    """
    array = self.numpy()
    if array.ndim != 4:
        raise ValueError("polar_transform requires a 4D-STEM dataset (ndim=4).")
    scan_y, scan_x, ny, nx = array.shape
    origin_row_f = float(origin_row)
    origin_col_f = float(origin_col)
    coords, phi_bins, radial_bins, radial_max_eff = _precompute_polar_coords(
        ny=ny,
        nx=nx,
        origin_row=origin_row_f,
        origin_col=origin_col_f,
        ellipse_params=ellipse_params,
        num_annular_bins=num_annular_bins,
        radial_min=radial_min,
        radial_max=radial_max,
        radial_step=radial_step,
        two_fold_rotation_symmetry=two_fold_rotation_symmetry,
    )
    n_phi = phi_bins.size
    n_r = radial_bins.size
    result_dtype = np.result_type(array.dtype, np.float32)
    out = np.empty((scan_y, scan_x, n_phi, n_r), dtype=result_dtype)
    for iy in range(scan_y):
        for ix in range(scan_x):
            dp = array[iy, ix]
            out[iy, ix] = map_coordinates(
                dp,
                coords,
                order=1,
                mode="constant",
                cval=0.0,
            )
    if two_fold_rotation_symmetry:
        phi_range = np.pi
    else:
        phi_range = 2.0 * np.pi
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
            "polar_origin_row": origin_row_f,
            "polar_origin_col": origin_col_f,
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
        _token=Polar4dstem._token,
    )
