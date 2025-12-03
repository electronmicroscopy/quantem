import numpy as np
from numpy.typing import NDArray
from typing import Any, TYPE_CHECKING

from quantem.core.datastructures.dataset4d import Dataset4d
# from quantem.core.datastructures.dataset4dstem import Dataset4dstem

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
        array = ensure_valid_array(array, ndim=4)
        if origin is None:
            origin = np.zeros(4)
        if sampling is None:
            sampling = np.ones(4)
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


def dataset4dstem_polar_transform(
    self: "Dataset4dstem",
    origin_row: float | NDArray,
    origin_col: float | NDArray,
    ellipse_params: tuple[float, float, float] | None = None,
    num_annular_bins: int = 180,
    radial_min: float = 0.0,
    radial_max: float | None = None,
    radial_step: float = 1.0,
    two_fold_rotation_symmetry: bool = False,
    name: str | None = None,
    signal_units: str | None = None,
) -> Polar4dstem:
    """Return a Polar4dstem with shape (scan_y, scan_x, phi, r)."""
    if self.array.ndim != 4:
        raise ValueError("polar_transform requires a 4D-STEM dataset (ndim=4).")

    scan_y, scan_x, ny, nx = self.array.shape

    mapping = _precompute_polar_mapping(
        ny=ny,
        nx=nx,
        origin_row=float(origin_row),
        origin_col=float(origin_col),
        ellipse_params=ellipse_params,
        num_annular_bins=num_annular_bins,
        radial_min=radial_min,
        radial_max=radial_max,
        radial_step=radial_step,
        two_fold_rotation_symmetry=two_fold_rotation_symmetry,
    )

    result_dtype = np.result_type(self.array.dtype, np.float32)
    out = np.empty(
        (scan_y, scan_x, mapping["n_phi"], mapping["n_r"]),
        dtype=result_dtype,
    )

    for iy in range(scan_y):
        for ix in range(scan_x):
            out[iy, ix] = _apply_polar_mapping_single(
                self.array[iy, ix],
                mapping,
                dtype=result_dtype,
            )

    phi_step_deg = mapping["phi_step"] * 180.0 / np.pi
    phi_units = "deg"
    radial_units = self.units[-1]

    sampling = np.array(
        [
            self.sampling[0],
            self.sampling[1],
            phi_step_deg,
            self.sampling[-1] * mapping["radial_step"],
        ],
        dtype=float,
    )
    origin = np.array(
        [
            self.origin[0],
            self.origin[1],
            0.0,
            self.sampling[-1] * mapping["radial_min"],
        ],
        dtype=float,
    )
    units = [
        self.units[0],
        self.units[1],
        phi_units,
        radial_units,
    ]

    metadata = dict(self.metadata)
    metadata.update(
        {
            "polar_radial_min": mapping["radial_min"],
            "polar_radial_max": mapping["radial_max"],
            "polar_radial_step": mapping["radial_step"],
            "polar_num_annular_bins": mapping["n_phi"],
            "polar_two_fold_rotation_symmetry": two_fold_rotation_symmetry,
            "polar_origin_row": float(origin_row),
            "polar_origin_col": float(origin_col),
            "polar_ellipse_params": tuple(ellipse_params)
            if ellipse_params is not None
            else None,
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


def _precompute_polar_mapping(
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
) -> dict[str, Any]:
    origin_row = float(origin_row)
    origin_col = float(origin_col)
    annular_range = np.pi if two_fold_rotation_symmetry else 2.0 * np.pi

    rows = np.arange(ny, dtype=np.float64)
    cols = np.arange(nx, dtype=np.float64)
    cc, rr = np.meshgrid(cols, rows, indexing="xy")
    x = cc - origin_col
    y = rr - origin_row

    if ellipse_params is None:
        rr_pix = np.sqrt(x * x + y * y)
        tt = np.mod(np.arctan2(y, x), annular_range)
    else:
        if len(ellipse_params) != 3:
            raise ValueError("ellipse_params must be a length-3 tuple (a, b, theta_deg).")
        a, b, theta_deg = ellipse_params
        theta = np.deg2rad(theta_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        xc = x * cos_t + y * sin_t
        yc = (y * cos_t - x * sin_t) * (a / b)
        rr_pix = (b / a) * np.hypot(xc, yc)
        tt = np.mod(np.arctan2(yc, xc) + theta, annular_range)

    if radial_step <= 0:
        raise ValueError("radial_step must be > 0.")
    radial_min = float(radial_min)

    if radial_max is None:
        radial_max_eff = float(rr_pix.max())
    else:
        radial_max_eff = float(radial_max)
    if radial_max_eff <= radial_min + radial_step:
        radial_max_eff = radial_min + radial_step

    radial_bins = np.arange(radial_min, radial_max_eff, radial_step, dtype=np.float64)
    n_r = radial_bins.size
    if n_r < 1:
        raise ValueError("No radial bins defined. Check radial_min, radial_max, and radial_step.")

    n_phi = int(num_annular_bins)
    if n_phi < 1:
        raise ValueError("num_annular_bins must be >= 1.")
    phi_step = annular_range / n_phi

    r_bin = (rr_pix - radial_min) / radial_step
    t_bin = tt / phi_step

    r0 = np.floor(r_bin).astype(np.int64)
    t0 = np.floor(t_bin).astype(np.int64)
    dr = (r_bin - r0).astype(np.float64)
    dt = (t_bin - t0).astype(np.float64)

    valid = (r0 >= 0) & (r0 < n_r - 1)
    t0 = np.clip(t0, 0, n_phi - 1)

    flat_valid = valid.ravel()
    r0v = r0.ravel()[flat_valid]
    t0v = t0.ravel()[flat_valid]
    drv = dr.ravel()[flat_valid]
    dtv = dt.ravel()[flat_valid]

    n_bins = n_phi * n_r
    idx00 = r0v + n_r * t0v
    idx01 = r0v + n_r * ((t0v + 1) % n_phi)
    idx10 = (r0v + 1) + n_r * t0v
    idx11 = (r0v + 1) + n_r * ((t0v + 1) % n_phi)

    w00 = (1.0 - drv) * (1.0 - dtv)
    w01 = (1.0 - drv) * dtv
    w10 = drv * (1.0 - dtv)
    w11 = drv * dtv

    weights_sum = np.bincount(idx00, weights=w00, minlength=n_bins)
    weights_sum += np.bincount(idx01, weights=w01, minlength=n_bins)
    weights_sum += np.bincount(idx10, weights=w10, minlength=n_bins)
    weights_sum += np.bincount(idx11, weights=w11, minlength=n_bins)
    weights_sum = weights_sum.reshape(n_phi, n_r)

    weights_inv = np.zeros_like(weights_sum, dtype=np.float64)
    mask_bins = weights_sum > 0
    weights_inv[mask_bins] = 1.0 / weights_sum[mask_bins]

    return {
        "flat_valid": flat_valid,
        "idx00": idx00,
        "idx01": idx01,
        "idx10": idx10,
        "idx11": idx11,
        "w00": w00,
        "w01": w01,
        "w10": w10,
        "w11": w11,
        "weights_inv": weights_inv,
        "n_phi": n_phi,
        "n_r": n_r,
        "radial_bins": radial_bins,
        "phi_step": phi_step,
        "annular_range": annular_range,
        "radial_min": radial_min,
        "radial_max": radial_min + radial_step * n_r,
        "radial_step": radial_step,
    }


def _apply_polar_mapping_single(
    image: NDArray,
    mapping: dict[str, Any],
    dtype: Any,
) -> NDArray:
    data = np.asarray(image, dtype=np.float64)
    flat = data.ravel()[mapping["flat_valid"]]
    n_bins = mapping["n_phi"] * mapping["n_r"]

    acc = np.bincount(mapping["idx00"], weights=flat * mapping["w00"], minlength=n_bins)
    acc += np.bincount(mapping["idx01"], weights=flat * mapping["w01"], minlength=n_bins)
    acc += np.bincount(mapping["idx10"], weights=flat * mapping["w10"], minlength=n_bins)
    acc += np.bincount(mapping["idx11"], weights=flat * mapping["w11"], minlength=n_bins)

    acc = acc.reshape(mapping["n_phi"], mapping["n_r"])
    acc *= mapping["weights_inv"]
    return acc.astype(dtype, copy=False)

