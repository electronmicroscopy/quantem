import warnings
from abc import abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Self, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from quantem.core import config
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.constraints import BaseConstraints, Constraints, parse_constraint_dict
from quantem.core.ml.optimizer_mixin import OptimizerMixin, OptimizerParams
from quantem.core.utils.utils import electron_wavelength_angstrom, tqdmnd
from quantem.core.utils.validators import (
    validate_array,
    validate_gt,
    validate_int,
    validate_tensor,
)
from quantem.core.visualization import show_2d
from quantem.diffractive_imaging.ptycho_utils import AffineTransform, fit_origin, shift_array

"""
Dataset models for ptychographic reconstruction.
"""


class PtychoDatasetConstraintParams:
    """
    Namespace class for ptychography dataset constraint dataclasses.

    Tab-complete on ``PtychoDatasetConstraintParams`` in a notebook to discover the
    available variants. Tab-complete inside a variant's constructor to see every
    constraint field with its default value.

    Variants
    --------
    Raster
        Constraints for ``PtychographyDatasetRaster`` (descan TV penalty, descan
        zero-out, scan position clipping and centering).
    """

    @dataclass
    class Raster(Constraints):
        """Constraints for raster-scan ptychography datasets (``PtychographyDatasetRaster``).

        Attributes
        ----------
        descan_shifts_zero : bool, default ``False``
            Forces all descan shifts to zero after each update. Useful when you
            want to keep the descan optimizer in the parameter group but freeze
            its effect. (Distinct from ``learn_descan=False``, which holds descan
            at its fitted value rather than zeroing it.)
        center_scan_positions : bool, default ``False``
            Shifts all scan positions uniformly so their mean sits at the object
            center after each update. Prevents the reconstruction from
            translating during long runs with ``lr_scan_positions > 0``.
        clip_scan_positions : bool, default ``True``
            Clamps scan positions to lie within ``[0, obj_shape - 1]`` after each
            update. On by default to prevent positions from drifting off the
            padded object during refinement.
        descan_tv_weight : float, default ``0.0``
            Soft penalty. Weight on the total-variation of the descan-shift
            sequence (x and y averaged). Encourages smoothly varying descan, but
            only contributes when ``learn_descan`` is on and the dataset has a
            descan optimizer attached.
        """

        # hard constraints
        descan_shifts_zero: bool = False
        center_scan_positions: bool = False
        clip_scan_positions: bool = True
        # soft constraints
        descan_tv_weight: float = 0.0
        _name: str = "raster"

        soft_constraint_keys = ["descan_tv_weight"]
        hard_constraint_keys = [
            "descan_shifts_zero",
            "center_scan_positions",
            "clip_scan_positions",
        ]

    @classmethod
    def parse_dict(cls, d: dict) -> "PtychoDatasetConstraintsType":
        """Instantiate the appropriate variant from a config dict.

        The dict must contain a ``'name'`` or ``'type'`` key (case-insensitive),
        with value ``'raster'``. All other keys are forwarded as keyword
        arguments to the chosen dataclass.
        """
        return cast(PtychoDatasetConstraintsType, parse_constraint_dict(cls, d, kind="dataset"))


PtychoDatasetConstraintsType = PtychoDatasetConstraintParams.Raster


class PtychographyDatasetBase(
    AutoSerialize, OptimizerMixin, torch.nn.Module, torch.utils.data.Dataset
):
    _token = object()
    _patch_indices: torch.Tensor

    # TODO make this a PPLR so different lrs can be used for different parameters
    DEFAULT_LRS = {
        "descan": 1e-3,
        "scan_positions": 1e-3,
    }

    def __init__(
        self,
        dset: Dataset3d,
        detector_mask: torch.Tensor | np.ndarray | None = None,
        verbose: int | bool = 1,
        learn_descan: bool = True,
        learn_scan_positions: bool = True,
        _token: object | None = None,
    ):
        AutoSerialize.__init__(self)
        OptimizerMixin.__init__(self)
        torch.nn.Module.__init__(self)

        if _token is not self._token:
            raise RuntimeError("Use PtychographyDatasetRaster.from_* to instantiate this class.")

        if dset.units[-1] != "A^-1":
            if dset.units[-1] == "mrad":
                pass
            else:
                raise ValueError(f"Expected diffraction units to be 'A^-1', got {dset.units[-1]}")

        self.dset = dset
        self.verbose = verbose
        # target_residency controls where loss targets live:
        #   "device" (default) — targets are kept resident on the compute device (current behavior)
        #   "cpu"              — targets live in CPU RAM and are streamed to the device per-batch,
        #                        enabling datasets larger than a single GPU's VRAM
        self._target_residency: Literal["device", "cpu"] = "device"
        self._preprocessed = False
        self._preprocessing_params = {}  # for serialization and reloading
        self._com_rotation_rad = 0  # default
        self._transpose = False  # default

        # scan_positions_px: [num_positions, 2] in pixels
        self._scan_positions_px = nn.Parameter(
            torch.zeros((self.num_gpts, 2), dtype=getattr(torch, config.get("dtype_real"))),
            requires_grad=learn_scan_positions,
        )
        self.learn_scan_positions = learn_scan_positions

        # descan_shifts: [self.num_gpts, 2] descan shifts in pixels
        self._descan_shifts = nn.Parameter(
            torch.zeros((self.num_gpts, 2), dtype=getattr(torch, config.get("dtype_real"))),
            requires_grad=learn_descan,
        )
        self.learn_descan = learn_descan

        # Store initial values for reset
        self._initial_scan_positions_px = torch.zeros_like(self._scan_positions_px)
        self._initial_descan_shifts = torch.zeros_like(self._descan_shifts)

        # _targets is a plain attribute so that its device can be managed explicitly per
        # target_residency. Initialized empty (rather than (num_gpts, *roi_shape) buffer) to avoid
        # a redundant allocation during preprocessing
        self._targets = torch.zeros(0, dtype=getattr(torch, config.get("dtype_real")))
        self.register_buffer("_patch_indices", torch.zeros(0, dtype=torch.int32))
        self.register_buffer("_last_patch_positions_px", torch.zeros(self.num_gpts, 2))
        self.register_buffer("_detector_mask", torch.ones(*self.roi_shape))
        self.positions_mask = torch.ones(self.num_gpts, dtype=torch.bool)
        self.detector_mask = detector_mask
        self._constraints = {}
        self._probe_energy = None
        # Set by the ptychography wiring from obj_model.is_implicit. When True, forward()
        # emits continuous per-patch coordinates instead of integer patch_indices (and zeroed
        # fractional positions), so the probe is not subpixel-shifted.
        self._implicit_object = False

        # Measurement arrays. Only ``centered_amplitudes`` is materialized during normalization;
        self._centered_amplitudes: torch.Tensor | None = None
        self._amplitudes: torch.Tensor | None = None
        self._intensities: torch.Tensor | None = None
        self._centered_intensities: torch.Tensor | None = None
        self._pattern_crop_mask: np.ndarray | None = None
        self._pattern_crop_mask_shape: tuple[int, int] = (
            int(self.roi_shape[0]),
            int(self.roi_shape[1]),
        )

    def get_optimization_parameters(self) -> "dict[str, list[torch.Tensor]]":
        """Descan and scan-position parameters as separate PPLR groups.

        Returns one group per *learnable* parameter set; ``{}`` when neither is learnable
        (``set_optimizer`` then short-circuits to removing the optimizer).
        """
        groups: dict[str, list[torch.Tensor]] = {}
        if self.learn_descan:
            groups["descan"] = [self._descan_shifts]
        if self.learn_scan_positions:
            groups["scan_positions"] = [self._scan_positions_px]
        return groups

    def _normalize_optimizer_params(self, params):
        """Broadcast a single optimizer spec to the learnable descan/scan_position groups.

        A single ``OptimizerParamsType`` / single-optimizer dict (normalized to the ``"default"`` key)
        is fanned out to whichever groups are currently learnable, so the common single-LR caller
        keeps working. An explicit PPLR dict (keyed by ``descan``/``scan_positions``) passes through.
        """
        norm = super()._normalize_optimizer_params(params)
        if set(norm) == {self.DEFAULT_OPTIMIZER_KEY}:
            spec = norm[self.DEFAULT_OPTIMIZER_KEY]
            learnable = [
                key
                for key, on in (
                    ("descan", self.learn_descan),
                    ("scan_positions", self.learn_scan_positions),
                )
                if on
            ]
            if not learnable and not isinstance(spec, OptimizerParams.NoneOptimizer):
                warnings.warn(
                    f"{type(self).__name__}: an optimizer was requested but nothing is "
                    "learnable (both learn_descan and learn_scan_positions are False); "
                    "the optimizer will be removed. Enable learn_descan and/or "
                    "learn_scan_positions to optimize.",
                    stacklevel=2,
                )
            return {key: replace(spec) for key in learnable} if learnable else {}
        return norm

    def to(self, *args, **kwargs):
        """Move all relevant tensors to a different device."""
        # Call parent's to() method to handle PyTorch's internal device management
        super().to(*args, **kwargs)
        # _targets is a plain attribute, so nn.Module.to() does not move it; do so explicitly
        # unless residency is "cpu" (in which case targets intentionally stay on CPU and are
        # streamed to the device per-batch).
        if (
            getattr(self, "target_residency", "device") != "cpu"
            and getattr(self, "_targets", None) is not None
        ):
            # After super().to(), self.device reflects the new device (it reads off a Parameter).
            self._targets = self._targets.to(self.device)
        # Reconnect optimizer to parameters on the new device
        self.reconnect_optimizer_to_parameters()
        return self

    # region --- optimizable parameters ---
    @property
    def descan_shifts(self) -> nn.Parameter:
        return self._descan_shifts

    @descan_shifts.setter
    def descan_shifts(self, shifts: torch.Tensor | np.ndarray) -> None:
        shifts = validate_tensor(
            shifts,
            name="descan_shifts",
            dtype=getattr(torch, config.get("dtype_real")),
            shape=(self.num_positions, 2),
        )
        self._descan_shifts.data = shifts.to(self.device)

    @property
    def learn_descan(self) -> bool:
        return self._learn_descan

    @learn_descan.setter
    def learn_descan(self, learn_descan: bool) -> None:
        self._learn_descan = bool(learn_descan)

    @property
    def scan_positions_px(self) -> nn.Parameter:
        return self._scan_positions_px

    @scan_positions_px.setter
    def scan_positions_px(self, positions: torch.Tensor | np.ndarray) -> None:
        positions = validate_tensor(
            positions,
            name="scan_positions_px",
            dtype=getattr(torch, config.get("dtype_real")),
            shape=(self.num_positions, 2),
        )
        self._scan_positions_px.data = positions.to(self.device)

    @property
    def learn_scan_positions(self) -> bool:
        return self._learn_scan_positions

    @learn_scan_positions.setter
    def learn_scan_positions(self, learn_scan_positions: bool) -> None:
        self._learn_scan_positions = bool(learn_scan_positions)

    @property
    def positions_px_fractional(self) -> torch.Tensor:
        """fractional component of positions_px_fractional"""
        return self.scan_positions_px - torch.round(self.scan_positions_px)

    # endregion --- optimizable parameters ---

    # region --- buffers ---
    @property
    def detector_mask(self) -> torch.Tensor:
        return self._detector_mask

    @detector_mask.setter
    def detector_mask(self, mask: torch.Tensor | np.ndarray | None) -> None:
        if mask is None:
            mask = torch.ones(*self.roi_shape, dtype=getattr(torch, config.get("dtype_real")))
        else:
            mask = validate_tensor(
                mask,
                "detector_mask",
                dtype=getattr(torch, config.get("dtype_real")),
                shape=self.roi_shape,
            )
        self._detector_mask = mask.to(self.device)

    @property
    def initial_descan_shifts(self) -> torch.Tensor:
        """Initial descan shifts, used for resetting the dataset"""
        return self._initial_descan_shifts

    @initial_descan_shifts.setter
    def initial_descan_shifts(self, shifts: torch.Tensor | np.ndarray) -> None:
        shifts = validate_tensor(
            shifts,
            name="initial_descan_shifts",
            dtype=getattr(torch, config.get("dtype_real")),
            shape=(self.num_positions, 2),
        )
        self._initial_descan_shifts = shifts

    @property
    def initial_scan_positions_px(self) -> torch.Tensor:
        """Initial scan positions in pixels, used for resetting the dataset"""
        return self._initial_scan_positions_px

    @initial_scan_positions_px.setter
    def initial_scan_positions_px(self, positions: torch.Tensor | np.ndarray) -> None:
        positions = validate_tensor(
            positions,
            name="initial_scan_positions_px",
            dtype=getattr(torch, config.get("dtype_real")),
            shape=(self.num_positions, 2),
        )
        self._initial_scan_positions_px = positions

    @property
    def targets(self) -> torch.Tensor:
        if not self._preprocessed:
            raise ValueError("dset must be preprocessed before targets can be accessed")
        return self._targets

    @property
    def target_residency(self) -> Literal["device", "cpu"]:
        """Where the loss targets live: ``"device"`` (resident, fastest) or
        ``"cpu"`` (streamed per-batch, enables datasets larger than VRAM)."""
        return self._target_residency

    @target_residency.setter
    def target_residency(self, value: str) -> None:
        if value not in ("device", "cpu"):
            raise ValueError(f"target_residency must be 'device' or 'cpu', got {value!r}")
        self._target_residency = value

    def _set_targets(self, target_space: Literal["amplitude", "intensity"]):
        """Build the per-position targets in the given measurement space.

        ``target_space`` is the space the data-fidelity criterion compares in (set from the
        criterion's ``target_space``); the comparison itself lives in the criterion, not here.
        """
        # When residency is "cpu", build targets on CPU so they can be streamed per-batch
        # (and read by DataLoader workers); otherwise keep them resident on the compute device.
        target_device = "cpu" if self.target_residency == "cpu" else self.device
        learn_descan = self.learn_descan and self.has_optimizer()
        if target_space == "amplitude":
            source = self.amplitudes if learn_descan else self.centered_amplitudes
        elif target_space == "intensity":
            source = self.intensities if learn_descan else self.centered_intensities
        else:
            raise ValueError(
                f"target_space must be 'amplitude' or 'intensity', got {target_space!r}"
            )
        self._targets = source.to(target_device)

    @property
    def patch_indices(self) -> torch.Tensor:
        return self._patch_indices

    # endregion --- buffers ---

    # region --- torch.utils.data.Dataset interface ---
    def __len__(self) -> int:
        return self.num_positions

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one sample for the DataLoader.

        The target is returned as-is on whatever device target_residency dictates (CPU when
        residency is "cpu" so DataLoader workers can read it). The integer index is collated
        into a LongTensor by the default collate_fn.
        """
        return {"index": idx, "target": self._targets[idx]}

    # endregion --- torch.utils.data.Dataset interface ---

    # region --- explicit properties (have setters) ---
    @property
    def dset(self) -> Dataset3d:
        return self._dset

    @dset.setter
    def dset(self, new_dset: Dataset3d):
        if not isinstance(new_dset, Dataset3d):
            raise TypeError(f"dset should be a Dataset3d, got {type(new_dset)}")
        self._dset = new_dset

    @property
    def centered_amplitudes(self) -> torch.Tensor:
        """gives the amplitudes that have had descan corrected and which are centered in the fov
        shaped as (rr*rc, qx, qy)
        """
        if self._centered_amplitudes is None:
            raise ValueError("centered_amplitudes is unset; preprocess the dataset first")
        return self._centered_amplitudes

    @centered_amplitudes.setter
    def centered_amplitudes(self, arr: "np.ndarray | torch.Tensor") -> None:
        arr = validate_tensor(
            arr,
            name="centered_amplitudes",
            dtype=getattr(torch, config.get("dtype_real")),
            ndim=3,
            shape=(self.num_positions, *self.roi_shape),
        )
        self._centered_amplitudes = arr

    def _recompute_raw_measurement(self, amplitude: bool) -> torch.Tensor:
        """
        Recompute the raw (un-centered) masked ``amplitudes``/``intensities`` from
        ``intensities_4d`` on demand. Values are identical to what
        ``_normalize_diffraction_intensities`` used to store; kept lazy so the full-size arrays are
        not held resident.

        Typically will be called only when changing the targets (using ``set_targets``).
        """
        dtype = config.get("dtype_real")
        mask = self.positions_mask.detach().cpu().numpy().ravel()
        roi = tuple(int(x) for x in self.roi_shape)
        arr = np.maximum(
            np.asarray(self.intensities_4d).reshape(self.num_gpts, *roi).astype(dtype), 0
        )
        if amplitude:
            arr = np.sqrt(arr)
        arr = arr[mask]
        if self._pattern_crop_mask is not None:
            arr = arr[:, self._pattern_crop_mask].reshape((-1, *self._pattern_crop_mask_shape))
        return torch.as_tensor(np.ascontiguousarray(arr), dtype=getattr(torch, dtype))

    @property
    def amplitudes(self) -> torch.Tensor:
        """raw intensities converted to amplitudes, as a torch tensor"""
        if self._amplitudes is not None:
            return self._amplitudes
        return self._recompute_raw_measurement(amplitude=True)

    @amplitudes.setter
    def amplitudes(self, arr: "np.ndarray | torch.Tensor") -> None:
        arr = validate_tensor(
            arr,
            name="amplitudes",
            dtype=getattr(torch, config.get("dtype_real")),
            ndim=3,
            shape=(self.num_positions, *self.roi_shape),
        )
        self._amplitudes = arr

    @property
    def centered_intensities(self) -> torch.Tensor:
        """intensities that have had descan corrected and which are centered in the fov
        shaped as (rr*rc, qx, qy)
        """
        if self._centered_intensities is not None:
            return self._centered_intensities
        return self.centered_amplitudes**2

    @centered_intensities.setter
    def centered_intensities(self, arr: "np.ndarray | torch.Tensor") -> None:
        arr = validate_tensor(
            arr,
            name="centered_intensities",
            dtype=getattr(torch, config.get("dtype_real")),
            ndim=3,
            shape=(self.num_positions, *self.roi_shape),
        )
        self._centered_intensities = arr

    @property
    def intensities(self) -> torch.Tensor:
        """raw intensities as a torch tensor"""
        if self._intensities is not None:
            return self._intensities
        return self._recompute_raw_measurement(amplitude=False)

    @intensities.setter
    def intensities(self, arr: "np.ndarray | torch.Tensor") -> None:
        arr = validate_tensor(
            arr,
            name="intensities",
            dtype=getattr(torch, config.get("dtype_real")),
            ndim=3,
            shape=(self.num_positions, *self.roi_shape),
        )
        self._intensities = arr

    @property
    def positions_mask(self) -> torch.Tensor:
        """1D boolean mask (length ``num_gpts``) of scan positions kept in the reconstruction.
        Initialized to all-True. Masked-out (False) positions are dropped from the targets,
        scan positions, descan shifts, and patch indices during preprocessing.
        """
        return self._positions_mask

    @positions_mask.setter
    def positions_mask(self, mask: torch.Tensor | np.ndarray) -> None:
        # accept a 2D real-space mask (gpts) or any array-like; flatten to the raster order
        mask = validate_tensor(mask, name="positions_mask", dtype=torch.bool).reshape(-1)
        if mask.shape[0] != self.num_gpts:
            raise ValueError(
                f"positions_mask must have {self.num_gpts} elements, got {mask.shape[0]}"
            )
        self._positions_mask = mask

    @property
    def verbose(self) -> int:
        return self._verbose

    @verbose.setter
    def verbose(self, v: bool | int | float) -> None:
        self._verbose = validate_int(validate_gt(v, -1, "verbose"), "verbose")

    @property
    def mean_diffraction_intensity(self) -> float:
        return self._mean_diffraction_intensity

    @mean_diffraction_intensity.setter
    def mean_diffraction_intensity(self, value: float) -> None:
        n = "mean_diffraction_intensity"
        self._mean_diffraction_intensity = validate_gt(value, 0, n)

    @property
    def com_transpose(self) -> bool:
        "whether or not the dset has been transposed"
        return self._transpose

    @com_transpose.setter
    def com_transpose(self, t: bool) -> None:
        self._transpose = bool(t)

    @property
    def com_rotation_rad(self) -> float:
        "Best fit rotation of the dc"
        return self._com_rotation_rad

    @com_rotation_rad.setter
    def com_rotation_rad(self, rot: float) -> None:
        self._com_rotation_rad = float(rot)

    @property
    def diffraction_padding(self) -> np.ndarray:
        return self._diffraction_padding

    @diffraction_padding.setter
    def diffraction_padding(self, padding: np.ndarray | tuple | list):
        pad = validate_array(padding, shape=(2,), dtype=int, name="diffraction_padding")
        self._diffraction_padding = pad

    @property
    def probe_energy(self) -> float | None:
        """Probe energy in eV, if known"""
        return self._probe_energy

    @probe_energy.setter
    def probe_energy(self, energy: float | None) -> None:
        if energy is None:
            self._probe_energy = None
        else:
            self._probe_energy = validate_gt(energy, 0, "probe_energy")

    # endregion --- explicit properties (have setters) ---

    # region --- implicit properties (no setters) ---
    @property
    def device(self) -> torch.device:
        return self.descan_shifts.device

    @property
    def preprocessed(self) -> bool:
        return self._preprocessed

    @property
    def implicit_object(self) -> bool:
        """Whether the paired object model is an implicit (coordinate-queried) representation.

        Set by the ptychography wiring from ``obj_model.is_implicit``. When True, ``forward``
        emits continuous per-patch coordinates (instead of integer ``patch_indices``) along with
        zeroed fractional positions, so the probe is not subpixel-shifted.
        """
        return self._implicit_object

    @implicit_object.setter
    def implicit_object(self, val: bool) -> None:
        self._implicit_object = bool(val)

    @property
    def shape(self) -> np.ndarray:
        return np.array(self.dset.shape)

    @property
    def roi_shape(self) -> np.ndarray:
        return np.array(self.dset.shape[-2:])

    @property
    def num_gpts(self) -> int:
        return int(self.dset.shape[0])

    @property
    def num_positions(self) -> int:
        """Number of active scan positions, i.e. ``positions_mask.sum()``.

        Equals ``num_gpts`` until a sparser ``positions_mask`` is applied. This is the length
        of ``targets``/``scan_positions_px``/``descan_shifts`` and the index range the
        reconstruction loops (DataLoader, train/val split) operate over.
        """
        return int(self.positions_mask.sum())

    @property
    def detector_sampling(self) -> np.ndarray:
        """Detector sampling in reciprocal space. Units of A^-1"""
        return self.dset.sampling[-2:]

    @property
    def detector_units(self) -> list[str]:
        """Detector units in reciprocal space"""
        return self.dset.units[-2:]

    @property
    def obj_sampling(self) -> np.ndarray:
        """Realspace sampling of the reconstruction. Units of A"""
        return 1 / (self.roi_shape * self.reciprocal_sampling)

    @property
    def reciprocal_sampling(self) -> np.ndarray:
        """
        Units A^-1 or raises error
        """
        sampling = self.detector_sampling
        units = self.detector_units
        if units[0] == "A^-1":
            pass
        elif units[0] == "mrad":
            if self.probe_energy is None:
                raise ValueError(
                    "dset Q units given in mrad but no probe energy defined to "
                    + "convert to A^-1. Please set probe_energy in preprocess() or convert to A^-1"
                )
            sampling = sampling / electron_wavelength_angstrom(self.probe_energy) / 1e3
        elif units[0] == "pixels":
            raise ValueError("dset Q units given in pixels, needs calibration")
        else:
            raise NotImplementedError(f"Unknown dset Q units: {units}")
        return sampling

    @property
    def reciprocal_units(self) -> list[str]:
        """Hardcoded to A^-1, self.reciprocal_sampling will raise an error if can't get A^-1"""
        return ["A^-1", "A^-1"]

    @property
    def _obj_shape_crop_2d(self) -> np.ndarray:
        """Allocated/expected object crop shape, in the object array's own axis order."""
        shp = np.floor(self.fov / self.obj_sampling)
        shp += shp % 2
        shp = shp.astype("int")
        if self.com_transpose:  # match the axis-flipped scan positions
            shp = shp[::-1]
        return shp

    @property
    def _obj_shape_rot_2d(self) -> np.ndarray:
        cshape = self._obj_shape_crop_2d.copy()
        rotshape = np.floor(
            [
                abs(cshape[-1] * np.sin(self.com_rotation_rad))
                + abs(cshape[-2] * np.cos(self.com_rotation_rad)),
                abs(cshape[-2] * np.sin(self.com_rotation_rad))
                + abs(cshape[-1] * np.cos(self.com_rotation_rad)),
            ]
        )
        rotshape += rotshape % 2
        return rotshape.astype("int")

    def _obj_shape_full_2d(self, obj_padding_px: np.ndarray | tuple) -> np.ndarray:
        rshape = self._obj_shape_rot_2d.copy()
        p = 2 * np.array(obj_padding_px)
        return (rshape + p).astype("int")

    # endregion --- implicit properties (no setters) ---

    # region --- abstract class methods ---
    @abstractmethod
    def forward(
        self,
        batch_indices: np.ndarray | torch.Tensor,
        obj_padding_px: np.ndarray | tuple,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Forward pass to compute the diffraction intensities from the object and scan positions."""
        # return patch_indices, positions_px, positions_px_fractional
        # positions_px and fractional can just return
        # check if need to set patch_indices again, or if can just return them
        #
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def preprocess(self, *args, **kwargs) -> None:
        """Preprocess the dataset."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def _set_initial_scan_positions_px(
        self, obj_padding_px: np.ndarray | tuple, positions_mask: np.ndarray | None
    ) -> None:
        """Set the scan positions in pixels based on the object shape. This will depend on scan
        type, e.g. raster scan, spiral scan, etc. and is therefore implemented in subclasses"""
        raise NotImplementedError("This method should be implemented in subclasses.")

    # endregion --- abstract class methods ---

    # region --- class methods ---
    def vprint(self, m: Any, level: int = 1, *args, **kwargs) -> None:
        """Print messages if verbose is enabled."""
        if self.verbose >= level:
            print(m, *args, **kwargs)

    def _set_patch_indices(self, obj_padding_px: np.ndarray | tuple) -> None:
        """Set the _patch_indices based on self.scan_positions_px"""
        obj_shape = self._obj_shape_full_2d(obj_padding_px)
        r0 = torch.round(self.scan_positions_px[:, 0]).type(torch.int32)
        c0 = torch.round(self.scan_positions_px[:, 1]).type(torch.int32)

        x_ind = torch.fft.fftfreq(self.roi_shape[0], d=1 / self.roi_shape[0]).to(self.device)
        y_ind = torch.fft.fftfreq(self.roi_shape[1], d=1 / self.roi_shape[1]).to(self.device)

        # Process positions in chunks to reduce memory usage
        chunk_size = min(1000, len(r0))
        patch_indices = torch.empty(
            (len(r0), int(self.roi_shape[0]), int(self.roi_shape[1])),
            dtype=torch.int32,
            device=r0.device,
        )

        for i in range(0, len(r0), chunk_size):
            end_idx = min(i + chunk_size, len(r0))
            r0_chunk = r0[i:end_idx]
            c0_chunk = c0[i:end_idx]

            row_chunk = (r0_chunk[:, None, None] + x_ind[None, :, None]) % obj_shape[-2]
            col_chunk = (c0_chunk[:, None, None] + y_ind[None, None, :]) % obj_shape[-1]

            patch_indices[i:end_idx] = (row_chunk * obj_shape[-1] + col_chunk).type(torch.int32)

        self._patch_indices = patch_indices
        self._last_patch_positions_px = self.scan_positions_px.detach().clone()

    def patch_indices_need_update(self) -> bool:
        """
        Returns True if scan_positions_px has changed enough to require updating patch indices.
        """
        old_pos = torch.round(self._last_patch_positions_px)
        new_pos = torch.round(self.scan_positions_px)
        return not torch.equal(old_pos, new_pos)

    def _scan_coords(
        self, batch_indices: np.ndarray | torch.Tensor, obj_padding_px: np.ndarray | tuple
    ) -> torch.Tensor:
        """Continuous normalized ``(row, col)`` patch coordinates for implicit objects.

        For each scan position (NOT rounded) we add the same integer ROI offsets used by
        ``_set_patch_indices``, then normalize to ``[-1, 1]`` over the padded object extent
        via ``idx / (N - 1) * 2 - 1`` (matching ``torch.linspace(-1, 1, N)``). Unlike
        ``_set_patch_indices`` there is no modulo wrap: samples that fall outside the object
        map outside ``[-1, 1]``, and the implicit object treats them as vacuum.

        Because the un-rounded (fractional) position is baked into the coordinates here, the
        probe is queried at zero fractional shift in this case.

        Returns
        -------
        torch.Tensor
            ``(batch, Hroi, Wroi, 2)`` normalized ``(row, col)`` coordinates.
        """
        obj_shape = self._obj_shape_full_2d(obj_padding_px)
        positions = self.scan_positions_px[batch_indices]  # (batch, 2), un-rounded
        hroi, wroi = int(self.roi_shape[0]), int(self.roi_shape[1])
        x_ind = torch.fft.fftfreq(hroi, d=1 / hroi).to(self.device)
        y_ind = torch.fft.fftfreq(wroi, d=1 / wroi).to(self.device)
        rows = positions[:, 0][:, None, None] + x_ind[None, :, None]  # (batch, Hroi, 1)
        cols = positions[:, 1][:, None, None] + y_ind[None, None, :]  # (batch, 1, Wroi)
        rows = rows.expand(-1, -1, wroi)
        cols = cols.expand(-1, hroi, -1)
        rows_n = rows / float(obj_shape[-2] - 1) * 2.0 - 1.0
        cols_n = cols / float(obj_shape[-1] - 1) * 2.0 - 1.0
        return torch.stack([rows_n, cols_n], dim=-1)  # (batch, Hroi, Wroi, 2)

    def reset(self) -> None:
        self.descan_shifts = self.initial_descan_shifts.clone().to(self.device)
        self.scan_positions_px = self.initial_scan_positions_px.clone().to(self.device)

    # endregion --- class methods ---


class DatasetConstraints(
    BaseConstraints[PtychoDatasetConstraintParams.Raster], PtychographyDatasetBase
):
    DEFAULT_CONSTRAINTS: PtychoDatasetConstraintParams.Raster = (
        PtychoDatasetConstraintParams.Raster()
    )

    def apply_soft_constraints(self, descan_shifts: torch.Tensor) -> torch.Tensor:
        self.reset_soft_constraint_losses()
        loss = self._get_zero_loss_tensor()

        if self.constraints.descan_tv_weight > 0 and self.learn_descan and self.has_optimizer():
            tv_loss = self.get_descan_tv_loss(descan_shifts, self.constraints.descan_tv_weight)
            loss = loss + tv_loss
            self.add_soft_constraint_loss("descan_tv_weight", tv_loss)

        self.accumulate_constraint_losses()
        return loss

    def get_descan_tv_loss(self, descan_shifts: torch.Tensor, weight: float = 0.0) -> torch.Tensor:
        loss = torch.tensor(0, device=self.device, dtype=getattr(torch, config.get("dtype_real")))
        if weight == 0:
            return loss

        x_loss = torch.mean(torch.abs(descan_shifts[:, 0].diff()))
        y_loss = torch.mean(torch.abs(descan_shifts[:, 1].diff()))
        return weight * (x_loss + y_loss) / 2

    def apply_descan_constraints(
        self,
        descan: torch.Tensor,
    ) -> torch.Tensor:
        if self.constraints.descan_shifts_zero:
            descan = torch.zeros_like(descan)
        return descan

    def apply_hard_constraints(self, obj_padding_px: np.ndarray | tuple) -> None:
        positions = self.scan_positions_px
        obj_shape = torch.tensor(self._obj_shape_full_2d(obj_padding_px), device=positions.device)
        if self.constraints.clip_scan_positions:
            positions = torch.clamp(positions, min=torch.zeros_like(obj_shape), max=obj_shape - 1)

        if self.constraints.center_scan_positions:
            positions = positions - positions.mean(dim=0, keepdim=True)
            positions = positions + obj_shape / 2

        self.scan_positions_px = positions


class PtychographyDatasetRaster(DatasetConstraints):
    """
    Currently calling this DatasetRaster because it only handles 4DSTEM datasets.
    This top-level class has methods for: forward, preprocess, and _set_initial_scan_positions_px
    along with classmethods for creating from 4DSTEM datasets and files.

    As to whether this should be expanded vs making other top level classes, I see it coming
    down to preprocessing, and how much is shared between the different dataset types. I don't
    know what the usecases will be, and i'm happy for all this to be heavily refactored. -ARCM
    """

    def __init__(
        self,
        dset: Dataset4dstem,
        detector_mask: torch.Tensor | np.ndarray | None = None,
        verbose: int | bool = 1,
        learn_descan: bool = True,
        learn_scan_positions: bool = True,
        _token: object | None = None,
    ):
        self.scan_sampling = dset.sampling[:2]
        self.scan_units = dset.units[:2]
        self.gpts = dset.shape[:2]
        # TODO remove after Dataset torch migration complete
        dset_numpy = dset.array if dset.array is not None else dset.tensor.cpu().numpy()

        self.intensities_4d = dset_numpy.copy()

        # convert to dataset3d
        shp = dset_numpy.shape
        dset3d = Dataset3d.from_array(
            array=dset_numpy.reshape((shp[0] * shp[1], shp[2], shp[3])),
            name=dset.name,
            origin=[0, *dset.origin[2:]],
            sampling=[0, *dset.sampling[2:]],
            units=["pix", *dset.units[2:]],
        )
        p = Path(dset.file_path).expanduser().resolve() if dset.file_path is not None else None
        dset3d.file_path = p  # any other attributes to transfer?

        super().__init__(
            dset=dset3d,
            detector_mask=detector_mask,
            verbose=verbose,
            learn_descan=learn_descan,
            learn_scan_positions=learn_scan_positions,
            _token=_token,
        )

    # region --- classmethods ---
    @classmethod
    def from_dataset4dstem(
        cls,
        dset: Dataset4dstem,
        detector_mask: torch.Tensor | np.ndarray | None = None,
        verbose: int | bool = 1,
        learn_descan: bool = True,
        learn_scan_positions: bool = True,
    ) -> Self:
        """
        Create a new Dataset4dstem from a Dataset4dstem.

        Parameters
        ----------
        dset : Dataset4dstem
            The underlying 4D array data

        Returns
        -------
        Dataset4dstem
            A new Dataset4dstem instance
        """
        return cls(
            dset=dset,
            detector_mask=detector_mask,
            verbose=verbose,
            learn_descan=learn_descan,
            learn_scan_positions=learn_scan_positions,
            _token=cls._token,
        )

    @classmethod
    def from_file(
        cls,
        file_path: str,
        file_type: str,
        detector_mask: torch.Tensor | np.ndarray | None = None,
        verbose: int | bool = 1,
        learn_descan: bool = True,
        learn_scan_positions: bool = True,
    ) -> Self:
        """
        Create a new Dataset4dstem from a file.

        Parameters
        ----------
        file_path : str
            Path to the data file
        file_type : str
            The type of file reader needed. See rosettasciio for supported formats
            https://hyperspy.org/rosettasciio/supported_formats/index.html

        Returns
        -------
        Dataset4dstem
            A new Dataset4dstem instance loaded from the file
        """
        # Import here to avoid circular imports
        from quantem.core.io.file_readers import read_4dstem

        dset = read_4dstem(file_path, file_type)
        return cls(
            dset=dset,
            detector_mask=detector_mask,
            verbose=verbose,
            learn_descan=learn_descan,
            learn_scan_positions=learn_scan_positions,
            _token=cls._token,
        )

    @classmethod
    def from_array(
        cls,
        array: np.ndarray | Any,
        name: str | None = None,
        origin: np.ndarray | tuple | list | float | int | None = None,
        sampling: np.ndarray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
        detector_mask: torch.Tensor | np.ndarray | None = None,
        verbose: int | bool = 1,
        learn_descan: bool = True,
        learn_scan_positions: bool = True,
    ) -> Self:
        """
        Create a new Dataset4dstem from an array.

        Parameters
        ----------
        array : np.ndarray | Any
            The underlying 4D array data
        name : str | None, optional
            A descriptive name for the dataset. If None, defaults to "4D-STEM dataset"
        origin : np.ndarray | tuple | list | float | int | None, optional
            The origin coordinates for each dimension in calibrated units. If None, defaults to zeros
        sampling : np.ndarray | tuple | list | float | int | None, optional
            The sampling rate/spacing for each dimension. If None, defaults to ones
        units : list[str] | tuple | list | None, optional
            Units for each dimension. If None, defaults to ["pixels"] * 4
        signal_units : str, optional
            Units for the array values, by default "arb. units"

        Returns
        -------
        Dataset4dstem
            A new Dataset4dstem instance
        """
        dset = Dataset4dstem.from_array(
            array=array,
            name=name if name is not None else "4D-STEM dataset",
            origin=origin if origin is not None else np.zeros(4),
            sampling=sampling if sampling is not None else np.ones(4),
            units=units if units is not None else ["pixels"] * 4,
            signal_units=signal_units,
        )
        return cls.from_dataset4dstem(
            dset=dset,
            detector_mask=detector_mask,
            verbose=verbose,
            learn_descan=learn_descan,
            learn_scan_positions=learn_scan_positions,
        )

    # endregion --- classmethods ---

    # region --- properties ---
    @property
    def intensities_4d(self) -> np.ndarray:
        """4D diffraction intensities"""
        return self._intensities_4d

    @intensities_4d.setter
    def intensities_4d(self, intensities: np.ndarray) -> None:
        # Keep raw detector counts in their native dtype
        dtype = validate_array(intensities, name="intensities_4d", ndim=4).dtype
        self._intensities_4d = validate_array(
            intensities, name="intensities_4d", ndim=4, dtype=dtype
        )

    @property
    def com_measured(self) -> np.ndarray:
        """Measured center of mass in pixels"""
        return self._com_measured

    @com_measured.setter
    def com_measured(self, com: np.ndarray | tuple) -> None:
        com = validate_array(
            com, name="com_measured", shape=(2, *self.gpts), dtype=config.get("dtype_real")
        )
        self._com_measured = com

    @property
    def com_fit(self) -> np.ndarray:
        """fit center of mass in pixels"""
        return self._com_fit

    @com_fit.setter
    def com_fit(self, com: np.ndarray | tuple) -> None:
        com = validate_array(
            com, name="com_fit", shape=(2, *self.gpts), dtype=config.get("dtype_real")
        )
        self._com_fit = com

    @property
    def com_normalized(self) -> np.ndarray:
        """normalized center of mass: (measured - fitted) * reciprocal_sampling"""
        difs = np.nan_to_num(self.com_measured - self.com_fit)
        return difs * self.reciprocal_sampling[:, None, None]

    @property
    def scan_sampling(self) -> np.ndarray:
        """Scan sampling in pixels"""
        return self._scan_sampling

    @scan_sampling.setter
    def scan_sampling(self, sampling: np.ndarray | tuple) -> None:
        sampling = validate_array(
            sampling,
            name="scan_sampling",
            shape=(2,),
            dtype=config.get("dtype_real"),
        )
        self._scan_sampling = sampling

    @property
    def scan_units(self) -> list[str]:
        """Scan units"""
        return self._scan_units

    @scan_units.setter
    def scan_units(self, units: list[str]) -> None:
        self._scan_units = units

    @property
    def gpts(self) -> np.ndarray:
        """Number of gpts"""
        return self._gpts

    @gpts.setter
    def gpts(self, gpts: np.ndarray | tuple | list) -> None:
        gpts = validate_array(gpts, name="gpts", shape=(2,), dtype=int)
        self._gpts = gpts

    @property
    def fov(self) -> np.ndarray:
        """
        Field of view in real space. Units of A matching self.obj_sampling
        untying this from gpts is actually a little tricky, as the fov needs to be fixed as
        it is used for calculating the object shape, which shouldn't change during the recon
        """
        # min_pos = torch.min(self.initial_scan_positions_px, dim=0)[0]
        # max_pos = torch.max(self.initial_scan_positions_px, dim=0)[0]
        # extent_px = max_pos - min_pos
        # return extent_px.cpu().detach().numpy() * self.obj_sampling
        return self.scan_sampling * (self.gpts - 1)

    @property
    def upsample_factor(self) -> float:
        # pair each crop axis with its own scan-grid axis so the factor stays
        # transpose-invariant (_obj_shape_crop_2d is transposed when com_transpose)
        gpts = self.gpts[::-1] if self.com_transpose else self.gpts
        return (self._obj_shape_crop_2d / gpts).mean()

    # endregion --- properties ---

    def _set_initial_scan_positions_px(
        self,
        obj_padding_px: np.ndarray | tuple | None,
        positions_mask: np.ndarray | None = None,
    ):
        """
        Method to compute the initial guess of scan positions in pixels.

        Parameters
        ----------
        positions: (J,2) np.ndarray or None
            Input probe positions in Å.
            If None, a raster scan using experimental parameters is constructed.
        positions_mask: np.ndarray, optional
            Boolean real space mask to select positions in datacube to skip for reconstruction
        obj_padding_px: Tuple[int,int], optional
            Pixel dimensions to pad object with
            If None, the padding is set to half the probe ROI dimensions
        positions_offset_ang, np.ndarray, optional
            Offset of positions in A
        """

        if obj_padding_px is None:
            obj_padding_px = np.array([0, 0])

        if positions_mask is not None:
            self.positions_mask = positions_mask

        nr, nc = self.gpts
        Sr, Sc = self._scan_sampling
        r = np.arange(nr) * Sr
        c = np.arange(nc) * Sc

        r, c = np.meshgrid(r, c, indexing="ij")

        pos_mask_2d = self.positions_mask.reshape(nr, nc).cpu().numpy()
        r = r[pos_mask_2d]
        c = c[pos_mask_2d]

        positions = np.stack((r.ravel(), c.ravel()), axis=-1).astype(config.get("dtype_real"))

        if self.com_rotation_rad != 0:
            tf = AffineTransform(angle=self.com_rotation_rad)
            positions = tf(positions, origin=positions.mean(0))

        sampling = self.obj_sampling
        if self.com_transpose:
            positions = np.flip(positions, axis=1)
            sampling = sampling[::-1]

        # anchor the used positions to the object origin
        m: np.ndarray = np.min(positions, axis=0)
        positions -= m

        # finally, switch to pixels
        positions[:, 0] /= sampling[0]
        positions[:, 1] /= sampling[1]

        # top-left padding
        positions[:, 0] += obj_padding_px[0]
        positions[:, 1] += obj_padding_px[1]
        self.scan_positions_px = positions
        self.initial_scan_positions_px = self.scan_positions_px.data.clone()
        return

    def preprocess(
        self,
        com_fit_function: Literal["none", "plane", "parabola", "constant", "no_shift"] = "plane",
        force_com_rotation: float | None = None,
        force_com_transpose: bool | None = None,
        bilinear: bool = False,
        padded_diffraction_intensities_shape: tuple[int, int] | None = None,
        obj_padding_px: tuple[int, int] | np.ndarray = (0, 0),
        plot_rotation: bool = True,
        plot_com: str | bool = True,
        vectorized: bool = True,
        probe_energy: float | None = None,
        positions_mask: np.ndarray | None = None,
    ):
        if positions_mask is not None:
            self.positions_mask = positions_mask

        # Store preprocessing parameters for serialization and reloading
        self._preprocessing_params = {
            "com_fit_function": com_fit_function,
            "force_com_rotation": force_com_rotation,
            "force_com_transpose": force_com_transpose,
            "bilinear": bilinear,
            "padded_diffraction_intensities_shape": padded_diffraction_intensities_shape,
            "obj_padding_px": obj_padding_px,
            "plot_rotation": False,
            "plot_com": False,
            "vectorized": vectorized,
            "positions_mask": self.positions_mask.cpu().numpy(),
        }

        if probe_energy is not None:
            self.probe_energy = probe_energy

        if padded_diffraction_intensities_shape is not None:
            self.diffraction_padding = (
                np.array(padded_diffraction_intensities_shape) - np.array(self.dset.shape[-2:])
            ) // 2
            self.dset.pad(
                output_shape=(
                    self.num_gpts,
                    *padded_diffraction_intensities_shape,
                ),
                modify_in_place=True,
            )
            # TODO remove after Dataset torch migration complete
            dset_numpy = (
                self.dset.array if self.dset.array is not None else self.dset.tensor.cpu().numpy()
            )
            self.intensities_4d = dset_numpy.reshape(
                (*self.gpts, *padded_diffraction_intensities_shape)
            )
            self.detector_mask = torch.nn.functional.pad(
                self.detector_mask,
                (
                    self.diffraction_padding[1],
                    self.diffraction_padding[1],
                    self.diffraction_padding[0],
                    self.diffraction_padding[0],
                ),
                mode="constant",
                value=0,
            )
        else:
            self.diffraction_padding = (0, 0)

        # calculate CoM
        self._set_intensities_com(
            self.intensities_4d,
            fit_function=com_fit_function,
            vectorized_calculation=vectorized,
        )
        self._set_com_relative_rotation(
            force_com_rotation=force_com_rotation,
            force_com_transpose=force_com_transpose,
            plot_rotation=plot_rotation,
            plot_com=plot_com,
        )

        # set the various amplitudese and intensities (can be stripped down later)
        self._normalize_diffraction_intensities(bilinear=bilinear)

        self._set_initial_scan_positions_px(obj_padding_px)
        self._set_patch_indices(obj_padding_px)

        self._set_targets("amplitude")  # default space; criterion-driven space set in reconstruct

        self._preprocessed = True
        return

    def _set_intensities_com(
        self,
        intensities: np.ndarray,
        dp_mask: np.ndarray | None = None,
        fit_function: Literal["none", "plane", "parabola", "constant", "no_shift"] = "plane",
        vectorized_calculation=True,
    ) -> None:
        """
        Common preprocessing function to compute and fit diffraction intensities CoM

        Parameters
        ----------
        intensities: (Rr,Rc,Qr,Qc) np.ndarray
            Raw intensities array stored on device, with dtype np.float32
        dp_mask: ndarray
            If not None, apply mask to datacube intensities
        fit_function: str, optional
            2D fitting function for CoM fitting. One of 'plane','parabola','bezier_two'
        vectorized_calculation: bool, optional
            If True (default), the calculation is vectorized

        Returns
        -------
        None
        """
        if dp_mask is not None:
            if dp_mask.shape != intensities.shape[-2:]:
                raise ValueError(
                    f"Mask shape should be (Qr,Qc) = {intensities.shape[-2:]} | got {dp_mask.shape}"
                )
            dp_mask = np.asarray(dp_mask, dtype=config.get("dtype_real"))

        # Coordinates
        kr = np.arange(intensities.shape[-2])
        kc = np.arange(intensities.shape[-1])
        krm, kcm = np.meshgrid(kr, kc, indexing="ij")

        if vectorized_calculation:
            if dp_mask is not None:
                intensities_mask = (intensities * dp_mask).astype(config.get("dtype_real"))
            else:
                intensities_mask = (intensities).astype(config.get("dtype_real"))
            com_measured_r = np.sum(intensities_mask * krm[None, None], axis=(-2, -1))
            com_measured_c = np.sum(intensities_mask * kcm[None, None], axis=(-2, -1))

            intensities_sum = np.sum(intensities_mask, axis=(-2, -1))
            intensities_mask = intensities_sum == 0
            intensities_sum[intensities_mask] = 1
            com_measured_r /= intensities_sum
            com_measured_c /= intensities_sum
            com_measured_r[intensities_mask] = np.nan
            com_measured_c[intensities_mask] = np.nan

        else:
            shape_r, shape_c = intensities.shape[:2]
            com_measured_r = np.zeros((shape_r, shape_c))
            com_measured_c = np.zeros((shape_r, shape_c))

            # loop of dps
            for Rr, Rc in tqdmnd(
                range(shape_r),
                range(shape_c),
                desc="Calculating center of mass",
                unit="probe position",
                disable=not self._verbose,
            ):
                masked_intensity = intensities[Rr, Rc]
                if dp_mask is not None:
                    masked_intensity = masked_intensity * dp_mask
                summed_intensity = masked_intensity.sum()
                if summed_intensity == 0:
                    com_measured_r[Rr, Rc] = np.nan
                    com_measured_c[Rr, Rc] = np.nan
                else:
                    com_measured_r[Rr, Rc] = np.sum(masked_intensity * krm) / summed_intensity
                    com_measured_c[Rr, Rc] = np.sum(masked_intensity * kcm) / summed_intensity

        if fit_function == "none":
            com_fit_r, com_fit_c = com_measured_r, com_measured_c
        elif fit_function == "no_shift":
            com_fit_r, com_fit_c = np.ones_like(com_measured_r), np.ones_like(com_measured_c)
            com_fit_r = com_fit_r * self.roi_shape[0] / 2
            com_fit_c = com_fit_c * self.roi_shape[1] / 2
        else:
            pos_mask_2d = self.positions_mask.reshape(self.gpts[0], self.gpts[1]).cpu().numpy()
            finite_mask = np.isfinite(com_measured_r) & pos_mask_2d
            com_fit_r, com_fit_c, _com_res_r, _com_res_c = fit_origin(
                data=(com_measured_r, com_measured_c),
                fit_function=fit_function,
                mask=finite_mask,
            )

        self.com_measured = (com_measured_r, com_measured_c)  # raw measured pixels
        self.com_fit = (com_fit_r, com_fit_c)  # fitted for descan, pixels
        return

    def _set_com_relative_rotation(
        self,
        rotation_angles_deg: np.ndarray | None = None,
        force_com_rotation: float | None = None,
        force_com_transpose: bool | None = None,
        plot_rotation: bool = True,
        plot_com: str | bool = "default",
        **kwargs,
    ):
        """
        Common method to solve for the relative rotation between scan directions
        and the reciprocal coordinate system. We do this by minimizing the curl of the
        CoM gradient vector field or, alternatively, maximizing the divergence.

        force_com_rotation: float (degrees), optional
            Force relative rotation angle between real and reciprocal space
        force_com_transpose: bool, optional
            Force whether diffraction intensities need to be transposed.
        plot_rotation: bool, optional
            If True, the CoM curl minimization search result will be displayed
        plot_com: str, optional
            If 'default', the corrected CoM arrays will be displayed
            If 'all', the computed and fitted CoM arrays will be displayed
        """

        # Helper functions
        def rotate_com_vectors(
            com: np.ndarray,
            angle_rad: float,
            transpose: bool = False,
        ) -> np.ndarray:
            """Rotate CoM vectors by angle_rad with optional transpose"""
            com_r, com_c = com
            if transpose:
                rot_r = np.cos(angle_rad) * com_c - np.sin(angle_rad) * com_r
                rot_c = np.sin(angle_rad) * com_c + np.cos(angle_rad) * com_r
            else:
                rot_r = np.cos(angle_rad) * com_r - np.sin(angle_rad) * com_c
                rot_c = np.sin(angle_rad) * com_r + np.cos(angle_rad) * com_c
            return np.array((rot_r, rot_c))

        def calculate_curl(com_r: np.ndarray, com_c: np.ndarray) -> float:
            """Calculate curl of CoM gradient vector field"""
            grad_r_c = com_r[1:-1, 2:] - com_r[1:-1, :-2]  # dVh/dw
            grad_c_r = com_c[2:, 1:-1] - com_c[:-2, 1:-1]  # dVw/dh
            return float(np.nanmean(np.abs(grad_c_r - grad_r_c)))

        def calculate_curl_for_angles(
            angles_rad: np.ndarray,
            com: np.ndarray,
            transpose: bool = False,
        ) -> np.ndarray:
            """Calculate curl for multiple angles"""
            angles_rad_expanded = angles_rad[:, None, None]
            com_r, com_c = com[0][None], com[1][None]
            if transpose:
                rot_r = np.cos(angles_rad_expanded) * com_c - np.sin(angles_rad_expanded) * com_r
                rot_c = np.sin(angles_rad_expanded) * com_c + np.cos(angles_rad_expanded) * com_r
            else:
                rot_r = np.cos(angles_rad_expanded) * com_r - np.sin(angles_rad_expanded) * com_c
                rot_c = np.sin(angles_rad_expanded) * com_r + np.cos(angles_rad_expanded) * com_c

            grad_r_c = rot_r[:, 1:-1, 2:] - rot_r[:, 1:-1, :-2]
            grad_c_r = rot_c[:, 2:, 1:-1] - rot_c[:, :-2, 1:-1]
            return np.nanmean(np.abs(grad_c_r - grad_r_c), axis=(-2, -1))

        def plot_curl_results(
            angles_deg: np.ndarray,
            curl_values: np.ndarray | tuple[np.ndarray, np.ndarray],
            best_angle: float,
            transpose: bool = False,
            **plot_kwargs,
        ) -> None:
            """Plot curl vs rotation angle"""
            figsize = plot_kwargs.get("figsize", (8, 2))
            fig, ax = plt.subplots(figsize=figsize)

            if isinstance(curl_values, tuple):
                ax.plot(angles_deg, curl_values[0], label="CoM")
                ax.plot(angles_deg, curl_values[1], label="CoM after transpose")
            else:
                label = "CoM after transpose" if transpose else "CoM"
                ax.plot(angles_deg, curl_values, label=label)

            y_range = ax.get_ylim()
            ax.plot(np.ones(2) * best_angle, y_range, color=(0, 0, 0, 1))

            ax.legend(loc="best")
            ax.set_xlabel("Rotation [degrees]")
            ax.set_ylabel("Mean Absolute Curl")

            if isinstance(curl_values, tuple):
                aspect_ratio = np.maximum(np.ptp(curl_values[0]), np.ptp(curl_values[1]))
            else:
                aspect_ratio = np.ptp(curl_values)
            ax.set_aspect(np.ptp(angles_deg) / aspect_ratio / 4)

            fig.tight_layout()
            plt.show()

        def plot_com_images(
            com_arrays: list[np.ndarray],
            titles: list[str],
            **plot_kwargs,
        ) -> None:
            """Plot CoM vector fields"""
            if len(com_arrays) == 6:  # All CoM arrays
                figsize = plot_kwargs.pop("figsize", (8, 12))
            else:  # Just corrected CoM
                figsize = plot_kwargs.pop("figsize", (8, 4))

            cmap = plot_kwargs.pop("cmap", "RdBu_r")
            show_2d(
                com_arrays,
                title=titles,
                cmap=cmap,
                figsize=figsize,
                scalebar={"sampling": self.obj_sampling[0], "units": "A"},
                norm={"interval_type": "manual"},
                force_show=True,
                **plot_kwargs,
            )

        if rotation_angles_deg is None:
            rotation_angles_deg = np.arange(-89.0, 90.0, 1.0)

        rotation_angles_deg = np.asarray(rotation_angles_deg)
        rotation_angles_rad = np.deg2rad(rotation_angles_deg)

        pos_mask_2d = self.positions_mask.reshape(self.gpts[0], self.gpts[1]).cpu().numpy()
        com_normalized_masked = self.com_normalized.copy()
        com_normalized_masked[:, ~pos_mask_2d] = np.nan

        # Case 1: Known rotation
        if force_com_rotation is not None:
            _rotation_best_rad = np.deg2rad(force_com_rotation)
            self.vprint(f"Forcing best fit rotation to {force_com_rotation:.0f} degrees.")

            # Case 1.1: Known rotation and transpose
            if force_com_transpose is not None:
                _rotation_best_transpose = force_com_transpose
                self.vprint(f"Forcing transpose of intensities to {force_com_transpose}.")

            # Case 1.2: Known rotation, unknown transpose
            else:
                # Calculate curl for both transpose options
                rot_r, rot_c = rotate_com_vectors(
                    com_normalized_masked, _rotation_best_rad, transpose=False
                )
                rotation_curl = calculate_curl(rot_r, rot_c)

                rot_r, rot_c = rotate_com_vectors(
                    com_normalized_masked, _rotation_best_rad, transpose=True
                )
                rotation_curl_transpose = calculate_curl(rot_r, rot_c)

                # Choose the option with minimum curl
                _rotation_best_transpose = rotation_curl_transpose < rotation_curl

                if _rotation_best_transpose:
                    self.vprint("Diffraction intensities should be transposed.")

        # Case 2: Unknown rotation
        else:
            # Case 2.1: Known transpose, unknown rotation
            if force_com_transpose is not None:
                _rotation_best_transpose = force_com_transpose
                self.vprint(f"Forcing transpose of intensities to {force_com_transpose}.")

                # Calculate curl for all angles with known transpose
                curl_values = calculate_curl_for_angles(
                    rotation_angles_rad,
                    com_normalized_masked,
                    transpose=_rotation_best_transpose,
                )

                # Find angle with minimum curl
                min_index = np.argmin(curl_values).item()
                rotation_best_deg = rotation_angles_deg[min_index]
                _rotation_best_rad = rotation_angles_rad[min_index]
                self.vprint(f"Calculated best fit rotation = {rotation_best_deg:.0f} degrees.")

                if plot_rotation:
                    plot_curl_results(
                        rotation_angles_deg,
                        curl_values,
                        rotation_best_deg,
                        transpose=_rotation_best_transpose,
                        **kwargs,
                    )

            else:
                # Case 2.2: Unknown rotation and transpose
                # Calculate curl for both transpose options
                rotation_curl = calculate_curl_for_angles(
                    rotation_angles_rad,
                    com_normalized_masked,
                    transpose=False,
                )

                rotation_curl_transpose = calculate_curl_for_angles(
                    rotation_angles_rad,
                    com_normalized_masked,
                    transpose=True,
                )

                # Minimize Curl
                ind_min = np.argmin(rotation_curl).item()
                ind_trans_min = np.argmin(rotation_curl_transpose).item()
                if rotation_curl[ind_min] <= rotation_curl_transpose[ind_trans_min]:
                    rotation_best_deg = rotation_angles_deg[ind_min]
                    _rotation_best_rad = rotation_angles_rad[ind_min]
                    _rotation_best_transpose = False
                else:
                    rotation_best_deg = rotation_angles_deg[ind_trans_min]
                    _rotation_best_rad = rotation_angles_rad[ind_trans_min]
                    _rotation_best_transpose = True

                self._rotation_angles_deg = rotation_angles_deg
                self.vprint(f"Calculated best fit rotation = {rotation_best_deg:.0f} degrees.")
                if _rotation_best_transpose:
                    self.vprint("Diffraction intensities should be transposed.")

                if plot_rotation:
                    plot_curl_results(
                        rotation_angles_deg,
                        (rotation_curl, rotation_curl_transpose),
                        rotation_best_deg,
                        **kwargs,
                    )

        _com_r, _com_c = rotate_com_vectors(
            self.com_normalized,
            _rotation_best_rad,
            transpose=_rotation_best_transpose,
        )

        if plot_com == "all":
            plot_com_images(
                [*self.com_measured, *self.com_normalized, _com_r, _com_c],
                [
                    "CoM_r",
                    "CoM_c",
                    "Normalized CoM_r",
                    "Normalized CoM_c",
                    "Corrected CoM_r",
                    "Corrected CoM_c",
                ],
                **kwargs,
            )
        elif plot_com == "default" or plot_com is True:
            plot_com_images(
                [_com_r, _com_c],
                ["Corrected CoM_r", "Corrected CoM_c"],
                **kwargs,
            )

        self.com_rotation_rad = _rotation_best_rad
        self.com_transpose = _rotation_best_transpose
        self._com = _com_r, _com_c  # com_normalized rotated by com_rotation_rad
        return

    def _normalize_diffraction_intensities(
        self,
        positions_mask: np.ndarray | None = None,
        crop_patterns: bool = False,
        bilinear: bool = False,
    ):
        dtype = config.get("dtype_real")
        np_dtype = np.dtype(dtype)
        raw = self._intensities_4d
        com_fit = self.com_fit
        if positions_mask is not None:
            self.positions_mask = positions_mask
        positions_mask_2d = self.positions_mask.reshape(self.gpts[0], self.gpts[1]).cpu().numpy()

        # Aggressive cropping for when off-centered high scattering angle data was recorded
        if crop_patterns:
            crop_r = int(np.minimum(raw.shape[2] - com_fit[0].max(), com_fit[0].min()))
            crop_c = int(np.minimum(raw.shape[3] - com_fit[1].max(), com_fit[1].min()))
            crop_m = int(np.minimum(crop_c, crop_r))

            pattern_crop_mask = np.zeros(self.roi_shape, dtype="bool")
            pattern_crop_mask[:crop_m, :crop_m] = True
            pattern_crop_mask[-crop_m:, :crop_m] = True
            pattern_crop_mask[:crop_m:, -crop_m:] = True
            pattern_crop_mask[-crop_m:, -crop_m:] = True
            pattern_crop_mask_shape = (crop_m * 2, crop_m * 2)

        else:
            pattern_crop_mask = None
            pattern_crop_mask_shape = (int(self.roi_shape[0]), int(self.roi_shape[1]))

        # Accumulate into a numpy buffer and convert once at the end
        n_out = int(positions_mask_2d.sum())
        centered_amplitudes = np.zeros((n_out, *pattern_crop_mask_shape), dtype=np_dtype)

        mean_intensity = 0
        mean_amplitude = 0
        out_i = 0
        current_Rr = -1
        row = None
        for Rr, Rc in tqdmnd(
            range(raw.shape[0]),
            range(raw.shape[1]),
            desc="Normalizing intensities",
            unit="probe position",
            disable=not self._verbose,
        ):
            if not positions_mask_2d[Rr, Rc]:
                continue

            # Cast to float32 one scan-row at a time--avoids a full 4D float32 copy
            if Rr != current_Rr:
                current_Rr = Rr
                row = raw[Rr]
                if row.dtype != np_dtype:
                    row = row.astype(dtype)
            intensity = np.maximum(row[Rc], 0)
            mean_intensity += np.sum(intensity)

            ### shifting amplitude rather than intensity to minimize ringing artifacts
            amplitude = np.maximum(np.sqrt(intensity), 0)
            mean_amplitude += np.sum(amplitude)
            shift_amplitude = shift_array(  # shifting to 0,0 then fftshift
                amplitude,
                -(com_fit[0, Rr, Rc] + 0.0),
                -(com_fit[1, Rr, Rc] + 0.0),
                bilinear=bilinear,
            )
            shift_amplitude = np.maximum(shift_amplitude, 0)
            shift_amplitude = np.fft.fftshift(shift_amplitude)
            if pattern_crop_mask is not None:
                shift_amplitude = shift_amplitude[pattern_crop_mask].reshape(
                    pattern_crop_mask_shape
                )

            centered_amplitudes[out_i] = shift_amplitude
            out_i += 1

        mean_intensity /= n_out
        mean_amplitude /= n_out

        # Wrap the numpy buffer zero-copy (``torch.from_numpy`` shares storage)
        self._centered_amplitudes = torch.from_numpy(centered_amplitudes)
        self._amplitudes = None
        self._intensities = None
        self._centered_intensities = None
        descan_shifts = -1 * com_fit.reshape((2, -1))  # (2, rr*rc)
        descan_shifts += self.roi_shape[:, None] / 2
        descan_shifts = descan_shifts.T  # (rr*rc, 2)
        descan_shifts = descan_shifts[positions_mask_2d.ravel()]
        self.descan_shifts = descan_shifts
        self.initial_descan_shifts = self.descan_shifts.data.clone()

        self.mean_diffraction_intensity = mean_intensity
        self.mean_diffraction_amplitude = mean_amplitude
        self._pattern_crop_mask = pattern_crop_mask
        self._pattern_crop_mask_shape = pattern_crop_mask_shape
        return

    def forward(
        self,
        batch_indices: np.ndarray | torch.Tensor,
        obj_padding_px: np.ndarray | tuple,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Forward pass to compute the diffraction intensities from the object and scan positions.

        The first return value is the object-query payload: integer ``patch_indices`` for a
        grid-based object, or continuous normalized coordinates for an implicit object
        (``implicit_object=True``). In the implicit case the fractional position is baked into
        the coordinates, so the returned fractional shift is zero (the probe is not shifted).
        """
        self.apply_hard_constraints(obj_padding_px)
        positions_px = self.scan_positions_px[batch_indices]
        if self.learn_descan and self.has_optimizer():
            descan_shifts = self.apply_descan_constraints(self.descan_shifts)[batch_indices]
        else:
            descan_shifts = None

        if self._implicit_object:
            patch_data = self._scan_coords(batch_indices, obj_padding_px)
            positions_px_fractional = torch.zeros_like(positions_px)
            return patch_data, positions_px, positions_px_fractional, descan_shifts

        positions_px_fractional = positions_px - torch.round(positions_px)
        with torch.no_grad():
            if self.patch_indices_need_update():
                self._set_patch_indices(obj_padding_px)
        patch_data = self.patch_indices[batch_indices]
        return patch_data, positions_px, positions_px_fractional, descan_shifts


DatasetModelType = PtychographyDatasetRaster  # | PtychographyDatasetSpiral
