from typing import Any, Literal, Sequence, TypedDict, cast
from warnings import warn

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from quantem.core import config
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.constraints import Constraints
from quantem.core.ml.dist_utils import all_reduce_params, worker_init_fn
from quantem.core.utils.rng import RNGMixin
from quantem.core.utils.utils import (
    electron_wavelength_angstrom,
    generate_batches,
    to_numpy,
)
from quantem.core.utils.validators import (
    validate_array,
    validate_gt,
    validate_int,
    validate_np_len,
    validate_tensor,
)
from quantem.diffractive_imaging.dataset_models import (
    DatasetModelType,
    PtychographyDatasetBase,
)
from quantem.diffractive_imaging.detector_models import DetectorBase, DetectorModelType
from quantem.diffractive_imaging.logger_ptychography import LoggerPtychography
from quantem.diffractive_imaging.object_models import ObjectBase, ObjectModelType
from quantem.diffractive_imaging.probe_models import ProbeBase, ProbeModelType, ProbePixelated
from quantem.diffractive_imaging.ptycho_losses import DataCriterion, get_data_criterion
from quantem.diffractive_imaging.ptycho_utils import (
    center_crop_arr,
    fourier_translation_operator,
    sum_patches,
)

"""
design patterns:
    - all outward facing properties ptycho.blah will give numpy arrays
        - hidden attributes, ptycho._blah will be torch, living on cpu/gpu depending on config
    - objects are always 3D, if doing a singleslice recon, the shape is just [1, :, :]
    - likewise probes are always stacks for mixed state, if single probe, then shape is [1, :, :]
    - all preprocessing will be done with torch tensors 
"""


class Snapshot(TypedDict):
    """
    A snapshot of the object and probe at a given iteration.

    Parameters
    ----------
    obj: np.ndarray
        The object at the given iteration.
    probe: np.ndarray
        The probe at the given iteration.
    iteration: int
        The iteration number.
    """

    obj: np.ndarray
    probe: np.ndarray
    iteration: int


class PtychographyBase(RNGMixin, AutoSerialize):
    """
    A base class for performing phase retrieval using the Ptychography algorithm.

    This class provides a basic framework for performing phase retrieval using the Ptychography algorithm.
    It is designed to be subclassed by specific Ptychography algorithms.
    """

    _token = object()

    def __init__(  # TODO prevent direct instantiation
        self,
        dset: DatasetModelType,
        obj_model: ObjectModelType,
        probe_model: ProbeModelType,
        detector_model: DetectorModelType,
        logger: LoggerPtychography | None = None,
        device: str | int = "cpu",  # "gpu" | "cpu" | "cuda:X"
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
        _token: None | object = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use Dataset.from_array() to instantiate this class.")

        if not config.get("has_torch"):
            raise RuntimeError("the quantEM Ptychography module requires torch to be installed.")

        super().__init__()

        # Pre-initialize private attributes so type checker sees them in __init__
        self._verbose: int = 0
        self._logger: LoggerPtychography | None = None
        self._batch_size: int = 1
        self._dset: DatasetModelType = dset
        self._obj_model: ObjectModelType = obj_model
        self._probe_model: ProbeModelType = probe_model
        self._detector_model: DetectorModelType = detector_model
        self._criterion: DataCriterion = get_data_criterion("l2_amplitude")

        self.verbose = verbose
        self.dset = dset
        self.device = device
        self.rng = rng

        # initializing default attributes
        self._multi_gpu_devices: list[int] | None = None
        self._preprocessed: bool = False
        self._obj_padding_force_power2_level: int = 3
        self._store_snapshots: bool = False
        self._store_snapshot_every: int = 1
        self._iter_losses: list[float] = []
        self._iter_val_losses: list[float] = []
        self._iter_recon_types: list[str] = []
        self._iter_lrs: dict[str, list[float]] = {}  # LRs/step_sizes across iterations
        self._snapshots: list[Snapshot] = []
        self._obj_padding_px = np.array([0, 0])
        self.obj_fov_mask = torch.ones(tuple(self.dset._obj_shape_full_2d(self.obj_padding_px)))
        self.batch_size = self.dset.num_gpts
        self._val_ratio = 0.0
        self._val_mode: Literal["grid", "random"] = "grid"

        if (
            isinstance(probe_model, ProbePixelated)
            and (probe_model.vacuum_probe_intensity is not None)
            # ``centered_amplitudes`` shares amplitudes' shape but is always resident (amplitudes is
            # recomputed lazily), so use it here to avoid materializing the full raw array.
            and (dset.centered_amplitudes.shape[1:] != probe_model.vacuum_probe_intensity.shape)
        ):
            probe_model.rescale_vacuum_probe(
                (dset.centered_amplitudes.shape[1], dset.centered_amplitudes.shape[2])
            )

        # Remove centralized optimizer storage - now managed by individual models
        self.probe_model = probe_model
        self.obj_model = obj_model
        self.detector_model = detector_model
        self.compute_propagator_arrays()
        self.logger = logger
        self.to(self._single_device)

    # region --- preprocessing ---
    ## hopefully will be able to remove some of these preprocessing flags,
    ## convert plotting and vectorized to kwargs
    def preprocess(
        self,
        obj_padding_px: tuple[int, int] = (0, 0),
        val_ratio: float = 0.0,
        val_mode: Literal["grid", "random"] = "grid",
        vectorized: bool = True,
        batch_size: int | None = None,
        com_fit_function: Literal[  # TODO replace with dataset kwaargs?
            "none", "plane", "parabola", "constant", "no_shift"
        ] = "constant",
        force_com_rotation: float | None = None,
        force_com_transpose: bool | None = None,
        padded_diffraction_intensities_shape: tuple[int, int] | None = None,
        plot_rotation: bool = True,
        plot_com: str | bool = True,
        plot_probe_overlap: bool = False,
    ):
        """
        Rather than passing 100 flags here, I'm going to suggest that if users want to run very
        customized pre-processing, they just call the functions themselves directly.
        """
        # self.to(self.device)
        if not self.dset.preprocessed:
            self.vprint("Dataset was not preprocessed, proceeding with defaults.")
            self.dset.preprocess(
                com_fit_function=com_fit_function,
                force_com_rotation=force_com_rotation,
                force_com_transpose=force_com_transpose,
                padded_diffraction_intensities_shape=padded_diffraction_intensities_shape,
                obj_padding_px=obj_padding_px,
                plot_rotation=plot_rotation,
                plot_com=plot_com,
                vectorized=vectorized,
            )
            self._probe_model.set_initial_probe(
                self.roi_shape,
                self.reciprocal_sampling,
                self.dset.mean_diffraction_intensity,
                device=self._single_device,
            )

        # change obj_padding_px and whatever else needs to be changed
        self.obj_padding_px = obj_padding_px  # also initializes the object model

        self.compute_propagator_arrays()
        self._set_obj_fov_mask(batch_size=batch_size)
        self._preprocessed = True
        # store validation split ratio for reconstruction step
        self.val_ratio = float(val_ratio)
        self.val_mode = val_mode
        # if self.num_iters == 0:
        #     self.reset_recon()  # if new models, reset to ensure shapes are correct
        return self

    def _set_obj_fov_mask(self, gaussian_sigma: float = 2.0, batch_size=None):
        overlap = self._get_probe_overlap(batch_size)
        ov = overlap > overlap.max() * 0.3
        ov = ndi.binary_closing(ov, iterations=5)
        dilation_iters = min(32, np.min(self.obj_padding_px) // 4)
        if dilation_iters > 0:
            ov = ndi.binary_dilation(ov, iterations=dilation_iters)
        ov = ndi.gaussian_filter(ov.astype(config.get("dtype_real")), sigma=gaussian_sigma)
        self.obj_fov_mask = ov
        self.obj_model.mask = ov
        return

    def _get_probe_overlap(self, max_batch_size: int | None = None) -> np.ndarray:
        prb = self.probe_model.probe[0]
        num_dps = self.dset.num_positions
        shifted_probes = prb.expand(num_dps, *self.roi_shape)

        batch_size = min(num_dps, 4096) if max_batch_size is None else int(max_batch_size)
        probe_overlap = torch.zeros(
            tuple(self.obj_shape_full[-2:]), dtype=self._dtype_real, device=self._single_device
        )
        for start, end in generate_batches(num_dps, max_batch=batch_size):
            probe_overlap += sum_patches(
                torch.abs(shifted_probes[start:end]) ** 2,
                self.dset.patch_indices[start:end],
                tuple(self.obj_shape_full[-2:]),
            )
        return self._to_numpy(probe_overlap)

    # endregion --- preprocessing ---

    # region --- explicit class properties ---
    @property
    def dset(self) -> DatasetModelType:
        return self._dset

    @dset.setter
    def dset(self, new_dset: DatasetModelType):
        if not isinstance(new_dset, PtychographyDatasetBase) and "PtychographyDataset" not in str(
            type(new_dset)
        ):
            raise TypeError(f"dset should be a PtychographyDataset, got {type(new_dset)}")
        self._dset = new_dset

    @property
    def criterion(self) -> DataCriterion:
        """Active data-fidelity criterion. Assign a registered name or a ``DataCriterion``.

        Transient config (re-set from ``loss_type`` each ``reconstruct``); not serialized, so it
        lazily re-defaults to L2 on a loaded object (where ``__init__`` did not run).
        """
        if getattr(self, "_criterion", None) is None:
            self._criterion = get_data_criterion("l2_amplitude")
        return self._criterion

    @criterion.setter
    def criterion(self, value: "str | DataCriterion") -> None:
        self._criterion = get_data_criterion(value)

    @property
    def detector_model(self) -> DetectorModelType:
        return self._detector_model

    @detector_model.setter
    def detector_model(self, new_detector_model: DetectorModelType):
        if not isinstance(new_detector_model, DetectorBase) and "Detector" not in str(
            type(new_detector_model)
        ):
            raise TypeError(f"detector_model should be a Detector, got {type(new_detector_model)}")
        self._detector_model = new_detector_model

    @property
    def obj_type(self) -> str:
        return self.obj_model._obj_type

    def set_obj_type(self, t: str | None, force: bool = False) -> None:
        new_obj_type = self.obj_model._process_obj_type(t)
        if self.num_iters > 0 and new_obj_type != self.obj_model.obj_type and not force:
            raise ValueError(
                "Cannot change object type after training. Run with reset=True or rerun preprocess."
            )
        self.obj_model.obj_type = new_obj_type

    @property
    def num_slices(self) -> int:
        """if num_slices > 1, then it is multislice reconstruction"""
        return self.obj_model.num_slices

    @property
    def propagators(self) -> torch.Tensor:
        if self.num_slices == 1:
            return torch.tensor([])
        else:
            return self._propagators

    @propagators.setter
    def propagators(
        self, prop: "np.ndarray | list[np.ndarray] | torch.Tensor | list[torch.Tensor]"
    ) -> None:
        if self.num_slices == 1:
            self._propagators = torch.tensor([])
        else:
            prop = validate_tensor(
                prop,
                name="propagators",
                dtype=config.get("dtype_complex"),
                ndim=3,
                shape=(self.num_slices - 1, *self.roi_shape),
                expand_dims=False,
            )
            self._propagators = self._to_torch(prop)

    @property
    def num_probes(self) -> int:
        """if num_probes > 1, then it is a mixed-state reconstruction"""
        return self.probe_model.num_probes

    @property
    def slice_thicknesses(self) -> np.ndarray:
        slice_thick = self._obj_model.slice_thicknesses
        if slice_thick is None:
            return np.array([])
        return self._to_numpy(slice_thick)

    @slice_thicknesses.setter
    def slice_thicknesses(self, val: float | Sequence[float] | None) -> None:
        self._obj_model.slice_thicknesses = val
        if hasattr(self, "_propagators"):  # propagators already set, update with new slices
            self.compute_propagator_arrays()

    @property
    def verbose(self) -> int:
        return self._verbose

    @verbose.setter
    def verbose(self, v: bool | int | float) -> None:
        self._verbose = validate_int(validate_gt(v, -1, "verbose"), "verbose")

    @property
    def obj(self) -> np.ndarray:
        """Object array in its native representation per ``obj_type``:

        - ``"complex"`` → complex ndarray (amp * exp(1j*phase)); phase recentered.
        - ``"pure_phase"`` → real ndarray of phase values.
        - ``"potential"`` → real ndarray of potential values.
        """
        obj = self._to_numpy(self.obj_model.obj)
        if self.obj_type == "complex":
            ph = np.angle(obj)
            obj = np.abs(obj) * np.exp(1j * (ph - ph.mean()))
        return obj

    @property
    def obj_padding_px(self) -> np.ndarray:
        return self._obj_padding_px

    @obj_padding_px.setter
    def obj_padding_px(self, pad: np.ndarray | tuple[int, int]):
        p2 = self._to_numpy(
            validate_array(
                validate_np_len(pad, 2, name="obj_padding_px"),
                dtype="int16",
                ndim=1,
                name="obj_padding_px",
            )
        )
        if self._obj_padding_force_power2_level > 0:
            p2 = adjust_padding_power2(
                p2,
                self.dset._obj_shape_full_2d((0, 0)),
                self._obj_padding_force_power2_level,
            )
        self._obj_padding_px = p2
        self.obj_model._initialize_obj(shape=self.obj_shape_full, sampling=self.sampling)
        self.dset._set_initial_scan_positions_px(self.obj_padding_px)
        self.dset._set_patch_indices(self.obj_padding_px)
        self.dset._preprocessing_params["obj_padding_px"] = self.obj_padding_px

    @property
    def obj_fov_mask(self) -> np.ndarray:
        return self._to_numpy(self._obj_fov_mask)

    @obj_fov_mask.setter
    def obj_fov_mask(self, mask: "np.ndarray|torch.Tensor"):
        mask = validate_tensor(
            mask,
            dtype=config.get("dtype_real"),
            ndim=3,
            name="obj_fov_mask",
            expand_dims=True,
        )
        self._obj_fov_mask = self._to_torch(mask)

    @property
    def iter_losses(self) -> np.ndarray:
        """
        Loss/MSE error for each iteration regardless of reconstruction method used
        """
        return np.array(self._iter_losses)

    @property
    def val_iter_losses(self) -> np.ndarray:
        """
        Validation loss (consistency) per iteration if a validation split was used.
        """
        return np.array(self._iter_val_losses)

    @property
    def val_ratio(self) -> float:
        return float(self._val_ratio)

    @val_ratio.setter
    def val_ratio(self, r: float) -> None:
        r = float(r)
        if r < 0.0 or r > 1.0:
            raise ValueError("val_ratio must satisfy 0 <= val_ratio <= 1")
        self._val_ratio = r

    @property
    def val_mode(self) -> Literal["grid", "random"]:
        return self._val_mode

    @val_mode.setter
    def val_mode(self, mode: Literal["grid", "random"]) -> None:
        if mode not in ["grid", "random"]:
            raise ValueError(f"val_mode must be either 'grid' or 'random', got {mode}")
        self._val_mode = mode

    @property
    def num_iters(self) -> int:
        """
        Number of iterations for which the recon has been run so far
        """
        return len(self.iter_losses)

    @property
    def iter_recon_types(self) -> np.ndarray:
        """
        Keeping track of what reconstruction type was used
        """
        return np.array(self._iter_recon_types)

    @property
    def iter_lrs(self) -> dict[str, np.ndarray]:
        """
        List of step sizes/LRs depending on recon type
        """
        return {k: np.array(v) for k, v in self._iter_lrs.items()}

    @property
    def probe(self) -> np.ndarray:
        """Complex valued probe(s). Shape [num_probes, roi_height, roi_width]"""
        return self._to_numpy(self.probe_model.probe)

    @property
    def store_snapshots(self) -> bool:
        return self._store_snapshots

    @store_snapshots.setter
    def store_snapshots(self, val: bool | None) -> None:
        if val is not None:
            self._store_snapshots = bool(val)

    @property
    def store_snapshot_every(self) -> int:
        return self._store_snapshot_every

    @store_snapshot_every.setter
    def store_snapshot_every(self, val: int | None) -> None:
        if val is not None:
            self._store_snapshot_every = int(val)

    @property
    def snapshots(self) -> list[Snapshot]:
        return self._snapshots

    def get_snapshot_by_iter(
        self, iteration: int, closest: bool = False, cropped: bool = False
    ) -> Snapshot:
        """
        Get a snapshot by iteration number.
        Parameters
        ----------
        iteration: int
            The iteration number.
        closest: bool
            Whether to return the closest snapshot if one is not found at the exact iteration.
        cropped: bool
            Whether to crop the object to the field of view. False (default) -> full object.

        Returns
        -------
        snapshot: Snapshot
            The snapshot at the given iteration.
        """
        if len(self.snapshots) == 0:
            raise ValueError(
                "No snapshots available. Use store_snapshots=True during reconstruction."
            )
        iteration = int(iteration)
        if iteration < 0:
            iteration = self.num_iters + iteration
        if closest:
            closest_snapshot = min(self.snapshots, key=lambda s: abs(s["iteration"] - iteration))
            snp = closest_snapshot
        else:
            for snp in self.snapshots:
                if snp["iteration"] == iteration:
                    break
            else:
                raise ValueError(
                    f"No snapshot found at iteration: {iteration}, "
                    + "to return the closest snapshot, set closest=True"
                )
        if cropped:
            snp2 = snp.copy()
            cropped_obj = self._crop_rotate_obj_fov(snp2["obj"])
            # same logic as self.obj_cropped: only re-center for complex (which
            # carries phase inside a complex tensor); pure_phase and potential
            # are already real and recentered upstream.
            if self.obj_type == "complex":
                ph = np.angle(cropped_obj)
                cropped_obj = np.abs(cropped_obj) * np.exp(1j * (ph - ph.mean()))
            snp2["obj"] = cropped_obj
            return snp2
        else:
            return snp

    # TODO is there a way to type hint proper object model type? probably not...
    @property
    def obj_model(self) -> ObjectModelType:
        return self._obj_model

    @obj_model.setter
    def obj_model(self, model: ObjectModelType | type):
        # Type checking with autoreload bug workaround
        if not (isinstance(model, ObjectBase) or "object" in str(type(model))):
            raise TypeError(f"obj_model must be a ObjectModelType, got {type(model)}")

        # Set object shape
        model.to(self._single_device)
        self._obj_model = cast(ObjectModelType, model)
        # Keep the dataset's forward path (coordinates vs. integer patch_indices) in sync with
        # the object representation. Implicit objects are queried at continuous coordinates.
        if hasattr(self, "_dset"):
            self.dset.implicit_object = model.is_implicit

    @property
    def probe_model(self) -> ProbeModelType:
        return self._probe_model

    @probe_model.setter
    def probe_model(self, model: ProbeModelType | type):
        # Type checking with autoreload bug workaround
        if not (isinstance(model, ProbeBase) or "probe" in str(type(model))):
            raise TypeError(f"probe_model must be a ProbeModelType, got {type(model)}")

        self._probe_model = cast(
            ProbeModelType, model
        )  # have before so that energy available to set initial probe
        if self.dset.preprocessed:
            self._probe_model.set_initial_probe(
                self.roi_shape,
                self.reciprocal_sampling,
                self.dset.mean_diffraction_intensity,
                device=self._single_device,
            )
        else:
            # will be set in ptycho.preprocess after dset is preprocessed
            pass
        self._probe_model.to(self._single_device)

    @property
    def constraints(self) -> dict[str, Any]:
        """Get current constraints from all models as a nested dictionary."""
        return {
            "object": self.obj_model.constraints,
            "probe": self.probe_model.constraints,
            "dataset": self.dset.constraints,
            "detector": {
                "detector_mask": getattr(self.detector_model, "detector_mask", None),
            },
        }

    @constraints.setter
    def constraints(self, c: dict[str, Any]):
        """Set constraints by forwarding to individual models.

        Each leaf value may be either a plain ``dict`` (validated per-key against
        the model's constraint dataclass) or a ``Constraints`` dataclass instance
        (assigned wholesale to the model).
        """
        constraint_handlers = {
            "object": self.obj_model,
            "probe": self.probe_model,
            "dataset": self.dset,
        }

        for key, value in c.items():
            if key in constraint_handlers:
                if isinstance(value, Constraints):
                    constraint_handlers[key].constraints = value
                elif isinstance(value, dict):
                    constraint_handlers[key].constraints = value
                else:
                    raise TypeError(
                        f"Constraints for '{key}' must be a dict or Constraints dataclass, "
                        f"got {type(value).__name__}"
                    )
            elif key == "detector" and isinstance(value, dict):
                warn("Detector constraints not implemented, skipping")
            else:
                valid_keys = list(constraint_handlers.keys()) + ["detector"]
                raise KeyError(
                    f"Invalid constraint category '{key}'. Valid categories are: {valid_keys}"
                )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, val: int | None) -> None:
        if val is not None:
            v = validate_gt(validate_int(val, "batch_size"), 0, "batch_size")
            self._batch_size = int(v)

    @property
    def logger(self) -> LoggerPtychography | None:
        return self._logger

    @logger.setter
    def logger(self, logger: LoggerPtychography | None):
        if logger is None:
            self._logger = None
        elif not isinstance(logger, LoggerPtychography) and "logger_pty" not in str(type(logger)):
            raise TypeError(f"Logger must be a LoggerPtychography, got {type(logger)}")

        self._logger = logger

    # endregion --- explicit class properties ---

    # region --- implicit class properties ---

    @property
    def device(self) -> str | list[int]:
        """Returns the active device: 'cuda:X'/'cpu' for single-GPU, or [gpu_ids] for multi-GPU."""
        if self._multi_gpu_devices is not None:
            return self._multi_gpu_devices
        if hasattr(self, "_device"):
            return self._device
        return config.get("device")

    @device.setter
    def device(self, device: str | int | None):
        # allow setting gpu/cpu, but not changing the device from the config gpu device
        if device is not None:
            dev, _id = config.validate_device(device)
            self._device = dev
            try:
                self.to(dev)
            except AttributeError:
                pass

    @property
    def _single_device(self) -> str:
        """Single-device string for internal tensor operations. Always str, never a list."""
        return self._device if hasattr(self, "_device") else str(config.get("device"))

    @property
    def _obj_dtype(self) -> "torch.dtype":
        return self.obj_model.dtype

    @property
    def _dtype_real(self) -> "torch.dtype":
        # necessary because torch doesn't like passing strings to convert dtypes
        return getattr(torch, config.get("dtype_real"))

    @property
    def _dtype_complex(self) -> "torch.dtype":
        return getattr(torch, config.get("dtype_complex"))

    @property
    def obj_cropped(self) -> np.ndarray:
        """Cropped + FOV-rotated object, in its native representation.

        - ``obj_type="complex"`` → complex array (amp * exp(1j*phase)); phase is
          recentered to zero mean here as a defensive duplicate of
          ``ObjectConstraints._apply_hard_complex``.
        - ``obj_type="pure_phase"`` → real array of phase values (already
          recentered upstream by ``_apply_hard_pure_phase``).
        - ``obj_type="potential"`` → real array of potential values.
        """
        cropped = self._crop_rotate_obj_fov(self.obj, padding=self.obj_padding_px)
        if self.obj_type == "complex":
            ph = np.angle(cropped)
            cropped = np.abs(cropped) * np.exp(1j * (ph - ph.mean()))
        return cropped

    @property  # FIXME depend on ptychodataset
    def roi_shape(self) -> np.ndarray:
        return self.dset.roi_shape

    @property  # FIXME depend on ptychodataset
    def gpts(self) -> np.ndarray:
        return self.dset.gpts

    @property
    def reciprocal_sampling(self) -> np.ndarray:
        """
        Units A^-1 or raises error
        """
        sampling = self.dset.detector_sampling
        units = self.dset.detector_units
        if units[0] == "A^-1":
            pass
        elif units[0] == "mrad":
            if self.probe_model.probe_params["energy"] is not None:  # convert mrad -> A^-1
                sampling = (
                    sampling
                    / electron_wavelength_angstrom(self.probe_model.probe_params["energy"])
                    / 1e3
                )
            else:
                raise ValueError("dc units given in mrad but no energy defined to convert to A^-1")
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
    def angular_sampling(self) -> np.ndarray:
        """
        Units mrad or raises error
        """
        sampling = self.dset.detector_sampling
        units = self.dset.detector_units
        if units[0] == "mrad":
            pass
        elif units[0] == "A^-1":
            if self.probe_model.probe_params["energy"] is not None:
                sampling = (
                    sampling
                    * electron_wavelength_angstrom(self.probe_model.probe_params["energy"])
                    * 1e3
                )
            else:
                raise ValueError("dc units given in A^-1 but no energy defined to convert to mrad")
        elif units[0] == "pixels":
            raise ValueError("dset Q units given in pixels, needs calibration")
        else:
            raise NotImplementedError(f"Unknown dset Q units: {units}")
        return sampling

    @property
    def angular_units(self) -> list[str]:
        """Hardcoded to mrad, self.angular_sampling will raise an error if can't get mrad"""
        return ["mrad", "mrad"]

    @property
    def sampling(self) -> np.ndarray:
        """Realspace sampling of the reconstruction. Units of A"""
        return self.dset.obj_sampling

    @property
    def obj_shape_crop(self) -> np.ndarray:
        """All object shapes are 3D"""
        shp = np.floor(self.dset.fov / self.sampling)
        shp += shp % 2
        shp = np.concatenate([[self.num_slices], shp])
        return shp.astype("int")

    @property
    def obj_shape_full(self) -> np.ndarray:
        rotshape = self.dset._obj_shape_full_2d(self.obj_padding_px)
        shape = np.concatenate([[self.num_slices], rotshape])
        return shape

    # endregion --- implicit class properties ---

    # region --- class methods ---
    def vprint(self, m: Any, level: int = 1, *args, **kwargs) -> None:
        """Print messages if verbose is enabled."""
        if self.verbose >= level:
            print(m, *args, **kwargs)

    def _check_preprocessed(self):
        if not self._preprocessed:
            raise AttributeError(
                "Preprocessing has not been completed. Please run Ptycho.preprocess()"
            )

    def _to_numpy(self, array: "np.ndarray | torch.Tensor") -> np.ndarray:
        return to_numpy(array)

    def _to_torch(
        self, array: "np.ndarray | torch.Tensor", dtype: "str | torch.dtype" = "same"
    ) -> "torch.Tensor":
        """
        dtype can be: "same": same as input array, default
                      "object": same as object type, real or complex determined by potential/complex
                      torch.dtype type
        """
        if isinstance(dtype, str):
            dtype = dtype.lower()
            if dtype == "same":
                dt = None
            elif dtype == "probe":
                dt = self._dtype_complex
            elif dtype in ["object", "obj"]:
                if np.iscomplexobj(array):
                    dt = self._dtype_complex
                else:
                    dt = self._dtype_real
            else:
                raise ValueError(
                    f"Unknown string passed {dtype}, dtype should be 'same', 'object' or torch.dtype"
                )
        elif isinstance(dtype, torch.dtype):
            dt = dtype
        else:
            raise TypeError(f"dtype should be string or torch.dtype, got {type(dtype)} {dtype}")

        if isinstance(array, np.ndarray):
            t = torch.tensor(array.copy(), device=self._single_device, dtype=dt)
        elif isinstance(array, torch.Tensor):
            t = array.to(self._single_device)
            if dt is not None:
                t = t.type(dt)
        elif isinstance(array, (list, tuple)):
            t = torch.tensor(array, device=self._single_device, dtype=dt)
        else:
            raise TypeError(f"arr should be ndarray or Tensor, got {type(array)}")
        return t

    def _crop_rotate_obj_fov(
        self,
        array: "np.ndarray",
        positions_px: np.ndarray | None = None,
        com_rotation_rad: float | None = None,
        transpose: bool | None = None,
        padding: np.ndarray | tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Un-rotates and un-transposes the object and crops it to the reconstruction FOV."""
        array = self._to_numpy(array).copy()
        com_rotation_rad = (
            self.dset.com_rotation_rad if com_rotation_rad is None else com_rotation_rad
        )
        transpose = self.dset.com_transpose if transpose is None else transpose

        angle = com_rotation_rad if transpose else -1 * com_rotation_rad

        rotated_array = ndi.rotate(array, np.rad2deg(-angle), order=1, reshape=True, axes=(-2, -1))

        if transpose:
            rotated_array = rotated_array.swapaxes(-2, -1)

        cropped = center_crop_arr(rotated_array, tuple(self.obj_shape_crop), pad_if_needed=False)

        return cropped

    def reset_recon(self) -> None:
        self._reset_rng()
        self.obj_model.reset()
        self.probe_model.reset()
        self.dset.reset()
        self.compute_propagator_arrays()
        # obj_model and its DEFAULT_CONSTRAINTS are correlated at runtime (each object type pairs
        # with its own constraint dataclass), which the union type can't express.
        self.obj_model.constraints = self.obj_model.DEFAULT_CONSTRAINTS  # pyright: ignore[reportAttributeAccessIssue]
        # detector reset if necessary
        self._iter_losses = []
        self._iter_val_losses = []
        self._iter_recon_types = []
        self._iter_lrs = {}
        self._snapshots = []

    def _store_current_iter_snapshot(
        self,
    ) -> None:
        probe = self.probe
        obj = self.obj
        snp = Snapshot(iteration=self.num_iters, obj=obj, probe=probe)
        self._snapshots.append(snp)

    def get_probe_intensities(
        self, probe: "torch.Tensor | np.ndarray | None" = None
    ) -> np.ndarray:
        """Returns the relative probe intensities for each probe in mixed state"""
        if probe is None:
            probe = self.probe
        if probe.ndim == 2:
            return np.array([1.0])
        else:
            probe = self._to_numpy(probe)
            intensities = np.abs(probe) ** 2
            return intensities.sum(axis=(-2, -1)) / intensities.sum()

    def _broadcast_parameters(self, src: int = 0) -> None:
        """Broadcast obj, probe, and dataset parameters from rank src to all other ranks.

        Uses .parameters() so it works for both pixelated and DIP/INR models. The dataset's
        learnable params (scan positions / descan shifts) must also be broadcast: with the
        DistributedSampler partitioning scan positions, the full position params are replicated
        on every rank, so they must start identical and stay synchronized.
        """
        for p in self.obj_model.parameters():
            dist.broadcast(p.data, src=src)
        for p in self.probe_model.parameters():
            dist.broadcast(p.data, src=src)
        for group in self.dset.get_optimization_parameters().values():
            for p in group:
                buf = p.data.contiguous()
                dist.broadcast(buf, src=src)
                p.data.copy_(buf)

    def _all_reduce_gradients(self) -> None:
        """Average obj, probe, and dataset gradients across all ranks (call after backward,
        before step).

        Uses .parameters() so it works for both pixelated and DIP/INR models. The dataset's
        learnable params are included because each scan position's gradient is nonzero on
        exactly one rank, so they must be reduced (AVG) to stay consistent across ranks.
        """
        dset_params = [
            p for group in self.dset.get_optimization_parameters().values() for p in group
        ]
        params = [
            p
            for p in (
                list(self.obj_model.parameters())
                + list(self.probe_model.parameters())
                + dset_params
            )
            if p.grad is not None
        ]
        if params:
            for p in params:
                if p.grad is not None and not p.grad.is_contiguous():
                    p.grad = p.grad.contiguous()
            all_reduce_params(*params)

    def to(self, device: str | int | torch.device):
        dev, _id = config.validate_device(device)
        self._device = dev
        self._multi_gpu_devices = None
        # Sync each sub-model's own device tracker so their reset() uses the correct device
        self.obj_model.device = dev
        self.probe_model.device = dev
        self.obj_model.to(dev)
        self.probe_model.to(dev)
        self.dset.to(dev)
        self._obj_fov_mask = self._to_torch(self._obj_fov_mask)
        self._propagators = self._to_torch(self._propagators)
        self._rng_to_device(dev)

    def _build_dataloaders(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        world_size: int,
        rank: int,
        num_workers: int,
    ) -> "tuple[DataLoader, DistributedSampler | None, DataLoader | None]":
        """Build train + (optional) val DataLoaders for both single- and multi-GPU paths.

        Mirrors the shape of ``DDPMixin.setup_dataloader`` but adapted to ptycho's device
        contract (``str | list[int]``) and ptycho's precomputed ``val_mode`` index split.
        ``world_size > 1`` uses ``DistributedSampler`` over a ``Subset``; ``world_size == 1``
        uses ``shuffle=True`` with a seeded ``torch.Generator`` for run-to-run determinism.
        ``__getitem__`` returns ``{"index": idx, ...}`` for the original dataset index, and
        ``Subset[i]`` calls ``dataset[indices[i]]``, so ``batch["index"]`` is the original
        dataset index under either branch.

        ``self.batch_size`` is the GLOBAL batch: the number of samples contributing to one
        optimizer step across all ranks. Each rank's DataLoader draws ``batch_size //
        world_size``, so the same ``batch_size`` gives the same optimization trajectory (and
        loss curve) on any GPU count.
        """
        pin_memory = self.dset.target_residency == "cpu" and str(self._single_device).startswith(
            "cuda"
        )
        per_rank_batch = self.batch_size
        if world_size > 1:
            per_rank_batch = max(1, self.batch_size // world_size)
            if self.batch_size % world_size != 0:
                warn(
                    f"batch_size={self.batch_size} is not divisible by world_size={world_size}; "
                    f"each rank uses {per_rank_batch}, so the effective global batch is "
                    f"{per_rank_batch * world_size}."
                )
        loader_kwargs: dict[str, Any] = {
            "batch_size": per_rank_batch,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": False,
        }
        if num_workers > 0:
            loader_kwargs.update(
                multiprocessing_context="spawn",
                persistent_workers=True,
                worker_init_fn=worker_init_fn,
            )

        train_subset = torch.utils.data.Subset(self.dset, train_indices.tolist())
        val_subset = (
            torch.utils.data.Subset(self.dset, val_indices.tolist())
            if len(val_indices) > 0
            else None
        )

        if world_size > 1:
            train_sampler = DistributedSampler(
                train_subset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=int(self.rng.integers(0, 2**31 - 1)),
                drop_last=False,
            )
            train_loader = DataLoader(train_subset, sampler=train_sampler, **loader_kwargs)
            if val_subset is not None:
                val_sampler = DistributedSampler(
                    val_subset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
                val_loader = DataLoader(val_subset, sampler=val_sampler, **loader_kwargs)
            else:
                val_loader = None
        else:
            train_sampler = None
            shuffle_gen = torch.Generator().manual_seed(int(self.rng.integers(0, 2**31 - 1)))
            train_loader = DataLoader(
                train_subset, shuffle=True, generator=shuffle_gen, **loader_kwargs
            )
            val_loader = (
                DataLoader(val_subset, shuffle=False, **loader_kwargs)
                if val_subset is not None
                else None
            )

        return train_loader, train_sampler, val_loader

    # endregion

    # region --- ptychography forward model ---

    def forward_operator(
        self,
        obj_patches: torch.Tensor,
        shifted_input_probes: torch.Tensor,
        descan: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.probe_model.learn_probe_tilt:
            self.compute_propagator_arrays()
        propagated_probes, overlap = self.overlap_projection(obj_patches, shifted_input_probes)
        ## prop_probes shape: (nslices, nprobes, batch_size, roi_shape[0], roi_shape[1])
        ## overlap shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        if descan is not None:
            shifts = fourier_translation_operator(descan, tuple(self.roi_shape))
            overlap *= shifts[None]
        return propagated_probes, overlap

    def error_estimate(
        self,
        pred_intensities: torch.Tensor,
        targets: torch.Tensor,
        global_n: int | None = None,
    ) -> torch.Tensor:
        """Data-fidelity loss for one batch via the active criterion (``self.criterion``).

        Maps predictions into the criterion's measurement space (amplitude or intensity),
        applies the detector mask, evaluates the criterion, and normalizes by the mean
        diffraction intensity. Which comparison is used (L2/L1/smooth-L1/Poisson/S3IM/...) is
        entirely the criterion's concern; see ``ptycho_losses``.
        """
        criterion = self.criterion
        if criterion.target_space == "amplitude":
            preds = torch.sqrt(pred_intensities + 1e-9)  # eps avoids diverging gradients at 0
        else:
            preds = pred_intensities

        mask = self.dset.detector_mask
        n = global_n if global_n is not None else self.dset.num_positions
        error = criterion(preds * mask, targets * mask, n)
        loss = error / self.dset.mean_diffraction_intensity
        return loss

    def overlap_projection(self, obj_patches, input_probe):
        """Multiplies `input_probes` with roi-shaped patches from `obj_array`.
        This version is for GD only -- AD does not require all the propagated probe
        slices and trying to store them causes in-place issues
        """
        propagated_probes = [input_probe]
        overlap = obj_patches[0] * input_probe
        for s in range(1, self.num_slices):
            propagated_probe = self._propagate_array(overlap, self._propagators[s - 1])
            overlap = obj_patches[s] * propagated_probe
            propagated_probes.append(propagated_probe)

        propagated_probes = torch.stack(propagated_probes, dim=0).to(overlap.device)
        return propagated_probes, overlap  # type:ignore

    def estimate_amplitudes(
        self, overlap_array: "torch.Tensor", corner_centered: bool = False
    ) -> "torch.Tensor":
        """Returns the estimated fourier amplitudes from real-valued `overlap_array`."""
        # overlap shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        # incoherent sum of all probe components
        eps = 1e-9  # this is to avoid diverging gradients at sqrt(0)
        overlap_fft = torch.fft.fft2(overlap_array, norm="ortho")
        amps = torch.sqrt(torch.sum(torch.abs(overlap_fft) ** 2, dim=0) + eps)
        if not corner_centered:  # default is shifted amplitudes matching exp data
            return torch.fft.fftshift(amps, dim=(-2, -1))
        else:
            return amps

    def estimate_intensities(self, overlap_array: "torch.Tensor") -> "torch.Tensor":
        """Returns the estimated fourier intensities from real-valued `overlap_array`."""
        # overlap shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        # incoherent sum of all probe components
        overlap_fft = torch.fft.fft2(overlap_array, norm="ortho")
        return torch.sum(torch.abs(overlap_fft) ** 2, dim=0)

    def _propagate_array(
        self, array: "torch.Tensor", propagator_array: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Propagates array by Fourier convolving array with propagator_array.

        Parameters
        ----------
        array: np.ndarray
            Wavefunction array to be convolved
        propagator_array: np.ndarray
            Propagator array to convolve array with

        Returns
        -------
        propagated_array: np.ndarray
            Fourier-convolved array
        """
        propagated = torch.fft.ifft2(torch.fft.fft2(array) * propagator_array)
        return propagated

    def compute_propagator_arrays(self):
        self.propagators = self.probe_model._compute_propagator_arrays(
            self.sampling, self.num_slices, self.slice_thicknesses
        )

    # endregion


# misc helpers to maybe move elsewhere


def adjust_padding_power2(pad, shape, power2_level):
    """
    Adjusts pad so that (shape + 2*pad) is divisible by 2**power2_level.
    """
    div = 2**power2_level
    rem0 = (shape[-2] + 2 * pad[-2]) % div
    rem1 = (shape[-1] + 2 * pad[-1]) % div
    if rem0 != 0:
        pad[-2] += (div - rem0) // 2
    if rem1 != 0:
        pad[-1] += (div - rem1) // 2

    if ((shape[-2] + 2 * pad[-2]) % div != 0) or ((shape[-1] + 2 * pad[-1]) % div != 0):
        raise ValueError(f"Adjustment failed to achieve divisibility by {div}")
    return pad
