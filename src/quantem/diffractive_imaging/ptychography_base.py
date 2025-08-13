from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import scipy.ndimage as ndi

import quantem.core.utils.array_funcs as arr
from quantem.core import config
from quantem.core.datastructures import Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
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
from quantem.diffractive_imaging.object_models import ObjectBase, ObjectModelType, ObjectPixelated
from quantem.diffractive_imaging.probe_models import ProbeBase, ProbeModelType
from quantem.diffractive_imaging.ptycho_utils import (
    AffineTransform,
    center_crop_arr,
    fourier_translation_operator,
    sum_patches,
)

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


"""
design patterns:
    - all outward facing properties ptycho.blah will give numpy arrays
        - hidden attributes, ptycho._blah will be torch, living on cpu/gpu depending on config
    - objects are always 3D, if doing a singleslice recon, the shape is just [1, :, :]
    - likewise probes are always stacks for mixed state, if single probe, then shape is [1, :, :]
    - all preprocessing will be done with torch tensors 
"""


class PtychographyBase(AutoSerialize):
    """
    A base class for performing phase retrieval using the Ptychography algorithm.

    This class provides a basic framework for performing phase retrieval using the Ptychography algorithm.
    It is designed to be subclassed by specific Ptychography algorithms.
    """

    DEFAULT_CONSTRAINTS = {
        "object": {
            "fix_potential_baseline": False,
            "identical_slices": False,
            "apply_fov_mask": False,
            "tv_weight_yx": 0.0,
            "tv_weight_z": 0.0,
            "surface_zero_weight": 0.0,
        },
        "probe": {
            "fix_probe": False,
            "fix_probe_com": False,
        },
        "dataset": {
            "descan_tv_weight": 0.0,
            "descan_shifts_constant": False,
        },
        "detector": {
            "detector_mask": None,
        },
    }

    _token = object()

    def __init__(  # TODO prevent direct instantiation
        self,
        dset: DatasetModelType,
        obj_model: ObjectModelType,
        probe_model: ProbeModelType,
        detector_model: DetectorModelType,
        device: str | int = "cpu",  # "gpu" | "cpu" | "cuda:X"
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
        _token: None | object = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use Dataset.from_array() to instantiate this class.")

        if not config.get("has_torch"):
            raise RuntimeError("the quantEM Ptychography module requires torch to be installed.")

        self.verbose = verbose
        self.dset = dset
        self.device = device
        self.rng = rng

        # initializing default attributes
        self._preprocessed: bool = False
        self._obj_padding_force_power2_level: int = 3
        self._store_iterations: bool = False
        self._store_iterations_every: int = 1
        self._epoch_losses: list[float] = []
        self._epoch_recon_types: list[str] = []
        self._epoch_lrs: dict[str, list] = {}  # LRs/step_sizes across epochs
        self._epoch_snapshots: list[dict[str, int | np.ndarray]] = []
        self._constraints = self.DEFAULT_CONSTRAINTS.copy()
        self._obj_padding_px = np.array([0, 0])
        self.obj_fov_mask = torch.ones(self.dset._obj_shape_full_2d(self.obj_padding_px).shape)

        self._schedulers = {}
        self._optimizers = {}
        self._scheduler_params = {}
        self._optimizer_params = {}

        self.set_probe_model(probe_model)
        self.set_obj_model(obj_model)
        self.detector_model = detector_model
        self._compute_propagator_arrays()
        self.to(self.device)

    # region --- preprocessing ---
    ## hopefully will be able to remove some of thes preprocessing flags,
    ## convert plotting and vectorized to kwargs
    ## could also force users to initialize object and probe models externally, but I prefer
    ## having the flexibility of passing the types in here and initializing them internally
    def preprocess(
        self,
        obj_model: ObjectModelType | type | None = None,
        probe_model: ProbeModelType | type | None = None,
        obj_padding_px: tuple[int, int] = (0, 0),
        com_fit_function: Literal[
            "none", "plane", "parabola", "bezier_two", "constant"
        ] = "constant",
        force_com_rotation: float | None = None,
        force_com_transpose: bool | None = None,
        padded_diffraction_intensities_shape: tuple[int, int] | None = None,
        plot_rotation: bool = True,
        plot_com: str | bool = True,
        plot_probe_overlap: bool = False,
        vectorized: bool = True,
    ):
        """
        Rather than passing 100 flags here, I'm going to suggest that if users want to run very
        customized pre-processing, they just call the functions themselves directly.
        """
        self.obj_padding_px = obj_padding_px
        if not self.dset.preprocessed:
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
        else:
            # change obj_padding_px and whatever else needs to be changed
            self.dset._set_initial_scan_positions_px(self.obj_padding_px)
            self.dset._set_patch_indices(self.obj_padding_px)

        if probe_model is not None:
            self.set_probe_model(probe_model)

        if obj_model is not None:
            self.set_obj_model(obj_model)
        # else:
        #     self._obj_model.shape = tuple(self.obj_shape_full)
        #     self._obj_model.reset()

        self._compute_propagator_arrays()
        self._set_obj_fov_mask()
        self._preprocessed = True
        self.reset_recon()  # force clear losses and everything
        return self

    def _compute_propagator_arrays(
        self,
        theta_r: float | None = None,
        theta_c: float | None = None,
    ):
        """
        Precomputes propagator arrays complex wave-function will be convolved by,
        for all slice thicknesses.

        Parameters
        ----------
        theta_r: float, optional
            tilt of propagator in mrad around the row-axis
        theta_c: float, optional
            tilt of propagator in mrad around the column-axis

        Returns
        -------
        propagator_arrays: np.ndarray
            (T,Sr,Sc) shape array storing propagator arrays
        """

        if self.num_slices == 1:
            self.propagators = torch.tensor([])
            return

        kr, kc = tuple(torch.fft.fftfreq(n, d) for n, d in zip(self.roi_shape, self.sampling))
        k2 = (kr[:, None] ** 2 + kc[None] ** 2).to(torch.complex64)  # broadcasting to match shape
        wavelength = electron_wavelength_angstrom(self.probe_model.probe_params["energy"])
        propagators = torch.empty(
            (self.num_slices - 1, kr.shape[0], kc.shape[0]), dtype=torch.complex64
        )

        # TODO vectorize -- allow for optimizing over tilts
        for i, dz in enumerate(self.slice_thicknesses):
            phase_factor = torch.tensor(-1.0j * np.pi * wavelength * dz)
            propagators[i] = torch.exp(phase_factor * k2)

            if theta_r is not None:
                propagators[i] *= torch.exp(
                    1.0j * (-2 * kr[:, None] * np.pi * dz * np.tan(theta_r / 1e3))
                )

            if theta_c is not None:
                propagators[i] *= torch.exp(
                    1.0j * (-2 * kc[None] * np.pi * dz * np.tan(theta_c / 1e3))
                )

        self.propagators = propagators
        return

    def _set_obj_fov_mask(self, gaussian_sigma: float = 2.0, batch_size=None):
        overlap = self._get_probe_overlap(batch_size)
        ov = overlap > overlap.max() * 0.3
        ov = ndi.binary_closing(ov, iterations=5)
        ov = ndi.binary_dilation(ov, iterations=min(32, np.min(self.obj_padding_px) // 4))
        ov = ndi.gaussian_filter(ov.astype(config.get("dtype_real")), sigma=gaussian_sigma)
        self.obj_fov_mask = ov
        self.obj_model.mask = ov
        return

    def _get_probe_overlap(self, max_batch_size: int | None = None) -> np.ndarray:
        prb = self.probe_model.probe[0]
        num_dps = int(np.prod(self.gpts))
        shifted_probes = prb.expand(num_dps, *self.roi_shape)

        batch_size = num_dps if max_batch_size is None else int(max_batch_size)
        probe_overlap = torch.zeros(
            tuple(self.obj_shape_full[-2:]), dtype=self._dtype_real, device=self.device
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
        if self.num_epochs > 0 and new_obj_type != self.obj_model.obj_type and not force:
            raise ValueError(
                "Cannot change object type after training. Run with reset=True or rerun preprocess."
            )
        self.obj_model.obj_type = new_obj_type

    @property
    def num_slices(self) -> int:
        """if num_slices > 1, then it is multislice reconstruction"""
        return self.obj_model.num_slices

    @property
    def propagators(self) -> np.ndarray:
        if self.num_slices == 1:
            return np.array([])
        else:
            return self._to_numpy(self._propagators)

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
        return self._to_numpy(self._obj_model.slice_thicknesses)
        # return self._to_numpy(self._slice_thicknesses)

    @slice_thicknesses.setter
    def slice_thicknesses(self, val: float | Sequence | None) -> None:
        self._obj_model.slice_thicknesses = val
        if hasattr(self, "_propagators"):  # propagators already set, update with new slices
            self._compute_propagator_arrays()

    @property
    def verbose(self) -> int:
        return self._verbose

    @verbose.setter
    def verbose(self, v: bool | int | float) -> None:
        self._verbose = validate_int(validate_gt(v, -1, "verbose"), "verbose")

    @property
    def obj(self) -> np.ndarray:
        return self._to_numpy(self.obj_model.obj)

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
                self.dset._obj_shape_full_2d(),
                self._obj_padding_force_power2_level,
            )
        self._obj_padding_px = p2
        self.obj_model.shape = tuple(self.obj_shape_full)
        self.dset._set_initial_scan_positions_px(self.obj_padding_px)
        self.dset._set_patch_indices(self.obj_padding_px)

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
    def epoch_losses(self) -> np.ndarray:
        """
        Loss/MSE error for each epoch regardless of reconstruction method used
        """
        return np.array(self._epoch_losses)

    @property
    def num_epochs(self) -> int:
        """
        Number of epochs for which the recon has been run so far
        """
        return len(self.epoch_losses)

    @property
    def epoch_recon_types(self) -> np.ndarray:
        """
        Keeping track of what reconstruction type was used
        """
        return np.array(self._epoch_recon_types)

    @property
    def epoch_lrs(self) -> dict[str, np.ndarray]:
        """
        List of step sizes/LRs depending on recon type
        """
        return {k: np.array(v) for k, v in self._epoch_lrs.items()}

    @property
    def probe(self) -> np.ndarray:
        """Complex valued probe(s). Shape [num_probes, roi_reight, roi_width]"""
        return self._to_numpy(self.probe_model.probe)

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, rng: np.random.Generator | int | None):
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, (int, float)):
            rng = np.random.default_rng(rng)
        elif not isinstance(rng, np.random.Generator):
            raise TypeError(f"rng should be a np.random.Generator or a seed, got {type(rng)}")
        self._rng = rng
        seed = rng.bit_generator._seed_seq.entropy  # type:ignore ## get the seed from the generator
        self._rng_torch = torch.Generator(device=self.device).manual_seed(seed % 2**32)

    @property
    def store_iterations(self) -> bool:
        return self._store_iterations

    @store_iterations.setter
    def store_iterations(self, val: bool | None) -> None:
        if val is not None:
            self._store_iterations = bool(val)

    @property
    def store_iterations_every(self) -> int:
        return self._store_iterations_every

    @store_iterations_every.setter
    def store_iterations_every(self, val: int | None) -> None:
        if val is not None:
            self._store_iterations_every = int(val)

    @property
    def epoch_snapshots(self) -> list[dict[str, int | np.ndarray]]:
        return self._epoch_snapshots

    def get_snapshot_by_iter(self, iteration: int):
        iteration = int(iteration)
        for snapshot in self.epoch_snapshots:
            if snapshot["iteration"] == iteration:
                return snapshot
        raise ValueError(f"No snapshot found at iteration: {iteration}")

    @property
    def obj_model(self) -> ObjectModelType:
        return self._obj_model

    # @obj_model.setter
    # def obj_model(self, *args):
    #     raise AttributeError("Use tycho.set_obj_model to set the obj_model")

    def set_obj_model(
        self,
        model: ObjectModelType | type | None,
        num_slices: int | None = None,
        slice_thicknesses: float | Sequence | None = None,
        obj_type: Literal["complex", "pure_phase", "potential"] = "complex",
    ):
        # TODO test with calling before preprocess, pass in "pixelized" or similar
        # TODO -- here can transfer obj from existing to new model if applicable?
        if model is None:
            if hasattr(self, "_obj_model"):
                return
            else:
                raise ValueError("obj_model must be a subclass of ObjectModelType")

        if isinstance(model, type):
            if not issubclass(model, ObjectModelType):
                raise TypeError(
                    f"obj_model must be a subclass of ObjectModelType, got {type(model)}"
                )
            if issubclass(model, ObjectPixelated):
                print("initializing new model obj")
                num_slices = self.num_slices if num_slices is None else int(num_slices)
                self._obj_model = model(
                    num_slices=num_slices,
                    slice_thicknesses=slice_thicknesses,
                    device=self.device,
                    obj_type=obj_type,
                )
            else:
                raise TypeError("please pre-intialize the object model")
        # autoreload bug leads to type issues, so checking str also
        elif isinstance(model, ObjectBase) or "object" in str(type(model)):
            self._obj_model = model
        else:
            raise TypeError(f"obj_modelect must be a ObjectModelType, got {type(model)}")

        # setting object shape manually here as haven't yet set slices
        rotshape = self.dset._obj_shape_full_2d(self.obj_padding_px)
        obj_shape_full = (self.num_slices, int(rotshape[0]), int(rotshape[1]))
        self._obj_model.shape = obj_shape_full
        self._obj_model.reset()

    @property
    def probe_model(self) -> ProbeModelType:
        return self._probe_model

    def set_probe_model(
        self,
        probe_model: ProbeModelType | type | None,
        num_probes: int = 1,
        probe_params: dict[str, float] = {},
        vacuum_probe_intensity: np.ndarray | Dataset4dstem | None = None,
        initial_probe: np.ndarray | None = None,
    ):
        if probe_model is None:
            if hasattr(self, "_probe_model"):
                return
            else:
                raise ValueError("probe_model must be a subclass of ProbeModelType")

        if isinstance(probe_model, type):
            if not issubclass(probe_model, ProbeModelType):
                raise TypeError(
                    f"probe_model must be a subclass of ProbeModelType, got {type(probe_model)}"
                )

            self._probe_model = probe_model(
                num_probes=num_probes,
                probe_params=probe_params,
                vacuum_probe_intensity=vacuum_probe_intensity,
                initial_probe_array=initial_probe,
                device=self.device,
                rng=self.rng,
            )
        # autoreload bug leads to type issues
        elif isinstance(probe_model, ProbeBase) or "probe" in str(type(probe_model)):
            # add protections for changing num_probes and such
            self._probe_model = probe_model
        else:
            raise TypeError(f"probe_model must be a ProbeModelType, got {type(probe_model)}")

        self._probe_model.set_initial_probe(
            self.roi_shape, self.reciprocal_sampling, self.dset.mean_diffraction_intensity
        )
        self._probe_model.to(self.device)
        self._probe_model.constraints = self._constraints["probe"]

    @property
    def constraints(self) -> dict[str, Any]:
        return self._constraints

    @constraints.setter
    def constraints(self, c: dict[str, Any]):
        """Sets both self._constraints as well as the constraints in the object and probe models"""
        for key, value in c.items():
            if key not in self.DEFAULT_CONSTRAINTS:
                raise KeyError(
                    f"Invalid constraint key '{key}', allowed keys are {list(self.DEFAULT_CONSTRAINTS.keys())}"
                )

            if not isinstance(value, dict):
                raise ValueError(f"Constraint '{key}' must be a dictionary.")

            allowed_subkeys = self.DEFAULT_CONSTRAINTS[key].keys()
            for subkey, subvalue in value.items():
                if subkey not in allowed_subkeys:
                    raise KeyError(
                        f"Invalid subkey '{subkey}' for constraint '{key}', allowed subkeys are {list(allowed_subkeys)}"
                    )

                self._constraints[key][subkey] = subvalue
        for k, v in self._constraints["object"].items():
            if k in self.obj_model.DEFAULT_CONSTRAINTS.keys():
                self.obj_model.add_constraint(k, v)
        for k, v in self._constraints["probe"].items():
            if k in self.probe_model.DEFAULT_CONSTRAINTS.keys():
                self.probe_model.add_constraint(k, v)
        for k, v in self._constraints["dataset"].items():
            if k in self.dset.DEFAULT_CONSTRAINTS.keys():
                self.dset.add_constraint(k, v)

    # endregion --- explicit class properties ---

    # region --- implicit class properties ---

    @property
    def device(self) -> str:
        """This should be of form 'cuda:X' or 'cpu', as defined by quantem.config"""
        if hasattr(self, "_device"):
            return self._device
        else:
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
        cropped = self._crop_rotate_obj_fov(self.obj)
        if self.obj_type == "pure_phase":
            cropped = np.exp(1j * np.angle(cropped))
        cropped = center_crop_arr(cropped, tuple(self.obj_shape_crop))  # sometimes 1 pixel off
        # TEMP testing for bugs
        if cropped.shape != tuple(self.obj_shape_crop):
            raise ValueError(
                f"Object shape {cropped.shape} does not match expected shape {self.obj_shape_crop}"
            )
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

    def _check_rm_preprocessed(self, new_val: Any, name: str) -> None:
        if hasattr(self, name):
            if getattr(self, name) != new_val:
                self._preprocessed = False

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
            t = torch.tensor(array.copy(), device=self.device, dtype=dt)
        elif isinstance(array, torch.Tensor):
            t = array.to(self.device)
            if dt is not None:
                t = t.type(dt)
        elif isinstance(array, (list, tuple)):
            t = torch.tensor(array, device=self.device, dtype=dt)
        else:
            raise TypeError(f"arr should be ndarray or Tensor, got {type(array)}")
        return t

    def _crop_rotate_obj_fov(
        self,
        array: "np.ndarray",
        positions_px: np.ndarray | None = None,
        com_rotation_rad: float | None = None,
        transpose: bool | None = None,
        padding: int = 0,
    ) -> np.ndarray:
        """
        Crops and rotated object to FOV bounded by current pixel positions.
        """
        array = self._to_numpy(array).copy()
        com_rotation_rad = (
            self.dset.com_rotation_rad if com_rotation_rad is None else com_rotation_rad
        )
        transpose = self.dset.com_transpose if transpose is None else transpose

        angle = com_rotation_rad if transpose else -1 * com_rotation_rad

        if positions_px is None:
            positions = self.dset.scan_positions_px.cpu().detach().numpy()
        else:
            positions = positions_px

        tf = AffineTransform(angle=angle)
        rotated_points = tf(positions, origin=positions.mean(0))

        min_x, min_y = np.floor(np.amin(rotated_points, axis=0) - padding).astype("int")
        min_x = min_x if min_x > 0 else 0
        min_y = min_y if min_y > 0 else 0
        max_x, max_y = np.ceil(np.amax(rotated_points, axis=0) + padding).astype("int")

        rotated_array = ndi.rotate(
            array, np.rad2deg(-angle), order=1, reshape=False, axes=(-2, -1)
        )[..., min_x:max_x, min_y:max_y]

        if transpose:
            rotated_array = rotated_array.swapaxes(-2, -1)

        return rotated_array

    def _repeat_arr(
        self, arr: "np.ndarray|torch.Tensor", repeats: int, axis: int
    ) -> "np.ndarray|torch.Tensor":
        """repeat the input array along the desired axis."""
        if config.get("has_torch"):
            if isinstance(arr, torch.Tensor):
                return torch.repeat_interleave(arr, repeats, dim=axis)
        return np.repeat(arr, repeats, axis=axis)

    def reset_recon(self) -> None:
        self.obj_model.reset()
        self.probe_model.reset()
        self.dset.reset()
        self._epoch_losses = []
        self._epoch_recon_types = []
        self._epoch_snapshots = []
        self._epoch_lrs = {}
        self.constraints = self.DEFAULT_CONSTRAINTS.copy()

    def append_recon_iteration(
        self,
        obj: "torch.Tensor | np.ndarray | None" = None,
        probe: "torch.Tensor | np.ndarray | None" = None,
    ) -> None:
        if probe is None:
            probe = self.probe
        else:
            probe = self._to_numpy(probe)
        if obj is None:
            obj = self.obj
        else:
            obj = self._to_numpy(obj)
        self._epoch_snapshots.append(
            {
                "iteration": self.num_epochs,
                "obj": obj,
                "probe": probe,
            }
        )
        return

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

    def save(
        self,
        path: str | Path,
        mode: Literal["w", "o"] = "w",
        store: Literal["auto", "zip", "dir"] = "auto",
        skip: str | type | Sequence[str | type] = (),
        compression_level: int | None = 4,
    ):
        if isinstance(skip, (str, type)):
            skip = [skip]
        skip = list(skip)
        skips = skip + [torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]
        super().save(
            path,
            mode=mode,
            store=store,
            compression_level=compression_level,
            # skip=["optimizers", "_optimizers"],
            # skip=torch.optim.Optimizer,
            skip=skips,
        )

    def to(self, device: str | int | torch.device):
        dev, _id = config.validate_device(device)
        if dev != self.device:
            self._device = dev
        self.obj_model.to(dev)
        self.probe_model.to(dev)
        self.dset.to(dev)
        self._obj_fov_mask = self._to_torch(self._obj_fov_mask)
        self._propagators = self._to_torch(self._propagators)

    # endregion

    # region --- ptychography foRcard model ---

    def forward_operator(
        self,
        obj_patches: torch.Tensor,
        shifted_input_probes: torch.Tensor,
        descan: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        batch_indices: np.ndarray,
        amplitude_error: bool = True,
        use_unshifted: bool = True,
        loss_type: Literal["l1", "l2"] = "l2",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if amplitude_error:
            if use_unshifted:
                targets = self.dset.amplitudes[batch_indices]
            else:
                targets = self.dset.centered_amplitudes[batch_indices]
            preds = torch.sqrt(pred_intensities + 1e-9)  # add eps to avoid diverging gradients
        else:
            if use_unshifted:
                targets = self.dset.intensities[batch_indices]
            else:
                targets = self.dset.centered_intensities[batch_indices]
            preds = pred_intensities
        if loss_type == "l1":
            error = arr.sum(arr.abs(preds - targets))
        elif loss_type == "l2":
            error = arr.sum(arr.abs(preds - targets) ** 2)
        else:
            raise ValueError(f"Unknown loss type {loss_type}, should be 'l1' or 'l2'")
        loss = error / self.dset.mean_diffraction_intensity / targets.shape[0]
        return loss, targets

    # def error_estimate(
    #     self,
    #     overlap: torch.Tensor,
    #     true_amplitudes: torch.Tensor,
    # ) -> torch.Tensor:
    #     farfield_amplitudes = self.estimate_amplitudes(overlap)
    #     raw_error = arr.sum(arr.abs(farfield_amplitudes - true_amplitudes) ** 2)
    #     ave_error = raw_error / farfield_amplitudes.shape[0]  # normalize by # patterns
    #     return ave_error

    def error_estimate_intensities(
        self,
        overlap: torch.Tensor,
        true_intensities: torch.Tensor,
    ) -> torch.Tensor:
        farfield_intensities = self.estimate_intensities(overlap)
        raw_error = arr.sum(arr.abs(farfield_intensities - true_intensities) ** 2)
        ave_error = raw_error / farfield_intensities.shape[0]  # normalize by # patterns
        return ave_error

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

        return arr.match_device(propagated_probes, overlap), overlap  # type:ignore

    def estimate_amplitudes(
        self, overlap_array: "torch.Tensor", corner_centered: bool = False
    ) -> "torch.Tensor":
        """Returns the estimated fourier amplitudes from real-valued `overlap_array`."""
        # overlap shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        # incoherent sum of all probe components
        eps = 1e-9  # this is to avoid diverging gradients at sqrt(0)
        overlap_fft = torch.fft.fft2(overlap_array)
        amps = torch.sqrt(torch.sum(torch.abs(overlap_fft + eps) ** 2, dim=0))
        if not corner_centered:  # default is shifted amplitudes matching exp data
            return torch.fft.fftshift(amps, dim=(-2, -1))
        else:
            return amps

    def estimate_intensities(self, overlap_array: "torch.Tensor") -> "torch.Tensor":
        """Returns the estimated fourier amplitudes from real-valued `overlap_array`."""
        # overlap shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        # incoherent sum of all probe components
        overlap_fft = torch.fft.fft2(overlap_array)
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
