from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Callable, Self, Union, cast
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.datastructures import Dataset2d, Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.blocks import reset_weights
from quantem.core.ml.constraints import BaseConstraints, Constraints, parse_constraint_dict
from quantem.core.ml.loss_functions import get_loss_module
from quantem.core.ml.optimizer_mixin import (
    OptimizerMixin,
    OptimizerParams,
    OptimizerParamsType,
    SchedulerParamsType,
)
from quantem.core.utils.rng import RNGMixin
from quantem.core.utils.utils import electron_wavelength_angstrom, to_numpy
from quantem.core.utils.validators import (
    validate_arr_gt,
    validate_array,
    validate_dict_keys,
    validate_gt,
    validate_np_len,
    validate_tensor,
)
from quantem.core.visualization import show_2d
from quantem.diffractive_imaging._natural_neighbors_interpolation import (
    beamlet_weights,
    fourier_beamlet_weights,
    one_hot_beamlet_weights,
)
from quantem.diffractive_imaging.complex_probe import (
    POLAR_ALIASES,
    POLAR_SYMBOLS,
    fourier_space_probe,
    real_space_probe,
    spatial_frequencies,
    standardize_aberration_coefs,
)
from quantem.diffractive_imaging.ptycho_utils import (
    fourier_shift_expand,
    shift_array,
)

DeviceType = Union[str, torch.device, int]


class PtychoProbeConstraintParams:
    """
    Namespace class for ptychography probe constraint dataclasses.

    Tab-complete on ``PtychoProbeConstraintParams`` in a notebook to discover the
    available variants. Tab-complete inside a variant's constructor to see every
    constraint field with its default value.

    Variants
    --------
    Raster
        Constraints for grid-based probe representations (``ProbePixelated`` and
        ``ProbeDIP`` share this set today).
    Parametric
        Placeholder for parametric probe models, where Gram-Schmidt orthogonalization
        and pixel-domain TV are moot.
    """

    @dataclass
    class Raster(Constraints):
        """Constraints for grid-based ptychography probe models (``ProbePixelated``,
        ``ProbeDIP``).

        Attributes
        ----------
        orthogonalize_probe : bool, default ``True``
            Mixed-state probe (``num_probes > 1``) only. After each update applies
            Gram-Schmidt orthogonalization across the probe stack and then sorts
            the resulting probes by total intensity (descending). For
            ``num_probes == 1`` this is effectively a renormalization no-op.
        center_probe : bool, default ``False``
            Shifts the probe's intensity center-of-mass back to the array center
            via a Fourier shift after each update. Useful when probe drift
            competes with scan-position refinement; if both move freely the
            reconstruction can wander while still fitting the diffraction data.
        tv_weight : float, default ``0.0``
            Soft penalty. Weight on the in-plane total-variation of the (complex)
            probe; encourages smooth probe magnitude / phase.
        """

        # hard constraints
        orthogonalize_probe: bool = True
        center_probe: bool = False
        # soft constraints
        tv_weight: float = 0.0
        _name: str = "raster"

        soft_constraint_keys = ["tv_weight"]
        hard_constraint_keys = ["orthogonalize_probe", "center_probe"]

    @dataclass
    class Parametric(Constraints):
        """Placeholder for parametric probe constraints (``ProbeParametric``).

        Parametric probes are pure functions of aberration / aperture coefficients,
        so pixel-domain projections like ``orthogonalize_probe`` and ``tv_weight``
        don't apply. Parametric-specific fields (e.g. bounds on individual
        aberration coefficients) will land here when needed.
        """

        _name: str = "parametric"

        soft_constraint_keys = []
        hard_constraint_keys = []

    @classmethod
    def parse_dict(cls, d: dict) -> "PtychoProbeConstraintsType":
        """Instantiate the appropriate variant from a config dict.

        The dict must contain a ``'name'`` or ``'type'`` key (case-insensitive),
        with value ``'raster'`` or ``'parametric'``. All other keys are forwarded
        as keyword arguments to the chosen dataclass.
        """
        return cast(PtychoProbeConstraintsType, parse_constraint_dict(cls, d, kind="probe"))


PtychoProbeConstraintsType = (
    PtychoProbeConstraintParams.Raster | PtychoProbeConstraintParams.Parametric
)


class ProbeBase(nn.Module, RNGMixin, OptimizerMixin, AutoSerialize):
    DEFAULT_PROBE_PARAMS = {
        "energy": None,
        "defocus": None,
        "semiangle_cutoff": None,
        "soft_edges": True,
        "aberration_coefs": {},
    }
    DEFAULT_LRS = {
        "probe": 1e-3,
    }
    _token = object()

    def __init__(
        self,
        num_probes: int = 1,
        probe_params: dict = {},
        probe_tilt: tuple[float, float] | torch.Tensor = (0, 0),
        learn_probe_tilt: bool = False,
        roi_shape: tuple[int, int] | np.ndarray | None = None,
        device: DeviceType = "cpu",
        rng: np.random.Generator | int | None = None,
        max_aberrations_order=5,
        _token: object | None = None,
        *args,
        **kwargs,
    ):
        if _token is not self._token:
            raise RuntimeError("Use a factory method to instantiate this class.")
        # Initialize nn.Module first
        nn.Module.__init__(self)
        RNGMixin.__init__(self, rng=rng, device=device)
        OptimizerMixin.__init__(self)

        self.num_probes = num_probes
        self.device = device
        self._probe_params = self.DEFAULT_PROBE_PARAMS
        self._max_aberrations_order = max_aberrations_order
        self.probe_params = probe_params
        self._constraints = {}
        self.rng = rng
        self._probe_tilt = nn.Parameter(
            torch.tensor(probe_tilt, dtype=getattr(torch, config.get("dtype_real"))),
            requires_grad=learn_probe_tilt,
        )
        self.learn_probe_tilt = learn_probe_tilt
        self._initial_probe_tilt = torch.tensor(
            probe_tilt, dtype=getattr(torch, config.get("dtype_real"))
        )
        if roi_shape is not None:
            self.roi_shape = roi_shape

    def get_optimization_parameters(self) -> "dict[str, list[torch.Tensor]]":
        """Get the parameters that should be optimized for this model, keyed by group."""
        params = self.params
        if params is None:
            return {}
        return {self.DEFAULT_OPTIMIZER_KEY: list(params)}

    @property
    def learn_probe_tilt(self) -> bool:
        return self._learn_probe_tilt

    @learn_probe_tilt.setter
    def learn_probe_tilt(self, learn_probe_tilt: bool):
        with torch.no_grad():
            self._learn_probe_tilt = bool(learn_probe_tilt)
            self._probe_tilt.requires_grad = learn_probe_tilt

    @property
    def probe_tilt(self) -> nn.Parameter:
        """tilt of the probe in mrad"""
        return self._probe_tilt

    @probe_tilt.setter
    def probe_tilt(self, tilt: torch.Tensor | tuple[float, float]):
        tilt = validate_tensor(
            tilt,
            name="probe_tilt",
            dtype=getattr(torch, config.get("dtype_real")),
            shape=(2,),
        )
        self._probe_tilt.data = tilt.to(self.device)

    @property
    def shape(self) -> np.ndarray:
        return to_numpy((self.num_probes, *self.roi_shape))

    @property
    def roi_shape(self) -> np.ndarray:
        """shape of the probe"""
        return self._roi_shape

    @roi_shape.setter
    def roi_shape(self, shape: tuple[int, int] | np.ndarray) -> None:
        arr = validate_array(
            shape,
            name="roi_shape",
            shape=(2,),
        )
        arr = validate_arr_gt(arr, 0, "roi_shape")
        self._roi_shape = arr

    @property
    def probe_params(self) -> dict[str, Any]:
        return self._probe_params

    @probe_params.setter
    def probe_params(self, params: dict[str, Any]):
        validate_dict_keys(
            params,
            [*self.DEFAULT_PROBE_PARAMS.keys(), *POLAR_SYMBOLS, *POLAR_ALIASES.keys()],
        )

        def set_aberrations(
            params: dict[str, Any], max_order: int | None = None
        ) -> dict[str, float]:
            """Standardize aberration coefficients with optional max order filling."""

            def process_polar_params(p: dict):
                bads = []
                for symbol, value in p.items():
                    if isinstance(value, dict):
                        process_polar_params(value)
                    elif value is None:
                        continue
                    elif symbol in POLAR_SYMBOLS:
                        polar_parameters[symbol] = float(value)
                        bads.append(symbol)
                    elif symbol == "defocus":
                        polar_parameters["C10"] = -float(value)
                        bads.append(symbol)
                    elif symbol in POLAR_ALIASES:
                        polar_parameters[POLAR_ALIASES[symbol]] = float(value)
                        bads.append(symbol)
                [p.pop(bad, None) for bad in bads]

            # Start only with explicitly passed aberrations
            polar_parameters = {}
            process_polar_params(params)

            # Optionally fill all up to a given order with zeros
            if max_order is not None:
                for sym in POLAR_SYMBOLS:
                    if sym.startswith(("C", "phi")):
                        order = int(sym[-2])
                    else:
                        continue
                    if order <= max_order and sym not in polar_parameters:
                        polar_parameters[sym] = 0.0

            return polar_parameters

        polar_parameters = set_aberrations(deepcopy(params), self._max_aberrations_order)
        params["aberration_coefs"] = polar_parameters
        self._probe_params = self.DEFAULT_PROBE_PARAMS | self._probe_params | params

    @property
    def mean_diffraction_intensity(self) -> float:
        """mean diffraction intensity"""
        return self._mean_diffraction_intensity

    @mean_diffraction_intensity.setter
    def mean_diffraction_intensity(self, m: float):
        validate_gt(m, 0.0, "mean_diffraction_intensity")
        self._mean_diffraction_intensity = m

    @property
    def reciprocal_sampling(self) -> np.ndarray:
        """reciprocal sampling of the probe"""
        return to_numpy(self._reciprocal_sampling)

    @reciprocal_sampling.setter
    def reciprocal_sampling(self, sampling: np.ndarray | list | tuple):
        val = validate_array(
            validate_np_len(sampling, 2, name="reciprocal_sampling"),
            dtype=config.get("dtype_real"),
            ndim=1,
            name="reciprocal_sampling",
        )
        self._reciprocal_sampling = self._to_torch(val)

    @property
    def num_probes(self) -> int:
        """if num_probes > 1, then it is a mixed-state reconstruction"""
        return self._num_probes

    @num_probes.setter
    def num_probes(self, n: int):
        validate_gt(n, 0, "num_probes")
        self._num_probes = int(n)

    @property
    def dtype(self) -> torch.dtype:
        dtype_str = config.get("dtype_complex")
        if isinstance(dtype_str, str):
            return getattr(torch, dtype_str)
        return dtype_str

    @property
    def device(self) -> DeviceType:
        return self._device

    @device.setter
    def device(self, device: DeviceType):
        dev, _id = config.validate_device(device)
        self._device = dev

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
                dt = self.dtype
            else:
                raise ValueError(
                    f"Unknown string passed {dtype}, dtype should be 'same', 'probe', or torch.dtype"
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

    @property
    @abstractmethod
    def probe(self) -> torch.Tensor:
        """get the full probe"""
        raise NotImplementedError()

    @property
    def params(self) -> list[nn.Parameter]:
        """optimization parameters"""
        params = []
        if self.learn_probe_tilt:
            params.append(self.probe_tilt)
        return params

    @property
    def model_input(self):
        """get the model input"""
        raise NotImplementedError()

    def forward(self, fract_positions: torch.Tensor) -> torch.Tensor:
        """Get probe positions"""
        raise NotImplementedError()

    def reset(self):
        """Reset the probe"""
        self.probe_tilt = self._initial_probe_tilt.clone().to(self.device)

    def set_initial_probe(
        self,
        roi_shape: np.ndarray | tuple,
        reciprocal_sampling: np.ndarray,
        mean_diffraction_intensity: float,
        device: str | None = None,
    ):
        if device is not None:
            self._device = device

        # Only update roi_shape if it wasn't already set during initialization
        if not hasattr(self, "_roi_shape"):
            self.roi_shape = np.array(roi_shape)
        else:
            # Verify that the provided roi_shape matches the initialized one
            if not np.array_equal(self.roi_shape, np.array(roi_shape)):
                raise ValueError(
                    f"roi_shape {roi_shape} conflicts with initialized roi_shape {self.roi_shape}."
                )

        self.reciprocal_sampling = reciprocal_sampling
        self.mean_diffraction_intensity = mean_diffraction_intensity

    def check_probe_params(self):
        for k in self.DEFAULT_PROBE_PARAMS.keys():
            if self.probe_params[k] is None:
                if k == "defocus":
                    if self.probe_params["aberration_coefs"]["C10"] != 0:
                        self.probe_params[k] = -1 * self.probe_params["aberration_coefs"]["C10"]
                        continue
                print(f"Missing probe parameter '{k}' in probe_params")
                # raise ValueError(f"Missing probe parameter '{k}' in probe_params")

    def to(self, *args, **kwargs) -> Self:
        """Move all relevant tensors to a different device. Overrides nn.Module.to()."""
        # Call parent's to() method first to handle PyTorch's internal device management
        super().to(*args, **kwargs)

        device = kwargs.get("device", args[0] if args else None)
        if device is not None:
            self.device = device
            self._rng_to_device(device)
            self.reconnect_optimizer_to_parameters()

        return self

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the object model."""
        raise NotImplementedError()

    def backward(self, *args, **kwargs):
        raise NotImplementedError(
            f"Analytical gradients are not implemented for {Self}, use autograd=True"
        )

    def _compute_propagator_arrays(
        self,
        sampling: tuple[float, float] | np.ndarray,
        num_slices: int,
        slice_thicknesses: torch.Tensor | np.ndarray,
        gpts: tuple[int, int] | np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Precomputes propagator arrays complex wave-function will be convolved by,
        for all slice thicknesses.

        Parameters
        ----------
        sampling: tuple[float, float] | np.ndarray
            sampling of the probe in pixels
        num_slices: int
            number of slices
        slice_thicknesses: torch.Tensor | np.ndarray
            thickness of each slice in angstrom
        gpts: tuple[int, int] | np.ndarray | None
            grid shape the propagators act on; defaults to the probe roi_shape

        Returns
        -------
        propagators: torch.Tensor
            (T,Sr,Sc) shape array storing propagator arrays
        """

        if num_slices == 1:
            return torch.tensor([])

        if gpts is None:
            gpts = self.roi_shape
        kr, kc = tuple(torch.fft.fftfreq(n, d, device=self.device) for n, d in zip(gpts, sampling))
        k2 = (kr[:, None] ** 2 + kc[None] ** 2).to(torch.complex64)  # broadcasting to (Sr, Sc)
        probe_energy = self.probe_params["energy"]
        if probe_energy is None:
            raise ValueError("probe_model energy must be set to compute propagators.")
        wavelength = electron_wavelength_angstrom(probe_energy)
        propagators = torch.empty(
            (num_slices - 1, kr.shape[0], kc.shape[0]), dtype=torch.complex64, device=self.device
        )

        theta_r, theta_c = self.probe_tilt
        dz = torch.tensor(slice_thicknesses, device=self.device, dtype=k2.dtype)  # (T,)
        phase_factor = -1.0j * torch.pi * wavelength * dz[:, None, None]  # (T,1,1)
        propagators = torch.exp(phase_factor * k2)  # (T, Sr, Sc)
        if theta_r != 0:
            kr_term = 1.0j * (-2 * torch.pi * dz[:, None, None] * torch.tan(theta_r / 1e3))
            propagators = propagators * torch.exp(kr_term * kr[None, :, None])
        if theta_c != 0:
            kc_term = 1.0j * (-2 * torch.pi * dz[:, None, None] * torch.tan(theta_c / 1e3))
            propagators = propagators * torch.exp(kc_term * kc[None, None, :])

        return propagators


class ProbeConstraints(BaseConstraints[PtychoProbeConstraintParams.Raster], ProbeBase):
    DEFAULT_CONSTRAINTS: PtychoProbeConstraintParams.Raster = PtychoProbeConstraintParams.Raster()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_soft_constraints(self, probe: torch.Tensor) -> torch.Tensor:
        self.reset_soft_constraint_losses()
        loss = self._get_zero_loss_tensor()
        if self.constraints.tv_weight:
            loss_tv = self._probe_tv_constraint(probe, self.constraints.tv_weight)
            self.add_soft_constraint_loss("tv_loss", loss_tv)
            loss = loss + loss_tv

        self.accumulate_constraint_losses()
        return loss

    def apply_hard_constraints(self, probe: torch.Tensor) -> torch.Tensor:
        if self.constraints.orthogonalize_probe:
            probe = self._probe_orthogonalization_constraint(probe)
        if self.constraints.center_probe:
            probe = self._probe_center_of_mass_constraint(probe)
        return probe

    def _probe_tv_constraint(self, probe: torch.Tensor, weight: float) -> torch.Tensor:
        tv = self._get_zero_loss_tensor()
        if weight == 0:
            return tv
        for dim in (-1, -2):
            tv = tv + torch.mean(torch.abs(torch.diff(probe, dim=dim)))
        return weight * tv

    def _probe_center_of_mass_constraint(self, start_probe: torch.Tensor) -> torch.Tensor:
        probe_int = torch.fft.fftshift(torch.abs(start_probe).square(), dim=(-2, -1))
        # TODO -- move this to a util function
        y_coords = torch.arange(probe_int.shape[-2], device=probe_int.device)
        x_coords = torch.arange(probe_int.shape[-1], device=probe_int.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
        total_intensity = torch.sum(probe_int, dim=(-2, -1))
        com_y = torch.sum(probe_int * y_grid[None,], dim=(-2, -1)) / total_intensity
        com_x = torch.sum(probe_int * x_grid[None,], dim=(-2, -1)) / total_intensity

        probe_int_com = torch.stack([com_y, com_x], dim=-1) - torch.tensor(
            [s // 2 for s in self.roi_shape], device=self.device
        )
        return fourier_shift_expand(start_probe, -probe_int_com, expand_dim=False)

    def _probe_orthogonalization_constraint(self, start_probe: torch.Tensor) -> torch.Tensor:
        ### this is not very efficient with Adam, should find a better way
        n_probes = start_probe.shape[0]
        orthogonal_probes = []
        # Equivalent to torch.norm(..., dim=(-2,-1), keepdim=True)
        # original_norms = torch.norm(start_probe, dim=(-2, -1), keepdim=True)
        original_norms = torch.sqrt(
            torch.sum(
                start_probe.real.square() + start_probe.imag.square(), dim=(-2, -1), keepdim=True
            )
        )

        # Apply Gram-Schmidt process
        for i in range(n_probes):
            probe_i = start_probe[i]

            # Subtract projections onto previously computed orthogonal probes
            for j in range(len(orthogonal_probes)):
                projection = (
                    torch.sum(orthogonal_probes[j].conj() * probe_i) * orthogonal_probes[j]
                )
                probe_i = probe_i - projection

            # norm = torch.norm(probe_i)
            norm = torch.sqrt(torch.sum(probe_i.real.square() + probe_i.imag.square())).clamp_min(
                1e-12
            )
            orthogonal_probes.append(probe_i / norm)

        orthogonal_probes = torch.stack(orthogonal_probes)
        orthogonal_probes = orthogonal_probes * original_norms.view(-1, 1, 1)

        # Sort probes by real-space intensity
        intensities = torch.sum(torch.abs(orthogonal_probes).square(), dim=(-2, -1))
        intensities_order = torch.argsort(intensities, descending=True)

        # MPS-safe fancy indexing
        real_sorted = orthogonal_probes.real[intensities_order]
        imag_sorted = orthogonal_probes.imag[intensities_order]
        orthogonal_probes_sorted = torch.complex(real_sorted, imag_sorted)

        return orthogonal_probes_sorted


#    def _probe_orthogonalization_constraint(self, start_probe: torch.Tensor) -> torch.Tensor:
#        """
#        """
#        n_probes = start_probe.shape[0]
#
#        # Gram matrix, G = P @ P.H
#        P = start_probe.view(n_probes,-1)
#        G = P @ P.conj().T
#
#        # eigen-decomposition of G
#        _, eigenvecs = torch.linalg.eigh(G)
#
#        # rotate probes into orthogonal basis
#        orthogonal_probes = torch.tensordot(eigenvecs.T, start_probe, dims=1)
#
#        # sort by intensity
#        intensities = torch.sum(torch.abs(orthogonal_probes) ** 2, dim=(-2,-1))
#        order = torch.argsort(intensities, descending=True)
#
#        return orthogonal_probes[order]


class ProbePixelated(ProbeConstraints):
    def __init__(
        self,
        num_probes: int = 1,
        probe_params: dict = {},
        probe_tilt: tuple[float, float] | torch.Tensor = (0, 0),
        learn_probe_tilt: bool = False,
        roi_shape: tuple[int, int] | np.ndarray | None = None,
        dtype: torch.dtype = torch.complex64,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        initial_probe_weights: list[float] | np.ndarray | None = None,
        vacuum_probe_intensity: Dataset4dstem | np.ndarray | None = None,
        _from_params: bool = False,
        _token: object | None = None,
        *args,
    ):
        super().__init__(
            num_probes=num_probes,
            probe_params=probe_params.copy(),
            probe_tilt=probe_tilt,
            learn_probe_tilt=learn_probe_tilt,
            roi_shape=roi_shape,
            dtype=dtype,
            device=device,
            rng=rng,
            _token=_token,
        )
        self.initial_probe_weights = initial_probe_weights
        self._from_params = _from_params
        self.vacuum_probe_intensity = vacuum_probe_intensity

    @classmethod
    def from_array(
        cls,
        probe_array: np.ndarray | torch.Tensor,
        num_probes: int | None = None,
        probe_params: dict = {},  # not sure if necessary
        probe_tilt: tuple[float, float] | torch.Tensor = (0, 0),
        learn_probe_tilt: bool = False,
        dtype: torch.dtype = torch.complex64,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        initial_probe_weights: list[float] | np.ndarray | None = None,
    ):
        if isinstance(probe_array, np.ndarray):
            probe_array = torch.tensor(probe_array, dtype=dtype, device=device)
        else:
            probe_array = probe_array.to(dtype=dtype, device=device)
        if probe_array.ndim == 3:
            if num_probes is None:
                num_probes = probe_array.shape[0]
            elif num_probes != probe_array.shape[0]:
                raise ValueError(
                    f"num_probes {num_probes} must match probe_array shape {probe_array.shape[0]}"
                )
        else:
            num_probes = 1 if num_probes is None else num_probes
            probe_array = torch.stack([probe_array] * num_probes, dim=0)

        probe_model = cls(
            num_probes=num_probes,
            probe_params=probe_params.copy(),
            roi_shape=(int(probe_array.shape[-2]), int(probe_array.shape[-1])),
            dtype=dtype,
            probe_tilt=probe_tilt,
            learn_probe_tilt=learn_probe_tilt,
            device=device,
            rng=rng,
            initial_probe_weights=initial_probe_weights,
            _from_params=False,
            _token=cls._token,
        )

        probe_model.initial_probe = probe_array
        probe_model._probe = nn.Parameter(probe_array.clone(), requires_grad=True)
        return probe_model

    @classmethod
    def from_params(
        cls,
        probe_params: dict,
        num_probes: int = 1,
        roi_shape: tuple[int, int] | None = None,  # can be set later when set_initial_probe
        probe_tilt: tuple[float, float] | torch.Tensor = (0, 0),
        learn_probe_tilt: bool = False,
        dtype: torch.dtype = torch.complex64,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        initial_probe_weights: list[float] | np.ndarray | None = None,
        vacuum_probe_intensity: Dataset4dstem | np.ndarray | None = None,
    ):
        probe_model = cls(
            num_probes=num_probes,
            probe_params=deepcopy(probe_params),  # this seems to be needed for nested dicts
            roi_shape=roi_shape,
            dtype=dtype,
            probe_tilt=probe_tilt,
            learn_probe_tilt=learn_probe_tilt,
            device=device,
            rng=rng,
            initial_probe_weights=initial_probe_weights,
            vacuum_probe_intensity=vacuum_probe_intensity,
            _from_params=True,
            _token=cls._token,
        )
        probe_model._initial_probe = None

        return probe_model

    @property
    def probe(self) -> torch.Tensor:
        """get the full probe"""
        return self.apply_hard_constraints(self._probe)

    @probe.setter
    def probe(self, prb: "np.ndarray|torch.Tensor"):
        prb = validate_tensor(
            prb,
            name="probe",
            dtype=config.get("dtype_complex"),
            ndim=3,
            shape=(self.num_probes, *self.roi_shape),
            expand_dims=True,
        )
        probe_tensor = self._to_torch(prb)
        # Update the probe parameter data
        with torch.no_grad():
            self._probe.data = probe_tensor

    @property
    def initial_probe_weights(self) -> torch.Tensor:
        return self._initial_probe_weights

    @initial_probe_weights.setter
    def initial_probe_weights(self, weights: list[float] | np.ndarray | None):
        if weights is None:
            self._initial_probe_weights = torch.tensor(
                [1 - 0.02 * (self.num_probes - 1)] + [0.02] * (self.num_probes - 1)
            )
        else:
            if len(weights) != self.num_probes:
                raise ValueError(
                    f"initial_probe_weights must be a list of length {self.num_probes}"
                )
            w2 = validate_tensor(weights, name="initial_probe_weights", dtype=torch.float32)
            self._initial_probe_weights = w2 / torch.sum(w2)

    @property
    def params(self) -> list[nn.Parameter]:
        """optimization parameters"""
        params = super().params
        params.extend([self._probe])
        return params

    @property
    def initial_probe(self) -> torch.Tensor:
        return self._initial_probe

    @initial_probe.setter
    def initial_probe(self, initial_probe: np.ndarray | torch.Tensor):
        probe = validate_tensor(
            initial_probe,
            name="initial_probe",
            dtype=config.get("dtype_complex"),
        )
        self._initial_probe = probe

    def forward(self, fract_positions: torch.Tensor) -> torch.Tensor:
        shifted_probes = fourier_shift_expand(self.probe, fract_positions).swapaxes(0, 1)
        return shifted_probes

    def set_initial_probe(
        self,
        roi_shape: np.ndarray | tuple,
        reciprocal_sampling: np.ndarray,
        mean_diffraction_intensity: float,
        device: str | None = None,
    ):
        super().set_initial_probe(
            roi_shape, reciprocal_sampling, mean_diffraction_intensity, device
        )

        if self._from_params:
            self.check_probe_params()
            prb = real_space_probe(
                gpts=tuple(self.roi_shape.astype("int")),
                sampling=tuple(1 / (self.roi_shape * self.reciprocal_sampling).astype(np.float64)),
                energy=self.probe_params["energy"],
                semiangle_cutoff=self.probe_params["semiangle_cutoff"],
                vacuum_probe_intensity=self.vacuum_probe_intensity,
                aberration_coefs=self.probe_params["aberration_coefs"],
                soft_edges=self.probe_params["soft_edges"],
            )
            probes = prb.to(dtype=self.dtype, device=self.device)
        else:
            probes = self.initial_probe.clone()

        if probes.ndim != 3:
            probes = probes[None]
        if probes.shape[0] != self.num_probes:
            # probes = torch.tile(probes, (self.num_probes, 1, 1))
            probes = torch.cat([probes] * self.num_probes, dim=0)

        probes = self._apply_random_phase_shifts(probes)
        probes = self._apply_weights(probes)

        self._initial_probe = self._to_torch(probes)
        self._probe = nn.Parameter(self._initial_probe.clone().to(self.device), requires_grad=True)
        return

    def reset(self):
        super().reset()
        self.probe = self._initial_probe.clone()
        self._probe = nn.Parameter(self._initial_probe.clone().to(self.device), requires_grad=True)

    def to(self, *args, **kwargs) -> Self:
        super().to(*args, **kwargs)
        return self

    @property
    def name(self) -> str:
        return "ProbePixelated"

    def backward(self, propagated_gradient, obj_patches):
        obj_normalization = torch.sum(torch.abs(obj_patches).square(), dim=(-2, -1)).max()
        if self.num_probes == 1:
            # this is wrong--but it fixes the issue with multiple probes sgd + analytical--TODO fix
            # basically it screws up the amplitude grad but fixes the phase grad
            ortho_norm: float = 2 * np.prod(self.roi_shape) ** 0.5  # from ortho fft2 # type:ignore
        else:
            ortho_norm: float = 1 / (2 * np.prod(self.roi_shape) ** 0.5)  # type:ignore
        probe_grad = torch.sum(propagated_gradient, dim=1) / obj_normalization / ortho_norm
        self._probe.grad = -1 * probe_grad.clone().detach()

    @property
    def vacuum_probe_intensity(self) -> torch.Tensor | None:
        """corner centered vacuum probe"""
        if self._vacuum_probe_intensity is None:
            return None
        return self._vacuum_probe_intensity

    @vacuum_probe_intensity.setter
    def vacuum_probe_intensity(self, vp: np.ndarray | torch.Tensor | Dataset4dstem | None):
        """overwritten, clean up"""
        if vp is None:
            self._vacuum_probe_intensity = None
            return
        elif isinstance(vp, np.ndarray):
            vp2 = vp.astype(config.get("dtype_real"))
        elif isinstance(vp, (Dataset4dstem, Dataset2d)):
            vp2 = cast(np.ndarray, vp.array)  # TODO when finished Dataset->torch fix here
        elif isinstance(vp, torch.Tensor):
            vp2 = vp.cpu().detach().numpy()
        else:
            raise NotImplementedError(f"Unknown vacuum probe type: {type(vp)}")

        if vp2.ndim == 4:
            vp2 = np.mean(vp2, axis=(0, 1))
        elif vp2.ndim != 2:
            raise ValueError(f"Weird number of dimensions for vacuum probe, shape: {vp.shape}")

        # vacuum probe will end up corner centered, but if it starts corner centered then
        # we want to fftshift it be centered, so that we can use com to corner center it properly
        corner_vals = vp2[:10, :10].mean()
        if corner_vals > 0.01 * vp2.max():
            vp2 = np.fft.fftshift(vp2)

        # fix centering
        com: list | tuple = ndi.center_of_mass(vp2)
        vp2 = shift_array(
            vp2,
            -com[0],
            -com[1],
            bilinear=True,
        )

        self._vacuum_probe_intensity = torch.tensor(vp2, dtype=torch.float32, device=self.device)

    def rescale_vacuum_probe(self, shape: tuple[int, int]):
        """hack, should be fixed"""
        if self.vacuum_probe_intensity is None:
            return
        scale_output = (
            shape[0] / self.vacuum_probe_intensity.shape[0],
            shape[1] / self.vacuum_probe_intensity.shape[1],
        )
        self._vacuum_probe_intensity = torch.tensor(
            ndi.zoom(
                self.vacuum_probe_intensity.cpu().detach().numpy(),
                scale_output,
            ),
            dtype=getattr(torch, config.get("dtype_real")),
            device=self.device,
        )

    def _apply_random_phase_shifts(self, probe_array: torch.Tensor | np.ndarray) -> torch.Tensor:
        probes = self._to_torch(probe_array)
        for a0 in range(1, self.num_probes):
            shift_y = torch.exp(
                -2j * torch.pi * (self.rng.random() - 0.5) * torch.fft.fftfreq(self.roi_shape[0])
            )
            shift_x = torch.exp(
                -2j * torch.pi * (self.rng.random() - 0.5) * torch.fft.fftfreq(self.roi_shape[1])
            )
            shift_y = shift_y.to(self.device)
            shift_x = shift_x.to(self.device)
            probes[a0] = probes[a0] * shift_y[:, None] * shift_x[None]
        return probes

    def _apply_weights(self, probe_array: torch.Tensor | np.ndarray) -> torch.Tensor:
        probes = self._to_torch(probe_array)
        probe_intensity = torch.sum(torch.abs(torch.fft.fft2(probes, norm="ortho")).square())
        intensity_norm = torch.sqrt(self.mean_diffraction_intensity / probe_intensity)
        probes *= intensity_norm

        current_weights = torch.sum(torch.abs(probes).square(), dim=(1, 2))
        current_weights = current_weights / torch.sum(current_weights)
        weight_scaling = torch.sqrt(self.initial_probe_weights.to(self.device) / current_weights)
        probes = probes * self._to_torch(weight_scaling)[:, None, None]

        # self._initial_probe = self._to_torch(probes)
        # self._probe = self._initial_probe.clone()
        return probes


class ProbeParametric(ProbeConstraints):
    def __init__(
        self,
        num_probes: int = 1,
        probe_params: dict = {},
        probe_tilt: tuple[float, float] | torch.Tensor = (0, 0),
        learn_probe_tilt: bool = False,
        roi_shape: tuple[int, int] | np.ndarray | None = None,
        dtype: torch.dtype = torch.complex64,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        vacuum_probe_intensity: np.ndarray | Dataset4dstem | None = None,
        max_aberrations_order: int | None = None,
        learn_aberrations: bool = True,
        learn_cutoff: bool = False,
        _token: object | None = None,
    ):
        if num_probes > 1:
            raise NotImplementedError()

        super().__init__(
            num_probes=num_probes,
            probe_params=probe_params.copy(),
            probe_tilt=probe_tilt,
            learn_probe_tilt=learn_probe_tilt,
            max_aberrations_order=max_aberrations_order,
            roi_shape=roi_shape,
            dtype=dtype,
            device=device,
            rng=rng,
            _token=_token,
        )

        self.learn_aberrations = learn_aberrations
        self.learn_cutoff = learn_cutoff
        self._vacuum_probe_intensity = None

        self.vacuum_probe_intensity = vacuum_probe_intensity

        if learn_cutoff and self.vacuum_probe_intensity is None:
            self.semiangle_cutoff = nn.Parameter(
                torch.tensor(float(self.probe_params["semiangle_cutoff"]), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "semiangle_cutoff",
                torch.tensor(float(self.probe_params["semiangle_cutoff"]), dtype=torch.float32),
            )

        aberration_coefs = self.probe_params.get("aberration_coefs", {})
        self.aberration_names = list(aberration_coefs.keys())
        self.aberration_coefs = nn.ParameterDict()

        for k, v in aberration_coefs.items():
            if learn_aberrations:
                self.aberration_coefs[k] = nn.Parameter(
                    torch.tensor(float(v), dtype=torch.float32)
                )
            else:
                self.register_buffer(k, torch.tensor(float(v), dtype=torch.float32))

        self._store_initial_params()

    def _store_initial_params(self):
        """Store initial learnable parameter values for later reset."""
        if hasattr(self, "semiangle_cutoff"):
            self.register_buffer(
                "_initial_semiangle_cutoff", self.semiangle_cutoff.detach().clone()
            )
        if hasattr(self, "aberration_coefs"):
            for name, tensor in self.aberration_coefs.items():
                self.register_buffer(f"_initial_aberration_coefs_{name}", tensor.detach().clone())

    @classmethod
    def from_params(
        cls,
        probe_params: dict,
        num_probes: int = 1,
        roi_shape: tuple[int, int] | None = None,
        probe_tilt: tuple[float, float] | torch.Tensor = (0, 0),
        learn_probe_tilt: bool = False,
        dtype: torch.dtype = torch.complex64,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        vacuum_probe_intensity: np.ndarray | Dataset4dstem | None = None,
        max_aberrations_order: int | None = None,
        learn_aberrations: bool = True,
        learn_cutoff: bool = False,
    ):
        return cls(
            num_probes=num_probes,
            probe_params=probe_params.copy(),
            probe_tilt=probe_tilt,
            learn_probe_tilt=learn_probe_tilt,
            roi_shape=roi_shape,
            dtype=dtype,
            device=device,
            rng=rng,
            vacuum_probe_intensity=vacuum_probe_intensity,
            max_aberrations_order=max_aberrations_order,
            learn_aberrations=learn_aberrations,
            learn_cutoff=learn_cutoff,
            _token=cls._token,
        )

    @property
    def vacuum_probe_intensity(self) -> np.ndarray | None:
        if self._vacuum_probe_intensity is None:
            return None
        return self._vacuum_probe_intensity

    @vacuum_probe_intensity.setter
    def vacuum_probe_intensity(self, vp: np.ndarray | Dataset4dstem | None):
        if vp is None:
            self._vacuum_probe_intensity = None
            return
        elif isinstance(vp, np.ndarray):
            vp2 = vp.astype(config.get("dtype_real"))
        elif isinstance(vp, (Dataset4dstem, Dataset2d)):
            vp2 = cast(np.ndarray, vp.array)  # TODO when finished Dataset->torch fix here
        else:
            raise NotImplementedError(f"Unknown vacuum probe type: {type(vp)}")

        if vp2.ndim == 4:
            vp2 = np.mean(vp2, axis=(0, 1))
        elif vp2.ndim != 2:
            raise ValueError(f"Unexpected shape for vacuum probe: {vp2.shape}")

        self._vacuum_probe_intensity = vp2

    @property
    def params(self) -> list[nn.Parameter]:
        """Optimization parameters."""
        params = super().params
        if isinstance(self.semiangle_cutoff, nn.Parameter):
            params.append(self.semiangle_cutoff)
        params.extend(list(self.aberration_coefs.values()))
        return params

    @property
    def probe(self) -> torch.Tensor:
        """get the full probe"""
        return self.apply_hard_constraints(self._build_probe())

    @property
    def name(self) -> str:
        return "ProbeParametric"

    def _build_probe(self) -> torch.Tensor:
        """Build the probe array on the fly from current parameters."""
        # collect aberration coefficients
        coefs = {}
        for k in self.aberration_names:
            if hasattr(self.aberration_coefs, k):
                coefs[k] = getattr(self.aberration_coefs, k)
            elif hasattr(self, k):
                coefs[k] = getattr(self, k)
            else:
                raise KeyError(f"Unknown aberration key {k}")

        probe = real_space_probe(
            gpts=tuple(self.roi_shape.astype("int")),
            sampling=tuple(1 / (self.roi_shape * self.reciprocal_sampling).astype(np.float64)),
            energy=self.probe_params["energy"],
            semiangle_cutoff=self.semiangle_cutoff,  # type:ignore
            vacuum_probe_intensity=self.vacuum_probe_intensity,  # type:ignore
            aberration_coefs=coefs,
            soft_edges=self.probe_params["soft_edges"],
            device=self.device,  # type:ignore
        )
        probe = probe.to(dtype=self.dtype, device=self.device)
        mean_diffraction_intensity = getattr(self, "_mean_diffraction_intensity", 1.0)
        return probe[None] * np.sqrt(mean_diffraction_intensity)

    def forward(self, fract_positions: torch.Tensor) -> torch.Tensor:
        """Generate probe on the fly and apply subpixel shifts."""
        shifted_probes = fourier_shift_expand(self.probe, fract_positions).swapaxes(0, 1)
        return shifted_probes

    def reset(self):
        """Reset learnable parameters to their initial values."""
        super().reset()
        with torch.no_grad():
            if hasattr(self, "semiangle_cutoff"):
                self.semiangle_cutoff.copy_(self._initial_semiangle_cutoff.to(self.device))  # type:ignore
            if hasattr(self, "aberration_coefs"):
                for name, param in self.aberration_coefs.items():
                    initial = getattr(self, f"_initial_aberration_coefs_{name}")
                    param.data.copy_(initial)


class ProbeDIP(ProbeConstraints):
    """
    DIP/model based probe model.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        num_probes: int = 1,
        probe_params: dict = {},
        probe_tilt: tuple[float, float] | torch.Tensor = (0, 0),
        learn_probe_tilt: bool = False,
        roi_shape: tuple[int, int] | np.ndarray | None = None,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        _token: object | None = None,
    ):
        super().__init__(
            num_probes=num_probes,
            probe_params=probe_params.copy(),
            probe_tilt=probe_tilt,
            learn_probe_tilt=learn_probe_tilt,
            roi_shape=roi_shape,
            device=device,
            rng=rng,
            _token=_token,
        )
        self.register_buffer("_model_input", torch.tensor([]))
        self.register_buffer("_pretrain_target", torch.tensor([]))

        self._model = model.to(self._device)
        self._check_roi_shape()
        self.set_pretrained_weights(self._model)

        self._optimizer = None
        self._scheduler = None
        self._pretrain_losses: list[float] = []
        self._pretrain_lrs: list[float] = []

    @classmethod
    def from_model(
        cls,
        model: "torch.nn.Module",
        probe_tilt: tuple[float, float] | torch.Tensor = (0, 0),
        learn_probe_tilt: bool = False,
        model_input: torch.Tensor | None = None,
        num_probes: int = 1,
        probe_params: dict = {},
        roi_shape: tuple[int, int] | np.ndarray | None = None,
        input_noise_std: float = 0.025,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        probe_model = cls(
            model=model,
            num_probes=num_probes,
            probe_params=probe_params.copy(),
            probe_tilt=probe_tilt,
            learn_probe_tilt=learn_probe_tilt,
            roi_shape=roi_shape,
            device=device,
            rng=rng,
            _token=cls._token,
        )

        if model_input is None:
            # Create default model input - use roi_shape if provided, otherwise placeholder
            if roi_shape is not None:
                input_shape = (1, num_probes, *np.array(roi_shape))
            else:
                input_shape = (1, num_probes, 1, 1)  # will be set properly in set_initial_probe
            probe_model.model_input = torch.randn(
                input_shape, dtype=torch.complex64, device=device, generator=probe_model._rng_torch
            )
        else:
            probe_model.model_input = model_input.clone().detach()

        probe_model.pretrain_target = probe_model.model_input.clone().detach()
        probe_model._model_input_noise_std = input_noise_std
        return probe_model

    @classmethod
    def from_pixelated(
        cls,
        model: "torch.nn.Module",
        pixelated: "ProbeModelType",  # ProbePixelated upsets linter when ptycho.probe_model is used
        input_noise_std: float = 0.025,
        device: str = "cpu",
    ) -> "ProbeDIP":
        if not isinstance(pixelated, ProbePixelated) and "ProbePixelated" not in str(
            type(pixelated)
        ):
            raise TypeError(f"dset should be a ProbePixelated, got {type(pixelated)}")

        probe_model = cls(
            model=model,
            num_probes=pixelated.num_probes,
            probe_params=pixelated.probe_params.copy(),
            probe_tilt=tuple(pixelated.probe_tilt.detach().cpu().numpy()),
            learn_probe_tilt=pixelated.learn_probe_tilt,
            roi_shape=pixelated.roi_shape,
            device=device,
            rng=pixelated._rng_seed,
            _token=cls._token,
        )

        probe_model.model_input = pixelated.probe.clone().detach()
        probe_model.pretrain_target = probe_model.model_input.clone().detach()
        probe_model._model_input_noise_std = input_noise_std
        return probe_model

    @property
    def name(self) -> str:
        return "ProbeDIP"

    @property
    def dtype(self) -> "torch.dtype":
        if hasattr(self.model, "dtype"):
            return getattr(self.model, "dtype")
        else:
            return self.model_input.dtype

    @property
    def model(self) -> "torch.nn.Module":
        """get the DIP model"""
        return self._model

    @model.setter
    def model(self, dip: "torch.nn.Module"):
        """
        This actually doesn't work -- can't have setters for torch sub modules
        https://github.com/pytorch/pytorch/issues/52664
        """
        print("probe model setter hi")
        if not isinstance(dip, torch.nn.Module):
            raise TypeError(f"DIP must be a torch.nn.Module, got {type(dip)}")
        if hasattr(dip, "dtype"):
            dt = getattr(dip, "dtype")
            if not dt.is_complex:
                raise ValueError("DIP model must be a complex-valued model for probe objects")
        self._model = dip.to(self.device)
        self.set_pretrained_weights(self._model)

    @property
    def pretrained_weights(self) -> dict[str, torch.Tensor]:
        """get the pretrained weights of the DIP model"""
        return self._pretrained_weights

    def set_pretrained_weights(self, model: torch.nn.Module):
        """set the pretrained weights of the DIP model"""
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Pretrained model must be a torch.nn.Module, got {type(model)}")
        self._pretrained_weights = deepcopy(model.state_dict())

    @property
    def model_input(self) -> torch.Tensor:
        """get the model input"""
        return self._model_input

    @model_input.setter
    def model_input(self, input_tensor: torch.Tensor):
        """set the model input"""
        inp = validate_tensor(
            input_tensor,
            name="model_input",
            dtype=torch.complex64,
            ndim=4,
            expand_dims=True,
        )
        self._model_input = inp.to(self.device)

    @property
    def pretrain_target(self) -> torch.Tensor:
        """get the pretrain target"""
        return self._pretrain_target

    @pretrain_target.setter
    def pretrain_target(self, target: torch.Tensor):
        """set the pretrain target"""
        if target.ndim == 4:
            target = target.squeeze(0)
        target = validate_tensor(
            target,
            name="pretrain_target",
            ndim=3,
            dtype=torch.complex64,
            expand_dims=True,
        )
        if target.shape[-3:] != self.model_input.shape[-3:]:
            raise ValueError(
                f"Pretrain target shape {target.shape} does not match model input shape {self.model_input.shape}"
            )
        self._pretrain_target = target.to(self.device)

    @property
    def _model_input_noise_std(self) -> float:
        """standard deviation of the gaussian noise added to the model input each forward call"""
        return self._input_noise_std

    @_model_input_noise_std.setter
    def _model_input_noise_std(self, std: float):
        validate_gt(std, 0.0, "input_noise_std", geq=True)
        self._input_noise_std = std

    @property
    def pretrain_losses(self) -> np.ndarray:
        return np.array(self._pretrain_losses)

    @property
    def pretrain_lrs(self) -> np.ndarray:
        return np.array(self._pretrain_lrs)

    @property
    def probe(self) -> torch.Tensor:
        """get the full probe"""
        probe = self.model(self._model_input)[0]
        return self.apply_hard_constraints(probe)

    @property
    def _probe(self) -> torch.Tensor:
        return self.forward(None)  # type: ignore

    def forward(self, fract_positions: torch.Tensor) -> torch.Tensor:
        """Get shifted probes at fractional positions"""
        if self._input_noise_std > 0.0:
            noise = (
                torch.randn(
                    self.model_input.shape,
                    dtype=self.dtype,
                    device=self.device,
                    generator=self._rng_torch,
                )
                * self._input_noise_std
            )
            model_input = self.model_input + noise
        else:
            model_input = self.model_input

        probe = self.model(model_input)[0]
        shifted_probes = fourier_shift_expand(probe, fract_positions).swapaxes(0, 1)
        return shifted_probes

    def set_initial_probe(
        self,
        roi_shape: np.ndarray | tuple,
        reciprocal_sampling: np.ndarray,
        mean_diffraction_intensity: float,
        device: str | None = None,
        *args,
    ):
        """Set initial probe and create appropriate model input"""
        super().set_initial_probe(
            roi_shape, reciprocal_sampling, mean_diffraction_intensity, device
        )

        # could check if num_probes corresponds to out_channels of model

        # Only create new model_input if it's still the placeholder (shape [1, num_probes, 1, 1])
        if self.model_input.shape[-2:] == (1, 1):
            self.model_input = torch.randn(
                (1, self.num_probes, *self.roi_shape),
                dtype=self.dtype,
                device=self.device,
                generator=self._rng_torch,
            )

    def to(self, *args, **kwargs) -> Self:
        """Move all relevant tensors to a different device."""
        super().to(*args, **kwargs)
        device = kwargs.get("device", args[0] if args else None)
        if device is not None:
            self._model = self.model.to(self.device)
            self._model_input = self._model_input.to(self.device)
            if hasattr(self, "_initial_probe"):
                self._initial_probe = self._initial_probe.to(self.device)
        return self

    @property
    def params(self) -> list[nn.Parameter]:
        """optimization parameters"""
        params = super().params
        params.extend(list(self.model.parameters()))
        return params

    def reset(self):
        """Reset the object model to its initial or pre-trained state"""
        super().reset()
        self.model.load_state_dict(self.pretrained_weights.copy())

    def pretrain(
        self,
        model_input: torch.Tensor | None = None,
        pretrain_target: torch.Tensor | None = None,
        reset: bool = False,
        num_iters: int = 100,
        optimizer_params: dict | OptimizerParamsType | None = None,
        scheduler_params: dict | SchedulerParamsType | None = None,
        loss_fn: Callable | str = "l2",
        apply_constraints: bool = False,
        show: bool = True,
        device: str | int | None = None,
    ):
        if device is not None:
            dev, _ = config.validate_device(device)
            self.to(dev)

        if optimizer_params is not None:
            self.set_optimizer(optimizer_params)

        if scheduler_params is not None:
            self.set_scheduler(scheduler_params, num_iters)

        if reset:
            self.model.apply(reset_weights)
            self._pretrain_losses = []
            self._pretrain_lrs = []

        if model_input is not None:
            self.model_input = model_input
        if pretrain_target is not None:
            if pretrain_target.shape[-3:] != self.model_input.shape[-3:]:
                raise ValueError(
                    f"Model target shape {pretrain_target.shape} does not match model input shape {self.model_input.shape}"
                )
            self.pretrain_target = pretrain_target.clone().detach().to(self.device)
        elif self.pretrain_target is None:
            self.pretrain_target = self._initial_probe.clone().detach()

        loss_fn = get_loss_module(loss_fn, self.dtype)
        self._pretrain(
            num_iters=num_iters,
            loss_fn=loss_fn,
            apply_constraints=apply_constraints,
            show=show,
        )
        self.set_pretrained_weights(self.model)

    def _pretrain(
        self,
        num_iters: int,
        loss_fn: Callable,
        apply_constraints: bool = False,
        show: bool = False,
    ):
        """Pretrain the DIP model."""
        if not hasattr(self, "pretrain_target"):
            raise ValueError("Pretrain target is not set. Use pretrain_target to set it.")

        self.model.train()
        optimizer = self.optimizer
        if optimizer is None:
            raise ValueError("Optimizer not set. Call set_optimizer() first.")

        sch = self.scheduler
        pbar = tqdm(range(num_iters))
        output = self.probe

        for a0 in pbar:
            if self._input_noise_std > 0.0:
                noise = (
                    torch.randn(
                        self.model_input.shape,
                        dtype=self.dtype,
                        device=self.device,
                        generator=self._rng_torch,
                    )
                    * self._input_noise_std
                )
                model_input = self.model_input + noise
            else:
                model_input = self.model_input

            if apply_constraints:
                output = self.apply_hard_constraints(self.model(model_input)[0])
            else:
                output = self.model(model_input)[0]
            loss: torch.Tensor = loss_fn(output, self.pretrain_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if sch is not None:
                if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sch.step(loss.item())
                else:
                    sch.step()

            self._pretrain_losses.append(loss.item())
            self._pretrain_lrs.append(optimizer.param_groups[0]["lr"])
            pbar.set_description(f"Iter {a0 + 1}/{num_iters}, Loss: {loss.item():.3e}, ")

        if show:
            self.visualize_pretrain(output)

    def visualize_pretrain(self, pred_probe: torch.Tensor):
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
        ax = fig.add_subplot(gs[0])
        lines = []
        lines.extend(
            ax.semilogy(
                np.arange(len(self._pretrain_losses)), self._pretrain_losses, c="k", label="loss"
            )
        )
        ax.set_ylabel("Loss", color="k")
        ax.tick_params(axis="y", which="both", colors="k")
        ax.spines["left"].set_color("k")
        ax.set_xlabel("Iterations")
        nx = ax.twinx()
        nx.spines["left"].set_visible(False)
        lines.extend(
            nx.semilogy(
                np.arange(len(self._pretrain_lrs)),
                self._pretrain_lrs,
                c="tab:orange",
                label="LR",
            )
        )
        labs = [lin.get_label() for lin in lines]
        nx.legend(lines, labs, loc="upper center")
        nx.set_ylabel("LRs")

        n_bot = 2
        gs_bot = gridspec.GridSpecFromSubplotSpec(1, n_bot, subplot_spec=gs[1])
        axs_bot = np.array([fig.add_subplot(gs_bot[0, i]) for i in range(n_bot)])
        target = self.pretrain_target
        show_2d(
            [
                np.fft.fftshift(pred_probe.mean(0).cpu().detach().numpy()),
                np.fft.fftshift(target.mean(0).cpu().detach().numpy()),
            ],
            figax=(fig, axs_bot),
            title=[
                "Predicted Probe",
                "Target Probe",
            ],
            cmap="magma",
            cbar=False,
        )
        plt.suptitle(
            f"Final loss: {self._pretrain_losses[-1]:.3e} | Iters: {len(self._pretrain_losses)}",
            fontsize=14,
            y=0.94,
        )
        plt.show()

    def backward(self, propagated_gradient, obj_patches):
        """Backward pass for analytical gradients (not implemented for DIP)"""
        raise NotImplementedError(
            f"Analytical gradients are not implemented for {self.name}, use autograd=True"
        )

    def _check_roi_shape(self):
        num_layers = getattr(self.model, "num_layers", None)
        if num_layers is not None:
            if not np.all(np.array(self.roi_shape) % 2**num_layers == 0):
                raise ValueError(
                    f"Model has {num_layers} layers, but ROI shape {self.roi_shape} is not "
                    f"divisible by 2^{num_layers} for all dimensions.\nPlease crop or pad the "
                    "dataset in reciprocal space to fix this error."
                )
        else:
            warn(
                "Model has no num_layers attribute, so cannot check if the ROI shape is "
                "compatible with the model.\nIf using a ConvNet, ensure that the ROI shape is "
                "divisble by 2^num_layers for all dimensions."
            )


def _prism_wave_vectors(
    cutoff: float,
    extent: np.ndarray | tuple[float, float],
    wavelength: float,
) -> np.ndarray:
    """Dense beamlet wave vectors: all reciprocal-grid points inside the cutoff disk.

    Parameters
    ----------
    cutoff : float
        Convergence semi-angle cutoff in mrad.
    extent : np.ndarray | tuple[float, float]
        Real-space extent in Angstroms.
    wavelength : float
        Electron wavelength in Angstroms.

    Returns
    -------
    np.ndarray
        Wave vectors in inverse Angstroms, shape (num_beamlets, 2).
    """
    cutoff_rad = cutoff * 1e-3
    n_max = int(np.ceil(cutoff_rad / (wavelength / extent[0])))
    m_max = int(np.ceil(cutoff_rad / (wavelength / extent[1])))

    kx = np.arange(-n_max, n_max + 1, dtype=np.float64) / extent[0]
    ky = np.arange(-m_max, m_max + 1, dtype=np.float64) / extent[1]

    mask = kx[:, None] ** 2 + ky[None, :] ** 2 < (cutoff_rad / wavelength) ** 2
    kxg, kyg = np.meshgrid(kx, ky, indexing="ij")

    return np.stack((kxg[mask], kyg[mask]), axis=-1)


def _partitioned_prism_wave_vectors(
    cutoff: float,
    wavelength: float,
    num_rings: int = 4,
    num_points_per_ring: int = 6,
) -> np.ndarray:
    """Parent beam wave vectors: a center point plus hexagonal rings up to the cutoff.

    Parameters
    ----------
    cutoff : float
        Convergence semi-angle cutoff in mrad.
    wavelength : float
        Electron wavelength in Angstroms.
    num_rings : int
        Number of rings including the center point.
    num_points_per_ring : int
        Points on the innermost ring; each subsequent ring adds this many more.

    Returns
    -------
    np.ndarray
        Wave vectors in inverse Angstroms, shape (num_parent, 2).
    """
    if num_rings < 2:
        raise ValueError(f"num_rings must be >= 2, got {num_rings}")

    rings = [np.array([[0.0, 0.0]])]
    n = num_points_per_ring

    for r in np.linspace(cutoff / (num_rings - 1), cutoff, num_rings - 1):
        angles = np.arange(n, dtype=np.float64) * 2 * np.pi / n + np.pi / 2
        kx = r * np.sin(angles) / 1000.0 / wavelength
        ky = r * np.cos(-angles) / 1000.0 / wavelength
        rings.append(np.stack([kx, ky], axis=1))
        n += num_points_per_ring

    return np.vstack(rings)


# coarse grid parents are kept within this normalized radius beyond the aperture
# edge (unity at the cutoff), or one coarse cell, whichever is larger; matches
# abTEM's disk-shaped coarse support (C-PRISM PR #318).
_GRID_SUPPORT_MARGIN = 0.45


def _grid_prism_wave_vectors(
    cutoff: float,
    extent: np.ndarray | tuple[float, float],
    wavelength: float,
    interpolation_factor: int,
    margin: float = _GRID_SUPPORT_MARGIN,
) -> np.ndarray:
    """Parent beams on a regular reciprocal sublattice ``k = f n / extent``.

    Used by the Fourier-interpolation (``"fourier"``, ``"nearest"``) and grid
    Sibson PRISM schemes. The lattice is subsampled by ``interpolation_factor``
    relative to the dense beamlet grid and restricted to a disk around the
    aperture (radius ``1 + max(coarse_cell, margin)``, unity at the cutoff), so the
    dropped rectangle corners are neither built (multislice cost) nor overfit by
    the interpolation. Pass ``margin=0`` for the one-hot classic-PRISM parents,
    where beams beyond the aperture carry ~zero CTF weight.

    Parameters
    ----------
    cutoff : float
        Convergence semi-angle cutoff in mrad.
    extent : np.ndarray | tuple[float, float]
        Real-space extent in Angstroms.
    wavelength : float
        Electron wavelength in Angstroms.
    interpolation_factor : int
        Coarse-lattice subsampling factor ``f`` (>= 1).
    margin : float
        Support margin beyond the aperture edge, in normalized radius.

    Returns
    -------
    np.ndarray
        Wave vectors in inverse Angstroms, shape (num_parent, 2).
    """
    f = int(interpolation_factor)
    if f < 1:
        raise ValueError(f"interpolation_factor must be >= 1, got {f}")

    cutoff_rad = cutoff * 1e-3
    n_max = int(np.ceil(cutoff_rad / (wavelength / extent[0])))
    m_max = int(np.ceil(cutoff_rad / (wavelength / extent[1])))

    bx = -(n_max // -f) + 1  # ceil(n_max / f) + 1
    by = -(m_max // -f) + 1

    n = np.arange(-bx, bx + 1)
    m = np.arange(-by, by + 1)

    kx = f * n / extent[0]
    ky = f * m / extent[1]

    radius_x = (n * f) / max(1, n_max)
    radius_y = (m * f) / max(1, m_max)
    radius = np.sqrt(radius_x[:, None] ** 2 + radius_y[None, :] ** 2)

    cell = max(f / max(1, n_max), f / max(1, m_max))
    keep = radius <= 1.0 + max(cell, margin)

    kxg, kyg = np.meshgrid(kx, ky, indexing="ij")
    return np.stack((kxg[keep], kyg[keep]), axis=-1)


class ProbePRISM(BaseConstraints[PtychoProbeConstraintParams.Parametric], ProbeBase):
    """
    Partitioned-PRISM probe model.

    Instead of a converged probe, the probe is represented as a set of tilted plane
    waves ("beamlets") inside the aperture, grouped under a sparse set of parent
    beams via frozen Sibson natural-neighbor interpolation weights. The engine
    (``PtychographyPRISM``) propagates the parent beams through the full object and
    reduces them with the ROI-sized coefficient maps this model produces:

        ``coef_map[p, b, n] = ifft2( CTF_p(k) * w_b(k) * c_pb * exp(-2 pi i k . dr_n) )``

    Interpolation schemes (``parent_layout`` x ``interpolation``) map onto the PRISM
    literature; all share the same reduction, differing only in the frozen weight
    maps and (for ``"nearest"``) a real-space crop window:

    ======================  =============  ===============  ==========================
    scheme                  parent_layout  interpolation    notes
    ======================  =============  ===============  ==========================
    partitioned PRISM       ``"rings"``    ``"sibson"``     default; hexagonal parents
    PRISM (Ophus 2017)      ``"grid"``     ``"nearest"``    one-hot + crop window
    C-PRISM w/o SVD (#318)  ``"grid"``     ``"fourier"``    trig. interp., full aperture
    grid ablation           ``"grid"``     ``"sibson"``     isolates interpolant/layout
    ======================  =============  ===============  ==========================

    C-PRISM's phase-removal tricks are already handled by the engine
    (carrier removal at parent propagation, ``thickness_compensation`` back-
    propagation); its SVD compression stage is intentionally not implemented (it
    adds no accuracy and its build-once/reduce-many amortization is broken by
    per-iteration object updates).

    Learnables (each gated by a flag):
    - per-mode aberration coefficients (``learn_aberrations``)
    - a free complex coefficient per parent beam per mode (``learn_beam_coefficients``),
      initialized so the initial probe equals the aberrated CTF; these capture
      partial coherence / mode structure.

    A measured aperture can be supplied via ``vacuum_probe_intensity``
    (corner-centered intensity; replaces the analytic soft/hard aperture, as in
    ``ProbeParametric``). Since the aperture profile is otherwise frozen, this is
    the way to reach exact agreement with data whose aperture edge differs from
    the analytic model.

    Notes
    -----
    ``forward`` returns ``(beamlets_fft, position_coefs)`` rather than the shifted
    probe stack of the other probe models: materializing per-position probes would
    defeat PRISM. This model therefore only works with ``PtychographyPRISM``.

    The two learnable groups have vastly different gradient scales and MUST use
    per-group learning rates (PPLR): aberration coefficients are Angstrom-scale
    parameters with tiny gradients (SGD lr ~1e-1 works well), while beam
    coefficients are O(1) parameters with O(0.1) gradients (SGD lr ~1e-3, or Adam
    lr ~1e-2; a shared lr of 0.1 diverges immediately). For example::

        optimizer_params={"probe": {
            "aberrations": {"type": "SGD", "lr": 0.125},
            "beam_coefficients": {"type": "SGD", "lr": 1e-3},
        }}
    """

    DEFAULT_CONSTRAINTS: PtychoProbeConstraintParams.Parametric = (
        PtychoProbeConstraintParams.Parametric()
    )

    def __init__(
        self,
        num_probes: int = 1,
        probe_params: dict = {},
        aberration_coefs: dict | list[dict] | None = None,
        num_partitions: int = 4,
        dense: bool = False,
        parent_layout: str = "rings",
        interpolation: str = "sibson",
        interpolation_factor: int | None = None,
        learn_aberrations: bool = True,
        learn_beam_coefficients: bool = False,
        initial_probe_weights: list[float] | np.ndarray | None = None,
        probe_tilt: tuple[float, float] | torch.Tensor = (0, 0),
        learn_probe_tilt: bool = False,
        roi_shape: tuple[int, int] | np.ndarray | None = None,
        vacuum_probe_intensity: np.ndarray | Dataset4dstem | None = None,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        max_aberrations_order: int | None = None,
        _token: object | None = None,
    ):
        super().__init__(
            num_probes=num_probes,
            probe_params=probe_params.copy(),
            probe_tilt=probe_tilt,
            learn_probe_tilt=learn_probe_tilt,
            max_aberrations_order=max_aberrations_order,
            roi_shape=roi_shape,
            device=device,
            rng=rng,
            _token=_token,
        )

        if self.probe_params.get("energy", None) is None:
            raise ValueError("probe_params must contain 'energy' for ProbePRISM")
        if self.probe_params.get("semiangle_cutoff", None) is None:
            raise ValueError("probe_params must contain 'semiangle_cutoff' for ProbePRISM")

        if parent_layout not in ("rings", "grid"):
            raise ValueError(f"parent_layout must be 'rings' or 'grid', got {parent_layout!r}")
        if interpolation not in ("sibson", "fourier", "nearest"):
            raise ValueError(
                f"interpolation must be 'sibson', 'fourier', or 'nearest', got {interpolation!r}"
            )
        if interpolation in ("fourier", "nearest") and parent_layout != "grid":
            raise ValueError(
                f"interpolation={interpolation!r} requires parent_layout='grid' "
                "(ring parents are off the reciprocal sublattice)"
            )
        if parent_layout == "grid":
            if interpolation_factor is None:
                raise ValueError("parent_layout='grid' requires interpolation_factor")
            if int(interpolation_factor) < 1:
                raise ValueError(f"interpolation_factor must be >= 1, got {interpolation_factor}")

        self.num_partitions = num_partitions
        self.dense = dense
        self.parent_layout = parent_layout
        self.interpolation = interpolation
        self.interpolation_factor = (
            None if interpolation_factor is None else int(interpolation_factor)
        )
        self.learn_aberrations = learn_aberrations
        self.learn_beam_coefficients = learn_beam_coefficients
        self._vacuum_probe_intensity = None
        self.vacuum_probe_intensity = vacuum_probe_intensity

        # per-mode aberration coefficients; probe_params aberrations are the shared default
        base_coefs = dict(self.probe_params.get("aberration_coefs", {}))
        if aberration_coefs is None:
            mode_coefs = [dict(base_coefs) for _ in range(num_probes)]
        elif isinstance(aberration_coefs, dict):
            merged = base_coefs | {
                k: float(v) for k, v in standardize_aberration_coefs(aberration_coefs).items()
            }
            mode_coefs = [dict(merged) for _ in range(num_probes)]
        else:
            if len(aberration_coefs) != num_probes:
                raise ValueError(
                    f"aberration_coefs list length {len(aberration_coefs)} does not match "
                    f"num_probes {num_probes}"
                )
            mode_coefs = [
                base_coefs | {k: float(v) for k, v in standardize_aberration_coefs(d).items()}
                for d in aberration_coefs
            ]

        self.aberration_coefs = nn.ModuleList()
        for coefs in mode_coefs:
            self.aberration_coefs.append(
                nn.ParameterDict(
                    {
                        k: nn.Parameter(
                            torch.tensor(float(v), dtype=torch.float32),
                            requires_grad=learn_aberrations,
                        )
                        for k, v in coefs.items()
                    }
                )
            )
        for p, param_dict in enumerate(self.aberration_coefs):
            for k, param in param_dict.items():
                self.register_buffer(f"_initial_aberration_coefs_{p}_{k}", param.detach().clone())

        weights = (
            np.full(num_probes, 1.0 / num_probes)
            if initial_probe_weights is None
            else np.asarray(initial_probe_weights, dtype=np.float64)
        )
        if len(weights) != num_probes:
            raise ValueError(
                f"initial_probe_weights length {len(weights)} does not match "
                f"num_probes {num_probes}"
            )
        self._initial_probe_weights = weights / weights.sum()

    @property
    def vacuum_probe_intensity(self) -> torch.Tensor | None:
        """Measured aperture intensity (corner-centered); replaces the analytic aperture."""
        return self._vacuum_probe_intensity

    @vacuum_probe_intensity.setter
    def vacuum_probe_intensity(self, vp: np.ndarray | torch.Tensor | Dataset4dstem | None):
        if vp is None:
            self._vacuum_probe_intensity = None
            return
        if isinstance(vp, (Dataset4dstem, Dataset2d)):
            vp = cast(np.ndarray, vp.array)
        vp_t = torch.as_tensor(np.asarray(vp), dtype=torch.float32)
        if vp_t.ndim == 4:
            vp_t = vp_t.mean(dim=(0, 1))
        elif vp_t.ndim != 2:
            raise ValueError(f"Unexpected shape for vacuum probe: {tuple(vp_t.shape)}")
        self._vacuum_probe_intensity = vp_t.to(self.device)

    @classmethod
    def from_params(
        cls,
        probe_params: dict,
        num_probes: int = 1,
        aberration_coefs: dict | list[dict] | None = None,
        num_partitions: int = 4,
        dense: bool = False,
        parent_layout: str = "rings",
        interpolation: str = "sibson",
        interpolation_factor: int | None = None,
        learn_aberrations: bool = True,
        learn_beam_coefficients: bool = False,
        initial_probe_weights: list[float] | np.ndarray | None = None,
        probe_tilt: tuple[float, float] | torch.Tensor = (0, 0),
        learn_probe_tilt: bool = False,
        roi_shape: tuple[int, int] | np.ndarray | None = None,
        vacuum_probe_intensity: np.ndarray | Dataset4dstem | None = None,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        max_aberrations_order: int | None = None,
    ) -> "ProbePRISM":
        return cls(
            num_probes=num_probes,
            probe_params=probe_params.copy(),
            aberration_coefs=aberration_coefs,
            num_partitions=num_partitions,
            dense=dense,
            parent_layout=parent_layout,
            interpolation=interpolation,
            interpolation_factor=interpolation_factor,
            learn_aberrations=learn_aberrations,
            learn_beam_coefficients=learn_beam_coefficients,
            initial_probe_weights=initial_probe_weights,
            probe_tilt=probe_tilt,
            learn_probe_tilt=learn_probe_tilt,
            roi_shape=roi_shape,
            vacuum_probe_intensity=vacuum_probe_intensity,
            device=device,
            rng=rng,
            max_aberrations_order=max_aberrations_order,
            _token=cls._token,
        )

    # region --- geometry ---

    @property
    def wavelength(self) -> float:
        """Electron wavelength in Angstroms."""
        return electron_wavelength_angstrom(self.probe_params["energy"])

    @property
    def extent(self) -> np.ndarray:
        """Real-space ROI extent in Angstroms."""
        return 1 / self.reciprocal_sampling

    @property
    def sampling(self) -> np.ndarray:
        """Real-space sampling in Angstroms."""
        return 1 / (self.roi_shape * self.reciprocal_sampling)

    @property
    def angular_sampling(self) -> np.ndarray:
        """Angular sampling in mrad."""
        return self.reciprocal_sampling * self.wavelength * 1e3

    @property
    def parent_wave_vectors(self) -> torch.Tensor:
        """Parent beam wave vectors in inverse Angstroms, shape (num_parent, 2)."""
        return self._parent_wave_vectors

    @property
    def beamlet_wave_vectors(self) -> torch.Tensor:
        """Dense beamlet wave vectors in inverse Angstroms, shape (num_beamlets, 2)."""
        return self._beamlet_wave_vectors

    @property
    def interpolation_weights(self) -> torch.Tensor:
        """Frozen Sibson weight maps, shape (num_parent, *roi_shape)."""
        return self._interpolation_weights

    @property
    def num_parent_beams(self) -> int:
        return self._parent_wave_vectors.shape[0]

    @property
    def coefficient_window(self) -> torch.Tensor | None:
        """Real-space crop mask for the classic ('nearest') scheme, else ``None``.

        The engine multiplies the reduced coefficient maps by this corner-centered
        box to discard the ghost probes that the one-hot coarse aperture replicates.
        """
        if getattr(self, "_has_coefficient_window", False):
            return self._coefficient_window_mask
        return None

    def _make_coefficient_window(self, gpts: tuple[int, int], f: int) -> torch.Tensor:
        """Corner-centered box mask of size ``gpts // f`` (classic-PRISM crop)."""
        wx = max(1, gpts[0] // f)
        wy = max(1, gpts[1] // f)
        # signed offset from the corner, where the (corner-centered) probe sits
        ox = ((torch.arange(gpts[0]) + gpts[0] // 2) % gpts[0]) - gpts[0] // 2
        oy = ((torch.arange(gpts[1]) + gpts[1] // 2) % gpts[1]) - gpts[1] // 2
        mask = (ox.abs()[:, None] <= wx // 2) & (oy.abs()[None, :] <= wy // 2)
        return mask.to(torch.float32)

    def _nearest_norm_correction(self) -> float:
        """Frozen scalar rescaling the windowed one-hot probe to the mean intensity.

        The one-hot ('nearest') aperture is subsampled by the interpolation factor
        and its probe is cropped to :attr:`coefficient_window`; both change the
        probe power relative to the dense aperture that ``_norm_factor`` assumes.
        """
        with torch.no_grad():
            basis_sum = (self._mode_ctf(0)[None] * self._interpolation_weights).sum(0)
            probe = torch.fft.ifft2(basis_sum * self._norm_factor) * self._coefficient_window_mask
            intensity = torch.sum(torch.abs(torch.fft.fft2(probe, norm="ortho")).square())
            target = torch.as_tensor(self.mean_diffraction_intensity, dtype=intensity.dtype)
            return float(torch.sqrt(target / intensity.clamp_min(1e-12)))

    # endregion --- geometry ---

    def _set_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """Register a buffer, replacing it if it already exists."""
        if hasattr(self, name):
            delattr(self, name)
        self.register_buffer(name, tensor)

    def set_initial_probe(
        self,
        roi_shape: np.ndarray | tuple,
        reciprocal_sampling: np.ndarray,
        mean_diffraction_intensity: float,
        device: str | None = None,
    ):
        """Compute the PRISM beam geometry and initialize the learnable coefficients."""
        super().set_initial_probe(
            roi_shape, reciprocal_sampling, mean_diffraction_intensity, device
        )

        gpts = tuple(self.roi_shape.astype(int))
        sampling = self.sampling
        cutoff = float(self.probe_params["semiangle_cutoff"]) + float(
            np.linalg.norm(self.angular_sampling)
        )

        beamlet_kv = _prism_wave_vectors(cutoff, self.extent, self.wavelength)
        coefficient_window: torch.Tensor | None = None
        f = self.interpolation_factor
        if self.dense:
            parent_kv = beamlet_kv
            weights = one_hot_beamlet_weights(beamlet_kv, gpts, sampling)
        elif self.parent_layout == "rings":
            parent_kv = _partitioned_prism_wave_vectors(
                cutoff, self.wavelength, num_rings=self.num_partitions
            )
            weights = beamlet_weights(parent_kv, beamlet_kv, gpts, sampling)
        elif self.interpolation == "nearest":
            # classic Fourier-interpolation PRISM: one-hot coarse beams (no dense
            # interpolation), the approximation is the real-space crop window
            parent_kv = _grid_prism_wave_vectors(
                cutoff, self.extent, self.wavelength, f, margin=0.0
            )
            weights = one_hot_beamlet_weights(parent_kv, gpts, sampling)
            coefficient_window = self._make_coefficient_window(gpts, f)
        elif self.interpolation == "fourier":
            # C-PRISM interpolant (minus SVD): trigonometric interpolation of the
            # coarse grid up to the full dense aperture
            parent_kv = _grid_prism_wave_vectors(cutoff, self.extent, self.wavelength, f)
            weights = fourier_beamlet_weights(parent_kv, beamlet_kv, gpts, sampling, f)
        else:  # grid parents with Sibson weights (ablation)
            parent_kv = _grid_prism_wave_vectors(cutoff, self.extent, self.wavelength, f)
            weights = beamlet_weights(parent_kv, beamlet_kv, gpts, sampling)

        self._has_coefficient_window = coefficient_window is not None
        if coefficient_window is not None:
            self._set_buffer("_coefficient_window_mask", coefficient_window.to(self.device))

        kxa, kya = spatial_frequencies(gpts, sampling, device=self.device)
        self._set_buffer("_kxa", kxa.contiguous())
        self._set_buffer("_kya", kya.contiguous())
        self._set_buffer(
            "_norm_factor",
            torch.sqrt(
                torch.tensor(mean_diffraction_intensity * gpts[0] * gpts[1], dtype=torch.float32)
            ).to(self.device),
        )

        already_initialized = hasattr(
            self, "_beam_coefficients"
        ) and self._beam_coefficients.shape[1] == len(parent_kv)
        if already_initialized:
            return

        def to_t(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(arr)).to(
                device=self.device, dtype=torch.float32
            )

        self._set_buffer("_parent_wave_vectors", to_t(parent_kv))
        self._set_buffer("_beamlet_wave_vectors", to_t(beamlet_kv))
        self._set_buffer("_interpolation_weights", to_t(weights))

        if self._vacuum_probe_intensity is not None:
            # beamlets exist only inside the padded semiangle disk; a measured
            # aperture extending beyond it would be silently truncated
            vacuum = self._vacuum_probe_intensity
            beamlet_support = self._interpolation_weights.sum(dim=0) > 0
            covered = vacuum[beamlet_support].sum() / vacuum.sum().clamp_min(1e-12)
            if covered < 0.999:
                warn(
                    f"vacuum_probe_intensity has {100 * (1 - covered.item()):.2f}% of its "
                    "weight outside the beamlet disk (semiangle_cutoff + padding); "
                    "increase semiangle_cutoff to cover the measured aperture."
                )

        mode_amplitudes = torch.sqrt(
            torch.from_numpy(self._initial_probe_weights).to(torch.float32)
        )
        # the one-hot 'nearest' scheme subsamples the aperture and crops the probe,
        # so its (windowed) probe power differs from the dense aperture; a frozen
        # init-time scalar rescales it to the mean diffraction intensity
        norm_correction = 1.0
        if self._has_coefficient_window:
            norm_correction = self._nearest_norm_correction()
        beam_coefs = torch.zeros(self.num_probes, len(parent_kv), 2, dtype=torch.float32)
        beam_coefs[:, :, 0] = mode_amplitudes[:, None] * norm_correction
        if self.num_probes > 1 and self.learn_beam_coefficients:
            # identical initial modes have parallel gradients and can never
            # differentiate under optimization; break the symmetry with a small
            # deterministic per-parent perturbation (rng-seeded)
            noise = torch.from_numpy(self.rng.standard_normal(tuple(beam_coefs.shape))).to(
                torch.float32
            )
            beam_coefs = beam_coefs + 0.1 * mode_amplitudes[:, None, None] * noise
        self._beam_coefficients = nn.Parameter(
            beam_coefs.to(self.device), requires_grad=self.learn_beam_coefficients
        )
        self._set_buffer("_initial_beam_coefficients", beam_coefs.clone().to(self.device))

    # region --- core compute ---

    def _compute_beamlet_basis_fft(self) -> torch.Tensor:
        """CTF- and coefficient-weighted beamlet basis in reciprocal space.

        Returns
        -------
        torch.Tensor
            Shape (num_probes, num_parent, *roi_shape), complex.
        """
        basis = [
            self._mode_ctf(p)[None] * self._interpolation_weights for p in range(self.num_probes)
        ]

        beam_coefs = torch.view_as_complex(self._beam_coefficients)
        return torch.stack(basis) * beam_coefs[:, :, None, None] * self._norm_factor

    def _mode_ctf(self, p: int) -> torch.Tensor:
        """Aberrated (normalized) Fourier-space aperture for probe mode ``p``."""
        coefs: dict[str, Any] = {k: v for k, v in self.aberration_coefs[p].items()}
        return fourier_space_probe(
            gpts=tuple(self.roi_shape.astype(int)),
            sampling=tuple(self.sampling),
            energy=self.probe_params["energy"],
            semiangle_cutoff=float(self.probe_params["semiangle_cutoff"]),
            soft_edges=self.probe_params["soft_edges"],
            vacuum_probe_intensity=self._vacuum_probe_intensity,
            aberration_coefs=coefs,
            normalized=True,
            device=self.device,
        )

    def _position_coefficients(self, fract_positions: torch.Tensor) -> torch.Tensor:
        """Fractional-position phase ramps exp(-2 pi i k . dr), shape (batch, *roi_shape)."""
        sampling = torch.as_tensor(self.sampling, dtype=self._kxa.dtype, device=self._kxa.device)
        positions_A = fract_positions.to(self._kxa.device) * sampling[None]
        phase = positions_A[:, 0, None, None] * self._kxa[None] + (
            positions_A[:, 1, None, None] * self._kya[None]
        )
        return torch.exp(-2j * torch.pi * phase)

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]  # PRISM returns basis + phases, engine owns the reduction
        self, fract_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(beamlets_fft, position_coefs)`` for the PRISM engine's reduction.

        Only consumed by ``PtychographyPRISM``; the base probe contract (shifted
        probe stacks) is intentionally not implemented.
        """
        return (
            self._compute_beamlet_basis_fft(),
            self._position_coefficients(fract_positions),
        )

    # endregion --- core compute ---

    # region --- contract ---

    @property
    def probe(self) -> torch.Tensor:
        """Real-space probe stack (num_probes, *roi_shape): summed beamlet basis.

        For the classic ('nearest') scheme the corner-centered crop window is
        applied, matching what the engine reduces (and discarding the ghost copies).
        """
        probe = torch.fft.ifft2(self._compute_beamlet_basis_fft().sum(dim=1))
        window = self.coefficient_window
        if window is not None:
            probe = probe * window[None]
        return probe

    @property
    def params(self) -> list[nn.Parameter]:
        """Optimization parameters."""
        params = super().params
        if self.learn_aberrations:
            params.extend(p for pd in self.aberration_coefs for p in pd.values())
        if self.learn_beam_coefficients and hasattr(self, "_beam_coefficients"):
            params.append(self._beam_coefficients)
        return params

    @property
    def name(self) -> str:
        return "ProbePRISM"

    def get_optimization_parameters(self) -> "dict[str, list[torch.Tensor]]":
        """Aberrations / beam coefficients / probe tilt as separate PPLR groups.

        Returns one group per *learnable* parameter set; ``{}`` when nothing is
        learnable (``set_optimizer`` then short-circuits to removing the optimizer).
        """
        groups: dict[str, list[torch.Tensor]] = {}
        if self.learn_aberrations:
            aberrations = [p for pd in self.aberration_coefs for p in pd.values()]
            if aberrations:
                groups["aberrations"] = aberrations
        if self.learn_beam_coefficients and hasattr(self, "_beam_coefficients"):
            groups["beam_coefficients"] = [self._beam_coefficients]
        if self.learn_probe_tilt:
            groups["probe_tilt"] = [self._probe_tilt]
        return groups

    def _normalize_optimizer_params(self, params):
        """Broadcast a single optimizer spec to the learnable PPLR groups.

        A single ``OptimizerParamsType`` / single-optimizer dict (normalized to the
        ``"default"`` key) is fanned out to whichever groups are currently learnable,
        so the common single-LR caller keeps working. An explicit PPLR dict (keyed by
        ``aberrations`` / ``beam_coefficients`` / ``probe_tilt``) passes through.
        """
        norm = super()._normalize_optimizer_params(params)
        if set(norm) == {self.DEFAULT_OPTIMIZER_KEY}:
            spec = norm[self.DEFAULT_OPTIMIZER_KEY]
            active = list(self.get_optimization_parameters().keys())
            if not active and not isinstance(spec, OptimizerParams.NoneOptimizer):
                warn(
                    f"{type(self).__name__}: an optimizer was requested but nothing is "
                    "learnable; the optimizer will be removed. Enable learn_aberrations, "
                    "learn_beam_coefficients and/or learn_probe_tilt to optimize.",
                    stacklevel=2,
                )
            return {key: replace(spec) for key in active} if active else {}
        return norm

    def reset(self):
        """Reset learnable parameters to their initial values."""
        super().reset()
        with torch.no_grad():
            for p, param_dict in enumerate(self.aberration_coefs):
                for k, param in param_dict.items():
                    initial = getattr(self, f"_initial_aberration_coefs_{p}_{k}")
                    param.data.copy_(initial)
            if hasattr(self, "_beam_coefficients"):
                self._beam_coefficients.data.copy_(
                    self._initial_beam_coefficients.to(self._beam_coefficients.device)
                )

    def apply_hard_constraints(self, probe: torch.Tensor) -> torch.Tensor:
        """Pixel-domain projections don't apply to coefficient-parameterized probes."""
        return probe

    def apply_soft_constraints(self, probe: torch.Tensor) -> torch.Tensor:
        self.reset_soft_constraint_losses()
        loss = self._get_zero_loss_tensor()
        self.accumulate_constraint_losses()
        return loss

    # endregion --- contract ---

    def show_interpolation_weights(self, ax: "plt.Axes | None" = None):
        """RGB overlay of the per-parent Sibson weight maps."""
        from matplotlib.colors import to_rgb

        weights = to_numpy(self._interpolation_weights)

        color_cycle = [["c", "r"], ["m", "g"], ["b", "y"]]
        colors = ["w"]
        i = 1
        while len(colors) < len(weights):
            colors += color_cycle[(i - 1) % 3] * (3 + (i - 1) * 3)
            i += 1

        rgb = np.array([to_rgb(color) for color in colors[: len(weights)]])
        color_map = np.tensordot(weights.transpose(1, 2, 0), rgb, axes=([2], [0]))
        color_map = np.fft.fftshift(color_map, axes=(0, 1))

        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(np.clip(color_map, 0, 1))
        ax.set(xticks=[], yticks=[], title="PRISM interpolation weights")
        return ax


ProbeModelType = ProbePixelated | ProbeDIP | ProbePRISM
