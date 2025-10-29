import math
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Callable, Self, Union

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
from quantem.core.ml.loss_functions import get_loss_function
from quantem.core.ml.optimizer_mixin import OptimizerMixin
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
from quantem.diffractive_imaging.complex_probe import (
    POLAR_ALIASES,
    POLAR_SYMBOLS,
    evaluate_probe,
    real_space_probe,
)
from quantem.diffractive_imaging.constraints import BaseConstraints
from quantem.diffractive_imaging.ptycho_utils import (
    fourier_shift_expand,
    shift_array,
)

DeviceType = Union[str, torch.device, int]


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
        self._device = device
        self._probe_params = self.DEFAULT_PROBE_PARAMS
        self._max_aberrations_order = max_aberrations_order
        self.probe_params = probe_params
        self._constraints = {}
        self.rng = rng
        if roi_shape is not None:
            self.roi_shape = roi_shape

    def get_optimization_parameters(self):
        """Get the parameters that should be optimized for this model."""
        try:
            params = self.params
            if params is None:
                return []
            return params
        except NotImplementedError:
            # This happens when params is not implemented yet in abstract base
            return []

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
    def probe_params(self, params: dict[str, Any] = {}):
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

        polar_parameters = set_aberrations(params.copy(), self._max_aberrations_order)
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
    def params(self):
        """optimization parameters"""
        raise NotImplementedError()

    @property
    def model_input(self):
        """get the model input"""
        raise NotImplementedError()

    def forward(self, fract_positions: torch.Tensor) -> torch.Tensor:
        """Get probe positions"""
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """Reset the probe"""
        raise NotImplementedError()

    def _initialize_probe(
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


class ProbeConstraints(BaseConstraints, ProbeBase):
    DEFAULT_CONSTRAINTS = {
        # "fix_probe": False,
        "orthogonalize_probe": True,
        "center_probe": False,
        "tv_weight": 0.0,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_soft_constraints(self, probe: torch.Tensor) -> torch.Tensor:
        self.reset_soft_constraint_losses()
        loss = self._get_zero_loss_tensor()
        if self.constraints["tv_weight"]:
            loss_tv = self._probe_tv_constraint(probe, self.constraints["tv_weight"])
            self.add_soft_constraint_loss("tv_loss", loss_tv)
            loss = loss + loss_tv

        self.accumulate_constraint_losses()
        return loss

    def apply_hard_constraints(self, probe: torch.Tensor) -> torch.Tensor:
        # if self.constraints["fix_probe"]:
        #     return self.initial_probe
        if self.constraints["orthogonalize_probe"]:
            probe = self._probe_orthogonalization_constraint(probe)
        if self.constraints["center_probe"]:
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
        dtype: torch.dtype = torch.complex64,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        initial_probe_weights: list[float] | np.ndarray | None = None,
    ):
        if isinstance(probe_array, np.ndarray):
            probe_array = torch.tensor(probe_array, dtype=dtype, device=device)
        if probe_array.ndim == 3:
            if num_probes is None:
                num_probes = probe_array.shape[0]
            elif num_probes != probe_array.shape[0]:
                raise ValueError(
                    f"num_probes {num_probes} must match probe_array shape {probe_array.shape[0]}"
                )
        else:
            num_probes = 1 if num_probes is None else num_probes
            probe_array = torch.tensor(probe_array, dtype=dtype, device=device)
            probe_array = torch.tile(probe_array, (num_probes, 1, 1))

        probe_model = cls(
            num_probes=num_probes,
            probe_params=probe_params.copy(),
            roi_shape=(int(probe_array.shape[-2]), int(probe_array.shape[-1])),
            dtype=dtype,
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
        dtype: torch.dtype = torch.complex64,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        initial_probe_weights: list[float] | np.ndarray | None = None,
        vacuum_probe_intensity: Dataset4dstem | np.ndarray | None = None,
    ):
        probe_model = cls(
            num_probes=num_probes,
            probe_params=probe_params.copy(),
            roi_shape=roi_shape,
            dtype=dtype,
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
    def params(self):
        """optimization parameters"""
        return self._probe

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
        super()._initialize_probe(
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
            probes = torch.tile(probes, (self.num_probes, 1, 1))

        probes = self._apply_random_phase_shifts(probes)
        probes = self._apply_weights(probes)

        self._initial_probe = self._to_torch(probes)
        self._probe = nn.Parameter(self._initial_probe.clone().to(self.device), requires_grad=True)
        return

    def reset(self):
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
            vp2 = vp.array
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

        self._vacuum_probe_intensity = torch.tensor(
            vp2, dtype=config.get("dtype_real"), device=self.device
        )

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
            dtype=config.get("dtype_real"),
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
            vp2 = vp.array
        else:
            raise NotImplementedError(f"Unknown vacuum probe type: {type(vp)}")

        if vp2.ndim == 4:
            vp2 = np.mean(vp2, axis=(0, 1))
        elif vp2.ndim != 2:
            raise ValueError(f"Unexpected shape for vacuum probe: {vp2.shape}")

        self._vacuum_probe_intensity = vp2

    @property
    def params(self):
        """Optimization parameters."""
        params = []
        if isinstance(self.semiangle_cutoff, nn.Parameter):
            params.append(self.semiangle_cutoff)
        params += list(self.aberration_coefs.values())
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
        num_probes: int = 1,
        probe_params: dict = {},
        roi_shape: tuple[int, int] | np.ndarray | None = None,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        _token: object | None = None,
    ):
        super().__init__(
            num_probes=num_probes,
            probe_params=probe_params.copy(),
            roi_shape=roi_shape,
            device=device,
            rng=rng,
            _token=_token,
        )
        self.register_buffer("_model_input", torch.tensor([]))
        self.register_buffer("_pretrain_target", torch.tensor([]))

        self._optimizer = None
        self._scheduler = None
        self._pretrain_losses: list[float] = []
        self._pretrain_lrs: list[float] = []

    @classmethod
    def from_model(
        cls,
        model: "torch.nn.Module",
        model_input: torch.Tensor | None = None,
        num_probes: int = 1,
        probe_params: dict = {},
        roi_shape: tuple[int, int] | np.ndarray | None = None,
        input_noise_std: float = 0.025,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        probe_model = cls(
            num_probes=num_probes,
            probe_params=probe_params.copy(),
            roi_shape=roi_shape,
            device=device,
            rng=rng,
            _token=cls._token,
        )
        probe_model._model = model.to(device)
        probe_model.set_pretrained_weights(model)

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
        if not isinstance(pixelated, ProbePixelated):
            raise ValueError(f"Pixelated must be an ObjectPixelated, got {type(pixelated)}")

        probe_model = cls(
            num_probes=pixelated.num_probes,
            probe_params=pixelated.probe_params.copy(),
            roi_shape=pixelated.roi_shape,
            device=device,
            rng=pixelated._rng_seed,
            _token=cls._token,
        )

        probe_model._model = model.to(device)
        probe_model.set_pretrained_weights(model)

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
        super()._initialize_probe(
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
    def params(self):
        """optimization parameters"""
        return self.model.parameters()

    def get_optimization_parameters(self):
        """Get the parameters that should be optimized for this model."""
        # Return a fresh list of parameters each time to avoid generator exhaustion
        return list(self.model.parameters())

    def reset(self):
        """Reset the object model to its initial or pre-trained state"""
        self.model.load_state_dict(self.pretrained_weights.copy())

    def pretrain(
        self,
        model_input: torch.Tensor | None = None,
        pretrain_target: torch.Tensor | None = None,
        reset: bool = False,
        num_epochs: int = 100,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        loss_fn: Callable | str = "l2",
        apply_constraints: bool = False,
        show: bool = True,
        device: str | None = None,  # allow overwriting of device
    ):
        if device is not None:
            self.to(device)

        if optimizer_params is not None:
            self.set_optimizer(optimizer_params)

        if scheduler_params is not None:
            self.set_scheduler(scheduler_params, num_epochs)

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

        loss_fn = get_loss_function(loss_fn, self.dtype)
        self._pretrain(
            num_epochs=num_epochs,
            loss_fn=loss_fn,
            apply_constraints=apply_constraints,
            show=show,
        )
        self.set_pretrained_weights(self.model)

    def _pretrain(
        self,
        num_epochs: int,
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
        pbar = tqdm(range(num_epochs))
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
            pbar.set_description(f"Epoch {a0 + 1}/{num_epochs}, Loss: {loss.item():.3e}, ")

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
        ax.set_xlabel("Epochs")
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
            cbar=True,
        )
        plt.suptitle(
            f"Final loss: {self._pretrain_losses[-1]:.3e} | Epochs: {len(self._pretrain_losses)}",
            fontsize=14,
            y=0.94,
        )
        plt.show()

    def backward(self, propagated_gradient, obj_patches):
        """Backward pass for analytical gradients (not implemented for DIP)"""
        raise NotImplementedError(
            f"Analytical gradients are not implemented for {self.name}, use autograd=True"
        )


class ProbePRISM(ProbeBase):
    """
    PRISM probe model that computes probe coefficients for plane wave decomposition.
    """

    def __init__(
        self,
        num_probes: int | None = None,
        probe_params: dict = {},
        roi_shape: tuple[int, int] | np.ndarray | None = None,
        interpolation: tuple[int, int] = (1, 1),
        learn_aberrations: bool = True,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        _token: object | None = None,
    ):
        """
        Parameters
        ----------
        aberration_coefs_list : Sequence[dict]
            List of aberration coefficient dictionaries, one per probe mode
        num_probes : int | None
            Number of probe modes (if None, inferred from aberration_coefs_list)
        probe_params : dict
            Additional probe parameters (must contain 'energy')
        roi_shape : tuple[int, int] | np.ndarray | None
            Shape of the probe ROI
        interpolation : tuple[int, int]
            PRISM interpolation factors (f_x, f_y)
        device : str
            Device to use ('cpu' or 'cuda')
        rng : np.random.Generator | int | None
            Random number generator
        """

        # handle list of aberrations or defocus
        coefs = self._extract_aberration_coefs(probe_params.copy())

        if num_probes is None:
            num_probes = len(coefs)
        elif num_probes != len(coefs):
            raise ValueError(
                f"num_probes {num_probes} does not match length of aberration_coefs {len(coefs)}"
            )

        super().__init__(
            num_probes=num_probes,
            probe_params=probe_params,
            roi_shape=roi_shape,
            device=device,
            rng=rng,
            _token=_token,
        )

        self.interpolation = interpolation
        self.aberration_coefs_list = coefs
        self.learn_aberrations = learn_aberrations
        self._wave_vectors = None
        self._plane_waves = None

        self._ctf_params = None
        self._intensity_norm_factor = 1.0

    @classmethod
    def from_params(
        cls,
        probe_params: dict,
        num_probes: int | None = None,
        roi_shape: tuple[int, int] | None = None,
        interpolation: tuple[int, int] = (1, 1),
        learn_aberrations: bool = True,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        """
        Create ProbePRISM from parameters.

        Parameters
        ----------
        probe_params : dict
            Probe parameters including 'energy' (in eV)
        num_probes : int | None
            Number of probe modes
        roi_shape : tuple[int, int] | None
            Shape of the probe ROI
        interpolation : tuple[int, int]
            PRISM interpolation factors
        device : str
            Device to use
        rng : np.random.Generator | int | None
            Random number generator
        """
        return cls(
            num_probes=num_probes,
            probe_params=probe_params,
            roi_shape=roi_shape,
            interpolation=interpolation,
            learn_aberrations=learn_aberrations,
            device=device,
            rng=rng,
            _token=cls._token,
        )

    @property
    def interpolation(self) -> tuple[int, int]:
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interp: tuple[int, int] | list):
        if len(interp) != 2:
            raise ValueError(f"interpolation must have length 2, got {len(interp)}")
        self._interpolation = (int(interp[0]), int(interp[1]))

    @property
    def wave_vectors(self) -> torch.Tensor:
        """Wave vectors for plane wave decomposition [num_waves, 2]"""
        return self._wave_vectors

    @property
    def plane_waves(self) -> torch.Tensor:
        """Plane waves in real space [num_waves, roi_height, roi_width]"""
        return self._plane_waves

    @property
    def ctf_coefs(self) -> torch.Tensor:
        """
        Compute CTF coefficients on-the-fly from learnable parameters.
        Returns [num_probes, num_waves]
        """
        if self._ctf_params is None:
            raise RuntimeError("CTF parameters not initialized")

        # Recompute CTF from current parameter values
        return self._compute_ctf_from_params()

    @property
    def probe(self) -> torch.Tensor:
        """
        Get probe in real space by reconstructing from PRISM basis at origin.
        Returns [num_probes, roi_height, roi_width]
        """

        # Reconstruct probes
        probes = torch.tensordot(
            self.ctf_coefs,
            self.plane_waves,
            dims=((1,), (0,)),
        )  # [num_probes, roi_height, roi_width]

        return probes

    @property
    def params(self):
        """Return learnable parameters for optimization"""

        if self.learn_aberrations:
            params = []
            # Return all aberration parameters
            for probe_idx in range(self.num_probes):
                params.extend(self._ctf_params[probe_idx].values())
            return params
        else:
            return None

        return params

    @property
    def name(self) -> str:
        return "ProbePRISM"

    def set_initial_probe(
        self,
        roi_shape: np.ndarray | tuple,
        reciprocal_sampling: np.ndarray,
        mean_diffraction_intensity: float,
        device: str | None = None,
    ):
        """
        Initialize PRISM probe by computing plane wave basis and CTF coefficients.
        """
        super()._initialize_probe(
            roi_shape, reciprocal_sampling, mean_diffraction_intensity, device
        )

        # Check required probe parameters
        if self.probe_params.get("energy", None) is None:
            raise ValueError("probe_params must contain 'energy' for PRISM")
        if self.probe_params.get("semiangle_cutoff", None) is None:
            raise ValueError("probe_params must contain 'semiangle_cutoff' for PRISM")

        energy = self.probe_params.get("energy")
        semiangle_cutoff = self.probe_params.get("semiangle_cutoff")

        wavelength = electron_wavelength_angstrom(energy)
        extent = tuple((1 / self.reciprocal_sampling).astype(np.float32))
        angular_sampling = wavelength * 1e3 * self.reciprocal_sampling

        # Compute PRISM wave vectors
        self._wave_vectors = self._prism_wave_vectors(
            semiangle_cutoff=semiangle_cutoff + np.linalg.norm(angular_sampling),
            extent=extent,
            wavelength=wavelength,
            interpolation=self.interpolation,
        )

        # Compute plane waves
        self._plane_waves = self._prism_plane_waves(
            wave_vectors=self.wave_vectors,
            extent=extent,
            gpts=tuple(self.roi_shape.astype(int)),
        ) / math.sqrt(math.prod(self.roi_shape))

        # Initialize learnable CTF parameters for each probe mode
        self._ctf_params = nn.ModuleList()
        for probe_idx, aberration_coefs in enumerate(self.aberration_coefs_list):
            probe_params = nn.ParameterDict()

            for key, value in aberration_coefs.items():
                if self.learn_aberrations:
                    probe_params[key] = nn.Parameter(
                        torch.tensor(float(value), dtype=torch.float32, device=self.device)
                    )
                else:
                    probe_params.register_buffer(
                        key, torch.tensor(float(value), dtype=torch.float32, device=self.device)
                    )

            self._ctf_params.append(probe_params)

        # Store initial values for reset
        self._store_initial_ctf_params()

        # Normalize initial probe
        self._normalize_probe_intensity(mean_diffraction_intensity)

    def _store_initial_ctf_params(self):
        """Store initial parameter values for reset."""
        self._initial_ctf_params = []
        for probe_params in self._ctf_params:
            initial_dict = {}
            for key, val in probe_params.items():
                initial_dict[key] = val.detach().clone()
            self._initial_ctf_params.append(initial_dict)

    def _compute_ctf_from_params(self) -> torch.Tensor:
        """
        Compute CTF coefficients from current learnable parameters.
        Returns [num_probes, num_waves]
        """
        ctf_coefs_list = []

        # Get current semiangle
        semiangle = self.probe_params.get("semiangle_cutoff")
        energy = self.probe_params.get("energy")
        wavelength = electron_wavelength_angstrom(energy)

        angular_sampling = wavelength * 1e3 * self.reciprocal_sampling

        for probe_idx in range(self.num_probes):
            # Get current aberration coefficients for this probe
            aberration_coefs = {key: val for key, val in self._ctf_params[probe_idx].items()}

            # Compute CTF for this probe mode
            ctf_coef = self._ctf_coefficients(
                wave_vectors=self.wave_vectors,
                semiangle_cutoff=semiangle,
                angular_sampling=tuple(angular_sampling.astype(float)),
                wavelength=wavelength,
                aberration_coefs=aberration_coefs,
            )
            ctf_coefs_list.append(ctf_coef)

        return torch.stack(ctf_coefs_list) * self._intensity_norm_factor

    def _normalize_probe_intensity(self, mean_diffraction_intensity: float):
        """Normalize probe intensity to match experimental data."""
        # Compute current probe
        current_ctf = self._compute_ctf_from_params()
        probe = torch.tensordot(current_ctf, self.plane_waves, dims=((1,), (0,)))

        # Compute intensity normalization
        probe_intensity = torch.sum(torch.abs(probe).square())
        intensity_norm = torch.sqrt(mean_diffraction_intensity / probe_intensity)
        self._intensity_norm_factor = intensity_norm.detach().item()

    def reset(self):
        """Reset learnable parameters to initial values."""
        if not hasattr(self, "_initial_ctf_params"):
            return

        with torch.no_grad():
            for probe_idx, initial_dict in enumerate(self._initial_ctf_params):
                for key, initial_val in initial_dict.items():
                    self._ctf_params[probe_idx][key].copy_(initial_val)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute PRISM coefficients for given probe positions.

        Parameters
        ----------
        fract_positions : torch.Tensor
            Fractional probe positions [batch_size, 2]

        Returns
        -------
        torch.Tensor
            PRISM coefficients [num_probes, batch_size, num_waves]
        """
        # Compute position coefficients
        position_coefs = self._position_coefficients(
            positions=positions,
            wave_vectors=self.wave_vectors,
        )  # [batch_size, num_waves]

        # Combine with CTF coefficients
        coefs = position_coefs[None, :, :] * self.ctf_coefs[:, None, :]
        # [num_probes, batch_size, num_waves]

        return coefs

    # --- PRISM helper functions ---

    def _prism_wave_vectors(
        self,
        semiangle_cutoff: float,
        extent: tuple[float, float],
        wavelength: float,
        interpolation: tuple[int, int],
    ) -> torch.Tensor:
        """
        Returns planewave wave_vectors.

        Parameters
        ----------
        semiangle_cutoff : float
            Convergence semi-angle in mrad
        extent : tuple[float, float]
            Real-space extent in Angstroms
        wavelength : float
            Electron wavelength in Angstroms
        interpolation : tuple[int, int]
            PRISM interpolation factors

        Returns
        -------
        torch.Tensor
            Wave vectors [num_waves, 2]
        """
        cutoff = semiangle_cutoff * 1e-3
        n_max = math.ceil(cutoff / (wavelength / extent[0] * interpolation[0]))
        m_max = math.ceil(cutoff / (wavelength / extent[1] * interpolation[1]))

        n = torch.arange(-n_max, n_max + 1, dtype=torch.float32, device=self.device)
        m = torch.arange(-m_max, m_max + 1, dtype=torch.float32, device=self.device)
        w, h = extent[0], extent[1]
        int_n, int_m = float(interpolation[0]), float(interpolation[1])

        kx = n / w * int_n
        ky = m / h * int_m

        mask = kx[:, None].square() + ky[None, :].square() < (cutoff / wavelength) ** 2

        kx, ky = torch.meshgrid(kx, ky, indexing="ij")
        kx = kx[mask]
        ky = ky[mask]

        return torch.stack((kx, ky), dim=-1)

    def _prism_plane_waves(
        self,
        wave_vectors: torch.Tensor,
        extent: tuple[float, float],
        gpts: tuple[int, int],
    ) -> torch.Tensor:
        """
        Compute plane waves in real space.

        Parameters
        ----------
        wave_vectors : torch.Tensor
            Wave vectors [num_waves, 2]
        extent : tuple[float, float]
            Real-space extent in Angstroms
        gpts : tuple[int, int]
            Grid points

        Returns
        -------
        torch.Tensor
            Plane waves [num_waves, roi_height, roi_width]
        """
        x = torch.linspace(0, extent[0], gpts[0] + 1, dtype=torch.float32, device=self.device)[:-1]
        y = torch.linspace(0, extent[1], gpts[1] + 1, dtype=torch.float32, device=self.device)[:-1]

        array = torch.exp(
            1.0j * 2 * math.pi * wave_vectors[:, 0, None, None] * x[:, None]
        ) * torch.exp(1.0j * 2 * math.pi * wave_vectors[:, 1, None, None] * y[None, :])

        return array

    def _position_coefficients(
        self, positions: torch.Tensor, wave_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute position-dependent phase shifts for PRISM.

        Parameters
        ----------
        positions : torch.Tensor
            Probe positions [batch_size, 2]
        wave_vectors : torch.Tensor
            Wave vectors [num_waves, 2]

        Returns
        -------
        torch.Tensor
            Position coefficients [batch_size, num_waves]
        """
        coefficients = torch.exp(
            -2.0j * math.pi * positions[..., 0, None] * wave_vectors[:, 0][None]
        ) * torch.exp(-2.0j * math.pi * positions[..., 1, None] * wave_vectors[:, 1][None])

        return coefficients

    def _ctf_coefficients(
        self,
        wave_vectors: torch.Tensor,
        semiangle_cutoff: float,
        angular_sampling: tuple[float, float],
        wavelength: float,
        aberration_coefs: dict,
    ) -> torch.Tensor:
        """
        Compute CTF coefficients for a single probe mode.

        Parameters
        ----------
        wave_vectors : torch.Tensor
            Wave vectors [num_waves, 2]
        semiangle_cutoff : float
            Convergence semi-angle in mrad
        angular_sampling : tuple[float, float]
            Angular sampling in mrad
        wavelength : float
            Electron wavelength in Angstroms
        aberration_coefs : dict
            Aberration coefficients

        Returns
        -------
        torch.Tensor
            CTF coefficients [num_waves]
        """
        alpha = torch.sqrt(wave_vectors[:, 0].square() + wave_vectors[:, 1].square()) * wavelength
        phi = torch.arctan2(wave_vectors[:, 1], wave_vectors[:, 0])

        array = evaluate_probe(
            alpha,
            phi,
            semiangle_cutoff,
            angular_sampling,
            wavelength,
            soft_edges=True,
            vacuum_probe_intensity=None,
            aberration_coefs=aberration_coefs,
        )
        return array / array.abs().square().sum().sqrt()

    def _extract_aberration_coefs(self, probe_params: dict) -> list[dict]:
        """
        Extract and standardize aberration coefficients from probe_params.

        Handles multiple input formats:
        - Direct: {"C10": 100, "C30": 1e7}
        - Nested: {"aberration_coefs": {"C10": 100}}
        - List: {"aberration_coefs": [{"C10": 50}, {"C10": 100}]}
        - Defocus: {"defocus": 100} or {"defocus": [50, 100]}
        - Aliases: {"defocus": 100, "Cs": 1e7}

        Returns
        -------
        list[dict]
            List of aberration coefficient dictionaries, one per probe mode
        """
        # Start with empty list
        coefs_list = []

        # Check if aberration_coefs is explicitly provided
        if "aberration_coefs" in probe_params:
            aberration_coefs = probe_params.pop("aberration_coefs")

            # Handle list vs single dict
            if isinstance(aberration_coefs, list):
                coefs_list = [self._standardize_aberration_dict(c) for c in aberration_coefs]
            else:
                coefs_list = [self._standardize_aberration_dict(aberration_coefs)]

        # Check for defocus shorthand
        elif "defocus" in probe_params:
            defocus = probe_params.pop("defocus")

            # Handle list vs single value
            if isinstance(defocus, list):
                coefs_list = [{"C10": -float(df)} for df in defocus]
            else:
                coefs_list = [{"C10": -float(defocus)}]

        # Otherwise, look for aberration coefficients directly in probe_params
        else:
            # Extract any keys that look like aberrations
            aberration_dict = {}
            keys_to_remove = []

            for key in probe_params.keys():
                # Check if this is an aberration coefficient or alias
                canonical = POLAR_ALIASES.get(key, key)
                if canonical in POLAR_SYMBOLS or key in POLAR_ALIASES:
                    aberration_dict[key] = probe_params[key]
                    keys_to_remove.append(key)

            # Remove aberration keys from probe_params
            for key in keys_to_remove:
                probe_params.pop(key)

            # If we found aberrations, standardize them
            if aberration_dict:
                coefs_list = [self._standardize_aberration_dict(aberration_dict)]
            else:
                # No aberrations specified, use empty dict
                coefs_list = [{}]

        return coefs_list

    def _standardize_aberration_dict(self, aberration_dict: dict) -> dict:
        """
        Standardize a single aberration coefficient dictionary.

        Converts aliases to canonical names and handles special cases.

        Parameters
        ----------
        aberration_dict : dict
            Dictionary with aberration coefficients (may contain aliases)

        Returns
        -------
        dict
            Dictionary with canonical aberration coefficient names
        """
        standardized = {}

        for key, val in aberration_dict.items():
            # Get canonical name
            canonical = POLAR_ALIASES.get(key, key)

            # Special handling for defocus (sign convention)
            if key == "defocus":
                standardized["C10"] = -float(val)
            elif canonical in POLAR_SYMBOLS:
                standardized[canonical] = float(val)
            else:
                raise KeyError(
                    f"Unknown aberration key '{key}'. "
                    f"Expected one of: {', '.join(POLAR_SYMBOLS + tuple(POLAR_ALIASES))}"
                )

        return standardized


ProbeModelType = ProbePixelated | ProbeDIP | ProbeParametric | ProbePRISM
