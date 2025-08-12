from abc import abstractmethod
from copy import deepcopy
from typing import Any, Callable, Self
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import center_of_mass
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.datastructures import Dataset2d, Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.blocks import reset_weights
from quantem.core.ml.loss_functions import get_loss_function
from quantem.core.ml.optimizer_mixin import OptimizerMixin
from quantem.core.utils.utils import RNGMixin, to_numpy
from quantem.core.utils.validators import (
    validate_arr_gt,
    validate_array,
    validate_dict_keys,
    validate_gt,
    validate_np_len,
    validate_tensor,
)
from quantem.core.visualization import show_2d
from quantem.diffractive_imaging.complexprobe import (
    POLAR_ALIASES,
    POLAR_SYMBOLS,
    ComplexProbe,
)
from quantem.diffractive_imaging.constraints import BaseConstraints
from quantem.diffractive_imaging.ptycho_utils import (
    fourier_shift_expand,
    shift_array,
)

# TODO class methods for probe models
# - pixelated should have from_params and from_array
# - DIP should have from_model


class ProbeBase(RNGMixin, OptimizerMixin, AutoSerialize):
    DEFAULT_PROBE_PARAMS = {
        "energy": None,
        "defocus": None,
        "semiangle_cutoff": None,
        "rolloff": 2,
        "polar_parameters": {},
    }
    DEFAULT_LRS = {
        "probe": 1e-3,
    }

    def __init__(
        self,
        num_probes: int = 1,
        probe_params: dict = {},
        vacuum_probe_intensity: Dataset4dstem | np.ndarray | None = None,
        roi_shape: tuple[int, int] | np.ndarray | None = None,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # self._shape = shape
        self.num_probes = num_probes
        self._device = device
        self._probe_params = self.DEFAULT_PROBE_PARAMS
        self.probe_params = probe_params
        self.vacuum_probe_intensity = vacuum_probe_intensity
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
        polar_parameters: dict[str, float] = dict(zip(POLAR_SYMBOLS, [0.0] * len(POLAR_SYMBOLS)))

        def process_polar_params(p: dict):
            bads = []
            for symbol, value in p.items():
                if isinstance(value, dict):
                    process_polar_params(value)  # Recursively process nested dictionaries
                elif value is None:
                    continue
                elif symbol in polar_parameters.keys():
                    polar_parameters[symbol] = float(value)
                    bads.append(symbol)
                elif symbol == "defocus":
                    polar_parameters[POLAR_ALIASES[symbol]] = -1 * float(value)
                elif symbol in POLAR_ALIASES:
                    polar_parameters[POLAR_ALIASES[symbol]] = float(value)
                    bads.append(symbol)
            [p.pop(bad) for bad in bads]
            # Ignore other parameters (energy, semiangle_cutoff, etc.)

        process_polar_params(params)
        params["polar_parameters"] = polar_parameters
        self._probe_params = self.DEFAULT_PROBE_PARAMS | self._probe_params | params

    @property
    def vacuum_probe_intensity(self) -> np.ndarray | None:
        """corner centered vacuum probe"""
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
            raise ValueError(f"Weird number of dimensions for vacuum probe, shape: {vp.shape}")

        # vacuum probe should be corner centered
        corner_vals = vp2[:10, :10].mean()
        if corner_vals < 0.01 * vp2.max():
            warn("Looks like vacuum probe is not corner centered, fft shifting now)")
        else:
            vp2 = np.fft.fftshift(vp2)

        # fix centering
        com: list | tuple = center_of_mass(vp2)
        vp2 = shift_array(
            vp2,
            -com[0],
            -com[1],
            bilinear=True,
        )

        self._vacuum_probe_intensity = vp2

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
        return config.get("dtype_complex")

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device: str | torch.device):
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
    def initial_probe(self) -> torch.Tensor:
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

    @abstractmethod
    def set_initial_probe(self, *args, **kwargs):
        raise NotImplementedError()

    def check_probe_params(self):
        for k in self.DEFAULT_PROBE_PARAMS.keys():
            if self.probe_params[k] is None:
                if k == "defocus":
                    if self.probe_params["polar_parameters"]["C10"] != 0:
                        self.probe_params[k] = -1 * self.probe_params["polar_parameters"]["C10"]
                        continue
                print(f"Missing probe parameter '{k}' in probe_params")
                # raise ValueError(f"Missing probe parameter '{k}' in probe_params")

    @abstractmethod
    def to(self, device: str | torch.device):
        """Move all relevant tensors to a different device."""
        self.device = device
        self._rng_to_device(device)

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
        probe_int = torch.abs(start_probe) ** 2
        probe_int_com = self._to_torch(
            np.array(
                [
                    center_of_mass(torch.fft.fftshift(probe_int[i]).detach().cpu().numpy())
                    for i in range(self.num_probes)
                ]
            )
        )
        probe_int_com = probe_int_com - torch.tensor(
            [s // 2 for s in self.roi_shape], device=self.device
        )
        return fourier_shift_expand(start_probe, -probe_int_com, expand_dim=False)

    def _probe_orthogonalization_constraint(self, start_probe: torch.Tensor) -> torch.Tensor:
        n_probes = start_probe.shape[0]
        orthogonal_probes = []
        original_norms = torch.norm(start_probe.view(n_probes, -1), dim=1, keepdim=True)

        # Apply Gram-Schmidt process
        for i in range(n_probes):
            probe_i = start_probe[i]

            # Subtract projections onto previously computed orthogonal probes
            for j in range(len(orthogonal_probes)):
                projection = (
                    torch.sum(orthogonal_probes[j].conj() * probe_i) * orthogonal_probes[j]
                )
                probe_i = probe_i - projection

            orthogonal_probes.append(probe_i / torch.norm(probe_i))

        orthogonal_probes = torch.stack(orthogonal_probes)
        orthogonal_probes = orthogonal_probes * original_norms.view(-1, 1, 1)

        # Sort probes by real-space intensity
        intensities = torch.sum(torch.abs(orthogonal_probes) ** 2, dim=(-2, -1))
        intensities_order = torch.argsort(intensities, descending=True)

        return orthogonal_probes[intensities_order]


class ProbePixelated(ProbeConstraints):
    def __init__(
        self,
        num_probes: int = 1,
        probe_params: dict = {},
        vacuum_probe_intensity: Dataset4dstem | np.ndarray | None = None,
        initial_probe_array: np.ndarray | None = None,
        roi_shape: tuple[int, int] | np.ndarray | None = None,
        dtype: torch.dtype = torch.complex64,
        device: str = "cpu",
        rng: np.random.Generator = np.random.default_rng(),
        initial_probe_weights: list[float] | np.ndarray | None = None,
        *args,
    ):
        super().__init__(
            num_probes=num_probes,
            probe_params=probe_params,
            vacuum_probe_intensity=vacuum_probe_intensity,
            roi_shape=roi_shape,
            dtype=dtype,
            device=device,
            rng=rng,
        )
        self.initial_probe_array = initial_probe_array
        self.initial_probe_weights = initial_probe_weights
        # Handle roi_shape setting priority: initial_probe_array > roi_shape parameter > unset
        if initial_probe_array is not None:
            probe_shape = np.array(initial_probe_array.shape)
            if roi_shape is not None and not np.array_equal(probe_shape, np.array(roi_shape)):
                warn(
                    f"roi_shape {roi_shape} conflicts with initial_probe_array shape {probe_shape}. Using probe array shape."
                )
            self.roi_shape = probe_shape
        elif roi_shape is not None:
            # roi_shape was already set in parent __init__
            pass
        # If neither is provided, roi_shape will be set later in set_initial_probe

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
        self._probe = self._to_torch(prb)

    @property
    def initial_probe_weights(self) -> np.ndarray:
        return self._initial_probe_weights

    @initial_probe_weights.setter
    def initial_probe_weights(self, weights: list[float] | np.ndarray | None):
        if weights is None:
            self._initial_probe_weights = np.array(
                [1 - 0.02 * (self.num_probes - 1)] + [0.02] * (self.num_probes - 1)
            )
        else:
            if len(weights) != self.num_probes:
                raise ValueError(
                    f"initial_probe_weights must be a list of length {self.num_probes}"
                )
            self._initial_probe_weights = np.array(weights) / np.sum(weights)

    @property
    def params(self):
        """optimization parameters"""
        return self._probe

    @property
    def initial_probe(self) -> torch.Tensor:
        return self._initial_probe

    @property
    def initial_probe_array(self) -> np.ndarray | None:
        return self._initial_probe_array

    @initial_probe_array.setter
    def initial_probe_array(self, initial_probe: np.ndarray | ComplexProbe | None):
        if isinstance(initial_probe, ComplexProbe):
            raise NotImplementedError
        if initial_probe is None:
            self._initial_probe_array = None
        else:
            probe = validate_array(
                initial_probe,
                name="initial_probe",
                dtype=config.get("dtype_complex"),
            )
            self._initial_probe_array = probe

    def forward(self, fract_positions: torch.Tensor) -> torch.Tensor:
        shifted_probes = fourier_shift_expand(self.probe, fract_positions).swapaxes(0, 1)
        return shifted_probes

    def set_initial_probe(
        self,
        roi_shape: np.ndarray | tuple,
        reciprocal_sampling: np.ndarray,
        mean_diffraction_intensity: float,
        device: str | None = None,
        *args,
    ):
        if device is not None:
            self._device = device

        # Only update roi_shape if it wasn't already set during initialization
        if not hasattr(self, "_roi_shape"):
            self.roi_shape = np.array(roi_shape)
        else:
            # Verify that the provided roi_shape matches the initialized one
            if not np.array_equal(self.roi_shape, np.array(roi_shape)):
                warn(
                    f"roi_shape {roi_shape} conflicts with initialized roi_shape {self.roi_shape}. Using initialized value."
                )

        self.reciprocal_sampling = reciprocal_sampling
        self.mean_diffraction_intensity = mean_diffraction_intensity

        if self.initial_probe_array is not None:
            probes = self.initial_probe_array
        elif self.probe_params is not None:
            self.check_probe_params()
            prb = ComplexProbe(
                gpts=tuple(self.roi_shape),
                sampling=tuple(1 / (self.roi_shape * self.reciprocal_sampling)),
                energy=self.probe_params["energy"],
                semiangle_cutoff=self.probe_params["semiangle_cutoff"],
                defocus=self.probe_params["defocus"],
                rolloff=self.probe_params["rolloff"],
                vacuum_probe_intensity=self.vacuum_probe_intensity,
                parameters=self.probe_params["polar_parameters"],
                device="cpu",
            )
            probes: np.ndarray = prb.build()._array
        else:
            raise ValueError(
                "must provide either probe_params or probe in the form of a numpy array or ComplexProbe"
            )

        if probes.ndim != 3:
            probes = probes[None]
        if probes.shape[0] != self.num_probes:
            probes = np.tile(probes, (self.num_probes, 1, 1))

        # apply random phase shifts for initializing mixed state
        for a0 in range(1, self.num_probes):
            shift_y = np.exp(
                -2j * np.pi * (self.rng.random() - 0.5) * np.fft.fftfreq(self.roi_shape[0])
            ).astype(config.get("dtype_complex"))
            shift_x = np.exp(
                -2j * np.pi * (self.rng.random() - 0.5) * np.fft.fftfreq(self.roi_shape[1])
            ).astype(config.get("dtype_complex"))
            probes[a0] = probes[a0] * shift_y[:, None] * shift_x[None]

        # apply weights
        probe_intensity = np.sum(np.abs(np.fft.fft2(probes, norm="ortho")) ** 2)
        intensity_norm = np.sqrt(mean_diffraction_intensity / probe_intensity)
        probes *= intensity_norm

        current_weights = np.sum(np.abs(probes) ** 2, axis=(1, 2))
        current_weights = current_weights / np.sum(current_weights)
        weight_scaling = np.sqrt(self.initial_probe_weights / current_weights)
        probes = probes * weight_scaling[:, None, None]

        self._initial_probe = self._to_torch(probes)
        self._probe = self._initial_probe.clone()
        return

    def reset(self):
        self.probe = self._initial_probe.clone()

    def to(self, device: str | torch.device):
        super().to(device)
        self._probe = self._probe.to(self.device)

    @property
    def name(self) -> str:
        return "ProbePixelized"

    def backward(self, propagated_gradient, obj_patches):
        obj_normalization = torch.sum(torch.abs(obj_patches) ** 2, dim=(-2, -1)).max()
        ortho_norm: float = np.prod(self.roi_shape) ** 0.5  # from ortho fft2 # type:ignore
        probe_grad = torch.sum(propagated_gradient, dim=1) / obj_normalization / ortho_norm
        self._probe.grad = -1 * probe_grad.clone().detach()


class ProbeDIP(ProbeConstraints):
    """
    DIP/model based probe model.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        model_input: torch.Tensor | None = None,
        num_probes: int = 1,
        probe_params: dict = {},
        vacuum_probe_intensity: Dataset4dstem | np.ndarray | None = None,
        roi_shape: tuple[int, int] | np.ndarray | None = None,
        input_noise_std: float = 0.025,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ):
        super().__init__(
            num_probes=num_probes,
            probe_params=probe_params,
            vacuum_probe_intensity=vacuum_probe_intensity,
            roi_shape=roi_shape,
            device=device,
        )
        self.rng = rng
        self.model = model

        if model_input is None:
            # Create default model input - use roi_shape if provided, otherwise placeholder
            if roi_shape is not None:
                input_shape = (1, num_probes, *np.array(roi_shape))
            else:
                input_shape = (1, num_probes, 1, 1)  # will be set properly in set_initial_probe
            self.model_input = torch.randn(
                input_shape, dtype=torch.complex64, device=self.device, generator=self._rng_torch
            )
        else:
            self.model_input = model_input.clone().detach()

        self.pretrain_target = self.model_input.clone().detach()
        self._optimizer = None
        self._scheduler = None
        self._pretrain_losses: list[float] = []
        self._pretrain_lrs: list[float] = []
        self._model_input_noise_std = input_noise_std
        self.training = False

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
        """set the DIP model"""
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
        if device is not None:
            self._device = device

        # Only update roi_shape if it wasn't already set during initialization
        if not hasattr(self, "_roi_shape"):
            self.roi_shape = np.array(roi_shape)
        else:
            # Verify that the provided roi_shape matches the initialized one
            if not np.array_equal(self.roi_shape, np.array(roi_shape)):
                warn(
                    f"roi_shape {roi_shape} conflicts with initialized roi_shape {self.roi_shape}. Using initialized value."
                )

        self.reciprocal_sampling = reciprocal_sampling
        self.mean_diffraction_intensity = mean_diffraction_intensity

        # could check if num_probes corresponds to out_channels of model

        # Only create new model_input if it's still the placeholder (shape [1, num_probes, 1, 1])
        if self.model_input.shape[-2:] == (1, 1):
            self.model_input = torch.randn(
                (1, self.num_probes, *self.roi_shape),
                dtype=self.dtype,
                device=self.device,
                generator=self._rng_torch,
            )

    def to(self, device: str | torch.device):
        """Move all relevant tensors to a different device."""
        super().to(device)
        self._model = self._model.to(self.device)
        self._model_input = self._model_input.to(self.device)
        if hasattr(self, "_initial_probe"):
            self._initial_probe = self._initial_probe.to(self.device)

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
    ):
        if optimizer_params is not None:
            self.set_optimizer(optimizer_params)

        if scheduler_params is not None:
            self.set_scheduler(scheduler_params, num_epochs)

        if reset:
            self._model.apply(reset_weights)
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

        self._model.train()
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


ProbeModelType = ProbePixelated | ProbeDIP
