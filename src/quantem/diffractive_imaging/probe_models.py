from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Self
from warnings import warn

import numpy as np
import scipy.ndimage as ndi

from quantem.core import config
from quantem.core.datastructures import Dataset2d, Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.utils import to_numpy
from quantem.core.utils.validators import (
    validate_arr_gt,
    validate_array,
    validate_dict_keys,
    validate_gt,
    validate_tensor,
)
from quantem.diffractive_imaging.complexprobe import (
    POLAR_ALIASES,
    POLAR_SYMBOLS,
    ComplexProbe,
)
from quantem.diffractive_imaging.ptycho_utils import (
    fourier_shift_expand,
    get_com_2d,
    shift_array,
)

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


class ProbeBase(AutoSerialize):
    DEFAULT_PROBE_PARAMS = {
        "energy": None,
        "defocus": None,
        "semiangle_cutoff": None,
        "rolloff": 2,
        "polar_parameters": {},
    }

    def __init__(
        self,
        num_probes: int = 1,
        probe_params: dict = {},
        vacuum_probe_intensity: Dataset4dstem | np.ndarray | None = None,
        device: str = "cpu",
        rng: np.random.Generator = np.random.default_rng(),
        *args,
        **kwargs,
    ):
        # self._shape = shape
        self.num_probes = num_probes
        self._device = device
        self._probe_params = self.DEFAULT_PROBE_PARAMS
        self.probe_params = probe_params
        self.vacuum_probe_intensity = vacuum_probe_intensity
        self._constraints = {}
        self.rng = rng

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
        com: list | tuple = ndi.center_of_mass(vp2)
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
    def reciprocal_sampling(self) -> np.ndarray:
        """reciprocal sampling of the probe"""
        return to_numpy(self._reciprocal_sampling)

    @reciprocal_sampling.setter
    def reciprocal_sampling(self, sampling: np.ndarray | list | tuple):
        sampling = validate_array(
            sampling,
            name="reciprocal_sampling",
            dtype=config.get("dtype_real"),
            shape=(2,),
            expand_dims=True,
        )
        validate_arr_gt(sampling, 0.0, "reciprocal_sampling")
        self._reciprocal_sampling = sampling

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
                raise ValueError(f"Missing probe parameter '{k}' in probe_params")

    @abstractmethod
    def to(self, device: str | torch.device):
        """Move all relevant tensors to a different device."""
        self.device = device
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the object model."""
        raise NotImplementedError()

    def backward(self, *args, **kwargs):
        raise NotImplementedError(
            f"Analytical gradients are not implemented for {Self}, use autograd=True"
        )


class ProbeConstraints(ProbeBase):
    DEFAULT_CONSTRAINTS = {
        "fix_probe": False,
        "fix_probe_com": False,
    }
    # Defaults are kept in PtychographyConstraints for now, not sure where they'll end up
    # eventually, as it contains hard + soft constraints, but if adding new constraints here
    # also need to be added along with default values there

    @property
    def constraints(self) -> dict[str, Any]:
        return self._constraints

    @constraints.setter
    def constraints(self, c: dict[str, Any]):
        gkeys = self.DEFAULT_CONSTRAINTS.keys()
        for key, value in c.items():
            if key not in gkeys:
                raise KeyError(f"Invalid constraint key '{key}', allowed keys are {gkeys}")
            self._constraints[key] = value

    def add_constraint(self, key: str, value: Any):
        """Add a constraint to the object model."""
        gkeys = self.DEFAULT_CONSTRAINTS.keys()
        if key not in gkeys:
            raise KeyError(f"Invalid constraint key '{key}', allowed keys are {gkeys}")
        self._constraints[key] = value

    def apply_constraints(self, probe: torch.Tensor) -> torch.Tensor:
        """
        Apply constraints to the object model.
        """
        if self.num_probes > 1:
            probe = self._probe_orthogonalization_constraint(probe)

        if self.constraints["fix_probe_com"]:
            probe = self._probe_center_of_mass_constraint(probe)

        return probe

    def _probe_center_of_mass_constraint(self, start_probe: torch.Tensor) -> torch.Tensor:
        """
        Ptychographic center of mass constraint.
        Used for centering corner-centered probe intensity.
        """
        probe_intensity = torch.abs(start_probe) ** 2
        com = get_com_2d(probe_intensity, corner_centered=True)
        print("com shape: ", com.shape)
        shifted_probe = fourier_shift_expand(start_probe, -1 * com, expand_dim=False)
        print("shifted probe shape: ", shifted_probe.shape)
        return shifted_probe

    def _probe_orthogonalization_constraint(self, start_probe: torch.Tensor) -> torch.Tensor:
        """
        Ptychographic probe-orthogonalization constraint.
        Used to ensure mixed states are orthogonal to each other.
        Adapted from https://github.com/AdvancedPhotonSource/tike/blob/main/src/tike/ptycho/probe.py#L690
        """

        n_probes = start_probe.shape[0]
        orthogonal_probes = []

        original_norms = torch.norm(
            torch.reshape(start_probe, (n_probes, -1)), dim=1, keepdim=True
        )

        # Gram-Schmidt orthogonalization
        for i in range(n_probes):
            probe_i = start_probe[i]
            for j in range(len(orthogonal_probes)):
                projection = (
                    torch.sum(orthogonal_probes[j].conj() * probe_i) * orthogonal_probes[j]
                )
                probe_i = probe_i - projection
            orthogonal_probes.append(probe_i / torch.norm(probe_i))

        orthogonal_probes = torch.stack(orthogonal_probes)
        orthogonal_probes = orthogonal_probes * torch.reshape(original_norms, (-1, 1, 1))

        # Sort probes by real-space intensity
        intensities = torch.sum(torch.abs(orthogonal_probes) ** 2, dim=(-2, -1))
        intensities_order = torch.flip(torch.argsort(intensities), dims=(0,))

        return orthogonal_probes[intensities_order]


class ProbePixelated(ProbeConstraints, ProbeBase):
    def __init__(
        self,
        num_probes: int = 1,
        probe_params: dict = {},
        vacuum_probe_intensity: Dataset4dstem | np.ndarray | None = None,
        initial_probe_array: np.ndarray | None = None,
        dtype: torch.dtype = torch.complex64,
        device: str = "cpu",
        rng: np.random.Generator = np.random.default_rng(),
        *args,
    ):
        super().__init__(
            num_probes=num_probes,
            probe_params=probe_params,
            vacuum_probe_intensity=vacuum_probe_intensity,
            dtype=dtype,
            device=device,
            rng=rng,
        )
        self.constraints = self.DEFAULT_CONSTRAINTS.copy()
        self.initial_probe_array = initial_probe_array
        if initial_probe_array is not None:
            self.roi_shape = initial_probe_array.shape

    # TODO write classmethods for from_params and from_array
    # from_params should accept vacuum probe as well

    @property
    def probe(self) -> torch.Tensor:
        """get the full probe"""
        return self.apply_constraints(self._probe)
        # return self._probe

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
        self.roi_shape = np.array(roi_shape)
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
            probes = prb.build()._array
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

        probe_intensity = np.sum(np.abs(np.fft.fft2(probes)) ** 2)
        intensity_norm = np.sqrt(mean_diffraction_intensity / probe_intensity)
        probes *= intensity_norm
        self._initial_probe = self._to_torch(probes)
        self._probe = self._initial_probe.clone()
        return

    def reset(self):
        self.probe = self._initial_probe.clone()

    def to(self, device: str | torch.device):
        self.device = device
        self._probe = self._probe.to(self.device)

    @property
    def name(self) -> str:
        return "ProbePixelized"

    def backward(self, propagated_gradient, obj_patches):
        obj_normalization = torch.sum(torch.abs(obj_patches[0]) ** 2, dim=0).max()
        probe_grad = torch.sum(propagated_gradient, dim=1) / obj_normalization
        self._probe.grad = -1 * probe_grad.clone().detach()


ProbeModelType = ProbePixelated  # | ProbeParameterized
