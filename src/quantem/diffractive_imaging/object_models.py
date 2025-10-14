from abc import abstractmethod
from copy import deepcopy
from typing import Callable, Literal, Self, Sequence, cast
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.blocks import reset_weights
from quantem.core.ml.loss_functions import get_loss_function
from quantem.core.ml.optimizer_mixin import OptimizerMixin
from quantem.core.utils.rng import RNGMixin
from quantem.core.utils.validators import (
    validate_arr_gt,
    validate_gt,
    validate_tensor,
)
from quantem.core.visualization import show_2d
from quantem.diffractive_imaging.constraints import BaseConstraints
from quantem.diffractive_imaging.ptycho_utils import sum_patches

object_type = Literal["potential", "pure_phase", "complex"]

"""
Currently all object models.obj are complex valued for "complex" or "pure_phase" object types,
and real valued for "potential" object types. This could be changed to be always complex valued, 
(after applying constraints) as currently the real-valued potential is made complex in get_obj_patches, 
which will not be used for implicit NNs, which leads to an inconsistency. Leaving for now as I'm not 
sure if this would lead to other issues, so a bit of testing will be needed.
"""


class ObjectBase(nn.Module, RNGMixin, OptimizerMixin, AutoSerialize):
    """
    Base class for all ObjectModels to inherit from.
    """

    DEFAULT_LRS = {
        "object": 5e-3,
        "tv_weight_z": 0,
        "tv_weight_yx": 0,
    }
    _token = object()

    def __init__(
        self,
        device: str = "cpu",
        obj_type: object_type = "complex",
        rng: np.random.Generator | int | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use a factory method to instantiate this class.")

        # Initialize nn.Module first
        nn.Module.__init__(self)
        RNGMixin.__init__(self, rng=rng, device=device)
        OptimizerMixin.__init__(self)

        self.register_buffer("_mask", torch.tensor([]))
        self.device = device
        self._obj_type = obj_type

    # there is some redundancy with shape, shape_2d, and num_slices, but I think it's okay
    # to just allow things to be set in multiple ways as long as they chain to each other
    @property
    def shape(self) -> tuple[int, int, int]:
        return self.obj.shape
        # if None in self._shape:
        #     raise ValueError("Shape has not yet been set")
        # return self._shape  # type: ignore ## idk why this is needed

    # @shape.setter
    # def shape(self, s: tuple | np.ndarray | None) -> None:
    #     print("shape setting directly: ", s)
    #     if s is None:
    #         self._shape = (1, None, None)
    #     else:
    #         s = tuple(s)
    #         if len(s) != 3:
    #             raise ValueError(
    #                 f"Shape must be a tuple of length 3 (depth, row, col), got {len(s)}: {s}"
    #             )
    #         self._shape = (int(s[0]), int(s[1]), int(s[2]))

    @property
    @abstractmethod
    def num_slices(self) -> int:
        # different for pixelated vs DIP so abstract
        raise NotImplementedError()

    # @num_slices.setter
    # def num_slices(self, n: int) -> None:
    #     validate_gt(n, 0, "num_slices")
    #     self._shape = (n, *self._shape[1:])

    @property
    def shape_2d(self) -> tuple[int, int]:
        return self.shape[1:]

    # @shape_2d.setter
    # def shape_2d(self, s: tuple | np.ndarray) -> None:
    #     print("setting shape2d: ", s)
    #     s = tuple(int(x) for x in s)
    #     if len(s) != 2:
    #         raise ValueError(f"shape_2d must be a tuple of length 2 (row, col), got {len(s)}: {s}")
    #     if None in self._shape:
    #         # first time setting shape_2d, set shape to (num_slices, *s)
    #         self.shape = (self.num_slices, *s)
    #         self._initialize_obj()
    #     else:
    #         self.shape = (self.num_slices, *s)
    #         print('none not in shape', self.shape_2d, s, self.shape)
    #         if self.shape_2d != s:
    #             print(f"temp warning -- overrriding shape_2d {self.shape_2d} -> shape_2d")
    #             self._initialize_obj()

    @property
    def dtype(self) -> "torch.dtype":
        if self.obj_type == "potential":
            return getattr(torch, config.get("dtype_real"))
        else:
            return getattr(torch, config.get("dtype_complex"))

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device: str | torch.device):
        dev, _id = config.validate_device(device)
        self._device = dev

    @property
    def obj_type(self) -> object_type:
        return cast(object_type, self._obj_type)

    @obj_type.setter
    def obj_type(self, t: str | None) -> None:
        self._obj_type = self._process_obj_type(t)

    def _process_obj_type(self, obj_type: str | None) -> object_type:
        if obj_type is None:
            return self.obj_type
        t_str = str(obj_type).lower()
        if t_str in ["potential", "potentials"]:
            return "potential"
        elif t_str in ["pure_phase", "purephase", "pure phase"]:
            return "pure_phase"
        elif t_str in ["complex"]:
            return "complex"
        else:
            raise ValueError(
                f"Object type should be 'potential', 'complex', or 'pure_phase', got {obj_type}"
            )

    @property
    def slice_thicknesses(self) -> torch.Tensor | None:
        return self._slice_thicknesses

    @slice_thicknesses.setter
    def slice_thicknesses(self, val: float | Sequence | torch.Tensor | None) -> None:
        if val is None:
            thicknesses = []
        elif isinstance(val, (float, int)):
            thicknesses = [val]
        else:
            thicknesses = val

        if len(thicknesses) == 0:
            if self.num_slices > 1:
                raise ValueError(
                    f"num slices = {self.num_slices}, so slice_thicknesses cannot be None"
                )
            thicknesses = torch.tensor([])
        elif len(thicknesses) == 1:
            thk = validate_gt(float(thicknesses[0]), 0, "slice_thicknesses")
            thicknesses = thk * torch.ones(self.num_slices - 1)
        else:
            if self.num_slices == 1:
                warn("Single slice reconstruction so not setting slice_thicknesses")
            thicknesses = validate_tensor(
                thicknesses,
                name="slice_thicknesses",
                dtype=config.get("dtype_real"),
                ndim=1,
                shape=(self.num_slices - 1,),
            )
            thicknesses = validate_arr_gt(thicknesses, 0, "slice_thicknesses")

        dt = getattr(torch, config.get("dtype_real"))
        self._slice_thicknesses = thicknesses.type(dt).to(self.device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    @mask.setter
    def mask(self, mask: torch.Tensor | np.ndarray):
        mask = validate_tensor(
            mask,
            name="mask",
            dtype=self.dtype,
            ndim=3,
            expand_dims=True,
        )
        self._mask = mask.to(self.device).expand(self.num_slices, -1, -1)

    @property
    def obj(self):
        raise NotImplementedError()

    @property
    def params(self):
        raise NotImplementedError()

    @abstractmethod
    def forward(self, patch_indices: torch.Tensor):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def _initialize_obj(self, shape: tuple[int, int, int] | np.ndarray) -> None:
        raise NotImplementedError()

    def to(self, *args, **kwargs):
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
        raise NotImplementedError()

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

    def _propagate_array(
        self, array: "torch.Tensor", propagator_array: "torch.Tensor"
    ) -> "torch.Tensor":
        propagated = torch.fft.ifft2(torch.fft.fft2(array) * propagator_array)
        return propagated

    def _get_obj_patches(self, obj_array, patch_indices):
        if not obj_array.is_complex():  # potential or pure_phase DIP -> float
            obj_array2 = torch.exp(1.0j * obj_array)
        else:
            obj_array2 = obj_array
        obj_flat = obj_array2.reshape(obj_array.shape[0], -1)
        patches = obj_flat[:, patch_indices]
        return patches

    def backward(self, *args, **kwargs):
        raise NotImplementedError(
            f"Analytical gradients are not implemented for {Self}, use autograd=True"
        )


class ObjectConstraints(BaseConstraints, ObjectBase):
    DEFAULT_CONSTRAINTS = {
        "positivity": True,
        "fix_potential_baseline": False,
        "fix_potential_baseline_factor": 1.0,
        "identical_slices": False,
        "apply_fov_mask": False,
        "tv_weight_z": 0,
        "tv_weight_yx": 0,
        "surface_zero_weight": 0,
    }

    def apply_hard_constraints(
        self, obj: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.obj_type in ["complex", "pure_phase"]:
            if self.obj_type == "complex":
                amp = torch.clamp(torch.abs(obj), 0.0, 1.0)
            else:
                amp = 1.0
            phase = obj.angle() - obj.angle().mean()
            if mask is not None and self.constraints["apply_fov_mask"]:
                obj2 = amp * mask * torch.exp(1.0j * phase * mask)
            else:
                obj2 = amp * torch.exp(1.0j * phase)
        else:  # potential
            if self.constraints["fix_potential_baseline"]:
                if mask is not None:
                    offset = obj[mask < 0.5 * mask.max()].mean()
                else:
                    offset = torch.min(obj)
                offset *= self.constraints["fix_potential_baseline_factor"]
            else:
                offset = 0

            if self.constraints["positivity"]:
                obj2 = torch.clamp(obj - offset, min=0.0)
            else:
                obj2 = obj - offset

        if self.constraints["apply_fov_mask"] and mask is not None:
            obj2 *= mask

        if self.num_slices > 1:
            if self.constraints["identical_slices"]:
                with torch.no_grad():
                    obj2[:] = torch.mean(obj2, dim=0, keepdim=True)

        return obj2

    def apply_soft_constraints(
        self, obj: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # reset recorded losses each call
        self.reset_soft_constraint_losses()

        tv_loss = self.get_tv_loss(
            obj,
            weights=(
                self.constraints["tv_weight_z"],
                self.constraints["tv_weight_yx"],
            ),
        )
        self.add_soft_constraint_loss("tv_loss", tv_loss)

        surface_zero_loss = self.get_surface_zero_loss(
            obj,
            weight=self.constraints["surface_zero_weight"],
        )
        self.add_soft_constraint_loss("surface_zero_loss", surface_zero_loss)

        self.accumulate_constraint_losses()
        return tv_loss + surface_zero_loss

    def get_tv_loss(
        self, array: torch.Tensor, weights: None | tuple[float, float] = None
    ) -> torch.Tensor:
        loss = self._get_zero_loss_tensor()
        if weights is None:
            w = (
                self.constraints["tv_weight_z"],
                self.constraints["tv_weight_yx"],
            )
        elif isinstance(weights, (float, int)):
            if weights == 0:
                return loss
            w = (weights, weights)
        else:
            if not any(weights):
                return loss
            if len(weights) != 2:
                raise ValueError(f"weights must be a tuple of length 2, got {weights}")
            w = weights

        if self.num_slices == 1:
            w = (0, w[1])

        if array.is_complex():
            ph = array.angle()
            loss = loss + self._calc_tv_loss(ph, w)
            amp = array.abs()
            if self.obj_type == "complex":
                loss = loss + self._calc_tv_loss(amp, w)
        else:
            loss = loss + self._calc_tv_loss(array, w)

        return loss

    def _calc_tv_loss(self, array: torch.Tensor, weight: tuple[float, float]) -> torch.Tensor:
        loss = self._get_zero_loss_tensor()
        calc_dim = 0
        for dim in range(array.ndim):
            if dim == 0 and array.ndim == 3:  # could be cleaner...
                w = weight[0]
            else:
                w = weight[1]
            if w > 0:
                calc_dim += 1
                loss = loss + w * torch.mean(torch.abs(array.diff(dim=dim)))
        if calc_dim > 0:
            loss = loss / calc_dim
        return loss

    def get_surface_zero_loss(
        self, array: torch.Tensor, weight: float | int = 0.0
    ) -> torch.Tensor:
        loss = self._get_zero_loss_tensor()
        if weight == 0:
            return loss
        if array.shape[0] < 3:
            return loss
        if array.is_complex():
            ph = array.angle().abs()
            if self.obj_type == "complex":
                amp = array.abs()
                loss = loss + weight * (torch.mean(1.0 - amp[0]) + torch.mean(1.0 - amp[-1]))
            loss = loss + weight * (torch.mean(torch.diff(ph[0])) + torch.mean(torch.diff(ph[-1])))
        else:
            loss = loss + weight * (
                torch.mean(torch.abs(array[0])) + torch.mean(torch.abs(array[-1]))
            )
        return loss


class ObjectPixelated(ObjectConstraints):
    """
    Object model for pixelized objects.
    """

    def __init__(
        self,
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | None = None,
        obj_type: Literal["complex", "pure_phase", "potential"] = "complex",
        initialize_mode: Literal["uniform", "random", "array"] = "uniform",
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
        _token: object | None = None,
    ):
        super().__init__(
            device=device,
            obj_type=obj_type,
            rng=rng,
            _token=_token,
        )
        self._initialize_mode = initialize_mode
        self._obj = nn.Parameter(torch.ones(num_slices, 1, 1), requires_grad=True)
        self.slice_thicknesses = slice_thicknesses

    @classmethod
    def from_uniform(
        cls,
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | None = None,
        device: str = "cpu",
        obj_type: Literal["complex", "pure_phase", "potential"] = "complex",
        rng: np.random.Generator | int | None = None,
    ):
        """
        Create ObjectPixelated from a uniform initialization.
        """
        obj_model = cls(
            num_slices=num_slices,
            slice_thicknesses=slice_thicknesses,
            device=device,
            obj_type=obj_type,
            initialize_mode="uniform",
            rng=rng,
            _token=cls._token,
        )

        return obj_model

    @classmethod
    def from_random(
        cls,
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | None = None,
        device: str = "cpu",
        obj_type: Literal["complex", "pure_phase", "potential"] = "complex",
        rng: np.random.Generator | int | None = None,
    ):
        """
        Create ObjectPixelated from a random initialization.
        """
        obj_model = cls(
            num_slices=num_slices,
            slice_thicknesses=slice_thicknesses,
            device=device,
            obj_type=obj_type,
            initialize_mode="random",
            rng=rng,
            _token=cls._token,
        )

        return obj_model

    @classmethod
    def from_array(
        cls,
        initial_obj: torch.Tensor | np.ndarray,
        slice_thicknesses: float | Sequence | None = None,
        device: str = "cpu",
        obj_type: Literal["complex", "pure_phase", "potential"] = "complex",
        rng: np.random.Generator | int | None = None,
    ):
        """
        Create ObjectPixelated from an array. Shape must match the correct recon shape,
        and so for a demo of this use the pdset.obj_shape_full + padding to confirm is correct.
        """
        num_slices = initial_obj.shape[0]

        obj_model = cls(
            num_slices=num_slices,
            slice_thicknesses=slice_thicknesses,
            device=device,
            obj_type=obj_type,
            initialize_mode="array",
            rng=rng,
            _token=cls._token,
        )
        obj_model._initial_obj = torch.tensor(
            initial_obj, dtype=obj_model.dtype, device=obj_model.device
        )

        return obj_model

    @property
    def obj(self):
        return self.apply_hard_constraints(self._obj, mask=self.mask)

    @property
    def num_slices(self) -> int:
        return self._obj.shape[0]

    @property
    def params(self):
        """optimization parameters"""
        return self._obj

    @property
    def initial_obj(self):
        return self._initial_obj

    def _initialize_obj(self, shape: tuple[int, int, int] | np.ndarray) -> None:
        init_shape = tuple(int(x) for x in shape)
        if self._initialize_mode == "uniform":
            if self.obj_type in ["complex", "pure_phase"]:
                arr = torch.ones(init_shape) * torch.exp(1.0j * torch.zeros(init_shape))
            else:
                arr = torch.zeros(init_shape)
        elif self._initialize_mode == "random":
            ph = (
                torch.randn(init_shape, dtype=torch.float32, generator=self._rng_torch) - 0.5
            ) * 1e-6
            if self.obj_type == "potential":
                arr = ph
            else:
                arr = torch.exp(1.0j * ph)
        elif self._initialize_mode == "array":
            arr = self._initial_obj
        else:
            raise ValueError(f"Invalid initialize mode: {self._initialize_mode}")

        self._initial_obj = arr.type(self.dtype)
        self.reset()

    def reset(self):
        """Reset the object model to its initial or pre-trained state"""
        self._obj = nn.Parameter(self.initial_obj.clone().to(self.device), requires_grad=True)

    def forward(self, patch_indices: torch.Tensor):
        """Get patch indices of the object"""
        return self._get_obj_patches(self.obj, patch_indices)

    @property
    def name(self) -> str:
        return "ObjPixelized"

    def backward(
        self,
        gradient: torch.Tensor,
        obj_patches: torch.Tensor,
        shifted_probes: torch.Tensor,
        propagators: torch.Tensor,
        patch_indices: torch.Tensor,
    ):
        obj_shape = self._obj.shape[-2:]
        obj_gradient = torch.zeros_like(self._obj)
        for s in reversed(range(self.num_slices)):
            probe_slice = shifted_probes[s]
            obj_slice = obj_patches[s]
            probe_normalization = torch.zeros_like(self._obj[s])
            obj_update = torch.zeros_like(self._obj[s])
            for a0 in range(shifted_probes.shape[1]):
                probe = probe_slice[a0]
                grad = gradient[a0]
                probe_normalization += sum_patches(
                    torch.abs(probe) ** 2, patch_indices, obj_shape
                ).max()

                if self.obj_type == "potential":
                    obj_update += sum_patches(
                        torch.real(-1j * torch.conj(obj_slice) * torch.conj(probe) * grad),
                        patch_indices,
                        obj_shape,
                    )
                else:
                    obj_update += sum_patches(torch.conj(probe) * grad, patch_indices, obj_shape)

            obj_gradient[s] = obj_update / probe_normalization

            # back-transmit and back-propagate
            gradient *= torch.conj(obj_slice)
            if s > 0:
                gradient = self._propagate_array(gradient, torch.conj(propagators[s - 1]))

        self._obj.grad = -1 * obj_gradient.clone().detach()
        return gradient


class ObjectDIP(ObjectConstraints):
    """
    DIP/model based object model.
    TODO -- handle 2/3D models more gracefully
        - start with just 2D CNN, allow for single channel output if identical_slices = True
        ( or multi-channel output also, if wanting to then relax the identical_slices constraint)
        - then allow for 3D models, single channel output
    """

    def __init__(
        self,
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | torch.Tensor | None = None,
        input_noise_std: float = 0.025,
        device: str = "cpu",
        obj_type: object_type = "complex",
        rng: np.random.Generator | int | None = None,
        _token: object | None = None,
    ):
        super().__init__(
            device=device,
            obj_type=obj_type,
            rng=rng,
            _token=_token,
        )
        self.register_buffer("_model_input", torch.tensor([]))
        self.register_buffer("_pretrain_target", torch.tensor([]))

        if num_slices < 1:  # no setter cuz shouldn't change after initialization
            raise ValueError(f"num_slices must be greater than 0, got {num_slices}")
        self._num_slices = int(num_slices)
        self.slice_thicknesses = slice_thicknesses

        self._pretrain_losses = []
        self._pretrain_lrs = []
        self._model_input_noise_std = input_noise_std
        self._model_input = torch.tensor([])
        self._pretrain_target = torch.tensor([])

    @classmethod
    def from_model(
        cls,
        model: "torch.nn.Module",
        model_input: torch.Tensor,
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | torch.Tensor | None = None,
        input_noise_std: float = 0.025,
        device: str = "cpu",
        obj_type: object_type = "complex",
        rng: np.random.Generator | int | None = None,
    ):
        """Create ObjectDIP from a CNN and model input."""
        obj_model = cls(
            num_slices=num_slices,
            slice_thicknesses=slice_thicknesses,
            input_noise_std=input_noise_std,
            device=device,
            obj_type=obj_type,
            rng=rng,
            _token=cls._token,
        )
        obj_model.model = model.to(device)
        obj_model.model_input = model_input
        obj_model._set_pretrained_weights(model)

        return obj_model

    @classmethod
    def from_pixelated(
        cls,
        model: "torch.nn.Module",
        pixelated: "ObjectModelType",  # ObjectPixelated upsets linter when ptycho.obj_model is used
        input_noise_std: float = 0.025,
        device: str = "cpu",
    ) -> "ObjectDIP":
        """
        Create ObjectDIP from a pixelated object model.
        """
        if not (
            isinstance(pixelated, ObjectPixelated) or "ObjectPixelated" in str(type(pixelated))
        ):
            raise ValueError(f"Pixelated must be an ObjectPixelated, got {type(pixelated)}")

        model_dtype = "complex" if pixelated.obj_type == "complex" else "real"
        if hasattr(model, "dtype"):  # allow overwriting of dtype based on model
            if "complex" in str(model.dtype):
                model_dtype = "complex"
            else:
                model_dtype = "real"

        if pixelated.obj_type == "pure_phase" and model_dtype == "real":
            obj = pixelated.obj.angle().clone().detach()
        else:
            obj = pixelated.obj.clone().detach()

        obj_model = cls.from_model(
            model=model,
            model_input=obj,
            num_slices=pixelated.num_slices,
            slice_thicknesses=pixelated.slice_thicknesses,
            input_noise_std=input_noise_std,
            device=pixelated.device,
            obj_type=pixelated.obj_type,
            rng=pixelated._rng_seed,
        )
        obj_model.pretrain_target = obj

        return obj_model

    @property
    def num_slices(self) -> int:
        return self._num_slices

    @property
    def name(self) -> str:
        return "ObjectDIP"

    @property
    def dtype(self) -> "torch.dtype":
        if hasattr(self.model, "dtype"):
            return getattr(self.model, "dtype")
        else:
            if self.obj_type in ["complex"]:
                return config.get("dtype_complex")
            else:
                return config.get("dtype_real")

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
        print("\n\n\nsetting model, this is not reachable???\n\n\n")
        # if not isinstance(dip, torch.nn.Module):
        #     raise TypeError(f"DIP must be a torch.nn.Module, got {type(dip)}")
        # if hasattr(dip, "dtype"):
        #     dt = getattr(dip, "dtype")
        #     if self.obj_type in ["complex"] and not dt.is_complex:
        #         raise ValueError("DIP model must be a complex-valued model for complex objects")
        # self._model = dip.to(self.device)
        # self._set_pretrained_weights(self._model)

    @property
    def pretrained_weights(self) -> dict[str, torch.Tensor]:
        """get the pretrained weights of the DIP model"""
        return self._pretrained_weights

    def _set_pretrained_weights(self, model: torch.nn.Module):
        """set the pretrained weights of the DIP model"""
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Pretrained model must be a torch.nn.Module, got {type(model)}")
        self._pretrained_weights = deepcopy(model.state_dict())

    @property
    def model_input(self) -> torch.Tensor:
        """get the model input"""
        return cast(torch.Tensor, self._model_input)

    @model_input.setter
    def model_input(self, input_tensor: torch.Tensor | np.ndarray):
        """set the model input, for a CNN2D should be (1, num_slices, h, w)"""
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.tensor(input_tensor)
        else:
            input_tensor = input_tensor.clone().detach()
        if input_tensor.shape[-3] != self.num_slices:
            raise ValueError(
                f"model_input.shape[-3] {input_tensor.shape[-3]} does not match num_slices {self.num_slices}"
            )
        if input_tensor.ndim == 3:
            input_tensor = input_tensor[None]
        elif input_tensor.ndim != 4:
            raise ValueError(
                f"model_input must be a 3D tensor of shape (num_slices, h, w), got {input_tensor.ndim}D of shape {input_tensor.shape}"
            )

        self._model_input = input_tensor.type(self.dtype).to(self.device)

    # def _generate_model_input(self, mode: Literal["random", "zeros", "ones"]) -> None:
    #     input_shape = (1, *self.shape)
    #     # TODO -- support for 3D CNN models, single channel 2D with identical slices
    #     if mode == "random":
    #         inp = torch.randn(
    #             input_shape, device=self.device, dtype=self.dtype, generator=self._rng_torch
    #         )
    #     elif mode == "zeros":
    #         inp = torch.zeros(input_shape, device=self.device, dtype=self.dtype)
    #     elif mode == "ones":
    #         inp = torch.ones(input_shape, device=self.device, dtype=self.dtype)
    #     else:
    #         raise ValueError(f"Invalid mode: {mode} | must be one of: 'random', 'zeros', 'ones'")
    #     self._model_input = inp

    @property
    def pretrain_target(self) -> torch.Tensor:
        """get the pretrain target"""
        return self._pretrain_target

    @pretrain_target.setter
    def pretrain_target(self, target: torch.Tensor | None):
        """set the pretrain target"""
        if target is None:
            self._pretrain_target = torch.tensor([])
            return

        if target.ndim == 4:
            target = target.squeeze(0)
        target = validate_tensor(
            target,
            name="pretrain_target",
            ndim=3,
            dtype=self.dtype,
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
    def obj(self):
        """get the full object"""
        obj = self.model(self._model_input)[0]
        if self.obj_type == "pure_phase" and "complex" not in str(self.dtype):
            # using a real-valued model for a pure-phase (complex) object
            obj = torch.ones_like(obj) * torch.exp(1j * obj)
        # TODO -- single channel 2D with identical slices, view as 3D num_slices
        return self.apply_hard_constraints(obj, mask=self.mask)

    @property
    def _obj(self):
        # TODO -- single channel 2D with identical slices, view as 3D num_slices??
        return self.model(self._model_input)[0]

    def forward(self, patch_indices: torch.Tensor):
        """Get object patches at given indices"""
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

        obj_array = self.model(model_input)[0]
        if self.mask.numel() > 0:
            obj_array = obj_array * self._mask
        return self._get_obj_patches(obj_array, patch_indices)

    def to(self, *args, **kwargs):
        """Move all relevant tensors to a different device."""
        # Call parent's to() method first to handle PyTorch's internal device management
        # This will automatically move the registered module and buffers
        super().to(*args, **kwargs)
        self._model = self.model.to(*args, **kwargs)

        # Update device property
        device = kwargs.get("device", args[0] if args else None)
        if device is not None:
            self.device = device
            self._rng_to_device(device)
            self.reconnect_optimizer_to_parameters()

        return self

    @property
    def params(self):
        """optimization parameters"""
        return self.model.parameters()

    def get_optimization_parameters(self):
        """Get the parameters that should be optimized for this model."""
        # Since model is registered as a submodule, we can use the parent's parameters() method
        # which will automatically include all submodule parameters
        return list(self.parameters())

    def reset(self):
        """Reset the object model to its initial or pre-trained state"""
        self.model.load_state_dict(self.pretrained_weights.copy())

    def _initialize_obj(self, shape: tuple[int, int, int] | np.ndarray) -> None:
        if not np.array_equal(shape, self.model_input.shape[1:]):
            raise ValueError(
                f"shape {shape} does not match model_input.shape {self.model_input.shape}"
            )

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
        elif self.pretrain_target.numel() == 0:
            # self.pretrain_target = self.model_input.clone().detach().to(self.device)
            raise ValueError(
                "No pretrain target set. Provide pretrain_target or set it beforehand."
            )

        loss_fn = get_loss_function(loss_fn, self.dtype)
        self._pretrain(
            num_epochs=num_epochs,
            loss_fn=loss_fn,
            apply_constraints=apply_constraints,
            show=show,
        )
        self._set_pretrained_weights(self.model)

    def _pretrain(
        self,
        num_epochs: int,
        loss_fn: Callable,
        apply_constraints: bool = False,
        show: bool = False,
    ):
        """Pretrain the DIP model."""
        if self.pretrain_target is None:
            raise ValueError("Pretrain target is not set. Use pretrain_target to set it.")

        self.model.train()
        optimizer = self.optimizer
        if optimizer is None:
            raise ValueError("Optimizer not set. Call set_optimizer() first.")

        scheduler = self.scheduler
        pbar = tqdm(range(num_epochs))
        output = self.obj

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
                if self.obj_type == "pure_phase":
                    output = output.angle()
            else:
                output = self.model(model_input)[0]
            loss: torch.Tensor = loss_fn(output, self.pretrain_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss.item())
                else:
                    scheduler.step()

            self._pretrain_losses.append(loss.item())
            self._pretrain_lrs.append(optimizer.param_groups[0]["lr"])
            pbar.set_description(f"Epoch {a0 + 1}/{num_epochs}, Loss: {loss.item():.3e}, ")

        if show:
            self.visualize_pretrain(output)

    def visualize_pretrain(self, pred_obj: torch.Tensor):
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

        n_bot = 4 if self.obj_type == "complex" else 2
        gs_bot = gridspec.GridSpecFromSubplotSpec(1, n_bot, subplot_spec=gs[1])
        axs_bot = np.array([fig.add_subplot(gs_bot[0, i]) for i in range(n_bot)])
        target = self.pretrain_target
        if target is None:
            raise ValueError("Model has not been pre-trained")
        if n_bot == 4:
            show_2d(
                [
                    pred_obj.mean(0).angle().cpu().detach().numpy(),
                    target.mean(0).angle().cpu().detach().numpy(),
                    pred_obj.mean(0).abs().cpu().detach().numpy(),
                    target.mean(0).abs().cpu().detach().numpy(),
                ],
                figax=(fig, axs_bot),
                title=[
                    "Predicted Phase",
                    "Target Phase",
                    "Predicted Amplitude",
                    "Target Amplitude",
                ],
                cmap="magma",
                cbar=True,
            )
        else:
            show_2d(
                [
                    pred_obj.mean(0).cpu().detach().numpy(),
                    target.mean(0).cpu().detach().numpy(),
                ],
                figax=(fig, axs_bot),
                title=[f"Pred obj ({self.obj_type})", f"Target obj ({self.obj_type})"],
                cmap="magma",
                cbar=True,
            )
        plt.suptitle(
            f"Final loss: {self._pretrain_losses[-1]:.3e} | Epochs: {len(self._pretrain_losses)}",
            fontsize=14,
            y=0.94,
        )
        plt.show()


# class ObjectImplicit(ObjectBase):
#     """
#     Object model for implicit objects. Importantly, the forward call from scan positions
#     for this model will not require subpixel shifting of the object probe, as subpixel shifting
#     will be done in the object model itself, so it is properly aligned around the probe positions
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._obj = None
#         self._obj_shape = None
#         self._num_slices = None

#     def pretrain(self, *args, **kwargs):


#     ### here the forward call will take the batch indices and create the appropriate
#     ### input (which maybe is just the raw patch indices? tbd) for the implicit input
#     ### so it will be parallelized inference across the batches rather than inference once
#     ### and then patching that, like it will be for DIP

ObjectModelType = ObjectPixelated | ObjectDIP  # | ObjectImplicit
