import math
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Literal, Sequence, cast
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.blocks import reset_weights
from quantem.core.ml.constraints import BaseConstraints, Constraints, parse_constraint_dict
from quantem.core.ml.inr import HSiren
from quantem.core.ml.loss_functions import get_loss_module
from quantem.core.ml.models.kplanes import CPTilted, KPlanes, KPlanesTILTED, KPlanesType
from quantem.core.ml.optimizer_mixin import (
    OptimizerMixin,
    OptimizerParams,
    OptimizerParamsType,
    SchedulerParamsType,
)
from quantem.core.utils.rng import RNGMixin
from quantem.core.utils.validators import (
    validate_arr_gt,
    validate_gt,
    validate_tensor,
)
from quantem.core.visualization import show_2d
from quantem.core.visualization.custom_normalizations import CustomNormalization
from quantem.diffractive_imaging.ptycho_utils import add_input_noise, sum_patches

object_type = Literal["potential", "pure_phase", "complex"]


class PtychoObjConstraintParams:
    """
    Namespace class for ptychography object constraint dataclasses.

    Tab-complete on ``PtychoObjConstraintParams`` in a notebook to discover the
    available variants. Tab-complete inside a variant's constructor to see every
    constraint field with its default value.

    Variants
    --------
    Raster
        Constraints for grid-based object representations (``ObjectPixelated`` and
        ``ObjectDIP`` share this set today).
    INR
        Constraints for the implicit-neural-representation objects (``ObjectINR``,
        ``ObjectTensorDecomp``).

    Examples
    --------
    >>> PtychoObjConstraintParams.Raster(tv_weight_z=5.0, identical_slices=True)
    >>> PtychoObjConstraintParams.parse_dict({"name": "raster", "positivity": False})
    """

    @dataclass
    class Raster(Constraints):
        """Constraints for grid-based ptychography object models (``ObjectPixelated``,
        ``ObjectDIP``).

        Fields are applied each iteration in two flavors: **hard** constraints
        project / filter the object after the optimizer step; **soft** constraints
        add a penalty term to the training loss.

        Attributes
        ----------
        positivity : bool, default ``True``
            Clamps the object to be non-negative after each update.
            Only consulted when ``obj_type="potential"``; for ``"complex"`` /
            ``"pure_phase"`` the amplitude is clamped to ``[0, 1]`` (or fixed to 1)
            regardless of this flag.
        positivity_mode: Literal["clamp", "shrink"], default ``"clamp"``
            How to enforce positivity. "clamp" clamps the object to be non-negative after each
            update, does not move the parameter, only how it is shown/used.
            "shrink" subtracts a per-slice background offset from the object so background regions
            sit at zero, is applied to the parameter after the update step.
            If an FOV mask is set the offset is the per-slice mean of the background
            (``mask < 0.5 * mask.max()``); otherwise it's the per-slice 10th percentile. The
            offset is scaled by ``fix_potential_baseline_factor`` even when
            ``fix_potential_baseline`` is False.
        fix_potential_baseline : bool, default ``False``
            ``obj_type="potential"`` only. Subtracts an offset from the object so
            background regions sit at zero. If an FOV mask is set the offset is
            the mean of the background (``mask < 0.5 * mask.max()``); otherwise
            it's ``obj.min()``.
        fix_potential_baseline_factor : float, default ``1.0``
            Scales the baseline offset. Values ``<1`` relax the anchoring
            (subtract less of the background); ``>1`` over-correct.
        identical_slices : bool, default ``False``
            Multislice (``num_slices > 1``) only. Replaces every slice with the
            mean across slices, forcing an effectively 2D object.
        apply_fov_mask : bool, default ``False``
            Multiplies the object by the precomputed FOV mask after each update.
            Useful when the scan does not cover the full padded object area.
        gaussian_sigma : float | None, default ``None``
            Standard deviation (in pixels) of a 2D Gaussian blur applied to each
            slice after each update. Smoothing prior; ``None`` disables.
        butterworth_order : int, default ``4``
            Order of the Butterworth filter used by ``q_lowpass`` / ``q_highpass``.
        q_lowpass : float | None, default ``None``
            Lowpass cutoff in inverse Angstroms. Fourier components above this
            spatial frequency are suppressed via a Butterworth filter.
        q_highpass : float | None, default ``None``
            Highpass cutoff in inverse Angstroms. Components below this frequency
            are suppressed; typically used to remove a slowly varying background.
        tv_weight_z : float, default ``0.0``
            Soft penalty. Weight on the depth-axis total-variation term in the
            loss. Multislice (``num_slices > 1``) only.
        tv_weight_xy : float, default ``0.0``
            Soft penalty. Weight on the in-plane total-variation term;
            encourages piecewise-smooth regions while preserving edges.
        surface_zero_weight : float, default ``0.0``
            Soft penalty pulling the first and last slices toward zero. Useful
            for thick samples embedded in vacuum. Multislice only and requires
            ``num_slices >= 3``.
        """

        # hard constraints
        positivity: bool = True
        positivity_mode: Literal["clamp", "shrink"] = "clamp"
        fix_potential_baseline: bool = False
        fix_potential_baseline_factor: float = 1.0
        identical_slices: bool = False
        apply_fov_mask: bool = False
        # filtering (treated as hard, applied post-update)
        gaussian_sigma: float | None = None  # pixels
        butterworth_order: int = 4
        q_lowpass: float | None = None  # A^-1
        q_highpass: float | None = None  # A^-1
        # soft constraints
        tv_weight_z: float = 0.0
        tv_weight_xy: float = 0.0
        surface_zero_weight: float = 0.0
        _name: str = "raster"

        soft_constraint_keys = ["tv_weight_z", "tv_weight_xy", "surface_zero_weight"]
        hard_constraint_keys = [
            "positivity",
            "positivity_mode",
            "fix_potential_baseline",
            "fix_potential_baseline_factor",
            "identical_slices",
            "apply_fov_mask",
            "gaussian_sigma",
            "butterworth_order",
            "q_lowpass",
            "q_highpass",
        ]

    @dataclass
    class INR(Constraints):
        """Constraints for the implicit (``ObjectINR`` / ``ObjectTensorDecomp``) object.

        An implicit object has no grid to clamp/filter in place, so the grid-based hard
        constraints of ``Raster`` do not apply directly. Instead: positivity is a **soft**
        penalty evaluated at sampled coordinates (keeps the network output linear so a
        zero-background ``potential`` fits without the vanishing/dead gradients of a
        softplus/relu output activation), and the potential baseline is a **display gauge** on
        the materialized object (a constant potential offset is a global phase, i.e.
        diffraction-invariant).

        Attributes
        ----------
        tv_weight_z : float, default ``0.0``
            Soft penalty. Depth-axis (``z``) total variation, finite-differenced at randomly
            sampled coordinates. Multislice (``num_slices > 1``) only.
        tv_weight_xy : float, default ``0.0``
            Soft penalty. In-plane (``y``, ``x``) total variation at sampled coordinates.
        positivity_weight : float, default ``0.0``
            Soft penalty. Weight on ``mean(relu(-value))`` at sampled coordinates -- drives the
            ``potential`` non-negative. ``obj_type="potential"`` only; ignored otherwise. Scale it
            relative to the data-loss magnitude; increase if negativity persists.
        fix_potential_baseline : bool, default ``False``
            ``obj_type="potential"`` only. Subtracts a background offset from the *materialized*
            object so background sits at zero (then clamps >= 0). Display gauge only -- it does not
            perturb the reconstruction (the forward queries the network directly), mirroring the
            ``Raster`` constraint of the same name.
        fix_potential_baseline_factor : float, default ``1.0``
            Scales the subtracted baseline offset (``<1`` relaxes the anchoring).
        """

        # soft constraints (evaluated at sampled coordinates)
        tv_weight_z: float = 0.0
        tv_weight_xy: float = 0.0
        positivity_weight: float = 0.0
        # hard / display constraints (applied to the materialized object only)
        fix_potential_baseline: bool = False
        fix_potential_baseline_factor: float = 1.0
        _name: str = "inr"

        soft_constraint_keys = ["tv_weight_z", "tv_weight_xy", "positivity_weight"]
        hard_constraint_keys = ["fix_potential_baseline", "fix_potential_baseline_factor"]

    @classmethod
    def parse_dict(cls, d: dict) -> "PtychoObjConstraintsType":
        """Instantiate the appropriate variant from a config dict.

        The dict must contain a ``'name'`` or ``'type'`` key (case-insensitive),
        with value ``'raster'`` or ``'inr'``. All other keys are forwarded as
        keyword arguments to the chosen dataclass.
        """
        return cast(PtychoObjConstraintsType, parse_constraint_dict(cls, d, kind="object"))


PtychoObjConstraintsType = PtychoObjConstraintParams.Raster | PtychoObjConstraintParams.INR

"""
Object representation by obj_type:
- "complex"    : _obj is complex (amplitude * exp(1j * phase))
- "pure_phase" : _obj is a real, unwrapped phase array
- "potential"  : _obj is a real potential array

The forward boundary (`_get_obj_patches`) wraps real `_obj` to `exp(1j * _obj)` for
both pure_phase and potential, so the rest of the forward model never has to
branch on obj_type.
"""


class ObjectBase(nn.Module, RNGMixin, OptimizerMixin, AutoSerialize):
    """
    Base class for all ObjectModels to inherit from.
    """

    DEFAULT_LRS = {
        "object": 5e-3,
        "tv_weight_z": 0,
        "tv_weight_xy": 0,
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
        self._sampling = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.obj.shape

    @property
    def is_implicit(self) -> bool:
        """Whether this is an implicit (coordinate-queried) object representation.

        Pixelated/DIP objects are grid-based and consume integer ``patch_indices``;
        an implicit object (``ObjectINR``) instead consumes continuous coordinates,
        which the paired dataset produces when this is True. Overridden to True by
        implicit subclasses.
        """
        return False

    @property
    @abstractmethod
    def num_slices(self) -> int:
        # different for pixelated vs DIP so abstract
        raise NotImplementedError()

    @property
    def shape_2d(self) -> tuple[int, int]:
        return self.shape[1:]

    @property
    def dtype(self) -> "torch.dtype":
        if self.obj_type == "complex":
            return getattr(torch, config.get("dtype_complex"))
        return getattr(torch, config.get("dtype_real"))

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

    @property
    def sampling(self) -> tuple[float, float]:
        """Realspace in-plane sampling in A"""
        if self._sampling is None:
            raise ValueError("ObjectModel sampling not set, call _initialize_obj() first")
        return self._sampling

    @sampling.setter
    def sampling(self, sampling: tuple[float, float] | np.ndarray | torch.Tensor):
        smp = validate_arr_gt(
            validate_tensor(sampling, name="sampling", ndim=1, shape=(2,)), 0, "sampling"
        )
        self._sampling = (smp[0].item(), smp[1].item())

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
    def slice_thicknesses(self, val: float | Sequence | torch.Tensor | np.ndarray | None) -> None:
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
        self._mask = mask.to(self.device).expand(self.num_slices, -1, -1).contiguous()

    @property
    @abstractmethod
    def obj(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def params(self) -> list[nn.Parameter]:
        raise NotImplementedError()

    @abstractmethod
    def forward(self, patch_indices: torch.Tensor, /):
        # positional-only: implicit object models accept coordinates here instead of indices
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    def project_parameters(self) -> None:
        """In-place hard projection of the underlying parameters after an optimizer step.

        No-op by default. ``ObjectPixelated`` overrides this to enforce
        ``positivity_mode="shrink"`` (proximal per-slice background shrinkage on the potential
        grid). Called once per optimizer step from the reconstruction loop.
        """
        return

    @abstractmethod
    def _initialize_obj(
        self,
        shape: tuple[int, int, int] | np.ndarray,
        sampling: np.ndarray | tuple[float, float] | None = None,
    ) -> None:
        if sampling is not None:
            self.sampling = sampling

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

    def get_optimization_parameters(self) -> "dict[str, list[torch.Tensor]]":
        """Get the parameters that should be optimized for this model, keyed by group."""
        params = self.params
        if params is None:
            return {}
        return {self.DEFAULT_OPTIMIZER_KEY: list(params)}

    def _propagate_array(
        self, array: "torch.Tensor", propagator_array: "torch.Tensor"
    ) -> "torch.Tensor":
        propagated = torch.fft.ifft2(torch.fft.fft2(array) * propagator_array)
        return propagated

    def _get_obj_patches(self, obj_array, patch_indices):
        """Forward boundary: wrap real obj to ``exp(1j * obj)`` and gather patches.

        ``obj_array`` may be complex (``obj_type="complex"``) or real (``"pure_phase"``,
        ``"potential"``). Real inputs are wrapped to the complex transmission
        function ``exp(1j * obj_array)`` here, so the rest of the forward model
        never has to branch on ``obj_type``. ``patch_indices`` is a
        ``(num_gpts, Hroi, Wroi)`` int tensor of flattened-index lookups into the
        2D padded object.
        """
        if not obj_array.is_complex():  # potential or pure_phase DIP -> float
            obj_array2 = torch.exp(1.0j * obj_array)
        else:
            obj_array2 = obj_array
        obj_flat = obj_array2.reshape(obj_array.shape[0], -1)

        # patches = obj_flat[:, patch_indices]
        # MPS does not support complex scatter kernel..
        real = obj_flat.real
        imag = obj_flat.imag
        patches = torch.complex(real[:, patch_indices], imag[:, patch_indices])

        return patches

    def backward(self, *args, **kwargs):
        raise NotImplementedError(
            f"Analytical gradients are not implemented for {type(self).__name__}, "
            "use autograd=True"
        )


class ObjectConstraints(BaseConstraints[PtychoObjConstraintParams.Raster], ObjectBase):
    DEFAULT_CONSTRAINTS: PtychoObjConstraintParams.Raster = PtychoObjConstraintParams.Raster()

    def apply_hard_constraints(
        self, raw: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Apply hard constraints: range clamping and filtering. All hard constaints are applied in
        place with torch.no_grad().
        """
        c = self.constraints
        with torch.no_grad():
            if self.obj_type == "complex":
                constrained = self._apply_hard_complex(raw, c)
            elif self.obj_type == "pure_phase":
                constrained = self._apply_hard_pure_phase(raw, c)
            else:  # potential
                constrained = self._apply_hard_potential(raw, c, mask)
            constrained = self._apply_shared_hard(constrained, c, mask)
        return raw + (constrained - raw).detach()

    def _apply_hard_complex(
        self, obj: torch.Tensor, c: PtychoObjConstraintParams.Raster
    ) -> torch.Tensor:
        amp = torch.clamp(torch.abs(obj), 0.0, 1.0)
        phase = obj.angle() - obj.angle().mean()
        return amp * torch.exp(1.0j * phase)

    def _apply_hard_pure_phase(
        self, obj: torch.Tensor, c: PtychoObjConstraintParams.Raster
    ) -> torch.Tensor:
        # phase stored directly as a real tensor; recenter to zero mean
        return obj - obj.mean()

    def _apply_hard_potential(
        self,
        obj: torch.Tensor,
        c: PtychoObjConstraintParams.Raster,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # "shrink" manages the _obj parameter directly in project_parameters() (post-step), so the
        # forward just passes the (already non-negative) parameter through.
        if c.positivity_mode == "shrink":
            return obj
        offset = self._potential_baseline_offset(obj, c, mask)
        obj = obj - offset
        # "clamp" clamps here; the apply_hard_constraints wrapper makes it a straight-through op
        # (the forward/display is non-negative but the _obj parameter is left untouched).
        if c.positivity:
            return torch.clamp(obj, min=0.0)
        return obj

    def _potential_baseline_offset(
        self,
        obj: torch.Tensor,
        c: PtychoObjConstraintParams.Raster,
        mask: torch.Tensor | None,
    ) -> torch.Tensor | float:
        """Background offset subtracted by ``fix_potential_baseline`` (``0`` when disabled).

        Estimated from the FOV-mask background mean (else the global min), detached, and scaled by
        ``fix_potential_baseline_factor``.
        """
        if not c.fix_potential_baseline:
            return 0.0
        # mask is an empty tensor (not None) when no FOV mask is set, so guard on numel().
        if mask is not None and mask.numel() and (mask < 0.5 * mask.max()).any():
            offset = obj[mask < 0.5 * mask.max()].mean()
        else:
            offset = obj.min()
        offset = offset.detach()
        return offset * c.fix_potential_baseline_factor

    def _apply_shared_hard(
        self,
        obj: torch.Tensor,
        c: PtychoObjConstraintParams.Raster,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if c.apply_fov_mask and mask is not None:
            obj = obj * mask

        if c.gaussian_sigma is not None:
            obj = self.gaussian_blur_2d(obj, sigma=c.gaussian_sigma)

        if any([c.q_lowpass, c.q_highpass]):
            obj = self.butterworth_constraint(obj, sampling=self.sampling)

        if self.num_slices > 1 and c.identical_slices:
            # In-place mutation is safe because apply_hard_constraints is
            # always called under outer torch.no_grad (see its docstring).
            obj[:] = torch.mean(obj, dim=0, keepdim=True)
        return obj

    def apply_soft_constraints(
        self, obj: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sum of the per-iteration soft penalties.

        Returns a scalar tensor that is added to the data-fidelity loss before
        ``backward()``. Individual contributions are also recorded via
        ``add_soft_constraint_loss`` for logging.
        """
        # reset recorded losses each call
        self.reset_soft_constraint_losses()

        tv_loss = self.get_tv_loss(obj)
        self.add_soft_constraint_loss("tv_loss", tv_loss)

        surface_zero_loss = self.get_surface_zero_loss(
            obj,
            weight=self.constraints.surface_zero_weight,
        )
        self.add_soft_constraint_loss("surface_zero_loss", surface_zero_loss)
        self.accumulate_constraint_losses()
        return tv_loss + surface_zero_loss

    def get_tv_loss(
        self, array: torch.Tensor, weights: None | tuple[float, float] = None
    ) -> torch.Tensor:
        """Total-variation soft penalty on the object.

        ``weights`` is a ``(z_weight, xy_weight)`` tuple. When ``None``, defaults
        to ``(self.constraints.tv_weight_z, self.constraints.tv_weight_xy)``. A single
        scalar is broadcast to both axes. The z weight is zeroed for
        ``num_slices == 1``.
        """
        loss = self._get_zero_loss_tensor()
        w = self._resolve_tv_weights(weights)
        if not any(w):
            return loss

        if self.obj_type == "complex":
            return self._tv_complex(array, w)
        # pure_phase and potential are both real tensors; phase wrapping is gone.
        return self._calc_tv_loss(array, w)

    def _resolve_tv_weights(
        self, weights: None | tuple[float, float] | float | int
    ) -> tuple[float, float]:
        if weights is None:
            w: tuple[float, float] = (
                self.constraints.tv_weight_z,
                self.constraints.tv_weight_xy,
            )
        elif isinstance(weights, (float, int)):
            w = (float(weights), float(weights))
        else:
            if len(weights) != 2:
                raise ValueError(f"weights must be a tuple of length 2, got {weights}")
            w = (float(weights[0]), float(weights[1]))
        if self.num_slices == 1:
            w = (0.0, w[1])
        return w

    def _tv_complex(self, array: torch.Tensor, w: tuple[float, float]) -> torch.Tensor:
        # complex objects carry information in both amplitude and phase. We
        # still extract phase via angle() here, so the wrap warning stays —
        # but only for obj_type == "complex".
        loss = self._get_zero_loss_tensor()
        ph = array.angle()
        warn(
            "calculating TV loss for phase of complex object, "
            "phase wrapping may distort the gradient. Consider obj_type='pure_phase'."
        )
        # TODO: amp and phase share `w` here. Consider splitting `tv_weight_xy`
        # into separate amp/phase weights on PtychoObjConstraintParams.Raster
        # so users can tune them independently for obj_type="complex".
        loss = loss + self._calc_tv_loss(ph, w)
        amp = array.abs()
        loss = loss + self._calc_tv_loss(amp, w)
        return loss

    def _calc_tv_loss(self, array: torch.Tensor, weight: tuple[float, float]) -> torch.Tensor:
        """Mean-|diff| TV on a real array. ``weight = (w_z, w_xy)``.

        For a 3D ``(slices, H, W)`` array, dim 0 uses ``w_z`` and dims 1+2 use
        ``w_xy``. The result is averaged over the number of axes that actually
        contributed (i.e. had a non-zero weight).
        """
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
        """Penalize the first and last slices to be near vacuum.

        Real ``pure_phase`` / ``potential`` arrays: penalizes ``|array[0]|`` and
        ``|array[-1]|`` directly. ``complex`` arrays pull amplitude toward 1
        and phase toward its mean (see ``_surface_zero_complex``). A no-op for
        single- or double-slice objects (``array.shape[0] < 3``).
        """
        loss = self._get_zero_loss_tensor()
        if weight == 0 or array.shape[0] < 3:
            return loss
        if self.obj_type == "complex":
            return self._surface_zero_complex(array, weight)
        # pure_phase and potential: real array, penalize first/last slice magnitude
        return loss + weight * (torch.mean(torch.abs(array[0])) + torch.mean(torch.abs(array[-1])))

    def _surface_zero_complex(self, array: torch.Tensor, weight: float | int) -> torch.Tensor:
        # complex: pull amp toward 1 (vacuum) at the surfaces, and phase toward its mean
        loss = self._get_zero_loss_tensor()
        amp = array.abs()
        loss = loss + weight * (torch.mean(1.0 - amp[0]) + torch.mean(1.0 - amp[-1]))
        ph = array.angle().abs()
        warn(
            "calculating surface zero loss for phase of complex object, "
            "phase wrapping may distort the gradient. Consider obj_type='pure_phase'."
        )
        loss = loss + weight * (
            torch.mean(torch.abs(ph[0] - ph[0].mean()))
            + torch.mean(torch.abs(ph[-1] - ph[-1].mean()))
        )
        return loss

    def gaussian_blur_2d(self, tensor, sigma=1.0):
        """Separable 2D Gaussian blur over the last two dimensions.

        Parameters
        ----------
        tensor : torch.Tensor
            Real or complex, shape ``(slices, H, W)``. Complex inputs are
            filtered as independent real/imag channels.
        sigma : float
            Standard deviation of the Gaussian kernel, in pixels. The
            kernel size is ``2 * ceil(3 * sigma) + 1``.
        """
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0, device=tensor.device)
        gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss = gauss / gauss.sum()

        kernel_h = gauss.view(1, 1, -1, 1)
        kernel_v = gauss.view(1, 1, 1, -1)

        if tensor.is_complex():
            real = tensor.real.unsqueeze(1)
            imag = tensor.imag.unsqueeze(1)

            real_h = nn.functional.conv2d(real, kernel_h, padding=(kernel_size // 2, 0))
            real_blurred = nn.functional.conv2d(
                real_h, kernel_v, padding=(0, kernel_size // 2)
            ).squeeze(1)

            imag_h = nn.functional.conv2d(imag, kernel_h, padding=(kernel_size // 2, 0))
            imag_blurred = nn.functional.conv2d(
                imag_h, kernel_v, padding=(0, kernel_size // 2)
            ).squeeze(1)

            return torch.complex(real_blurred, imag_blurred)
        else:
            x = tensor.unsqueeze(1)

            x_h = nn.functional.conv2d(x, kernel_h, padding=(kernel_size // 2, 0))
            x_blurred = nn.functional.conv2d(x_h, kernel_v, padding=(0, kernel_size // 2)).squeeze(
                1
            )

            return x_blurred

    def butterworth_constraint(
        self,
        tensor: torch.Tensor,
        sampling: tuple[float, float],
    ) -> torch.Tensor:
        """Apply a Fourier-domain Butterworth low/high-pass to each 2D slice.

        Reads ``q_lowpass``, ``q_highpass``, and ``butterworth_order`` off
        ``self.constraints``. The DC component is subtracted before filtering
        and added back so the mean is preserved.

        Parameters
        ----------
        tensor : torch.Tensor
            Shape ``(slices, H, W)``. May be real or complex. Real inputs are
            re-cast to real after the FFT round-trip when ``obj_type != "complex"``.
        sampling : tuple[float, float]
            ``(dy, dx)`` real-space sampling in Ångström per pixel. Sets the
            inverse-Å scale of the Butterworth response; ``q_lowpass`` and
            ``q_highpass`` are in *inverse Ångström (cycles / Å).
        """

        q_lowpass = self.constraints.q_lowpass
        q_highpass = self.constraints.q_highpass
        butterworth_order = self.constraints.butterworth_order

        qx = torch.fft.fftfreq(tensor.shape[-2], sampling[0], device=tensor.device)
        qy = torch.fft.fftfreq(tensor.shape[-1], sampling[1], device=tensor.device)

        qya, qxa = torch.meshgrid(qy, qx, indexing="xy")
        qra = torch.sqrt(qxa**2 + qya**2)

        env = torch.ones_like(qra)

        if q_highpass:
            env *= 1 - 1 / (1 + (qra / q_highpass) ** (2 * butterworth_order))

        if q_lowpass:
            env *= 1 / (1 + (qra / q_lowpass) ** (2 * butterworth_order))

        tensor_mean = tensor.mean(dim=(-2, -1), keepdim=True)
        tensor = tensor - tensor_mean

        # Apply filter in Fourier space
        tensor = torch.fft.ifft2(torch.fft.fft2(tensor) * env)

        tensor = tensor + tensor_mean

        # FFT-based filter returns complex even for real inputs; cast back to real
        # for any non-complex object type (pure_phase, potential).
        if self.obj_type != "complex":
            tensor = tensor.real

        return tensor


class ObjectPixelated(ObjectConstraints):
    """
    Object model for pixelized objects.
    """

    def __init__(
        self,
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | None | np.ndarray = None,
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
        slice_thicknesses: float | Sequence | None | np.ndarray = None,
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
        slice_thicknesses: float | Sequence | None | np.ndarray = None,
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
        initial = torch.as_tensor(initial_obj)
        if initial.is_complex() and obj_type != "complex":
            if obj_type == "pure_phase":
                # Convert legacy complex initial_obj (amp*exp(1j*phase)) to bare phase
                initial = initial.angle()
            else:
                raise ValueError(f"Complex initial_obj is not valid for obj_type '{obj_type}'")
        obj_model._initial_obj = initial.detach().to(
            dtype=obj_model.dtype, device=obj_model.device, copy=True
        )

        return obj_model

    @property
    def obj(self):
        return self.apply_hard_constraints(self._obj, mask=self.mask)

    @property
    def num_slices(self) -> int:
        return self._obj.shape[0]

    @property
    def params(self) -> list[nn.Parameter]:
        """optimization parameters"""
        return [self._obj]

    def project_parameters(self) -> None:
        """Post-step proximal shrinkage of the potential grid parameter, for
        ``positivity_mode="shrink"`` (``obj_type="potential"`` with ``positivity=True``); a no-op
        otherwise.

        Subtracts a per-slice background offset (``fix_potential_baseline_factor * per-slice
        background``) from ``_obj`` then clamps ``>= 0``. Unlike the straight-through ``clamp`` mode
        this moves ``_obj`` itself, so the reconstruction (not just the display) is shrunk toward a
        zero background. Per-slice keeps it on the diffraction-invariant gauge (loss-neutral for
        multislice), and the offset shrinks with the background so it self-limits at zero.
        """
        c = self.constraints
        if not (self.obj_type == "potential" and c.positivity and c.positivity_mode == "shrink"):
            return
        with torch.no_grad():
            bg = self._per_slice_background(self._obj, self.mask)
            offset = (c.fix_potential_baseline_factor * bg.clamp_min(0.0)).view(-1, 1, 1)
            self._obj.data = (self._obj.data - offset).clamp_min(0.0)

    def _per_slice_background(self, obj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-slice background level, shape ``[num_slices]``: the FOV-mask background mean when a
        mask is set, else a robust per-slice 10th percentile."""
        s = obj.shape[0]
        flat = obj.reshape(s, -1)
        if mask is not None and mask.numel() and (mask < 0.5 * mask.max()).any():
            bg = (mask < 0.5 * mask.max()).reshape(s, -1).to(obj.dtype)
            return (flat * bg).sum(1) / bg.sum(1).clamp_min(1.0)
        return torch.quantile(flat, 0.1, dim=1)

    @property
    def initial_obj(self):
        return self._initial_obj

    def _initialize_obj(
        self,
        shape: tuple[int, int, int] | np.ndarray,
        sampling: tuple[float, float] | np.ndarray | None = None,
    ) -> None:
        super()._initialize_obj(shape, sampling)
        if self.obj.numel() > self.num_slices and np.array_equal(self.shape, shape):
            return
        init_shape = tuple(int(x) for x in shape)
        if self._initialize_mode == "uniform":
            if self.obj_type == "complex":
                # amp=1, phase=0 -> complex ones
                arr = torch.ones(init_shape) * torch.exp(1.0j * torch.zeros(init_shape))
            else:
                # pure_phase (phase=0) and potential start as real zeros
                arr = torch.zeros(init_shape)
        elif self._initialize_mode == "random":
            ph = (
                torch.randn(init_shape, dtype=torch.float32, generator=self._rng_torch) - 0.5
            ) * 1e-6
            if self.obj_type == "complex":
                arr = torch.exp(1.0j * ph)
            else:
                # pure_phase stores phase directly; potential stores real values
                arr = ph
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
        obj_model.model = model.to(obj_model.device)
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

        if pixelated.obj_type == "complex" and model_dtype == "real":
            obj = pixelated.obj.angle().clone().detach()
        else:
            obj = pixelated.obj.clone().detach()

        obj_model = cls.from_model(
            model=model,
            model_input=obj,
            num_slices=pixelated.num_slices,
            slice_thicknesses=pixelated.slice_thicknesses,
            input_noise_std=input_noise_std,
            device=device,
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
        return super().dtype

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
        raise RuntimeError("\n\n\nsetting model, this shouldn't be reachable???\n\n\n")
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
        raw = self.model(self._model_input)[0]
        # TODO -- single channel 2D with identical slices, view as 3D num_slices
        return self.apply_hard_constraints(raw, mask=self.mask)

    @property
    def _obj(self):
        return self.model(self._model_input)[0]

    def forward(self, patch_indices: torch.Tensor):
        """Get object patches at given indices"""
        model_input = add_input_noise(
            self.model_input, self._input_noise_std, self.dtype, self.device, self._rng_torch
        )
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
    def params(self) -> list[nn.Parameter]:
        """optimization parameters"""
        return list(self.model.parameters())

    def reset(self):
        """Reset the object model to its initial or pre-trained state"""
        self.model.load_state_dict(self.pretrained_weights.copy())

    def _initialize_obj(
        self,
        shape: tuple[int, int, int] | np.ndarray,
        sampling: tuple[float, float] | np.ndarray | None = None,
    ) -> None:
        super()._initialize_obj(shape, sampling)
        if not np.array_equal(shape, self.model_input.shape[1:]):
            raise ValueError(
                f"shape {shape} does not match model_input.shape {self.model_input.shape}"
            )

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
        normalize_object_plotting: bool = True,
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
        elif self.pretrain_target.numel() == 0:
            # self.pretrain_target = self.model_input.clone().detach().to(self.device)
            raise ValueError(
                "No pretrain target set. Provide pretrain_target or set it beforehand."
            )

        loss_fn = get_loss_module(loss_fn, self.dtype)
        self._pretrain(
            num_iters=num_iters,
            loss_fn=loss_fn,
            apply_constraints=apply_constraints,
            show=show,
            normalize_object_plotting=normalize_object_plotting,
        )
        self._set_pretrained_weights(self.model)

    def _pretrain(
        self,
        num_iters: int,
        loss_fn: Callable,
        apply_constraints: bool = False,
        show: bool = False,
        normalize_object_plotting: bool = True,
    ):
        """Pretrain the DIP model."""
        if self.pretrain_target is None:
            raise ValueError("Pretrain target is not set. Use pretrain_target to set it.")

        self.model.train()
        optimizer = self.optimizer
        if optimizer is None:
            raise ValueError("Optimizer not set. Call set_optimizer() first.")

        scheduler = self.scheduler
        pbar = tqdm(range(num_iters))
        output = self.obj

        for a0 in pbar:
            model_input = add_input_noise(
                self.model_input, self._input_noise_std, self.dtype, self.device, self._rng_torch
            )
            if apply_constraints:
                output = self.apply_hard_constraints(self.model(model_input)[0])
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
            pbar.set_description(f"Iter {a0 + 1}/{num_iters}, Loss: {loss.item():.3e}, ")

        if show:
            self.visualize_pretrain(
                output,
                normalize_object_plotting=normalize_object_plotting,
            )

    def visualize_pretrain(
        self,
        pred_obj: torch.Tensor,
        normalize_object_plotting: bool = True,
    ):
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

        n_bot = 4 if self.obj_type == "complex" else 2
        gs_bot = gridspec.GridSpecFromSubplotSpec(1, n_bot, subplot_spec=gs[1])
        axs_bot = np.array([fig.add_subplot(gs_bot[0, i]) for i in range(n_bot)])
        target = self.pretrain_target
        if target is None:
            raise ValueError("Model has not been pre-trained")
        if n_bot == 4:
            norm_angle = None
            norm_abs = None
            if normalize_object_plotting:
                target_mean_angle = target.mean(0).angle().cpu().detach().numpy()
                target_mean_abs = target.mean(0).abs().cpu().detach().numpy()

                target_norm_angle = CustomNormalization(
                    interval_type="quantile",
                    data=target_mean_angle,
                )
                norm_angle = {
                    "interval_type": "manual",
                    "vmin": target_norm_angle.vmin,
                    "vmax": target_norm_angle.vmax,
                }

                target_norm_abs = CustomNormalization(
                    interval_type="quantile",
                    data=target_mean_abs,
                )
                norm_abs = {
                    "interval_type": "manual",
                    "vmin": target_norm_abs.vmin,
                    "vmax": target_norm_abs.vmax,
                }
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
                norm=[norm_angle, norm_angle, norm_abs, norm_abs],  # type:ignore
            )
        else:
            norm = None
            if normalize_object_plotting:
                target_mean = target.mean(0).cpu().detach().numpy()
                target_norm = CustomNormalization(
                    interval_type="quantile",
                    data=target_mean,
                )
                norm = {
                    "interval_type": "manual",
                    "vmin": target_norm.vmin,
                    "vmax": target_norm.vmax,
                }

            show_2d(
                [
                    pred_obj.mean(0).cpu().detach().numpy(),
                    target.mean(0).cpu().detach().numpy(),
                ],
                figax=(fig, axs_bot),
                title=[f"Pred obj ({self.obj_type})", f"Target obj ({self.obj_type})"],
                cmap="magma",
                cbar=True,
                norm=norm,
            )
        plt.suptitle(
            f"Final loss: {self._pretrain_losses[-1]:.3e} | Iters: {len(self._pretrain_losses)}",
            fontsize=14,
            y=0.94,
        )
        plt.show()


class ObjectINR(BaseConstraints[PtychoObjConstraintParams.INR], ObjectBase):
    """Implicit (coordinate-queried) object model.

    Wraps an implicit neural representation (INR; an ``HSiren`` by default) that maps
    normalized 3D coordinates ``(z, y, x)`` in ``[-1, 1]`` to a real-valued object — the phase
    for ``obj_type="pure_phase"`` or the potential for ``obj_type="potential"``, both wrapped to
    the complex transmission ``exp(1j * value)``. Rather than gathering grid-aligned patches at integer
    scan positions like ``ObjectPixelated``, the paired dataset produces continuous
    per-patch ``(y, x)`` coordinates at the *true* (fractional) scan positions; this
    model augments them with each slice's ``z`` coordinate, queries the INR, and returns
    the complex transmission patches ``exp(1j * phase)``. Because the object is sampled
    directly at the true position, the probe no longer needs subpixel shifting.

    Coordinate convention
    ----------------------
    Coordinates are normalized to ``[-1, 1]`` over the padded object extent (matching
    ``torch.linspace(-1, 1, N)``). The ``z`` axis spans the slices: a single slice sits
    at ``z = 0``; multislice z-positions come from the cumulative ``slice_thicknesses``,
    mapped so the first slice is at ``-1`` and the last at ``+1``. Samples that fall
    outside the object (``|y| > 1`` or ``|x| > 1``) are treated as vacuum (phase 0,
    transmission 1) rather than wrapped toroidally as the pixelated path does.

    Notes
    -----
    - Autograd-only: analytical gradients are not implemented (see ``backward``).
    - Hard constraints have no grid to project; only the soft, coordinate-sampled TV
      penalties of ``PtychoObjConstraintParams.INR`` apply.
    """

    DEFAULT_LRS = {
        "object": 1e-3,
        "tv_weight_z": 0,
        "tv_weight_xy": 0,
    }
    DEFAULT_CONSTRAINTS: PtychoObjConstraintParams.INR = PtychoObjConstraintParams.INR()

    def __init__(
        self,
        model: "torch.nn.Module",
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | torch.Tensor | None = None,
        obj_type: object_type = "pure_phase",
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
        if self.obj_type == "complex":
            raise NotImplementedError(
                "ObjectINR does not support obj_type='complex' yet (planned); use 'pure_phase' "
                "or 'potential' (both real-valued, wrapped to exp(1j * value))."
            )
        if num_slices < 1:
            raise ValueError(f"num_slices must be greater than 0, got {num_slices}")
        self._num_slices = int(num_slices)
        self._model = model.to(self._device)
        self.slice_thicknesses = slice_thicknesses
        self._set_pretrained_weights(self._model)

        # Padded object extent [num_slices, H, W]; set in _initialize_obj. Defines the
        # [-1, 1] coordinate domain and the grid on which .obj is materialized.
        self._obj_shape: tuple[int, int, int] | None = None
        # Lazily materialized full-grid object (detached); invalidated each forward().
        self._obj_cache: torch.Tensor | None = None
        # Pretraining state (used by pretrain() / from_pixelated()).
        self.register_buffer("_pretrain_target", torch.tensor([]))
        self._pretrain_losses: list[float] = []
        self._pretrain_lrs: list[float] = []

    @classmethod
    def from_inr(
        cls,
        model: "torch.nn.Module",
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | torch.Tensor | None = None,
        obj_type: object_type = "pure_phase",
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ) -> "ObjectINR":
        """Create an ObjectINR from a user-supplied INR ``nn.Module``.

        The model must map coordinates of shape ``(N, 3)`` (``z, y, x``) to a single
        real output ``(N, 1)``.
        """
        return cls(
            model=model,
            num_slices=num_slices,
            slice_thicknesses=slice_thicknesses,
            obj_type=obj_type,
            device=device,
            rng=rng,
            _token=cls._token,
        )

    @classmethod
    def from_uniform(
        cls,
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | torch.Tensor | None = None,
        hidden_features: int = 128,
        hidden_layers: int = 3,
        first_omega_0: float = 10.0,
        hidden_omega_0: float = 10.0,
        obj_type: object_type = "pure_phase",
        final_activation: str | Callable | None = None,
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ) -> "ObjectINR":
        """Create an ObjectINR backed by a default ``HSiren``, initialized to vacuum.

        The HSiren's final layer is zero-initialized so the object starts uniform (a
        diffraction-equivalent vacuum), matching ``ObjectPixelated.from_uniform``.

        ``final_activation`` sets the output nonlinearity. The default (``None``) is ``"identity"``
        for both ``pure_phase`` and ``potential``: enforcing positivity at the output (softplus /
        relu) makes a zero-background potential hard to fit (vanishing / dead gradients), so for
        ``potential`` positivity is instead the soft ``positivity_weight`` constraint. With the
        zeroed final layer the object starts at 0 (vacuum). Pass ``final_activation="softplus"`` to
        opt back into output-activation positivity.

        Note
        ----
        ``first_omega_0`` / ``hidden_omega_0`` set the SIREN's frequency content and are the
        most important knobs to tune (INRs are sensitive to this, more so than DIPs). The
        default of ``10`` suits smooth phase objects; the SIREN image-fitting default of ``30``
        is typically too high here (optimization stalls near vacuum), while objects with fine
        features may want a larger value. Pair omega_0 with the object learning rate.
        """
        if final_activation is None:
            final_activation = "identity"
        model = HSiren(
            in_features=3,
            out_features=1,
            hidden_layers=hidden_layers,
            hidden_features=hidden_features,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            final_activation=final_activation,
            dtype=getattr(torch, config.get("dtype_real")),
        )
        # Zero the final linear layer so the INR output is uniform at init (vacuum / global phase).
        with torch.no_grad():
            final_linear = cast(nn.Linear, model.net[-2])
            final_linear.weight.zero_()
            if final_linear.bias is not None:
                final_linear.bias.zero_()
        return cls.from_inr(
            model=model,
            num_slices=num_slices,
            slice_thicknesses=slice_thicknesses,
            obj_type=obj_type,
            device=device,
            rng=rng,
        )

    @classmethod
    def from_pixelated(
        cls,
        pixelated: "ObjectModelType",
        model: "torch.nn.Module | None" = None,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        first_omega_0: float = 10.0,
        hidden_omega_0: float = 10.0,
        device: str | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> "ObjectINR":
        """Create an ObjectINR matching a pixelated object, with it as the pretrain target.

        The INR is built to the pixelated object's geometry (``num_slices``,
        ``slice_thicknesses``, ``obj_type``, padded shape) and the current pixelated object is
        stored as the pretrain target, so ``pretrain()`` warm-starts the INR to reproduce the
        pixelated reconstruction -- mirroring ``ObjectDIP.from_pixelated`` + ``pretrain``.

        Pass ``model`` to wrap a custom INR ``nn.Module`` directly (mapping ``(N, 3)`` coords to
        ``(N, 1)``), as with ``ObjectDIP.from_pixelated`` -- handy for testing architectures. When
        ``model`` is ``None`` a default zero-initialized ``HSiren`` is built from the
        ``hidden_features`` / ``hidden_layers`` / ``omega_0`` args (with identity output
        activations); when a ``model`` is given those args and the activation are its own.
        """
        if not (
            isinstance(pixelated, ObjectPixelated) or "ObjectPixelated" in str(type(pixelated))
        ):
            raise ValueError(f"pixelated must be an ObjectPixelated, got {type(pixelated)}")
        dev = pixelated.device if device is None else device
        seed = pixelated._rng_seed if rng is None else rng
        if model is not None:
            inr = cls.from_inr(
                model=model,
                num_slices=pixelated.num_slices,
                slice_thicknesses=pixelated.slice_thicknesses,
                obj_type=pixelated.obj_type,
                device=dev,
                rng=seed,
            )
        else:
            inr = cls.from_uniform(
                num_slices=pixelated.num_slices,
                slice_thicknesses=pixelated.slice_thicknesses,
                obj_type=pixelated.obj_type,
                hidden_features=hidden_features,
                hidden_layers=hidden_layers,
                first_omega_0=first_omega_0,
                hidden_omega_0=hidden_omega_0,
                device=dev,
                rng=seed,
            )
        target = pixelated.obj.detach().to(dev)  # (num_slices, H, W) real phase / potential
        inr._obj_shape = tuple(int(x) for x in target.shape)  # type: ignore[assignment]
        if pixelated._sampling is not None:
            inr.sampling = pixelated.sampling
        inr.pretrain_target = target
        return inr

    # region --- properties ---
    @property
    def is_implicit(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "ObjectINR"

    @property
    def num_slices(self) -> int:
        return self._num_slices

    @property
    def model(self) -> "torch.nn.Module":
        return self._model

    @property
    def params(self) -> list[nn.Parameter]:
        """optimization parameters"""
        return list(self._model.parameters())

    @property
    def pretrained_weights(self) -> dict[str, torch.Tensor]:
        return self._pretrained_weights

    def _set_pretrained_weights(self, model: "torch.nn.Module") -> None:
        self._pretrained_weights = deepcopy(model.state_dict())

    @property
    def pretrain_target(self) -> torch.Tensor:
        """Target object (real phase, ``(num_slices, H, W)``) fitted by ``pretrain()``."""
        return self._pretrain_target

    @pretrain_target.setter
    def pretrain_target(self, target: torch.Tensor | np.ndarray | None) -> None:
        if target is None:
            self._pretrain_target = torch.tensor([], device=self.device)
            return
        t = validate_tensor(
            target,
            name="pretrain_target",
            ndim=3,
            dtype=config.get("dtype_real"),
            expand_dims=True,
        )
        self._pretrain_target = t.to(self.device)

    @property
    def pretrain_losses(self) -> np.ndarray:
        return np.array(self._pretrain_losses)

    @property
    def pretrain_lrs(self) -> np.ndarray:
        return np.array(self._pretrain_lrs)

    @property
    def _z_coords(self) -> torch.Tensor:
        """Normalized z-coordinate of each slice in [-1, 1], shape (num_slices,)."""
        real_dtype = getattr(torch, config.get("dtype_real"))
        s = self.num_slices
        if s == 1:
            return torch.zeros(1, device=self.device, dtype=real_dtype)
        thick = self._slice_thicknesses.to(self.device)  # (S-1,)
        zeros = torch.zeros(1, device=self.device, dtype=real_dtype)
        z_pos = torch.cat([zeros, torch.cumsum(thick, dim=0)])  # (S,)
        total = z_pos[-1]
        if total <= 0:
            return torch.linspace(-1.0, 1.0, s, device=self.device, dtype=real_dtype)
        return (z_pos / total) * 2.0 - 1.0

    @property
    def obj(self):
        """Materialized full object on the padded grid (real phase for pure_phase).

        Cold-path only (display / logging / serialization); the training loop queries
        the INR directly via ``forward`` and does not touch this. Cached and invalidated
        on each ``forward`` call.
        """
        if self._obj_cache is None:
            raw = self._materialize_obj()
            self._obj_cache = self.apply_hard_constraints(raw, mask=self.mask)
        return self._obj_cache

    # endregion --- properties ---

    def _invalidate_obj_cache(self) -> None:
        self._obj_cache = None

    def _query_phase(self, coords_xy: torch.Tensor) -> torch.Tensor:
        """Query the INR at normalized (y, x) coords for every slice.

        ``coords_xy`` has shape ``(..., 2)`` (normalized row=y, col=x in [-1, 1]).
        Returns phase of shape ``(num_slices, ...)``.
        """
        s = self.num_slices
        lead = coords_xy.shape[:-1]
        xy = coords_xy.reshape(-1, 2)  # (M, 2)
        m = xy.shape[0]
        z = self._z_coords  # (S,)
        z_full = z.view(s, 1, 1).expand(s, m, 1)
        xy_full = xy.view(1, m, 2).expand(s, m, 2)
        coords3d = torch.cat([z_full, xy_full], dim=-1).reshape(-1, 3)  # (S*M, 3)
        phase = self._model(coords3d).squeeze(-1).reshape(s, *lead)
        return phase

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Query the object at continuous per-patch coordinates.

        ``coords`` has shape ``(batch, Hroi, Wroi, 2)``: normalized ``(y, x)`` positions
        in ``[-1, 1]`` produced by the dataset at the true (fractional) scan positions.
        Returns complex transmission patches of shape ``(num_slices, batch, Hroi, Wroi)``.
        """
        self._invalidate_obj_cache()
        inside = (coords[..., 0].abs() <= 1.0) & (coords[..., 1].abs() <= 1.0)  # (batch, H, W)
        phase = self._query_phase(coords)  # (S, batch, H, W)
        phase = phase * inside.unsqueeze(0)  # off-object -> phase 0 (vacuum)
        return torch.exp(1.0j * phase)

    def _grid_coords_xy(self) -> torch.Tensor:
        """Normalized ``(row, col)`` grid over the padded object, shape ``(H, W, 2)``."""
        if self._obj_shape is None:
            raise ValueError("ObjectINR shape not set, call _initialize_obj() first")
        real_dtype = getattr(torch, config.get("dtype_real"))
        _, h, w = (int(x) for x in self._obj_shape)
        ys = torch.linspace(-1.0, 1.0, h, device=self.device, dtype=real_dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=self.device, dtype=real_dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gy, gx], dim=-1)  # (H, W, 2)

    def _materialize_obj(self) -> torch.Tensor:
        with torch.no_grad():
            phase = self._query_phase(self._grid_coords_xy())  # (S, H, W)
        return phase

    def pretrain(
        self,
        pretrain_target: torch.Tensor | np.ndarray | None = None,
        num_iters: int = 200,
        optimizer_params: "dict | OptimizerParamsType | None" = None,
        scheduler_params: "dict | SchedulerParamsType | None" = None,
        loss_fn: Callable | str = "l2",
        device: str | int | None = None,
        show: bool = True,
        normalize_object_plotting: bool = True,
    ) -> None:
        """Warm-start the INR by fitting it to a target object (e.g. a pixelated recon).

        Queries the INR on the full normalized grid and regresses it onto ``pretrain_target``
        (a real ``(num_slices, H, W)`` phase array, typically the pixelated reconstruction set by
        ``from_pixelated``). The fitted weights become the reset state, so a subsequent
        ``reconstruct(reset=True)`` resumes from this warm start instead of vacuum. A learning-rate
        scheduler is recommended (pass ``scheduler_params``) as INRs converge better with one.
        """
        if device is not None:
            dev, _ = config.validate_device(device)
            self.to(dev)
        if pretrain_target is not None:
            self.pretrain_target = pretrain_target
        if self._pretrain_target is None or self._pretrain_target.numel() == 0:
            raise ValueError(
                "No pretrain target set; pass pretrain_target or use from_pixelated()."
            )
        if self._obj_shape is None:
            self._obj_shape = tuple(int(x) for x in self._pretrain_target.shape)  # type: ignore[assignment]
        if optimizer_params is not None:
            self.set_optimizer(optimizer_params)
        if scheduler_params is not None:
            self.set_scheduler(scheduler_params, num_iters)
        loss_module = get_loss_module(loss_fn, getattr(torch, config.get("dtype_real")))
        self._pretrain(
            num_iters, loss_module, show=show, normalize_object_plotting=normalize_object_plotting
        )
        self._set_pretrained_weights(self._model)
        self._invalidate_obj_cache()

    def _pretrain(
        self,
        num_iters: int,
        loss_fn: Callable,
        show: bool = False,
        normalize_object_plotting: bool = True,
    ) -> None:
        optimizer = self.optimizer
        if optimizer is None:
            raise ValueError("Optimizer not set. Pass optimizer_params to pretrain().")
        scheduler = self.scheduler
        coords_xy = self._grid_coords_xy()
        target = self._pretrain_target.to(self.device)
        self._model.train()
        pbar = tqdm(range(num_iters))
        output = self._query_phase(coords_xy)
        for _ in pbar:
            optimizer.zero_grad()
            output = self._query_phase(coords_xy)  # (S, H, W), differentiable
            loss: torch.Tensor = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss.item())
                else:
                    scheduler.step()
            self._pretrain_losses.append(loss.item())
            self._pretrain_lrs.append(optimizer.param_groups[0]["lr"])
            pbar.set_description(
                f"Iter {len(self._pretrain_losses)}/{num_iters}, Loss: {loss.item():.3e}"
            )
        if show:
            self.visualize_pretrain(output.detach(), normalize_object_plotting)

    def visualize_pretrain(
        self, pred_obj: torch.Tensor, normalize_object_plotting: bool = True
    ) -> None:
        """Plot the pretraining loss / learning-rate curves and the pred vs target object."""
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

        gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1])
        axs_bot = np.array([fig.add_subplot(gs_bot[0, i]) for i in range(2)])
        target = self._pretrain_target
        norm = None
        if normalize_object_plotting:
            target_mean = target.mean(0).cpu().detach().numpy()
            target_norm = CustomNormalization(interval_type="quantile", data=target_mean)
            norm = {
                "interval_type": "manual",
                "vmin": target_norm.vmin,
                "vmax": target_norm.vmax,
            }
        show_2d(
            [
                pred_obj.mean(0).cpu().detach().numpy(),
                target.mean(0).cpu().detach().numpy(),
            ],
            figax=(fig, axs_bot),
            title=[f"Pred obj ({self.obj_type})", f"Target obj ({self.obj_type})"],
            cmap="magma",
            cbar=True,
            norm=norm,
        )
        plt.suptitle(
            f"Final loss: {self._pretrain_losses[-1]:.3e} | Iters: {len(self._pretrain_losses)}",
            fontsize=14,
            y=0.94,
        )
        plt.show()

    def _initialize_obj(
        self,
        shape: tuple[int, int, int] | np.ndarray,
        sampling: tuple[float, float] | np.ndarray | None = None,
    ) -> None:
        super()._initialize_obj(shape, sampling)
        shape_t = tuple(int(x) for x in shape)
        if shape_t[0] != self.num_slices:
            raise ValueError(
                f"shape[0] ({shape_t[0]}) does not match num_slices ({self.num_slices})"
            )
        self._obj_shape = shape_t  # type: ignore[assignment]
        self._invalidate_obj_cache()

    def reset(self) -> None:
        """Reset the INR weights to their initial (pretrained) state."""
        self._model.load_state_dict(deepcopy(self._pretrained_weights))
        self._invalidate_obj_cache()

    def to(self, *args, **kwargs):
        """Move all relevant tensors to a different device."""
        super().to(*args, **kwargs)
        self._model = self._model.to(*args, **kwargs)
        device = kwargs.get("device", args[0] if args else None)
        if device is not None:
            self.device = device
            self._rng_to_device(device)
            self.reconnect_optimizer_to_parameters()
        self._invalidate_obj_cache()
        return self

    # region --- constraints ---
    def apply_hard_constraints(
        self, raw: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Project the materialized object (display only).

        Unlike the grid-based ``Raster`` constraints, an INR has nothing to clamp or filter in
        place. For ``pure_phase`` we recenter the phase to zero mean (a global-phase gauge) so the
        displayed object matches the pixelated convention. For ``potential``, if
        ``fix_potential_baseline`` is set, subtract a background offset (mask background mean, else
        the global min) scaled by ``fix_potential_baseline_factor`` and clamp ``>= 0`` -- a display
        gauge (a constant potential offset is a global phase, hence diffraction-invariant), so it
        does not affect the reconstruction. Positivity *during* the reconstruction is the soft
        ``positivity_weight`` penalty, not a projection here.
        """
        with torch.no_grad():
            if self.obj_type == "pure_phase":
                return raw - raw.mean()
            if self.constraints.fix_potential_baseline:
                if mask is not None and mask.numel() and (mask < 0.5 * mask.max()).any():
                    offset = raw[mask < 0.5 * mask.max()].mean()
                else:
                    offset = raw.amin()
                offset = offset * self.constraints.fix_potential_baseline_factor
                return torch.clamp(raw - offset, min=0.0)
            return raw

    def apply_soft_constraints(
        self, obj: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Coordinate-sampled total-variation penalty.

        The ``obj`` argument is accepted for interface parity with the grid-based object
        models but ignored: TV is evaluated at randomly sampled coordinates so the
        penalty is differentiable w.r.t. the INR weights without materializing the full
        grid.
        """
        self.reset_soft_constraint_losses()
        loss = self._get_zero_loss_tensor()
        w_z = self.constraints.tv_weight_z if self.num_slices > 1 else 0.0
        w_xy = self.constraints.tv_weight_xy
        if w_z > 0 or w_xy > 0:
            tv_loss = self._sampled_tv_loss(w_z, w_xy)
            loss = loss + tv_loss
            self.add_soft_constraint_loss("tv_loss", tv_loss)
        w_pos = self.constraints.positivity_weight
        if w_pos > 0 and self.obj_type == "potential":
            pos_loss = self._sampled_positivity_loss(w_pos)
            loss = loss + pos_loss
            self.add_soft_constraint_loss("positivity_loss", pos_loss)
        self.accumulate_constraint_losses()
        return loss

    def _sampled_positivity_loss(self, weight: float, num_samples: int = 4096) -> torch.Tensor:
        """Differentiable positivity penalty for ``potential``: ``weight * mean(relu(-value))`` at
        random coordinates. Keeps the network output linear (identity activation), so a
        zero-background potential fits without the vanishing (softplus) / dead (relu) gradients of
        an output activation; negative regions get a constant linear restoring force toward 0.
        """
        real_dtype = getattr(torch, config.get("dtype_real"))
        coords_xy = (
            torch.rand(
                num_samples, 2, device=self.device, dtype=real_dtype, generator=self._rng_torch
            )
            * 2.0
            - 1.0
        )
        value = self._query_phase(coords_xy)  # (S, num_samples) -- the queried potential
        return weight * torch.relu(-value).mean()

    def _sampled_tv_loss(self, w_z: float, w_xy: float, num_samples: int = 4096) -> torch.Tensor:
        """Finite-difference TV over (z, y, x) at randomly sampled coordinates."""
        real_dtype = getattr(torch, config.get("dtype_real"))
        coords_xy = (
            torch.rand(
                num_samples, 2, device=self.device, dtype=real_dtype, generator=self._rng_torch
            )
            * 2.0
            - 1.0
        )
        phase = self._query_phase(coords_xy)  # (S, num_samples)
        loss = self._get_zero_loss_tensor()
        # finite-difference step ~ one pixel in normalized coords
        if self._obj_shape is not None:
            h = 2.0 / max(int(self._obj_shape[-1]), int(self._obj_shape[-2]))
        else:
            h = 1e-2
        if w_xy > 0:
            for axis in range(2):
                offset = torch.zeros(2, device=self.device, dtype=real_dtype)
                offset[axis] = h
                shifted = self._query_phase(coords_xy + offset)
                loss = loss + w_xy * torch.mean(torch.abs(shifted - phase))
            loss = loss / 2
        if w_z > 0 and self.num_slices > 1:
            loss = loss + w_z * torch.mean(torch.abs(phase[1:] - phase[:-1]))
        return loss

    # endregion --- constraints ---

    def backward(self, *args, **kwargs):
        raise NotImplementedError(
            f"Analytical gradients are not implemented for {self.name}, use autograd=True"
        )


class ObjectTensorDecomp(ObjectINR):
    """Implicit object model backed by a tensor-decomposition network (K-Planes family).

    A thin subclass of :class:`ObjectINR` that swaps the SIREN for a tensor-decomposition model
    from :mod:`quantem.core.ml.models.kplanes` — plain :class:`KPlanes`, the tilted
    :class:`KPlanesTILTED` (T learned SO(3) rotations), or the :class:`CPTilted` bottleneck.
    Because these consume the same ``(N, 3)`` ``(z, y, x)`` coordinates and return ``(N, 1)``,
    every coordinate-query path of ``ObjectINR`` (``forward``, ``_query_phase``, materialization,
    sampled-TV soft constraints, ``pretrain``) is reused unchanged; only the optimizer wiring
    differs.

    These models expose multiple parameter groups (``grids``/``sigma_net``, plus ``so3`` for the
    tilted variants), so the model uses per-parameter-group learning rates (PPLR):
    ``optimizer_params`` must be a dict keyed by ``model.param_keys`` (see
    :meth:`get_optimization_parameters` / :meth:`_normalize_optimizer_params`), e.g.
    ``{"grids": OptimizerParams.Adam(lr=1e-2), "sigma_net": OptimizerParams.Adam(lr=1e-3)}``.

    The object grid is treated as a 3D ``(z, y, x)`` volume with ``z`` the slice axis: a single
    slice is queried at ``z = 0`` (the in-plane K-plane already provides the 2D feature grid),
    multislice spans ``z`` via the slice thicknesses. A model's ``resolution`` is the
    *feature-plane* resolution, set by the user and decoupled from the padded object grid (which
    comes from ``_initialize_obj``).

    Notes
    -----
    - The ``density_activation`` must be a picklable ``nn.Module`` (``nn.Identity`` for
      ``pure_phase``, ``nn.Softplus`` for ``potential``); a bare lambda will break ``save()``
      because AutoSerialize pickles the whole module. ``from_uniform`` sets this automatically;
      ``from_model`` warns otherwise.
    - K-Planes converge from scratch, but ``from_pixelated`` + ``pretrain`` can still warm-start
      the grid to a pixelated reconstruction (cheap, useful for finding hyperparameters quickly).
    """

    DEFAULT_CONSTRAINTS: PtychoObjConstraintParams.INR = PtychoObjConstraintParams.INR()

    @classmethod
    def from_model(
        cls,
        model: KPlanesType,
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | torch.Tensor | None = None,
        obj_type: object_type = "pure_phase",
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ) -> "ObjectTensorDecomp":
        """Wrap a user-built tensor-decomposition model as a ptychography object model.

        ``model`` is a :class:`KPlanes`, :class:`KPlanesTILTED`, or :class:`CPTilted` mapping
        ``(N, 3)`` ``(z, y, x)`` coordinates to ``(N, 1)`` and exposing the PPLR interface
        (``param_keys`` / ``get_params``).
        """
        if not isinstance(model, (KPlanes, CPTilted)):  # KPlanesTILTED is a KPlanes subclass
            raise TypeError(
                f"model must be a KPlanes/KPlanesTILTED/CPTilted instance, got {type(model)}"
            )
        activation = getattr(model, "density_activation", None)
        if activation is not None and not isinstance(activation, nn.Module):
            warn(
                "KPlanes.density_activation is a plain callable (e.g. a lambda); saving this "
                "object will fail because AutoSerialize pickles the whole module. Use an "
                "nn.Module activation (nn.Identity for pure_phase, nn.Softplus for potential), "
                "e.g. via ObjectTensorDecomp.from_uniform.",
                stacklevel=2,
            )
        obj = cls(
            model=model,
            num_slices=num_slices,
            slice_thicknesses=slice_thicknesses,
            obj_type=obj_type,
            device=device,
            rng=rng,
            _token=cls._token,
        )
        obj.to(device)
        return obj

    @classmethod
    def from_uniform(  # pyright: ignore[reportIncompatibleMethodOverride]  # KPlanes factory, intentionally diverges from ObjectINR.from_uniform
        cls,
        num_slices: int = 1,
        slice_thicknesses: float | Sequence | torch.Tensor | None = None,
        M_features: int = 16,
        resolution: Sequence[int] = (64, 64, 64),
        multiscale_res_multipliers: Sequence[float] | None = (0.25, 0.5, 1.0),
        use_hybrid_mlp: bool = False,
        hybrid_hidden_dim: int = 64,
        hybrid_num_layers: int = 2,
        tilted: bool = False,
        T: int = 4,
        obj_type: object_type = "pure_phase",
        device: str = "cpu",
        rng: np.random.Generator | int | None = None,
    ) -> "ObjectTensorDecomp":
        """Build a default K-Planes-backed object, initialized to vacuum.

        ``tilted=False`` builds a plain :class:`KPlanes`; ``tilted=True`` builds a
        :class:`KPlanesTILTED` with ``T`` learned SO(3) rotations. The decoder's final layer is
        zeroed so the object starts at 0 (vacuum), matching ``ObjectINR.from_uniform``. The decoder
        is **identity**-activated for both ``pure_phase`` and ``potential``: a softplus/relu output
        activation makes a zero-background potential hard to fit (vanishing/dead gradients), so for
        ``potential`` enforce positivity with the soft ``positivity_weight`` constraint instead (use
        ``from_model`` with an ``nn.Softplus`` activation to opt back in). ``resolution`` is the
        feature-plane resolution ``(z, y, x)`` and is independent of the reconstructed object grid;
        for multislice set ``resolution[0]`` to span the slices.
        """
        density_activation: nn.Module = nn.Identity()
        ms = list(multiscale_res_multipliers) if multiscale_res_multipliers is not None else None
        model: KPlanesType
        if tilted:
            model = KPlanesTILTED(
                M_features=M_features,
                resolution=resolution,
                multiscale_res_multipliers=ms,
                density_activation=density_activation,
                T=T,
                use_hybrid_mlp=use_hybrid_mlp,
                hybrid_hidden_dim=hybrid_hidden_dim,
                hybrid_num_layers=hybrid_num_layers,
            )
        else:
            model = KPlanes(
                M_features=M_features,
                resolution=resolution,
                multiscale_res_multipliers=ms,
                density_activation=density_activation,
                use_hybrid_mlp=use_hybrid_mlp,
                hybrid_hidden_dim=hybrid_hidden_dim,
                hybrid_num_layers=hybrid_num_layers,
            )
        # Zero the final decoder layer so the object starts at vacuum (global phase).
        with torch.no_grad():
            final_linear = (
                model.sigma_net[-1]
                if isinstance(model.sigma_net, nn.Sequential)
                else model.sigma_net
            )
            final_linear = cast(nn.Linear, final_linear)
            final_linear.weight.zero_()
            if final_linear.bias is not None:
                final_linear.bias.zero_()
        return cls.from_model(
            model,
            num_slices=num_slices,
            slice_thicknesses=slice_thicknesses,
            obj_type=obj_type,
            device=device,
            rng=rng,
        )

    @property
    def name(self) -> str:
        return "ObjectTensorDecomp"

    @property
    def model(self) -> KPlanesType:
        return cast(KPlanesType, self._model)

    def get_optimization_parameters(self) -> "dict[str, list[torch.Tensor]]":
        """PPLR: one param group per ``model.param_keys`` (hyperparameters baked by set_optimizer)."""
        model = self.model
        groups = model.get_params()
        return {key: list(groups[key]) for key in model.param_keys}

    def _normalize_optimizer_params(self, params):
        """Require a dict keyed by ``model.param_keys`` (PPLR); reject single-optimizer specs.

        The framework's "disabled" sentinel — a bare ``NoneOptimizer`` or a dict whose values
        are all ``NoneOptimizer`` (e.g. the ``{"default": NoneOptimizer()}`` set at init / by
        ``remove_optimizer`` and replayed through ``reset_optimizer`` on ``reconstruct(reset=True)``)
        — is passed straight to the base normalizer so the optimizer can be cleanly disabled
        without matching ``param_keys``.
        """
        if isinstance(params, OptimizerParams.NoneOptimizer) or (
            isinstance(params, dict)
            and len(params) > 0
            and all(isinstance(v, OptimizerParams.NoneOptimizer) for v in params.values())
        ):
            return super()._normalize_optimizer_params(params)
        if not isinstance(params, dict) or self._is_single_optimizer_dict(params):
            raise TypeError(
                f"{type(self).__name__} requires dict[str, OptimizerParamsType] keyed by "
                f"param_keys {self.model.param_keys}; got {type(params)}"
            )
        expected = set(self.model.param_keys)
        got = set(params.keys())
        if got != expected:
            raise ValueError(
                f"optimizer_params keys must match model.param_keys: got {got}, expected {expected}"
            )
        return super()._normalize_optimizer_params(params)

    @classmethod
    def from_pixelated(  # pyright: ignore[reportIncompatibleMethodOverride]  # K-Planes factory, intentionally diverges from ObjectINR.from_pixelated
        cls,
        pixelated: "ObjectModelType",
        model: KPlanesType | None = None,
        M_features: int = 16,
        resolution: Sequence[int] = (128, 128, 128),
        multiscale_res_multipliers: Sequence[float] | None = (0.25, 0.5, 1.0),
        use_hybrid_mlp: bool = False,
        tilted: bool = False,
        T: int = 4,
        device: str | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> "ObjectTensorDecomp":
        """Build a K-Planes object matching a pixelated object, with it as the pretrain target.

        Mirrors :meth:`ObjectINR.from_pixelated`: the K-Planes model is built to the pixelated
        object's geometry (``num_slices``, ``slice_thicknesses``, ``obj_type``, padded shape) and
        the current pixelated object is stored as the pretrain target, so ``pretrain()``
        warm-starts the grid to reproduce the pixelated reconstruction. Pass ``model`` to wrap a
        custom tensor-decomposition ``nn.Module`` directly; otherwise one is built from the
        ``M_features`` / ``resolution`` / ``tilted`` / ``T`` args.
        """
        if not (
            isinstance(pixelated, ObjectPixelated) or "ObjectPixelated" in str(type(pixelated))
        ):
            raise ValueError(f"pixelated must be an ObjectPixelated, got {type(pixelated)}")
        dev = pixelated.device if device is None else device
        seed = pixelated._rng_seed if rng is None else rng
        if model is not None:
            obj = cls.from_model(
                model=model,
                num_slices=pixelated.num_slices,
                slice_thicknesses=pixelated.slice_thicknesses,
                obj_type=pixelated.obj_type,
                device=dev,
                rng=seed,
            )
        else:
            obj = cls.from_uniform(
                num_slices=pixelated.num_slices,
                slice_thicknesses=pixelated.slice_thicknesses,
                M_features=M_features,
                resolution=resolution,
                multiscale_res_multipliers=multiscale_res_multipliers,
                use_hybrid_mlp=use_hybrid_mlp,
                tilted=tilted,
                T=T,
                obj_type=pixelated.obj_type,
                device=dev,
                rng=seed,
            )
        target = pixelated.obj.detach().to(dev)  # (num_slices, H, W) real phase / potential
        obj._obj_shape = tuple(int(x) for x in target.shape)  # type: ignore[assignment]
        if pixelated._sampling is not None:
            obj.sampling = pixelated.sampling
        obj.pretrain_target = target
        return obj

    def pretrain(
        self,
        pretrain_target: torch.Tensor | np.ndarray | None = None,
        num_iters: int = 200,
        optimizer_params: "dict | OptimizerParamsType | None" = None,
        scheduler_params: "dict | SchedulerParamsType | None" = None,
        loss_fn: Callable | str = "l2",
        device: str | int | None = None,
        show: bool = True,
        normalize_object_plotting: bool = True,
    ) -> None:
        """Warm-start the K-Planes grid by regressing it onto a target object (PPLR).

        Same direct grid->target regression as ``ObjectINR.pretrain`` (no forward model), but
        ``optimizer_params`` is PPLR-keyed. When ``None`` it defaults to per-group Adam
        (``grids`` lr 1e-2, others 1e-3) so pretraining works out of the box; the fitted weights
        become the ``reset()`` state.
        """
        if optimizer_params is None:
            optimizer_params = {
                key: OptimizerParams.Adam(lr=1e-2 if key == "grids" else 1e-3)
                for key in self.model.param_keys
            }
        super().pretrain(
            pretrain_target=pretrain_target,
            num_iters=num_iters,
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params,
            loss_fn=loss_fn,
            device=device,
            show=show,
            normalize_object_plotting=normalize_object_plotting,
        )


ObjectModelType = ObjectPixelated | ObjectDIP | ObjectINR | ObjectTensorDecomp
