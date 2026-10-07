from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import optuna
from numpy.typing import NDArray
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.rng import RNGMixin
from quantem.core.utils.utils import electron_wavelength_angstrom, to_numpy
from quantem.core.utils.validators import (
    validate_aberration_coefficients,
    validate_gt,
    validate_int,
    validate_tensor,
)
from quantem.core.visualization import show_2d
from quantem.diffractive_imaging.complex_probe import (
    FourierProbe,
    aberration_surface_cartesian_gradients,
    evaluate_probe,
    polar_coordinates,
    spatial_frequencies,
)
from quantem.diffractive_imaging.direct_ptycho_utils import _rotation_degrees_to_radians
from quantem.diffractive_imaging.ptycho_utils import (
    OptimizationParameter as OptimizationParameter,  # re-export: the documented path
)

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class HyperparameterState:
    initial_aberrations: dict[str, float] = field(default_factory=dict)
    initial_rotation_angle: float | None = None
    optimized_aberrations: dict[str, float] = field(default_factory=dict)
    optimized_rotation_angle: float | None = None
    optimized_keys: set[str] = field(default_factory=set)
    study: optuna.Study | None = None

    def __post_init__(self):
        self.initial_aberrations = validate_aberration_coefficients(dict(self.initial_aberrations))
        self.optimized_aberrations = validate_aberration_coefficients(self.optimized_aberrations)

        if self.optimized_keys:
            canonical = validate_aberration_coefficients(
                {k: 0.0 for k in self.optimized_keys if k != "rotation_angle"}
            )
            self.optimized_keys = set(canonical.keys()) | {
                k for k in self.optimized_keys if k == "rotation_angle"
            }

    def __repr__(self) -> str:
        return self.summarize(which="all")

    def current_aberrations(
        self, override_fixed: dict[str, float] | None = None
    ) -> Dict[str, float]:
        """Return full aberration dictionary (fixed ⊕ optimized)."""
        out = dict(self.initial_aberrations)
        out.update(self.optimized_aberrations)
        if override_fixed is not None:
            out.update(validate_aberration_coefficients(override_fixed))
        return out

    def current_rotation_angle(self, override_fixed: float | None = None) -> float:
        """Return rotation angle (optimized takes precedence)."""
        if override_fixed is not None:
            return override_fixed
        if self.optimized_rotation_angle is not None:
            return self.optimized_rotation_angle
        if self.initial_rotation_angle is not None:
            return self.initial_rotation_angle
        return 0.0

    def clear_optimized(self):
        """Clear all optimized aberrations and rotation angle."""
        self.optimized_aberrations.clear()
        self.optimized_rotation_angle = None
        self.optimized_keys.clear()
        self.study = None

    def clear_all(self):
        """Clear everything: initial and optimized hyperparameters."""
        self.initial_aberrations.clear()
        self.initial_rotation_angle = None
        self.clear_optimized()

    def copy(self):
        """ """
        return HyperparameterState(
            initial_aberrations=self.initial_aberrations,
            optimized_aberrations=self.optimized_aberrations,
            initial_rotation_angle=self.initial_rotation_angle,
            optimized_rotation_angle=self.optimized_rotation_angle,
            optimized_keys=self.optimized_keys,
            study=self.study,
        )

    def summarize(
        self,
        *,
        which: str = "current",
        override_aberration_coefs: dict[str, float] | None = None,
        override_rotation_angle: float | None = None,
    ) -> str:
        cls = self.__class__.__name__
        lines: list[str] = []

        def add(name: str, value):
            lines.append(f"  {name}={value!r},")

        if which == "initial":
            if self.initial_aberrations:
                add("initial_aberrations", self.initial_aberrations)
            if self.initial_rotation_angle is not None:
                add("initial_rotation_angle_deg", self.initial_rotation_angle)

        elif which == "optimized":
            if self.optimized_aberrations:
                add("optimized_aberrations", self.optimized_aberrations)
            if self.optimized_rotation_angle is not None:
                add("optimized_rotation_angle_deg", self.optimized_rotation_angle)

        elif which == "current":
            current_abers = self.current_aberrations(override_aberration_coefs)
            current_rot = self.current_rotation_angle(override_rotation_angle)

            if current_abers:
                add("current_aberrations", current_abers)
            if current_rot is not None:
                add("current_rotation_angle_deg", current_rot)

        elif which == "all":
            if self.initial_aberrations:
                add("initial_aberrations", self.initial_aberrations)
            if self.initial_rotation_angle is not None:
                add("initial_rotation_angle_deg", self.initial_rotation_angle)
            if self.optimized_aberrations:
                add("optimized_aberrations", self.optimized_aberrations)
            if self.optimized_rotation_angle is not None:
                add("optimized_rotation_angle_deg", self.optimized_rotation_angle)

        else:
            raise ValueError(
                f"`which` must be one of "
                f'{{"initial", "optimized", "current", "all"}}, got {which!r}'
            )

        if not lines:
            return f"{cls}()"

        body = "\n".join(lines)
        return f"{cls}(\n{body}\n)"


@dataclass(frozen=True)
class BrightFieldContext:
    bf_mask: torch.Tensor
    bf_inds_i: torch.Tensor
    bf_inds_j: torch.Tensor
    num_bf: int
    vbf_index_mapping: torch.Tensor


class DirectPtychographyBase(RNGMixin, AutoSerialize):
    """
    Shared state and hyperparameter machinery for direct-ptychography reconstructions.

    This class holds everything that does not depend on *how* the deconvolution is
    performed: geometry/sampling bookkeeping, the aberration and rotation
    :class:`HyperparameterState`, bright-field mask indexing, visualization, and the
    optuna / grid-search drivers. Subclasses supply the reconstruction itself.

    Subclass contract
    -----------------
    Attributes a subclass must set in ``__init__``, in this order (the ``scan_sampling``
    and ``reciprocal_sampling`` setters read the units and wavelength):

    1. ``device``, ``verbose``, ``rng``
    2. ``wavelength``, ``scan_units``, ``detector_units``
    3. ``scan_sampling``, ``reciprocal_sampling``, ``angular_sampling``
    4. ``gpts``, ``sampling`` (detector grid), ``bf_mask``, ``semiangle_cutoff``
    5. ``hyperparameter_state``

    Methods a subclass must implement:

    - ``reconstruct(*, override_aberration_coefs, override_rotation_angle, verbose, ...)``
      returning ``self``
    - ``variance_loss()`` returning a scalar tensor (a *method*, not a property --
      the optimizers call ``float(self.variance_loss())``)
    - ``corrected_bf`` property returning the reconstructed image, or ``None``

    A subclass whose reconstruction spans more than the scan field of view (e.g. a padded
    canvas) must override ``_obj_fov``; upsampling needs no bookkeeping, since
    ``_obj_sampling`` reads the object's own shape.
    """

    # --- state the subclass __init__ must provide (annotation only, no default) ---
    hyperparameter_state: HyperparameterState
    scan_units: Tuple[str, str]
    detector_units: Tuple[str, str]
    scan_gpts: Tuple[int, int]
    gpts: Tuple[int, int]
    sampling: Tuple[float, float]
    angular_sampling: Tuple[float, float]

    @property
    def wavelength(self) -> float:
        """Probe wavelength in Angstrom."""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: float) -> None:
        self._wavelength = float(validate_gt(value, 0.0, "wavelength"))

    @property
    def fourier_probe(self) -> "FourierProbe | None":
        """An empirical complex probe, or ``None`` when the probe is analytic.

        Set it to reconstruct with a measured ``psi(k)`` instead of an aperture plus
        aberrations. Everything that reads the aberration surface -- the parallax kernel,
        the ``sign(sin(chi))`` phase flip, the defocus gradient, any hyperparameter search
        over aberrations -- has no meaning then, and raises.
        """
        return getattr(self, "_fourier_probe", None)

    @fourier_probe.setter
    def fourier_probe(self, value) -> None:
        if value is None:
            self._fourier_probe = None
            return
        if not isinstance(value, FourierProbe):
            raise TypeError(
                "`fourier_probe` must be a FourierProbe; build one with "
                "`FourierProbe.from_array(psi, reciprocal_sampling, wavelength)`."
            )
        if value.array is None:
            raise ValueError(
                "`fourier_probe` is for an empirical probe; an analytic one is already "
                "described by `semiangle_cutoff` and the aberration coefficients."
            )
        if tuple(value.array.shape) != tuple(self.gpts):
            raise ValueError(
                f"`fourier_probe` has shape {tuple(value.array.shape)} but the detector grid "
                f"is {tuple(self.gpts)}. They must match -- crop the probe the same way the "
                "diffraction patterns were cropped."
            )
        self._fourier_probe = value.to(self.device)

    def _require_analytic_probe(self, what: str) -> None:
        """Guard for everything that only means something for an aperture plus aberrations."""
        if self.fourier_probe is not None:
            raise NotImplementedError(
                f"{what} is defined by the aberration surface chi(k), which an empirical "
                "`fourier_probe` does not have. Use a deconvolution kernel that does not "
                "need it -- 'ssb', 'obf' and 'mf' read only the probe itself, and 'icom' "
                "does not read the probe at all."
            )

    def _return_probe(self, aberration_coefs) -> "FourierProbe":
        """The probe object the overlap function samples, empirical or analytic."""
        probe = self.fourier_probe
        if probe is not None:
            return probe
        return FourierProbe.from_aberrations(
            self.wavelength,
            self.semiangle_cutoff,
            self.angular_sampling,
            aberration_coefs,
            self.soft_edges,
        )

    def _return_probe_on_grid(self, k, phi, aberration_coefs):
        """``psi(k)`` on the detector grid, whose squared sum is the bright-field weight.

        Leaves ``soft_edges`` at ``evaluate_probe``'s default rather than taking
        ``self.soft_edges``, so the normalization matches between the two classes. This is
        an inconsistency, kept for compatibility.
        """
        probe = self.fourier_probe
        if probe is not None:
            return probe.array
        return evaluate_probe(
            k * self.wavelength,
            phi,
            self.semiangle_cutoff,
            self.angular_sampling,
            self.wavelength,
            aberration_coefs=aberration_coefs,
        )

    @staticmethod
    def _resolve_wavelength(energy: float | None, wavelength: float | None) -> float:
        """Wavelength in Angstrom, from an electron accelerating voltage or given directly.

        ``energy`` goes through the relativistic electron de Broglie formula, which is what
        every electron entry point wants. ``wavelength`` skips it, which is what anything
        that is not an electron needs: for a 7.9 keV photon the electron formula returns
        0.137 Angstrom against the correct ``hc/E`` = 1.569.
        """
        if (energy is None) == (wavelength is None):
            raise ValueError(
                "Pass exactly one of `energy` (electron accelerating voltage, in volts) or "
                "`wavelength` (in Angstrom, for photons or anything else non-electron), "
                f"got energy={energy!r} and wavelength={wavelength!r}."
            )
        if wavelength is not None:
            return float(validate_gt(wavelength, 0.0, "wavelength"))
        return electron_wavelength_angstrom(validate_gt(energy, 0.0, "energy"))

    # ------------------------------------------------------------------
    # subclass hooks
    # ------------------------------------------------------------------

    def reconstruct(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} does not implement reconstruct().")

    def variance_loss(self):
        raise NotImplementedError(f"{type(self).__name__} does not implement variance_loss().")

    def rms_gradient_loss(self):
        """
        Negated RMS gradient of the reconstruction, per Angstrom -- a sharpness objective.

        The classic autofocus metric: a correctly deconvolved image has sharp edges and a
        large gradient, a mis-set aberration blurs them. Negated so that, like
        :meth:`variance_loss`, it is minimized.

        It is better conditioned than the variance loss -- 28% dynamic range against 0.08%
        over a defocus series, agreeing on the optimum -- because the variance loss compares
        bright-field images with each other and saturates once they agree, while this
        measures the reconstruction. It is also insensitive to how the canvas is sized, so it
        needs no pinning the way a patch fit does.

        Two caveats:

        - It rewards *amplitude*, not only sharpness, since it is not normalized by the
          image's own spread. Aberrations and rotation barely change the overall scale, so
          this is safe for the searches here, but a hyperparameter that could inflate the
          object would game it.
        - It is defined for every deconvolution kernel, where
          ``DirectPtychographyMontage.variance_loss`` is defined only for the parallax one.
          That makes it the way to drive a search over ``"ssb"`` or ``"obf"``.

        Returns
        -------
        float or None
            ``None`` before :meth:`reconstruct`, mirroring :meth:`variance_loss`.
        """
        obj = self.corrected_bf
        if obj is None:
            return None
        if min(obj.shape[-2:]) < 2:
            raise ValueError(
                f"An object of shape {tuple(obj.shape)} has no gradient to measure; "
                "reconstruct onto a canvas at least 2x2."
            )

        # per Angstrom rather than per pixel, so the value is comparable across upsampling
        # factors and samplings rather than only within one search
        spacing = tuple(float(s) for s in self._obj_sampling)
        grad_rows, grad_cols = torch.gradient(obj.to(torch.float32), spacing=spacing, dim=(-2, -1))
        return -float(torch.sqrt((grad_rows.square() + grad_cols.square()).mean()))

    #: objectives the hyperparameter searches accept by name, all minimized
    _LOSS_FUNCTIONS = {
        "variance": "variance_loss",
        "rms_gradient": "rms_gradient_loss",
    }

    def _return_loss_value(self, loss) -> float:
        """Evaluate a search objective on the current reconstruction."""
        if callable(loss):
            return float(loss(self))
        try:
            method = self._LOSS_FUNCTIONS[loss]
        except (KeyError, TypeError):
            raise ValueError(
                f"`loss` must be a callable or one of {sorted(self._LOSS_FUNCTIONS)}, got {loss!r}"
            ) from None
        return float(getattr(self, method)())

    @property
    def corrected_bf(self):
        raise NotImplementedError(f"{type(self).__name__} does not implement corrected_bf.")

    @property
    def fov(self) -> tuple[float, float]:
        """Field of view of the scan, in Angstrom. Fixed by the acquisition."""
        return tuple(n * s for n, s in zip(self.scan_gpts, self.scan_sampling))

    @property
    def scan_origin(self) -> tuple[float, float]:
        """Position of scan pixel ``(0, 0)``, in Angstrom, in the caller's coordinates.

        Zero for a raster acquisition, whose grid *defines* the coordinates. The ungridded
        constructors anchor the scan grid at the corner of the probe-position bounding box,
        and record that corner here so pixels can be mapped back to the positions that were
        passed in -- which is what makes two acquisitions of the same region comparable.
        """
        return getattr(self, "_scan_origin", (0.0, 0.0))

    @scan_origin.setter
    def scan_origin(self, value) -> None:
        if value is None:
            self._scan_origin = (0.0, 0.0)
            return
        origin = tuple(float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1))
        if len(origin) != 2:
            raise ValueError(f"`scan_origin` must be a (row, col) pair, got {value!r}")
        self._scan_origin = origin

    @property
    def _obj_fov(self) -> tuple[float, float]:
        """Field of view the reconstructed object spans, in Angstrom.

        Defaults to the scan field of view, which is what a reconstruction sampled on the
        scan grid covers at any upsampling factor. Override where the object spans more.
        """
        return self.fov

    @property
    def obj_origin(self) -> tuple[float, float]:
        """Position of object pixel ``(0, 0)``, in Angstrom, in the caller's coordinates.

        Together with :attr:`_obj_sampling` this is the full map from object pixels back to
        the probe positions that were passed in: ``origin + pixel * sampling``. Defaults to
        :attr:`scan_origin`, for a reconstruction sampled on the scan grid; a class whose
        canvas starts elsewhere must override it.
        """
        return self.scan_origin

    @property
    def _obj_sampling(self) -> tuple[float, float]:
        """Real-space sampling of the reconstructed object, in Angstrom.

        Derived from the object's own shape rather than tracked across reconstructions, so
        upsampling needs no bookkeeping and the scalebar cannot fall out of sync.
        """
        obj = self.corrected_bf
        if obj is None:
            return tuple(self.scan_sampling)
        return tuple(f / n for f, n in zip(self._obj_fov, obj.shape[-2:]))

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def verbose(self) -> int:
        return self._verbose

    @verbose.setter
    def verbose(self, v: bool | int | float) -> None:
        self._verbose = validate_int(validate_gt(v, -1, "verbose"), "verbose")

    @property
    def bf_mask(self) -> torch.Tensor:
        return self._bf_mask

    @bf_mask.setter
    def bf_mask(self, value: torch.Tensor):
        self._bf_mask = validate_tensor(value, "bf_mask", dtype=torch.bool).to(device=self.device)

    @property
    def rotation_angle(self) -> float:
        """Current detector rotation angle in degrees."""
        return self.hyperparameter_state.current_rotation_angle()

    @property
    def aberration_coefs(self) -> dict:
        return self.hyperparameter_state.current_aberrations()

    @property
    def semiangle_cutoff(self) -> float:
        return self._semiangle_cutoff

    @semiangle_cutoff.setter
    def semiangle_cutoff(self, value: float):
        if value is None:
            # an empirical probe already carries its own aperture, whatever shape it is
            if self.fourier_probe is not None:
                self._semiangle_cutoff = None
                return
            raise ValueError(
                "`semiangle_cutoff` is required, in mrad: it sets the aperture used to build "
                "the probe and the deconvolution kernels. Pass a `fourier_probe` instead if "
                "the probe is measured rather than described by an aperture."
            )
        validate_gt(value, 0.0, "semiangle_cutoff")
        self._semiangle_cutoff = value

    @property
    def device(self) -> str | torch.device:
        """This should be of form 'cuda:X' or 'cpu', as defined by quantem.config"""
        if hasattr(self, "_device"):
            return self._device  # ty:ignore[invalid-return-type]
        else:
            return config.get("device")

    @device.setter
    def device(self, device: str | int | None):
        if device is not None:
            dev, _id = config.validate_device(device)
            self._device = dev

    @property
    def scan_sampling(self) -> NDArray:
        return self._scan_sampling  # ty:ignore[invalid-return-type]

    @scan_sampling.setter
    def scan_sampling(self, value: NDArray | tuple | list) -> None:
        """
        Units A or raises error
        """
        units = self.scan_units
        if units[0] == "A":
            self._scan_sampling = value
        else:
            raise ValueError("real-space needs to be given in 'A'")

    @property
    def reciprocal_sampling(self) -> NDArray:
        return self._reciprocal_sampling  # ty:ignore[invalid-return-type]

    @reciprocal_sampling.setter
    def reciprocal_sampling(self, value: NDArray | tuple | list) -> None:
        """
        Units A or raises error
        """
        units = self.detector_units
        if units[0] == "A^-1":
            self._reciprocal_sampling = value
        elif units[0] == "mrad":
            self._reciprocal_sampling = tuple(val / self.wavelength / 1e3 for val in value)
        else:
            raise ValueError("reciprocal-space needs to be given in 'A^-1' or 'mrad'")

    @property
    def obj(self) -> np.ndarray | None:
        """Reconstructed object as a numpy array, or ``None`` before :meth:`reconstruct`.

        Mirrors ``corrected_bf`` rather than raising: ``AutoSerialize._recursive_load``
        walks ``dir(obj)`` and evaluates every property, so a ``to_numpy(None)`` here made
        save/load fail for a instance that had not been reconstructed yet.
        """
        corrected_bf = self.corrected_bf
        if corrected_bf is None:
            return None
        return to_numpy(corrected_bf)

    # ------------------------------------------------------------------
    # bright-field geometry
    # ------------------------------------------------------------------

    def _return_bf_context(self, bf_mask):
        """
        Given a BF mask, compute all BF-dependent geometry and indexing.
        """
        bf_mask = torch.as_tensor(bf_mask, dtype=torch.bool, device=self.device)

        bf_inds_i, bf_inds_j = torch.nonzero(bf_mask, as_tuple=True)
        vbf_index_mapping = torch.where(bf_mask[self.bf_mask])[0]
        num_bf = bf_inds_i.numel()

        return BrightFieldContext(
            bf_mask=bf_mask,
            bf_inds_i=bf_inds_i,
            bf_inds_j=bf_inds_j,
            num_bf=num_bf,
            vbf_index_mapping=vbf_index_mapping,
        )

    def _make_checkerboard_bf_masks(self, gpts, bf_mask):
        """ """
        i_coords = torch.arange(gpts[0], device=self.device)
        j_coords = torch.arange(gpts[1], device=self.device)
        i_grid, j_grid = torch.meshgrid(i_coords, j_coords, indexing="ij")
        checkerboard = torch.fft.ifftshift(((i_grid + j_grid) % 2).bool())

        bf1 = bf_mask & checkerboard
        bf2 = bf_mask & (~checkerboard)

        return [bf1, bf2]

    def _normalize_kernel_name(self, kernel):
        kernel = kernel.lower()

        aliases = {
            "ssb": "ssb",
            "single-sideband": "ssb",
            "acbf": "ssb",
            "aberration-corrected-bright-field": "ssb",
            "obf": "obf",
            "optimum-bright-field": "obf",
            "mf": "mf",
            "matched-filter": "mf",
            "prlx": "prlx",
            "parallax": "prlx",
            "tcbf": "prlx",
            "tilt-corrected-bright-field": "prlx",
            "icom": "icom",
            "center-of-mass": "icom",
        }

        if kernel not in aliases:
            raise ValueError(f"Unknown deconvolution kernel '{kernel}'")

        return aliases[kernel]

    def _return_lateral_shifts(
        self,
        rotation_angle,
        aberration_coefs,
        bf_mask,
    ):
        """Aberration-induced lateral shift of each BF pixel, in Angstrom."""
        # Get initial shifts
        kxa, kya = spatial_frequencies(
            self.gpts,
            self.sampling,
            rotation_angle=_rotation_degrees_to_radians(rotation_angle),
            device=self.device,
        )
        k, phi = polar_coordinates(kxa, kya)

        dx, dy = aberration_surface_cartesian_gradients(
            k * self.wavelength,
            phi,
            aberration_coefs=aberration_coefs,
        )
        grad_k = torch.stack((dx[bf_mask], dy[bf_mask]), -1)
        lateral_shifts = grad_k / 2 / np.pi
        return lateral_shifts

    # ------------------------------------------------------------------
    # visualization
    # ------------------------------------------------------------------

    def visualize(
        self,
        return_fig: bool = False,
        show_obj_fft: bool = True,
        apply_hanning_window: bool = False,
        **kwargs,
    ):
        """
        Show the reconstructed object and its Hann-windowed Fourier transform.

        Parameters
        ----------
        cbar : bool, optional
            Whether to show colorbars, by default True.
        return_fig : bool, optional
            If True, return ``(fig, axs)``.
        fft_norm : str | dict, optional
            Normalization passed to ``show_2d`` for the object FFT.
        **kwargs
            Additional arguments passed to ``show_2d``.
        """
        if self.corrected_bf is None:
            raise RuntimeError("Run reconstruct() before visualize().")

        obj = self.obj
        obj_sampling = self._obj_sampling
        obj_scalebar = {"sampling": obj_sampling[1], "units": "Å"}

        if show_obj_fft:
            if apply_hanning_window:
                window = np.hanning(obj.shape[-2])[:, None] * np.hanning(obj.shape[-1])[None, :]
                obj_fft = np.fft.fftshift(np.abs(np.fft.fft2(obj * window)))
            else:
                obj_fft = np.fft.fftshift(np.abs(np.fft.fft2(obj)))

            fft_sampling = 1 / (obj_sampling[1] * obj.shape[-1])
            fft_scalebar = {"sampling": fft_sampling, "units": r"$\mathrm{A^{-1}}$"}

            fig, axs = show_2d(
                [obj, obj_fft],
                title=["Object phase", "Object phase FFT"],
                scalebar=[obj_scalebar, fft_scalebar],
                **kwargs,
            )
            axs[1].set_aspect(obj.shape[-1] / obj.shape[-2])
        else:
            fig, axs = show_2d(
                obj,
                title="Object phase",
                scalebar=obj_scalebar,
                **kwargs,
            )

        if return_fig:
            return fig, axs
        return None

    # ------------------------------------------------------------------
    # hyperparameter optimization
    # ------------------------------------------------------------------

    def optimize_hyperparameters(
        self,
        aberration_coefs: dict[str, float | OptimizationParameter] | None = None,
        rotation_angle: float | OptimizationParameter | None = None,
        n_trials=50,
        sampler=None,
        loss="variance",
        verbose=None,
        **reconstruct_kwargs,
    ):
        """
        Optimize hyperparameters (aberrations and/or rotation) using Optuna.

        Parameters
        ----------
        aberration_coefs : dict[str, float|OptimizationParameter]
            Dict of aberration names to either fixed values or optimization ranges.
        rotation_angle : float|OptimizationParameter
            Fixed rotation or optimization range, in degrees.
        n_trials : int
            Number of Optuna trials.
        sampler : optuna.samplers.BaseSampler, optional
            Custom Optuna sampler.
        loss : {"variance", "rms_gradient"} or callable
            Objective to minimize. ``"variance"`` is :meth:`variance_loss`, the spread
            between bright-field images. ``"rms_gradient"`` is
            :meth:`rms_gradient_loss`, an image-sharpness objective that is better
            conditioned -- 28% dynamic range against 0.08% over a defocus series -- and is
            defined for every deconvolution kernel. A callable is passed the reconstruction
            and must return a float to minimize.
        verbose : bool, optional
            Report the search and show Optuna's progress bar. Defaults to :attr:`verbose`.
        **reconstruct_kwargs :
            Extra arguments passed to reconstruct().
        """

        if verbose is None:
            verbose = self.verbose

        sampler = sampler or optuna.samplers.TPESampler()

        state = self.hyperparameter_state
        aberration_coefs = aberration_coefs or {}

        # Reset optimized bookkeeping
        state.clear_optimized()

        # Partition inputs
        fixed_override_aberrations = {}
        optimizable_aberrations = {}

        for name, val in aberration_coefs.items():
            if isinstance(val, OptimizationParameter):
                optimizable_aberrations[name] = val
                state.optimized_keys.add(name)
            else:
                fixed_override_aberrations[name] = val

        if isinstance(rotation_angle, OptimizationParameter):
            state.optimized_keys.add("rotation_angle")

        def objective(trial):
            trial_aberrations = {}
            for name, val in optimizable_aberrations.items():
                trial_aberrations[name] = trial.suggest_float(name, val.low, val.high, log=val.log)

            trial_aberrations |= fixed_override_aberrations

            if isinstance(rotation_angle, OptimizationParameter):
                rot = trial.suggest_float(
                    "rotation_angle",
                    rotation_angle.low,
                    rotation_angle.high,
                    log=rotation_angle.log,
                )
            else:
                rot = rotation_angle

            self.reconstruct(
                override_aberration_coefs=trial_aberrations,
                override_rotation_angle=rot,
                verbose=False,
                **reconstruct_kwargs,
            )
            return self._return_loss_value(loss)

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=bool(verbose))

        # Write back optimized results
        best = study.best_params.copy()
        state.optimized_rotation_angle = best.pop("rotation_angle", None)

        if state.optimized_rotation_angle is None and rotation_angle is not None:
            state.optimized_rotation_angle = rotation_angle  # ty:ignore[invalid-assignment]

        state.optimized_aberrations = best
        state.optimized_aberrations = state.current_aberrations(fixed_override_aberrations)
        state.study = study

        if verbose:
            print("Optimized state:\n\n", self.hyperparameter_state)

        self.reconstruct(verbose=False, **reconstruct_kwargs)
        return self

    def grid_search_hyperparameters(
        self,
        aberration_coefs: dict[str, float | OptimizationParameter] | None = None,
        rotation_angle: float | OptimizationParameter | None = None,
        loss="variance",
        verbose=None,
        **reconstruct_kwargs,
    ):
        """
        Exhaustive search over a grid of hyperparameter values.

        Parameters
        ----------
        loss : {"variance", "rms_gradient"} or callable
            Objective to minimize; see :meth:`optimize_hyperparameters`.
        """
        if verbose is None:
            verbose = self.verbose

        aberration_coefs = aberration_coefs or {}
        state = self.hyperparameter_state

        # Reset optimized bookkeeping
        state.clear_optimized()

        # Partition inputs
        fixed_override_aberrations: dict[str, float] = {}
        optimizable_aberrations: dict[str, OptimizationParameter] = {}

        for name, val in aberration_coefs.items():
            if isinstance(val, OptimizationParameter):
                optimizable_aberrations[name] = val
                state.optimized_keys.add(name)
            else:
                fixed_override_aberrations[name] = val

        optimize_rotation = isinstance(rotation_angle, OptimizationParameter)
        if optimize_rotation:
            state.optimized_keys.add("rotation_angle")

        # Build parameter grid (only over optimizable parameters)
        param_grid: dict[str, list[float]] = {}

        for name, param in optimizable_aberrations.items():
            param_grid[name] = param.grid_values()

        # isinstance inline rather than reusing `optimize_rotation`, so the type narrows
        if isinstance(rotation_angle, OptimizationParameter):
            param_grid["rotation_angle"] = rotation_angle.grid_values()

        # Cartesian product
        keys = list(param_grid.keys())
        grid = list(product(*(param_grid[k] for k in keys)))

        best_loss = float("inf")
        best_params: dict[str, float] | None = None
        results = []

        for combo in tqdm(grid, disable=not verbose):
            trial_params = dict(zip(keys, combo))

            trial_aberrations = dict(fixed_override_aberrations)
            trial_aberrations.update(
                {k: v for k, v in trial_params.items() if k != "rotation_angle"}
            )

            if optimize_rotation:
                rot = trial_params["rotation_angle"]
            else:
                rot = rotation_angle

            self.reconstruct(
                override_aberration_coefs=trial_aberrations,
                override_rotation_angle=rot,
                verbose=False,
                **reconstruct_kwargs,
            )

            loss_value = self._return_loss_value(loss)
            results.append((trial_params, loss_value))

            if loss_value < best_loss:
                best_loss = loss_value
                best_params = trial_params

        self._grid_search_results = results

        # Write back best optimized values
        if best_params is not None:
            best_params = best_params.copy()
            state.optimized_rotation_angle = best_params.pop("rotation_angle", None)
            state.optimized_aberrations = best_params

        if state.optimized_rotation_angle is None and rotation_angle is not None:
            state.optimized_rotation_angle = rotation_angle  # ty:ignore[invalid-assignment]

        state.optimized_aberrations = state.current_aberrations(fixed_override_aberrations)

        if verbose:
            print("Optimized state:\n\n", self.hyperparameter_state)

        # Final reconstruction using merged state
        self.reconstruct(verbose=False, **reconstruct_kwargs)
        return self
