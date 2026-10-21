from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Dict, Mapping, Tuple

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
from quantem.diffractive_imaging.direct_ptycho_utils import (
    EIGENVALUE_FLOOR,
    OBJ_TYPES,
    _rotation_degrees_to_radians,
    _to,
    alias_tiles,
    column_factors,
    explained_energy,
    frequency_rings,
    from_alias_tiles,
    model_aperture,
    noise_variance_from_residual,
    residual_per_degree_of_freedom,
    slice_depths,
    solve_normal_equations,
    wiener_estimate,
)
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
class HyperparameterState(AutoSerialize):
    """The initial and optimized hyperparameters of a direct reconstruction.

    It saves as a native subtree of its reconstruction, so the values can be read from the file.
    An optuna ``study``, when an optimization left one, is pickled instead.
    """

    initial_aberrations: dict[str, float] = field(default_factory=dict)
    initial_rotation_angle: float | None = None
    optimized_aberrations: dict[str, float] = field(default_factory=dict)
    optimized_rotation_angle: float | None = None
    optimized_keys: set[str] = field(default_factory=set)
    study: optuna.Study | None = None
    optimized_slice_thicknesses: list[float] | None = None

    #: optimized keys that are not aberration coefficients
    _OTHER_KEYS = ("rotation_angle", "slice_thicknesses")

    def __post_init__(self):
        self.initial_aberrations = validate_aberration_coefficients(dict(self.initial_aberrations))
        self.optimized_aberrations = validate_aberration_coefficients(self.optimized_aberrations)

        if self.optimized_keys:
            canonical = validate_aberration_coefficients(
                {k: 0.0 for k in self.optimized_keys if k not in self._OTHER_KEYS}
            )
            self.optimized_keys = set(canonical.keys()) | {
                k for k in self.optimized_keys if k in self._OTHER_KEYS
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

    def current_slice_thicknesses(self, override_fixed=None):
        """Slice thicknesses to reconstruct with: the override, else the optimized ones, else ``None``."""
        if override_fixed is not None:
            return override_fixed
        return self.optimized_slice_thicknesses

    def clear_optimized(self):
        """Clear all optimized aberrations, rotation angle and slice thicknesses."""
        self.optimized_aberrations.clear()
        self.optimized_rotation_angle = None
        self.optimized_slice_thicknesses = None
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
            optimized_slice_thicknesses=self.optimized_slice_thicknesses,
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
            if self.optimized_slice_thicknesses is not None:
                add("optimized_slice_thicknesses", self.optimized_slice_thicknesses)

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
            if self.optimized_slice_thicknesses is not None:
                add("optimized_slice_thicknesses", self.optimized_slice_thicknesses)

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
      returning ``self``, and storing ``_residual`` for :meth:`residual_loss`
    - ``corrected_bf`` property returning the reconstructed phase, or ``None``

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

        ``psi(k)`` is in the scan frame, the frame the aberration coefficients are defined
        in, and is read at the detector's k-grid rotated by ``rotation_angle``. A rotation
        that does not map the grid onto itself puts those reads between grid points, so build
        the probe with ``interpolation="bilinear"`` and refine it with
        ``FourierProbe.resampled_to`` until the error is acceptable.

        It can be a mixed state, ``FourierProbe.from_array`` of ``(M, Ny, Nx)`` incoherent
        modes such as a mixed-state ptychographic reconstruction returns. The overlap
        function is then summed over the modes, for every kernel that reads the probe and
        for complex and multislice reconstructions alike. The slices are centred on the
        plane the modes describe.
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
        shape, gpts = tuple(value.array.shape), tuple(int(n) for n in self.gpts)
        factors = {n // g for n, g in zip(shape, gpts)}
        if any(n % g for n, g in zip(shape, gpts)) or len(factors) != 1:
            raise ValueError(
                f"`fourier_probe` has shape {shape} but the detector grid is {gpts}. They must "
                "match, or the probe must be the same grid refined by a whole number with "
                "`FourierProbe.resampled_to` -- crop the probe the same way the diffraction "
                "patterns were cropped."
            )
        self._fourier_probe = value.to(self.device)

    def _require_analytic_probe(self, what: str, aberration_coefs: Mapping | None = None) -> None:
        """Guard for everything that only means something for an aperture plus aberrations.

        An empirical ``fourier_probe`` carries a phase but no aberration surface, so there is
        nothing to differentiate. Passing ``aberration_coefs`` alongside one says the caller
        has a model of that phase. The coefficients then supply the shift field, while the
        measured probe still supplies the amplitude weighting and the bright-field mask.
        :func:`~quantem.diffractive_imaging.direct_ptycho_utils.fit_aberrations_from_probe`
        fits those coefficients and reports whether the probe justified fitting them.
        """
        if self.fourier_probe is None:
            return
        if aberration_coefs and any(
            not key.startswith("phi") and float(value) != 0.0
            for key, value in aberration_coefs.items()
        ):
            return
        raise NotImplementedError(
            f"{what} is defined by the aberration surface chi(k), which an empirical "
            "`fourier_probe` does not have. Either pass `aberration_coefs` modelling its "
            "phase -- `fit_aberrations_from_probe` fits them and reports whether the phase "
            "is smooth enough to be worth modelling -- or use a deconvolution kernel that "
            "does not need chi: 'ssb' and 'obf' read only the probe itself, and "
            "'icom' does not read the probe at all."
        )

    @property
    def focal_spread(self) -> float:
        """Rms spread of defocus in Angstrom, from the energy spread and lens instabilities.

        With :attr:`source_size` it describes partial coherence by a closed-form envelope on
        the overlap function (see
        :func:`~quantem.diffractive_imaging.complex_probe.partial_coherence_envelope`), the
        parametric alternative to a mixed-state :attr:`fourier_probe`. It weights each
        bright-field pixel differently, so it changes every kernel that reads the overlap
        function and :meth:`residual_loss`, which can therefore fit it: on simulated data the
        residual was smallest at the true 50 A. SSB keeps its unit-modulus weights from the
        coherent overlap, since the envelope gave the in-focus triple overlap a weight it has
        no signal for and lowered its correlation with the object (0.920 to 0.893).
        """
        return getattr(self, "_focal_spread", 0.0)

    @focal_spread.setter
    def focal_spread(self, value) -> None:
        self._focal_spread = self._validate_spread(value, "focal_spread")

    @property
    def source_size(self) -> float:
        """Rms spread of the probe position per axis in Angstrom, from the finite source.

        It multiplies the data at scan frequency ``q`` by ``exp(-2 pi^2 source_size^2 |q|^2)``
        whatever the bright-field pixel, so the least-squares fit, :meth:`residual_loss`, SSB
        and OBF are all unchanged by it, and the data cannot determine it: it must come from a
        calibration. It matters to the kernel that estimates the object itself, ``"wiener"``,
        which divides it back out where the signal allows; WDD's fixed ridge does not.
        """
        return getattr(self, "_source_size", 0.0)

    @source_size.setter
    def source_size(self, value) -> None:
        self._source_size = self._validate_spread(value, "source_size")

    @staticmethod
    def _validate_spread(value, name):
        if isinstance(value, torch.Tensor) and value.requires_grad:
            return value
        value = float(value)
        if not value >= 0:
            raise ValueError(
                f"`{name}` must be a non-negative rms spread in Angstrom, got {value}"
            )
        return value

    def _return_probe(self, aberration_coefs) -> "FourierProbe":
        """The probe object the overlap function samples, empirical or analytic, with the
        partial-coherence envelope of :attr:`focal_spread` and :attr:`source_size`."""
        probe = self.fourier_probe
        if probe is None:
            probe = FourierProbe.from_aberrations(
                self.wavelength,
                self.semiangle_cutoff,
                self.angular_sampling,
                aberration_coefs,
                self.soft_edges,
            )
        return probe.with_partial_coherence(self.focal_spread, self.source_size)

    def _return_probe_on_grid(self, k, phi, aberration_coefs, probe=None):
        """``psi(k)`` on the detector grid, whose squared sum is the bright-field weight.

        Leaves ``soft_edges`` at ``evaluate_probe``'s default rather than taking
        ``self.soft_edges``, so the normalization matches between the two classes. This is
        an inconsistency, kept for compatibility.

        An array probe, :attr:`fourier_probe` or the ``probe`` given, is in the scan frame, as
        the aberration surface is, so it is read at the rotated k-grid where the overlap
        function also reads it. The montage passes the probe it has refined onto its canvas.
        """
        probe = self.fourier_probe if probe is None else probe
        if probe is not None and probe.array is not None:
            return probe.read(k * torch.cos(phi), k * torch.sin(phi))
        return evaluate_probe(
            k * self.wavelength,
            phi,
            self.semiangle_cutoff,
            self.angular_sampling,
            self.wavelength,
            aberration_coefs=aberration_coefs,
        )

    @classmethod
    def _model_aperture_for(cls, energy, wavelength, semiangle_cutoff, fourier_probe, device):
        """The probe model's ``|psi(k)|`` as the stack builders evaluate it, or ``None``.

        The builders call it once the detector rotation is known, so that an empirical probe,
        which lives in the scan frame, is read where the reconstruction will read it.
        """
        if fourier_probe is None and (semiangle_cutoff is None or not semiangle_cutoff > 0):
            return None  # the constructor reports a bad semiangle in its own terms
        resolved = cls._resolve_wavelength(energy, wavelength)

        def aperture(gpts, reciprocal_sampling, detector_units, rotation_angle):
            return model_aperture(
                gpts,
                reciprocal_sampling,
                detector_units,
                resolved,
                semiangle_cutoff,
                fourier_probe,
                rotation_angle,
                device,
            )

        return aperture

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

    def residual_loss(self):
        """
        Fraction of the data the reconstruction's forward model leaves unexplained.

        Every kernel is a linear model of the scan-space Fourier transform of the bright-field
        stack, ``G(q, k)``, with one or more columns per spatial frequency. The least-squares
        residual of that model is ``sum |G|^2 - sum_q b^H M^+ b``, with ``M`` the Gram matrix
        of the columns and ``b`` their projections onto the data, and this reports it as a
        fraction of ``sum |G|^2``. Minimizing it over aberrations, rotation or slice depths is
        the maximum-likelihood fit of those hyperparameters under white noise.

        The columns are the kernel's own model:

        - ``ssb``, ``obf``, ``wiener`` and ``wdd``: the weak-object overlap functions,
          ``Gamma-`` per slice and, for ``obj_type="complex"``, ``Gamma+`` per slice. These
          kernels share the model, so at the same hyperparameters they report the same value.
        - ``prlx``: one shifted copy of the image per bright-field pixel, so this is the spread
          of the parallax-corrected images about their mean, which the earlier
          ``variance_loss`` measured.
        - ``icom``: the first moment ``k . q``.

        Only frequencies inside the band limit ``2 alpha / lambda`` and away from ``q = 0``
        are counted. The value depends on the model, the bright-field mask and the
        upsampling factor, so compare it only between reconstructions that share those.

        Returns
        -------
        float or None
            ``None`` before :meth:`reconstruct`, and after a ``slice_mode="sectioned"`` one,
            which is a reconstruction per depth rather than one model; see
            :attr:`section_residuals`.
        """
        return getattr(self, "_residual", None)

    @property
    def noise_variance(self) -> float | None:
        """The noise variance the last ``prior="ard"`` reconstruction used, per unit weight of
        the normalized bright-field data; ``None`` otherwise."""
        return getattr(self, "_noise_variance", None)

    @property
    def obj_type(self) -> str | None:
        """``"pure_phase"`` or ``"complex"``, as last reconstructed; ``None`` before."""
        return getattr(self, "_obj_type", None)

    @property
    def num_slices(self) -> int | None:
        """Number of slices last reconstructed; ``None`` before :meth:`reconstruct`."""
        return getattr(self, "_num_slices", None)

    @property
    def slice_depths(self) -> NDArray | None:
        """Depth of each reconstructed slice in Angstrom, downstream positive, centred on C10."""
        depths = getattr(self, "_slice_depths", None)
        return None if depths is None else np.asarray(depths, dtype=np.float64)

    @property
    def corrected_amplitude(self):
        """The reconstructed log-amplitude ``ln|t|``, zero-mean, for ``obj_type="complex"``.

        Shaped like :attr:`corrected_bf`; ``None`` for a pure-phase reconstruction.
        """
        return getattr(self, "_corrected_amplitude", None)

    # ------------------------------------------------------------------
    # per-frequency sums -> images, shared by the Fourier and montage FFT routes
    # ------------------------------------------------------------------

    def _fourier_channels(
        self,
        ctx,
        sums,
        num_slices,
        obj_type,
        ridge=EIGENVALUE_FLOOR,
        prior=None,
        noise_variance=None,
    ):
        """``(phase_q, amplitude_q)``, each ``(num_slices, Ny, Nx)``, before the Butterworth filter."""
        kernel = ctx["kernel"]
        bf_weights = ctx["bf_weights"]
        complex_obj = obj_type == "complex"
        self._noise_variance = None
        # the normal equations hold every alias tile of a scan frequency side by side when the
        # reconstruction is upsampled (see `alias_tiles`); each tile has `channels` columns
        alias_factor = ctx.get("alias_factor", 1)
        channels = num_slices * (2 if complex_obj else 1)

        def to_grid(coefficients):
            if alias_factor > 1:
                coefficients = from_alias_tiles(coefficients, alias_factor, channels)
            return coefficients.permute(2, 0, 1)

        if kernel == "obf":
            # the inverse-variance Gram: for one column the solve keeps the unweighted one
            weighted = sums.get("residual_gram", sums["gram"])
            if prior is not None and noise_variance is None:
                noise_variance = noise_variance_from_residual(
                    weighted,
                    sums["numerator"],
                    sums["weighted_energy"],
                    ctx["bf"].num_bf,
                    ctx["residual_mask"],
                    ridge,
                )
            if prior is not None:
                self._noise_variance = float(noise_variance)
            coefficients = solve_normal_equations(
                sums["gram"],
                sums["numerator"],
                bf_weights,
                num_phase=num_slices,
                eigenvalue_floor=ridge,
                prior=prior,
                noise_variance=noise_variance,
                weighted_gram=weighted,
                channels=channels,
            )
            coefficients = to_grid(coefficients)
            phase_q = coefficients[:num_slices]
            amplitude_q = coefficients[num_slices:] if complex_obj else None
        elif kernel == "wiener":
            weighted = sums.get("residual_gram", sums["gram"])
            residual = residual_per_degree_of_freedom(
                weighted, sums["numerator"], sums["weighted_energy"], ctx["bf"].num_bf, ridge
            )
            mask = ctx["residual_mask"].to(residual.device)
            if noise_variance is None:
                noise_variance = float(residual[mask > 0].median())
            self._noise_variance = float(noise_variance)
            numerator = sums["numerator"]
            rings = frequency_rings(ctx["qxa"], ctx["qya"])
            if alias_factor > 1:
                # each tile's columns sit at their own |q| on the upsampled grid
                rings = alias_tiles(rings[..., None], alias_factor)
                rings = rings.repeat_interleave(channels, -1)
            estimate = wiener_estimate(
                weighted,
                numerator,
                noise_variance,
                rings,
                eigenvalue_floor=ridge,
                channels=channels,
            )
            # the columns model the mean-normalized data relative to the peak intensity, and an
            # upsampled grid's inverse transform divides by U^2 more points than the scan holds
            scale = ctx["peak_intensity"] * ctx["upsampling_factor"] ** 2
            factors = column_factors(
                estimate.shape[-1], num_slices, 1.0, estimate.device, channels
            ).to(estimate.dtype)
            coefficients = _to(estimate * factors * scale, numerator.device, numerator.dtype)
            coefficients = to_grid(coefficients)
            phase_q = coefficients[:num_slices]
            amplitude_q = coefficients[num_slices:] if complex_obj else None
        elif kernel == "ssb":
            if complex_obj:
                phase_q = -1.0j * sums["phase"] / bf_weights
                amplitude_q = sums["amplitude"] / 2 / bf_weights
            else:
                phase_q = sums["phase"] / bf_weights
                amplitude_q = None
            phase_q = phase_q[None]
            amplitude_q = None if amplitude_q is None else amplitude_q[None]
        elif kernel == "prlx":
            phase_q = (sums["phase"] * ctx["sign_sin_chi_q"] / bf_weights)[None]
            amplitude_q = None
        else:
            phase_q = (sums["phase"] / bf_weights)[None]
            # the zeroth moment: the plain bright-field sum, which sum_k Gamma- = 0 leaves free
            # of phase for any aberration. Gamma+ -> 2 |psi(k)|^2 as q -> 0, hence the 1/2
            amplitude_q = (sums["amplitude"] / 2 / bf_weights)[None] if complex_obj else None
        return phase_q, amplitude_q

    def _residual_from_sums(self, ctx, sums):
        """The relative least-squares residual of the kernel's own forward model, as a tensor.

        It stays connected to the graph when the sums are, which is what a gradient fit of the
        aberrations, rotation or slice thicknesses differentiates.
        """
        mask = ctx["residual_mask"]
        kernel = ctx["kernel"]
        woa = kernel in self._WOA_KERNELS
        # the weak-object columns model the mean-normalized data as c_k x / |psi(k)|^2, so
        # their residual is weighted by |psi(k)|^2 (see `woa_kernel_terms`)
        energy = sums["weighted_energy"] if woa else sums["energy"]
        total = (energy * mask.to(energy.device)).sum()
        if woa:
            gram = sums.get("residual_gram", sums["gram"])
            explained = explained_energy(gram, sums["numerator"], mask)
        elif kernel == "prlx":
            # one column of unit modulus per bright-field pixel
            count = torch.full_like(energy, float(ctx["bf"].num_bf))
            explained = explained_energy(count, sums["phase"], mask)
        else:
            explained = explained_energy(sums["power"], sums["phase"], mask)
        return 1 - _to(explained, total.device, total.dtype) / total

    def _residual_mask(self, qxa, qya):
        """Frequencies the residual counts: inside the band the data can carry, away from DC."""
        q = torch.hypot(qxa, qya)
        mask = q > 0
        band = self._return_band_limit()
        if band is not None:
            mask &= q <= band
        return mask

    #: kernels that solve the per-frequency normal equations, and so support several slices
    _MULTISLICE_KERNELS = frozenset({"obf", "wiener"})

    #: kernels whose forward model is the weak-object overlap function, which accumulate the
    #: normal equations and so share :meth:`residual_loss`
    _WOA_KERNELS = frozenset({"ssb", "obf", "wiener", "wdd"})

    def _validate_reconstruction_mode(self, kernel: str, obj_type: str, num_slices: int):
        """Raise for a combination of kernel, ``obj_type`` and ``num_slices`` with no meaning."""
        if obj_type not in OBJ_TYPES:
            raise ValueError(f"`obj_type` must be one of {OBJ_TYPES}, got {obj_type!r}")
        if obj_type == "complex" and kernel == "prlx":
            raise ValueError(
                "The parallax kernel cannot separate amplitude from phase. Tilt-corrected bright "
                "field sums the two sidebands, so each frequency records only "
                "cos(chi) Psi_a + sin(chi) Psi_phi (Ma et al., Ultramicroscopy 283, eqs. 18 and "
                "31), which one dataset cannot split. Use 'obf' or 'ssb' for a complex object, or "
                "obj_type='pure_phase' for parallax."
            )
        if int(num_slices) != num_slices or num_slices < 1:
            raise ValueError(f"`num_slices` must be a positive integer, got {num_slices!r}")
        if num_slices > 1 and kernel == "wdd":
            raise ValueError(
                "WDD deconvolves a single multiplicative object, so it cannot solve for slices "
                "jointly. The default, slice_mode='sectioned', reconstructs one plane per depth; "
                "'obf' and 'wiener' solve for slices jointly."
            )
        if num_slices > 1 and kernel not in self._MULTISLICE_KERNELS:
            raise ValueError(
                f"slice_mode='joint' needs the per-frequency least-squares solve, which only "
                f"'obf' and 'wiener' use; got {kernel!r}. SSB normalizes every bright-field "
                "pixel on its own, and parallax and iCoM have no depth-resolved model. The "
                "default, slice_mode='sectioned', reconstructs one plane per depth with any "
                "kernel but 'icom'."
            )

    #: spellings of the two ways to reconstruct several slices
    _SLICE_MODES = {
        "joint": "joint",
        "joint-solve": "joint",
        "sectioned": "sectioned",
        "section": "sectioned",
        "depth-section": "sectioned",
        "depth-sectioning": "sectioned",
        "sweep": "sectioned",
    }

    def _normalize_slice_mode(self, slice_mode) -> str:
        mode = self._SLICE_MODES.get(str(slice_mode).lower())
        if mode is None:
            raise ValueError(f"`slice_mode` must be 'joint' or 'sectioned', got {slice_mode!r}")
        return mode

    @property
    def slice_mode(self) -> str | None:
        """``"joint"`` or ``"sectioned"``, as last reconstructed; ``None`` before."""
        return getattr(self, "_slice_mode", None)

    @property
    def section_residuals(self):
        """Each section's :meth:`residual_loss`, after a ``slice_mode="sectioned"``
        reconstruction; ``None`` otherwise.

        Every section is a single plane fitted to all the data, so this is how well a plane at
        each depth explains it, smallest near the specimen's mean plane.
        """
        return getattr(self, "_section_residuals", None)

    def _reconstruct_sections(self, reconstruct, arguments):
        """One single-plane reconstruction per slice depth, stacked: ``slice_mode="sectioned"``.

        ``arguments`` are the caller's :meth:`reconstruct` arguments. The slices are placed as
        for the joint solve, centred on the plane ``C10`` refers to, and slice ``n`` is
        reconstructed with the probe at ``C10 + z_n``.
        """
        kernel = self._normalize_kernel_name(arguments["deconvolution_kernel"])
        if kernel == "icom":
            raise ValueError(
                "iCoM does not depend on the aberrations, so every section of it would be the "
                "same image. Use another kernel to section in depth."
            )
        state = self.hyperparameter_state
        if arguments.get("use_initial_state"):
            aberrations = dict(state.initial_aberrations)
            rotation = state.initial_rotation_angle
            rotation = 0.0 if rotation is None else rotation
        else:
            aberrations = state.current_aberrations(arguments.get("override_aberration_coefs"))
            rotation = state.current_rotation_angle(arguments.get("override_rotation_angle"))
        thicknesses = state.current_slice_thicknesses(arguments.get("slice_thicknesses"))
        depths = slice_depths(int(arguments["num_slices"]), thicknesses, device=self.device)
        reference = aberrations.get("C10", 0.0)

        phases, amplitudes, residuals = [], [], []
        for depth in depths:
            offset = depth if depth.requires_grad else float(depth)
            reconstruct(
                **dict(
                    arguments,
                    num_slices=1,
                    slice_thicknesses=None,
                    slice_mode="joint",
                    override_aberration_coefs=dict(aberrations, C10=reference + offset),
                    override_rotation_angle=rotation,
                    use_initial_state=False,
                )
            )
            phases.append(self._corrected_bf)
            amplitudes.append(self._corrected_amplitude)
            residuals.append(self._residual)

        self._corrected_bf = torch.stack(phases)
        self._corrected_amplitude = None if amplitudes[0] is None else torch.stack(amplitudes)
        self._num_slices = len(phases)
        self._slice_depths = [float(z) for z in depths.detach().cpu()]
        self._slice_mode = "sectioned"
        self._section_residuals = residuals
        # one reconstruction per depth, not one model of the data
        self._residual = None
        return self

    def _check_loss_for_slice_mode(self, loss, reconstruct_kwargs):
        """A sectioned stack has no residual of its own, so it cannot be fitted by one."""
        sectioned = (
            int(reconstruct_kwargs.get("num_slices", 1)) > 1
            and self._normalize_slice_mode(reconstruct_kwargs.get("slice_mode", "sectioned"))
            == "sectioned"
        )
        if sectioned and loss == "residual":
            raise ValueError(
                "A sectioned stack is one reconstruction per depth rather than one model of the "
                "data, so it has no residual to minimize. Fit with num_slices=1, with "
                "slice_mode='joint', or with loss='rms_gradient'."
            )

    def _validate_prior(self, kernel: str, prior, noise_variance):
        """Raise for a ``prior`` the kernel cannot use, or a ``noise_variance`` that is not one."""
        if noise_variance is not None and not float(noise_variance) > 0:
            raise ValueError(f"`noise_variance` must be positive, got {noise_variance!r}")
        if prior is None:
            if noise_variance is not None and kernel != "wiener":
                raise ValueError(
                    "`noise_variance` is only used with `prior='ard'` or the 'wiener' kernel."
                )
            return
        if prior != "ard":
            raise ValueError(f"`prior` must be None or 'ard', got {prior!r}")
        if kernel != "obf":
            raise ValueError(
                f"prior='ard' acts on the per-frequency normal equations of 'obf'; got {kernel!r}. "
                "The 'wiener' kernel has its own prior, the data's radial spectrum."
            )

    def rms_gradient_loss(self):
        """
        Negated RMS gradient of the reconstruction, per Angstrom -- a sharpness objective.

        The classic autofocus metric: a correctly deconvolved image has sharp edges and a
        large gradient, a mis-set aberration blurs them. Negated so that, like
        :meth:`residual_loss`, it is minimized. Over several slices it averages the gradient of
        all of them. It is insensitive to how the canvas is sized, so it needs no pinning the
        way a patch fit does.

        It rewards *amplitude*, not only sharpness, since it is not normalized by the image's
        own spread. Aberrations and rotation barely change the overall scale, so this is safe
        for those searches, but a hyperparameter that can inflate the object will game it. Slice
        thicknesses are one: an assumed spacing ``s`` times too small scales the difference
        between the slices by ``1 / s``. Fit thicknesses with :meth:`residual_loss` instead.

        Returns
        -------
        float or None
            ``None`` before :meth:`reconstruct`, mirroring :meth:`residual_loss`.
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
        value = -torch.sqrt((grad_rows.square() + grad_cols.square()).mean())
        # a differentiable reconstruction keeps the graph, so a gradient fit can use this too
        return value if value.requires_grad else float(value)

    #: objectives the hyperparameter searches accept by name, all minimized
    _LOSS_FUNCTIONS = {
        "residual": "residual_loss",
        "rms_gradient": "rms_gradient_loss",
    }

    def _return_loss_value(self, loss) -> float:
        """Evaluate a search objective on the current reconstruction."""
        if callable(loss):
            return float(loss(self))
        if loss == "variance":
            raise ValueError(
                "loss='variance' has been removed. Use loss='residual', which measures the same "
                "spread of the parallax-corrected images for 'prlx' and is defined for every "
                "kernel."
            )
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

        The phase for ``obj_type="pure_phase"``, and the complex transmission
        ``exp(a + i phi)`` for ``obj_type="complex"``, with ``a`` the log-amplitude. Shaped
        ``(Ny, Nx)`` for one slice and ``(num_slices, Ny, Nx)`` for several.

        Mirrors ``corrected_bf`` rather than raising: ``AutoSerialize._recursive_load``
        walks ``dir(obj)`` and evaluates every property, so a ``to_numpy(None)`` here made
        save/load fail for a instance that had not been reconstructed yet.
        """
        corrected_bf = self.corrected_bf
        if corrected_bf is None:
            return None
        phase = to_numpy(corrected_bf)
        amplitude = self.corrected_amplitude
        if amplitude is None:
            return phase
        return np.exp(to_numpy(amplitude) + 1j * phase).astype(np.complex64)

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
            "wiener": "wiener",
            "wdd": "wdd",
            "wigner-distribution-deconvolution": "wdd",
            "prlx": "prlx",
            "parallax": "prlx",
            "tcbf": "prlx",
            "tilt-corrected-bright-field": "prlx",
            "icom": "icom",
            "center-of-mass": "icom",
        }

        if kernel in ("mf", "matched-filter"):
            raise ValueError(
                "The matched-filter kernel has been removed. Its transfer, s / (s + eps max s), "
                "sat at 0.7-0.9 rather than one at the default regularization, so its values were "
                "not the calibrated radians it was kept for. Use 'obf'."
            )
        if kernel not in aliases:
            raise ValueError(f"Unknown deconvolution kernel '{kernel}'")

        return aliases[kernel]

    #: kernels carrying a ``gamma`` factor, which is already zero past ``2 alpha / lambda``
    _BAND_LIMITED_KERNELS = frozenset({"ssb", "obf", "wiener", "wdd"})

    def _return_band_limit(self) -> float | None:
        """``2 alpha / lambda``, the largest object frequency the data can carry.

        Every kernel is a multiplier on ``G(k, q)``, which is an overlap of two probe disks
        displaced by ``q`` and is empty once that displacement exceeds twice the probe's own
        extent. An empirical probe has no ``semiangle_cutoff``, so its extent is read off the
        bright-field mask instead.
        """
        if self.semiangle_cutoff is not None:
            return 2 * self.semiangle_cutoff * 1e-3 / self.wavelength

        bf_mask = self.bf_mask
        if bf_mask is None or not bool(bf_mask.any()):
            return None
        kxa, kya = spatial_frequencies(self.gpts, self.sampling, device=self.device)
        return 2 * float(torch.hypot(kxa[bf_mask], kya[bf_mask]).max())

    def _resolve_q_lowpass(self, q_lowpass, deconvolution_kernel, band_limit=True) -> float | None:
        """Clamp ``q_lowpass`` to the band limit for kernels that do not impose it themselves.

        ``ssb``, ``obf``, ``wiener`` and ``wdd`` multiply by ``gamma``, so they are already zero
        beyond ``2 alpha / lambda``; for an array probe, whose ``gamma`` holds round-off there,
        through :data:`~quantem.diffractive_imaging.direct_ptycho_utils.OVERLAP_FLOOR`.
        ``prlx`` is a pure shift of unit magnitude at every ``q`` and
        ``icom`` only rolls off as ``1/|q|``, so both would otherwise pass whatever lies past
        it.

        Upsampling tiles the bright-field spectrum, which places the aliased replicas at those
        frequencies. A ``gamma`` factor uses ``k`` to distinguish a real high frequency from
        its alias, which is what gives resolution beyond the scan Nyquist. A pure shift
        cannot, so for ``prlx`` the replicas are only alias.
        """
        if not band_limit or deconvolution_kernel in self._BAND_LIMITED_KERNELS:
            return q_lowpass
        cutoff = self._return_band_limit()
        if cutoff is None:
            return q_lowpass
        return cutoff if q_lowpass is None else min(q_lowpass, cutoff)

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
        Show the reconstructed object and its Fourier transform.

        One row per slice. A complex reconstruction shows the phase and the log-amplitude,
        each read from the stored arrays rather than from ``np.angle``, which would wrap.

        Parameters
        ----------
        return_fig : bool, optional
            If True, return ``(fig, axs)``.
        show_obj_fft : bool, optional
            Also show the magnitude of each image's Fourier transform.
        apply_hanning_window : bool, optional
            Window the images before transforming them.
        **kwargs
            Additional arguments passed to ``show_2d``, e.g. ``cbar`` or ``fft_norm``.
        """
        if self.corrected_bf is None:
            raise RuntimeError("Run reconstruct() before visualize().")

        phase = to_numpy(self.corrected_bf)
        amplitude = self.corrected_amplitude
        amplitude = None if amplitude is None else to_numpy(amplitude)
        if phase.ndim == 2:
            phase = phase[None]
            amplitude = None if amplitude is None else amplitude[None]

        obj_sampling = self._obj_sampling
        obj_scalebar = {"sampling": obj_sampling[1], "units": "Å"}
        fft_sampling = 1 / (obj_sampling[1] * phase.shape[-1])
        fft_scalebar = {"sampling": fft_sampling, "units": r"$\mathrm{A^{-1}}$"}
        window = (
            np.hanning(phase.shape[-2])[:, None] * np.hanning(phase.shape[-1])[None, :]
            if apply_hanning_window
            else 1.0
        )

        def fourier(image):
            return np.fft.fftshift(np.abs(np.fft.fft2(image * window)))

        depths = self.slice_depths
        rows, titles, scalebars = [], [], []
        for index in range(phase.shape[0]):
            label = "" if phase.shape[0] == 1 else f", z = {depths[index]:+.1f} Å"
            channels = [("Object phase", phase[index])]
            if amplitude is not None:
                channels.append(("Object log-amplitude", amplitude[index]))
            row, title, scalebar = [], [], []
            for name, image in channels:
                row.append(image)
                title.append(name + label)
                scalebar.append(obj_scalebar)
                if show_obj_fft:
                    row.append(fourier(image))
                    title.append(name + " FFT" + label)
                    scalebar.append(fft_scalebar)
            rows.append(row)
            titles.append(title)
            scalebars.append(scalebar)

        if len(rows) == 1 and len(rows[0]) == 1:
            fig, axs = show_2d(rows[0][0], title=titles[0][0], scalebar=scalebars[0][0], **kwargs)
        elif len(rows) == 1:
            fig, axs = show_2d(rows[0], title=titles[0], scalebar=scalebars[0], **kwargs)
        else:
            fig, axs = show_2d(rows, title=titles, scalebar=scalebars, **kwargs)

        if show_obj_fft:
            aspect = phase.shape[-1] / phase.shape[-2]
            for ax, title in zip(np.ravel(axs), [t for row in titles for t in row]):
                if title.split(",")[0].endswith("FFT"):
                    ax.set_aspect(aspect)

        if return_fig:
            return fig, axs
        return None

    # ------------------------------------------------------------------
    # hyperparameter optimization
    # ------------------------------------------------------------------

    @staticmethod
    def _with_slice_thicknesses(reconstruct_kwargs: dict, thicknesses) -> dict:
        """``reconstruct_kwargs`` with ``slice_thicknesses`` set, when a search supplies one."""
        if thicknesses is None:
            return reconstruct_kwargs
        return {**reconstruct_kwargs, "slice_thicknesses": thicknesses}

    @staticmethod
    def _resolved_thicknesses(best, requested):
        """What a search writes back: the best searched gap, else a fixed one, else ``None``."""
        if best is not None:
            return [float(best)]
        if requested is not None and not isinstance(requested, OptimizationParameter):
            return [float(t) for t in np.atleast_1d(requested)]
        return None

    def optimize_hyperparameters(
        self,
        aberration_coefs: dict[str, float | OptimizationParameter] | None = None,
        rotation_angle: float | OptimizationParameter | None = None,
        slice_thicknesses: float | OptimizationParameter | None = None,
        n_trials=50,
        sampler=None,
        loss="residual",
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
        slice_thicknesses : float|OptimizationParameter, optional
            Fixed gap between slices or an optimization range, in Angstrom, for a
            reconstruction with ``num_slices > 1`` and ``slice_mode="joint"`` passed in
            ``reconstruct_kwargs``. One value
            is used for every gap. Fit ``C10`` with one slice first: the residual of several
            slices barely depends on the reference plane.
        n_trials : int
            Number of Optuna trials.
        sampler : optuna.samplers.BaseSampler, optional
            Custom Optuna sampler.
        loss : {"residual", "rms_gradient"} or callable
            Objective to minimize. ``"residual"`` is :meth:`residual_loss`, the fraction of
            the data the kernel's forward model leaves unexplained. ``"rms_gradient"`` is
            :meth:`rms_gradient_loss`, an image-sharpness objective. A callable is passed the
            reconstruction and must return a float to minimize.
        verbose : bool, optional
            Report the search and show Optuna's progress bar. Defaults to :attr:`verbose`.
        **reconstruct_kwargs :
            Extra arguments passed to reconstruct().
        """

        self._check_loss_for_slice_mode(loss, reconstruct_kwargs)
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
        if isinstance(slice_thicknesses, OptimizationParameter):
            state.optimized_keys.add("slice_thicknesses")

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

            if isinstance(slice_thicknesses, OptimizationParameter):
                thicknesses = trial.suggest_float(
                    "slice_thicknesses",
                    slice_thicknesses.low,
                    slice_thicknesses.high,
                    log=slice_thicknesses.log,
                )
            else:
                thicknesses = slice_thicknesses

            self.reconstruct(
                override_aberration_coefs=trial_aberrations,
                override_rotation_angle=rot,
                verbose=False,
                **self._with_slice_thicknesses(reconstruct_kwargs, thicknesses),
            )
            return self._return_loss_value(loss)

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=bool(verbose))

        # Write back optimized results
        best = study.best_params.copy()
        state.optimized_rotation_angle = best.pop("rotation_angle", None)
        best_thickness = best.pop("slice_thicknesses", None)

        if state.optimized_rotation_angle is None and rotation_angle is not None:
            state.optimized_rotation_angle = rotation_angle  # ty:ignore[invalid-assignment]

        state.optimized_aberrations = best
        state.optimized_aberrations = state.current_aberrations(fixed_override_aberrations)
        state.optimized_slice_thicknesses = self._resolved_thicknesses(
            best_thickness, slice_thicknesses
        )
        state.study = study

        if verbose:
            print("Optimized state:\n\n", self.hyperparameter_state)

        self.reconstruct(verbose=False, **reconstruct_kwargs)
        return self

    def grid_search_hyperparameters(
        self,
        aberration_coefs: dict[str, float | OptimizationParameter] | None = None,
        rotation_angle: float | OptimizationParameter | None = None,
        slice_thicknesses: float | OptimizationParameter | None = None,
        loss="residual",
        verbose=None,
        **reconstruct_kwargs,
    ):
        """
        Exhaustive search over a grid of hyperparameter values.

        Parameters
        ----------
        loss : {"residual", "rms_gradient"} or callable
            Objective to minimize; see :meth:`optimize_hyperparameters`.
        slice_thicknesses : float|OptimizationParameter, optional
            Fixed gap between slices or a grid of them; see :meth:`optimize_hyperparameters`.
        """
        self._check_loss_for_slice_mode(loss, reconstruct_kwargs)
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
        if isinstance(slice_thicknesses, OptimizationParameter):
            param_grid["slice_thicknesses"] = slice_thicknesses.grid_values()
            state.optimized_keys.add("slice_thicknesses")

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
                {
                    k: v
                    for k, v in trial_params.items()
                    if k not in ("rotation_angle", "slice_thicknesses")
                }
            )
            thicknesses = trial_params.get("slice_thicknesses", slice_thicknesses)

            if optimize_rotation:
                rot = trial_params["rotation_angle"]
            else:
                rot = rotation_angle

            self.reconstruct(
                override_aberration_coefs=trial_aberrations,
                override_rotation_angle=rot,
                verbose=False,
                **self._with_slice_thicknesses(reconstruct_kwargs, thicknesses),
            )

            loss_value = self._return_loss_value(loss)
            results.append((trial_params, loss_value))

            if loss_value < best_loss:
                best_loss = loss_value
                best_params = trial_params

        self._grid_search_results = results

        # Write back best optimized values
        best_thickness = None
        if best_params is not None:
            best_params = best_params.copy()
            state.optimized_rotation_angle = best_params.pop("rotation_angle", None)
            best_thickness = best_params.pop("slice_thicknesses", None)
            state.optimized_aberrations = best_params
        state.optimized_slice_thicknesses = self._resolved_thicknesses(
            best_thickness, slice_thicknesses
        )

        if state.optimized_rotation_angle is None and rotation_angle is not None:
            state.optimized_rotation_angle = rotation_angle  # ty:ignore[invalid-assignment]

        state.optimized_aberrations = state.current_aberrations(fixed_override_aberrations)

        if verbose:
            print("Optimized state:\n\n", self.hyperparameter_state)

        # Final reconstruction using merged state
        self.reconstruct(verbose=False, **reconstruct_kwargs)
        return self
