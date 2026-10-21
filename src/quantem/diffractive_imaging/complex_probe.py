import math
from collections import defaultdict
from typing import Mapping, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core.utils.utils import electron_wavelength_angstrom

# fmt: off
POLAR_ALIASES = {
    "defocus": "C10",
    "astigmatism": "C12",
    "astigmatism_angle": "phi12",
    "coma": "C21",
    "coma_angle": "phi21",
    "Cs": "C30",
    "C5": "C50",
}

POLAR_SYMBOLS = (
    "C10", "C12", "phi12",
    "C21", "phi21", "C23", "phi23",
    "C30", "C32", "phi32", "C34", "phi34",
    "C41", "phi41", "C43", "phi43", "C45", "phi45",
    "C50", "C52", "phi52", "C54", "phi54", "C56", "phi56",
)
# fmt: on


def hard_aperture(alpha: torch.Tensor, semiangle_cutoff: float) -> torch.Tensor:
    """
    Calculates circular aperture with hard edges.

    Parameters
    ----------
    alpha: torch.Tensor
        Radial component of the polar frequencies [rad].
    semiangle_cutoff: float
        The semiangle cutoff describes the sharp Fourier space cutoff due to the objective aperture [mrad].

    Returns
    -------
    aperture: torch.Tensor
        circular aperture tensor with hard edges.
    """
    semiangle_rad = semiangle_cutoff * 1e-3
    return (alpha <= semiangle_rad).to(torch.float32)


def soft_aperture(
    alpha: torch.Tensor,
    phi: torch.Tensor,
    semiangle_cutoff: float,
    angular_sampling: Tuple[float, float],
) -> torch.Tensor:
    """
    Calculates circular aperture with soft edges.

    Parameters
    ----------
    alpha: torch.Tensor
        Radial component of the polar frequencies [rad].
    phi: torch.Tensor
        Angular component of the polar frequencies.
    semiangle_cutoff: float
        The semiangle cutoff describes the sharp Fourier space cutoff due to the objective aperture [mrad].
    angular_sampling: Tuple[float,float]
        Sampling of the polar frequencies grid in mrad.

    Returns
    -------
    aperture: torch.Tensor
        circular aperture tensor with soft edges.
    """
    semiangle_rad = semiangle_cutoff * 1e-3
    denominator = torch.sqrt(
        (torch.cos(phi) * angular_sampling[0] * 1e-3).square()
        + (torch.sin(phi) * angular_sampling[1] * 1e-3).square()
    )
    array = torch.clip(
        (semiangle_rad - alpha) / denominator + 0.5,
        0,
        1,
    )
    return array.to(torch.float32)


def aperture(
    alpha: torch.Tensor,
    phi: torch.Tensor,
    semiangle_cutoff: float,
    angular_sampling: Tuple[float, float],
    soft_edges: bool = True,
    vacuum_probe_intensity: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Calculates circular aperture.

    Parameters
    ----------
    alpha: torch.Tensor
        Radial component of the polar frequencies [rad].
    phi: torch.Tensor
        Angular component of the polar frequencies.
    semiangle_cutoff: float
        The semiangle cutoff describes the sharp Fourier space cutoff due to the objective aperture [mrad].
    angular_sampling: Tuple[float,float]
        Sampling of the polar frequencies grid in mrad.
    soft_edges: bool
        If True, uses soft edges.
    vacuum_probe_intensity: torch.Tensor
        If not None, uses sqrt of vacuum_probe_intensity as aperture. Assumed to be corner-centered.

    Returns
    -------
    aperture: torch.Tensor
        aperture tensor.
    """
    if vacuum_probe_intensity is not None:
        return torch.sqrt(vacuum_probe_intensity).to(torch.float32)
    if soft_edges:
        return soft_aperture(alpha, phi, semiangle_cutoff, angular_sampling)
    else:
        return hard_aperture(alpha, semiangle_cutoff)


def standardize_aberration_coefs(aberration_coefs: Mapping[str, float]) -> dict[str, torch.Tensor]:
    """
    Convert user-supplied aberration coefficient dictionary into canonical
    polar-aberration symbols (C_nm, phi_nm), resolving aliases and conventions.

    Parameters
    ----------
    coefs : dict
        May contain canonical symbols (e.g. 'C10', 'phi12') or aliases
        (e.g. 'defocus', 'astigmatism', 'coma', 'Cs').

    Returns
    -------
    dict
        Dictionary with canonical polar keys only.
    """
    out = {}

    for key, val in aberration_coefs.items():
        canonical = POLAR_ALIASES.get(key, key)

        if key == "defocus":
            out["C10"] = -float(val)

        elif canonical in POLAR_SYMBOLS:
            out[canonical] = float(val)

        else:
            raise KeyError(
                f"Unknown aberration key '{key}'. "
                f"Expected one of: {', '.join(POLAR_SYMBOLS + tuple(POLAR_ALIASES))}"
            )

    return {k: torch.tensor(v, dtype=torch.float32) for k, v in out.items()}


def aberration_surface(
    alpha: torch.Tensor,
    phi: torch.Tensor,
    wavelength: float,
    aberration_coefs: Mapping[str, float | torch.Tensor],
):
    """ """

    pi = math.pi
    alpha2 = alpha.square()
    chi = torch.zeros_like(alpha)

    # coefs = standardize_aberration_coefs(aberration_coefs)
    coefs = aberration_coefs

    def get(name, default=0.0):
        val = coefs.get(name, default)
        return val

    if any(k in coefs for k in ("C10", "C12", "phi12")):
        chi = chi + 0.5 * alpha2 * (get("C10") + get("C12") * torch.cos(2 * (phi - get("phi12"))))

    if any(k in coefs for k in ("C21", "phi21", "C23", "phi23")):
        chi = chi + (1 / 3) * alpha2 * alpha * (
            get("C21") * torch.cos(phi - get("phi21"))
            + get("C23") * torch.cos(3 * (phi - get("phi23")))
        )

    if any(k in coefs for k in ("C30", "C32", "phi32", "C34", "phi34")):
        chi = chi + (1 / 4) * alpha2.square() * (
            get("C30")
            + get("C32") * torch.cos(2 * (phi - get("phi32")))
            + get("C34") * torch.cos(4 * (phi - get("phi34")))
        )

    if any(k in coefs for k in ("C41", "phi41", "C43", "phi43", "C45", "phi45")):
        chi = chi + (1 / 5) * alpha2.square() * alpha * (
            get("C41") * torch.cos(phi - get("phi41"))
            + get("C43") * torch.cos(3 * (phi - get("phi43")))
            + get("C45") * torch.cos(5 * (phi - get("phi45")))
        )

    if any(k in coefs for k in ("C50", "C52", "phi52", "C54", "phi54", "C56", "phi56")):
        chi = chi + (1 / 6) * alpha2 * alpha2 * alpha2 * (
            get("C50")
            + get("C52") * torch.cos(2 * (phi - get("phi52")))
            + get("C54") * torch.cos(4 * (phi - get("phi54")))
            + get("C56") * torch.cos(6 * (phi - get("phi56")))
        )

    chi = 2 * pi / wavelength * chi
    return chi


def aberration_surface_polar_gradients(
    alpha: torch.Tensor,
    phi: torch.Tensor,
    aberration_coefs: Mapping[str, float | torch.Tensor],
):
    """ """

    pi = math.pi
    alpha2 = alpha.square()
    dchi_dk = torch.zeros_like(alpha)
    dchi_dphi = torch.zeros_like(alpha)

    # coefs = standardize_aberration_coefs(aberration_coefs)
    coefs = aberration_coefs

    def get(name, default=0.0):
        val = coefs.get(name, default)
        return val

    if any(k in coefs for k in ("C10", "C12", "phi12")):
        dchi_dk = dchi_dk + alpha * (get("C10") + get("C12") * torch.cos(2 * (phi - get("phi12"))))
        dchi_dphi = dchi_dphi - 1 / 2.0 * alpha * (
            2.0 * get("C12") * torch.sin(2 * (phi - get("phi12")))
        )

    if any(k in coefs for k in ("C21", "phi21", "C23", "phi23")):
        dchi_dk = dchi_dk + alpha2 * (
            get("C21") * torch.cos(1 * (phi - get("phi21")))
            + get("C23") * torch.cos(3 * (phi - get("phi23")))
        )
        dchi_dphi = dchi_dphi - 1 / 3.0 * alpha2 * (
            1.0 * get("C21") * torch.sin(1 * (phi - get("phi21")))
            + 3.0 * get("C23") * torch.sin(3 * (phi - get("phi23")))
        )

    if any(k in coefs for k in ("C30", "C32", "phi32", "C34", "phi34")):
        dchi_dk = dchi_dk + alpha2 * alpha * (
            get("C30")
            + get("C32") * torch.cos(2 * (phi - get("phi32")))
            + get("C34") * torch.cos(4 * (phi - get("phi34")))
        )
        dchi_dphi = dchi_dphi - 1 / 4.0 * alpha2 * alpha * (
            2.0 * get("C32") * torch.sin(2 * (phi - get("phi32")))
            + 4.0 * get("C34") * torch.sin(4 * (phi - get("phi34")))
        )

    if any(k in coefs for k in ("C41", "phi41", "C43", "phi43", "C45", "phi45")):
        dchi_dk = dchi_dk + alpha2 * alpha2 * (
            get("C41") * torch.cos(1 * (phi - get("phi41")))
            + get("C43") * torch.cos(3 * (phi - get("phi43")))
            + get("C45") * torch.cos(5 * (phi - get("phi45")))
        )
        dchi_dphi = dchi_dphi - 1 / 5.0 * alpha2 * alpha2 * (
            1.0 * get("C41") * torch.sin(1 * (phi - get("phi41")))
            + 3.0 * get("C43") * torch.sin(3 * (phi - get("phi43")))
            + 5.0 * get("C45") * torch.sin(5 * (phi - get("phi45")))
        )

    if any(k in coefs for k in ("C50", "C52", "phi52", "C54", "phi54", "C56", "phi56")):
        dchi_dk = dchi_dk + alpha2 * alpha2 * alpha * (
            get("C50")
            + get("C52") * torch.cos(2 * (phi - get("phi52")))
            + get("C54") * torch.cos(4 * (phi - get("phi54")))
            + get("C56") * torch.cos(6 * (phi - get("phi56")))
        )
        dchi_dphi = dchi_dphi - 1 / 6.0 * alpha2 * alpha2 * alpha * (
            2.0 * get("C52") * torch.sin(2 * (phi - get("phi52")))
            + 4.0 * get("C54") * torch.sin(4 * (phi - get("phi54")))
            + 6.0 * get("C56") * torch.sin(6 * (phi - get("phi56")))
        )

    scale = 2 * pi
    return scale * dchi_dk, scale * dchi_dphi


def aberration_surface_cartesian_gradients(
    alpha: torch.Tensor,
    phi: torch.Tensor,
    aberration_coefs: Mapping[str, float | torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute dchi/dx and dchi/dy from the polar derivatives.
    """
    dchi_dk, dchi_dphi = aberration_surface_polar_gradients(alpha, phi, aberration_coefs)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    dchi_dx = cos_phi * dchi_dk - sin_phi * dchi_dphi
    dchi_dy = sin_phi * dchi_dk + cos_phi * dchi_dphi

    return dchi_dx, dchi_dy


class FourierProbe:
    """``psi(k)``, the probe in the detector plane, sampled wherever ``gamma_factor`` needs it.

    Two flavours behind one interface:

    - :meth:`from_aberrations` -- an aperture times ``exp(-i chi)``, evaluated in closed form
      at any ``k``. What every electron entry point uses, and identical to calling
      :func:`evaluate_probe` directly.
    - :meth:`from_array` -- a measured or independently reconstructed complex array, sampled
      on its own reciprocal grid. For a probe that no aperture and low-order aberrations
      describe: a zone plate with a central stop, a speckled illumination, an X-ray optic.

    The difference that matters is that an analytic probe can be *evaluated* at the arbitrary
    ``k -/+ q`` that the overlap function asks for, while an array can only be *sampled*. See
    :meth:`at` for what that costs.

    An array probe can also be a mixed state: ``(M, Ny, Nx)`` incoherent modes, as a
    mixed-state ptychographic reconstruction returns them. It has no single ``psi(k)``, so
    :meth:`at` refuses it and :func:`overlap_terms` sums the overlap function over
    :meth:`coherent_modes` instead. Its :attr:`array` is ``sqrt(sum_m |psi_m|^2)``, the
    amplitude whose square the detector sees, which is all the intensity weights need.

    Partial coherence can also be described by two parameters rather than by modes:
    ``focal_spread`` and ``source_size``, the rms spreads of defocus and of probe position in
    Angstrom (see :meth:`with_partial_coherence`). Neither changes ``|psi(k)|``, so only the
    overlap function sees them.
    """

    def __init__(
        self,
        wavelength: float,
        *,
        array: torch.Tensor | None = None,
        reciprocal_sampling: Tuple[float, float] | None = None,
        semiangle_cutoff: float | None = None,
        aberration_coefs: Mapping[str, float | torch.Tensor] | None = None,
        angular_sampling: Tuple[float, float] | None = None,
        soft_edges: bool = True,
        interpolation: str = "exact",
        modes: torch.Tensor | None = None,
        focal_spread: float = 0.0,
        source_size: float = 0.0,
    ):
        self.wavelength = float(wavelength)
        # complex64, the precision the reconstructions accumulate in and the widest MPS supports
        if modes is not None:
            modes = modes.to(torch.complex64)
            # the amplitude the detector sees; the phase has no meaning for a mixed state
            array = modes.abs().square().sum(0).sqrt().to(modes.dtype)
        elif array is not None:
            array = array.to(torch.complex64)
        self.array = array
        self.modes = modes
        self.reciprocal_sampling = reciprocal_sampling
        self.semiangle_cutoff = semiangle_cutoff
        self.aberration_coefs = aberration_coefs or {}
        self.angular_sampling = angular_sampling
        self.soft_edges = soft_edges
        self.interpolation = interpolation
        self.focal_spread = focal_spread
        self.source_size = source_size

    @classmethod
    def from_aberrations(
        cls,
        wavelength: float,
        semiangle_cutoff: float,
        angular_sampling: Tuple[float, float],
        aberration_coefs: Mapping[str, float | torch.Tensor] = {},
        soft_edges: bool = True,
    ) -> "FourierProbe":
        """The analytic probe: a soft or hard aperture times ``exp(-i chi(k))``."""
        return cls(
            wavelength,
            semiangle_cutoff=semiangle_cutoff,
            angular_sampling=angular_sampling,
            aberration_coefs=aberration_coefs,
            soft_edges=soft_edges,
        )

    @classmethod
    def from_array(
        cls,
        array: torch.Tensor | NDArray,
        reciprocal_sampling: Tuple[float, float],
        wavelength: float,
        normalize: bool = True,
        interpolation: str = "exact",
    ) -> "FourierProbe":
        """An empirical complex probe on the detector's own reciprocal grid.

        Parameters
        ----------
        array : complex array
            ``psi(k)``, corner-centered to match :func:`spatial_frequencies`. A probe
            saved with its maximum at the array centre needs ``ifftshift`` first. A 3D array
            ``(M, Ny, Nx)`` is a mixed state of ``M`` incoherent modes, each the Fourier
            transform of one real-space mode of a mixed-state reconstruction. It is stored
            as complex64 whatever its input precision.
        reciprocal_sampling : tuple of float
            ``dq`` per axis, in inverse Angstrom. Its reciprocal, ``1 / dq`` per axis, is the
            probe's real-space field of view, which :meth:`at` needs.
        normalize : bool
            Scale to unit total intensity, summed over modes, matching
            :func:`fourier_space_probe`. The
            deconvolution kernels are scale invariant but the bright-field weight that
            normalizes the finished object is not, so leaving this off changes the object's
            absolute scale.
        interpolation : {"exact", "bilinear"}
            What to do when ``k -/+ q`` falls between grid points -- see :meth:`at`.
        """
        tensor = array if isinstance(array, torch.Tensor) else torch.as_tensor(array)
        if not torch.is_complex(tensor):
            raise ValueError(
                f"`array` must be a complex probe psi(k), got dtype {tensor.dtype}. Pass "
                "`amplitude * exp(1j * phase)`, or a real array cast to complex if the "
                "probe really has no phase structure."
            )
        if tensor.ndim not in (2, 3):
            raise ValueError(
                "`array` must be 2D (Ny, Nx), or 3D (M, Ny, Nx) for M incoherent modes, got "
                f"shape {tuple(tensor.shape)}"
            )
        if normalize:
            tensor = tensor / tensor.abs().square().sum().sqrt()

        sampling = tuple(float(s) for s in reciprocal_sampling)
        if tensor.ndim == 3:
            return cls(
                wavelength, modes=tensor, reciprocal_sampling=sampling, interpolation=interpolation
            )
        return cls(
            wavelength, array=tensor, reciprocal_sampling=sampling, interpolation=interpolation
        )

    @property
    def num_modes(self) -> int:
        """How many incoherent modes the probe has: one unless it is a mixed state."""
        return 1 if self.modes is None else int(self.modes.shape[0])

    def coherent_modes(self) -> list["FourierProbe"]:
        """The incoherent modes, each a coherent probe; ``[self]`` for a coherent probe.

        The modes leave out :attr:`focal_spread` and :attr:`source_size`, which
        :func:`overlap_terms` applies once to their sum.
        """
        if self.modes is None:
            return [self]
        return [
            FourierProbe(
                self.wavelength,
                array=mode,
                reciprocal_sampling=self.reciprocal_sampling,
                interpolation=self.interpolation,
            )
            for mode in self.modes
        ]

    def with_partial_coherence(self, focal_spread=0.0, source_size=0.0) -> "FourierProbe":
        """The same probe with a parametric partial-coherence envelope on its overlap function.

        ``focal_spread`` is the rms spread of defocus, from the energy spread and lens current
        instabilities, and ``source_size`` the rms spread of the probe position per axis, from
        the finite source, both Gaussian and in Angstrom. Averaging the overlap function over
        them has a closed form on each half (see :func:`partial_coherence_envelope`). It
        multiplies the mode sum of a mixed state as well, so modes that already describe the
        incoherence should not be given these too. Either may be a tensor that requires grad.
        """
        copy = FourierProbe.__new__(FourierProbe)
        copy.__dict__.update(self.__dict__)
        copy.focal_spread = focal_spread
        copy.source_size = source_size
        return copy

    @property
    def is_partially_coherent(self) -> bool:
        """Whether the overlap function carries a focal-spread or source-size envelope."""
        return _nonzero(self.focal_spread) or _nonzero(self.source_size)

    def read(self, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
        """``psi(k)``, or ``sqrt(sum_m |psi_m(k)|^2)`` for a mixed state, at given frequencies.

        Each mode is read with its own sampling, an exact gather where the frequencies are on
        its grid, and bilinearly where they fall between grid points, as they do under a
        rotation that does not map the grid onto itself.
        """

        def sample(mode):
            try:
                return mode.at(kx, ky)
            except ValueError:
                return mode.with_interpolation("bilinear").at(kx, ky)

        if self.modes is None:
            return sample(self)
        power = sum(sample(mode).abs().square() for mode in self.coherent_modes())
        return power.sqrt().to(self.modes.dtype)

    def with_interpolation(self, interpolation: str) -> "FourierProbe":
        """The same array probe, sampled with ``interpolation``."""
        if self.array is None:
            return self
        if self.modes is not None:
            return FourierProbe(
                self.wavelength,
                modes=self.modes,
                reciprocal_sampling=self.reciprocal_sampling,
                interpolation=interpolation,
                focal_spread=self.focal_spread,
                source_size=self.source_size,
            )
        return FourierProbe(
            self.wavelength,
            array=self.array,
            reciprocal_sampling=self.reciprocal_sampling,
            interpolation=interpolation,
            focal_spread=self.focal_spread,
            source_size=self.source_size,
        )

    @property
    def is_analytic(self) -> bool:
        """Whether the probe has aberration coefficients behind it.

        The parallax kernel, the ``sign(sin(chi))`` phase flip, the defocus gradient and any
        hyperparameter search over aberrations are only defined when this is true.
        """
        return self.array is None

    def to(self, device) -> "FourierProbe":
        if self.array is None:
            return self
        if self.modes is not None:
            return FourierProbe(
                self.wavelength,
                modes=self.modes.to(device),
                reciprocal_sampling=self.reciprocal_sampling,
                interpolation=self.interpolation,
                focal_spread=self.focal_spread,
                source_size=self.source_size,
            )
        moved = FourierProbe(
            self.wavelength,
            array=self.array.to(device),
            reciprocal_sampling=self.reciprocal_sampling,
            interpolation=self.interpolation,
            focal_spread=self.focal_spread,
            source_size=self.source_size,
        )
        return moved

    def resampled_to(self, reciprocal_sampling: Tuple[float, float]) -> "FourierProbe":
        """The same probe on a finer reciprocal grid, by zero-padding it in real space.

        This is not an approximation. ``psi`` is the transform of a probe confined to the
        field of view its sampling implies, so it is band limited, and zero-padding then
        transforming back is exactly the sinc interpolation that band limit licenses --
        unlike bilinear sampling, which is not.

        The requested step must divide the current one by a whole number per axis: refining
        by a non-integer factor would need the real-space probe resampled rather than merely
        extended, which *is* an approximation.

        This is what makes an empirical probe usable on a canvas larger than the probe's own
        field of view: the canvas needs ``q`` spaced ``1 / canvas_fov``, and the probe
        supplies ``1 / probe_fov``, so the padding factor is ``canvas_fov / probe_fov``.
        """
        if self.array is None:
            return self  # analytic probes are evaluated, not sampled

        current = np.asarray(self.reciprocal_sampling, dtype=float)
        target = np.asarray(reciprocal_sampling, dtype=float)
        ratio = current / target
        rounded = np.round(ratio)
        if np.any(rounded < 1) or np.any(np.abs(ratio - rounded) > 1e-6):
            raise ValueError(
                f"An empirical probe can only be refined onto a reciprocal grid whose step "
                f"divides its own by a whole number, but "
                f"{tuple(self.reciprocal_sampling)} / {tuple(float(t) for t in target)} = "  # ty:ignore[not-iterable]
                f"{tuple(np.round(ratio, 4))}. Equivalently, the canvas field of view must "
                f"be a whole multiple of the probe's."
            )

        factor = rounded.astype(int)
        if np.all(factor == 1):
            return self

        sampling = tuple(float(t) for t in target)  # ty:ignore[not-iterable]
        if self.modes is not None:
            return FourierProbe(
                self.wavelength,
                modes=torch.stack([self._refined(mode, factor) for mode in self.modes]),
                reciprocal_sampling=sampling,
                interpolation=self.interpolation,
                focal_spread=self.focal_spread,
                source_size=self.source_size,
            )
        return FourierProbe(
            self.wavelength,
            array=self._refined(self.array, factor),
            reciprocal_sampling=sampling,
            interpolation=self.interpolation,
            focal_spread=self.focal_spread,
            source_size=self.source_size,
        )

    @staticmethod
    def _refined(array, factor):
        """``array`` on a grid ``factor`` times finer, by zero-padding it in real space."""
        shape = tuple(int(n * f) for n, f in zip(array.shape, factor))
        real_space = torch.fft.ifft2(array)
        padded = torch.zeros(shape, dtype=real_space.dtype, device=real_space.device)
        # keep the corner-centered quadrants where they belong on the larger grid
        n_rows, n_cols = array.shape
        for row_slice, row_source in (
            (slice(0, (n_rows + 1) // 2), slice(0, (n_rows + 1) // 2)),
            (slice(shape[0] - n_rows // 2, shape[0]), slice((n_rows + 1) // 2, n_rows)),
        ):
            for col_slice, col_source in (
                (slice(0, (n_cols + 1) // 2), slice(0, (n_cols + 1) // 2)),
                (slice(shape[1] - n_cols // 2, shape[1]), slice((n_cols + 1) // 2, n_cols)),
            ):
                padded[row_slice, col_slice] = real_space[row_source, col_source]

        # no rescaling: the padded transform evaluated on the coarse sub-lattice is
        # `sum_m real[m] exp(-2i.pi.k.m/n)`, which is the original psi exactly
        return torch.fft.fft2(padded)

    def at(self, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
        """``psi`` at the given spatial frequencies, in inverse Angstrom.

        For an analytic probe this is a closed-form evaluation and any ``k`` is fine.

        An array probe can only be read at its own grid points, and ``gamma_factor`` asks for
        ``k -/+ q`` -- detector frequencies offset by canvas frequencies. Those coincide only
        when the two grids are commensurate, that is when

            (probe field of view) / (canvas field of view)

        is an integer, the probe field of view being ``1 / reciprocal_sampling``. When it is,
        every lookup is an exact gather. When it is not, ``interpolation="exact"`` raises
        rather than quietly interpolating: a speckled probe varies over a few detector pixels
        (0.83 amplitude spread pixel-to-pixel on the X-ray data this was written for), so
        bilinear sampling is an approximation worth opting into explicitly.

        The usual fix is to zero-pad the probe in real space -- which refines its reciprocal
        grid without changing the probe -- until the ratio is an integer.
        """
        if self.modes is not None:
            raise ValueError(
                "A mixed-state probe has no single psi(k). Its overlap function is a sum over "
                "`coherent_modes()`, which `overlap_terms` takes."
            )
        if self.array is None:
            k, phi = polar_coordinates(kx, ky)
            return evaluate_probe(
                k * self.wavelength,
                phi,
                self.semiangle_cutoff,
                self.angular_sampling,
                self.wavelength,
                self.soft_edges,
                None,
                self.aberration_coefs,
            )

        n_rows, n_cols = self.array.shape
        dq_row, dq_col = self.reciprocal_sampling  # ty:ignore[not-iterable]
        row = kx / dq_row
        col = ky / dq_col

        if self.interpolation == "bilinear":
            return self._bilinear(row, col)

        rounded_row, rounded_col = torch.round(row), torch.round(col)
        # 1e-3 of a detector pixel: loose enough for float32 k-grids, tight enough that a
        # genuinely incommensurate canvas never slips through
        offset = torch.maximum((row - rounded_row).abs().max(), (col - rounded_col).abs().max())
        if float(offset) > 1e-3:
            fov = (1 / dq_row, 1 / dq_col)
            raise ValueError(
                f"An empirical probe can only be sampled on its own reciprocal grid, but the "
                f"requested frequencies miss it by up to {float(offset):.3f} of a detector "
                f"pixel. The canvas field of view must divide the probe's, "
                f"{fov[0]:.2f} x {fov[1]:.2f} Angstrom. Choose a canvas that does, refine the "
                f"probe's grid with `resampled_to` (zero-padding it in real space), or pass "
                f"`interpolation='bilinear'` to accept the sampling error."
            )

        return self._gather(rounded_row, rounded_col)

    def _gather(self, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        """``psi`` at signed frequency indices, zero outside the detector.

        The indices are frequencies in units of ``dq``, so they run over
        ``[-n//2, (n+1)//2)`` -- the same range ``fftfreq`` produces. Beyond it the detector
        measured nothing, so the probe is zero there. Wrapping instead would alias the
        opposite edge of the aperture into the answer, which is wrong wherever the grid is
        cropped close to the probe: on the electron fixtures, where the bright-field mask is
        cropped to the disk, that alone moved the reconstruction by 13%.
        """
        array = self.array
        n_rows, n_cols = array.shape  # ty:ignore[possibly-unbound-attribute]
        row, col = row.long(), col.long()
        inside = (
            (row >= -(n_rows // 2))
            & (row < (n_rows + 1) // 2)
            & (col >= -(n_cols // 2))
            & (col < (n_cols + 1) // 2)
        )
        values = array[row % n_rows, col % n_cols]
        return values * inside

    def _bilinear(self, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        row0, col0 = torch.floor(row), torch.floor(col)
        drow, dcol = row - row0, col - col0
        return (
            self._gather(row0, col0) * ((1 - drow) * (1 - dcol))
            + self._gather(row0 + 1, col0) * (drow * (1 - dcol))
            + self._gather(row0, col0 + 1) * ((1 - drow) * dcol)
            + self._gather(row0 + 1, col0 + 1) * (drow * dcol)
        )


def _nonzero(value) -> bool:
    """Whether a float or tensor parameter is set: a tensor that requires grad always counts."""
    if isinstance(value, torch.Tensor):
        return value.requires_grad or bool((value != 0).any())
    return value is not None and float(value) != 0.0


def partial_coherence_envelope(kx, ky, qx, qy, wavelength: float, focal_spread, source_size):
    """``(env_minus, env_plus)``, the factors partial coherence puts on ``t_minus`` and ``t_plus``.

    A probe whose position spreads by a Gaussian of rms ``source_size`` per axis and whose
    defocus spreads by a Gaussian of rms ``focal_spread`` illuminates with the cross-spectral
    density ``J(k, k') = psi*(k) psi(k') mu(k' - k) exp(-(pi lambda sigma)^2 (|k'|^2 - |k|^2)^2 / 2)``.
    On the two halves of the overlap function this is

        ``exp(-2 pi^2 (source_size^2 |q|^2 + focal_spread^2 qz^2))``,

    with ``qz = lambda (|k|^2 - |k - q|^2) / 2`` for ``t_minus`` and
    ``lambda (|k + q|^2 - |k|^2) / 2`` for ``t_plus``, the same ``qz`` whose phase
    :func:`~quantem.diffractive_imaging.direct_ptycho_utils.depth_phase_factors` applies to a
    slice. The source factor depends on ``q`` alone, the focal factor on each bright-field
    pixel. It agreed with a 41-mode quadrature of the focal spread to 1e-15.
    """
    q2 = qx.square() + qy.square()
    k2 = kx.square() + ky.square()
    qz_minus = (k2 - ((kx - qx).square() + (ky - qy).square())) * (wavelength / 2)
    qz_plus = (((kx + qx).square() + (ky + qy).square()) - k2) * (wavelength / 2)
    scale = 2 * math.pi**2
    source = scale * source_size**2 * q2
    focal = scale * focal_spread**2
    return torch.exp(-(source + focal * qz_minus.square())), torch.exp(
        -(source + focal * qz_plus.square())
    )


def overlap_terms(
    probe: FourierProbe, cmplx_probe_at_k, kx, ky, qx, qy, partial_coherence: bool = True
):
    """``(t_minus, t_plus, psi_minus, psi_plus)``, the two halves of the overlap function.

    ``t_minus = psi(k - q) psi*(k)`` and ``t_plus = psi*(k + q) psi(k)``, so that
    ``Gamma-+ = t_minus -+ t_plus``. These are what the weak-object expansion of the
    bright-field intensity at detector frequency ``k`` gives at scan frequency ``q``, for
    transforms with the ``exp(-2 pi i ...)`` sign of :func:`torch.fft.fft2`. Writing
    ``psi(q - k)`` instead agrees only for a centrosymmetric probe: shifting the probe by
    ``s`` must multiply both terms by the same ``exp(-2 pi i q.s)``, which the object then
    absorbs, and ``psi(q - k)`` gives ``t_minus`` a ``k``-dependent phase instead.

    ``kx, ky`` are the bright-field frequencies, ``(B, 1, 1)``, and ``qx, qy`` the scan
    frequencies, ``(H, W)``. ``cmplx_probe_at_k`` is ``psi(k)``, ``(B, 1, 1)``.

    For a mixed state both terms are summed over the incoherent modes, each evaluated at
    ``k`` itself rather than read from ``cmplx_probe_at_k``. ``psi_minus`` and ``psi_plus``
    are then ``sqrt(sum_m |psi_m|^2)`` at ``k -/+ q``, which is all their callers use them for:
    whether the shifted aperture reaches there.

    A probe with :attr:`~FourierProbe.focal_spread` or :attr:`~FourierProbe.source_size` has
    both terms multiplied by :func:`partial_coherence_envelope`, unless ``partial_coherence``
    is false.
    """
    kmq = (kx - qx, ky - qy)
    kpq = (kx + qx, ky + qy)
    if probe.modes is None:
        psi_minus = probe.at(*kmq)
        psi_plus = probe.at(*kpq)
        t_minus = psi_minus * cmplx_probe_at_k.conj()
        t_plus = psi_plus.conj() * cmplx_probe_at_k
    else:
        t_minus = t_plus = power_minus = power_plus = 0
        for mode in probe.coherent_modes():
            psi_k = mode.at(kx, ky)
            psi_minus = mode.at(*kmq)
            psi_plus = mode.at(*kpq)
            t_minus = t_minus + psi_minus * psi_k.conj()
            t_plus = t_plus + psi_plus.conj() * psi_k
            power_minus = power_minus + psi_minus.abs().square()
            power_plus = power_plus + psi_plus.abs().square()
        psi_minus, psi_plus = power_minus.sqrt(), power_plus.sqrt()

    if partial_coherence and probe.is_partially_coherent:
        env_minus, env_plus = partial_coherence_envelope(
            kx, ky, qx, qy, probe.wavelength, probe.focal_spread, probe.source_size
        )
        t_minus = t_minus * env_minus.to(t_minus.real.dtype)
        t_plus = t_plus * env_plus.to(t_plus.real.dtype)
    return t_minus, t_plus, psi_minus, psi_plus


def gamma_factor(
    probe: FourierProbe,
    cmplx_probe_at_k: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
    qx: torch.Tensor,
    qy: torch.Tensor,
    asymmetric_version: bool = True,
    normalize: bool = True,
):
    """The overlap function ``Gamma-+(k, q) = t_minus -+ t_plus`` of :func:`overlap_terms`."""
    t_minus, t_plus, _, _ = overlap_terms(probe, cmplx_probe_at_k, kx, ky, qx, qy)
    gamma = t_minus - t_plus if asymmetric_version else t_minus + t_plus
    if normalize:
        gamma = gamma / gamma.abs().clamp(min=1e-8)
    return gamma


def evaluate_probe(
    alpha: torch.Tensor,
    phi: torch.Tensor,
    semiangle_cutoff: float,
    angular_sampling: Tuple[float, float],
    wavelength: float,
    soft_edges: bool = True,
    vacuum_probe_intensity: torch.Tensor | None = None,
    aberration_coefs: Mapping[str, float | torch.Tensor] = {},
) -> torch.Tensor:
    """ """

    probe_aperture = aperture(
        alpha, phi, semiangle_cutoff, angular_sampling, soft_edges, vacuum_probe_intensity
    )

    probe_aberrations = aberration_surface(alpha, phi, wavelength, aberration_coefs)

    return probe_aperture * torch.exp(-1j * probe_aberrations)


def _passively_rotate_grid(
    kxa: torch.Tensor,
    kya: torch.Tensor,
    rotation_angle: float,
):
    """ """

    if isinstance(rotation_angle, torch.Tensor):
        cos_a = torch.cos(-rotation_angle)
        sin_a = torch.sin(-rotation_angle)
    else:
        cos_a = math.cos(-rotation_angle)
        sin_a = math.sin(-rotation_angle)
    kxa, kya = (
        kxa * cos_a + kya * sin_a,
        -kxa * sin_a + kya * cos_a,
    )

    return kxa, kya


def spatial_frequencies(
    gpts: Tuple[int, int],
    sampling: Tuple[float, float] | NDArray,
    rotation_angle: float | None = None,
    device: str | torch.device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ """
    kxa = torch.fft.fftfreq(gpts[0], sampling[0], device=device, dtype=torch.float32)
    kya = torch.fft.fftfreq(gpts[1], sampling[1], device=device, dtype=torch.float32)
    kxa = kxa[:, None].broadcast_to(*gpts)
    kya = kya[None, :].broadcast_to(*gpts)

    # passive grid rotation
    if rotation_angle is not None:
        kxa, kya = _passively_rotate_grid(kxa, kya, rotation_angle)

    return kxa, kya


def polar_coordinates(kx: torch.Tensor, ky: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(|k|, angle)``, with a zero rather than NaN gradient at the origin.

    ``sqrt`` and ``atan2`` both have an undefined derivative at ``k = 0``, which a gradient
    with respect to a rotation angle or an aberration would otherwise turn into NaN. Routing
    the origin through ``torch.where`` on both sides leaves the values unchanged.
    """
    k2 = kx.square() + ky.square()
    nonzero = k2 > 0
    k = torch.where(nonzero, torch.sqrt(torch.where(nonzero, k2, torch.ones_like(k2))), 0)
    phi = torch.where(
        nonzero,
        torch.atan2(
            torch.where(nonzero, ky, torch.zeros_like(ky)),
            torch.where(nonzero, kx, torch.ones_like(kx)),
        ),
        0,
    )
    return k, phi


def polar_spatial_frequencies(
    gpts: Tuple[int, int],
    sampling: Tuple[float, float],
    rotation_angle: float | None = None,
    device: str | torch.device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ """
    kx, ky = spatial_frequencies(gpts, sampling, rotation_angle=rotation_angle, device=device)
    return polar_coordinates(kx, ky)


def fourier_space_probe(
    gpts: Tuple[int, int],
    sampling: Tuple[float, float],
    energy: float,
    semiangle_cutoff: float,
    rotation_angle: float | None = None,
    soft_edges: bool = True,
    vacuum_probe_intensity: torch.Tensor | None = None,
    aberration_coefs: Mapping[str, float | torch.Tensor] = {},
    normalized: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """ """
    wavelength = electron_wavelength_angstrom(energy)
    k, phi = polar_spatial_frequencies(
        gpts, sampling, rotation_angle=rotation_angle, device=device
    )
    alpha = k * wavelength
    angular_sampling = (alpha[1, 0] * 1e3, alpha[0, 1] * 1e3)

    vacuum = (
        vacuum_probe_intensity.to(device=device) if vacuum_probe_intensity is not None else None
    )

    fourier_probe = evaluate_probe(
        alpha,
        phi,
        semiangle_cutoff,
        angular_sampling,
        wavelength,
        soft_edges=soft_edges,
        vacuum_probe_intensity=vacuum,
        aberration_coefs=aberration_coefs,
    )

    if normalized:
        fourier_probe = fourier_probe / fourier_probe.abs().square().sum().sqrt()

    return fourier_probe


def real_space_probe(
    gpts: Tuple[int, int],
    sampling: Tuple[float, float],
    energy: float,
    semiangle_cutoff: float,
    rotation_angle: float | None = None,
    soft_edges: bool = True,
    vacuum_probe_intensity: torch.Tensor | None = None,
    aberration_coefs: Mapping[str, float | torch.Tensor] = {},
    normalized: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """ """

    fourier_probe = fourier_space_probe(
        gpts,
        sampling,
        energy,
        semiangle_cutoff,
        rotation_angle=rotation_angle,
        soft_edges=soft_edges,
        vacuum_probe_intensity=vacuum_probe_intensity,
        aberration_coefs=aberration_coefs,
        normalized=True,
        device=device,
    )

    probe = torch.fft.ifft2(fourier_probe)

    if normalized:
        probe = probe / probe.abs().square().sum().sqrt()

    return probe


def aberration_surface_grad(
    gpts: Tuple[int, int],
    sampling: Tuple[float, float],
    energy: float,
    rotation_angle: float | None = None,
    aberration_coefs: Mapping[str, float | torch.Tensor] = {},
    device: str | torch.device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ """
    wavelength = electron_wavelength_angstrom(energy)
    k, phi = polar_spatial_frequencies(
        gpts, sampling, rotation_angle=rotation_angle, device=device
    )
    alpha = k * wavelength

    dx, dy = aberration_surface_cartesian_gradients(alpha, phi, aberration_coefs)
    return dx, dy


def polar_to_cartesian_aberrations(polar, max_order=5, device=None, dtype=None):
    polar = defaultdict(lambda: torch.tensor(0.0, device=device, dtype=dtype), polar)
    cart = {}

    for n in range(1, max_order + 1):
        for s in range(0, n + 2):
            m = 2 * s - n - 1
            if m < 0:
                continue
            name = f"C{n}{m}"
            if m == 0:
                cart[name] = polar[name]
            else:
                phi = polar[f"phi{n}{m}"]
                C = polar[name]
                cart[f"{name}_a"] = C * torch.cos(m * phi)
                cart[f"{name}_b"] = C * torch.sin(m * phi)

    return cart


def cartesian_to_polar_aberrations(cart, max_order=5):
    cart = defaultdict(lambda: torch.tensor(0.0), cart)
    polar = {}

    for n in range(1, max_order + 1):
        for s in range(0, n + 2):
            m = 2 * s - n - 1
            if m < 0:
                continue
            name = f"C{n}{m}"
            if m == 0:
                polar[name] = cart[name]
            else:
                Ca = cart[f"{name}_a"]
                Cb = cart[f"{name}_b"]
                polar[name] = torch.sqrt(Ca**2 + Cb**2)
                polar[f"phi{n}{m}"] = torch.atan2(Cb, Ca) / m

    return polar


def merge_aberration_coefficients(
    init_coefs_polar: dict,
    delta_coefs_cartesian: dict,
):
    """
    Convert cartesian aberration deltas to polar and merge with initial coefficients.

    Parameters
    ----------
    aberration_coefs_init : dict
        Polar aberration coefficients (e.g. C10, C12, phi12, ...)
    delta_cartesian : dict
        Fitted cartesian deltas (Cnm_a, Cnm_b)

    Returns
    -------
    dict
        Updated polar aberration coefficients
    """
    updated_coefs_cartesian = polar_to_cartesian_aberrations(init_coefs_polar)

    for k, v in delta_coefs_cartesian.items():
        if k in updated_coefs_cartesian:
            updated_coefs_cartesian[k] = updated_coefs_cartesian[k] + v
        else:
            updated_coefs_cartesian[k] = v

    updated_coefs_polar = cartesian_to_polar_aberrations(updated_coefs_cartesian)

    return updated_coefs_polar


def parse_cartesian_aberration_label(label: str) -> tuple[int, int, str | None]:
    """
    Parse 'Cnm', 'Cnm_a', 'Cnm_b'
    Returns (n, m, kind) where kind ∈ {None, 'a', 'b'}
    """

    base, *rest = label.split("_")
    kind = rest[0] if rest else None
    n = int(base[1])
    m = int(base[2])

    return n, m, kind


def aberration_surface_cartesian_basis(
    alpha: torch.Tensor, phi: torch.Tensor, wavelength: float, cartesian_basis: list[str]
) -> torch.Tensor:
    """
    Cartesian aberration chi basis.

    Parameters
    ----------
    alpha, phi : torch.Tensor
        Polar k-space coordinates
    wavelength : float
    cartesian_basis : list[str]
        e.g. ['C10', 'C12_a', 'C12_b', 'C21_a', 'C21_b']

    Returns
    -------
    dict[str, torch.Tensor]
        chi basis functions
    """
    k = 2 * math.pi / wavelength
    out = []

    for label in cartesian_basis:
        n, m, kind = parse_cartesian_aberration_label(label)
        pref = k / (n + 1)
        radial = alpha ** (n + 1)

        if kind is None:
            out.append(pref * radial)
        elif kind == "a":
            out.append(pref * radial * torch.cos(m * phi))
        elif kind == "b":
            out.append(pref * radial * torch.sin(m * phi))
        else:
            raise ValueError(f"Invalid aberration label: {label}")

    return torch.stack(out, dim=-1)
