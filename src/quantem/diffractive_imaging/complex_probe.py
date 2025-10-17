import math
from typing import Mapping, Tuple

import torch

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
        (torch.cos(phi) * angular_sampling[0] * 1e-3) ** 2
        + (torch.sin(phi) * angular_sampling[1] * 1e-3) ** 2
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


def standardize_aberration_coefs(aberration_coefs: Mapping[str, float]) -> dict[str, float]:
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
    aberration_coefs: dict[str, float | torch.Tensor],
):
    """ """

    pi = math.pi
    alpha2 = alpha**2
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
        chi = chi + (1 / 4) * alpha2**2 * (
            get("C30")
            + get("C32") * torch.cos(2 * (phi - get("phi32")))
            + get("C34") * torch.cos(4 * (phi - get("phi34")))
        )

    if any(k in coefs for k in ("C41", "phi41", "C43", "phi43", "C45", "phi45")):
        chi = chi + (1 / 5) * alpha2**2 * alpha * (
            get("C41") * torch.cos(phi - get("phi41"))
            + get("C43") * torch.cos(3 * (phi - get("phi43")))
            + get("C45") * torch.cos(5 * (phi - get("phi45")))
        )

    if any(k in coefs for k in ("C50", "C52", "phi52", "C54", "phi54", "C56", "phi56")):
        chi = chi + (1 / 6) * alpha2**3 * (
            get("C50")
            + get("C52") * torch.cos(2 * (phi - get("phi52")))
            + get("C54") * torch.cos(4 * (phi - get("phi54")))
            + get("C56") * torch.cos(6 * (phi - get("phi56")))
        )

    chi = 2 * pi / wavelength * chi
    return chi


def evaluate_probe(
    alpha: torch.Tensor,
    phi: torch.Tensor,
    semiangle_cutoff: float,
    angular_sampling: Tuple[float, float],
    wavelength: float,
    soft_edges: bool = True,
    vacuum_probe_intensity: torch.Tensor | None = None,
    aberration_coefs: dict[str, float | torch.Tensor] = {},
) -> torch.Tensor:
    """ """

    probe_aperture = aperture(
        alpha, phi, semiangle_cutoff, angular_sampling, soft_edges, vacuum_probe_intensity
    )

    probe_aberrations = aberration_surface(alpha, phi, wavelength, aberration_coefs)

    return probe_aperture * torch.exp(-1j * probe_aberrations)


def spatial_frequencies(
    gpts: Tuple[int, int], sampling: Tuple[float, float], device: str | torch.device = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ """
    kx = torch.fft.fftfreq(gpts[0], sampling[0], device=device, dtype=torch.float32)
    ky = torch.fft.fftfreq(gpts[1], sampling[1], device=device, dtype=torch.float32)
    return kx, ky


def polar_spatial_frequencies(
    gpts: Tuple[int, int], sampling: Tuple[float, float], device: str | torch.device = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ """
    kx, ky = spatial_frequencies(gpts, sampling, device=device)
    k = torch.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    phi = torch.arctan2(ky[None, :], kx[:, None])
    return k, phi


def fourier_space_probe(
    gpts: Tuple[int, int],
    sampling: Tuple[float, float],
    energy: float,
    semiangle_cutoff: float,
    soft_edges: bool = True,
    vacuum_probe_intensity: torch.Tensor | None = None,
    aberration_coefs: dict[str, float | torch.Tensor] = {},
    normalized: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """ """
    wavelength = electron_wavelength_angstrom(energy)
    k, phi = polar_spatial_frequencies(gpts, sampling, device=device)
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
    soft_edges: bool = True,
    vacuum_probe_intensity: torch.Tensor | None = None,
    aberration_coefs: dict[str, float | torch.Tensor] = {},
    normalized: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """ """

    fourier_probe = fourier_space_probe(
        gpts,
        sampling,
        energy,
        semiangle_cutoff,
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
