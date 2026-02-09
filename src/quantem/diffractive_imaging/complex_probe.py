import math
from collections import defaultdict
from typing import Mapping, Tuple

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


def gamma_factor(
    qmks: tuple[torch.Tensor, torch.Tensor],
    qpks: tuple[torch.Tensor, torch.Tensor],
    cmplx_probe_at_k: torch.Tensor,
    wavelength: float,
    semiangle_cutoff: float,
    soft_edges: bool,
    aberration_coefs: Mapping[str, float | torch.Tensor],
    angular_sampling: Tuple[float, float],
    asymmetric_version: bool = True,
    normalize: bool = True,
):
    """ """

    q_m, phi_m = polar_coordinates(*qmks)
    q_p, phi_p = polar_coordinates(*qpks)

    probe_m = evaluate_probe(
        q_m * wavelength,
        phi_m,
        semiangle_cutoff,
        angular_sampling,
        wavelength,
        soft_edges,
        None,
        aberration_coefs,
    )

    probe_p = evaluate_probe(
        q_p * wavelength,
        phi_p,
        semiangle_cutoff,
        angular_sampling,
        wavelength,
        soft_edges,
        None,
        aberration_coefs,
    )

    if asymmetric_version:
        gamma = probe_m * cmplx_probe_at_k.conj() - probe_p.conj() * cmplx_probe_at_k
    else:
        gamma = probe_m * cmplx_probe_at_k.conj() + probe_p.conj() * cmplx_probe_at_k
    if normalize:
        gamma /= gamma.abs().clamp(min=1e-8)
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
    """ """
    k = torch.sqrt(kx.square() + ky.square())
    phi = torch.arctan2(ky, kx)
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
