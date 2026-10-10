"""Illumination-averaged excitation envelopes for kinematical patterns.

Precession rotates the incident beam on a cone and a convergent probe fills
a disk of directions; the recorded intensity of a reflection is the
average of its rocking curve over that support. For a ring centered on the
optic axis the excitation error of reflection g is exactly
s_g(phi) = c_g + a_g cos(phi - delta_g), and for a small disk it is affine
in the incident direction to leading order, so the average of a Gaussian
excitation envelope over ring and disk reduces to one scalar transform,

    G(c, a, b; sigma) = sqrt(2/pi) int_0^inf exp(-x^2/2) cos(c x / sigma)
                        J0(a x / sigma) jinc(b x / sigma) dx,

with jinc(x) = 2 J1(x) / x; G(c, 0, 0) = exp(-c^2 / 2 sigma^2) recovers the
static envelope. The functions here evaluate G for whole reflection lists
at once (vectorized Gauss-Legendre quadrature of the transform, exact to
~1e-9 over the parameter range of electron diffraction), give the ring and
disk coefficients (c, a, b) from the geometry, and are shared by the
pattern simulation, the orientation library and the refinements.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import j0, j1

from quantem.core.utils.utils import electron_wavelength_angstrom

_X_NODES, _X_WEIGHTS = np.polynomial.legendre.leggauss(400)
_X_MAX = 9.0
_X = 0.5 * _X_MAX * (_X_NODES + 1)
_W = 0.5 * _X_MAX * _X_WEIGHTS * np.exp(-0.5 * _X**2) * np.sqrt(2 / np.pi)


def _jinc(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-12
    out[nz] = 2 * j1(x[nz]) / x[nz]
    return out


def gaussian_envelope(c, a, b, sigma: float) -> np.ndarray:
    """Illumination-averaged Gaussian excitation envelope G(c, a, b; sigma).

    Parameters
    ----------
    c, a, b : array-like
        Central excitation error, ring amplitude and disk amplitude of each
        reflection (1/Angstroms), from `excitation_coefficients`.
    sigma : float
        Width of the excitation envelope (1/Angstroms).

    Returns
    -------
    np.ndarray
        The averaged envelope in [0, 1], same shape as c.
    """
    c = np.asarray(c, dtype=float)
    a = np.broadcast_to(np.asarray(a, dtype=float), c.shape)
    b = np.broadcast_to(np.asarray(b, dtype=float), c.shape)
    x = _X / sigma
    integrand = np.cos(c[..., None] * x) * j0(a[..., None] * x) * _jinc(b[..., None] * x)
    return np.clip(integrand @ _W, 0.0, 1.0)


def excitation_coefficients(
    g_lab: torch.Tensor | np.ndarray,
    energy_ev: float,
    precession_deg: float = 0.0,
    semiconv_mrad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central excitation error and illumination amplitudes of each reflection.

    For a precession ring of radius r = k0 sin(theta_p) centered on the optic
    axis, with K = sqrt(k0^2 - r^2),

        c_g = (2 K g_z - |g|^2) / (2 (K - g_z)),   a_g = r |g_xy| / |K - g_z|,

    which is exact: s_g(phi) = c_g + a_g cos(phi - delta_g). The convergence
    disk of radius R = k0 sin(alpha) adds b_g = R |g_xy| / |K - g_z| under
    the same affine model. Without illumination, c_g is the static
    excitation error and a_g = b_g = 0.

    Parameters
    ----------
    g_lab : torch.Tensor | np.ndarray
        Lab-frame reciprocal vectors (N, 3), 1/Angstroms, with the beam
        along -z.
    energy_ev : float
        Beam energy, eV.
    precession_deg : float, default=0.0
        Precession semi-angle, degrees.
    semiconv_mrad : float, default=0.0
        Convergence semi-angle, mrad.

    Returns
    -------
    c, a, b : np.ndarray
        Central excitation error, ring amplitude and disk amplitude (N,),
        1/Angstroms.
    """
    g = np.asarray(
        g_lab.detach().cpu().numpy() if isinstance(g_lab, torch.Tensor) else g_lab, dtype=float
    )
    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    r = k0 * np.sin(np.deg2rad(precession_deg))
    R = k0 * np.sin(semiconv_mrad * 1e-3)
    K = np.sqrt(k0**2 - r**2)
    den = K - g[:, 2]
    g2 = (g**2).sum(axis=1)
    gxy = np.hypot(g[:, 0], g[:, 1])
    c = (2 * K * g[:, 2] - g2) / (2 * den)
    a = r * gxy / np.abs(den)
    b = R * gxy / np.abs(den)
    return c, a, b


def relrod_factor(g_lab, n_lab, energy_ev: float, precession_deg: float = 0.0):
    """How far along a relrod the Ewald sphere is, per unit excitation error.

    A plate-shaped crystal spreads every reciprocal lattice point into a rod
    along the plate normal n, a long one for a 2D material. The sphere meets
    the rod through g at g + t n, with, to first order,

        t = -f s_g,   f = (K - g_z) / (K n_z - n . g),

    where s_g is the excitation error measured along the beam and
    K = sqrt(k0^2 - r^2) as in :func:`excitation_coefficients`. For n along
    the beam f = 1. A rod nearly tangent to the sphere (an edge-on plate) is
    never excited; f is set huge there so the reflection drops out.

    Parameters
    ----------
    g_lab : torch.Tensor | np.ndarray
        Lab-frame reciprocal vectors (..., 3), 1/Angstroms.
    n_lab : torch.Tensor | np.ndarray
        Lab-frame unit plate normal, broadcastable to `g_lab`.
    energy_ev : float
        Beam energy, eV.
    precession_deg : float, default=0.0
        Precession semi-angle, degrees; sets K as above.

    Returns
    -------
    torch.Tensor | np.ndarray
        f (...,), dimensionless, the same type as `g_lab`. 1e6 where the rod
        is within 0.05 K of tangent to the sphere.
    """
    k0 = 1.0 / electron_wavelength_angstrom(energy_ev)
    r = k0 * np.sin(np.deg2rad(precession_deg))
    K = float(np.sqrt(k0**2 - r**2))
    if isinstance(g_lab, torch.Tensor):
        n = torch.as_tensor(n_lab, dtype=g_lab.dtype, device=g_lab.device)
        den_n = K * n[..., 2] - (g_lab * n).sum(-1)
        tangent = den_n.abs() < 0.05 * K
        f = (K - g_lab[..., 2]) / torch.where(tangent, torch.ones_like(den_n), den_n)
        return torch.where(tangent, torch.full_like(f, 1e6), f)
    g = np.asarray(g_lab, dtype=float)
    n = np.asarray(n_lab, dtype=float)
    den_n = K * n[..., 2] - (g * n).sum(-1)
    tangent = np.abs(den_n) < 0.05 * K
    f = (K - g[..., 2]) / np.where(tangent, 1.0, den_n)
    return np.where(tangent, 1e6, f)


def averaged_gaussian_intensity_envelope(
    g_lab,
    energy_ev: float,
    sigma: float,
    precession_deg: float = 0.0,
    semiconv_mrad: float = 0.0,
    foil_normal_lab=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Illumination-averaged Gaussian envelope of every reflection.

    With `foil_normal_lab` the excitation is measured along the relrod
    instead of along the beam: c, a and b are the distances along the rod
    (see :func:`relrod_factor`).

    Parameters
    ----------
    g_lab : torch.Tensor | np.ndarray
        Lab-frame reciprocal vectors (N, 3), 1/Angstroms.
    energy_ev : float
        Beam energy, eV.
    sigma : float
        Width of the excitation envelope, 1/Angstroms.
    precession_deg : float, default=0.0
        Precession semi-angle, degrees.
    semiconv_mrad : float, default=0.0
        Convergence semi-angle, mrad.
    foil_normal_lab : array-like, optional
        Lab-frame unit plate normal (3,).

    Returns
    -------
    envelope : np.ndarray
        Averaged envelope (N,) in [0, 1].
    c, a, b : np.ndarray
        Central excitation error and ring and disk amplitudes (N,),
        1/Angstroms; a + b is the half-width of the swept range about c.
    """
    c, a, b = excitation_coefficients(g_lab, energy_ev, precession_deg, semiconv_mrad)
    if foil_normal_lab is not None:
        g_np = g_lab.detach().cpu().numpy() if isinstance(g_lab, torch.Tensor) else g_lab
        f = relrod_factor(
            np.asarray(g_np, dtype=float), foil_normal_lab, energy_ev, precession_deg
        )
        c, a, b = c * f, a * np.abs(f), b * np.abs(f)
    if precession_deg <= 0 and semiconv_mrad <= 0:
        return np.exp(-0.5 * (c / sigma) ** 2), c, a, b
    # A reflection farther from the Ewald sphere than the illumination sweeps
    # it, plus six envelope widths, is never excited (the envelope there is
    # below 1e-8). At k_max = 2 that is nearly all of them, and the quadrature
    # below costs 400 Bessel evaluations per reflection.
    env = np.zeros_like(c)
    live = np.abs(c) < a + b + 6.0 * sigma
    if live.any():
        if semiconv_mrad <= 0:
            # a pure precession ring has the fast Bessel series the orientation
            # plan uses; the disk quadrature is needed only with convergence
            env[live] = (
                gaussian_envelope_ring_torch(
                    torch.as_tensor(c[live]), torch.as_tensor(a[live]), sigma
                )
                .cpu()
                .numpy()
            )
        else:
            env[live] = gaussian_envelope(c[live], a[live], b[live], sigma)
    return env, c, a, b


def ring_disk_quadrature(r: float, R: float, n_phi: int = 128, n_r: int = 8, n_psi: int = 32):
    """Positive quadrature of the ring x disk illumination.

    The reference against which the analytic envelopes are checked.

    Parameters
    ----------
    r : float
        Ring radius, in the units the tilts are wanted in.
    R : float
        Disk radius, same units.
    n_phi : int, default=128
        Equally spaced points on the ring.
    n_r : int, default=8
        Gauss-Legendre radial nodes of the disk (in r^2).
    n_psi : int, default=32
        Equally spaced azimuths of the disk.

    Returns
    -------
    tilts : np.ndarray
        In-plane tilts (M, 2), ring point plus disk point.
    weights : np.ndarray
        Positive weights (M,) summing to 1.
    """
    phi = 2 * np.pi * np.arange(n_phi) / n_phi if r > 0 else np.zeros(1)
    ring = r * np.column_stack((np.cos(phi), np.sin(phi)))
    if R > 0:
        x, w = np.polynomial.legendre.leggauss(n_r)
        radius = R * np.sqrt((x + 1) / 2)
        psi = 2 * np.pi * np.arange(n_psi) / n_psi
        disk = (radius[:, None, None] * np.column_stack((np.cos(psi), np.sin(psi)))[None]).reshape(
            -1, 2
        )
        wd = np.repeat(w / 2 / n_psi, n_psi)
    else:
        disk, wd = np.zeros((1, 2)), np.ones(1)
    t = (ring[:, None] + disk[None]).reshape(-1, 2)
    weights = np.tile(wd, len(ring)) / len(ring)
    return t, weights


def gaussian_envelope_ring_series(c, a, sigma: float, n_terms: int = 6) -> np.ndarray:
    """Ring-averaged Gaussian envelope, b = 0, by the Bessel series

        G = exp(-c^2/2 sigma^2 - v) [I0(u) I0(v) + 2 sum_n (-1)^n I_2n(u) I_n(v)],
        u = c a / sigma^2,  v = a^2 / 4 sigma^2,

    which converges in a few terms for v < 1 (the precession sweep of the
    excitation error smaller than the envelope width, the electron
    diffraction regime); larger v falls back to the transform.

    Parameters
    ----------
    c : np.ndarray | torch.Tensor
        Central excitation errors, any shape, 1/Angstroms.
    a : np.ndarray | torch.Tensor | float
        Ring amplitudes, broadcast to `c`, 1/Angstroms.
    sigma : float
        Width of the excitation envelope, 1/Angstroms.
    n_terms : int, default=6
        Terms of the sum over n.

    Returns
    -------
    np.ndarray | torch.Tensor
        Envelope in [0, 1], shape of `c`; a float64 tensor when `c` is a
        tensor.
    """
    from scipy.special import iv

    is_torch = isinstance(c, torch.Tensor)
    c_np = np.asarray(c.detach().cpu().numpy() if is_torch else c, dtype=float)
    a_np = np.broadcast_to(
        np.asarray(a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a, dtype=float),
        c_np.shape,
    )
    u = c_np * a_np / sigma**2
    v = a_np**2 / (4 * sigma**2)
    total = iv(0, u) * iv(0, v)
    for n in range(1, n_terms + 1):
        total = total + 2 * (-1) ** n * iv(2 * n, u) * iv(n, v)
    out = np.exp(-0.5 * (c_np / sigma) ** 2 - v) * total
    big = v > 1.0
    if np.any(big):
        out[big] = gaussian_envelope(c_np[big], a_np[big], 0.0, sigma)
    out = np.clip(out, 0.0, 1.0)
    return torch.as_tensor(out, dtype=torch.float64) if is_torch else out


def excitation_amplitudes(
    g_lab: torch.Tensor, energy_ev: float, precession_deg: float, semiconv_mrad: float
):
    """Ring and disk amplitudes of lab-frame reflections, in torch.

    The a and b of :func:`excitation_coefficients`, without c, for any
    leading shape and differentiable in `g_lab`.

    Parameters
    ----------
    g_lab : torch.Tensor
        Lab-frame reciprocal vectors (..., 3), 1/Angstroms.
    energy_ev : float
        Beam energy, eV.
    precession_deg : float
        Precession semi-angle, degrees.
    semiconv_mrad : float
        Convergence semi-angle, mrad.

    Returns
    -------
    a, b : torch.Tensor
        Ring and disk amplitudes (...,), 1/Angstroms; zero without
        precession or convergence.
    """
    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    r = k0 * np.sin(np.deg2rad(precession_deg))
    R = k0 * np.sin(semiconv_mrad * 1e-3)
    K = np.sqrt(k0**2 - r**2)
    den = (K - g_lab[..., 2]).abs().clamp_min(1e-12)
    gxy = torch.hypot(g_lab[..., 0], g_lab[..., 1])
    return r * gxy / den, R * gxy / den


_V_NODES, _V_WEIGHTS = np.polynomial.legendre.leggauss(200)
_V = 0.5 * (_V_NODES + 1)
_WV = 0.5 * _V_WEIGHTS * 2 * (1 - _V)


def slab_envelope(c, a, b, thickness_A: float) -> np.ndarray:
    """Illumination-averaged finite-thickness (first Born) rocking curve,

        S(c, a, b; z) = 2 int_0^1 (1 - v) cos(2 pi c z v) J0(2 pi a z v)
                        jinc(2 pi b z v) dv,

    which reduces to sinc(c z)^2 without illumination; the Born intensity
    of reflection g is (pi |U_g| z / k0)^2 S. Vectorized Gauss-Legendre
    quadrature over v, exact to ~1e-10 for the phase ranges of electron
    diffraction (c z below ~20).

    Parameters
    ----------
    c, a, b : array-like
        Central excitation error, ring amplitude and disk amplitude
        (1/Angstroms), from :func:`excitation_coefficients`; a and b are
        broadcast to the shape of c.
    thickness_A : float
        Specimen thickness z, Angstroms.

    Returns
    -------
    np.ndarray
        S in [0, 1], same shape as c.
    """
    c = np.asarray(c, dtype=float)
    a = np.broadcast_to(np.asarray(a, dtype=float), c.shape)
    b = np.broadcast_to(np.asarray(b, dtype=float), c.shape)
    x = 2 * np.pi * thickness_A * _V
    integrand = np.cos(c[..., None] * x) * j0(a[..., None] * x) * _jinc(b[..., None] * x)
    return np.clip(integrand @ _WV, 0.0, 1.0)


def gaussian_envelope_ring_torch(c: torch.Tensor, a: torch.Tensor, sigma: float) -> torch.Tensor:
    """Ring-averaged Gaussian envelope (b = 0) in torch, for the refinements.

    The Bessel series of :func:`gaussian_envelope_ring_series` truncated
    after the I_4 term, with I_n(u) from the I_0 / I_1 recurrences (and
    their small-argument series where the recurrence would cancel). Below
    v = a^2 / 4 sigma^2 = 0.3 the truncation error is under 1e-5; larger
    sweeps are averaged over the ring directly by quadrature. Stays on the
    device of `c` and is differentiable.

    Parameters
    ----------
    c : torch.Tensor
        Central excitation errors, any shape, 1/Angstroms.
    a : torch.Tensor
        Ring amplitudes, broadcastable to `c`, 1/Angstroms.
    sigma : float
        Width of the excitation envelope, 1/Angstroms.

    Returns
    -------
    torch.Tensor
        float64 envelope in [0, 1], the broadcast shape of `c` and `a`.
    """
    c = c.to(torch.float64)
    a = torch.as_tensor(a, dtype=torch.float64)
    u = c * a / sigma**2
    v = a * a / (4 * sigma**2)
    i0u = torch.special.i0(u)
    i1u = torch.special.i1(u)
    small2 = u.abs() < 1e-3
    u_safe = torch.where(small2, torch.ones_like(u), u)
    i2u = torch.where(small2, u * u / 8, i0u - 2 * i1u / u_safe)
    small4 = u.abs() < 5e-2
    i3u = torch.where(small4, u**3 / 48, i1u - 4 * i2u / u_safe)
    i4u = torch.where(small4, u**4 / 384, i2u - 6 * i3u / u_safe)
    i0v = torch.special.i0(v)
    i1v = torch.special.i1(v)
    smallv = v < 1e-3
    v_safe = torch.where(smallv, torch.ones_like(v), v)
    i2v = torch.where(smallv, v * v / 8, i0v - 2 * i1v / v_safe)
    out = torch.exp(-0.5 * (c / sigma) ** 2 - v) * (i0u * i0v - 2 * i2u * i1v + 2 * i4u * i2v)
    # v follows the shape of a, which may be narrower than the output: the
    # fallback mask has to be taken in the broadcast shape or it indexes the
    # wrong elements (a narrow sigma or a large sweep reaches this branch)
    big = torch.broadcast_to(v > 0.3, out.shape)
    if bool(big.any()):
        out = out.clone()
        c_b = torch.broadcast_to(c, out.shape)[big]
        a_b = torch.broadcast_to(a, out.shape)[big]
        # the ring average itself, (1/pi) int_0^pi exp(-(c - a cos phi)^2 /
        # 2 sigma^2) dphi, by the midpoint rule: exact to rounding for this
        # periodic integrand once the nodes resolve a / sigma, and it stays
        # in torch on the input's device
        n = int(min(256, max(16, np.ceil(8 + 4 * float(a_b.max()) / sigma))))
        phi = (torch.arange(n, dtype=torch.float64, device=c_b.device) + 0.5) * (np.pi / n)
        d = c_b[:, None] - a_b[:, None] * torch.cos(phi)
        out[big] = torch.exp(-0.5 * (d / sigma) ** 2).mean(dim=-1).to(out.dtype)
    return out.clamp(0.0, 1.0)
