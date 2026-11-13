"""
Repeat-slice multislice propagation for the PADF test lattices.

Treats a 2D lattice image as the projected potential of one slice and
propagates a plane wave through n_slices identical copies (AA stacking) -
a minimal dynamical-diffraction model of a column structure. With a
NON-centrosymmetric projection (e.g. the boron_nitride lattice), multiple
scattering violates Friedel symmetry: odd angular harmonics appear in the
diffraction rings and grow with thickness. Centrosymmetric projections
(triangular, honeycomb, square) keep I(q) = I(-q) at any thickness.

For quantitative work use abTEM with a real 3D structure; this module is
for fast, controlled tests of how dynamical scattering perturbs
correlation / PADF / bispectrum observables.
"""

import numpy as np


def circular_window(shape):
    """
    Radially symmetric Hann window: w = cos^2(pi rho / 2R) with R half the
    smaller image dimension, zero outside. Unlike the separable 2D Hann
    (outer product of 1D windows), its FFT skirt is isotropic, so the
    windowed central beam carries no spurious 4-fold angular structure
    into the correlations.
    """
    n_row, n_col = shape
    row = np.arange(n_row) - (n_row - 1) / 2
    col = np.arange(n_col) - (n_col - 1) / 2
    rho = np.hypot(row[:, None], col[None, :])
    R = min(n_row, n_col) / 2.0
    w = np.cos(0.5 * np.pi * np.minimum(rho / R, 1.0)) ** 2
    return w


def electron_wavelength(voltage_kv: float) -> float:
    """Relativistic electron wavelength in Angstroms."""
    v = voltage_kv * 1e3
    h = 6.62607015e-34
    m0 = 9.1093837015e-31
    e = 1.602176634e-19
    c = 2.99792458e8
    lam = h / np.sqrt(2 * m0 * e * v * (1 + e * v / (2 * m0 * c**2)))
    return lam * 1e10


def multislice(
    potential,
    pixel_size_ang: float,
    n_slices: int,
    slice_thickness_ang: float = 3.33,
    voltage_kv: float = 80.0,
    phase_per_slice: float = 0.15,
):
    """
    Propagate a plane wave through n_slices copies of the same 2D potential.

    Parameters
    ----------
    potential : (N, N) array
        Projected potential of one slice (arbitrary units; normalized
        internally so its maximum produces `phase_per_slice` radians).
    pixel_size_ang : float
        Real-space pixel size in Angstroms.
    n_slices : int
        Number of identical slices (thickness = n_slices * slice_thickness).
    slice_thickness_ang : float
        Slice spacing in Angstroms (3.33 = graphite/h-BN interlayer).
    voltage_kv : float
        Accelerating voltage in kV (sets the Fresnel propagator).
    phase_per_slice : float
        Peak phase shift (radians) of one slice at the strongest atom.
        n_slices = 1 with a small value approaches the kinematic limit.

    Returns
    -------
    psi : (N, N) complex exit wave.
    """
    pot = np.asarray(potential, dtype=np.float64)
    n_row, n_col = pot.shape
    lam = electron_wavelength(voltage_kv)

    transmission = np.exp(1j * phase_per_slice * pot / pot.max())

    q_row = np.fft.fftfreq(n_row, d=pixel_size_ang)
    q_col = np.fft.fftfreq(n_col, d=pixel_size_ang)
    q2 = q_row[:, None] ** 2 + q_col[None, :] ** 2
    propagator = np.exp(-1j * np.pi * lam * slice_thickness_ang * q2)
    # standard 2/3 bandwidth limit to prevent aliasing of scattered orders
    q_max = min(np.abs(q_row).max(), np.abs(q_col).max())
    propagator *= q2 <= (2.0 / 3.0 * q_max) ** 2

    psi = np.ones((n_row, n_col), dtype=np.complex128)
    for _ in range(n_slices):
        psi = np.fft.ifft2(np.fft.fft2(psi * transmission) * propagator)
    return psi


def diffraction_intensity(psi, window: bool = True):
    """
    Diffraction intensity of an exit wave: |FFT(psi * w)|^2, fftshifted,
    with a circular (radial Hann) window so the finite-field skirt is
    isotropic. The DC (unscattered beam) still dominates; downstream
    analysis should treat it like the experimental central beam.
    """
    if window:
        psi = psi * circular_window(psi.shape)
    return np.abs(np.fft.fftshift(np.fft.fft2(psi))) ** 2
