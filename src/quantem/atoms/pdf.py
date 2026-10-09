"""Pair distribution functions, neighbor lists and bond-length calibration.

The radial distribution function (RDF) of a 3D atomic model is computed with a
KD-tree cumulative pair count, which is fast (tens of milliseconds for tens of
thousands of atoms) and needs no pair list.  The first RDF peak is fit with an
asymmetric generalized Gaussian so the mean nearest-neighbor (NN) distance and a
first-shell cutoff can be estimated robustly, following the approach of the
original MATLAB cluster analysis scripts.

Calibration maps the measured NN distance to the NN distance of a reference
crystal, giving the physical size of one voxel.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

__all__ = [
    "radial_distribution",
    "fit_first_peak",
    "find_neighbors",
    "nn_distance_from_lattice",
]


def radial_distribution(
    xyz: NDArray,
    r_max: float,
    dr: float,
    sigma: float | None = None,
) -> dict[str, NDArray]:
    """Compute the radial distribution function of a set of 3D points.

    Parameters
    ----------
    xyz : ndarray
        ``(N, 3)`` coordinates.
    r_max : float
        Maximum radius.
    dr : float
        Radial bin width.
    sigma : float, optional
        Gaussian smoothing width (same units as ``r``).  Default ``2 * dr``.

    Returns
    -------
    dict
        ``r`` bin centers, ``counts`` pair counts per bin (each unordered pair
        counted once), ``g`` the RDF normalized by the shell volume and mean
        number density, and ``g_smooth`` a Gaussian-smoothed copy of ``g``.
    """
    xyz = np.asarray(xyz, dtype=float)
    tree = cKDTree(xyz)
    edges = np.arange(0.0, r_max + dr, dr)
    pairs = tree.query_pairs(r_max, output_type="ndarray")
    if pairs.shape[0]:
        dist = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
        counts = np.histogram(dist, bins=edges)[0].astype(float)
    else:
        counts = np.zeros(edges.size - 1)
    r = 0.5 * (edges[1:] + edges[:-1])
    shell_volume = 4.0 * np.pi * r**2 * dr
    extent = xyz.max(0) - xyz.min(0)
    density = xyz.shape[0] / max(float(np.prod(extent)), 1e-12)
    g = counts / (shell_volume * density * xyz.shape[0] / 2.0)
    if sigma is None:
        sigma = 2.0 * dr
    g_smooth = gaussian_filter1d(g, sigma / dr, mode="nearest") if sigma > 0 else g.copy()
    return {"r": r, "counts": counts, "g": g, "g_smooth": g_smooth}


def _asymmetric_peak(r, amp, r0, p_lo, p_hi, w_lo, w_hi):
    lo = amp * np.exp(-((np.abs(r - r0) / w_lo) ** p_lo))
    hi = amp * np.exp(-((np.abs(r - r0) / w_hi) ** p_hi))
    return np.where(r < r0, lo, hi)


def fit_first_peak(
    r: NDArray,
    g: NDArray,
    fit_radius: float = 1.25,
    cutoff_sigma: float = 2.0,
    r_min: float | None = None,
) -> dict[str, float | NDArray]:
    """Fit the first RDF peak with an asymmetric generalized Gaussian.

    Parameters
    ----------
    r, g : ndarray
        Radial bins and (smoothed) RDF.
    fit_radius : float
        Fit window extends from ``r_min`` to ``fit_radius * r_peak``.
    cutoff_sigma : float
        First-shell cutoffs are placed ``cutoff_sigma`` widths below / above
        the peak position.
    r_min : float, optional
        Ignore the RDF below this radius (e.g. tracing artifacts).  Defaults to
        one third of the peak radius.

    Returns
    -------
    dict
        ``r_nn`` peak position, ``width_lo``/``width_hi`` asymmetric widths,
        ``cutoff`` = ``(lo, hi)`` first-shell radial cutoffs, ``fit`` the
        fitted curve on ``r`` and ``coefs`` the raw fit coefficients.
    """
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)
    p = np.where(r > 0, g / np.maximum(r, 1e-12) ** 2, 0.0)
    # first significant peak: first local maximum of g above 50% of global max
    i_max = int(np.argmax(g))
    peaks = np.where((g[1:-1] > g[:-2]) & (g[1:-1] >= g[2:]) & (g[1:-1] > 0.5 * g[i_max]))[0] + 1
    i_peak = int(peaks[0]) if peaks.size else i_max
    del p
    r0 = float(r[i_peak])
    amp0 = float(g[i_peak])
    if r_min is None:
        r_min = r0 / 3.0
    sub = (r >= r_min) & (r <= fit_radius * r0)
    w0 = 0.15 * r0
    coefs0 = [amp0, r0, 2.0, 2.0, w0, w0]
    lower = [0, r_min, 1.0, 1.0, 1e-3 * r0, 1e-3 * r0]
    upper = [np.inf, fit_radius * r0, 8.0, 8.0, r0, r0]
    try:
        coefs, _ = curve_fit(
            _asymmetric_peak, r[sub], g[sub], p0=coefs0, bounds=(lower, upper), maxfev=20000
        )
    except (RuntimeError, ValueError):
        coefs = np.array(coefs0)
    amp, r_nn, _, _, w_lo, w_hi = coefs
    fit = _asymmetric_peak(r, *coefs)
    return {
        "r_nn": float(r_nn),
        "amplitude": float(amp),
        "width_lo": float(w_lo),
        "width_hi": float(w_hi),
        "cutoff": (float(r_nn - cutoff_sigma * w_lo), float(r_nn + cutoff_sigma * w_hi)),
        "fit": fit,
        "coefs": np.asarray(coefs),
    }


def find_neighbors(xyz: NDArray, num_neighbors: int) -> tuple[NDArray, NDArray]:
    """Return the ``num_neighbors`` nearest neighbors of every point.

    Parameters
    ----------
    xyz : ndarray
        ``(N, 3)`` coordinates.
    num_neighbors : int
        Neighbors per point (self excluded).

    Returns
    -------
    distances, indices : ndarray
        ``(N, num_neighbors)`` arrays sorted by distance.  If fewer than
        ``num_neighbors`` points exist, missing entries have index ``-1`` and
        infinite distance.
    """
    xyz = np.asarray(xyz, dtype=float)
    n = xyz.shape[0]
    k = min(num_neighbors + 1, n)
    tree = cKDTree(xyz)
    dist, idx = tree.query(xyz, k=k)
    # drop the query point itself (not necessarily in column 0 when points coincide)
    is_self = idx == np.arange(n)[:, None]
    dist = np.where(is_self, -np.inf, dist)
    order = np.argsort(dist, axis=1, kind="stable")
    dist = np.take_along_axis(dist, order, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    has_self = is_self.any(axis=1)
    dist = np.where(has_self[:, None], dist[:, 1:], dist[:, : k - 1])
    idx = np.where(has_self[:, None], idx[:, 1:], idx[:, : k - 1])
    if k - 1 < num_neighbors:
        pad = num_neighbors - (k - 1)
        dist = np.pad(dist, ((0, 0), (0, pad)), constant_values=np.inf)
        idx = np.pad(idx, ((0, 0), (0, pad)), constant_values=-1)
    return dist, idx


_NN_FACTORS = {
    "fcc": 1.0 / np.sqrt(2.0),
    "bcc": np.sqrt(3.0) / 2.0,
    "sc": 1.0,
    "hcp": 1.0,
    "diamond": np.sqrt(3.0) / 4.0,
    "zincblende": np.sqrt(3.0) / 4.0,
    "wurtzite": np.sqrt(3.0 / 8.0) * np.sqrt(8.0 / 3.0) * 3.0 / 8.0,  # u*c with c = 1.633a
}


def nn_distance_from_lattice(structure: str, lattice_constant: float) -> float:
    """Nearest-neighbor distance of a reference crystal.

    Parameters
    ----------
    structure : str
        ``"fcc"``, ``"bcc"``, ``"sc"``, ``"hcp"``, ``"diamond"``, ``"zincblende"``
        or ``"wurtzite"``.  For the hexagonal structures ``lattice_constant``
        is ``a`` and ideal ``c/a`` is assumed.
    lattice_constant : float
        Cubic lattice constant ``a`` (or hexagonal ``a``).

    Returns
    -------
    float
        Nearest-neighbor distance in the units of ``lattice_constant``.
    """
    key = structure.lower()
    if key not in _NN_FACTORS:
        raise ValueError(f"Unknown structure {structure!r}; choose from {list(_NN_FACTORS)}")
    return float(_NN_FACTORS[key] * lattice_constant)
