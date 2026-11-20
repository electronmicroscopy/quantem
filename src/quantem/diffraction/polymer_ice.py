"""Flag crystalline-ice reflections among peaks detected in polymer 4D-STEM diffraction.

Ice contaminating a polymer scan gives six-fold sets of sharp reflections. Its
strongest ring (d ~ 3.66 A) overlaps the pi-pi stacking peak, so q alone cannot
separate them, and ``detect_ice`` tests each peak against four gates in turn:

1. q window: within ``dq_invA`` of ``q_target_invA``.
2. Sharpness (optional): the ``width_q`` and ``width_theta`` fields, measured with
   :func:`measure_peak_widths`, at or below the ``max_width_*`` ceilings. Ice is
   annularly sharp, while polymer at the same q is an annularly broad arc.
3. Intensity: at or above ``intensity_cutoff``, or a scan-wide percentile.
4. Six-fold geometry: candidates must sit on arms 60 degrees apart, with at least
   ``min_matches`` arms populated. Theta is folded modulo 180 degrees for this
   test, so a Friedel pair shares one arm and at most three arms exist.

Up to ``max_crystallites`` lattices are peeled from each pattern, strongest first.
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from quantem.core.datastructures import Vector


@dataclass(frozen=True)
class IceFlaggerParams:
    """Settings for :func:`detect_ice` and :func:`measure_peak_widths`.

    Parameters
    ----------
    q_target_invA, dq_invA : float
        Centre and half-width of the q window, in 1/Angstrom.
    dtheta_deg : float
        Angular tolerance for a peak to sit on a six-fold arm.
    min_matches : int
        Arms, out of the three on the folded 0-180 degree axis, that must be populated.
    min_peaks_per_arm : int
        Peaks an arm must carry to count. 2 demands a pair, but two nearby peaks also pass.
    require_friedel_pair : bool
        Count only arms holding two peaks whose unfolded angles differ by 180 +/- dtheta_deg.
    max_crystallites : int
        Lattices at unrelated orientations to peel from one pattern.
    min_phi_separation_deg : float, optional
        Orientation difference, on the 0-60 degree wedge, for a further lattice to count
        as a new crystallite. Defaults to ``dtheta_deg``.
    conservative : bool
        If False, also flag in-window peaks below the intensity threshold that sit on a
        found lattice.
    intensity_cutoff : float, optional
        Absolute intensity threshold. If None, the ``intensity_percentile_global``
        percentile of all peak intensities in the analysed scan positions is used.
    intensity_percentile_global : float
        Scan-wide percentile used when ``intensity_cutoff`` is None.
    max_width_q_invA, max_width_theta_deg : float, optional
        Ceilings on the radial (1/Angstrom) and annular (degrees) FWHM. None leaves that
        axis ungated. The annular ceiling is the one that separates ice from polymer.
    sharpness_window_q_invA, sharpness_window_theta_deg : float
        Half-widths of the profiles used to measure each FWHM. A profile that never falls
        to half maximum inside its window reads as wider than the window.
    sharpness_baseline_quantile : float
        Quantile of the profile taken as local background below the half maximum.
    sharpness_refine_bins : int
        Samples of argmax refinement around the detected peak position.
    sharpness_smooth_sigma_bins : float
        Gaussian smoothing of each profile, in samples, removed from the FWHM in quadrature.
    sharpness_crossing_persistence : int
        Samples a half-maximum crossing must stay below half to count.
    """

    q_target_invA: float = 1.61
    dq_invA: float = 0.05
    dtheta_deg: float = 6.0
    min_matches: int = 2
    min_peaks_per_arm: int = 1
    require_friedel_pair: bool = False
    max_crystallites: int = 1
    min_phi_separation_deg: float | None = None
    conservative: bool = True
    intensity_cutoff: float | None = None
    intensity_percentile_global: float = 99.0
    max_width_q_invA: float | None = None
    max_width_theta_deg: float | None = None
    sharpness_window_q_invA: float = 0.06
    sharpness_window_theta_deg: float = 90.0
    sharpness_baseline_quantile: float = 0.05
    sharpness_refine_bins: int = 2
    sharpness_smooth_sigma_bins: float = 1.0
    sharpness_crossing_persistence: int = 3

    def __post_init__(self):
        rules = [
            ("> 0", lambda v: v > 0, "dq_invA dtheta_deg sharpness_window_q_invA"),
            ("> 0", lambda v: v > 0, "sharpness_window_theta_deg"),
            (">= 1", lambda v: v >= 1, "min_peaks_per_arm max_crystallites"),
            (">= 1", lambda v: v >= 1, "sharpness_crossing_persistence"),
            (">= 0", lambda v: v >= 0, "sharpness_refine_bins sharpness_smooth_sigma_bins"),
            ("None or > 0", lambda v: v is None or v > 0, "max_width_q_invA max_width_theta_deg"),
            ("None or > 0", lambda v: v is None or v > 0, "min_phi_separation_deg"),
            ("1, 2 or 3, the arms on the folded axis", lambda v: 1 <= v <= 3, "min_matches"),
            ("in [0, 100]", lambda v: 0 <= v <= 100, "intensity_percentile_global"),
            ("in [0, 1)", lambda v: 0 <= v < 1, "sharpness_baseline_quantile"),
        ]
        for rule, ok, names in rules:
            for name in names.split():
                if not ok(getattr(self, name)):
                    raise ValueError(f"{name} must be {rule}; got {getattr(self, name)!r}")


@dataclass(frozen=True)
class IceDetectionResult:
    """Output of :func:`detect_ice`.

    Attributes
    ----------
    mask : Vector
        Field ``is_ice`` (1 for ice), one row per input peak, aligned with the peaks.
    flagged_count_map : ndarray of int
        Ice peaks per scan position.
    matched_arm_count_map : ndarray of int
        Populated six-fold arms per scan position, summed over crystallites.
    intensity_threshold : float
        Intensity threshold applied to candidates.
    debug : dict or None
        Per-position diagnostics keyed by scan index, when requested.
    """

    mask: Vector
    flagged_count_map: NDArray[np.integer]
    matched_arm_count_map: NDArray[np.integer]
    intensity_threshold: float
    debug: dict | None = None

    def filter(self, peaks: Vector, invert: bool = False) -> Vector:
        """Return a copy of ``peaks`` without the ice peaks, or with only them if ``invert``."""
        counts = np.asarray(peaks.row_counts())
        if peaks.shape != self.mask.shape or counts.tolist() != self.mask.row_counts():
            raise ValueError("peaks are not aligned with the ice mask.")
        keep = (self.mask.numpy()[:, 0] > 0) == invert
        kept = np.bincount(np.repeat(np.arange(len(counts)), counts)[keep], minlength=len(counts))
        return _from_flat(
            peaks.numpy()[keep],
            kept,
            peaks.shape,
            fields=peaks.fields,
            units=peaks.units,
            name=peaks.name,
            metadata=copy.deepcopy(peaks.metadata),
            dtype=peaks.dtype,
            device=peaks.device,
        )


def measure_peak_widths(
    dp: NDArray,
    peaks_rc: NDArray,
    origin: tuple[float, float],
    pixel_size: float,
    params: IceFlaggerParams | None = None,
) -> tuple[NDArray, NDArray]:
    """Radial and annular FWHM of each peak in one diffraction pattern.

    The pattern is sampled with bilinear interpolation along the arc about ``origin``
    through each peak, in 1 degree steps, and then along the radial line through the
    arc maximum, in 1 pixel steps. Each profile's FWHM is measured above a local
    baseline, with the settings in ``params``.

    Parameters
    ----------
    dp : ndarray
        Diffraction pattern, (H, W) detector pixels.
    peaks_rc : ndarray
        (M, 2) peak positions (row, col) in detector pixels.
    origin : tuple of float
        (row, col) position of the direct beam.
    pixel_size : float
        Detector pixel size in 1/Angstrom.
    params : IceFlaggerParams, optional
        Window, baseline, smoothing and persistence settings.

    Returns
    -------
    width_q : ndarray
        (M,) radial FWHM in 1/Angstrom.
    width_theta : ndarray
        (M,) annular FWHM in radians. Profiles that never fall to half maximum inside
        their window, or run off the detector first, read as wider than the window,
        and peaks with no maximum above the baseline are NaN.
    """
    p = IceFlaggerParams() if params is None else params
    image = torch.as_tensor(np.asarray(dp, dtype=np.float64))
    peaks = np.asarray(peaks_rc, dtype=np.float64).reshape(-1, 2)
    width_q, width_theta = np.full(len(peaks), np.nan), np.full(len(peaks), np.nan)
    ok = np.isfinite(peaks).all(1)
    if not ok.any():
        return width_q, width_theta
    r0, c0 = float(origin[0]), float(origin[1])
    radius = np.hypot(peaks[ok, 0] - r0, peaks[ok, 1] - c0)[:, None]
    phi = np.arctan2(peaks[ok, 0] - r0, peaks[ok, 1] - c0)[:, None]

    # 1 degree samples; at most just under a full circle, so no sample repeats.
    n_arc = min(max(2, math.ceil(p.sharpness_window_theta_deg)), 179)
    angles = phi + np.deg2rad(np.arange(-n_arc, n_arc + 1))
    arc = _sample(image, r0 + radius * np.sin(angles), c0 + radius * np.cos(angles))
    fwhm_theta, at = _fwhm(arc, n_arc, p)

    n_line = max(2, math.ceil(p.sharpness_window_q_invA / pixel_size))
    radii = radius + np.arange(-n_line, n_line + 1)
    angle = angles[np.arange(len(angles)), at][:, None]
    line = _sample(image, r0 + radii * np.sin(angle), c0 + radii * np.cos(angle))
    line[radii < 0] = np.nan
    fwhm_q, _ = _fwhm(line, n_line, p)

    width_q[ok] = fwhm_q * pixel_size
    width_theta[ok] = np.deg2rad(fwhm_theta)
    return width_q, width_theta


def detect_ice(
    peaks: Vector,
    params: IceFlaggerParams | None = None,
    scan_mask: NDArray[np.bool_] | None = None,
    return_debug: bool = False,
) -> IceDetectionResult:
    """Flag the peaks that belong to six-fold ice lattices.

    Parameters
    ----------
    peaks : Vector
        Peaks per scan position, with fields ``q`` (1/Angstrom), ``theta`` (radians,
        unfolded) and ``intensity``, plus ``width_q`` and ``width_theta`` from
        :func:`measure_peak_widths` when ``params`` sets a width ceiling.
    params : IceFlaggerParams, optional
        Detection settings.
    scan_mask : ndarray of bool, optional
        Scan positions to analyse. Peaks elsewhere are never flagged and do not
        enter the intensity percentile.
    return_debug : bool
        Also return per-position counts and the lattices found.

    Returns
    -------
    IceDetectionResult
    """
    p = IceFlaggerParams() if params is None else params
    ceilings = {"width_q": p.max_width_q_invA, "width_theta": p.max_width_theta_deg}
    fields = ["q", "theta", "intensity"] + [f for f, c in ceilings.items() if c is not None]
    missing = [f for f in fields if f not in peaks.fields]
    if missing:
        raise ValueError(
            f"peaks has no field(s) {missing}; width fields come from measure_peak_widths."
        )
    shape = peaks.shape
    selected = np.ones(shape, bool) if scan_mask is None else np.asarray(scan_mask, bool)
    if selected.shape != shape:
        raise ValueError(f"scan_mask shape {selected.shape} must match the scan shape {shape}.")

    counts = np.asarray(peaks.row_counts(), dtype=int)
    starts = np.concatenate([[0], np.cumsum(counts)])
    cell_of_row = np.repeat(np.arange(len(counts)), counts)
    data = peaks.select_fields(fields).numpy().astype(np.float64)
    q, theta, intensity = data[:, 0], data[:, 1], data[:, 2]
    analysed = selected.ravel()[cell_of_row]

    in_q = analysed & np.isfinite(theta) & (np.abs(q - p.q_target_invA) <= p.dq_invA)
    sharp = in_q.copy()
    if p.max_width_q_invA is not None:
        sharp &= data[:, fields.index("width_q")] <= p.max_width_q_invA
    if p.max_width_theta_deg is not None:
        sharp &= np.rad2deg(data[:, fields.index("width_theta")]) <= p.max_width_theta_deg
    if p.intensity_cutoff is not None:
        threshold = float(p.intensity_cutoff)
    else:
        values = intensity[analysed & np.isfinite(intensity)]
        threshold = (
            float(np.percentile(values, p.intensity_percentile_global)) if values.size else np.inf
        )
    candidate = sharp & (intensity >= threshold)
    n_candidates = np.bincount(cell_of_row[candidate], minlength=len(counts))

    flags = np.zeros(len(data), bool)
    lattices: dict[int, list] = {}
    for cell in np.flatnonzero(n_candidates >= p.min_matches):
        rows = slice(starts[cell], starts[cell + 1])
        flags[rows], lattices[cell] = _fit_lattices(theta[rows], sharp[rows], candidate[rows], p)
    arms = np.zeros(len(counts), int)
    for cell, found in lattices.items():
        arms[cell] = sum(len(a) for _, a in found)

    debug = None
    if return_debug:
        n_q, n_sharp = (np.bincount(cell_of_row[m], minlength=len(counts)) for m in (in_q, sharp))
        debug = {
            tuple(int(i) for i in np.unravel_index(cell, shape)): {
                "n_peaks": int(counts[cell]),
                "n_q": int(n_q[cell]),
                "n_sharp": int(n_sharp[cell]),
                "n_candidates": int(n_candidates[cell]),
                "phi_deg": [phi for phi, _ in lattices.get(cell, [])],
                "arms": [a for _, a in lattices.get(cell, [])],
            }
            for cell in np.flatnonzero(selected.ravel())
        }
    mask = _from_flat(
        flags[:, None].astype(np.float32),
        counts,
        shape,
        fields=["is_ice"],
        units=["bool"],
        name="ice_mask",
    )
    flagged = np.bincount(cell_of_row[flags], minlength=len(counts)).reshape(shape)
    return IceDetectionResult(mask, flagged, arms.reshape(shape), threshold, debug)


def _fit_lattices(theta, sharp, candidate, p):
    """Peel six-fold lattices from one pattern; return the flags and (phi_deg, arms) per lattice."""
    folded = np.rad2deg(theta) % 180.0
    flags = np.zeros(theta.shape, bool)
    found: list[tuple[float, list[int]]] = []
    separation = p.dtheta_deg if p.min_phi_separation_deg is None else p.min_phi_separation_deg
    remaining = np.flatnonzero(candidate)
    while remaining.size and len(found) < p.max_crystallites:
        # Orientation from the best-supported cluster on the 0-60 degree wedge.
        wedge = folded[remaining] % 60.0
        support = np.abs(_wrap(wedge[:, None] - wedge, 60.0)) <= p.dtheta_deg
        phi = _circular_mean(wedge[support[support.sum(1).argmax()]], 60.0)
        if any(abs(_wrap(phi - other, 60.0)) < separation for other, _ in found):
            break
        arm, error = _arms(folded[remaining], phi)
        aligned = error <= p.dtheta_deg
        good = [a for a in range(3) if _arm_counts(theta[remaining[aligned & (arm == a)]], p)]
        if len(good) < p.min_matches:
            break
        claimed = aligned & np.isin(arm, good)
        flags[remaining[claimed]] = True
        if not p.conservative:
            arm_all, error_all = _arms(folded, phi)
            flags |= sharp & (error_all <= p.dtheta_deg) & np.isin(arm_all, good)
        found.append((phi, good))
        remaining = remaining[~claimed]
    return flags, found


def _arms(folded_deg, phi):
    """Nearest arm (0, 1, 2) of the lattice at ``phi`` and the angular error, on the folded axis."""
    steps = np.round((folded_deg - phi) / 60.0)
    return (steps % 3).astype(int), np.abs(folded_deg - phi - 60.0 * steps)


def _arm_counts(theta, p):
    """Whether an arm holding peaks at unfolded angles ``theta`` (radians) counts as matched."""
    if len(theta) < p.min_peaks_per_arm:
        return False
    if not p.require_friedel_pair:
        return True
    separation = np.abs(_wrap(np.rad2deg(theta[:, None] - theta), 360.0))
    return bool(np.any(np.abs(separation - 180.0) <= p.dtheta_deg))


def _wrap(delta, period):
    """Signed angular difference wrapped to [-period/2, period/2)."""
    return (delta + period / 2) % period - period / 2


def _circular_mean(angles, period):
    """Mean of angles on a circle of circumference ``period``, in [0, period)."""
    a = 2 * np.pi * np.asarray(angles) / period
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()) * period / (2 * np.pi) % period)


def _sample(image, rows, cols):
    """Bilinear samples of ``image`` at pixel positions (rows, cols), NaN off the detector."""
    h, w = image.shape
    grid = torch.as_tensor(np.stack([2 * cols / (w - 1) - 1, 2 * rows / (h - 1) - 1], -1))
    values = F.grid_sample(image[None, None], grid[None], mode="bilinear", align_corners=True)
    inside = (rows >= 0) & (rows <= h - 1) & (cols >= 0) & (cols <= w - 1)
    return np.where(inside, values[0, 0].numpy(), np.nan)


def _fwhm(profiles, center, p):
    """FWHM in samples of each row of ``profiles`` about sample ``center``, and the refined peak."""
    sigma = p.sharpness_smooth_sigma_bins
    if sigma > 0:
        half = int(3 * sigma + 0.5)
        kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
        padded = np.pad(profiles, ((0, 0), (half, half)), mode="edge")
        profiles = sliding_window_view(padded, 2 * half + 1, axis=1) @ (kernel / kernel.sum())
    rows = np.arange(len(profiles))
    lo = max(0, center - p.sharpness_refine_bins)
    near = profiles[:, lo : center + p.sharpness_refine_bins + 1]
    at = lo + np.where(np.isfinite(near), near, -np.inf).argmax(1)
    peak = profiles[rows, at]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        baseline = np.nanquantile(profiles, p.sharpness_baseline_quantile, axis=1)
    level = baseline + 0.5 * (peak - baseline)
    right = _half_width(profiles, at, level, p)
    left = _half_width(profiles[:, ::-1], profiles.shape[1] - 1 - at, level, p)
    width = np.sqrt(np.maximum((left + right) ** 2 - (2.3548 * sigma) ** 2, 0.0))
    return np.where(np.isfinite(peak) & (peak > baseline), width, np.nan), at


def _half_width(profiles, center, level, p):
    """Samples from ``center`` rightwards to a drop below ``level`` that persists.

    Running off the window or the detector first returns the window length.
    """
    m, n = profiles.shape
    k = p.sharpness_crossing_persistence
    above = np.pad(profiles > level[:, None], ((0, 0), (0, k - 1)))
    sustained = ~sliding_window_view(above, k, axis=1).any(-1)
    stop = (np.arange(n) > center[:, None]) & (sustained | ~np.isfinite(profiles))
    j = stop.argmax(1)
    before, after = profiles[np.arange(m), j - 1], profiles[np.arange(m), j]
    span = before - after
    fraction = np.clip(np.divide(before - level, span, out=np.zeros(m), where=span > 0), 0, 1)
    return np.where(stop.any(1) & np.isfinite(after), j - 1 - center + fraction, n)


def _from_flat(rows, counts, shape, **kwargs) -> Vector:
    """Vector of ``shape`` whose cells hold consecutive blocks of ``rows``, ``counts`` long."""
    cells = np.split(rows, np.cumsum(counts)[:-1])

    def nest(block, dims):
        if not dims:
            return block[0]
        step = len(block) // dims[0]
        return [nest(block[i * step : (i + 1) * step], dims[1:]) for i in range(dims[0])]

    return Vector.from_data(nest(cells, tuple(shape)), **kwargs)


__all__ = ["IceDetectionResult", "IceFlaggerParams", "detect_ice", "measure_peak_widths"]
