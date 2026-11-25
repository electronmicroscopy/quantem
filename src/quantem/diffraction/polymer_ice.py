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
import dataclasses
import math
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
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
    sharpness_friedel_sum : bool
        Add the pattern at the Friedel-opposite position to each profile before
        measuring it, which doubles the signal of a centrosymmetric pattern.
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
    sharpness_friedel_sum: bool = True

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
    ellipse: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Radial and annular FWHM of each peak in one diffraction pattern.

    Each peak is measured in the ellipse-corrected frame, where rings are circles.
    The pattern is sampled with bilinear interpolation along the arc about
    ``origin`` through the peak, in steps of 1 pixel of arc, and then along the
    radial line through the arc maximum, in 1 pixel steps. Each profile's FWHM is
    measured above a local baseline, with the settings in ``params``.

    Parameters
    ----------
    dp : ndarray
        Diffraction pattern, (H, W) detector pixels.
    peaks_rc : ndarray
        (M, 2) peak positions (row, col) in detector pixels.
    origin : tuple of float
        (row, col) position of the direct beam.
    pixel_size : float
        Pixel size of the ellipse-corrected frame in 1/Angstrom, as in
        :class:`~quantem.diffraction.calibration.DiffractionCalibration`.
    params : IceFlaggerParams, optional
        Window, baseline, smoothing, persistence and Friedel-sum settings.
    ellipse : array-like, optional
        ``[e11, e12]`` elliptic distortion correction of
        :class:`~quantem.diffraction.calibration.DiffractionCalibration`. None
        treats the rings as circles.

    Returns
    -------
    width_q : ndarray
        (M,) radial FWHM in 1/Angstrom.
    width_theta : ndarray
        (M,) annular FWHM in radians. Profiles that never fall to half maximum inside
        their window, or run off the detector first, read as wider than the window,
        and peaks with no maximum above the baseline are NaN.
    """
    from quantem.diffraction.calibration import _ellipse_matrix

    p = IceFlaggerParams() if params is None else params
    image = torch.as_tensor(np.asarray(dp, dtype=np.float64))
    peaks = np.asarray(peaks_rc, dtype=np.float64).reshape(-1, 2)
    width_q, width_theta = np.full(len(peaks), np.nan), np.full(len(peaks), np.nan)
    origin = np.asarray(origin, dtype=np.float64)
    to_circle = np.eye(2) if ellipse is None else _ellipse_matrix(ellipse)
    to_detector = np.linalg.inv(to_circle)

    def profile(rows, cols):
        # (row, col) offsets in the corrected frame -> samples of the pattern
        offsets = np.stack([rows, cols], -1) @ to_detector.T
        values = _sample(image, *(origin + offsets).transpose(2, 0, 1))
        if p.sharpness_friedel_sum:
            values = values + _sample(image, *(origin - offsets).transpose(2, 0, 1))
        return values

    n_line = max(2, math.ceil(p.sharpness_window_q_invA / pixel_size))
    for i in np.flatnonzero(np.isfinite(peaks).all(1)):
        d = (peaks[i] - origin) @ to_circle.T
        radius, phi = np.hypot(*d), np.arctan2(*d)
        if radius == 0:
            continue
        step = 1.0 / radius  # 1 pixel of arc, in radians
        # at most just under half a circle each way, so no sample repeats
        n_arc = min(
            max(2, math.ceil(np.deg2rad(p.sharpness_window_theta_deg) / step)),
            int((np.pi - 1e-6) / step),
        )
        angles = phi + step * np.arange(-n_arc, n_arc + 1)[None]
        fwhm_theta, at = _fwhm(profile(radius * np.sin(angles), radius * np.cos(angles)), n_arc, p)

        radii = radius + np.arange(-n_line, n_line + 1)[None]
        angle = angles[0, at[0]]
        line = profile(radii * np.sin(angle), radii * np.cos(angle))
        line[radii < 0] = np.nan
        fwhm_q, _ = _fwhm(line, n_line, p)

        width_q[i] = fwhm_q[0] * pixel_size
        width_theta[i] = fwhm_theta[0] * step
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
    threshold = _intensity_threshold(intensity[analysed], p)
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


def _intensity_threshold(intensity, p):
    """``intensity_cutoff``, or the ``intensity_percentile_global`` percentile of ``intensity``."""
    if p.intensity_cutoff is not None:
        return float(p.intensity_cutoff)
    values = intensity[np.isfinite(intensity)]
    return float(np.percentile(values, p.intensity_percentile_global)) if values.size else np.inf


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


# ---- figures ----


def _analyzed_rows(peaks, scan_mask):
    """Per-peak flag: the peak's scan position is in ``scan_mask``."""
    if scan_mask is None:
        return np.ones(int(np.sum(peaks.row_counts())), bool)
    return np.repeat(np.asarray(scan_mask, bool).ravel(), peaks.row_counts())


def plot_q_intensity_density(
    peaks: Vector,
    params: IceFlaggerParams | None = None,
    scan_mask: NDArray[np.bool_] | None = None,
    q_range=None,
    q_bins: int = 200,
    i_bins: int = 200,
    color: str = "cyan",
):
    """Density of peaks in (q, intensity), with the ice q window and intensity threshold.

    Use it to choose ``intensity_cutoff``: ice reflections stand out as bright
    peaks at ``q_target_invA``.

    Parameters
    ----------
    peaks : Vector
        Polar peaks with fields ``(q, theta, intensity)``.
    params : IceFlaggerParams, optional
        The q window and intensity threshold to draw.
    scan_mask : ndarray of bool, optional
        Scan positions to include, as in :func:`detect_ice`.
    q_range : (float, float), optional
        q axis limits in 1/Angstrom; defaults to zero up to the largest peak q or
        the window edge.
    q_bins, i_bins : int
        Histogram bins along q and intensity.
    color : str
        Color of the window and threshold.

    Returns
    -------
    fig, ax
    """
    p = IceFlaggerParams() if params is None else params
    data = peaks.select_fields(["q", "intensity"]).numpy().astype(np.float64)
    data = data[_analyzed_rows(peaks, scan_mask) & np.isfinite(data).all(1)]
    if len(data) == 0:
        raise ValueError("No peaks with finite q and intensity to plot.")
    q, intensity = data[:, 0], data[:, 1]
    threshold = _intensity_threshold(intensity, p)
    i_max = max(1.0, float(intensity.max()))
    if q_range is None:
        q_range = (0.0, max(float(q.max()), p.q_target_invA + p.dq_invA))

    fig, ax = plt.subplots(figsize=(8, 4))
    hist = ax.hist2d(
        q,
        intensity,
        bins=(q_bins, i_bins),
        range=(q_range, (0, i_max)),
        norm=LogNorm(),
        cmap="magma",
    )
    fig.colorbar(hist[3], ax=ax, label="peaks (log scale)")
    ax.add_patch(
        Rectangle(
            (p.q_target_invA - p.dq_invA, threshold),
            2 * p.dq_invA,
            i_max - threshold,
            facecolor=color,
            edgecolor=color,
            alpha=0.18,
        )
    )
    ax.axvline(p.q_target_invA, color=color, ls=":", lw=2)
    ax.axhline(threshold, color=color, ls="--", lw=2)
    ax.text(q_range[0], threshold, f" threshold {threshold:.3g}", color=color, va="bottom")
    ax.set(xlabel="q (1/Å)", ylabel="intensity", xlim=q_range, ylim=(0, i_max))
    fig.tight_layout()
    return fig, ax


def selection_box(
    peaks: Vector,
    params: IceFlaggerParams | None = None,
    max_width_q_invA: float | None = None,
    max_width_theta_deg: float | None = None,
    scan_mask: NDArray[np.bool_] | None = None,
    fig_path=None,
) -> IceFlaggerParams:
    """Show which q-window peaks pass the sharpness and intensity gates, and return the params.

    The width ceilings are tried on peaks measured once with
    :meth:`~quantem.diffraction.polymer.BraggPeaksPolymer.measure_peak_widths`::

        peaks = bp.measure_peak_widths(params)
        params = selection_box(peaks, params, max_width_theta_deg=8.0)
        result = bp.detect_ice(params, peaks=peaks)

    Parameters
    ----------
    peaks : Vector
        Polar peaks with ``width_q`` and ``width_theta`` fields.
    params : IceFlaggerParams, optional
        Settings to start from; the width ceilings are replaced.
    max_width_q_invA, max_width_theta_deg : float, optional
        Ceilings to try; None leaves that width ungated.
    scan_mask : ndarray of bool, optional
        Scan positions to include, as in :func:`detect_ice`.
    fig_path : str or Path, optional
        Also save the figure here.

    Returns
    -------
    IceFlaggerParams
        ``params`` with the two ceilings set.
    """
    p = dataclasses.replace(
        IceFlaggerParams() if params is None else params,
        max_width_q_invA=max_width_q_invA,
        max_width_theta_deg=max_width_theta_deg,
    )
    fields = ["q", "intensity", "width_q", "width_theta"]
    missing = [f for f in fields if f not in peaks.fields]
    if missing:
        raise ValueError(
            f"peaks has no field(s) {missing}; measure them with measure_peak_widths."
        )
    data = peaks.select_fields(fields).numpy().astype(np.float64)
    analyzed = _analyzed_rows(peaks, scan_mask)
    cell = np.repeat(np.arange(int(np.prod(peaks.shape))), peaks.row_counts())
    threshold = _intensity_threshold(data[analyzed, 1], p)
    band = analyzed & (np.abs(data[:, 0] - p.q_target_invA) <= p.dq_invA)
    intensity, width_q, width_theta = (
        data[band, 1],
        data[band, 2],
        np.rad2deg(data[band, 3]),
    )
    selected = intensity >= threshold
    if p.max_width_q_invA is not None:
        selected &= width_q <= p.max_width_q_invA
    if p.max_width_theta_deg is not None:
        selected &= width_theta <= p.max_width_theta_deg

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))

    def density(ax, x, y, x_max, y_min, y_max, xlabel, ylabel):
        finite = np.isfinite(x) & np.isfinite(y)
        hist = ax.hist2d(x[finite], y[finite], bins=80, norm=LogNorm(), cmap="magma")
        fig.colorbar(hist[3], ax=ax, label="peaks (log scale)")
        (x_lo, x_hi), (y_lo, y_hi) = ax.get_xlim(), ax.get_ylim()
        right = x_hi if x_max is None else x_max
        bottom = y_lo if y_min is None else y_min
        top = y_hi if y_max is None else y_max
        ax.add_patch(
            Rectangle((x_lo, bottom), right - x_lo, top - bottom, facecolor="cyan", alpha=0.15)
        )
        for value, line in (
            (x_max, ax.axvline),
            (y_min, ax.axhline),
            (y_max, ax.axhline),
        ):
            if value is not None:
                line(value, color="cyan", ls="--", lw=2)
        ax.set(xlabel=xlabel, ylabel=ylabel)

    floor = threshold if np.isfinite(threshold) else None
    density(
        axes[0, 0],
        width_q,
        intensity,
        p.max_width_q_invA,
        floor,
        None,
        "radial FWHM (1/Å)",
        "intensity",
    )
    density(
        axes[0, 1],
        width_theta,
        intensity,
        p.max_width_theta_deg,
        floor,
        None,
        "annular FWHM (deg)",
        "intensity",
    )
    bright = intensity >= threshold
    density(
        axes[1, 0],
        width_q[bright],
        width_theta[bright],
        p.max_width_q_invA,
        None,
        p.max_width_theta_deg,
        "radial FWHM (1/Å)",
        "annular FWHM (deg)",
    )
    axes[1, 0].set_title(f"intensity >= {threshold:.3g}")

    counts = np.bincount(cell[band][selected], minlength=int(np.prod(peaks.shape)))
    image = axes[1, 1].imshow(counts.reshape(peaks.shape), cmap="inferno", interpolation="nearest")
    fig.colorbar(image, ax=axes[1, 1], label="selected peaks")
    axes[1, 1].set_title("selected peaks across the scan")
    axes[1, 1].axis("off")
    fig.suptitle(f"{selected.sum()} of {band.sum()} peaks in the q window pass both gates")
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches="tight")
    plt.show()
    return p


def orientation_histograms(
    peaks: Vector,
    result: IceDetectionResult,
    params: IceFlaggerParams | None = None,
    theta_step_deg: float = 2.0,
    sigma_theta: float = 3.0,
    fig_path=None,
) -> dict[str, NDArray]:
    """Orientation histograms in the ice q window for all, kept and removed peaks.

    Ice shows as compact spots on the scan and narrow spikes in theta; polymer
    signal is spread out and broad, so it should stay in the kept panel.

    Parameters
    ----------
    peaks : Vector
        The polar peaks that ``result`` was computed from.
    result : IceDetectionResult
        Output of :func:`detect_ice`.
    params : IceFlaggerParams, optional
        Gives the q window.
    theta_step_deg, sigma_theta : float
        Theta bin size and smoothing in degrees.
    fig_path : str or Path, optional
        Also save the figure here.

    Returns
    -------
    dict
        ``(scan_row, scan_col, num_theta)`` histograms keyed ``"all"``, ``"kept"``
        and ``"removed"``.
    """
    from quantem.diffraction.polymer_orientation import make_orientation_histogram

    p = IceFlaggerParams() if params is None else params
    window = (p.q_target_invA - p.dq_invA, p.q_target_invA + p.dq_invA)
    subsets = {
        "all": peaks,
        "kept": result.filter(peaks),
        "removed": result.filter(peaks, invert=True),
    }
    hists = {
        label: make_orientation_histogram(
            subset,
            window,
            upsample_factor=1,
            theta_step_deg=theta_step_deg,
            sigma_x=0,
            sigma_y=0,
            sigma_theta=sigma_theta,
            normalize=None,
        )[0]
        for label, subset in subsets.items()
    }

    theta = np.arange(0, 180, theta_step_deg)
    vmax = max(float(h.max()) for h in hists.values()) or 1.0
    ymax = max(float(h.sum(axis=(0, 1)).max()) for h in hists.values()) or 1.0
    fig, axes = plt.subplots(3, 2, figsize=(9, 9.6), gridspec_kw={"width_ratios": [1, 1.3]})
    for (label, hist), (ax_map, ax_theta) in zip(hists.items(), axes):
        image = ax_map.imshow(
            hist.max(axis=2), cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest"
        )
        fig.colorbar(image, ax=ax_map, fraction=0.046)
        ax_map.set_title(f"{label}: maximum over theta", fontsize=9)
        ax_map.axis("off")
        ax_theta.plot(theta, hist.sum(axis=(0, 1)), lw=1.2)
        ax_theta.set(
            xlim=(0, 180),
            ylim=(0, 1.05 * ymax),
            xlabel="theta (deg)",
            ylabel="summed intensity",
        )
        ax_theta.set_title(f"{label}: summed over the scan", fontsize=9)
    fig.suptitle(
        f"q = {window[0]:.3f}-{window[1]:.3f} 1/Å (d = {1 / window[1]:.2f}-{1 / window[0]:.2f} Å)",
        fontsize=10,
    )
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches="tight")
    plt.show()
    return hists
