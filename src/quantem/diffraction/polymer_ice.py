"""Ice-peak detection for polymer diffraction analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from quantem.core.datastructures import Vector


@dataclass(frozen=True)
class IceFlaggerParams:
    q_target_invA: float = 1.61
    dq_invA: float = 0.05
    dtheta_deg: float = 6.0
    min_matches: int = 2
    intensity_field: str = "intensities"
    intensity_percentile_global: float = 99.0
    intensity_cutoff: float | None = None
    intensity_cutoff_mode: Literal["absolute", "percentile"] = "absolute"
    conservative: bool = True

    # --- Sharpness gate ---------------------------------------------------
    # Ice reflections are small and sharp; polymer signal is a larger dot or a
    # broad diffuse region. Widths are full-width-at-half-maximum measured on
    # the polar intensity volume at each candidate's (r, theta), radially in
    # 1/A and annularly in degrees. Both ceilings default to None, which
    # disables the gate entirely and reproduces the previous behaviour.
    max_width_r_invA: float | None = None
    max_width_theta_deg: float | None = None
    # "both"   -- a candidate must be sharp radially AND annularly (compact dots).
    # "either" -- sharp in one direction is enough, which also keeps the thin
    #             streaks that are narrow across their length but not along it.
    sharpness_mode: Literal["both", "either"] = "both"
    # Half-width of the search window used to measure each FWHM. A candidate
    # whose profile never falls to half maximum inside the window is reported
    # as wider than the window, i.e. broad, and is rejected.
    sharpness_window_r_invA: float = 0.06
    sharpness_window_theta_deg: float = 40.0
    # Local background level, as a quantile of the windowed profile. The half
    # maximum is taken above this, so a peak riding on the amorphous ring is
    # measured against the ring rather than against zero.
    sharpness_baseline_quantile: float = 0.25
    # Bins of local argmax refinement, to absorb the sub-bin offset between a
    # detected peak position and the polar volume's sampling grid.
    sharpness_refine_bins: int = 2

    # --- Multiple crystallites -------------------------------------------
    # One pattern can contain several ice crystallites at unrelated orientations,
    # giving overlaid six-fold lattices. Each pass claims the best-supported
    # lattice and peels its peaks away before looking again. 1 keeps the previous
    # single-lattice behaviour.
    max_crystallites: int = 1
    # How far apart two lattices' phi must be, in degrees on the 0-60 wedge, to
    # count as separate crystallites. None uses dtheta_deg, i.e. lattices that the
    # matcher could not tell apart anyway are not treated as distinct.
    min_phi_separation_deg: float | None = None
    # Angular period of the peak thetas, in degrees. process_polar(two_fold_symmetry=True)
    # folds theta to [0, 180), collapsing every Friedel pair onto one angle, so only three
    # of the six lattice arms are distinguishable and min_matches cannot exceed 3. None lets
    # detect_ice read it off the BraggPeaksPolymer; set 360.0 or 180.0 to force it.
    theta_period_deg: float | None = None
    # Peaks an arm must carry to count towards min_matches. On a folded theta axis a
    # Friedel pair (theta and theta+180) lands on one arm as two peaks, while an
    # isolated reflection lands as one -- so 2 demands a pair and rejects lone peaks.
    # Caveat: two peaks that merely fall within dtheta_deg/dq_invA of each other also
    # satisfy it; folding makes them indistinguishable from a true opposed pair.
    min_peaks_per_arm: int = 1


@dataclass(frozen=True)
class IceFlaggerDebug:
    n_peaks_total: int
    n_candidates_q: int
    n_candidates_q_int: int
    intensity_threshold_used: float
    best_phi_deg: float | None
    matched_bins: list[int]
    matched_peak_indices: list[int]
    n_candidates_sharp: int | None = None
    # One entry per crystallite found, strongest first. ``best_phi_deg`` is the
    # first of these, retained so single-lattice call sites keep working.
    phi_deg: list[float] | None = None


@dataclass(frozen=True)
class IceDetectionResult:
    """Ice mask and diagnostics produced for an entire scan."""

    mask: Vector
    flagged_peaks_count_map: NDArray[np.integer]
    matched_bins_count_map: NDArray[np.integer]
    intensity_threshold: float
    debug_records: dict[tuple[int, int], IceFlaggerDebug] | None = None

    @property
    def ice_mask(self) -> Vector:
        """Alias retained for discoverability."""
        return self.mask

    @property
    def threshold(self) -> float:
        return self.intensity_threshold

    @property
    def flagged_count_map(self) -> NDArray[np.integer]:
        return self.flagged_peaks_count_map

    @property
    def matched_count_map(self) -> NDArray[np.integer]:
        return self.matched_bins_count_map

    def filter(self, vector: Vector, *, invert: bool = False) -> Vector:
        """Return a filtered copy; the source vector is never mutated."""

        if vector.shape != self.mask.shape:
            raise ValueError(
                f"shape mismatch: vector.shape={vector.shape} vs mask.shape={self.mask.shape}"
            )
        out = vector.copy()
        for iy in range(vector.shape[0]):
            for ix in range(vector.shape[1]):
                source = vector[iy, ix].array
                mask_cell = self.mask[iy, ix].array
                if source is None or len(source) == 0 or mask_cell is None:
                    continue
                flags = np.asarray(mask_cell)[:, 0].astype(bool, copy=False)
                if len(flags) != len(source):
                    raise ValueError(
                        f"Row count mismatch at ({iy},{ix}): vector has {len(source)} "
                        f"rows but ice mask has {len(flags)}."
                    )
                out[iy, ix] = source[flags if invert else ~flags]
        return out


def _half_width_bins(profile: NDArray[np.floating], center: int, direction: int, half: float) -> float:
    """Bins from ``center`` to where ``profile`` first falls to ``half``, interpolated.

    Returns the window half-length when no crossing is found, so a profile that
    never comes back down reads as at least as broad as the window.
    """

    previous = float(profile[center])
    for step in range(1, len(profile)):
        index = center + direction * step
        if index < 0 or index >= len(profile):
            return float(step - 1)
        value = float(profile[index])
        if not np.isfinite(value) or value <= half:
            span = previous - value
            fraction = (previous - half) / span if span > 0 else 0.0
            return (step - 1) + float(np.clip(fraction, 0.0, 1.0))
        previous = value
    return float(len(profile))


def _profile_fwhm(
    profile: NDArray[np.floating], center: int, *, baseline_quantile: float, refine_bins: int
) -> tuple[float, int]:
    """FWHM of ``profile`` in bins about ``center``, plus the refined peak bin.

    The profile is a window already cut out of the polar volume, so running off
    its end means "wider than the window" rather than "edge of the detector".
    """

    finite = profile[np.isfinite(profile)]
    if not len(finite):
        return float("inf"), center
    if refine_bins > 0:
        low = max(0, center - refine_bins)
        high = min(len(profile), center + refine_bins + 1)
        center = low + int(np.nanargmax(profile[low:high]))
    peak = float(profile[center])
    baseline = float(np.quantile(finite, baseline_quantile))
    if not np.isfinite(peak) or peak <= baseline:
        return float("inf"), center
    half = baseline + 0.5 * (peak - baseline)
    left = _half_width_bins(profile, center, -1, half)
    right = _half_width_bins(profile, center, +1, half)
    return left + right, center


def measure_peak_widths(
    r_invA,
    theta_rad,
    polar_intensity: NDArray[np.floating],
    r_axis: NDArray[np.floating],
    theta_axis: NDArray[np.floating],
    *,
    params: IceFlaggerParams = IceFlaggerParams(),
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Radial (1/Å) and annular (degrees) FWHM for each peak of one pattern.

    ``polar_intensity`` is that pattern's polar transform, indexed
    ``[radial_bin, annular_bin]``; ``r_axis`` and ``theta_axis`` are its
    coordinate axes (1/Å and radians). The annular axis is treated as periodic,
    the radial axis is not. Peaks that fall outside the sampled radial range
    get ``inf``, so they never pass a sharpness ceiling.
    """

    radius = np.asarray(r_invA, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)
    width_r = np.full(radius.shape, np.inf)
    width_theta = np.full(radius.shape, np.inf)
    if not radius.size or polar_intensity.size == 0 or len(r_axis) < 2 or len(theta_axis) < 2:
        return width_r, width_theta

    r_step = float(r_axis[1] - r_axis[0])
    theta_step_deg = float(np.rad2deg(theta_axis[1] - theta_axis[0]))
    n_r, n_theta = polar_intensity.shape
    # Window half-widths in bins; at least 2 so a FWHM is measurable at all.
    window_r = max(2, int(np.ceil(params.sharpness_window_r_invA / max(r_step, 1e-12))))
    window_theta = max(2, int(np.ceil(params.sharpness_window_theta_deg / max(theta_step_deg, 1e-12))))
    theta_period = float(theta_axis[-1] - theta_axis[0]) + (theta_axis[1] - theta_axis[0])

    for index in range(radius.size):
        if not (np.isfinite(radius[index]) and np.isfinite(theta[index])):
            continue
        r_bin = int(np.round((radius[index] - r_axis[0]) / r_step))
        if not 0 <= r_bin < n_r:
            continue
        theta_bin = int(np.round(np.mod(theta[index], theta_period) / (theta_period / n_theta))) % n_theta

        # Annular cut first: it is periodic, so the window is always full length
        # and the refined bin it returns anchors the radial cut.
        theta_indices = np.mod(np.arange(theta_bin - window_theta, theta_bin + window_theta + 1), n_theta)
        annular = polar_intensity[r_bin, theta_indices]
        fwhm_theta, refined = _profile_fwhm(
            annular,
            window_theta,
            baseline_quantile=params.sharpness_baseline_quantile,
            refine_bins=params.sharpness_refine_bins,
        )
        theta_bin = int(theta_indices[min(refined, len(theta_indices) - 1)])

        low = max(0, r_bin - window_r)
        radial = polar_intensity[low : min(n_r, r_bin + window_r + 1), theta_bin]
        fwhm_r, _ = _profile_fwhm(
            radial,
            r_bin - low,
            baseline_quantile=params.sharpness_baseline_quantile,
            refine_bins=params.sharpness_refine_bins,
        )
        width_r[index] = fwhm_r * r_step
        width_theta[index] = fwhm_theta * theta_step_deg
    return width_r, width_theta


def _resolve_theta_period(params: IceFlaggerParams, fallback: float | None) -> float:
    """Angular period of the peak thetas: explicit params win, then the caller's value."""

    if params.theta_period_deg is not None:
        return float(params.theta_period_deg)
    return 360.0 if fallback is None else float(fallback)


def _sharpness_enabled(params: IceFlaggerParams) -> bool:
    return params.max_width_r_invA is not None or params.max_width_theta_deg is not None


def sharpness_mask(
    width_r: NDArray[np.floating], width_theta: NDArray[np.floating], params: IceFlaggerParams
) -> NDArray[np.bool_]:
    """Which peaks pass the configured width ceilings.

    Public so a tuning preview can apply exactly the gate the flagger applies,
    rather than reimplementing it. Non-finite widths fail any ceiling that is set,
    and pass an axis with no ceiling.
    """

    radial_ok = (
        np.ones(width_r.shape, dtype=bool)
        if params.max_width_r_invA is None
        else width_r <= params.max_width_r_invA
    )
    annular_ok = (
        np.ones(width_theta.shape, dtype=bool)
        if params.max_width_theta_deg is None
        else width_theta <= params.max_width_theta_deg
    )
    if params.sharpness_mode == "both":
        return radial_ok & annular_ok
    if params.sharpness_mode == "either":
        # With only one ceiling set, "either" would pass everything through the
        # unset direction; fall back to the ceiling that was actually given.
        if params.max_width_r_invA is None:
            return annular_ok
        if params.max_width_theta_deg is None:
            return radial_ok
        return radial_ok | annular_ok
    raise ValueError("sharpness_mode must be 'both' or 'either'.")


def _angle_distance(
    angles: NDArray[np.floating], target: float, period: float = 360.0
) -> NDArray[np.floating]:
    """Separation on a circle of circumference ``period`` degrees.

    ``period`` is 180 when the polar transform folded theta with two-fold
    symmetry, which maps every Friedel pair onto a single angle.
    """

    delta = np.abs(np.mod(angles, period) - np.mod(target, period))
    return np.minimum(delta, period - delta)


def _phi_distance(first: float, second: float) -> float:
    """Separation of two six-fold orientations, which live on a 0-60 degree wedge."""

    delta = abs(np.mod(first, 60.0) - np.mod(second, 60.0))
    return float(min(delta, 60.0 - delta))


def _global_threshold(
    intensities: Vector, field: str, percentile: float, scan_mask: NDArray[np.bool_]
) -> float:
    index = intensities.fields.index(field)
    values = []
    for iy, ix in np.argwhere(scan_mask):
        cell = intensities[int(iy), int(ix)].array
        if cell is not None and len(cell):
            finite = np.asarray(cell)[:, index]
            finite = finite[np.isfinite(finite)]
            if len(finite):
                values.append(finite)
    return float(np.percentile(np.concatenate(values), percentile)) if values else float("inf")


def compute_global_intensity_threshold(
    peak_intensities: Vector,
    intensity_field: str = "intensities",
    percentile: float = 99.0,
    scan_mask=None,
) -> float:
    """Compute a scan-wide intensity percentile for ice candidate selection."""

    if intensity_field not in peak_intensities.fields:
        raise KeyError(
            f"Intensity field {intensity_field!r} is absent from peak_intensities."
        )
    selected = (
        np.ones(peak_intensities.shape, dtype=bool)
        if scan_mask is None
        else np.asarray(scan_mask, dtype=bool)
    )
    if selected.shape != peak_intensities.shape:
        raise ValueError(
            f"scan_mask shape {selected.shape} must match {peak_intensities.shape}."
        )
    return _global_threshold(
        peak_intensities, intensity_field, percentile, selected
    )


def flag_ice_peaks_in_pattern(
    r_invA,
    theta_rad,
    intensities,
    *,
    params: IceFlaggerParams,
    intensity_threshold_global: float,
    return_debug: bool = True,
    polar_intensity: NDArray[np.floating] | None = None,
    r_axis: NDArray[np.floating] | None = None,
    theta_axis: NDArray[np.floating] | None = None,
    theta_period_deg: float | None = None,
):
    """Flag peaks belonging to an aligned, possibly incomplete six-fold ice pattern.

    ``polar_intensity`` / ``r_axis`` / ``theta_axis`` are this pattern's polar
    transform and its coordinate axes. They are required only when ``params``
    sets a sharpness ceiling, which is measured from that volume.
    """

    radius = np.asarray(r_invA, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)
    intensity = np.asarray(intensities, dtype=float)
    if radius.shape != theta.shape or radius.shape != intensity.shape:
        raise ValueError("r_invA, theta_rad, and intensities must have the same shape.")

    q_candidates = np.isfinite(radius) & (
        np.abs(radius - params.q_target_invA) <= params.dq_invA
    )

    # Sharpness gate. Applied to the q band before the six-fold search, so the
    # broad polymer peaks neither get flagged nor drag the phi estimate around.
    n_sharp = None
    if _sharpness_enabled(params):
        if polar_intensity is None or r_axis is None or theta_axis is None:
            raise ValueError(
                "A sharpness ceiling (max_width_r_invA / max_width_theta_deg) requires the "
                "polar intensity volume; pass polar_data through detect_ice()."
            )
        width_r, width_theta = measure_peak_widths(
            radius, theta, polar_intensity, r_axis, theta_axis, params=params
        )
        q_candidates &= sharpness_mask(width_r, width_theta, params)
        n_sharp = int(np.count_nonzero(q_candidates))
    if params.intensity_cutoff is None:
        threshold = float(intensity_threshold_global)
    elif params.intensity_cutoff_mode == "absolute":
        threshold = float(params.intensity_cutoff)
    elif params.intensity_cutoff_mode == "percentile":
        finite = intensity[np.isfinite(intensity)]
        threshold = (
            float(np.percentile(finite, params.intensity_cutoff))
            if len(finite)
            else float("inf")
        )
    else:
        raise ValueError("intensity_cutoff_mode must be 'absolute' or 'percentile'.")

    candidate_indices = np.flatnonzero(
        q_candidates & np.isfinite(intensity) & (intensity >= threshold)
    )
    result = np.zeros(radius.shape, dtype=bool)
    bins: list[int] = []
    matched: list[int] = []
    phis: list[float] = []

    # Greedy peel: fit the best-supported six-fold lattice, claim its peaks, remove
    # them, and look again in what is left. A pattern can contain several ice
    # crystallites at unrelated orientations, and one pass only ever sees the
    # strongest. max_crystallites=1 reproduces the single-lattice behaviour.
    separation = (
        params.dtheta_deg
        if params.min_phi_separation_deg is None
        else params.min_phi_separation_deg
    )
    # A folded theta axis (period 180) makes only three of the six arms distinguishable,
    # because each arm and its Friedel partner share one angle.
    period = _resolve_theta_period(params, theta_period_deg)
    n_arms = max(1, int(round(period / 60.0)))
    remaining = candidate_indices
    for _ in range(max(1, params.max_crystallites)):
        if not len(remaining):
            break
        angles = np.mod(np.rad2deg(theta[remaining]), period)
        modulo = np.mod(angles, 60.0)
        supports = [
            _angle_distance(modulo, float(center)) <= params.dtheta_deg
            for center in modulo
        ]
        inliers = supports[int(np.argmax([np.count_nonzero(x) for x in supports]))]
        radians = np.deg2rad(modulo[inliers])
        phi = float(
            np.mod(np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))), 60)
        )
        # A lattice indistinguishable from one already claimed means the leftovers
        # are stragglers of it, not a new crystallite. Stop rather than double-count.
        if any(_phi_distance(phi, previous) < separation for previous in phis):
            break

        expected = phi + 60.0 * np.arange(n_arms)
        errors = np.stack(
            [_angle_distance(angles, value, period) for value in expected], axis=1
        )
        closest = np.argmin(errors, axis=1)
        aligned = errors[np.arange(len(angles)), closest] <= params.dtheta_deg
        # Keep only arms carrying enough peaks, then re-restrict the matched set to them.
        arm_counts = np.bincount(closest[aligned].astype(int), minlength=n_arms)
        good_arms = np.flatnonzero(arm_counts >= params.min_peaks_per_arm)
        if len(good_arms) < params.min_matches:
            break
        aligned &= np.isin(closest, good_arms)
        lattice_bins = sorted(good_arms.tolist())

        result[remaining[aligned]] = True
        if not params.conservative:
            # Sweep in sub-threshold peaks of the q band that sit on this lattice.
            q_indices = np.flatnonzero(q_candidates & ~result)
            if len(q_indices):
                q_angles = np.mod(np.rad2deg(theta[q_indices]), period)
                q_errors = np.stack(
                    [_angle_distance(q_angles, value, period) for value in expected[good_arms]],
                    axis=1,
                )
                result[q_indices[np.min(q_errors, axis=1) <= params.dtheta_deg]] = True
        phis.append(phi)
        bins.extend(lattice_bins)
        remaining = remaining[~aligned]
    matched = np.flatnonzero(result).astype(int).tolist()

    debug = IceFlaggerDebug(
        n_peaks_total=int(radius.size),
        n_candidates_q=int(np.count_nonzero(q_candidates)),
        n_candidates_q_int=int(len(candidate_indices)),
        intensity_threshold_used=threshold,
        best_phi_deg=phis[0] if phis else None,
        matched_bins=bins,
        matched_peak_indices=matched,
        n_candidates_sharp=n_sharp,
        phi_deg=phis,
    )
    return result, debug if return_debug else None


def detect_ice(
    polar_peaks: Vector,
    peak_intensities: Vector,
    *,
    params: IceFlaggerParams = IceFlaggerParams(),
    scan_mask=None,
    intensity_threshold_global: float | None = None,
    return_debug: bool = False,
    polar_data: dict | None = None,
    theta_period_deg: float | None = None,
) -> IceDetectionResult:
    """Detect ice peaks across aligned ragged peak and intensity vectors.

    ``polar_data`` is the polar transform dict produced by ``process_polar``
    (keys ``intensity``, ``r_invA``, ``theta``). It is required only when
    ``params`` sets a sharpness ceiling.
    """

    if polar_peaks.shape != peak_intensities.shape:
        raise ValueError("polar_peaks and peak_intensities must have matching shapes.")
    for field in ("r_invA", "theta"):
        if field not in polar_peaks.fields:
            raise KeyError(f"Required field {field!r} is absent from polar_peaks.")
    if params.intensity_field not in peak_intensities.fields:
        raise KeyError(
            f"Intensity field {params.intensity_field!r} is absent from peak_intensities."
        )
    shape = polar_peaks.shape
    selected = np.ones(shape, dtype=bool) if scan_mask is None else np.asarray(scan_mask, bool)
    if selected.shape != shape:
        raise ValueError(f"scan_mask shape {selected.shape} must match {shape}.")
    if params.intensity_cutoff is None:
        threshold = (
            compute_global_intensity_threshold(
                peak_intensities,
                intensity_field=params.intensity_field,
                percentile=params.intensity_percentile_global,
                scan_mask=selected,
            )
            if intensity_threshold_global is None
            else float(intensity_threshold_global)
        )
    elif params.intensity_cutoff_mode == "absolute":
        threshold = float(params.intensity_cutoff)
    else:
        threshold = float("nan")

    # A folded theta axis collapses Friedel pairs, so only period/60 arms are
    # distinguishable. Catch an unsatisfiable min_matches here rather than letting
    # every pattern silently fail to match.
    period = _resolve_theta_period(params, theta_period_deg)
    n_arms = max(1, int(round(period / 60.0)))
    if params.min_matches > n_arms:
        raise ValueError(
            f"min_matches={params.min_matches} can never be reached: theta has period "
            f"{period:g} degrees, which leaves only {n_arms} distinguishable six-fold arms. "
            "process_polar(two_fold_symmetry=True) folds theta to [0, 180), mapping each "
            f"Friedel pair onto one angle. Use min_matches <= {n_arms}."
        )

    polar_intensity_stack = r_axis = theta_axis = None
    if _sharpness_enabled(params):
        if polar_data is None:
            raise ValueError(
                "A sharpness ceiling (max_width_r_invA / max_width_theta_deg) requires "
                "polar_data; run process_polar() first, or clear the ceilings."
            )
        polar_intensity_stack = np.asarray(polar_data["intensity"])
        if polar_intensity_stack.shape[:2] != shape:
            raise ValueError(
                f"polar_data intensity has scan shape {polar_intensity_stack.shape[:2]}, "
                f"which must match {shape}."
            )
        # process_polar stores the coordinate grids as [radial_bin, annular_bin] meshes.
        r_axis = np.asarray(polar_data["r_invA"])[:, 0]
        theta_axis = np.asarray(polar_data["theta"])[0, :]

    mask = Vector.from_shape(shape=shape, fields=["is_ice"], units=["bool"], name="ice_peak_mask")
    flagged = np.zeros(shape, dtype=int)
    matched_bins = np.zeros(shape, dtype=int)
    records = {} if return_debug else None
    r_index = polar_peaks.fields.index("r_invA")
    theta_index = polar_peaks.fields.index("theta")
    intensity_index = peak_intensities.fields.index(params.intensity_field)
    for iy, ix in np.argwhere(selected):
        iy, ix = int(iy), int(ix)
        polar_cell = polar_peaks[iy, ix].array
        intensity_cell = peak_intensities[iy, ix].array
        if polar_cell is None or intensity_cell is None:
            continue
        if len(polar_cell) != len(intensity_cell):
            raise ValueError(
                f"Row count mismatch at ({iy},{ix}): polar peaks have {len(polar_cell)} "
                f"rows and intensities have {len(intensity_cell)}."
            )
        flags, debug = flag_ice_peaks_in_pattern(
            np.asarray(polar_cell)[:, r_index],
            np.asarray(polar_cell)[:, theta_index],
            np.asarray(intensity_cell)[:, intensity_index],
            params=params,
            intensity_threshold_global=threshold,
            return_debug=return_debug,
            polar_intensity=None if polar_intensity_stack is None else polar_intensity_stack[iy, ix],
            r_axis=r_axis,
            theta_axis=theta_axis,
            theta_period_deg=period,
        )
        if len(flags):
            mask[iy, ix] = flags[:, None]
        flagged[iy, ix] = np.count_nonzero(flags)
        if debug is not None:
            matched_bins[iy, ix] = len(debug.matched_bins)
            records[(iy, ix)] = debug
    return IceDetectionResult(mask, flagged, matched_bins, threshold, records)


def plot_q_intensity_density(
    polar_peaks: Vector,
    peak_intensities: Vector,
    *,
    q_field="r_invA",
    intensity_field="intensities",
    q_bins=250,
    i_bins=200,
    q_range=None,
    q_max=0.5,
    cutoff=None,
    cutoff_mode="absolute",
    cutoff_color="cyan",
    q_value=None,
    q_window=None,
    q_value_color="cyan",
    q_window_color="cyan",
    q_window_alpha=0.18,
    q_value_lw=2.0,
    q_window_lw=1.5,
):
    """Plot q versus intensity density for aligned ragged peak vectors."""

    import matplotlib.pyplot as plt

    q_index = polar_peaks.fields.index(q_field)
    intensity_index = peak_intensities.fields.index(intensity_field)
    qs, values = [], []
    for iy in range(polar_peaks.shape[0]):
        for ix in range(polar_peaks.shape[1]):
            q_cell = polar_peaks[iy, ix].array
            i_cell = peak_intensities[iy, ix].array
            if q_cell is None or i_cell is None:
                continue
            if len(q_cell) != len(i_cell):
                raise ValueError(f"Row count mismatch at ({iy},{ix}).")
            q = np.asarray(q_cell)[:, q_index]
            intensity = np.asarray(i_cell)[:, intensity_index]
            valid = np.isfinite(q) & np.isfinite(intensity) & (intensity >= 0) & (intensity <= 1)
            qs.extend(q[valid])
            values.extend(intensity[valid])
    if not qs:
        raise ValueError("No valid (q, intensity) pairs found.")
    qs = np.asarray(qs)
    values = np.asarray(values)
    fig, ax = plt.subplots(figsize=(8, 4))
    histogram = ax.hist2d(
        qs,
        values,
        bins=(q_bins, i_bins),
        range=((0, q_max) if q_range is None else q_range, (0, 1)),
        norm=LogNorm(),
        cmap="magma",
    )
    ax.set(xlabel="q (1/Å)", ylabel=intensity_field, ylim=(0, 1))
    fig.colorbar(histogram[3], ax=ax, label="count (log colormap)")
    if q_window is not None and q_value is None:
        raise ValueError("q_window requires q_value.")

    # Resolve the intensity floor first: it is the bottom edge of the shaded region.
    level = label = None
    if cutoff is not None:
        if cutoff_mode == "absolute":
            level, label = float(cutoff), f"cutoff={float(cutoff):.3g}"
        elif cutoff_mode == "percentile":
            level = float(np.percentile(values, cutoff))
            label = f"p{float(cutoff):g}={level:.3g}"
        else:
            raise ValueError("cutoff_mode must be 'absolute' or 'percentile'.")

    if q_value is not None:
        if q_window is not None:
            # Shade the candidate region itself -- the q window, above the intensity
            # floor -- rather than drawing bare edge lines. Cyan reads cleanly on magma.
            bottom = 0.0 if level is None else level
            top = ax.get_ylim()[1]
            ax.add_patch(
                Rectangle(
                    (q_value - q_window, bottom),
                    2.0 * q_window,
                    top - bottom,
                    facecolor=q_window_color,
                    alpha=q_window_alpha,
                    edgecolor=q_window_color,
                    lw=q_window_lw,
                    zorder=2,
                )
            )
        ax.axvline(q_value, color=q_value_color, lw=q_value_lw, ls=":", zorder=3)

    if level is not None:
        ax.axhline(level, color=cutoff_color, lw=2, ls="--", zorder=3)
        ax.text(ax.get_xlim()[0], level, " " + label, color=cutoff_color, va="bottom", zorder=3)
    fig.tight_layout()
    return fig, ax


def collect_peak_widths(
    polar_peaks: Vector,
    peak_intensities: Vector,
    polar_data: dict,
    *,
    params: IceFlaggerParams = IceFlaggerParams(),
    scan_mask=None,
    q_band_only: bool = True,
) -> dict[str, NDArray]:
    """Measure every peak's radial/annular width, flattened across the scan.

    This is the tuning counterpart to the sharpness ceilings: histogram
    ``width_r_invA`` against ``width_theta_deg`` to see where the sharp ice
    population separates from the broad polymer one, then set
    ``max_width_r_invA`` / ``max_width_theta_deg`` between them.

    With ``q_band_only`` the measurement is restricted to the flagger's q window,
    which is both far cheaper and the only population the gate ever sees.
    Returns flat arrays keyed ``iy``, ``ix``, ``q_invA``, ``theta_deg``,
    ``intensity``, ``width_r_invA``, ``width_theta_deg``.
    """

    shape = polar_peaks.shape
    selected = np.ones(shape, dtype=bool) if scan_mask is None else np.asarray(scan_mask, bool)
    if selected.shape != shape:
        raise ValueError(f"scan_mask shape {selected.shape} must match {shape}.")
    intensity_stack = np.asarray(polar_data["intensity"])
    r_axis = np.asarray(polar_data["r_invA"])[:, 0]
    theta_axis = np.asarray(polar_data["theta"])[0, :]
    r_index = polar_peaks.fields.index("r_invA")
    theta_index = polar_peaks.fields.index("theta")
    intensity_index = peak_intensities.fields.index(params.intensity_field)

    columns: dict[str, list] = {key: [] for key in
                                ("iy", "ix", "q_invA", "theta_deg", "intensity",
                                 "width_r_invA", "width_theta_deg")}
    for iy, ix in np.argwhere(selected):
        iy, ix = int(iy), int(ix)
        polar_cell = polar_peaks[iy, ix].array
        intensity_cell = peak_intensities[iy, ix].array
        if polar_cell is None or intensity_cell is None or not len(polar_cell):
            continue
        radius = np.asarray(polar_cell)[:, r_index]
        theta = np.asarray(polar_cell)[:, theta_index]
        values = np.asarray(intensity_cell)[:, intensity_index]
        keep = (
            np.isfinite(radius) & (np.abs(radius - params.q_target_invA) <= params.dq_invA)
            if q_band_only
            else np.isfinite(radius)
        )
        if not keep.any():
            continue
        radius, theta, values = radius[keep], theta[keep], values[keep]
        width_r, width_theta = measure_peak_widths(
            radius, theta, intensity_stack[iy, ix], r_axis, theta_axis, params=params
        )
        columns["iy"].extend([iy] * len(radius))
        columns["ix"].extend([ix] * len(radius))
        columns["q_invA"].extend(radius)
        columns["theta_deg"].extend(np.rad2deg(theta))
        columns["intensity"].extend(values)
        columns["width_r_invA"].extend(width_r)
        columns["width_theta_deg"].extend(width_theta)
    return {key: np.asarray(value) for key, value in columns.items()}


# Compatibility names used by existing analyses.
flag_ice_peaks_in_dataset = detect_ice


def apply_ice_mask_to_vector(vector: Vector, ice_mask_vector: Vector, *, invert=False) -> Vector:
    result = IceDetectionResult(
        ice_mask_vector,
        np.zeros(vector.shape, dtype=int),
        np.zeros(vector.shape, dtype=int),
        float("nan"),
    )
    return result.filter(vector, invert=invert)


__all__ = [
    "IceDetectionResult",
    "IceFlaggerDebug",
    "IceFlaggerParams",
    "apply_ice_mask_to_vector",
    "collect_peak_widths",
    "compute_global_intensity_threshold",
    "detect_ice",
    "flag_ice_peaks_in_dataset",
    "flag_ice_peaks_in_pattern",
    "measure_peak_widths",
    "sharpness_mask",
    "plot_q_intensity_density",
]
