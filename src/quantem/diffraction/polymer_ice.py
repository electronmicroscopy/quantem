"""Ice-peak detection for polymer diffraction analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib.colors import LogNorm
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


@dataclass(frozen=True)
class IceFlaggerDebug:
    n_peaks_total: int
    n_candidates_q: int
    n_candidates_q_int: int
    intensity_threshold_used: float
    best_phi_deg: float | None
    matched_bins: list[int]
    matched_peak_indices: list[int]


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


def _angle_distance(angles: NDArray[np.floating], target: float) -> NDArray[np.floating]:
    delta = np.abs(np.mod(angles, 360.0) - np.mod(target, 360.0))
    return np.minimum(delta, 360.0 - delta)


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
):
    """Flag peaks belonging to an aligned, possibly incomplete six-fold ice pattern."""

    radius = np.asarray(r_invA, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)
    intensity = np.asarray(intensities, dtype=float)
    if radius.shape != theta.shape or radius.shape != intensity.shape:
        raise ValueError("r_invA, theta_rad, and intensities must have the same shape.")

    q_candidates = np.isfinite(radius) & (
        np.abs(radius - params.q_target_invA) <= params.dq_invA
    )
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
    phi = None
    bins: list[int] = []
    matched: list[int] = []
    if len(candidate_indices):
        angles = np.mod(np.rad2deg(theta[candidate_indices]), 360.0)
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
        expected = phi + 60.0 * np.arange(6)
        errors = np.stack([_angle_distance(angles, value) for value in expected], axis=1)
        closest = np.argmin(errors, axis=1)
        aligned = errors[np.arange(len(angles)), closest] <= params.dtheta_deg
        bins = sorted(set(closest[aligned].astype(int).tolist()))
        if len(bins) >= params.min_matches:
            matched = candidate_indices[aligned].astype(int).tolist()
            result[matched] = True
            if not params.conservative:
                q_indices = np.flatnonzero(q_candidates)
                q_angles = np.mod(np.rad2deg(theta[q_indices]), 360.0)
                q_errors = np.stack(
                    [_angle_distance(q_angles, value) for value in expected], axis=1
                )
                result[q_indices[np.min(q_errors, axis=1) <= params.dtheta_deg]] = True
                matched = np.flatnonzero(result).astype(int).tolist()

    debug = IceFlaggerDebug(
        n_peaks_total=int(radius.size),
        n_candidates_q=int(np.count_nonzero(q_candidates)),
        n_candidates_q_int=int(len(candidate_indices)),
        intensity_threshold_used=threshold,
        best_phi_deg=phi,
        matched_bins=bins,
        matched_peak_indices=matched,
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
) -> IceDetectionResult:
    """Detect ice peaks across aligned ragged peak and intensity vectors."""

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
    q_window_alpha=0.35,
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
    if q_value is not None:
        ax.axvline(q_value, color=q_value_color, lw=q_value_lw, ls=":")
        if q_window is not None:
            for edge in (q_value - q_window, q_value + q_window):
                ax.axvline(
                    edge,
                    color=q_window_color,
                    lw=q_window_lw,
                    ls=":",
                    alpha=q_window_alpha,
                )
    if cutoff is not None:
        if cutoff_mode == "absolute":
            level, label = float(cutoff), f"cutoff={float(cutoff):.3g}"
        elif cutoff_mode == "percentile":
            level = float(np.percentile(values, cutoff))
            label = f"p{float(cutoff):g}={level:.3g}"
        else:
            raise ValueError("cutoff_mode must be 'absolute' or 'percentile'.")
        ax.axhline(level, color=cutoff_color, lw=2, ls="--")
        ax.text(ax.get_xlim()[0], level, " " + label, color=cutoff_color, va="bottom")
    fig.tight_layout()
    return fig, ax


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
    "compute_global_intensity_threshold",
    "detect_ice",
    "flag_ice_peaks_in_dataset",
    "flag_ice_peaks_in_pattern",
    "plot_q_intensity_density",
]
