"""Elliptical-distortion fitting for diffraction patterns.

Detector geometry makes a nominally circular diffraction ring elliptical. These
fitters measure that distortion from a single pattern so a polar transform can
undo it. They take a plain 2D array and a centre, so they apply to any diffuse
or crystalline ring -- amorphous halo, polymer scattering, powder ring -- and
need no acquisition- or material-specific object.

Two methods, both returning an :class:`EllipseFit`:

``fit_ellipse_from_ring``
    Angular-variance search (Karen Ehrhardt's). Holds the centre fixed and
    solves (b/a, theta) by minimising azimuthal variation in a radial band.

``fit_ellipse_from_ridge``
    Measures the ring radius independently at every azimuth after log
    compression and hot-spot clipping, then jointly refines the residual centre
    and the ellipse. Use when the centre itself is uncertain.

Both accept the fit only when held-out angular sectors improve over a circle;
``EllipseFit.accepted`` reports that, and the caller decides whether to apply a
rejected fit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from quantem.diffraction.polar_transform import polar_transform

__all__ = ["EllipseFit", "fit_ellipse_from_ridge", "fit_ellipse_from_ring"]


@dataclass(frozen=True)
class EllipseFit:
    """Result of an elliptical-distortion fit.

    ``a``/``b`` are the semi-axes and ``theta_deg`` the major-axis orientation;
    ``(a, b, theta_deg)`` is the triple the polar transforms consume, available
    as :attr:`params`. ``ring_band`` is the radial band the fit was measured
    over, in pixels.

    ``accepted`` is False when the fit failed validation against a plain circle.
    The values are still returned so the caller can inspect them, but applying
    them is then a deliberate choice.
    """

    a: float
    b: float
    theta_deg: float
    ring_band: tuple[float, float]
    method: str
    accepted: bool
    center_initial: tuple[float, float]
    center_refined: tuple[float, float]
    details: dict = field(default_factory=dict)

    @property
    def params(self) -> tuple[float, float, float]:
        """``(a, b, theta_deg)``, the form the polar transforms take."""
        return (self.a, self.b, self.theta_deg)

    @property
    def ratio(self) -> float:
        """Axis ratio a/b; 1.0 is a perfect circle."""
        return self.a / self.b if self.b else float("nan")

    @property
    def diagnostics(self) -> dict:
        """The flat diagnostics mapping, as ``BraggPeaksPolymer`` has always exposed it."""
        return {
            "method": self.method,
            "accepted": self.accepted,
            "center_initial": self.center_initial,
            "center_refined": self.center_refined,
            **self.details,
        }


def fit_ellipse_from_ring(
    dp,
    center,
    *,
    radial_min=None,
    radial_max=None,
    radial_step=1.0,
    num_annular_bins=180,
    ratio_range=(0.85, 1.18),
    n_ratio=12,
    n_theta=24,
    refine=True,
    max_ring_candidates=3,
    min_fit_improvement=0.005,
    max_fit_score=0.25,
    min_angular_coverage=0.55,
    device="cpu",
    show=False,
    verbose=False,
):
    """Fit ring ellipticity ``(a, b, theta_deg)`` by minimising the azimuthal variance
    of a diffraction-ring annulus at a FIXED center -- Karen Ehrhardt's angular-
    uniformity criterion (see ``quantem.diffraction.polar_transform``).

    Unlike a probe-blob fit (``fit_probe_ellipse``) this samples an annulus out at the
    ring radius and never touches the central beam, so a smeared / off-center / doubled
    central beam does not bias the result. Only the axis ratio ``b/a`` and orientation
    ``theta`` are identifiable from a single ring, so the returned ``(a, b)`` are
    normalised to the ring radius (``a ~ R0``); downstream consumers (``polar_transform``
    / ``find_central_beams_4d``) use only ``b/a`` and ``theta``.

    Parameters
    ----------
    dp : ndarray
        Centered mean diffraction pattern (beam at ``center``).
    center : (float, float)
        Fixed origin ``(y, x)`` in detector pixels.
    radial_min, radial_max : float, optional
        Ring band in pixels. If either is None the band is auto-detected from the
        circular median radial profile and several prominent candidates are fitted
        and quality-ranked.
    ratio_range, n_ratio, n_theta, refine :
        Coarse grid over ``b/a`` and ``theta`` (degrees), then a clipped local
        refinement pass.
    max_ring_candidates : int
        Maximum prominent radial-profile peaks evaluated when the ring band is
        selected automatically.
    min_fit_improvement, max_fit_score, min_angular_coverage : float
        Quality gates. Fits that do not improve held-out angular alignment, retain
        excessive raw angular variance, lack ring coverage, or hit a ratio boundary
        fall back to a circular correction with a warning.

    Returns
    -------
    (a, b, theta_deg, (radial_min, radial_max))
    """
    from quantem.diffraction.polar_transform import polar_transform

    dp = np.asarray(dp, dtype=float)
    Qy, Qx = dp.shape
    origin = np.asarray(center, dtype=float)

    def _polar(image, ellipse_params, rmin, rmax):
        # polar_transform returns (n_phi, n_r) when scan_pos is given.
        return np.asarray(
            polar_transform(
                image,
                origin_array=origin,
                ellipse_params=ellipse_params,
                num_annular_bins=num_annular_bins,
                radial_min=float(rmin),
                radial_max=float(rmax),
                radial_step=radial_step,
                scan_pos=(0, 0),
                device=device,
                show_progress=False,
            ),
            dtype=float,
        )

    # Log compression plus global winsorisation strongly reduces the leverage of
    # isolated Bragg spots without erasing the broad diffuse calibration ring.
    fit_dp = np.log1p(np.clip(dp, 0.0, None))
    finite_fit = fit_dp[np.isfinite(fit_dp)]
    if finite_fit.size:
        fit_dp = np.minimum(fit_dp, np.percentile(finite_fit, 99.5))

    # 1. Find several plausible diffuse-ring bands. A median angular profile is much
    #    less likely than a mean profile to select a sparse constellation of Bragg
    #    spots. Candidate fits are quality-ranked below rather than trusting the
    #    single strongest radial feature.
    r_hi = float(min(Qy, Qx) / 2.0 - 1.0)
    explicit_band = radial_min is not None and radial_max is not None
    candidate_bands = []
    if radial_min is None or radial_max is None:
        from scipy.ndimage import uniform_filter1d

        prof = _polar(fit_dp, (1.0, 1.0, 0.0), 0.0, r_hi)
        radial_profile = uniform_filter1d(np.median(prof, axis=0), size=5)
        r_axis = np.arange(radial_profile.size) * radial_step
        r_exclude = max(6.0, 0.06 * r_hi)
        i0 = int(r_exclude / radial_step)
        search_profile = radial_profile.copy()
        search_profile[:i0] = np.min(search_profile)
        prominence_floor = max(
            1e-9, 0.03 * float(np.ptp(search_profile[i0:]))
        )
        peak_indices, properties = find_peaks(
            search_profile,
            prominence=prominence_floor,
            distance=max(3, int(round(6.0 / radial_step))),
        )
        valid = (
            (peak_indices >= i0)
            & (r_axis[peak_indices] <= 0.92 * r_hi)
        )
        peak_indices = peak_indices[valid]
        prominences = properties["prominences"][valid]
        if not peak_indices.size:
            peak_indices = np.asarray(
                [i0 + int(np.argmax(search_profile[i0:]))]
            )
            prominences = np.asarray([1.0])
        order = np.argsort(prominences)[::-1][:max_ring_candidates]
        for index in peak_indices[order]:
            r0 = float(r_axis[index])
            half = max(6.0, 0.20 * r0)
            band_min = (
                max(r_exclude, r0 - half)
                if radial_min is None
                else float(radial_min)
            )
            band_max = (
                min(r_hi, r0 + half)
                if radial_max is None
                else float(radial_max)
            )
            if band_max > band_min:
                candidate_bands.append((band_min, band_max, r0))
    else:
        candidate_bands.append(
            (float(radial_min), float(radial_max),
             0.5 * (float(radial_min) + float(radial_max)))
        )

    # Deduplicate overlapping candidates created by broad/shouldered peaks.
    unique_bands = []
    for band in candidate_bands:
        if not any(abs(band[2] - other[2]) < 3.0 for other in unique_bands):
            unique_bands.append(band)
    candidate_bands = unique_bands

    fit_angles = (np.arange(num_annular_bins) // 6) % 2 == 0
    validation_angles = ~fit_angles

    def _robust_score(ellipse_params, band, angle_mask):
        polar = _polar(
            fit_dp, ellipse_params, band[0], band[1]
        )[angle_mask]
        if not polar.size:
            return np.inf
        # Per-radius clipping removes angularly isolated hot pixels. Per-angle
        # normalisation then scores radial alignment rather than polymer texture.
        upper = np.percentile(polar, 90.0, axis=0, keepdims=True)
        polar = np.minimum(polar, upper)
        polar = polar - np.percentile(
            polar, 10.0, axis=1, keepdims=True
        )
        polar = np.clip(polar, 0.0, None)
        scale = np.percentile(polar, 90.0, axis=1, keepdims=True)
        valid_scale = scale[:, 0] > 1e-9
        if np.count_nonzero(valid_scale) < 4:
            return np.inf
        polar = polar[valid_scale] / (scale[valid_scale] + 1e-9)
        reference = np.median(polar, axis=0)
        return float(
            np.median(np.abs(polar - reference), axis=0).sum()
            / (np.abs(reference).sum() + 1e-9)
        )

    def _raw_score(ellipse_params, band):
        polar = _polar(dp, ellipse_params, band[0], band[1])
        return float(
            polar.std(axis=0).sum()
            / (np.abs(polar.mean(axis=0)).sum() + 1e-6)
        )

    def _angular_coverage(ellipse_params, band):
        polar = _polar(fit_dp, ellipse_params, band[0], band[1])
        contrast = np.percentile(polar, 95.0, axis=1) - np.percentile(
            polar, 20.0, axis=1
        )
        reference = np.percentile(contrast, 90.0)
        if not np.isfinite(reference) or reference <= 1e-9:
            return 0.0
        return float(np.mean(contrast >= 0.15 * reference))

    def _search(ratios, thetas, band):
        best = (np.inf, 1.0, 0.0)
        for th in thetas:
            for rat in ratios:
                s = _robust_score(
                    (1.0, float(rat), float(th)), band, fit_angles
                )
                if s < best[0]:
                    best = (s, float(rat), float(th))
        return best

    # 2. Fit every candidate, clip refinement to the declared search range, and
    #    validate on held-out angular blocks.
    coarse_ratios = np.linspace(ratio_range[0], ratio_range[1], n_ratio)
    coarse_thetas = np.linspace(0.0, 180.0, n_theta, endpoint=False)
    diagnostics = []
    ratio_step = (
        (ratio_range[1] - ratio_range[0]) / max(n_ratio - 1, 1)
    )
    for band in candidate_bands:
        best = _search(coarse_ratios, coarse_thetas, band)
        if refine:
            _, rat0, th0 = best
            dth = 180.0 / n_theta
            fine_ratios = np.unique(np.clip(
                np.linspace(rat0 - ratio_step, rat0 + ratio_step, 11),
                ratio_range[0],
                ratio_range[1],
            ))
            fine_thetas = (
                np.linspace(th0 - dth, th0 + dth, 11) % 180.0
            )
            fine = _search(fine_ratios, fine_thetas, band)
            best = min(best, fine, key=lambda item: item[0])
        fit_score, ratio, theta = best
        circle_validation = _robust_score(
            (1.0, 1.0, 0.0), band, validation_angles
        )
        ellipse_validation = _robust_score(
            (1.0, ratio, theta), band, validation_angles
        )
        improvement = (
            (circle_validation - ellipse_validation)
            / max(abs(circle_validation), 1e-9)
        )
        raw_score = _raw_score((1.0, ratio, theta), band)
        coverage = _angular_coverage((1.0, ratio, theta), band)
        boundary_limited = (
            ratio <= ratio_range[0] + 0.25 * ratio_step
            or ratio >= ratio_range[1] - 0.25 * ratio_step
        )
        accepted = (
            np.isfinite(fit_score)
            and improvement >= min_fit_improvement
            and raw_score <= max_fit_score
            and coverage >= min_angular_coverage
            and not boundary_limited
        )
        diagnostics.append({
            "band": (float(band[0]), float(band[1])),
            "r0": float(band[2]),
            "ratio_b_over_a": float(ratio),
            "theta_deg": float(theta % 180.0),
            "fit_score": float(fit_score),
            "raw_score": float(raw_score),
            "validation_improvement": float(improvement),
            "angular_coverage": float(coverage),
            "boundary_limited": bool(boundary_limited),
            "accepted": bool(accepted),
        })

    accepted_candidates = [item for item in diagnostics if item["accepted"]]
    if accepted_candidates:
        selected = min(
            accepted_candidates,
            key=lambda item: (
                item["raw_score"],
                -item["validation_improvement"],
            ),
        )
        fit_accepted = True
    else:
        selected = min(
            diagnostics,
            key=lambda item: (
                item["boundary_limited"],
                item["raw_score"],
                -item["validation_improvement"],
            ),
        )
        fit_accepted = False

    radial_min, radial_max = selected["band"]
    r0 = selected["r0"]
    ratio = selected["ratio_b_over_a"] if fit_accepted else 1.0
    theta_deg = selected["theta_deg"] if fit_accepted else 0.0
    score = selected["raw_score"]

    # 3. Normalise (a, b) to the selected ring radius; only b/a and theta are
    #    identifiable. A rejected fit deliberately becomes a circular correction.
    a_axis, b_axis = r0, r0 * ratio
    # Canonicalise so a is the MAJOR semi-axis (a/b >= 1): the (a, b, theta) and
    # (b, a, theta+90) parametrisations describe the same ellipse, so pick the one
    # with a >= b for an unambiguous a/b >= 1 readout.
    if b_axis > a_axis:
        a_axis, b_axis = b_axis, a_axis
        theta_deg += 90.0
    theta_deg = float(theta_deg % 180.0)
    diagnostics = {
        "method": "angular_variance",
        "accepted": fit_accepted,
        "selected": selected,
        "candidates": diagnostics,
        "explicit_band": explicit_band,
        "center_initial": tuple(float(v) for v in center),
        "center_refined": tuple(float(v) for v in center),
        "rejection_reasons": [],
        "quality_thresholds": {
            "min_fit_improvement": float(min_fit_improvement),
            "max_fit_score": float(max_fit_score),
            "min_angular_coverage": float(min_angular_coverage),
            "ratio_range": tuple(float(v) for v in ratio_range),
        },
    }
    if not fit_accepted:
        reasons = []
        if selected["boundary_limited"]:
            reasons.append("ratio search boundary")
        if selected["validation_improvement"] < min_fit_improvement:
            reasons.append(
                f"held-out improvement {selected['validation_improvement']:.3g}"
            )
        if selected["raw_score"] > max_fit_score:
            reasons.append(f"raw score {selected['raw_score']:.3g}")
        if selected["angular_coverage"] < min_angular_coverage:
            reasons.append(
                f"angular coverage {selected['angular_coverage']:.1%}"
            )
        message = (
            "Ellipse fit rejected; using a circular correction"
            + (f" ({', '.join(reasons)})." if reasons else ".")
        )
        diagnostics["rejection_reasons"] = reasons
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    if verbose:
        bands_text = ", ".join(
            f"{item['r0']:.1f}" for item in diagnostics
        )
        print(
            f"  ellipse ring candidates: r0=[{bands_text}] px; "
            f"selected band=[{radial_min:.1f}, {radial_max:.1f}] px"
        )
        print(
            f"  ellipse ring fit: {'accepted' if fit_accepted else 'rejected'} "
            f"a/b={a_axis / b_axis:.4f} theta={theta_deg:.2f} deg "
            f"(score={score:.4g}, held-out improvement="
            f"{selected['validation_improvement']:.2%}, "
            f"coverage={selected['angular_coverage']:.1%})"
        )

    if show:
        import matplotlib.pyplot as plt

        circ = _polar(dp, (1.0, 1.0, 0.0), radial_min, radial_max)
        corr = _polar(
            dp, (a_axis, b_axis, theta_deg), radial_min, radial_max
        )
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(dp, cmap="magma")
        axes[0].plot([origin[1]], [origin[0]], "c+", ms=10)
        axes[0].set_title("centered mean DP")
        axes[1].imshow(circ, aspect="auto", cmap="magma")
        axes[1].set_title("polar: circular (before)")
        axes[2].imshow(corr, aspect="auto", cmap="magma")
        axes[2].set_title(f"polar: ellipse-corrected\na/b={a_axis / b_axis:.4f}, θ={theta_deg:.1f}°")
        for ax in axes[1:]:
            ax.set_xlabel("radius (band)")
            ax.set_ylabel("φ bin")
        plt.tight_layout()
        plt.show()

    return EllipseFit(
        a=float(a_axis),
        b=float(b_axis),
        theta_deg=float(theta_deg),
        ring_band=(float(radial_min), float(radial_max)),
        method="angular_variance",
        accepted=bool(fit_accepted),
        center_initial=tuple(float(v) for v in center),
        center_refined=tuple(float(v) for v in center),
        details=diagnostics,
    )


def fit_ellipse_from_ridge(
    dp,
    center,
    *,
    radial_min=None,
    radial_max=None,
    radial_step=1.0,
    num_annular_bins=180,
    ratio_range=(0.85, 1.0),
    center_search_radius=2.5,
    max_ring_candidates=3,
    min_angular_coverage=0.55,
    min_validation_improvement=0.05,
    max_validation_residual=2.5,
    device="cpu",
    show=False,
    verbose=False,
):
    """Jointly refine ring center and ellipticity from a robust radial ridge.

    The diffuse-ring radius is measured independently at each azimuth after
    log compression and hot-spot clipping. A first/second-harmonic model
    initializes center and ellipse terms, followed by bounded robust geometric
    least squares. Fits are accepted only when held-out azimuthal sectors improve
    over an independently refined circle.
    """
    from scipy.optimize import least_squares
    from quantem.diffraction.polar_transform import polar_transform

    dp = np.asarray(dp, dtype=float)
    origin = np.asarray(center, dtype=float)
    qy, qx = dp.shape
    r_hi = float(min(qy, qx) / 2.0 - 1.0)
    fit_dp = np.log1p(np.clip(dp, 0.0, None))
    finite = fit_dp[np.isfinite(fit_dp)]
    if finite.size:
        fit_dp = np.minimum(fit_dp, np.percentile(finite, 99.5))

    def polar_at(image, candidate_center, rmin, rmax):
        return np.asarray(
            polar_transform(
                image,
                origin_array=np.asarray(candidate_center, dtype=float),
                ellipse_params=(1.0, 1.0, 0.0),
                num_annular_bins=num_annular_bins,
                radial_min=float(rmin),
                radial_max=float(rmax),
                radial_step=radial_step,
                scan_pos=(0, 0),
                device=device,
                show_progress=False,
            ),
            dtype=float,
        )

    # Candidate rings from an angular median profile; sparse Bragg spots largely
    # disappear in the median rather than becoming the selected calibration ring.
    explicit_band = radial_min is not None and radial_max is not None
    if explicit_band:
        bands = [(
            float(radial_min),
            float(radial_max),
            0.5 * (float(radial_min) + float(radial_max)),
        )]
    else:
        full = polar_at(fit_dp, origin, 0.0, r_hi)
        profile = gaussian_filter1d(np.median(full, axis=0), 2.0)
        r_axis = np.arange(profile.size, dtype=float) * radial_step
        exclude = max(6.0, 0.06 * r_hi)
        start = int(np.ceil(exclude / radial_step))
        prominence = max(1e-9, 0.03 * np.ptp(profile[start:]))
        indices, properties = find_peaks(
            profile,
            prominence=prominence,
            distance=max(3, int(round(6.0 / radial_step))),
        )
        valid_peaks = (
            (indices >= start) & (r_axis[indices] <= 0.92 * r_hi)
        )
        indices = indices[valid_peaks]
        prominences = properties["prominences"][valid_peaks]
        if not indices.size:
            indices = np.asarray([start + np.argmax(profile[start:])])
            prominences = np.ones(1)
        order = np.argsort(prominences)[::-1][:max_ring_candidates]
        bands = []
        for index in indices[order]:
            r0 = float(r_axis[index])
            half = max(6.0, 0.20 * r0)
            low = max(exclude, r0 - half) if radial_min is None else float(radial_min)
            high = min(r_hi, r0 + half) if radial_max is None else float(radial_max)
            if high > low and not any(abs(r0 - old[2]) < 3.0 for old in bands):
                bands.append((low, high, r0))
        if not bands:
            fallback_r0 = float(np.clip(r_axis[start], exclude, 0.92 * r_hi))
            fallback_half = max(6.0, 0.20 * fallback_r0)
            bands = [(
                max(exclude, fallback_r0 - fallback_half),
                min(r_hi, fallback_r0 + fallback_half),
                fallback_r0,
            )]

    phi = np.linspace(0.0, 2.0 * np.pi, num_annular_bins, endpoint=False)
    block_fit = (np.arange(num_annular_bins) // 6) % 2 == 0

    def extract_ridge(band):
        polar = polar_at(fit_dp, origin, band[0], band[1])
        polar = np.minimum(
            polar, np.percentile(polar, 90.0, axis=0, keepdims=True)
        )
        smooth = gaussian_filter1d(polar, 1.25, axis=1, mode="nearest")
        baseline = np.percentile(smooth, 20.0, axis=1, keepdims=True)
        signal = np.clip(smooth - baseline, 0.0, None)
        peak_index = np.argmax(signal, axis=1)
        ridge = np.empty(num_annular_bins, dtype=float)
        confidence = np.empty(num_annular_bins, dtype=float)
        for index in range(num_annular_bins):
            lo = max(0, peak_index[index] - 2)
            hi = min(signal.shape[1], peak_index[index] + 3)
            weights = signal[index, lo:hi]
            bins = np.arange(lo, hi, dtype=float)
            ridge[index] = (
                np.average(bins, weights=weights)
                if weights.sum() > 1e-12
                else float(peak_index[index])
            )
            noise = (
                1.4826 * np.median(np.abs(np.diff(smooth[index])))
                + 1e-9
            )
            confidence[index] = signal[index, peak_index[index]] / noise
        ridge = band[0] + ridge * radial_step
        valid = np.isfinite(ridge) & (confidence >= 2.0)
        if np.count_nonzero(valid) >= 12:
            design = np.column_stack([
                np.ones(num_annular_bins),
                np.cos(phi),
                np.sin(phi),
                np.cos(2 * phi),
                np.sin(2 * phi),
            ])
            weights = np.clip(confidence / 10.0, 0.05, 1.0)
            beta = np.zeros(5)
            for _ in range(5):
                use = valid & np.isfinite(weights)
                root_weight = np.sqrt(weights[use])
                beta = np.linalg.lstsq(
                    design[use] * root_weight[:, None],
                    ridge[use] * root_weight,
                    rcond=None,
                )[0]
                residual = ridge - design @ beta
                scale = 1.4826 * np.median(
                    np.abs(residual[use] - np.median(residual[use]))
                ) + 1e-6
                robust = np.minimum(1.0, 1.5 * scale / (np.abs(residual) + 1e-9))
                weights = np.clip(confidence / 10.0, 0.05, 1.0) * robust
            valid &= np.abs(ridge - design @ beta) <= max(3.0, 4.0 * scale)
        else:
            beta = np.asarray([np.nan] * 5)
            weights = np.zeros_like(confidence)
        return ridge, confidence, valid, beta, weights

    def ellipse_residual(parameters, x, y, weights):
        cy, cx, axis_a, ratio, theta = parameters
        dx, dy = x - cx, y - cy
        cosine, sine = np.cos(theta), np.sin(theta)
        major = dx * cosine + dy * sine
        minor = -dx * sine + dy * cosine
        geometric = (
            np.sqrt(
                (major / axis_a) ** 2
                + (minor / (axis_a * ratio)) ** 2
            )
            - 1.0
        ) * axis_a
        return geometric * np.sqrt(np.clip(weights, 1e-3, None))

    def circle_residual(parameters, x, y, weights):
        cy, cx, radius = parameters
        return (
            np.hypot(x - cx, y - cy) - radius
        ) * np.sqrt(np.clip(weights, 1e-3, None))

    evaluated = []
    for band in bands:
        ridge, confidence, valid, harmonic, weights = extract_ridge(band)
        coverage = float(np.mean(valid))
        if np.count_nonzero(valid) < 12:
            evaluated.append({
                "band": tuple(float(v) for v in band[:2]),
                "r0": float(band[2]),
                "accepted": False,
                "angular_coverage": coverage,
                "rejection_reasons": ["insufficient ridge points"],
            })
            continue

        x = origin[1] + ridge * np.cos(phi)
        y = origin[0] + ridge * np.sin(phi)
        fit = valid & block_fit
        validate = valid & ~block_fit
        if np.count_nonzero(validate) < 6:
            fit = valid
            validate = valid

        r0, c1, s1, c2, s2 = harmonic
        center_initial = np.asarray([
            origin[0] + np.clip(s1, -center_search_radius, center_search_radius),
            origin[1] + np.clip(c1, -center_search_radius, center_search_radius),
        ])
        second = float(np.hypot(c2, s2))
        ratio_initial = np.clip(
            (max(r0, 1.0) - second) / (max(r0, 1.0) + second),
            ratio_range[0],
            ratio_range[1],
        )
        theta_initial = 0.5 * np.arctan2(s2, c2)
        center_low = origin - center_search_radius
        center_high = origin + center_search_radius
        axis_low = max(2.0, 0.65 * band[0])
        axis_high = min(r_hi * 1.5, 1.45 * band[1])

        circle = least_squares(
            circle_residual,
            [*center_initial, np.median(ridge[fit])],
            args=(x[fit], y[fit], weights[fit]),
            bounds=([
                center_low[0], center_low[1], axis_low
            ], [
                center_high[0], center_high[1], axis_high
            ]),
            loss="soft_l1",
            f_scale=1.0,
        )
        ellipse = least_squares(
            ellipse_residual,
            [
                *center_initial,
                np.clip(np.max(ridge[fit]), axis_low, axis_high),
                ratio_initial,
                theta_initial,
            ],
            args=(x[fit], y[fit], weights[fit]),
            bounds=([
                center_low[0], center_low[1], axis_low,
                ratio_range[0], -np.pi,
            ], [
                center_high[0], center_high[1], axis_high,
                ratio_range[1], np.pi,
            ]),
            loss="soft_l1",
            f_scale=1.0,
        )
        circle_validation = np.median(np.abs(circle_residual(
            circle.x, x[validate], y[validate], np.ones(np.count_nonzero(validate))
        )))
        ellipse_validation = np.median(np.abs(ellipse_residual(
            ellipse.x, x[validate], y[validate], np.ones(np.count_nonzero(validate))
        )))
        improvement = float(
            (circle_validation - ellipse_validation)
            / max(circle_validation, 1e-9)
        )
        center_boundary = bool(np.any(
            np.isclose(ellipse.x[:2], center_low, atol=0.05)
            | np.isclose(ellipse.x[:2], center_high, atol=0.05)
        ))
        ratio_boundary = bool(
            ellipse.x[3] <= ratio_range[0] + 0.005
            or ellipse.x[3] >= ratio_range[1] - 0.001
        )
        reasons = []
        if coverage < min_angular_coverage:
            reasons.append(f"angular coverage {coverage:.1%}")
        if improvement < min_validation_improvement:
            reasons.append(f"held-out improvement {improvement:.2%}")
        if ellipse_validation > max_validation_residual:
            reasons.append(
                f"held-out residual {ellipse_validation:.2f} px"
            )
        if center_boundary:
            reasons.append("center search boundary")
        if ratio_boundary:
            reasons.append("ratio search boundary")
        if not reasons:
            ellipse = least_squares(
                ellipse_residual,
                ellipse.x,
                args=(x[valid], y[valid], weights[valid]),
                bounds=([
                    center_low[0], center_low[1], axis_low,
                    ratio_range[0], -np.pi,
                ], [
                    center_high[0], center_high[1], axis_high,
                    ratio_range[1], np.pi,
                ]),
                loss="soft_l1",
                f_scale=1.0,
            )
            center_boundary = bool(np.any(
                np.isclose(ellipse.x[:2], center_low, atol=0.05)
                | np.isclose(ellipse.x[:2], center_high, atol=0.05)
            ))
            ratio_boundary = bool(
                ellipse.x[3] <= ratio_range[0] + 0.005
                or ellipse.x[3] >= ratio_range[1] - 0.001
            )
            if center_boundary:
                reasons.append("center search boundary after full refit")
            if ratio_boundary:
                reasons.append("ratio search boundary after full refit")
        evaluated.append({
            "band": tuple(float(v) for v in band[:2]),
            "r0": float(band[2]),
            "accepted": not reasons,
            "angular_coverage": coverage,
            "center_initial": tuple(float(v) for v in center_initial),
            "center_refined": tuple(float(v) for v in ellipse.x[:2]),
            "a_pixels": float(ellipse.x[2]),
            "b_pixels": float(ellipse.x[2] * ellipse.x[3]),
            "ratio_b_over_a": float(ellipse.x[3]),
            "theta_deg": float(np.rad2deg(ellipse.x[4]) % 180.0),
            "circle_validation_residual": float(circle_validation),
            "ellipse_validation_residual": float(ellipse_validation),
            "validation_improvement": improvement,
            "center_boundary": center_boundary,
            "ratio_boundary": ratio_boundary,
            "ridge_point_count": int(np.count_nonzero(valid)),
            "rejection_reasons": reasons,
            "_ridge": ridge,
            "_valid": valid,
            "_x": x,
            "_y": y,
        })

    accepted = [item for item in evaluated if item["accepted"]]
    candidates_with_fit = [item for item in evaluated if "a_pixels" in item]
    if accepted:
        selected = min(
            accepted,
            key=lambda item: (
                item["ellipse_validation_residual"],
                -item["validation_improvement"],
            ),
        )
        fit_accepted = True
    elif candidates_with_fit:
        selected = min(
            candidates_with_fit,
            key=lambda item: item["ellipse_validation_residual"],
        )
        fit_accepted = False
    else:
        selected = evaluated[0]
        fit_accepted = False

    public_candidates = [
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in evaluated
    ]
    if fit_accepted:
        a_axis = selected["a_pixels"]
        b_axis = selected["b_pixels"]
        theta_deg = selected["theta_deg"]
        refined_center = selected["center_refined"]
    else:
        a_axis = b_axis = selected["r0"]
        theta_deg = 0.0
        refined_center = tuple(float(v) for v in origin)
        reasons = selected.get("rejection_reasons", ["no valid ridge fit"])
        warnings.warn(
            "Ridge ellipse fit rejected; using a circular correction "
            f"({', '.join(reasons)}).",
            RuntimeWarning,
            stacklevel=2,
        )

    selected_public = {
        key: value for key, value in selected.items() if not key.startswith("_")
    }
    diagnostics = {
        "method": "ridge",
        "accepted": fit_accepted,
        "selected": selected_public,
        "candidates": public_candidates,
        "explicit_band": explicit_band,
        "center_initial": tuple(float(v) for v in origin),
        "center_refined": tuple(float(v) for v in refined_center),
        "rejection_reasons": (
            [] if fit_accepted else selected_public.get("rejection_reasons", [])
        ),
    }

    if verbose:
        print(
            "  ridge ellipse candidates: "
            + ", ".join(f"{item['r0']:.1f}" for item in public_candidates)
            + " px"
        )
        print(
            f"  ridge ellipse fit: {'accepted' if fit_accepted else 'rejected'} "
            f"a/b={a_axis / b_axis:.4f} theta={theta_deg:.2f} deg "
            f"center=({refined_center[0]:.2f}, {refined_center[1]:.2f})"
        )

    if show and "_ridge" in selected:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(np.log1p(np.clip(dp, 0, None)), cmap="magma")
        axes[0].scatter(
            selected["_x"][selected["_valid"]],
            selected["_y"][selected["_valid"]],
            s=5,
            c="cyan",
            alpha=0.7,
            label="ridge inliers",
        )
        axes[0].add_patch(Ellipse(
            (refined_center[1], refined_center[0]),
            2 * a_axis,
            2 * b_axis,
            angle=theta_deg,
            fill=False,
            color="lime",
            linewidth=1.5,
            label="ridge fit",
        ))
        axes[0].legend(fontsize=8)
        axes[0].set_title("diffuse-ring ridge and robust ellipse")
        before = polar_at(dp, origin, selected["band"][0], selected["band"][1])
        after = np.asarray(
            polar_transform(
                dp,
                origin_array=np.asarray(refined_center),
                ellipse_params=(a_axis, b_axis, theta_deg),
                num_annular_bins=num_annular_bins,
                radial_min=selected["band"][0],
                radial_max=selected["band"][1],
                radial_step=radial_step,
                scan_pos=(0, 0),
                device=device,
                show_progress=False,
            )
        )
        axes[1].imshow(before, aspect="auto", cmap="magma")
        axes[1].set_title("circular polar before")
        axes[2].imshow(after, aspect="auto", cmap="magma")
        axes[2].set_title("ridge-refined polar after")
        fig.tight_layout()
        plt.show()

    band = selected["band"]
    return EllipseFit(
        a=float(a_axis),
        b=float(b_axis),
        theta_deg=float(theta_deg),
        ring_band=(float(band[0]), float(band[1])),
        method="ridge",
        accepted=bool(fit_accepted),
        center_initial=tuple(float(v) for v in origin),
        center_refined=tuple(float(v) for v in refined_center),
        details=diagnostics,
    )
