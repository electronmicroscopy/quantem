"""Diffraction-space calibration against known crystal structures.

Functions here operate on detected Bragg peaks (a quantem Vector) and refine
the reciprocal-space pixel size by comparing the radial peak histogram with
the ring positions of a reference crystal.
"""

from __future__ import annotations

import numpy as np
import torch

from quantem.core.datastructures.vector import Vector
from quantem.diffraction.crystal import Crystal


def _measure_raw_origins(bragg_vectors, search_radius: float) -> np.ndarray:
    """Brightest-peak origin per position, NaN where nothing is found."""
    peaks = bragg_vectors.peaks
    scan_r, scan_c = peaks.shape[0], peaks.shape[1]
    H, W = int(bragg_vectors.dataset.shape[-2]), int(bragg_vectors.dataset.shape[-1])
    c0 = np.array([H / 2, W / 2])
    meas = np.full((scan_r, scan_c, 2), np.nan)
    for r in range(scan_r):
        for c in range(scan_c):
            arr = peaks[r, c].array
            if arr.shape[0] == 0:
                continue
            d = np.hypot(arr[:, 0] - c0[0], arr[:, 1] - c0[1])
            near = d < search_radius
            if not near.any():
                continue
            sub = arr[near]
            meas[r, c] = sub[np.argmax(sub[:, 2]), :2]
    return meas


def plot_origin_fit(bragg_vectors, origins: np.ndarray, search_radius: float = 6.0):
    """Diagnostic for measure_origins: measured vs fit vs residual, both axes."""
    import matplotlib.pyplot as plt

    meas = _measure_raw_origins(bragg_vectors, search_radius)
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 5.6))
    names = ["row", "col"]
    for k in range(2):
        m, f = meas[..., k], origins[..., k]
        resid = m - f
        center = np.nanmean(m)
        span = max(np.nanstd(m) * 3, 1e-3)
        for j, (img, title) in enumerate(
            [
                (m, f"measured origin {names[k]} (px)"),
                (f, f"plane fit {names[k]} (px)"),
                (resid, f"residual {names[k]} (px)"),
            ]
        ):
            c0 = 0.0 if j == 2 else center
            sp = max(np.nanstd(resid) * 3, 1e-3) if j == 2 else span
            im = axs[k, j].imshow(
                img, cmap="RdBu_r", vmin=c0 - sp, vmax=c0 + sp,
                interpolation="nearest",
            )
            axs[k, j].set_title(title, fontsize=10)
            axs[k, j].set_xticks([])
            axs[k, j].set_yticks([])
            fig.colorbar(im, ax=axs[k, j], shrink=0.85)
    fig.tight_layout()
    return fig, axs


def measure_origins(
    bragg_vectors,
    search_radius: float = 6.0,
    robust: bool = True,
    plot: bool = False,
):
    """Per-position diffraction origin from the brightest central peak.

    At each scan position the most intense detected peak within
    `search_radius` pixels of the detector center is taken as the direct
    beam; a plane is fit over the scan (least squares, optionally with one
    outlier-rejection pass) to model the descan.

    Parameters
    ----------
    plot : bool, default=False
        Show the fitted origin planes and the residuals of the measured
        origins against the fit.

    Returns
    -------
    np.ndarray
        (scan_row, scan_col, 2) plane-fit origins, ready for
        BraggVectors.correct_peak_origins(). With plot=True, also returns
        (fig, axs).
    """
    meas = _measure_raw_origins(bragg_vectors, search_radius)
    scan_r, scan_c = meas.shape[0], meas.shape[1]
    ry, rx = np.mgrid[0:scan_r, 0:scan_c]

    def plane(z, ok):
        A = np.stack([np.ones(ok.sum()), ry[ok], rx[ok]], axis=1)
        coef, *_ = np.linalg.lstsq(A, z[ok], rcond=None)
        return coef[0] + coef[1] * ry + coef[2] * rx

    out = np.zeros((scan_r, scan_c, 2))
    for k in range(2):
        z = meas[..., k]
        ok = np.isfinite(z)
        fit = plane(z, ok)
        if robust:
            resid = np.abs(z - fit)
            thresh = 3 * np.nanmedian(resid[ok]) + 1e-9
            ok = ok & (resid < thresh)
            fit = plane(z, ok)
        out[..., k] = fit

    if plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 3, figsize=(13.5, 5.6))
        names = ["row", "col"]
        for k in range(2):
            m, f = meas[..., k], out[..., k]
            resid = m - f
            center = np.nanmean(m)
            span = max(np.nanstd(m) * 3, 1e-3)
            for j, (img, title) in enumerate(
                [
                    (m, f"measured origin {names[k]} (px)"),
                    (f, f"plane fit {names[k]} (px)"),
                    (resid, f"residual {names[k]} (px)"),
                ]
            ):
                c0 = 0.0 if j == 2 else center
                s = max(np.nanstd(resid) * 3, 1e-3) if j == 2 else span
                im = axs[k, j].imshow(
                    img, cmap="RdBu_r", vmin=c0 - s, vmax=c0 + s,
                    interpolation="nearest",
                )
                axs[k, j].set_title(title, fontsize=10)
                axs[k, j].set_xticks([])
                axs[k, j].set_yticks([])
                fig.colorbar(im, ax=axs[k, j], shrink=0.85)
        fig.tight_layout()
        return out, fig, axs
    return out


def peaks_to_calibrated(
    peaks_px,
    pixel_size_inv_A: float,
    rotation_ccw_deg: float = 0.0,
    ellipse=None,
    name: str = "bragg_peaks_calibrated",
):
    """Convert origin-corrected pixel peaks to a calibrated (qx, qy) Vector.

    Parameters
    ----------
    peaks_px : Vector
        Peaks with fields (q_row, q_col, intensity) in detector pixels,
        already origin-corrected so (0, 0) is the direct beam.
    pixel_size_inv_A : float
        Reciprocal pixel size in 1/Angstroms.
    rotation_ccw_deg : float, default=0.0
        Diffraction-to-scan rotation: the detector coordinates are rotated
        by this angle so the pattern axes align with the scan axes. With
        this applied, in-plane orientations and strain axes are reported in
        the image frame.
    ellipse : array-like | None
        [e11, e12] elliptic distortion correction from calibrate_ellipse(),
        applied in the detector frame before the rotation.

    Returns
    -------
    Vector
        Fields (qx, qy, intensity) in 1/Angstroms.
    """
    scan_r, scan_c = peaks_px.shape[0], peaks_px.shape[1]
    flat = peaks_px.select_fields("q_row", "q_col", "intensity").flatten()
    row_counts = np.asarray(peaks_px.row_counts(), dtype=int)
    qrc = flat[:, :2] * pixel_size_inv_A
    if ellipse is not None:
        e11, e12 = float(ellipse[0]), float(ellipse[1])
        A = np.array([[1 + e11, e12], [e12, 1 - e11]])
        qrc = qrc @ A.T
    if rotation_ccw_deg != 0.0:
        th = np.deg2rad(rotation_ccw_deg)
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        qrc = qrc @ rot.T
    data = np.column_stack([qrc, flat[:, 2]])
    cells = np.split(data, np.cumsum(row_counts)[:-1])
    nested = [cells[r * scan_c : (r + 1) * scan_c] for r in range(scan_r)]
    out = Vector.from_data(
        nested,
        fields=["qx", "qy", "intensity"],
        units=["A^-1", "A^-1", "counts"],
        name=name,
    )
    # record the detector-to-scan rotation so pattern-overlay plots can put
    # peaks back into the raw detector frame
    out.metadata["rotation_ccw_deg"] = float(rotation_ccw_deg)
    return out


def scale_peaks(peaks, scale: float):
    """Return a copy of a (qx, qy, intensity) Vector with q scaled."""
    out = peaks.copy()
    flat = out.flatten()
    flat[:, :2] *= scale
    out.set_flattened(flat)
    return out


def radial_histogram(
    peaks: Vector,
    k_min: float = 0.05,
    k_max: float = 1.5,
    k_step: float = 0.002,
    bragg_k_power: float = 2.0,
    bragg_intensity_power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Intensity-weighted histogram of Bragg peak radii over all positions.

    Returns
    -------
    k : np.ndarray
        Bin centers (1/Angstroms).
    hist : np.ndarray
        Weighted counts, with linear interpolation between adjacent bins.
    """
    flat = peaks.select_fields("qx", "qy", "intensity").flatten()
    qr = np.hypot(flat[:, 0], flat[:, 1])
    weight = flat[:, 2] ** bragg_intensity_power * qr**bragg_k_power

    k = np.arange(k_min, k_max, k_step)
    frac = (qr - k_min) / k_step
    i0 = np.floor(frac).astype(int)
    w1 = frac - i0
    ok = (i0 >= 0) & (i0 < k.size - 1)
    hist = np.bincount(i0[ok], weights=weight[ok] * (1 - w1[ok]), minlength=k.size)
    hist += np.bincount(i0[ok] + 1, weights=weight[ok] * w1[ok], minlength=k.size)
    return k, hist


def simulated_ring_profile(
    crystal: Crystal,
    k: np.ndarray,
    k_broadening: float = 0.01,
    bragg_k_power: float = 2.0,
) -> np.ndarray:
    """1D ring profile of a crystal: Gaussians at |g| weighted by intensity."""
    g = crystal.g_len.numpy()
    w = crystal.struct_factors_int.numpy() * g**bragg_k_power
    prof = (
        w[None, :] * np.exp(-((k[:, None] - g[None, :]) ** 2) / (2 * k_broadening**2))
    ).sum(axis=1)
    return prof


def calibrate_pixel_size_matching(
    peaks,
    crystal: Crystal | list[Crystal],
    energy_ev: float,
    scales: np.ndarray | None = None,
    subsample: int = 8,
    angle_step_deg: float = 3.0,
    corr_kernel_size: float = 0.02,
    min_number_peaks: int = 6,
    plot: bool = False,
    return_scores: bool = False,
    returnfig: bool = False,
):
    """Refine the pixel size by maximizing the orientation-match correlation.

    The 1D radial fit can be fooled by ring-ratio degeneracies (e.g. the
    hexagonal-net radii shared by hcp prismatic rings and bcc {110}-family
    rings). Full-pattern matching is not: for each candidate scale, a
    subsampled grid of patterns is orientation-matched against the crystal
    and the median normalized correlation scored. Peaks in the score curve
    identify the true calibration.

    Parameters
    ----------
    peaks : Vector
        Calibrated peaks (qx, qy, intensity) in 1/Angstroms.
    crystal : Crystal | list[Crystal]
        Reference crystal(s). Pass ALL candidate phases for multi-phase
        samples: with a single reference, a scale that maps the majority
        phase's net onto the reference's (e.g. the bcc {110} ring onto the
        hcp prismatic ring, ratio 0.90 for Ti) can win the scan. Scoring the
        mean over phases of the per-phase median correlation removes the
        false optimum, since the other phases index nothing at the impostor
        scale.
    energy_ev : float
        Beam energy in eV.
    scales : np.ndarray | None
        Candidate scale factors; defaults to 0.90 ... 1.10 in 2% steps.
    subsample : int, default=8
        Stride of the probe-position grid used for scoring.

    Returns
    -------
    scale : float
        Best scale factor (parabolic refinement over the score maximum).
        With return_scores=True, also (scales, scores); with returnfig=True,
        also (fig, ax).
    """
    from quantem.diffraction.orientation import OrientationMap

    crystals = crystal if isinstance(crystal, (list, tuple)) else [crystal]
    if scales is None:
        scales = np.arange(0.90, 1.101, 0.02)
    sub = peaks[::subsample, ::subsample]
    scores = np.zeros(len(scales))
    for i, s in enumerate(scales):
        test = scale_peaks(sub, float(s))
        per_phase = []
        for xtl in crystals:
            om = OrientationMap.from_vectors(test, xtl, energy_ev=energy_ev)
            # detector_q_max must stay OFF here: the auto footprint shrinks
            # with the candidate scale, silently removing the unexplained
            # high-q template shells that penalize too-small scales -- the
            # score then rises monotonically as the pattern is compressed.
            # a tight kernel is essential for a wide scale scan: the default
            # matching kernel (0.05) hands partial credit to near-miss ring
            # coincidences of an impostor scale, while matches at the true
            # scale are exact to the detection noise (~0.005)
            om.build_plan(
                angle_step_zone_axis_deg=angle_step_deg,
                angle_step_in_plane_deg=angle_step_deg,
                corr_kernel_size=corr_kernel_size,
                detector_q_max=None,
                verbose=False,
            )
            om.match_orientations(progress_bar=False, min_number_peaks=min_number_peaks)
            corr = om.corr[..., 0]
            per_phase.append(float(corr[corr > 0].median()))
        scores[i] = float(np.mean(per_phase))

    i_best = int(np.argmax(scores))
    scale = float(scales[i_best])
    if 0 < i_best < len(scales) - 1:
        c0, c1, c2 = scores[i_best - 1 : i_best + 2]
        denom = 4 * c1 - 2 * c0 - 2 * c2
        step = scales[1] - scales[0]
        if abs(denom) > 1e-12:
            scale += (c2 - c0) / denom * step
    out: list = [scale]
    if return_scores:
        out += [scales, scores]
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(scales, scores, "k.-")
        ax.axvline(scale, color="r", ls="--")
        ax.set_xlabel("pixel size scale factor")
        ax.set_ylabel("median correlation")
        ax.set_title(f"best scale = {scale:.4f}")
        if returnfig:
            out += [fig, ax]
    return tuple(out) if len(out) > 1 else out[0]


def calibrate_pixel_size(
    peaks: Vector,
    crystal: Crystal,
    scale_range: tuple[float, float] = (0.8, 1.25),
    scale_step: float = 5e-4,
    k_min: float = 0.05,
    k_max: float = 1.3,
    k_broadening: float = 0.01,
    bragg_k_power: float = 2.0,
    plot: bool = False,
    returnfig: bool = False,
):
    """Refine the reciprocal pixel size against a reference crystal.

    Scans a multiplicative scale factor applied to the measured peak radii and
    maximizes the normalized overlap between the measured radial histogram and
    the crystal's simulated ring profile. Parabolic sub-step refinement of the
    best scale.

    Parameters
    ----------
    peaks : Vector
        Calibrated peaks with fields (qx, qy, intensity) in 1/Angstroms.
    crystal : Crystal
        Reference crystal with structure factors calculated. Choose the
        majority phase of the scan.
    scale_range : tuple, default=(0.8, 1.25)
        Search range of the scale factor.
    plot : bool, default=False
        Show the measured histogram against the crystal ring positions,
        before and after applying the scale.

    Returns
    -------
    scale : float
        Multiply existing q values (and the pixel size) by this factor,
        e.g. with scale_peaks(). With returnfig=True, also (fig, axs).
    """
    k, hist = radial_histogram(peaks, k_min=k_min * scale_range[0], k_max=k_max / scale_range[0])
    scales = np.arange(scale_range[0], scale_range[1], scale_step)
    score = np.zeros_like(scales)
    for i, s in enumerate(scales):
        prof = simulated_ring_profile(crystal, k * s, k_broadening, bragg_k_power)
        keep = (k * s > k_min) & (k * s < k_max)
        h, p = hist[keep], prof[keep]
        denom = np.linalg.norm(h) * np.linalg.norm(p)
        score[i] = (h * p).sum() / denom if denom > 0 else 0.0

    i_best = int(np.argmax(score))
    scale = float(scales[i_best])
    if 0 < i_best < scales.size - 1:
        c0, c1, c2 = score[i_best - 1 : i_best + 2]
        denom = 4 * c1 - 2 * c0 - 2 * c2
        if abs(denom) > 1e-12:
            scale += (c2 - c0) / denom * scale_step

    out: list = [scale]
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        prof = simulated_ring_profile(
            crystal, k * scale, k_broadening / 2, bragg_k_power
        )
        ax.fill_between(
            k * scale, hist / hist.max(), color="r", alpha=0.75, lw=0,
            label="measured (scaled)",
        )
        ax.plot(
            k * scale, prof / prof.max(), "k-", lw=1.0,
            label=f"{crystal.name} rings",
        )
        ax.set_ylabel("intensity (norm.)")
        ax.set_xlabel("scattering vector (1/$\\mathrm{\\AA}$)")
        ax.set_title(f"1D radial fit, scale = {scale:.4f}", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        if returnfig:
            out += [fig, ax]
    return tuple(out) if len(out) > 1 else out[0]


def measure_scan_rotation(
    dataset,
    origins: np.ndarray | None = None,
    mask_radius: float | None = None,
    plot: bool = False,
    returnfig: bool = False,
):
    """Detector-to-scan rotation from the curl of the center-of-mass field.

    The center of mass of each diffraction pattern (about the fitted origin)
    forms a vector field over the scan. In the correct common frame that
    field is (approximately) a gradient field, so its curl vanishes; rotating
    the detector axes by the unknown scan rotation and minimizing the summed
    squared curl recovers the angle.

    The curl is invariant under 180-degree rotation, so the sign of the
    measured field cannot distinguish theta from theta + 180. Both candidates
    are returned; pick the one consistent with a known feature (e.g. a
    Burgers orientation relationship, or the divergence sign convention of
    DPC). The returned angle is ready to pass to peaks_to_calibrated() as
    rotation_ccw_deg.

    Parameters
    ----------
    dataset : Dataset4dstem
        The 4D-STEM scan.
    origins : np.ndarray | None
        (scan_r, scan_c, 2) diffraction origins from measure_origins();
        defaults to the pattern center.
    mask_radius : float | None
        Restrict the center of mass to within this radius (pixels) of the
        origin -- i.e. the DPC signal of the direct beam only, excluding the
        Bragg disks. Recommended for crystalline data.
    plot : bool, default=False
        Plot the curl and divergence measures against the rotation angle.

    Returns
    -------
    rotation_ccw_deg : float
        Curl-minimizing rotation in [0, 180); the physical answer is either
        this angle or this angle + 180. With returnfig=True, also (fig, ax).
    """
    arr = np.asarray(dataset.array, dtype=float)
    scan_r, scan_c, H, W = arr.shape
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    if origins is None:
        origins = np.zeros((scan_r, scan_c, 2))
        origins[..., 0] = H / 2
        origins[..., 1] = W / 2
    if mask_radius is not None:
        rr = rows[None, None] - origins[..., 0][..., None, None]
        cc = cols[None, None] - origins[..., 1][..., None, None]
        arr = arr * (rr**2 + cc**2 <= mask_radius**2)
    tot = arr.sum(axis=(-2, -1))
    tot[tot <= 0] = 1.0
    com_r = (arr * rows).sum(axis=(-2, -1)) / tot - origins[..., 0]
    com_c = (arr * cols).sum(axis=(-2, -1)) / tot - origins[..., 1]

    # spatial derivatives of both components over the scan
    d_rr = np.gradient(com_r, axis=0)
    d_rc = np.gradient(com_r, axis=1)
    d_cr = np.gradient(com_c, axis=0)
    d_cc = np.gradient(com_c, axis=1)

    theta = np.deg2rad(np.arange(0, 180, 0.25))
    ct, st = np.cos(theta)[:, None, None], np.sin(theta)[:, None, None]
    # rotated field: (r', c') = (ct * r - st * c, st * r + ct * c)
    curl = (st * d_rr + ct * d_cr) - (ct * d_rc - st * d_cc)
    div = (ct * d_rr - st * d_cr) + (st * d_rc + ct * d_cc)
    curl_sq = (curl**2).mean(axis=(1, 2))
    div_sq = (div**2).mean(axis=(1, 2))

    i_best = int(np.argmin(curl_sq))
    rotation_ccw_deg = float(np.rad2deg(theta[i_best]))

    out: list = [rotation_ccw_deg]
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        deg = np.rad2deg(theta)
        ax.plot(deg, curl_sq, "k-", label="mean squared curl")
        ax.plot(deg, div_sq, "-", color="0.6", label="mean squared divergence")
        ax.axvline(rotation_ccw_deg, color="r", ls="--")
        ax.set_xlabel("rotation (degrees)")
        ax.set_ylabel("field measure")
        ax.set_title(
            "scan rotation = %.1f deg (or %.1f)"
            % (rotation_ccw_deg, rotation_ccw_deg + 180)
        )
        ax.legend()
        if returnfig:
            out += [fig, ax]
    return tuple(out) if len(out) > 1 else out[0]


def calibrate_ellipse(
    peaks,
    k_min: float = 0.15,
    k_max: float = 1.4,
    n_bins: int = 800,
    bragg_k_power: float = 2.0,
    bragg_intensity_power: float = 1.0,
    plot: bool = False,
    returnfig: bool = False,
):
    """Elliptic distortion (e11, e12) from radial histogram sharpness.

    Fits the traceless linear distortion A = [[1 + e11, e12], [e12, 1 - e11]]
    that, applied to the measured peaks, maximizes the sharpness of the
    radial peak histogram: an elliptic distortion smears every diffraction
    ring, and undoing it re-focuses them. The histogram is accumulated in
    log-radius bins, where a pure scale change is only a translation -- so
    the ellipse fit is independent of the pixel size, and the calibration
    workflow stays sequential: rough scale, then ellipse, then absolute
    scale by pattern matching.

    Parameters
    ----------
    peaks : Vector
        Calibrated peaks (qx, qy, intensity), approximate scale is fine.
    k_min, k_max : float
        Radial range (1/Angstroms) included in the sharpness measure.
    plot : bool, default=False
        Show the radial histogram before and after the correction.

    Returns
    -------
    ellipse : np.ndarray
        [e11, e12]; pass to peaks_to_calibrated(ellipse=...) or
        apply_ellipse(). With returnfig=True, also (fig, ax).
    """
    from scipy.optimize import minimize

    flat = peaks.select_fields("qx", "qy", "intensity").flatten()
    q = flat[:, :2]
    log_lo, log_hi = np.log(k_min), np.log(k_max)
    bin_w = (log_hi - log_lo) / n_bins

    def histogram(e):
        A = np.array([[1 + e[0], e[1]], [e[1], 1 - e[0]]])
        qe = q @ A.T
        r = np.hypot(qe[:, 0], qe[:, 1])
        ok = (r > k_min) & (r < k_max)
        w = flat[ok, 2] ** bragg_intensity_power * r[ok] ** bragg_k_power
        f = (np.log(r[ok]) - log_lo) / bin_w
        i0 = np.floor(f).astype(int)
        w1 = f - i0
        h = np.bincount(i0, weights=w * (1 - w1), minlength=n_bins + 1)
        h += np.bincount(i0 + 1, weights=w * w1, minlength=n_bins + 1)
        return h

    def cost(e):
        h = histogram(e)
        total = h.sum()
        if total <= 0:
            return 0.0
        return -float((h**2).sum()) / total**2

    res = minimize(
        cost,
        x0=np.zeros(2),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-12, "maxiter": 400},
    )
    ellipse = res.x

    out: list = [ellipse]
    if plot:
        import matplotlib.pyplot as plt

        k_bins = np.exp(np.linspace(log_lo, log_hi, n_bins + 1))
        h0 = histogram(np.zeros(2))
        h1 = histogram(ellipse)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(
            k_bins, h0 / h0.max(), color="0.7", lw=0, label="measured"
        )
        ax.plot(k_bins, h1 / h1.max(), "r-", lw=1.0, label="ellipse corrected")
        ax.set_xlabel("scattering vector (1/$\\mathrm{\\AA}$)")
        ax.set_ylabel("intensity (norm.)")
        mag = np.hypot(*ellipse)
        ax.set_title(
            "e11 = %.2e, e12 = %.2e  (%.2f%% ellipticity)"
            % (ellipse[0], ellipse[1], 200 * mag)
        )
        ax.legend()
        if returnfig:
            out += [fig, ax]
    return tuple(out) if len(out) > 1 else out[0]


def apply_ellipse(peaks, ellipse):
    """Return a copy of (qx, qy, intensity) peaks with the ellipse applied."""
    out = peaks.copy()
    flat = out.flatten()
    e11, e12 = float(ellipse[0]), float(ellipse[1])
    A = np.array([[1 + e11, e12], [e12, 1 - e11]])
    flat[:, :2] = flat[:, :2] @ A.T
    out.set_flattened(flat)
    return out


def _hkl_label(hkl: np.ndarray, hexagonal: bool) -> str:
    """Compact (hkl) / (hkil) plane label with unicode overbars."""

    def digit(v: int) -> str:
        v = int(round(v))
        txt = str(abs(v))
        return txt + "̅" if v < 0 else txt

    h, k, l = (int(round(v)) for v in hkl)
    if hexagonal:
        return "(" + digit(h) + digit(k) + digit(-(h + k)) + digit(l) + ")"
    return "(" + digit(h) + digit(k) + digit(l) + ")"


def plot_ring_comparison(
    peaks,
    crystals,
    k_min: float = 0.1,
    k_max: float = 1.5,
    k_broadening: float | None = None,
    bragg_k_power: float = 2.0,
    label_hkl: bool = True,
    label_min_intensity: float = 0.05,
    figax=None,
):
    """Measured radial peak histogram against crystal ring positions.

    One panel per crystal: the measured histogram is the red fill, the
    crystal's rings are black -- sharp vertical lines by default, or a
    Gaussian profile of width `k_broadening` when set (use after
    calibration, where the rings should sit inside the measured peaks). The
    strongest rings are labeled by (hkl), 4-index (hkil) for hexagonal
    crystals.

    Parameters
    ----------
    peaks : Vector
        Calibrated peaks (qx, qy, intensity) in 1/Angstroms.
    crystals : Crystal | list[Crystal]
        Reference crystal(s) with structure factors calculated.
    k_broadening : float | None
        None draws sharp lines at the ring positions; a value (1/Angstroms)
        draws the broadened ring profile instead.
    label_min_intensity : float, default=0.05
        Label rings whose summed intensity exceeds this fraction of the
        strongest ring.
    """
    import matplotlib.pyplot as plt

    xtls = crystals if isinstance(crystals, (list, tuple)) else [crystals]
    k, hist = radial_histogram(peaks, k_min=k_min, k_max=k_max)

    n = len(xtls)
    if figax is None:
        fig, axs = plt.subplots(n, 1, figsize=(11, 3.6 * n), sharex=True, squeeze=False)
        axs = axs[:, 0]
    else:
        fig, axs = figax
        axs = np.atleast_1d(axs)

    for ci, (ax, xtl) in enumerate(zip(axs, xtls)):
        ax.fill_between(
            k, hist / hist.max(), color="r", alpha=0.75, lw=0, label="measured"
        )
        hexagonal = xtl.laue_group in ("6/m", "6/mmm", "-3", "-3m")
        g_len = xtl.g_len.numpy()
        ints = xtl.struct_factors_int.numpy() * g_len**bragg_k_power
        hkl_np = xtl.hkl.numpy()
        shells = np.round(g_len / 0.01) * 0.01
        uniq = np.unique(shells)
        shell_int = np.array([ints[shells == u].sum() for u in uniq])
        shell_int = shell_int / shell_int.max()

        if k_broadening is not None:
            prof = simulated_ring_profile(xtl, k, k_broadening, bragg_k_power)
            ax.plot(
                k, prof / prof.max(), "k-", lw=1.0, label=f"{xtl.name} rings"
            )
        else:
            keep = (uniq > k_min) & (uniq < k_max)
            ax.vlines(
                uniq[keep], 0, shell_int[keep], colors="k", lw=1.0,
                label=f"{xtl.name} rings",
            )

        if label_hkl:
            rows = 0
            labeled_g: list[float] = []
            for u, si in zip(uniq, shell_int):
                if u < k_min or u > k_max or si < label_min_intensity:
                    continue
                if any(abs(u - g0) < 0.03 for g0 in labeled_g):
                    continue
                in_shell = shells == u
                idx = np.nonzero(in_shell)[0]
                idx = idx[ints[idx] > 0.99 * ints[idx].max()]
                key = [tuple(-hkl_np[i]) for i in idx]
                best = idx[int(np.lexsort(np.array(key).T[::-1])[0])]
                labeled_g.append(u)
                y = 1.05 + 0.11 * (rows % 2)
                rows += 1
                ax.text(
                    u, y, _hkl_label(hkl_np[best], hexagonal),
                    fontsize=8, ha="center", va="bottom",
                )
        ax.set_ylabel("intensity (norm.)")
        ax.set_ylim(0, 1.32)
        ax.legend(loc="upper right", fontsize=9)
    axs[-1].set_xlabel("scattering vector (1/$\\mathrm{\\AA}$)")
    fig.tight_layout()
    return fig, axs


def transform_peaks(peaks, M: np.ndarray):
    """Return a copy of a (qx, qy, intensity) Vector with q mapped by M (2x2)."""
    out = peaks.copy()
    flat = out.flatten()
    flat[:, :2] = flat[:, :2] @ np.asarray(M, dtype=float).T
    out.set_flattened(flat)
    return out


def refine_calibration(
    strain_maps,
    masks=None,
    max_strain: float = 0.05,
):
    """Global calibration residual from matched orientations.

    The per-position deformation A fitted by
    OrientationMap.calculate_strain() maps ideal simulated peaks onto the
    measured ones, so it contains both the local strain and any global
    calibration error. The element-wise median of A over many differently
    oriented grains (across all phases) averages the strain away and leaves
    the calibration residual: scale and ellipticity. A global detector
    rotation is NOT observable this way -- the in-plane refinement absorbs
    it into every orientation, so the reported rotation_deg is ~0 by
    construction; measure the scan rotation independently
    (measure_scan_rotation, or a known texture). Apply the returned
    correction with transform_peaks() and re-match to close the loop.

    Parameters
    ----------
    strain_maps : list[StrainMap]
        One per phase, from calculate_strain() on the SAME calibrated peaks.
    masks : list[np.ndarray] | None
        Per-phase inclusion masks (e.g. phase == i and reliable); defaults
        to all positions where the strain fit succeeded.
    max_strain : float, default=0.05
        Discard positions whose deformation differs from the identity by
        more than this (failed fits, overlaps).

    Returns
    -------
    dict with:
        'M' : the median deformation (2, 2),
        'correction' : inv(M), ready for transform_peaks(),
        'scale' : multiply the pixel size by this,
        'rotation_deg' : residual detector rotation,
        'ellipse' : (e11, e12) traceless ellipticity components,
        'num_positions' : positions used.
    """
    As = []
    for i, sm in enumerate(strain_maps):
        A = np.stack([sm.u_array, sm.v_array], axis=-1)  # (R, C, 2, 2)
        ok = np.isfinite(A).all(axis=(-2, -1))
        dev = np.abs(A - np.eye(2)).max(axis=(-2, -1))
        ok &= dev < max_strain
        if masks is not None and masks[i] is not None:
            ok &= np.asarray(masks[i]) > 0
        As.append(A[ok])
    A_all = np.concatenate(As, axis=0)
    M = np.median(A_all, axis=0)

    scale = float(np.sqrt(np.abs(np.linalg.det(M))))
    theta = 0.5 * (M[1, 0] - M[0, 1]) / scale
    sym = 0.5 * (M + M.T) / scale
    e11 = float(0.5 * (sym[0, 0] - sym[1, 1]))
    e12 = float(sym[0, 1])
    return {
        "M": M,
        "correction": np.linalg.inv(M),
        "scale": scale,
        "rotation_deg": float(np.rad2deg(theta)),
        "ellipse": (e11, e12),
        "num_positions": int(A_all.shape[0]),
    }


def plot_bragg_rings(
    peaks,
    crystals,
    n_rings: int = 8,
    q_max: float | None = None,
    bins: int = 400,
    power: float = 0.25,
    figax=None,
):
    """2D histogram of all Bragg peaks with crystal rings overlaid.

    The Bragg vector map (histogram of every detected peak over the scan)
    shows the calibration directly in 2D: the crystal's strongest rings are
    drawn as thin circles, which should thread through the measured spot
    density -- a radius mismatch is a pixel size error, and a direction-
    dependent mismatch is elliptic distortion.

    Parameters
    ----------
    peaks : Vector
        Calibrated peaks (qx, qy, intensity) in 1/Angstroms.
    crystals : Crystal | list[Crystal]
        Reference crystal(s); the n_rings strongest rings of each are drawn
        (solid, then dashed line styles).
    n_rings : int, default=8
        Number of rings per crystal, strongest first.
    """
    import matplotlib.pyplot as plt

    xtls = crystals if isinstance(crystals, (list, tuple)) else [crystals]
    flat = peaks.select_fields("qx", "qy", "intensity").flatten()
    if q_max is None:
        q_max = float(np.hypot(flat[:, 0], flat[:, 1]).max()) * 1.02
    H, xe, ye = np.histogram2d(
        flat[:, 0],
        flat[:, 1],
        bins=bins,
        range=[[-q_max, q_max], [-q_max, q_max]],
    )

    if figax is None:
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
    else:
        fig, ax = figax
    ax.imshow(
        H**power,
        cmap="gray_r",
        extent=(ye[0], ye[-1], xe[-1], xe[0]),
        interpolation="nearest",
    )
    styles = ["-", "--", ":"]
    colors = ["r", "b", "g"]
    th = np.linspace(0, 2 * np.pi, 361)
    for ci, xtl in enumerate(xtls):
        g_len = xtl.g_len.numpy()
        ints = xtl.struct_factors_int.numpy() * g_len**2
        shells = np.round(g_len / 0.01) * 0.01
        uniq = np.unique(shells)
        shell_int = np.array([ints[shells == u].sum() for u in uniq])
        keep = uniq < q_max
        uniq, shell_int = uniq[keep], shell_int[keep]
        order = np.argsort(shell_int)[::-1][:n_rings]
        for k, u in enumerate(np.sort(uniq[order])):
            ax.plot(
                u * np.sin(th), u * np.cos(th),
                ls=styles[ci % 3], color=colors[ci % 3], lw=0.5, alpha=0.6,
                label=f"{xtl.name} rings" if k == 0 else None,
            )
    ax.set_xlabel("$q_c$ (1/$\\mathrm{\\AA}$)")
    ax.set_ylabel("$q_r$ (1/$\\mathrm{\\AA}$)")
    ax.legend(loc="upper right", fontsize=9)
    return fig, ax
