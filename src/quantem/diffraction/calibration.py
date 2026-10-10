"""Diffraction-space calibration of 4D-STEM Bragg peaks.

Each step works on detected Bragg peaks (a quantem Vector):

- Origins: :func:`measure_origins` fits a plane to the direct-beam position
  over the scan, for ``BraggVectors.correct_peak_origins``.
- Scan rotation: :func:`measure_scan_rotation` finds the detector-to-scan
  rotation from the curl of the center-of-mass field.
- Pixel size and elliptic distortion: :func:`calibrate` matches the radial
  peak histogram against the rings of one or more reference crystals and
  returns a :class:`DiffractionCalibration`.
- :class:`DiffractionCalibration` holds the pixel size (1/Angstroms per
  pixel), the ellipse and the rotation, converts pixel peaks to calibrated
  (qx, qy) peaks with :meth:`DiffractionCalibration.apply`, and can be saved
  from a standard and reused on another dataset.

:func:`calibrate` is the main entry point. The other functions are
lower-level steps that act on peaks directly: :func:`peaks_to_calibrated`
and :func:`calibrate_ellipse` (both used by :func:`calibrate`),
:func:`calibrate_pixel_size` (radial histogram fit of the scale),
:func:`calibrate_pixel_size_matching` (scale fit by full orientation
matching), :func:`scale_peaks` and :func:`apply_ellipse`.
:func:`refine_calibration` measures the remaining calibration error from
strain maps after orientation matching.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import AutoSerialize
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.defaults import MIN_NUMBER_PEAKS


def _measure_raw_origins(bragg_vectors, search_radius: float, center=None) -> np.ndarray:
    """Brightest-peak origin per position, NaN where nothing is found."""
    peaks = bragg_vectors.peaks
    scan_r, scan_c = peaks.shape[0], peaks.shape[1]
    H, W = int(bragg_vectors.dataset.shape[-2]), int(bragg_vectors.dataset.shape[-1])
    c0 = np.array([H / 2, W / 2]) if center is None else np.asarray(center, dtype=float)
    meas = np.full((scan_r, scan_c, 2), np.nan)
    for r in range(scan_r):
        for c in range(scan_c):
            arr = peaks[r, c].numpy().astype(np.float64)
            if arr.shape[0] == 0:
                continue
            d = np.hypot(arr[:, 0] - c0[0], arr[:, 1] - c0[1])
            near = d < search_radius
            if not near.any():
                continue
            sub = arr[near]
            meas[r, c] = sub[np.argmax(sub[:, 2]), :2]
    return meas


def _plot_origin_panels(meas: np.ndarray, origins: np.ndarray):
    """Measured origins, plane fit and residual for both detector axes."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(13.5, 5.6))
    names = ["row", "col"]
    for k in range(2):
        m, f = meas[..., k], origins[..., k]
        resid = m - f
        mean_m = np.nanmean(m)
        span = max(np.nanstd(m) * 3, 1e-3)
        for j, (img, title) in enumerate(
            [
                (m, f"measured origin {names[k]} (px)"),
                (f, f"plane fit {names[k]} (px)"),
                (resid, f"residual {names[k]} (px)"),
            ]
        ):
            c0 = 0.0 if j == 2 else mean_m
            sp = max(np.nanstd(resid) * 3, 1e-3) if j == 2 else span
            im = axs[k, j].imshow(
                img,
                cmap="RdBu_r",
                vmin=c0 - sp,
                vmax=c0 + sp,
                interpolation="nearest",
            )
            axs[k, j].set_title(title, fontsize=10)
            axs[k, j].set_xticks([])
            axs[k, j].set_yticks([])
            fig.colorbar(im, ax=axs[k, j], shrink=0.85)
    fig.tight_layout()
    return fig, axs


def plot_origin_fit(bragg_vectors, origins: np.ndarray, search_radius: float = 6.0, center=None):
    """Measured origins against the plane fit of :func:`measure_origins`.

    Parameters
    ----------
    bragg_vectors : BraggVectors
        With detected peaks.
    origins : np.ndarray
        (scan_row, scan_col, 2) fitted origins from :func:`measure_origins`.
    search_radius : float, default=6.0
        Radius in detector pixels searched around `center`, as passed to
        :func:`measure_origins`.
    center : tuple of float, optional
        ``(row, col)`` detector position searched around, as passed to
        :func:`measure_origins`. Defaults to the detector center.

    Returns
    -------
    tuple
        ``(fig, axs)``, axs of shape (2, 3): rows are the detector row and
        column, columns are measured, fit and residual, all in pixels.
    """
    meas = _measure_raw_origins(bragg_vectors, search_radius, center)
    return _plot_origin_panels(meas, origins)


def measure_origins(
    bragg_vectors,
    search_radius: float = 6.0,
    robust: bool = True,
    plot: bool = False,
    center=None,
    min_coverage: float = 0.1,
):
    """Per-position diffraction origin from the brightest central peak.

    At each scan position the most intense detected peak within
    `search_radius` pixels of `center` is taken as the direct beam; a plane
    is fit over the scan (least squares, optionally with one
    outlier-rejection pass) to model the descan.

    Parameters
    ----------
    bragg_vectors : BraggVectors
        With detected peaks, fields (q_row, q_col, intensity) in detector
        pixels, not yet origin-corrected.
    search_radius : float, default=6.0
        Radius in detector pixels searched around `center`.
    robust : bool, default=True
        Reject outliers before the plane fit.
    plot : bool, default=False
        Show the fitted origin planes and the residuals of the measured
        origins against the fit.
    center : tuple of float, optional
        ``(row, col)`` detector position to search around, e.g. from
        :func:`~quantem.diffraction.disk_detection.estimate_central_beam`.
        Defaults to the detector centre, which misses a beam that sits
        further than `search_radius` from it.
    min_coverage : float, default=0.1
        Fraction of scan positions that must yield an origin. Below it the
        search has missed the beam, and the fit would be meaningless, so an
        error is raised rather than returning a plane through nothing.

    Returns
    -------
    np.ndarray
        (scan_row, scan_col, 2) plane-fit origins, ready for
        BraggVectors.correct_peak_origins(). With plot=True, also returns
        (fig, axs).

    Raises
    ------
    ValueError
        If fewer than `min_coverage` of the positions have a peak within
        `search_radius` of `center`.
    """
    meas = _measure_raw_origins(bragg_vectors, search_radius, center)
    scan_r, scan_c = meas.shape[0], meas.shape[1]
    coverage = float(np.isfinite(meas[..., 0]).mean())
    if coverage < min_coverage:
        H, W = int(bragg_vectors.dataset.shape[-2]), int(bragg_vectors.dataset.shape[-1])
        c0 = (H / 2, W / 2) if center is None else tuple(float(v) for v in center)
        raise ValueError(
            f"only {coverage:.1%} of scan positions have a peak within "
            f"{search_radius:g} px of ({c0[0]:.1f}, {c0[1]:.1f}); the direct beam is "
            "elsewhere. Pass center= from estimate_central_beam(dataset.dp_mean), "
            "or widen search_radius."
        )
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
        fig, axs = _plot_origin_panels(meas, out)
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
    name : str, default="bragg_peaks_calibrated"
        Name of the returned Vector.

    Returns
    -------
    Vector
        Fields (qx, qy, intensity) in 1/Angstroms.
    """
    scan_r, scan_c = peaks_px.shape[0], peaks_px.shape[1]
    flat = peaks_px.select_fields("q_row", "q_col", "intensity").numpy().astype(np.float64)
    row_counts = np.asarray(peaks_px.row_counts(), dtype=int)
    qrc = flat[:, :2] * pixel_size_inv_A
    if ellipse is not None:
        qrc = qrc @ _ellipse_matrix(ellipse).T
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
        dtype=peaks_px.dtype,
    )
    # carry the scan calibration through, so maps keep their scale bar, and
    # record the detector-to-scan rotation so pattern-overlay plots can put
    # peaks back into the raw detector frame
    for key in ("scan_sampling", "scan_units", "origins", "origin_ref"):
        if key in (peaks_px.metadata or {}):
            out.metadata[key] = peaks_px.metadata[key]
    out.metadata["rotation_ccw_deg"] = float(rotation_ccw_deg)
    return out


def scale_peaks(peaks, scale: float):
    """Return a copy of a (qx, qy, intensity) Vector with q scaled.

    Parameters
    ----------
    peaks : Vector
        Peaks with fields (qx, qy, intensity).
    scale : float
        Factor applied to qx and qy, e.g. from :func:`calibrate_pixel_size`.

    Returns
    -------
    Vector
        Scaled copy of `peaks`.
    """
    out = peaks.copy()
    flat = out.numpy().astype(np.float64)
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

    Parameters
    ----------
    peaks : Vector
        Calibrated peaks with fields (qx, qy, intensity) in 1/Angstroms.
    k_min, k_max : float, default=0.05, 1.5
        Range of the bins, 1/Angstroms.
    k_step : float, default=0.002
        Bin width, 1/Angstroms.
    bragg_k_power : float, default=2.0
        Each peak is weighted by ``|q| ** bragg_k_power``, which offsets the
        fall-off of the scattering factors with q.
    bragg_intensity_power : float, default=1.0
        Each peak is weighted by ``intensity ** bragg_intensity_power``; 0
        counts every peak equally.

    Returns
    -------
    k : np.ndarray
        Bin centers (1/Angstroms).
    hist : np.ndarray
        Weighted counts, with linear interpolation between adjacent bins.
    """
    flat = peaks.select_fields("qx", "qy", "intensity").numpy().astype(np.float64)
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
    """1D ring profile of a crystal: Gaussians at |g| weighted by intensity.

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated.
    k : np.ndarray
        Scattering vectors to evaluate at, 1/Angstroms.
    k_broadening : float, default=0.01
        Gaussian standard deviation of each ring, 1/Angstroms.
    bragg_k_power : float, default=2.0
        Each ring is weighted by ``|g| ** bragg_k_power`` times its intensity.

    Returns
    -------
    np.ndarray
        Profile, same shape as `k`.
    """
    g = crystal.g_len.numpy()
    w = crystal.struct_factors_int.numpy() * g**bragg_k_power
    prof = (w[None, :] * np.exp(-((k[:, None] - g[None, :]) ** 2) / (2 * k_broadening**2))).sum(
        axis=1
    )
    return prof


def calibrate_pixel_size_matching(
    peaks,
    crystal: Crystal | list[Crystal],
    energy_ev: float,
    scales: np.ndarray | None = None,
    subsample: int = 8,
    angle_step_deg: float = 3.0,
    corr_kernel_size: float = 0.02,
    min_number_peaks: int = MIN_NUMBER_PEAKS,
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
    angle_step_deg : float, default=3.0
        Zone-axis and in-plane angular step of the orientation plan, degrees.
    corr_kernel_size : float, default=0.02
        Matching kernel width, 1/Angstroms. Keep it near the peak position
        noise: a wide kernel gives partial credit to near-coincident rings
        at a wrong scale.
    min_number_peaks : int, default=MIN_NUMBER_PEAKS
        Positions with fewer peaks are not matched.
    plot : bool, default=False
        Plot the score against the scale factor.
    return_scores : bool, default=False
        Also return the scale factors and their scores.
    returnfig : bool, default=False
        With `plot`, also return the figure and axes.

    Returns
    -------
    scale : float
        Best scale factor (parabolic refinement over the score maximum).
        Multiply the pixel size by it.
    scales, scores : np.ndarray
        Only with `return_scores`. The score is the mean over phases of the
        median correlation of the matched positions, 0 for a phase where
        nothing matched.
    fig, ax
        Only with `plot` and `returnfig`.
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
            matched = corr[corr > 0]
            # a phase that indexes nothing at this scale scores zero
            per_phase.append(float(matched.median()) if matched.numel() > 0 else 0.0)
        scores[i] = float(np.mean(per_phase))

    i_best = int(np.nanargmax(scores)) if np.isfinite(scores).any() else 0
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
    scale_step : float, default=5e-4
        Step of the scale factor scan.
    k_min, k_max : float, default=0.05, 1.3
        Range of scattering vectors compared, 1/Angstroms, after scaling.
    k_broadening : float, default=0.01
        Gaussian standard deviation of the simulated rings, 1/Angstroms.
    bragg_k_power : float, default=2.0
        Rings are weighted by ``|g| ** bragg_k_power``; see
        :func:`simulated_ring_profile`.
    plot : bool, default=False
        Show the scaled histogram against the crystal ring profile.
    returnfig : bool, default=False
        With `plot`, also return the figure and axes.

    Returns
    -------
    scale : float
        Multiply existing q values (and the pixel size) by this factor,
        e.g. with scale_peaks().
    fig, ax
        Only with `plot` and `returnfig`.
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
        prof = simulated_ring_profile(crystal, k * scale, k_broadening / 2, bragg_k_power)
        ax.fill_between(
            k * scale,
            hist / hist.max(),
            color="r",
            alpha=0.75,
            lw=0,
            label="measured (scaled)",
        )
        ax.plot(
            k * scale,
            prof / prof.max(),
            "k-",
            lw=1.0,
            label=f"{crystal.name} rings",
        )
        ax.set_ylabel("intensity (norm.)")
        ax.set_xlabel(r"scattering vector (1/$\mathrm{\AA}$)")
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
    measured field cannot distinguish theta from theta + 180. Only the
    candidate in [0, 180) is returned; the other is that angle + 180. Pick
    the one consistent with a known feature (e.g. a Burgers orientation
    relationship, or the divergence sign convention of DPC). The returned
    angle is ready to pass to peaks_to_calibrated() as rotation_ccw_deg.

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
    returnfig : bool, default=False
        With `plot`, also return the figure and axes.

    Returns
    -------
    rotation_ccw_deg : float
        Curl-minimizing rotation in [0, 180), degrees, on a 0.25 degree
        grid; the physical answer is either this angle or this angle + 180.
    fig, ax
        Only with `plot` and `returnfig`.
    """
    arr = dataset.array
    scan_r, scan_c, H, W = arr.shape
    rows = np.arange(H, dtype=float)[:, None]
    cols = np.arange(W, dtype=float)[None, :]
    if origins is None:
        origins = np.zeros((scan_r, scan_c, 2))
        origins[..., 0] = H / 2
        origins[..., 1] = W / 2
    origins = np.asarray(origins, dtype=float)
    # one scan row at a time: a float64 copy of a full scan is several times
    # the size of the data (19 GB for 256 x 256 x 192 x 192), and the center
    # of mass only ever needs one pattern
    com_r = np.zeros((scan_r, scan_c))
    com_c = np.zeros((scan_r, scan_c))
    for i in range(scan_r):
        block = np.asarray(arr[i], dtype=float)  # (scan_c, H, W)
        if mask_radius is not None:
            rr = rows[None] - origins[i, :, 0][:, None, None]
            cc = cols[None] - origins[i, :, 1][:, None, None]
            block = block * (rr**2 + cc**2 <= mask_radius**2)
        tot = block.sum(axis=(-2, -1))
        tot[tot <= 0] = 1.0
        com_r[i] = (block * rows).sum(axis=(-2, -1)) / tot - origins[i, :, 0]
        com_c[i] = (block * cols).sum(axis=(-2, -1)) / tot - origins[i, :, 1]

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
            "scan rotation = %.1f deg (or %.1f)" % (rotation_ccw_deg, rotation_ccw_deg + 180)
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
    k_min, k_max : float, default=0.15, 1.4
        Radial range (1/Angstroms) included in the sharpness measure.
    n_bins : int, default=800
        Number of log-radius bins between `k_min` and `k_max`.
    bragg_k_power : float, default=2.0
        Each peak is weighted by ``|q| ** bragg_k_power``.
    bragg_intensity_power : float, default=1.0
        Each peak is weighted by ``intensity ** bragg_intensity_power``.
    plot : bool, default=False
        Show the radial histogram before and after the correction.
    returnfig : bool, default=False
        With `plot`, also return the figure and axes.

    Returns
    -------
    ellipse : np.ndarray
        [e11, e12], the correction matrix [[1 + e11, e12], [e12, 1 - e11]]
        that undoes the distortion; pass to peaks_to_calibrated(ellipse=...)
        or apply_ellipse().
    fig, ax
        Only with `plot` and `returnfig`.
    """
    from scipy.optimize import minimize

    flat = peaks.select_fields("qx", "qy", "intensity").numpy().astype(np.float64)
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
        ax.fill_between(k_bins, h0 / h0.max(), color="0.7", lw=0, label="measured")
        ax.plot(k_bins, h1 / h1.max(), "r-", lw=1.0, label="ellipse corrected")
        ax.set_xlabel(r"scattering vector (1/$\mathrm{\AA}$)")
        ax.set_ylabel("intensity (norm.)")
        mag = np.hypot(*ellipse)
        ax.set_title(
            "e11 = %.2e, e12 = %.2e  (%.2f%% ellipticity)" % (ellipse[0], ellipse[1], 200 * mag)
        )
        ax.legend()
        if returnfig:
            out += [fig, ax]
    return tuple(out) if len(out) > 1 else out[0]


def _ellipse_matrix(ellipse) -> np.ndarray:
    """The correction matrix [[1 + e11, e12], [e12, 1 - e11]] of an ellipse."""
    e11, e12 = float(ellipse[0]), float(ellipse[1])
    return np.array([[1 + e11, e12], [e12, 1 - e11]])


def _compose_ellipse(ellipse_new, ellipse_prev) -> np.ndarray:
    """One ellipse equivalent to applying `ellipse_prev`, then `ellipse_new`.

    The product of the two correction matrices is reduced to the traceless
    symmetric form [[1 + e11, e12], [e12, 1 - e11]]: its isotropic part is
    a pixel size change, fit separately, and its antisymmetric part is a
    rotation of second order in the ellipse components.

    Parameters
    ----------
    ellipse_new : array-like
        [e11, e12] fit on peaks that already have `ellipse_prev` applied.
    ellipse_prev : array-like | None
        [e11, e12] already applied, or None.

    Returns
    -------
    np.ndarray
        Combined [e11, e12].
    """
    if ellipse_prev is None:
        return np.asarray(ellipse_new, dtype=float)
    A = _ellipse_matrix(ellipse_new) @ _ellipse_matrix(ellipse_prev)
    sym = 0.5 * (A + A.T) / (0.5 * np.trace(A))
    return np.array([0.5 * (sym[0, 0] - sym[1, 1]), sym[0, 1]])


def apply_ellipse(peaks, ellipse):
    """Return a copy of (qx, qy, intensity) peaks with the ellipse applied.

    Parameters
    ----------
    peaks : Vector
        Peaks with fields (qx, qy, intensity).
    ellipse : array-like
        [e11, e12] from :func:`calibrate_ellipse`. Each q is mapped by the
        correction matrix [[1 + e11, e12], [e12, 1 - e11]].

    Returns
    -------
    Vector
        Corrected copy of `peaks`.
    """
    out = peaks.copy()
    flat = out.numpy().astype(np.float64)
    flat[:, :2] = flat[:, :2] @ _ellipse_matrix(ellipse).T
    out.set_flattened(flat)
    return out


def zone_reflections(crystal: Crystal, zone_axis) -> Crystal:
    """The crystal restricted to the reflections of one zone.

    A specimen that sits near one zone axis everywhere -- a flake lying flat,
    a textured film -- only ever shows the reflections of that zone, so its
    radial histogram holds those rings and no others. Calibrating or
    plotting against every ring of the crystal then compares the peaks with
    rings that cannot appear. This keeps the reflections hkl with
    h u + k v + l w = 0 for the zone axis [uvw] and drops the rest.

    Parameters
    ----------
    crystal : Crystal
        With structure factors calculated.
    zone_axis : sequence of int
        Zone axis direction in the crystal's own cell, [uvw], or [UVTW] for a
        hexagonal or trigonal cell, e.g. (0, 0, 0, 1) for the basal plane.

    Returns
    -------
    Crystal
        A shallow copy whose reflection list holds only that zone, named
        after it; the original is unchanged.
    """
    import copy

    if crystal.g_vec is None:
        raise RuntimeError(f"{crystal.name}: run calculate_structure_factors() first")
    z = np.asarray(zone_axis, dtype=float).ravel()
    if z.size == 4:
        U, V, T, W = z
        uvw = np.array([U - T, V - T, W])
        label = "[" + "".join(str(int(round(v))) for v in z) + "]"
    elif z.size == 3:
        uvw = z
        label = "[" + "".join(str(int(round(v))) for v in z) + "]"
    else:
        raise ValueError(f"zone_axis must have 3 or 4 indices, got {zone_axis}")
    keep = torch.as_tensor(np.abs(crystal.hkl.numpy() @ uvw) < 1e-6)
    out = copy.copy(crystal)
    for name in ("hkl", "g_vec", "g_len", "struct_factors", "struct_factors_int"):
        setattr(out, name, getattr(crystal, name)[keep])
    out.name = f"{crystal.name} {label}"
    return out


def _restrict(crystals, zone_axis):
    """Crystal list, each restricted to `zone_axis` when one is given."""
    xtls = [crystals] if isinstance(crystals, Crystal) else list(crystals)
    return xtls if zone_axis is None else [zone_reflections(x, zone_axis) for x in xtls]


def _hkl_label(hkl: np.ndarray, hexagonal: bool) -> str:
    """Compact (hkl) / (hkil) plane label with unicode overbars."""

    def digit(v: int) -> str:
        v = int(round(v))
        txt = str(abs(v))
        return txt + "̅" if v < 0 else txt

    h, k, ll = (int(round(v)) for v in hkl)
    if hexagonal:
        return "(" + digit(h) + digit(k) + digit(-(h + k)) + digit(ll) + ")"
    return "(" + digit(h) + digit(k) + digit(ll) + ")"


def _crystal_rings(crystal: Crystal, k_min: float, k_max: float) -> np.ndarray:
    """Distinct ring radii of a crystal with non-zero structure factor."""
    g = np.linalg.norm(np.asarray(crystal.g_vec), axis=1)
    f = np.asarray(crystal.struct_factors_int)
    keep = (g > k_min) & (g < k_max) & (f > 1e-6 * f.max())
    return np.array(sorted(set(np.round(g[keep], 4))))


def _histogram_maxima(peaks, k_min: float, k_max: float, bragg_k_power: float) -> np.ndarray:
    """Radii of the local maxima of the measured radial histogram."""
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d

    k, hist = radial_histogram(peaks, k_min=k_min, k_max=k_max, bragg_k_power=bragg_k_power)
    h = gaussian_filter1d(hist, 2.0)
    loc = (h == maximum_filter1d(h, 15)) & (h > 0.05 * h.max())
    return k[loc]


class DiffractionCalibration(AutoSerialize):
    """Reciprocal-space calibration of a detector, measured once and reused.

    Holds the reciprocal pixel size, the elliptic distortion and the
    diffraction-to-scan rotation, with the evidence behind them. Strained
    samples cannot calibrate themselves, so the normal route is to measure
    this on a standard such as nanocrystalline gold, save it, and apply it
    to the peaks of the sample of interest::

        cal = calibrate(peaks_au, gold, 0.01)
        cal.save("detector_300kV_80cm.zip", mode="o")
        ...
        cal = load("detector_300kV_80cm.zip")
        peaks = cal.apply(bv_centered.peaks)

    A calibration is tied to the detector binning it was measured at, which
    is recorded in `metadata`; `rebin` converts it to another binning.

    Parameters
    ----------
    pixel_size : float
        Reciprocal pixel size, 1/Angstroms per detector pixel.
    ellipse : array-like | None
        [e11, e12] elliptic distortion correction, see
        :func:`calibrate_ellipse`. None applies no correction.
    rotation_ccw_deg : float, default=0.0
        Diffraction-to-scan rotation in degrees, see
        :func:`measure_scan_rotation`.
    metadata : dict | None
        Evidence and provenance, e.g. the reference phases, the matched
        rings and their residual, and the detector binning.
    """

    def __init__(
        self,
        pixel_size: float,
        ellipse=None,
        rotation_ccw_deg: float = 0.0,
        metadata: dict | None = None,
    ):
        self.pixel_size = float(pixel_size)
        self.ellipse = None if ellipse is None else np.asarray(ellipse, dtype=float)
        self.rotation_ccw_deg = float(rotation_ccw_deg)
        self.metadata: dict = dict(metadata or {})

    def apply(self, peaks_px, name: str = "bragg_peaks_calibrated"):
        """Calibrated (qx, qy) peaks from origin-corrected pixel peaks.

        Parameters
        ----------
        peaks_px : Vector
            Origin-corrected peaks in detector pixels, from
            :meth:`~quantem.diffraction.BraggVectors.correct_peak_origins`.
        name : str, default="bragg_peaks_calibrated"
            Name of the returned Vector.

        Returns
        -------
        Vector
            Peaks in 1/Angstroms, carrying the measured pixel size and the
            scan calibration forward so plots downstream need neither passed
            to them.
        """
        out = peaks_to_calibrated(
            peaks_px,
            self.pixel_size,
            rotation_ccw_deg=self.rotation_ccw_deg,
            ellipse=self.ellipse,
            name=name,
        )
        # the measured pixel size replaces whatever the dataset was carrying
        out.metadata["pixel_size"] = float(self.pixel_size)
        out.metadata["pixel_size_units"] = "A^-1"
        return out

    def rebin(self, factor: float) -> "DiffractionCalibration":
        """The same calibration for data binned by `factor` more than this one.

        Parameters
        ----------
        factor : float
            Additional detector binning. The pixel size is multiplied by it;
            the ellipse and rotation do not depend on binning.

        Returns
        -------
        DiffractionCalibration
            New calibration with ``metadata["binning"]`` updated.
        """
        md = dict(self.metadata)
        md["binning"] = md.get("binning", 1) * factor
        return DiffractionCalibration(
            self.pixel_size * factor, self.ellipse, self.rotation_ccw_deg, md
        )

    def __repr__(self) -> str:
        e = "none" if self.ellipse is None else "e11 %+.5f e12 %+.5f" % tuple(self.ellipse)
        rms = self.metadata.get("residual_rms")
        q = (
            ""
            if rms is None
            else ", %d rings, rms %.2f%%"
            % (
                self.metadata.get("n_rings", 0),
                100 * rms,
            )
        )
        return (
            f"DiffractionCalibration(pixel_size={self.pixel_size:.6f} 1/A/px, "
            f"ellipse: {e}, rotation {self.rotation_ccw_deg:g} deg{q})"
        )


def _ring_profile_scores(
    peaks,
    crystals,
    scales: np.ndarray,
    k_broadening: float,
    k_min: float,
    k_max: float,
    bragg_k_power: float,
) -> np.ndarray:
    """Normalized overlap of the measured rings with the reference rings, for
    each trial scale.

    The score is the fraction of the total measured peak weight that lands on
    a reference ring. Two normalizations matter and both are traps. Scaling
    the reference rather than the data moves the comparison window with the
    scale, and re-normalizing to the weight left inside the window rewards a
    scale for pushing peaks out of it: either one lets a wrong scale beat the
    truth. Dividing by the total weight, counted once and independent of the
    scale, makes a lost peak a loss.
    """
    flat = peaks.select_fields("qx", "qy", "intensity").numpy().astype(np.float64)
    qr = np.hypot(flat[:, 0], flat[:, 1])
    weight = flat[:, 2] * qr**bragg_k_power
    k_step = 0.002
    k = np.arange(k_min, k_max, k_step)
    prof = np.zeros_like(k)
    for xtl in crystals:
        prof += simulated_ring_profile(xtl, k, k_broadening, bragg_k_power)
    prof = prof / max(prof.max(), 1e-12)
    total = float(weight.sum())
    scores = np.zeros_like(scales)
    for i, sc in enumerate(scales):
        frac = (qr * sc - k_min) / k_step
        i0 = np.floor(frac).astype(int)
        w1 = frac - i0
        ok = (i0 >= 0) & (i0 < k.size - 1)
        h = np.bincount(i0[ok], weights=weight[ok] * (1 - w1[ok]), minlength=k.size)
        h += np.bincount(i0[ok] + 1, weights=weight[ok] * w1[ok], minlength=k.size)
        scores[i] = float((h[: k.size] * prof).sum() / total) if total > 0 else 0.0
    return scores


def _fit_scale(peaks, crystals, lo, hi, k_broadening, k_min, k_max, bragg_k_power, n=241):
    """Best scale in [lo, hi] with parabolic refinement of the maximum."""
    scales = np.linspace(lo, hi, n)
    scores = _ring_profile_scores(
        peaks, crystals, scales, k_broadening, k_min, k_max, bragg_k_power
    )
    i = int(np.argmax(scores))
    best = float(scales[i])
    if 0 < i < n - 1:
        c0, c1, c2 = scores[i - 1 : i + 2]
        denom = 4 * c1 - 2 * c0 - 2 * c2
        if abs(denom) > 1e-12:
            best += (c2 - c0) / denom * (scales[1] - scales[0])
    return best, scales, scores


def calibrate(
    peaks_px,
    crystal,
    pixel_size_guess: float,
    rotation_ccw_deg: float = 0.0,
    fit_ellipse: bool = True,
    n_iter: int = 3,
    scale_search: tuple[float, float] = (0.6, 1.7),
    k_min: float = 0.05,
    k_max: float = 1.3,
    k_broadening: float = 0.01,
    bragg_k_power: float = 2.0,
    residual_tol: float = 0.01,
    plot: bool = False,
    figsize: tuple[float, float] = (13.0, 6.4),
    marker_size: float = 8.0,
    zone_axis=None,
    returnfig: bool = False,
):
    """Measure the reciprocal pixel size and the elliptic distortion.

    Three stages, each one removing the reason the next could fail. A coarse
    scan over `scale_search` with a deliberately broadened ring profile finds
    the right ring assignment even when the starting pixel size is far out;
    a broad profile has one maximum where a sharp one has many. The ellipse
    and the scale are then refined in turn, the ellipse in log-radius bins
    where it does not depend on the scale, and the scale against
    progressively sharper rings. Finally every measured ring is matched to
    its reference ring separately, which is the only check that can tell a
    correct calibration from a plausible one: a single pixel size that
    explains the pattern gives per-ring scale factors agreeing to a few
    tenths of a percent with no trend in k.

    Parameters
    ----------
    peaks_px : Vector
        Origin-corrected peaks in detector pixels, (q_row, q_col, intensity).
    crystal : Crystal | list[Crystal]
        Reference phase or phases, structure factors calculated.
    pixel_size_guess : float
        Starting reciprocal pixel size (1/Angstroms per pixel). Only the
        order of magnitude matters; the coarse scan covers `scale_search`.
    rotation_ccw_deg : float, default=0.0
        Diffraction-to-scan rotation recorded on the calibration.
    fit_ellipse : bool, default=True
        Fit the elliptic distortion as well as the scale.
    n_iter : int, default=3
        Ellipse and scale refinement rounds. Each round fits the ellipse
        left over after the current correction and composes the two.
    scale_search : tuple, default=(0.6, 1.7)
        Capture range of the coarse scan, as a multiple of the guess.
    k_min, k_max : float, default=0.05, 1.3
        Range of scattering vectors fit, 1/Angstroms. The ellipse fit uses
        at least 0.15 as its lower limit.
    k_broadening : float, default=0.01
        Gaussian standard deviation of the reference rings in the fine scale
        fit, 1/Angstroms; the coarse scan uses six times this. Measured rings
        further than four times this from any reference ring are not counted
        in the per-ring check.
    bragg_k_power : float, default=2.0
        Peaks and rings are weighted by ``|q| ** bragg_k_power``.
    residual_tol : float, default=0.01
        Per-ring residual rms above which the fit is reported as unreliable.
    plot : bool, default=False
        Show the ring comparison before and after the fit, for the first
        reference crystal.
    figsize : tuple, default=(13, 6.4)
        Figure size.
    marker_size : float, default=8.0
        Area of the brightest peak in the azimuth panels, where every peak is
        drawn with area proportional to its intensity. Raise it to bring out
        weak spots, lower it when strong ones hide the reference lines.
    zone_axis : sequence of int, optional
        Fit only the rings of this zone, [uvw] or [UVTW] (see
        :func:`zone_reflections`). For a specimen near one zone axis
        everywhere, whose peaks hold no other rings.
    returnfig : bool, default=False
        With `plot`, also return the figure and axes.

    Returns
    -------
    DiffractionCalibration
        The pixel size, ellipse and rotation. ``metadata`` holds the matched
        rings as (measured k, reference k, ratio), their residual rms and
        whether the fit passed `residual_tol`. With `plot` and `returnfig`,
        ``(cal, fig, axs)`` instead, axs of shape (2, 2).

    Warns
    -----
    UserWarning
        If fewer than three rings match or their residual rms exceeds
        `residual_tol`.

    Notes
    -----
    The figure shows the first reference crystal before and after the fit.
    To check every candidate phase against the result, including phases the
    calibration was not fit to, plot :func:`plot_calibration` on the
    calibrated peaks, or :meth:`CrystalMap.plot_calibration`.
    """
    crystals = [crystal] if isinstance(crystal, Crystal) else list(crystal)
    for xtl in crystals:
        if xtl.g_vec is None:
            raise RuntimeError(f"{xtl.name}: run calculate_structure_factors() first")
    crystals = _restrict(crystals, zone_axis)

    peaks_0 = peaks_to_calibrated(peaks_px, pixel_size_guess)
    # coarse: broad rings so the score has a single maximum over a wide range
    scale, _, _ = _fit_scale(
        peaks_0,
        crystals,
        scale_search[0],
        scale_search[1],
        6 * k_broadening,
        k_min,
        k_max,
        bragg_k_power,
    )
    ellipse = None
    for it in range(max(1, n_iter)):
        if fit_ellipse:
            pk = peaks_to_calibrated(peaks_px, pixel_size_guess * scale, ellipse=ellipse)
            # the fit sees peaks with the current ellipse already applied, so
            # it returns the residual distortion: compose it with the current
            # correction instead of replacing it
            residual = calibrate_ellipse(
                pk, k_min=max(k_min, 0.15), k_max=k_max, bragg_k_power=bragg_k_power
            )
            ellipse = _compose_ellipse(residual, ellipse)
        pk = peaks_to_calibrated(peaks_px, pixel_size_guess, ellipse=ellipse)
        half = 0.08 / (it + 1)
        scale, _, _ = _fit_scale(
            pk,
            crystals,
            scale * (1 - half),
            scale * (1 + half),
            k_broadening,
            k_min,
            k_max,
            bragg_k_power,
        )

    pixel_size = pixel_size_guess * scale
    peaks = peaks_to_calibrated(
        peaks_px, pixel_size, rotation_ccw_deg=rotation_ccw_deg, ellipse=ellipse
    )

    rings = np.concatenate([_crystal_rings(x, k_min, k_max * 1.15) for x in crystals])
    rings = np.array(sorted(set(np.round(rings, 4))))
    k_meas = _histogram_maxima(peaks, k_min, k_max * 1.15, bragg_k_power)
    table = []
    for km in k_meas:
        j = int(np.argmin(np.abs(rings - km)))
        if abs(rings[j] - km) < 4 * k_broadening:
            table.append((float(km), float(rings[j]), float(rings[j] / km)))
    per_ring = np.array([t[2] for t in table]) if table else np.array([np.nan])
    residual_rms = float(np.std(per_ring / np.median(per_ring))) if table else float("nan")
    reliable = len(table) >= 3 and residual_rms <= residual_tol

    cal = DiffractionCalibration(
        pixel_size,
        ellipse,
        rotation_ccw_deg,
        metadata=dict(
            reference=[x.name for x in crystals],
            pixel_size_guess=float(pixel_size_guess),
            scale=float(scale),
            n_rings=len(table),
            residual_rms=residual_rms,
            rings=table,
            reliable=bool(reliable),
            binning=1,
        ),
    )
    if not reliable:
        warnings.warn(
            f"calibration looks unreliable: {len(table)} rings matched, residual rms "
            f"{100 * residual_rms:.2f}% (tolerance {100 * residual_tol:.2f}%). Check the "
            "reference phase and the peak detection before using this pixel size.",
            stacklevel=2,
        )
    if not plot:
        return cal

    import matplotlib.pyplot as plt

    k_hi = k_max * 1.15
    fig, axs = plt.subplots(
        2,
        2,
        figsize=figsize,
        sharex="col",
        gridspec_kw={"height_ratios": [1, 1.3]},
    )
    for col, (pk, ttl, kb) in enumerate(
        (
            (peaks_0, f"before: {pixel_size_guess:.5f} " + r"$\mathrm{\AA}^{-1}$/px", None),
            (peaks, f"after: {pixel_size:.5f} " + r"$\mathrm{\AA}^{-1}$/px", k_broadening),
        )
    ):
        _calibration_panels(
            axs[0, col],
            axs[1, col],
            pk,
            crystals[0],
            k_min=k_min,
            k_max=k_hi,
            k_broadening=kb,
            bragg_k_power=bragg_k_power,
            marker_size=marker_size,
        )
        axs[0, col].set_title(ttl, fontsize=10)
    axs[1, 1].set_ylabel("")
    if ellipse is not None:
        axs[1, 0].text(
            0.02,
            0.97,
            "ellipse e11 %+.4f  e12 %+.4f" % tuple(ellipse),
            transform=axs[1, 0].transAxes,
            va="top",
            fontsize=9,
        )
    axs[1, 1].text(
        0.02,
        0.97,
        f"{len(table)} rings, rms {100 * residual_rms:.2f}%"
        + ("" if reliable else "  UNRELIABLE"),
        transform=axs[1, 1].transAxes,
        va="top",
        fontsize=9,
        color="k" if reliable else "tab:red",
    )
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.08)
    return (cal, fig, axs) if returnfig else cal


# reflections closer than this in |g| (1/Angstroms) are drawn as one ring
_SHELL_STEP = 0.005


def _shells(crystal: Crystal, bragg_k_power: float):
    """Group a crystal's reflections into rings.

    Returns
    -------
    radius : np.ndarray
        Mean |g| of each ring, 1/Angstroms, ascending.
    intensity : np.ndarray
        Summed intensity of each ring, weighted by ``|g| ** bragg_k_power``.
    index : np.ndarray
        Ring index of every reflection.
    """
    g = crystal.g_len.numpy()
    ints = crystal.struct_factors_int.numpy() * g**bragg_k_power
    _, index = np.unique(np.round(g / _SHELL_STEP).astype(np.int64), return_inverse=True)
    index = index.ravel()
    count = np.bincount(index)
    radius = np.bincount(index, weights=g) / count
    intensity = np.bincount(index, weights=ints)
    return radius, intensity, index


def _ring_shells(crystal: Crystal, k_min: float, k_max: float, bragg_k_power: float):
    """Ring radii of a crystal in (k_min, k_max) and their summed intensity,
    relative to the strongest ring in that range."""
    radius, intensity, _ = _shells(crystal, bragg_k_power)
    keep = (radius > k_min) & (radius < k_max)
    uniq, tot = radius[keep], intensity[keep]
    return uniq, tot / max(float(tot.max()), 1e-30) if tot.size else tot


def _calibration_panels(
    ax_hist,
    ax_az,
    peaks,
    crystal: Crystal,
    k_min: float,
    k_max: float,
    k_broadening: float | None = None,
    bragg_k_power: float = 2.0,
    marker_size: float = 8.0,
    n_rings: int = 12,
) -> None:
    """The radial histogram against one crystal's rings (top) and every peak
    as azimuth against scattering vector with its strongest rings (bottom)."""
    plot_ring_comparison(
        peaks,
        [crystal],
        k_min=k_min,
        k_max=k_max,
        k_broadening=k_broadening,
        bragg_k_power=bragg_k_power,
        n_labels=n_rings,
        figax=(ax_hist.figure, [ax_hist]),
    )
    ax_hist.set_xlabel("")
    # azimuth against scattering vector: a pixel size error shifts every
    # ring, the elliptic distortion is a cos(2 phi) wobble of each one
    flat = peaks.select_fields("qx", "qy", "intensity").numpy().astype(np.float64)
    r = np.hypot(flat[:, 0], flat[:, 1])
    phi = np.degrees(np.arctan2(flat[:, 1], flat[:, 0]))
    sel = (r > k_min) & (r < k_max)
    w = flat[sel, 2]
    hi = float(np.percentile(w, 99.5)) if w.size else 1.0
    ax_az.scatter(
        r[sel],
        phi[sel],
        s=marker_size * np.clip(w / max(hi, 1e-12), 0.03, 1.0),
        c="r",
        alpha=0.5,
        lw=0,
        rasterized=True,
    )
    radii, rel = _ring_shells(crystal, k_min, k_max, bragg_k_power)
    for g0 in radii[np.argsort(-rel)[:n_rings]]:
        ax_az.axvline(g0, color="k", lw=0.7, alpha=0.8)
    ax_az.set_xlim(k_min, k_max)
    ax_az.set_ylim(-180, 180)
    ax_az.set_yticks([-180, -90, 0, 90, 180])
    ax_az.set_xlabel(r"scattering vector (1/$\mathrm{\AA}$)")
    ax_az.set_ylabel("azimuth (deg)")


def plot_calibration(
    peaks,
    crystals,
    k_min: float = 0.05,
    k_max: float = 1.5,
    k_broadening: float | None = None,
    bragg_k_power: float = 2.0,
    marker_size: float = 8.0,
    n_rings: int = 12,
    zone_axis=None,
    axsize: tuple[float, float] = (6.0, 6.4),
    figax=None,
):
    """Calibrated peaks against the rings of every crystal, one column each.

    The calibration check for all candidate phases, however many of them the
    calibration itself was fit to. Each column holds the radial histogram of
    every peak against that crystal's rings (top) and every peak as azimuth
    against scattering vector with the same rings (bottom). A ring beside the
    measured peaks in every direction is a lattice parameter or pixel size
    error; a ring that wobbles with azimuth is elliptic distortion.

    Parameters
    ----------
    peaks : Vector
        Calibrated peaks (qx, qy, intensity) in 1/Angstroms.
    crystals : Crystal | list[Crystal]
        Candidate phases, structure factors calculated.
    k_min, k_max : float
        Range of scattering vectors shown, 1/Angstroms.
    k_broadening : float | None
        None draws sharp ring lines in the histograms; a width (1/Angstroms)
        draws the broadened ring profile instead.
    marker_size : float, default=8.0
        Area of the brightest peak in the azimuth panels.
    n_rings : int, default=12
        Rings drawn in the azimuth panels and labeled in the histograms,
        strongest first; a large cell would otherwise fill both with its
        weak superstructure rings.
    zone_axis : sequence of int, optional
        Show only the rings of this zone, [uvw] or [UVTW] (see
        :func:`zone_reflections`).
    axsize : tuple, default=(6.0, 6.4)
        Size of one column.
    figax : (fig, axs) | None
        Existing figure and a (2, n_crystals) array of axes.

    Returns
    -------
    tuple
        ``(fig, axs)``, axs of shape (2, n_crystals).
    """
    import matplotlib.pyplot as plt

    xtls = _restrict(crystals, zone_axis)
    if figax is None:
        fig, axs = plt.subplots(
            2,
            len(xtls),
            figsize=(axsize[0] * len(xtls), axsize[1]),
            sharex=True,
            squeeze=False,
            gridspec_kw={"height_ratios": [1, 1.3]},
        )
    else:
        fig, axs = figax
        axs = np.asarray(axs).reshape(2, len(xtls))
    for col, xtl in enumerate(xtls):
        _calibration_panels(
            axs[0, col],
            axs[1, col],
            peaks,
            xtl,
            k_min=k_min,
            k_max=k_max,
            k_broadening=k_broadening,
            bragg_k_power=bragg_k_power,
            marker_size=marker_size,
            n_rings=n_rings,
        )
        axs[0, col].set_title(xtl.name, fontsize=10)
        if col:
            axs[0, col].set_ylabel("")
            axs[1, col].set_ylabel("")
    if figax is None:
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.08)
    return fig, axs


def plot_ring_comparison(
    peaks,
    crystals,
    k_min: float = 0.1,
    k_max: float = 1.5,
    k_broadening: float | None = None,
    bragg_k_power: float = 2.0,
    label_hkl: bool = True,
    label_min_intensity: float = 0.05,
    n_labels: int = 12,
    zone_axis=None,
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
    k_min, k_max : float, default=0.1, 1.5
        Range of scattering vectors shown, 1/Angstroms.
    k_broadening : float | None
        None draws sharp lines at the ring positions; a value (1/Angstroms)
        draws the broadened ring profile instead.
    bragg_k_power : float, default=2.0
        Measured peaks and reference rings are weighted by
        ``|q| ** bragg_k_power``.
    label_hkl : bool, default=True
        Label the strongest rings by their Miller indices.
    label_min_intensity : float, default=0.05
        Label rings whose summed intensity exceeds this fraction of the
        strongest ring.
    n_labels : int, default=12
        Label at most this many rings, strongest first, which keeps a large
        cell with many rings readable.
    zone_axis : sequence of int, optional
        Show only the rings of this zone, [uvw] or [UVTW] (see
        :func:`zone_reflections`).
    figax : (fig, axs) | None
        Existing figure and one axis per crystal.

    Returns
    -------
    tuple
        ``(fig, axs)``, axs a 1D array with one axis per crystal.
    """
    import matplotlib.pyplot as plt

    xtls = _restrict(crystals, zone_axis)
    k, hist = radial_histogram(peaks, k_min=k_min, k_max=k_max, bragg_k_power=bragg_k_power)

    n = len(xtls)
    if figax is None:
        fig, axs = plt.subplots(n, 1, figsize=(11, 3.6 * n), sharex=True, squeeze=False)
        axs = axs[:, 0]
    else:
        fig, axs = figax
        axs = np.atleast_1d(axs)

    for ax, xtl in zip(axs, xtls):
        ax.fill_between(k, hist / hist.max(), color="r", alpha=0.75, lw=0, label="measured")
        hexagonal = xtl.hexagonal_matching
        g_len = xtl.g_len.numpy()
        ints = xtl.struct_factors_int.numpy() * g_len**bragg_k_power
        hkl_np = xtl.hkl.numpy()
        uniq, shell_int, shells = _shells(xtl, bragg_k_power)
        shell_int = shell_int / shell_int.max()

        if k_broadening is not None:
            prof = simulated_ring_profile(xtl, k, k_broadening, bragg_k_power)
            ax.plot(k, prof / prof.max(), "k-", lw=1.0, label=f"{xtl.name} rings")
        else:
            keep = (uniq > k_min) & (uniq < k_max)
            ax.vlines(
                uniq[keep],
                0,
                shell_int[keep],
                colors="k",
                lw=1.0,
                label=f"{xtl.name} rings",
            )

        if label_hkl:
            # the strongest rings first, each at least 0.03 1/A from the
            # last, then drawn left to right
            labeled: list[int] = []
            for j in np.argsort(-shell_int):
                u, si = uniq[j], shell_int[j]
                if len(labeled) >= n_labels:
                    break
                if u < k_min or u > k_max or si < label_min_intensity:
                    continue
                if any(abs(u - uniq[j0]) < 0.03 for j0 in labeled):
                    continue
                labeled.append(int(j))
            for rows, j in enumerate(sorted(labeled)):
                u = uniq[j]
                idx = np.nonzero(shells == j)[0]
                idx = idx[ints[idx] > 0.99 * ints[idx].max()]
                key = [tuple(-hkl_np[i]) for i in idx]
                best = idx[int(np.lexsort(np.array(key).T[::-1])[0])]
                # three staggered rows keep neighbouring labels apart
                y = 1.04 + 0.1 * (rows % 3)
                ax.text(
                    u,
                    y,
                    _hkl_label(hkl_np[best], hexagonal),
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )
        ax.set_ylabel("intensity (norm.)")
        ax.set_ylim(0, 1.42)
        # below the band of ring labels at the top
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.72), fontsize=8)
    axs[-1].set_xlabel(r"scattering vector (1/$\mathrm{\AA}$)")
    if figax is None:
        fig.tight_layout()
    return fig, axs


def transform_peaks(peaks, M: np.ndarray):
    """Return a copy of a (qx, qy, intensity) Vector with q mapped by M (2x2)."""
    out = peaks.copy()
    flat = out.numpy().astype(np.float64)
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

    Raises
    ------
    ValueError
        If `strain_maps` is empty or no position passes the filters.
    """
    As = []
    for i, sm in enumerate(strain_maps):
        A = np.stack([sm.g1_array, sm.g2_array], axis=-1)  # (R, C, 2, 2)
        ok = np.isfinite(A).all(axis=(-2, -1))
        dev = np.abs(A - np.eye(2)).max(axis=(-2, -1))
        ok &= dev < max_strain
        if masks is not None and masks[i] is not None:
            ok &= np.asarray(masks[i]) > 0
        As.append(A[ok])
    if not As:
        raise ValueError("strain_maps is empty: pass at least one StrainMap.")
    A_all = np.concatenate(As, axis=0)
    if A_all.shape[0] == 0:
        raise ValueError(
            "no positions left for the calibration residual: every strain fit "
            f"failed, was masked out, or deviates from the identity by more than "
            f"max_strain={max_strain:g}."
        )
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
    zone_axis=None,
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
    q_max : float | None
        Half-width of the histogram, 1/Angstroms. Defaults to just beyond
        the largest peak radius.
    bins : int, default=400
        Number of histogram bins along each axis.
    power : float, default=0.25
        The histogram is shown raised to this power, which brings out weak
        rings next to the direct beam.
    zone_axis : sequence of int, optional
        Draw only the rings of this zone, [uvw] or [UVTW] (see
        :func:`zone_reflections`).
    figax : (fig, ax) | None
        Existing figure and axis.

    Returns
    -------
    tuple
        ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    xtls = _restrict(crystals, zone_axis)
    flat = peaks.select_fields("qx", "qy", "intensity").numpy().astype(np.float64)
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
        uniq, shell_int, _ = _shells(xtl, 2.0)
        keep = uniq < q_max
        uniq, shell_int = uniq[keep], shell_int[keep]
        order = np.argsort(shell_int)[::-1][:n_rings]
        for k, u in enumerate(np.sort(uniq[order])):
            ax.plot(
                u * np.sin(th),
                u * np.cos(th),
                ls=styles[ci % 3],
                color=colors[ci % 3],
                lw=0.5,
                alpha=0.6,
                label=f"{xtl.name} rings" if k == 0 else None,
            )
    ax.set_xlabel(r"$q_c$ (1/$\mathrm{\AA}$)")
    ax.set_ylabel(r"$q_r$ (1/$\mathrm{\AA}$)")
    ax.legend(loc="upper right", fontsize=9)
    return fig, ax
