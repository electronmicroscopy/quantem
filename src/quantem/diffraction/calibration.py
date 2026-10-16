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

from quantem.core.datastructures.dataset4dstem import (  # noqa: F401  (re-exported)
    q_to_r_matrix,
    scan_rotation_from_metadata,
)
from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import AutoSerialize
from quantem.diffraction.crystal import Crystal


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
    transpose: bool = False,
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
        the image frame. The same angle as a dataset's
        ``q_to_r_rotation_ccw_deg``; see :func:`q_to_r_matrix`.
    ellipse : array-like | None
        [e11, e12] elliptic distortion correction from calibrate_ellipse(),
        applied in the detector frame before the rotation.
    name : str, default="bragg_peaks_calibrated"
        Name of the returned Vector.
    transpose : bool, default=False
        Swap the detector row and column axes before the rotation (the
        dataset's ``q_transpose``).

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
    if rotation_ccw_deg or transpose:
        qrc = qrc @ q_to_r_matrix(rotation_ccw_deg, transpose).T
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
    out.metadata["q_transpose"] = bool(transpose)
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
    bragg_k_power: float = 0.0,
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
    bragg_k_power : float, default=0.0
        Each peak is weighted by ``|q| ** bragg_k_power``. Keep 0 for
        calibration: a power of q makes the outer rings dominate.
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
    bragg_k_power: float = 0.0,
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
    bragg_k_power : float, default=0.0
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
    min_number_peaks: int = 5,
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
    min_number_peaks : int, default=5
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
    bragg_k_power: float = 0.0,
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
    bragg_k_power : float, default=0.0
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
    transpose: bool | None = None,
    plot: bool = False,
    returnfig: bool = False,
    return_transpose: bool = False,
):
    """Detector-to-scan rotation from the curl of the center-of-mass field.

    The center of mass of each diffraction pattern (about the fitted origin)
    forms a vector field over the scan. In the correct common frame that
    field is (approximately) a gradient field, so its curl vanishes; rotating
    the detector axes by the unknown scan rotation and minimizing the summed
    squared curl recovers the angle.

    The curl is invariant under 180-degree rotation, so the sign of the
    measured field cannot distinguish theta from theta + 180. Only the
    candidate in [-90, 90) is returned; the other is 180 degrees away. Pick
    the one consistent with a known feature (e.g. a Burgers orientation
    relationship, or the divergence sign convention of DPC). The returned
    angle is ready to pass to peaks_to_calibrated() as rotation_ccw_deg.

    A detector read out with its rows and columns swapped is a reflection,
    not a rotation, and no angle removes the curl it leaves; both
    possibilities are tested, as ptychography does, and a transpose is
    reported when it lowers the curl by more than 10 %. A warning flags a
    transpose that cannot be decided, and a curl that hardly depends on the
    angle (a ratio below 2 between its largest and smallest values), where
    the center of mass is dominated by diffraction contrast rather than the
    projected potential and the angle cannot be trusted. The angle and the transpose follow the convention of
    :func:`q_to_r_matrix`, the same as ptychography's ``com_rotation_rad`` and
    ``com_transpose``.

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
    transpose : bool | None, default=None
        Whether the detector axes are swapped; None tests both.
    plot : bool, default=False
        Plot the curl and divergence measures against the rotation angle.
    returnfig : bool, default=False
        With `plot`, also return the figure and axes.
    return_transpose : bool, default=False
        Also return the transpose.

    Returns
    -------
    rotation_ccw_deg : float
        Curl-minimizing rotation in [-90, 90), degrees, on a 0.25 degree
        grid; the physical answer is either this angle or 180 degrees away.
    transpose : bool
        Only with `return_transpose`.
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

    theta = np.deg2rad(np.arange(-90, 90, 0.25))
    ct, st = np.cos(theta)[:, None, None], np.sin(theta)[:, None, None]

    def measures(d_rr, d_rc, d_cr, d_cc):
        # rotated field: (r', c') = (ct * r - st * c, st * r + ct * c)
        curl = (st * d_rr + ct * d_cr) - (ct * d_rc - st * d_cc)
        div = (ct * d_rr - st * d_cr) + (st * d_rc + ct * d_cc)
        return (curl**2).mean(axis=(1, 2)), (div**2).mean(axis=(1, 2))

    # the transposed field is (com_c, com_r): swap the components
    options = {False: (d_rr, d_rc, d_cr, d_cc), True: (d_cr, d_cc, d_rr, d_rc)}
    tested = [transpose] if transpose is not None else [False, True]
    results = {t: measures(*options[t]) for t in tested}
    if len(tested) == 1:
        transpose = tested[0]
    else:
        # a transpose is a strong claim: it must lower the curl clearly
        c_false, c_true = results[False][0].min(), results[True][0].min()
        transpose = bool(c_true < 0.9 * c_false)
        if 0.9 * c_false <= c_true <= c_false / 0.9:
            warnings.warn(
                "the transposed and untransposed detector leave nearly the same curl "
                f"({c_true / c_false:.2f}): the transpose is undetermined, kept "
                f"{transpose}. Pass transpose= if it is known.",
                stacklevel=2,
            )
    curl_sq, div_sq = results[transpose]
    contrast = float(curl_sq.max() / max(curl_sq.min(), 1e-300))
    if contrast < 2.0:
        warnings.warn(
            f"the curl changes by only {contrast:.2f}x over all angles: the center of mass "
            "does not follow the projected potential here (diffraction contrast, a "
            "crystalline specimen), so this rotation is unreliable. Use mask_radius to "
            "keep to the direct beam, or a feature of known orientation.",
            stacklevel=2,
        )

    i_best = int(np.argmin(curl_sq))
    rotation_ccw_deg = float(np.rad2deg(theta[i_best]))

    out: list = [rotation_ccw_deg] + ([bool(transpose)] if return_transpose else [])
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        deg = np.rad2deg(theta)
        ax.plot(deg, curl_sq, "k-", label="mean squared curl")
        other = [t for t in results if t != transpose]
        if other:
            ax.plot(
                deg,
                results[other[0]][0],
                "k:",
                label=f"mean squared curl, transpose={other[0]}",
            )
        ax.plot(deg, div_sq, "-", color="0.6", label="mean squared divergence")
        ax.axvline(rotation_ccw_deg, color="r", ls="--")
        ax.set_xlabel("rotation (degrees)")
        ax.set_ylabel("field measure")
        ax.set_title(
            "scan rotation = %.1f deg (or %.1f)%s"
            % (
                rotation_ccw_deg,
                rotation_ccw_deg + (180 if rotation_ccw_deg < 0 else -180),
                ", transposed" if transpose else "",
            )
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
    bragg_k_power: float = 0.0,
    bragg_intensity_power: float = 0.5,
    max_ellipticity: float = 0.03,
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
    bragg_k_power : float, default=0.0
        Each peak is weighted by ``|q| ** bragg_k_power``.
    bragg_intensity_power : float, default=0.5
        Each peak is weighted by ``intensity ** bragg_intensity_power``. Full
        intensity weighting (1) lets a few strong peaks pull the fit.
    max_ellipticity : float, default=0.03
        Half range of the coarse grid over (e11, e12) that seeds the local
        optimization, so it cannot stop in a local maximum of the sharpness.
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

    # coarse grid first: the sharpness has local maxima between ring families
    grid = np.linspace(-max_ellipticity, max_ellipticity, 9)
    costs = [(cost((a, b)), a, b) for a in grid for b in grid]
    x0 = np.array(min(costs)[1:])
    res = minimize(
        cost,
        x0=x0,
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


def _warn_large_calibration_adjustment(scale: float, ellipse) -> None:
    """Warn when a calibration fit needs unusually large corrections."""
    scale_change_pct = 100 * (float(scale) - 1.0)
    ellipticity_pct = 0.0 if ellipse is None else 200 * float(np.hypot(*ellipse))
    adjustments = []
    if ellipticity_pct > 3.0:
        adjustments.append(f"ellipticity is {ellipticity_pct:.2f}% (limit 3%)")
    if abs(scale_change_pct) > 10.0:
        adjustments.append(f"pixel size changed by {scale_change_pct:+.1f}% (limit +/-10%)")
    if adjustments:
        warnings.warn(
            "Large calibration adjustment: "
            + "; ".join(adjustments)
            + ". Check the reference crystal, peak detection, and starting pixel size.",
            UserWarning,
            stacklevel=3,
        )


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
        :func:`measure_scan_rotation` and :func:`q_to_r_matrix`.
    metadata : dict | None
        Evidence and provenance, e.g. the reference phases, the matched
        rings and their residual, and the detector binning.
    transpose : bool, default=False
        Detector row and column axes swapped before the rotation.
    """

    # calibrations saved before the transpose existed load without one
    transpose = False

    def __init__(
        self,
        pixel_size: float,
        ellipse=None,
        rotation_ccw_deg: float = 0.0,
        metadata: dict | None = None,
        transpose: bool = False,
    ):
        self.pixel_size = float(pixel_size)
        self.ellipse = None if ellipse is None else np.asarray(ellipse, dtype=float)
        self.rotation_ccw_deg = float(rotation_ccw_deg)
        self.metadata: dict = dict(metadata or {})
        self.transpose = bool(transpose)

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
            transpose=self.transpose,
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
            self.pixel_size * factor, self.ellipse, self.rotation_ccw_deg, md, self.transpose
        )

    def __repr__(self) -> str:
        e = "none" if self.ellipse is None else "e11 %+.5f e12 %+.5f" % tuple(self.ellipse)
        rms = self.metadata.get("residual_rms")
        if rms is None:
            q = ""
        elif self.metadata.get("method") == "2d":
            q = ", %s, %d spots, rms %.4f 1/A" % (
                self.metadata.get("phase", ""),
                self.metadata.get("n_pairs", 0),
                rms,
            )
        else:
            q = ", %d rings, rms %.2f%%" % (self.metadata.get("n_rings", 0), 100 * rms)
        return (
            f"DiffractionCalibration(pixel_size={self.pixel_size:.6f} 1/A/px, "
            f"ellipse: {e}, rotation {self.rotation_ccw_deg:g} deg"
            f"{', transposed' if self.transpose else ''}{q})"
        )


def _allowed_rings(crystals, k_max: float, min_ring_intensity: float) -> np.ndarray:
    """Radii (1/Angstroms) of the rings a crystal can show, every one counted alike.

    A ring is kept when its strongest single reflection reaches
    `min_ring_intensity` of the crystal's strongest reflection. Only single
    reflections count -- no multiplicity, no ``|g|`` weighting -- because a
    textured or single-crystal specimen shows any allowed reflection at full
    strength or not at all, whatever a powder would give it.
    """
    out = []
    for xtl in crystals:
        g = xtl.g_len.numpy()
        f = xtl.struct_factors_int.numpy()
        nz = g > 1e-6
        if not nz.any():
            continue
        keep = nz & (g < k_max) & (f >= min_ring_intensity * f[nz].max())
        out.append(np.round(g[keep], 4))
    return np.unique(np.concatenate(out)) if out else np.zeros(0)


def _ring_distance(k: np.ndarray, rings: np.ndarray) -> np.ndarray:
    """Distance from each radius to the nearest ring."""
    j = np.clip(np.searchsorted(rings, k), 1, len(rings) - 1)
    return np.minimum(np.abs(k - rings[j - 1]), np.abs(k - rings[j]))


def _ring_match_scores(
    k: np.ndarray, w: np.ndarray, rings: np.ndarray, scales: np.ndarray, sigma: float
) -> np.ndarray:
    """Fraction of the peak weight on an allowed ring, for each trial scale.

    Positions only: every allowed ring has the same height, and a ring the
    data does not show costs nothing, so texture, a single zone axis or
    dynamical intensities do not pull the scale. The peak set and the
    denominator are fixed before scaling, so a scale cannot win by pushing
    peaks out of a window.
    """
    total = float(w.sum())
    if total <= 0 or len(rings) < 2:
        return np.zeros(len(scales))
    return np.array(
        [
            float((w * np.exp(-0.5 * (_ring_distance(k * s, rings) / sigma) ** 2)).sum()) / total
            for s in scales
        ]
    )


def _radial_maxima(k: np.ndarray, w: np.ndarray, sigma: float, min_frac: float = 0.01):
    """Maxima of the smoothed radial histogram: (radius, height) of each
    measured ring, sub-bin by a parabola. A tail, a diffuse band or a rising
    background contributes no maxima where it has no ring."""
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d

    step = 0.001
    if k.size == 0:
        return np.zeros(0), np.zeros(0)
    edges = np.arange(max(k.min(), 1e-3) - 5 * step, k.max() + 5 * step, step)
    h, _ = np.histogram(k, edges, weights=w)
    h = gaussian_filter1d(h.astype(np.float64), max(sigma / 3, step) / step)
    centers = 0.5 * (edges[1:] + edges[:-1])
    span = max(int(round(1.5 * sigma / step)), 3)
    loc = np.nonzero((h == maximum_filter1d(h, 2 * span + 1)) & (h > min_frac * h.max()))[0]
    loc = loc[(loc > 0) & (loc < len(h) - 1)]
    c0, c1, c2 = h[loc - 1], h[loc], h[loc + 1]
    den = c0 - 2 * c1 + c2
    shift = np.where(den < 0, 0.5 * (c0 - c2) / np.where(den < 0, den, -1.0), 0.0)
    return centers[loc] + shift * step, c1


def _ring_table(k: np.ndarray, w: np.ndarray, rings: np.ndarray, sigma: float, min_frac=0.01):
    """Measured rings matched to allowed rings: (k measured, k reference, ratio, weight).

    The measured ring is the maximum of the smoothed radial histogram, not a
    mean, so a tail on one side of a ring (overlapping disks near the beam,
    a second phase) does not move it. Maxima below `min_frac` of the
    strongest, or further than ``2 sigma`` from any allowed ring, are left
    out; of several maxima on one ring the strongest is kept.
    """
    best: dict[int, tuple] = {}
    for km, hm in zip(*_radial_maxima(k, w, sigma, min_frac)):
        j = int(np.argmin(np.abs(rings - km)))
        if abs(rings[j] - km) > 2 * sigma:
            continue
        if j not in best or hm > best[j][3]:
            best[j] = (float(km), float(rings[j]), float(rings[j] / km), float(hm))
    return [best[j] for j in sorted(best)]


def select_positions(peaks, positions):
    """The peaks of some scan positions only, on the same scan grid.

    Parameters
    ----------
    peaks : Vector
        Ragged peaks over the scan.
    positions : list of (row, col) | np.ndarray | None
        A list, tuple or array of (row, col), or an (R, C) boolean mask.
        None keeps every position.

    Returns
    -------
    peaks : Vector
        A copy holding only the peaks at the selected positions; the other
        positions are empty.
    mask : np.ndarray
        (R, C) boolean mask of the selected positions.
    """
    from quantem.core.utils.clustering import filter_rows
    from quantem.diffraction.orientation import position_mask

    shape = (int(peaks.shape[0]), int(peaks.shape[1]))
    mask = position_mask(positions, shape).numpy()
    if positions is None:
        return peaks, mask
    counts = np.asarray(peaks.row_counts(), dtype=int)
    keep = np.repeat(mask.reshape(-1), counts)
    if not keep.any():
        raise ValueError("no peaks at the selected positions")
    return filter_rows(peaks, keep), mask


def calibrate(
    peaks_px,
    crystal,
    pixel_size_guess: float,
    rotation_ccw_deg: float = 0.0,
    transpose: bool = False,
    fit_ellipse: bool = True,
    n_iter: int = 3,
    scale_search: tuple[float, float] = (0.9, 1.1),
    k_min: float = 0.05,
    k_max: float = 1.3,
    k_broadening: float = 0.01,
    power_intensity: float = 0.5,
    min_ring_intensity: float = 0.01,
    residual_tol: float = 0.01,
    plot: bool = False,
    figsize: tuple[float, float] = (13.0, 6.4),
    marker_size: float = 8.0,
    zone_axis=None,
    returnfig: bool = False,
    shell_step: float = 0.005,
    positions=None,
):
    """Measure the reciprocal pixel size and the elliptic distortion.

    Matches the radii of the detected peaks to the rings a reference crystal
    can show, using ring positions only. Every allowed ring counts alike and a
    ring missing from the data costs nothing, so the result does not depend
    on texture, a single zone axis or dynamical intensities; the data is
    never assumed to look like a powder pattern.

    Three stages. The scale is first scanned over `scale_search` around the
    guess for the largest fraction of the measured rings (the maxima of the
    radial histogram, by height) that land on an allowed ring.
    The ellipse and the scale are then refined in turn, the ellipse from the
    sharpness of the log-radius histogram, where it does not depend on the
    scale, and the scale as the median over rings of reference radius /
    measured radius, each measured ring taken at its histogram maximum.
    Finally every measured ring is checked separately, which is what tells a
    correct calibration from a plausible one: a single pixel size that
    explains the pattern gives per-ring ratios agreeing to a few tenths of a
    percent with no trend in k.

    Ring radii of one lattice repeat at fixed ratios (a bcc pattern scaled by
    sqrt(2) still lands most peaks on rings), so a position-only match is only
    unique over a limited range of scales. The default search, +-10 %, is
    free of such aliases for the low-order rings; widen it only for a guess
    that is far out, and check the result. A second, competing scale within
    the search is reported as a warning.

    Parameters
    ----------
    peaks_px : Vector
        Origin-corrected peaks in detector pixels, (q_row, q_col, intensity).
    crystal : Crystal | list[Crystal]
        Reference phase or phases, structure factors calculated.
    pixel_size_guess : float
        Starting reciprocal pixel size (1/Angstroms per pixel).
    rotation_ccw_deg : float, default=0.0
        Diffraction-to-scan rotation recorded on the calibration (the
        dataset's ``q_to_r_rotation_ccw_deg``, see :func:`q_to_r_matrix`).
    transpose : bool, default=False
        Detector axes swapped before the rotation, recorded on the
        calibration (the dataset's ``q_transpose``).
    fit_ellipse : bool, default=True
        Fit the elliptic distortion as well as the scale.
    n_iter : int, default=3
        Ellipse and scale refinement rounds. Each round fits the ellipse
        left over after the current correction and composes the two.
    scale_search : tuple, default=(0.9, 1.1)
        Range of the scale scan, as a multiple of the guess.
    k_min, k_max : float, default=0.05, 1.3
        Range of scattering vectors fit, 1/Angstroms. The ellipse fit uses
        at least 0.15 as its lower limit.
    k_broadening : float, default=0.01
        Width of a ring, 1/Angstroms: the Gaussian tolerance of the scale
        scan, and how far a measured ring may sit from its reference ring.
    power_intensity : float, default=0.5
        Peaks are weighted by ``intensity ** power_intensity``; 0 counts
        every peak alike.
    min_ring_intensity : float, default=0.01
        Rings whose strongest reflection is weaker than this fraction of the
        strongest reflection are left out (single reflections, without
        multiplicity).
    residual_tol : float, default=0.01
        Per-ring residual rms above which the fit is reported as unreliable.
    plot : bool, default=False
        Show the ring comparison before and after the fit, for the first
        reference crystal.
    figsize : tuple, default=(13, 6.4)
        Figure size.
    marker_size : float, default=8.0
        Area of the brightest peak in the azimuth panels, where every peak is
        drawn with area proportional to its intensity.
    zone_axis : sequence of int, optional
        Fit only the rings of this zone, [uvw] or [UVTW] (see
        :func:`zone_reflections`).
    shell_step : float, default=0.005
        Maximum range in |g|, in 1/Angstroms, grouped into one ring in the
        diagnostic plots. This affects display grouping only; calibration
        fitting uses the individual reflections.
    positions : list of (row, col) | np.ndarray | None
        Scan positions to calibrate from: a list, tuple or array of
        (row, col), or an (R, C) boolean mask. None (default) uses every
        position. Calibrate on a region whose crystallography is known, then
        apply the result to the whole scan with
        :meth:`DiffractionCalibration.apply`.
    returnfig : bool, default=False
        With `plot`, also return the figure and axes.

    Returns
    -------
    DiffractionCalibration
        The pixel size, ellipse and rotation. ``metadata`` holds the matched
        rings as (measured k, reference k, ratio), their residual rms,
        whether the fit passed `residual_tol`, and the score of the best
        competing scale relative to the chosen one (`alias_ratio`). With
        `plot` and `returnfig`, ``(cal, fig, axs)`` instead, axs of shape
        (2, 2).

    Warns
    -----
    UserWarning
        If fewer than three rings match, their residual rms exceeds
        `residual_tol`, the best scale sits at the edge of `scale_search`, a
        competing scale scores nearly as well, the peaks cover too few
        azimuths to fix the ellipse (one dominant orientation: use
        :func:`calibrate_2d`), or the fitted ellipticity exceeds 3% or the
        fitted pixel size differs from `pixel_size_guess` by more than 10%.

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
    peaks_px, pos_mask = select_positions(peaks_px, positions)
    rings = _allowed_rings(crystals, k_max * scale_search[1] * 1.2, min_ring_intensity)
    if len(rings) < 2:
        raise ValueError("fewer than two allowed rings below k_max")

    flat = peaks_px.select_fields("q_row", "q_col", "intensity").numpy().astype(np.float64)
    q_px = flat[:, :2]
    weight = np.clip(flat[:, 2], 0.0, None) ** power_intensity

    def radii(ellipse):
        q = q_px if ellipse is None else q_px @ _ellipse_matrix(ellipse).T
        return np.hypot(q[:, 0], q[:, 1]) * pixel_size_guess

    # the peak set is fixed once, at the guess
    k0 = radii(None)
    sel = (k0 > k_min) & (k0 < k_max)
    w = weight[sel]

    # coarse: 0.05 % steps over the search range
    scales = np.arange(scale_search[0], scale_search[1] + 2.5e-4, 5e-4)
    # the measured rings are the maxima of the radial histogram, weighted by
    # their height; every peak scored individually would let a diffuse band
    # or a rising background reward the scales that crowd it into dense rings
    k_meas, h_meas = _radial_maxima(k0[sel], w, k_broadening)
    scores = _ring_match_scores(k_meas, h_meas, rings, scales, k_broadening)
    i_best = int(np.argmax(scores))
    scale = float(scales[i_best])
    far = np.abs(scales / scale - 1) > 0.03
    alias_ratio = float(scores[far].max() / scores[i_best]) if far.any() else 0.0
    at_edge = i_best in (0, len(scales) - 1)
    # azimuthal coverage (effective fraction of 5 degree bins over 180): one
    # dominant orientation leaves the ellipse undetermined by ring sharpness
    phi = np.mod(np.degrees(np.arctan2(q_px[sel, 1], q_px[sel, 0])), 180.0)
    h_phi, _ = np.histogram(phi, 36, (0.0, 180.0), weights=w)
    p_phi = h_phi[h_phi > 0] / h_phi.sum() if h_phi.sum() > 0 else np.ones(1)
    coverage = float(np.exp(-(p_phi * np.log(p_phi)).sum()) / 36)

    ellipse = None
    table: list = []
    for _ in range(max(1, n_iter)):
        if fit_ellipse:
            pk = peaks_to_calibrated(peaks_px, pixel_size_guess * scale, ellipse=ellipse)
            residual = calibrate_ellipse(
                pk,
                k_min=max(k_min, 0.15),
                k_max=k_max,
                bragg_k_power=0.0,
                bragg_intensity_power=power_intensity,
            )
            ellipse = _compose_ellipse(residual, ellipse)
        k = radii(ellipse)[sel]
        table = _ring_table(k * scale, w, rings, k_broadening)
        if table:
            # measured radii are at the current scale: refine by the median ratio
            scale *= float(np.median([t[2] for t in table]))

    pixel_size = pixel_size_guess * scale
    _warn_large_calibration_adjustment(scale, ellipse)
    k = radii(ellipse)[sel] * scale
    table = [t[:3] for t in _ring_table(k, w, rings, k_broadening)]
    per_ring = np.array([t[2] for t in table]) if table else np.array([np.nan])
    residual_rms = float(np.std(per_ring / np.median(per_ring))) if table else float("nan")
    reliable = len(table) >= 3 and residual_rms <= residual_tol and not at_edge
    peaks_0 = peaks_to_calibrated(peaks_px, pixel_size_guess)
    peaks = peaks_to_calibrated(
        peaks_px,
        pixel_size,
        rotation_ccw_deg=rotation_ccw_deg,
        ellipse=ellipse,
        transpose=transpose,
    )

    cal = DiffractionCalibration(
        pixel_size,
        ellipse,
        rotation_ccw_deg,
        transpose=transpose,
        metadata=dict(
            reference=[x.name for x in crystals],
            pixel_size_guess=float(pixel_size_guess),
            scale=float(scale),
            n_rings=len(table),
            residual_rms=residual_rms,
            rings=table,
            alias_ratio=alias_ratio,
            azimuthal_coverage=coverage,
            n_positions=int(pos_mask.sum()),
            reliable=bool(reliable),
            binning=1,
        ),
    )
    if at_edge:
        warnings.warn(
            f"the best scale ({scale:.3f} x the guess) is at the edge of scale_search "
            f"{tuple(scale_search)}: the guess is too far out, or the reference phase is wrong.",
            stacklevel=2,
        )
    if fit_ellipse and coverage < 0.5:
        warnings.warn(
            f"the peaks cover few azimuths (coverage {coverage:.2f}): one orientation "
            "dominates, so ring sharpness cannot fix the ellipse. Use calibrate_2d.",
            stacklevel=2,
        )
    if alias_ratio > 0.9:
        warnings.warn(
            f"a scale more than 3 % away scores {100 * alias_ratio:.0f} % as well as the "
            "chosen one: check the ring assignment in the plot, or narrow scale_search.",
            stacklevel=2,
        )
    if not reliable and not at_edge:
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
            power_intensity=power_intensity,
            marker_size=marker_size,
            shell_step=shell_step,
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


def _histogram_spots(
    q_px: np.ndarray,
    weight: np.ndarray,
    r_max: float,
    bin_px: float = 0.25,
    sigma_px: float = 0.5,
    min_spacing_px: float = 2.0,
    min_rel: float = 0.02,
    max_spots: int = 80,
):
    """Spots of the summed 2D Bragg-vector histogram, in detector pixels.

    Returns (positions (N, 2), weights (N,)): each spot is the weighted mean
    of the raw peaks within ``min_spacing_px / 2`` of a histogram maximum,
    strongest first. The direct beam is included.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter

    edges = np.arange(-r_max, r_max + bin_px, bin_px)
    H, _, _ = np.histogram2d(q_px[:, 0], q_px[:, 1], [edges, edges], weights=weight)
    H = gaussian_filter(H, sigma_px / bin_px)
    size = max(int(round(min_spacing_px / bin_px)), 3) | 1
    loc = np.argwhere((H == maximum_filter(H, size)) & (H > 0))
    c = 0.5 * (edges[1:] + edges[:-1])
    pos = np.column_stack([c[loc[:, 0]], c[loc[:, 1]]])
    h = H[loc[:, 0], loc[:, 1]]
    off = np.hypot(pos[:, 0], pos[:, 1]) > min_spacing_px
    ref = h[off].max() if off.any() else h.max()
    keep = h >= min_rel * ref
    pos, h = pos[keep], h[keep]
    order = np.argsort(-h)[:max_spots]
    pos, h = pos[order], h[order]
    # refine each on the raw peaks around it
    from scipy.spatial import cKDTree

    tree = cKDTree(q_px)
    out = []
    for p0 in pos:
        idx = tree.query_ball_point(p0, 0.5 * min_spacing_px)
        if idx:
            ww = weight[idx]
            out.append((ww[:, None] * q_px[idx]).sum(0) / max(ww.sum(), 1e-30))
        else:
            out.append(p0)
    return np.asarray(out), h


def _fit_affine(p: np.ndarray, g: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares 2x2 M with M p ~ g."""
    sw = np.sqrt(w)[:, None]
    X, *_ = np.linalg.lstsq(p * sw, g * sw, rcond=None)
    return X.T


def _pair_spots(qm: np.ndarray, gs: np.ndarray, tol: float):
    """Nearest simulated spot to each measured spot within `tol`, each
    simulated spot used once (by its nearest measured spot)."""
    d = np.linalg.norm(qm[:, None, :] - gs[None, :, :], axis=-1)
    j = np.argmin(d, axis=1)
    ok = d[np.arange(len(qm)), j] < tol
    best: dict[int, int] = {}
    for i in np.nonzero(ok)[0]:
        if j[i] not in best or d[i, j[i]] < d[best[j[i]], j[i]]:
            best[int(j[i])] = int(i)
    mi = np.array(sorted(best.values()), dtype=int)
    return mi, j[mi]


def calibrate_2d(
    peaks_px,
    crystal,
    pixel_size_guess: float,
    energy_ev: float,
    rotation_ccw_deg: float = 0.0,
    transpose: bool = False,
    precession_deg: float = 0.0,
    power_intensity: float = 0.5,
    k_max: float = 1.3,
    pair_tol: tuple[float, ...] = (0.04, 0.02, 0.012),
    min_spot_rel: float = 0.02,
    plan_params: dict | None = None,
    min_pairs: int = 6,
    residual_tol: float = 0.005,
    plot: bool = False,
    figsize: tuple[float, float] = (13.0, 12.0),
    returnfig: bool = False,
    positions=None,
):
    """Pixel size and elliptic distortion from the dominant orientation, in 2D.

    For a specimen whose patterns share one dominant orientation -- a single
    crystal, a matrix grain, a strong fiber texture seen along its axis --
    the sum of every detected peak over the scan is a sharp spot pattern.
    This indexes that pattern once and fits the detector distortion to it
    directly in 2D, with no radial (polar) histogram:

    1. The peaks of every position are summed into a 2D histogram in
       detector pixels (weight ``intensity ** power_intensity``) and its
       spots found.
    2. The spot pattern, at the guessed pixel size, is matched against each
       candidate crystal with :class:`OrientationMap`; the best correlation
       picks the phase and its orientation.
    3. Measured spots are paired with the simulated reflections of that
       orientation, and a general 2x2 map ``q = M p`` from pixels to
       1/Angstroms is fit by least squares, re-pairing at each tolerance of
       `pair_tol`. Its symmetric part ``P = sqrt(M^T M)`` is the calibration:
       pixel size ``trace(P) / 2`` and ellipse ``P / pixel_size``. Its
       rotation belongs to the crystal orientation, not to the detector.

    An anisotropic lattice error of the reference (a strained or
    lower-symmetry cell) is indistinguishable from detector ellipticity in a
    single zone, so the reference should be a well-known structure.

    Parameters
    ----------
    peaks_px : Vector
        Origin-corrected peaks in detector pixels, (q_row, q_col, intensity).
    crystal : Crystal | list[Crystal]
        Candidate phase(s), structure factors calculated.
    pixel_size_guess : float
        Starting reciprocal pixel size (1/Angstroms per pixel), within a few
        percent; :func:`calibrate` gives one from the rings.
    energy_ev : float
        Beam energy in eV.
    rotation_ccw_deg : float, default=0.0
        Diffraction-to-scan rotation recorded on the calibration (the
        dataset's ``q_to_r_rotation_ccw_deg``, see :func:`q_to_r_matrix`).
    transpose : bool, default=False
        Detector axes swapped before the rotation, recorded on the
        calibration (the dataset's ``q_transpose``).
    precession_deg : float, default=0.0
        Precession semi-angle, for the simulated pattern.
    power_intensity : float, default=0.5
        Peaks are weighted by ``intensity ** power_intensity`` in the
        histogram.
    k_max : float, default=1.3
        Spots beyond this (1/Angstroms) are not fit.
    pair_tol : tuple of float, default=(0.04, 0.02, 0.012)
        Pairing tolerances (1/Angstroms) of the successive fits.
    min_spot_rel : float, default=0.02
        Histogram spots weaker than this fraction of the strongest (direct
        beam excluded) are not used.
    plan_params : dict, optional
        Overrides for :meth:`OrientationMap.build_plan` of the indexing step.
    min_pairs : int, default=6
        Fewer paired spots than this is reported as unreliable.
    residual_tol : float, default=0.005
        Paired-spot rms residual (1/Angstroms) above which the fit is
        reported as unreliable.
    plot : bool, default=False
        Show the ring comparison (top) and the 2D histogram of the peaks
        under the simulated spots (bottom), before and after.
    figsize : tuple, default=(13, 12)
        Figure size.
    returnfig : bool, default=False
        With `plot`, also return the figure and axes.
    positions : list of (row, col) | np.ndarray | None
        Scan positions to calibrate from: a list, tuple or array of
        (row, col), or an (R, C) boolean mask. None (default) uses every
        position. Calibrate on a region whose crystallography is known, then
        apply the result to the whole scan with
        :meth:`DiffractionCalibration.apply`.

    Returns
    -------
    DiffractionCalibration
        ``metadata`` records the method, the phase, the orientation
        quaternion (detector frame, at the fitted calibration), the paired
        spots and their rms residual. With `plot` and `returnfig`,
        ``(cal, fig, axs)``, axs of shape (2, 2).
    """
    import torch

    from quantem.diffraction.orientation import OrientationMap
    from quantem.diffraction.rotations import qmult, quat_from_matrix

    crystals = [crystal] if isinstance(crystal, Crystal) else list(crystal)
    for xtl in crystals:
        if xtl.g_vec is None:
            raise RuntimeError(f"{xtl.name}: run calculate_structure_factors() first")
    peaks_px, pos_mask = select_positions(peaks_px, positions)

    flat = peaks_px.select_fields("q_row", "q_col", "intensity").numpy().astype(np.float64)
    q_px = flat[:, :2]
    weight = np.clip(flat[:, 2], 0.0, None) ** power_intensity
    r_max = min(float(np.hypot(q_px[:, 0], q_px[:, 1]).max()), k_max / pixel_size_guess)
    inside = np.hypot(q_px[:, 0], q_px[:, 1]) < r_max
    spots, spot_w = _histogram_spots(q_px[inside], weight[inside], r_max, min_rel=min_spot_rel)

    # index the summed pattern once per candidate phase
    pattern = np.column_stack([spots * pixel_size_guess, spot_w / spot_w.max()])
    vec = Vector.from_data(
        [[pattern]],
        fields=["qx", "qy", "intensity"],
        units=["A^-1", "A^-1", "counts"],
        name="summed pattern",
    )
    plan = dict(
        angle_step_zone_axis_deg=1.0,
        angle_step_in_plane_deg=2.0,
        detector_q_max=None,
        verbose=False,
        progress_bar=False,
    )
    plan.update(plan_params or {})
    best = None
    for xtl in crystals:
        om = OrientationMap.from_vectors(
            vec, xtl, energy_ev=energy_ev, precession_deg=precession_deg
        )
        om.build_plan(**plan)
        om.match_orientations(num_matches=1, min_number_peaks=3, progress_bar=False)
        om.refine_orientations(progress_bar=False)
        corr = float(om.corr[0, 0, 0])
        if best is None or corr > best[0]:
            best = (corr, xtl, om)
    corr, xtl, om = best
    quat = om.quats[0, 0, 0].clone()
    sim = om.generate_pattern(0, 0, k_max=k_max)
    g = np.column_stack([sim["qx"].numpy(), sim["qy"].numpy()]).astype(np.float64)
    gi = sim["intensity"].numpy().astype(np.float64)
    g_keep = (np.hypot(g[:, 0], g[:, 1]) > 1e-6) & (gi > 1e-3 * gi.max())
    g, gi = g[g_keep], gi[g_keep]

    meas = np.hypot(spots[:, 0], spots[:, 1]) * pixel_size_guess > 1e-3  # no direct beam
    p, w = spots[meas], spot_w[meas]
    M = pixel_size_guess * np.eye(2)
    mi = gj = np.zeros(0, dtype=int)
    for tol in pair_tol:
        mi, gj = _pair_spots(p @ M.T, g, tol)
        if len(mi) < 3:
            break
        M = _fit_affine(p[mi], g[gj], w[mi])
    if len(mi) < 3:
        raise RuntimeError(
            f"only {len(mi)} spots of the summed pattern pair with {xtl.name}: no dominant "
            "orientation, or the pixel size guess is too far out"
        )
    # M = R P: P symmetric positive is the detector calibration
    evals, evecs = np.linalg.eigh(M.T @ M)
    P = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
    Rm = M @ np.linalg.inv(P)
    pixel_size = float(np.trace(P) / 2)
    A = P / pixel_size
    ellipse = np.array([0.5 * (A[0, 0] - A[1, 1]), 0.5 * (A[0, 1] + A[1, 0])])
    resid = p[mi] @ M.T - g[gj]
    rms = float(np.sqrt((w[mi][:, None] * resid**2).sum() / w[mi].sum() / 2))
    reliable = len(mi) >= min_pairs and rms <= residual_tol and np.linalg.det(Rm) > 0

    # the orientation in the calibrated detector frame: undo the in-plane
    # rotation R that M carried (lab z is the beam)
    R3 = np.eye(3)
    R3[:2, :2] = Rm.T
    q_cal = qmult(quat_from_matrix(torch.as_tensor(R3, dtype=quat.dtype)), quat)

    cal = DiffractionCalibration(
        pixel_size,
        ellipse,
        rotation_ccw_deg,
        transpose=transpose,
        metadata=dict(
            method="2d",
            reference=[x.name for x in crystals],
            phase=xtl.name,
            correlation=corr,
            orientation=[float(v) for v in q_cal],
            pixel_size_guess=float(pixel_size_guess),
            n_spots=int(len(p)),
            n_positions=int(pos_mask.sum()),
            n_pairs=int(len(mi)),
            residual_rms=rms,
            in_plane_rotation_deg=float(np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))),
            pairs=[
                (float(a), float(b), float(c), float(d)) for (a, b), (c, d) in zip(p[mi], g[gj])
            ],
            reliable=bool(reliable),
            binning=1,
        ),
    )
    if not reliable:
        warnings.warn(
            f"2D calibration looks unreliable: {len(mi)} spots paired with {xtl.name}, "
            f"rms {rms:.4f} 1/A (tolerance {residual_tol}). Check that one orientation "
            "dominates the scan and that the guess is within a few percent.",
            stacklevel=2,
        )
    if not plot:
        return cal

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(
        2, 2, figsize=figsize, gridspec_kw={"height_ratios": [1, 2.1]}, squeeze=False
    )
    peaks_0 = peaks_to_calibrated(peaks_px, pixel_size_guess)
    peaks_1 = peaks_to_calibrated(peaks_px, pixel_size, ellipse=ellipse)
    k_lo = 0.1
    for col, (pk, ttl) in enumerate(
        (
            (peaks_0, f"before: {pixel_size_guess:.5f} " + r"$\mathrm{\AA}^{-1}$/px"),
            (peaks_1, f"after: {pixel_size:.5f} " + r"$\mathrm{\AA}^{-1}$/px"),
        )
    ):
        plot_ring_comparison(
            pk,
            [xtl],
            k_min=k_lo,
            k_max=k_max,
            power_intensity=power_intensity,
            figax=(fig, [axs[0, col]]),
        )
        axs[0, col].set_title(ttl, fontsize=10)
    # before: the orientation as indexed at the guess; after: rotated into
    # the calibrated frame, P p = R^T g
    _spot_panel(axs[1, 0], q_px * pixel_size_guess, weight, g, gi, k_max, "before")
    _spot_panel(axs[1, 1], q_px @ P.T, weight, g @ Rm, gi, k_max, "after")
    axs[1, 1].text(
        0.02,
        0.98,
        f"{xtl.name}, {len(mi)} spots paired, rms {1e3 * rms:.1f}e-3 "
        + r"$\mathrm{\AA}^{-1}$"
        + f"\nellipse e11 {ellipse[0]:+.4f}  e12 {ellipse[1]:+.4f}"
        + ("" if reliable else "\nUNRELIABLE"),
        transform=axs[1, 1].transAxes,
        va="top",
        fontsize=9,
        color="k" if reliable else "tab:red",
        bbox=dict(facecolor="white", alpha=0.8, lw=0),
    )
    fig.tight_layout()
    return (cal, fig, axs) if returnfig else cal


def _spot_panel(ax, q, w, g, gi, k_max, title, bin_size=None):
    """Red 2D histogram of every peak, black simulated spots over it."""
    from scipy.ndimage import gaussian_filter

    bin_size = bin_size or k_max / 250
    edges = np.arange(-k_max, k_max + bin_size, bin_size)
    H, _, _ = np.histogram2d(q[:, 0], q[:, 1], [edges, edges], weights=w)
    H = gaussian_filter(H, 1.5)
    # scaled to the Bragg spots, not the direct beam
    c = 0.5 * (edges[1:] + edges[:-1])
    off = np.hypot(c[:, None], c[None, :]) > 0.1
    top = H[off].max() if (H[off] > 0).any() else max(H.max(), 1e-30)
    ax.imshow(
        np.clip(H / top, 0, 1) ** 0.4,
        cmap="Reds",
        origin="upper",
        extent=[-k_max, k_max, k_max, -k_max],
        interpolation="bilinear",
    )
    s = 3 + 12 * np.sqrt(gi / gi.max())
    ax.scatter(g[:, 1], g[:, 0], s=s, c="k", lw=0)
    ax.set_xlim(-k_max, k_max)
    ax.set_ylim(k_max, -k_max)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$q_\mathrm{col}$ (1/$\mathrm{\AA}$)")
    ax.set_ylabel(r"$q_\mathrm{row}$ (1/$\mathrm{\AA}$)")
    ax.set_title(title, fontsize=10)


def _shells(
    crystal: Crystal,
    bragg_k_power: float,
    shell_step: float = 0.005,
):
    """Group a crystal's reflections into rings.

    Parameters
    ----------
    crystal : Crystal
        Crystal with structure factors calculated.
    bragg_k_power : float
        Weight each reflection by ``|g| ** bragg_k_power`` when summing a ring.
    shell_step : float, default=0.005
        Maximum range in |g|, in 1/Angstroms, grouped into one ring.

    Returns
    -------
    radius : np.ndarray
        Mean |g| of each ring, 1/Angstroms, ascending.
    intensity : np.ndarray
        Summed intensity of each ring, weighted by ``|g| ** bragg_k_power``.
    index : np.ndarray
        Ring index of every reflection.
    """
    if not np.isfinite(shell_step) or shell_step <= 0:
        raise ValueError(f"shell_step must be finite and positive, got {shell_step}")
    g = crystal.g_len.numpy().ravel()
    ints = crystal.struct_factors_int.numpy().ravel() * g**bragg_k_power
    order = np.argsort(g)
    sorted_g = g[order]
    starts = [0]
    for i in range(1, sorted_g.size):
        if sorted_g[i] - sorted_g[starts[-1]] > shell_step:
            starts.append(i)
    starts = np.asarray(starts, dtype=int)
    sorted_index = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, g.size]))
    index = np.empty(g.size, dtype=int)
    index[order] = sorted_index
    count = np.bincount(index)
    radius = np.bincount(index, weights=g) / count
    intensity = np.bincount(index, weights=ints)
    return radius, intensity, index


def _ring_shells(
    crystal: Crystal,
    k_min: float,
    k_max: float,
    bragg_k_power: float,
    shell_step: float = 0.005,
):
    """Ring radii of a crystal in (k_min, k_max) and their summed intensity,
    relative to the strongest ring in that range."""
    radius, intensity, _ = _shells(crystal, bragg_k_power, shell_step=shell_step)
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
    bragg_k_power: float = 0.0,
    power_intensity: float = 0.5,
    marker_size: float = 8.0,
    n_rings: int = 12,
    shell_step: float = 0.005,
) -> None:
    """The radial histogram and, when requested, the azimuth view for one crystal."""
    plot_ring_comparison(
        peaks,
        [crystal],
        k_min=k_min,
        k_max=k_max,
        k_broadening=k_broadening,
        bragg_k_power=bragg_k_power,
        power_intensity=power_intensity,
        n_labels=n_rings,
        shell_step=shell_step,
        figax=(ax_hist.figure, [ax_hist]),
    )
    ax_hist.set_xlabel("")
    if ax_az is None:
        return
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
    radii, rel = _ring_shells(crystal, k_min, k_max, bragg_k_power, shell_step=shell_step)
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
    bragg_k_power: float = 0.0,
    power_intensity: float = 0.5,
    marker_size: float = 8.0,
    n_rings: int = 12,
    zone_axis=None,
    axsize: tuple[float, float] = (6.0, 6.4),
    figax=None,
    shell_step: float = 0.005,
    plot_azimuth: bool = True,
):
    """Calibrated peaks against the rings of every crystal, one column each.

    The calibration check for all candidate phases, however many of them the
    calibration itself was fit to. Each column holds the radial histogram of
    every peak against that crystal's rings and, when ``plot_azimuth=True``,
    every peak as azimuth against scattering vector with the same rings. A
    ring beside the measured peaks in every direction is a lattice parameter
    or pixel size error; a ring that wobbles with azimuth is elliptic
    distortion.

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
    power_intensity : float, default=0.5
        Peaks are weighted by ``intensity ** power_intensity`` in the histograms.
    marker_size : float, default=8.0
        Area of the brightest peak in the azimuth panels.
    n_rings : int, default=12
        Rings drawn in the azimuth panels and labeled in the histograms,
        strongest first; a large cell would otherwise fill both with its
        weak superstructure rings.
    shell_step : float, default=0.005
        Maximum range in |g|, in 1/Angstroms, grouped into one displayed ring.
    zone_axis : sequence of int, optional
        Show only the rings of this zone, [uvw] or [UVTW] (see
        :func:`zone_reflections`).
    plot_azimuth : bool, default=True
        Include the azimuth-versus-scattering-vector panels below the radial
        histograms. False returns only the radial histogram row.
    axsize : tuple, default=(6.0, 6.4)
        Size of one column.
    figax : (fig, axs) | None
        Existing figure and an array of axes with shape
        ``(2 if plot_azimuth else 1, n_crystals)``.

    Returns
    -------
    tuple
        ``(fig, axs)``, where ``axs`` has shape
        ``(2 if plot_azimuth else 1, n_crystals)``.
    """
    import matplotlib.pyplot as plt

    xtls = _restrict(crystals, zone_axis)
    n_rows = 2 if plot_azimuth else 1
    if figax is None:
        fig, axs = plt.subplots(
            n_rows,
            len(xtls),
            figsize=(
                axsize[0] * len(xtls),
                axsize[1] * (1.0 if plot_azimuth else 0.5),
            ),
            sharex=True,
            squeeze=False,
            gridspec_kw={"height_ratios": [1, 1.3] if plot_azimuth else [1]},
        )
    else:
        fig, axs = figax
        axs = np.asarray(axs)
        if not plot_azimuth and axs.ndim == 2 and axs.shape[0] == 2:
            axs = axs[:1]
        axs = axs.reshape(n_rows, len(xtls))
    for col, xtl in enumerate(xtls):
        _calibration_panels(
            axs[0, col],
            axs[1, col] if plot_azimuth else None,
            peaks,
            xtl,
            k_min=k_min,
            k_max=k_max,
            k_broadening=k_broadening,
            bragg_k_power=bragg_k_power,
            power_intensity=power_intensity,
            marker_size=marker_size,
            n_rings=n_rings,
            shell_step=shell_step,
        )
        axs[0, col].set_title(xtl.name, fontsize=10)
        if col:
            axs[0, col].set_ylabel("")
            if plot_azimuth:
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
    bragg_k_power: float = 0.0,
    power_intensity: float = 0.5,
    label_hkl: bool = True,
    label_min_intensity: float = 0.05,
    n_labels: int = 12,
    zone_axis=None,
    figax=None,
    shell_step: float = 0.005,
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
    bragg_k_power : float, default=0.0
        Measured peaks and reference rings are weighted by
        ``|q| ** bragg_k_power``.
    power_intensity : float, default=0.5
        Measured peaks are weighted by ``intensity ** power_intensity``.
    label_hkl : bool, default=True
        Label the strongest rings by their Miller indices.
    label_min_intensity : float, default=0.05
        Label rings whose summed intensity exceeds this fraction of the
        strongest ring.
    n_labels : int, default=12
        Label at most this many rings, strongest first, which keeps a large
        cell with many rings readable.
    shell_step : float, default=0.005
        Maximum range in |g|, in 1/Angstroms, grouped into one displayed ring.
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
    k, hist = radial_histogram(
        peaks,
        k_min=k_min,
        k_max=k_max,
        bragg_k_power=bragg_k_power,
        bragg_intensity_power=power_intensity,
    )

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
        uniq, shell_int, shells = _shells(xtl, bragg_k_power, shell_step=shell_step)
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

    Warns
    -----
    UserWarning
        If the fitted ellipticity exceeds 3% or the pixel size correction
        exceeds 10%.
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
    ellipse = (e11, e12)
    _warn_large_calibration_adjustment(scale, ellipse)
    return {
        "M": M,
        "correction": np.linalg.inv(M),
        "scale": scale,
        "rotation_deg": float(np.rad2deg(theta)),
        "ellipse": ellipse,
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
    shell_step: float = 0.005,
    sort_by: str = "intensity",
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
    shell_step : float, default=0.005
        Maximum range in |g|, in 1/Angstroms, grouped into one displayed ring.
    sort_by : {"intensity", "q_min"}, default="intensity"
        Select which rings to draw: the strongest rings by summed intensity,
        or the rings with the lowest nonzero |g|. Drawn rings are ordered by |g|.
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

    if sort_by not in ("intensity", "q_min"):
        raise ValueError("sort_by must be 'intensity' or 'q_min'")

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
        uniq, shell_int, _ = _shells(xtl, 2.0, shell_step=shell_step)
        keep = (uniq > 0) & (uniq < q_max)
        uniq, shell_int = uniq[keep], shell_int[keep]
        if sort_by == "intensity":
            selected = np.argsort(shell_int)[::-1][:n_rings]
        else:
            selected = np.argsort(uniq)[:n_rings]
        for k, u in enumerate(np.sort(uniq[selected])):
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
