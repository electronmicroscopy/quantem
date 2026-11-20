"""Radial peak families, orientation histograms and flowline maps.

Works on calibrated polar peaks, a :class:`~quantem.core.datastructures.Vector`
with fields ``(q, theta, intensity)`` such as
:meth:`~quantem.diffraction.polymer.BraggPeaksPolymer.polar_peaks` returns, so it
applies to any material with anisotropic scattering, not only polymers. The
pipeline is::

    centers, windows, _ = estimate_peak_windows(peaks)
    hist = make_orientation_histogram(peaks, windows)
    flow = make_flowline_map(hist)
    image = make_flowline_rainbow_image(flow)

Orientations are pi-periodic. The histogram axes are (radial window, scan row,
scan col, theta), with theta the scattering direction measured from the scan
column axis towards +row; flowlines run perpendicular to it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from scipy.signal import find_peaks, peak_widths
from tqdm.auto import tqdm

from quantem.core.datastructures.vector import Vector
from quantem.core.visualization import show_2d

__all__ = [
    "estimate_peak_windows",
    "radial_profile",
    "plot_radial_profile",
    "peak_maps",
    "make_orientation_histogram",
    "make_flowline_map",
    "make_flowline_rainbow_image",
    "make_flowline_rainbow_legend",
    "make_flowline_combined_image",
]


def _flat(peaks: Vector) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(q, theta, intensity)`` columns and the scan (row, col) of every peak."""
    data = peaks.select_fields("q", "theta", "intensity").numpy().astype(np.float64)
    cell = np.repeat(np.arange(int(np.prod(peaks.shape))), peaks.row_counts())
    return data, cell // peaks.shape[1], cell % peaks.shape[1]


def radial_profile(peaks: Vector, num_bins=200, q_range=None, mode="intensity"):
    """Peak intensity (or count) histogram over q.

    Returns
    -------
    q_centers, profile : np.ndarray
    """
    if mode not in ("intensity", "count"):
        raise ValueError(f"mode must be 'intensity' or 'count', got {mode!r}")
    data = _flat(peaks)[0]
    q_min, q_max = (0.0, data[:, 0].max()) if q_range is None else q_range
    edges = np.linspace(q_min, q_max, num_bins + 1)
    weights = data[:, 2] if mode == "intensity" else None
    profile, _ = np.histogram(data[:, 0], bins=edges, weights=weights)
    return 0.5 * (edges[:-1] + edges[1:]), profile


def estimate_peak_windows(
    peaks: Vector,
    num_bins=200,
    q_range=None,
    n_peaks=5,
    height_percentile=10,
    prominence_factor=0.1,
    width_factor=2.0,
    min_width=0.05,
    smoothing_sigma=2.0,
    mode="intensity",
    log_scale=False,
    verbose=True,
):
    """Find the strongest radial peak families and a q window around each.

    Parameters
    ----------
    peaks : Vector
        Polar peaks with fields ``(q, theta, intensity)``.
    num_bins : int
        Radial bins of the profile.
    q_range : (float, float), optional
        Profile range in 1/Angstrom; defaults to (0, max q).
    n_peaks : int
        Number of families kept, the most prominent first.
    height_percentile : float
        Profile percentile a family must exceed.
    prominence_factor : float
        Minimum prominence as a fraction of the profile maximum.
    width_factor : float
        Window width in units of the family's FWHM.
    min_width : float
        Minimum window width and family spacing, in 1/Angstrom.
    smoothing_sigma : float
        Gaussian smoothing of the profile, in bins.
    mode : {'intensity', 'count'}
        Profile of summed peak intensity or of peak count.
    log_scale : bool
        Detect on log(1 + profile), so weak families are not swamped.
    verbose : bool
        Print the families found.

    Returns
    -------
    centers : np.ndarray
        ``(n,)`` family q, ascending.
    windows : np.ndarray
        ``(n, 2)`` q windows.
    info : dict
        ``heights``, ``prominences``, ``widths_fwhm`` (aligned with ``centers``),
        and the smoothed ``profile`` on ``q``.
    """
    q, profile = radial_profile(peaks, num_bins, q_range, mode)
    if log_scale:
        profile = np.log1p(profile)
    if smoothing_sigma > 0:
        profile = _gaussian_1d(profile, smoothing_sigma, axis=0)
    dq = q[1] - q[0]
    found, props = find_peaks(
        profile,
        height=np.percentile(profile, height_percentile),
        prominence=prominence_factor * profile.max(),
        distance=max(1, int(min_width / dq)),
    )
    top = np.sort(np.argsort(props["prominences"])[::-1][:n_peaks])
    index = found[top]
    fwhm = peak_widths(profile, index, rel_height=0.5)[0] * dq
    half = np.maximum(width_factor * fwhm / 2, min_width / 2)
    centers = q[index]
    windows = np.clip(
        np.column_stack([centers - half, centers + half]), q[0] - dq / 2, q[-1] + dq / 2
    )
    info = {
        "heights": profile[index],
        "prominences": props["prominences"][top],
        "widths_fwhm": fwhm,
        "q": q,
        "profile": profile,
    }
    if verbose:
        for i, (c, w, f) in enumerate(zip(centers, windows, fwhm)):
            print(
                f"Family {i}: q = {c:.3f} 1/A (d = {1 / c:.2f} A), "
                f"window [{w[0]:.3f}, {w[1]:.3f}] 1/A, FWHM {f:.3f} 1/A"
            )
    return centers, windows, info


def plot_radial_profile(
    peaks: Vector,
    windows=None,
    centers=None,
    lines: dict | None = None,
    num_bins=200,
    q_range=None,
    mode="intensity",
    log_scale=False,
    show_d_spacing=False,
    figax=None,
):
    """Plot the radial profile with peak windows, centers and reference lines.

    Parameters
    ----------
    peaks : Vector
        Polar peaks with fields ``(q, theta, intensity)``.
    windows : np.ndarray, optional
        ``(n, 2)`` q windows, shaded with the profile filled inside them.
    centers : np.ndarray, optional
        Family centers, drawn as lines.
    lines : dict, optional
        ``{label: q values}`` reference lines, e.g. known reflections.
    mode, num_bins, q_range
        As in :func:`radial_profile`.
    log_scale : bool
        Logarithmic intensity axis.
    show_d_spacing : bool
        Add a d-spacing axis on top.
    figax : (Figure, Axes), optional
        Draw into existing axes.
    """
    q, profile = radial_profile(peaks, num_bins, q_range, mode)
    fig, ax = plt.subplots(figsize=(7, 4)) if figax is None else figax
    ax.plot(q, profile, color="black", lw=1.5)
    base = profile[profile > 0].min() if log_scale and np.any(profile > 0) else 0
    for i, (lo, hi) in enumerate(np.atleast_2d(windows) if windows is not None else []):
        inside = (q >= lo) & (q <= hi)
        ax.axvspan(lo, hi, color="tab:red", alpha=0.2, lw=0, label="windows" if i == 0 else None)
        ax.fill_between(q[inside], base, profile[inside], color="tab:red", alpha=0.5)
    for i, c in enumerate(np.atleast_1d(centers) if centers is not None else []):
        ax.axvline(c, color="tab:red", lw=1.5, label="centers" if i == 0 else None)
    colors = plt.cm.tab10(np.arange(len(lines or {})) % 10)
    for color, (label, values) in zip(colors, (lines or {}).items()):
        for j, v in enumerate(np.atleast_1d(values)):
            ax.axvline(v, color=color, ls="--", lw=1, label=label if j == 0 else None)
    if log_scale:
        ax.set_yscale("log")
    if show_d_spacing:
        inv = (lambda x: 1 / np.clip(x, 1e-12, None), lambda x: 1 / np.clip(x, 1e-12, None))
        ax.secondary_xaxis("top", functions=inv).set_xlabel("d-spacing (A)")
    ax.set(xlabel="q (1/A)", ylabel="peak intensity" if mode == "intensity" else "peak count")
    if windows is not None or centers is not None or lines:
        ax.legend(fontsize=8)
    return fig, ax


def peak_maps(peaks: Vector, q_ranges=None, mode="count", min_intensity=None, plot=True, **kwargs):
    """Per-position peak count or summed intensity, one map per q range.

    Parameters
    ----------
    peaks : Vector
        Polar peaks with fields ``(q, theta, intensity)``.
    q_ranges : np.ndarray, optional
        ``(n, 2)`` q windows; None counts every peak.
    mode : {'count', 'intensity'}
        What each map accumulates.
    min_intensity : float, optional
        Ignore weaker peaks.
    plot : bool
        Show the maps with :func:`~quantem.core.visualization.show_2d`.
    **kwargs
        Passed to ``show_2d``.

    Returns
    -------
    np.ndarray
        ``(n, scan_row, scan_col)`` maps.
    """
    data, rows, cols = _flat(peaks)
    ranges = np.array([[-np.inf, np.inf]]) if q_ranges is None else np.atleast_2d(q_ranges)
    keep = np.ones(len(data), bool) if min_intensity is None else data[:, 2] >= min_intensity
    weight = np.ones(len(data)) if mode == "count" else data[:, 2]
    maps = np.zeros((len(ranges), *peaks.shape[:2]))
    for k, (lo, hi) in enumerate(ranges):
        sel = keep & (data[:, 0] >= lo) & (data[:, 0] < hi)
        np.add.at(maps[k], (rows[sel], cols[sel]), weight[sel])
    if plot:
        titles = (
            [f"q {lo:.2f}-{hi:.2f} 1/A" for lo, hi in ranges] if q_ranges is not None else None
        )
        show_2d(list(maps), title=titles, cbar=True, **kwargs)
    return maps


def make_orientation_histogram(
    peaks: Vector,
    radial_ranges,
    upsample_factor=4.0,
    theta_step_deg=1.0,
    sigma_x=1.0,
    sigma_y=1.0,
    sigma_theta=3.0,
    normalize="stack",
):
    """Intensity histogram of peak orientation at each scan position.

    Each peak in a radial window is spread trilinearly over the upsampled scan
    grid and the theta bins (pi-periodic), and the histogram is then smoothed.

    Parameters
    ----------
    peaks : Vector
        Polar peaks with fields ``(q, theta, intensity)``.
    radial_ranges : np.ndarray
        ``(n, 2)`` q windows, or one ``(2,)`` window.
    upsample_factor : float
        Scan upsampling of the output.
    theta_step_deg : float
        Theta bin size in degrees.
    sigma_x, sigma_y : float
        Gaussian smoothing along scan rows and columns, in scan pixels.
    sigma_theta : float
        Gaussian smoothing along theta, in degrees (periodic).
    normalize : {'stack', 'image', None}
        Scale the maximum of the whole histogram, or of each window, to 1.

    Returns
    -------
    np.ndarray
        ``(n, scan_row * up, scan_col * up, num_theta)`` histogram.
    """
    ranges = np.atleast_2d(radial_ranges)
    num_theta = len(np.arange(0, 180, theta_step_deg))
    dtheta = np.deg2rad(theta_step_deg)
    size = np.round(np.array(peaks.shape[:2], float) * upsample_factor).astype(int)
    hist = np.zeros((len(ranges), size[0], size[1], num_theta))
    data, rows, cols = _flat(peaks)
    x = np.clip((rows + 0.5) * upsample_factor - 0.5, 0, size[0] - 2)
    y = np.clip((cols + 0.5) * upsample_factor - 0.5, 0, size[1] - 2)
    t = np.mod(data[:, 1], np.pi) / dtheta
    for k, (lo, hi) in enumerate(ranges):
        sel = (data[:, 0] ** 2 >= lo**2) & (data[:, 0] ** 2 < hi**2)
        _splat(hist[k], x[sel], y[sel], t[sel], data[sel, 2])
    if sigma_x:
        hist = _gaussian_1d(hist, sigma_x * upsample_factor, axis=1, mode="nearest", truncate=3.0)
    if sigma_y:
        hist = _gaussian_1d(hist, sigma_y * upsample_factor, axis=2, mode="nearest", truncate=3.0)
    if sigma_theta:
        hist = _gaussian_1d(hist, sigma_theta / theta_step_deg, axis=3, mode="wrap", truncate=2.0)
    if normalize == "stack" and hist.max() > 0:
        hist /= hist.max()
    elif normalize == "image":
        hist /= np.maximum(hist.max(axis=(1, 2, 3), keepdims=True), np.finfo(float).tiny)
    return hist


def make_flowline_map(
    orient_hist,
    thresh_seed=0.2,
    thresh_grow=0.05,
    thresh_collision=0.001,
    sep_seeds=None,
    sep_xy=6.0,
    sep_theta=5.0,
    sort_seeds="intensity",
    linewidth=2.0,
    step_size=0.5,
    min_steps=4,
    max_steps=1000,
    sigma_x=1.0,
    sigma_y=1.0,
    sigma_theta=2.0,
    seed=None,
    progress_bar=True,
):
    """Trace flowlines through an orientation histogram.

    Seeds are local maxima in theta above ``thresh_seed``. From each seed a line
    is grown both ways, perpendicular to the local orientation (a weighted mean
    over a Gaussian neighbourhood), until the histogram falls below
    ``thresh_grow`` or the line comes within ``sep_xy`` / ``sep_theta`` of one
    already drawn.

    Parameters
    ----------
    orient_hist : np.ndarray
        ``(n, rows, cols, num_theta)`` histogram from :func:`make_orientation_histogram`.
    thresh_seed, thresh_grow, thresh_collision : float
        Seed, growth and collision thresholds.
    sep_seeds : int, optional
        Seed spacing in pixels; defaults to half of ``sep_xy``.
    sep_xy, sep_theta : float or array
        Collision radius in pixels and theta bins, per window if an array.
    sort_seeds : {'intensity', 'random', None}
        Growth order of the seeds.
    linewidth : float
        Line width in pixels.
    step_size : float
        Growth step in pixels.
    min_steps, max_steps : int
        Shortest line drawn and longest line grown.
    sigma_x, sigma_y, sigma_theta : float
        Neighbourhood used to update the direction, in pixels and theta bins.
    seed : int, optional
        Random seed for ``sort_seeds='random'``.
    progress_bar : bool
        Show progress over seeds.

    Returns
    -------
    np.ndarray
        Flowline intensity, same shape as ``orient_hist``.
    """
    num_radii, n_r, n_c, n_t = orient_hist.shape
    sep_xy = np.broadcast_to(np.atleast_1d(sep_xy), (num_radii,))
    sep_theta = np.broadcast_to(np.atleast_1d(sep_theta), (num_radii,))
    sep_seeds = int(np.round(sep_xy.min() / 2 + 0.5)) if sep_seeds is None else int(sep_seeds)
    dtheta = np.pi / n_t
    theta = np.arange(n_t) * dtheta

    vx = np.arange(-np.ceil(2 * sigma_x), np.ceil(2 * sigma_x) + 1).astype(int)
    vy = np.arange(-np.ceil(2 * sigma_y), np.ceil(2 * sigma_y) + 1).astype(int)
    vt = np.arange(-np.ceil(2 * sigma_theta), np.ceil(2 * sigma_theta) + 1).astype(int)
    ax_, ay_, at_ = np.meshgrid(vx, vy, vt, indexing="ij")
    kernel = np.exp(-(ax_**2) / (2 * sigma_x**2) - ay_**2 / (2 * sigma_y**2))
    kernel *= np.exp(-(at_**2) / (2 * sigma_theta**2))
    neighbourhood = (
        vx[:, None, None],
        vy[None, :, None],
        vt[None, None, :],
        kernel / kernel.sum(),
    )
    rng = np.random.default_rng(seed)

    flow = np.zeros_like(orient_hist)
    for k in range(num_radii):
        orient = orient_hist[k]
        cr = np.arange(-np.ceil(sep_xy[k]), np.ceil(sep_xy[k]) + 1).astype(int)
        ct = np.arange(-np.ceil(sep_theta[k]), np.ceil(sep_theta[k]) + 1).astype(int)
        gx, gy, gt = np.meshgrid(cr, cr, ct, indexing="ij")
        ring = (gx**2 + gy**2) / sep_xy[k] ** 2 + gt**2 / sep_theta[k] ** 2
        collision = (cr[:, None, None], cr[None, :, None], ct[None, None, :])
        collision += (ring <= (1 + 1 / sep_xy[k]) ** 2,)

        seeds = (orient >= np.roll(orient, 1, axis=2)) & (orient >= np.roll(orient, -1, axis=2))
        seeds &= orient >= thresh_seed
        if sep_seeds > 1:
            seeds[np.arange(n_r) % sep_seeds != sep_seeds - 1] = False
            seeds[:, np.arange(n_c) % sep_seeds != sep_seeds - 1] = False
        xs, ys, ts = np.nonzero(seeds)
        if sort_seeds == "intensity":
            order = np.argsort(orient[seeds])[::-1]
        elif sort_seeds == "random":
            order = rng.permutation(len(xs))
        else:
            order = np.arange(len(xs))

        for i in tqdm(order, desc=f"Drawing flowlines {k}", disable=not progress_bar):
            xy0 = np.array([xs[i], ys[i]], float)
            t0 = _local_angle(orient, xy0, theta[ts[i]], dtheta, neighbourhood)
            args = (orient, flow[k], xy0, t0, dtheta, neighbourhood, collision, step_size)
            path = [
                p
                for sign in (1, -1)
                for p in _trace(*args, sign, thresh_grow, thresh_collision, max_steps)
            ]
            if len(path) + 2 > min_steps and path:
                _splat(flow[k], *np.array(path).T)

    flow *= step_size
    if linewidth > 1.0:
        s = linewidth - 1.0
        flow = _gaussian_1d(flow, s, axis=1, truncate=3.0)
        flow = _gaussian_1d(flow, s, axis=2, truncate=3.0) * s**2
    return flow


def _flowline_colors(theta, sym_rotation_order=2, theta_offset=np.pi):
    """RGB of each orientation: red, green and blue lobes 120 degrees apart."""
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.3, 1.0]])
    phase = theta_offset - np.asarray(theta) * sym_rotation_order
    lobes = [
        np.maximum(
            1
            - (np.mod(phase - 2 * np.pi * j / 3 + np.pi, 2 * np.pi) - np.pi) ** 2
            / (2 * np.pi / 3) ** 2,
            0,
        )
        for j in range(3)
    ]
    return np.stack(lobes, axis=-1) @ basis


def make_flowline_rainbow_image(
    orient_flowlines,
    int_range=(0.0, 0.2),
    sym_rotation_order=2,
    theta_offset=np.pi,
    greyscale=False,
    greyscale_max=True,
    white_background=False,
    power_scaling=1.0,
    sum_radial_bins=False,
    plot_images=True,
):
    """RGB image of the flowlines, coloured by orientation.

    Parameters
    ----------
    orient_flowlines : np.ndarray
        Output of :func:`make_flowline_map`.
    int_range : (float, float)
        Flowline intensity mapped to black and full colour.
    sym_rotation_order : int
        Rotational symmetry of the colouring.
    theta_offset : float or array
        Rotation of the colour wheel in radians, per window if an array. The
        default pi colours horizontal lines red and vertical lines cyan.
    greyscale : bool
        Grey levels instead of colour, from the max (``greyscale_max``) or mean
        over theta.
    white_background : bool
        Invert the background to white.
    power_scaling : float
        Power applied to the scaled intensity.
    sum_radial_bins : bool
        Combine all windows into one image.
    plot_images : bool
        Show the images.

    Returns
    -------
    np.ndarray
        ``(n, rows, cols, 3)`` RGB images.
    """

    def scaled(a):
        return np.clip((a - int_range[0]) / (int_range[1] - int_range[0]), 0, 1) ** power_scaling

    if greyscale:
        im = scaled(orient_flowlines.max(-1) if greyscale_max else orient_flowlines.mean(-1))
        image = np.repeat((1 - im if white_background else im)[..., None], 3, axis=-1)
    else:
        sig = scaled(orient_flowlines)
        theta = np.arange(sig.shape[-1]) * np.pi / sig.shape[-1]
        offsets = np.broadcast_to(np.atleast_1d(theta_offset), (sig.shape[0],))
        image = np.stack(
            [
                np.clip(s @ _flowline_colors(theta, sym_rotation_order, o), 0, 1)
                for s, o in zip(sig, offsets)
            ]
        )
        if white_background:
            hsv = rgb_to_hsv(image)
            hsv[..., 1], hsv[..., 2] = hsv[..., 2], 1.0
            image = hsv_to_rgb(hsv)
    return _finish(image, sum_radial_bins, white_background, plot_images)


def make_flowline_combined_image(
    orient_flowlines,
    int_range=(0.0, 0.2),
    cvals=((0.0, 0.7, 0.0), (1.0, 0.0, 0.0), (0.0, 0.7, 1.0)),
    white_background=False,
    power_scaling=1.0,
    sum_radial_bins=True,
    plot_images=True,
):
    """RGB image of the flowlines with one colour per radial window.

    Parameters
    ----------
    orient_flowlines : np.ndarray
        Output of :func:`make_flowline_map`.
    int_range : (float, float)
        Flowline intensity (summed over theta) mapped to black and full colour.
    cvals : array
        ``(n, 3)`` RGB colour of each window.
    white_background, power_scaling, sum_radial_bins, plot_images
        As in :func:`make_flowline_rainbow_image`.
    """
    sig = np.clip(
        (orient_flowlines.sum(axis=-1) - int_range[0]) / (int_range[1] - int_range[0]), 0, 1
    )
    sig = sig[..., None] ** power_scaling
    cvals = np.asarray(cvals, float)[: sig.shape[0], None, None, :]
    image = 1 - sig * (1 - cvals) if white_background else sig * cvals
    return _finish(image, sum_radial_bins, white_background, plot_images)


def make_flowline_rainbow_legend(
    im_size=256,
    sym_rotation_order=2,
    theta_offset=np.pi,
    white_background=False,
    radial_range=(0.45, 0.9),
    plot_legend=True,
):
    """Colour wheel for :func:`make_flowline_rainbow_image`.

    Each direction on the ring has the colour of a flowline running that way,
    computed with the same colour function as the image.

    Returns
    -------
    np.ndarray
        ``(im_size, im_size, 3)`` RGB image.
    """
    u = np.linspace(-1, 1, im_size)
    row, col = np.meshgrid(u, u, indexing="ij")
    radius = np.hypot(row, col)
    inside = (radius > radial_range[0]) & (radius < radial_range[1])
    # a line along direction psi is drawn for scattering at psi - 90 degrees
    theta = np.mod(np.arctan2(row, col) - np.pi / 2, np.pi)
    legend = _flowline_colors(theta, sym_rotation_order, theta_offset) * inside[..., None]
    if white_background:
        legend[~inside] = 1.0
    if plot_legend:
        _show_rgb(legend[None])
    return legend


# ---- helpers ----


def _finish(image, sum_radial_bins, white_background, plot_images):
    if sum_radial_bins:
        image = (
            image.min(axis=0, keepdims=True)
            if white_background
            else np.clip(image.sum(axis=0, keepdims=True), 0, 1)
        )
    if plot_images:
        _show_rgb(image)
    return image


def _show_rgb(images):
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5), squeeze=False)
    for ax, im in zip(axes[0], images):
        ax.imshow(im)
        ax.axis("off")
    return fig, axes


_PAD_MODES = {"nearest": "edge", "wrap": "wrap", "reflect": "symmetric"}


def _gaussian_1d(array, sigma, axis, mode="reflect", truncate=4.0):
    """Gaussian filter along one axis, as ``scipy.ndimage.gaussian_filter1d``."""
    radius = int(truncate * sigma + 0.5)
    x = torch.as_tensor(np.moveaxis(np.asarray(array, dtype=np.float64), axis, -1))
    shape = x.shape
    index = np.pad(np.arange(shape[-1]), radius, mode=_PAD_MODES[mode])
    k = torch.exp(-0.5 * (torch.arange(-radius, radius + 1, dtype=x.dtype) / sigma) ** 2)
    x = F.conv1d(x.reshape(-1, 1, shape[-1])[..., index], (k / k.sum()).view(1, 1, -1))
    return np.moveaxis(x.reshape(shape).numpy(), -1, axis)


def _splat(hist, x, y, t, weight):
    """Add ``weight`` trilinearly at fractional ``(x, y, t)``; t wraps."""
    n_x, n_y, n_t = hist.shape
    xf, yf, tf = np.floor(x).astype(int), np.floor(y).astype(int), np.floor(t).astype(int)
    dx, dy, dt = x - xf, y - yf, t - tf
    for ox, wx in ((0, 1 - dx), (1, dx)):
        for oy, wy in ((0, 1 - dy), (1, dy)):
            for ot, wt in ((0, 1 - dt), (1, dt)):
                np.add.at(
                    hist,
                    (
                        np.clip(xf + ox, 0, n_x - 1),
                        np.clip(yf + oy, 0, n_y - 1),
                        np.mod(tf + ot, n_t),
                    ),
                    weight * wx * wy * wt,
                )


def _interp(hist, x, y, t):
    """Trilinear value of ``hist`` at ``(x, y, t)``; x, y clamp and t wraps."""
    x, y = np.clip(x, 0, hist.shape[0] - 2), np.clip(y, 0, hist.shape[1] - 2)
    xf, yf, tf = int(x), int(y), int(np.floor(t))
    dx, dy, dt = x - xf, y - yf, t - tf
    t0, t1 = tf % hist.shape[2], (tf + 1) % hist.shape[2]
    block = hist[xf : xf + 2, yf : yf + 2][..., [t0, t1]]
    w = np.einsum("i,j,k->ijk", [1 - dx, dx], [1 - dy, dy], [1 - dt, dt])
    return float((block * w).sum())


def _local_angle(orient, xy, t, dtheta, neighbourhood):
    """Weighted mean orientation around ``(xy, t)``, unwrapped about ``t``."""
    vx, vy, vt, kernel = neighbourhood
    bins = np.round(t / dtheta).astype(int) + vt
    crop = (
        kernel
        * orient[
            np.clip(np.round(xy[0]).astype(int) + vx, 0, orient.shape[0] - 1),
            np.clip(np.round(xy[1]).astype(int) + vy, 0, orient.shape[1] - 1),
            np.mod(bins, orient.shape[2]),
        ]
    )
    total = crop.sum()
    return float((crop * bins * dtheta).sum() / total) if total > 0 else t


def _trace(
    orient,
    flow,
    xy0,
    t0,
    dtheta,
    neighbourhood,
    collision,
    step,
    sign,
    thresh_grow,
    thresh_collision,
    max_steps,
):
    """Grow one flowline from a seed; returns ``(x, y, t_bin, value)`` per step."""
    cx, cy, ct, mask = collision
    xy, t, path = xy0.copy(), t0, []
    for _ in range(max_steps):
        xy = xy + sign * step * np.array([np.cos(t), -np.sin(t)])
        if xy[0] < 0 or xy[1] < 0 or xy[0] > orient.shape[0] or xy[1] > orient.shape[1]:
            break
        value = _interp(orient, xy[0], xy[1], t / dtheta)
        near = flow[
            np.clip(np.round(xy[0]).astype(int) + cx, 0, orient.shape[0] - 1),
            np.clip(np.round(xy[1]).astype(int) + cy, 0, orient.shape[1] - 1),
            np.mod(np.round(t / dtheta).astype(int) + ct, orient.shape[2]),
        ]
        if value < thresh_grow or near[mask].max() > thresh_collision:
            break
        t = _local_angle(orient, xy, t, dtheta, neighbourhood)
        path.append((xy[0], xy[1], t / dtheta, value))
    return path
