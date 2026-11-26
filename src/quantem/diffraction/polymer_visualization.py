"""Static figures of single polymer diffraction patterns with their detected peaks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.diffraction.polar_transform import polar_transform


def save_peak_figures(
    dp,
    peaks_rc,
    intensities,
    origin,
    position,
    intensity_map,
    scan_shape,
    *,
    ellipse=None,
    map_title="",
    prefix="peaks",
    save_dir=".",
    vmin_cartesian=0.0,
    vmax_cartesian=7.0,
    norm_upper_quantile=None,
    norm_power=1.0,
    gaussian_filter_sigma=None,
    zoom=1.0,
    dp_cmap="gray",
    map_cmap="viridis",
    selected_peak_color="red",
    central_beam_color="red",
    peak_marker_size=None,
    peak_size_range=(30, 300),
    crosshair_width_peaks=2,
    peak_alpha=1.0,
    crosshair_color="r",
    crosshair_size=15,
    crosshair_width=2,
    show_polar=True,
    vmax_polar=None,
    num_annular_bins=180,
    two_fold_symmetry=True,
    figsize=(6, 6),
    dpi=400,
) -> list[Path]:
    """Save a scan map, a diffraction pattern and its polar transform, with the peaks marked.

    Three PDFs are written to ``save_dir`` as ``{prefix}_ry{ry}_rx{rx}_{name}.pdf``:

    - ``intensity_map``: ``intensity_map`` with the position circled and a zoomed inset.
    - ``diffraction``: the pattern with open circles on the peaks and a filled marker
      on the direct beam.
    - ``polar`` (if ``show_polar``): the pattern resampled on (radius, angle) about
      ``origin`` by :func:`~quantem.diffraction.polar_transform.polar_transform`,
      with the same peaks.

    Parameters
    ----------
    dp : ndarray
        ``(H, W)`` diffraction pattern.
    peaks_rc : ndarray
        ``(M, 2)`` peak positions ``(row, col)`` in detector pixels.
    intensities : ndarray or None
        ``(M,)`` peak intensities, which set the marker sizes unless
        ``peak_marker_size`` is given.
    origin : (float, float)
        Direct-beam position ``(row, col)`` in detector pixels.
    position : (int, int)
        Scan position ``(ry, rx)``.
    intensity_map : ndarray
        Scan image, ``(scan_row, scan_col)`` or an integer multiple of it, gray or RGB.
    scan_shape : (int, int)
        ``(scan_row, scan_col)``, to place ``position`` on an upsampled map.
    ellipse : array-like, optional
        ``[e11, e12]`` correction of
        :class:`~quantem.diffraction.calibration.DiffractionCalibration`; the polar
        transform then follows the corrected rings.
    map_title : str
        Title of the scan map.
    prefix, save_dir
        File name prefix and output folder (created if needed).
    vmin_cartesian, vmax_cartesian : float
        Display range of the pattern after normalization.
    norm_upper_quantile : float, optional
        Clip the pattern at this quantile.
    norm_power : float
        Raise the pattern, scaled to its maximum, to this power.
    gaussian_filter_sigma : float, optional
        Smooth the pattern, in pixels.
    zoom : float
        Show ``1 / zoom`` of the pattern about the origin, and the polar panel out
        to the same radius.
    dp_cmap, map_cmap : str
        Colormaps of the pattern and the scan map.
    selected_peak_color, central_beam_color : str
        Colors of the peak circles and the direct-beam marker.
    peak_marker_size : float, optional
        Fixed marker area; otherwise it scales with intensity over ``peak_size_range``.
    crosshair_width_peaks, peak_alpha : float
        Line width and opacity of the peak markers.
    crosshair_color, crosshair_size, crosshair_width
        Marker of the position on the scan map.
    show_polar : bool
        Also save the polar panel.
    vmax_polar : float, optional
        Display maximum of the polar panel; defaults to ``vmax_cartesian``.
    num_annular_bins : int
        Angular bins of the polar panel.
    two_fold_symmetry : bool
        Fold the polar angle onto 0-180 degrees.
    figsize, dpi
        Size and resolution of each figure.

    Returns
    -------
    list of Path
        The files written.
    """
    ry, rx = (int(i) for i in position)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(dp, dtype=np.float64)
    if norm_upper_quantile is not None:
        image = np.clip(image, 0, np.quantile(image, norm_upper_quantile))
    peak_value = np.nanmax(image)
    if norm_power != 1.0 and np.isfinite(peak_value) and peak_value > 0:
        image = (image / peak_value) ** norm_power * peak_value
    if gaussian_filter_sigma:
        image = gaussian_filter(image, gaussian_filter_sigma)
    peaks_rc = np.asarray(peaks_rc, dtype=np.float64).reshape(-1, 2)
    origin = np.asarray(origin, dtype=np.float64)
    central = _central_peak(peaks_rc, origin, image.shape)
    style = dict(
        sizes=_marker_sizes(intensities, len(peaks_rc), peak_marker_size, peak_size_range),
        color=selected_peak_color,
        central_color=central_beam_color,
        width=crosshair_width_peaks,
        alpha=peak_alpha,
    )
    paths = []

    def save(fig, name):
        path = save_dir / f"{prefix}_ry{ry}_rx{rx}_{name}.pdf"
        fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=dpi)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=figsize)
    _draw_map(
        ax,
        intensity_map,
        (ry, rx),
        scan_shape,
        cmap=map_cmap,
        color=crosshair_color,
        size=crosshair_size,
        width=crosshair_width,
    )
    ax.set_title(map_title)
    save(fig, "intensity_map")

    top, left, crop = _zoom_window(image, origin, zoom)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(crop, cmap=dp_cmap, vmin=vmin_cartesian, vmax=vmax_cartesian)
    _draw_peaks(
        ax,
        peaks_rc[:, 1] - left,
        peaks_rc[:, 0] - top,
        central,
        (origin[1] - left, origin[0] - top),
        **style,
    )
    ax.set(
        xlim=(-0.5, crop.shape[1] - 0.5), ylim=(crop.shape[0] - 0.5, -0.5), xticks=[], yticks=[]
    )
    ax.set_title(f"Diffraction Pattern (Ry={ry}, Rx={rx})")
    save(fig, "diffraction")

    if show_polar:
        axes = None if ellipse is None else _ellipse_axes(ellipse)
        polar = polar_transform(
            Dataset4dstem.from_array(image[None, None].astype(np.float32)),
            origin_array=origin,
            ellipse_params=axes,
            num_annular_bins=num_annular_bins,
            radial_max=None if zoom == 1 else min(crop.shape) / 2,
            two_fold_rotation_symmetry=two_fold_symmetry,
            scan_pos=(0, 0),
        )
        polar = polar.cpu().numpy()
        period = 180.0 if two_fold_symmetry else 360.0
        step = period / num_annular_bins
        radius, phi = _polar_coords(peaks_rc - origin, axes)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(
            polar,
            cmap=dp_cmap,
            vmin=vmin_cartesian,
            vmax=vmax_cartesian if vmax_polar is None else vmax_polar,
            aspect="auto",
            extent=(-0.5, polar.shape[1] - 0.5, (polar.shape[0] - 0.5) * step, -0.5 * step),
        )
        _draw_peaks(ax, radius, np.mod(np.rad2deg(phi), period), central, None, **style)
        ax.set(
            xlim=(-0.5, polar.shape[1] - 0.5),
            xlabel="radius (pixels)",
            ylabel="detector angle (deg)",
            title=f"Polar (Ry={ry}, Rx={rx})",
        )
        save(fig, "polar")
    return paths


def _ellipse_axes(ellipse) -> tuple[float, float, float]:
    """``(a, b, theta_deg)`` with ``b = 1`` that ``ellipse_from_axes`` maps onto ``ellipse``."""
    e11, e12 = np.asarray(ellipse, dtype=np.float64)
    e = np.hypot(e11, e12)
    return float((1 + e) / (1 - e)), 1.0, float(np.rad2deg(np.arctan2(-e12, e11) / 2))


def _polar_coords(offsets, axes):
    """``(radius, angle)`` of ``(row, col)`` offsets on the sampling grid of ``polar_transform``."""
    rows, cols = offsets[:, 0], offsets[:, 1]
    if axes is None:
        return np.hypot(rows, cols), np.arctan2(rows, cols)
    a, b, theta_deg = axes
    t = np.deg2rad(theta_deg)
    # undo polar_transform's rotation into the ellipse frame and its a/b stretch
    u = (cols * np.cos(t) + rows * np.sin(t)) * b / a
    v = rows * np.cos(t) - cols * np.sin(t)
    return np.hypot(u, v), np.arctan2(v, u) + t


def _central_peak(peaks_rc, origin, shape):
    """Index of the peak on the direct beam, or None."""
    if len(peaks_rc) == 0:
        return None
    distance = np.hypot(*(peaks_rc - origin).T)
    i = int(np.argmin(distance))
    return i if distance[i] <= max(4.0, 0.03 * min(shape)) else None


def _marker_sizes(intensities, n, fixed, size_range):
    if fixed is not None:
        return np.full(n, float(fixed))
    if intensities is None or n == 0:
        return np.full(n, 100.0)
    values = np.asarray(intensities, dtype=np.float64)
    span = values.max() - values.min()
    scaled = (values - values.min()) / span if span > 0 else np.ones(n)
    return size_range[0] + scaled * (size_range[1] - size_range[0])


def _draw_peaks(ax, x, y, central, center, *, sizes, color, central_color, width, alpha):
    """Open circles on the peaks; a filled marker on ``center``, else on the central peak."""
    beam = dict(s=120, facecolors=central_color, edgecolors="k", linewidths=width, zorder=10)
    others = np.ones(len(x), bool)
    if central is not None:
        others[central] = False
    if center is not None:
        ax.scatter(*center, alpha=alpha, **beam)
    elif central is not None:
        ax.scatter(x[central], y[central], alpha=alpha, **beam)
    ax.scatter(
        x[others],
        y[others],
        s=sizes[others],
        facecolors="none",
        edgecolors=color,
        linewidths=width,
        alpha=alpha,
        zorder=5,
    )


def _zoom_window(image, origin, zoom):
    """``(top, left, crop)``: the ``1 / zoom`` of ``image`` centered on ``origin``."""
    h, w = image.shape
    new_h, new_w = (max(1, min(n, int(n / zoom))) for n in (h, w))
    top = min(max(int(round(origin[0] - (new_h - 1) / 2)), 0), h - new_h)
    left = min(max(int(round(origin[1] - (new_w - 1) / 2)), 0), w - new_w)
    return top, left, image[top : top + new_h, left : left + new_w]


def _draw_map(ax, intensity_map, position, scan_shape, *, cmap, color, size, width, margin=4):
    """The scan map with ``position`` circled and a zoomed inset around it."""
    image = np.asarray(intensity_map)
    up = image.shape[0] // int(scan_shape[0])
    if up < 1 or image.shape[0] % scan_shape[0] or image.shape[1] != up * scan_shape[1]:
        raise ValueError(
            f"intensity_map shape {image.shape[:2]} is not a multiple of the scan {scan_shape}."
        )
    limits = {}
    if not (image.ndim == 3 and image.shape[2] in (3, 4)) and np.isfinite(image).any():
        vmin, vmax = np.quantile(image[np.isfinite(image)], [0.01, 0.99])
        limits = dict(vmin=vmin, vmax=vmax)
    show = dict(cmap=cmap, interpolation="nearest", **limits)
    ax.imshow(image, **show)
    # center of the position's block of map pixels
    row, col = ((i + 0.5) * up - 0.5 for i in position)
    ax.scatter(col, row, s=size, facecolors="none", edgecolors=color, linewidths=width)

    half = margin * up
    r0, r1 = max(0, int(row) - half), min(image.shape[0], int(row) + half + 1)
    c0, c1 = max(0, int(col) - half), min(image.shape[1], int(col) + half + 1)
    inset = inset_axes(ax, width="30%", height="30%", loc="upper right", borderpad=1.5)
    inset.imshow(image[r0:r1, c0:c1], extent=(c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5), **show)
    inset.add_patch(
        Rectangle(
            (position[1] * up - 0.5, position[0] * up - 0.5),
            up,
            up,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            zorder=10,
        )
    )
    inset.set(xticks=[], yticks=[])
    inset.set_title(f"{2 * margin + 1}x{2 * margin + 1} zoom", fontsize=8, pad=2)
    ax.add_patch(
        Rectangle(
            (c0 - 0.5, r0 - 0.5),
            c1 - c0,
            r1 - r0,
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
            linestyle="--",
            alpha=0.7,
        )
    )
    ax.set(
        xlabel="Rx (upsampled)" if up > 1 else "Rx", ylabel="Ry (upsampled)" if up > 1 else "Ry"
    )
