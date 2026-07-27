# from collections.abc import Sequence
import warnings
import tempfile
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter, map_coordinates, label
from tqdm import tqdm
import torch
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.diffraction.polymer_models import (
    PAPER_MODEL_ID,
    PAPER_MODEL_VERSION,
    MultiChannelCNN2d,
    build_polymer_model,
    resolve_polymer_model,
)
from quantem.diffraction.polymer_normalization import (
    LegacyNormalizationAdapter,
    NormalizationStrategy,
    resolve_normalization_strategy,
)
from quantem.core.datastructures import Vector
from quantem.core.visualization import show_2d
from quantem.diffraction.polar_transform import (
    centered_dp_mean,
    find_central_beams_4d,
    fit_origin_roi,
    find_origin as find_origin_angular_uniformity,
    polar_transform as karen_polar_transform,
    polar_transform_peaks as karen_polar_transform_peaks,
)
from quantem.diffraction.ellipse_fitting import fit_ellipse_from_ridge, fit_ellipse_from_ring
from quantem.diffraction.peak_detection import detect_blobs, find_central_beam_from_peaks
from quantem.diffraction.peak_visualization import (
    _intensity_display_limits,
    _normalized_dp,
    _resolve_intensity_map,
    peak_radial_count_plot as _peak_radial_count_plot,
    peak_radial_intensity_plot as _peak_radial_intensity_plot,
    plot_interactive_image_map as _plot_interactive_image_map,
    plot_peak_count_map as _plot_peak_count_map,
    plot_peak_histogram_map as _plot_peak_histogram_map,
    save_diffraction_figures as _save_diffraction_figures,
    visualize_selected_patterns as _visualize_selected_patterns,
)
from quantem.diffraction.vector_fields import (
    vector_field_cell as _vector_field_cell,
    vector_field_flat as _vector_field_flat,
)
from quantem.diffraction.orientation_correlation import (
    calculate_orientation_correlation as _calculate_orientation_correlation,
)
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffraction.polymer_utils import parse_reciprocal_units, sample_average_from_image
from emdfile import tqdmnd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
import ipywidgets as widgets
from ipywidgets import IntSlider, Button, HBox, VBox, interactive_output
from IPython.display import clear_output, display
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.colors import BoundaryNorm, hsv_to_rgb, rgb_to_hsv

# Default 1/e decay length for the orientation-correlation slope fit, as a
# fraction of the fitted lobe's distance span. Chosen on real scan data
# (pg3T2 07-03-2024/61): it holds the fitted intercept within ~1.7 degrees of
# the measured boundary at zero separation across all six ring pairs, versus
# up to 7 degrees for an unweighted fit, while keeping enough effective points
# for a stable slope. See plot_orientation_correlation(slope_weight_scale=...).
SLOPE_WEIGHT_FRACTION = 0.10


def _apply_zoom_crop(data, zoom_factor, center=None):
    """Crop data to center region based on zoom factor."""
    if zoom_factor == 1.0:
        return data, (0, data.shape[0], 0, data.shape[1])

    h, w = data.shape
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    new_h = max(1, min(h, new_h))
    new_w = max(1, min(w, new_w))
    
    if center is None:
        center_y, center_x = (h - 1) / 2, (w - 1) / 2
    else:
        center_y, center_x = center

    top = int(round(center_y - (new_h - 1) / 2))
    left = int(round(center_x - (new_w - 1) / 2))
    top = min(max(top, 0), h - new_h)
    left = min(max(left, 0), w - new_w)
    
    return data[top:top+new_h, left:left+new_w], (top, top+new_h, left, left+new_w)


def _display_center(image_centers, ry_data, rx_data, image_shape):
    center_y, center_x = image_shape[0] / 2, image_shape[1] / 2
    if image_centers is not None:
        stored_center = image_centers[:, ry_data, rx_data]
        if np.all(np.isfinite(stored_center)) and not np.allclose(stored_center, 0):
            center_y, center_x = stored_center
    return center_y, center_x


def _has_peak_positions(peaks_x, peaks_y):
    return (
        peaks_x is not None
        and peaks_y is not None
        and len(peaks_x) > 0
        and len(peaks_y) > 0
    )


def _central_peak_index(peaks_x, peaks_y, peaks_r_invA, center, max_dist=None):
    """Index of the detected central-beam peak, or ``None``.

    Defined as the detected peak nearest the calibrated beam ``center`` (from
    ``image_centers`` / ``find_central_beams_4d``), but only when it lies within
    ``max_dist`` pixels of it. The filled central-beam marker itself is always drawn at
    ``center``; this index only flags which detected peak, if any, to drop from the
    open-circle set so a ring is not drawn on top of the beam.

    ``peaks_r_invA`` is unused (kept for call-site compatibility): selecting the beam by
    smallest polar radius made the marker jump to an off-center low-q Bragg peak when the
    beam itself was not detected as a peak.
    """
    if not _has_peak_positions(peaks_x, peaks_y):
        return None
    center_y, center_x = center
    distances = np.sqrt(
        (np.asarray(peaks_x) - center_x) ** 2 + (np.asarray(peaks_y) - center_y) ** 2
    )
    idx = int(np.argmin(distances))
    if max_dist is not None and distances[idx] > max_dist:
        return None
    return idx


def _central_beam_max_dist(image_shape):
    """Pixel radius within which a detected peak counts as the central beam.

    Small enough that finite-q Bragg peaks are never mistaken for the beam, generous
    enough to absorb a few-pixel disagreement between center-finding and peak detection.
    """
    return max(4.0, 0.03 * min(image_shape[0], image_shape[1]))


def _zoom_peak_overlay(
    dp_data,
    peaks_x,
    peaks_y,
    peaks_r_invA,
    peak_ints,
    central_idx,
    zoom,
    fallback_center,
):
    if zoom == 1:
        return dp_data, peaks_x, peaks_y, peaks_r_invA, peak_ints, central_idx, fallback_center

    dp_data, ranges = _apply_zoom_crop(dp_data, zoom, center=fallback_center)
    top, bot, left, right = ranges
    display_center = (fallback_center[0] - top, fallback_center[1] - left)

    if _has_peak_positions(peaks_x, peaks_y):
        mask = (top <= peaks_y) & (peaks_y < bot) & (left <= peaks_x) & (peaks_x < right)
        kept_indices = np.flatnonzero(mask)
        if central_idx is not None:
            central_matches = np.flatnonzero(kept_indices == central_idx)
            central_idx = int(central_matches[0]) if len(central_matches) else None
        peaks_y = peaks_y[mask] - top
        peaks_x = peaks_x[mask] - left
        peaks_r_invA = peaks_r_invA[mask] if peaks_r_invA is not None else None
        peak_ints = peak_ints[mask] if peak_ints is not None else None

    return dp_data, peaks_x, peaks_y, peaks_r_invA, peak_ints, central_idx, display_center


def _polar_peak_bins(
    polar_r,
    polar_theta,
    max_radius_invA,
    num_radial_bins,
    num_annular_bins,
    two_fold_symmetry,
):
    r_bins = polar_r / max_radius_invA * num_radial_bins
    theta_period = np.pi if two_fold_symmetry else 2 * np.pi
    theta_bins = polar_theta / theta_period * num_annular_bins
    return r_bins, theta_bins


def _plot_bragg_peaks_on_ax(
    ax,
    peaks_x,
    peaks_y,
    peaks_r_invA,
    peak_intensities,
    central_idx,
    *,
    radial_range=None,
    show_all_peaks=False,
    selected_peak_color="red",
    other_peak_color="gray",
    central_beam_color="red",
    peak_intensity_mode="size",
    peak_size_range=(30, 300),
    peak_cmap="hot",
    peak_vmin=None,
    peak_vmax=None,
    crosshair_width_peaks=2,
    crosshair_scaling_peaks=1,
    crosshair_scaling_central_beam=1,
    peak_marker="o",
    peak_marker_facecolors="none",
    peak_marker_size=None,
    peak_alpha=0.8,
    central_alpha=0.95,
    central_linewidth=2,
    add_colorbar=False,
    center=None,
    show_center=True,
    show_central_beam=True,
):
    # show_central_beam=False fully suppresses the central-beam marker (both the
    # provided-center dot and the detected-central-peak dot); the central peak is
    # still excluded from the open-circle set via non_central below.
    plot_detected_center = (center is None or not show_center) and show_central_beam
    if center is not None and show_center and show_central_beam:
        center_y, center_x = center
        ax.scatter(
            center_x,
            center_y,
            s=120 * crosshair_scaling_central_beam,
            alpha=central_alpha,
            linewidths=central_linewidth,
            edgecolors="k",
            facecolors=central_beam_color,
            marker="o",
            zorder=10,
        )

    if peaks_r_invA is None or len(peaks_r_invA) == 0:
        return
    if not _has_peak_positions(peaks_x, peaks_y):
        return

    central_style = dict(
        edgecolors="k",
        facecolors=central_beam_color,
        marker="o",
        zorder=10,
    )

    if radial_range is not None:
        mask = (peaks_r_invA >= radial_range[0]) & (peaks_r_invA < radial_range[1])
        if show_all_peaks and np.any(~mask):
            out_indices = np.where(~mask)[0]
            if plot_detected_center and central_idx is not None and central_idx in out_indices:
                ax.scatter(
                    peaks_x[central_idx],
                    peaks_y[central_idx],
                    s=30,
                    alpha=0.95,
                    linewidths=2,
                    **central_style,
                )
            other_out_mask = out_indices != central_idx
            if np.any(other_out_mask):
                ax.scatter(
                    peaks_x[out_indices[other_out_mask]],
                    peaks_y[out_indices[other_out_mask]],
                    c=other_peak_color,
                    s=30,
                    alpha=0.5,
                    marker="x",
                    linewidths=1.5,
                )
        if not np.any(mask):
            return
        in_range_indices = np.where(mask)[0]
        if central_idx is not None and central_idx in in_range_indices:
            central_idx = np.where(in_range_indices == central_idx)[0][0]
        else:
            central_idx = None
        peaks_x, peaks_y = peaks_x[mask], peaks_y[mask]
        peak_intensities = peak_intensities[mask] if peak_intensities is not None else None

    if central_idx is not None:
        if plot_detected_center:
            ax.scatter(
                peaks_x[central_idx],
                peaks_y[central_idx],
                s=120 * crosshair_scaling_central_beam,
                alpha=central_alpha,
                linewidths=central_linewidth,
                **central_style,
            )
        non_central = np.ones(len(peaks_x), dtype=bool)
        non_central[central_idx] = False
    else:
        non_central = np.ones(len(peaks_x), dtype=bool)

    if not np.any(non_central):
        return

    if peak_intensities is not None and peak_intensity_mode is not None:
        int_subset = peak_intensities[non_central]
        int_min = peak_vmin if peak_vmin is not None else np.min(int_subset)
        int_max = peak_vmax if peak_vmax is not None else np.max(int_subset)
        norm_int = (
            (int_subset - int_min) / (int_max - int_min)
            if int_max > int_min
            else np.ones_like(int_subset)
        )
        if peak_intensity_mode == "color":
            colors, sizes = plt.cm.get_cmap(peak_cmap)(norm_int), 100
        elif peak_intensity_mode == "size":
            colors = selected_peak_color
            sizes = peak_size_range[0] + norm_int * (peak_size_range[1] - peak_size_range[0])
        elif peak_intensity_mode == "both":
            colors = plt.cm.get_cmap(peak_cmap)(norm_int)
            sizes = peak_size_range[0] + norm_int * (peak_size_range[1] - peak_size_range[0])
        else:
            colors, sizes = selected_peak_color, 100
        if peak_marker_size is not None:
            sizes = peak_marker_size

        scatter_kwargs = dict(
            s=sizes * crosshair_scaling_peaks,
            alpha=peak_alpha,
            marker=peak_marker,
            facecolors=peak_marker_facecolors,
            linewidths=crosshair_width_peaks,
            zorder=5,
        )
        if peak_marker_facecolors == "none":
            ax.scatter(
                peaks_x[non_central],
                peaks_y[non_central],
                edgecolors=colors,
                **scatter_kwargs,
            )
        else:
            ax.scatter(
                peaks_x[non_central],
                peaks_y[non_central],
                c=colors,
                **scatter_kwargs,
            )

        if peak_intensity_mode in ["color", "both"]:
            sm = plt.cm.ScalarMappable(
                cmap=peak_cmap, norm=plt.Normalize(vmin=int_min, vmax=int_max)
            )
            sm.set_array([])
            if add_colorbar:
                plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046).set_label(
                    "Peak Intensity", fontsize=8
                )
    else:
        scatter_kwargs = dict(
            s=100,
            alpha=peak_alpha,
            marker=peak_marker,
            facecolors=peak_marker_facecolors,
            linewidths=2,
            zorder=5,
        )
        if peak_marker_facecolors == "none":
            ax.scatter(
                peaks_x[non_central],
                peaks_y[non_central],
                edgecolors=selected_peak_color,
                **scatter_kwargs,
            )
        else:
            ax.scatter(
                peaks_x[non_central],
                peaks_y[non_central],
                c=selected_peak_color,
                **scatter_kwargs,
            )


def _draw_peaks_data_circles(
    ax,
    peaks_x,
    peaks_y,
    peak_intensities,
    central_idx,
    center,
    *,
    marker_scaled=True,
    marker_size=8.0,
    marker_size_min=4.0,
    marker_size_max=16.0,
    selected_peak_color="red",
    central_beam_color="red",
    show_central_beam=True,
    central_size=5.0,
    peak_linewidth=2.0,
    central_linewidth=1.5,
):
    """Draw Bragg-peak markers as circles in DATA coordinates (radius in detector
    pixels).

    Unlike ``_plot_bragg_peaks_on_ax`` (which sizes markers in fixed points**2, so they
    do not track the figure size), these circles are in data units and therefore cover a
    constant fraction of the diffraction pattern at any panel size -- matching the widget
    canvas, where the overlay sizes markers in data pixels scaled to the display (js
    ``drawDot`` / peak rings). Open circles for detected peaks (skipping the central-beam
    peak) plus a filled dot at the calibrated beam ``center`` (row, col).
    """
    from matplotlib.patches import Circle

    if _has_peak_positions(peaks_x, peaks_y):
        px = np.asarray(peaks_x)
        py = np.asarray(peaks_y)
        idxs = [
            i for i in range(len(px))
            if i != central_idx and np.isfinite(px[i]) and np.isfinite(py[i])
        ]
        ints = None if peak_intensities is None else np.asarray(peak_intensities, dtype=float)
        use_scaled = marker_scaled and ints is not None and len(idxs) > 0
        if use_scaled:
            vals = ints[idxs]
            imin, imax = float(np.nanmin(vals)), float(np.nanmax(vals))
            rng = (imax - imin) if imax > imin else 1.0
        for i in idxs:
            if use_scaled:
                norm = (ints[i] - imin) / rng
                if not np.isfinite(norm):
                    norm = 0.5
                r = marker_size_min + norm * (marker_size_max - marker_size_min)
            else:
                r = marker_size
            ax.add_patch(Circle(
                (px[i], py[i]), radius=r, fill=False,
                edgecolor=selected_peak_color, linewidth=peak_linewidth, zorder=5,
            ))

    if show_central_beam and center is not None:
        cy, cx = center
        ax.add_patch(Circle(
            (cx, cy), radius=central_size, facecolor=central_beam_color,
            edgecolor="k", linewidth=central_linewidth, zorder=10,
        ))


class ScanMaskEditor:
    """Interactive, persistent circular scan-mask editor.

    The horizontal X control maps directly to the scan-column coordinate. The
    vertical Y control is visually inverted relative to the array row index so
    moving the slider upward moves the probe marker upward on an ``origin="upper"``
    scan image.
    """

    SCHEMA_VERSION = 2
    GEOMETRIES = ("circle", "ellipse", "square", "rectangle")

    def __init__(
        self,
        analysis,
        *,
        initial_x=None,
        initial_y=None,
        initial_radius=None,
        initial_geometry="circle",
        initial_size_x=None,
        initial_size_y=None,
        reference_image=None,
        state_path=None,
        overlay_alpha=0.28,
        crosshair_width=2,
        crosshair_size=12,
        autosave=False,
        display_widget=True,
    ):
        self.analysis = analysis
        self.scan_shape = tuple(int(v) for v in analysis.dataset_cartesian.shape[:2])
        self.state_path = None if state_path is None else Path(state_path)
        self.autosave = bool(autosave)
        self.overlay_alpha = float(overlay_alpha)
        self.crosshair_width = float(crosshair_width)
        self.crosshair_size = float(crosshair_size)
        self._syncing = False
        self._dirty = False
        self._saved = False
        self._loaded = False

        rows, columns = self.scan_shape
        center_row = rows // 2 if initial_y is None else int(initial_y)
        center_column = columns // 2 if initial_x is None else int(initial_x)
        radius = (
            max(1, min(rows, columns) // 3)
            if initial_radius is None
            else int(initial_radius)
        )
        geometry = str(initial_geometry).lower()
        size_x = radius if initial_size_x is None else int(initial_size_x)
        size_y = radius if initial_size_y is None else int(initial_size_y)
        loaded_mask = None
        if self.state_path is not None and self.state_path.is_file():
            state = self._read_state(self.state_path)
            center_row = state["center_row"]
            center_column = state["center_column"]
            geometry = state["geometry"]
            size_x = state["size_x"]
            size_y = state["size_y"]
            loaded_mask = state["mask"]
            self._loaded = True
            self._saved = True

        self._validate_geometry(
            center_row, center_column, geometry, size_x, size_y
        )
        self.reference_image = self._resolve_reference_image(reference_image)
        self._preview_mask = (
            loaded_mask.copy()
            if loaded_mask is not None
            else self._geometry_mask(
                center_row, center_column, geometry, size_x, size_y
            )
        )

        maximum_radius = int(np.ceil(np.hypot(rows - 1, columns - 1))) + 1
        self.x_slider = widgets.IntSlider(
            value=center_column,
            min=0,
            max=columns - 1,
            step=1,
            description="X",
            continuous_update=True,
            readout=False,
            style={"description_width": "18px"},
            layout=widgets.Layout(width="375px"),
        )
        # Slider value increases upward, while array row indices increase downward.
        self.y_slider = widgets.IntSlider(
            value=rows - 1 - center_row,
            min=0,
            max=rows - 1,
            step=1,
            description="Y",
            orientation="vertical",
            continuous_update=True,
            readout=False,
            style={"description_width": "18px"},
            layout=widgets.Layout(height="281px", width="52px"),
        )
        self.geometry_selector = widgets.Dropdown(
            options=[
                ("Circular", "circle"),
                ("Elliptical", "ellipse"),
                ("Square", "square"),
                ("Rectangular", "rectangle"),
            ],
            value=geometry,
            description="Shape",
            style={"description_width": "42px"},
            layout=widgets.Layout(width="155px"),
        )
        self.size_x_slider = widgets.IntSlider(
            value=size_x,
            min=1,
            max=maximum_radius,
            step=1,
            description="Radius",
            continuous_update=True,
            readout=False,
            style={"description_width": "55px"},
            layout=widgets.Layout(width="375px"),
        )
        self.size_y_slider = widgets.IntSlider(
            value=size_y,
            min=1,
            max=maximum_radius,
            step=1,
            description="Y radius",
            continuous_update=True,
            readout=False,
            style={"description_width": "68px"},
            layout=widgets.Layout(width="375px"),
        )
        # Historical public attribute retained for callers that customize it.
        self.radius_slider = self.size_x_slider
        self.x_input = widgets.BoundedIntText(
            value=center_column,
            min=0,
            max=columns - 1,
            description="X column",
            style={"description_width": "62px"},
            layout=widgets.Layout(width="150px"),
        )
        self.y_input = widgets.BoundedIntText(
            value=center_row,
            min=0,
            max=rows - 1,
            description="Y row",
            style={"description_width": "52px"},
            layout=widgets.Layout(width="140px"),
        )
        self.size_x_input = widgets.BoundedIntText(
            value=size_x,
            min=1,
            max=maximum_radius,
            description="Radius",
            style={"description_width": "48px"},
            layout=widgets.Layout(width="130px"),
        )
        self.size_y_input = widgets.BoundedIntText(
            value=size_y,
            min=1,
            max=maximum_radius,
            description="Y radius",
            style={"description_width": "58px"},
            layout=widgets.Layout(width="140px"),
        )
        self.radius_input = self.size_x_input

        self.apply_button = widgets.Button(
            description="Apply",
            icon="check",
            button_style="primary",
            tooltip="Commit the preview mask to BraggPeaksPolymer.scan_mask",
            layout=widgets.Layout(width="72px"),
        )
        self.save_button = widgets.Button(
            description="Apply & Save",
            icon="save",
            button_style="success",
            tooltip="Commit and save this mask for the next notebook run",
            disabled=self.state_path is None,
            layout=widgets.Layout(width="105px"),
        )
        self.center_button = widgets.Button(
            description="Center",
            icon="crosshairs",
            tooltip="Center the circle",
            layout=widgets.Layout(width="75px"),
        )
        self.full_button = widgets.Button(
            description="Full",
            icon="expand",
            tooltip="Include the entire scan",
            layout=widgets.Layout(width="70px"),
        )
        self.reset_button = widgets.Button(
            description="Reset",
            icon="undo",
            tooltip="Restore the loaded/default state",
            layout=widgets.Layout(width="72px"),
        )
        self.status = widgets.HTML(layout=widgets.Layout(width="500px"))
        self.output = widgets.Output(
            layout=widgets.Layout(
                width="438px",
                height="356px",
                max_width="438px",
                overflow="hidden",
            )
        )

        self.figure, self.ax = plt.subplots(figsize=(4.0, 3.25))
        finite = self.reference_image[np.isfinite(self.reference_image)]
        if finite.size:
            vmin, vmax = np.percentile(finite, [1.0, 99.0])
            if not vmax > vmin:
                vmin, vmax = float(np.min(finite)), float(np.max(finite) + 1.0)
        else:
            vmin, vmax = 0.0, 1.0
        self.image_artist = self.ax.imshow(
            self.reference_image,
            cmap="gray",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        self.mask_artist = self.ax.imshow(
            np.ma.masked_where(~self._preview_mask, self._preview_mask),
            cmap="Reds",
            origin="upper",
            alpha=self.overlay_alpha,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        self.boundary_artist = None
        self.circle_artist = None
        self._replace_boundary_artist()
        (self.center_artist,) = self.ax.plot(
            center_column,
            center_row,
            marker="+",
            color="#ff3030",
            markersize=self.crosshair_size,
            markeredgewidth=self.crosshair_width,
        )
        self.ax.set(
            title="Scan-mask editor",
            xlabel="X — scan column",
            ylabel="Y — scan row",
            xlim=(-0.5, columns - 0.5),
            ylim=(rows - 0.5, -0.5),
        )
        self.ax.title.set_fontsize(10)
        self.ax.xaxis.label.set_fontsize(9)
        self.ax.yaxis.label.set_fontsize(9)
        self.ax.tick_params(labelsize=8)
        self.figure.tight_layout()

        self.x_slider.observe(self._on_x_slider, names="value")
        self.y_slider.observe(self._on_y_slider, names="value")
        self.geometry_selector.observe(self._on_geometry, names="value")
        self.size_x_slider.observe(self._on_size_x_slider, names="value")
        self.size_y_slider.observe(self._on_size_y_slider, names="value")
        self.x_input.observe(self._on_x_input, names="value")
        self.y_input.observe(self._on_y_input, names="value")
        self.size_x_input.observe(self._on_size_x_input, names="value")
        self.size_y_input.observe(self._on_size_y_input, names="value")
        self.apply_button.on_click(lambda _: self.apply())
        self.save_button.on_click(lambda _: self.save())
        self.center_button.on_click(
            lambda _: self.set_mask(x=columns // 2, y=rows // 2)
        )
        self.full_button.on_click(lambda _: self._set_full_scan())
        self.reset_button.on_click(lambda _: self._restore_initial())

        self._initial_geometry = (
            center_column, center_row, geometry, size_x, size_y
        )
        self._initial_mask = self._preview_mask.copy()
        # A loaded/default mask is immediately usable by Run All. Slider edits
        # remain previews until Apply, preventing repeated inference-cache invalidation.
        self.analysis.scan_mask = self._preview_mask.copy()
        self._render()
        self._refresh_status("Loaded saved mask" if self._loaded else "Default mask applied")

        position_row = widgets.HBox(
            [self.geometry_selector, self.x_input, self.y_input],
            layout=widgets.Layout(width="500px"),
        )
        self.size_input_row = widgets.HBox(
            [self.size_x_input, self.size_y_input],
            layout=widgets.Layout(width="500px"),
        )
        toolbar = widgets.HBox(
            [
                self.apply_button,
                self.save_button,
                self.center_button,
                self.full_button,
                self.reset_button,
            ],
            layout=widgets.Layout(flex_flow="row wrap"),
        )
        plot_row = widgets.HBox(
            [self.y_slider, self.output],
            layout=widgets.Layout(
                align_items="center", width="500px", overflow="hidden"
            ),
        )
        self.size_x_row = widgets.HBox(
            [
                widgets.Box(layout=widgets.Layout(width="52px")),
                self.size_x_slider,
            ],
            layout=widgets.Layout(align_items="center", width="500px"),
        )
        self.size_y_row = widgets.HBox(
            [
                widgets.Box(layout=widgets.Layout(width="52px")),
                self.size_y_slider,
            ],
            layout=widgets.Layout(align_items="center", width="500px"),
        )
        x_row = widgets.HBox(
            [
                widgets.Box(layout=widgets.Layout(width="52px")),
                self.x_slider,
            ],
            layout=widgets.Layout(align_items="center", width="500px"),
        )
        self.widget = widgets.VBox(
            [
                toolbar,
                position_row,
                self.size_input_row,
                self.size_x_row,
                self.size_y_row,
                plot_row,
                x_row,
                self.status,
            ],
            layout=widgets.Layout(width="500px", max_width="500px"),
        )
        self._refresh_geometry_controls()
        # Prevent the inline Matplotlib backend from appending a second copy of
        # the figure after the widget cell. The explicitly displayed Output copy
        # remains live and continues to update.
        plt.close(self.figure)
        if display_widget:
            display(self.widget)

    @property
    def x(self):
        """Horizontal scan-column coordinate."""
        return int(self.x_slider.value)

    @property
    def y(self):
        """Vertical scan-row coordinate."""
        return int(self.scan_shape[0] - 1 - self.y_slider.value)

    @property
    def radius(self):
        """Circle radius / square half-width compatibility value."""
        return self.size_x

    @property
    def geometry(self):
        return str(self.geometry_selector.value)

    @property
    def size_x(self):
        """Horizontal radius or half-width in scan pixels."""
        return int(self.size_x_slider.value)

    @property
    def size_y(self):
        """Vertical radius or half-height in scan pixels."""
        if self.geometry in {"circle", "square"}:
            return self.size_x
        return int(self.size_y_slider.value)

    @property
    def mask(self):
        """Current preview mask."""
        return self._preview_mask.copy()

    @property
    def applied_mask(self):
        return None if self.analysis.scan_mask is None else self.analysis.scan_mask.copy()

    def _effective_mask(self):
        """Return the committed mask for ndarray-style compatibility."""
        mask = self.analysis.scan_mask
        return self._preview_mask if mask is None else np.asarray(mask, dtype=bool)

    @property
    def shape(self):
        return self.scan_shape

    @property
    def dtype(self):
        return np.dtype(bool)

    @property
    def size(self):
        return int(np.prod(self.scan_shape))

    @property
    def ndim(self):
        return 2

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self._effective_mask(), dtype=dtype)
        if copy:
            array = array.copy()
        return array

    def __len__(self):
        return self.scan_shape[0]

    def sum(self, *args, **kwargs):
        """NumPy-compatible sum for legacy ``mask_arr = editor`` cells."""
        return self._effective_mask().sum(*args, **kwargs)

    def astype(self, *args, **kwargs):
        return self._effective_mask().astype(*args, **kwargs)

    def copy(self):
        return self._effective_mask().copy()

    def __getitem__(self, key):
        if not isinstance(key, str):
            return self._effective_mask()[key]
        # Compatibility with the historical returned dictionary. Its x0/y0
        # names represented row/column respectively despite their labels.
        values = {
            "mask": self.mask,
            "x0": self.y,
            "y0": self.x,
            "r": self.radius,
            "center_row": self.y,
            "center_column": self.x,
            "geometry": self.geometry,
            "size_x": self.size_x,
            "size_y": self.size_y,
        }
        return values[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def _validate_geometry(self, row, column, geometry, size_x, size_y):
        rows, columns = self.scan_shape
        if not 0 <= row < rows or not 0 <= column < columns:
            raise ValueError(
                f"Mask center (row={row}, column={column}) is outside scan shape "
                f"{self.scan_shape}."
            )
        if geometry not in self.GEOMETRIES:
            raise ValueError(
                f"Unknown mask geometry {geometry!r}; choose one of "
                f"{', '.join(self.GEOMETRIES)}."
            )
        if size_x < 1 or size_y < 1:
            raise ValueError("Mask half-sizes must be at least one scan pixel.")
        maximum_radius = int(np.ceil(np.hypot(rows - 1, columns - 1))) + 1
        if size_x > maximum_radius or size_y > maximum_radius:
            raise ValueError(
                f"Mask half-size ({size_x}, {size_y}) exceeds the supported maximum "
                f"{maximum_radius} for scan shape {self.scan_shape}."
            )

    def _geometry_mask(self, row, column, geometry, size_x, size_y):
        yy, xx = np.ogrid[: self.scan_shape[0], : self.scan_shape[1]]
        dy = yy - row
        dx = xx - column
        if geometry == "circle":
            return dy**2 + dx**2 <= size_x**2
        if geometry == "ellipse":
            return (dx / size_x) ** 2 + (dy / size_y) ** 2 <= 1
        if geometry == "square":
            return (np.abs(dx) <= size_x) & (np.abs(dy) <= size_x)
        if geometry == "rectangle":
            return (np.abs(dx) <= size_x) & (np.abs(dy) <= size_y)
        raise ValueError(f"Unknown mask geometry {geometry!r}.")

    def _resolve_reference_image(self, reference_image):
        if reference_image is None:
            virtual_images = getattr(self.analysis.dataset_cartesian, "virtual_images", {})
            if "virtual_image" in virtual_images:
                reference_image = virtual_images["virtual_image"]
                reference_image = getattr(
                    reference_image,
                    "array",
                    getattr(reference_image, "data", reference_image),
                )
            else:
                dataset = self.analysis.dataset_cartesian
                array = getattr(dataset, "array", None)
                if array is not None:
                    reference_image = np.asarray(array).mean(axis=(-2, -1))
                else:
                    reference_image = (
                        dataset.tensor.float().mean(dim=(-2, -1)).detach().cpu().numpy()
                    )
        reference_image = np.asarray(reference_image, dtype=float)
        if reference_image.shape != self.scan_shape:
            raise ValueError(
                f"reference_image shape {reference_image.shape} must match "
                f"scan shape {self.scan_shape}."
            )
        return reference_image

    def _read_state(self, path):
        try:
            with np.load(path, allow_pickle=False) as state:
                version = int(state["schema_version"])
                shape = tuple(int(v) for v in state["scan_shape"])
                if version not in {1, self.SCHEMA_VERSION}:
                    raise ValueError(
                        f"Unsupported scan-mask schema {version}; expected "
                        f"1 or {self.SCHEMA_VERSION}."
                    )
                if shape != self.scan_shape:
                    raise ValueError(
                        f"Saved scan-mask shape {shape} does not match current "
                        f"scan shape {self.scan_shape}."
                    )
                mask = np.asarray(state["mask"], dtype=bool)
                if mask.shape != self.scan_shape:
                    raise ValueError(
                        f"Saved mask array shape {mask.shape} does not match "
                        f"scan shape {self.scan_shape}."
                    )
                if version == 1:
                    geometry = "circle"
                    size_x = size_y = int(state["radius"])
                else:
                    geometry = str(state["geometry"].item())
                    size_x = int(state["size_x"])
                    size_y = int(state["size_y"])
                return {
                    "center_row": int(state["center_row"]),
                    "center_column": int(state["center_column"]),
                    "geometry": geometry,
                    "size_x": size_x,
                    "size_y": size_y,
                    "mask": mask,
                }
        except (OSError, KeyError) as exc:
            raise ValueError(f"Could not load scan-mask state from {path}: {exc}") from exc

    def set_mask(
        self,
        *,
        x=None,
        y=None,
        geometry=None,
        size_x=None,
        size_y=None,
    ):
        self._syncing = True
        try:
            if geometry is not None:
                geometry = str(geometry).lower()
                if geometry not in self.GEOMETRIES:
                    raise ValueError(
                        f"Unknown mask geometry {geometry!r}; choose one of "
                        f"{', '.join(self.GEOMETRIES)}."
                    )
                self.geometry_selector.value = geometry
            if x is not None:
                self.x_slider.value = int(x)
                self.x_input.value = int(x)
            if y is not None:
                self.y_slider.value = self.scan_shape[0] - 1 - int(y)
                self.y_input.value = int(y)
            if size_x is not None:
                self.size_x_slider.value = int(size_x)
                self.size_x_input.value = int(size_x)
            if size_y is not None:
                self.size_y_slider.value = int(size_y)
                self.size_y_input.value = int(size_y)
            if self.geometry in {"circle", "square"}:
                self.size_y_slider.value = self.size_x
                self.size_y_input.value = self.size_x
        finally:
            self._syncing = False
        self._refresh_geometry_controls()
        self._update_preview()
        return self

    def set_circle(self, *, x=None, y=None, radius=None):
        """Compatibility helper that explicitly selects circular geometry."""
        return self.set_mask(
            x=x,
            y=y,
            geometry="circle",
            size_x=radius,
            size_y=radius,
        )

    def apply(self):
        self.analysis.scan_mask = self._preview_mask.copy()
        self._dirty = False
        self._refresh_status("Mask applied")
        if self.autosave and self.state_path is not None:
            self.save(apply_first=False)
        return self.applied_mask

    def save(self, path=None, *, apply_first=True):
        path = self.state_path if path is None else Path(path)
        if path is None:
            raise ValueError("No scan-mask state_path was configured.")
        if apply_first:
            self.apply()
        path.parent.mkdir(parents=True, exist_ok=True)
        sampling = np.asarray(self.analysis.dataset_cartesian.sampling[:2], dtype=float)
        units = np.asarray(
            [str(value) for value in self.analysis.dataset_cartesian.units[:2]],
            dtype="U32",
        )
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", dir=path.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(
                stream,
                schema_version=np.asarray(self.SCHEMA_VERSION, dtype=np.int64),
                mask=self._preview_mask.astype(bool),
                mask_type=np.asarray(self.geometry),
                geometry=np.asarray(self.geometry),
                center_row=np.asarray(self.y, dtype=np.int64),
                center_column=np.asarray(self.x, dtype=np.int64),
                radius=np.asarray(self.radius, dtype=np.int64),
                size_x=np.asarray(self.size_x, dtype=np.int64),
                size_y=np.asarray(self.size_y, dtype=np.int64),
                scan_shape=np.asarray(self.scan_shape, dtype=np.int64),
                sampling=sampling,
                units=units,
            )
        temporary_path.replace(path)
        self.state_path = path
        self.save_button.disabled = False
        self._saved = True
        self._dirty = False
        self._refresh_status(f"Applied and saved to {path}")
        return path

    def close(self):
        plt.close(self.figure)

    def _restore_initial(self):
        x, y, geometry, size_x, size_y = self._initial_geometry
        self.set_mask(
            x=x,
            y=y,
            geometry=geometry,
            size_x=size_x,
            size_y=size_y,
        )
        self._preview_mask = self._initial_mask.copy()
        self._dirty = True
        self._render()
        self._refresh_status("Initial state restored; click Apply")

    def _set_full_scan(self):
        rows, columns = self.scan_shape
        if self.geometry == "circle":
            size_x = size_y = (
                int(np.ceil(np.hypot(rows - 1, columns - 1))) + 1
            )
        elif self.geometry == "ellipse":
            size_x, size_y = columns, rows
        elif self.geometry == "square":
            size_x = size_y = max(rows, columns)
        else:
            size_x, size_y = columns, rows
        self.set_mask(
            x=columns // 2,
            y=rows // 2,
            size_x=size_x,
            size_y=size_y,
        )

    def _on_x_slider(self, change):
        if self._syncing:
            return
        self._syncing = True
        self.x_input.value = int(change["new"])
        self._syncing = False
        self._update_preview()

    def _on_y_slider(self, change):
        if self._syncing:
            return
        self._syncing = True
        self.y_input.value = self.scan_shape[0] - 1 - int(change["new"])
        self._syncing = False
        self._update_preview()

    def _on_geometry(self, change):
        if self._syncing:
            return
        self._syncing = True
        try:
            if change["new"] in {"circle", "square"}:
                self.size_y_slider.value = self.size_x
                self.size_y_input.value = self.size_x
        finally:
            self._syncing = False
        self._refresh_geometry_controls()
        self._update_preview()

    def _on_size_x_slider(self, change):
        if self._syncing:
            return
        self._syncing = True
        try:
            self.size_x_input.value = int(change["new"])
            if self.geometry in {"circle", "square"}:
                self.size_y_slider.value = int(change["new"])
                self.size_y_input.value = int(change["new"])
        finally:
            self._syncing = False
        self._update_preview()

    def _on_size_y_slider(self, change):
        if self._syncing:
            return
        self._syncing = True
        self.size_y_input.value = int(change["new"])
        self._syncing = False
        self._update_preview()

    def _on_x_input(self, change):
        if self._syncing:
            return
        self._syncing = True
        self.x_slider.value = int(change["new"])
        self._syncing = False
        self._update_preview()

    def _on_y_input(self, change):
        if self._syncing:
            return
        self._syncing = True
        self.y_slider.value = self.scan_shape[0] - 1 - int(change["new"])
        self._syncing = False
        self._update_preview()

    def _on_size_x_input(self, change):
        if self._syncing:
            return
        self._syncing = True
        try:
            self.size_x_slider.value = int(change["new"])
            if self.geometry in {"circle", "square"}:
                self.size_y_slider.value = int(change["new"])
                self.size_y_input.value = int(change["new"])
        finally:
            self._syncing = False
        self._update_preview()

    def _on_size_y_input(self, change):
        if self._syncing:
            return
        self._syncing = True
        self.size_y_slider.value = int(change["new"])
        self._syncing = False
        self._update_preview()

    def _update_preview(self):
        self._preview_mask = self._geometry_mask(
            self.y, self.x, self.geometry, self.size_x, self.size_y
        )
        self._dirty = True
        self._saved = False
        self._render()
        self._refresh_status("Preview changed; click Apply or Apply & Save")

    def _refresh_geometry_controls(self):
        labels = {
            "circle": ("Radius", None),
            "ellipse": ("X radius", "Y radius"),
            "square": ("Half-size", None),
            "rectangle": ("Half-width", "Half-height"),
        }
        x_label, y_label = labels[self.geometry]
        self.size_x_slider.description = x_label
        self.size_x_input.description = x_label
        if y_label is None:
            self.size_y_slider.layout.display = "none"
            self.size_y_input.layout.display = "none"
            self.size_y_row.layout.display = "none"
        else:
            self.size_y_slider.description = y_label
            self.size_y_input.description = y_label
            self.size_y_slider.layout.display = ""
            self.size_y_input.layout.display = ""
            self.size_y_row.layout.display = ""

    def _replace_boundary_artist(self):
        if self.boundary_artist is not None:
            self.boundary_artist.remove()
        style = {
            "fill": False,
            "edgecolor": "#ff3030",
            "linewidth": 1.05,
            "linestyle": (0, (1.2, 5.5)),
            "alpha": 0.9,
        }
        if self.geometry in {"circle", "ellipse"}:
            size_y = self.size_x if self.geometry == "circle" else self.size_y
            artist = Ellipse(
                (self.x, self.y),
                width=2 * self.size_x,
                height=2 * size_y,
                **style,
            )
        else:
            size_y = self.size_x if self.geometry == "square" else self.size_y
            artist = Rectangle(
                (self.x - self.size_x - 0.5, self.y - size_y - 0.5),
                width=2 * self.size_x + 1,
                height=2 * size_y + 1,
                **style,
            )
        self.ax.add_patch(artist)
        self.boundary_artist = artist
        # Compatibility name retained even when the selected geometry is not circular.
        self.circle_artist = artist

    def _render(self):
        self.mask_artist.set_data(
            np.ma.masked_where(~self._preview_mask, self._preview_mask)
        )
        self._replace_boundary_artist()
        self.center_artist.set_data([self.x], [self.y])
        self.figure.canvas.draw_idle()
        with self.output:
            clear_output(wait=True)
            display(self.figure)

    def _refresh_status(self, message):
        count = int(self._preview_mask.sum())
        total = int(self._preview_mask.size)
        physical = ""
        try:
            row_sampling, column_sampling = (
                float(v) for v in self.analysis.dataset_cartesian.sampling[:2]
            )
            row_unit, column_unit = (
                str(v) for v in self.analysis.dataset_cartesian.units[:2]
            )
            if np.isclose(row_sampling, column_sampling) and row_unit == column_unit:
                if self.geometry == "circle":
                    physical = (
                        f" · radius ≈ {self.size_x * row_sampling:.4g} {row_unit}"
                    )
                else:
                    physical = (
                        f" · half-size ≈ "
                        f"{self.size_x * column_sampling:.4g} × "
                        f"{self.size_y * row_sampling:.4g} {row_unit}"
                    )
        except (TypeError, ValueError):
            pass
        save_state = "saved" if self._saved else "not saved"
        self.status.value = (
            f"<b>{message}</b><br>"
            f"{self.geometry.title()} · X column {self.x} · Y row {self.y} · "
            f"half-size {self.size_x} × {self.size_y} px"
            f"{physical} · {count:,}/{total:,} positions "
            f"({100.0 * count / total:.1f}%) · {save_state}"
        )


# TODO: Likely dataset4dSTEM rather than dataset4d input class
# Bragg peaks from crystalline vs polymer
# 
# TODO: "BraggPeaksPolymer" vs "BraggPeaksCrystal"
class BraggPeaksPolymer(AutoSerialize):
    """
    
    """

    _token = object()

    def __init__(
        self,
        dataset_cartesian: Dataset4dstem,
        compute_parameters: callable = None,
        normalize_data: callable = None,
        normalization_strategy: NormalizationStrategy | str | dict | None = None,
        model: MultiChannelCNN2d = None,
        final_shape: Tuple[int, int] = (256, 256),
        device: str = 'cpu',
        normalize_parameter_lower_percentile: float = 1.0,
        normalize_parameter_upper_percentile: float = 99.0,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use BraggPeaks.from_data() or .from_file() to instantiate this class."
            )

        self._dataset_cartesian = dataset_cartesian
        self._device = device
        self._final_shape = final_shape
        self.normalize_parameter_lower_percentile = normalize_parameter_lower_percentile
        self.normalize_parameter_upper_percentile = normalize_parameter_upper_percentile
        if (compute_parameters is None) != (normalize_data is None):
            raise ValueError(
                "compute_parameters and normalize_data must be supplied together."
            )
        if normalization_strategy is not None and compute_parameters is not None:
            raise ValueError(
                "Pass normalization_strategy or the legacy callback pair, not both."
            )
        if compute_parameters is not None:
            warnings.warn(
                "compute_parameters and normalize_data are deprecated; pass a "
                "normalization_strategy instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            normalization_strategy = LegacyNormalizationAdapter(
                compute_parameters,
                normalize_data,
                normalize_parameter_lower_percentile,
                normalize_parameter_upper_percentile,
            )
        self.compute_parameters = compute_parameters
        self.normalize_data = normalize_data
        self._normalization_strategy = (
            resolve_normalization_strategy(normalization_strategy)
            if normalization_strategy is not None
            else None
        )
        self._normalization_is_explicit = normalization_strategy is not None
        # To be set by class methods
        # self.resized_cartesian_data = None
        self.peak_coordinates_cartesian = None
        self.peak_intensities = None
        self.image_centers = None
        # Calibration parameters cached by preprocess() (lazy: applied downstream by
        # the polar transforms, the raw 4D data is left untouched).
        self.ellipse_params = None          # (a, b, theta_deg)
        self.ellipse_center = None          # (row, col) of the mean-DP ellipse fit
        self.descan_origin = None           # (2, Ry, Rx) plane-fitted CoM background
        self.origin_com_measured = None     # (2, Ry, Rx) raw per-pattern CoM
        self.detector_rotation_deg = None   # r->q rotation (clockwise, degrees)
        self.detector_transpose = None      # detector transpose flag
        self.sampling_inv_A = None          # detector-pixel sampling in 1/A
        self.polar_data = None
        self.polar_peaks = None
        self.max_radius = None
        self.num_radial_bins = None
        self.num_annular_bins = None
        self.orient_corr = None
        self.orient_corr_pairs = None
        # Cached dataset-level normalization stats (median, iqr). Computed once by
        # find_peaks_model / ensure_normalization_params and reused for live inference
        # so single-DP predictions reproduce the full-scan results exactly.
        self._normalization_parameters = None
        # Deprecated cache aliases retained for serialized historical objects.
        self._norm_median = None
        self._norm_iqr = None
        # True once BatchNorm running stats have been adapted to this dataset (for
        # eval-mode single-DP inference); see adapt_batchnorm / infer_peaks_single.
        self._bn_adapted = False
        # Set when an angular detector calibration must be converted to reciprocal
        # length. None means that the documented 300 kV default has not yet been
        # accepted or overridden by the user.
        self._accelerating_voltage_kv = None
        # Cache of the most recent train-mode chunk output for live inference
        # (bn_mode="train_batch"): (chunk_start, chunk_size, outs). Lets neighbouring
        # cursor positions in the same find_peaks_model chunk reuse one forward pass.
        self._live_chunk_cache = None
        # Scan mask (region of interest) remembered from find_peaks_model / process_polar,
        # so normalization + BN adaptation restrict to the sample ROI (see scan_mask).
        self._scan_mask = None

        if model is None:
            # Setup model
            input_channels = 1  # 1 for a greyscale image, 3 for RGB, 4 for RGBA, etc.
            k_size = 3
            # k_size = 7
            num_layers = 4
            start_filters = 32
            num_per_layer = 3
            # num_per_layer = 2
            use_skip_connections = True
            dtype = torch.float32
            # The immutable paper checkpoint was trained with dropout disabled.
            dropout = 0.0
            model = MultiChannelCNN2d(
                in_channels=input_channels,
                out_channels=2,
                start_filters=start_filters,
                num_layers=num_layers,
                num_per_layer=num_per_layer,
                use_skip_connections=use_skip_connections,
                dtype=dtype,
                dropout=dropout,
                final_activations=["sigmoid", "sigmoid"],
                conv_kernel_size=k_size,
            )
        self._model = model

    @property
    def model(self) -> MultiChannelCNN2d:
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model
        self._invalidate_inference_caches()

    @property
    def normalization_strategy(self):
        return self._normalization_strategy

    @normalization_strategy.setter
    def normalization_strategy(self, strategy):
        self._set_normalization_strategy(strategy, explicit=True)

    def _set_normalization_strategy(self, strategy, *, explicit):
        resolved = (
            resolve_normalization_strategy(strategy) if strategy is not None else None
        )
        if resolved != getattr(self, "_normalization_strategy", None):
            self._normalization_strategy = resolved
            self._invalidate_inference_caches()
        self._normalization_is_explicit = explicit

    def _invalidate_inference_caches(self):
        self._normalization_parameters = None
        self._norm_median = None
        self._norm_iqr = None
        self._bn_adapted = False
        self._live_chunk_cache = None

    def _require_normalization_strategy(self):
        if self._normalization_strategy is None:
            raise RuntimeError(
                "No inference normalization is configured. Load a registered model, "
                "or pass normalization_strategy (or the legacy compute_parameters and "
                "normalize_data callbacks) when using a custom checkpoint."
            )
        return self._normalization_strategy

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @property
    def dataset_cartesian(self) -> Dataset4dstem:
        return self._dataset_cartesian

    @dataset_cartesian.setter
    def dataset_cartesian(self, dataset_cartesian):
        self._dataset_cartesian = dataset_cartesian

    @property
    def final_shape(self) -> str:
        return self._final_shape

    @final_shape.setter
    def final_shape(self, final_shape):
        self._final_shape = final_shape

    @property
    def scan_mask(self):
        """Boolean (Ry, Rx) region-of-interest mask, or None for the whole scan.

        Remembered from ``find_peaks_model`` (and settable directly) so that
        ``ensure_normalization_params`` / ``adapt_batchnorm`` estimate their statistics
        from the sample ROI rather than off-sample regions (vacuum, edges, beam stop).
        """
        return self._scan_mask

    @scan_mask.setter
    def scan_mask(self, mask):
        if mask is None:
            new_mask = None
        else:
            new_mask = np.asarray(mask, dtype=bool)
            Ry, Rx = int(self._dataset_cartesian.shape[0]), int(self._dataset_cartesian.shape[1])
            if new_mask.shape != (Ry, Rx):
                raise ValueError(
                    f"scan_mask shape {new_mask.shape} must match scan shape ({Ry}, {Rx})"
                )
        # Only invalidate the lazily-cached stats if the mask actually changed, so
        # re-running find_peaks_model with the same mask doesn't needlessly recompute.
        changed = not (
            (self._scan_mask is None and new_mask is None)
            or (
                self._scan_mask is not None
                and new_mask is not None
                and np.array_equal(self._scan_mask, new_mask)
            )
        )
        self._scan_mask = new_mask
        if changed:
            self._invalidate_inference_caches()

    @classmethod
    def from_file(
        cls,
        file_path: str,
        device: str = "cpu",
        compute_parameters: callable = None,
        normalize_data: callable = None,
        normalization_strategy: NormalizationStrategy | str | dict | None = None,
        file_type: str | None = None,
        normalize_parameter_lower_percentile: float = 1.0,
        normalize_parameter_upper_percentile: float = 99.0,
    ) -> "BraggPeaksPolymer":
        dataset_cartesian = Dataset4dstem.from_file(file_path, file_type=file_type)
        return cls.from_data(
            dataset_cartesian=dataset_cartesian,
            device=device,
            compute_parameters=compute_parameters,
            normalize_data=normalize_data,
            normalization_strategy=normalization_strategy,
            normalize_parameter_lower_percentile=normalize_parameter_lower_percentile,
            normalize_parameter_upper_percentile=normalize_parameter_upper_percentile,
        )

    @classmethod
    def from_data(
        cls,
        dataset_cartesian: Dataset4dstem,
        device: str = "cpu",
        compute_parameters: callable = None,
        normalize_data: callable = None,
        normalization_strategy: NormalizationStrategy | str | dict | None = None,
        normalize_parameter_lower_percentile: float = 1.0,
        normalize_parameter_upper_percentile: float = 99.0,
    ) -> "BraggPeaksPolymer":
        return cls(
            dataset_cartesian=dataset_cartesian,
            _token=cls._token,
            device=device,
            compute_parameters=compute_parameters,
            normalize_data=normalize_data,
            normalization_strategy=normalization_strategy,
            normalize_parameter_lower_percentile=normalize_parameter_lower_percentile,
            normalize_parameter_upper_percentile=normalize_parameter_upper_percentile,
        )

    def pixels_to_inv_A(self, accelerating_voltage_kv: float = None):
        """Return the detector-pixel sampling in inverse angstroms.

        Angular calibrations in mrad require the electron wavelength. If no voltage
        has previously been supplied, 300 kV is assumed with an explicit warning.
        Supplying a voltage stores it for subsequent reciprocal-space operations.
        """
        unit = str(self.dataset_cartesian.units[2]).strip().lower()
        sampling = self.dataset_cartesian.sampling[2]

        if unit == "mrad":
            if accelerating_voltage_kv is None:
                accelerating_voltage_kv = self._accelerating_voltage_kv
            if accelerating_voltage_kv is None:
                accelerating_voltage_kv = 300.0
                warnings.warn(
                    "Detector calibration is in mrad; assuming an accelerating "
                    "voltage of 300 kV for conversion to 1/Å. Pass "
                    "accelerating_voltage_kv to find_peaks_model() to override it.",
                    UserWarning,
                    stacklevel=2,
                )
            if not np.isfinite(accelerating_voltage_kv) or accelerating_voltage_kv <= 0:
                raise ValueError("accelerating_voltage_kv must be a positive finite value")

            self._accelerating_voltage_kv = float(accelerating_voltage_kv)
            wavelength_angstrom = electron_wavelength_angstrom(
                self._accelerating_voltage_kv * 1e3
            )
            return sampling / (1e3 * wavelength_angstrom)

        _, sampling_angstrom_conversion_factor = parse_reciprocal_units(
            self.dataset_cartesian.units[2]
        )
        return sampling * sampling_angstrom_conversion_factor
    
    def preprocess(
        self,
        accelerating_voltage_kv: float | None = None,
        *,
        center_source: str = "descent",
        fit_ellipse: bool = True,
        ellipse_fit_method: str = "angular_variance",
        ellipse_threshold: float | None = None,
        ellipse_radial_min: float | None = None,
        ellipse_radial_max: float | None = None,
        ellipse_device: str | None = None,
        estimate_descan: bool = True,
        descan_fit_method: str = "plane",
        estimate_detector_rotation: bool = True,
        scan_mask: ArrayLike = None,
        center_device: str | None = None,
        com_device: str | None = None,
        com_batch_size: int | None = None,
        store_metadata: bool = True,
        show: bool = False,
        verbose: bool = True,
    ):
        """Calibrate the 4D-STEM scan (centers, ellipticity, descan, detector rotation).

        This mirrors the lazy design of the rest of the class: it *measures and caches*
        calibration parameters rather than re-warping the raw diffraction data (the ML
        peak-finder runs on raw patterns; centers/ellipticity are applied downstream by
        ``process_polar`` / the polar transforms). After ``preprocess`` you can call
        ``process_polar(center_ellipse_params=bp.ellipse_params)`` and the cached
        ``image_centers`` will be reused.

        Steps performed (each individually toggleable). Order matters: centering runs
        first so the ellipse is measured on an already-centered mean DP -- fitting the
        ellipse on the raw mean DP smears the ring by the descan drift and biases the
        fit toward the central beam.

        1. **Descan / detector rotation** (``estimate_descan`` /
           ``estimate_detector_rotation``) -- a ``CenterOfMassOriginModel`` measures the
           per-pattern centre of mass, fits a smooth background across scan positions
           (``descan_fit_method``), and estimates the r->q detector rotation + transpose.
           Results are cached on ``self.descan_origin`` (2, Ry, Rx),
           ``self.detector_rotation_deg``, ``self.detector_transpose`` and (optionally)
           ``dataset_cartesian.metadata["r_to_q_rotation_cw_deg"]``.
        2. **Image centers** -- ``self.image_centers`` (2, Ry, Rx), the per-pattern
           origins consumed by the polar transforms. ``center_source`` selects the
           estimator: ``"descent"`` / ``"grid"`` / ``"peaks"`` use
           ``find_central_beams_4d`` (angular-uniformity, the pipeline default),
           ``"com"`` uses the raw centre of mass, ``"descan"`` uses the plane-fitted
           (descanned) origin field.
        3. **Ellipticity** (``fit_ellipse``), fit LAST -- a diffuse-ring fit on a
           *centered* mean DP (each pattern shifted so its central beam sits at the
           detector center, then averaged; see ``_centered_dp_mean``). The
           ``"angular_variance"`` method searches ``(b/a, theta)`` to minimise the
           annulus' azimuthal variance. The ``"ridge"`` method extracts the ring radius
           independently at each azimuth, uses its first two harmonics to initialise the
           center and ellipse, and accepts a robust joint refinement only when it
           improves held-out angular sectors over a circle. Both methods ignore the
           central beam itself. The centered mean DP is cached on
           ``self.dp_mean_centered``; ``ellipse_radial_min`` /
           ``ellipse_radial_max`` bound the ring band (auto-detected from the radial
           profile when None). Stored as
           ``self.ellipse_params = (a, b, theta_deg)`` and (optionally) into
           ``dataset_cartesian.metadata["ellipticity"]``. ``ellipse_threshold`` is kept
           for backward compatibility but is unused by the ring fit.
        4. **Reciprocal sampling** -- caches ``self.sampling_inv_A`` via
           ``pixels_to_inv_A`` (accepts ``accelerating_voltage_kv`` for mrad detectors).

        Parameters
        ----------
        accelerating_voltage_kv : float, optional
            Beam voltage for mrad->1/A conversion (see ``pixels_to_inv_A``).
        center_source : {"descent", "grid", "peaks", "com", "descan"}
            Estimator backing ``self.image_centers``. Default "descent".
        fit_ellipse : bool
            Fit ellipticity from the mean DP. Default True.
        ellipse_fit_method : {"angular_variance", "ridge"}
            Angular-variance search or robust diffuse-ring ridge refinement.
            Default ``"angular_variance"`` during ridge-method validation.
        ellipse_threshold : float, optional
            Binarisation threshold for ``fit_probe_ellipse`` (Otsu if None).
        estimate_descan : bool
            Run the CoM + background-fit descan estimate. Default True.
        descan_fit_method : {"plane", "constant"}
            Background model for the descan fit. Default "plane".
        estimate_detector_rotation : bool
            Estimate the r->q detector rotation + transpose (requires the CoM model,
            so it forces ``estimate_descan``). Default True.
        scan_mask : ArrayLike, optional
            Boolean (Ry, Rx) ROI passed to ``find_central_beams_4d``.
        center_device, com_device : str, optional
            Device overrides for the angular-uniformity finder and the CoM model
            respectively (both default to ``self.device``). Note the CoM model loads
            the whole 4D tensor onto its device at once.
        com_batch_size : int, optional
            Batch size for the CoM origin calculation (whole scan if None).
        store_metadata : bool
            Write ellipticity / rotation into ``dataset_cartesian.metadata``. Default True.
        show : bool
            Show the ellipse-fit overlay. Default False.
        verbose : bool
            Print a short calibration summary. Default True.

        Returns
        -------
        dict
            The calibration parameters that were computed.
        """
        center_source = center_source.lower()
        valid_sources = ("descent", "grid", "peaks", "com", "descan")
        if center_source not in valid_sources:
            raise ValueError(f"center_source must be one of {valid_sources}, got {center_source!r}")
        ellipse_fit_method = str(ellipse_fit_method).lower()
        if ellipse_fit_method not in {"angular_variance", "ridge"}:
            raise ValueError(
                "ellipse_fit_method must be 'angular_variance' or 'ridge', "
                f"got {ellipse_fit_method!r}"
            )

        Ry, Rx, Qy, Qx = self._dataset_cartesian.shape
        need_com = (
            estimate_descan
            or estimate_detector_rotation
            or center_source in ("com", "descan")
        )

        results: dict = {}

        # 1. Descan (CoM + background fit) and detector rotation come FIRST: the
        #    per-pattern central-beam CoM and its smooth drift model are what let us
        #    build a properly centered mean DP for the ellipse fit in step 3.
        self.descan_origin = None
        self.origin_com_measured = None
        self.detector_rotation_deg = None
        self.detector_transpose = None
        com_model = None
        if need_com:
            from quantem.diffractive_imaging.origin_models import CenterOfMassOriginModel

            com_dev = com_device if com_device is not None else self.device
            com_model = CenterOfMassOriginModel.from_dataset(
                self._dataset_cartesian, device=com_dev
            )
            com_model.calculate_origin(max_batch_size=com_batch_size)
            measured = com_model.origin_measured.detach().cpu().numpy().reshape(Ry, Rx, 2)
            self.origin_com_measured = np.moveaxis(measured, -1, 0)  # (2, Ry, Rx)
            results["origin_com_measured"] = self.origin_com_measured

            # Restrict the descan/rotation fit to the ROI: out-of-mask patterns
            # (vacuum/substrate) have meaningless CoM that drags a global plane, leaving a
            # uniform residual inside the ROI. Fit the plane over in-mask patterns only
            # (sigma-clipped to reject hot/dead-pixel CoM outliers), push it back into the
            # CoM model, and zero the residual outside the ROI so the detector-rotation
            # curl isn't contaminated by junk patterns either.
            roi = (
                np.asarray(scan_mask, dtype=bool)
                if scan_mask is not None
                else np.ones((Ry, Rx), dtype=bool)
            )
            if estimate_descan or estimate_detector_rotation or center_source == "descan":
                import torch

                fitted = fit_origin_roi(measured, roi, fit_method=descan_fit_method)
                self.descan_origin = np.moveaxis(fitted, -1, 0)  # (2, Ry, Rx)
                results["descan_origin"] = self.descan_origin

                dev = com_model.device
                com_model.origin_fitted = torch.as_tensor(
                    fitted.reshape(-1, 2), dtype=torch.float, device=dev
                )
                meas_clean = measured.copy()
                meas_clean[~roi] = fitted[~roi]  # residual := 0 outside the ROI
                com_model.origin_measured = torch.as_tensor(
                    meas_clean.reshape(-1, 2), dtype=torch.float, device=dev
                )

            if estimate_detector_rotation:
                com_model.estimate_detector_rotation()
                self.detector_rotation_deg = float(com_model.detector_rotation_deg)
                self.detector_transpose = bool(com_model.detector_transpose)
                results["detector_rotation_deg"] = self.detector_rotation_deg
                results["detector_transpose"] = self.detector_transpose
                if store_metadata:
                    self._dataset_cartesian.metadata["r_to_q_rotation_cw_deg"] = (
                        self.detector_rotation_deg
                    )

        # 2. Per-pattern image centers consumed by the polar transforms. Centering
        #    runs BEFORE the ellipse fit (ellipse_params intentionally None here) so
        #    the ellipse is measured on an already-centered mean DP, not the reverse.
        if center_source in ("descent", "grid", "peaks"):
            self.image_centers = self.find_central_beams_4d(
                scan_mask=scan_mask,
                center_method=center_source,
                ellipse_params=None,
                center_device=center_device,
            )
        elif center_source == "com":
            self.image_centers = self.origin_com_measured.copy()
        else:  # "descan"
            self.image_centers = self.descan_origin.copy()
        results["image_centers"] = self.image_centers

        # 3. Ellipticity LAST, fit on a mean DP that has been centered so the central
        #    beam sits at the detector center and the diffraction ring is concentric.
        #    Both supported methods fit the diffuse ring rather than the probe blob;
        #    the ridge method may additionally remove a small residual center offset.
        self.ellipse_params = None
        self.ellipse_center = None
        self.dp_mean_centered = None
        self.ellipse_fit_diagnostics = None
        if fit_ellipse:
            self.dp_mean_centered = centered_dp_mean(
                self._dataset_cartesian, self.image_centers, com_model=com_model
            )
            Qy, Qx = self._dataset_cartesian.shape[-2:]
            center = ((Qy - 1) / 2.0, (Qx - 1) / 2.0)  # _centered_dp_mean puts the beam here
            fit_function = (
                fit_ellipse_from_ridge
                if ellipse_fit_method == "ridge"
                else fit_ellipse_from_ring
            )
            ellipse_fit = fit_function(
                self.dp_mean_centered,
                center,
                radial_min=ellipse_radial_min,
                radial_max=ellipse_radial_max,
                device=ellipse_device if ellipse_device is not None else "cpu",
                show=show,
                verbose=verbose,
            )
            ring_band = ellipse_fit.ring_band
            self.ellipse_params = ellipse_fit.params
            self.ellipse_fit_diagnostics = ellipse_fit.diagnostics
            self.ellipse_center = ellipse_fit.center_refined
            if ellipse_fit_method == "ridge" and ellipse_fit.accepted:
                center_delta = np.asarray(self.ellipse_center) - np.asarray(center)
                self.image_centers = np.asarray(
                    self.image_centers, dtype=float
                ).copy()
                valid_centers = (
                    (self.image_centers[0] != 0)
                    | (self.image_centers[1] != 0)
                )
                self.image_centers[0, valid_centers] += center_delta[0]
                self.image_centers[1, valid_centers] += center_delta[1]
                results["image_centers"] = self.image_centers
            self.ellipse_ring_band = ring_band
            results["ellipse_params"] = self.ellipse_params
            results["ellipse_center"] = self.ellipse_center
            results["ellipse_ring_band"] = ring_band
            results["ellipse_fit_diagnostics"] = self.ellipse_fit_diagnostics
            results["ellipse_fit_method"] = ellipse_fit_method
            if store_metadata:
                self._dataset_cartesian.metadata["ellipticity"] = self.ellipse_params

        # 4. Reciprocal-space sampling (pixels -> 1/A).
        try:
            self.sampling_inv_A = float(self.pixels_to_inv_A(accelerating_voltage_kv))
            results["sampling_inv_A"] = self.sampling_inv_A
        except Exception as exc:  # calibration/units may be unavailable
            self.sampling_inv_A = None
            if verbose:
                print(f"preprocess: reciprocal calibration skipped ({exc})")

        if verbose:
            print(f"preprocess: device={self.device}, scan=({Ry}, {Rx}), detector=({Qy}, {Qx})")
            print(f"  image_centers  <- {center_source}  shape {self.image_centers.shape}")
            if self.ellipse_params is not None:
                a, b, th = self.ellipse_params
                print(f"  ellipticity    a={a:.3f} b={b:.3f} theta={th:.2f} deg (a/b={a / b:.4f})")
            if self.descan_origin is not None:
                print(f"  descan         {descan_fit_method}-fit CoM background")
            if self.detector_rotation_deg is not None:
                print(
                    f"  r->q rotation  {self.detector_rotation_deg:.2f} deg "
                    f"(transpose={self.detector_transpose})"
                )
            if self.sampling_inv_A is not None:
                print(f"  sampling       {self.sampling_inv_A:.5g} 1/A per pixel")

        return results

    def resize_data(self, device:str = "cuda:0"):
        print(device)
        Ry, Rx, Qy, Qx = self._dataset_cartesian.shape
        scale_factor = (self._final_shape[0] * self._final_shape[1]) / (Qy * Qx)
        resized_data = np.zeros((Ry, Rx, self._final_shape[0], self._final_shape[1]))
        for i in tqdm(range(Ry), desc='rows'):
            inp = torch.tensor(self._dataset_cartesian[i].array, dtype=torch.float32).to(device)
            inp = torch.nn.functional.interpolate(inp[None, ...], size=self._final_shape, mode='bilinear', align_corners=False) * scale_factor
            resized_data[i, :, :, :] = inp.squeeze().detach().cpu().numpy()
        self.resized_cartesian_data = resized_data

    def resize_images(self, images, device: str = "cuda:0", initial_chunk_size: int = 100, show_progress=False):
        # Handle Dataset objects - extract array
        if hasattr(images, 'array'):
            images = images.array
        elif isinstance(images, Dataset3d):
            # If it's a Dataset3d, get the underlying array
            images = np.array([images[i].array for i in range(images.shape[0])])
        
        N, Qy, Qx = images.shape
        scale_factor = (self._final_shape[0] * self._final_shape[1]) / (Qy * Qx)
        resized_data = np.zeros((N, self._final_shape[0], self._final_shape[1]))
        
        chunk_size = initial_chunk_size
        i = 0
        
        with tqdm(total=N, desc='images', disable=not show_progress) as pbar:
            while i < N:
                try:
                    # Determine the end index for this chunk
                    end_idx = min(i + chunk_size, N)
                    chunk = images[i:end_idx]
                    
                    # Process chunk on GPU
                    inp = torch.tensor(chunk, dtype=torch.float32).to(device)
                    inp = torch.nn.functional.interpolate(
                        inp.unsqueeze(1),  # Add channel dimension
                        size=self._final_shape, 
                        mode='bilinear', 
                        align_corners=False
                    ) * scale_factor
                    
                    resized_data[i:end_idx, :, :] = inp.squeeze(1).detach().cpu().numpy()
                    
                    # Clear GPU cache
                    del inp
                    if 'cuda' in device:
                        torch.cuda.empty_cache()
                    
                    # Update progress and move to next chunk
                    pbar.update(end_idx - i)
                    i = end_idx
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        # Clear cache and reduce chunk size
                        if 'cuda' in device:
                            torch.cuda.empty_cache()
                        
                        chunk_size = max(1, chunk_size // 2)
                        print(f"\nGPU OOM! Reducing chunk size to {chunk_size}")
                        
                        if chunk_size == 1:
                            # If even single image fails, fall back to CPU
                            print("Falling back to CPU processing")
                            device = "cpu"
                    else:
                        raise e
        
        return resized_data

    def set_model_weights(
        self,
        path_to_weights: str = None,
        *,
        model_id: str = PAPER_MODEL_ID,
        version: str | None = None,
        latest: bool = False,
        local_model_dir: str | None = None,
        cache_dir: str | None = None,
    ) -> "BraggPeaksPolymer":
        """Load explicit weights or a checksum-verified named model.

        Explicit paths retain the historical behavior. Without a path, the
        immutable paper model is selected; ``latest=True`` is opt-in.
        """
        if path_to_weights is None:
            resolution = resolve_polymer_model(
                model_id=model_id,
                version=version,
                latest=latest,
                local_model_dir=local_model_dir,
                cache_dir=cache_dir,
            )
            self._model = build_polymer_model(resolution.specification)
            if not self._normalization_is_explicit:
                normalization_config = resolution.specification.get(
                    "experimental_normalization"
                )
                if normalization_config is None:
                    raise RuntimeError(
                        f"Registered model {resolution.model_id!r} does not declare "
                        "experimental_normalization."
                    )
                self._set_normalization_strategy(normalization_config, explicit=False)
            path_to_weights = str(resolution.weights_path)
            self.model_resolution = resolution
        self._model.load_state_dict(
            torch.load(path_to_weights, weights_only=True, map_location=self.device)
        )
        self._model.to(self.device)
        self._invalidate_inference_caches()
        return self

    def detect_ice(
        self,
        *,
        params=None,
        scan_mask=None,
        intensity_threshold_global=None,
        return_debug=False,
    ):
        """Detect ice peaks from this analysis's polar peaks and intensities."""

        from quantem.diffraction.polymer_ice import IceFlaggerParams, detect_ice

        if self.polar_peaks is None or self.peak_intensities is None:
            raise RuntimeError(
                "detect_ice() requires polar_peaks and peak_intensities to be computed first."
            )
        return detect_ice(
            self.polar_peaks,
            self.peak_intensities,
            params=IceFlaggerParams() if params is None else params,
            scan_mask=self.scan_mask if scan_mask is None else scan_mask,
            intensity_threshold_global=intensity_threshold_global,
            return_debug=return_debug,
            polar_data=getattr(self, "polar_data", None),
            # process_polar(two_fold_symmetry=True) folded theta to [0, 180).
            theta_period_deg=180.0 if getattr(self, "two_fold_symmetry", False) else 360.0,
        )

    def measure_ice_peak_widths(self, *, params=None, scan_mask=None, **kwargs):
        """Radial/annular widths of this analysis's peaks, for tuning the sharpness gate."""

        from quantem.diffraction.polymer_ice import IceFlaggerParams, collect_peak_widths

        if self.polar_peaks is None or self.peak_intensities is None or getattr(self, "polar_data", None) is None:
            raise RuntimeError(
                "measure_ice_peak_widths() requires polar_peaks, peak_intensities and polar_data."
            )
        return collect_peak_widths(
            self.polar_peaks,
            self.peak_intensities,
            self.polar_data,
            params=IceFlaggerParams() if params is None else params,
            scan_mask=self.scan_mask if scan_mask is None else scan_mask,
            **kwargs,
        )

    def plot_q_intensity_density(self, **kwargs):
        """Plot q/intensity density from this analysis's aligned peak vectors."""

        from quantem.diffraction.polymer_ice import plot_q_intensity_density

        if self.polar_peaks is None or self.peak_intensities is None:
            raise RuntimeError(
                "plot_q_intensity_density() requires polar_peaks and peak_intensities."
            )
        return plot_q_intensity_density(
            self.polar_peaks, self.peak_intensities, **kwargs
        )

    def _postprocess_single(self, position_map, intensity_map, sigma=1.0, threshold=0.25, show=False):
        """Process a single 2D image"""
        # Find peaks with subpixel-refinement
        peak_coords, peak_position_signal_intensities, refinement_success = detect_blobs(
            position_map,
            sigma=sigma,  # Sigma for Gaussian smoothing used in processing
            threshold=threshold,  # Threshold for strength of peak position signal to be valid peak
        )

        # If no peaks found, return empty lists
        if len(peak_coords) == 0:
            return np.array([]), np.array([])

        # map_coordinates expects coordinates in (row, col) = (y, x) order
        # peak_coords is already in [row, col] format from detect_blobs
        interpolated_intensities = map_coordinates(
            intensity_map, 
            peak_coords.T,  # Transpose to get [[all_y], [all_x]]
            order=1,  # 1 = bilinear interpolation
            mode='nearest'  # How to handle edges
        )
        
        # Optional: filter out peaks that were not successfully refined
        if np.any(refinement_success):
            pass
        
        if show:
            # Peak positions only
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(position_map, cmap='gray', alpha=0.8)
            ax.set_title("Input Position Map with Marked Peaks")
            ax.scatter(peak_coords[:, 1], peak_coords[:, 0], s=10, c='r', label="Peaks")
            ax.legend()
            plt.tight_layout()
            plt.show()

            # Peak positions with color representing intensity
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(position_map, cmap='gray', alpha=0.8)
            scatter = ax.scatter(
                peak_coords[:, 1],  # x coordinates
                peak_coords[:, 0],  # y coordinates
                c=interpolated_intensities,      # color by intensity
                s=10,
                cmap='turbo',    
                edgecolors='black', # white border for visibility
                linewidths=2,
                alpha=0.9,
                marker='o'
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Intensity', fontsize=12)
            ax.set_title('Peak Positions and Intensities', fontsize=14)
            ax.axis('off')
            plt.tight_layout()
            plt.show()

        return peak_coords, interpolated_intensities

    def ensure_normalization_params(
        self,
        device: str = None,
        n_normalize_samples: int = 1000,
        scan_mask: ArrayLike = None,
        recompute: bool = False,
    ):
        """Fit and cache the configured inference-normalization parameters.

        These are estimated once from a random sample of valid diffraction patterns and
        reused by both ``find_peaks_model`` (whole-scan) and ``infer_peaks_single``
        (live). Caching guarantees live single-DP inference reproduces the full-scan
        peaks exactly (same normalization). Parameters are intentionally opaque.
        """
        strategy = self._require_normalization_strategy()
        if not recompute and self._normalization_parameters is not None:
            return self._normalization_parameters

        device = device or self.device
        Ry, Rx, _, _ = self.dataset_cartesian.shape
        # Restrict to the stored ROI when no mask is passed explicitly (fall back to the
        # whole scan only if none is set); estimate stats from the sample region.
        if scan_mask is None:
            scan_mask = self._scan_mask
        if scan_mask is None:
            scan_mask = np.ones((Ry, Rx), dtype=bool)
        else:
            scan_mask = np.asarray(scan_mask, dtype=bool)
        valid_positions = np.argwhere(scan_mask)
        n_valid = len(valid_positions)

        n_normalize_samples = min(n_normalize_samples, n_valid)
        sample_indices = np.random.choice(n_valid, size=n_normalize_samples, replace=False)

        stats_patterns = np.array([
            self.dataset_cartesian[ry, rx].array
            for ry, rx in valid_positions[sample_indices]
        ])

        stats_patterns_resized = self.resize_images(stats_patterns, device=device)
        parameters = strategy.fit(stats_patterns_resized)
        self._normalization_parameters = parameters
        if isinstance(parameters, tuple) and len(parameters) == 2:
            self._norm_median, self._norm_iqr = parameters
        else:
            self._norm_median = self._norm_iqr = None
        return parameters

    def adapt_batchnorm(
        self,
        device: str = None,
        n_samples: int = 1000,
        scan_mask: ArrayLike = None,
        chunk_size: int = 100,
        recompute: bool = False,
    ):
        """Adapt the model's BatchNorm running statistics to THIS dataset, then eval.

        The model trains on synthetic data, so its stored BatchNorm running stats do not
        match the experimental scan; plain ``eval()`` inference then under-detects.
        ``find_peaks_model`` sidesteps this by running in train mode (per-chunk batch
        stats). For deterministic single-DP inference (``infer_peaks_single`` / the live
        widget), we instead estimate the running stats *once* from a representative sample
        of this dataset and freeze them: reset the BatchNorm buffers, run a sample through
        the model in train mode with ``momentum=None`` (so the buffers accumulate the
        cumulative mean/var over the sample), then switch to eval. Uses the same input
        normalization pipeline (resize + ``normalize_data`` with the cached median/iqr).
        Idempotent unless ``recompute=True``. Leaves the model in eval mode.
        """
        if self._bn_adapted and not recompute:
            return
        import torch.nn as nn

        device = device or self.device
        parameters = self.ensure_normalization_params(
            device=device, n_normalize_samples=max(n_samples, 1000), scan_mask=scan_mask
        )
        strategy = self._require_normalization_strategy()

        Ry, Rx, _, _ = self.dataset_cartesian.shape
        # Restrict the adaptation sample to the stored ROI when none is passed.
        if scan_mask is None:
            scan_mask = self._scan_mask
        if scan_mask is None:
            scan_mask = np.ones((Ry, Rx), dtype=bool)
        else:
            scan_mask = np.asarray(scan_mask, dtype=bool)
        valid_positions = np.argwhere(scan_mask)
        n_valid = len(valid_positions)
        n_samples = min(n_samples, n_valid)
        sample_indices = np.random.choice(n_valid, size=n_samples, replace=False)
        sample_positions = valid_positions[sample_indices]

        # Temporarily switch BatchNorm layers to cumulative-average mode so the running
        # buffers become the exact mean/var over the sample (not an EMA of the last batch).
        self.model.to(device)
        bn_layers = [m for m in self.model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        saved_momentum = [m.momentum for m in bn_layers]
        for m in bn_layers:
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average
        self.model.train()
        try:
            with torch.no_grad():
                for i in range(0, n_samples, chunk_size):
                    chunk = np.array([
                        self.dataset_cartesian[ry, rx].array
                        for ry, rx in sample_positions[i : i + chunk_size]
                    ])
                    resized = self.resize_images(chunk, device=device, initial_chunk_size=chunk_size)
                    ins = torch.tensor(resized, dtype=torch.float32).to(device)
                    ins_batch = strategy.transform(ins, parameters)[:, None, ...]
                    self.model(ins_batch)  # updates BN running stats only
        finally:
            for m, mom in zip(bn_layers, saved_momentum):
                m.momentum = mom
            self.model.eval()
        self._bn_adapted = True

    def prepare_inference(self, device: str = None, n_samples: int = 1000, scan_mask: ArrayLike = None):
        """Convenience: compute input-normalization stats + adapt BatchNorm in one call.

        Run after the model weights are loaded to ready the object for deterministic
        eval-mode single-DP inference (``infer_peaks_single``).
        """
        self.ensure_normalization_params(device=device, n_normalize_samples=n_samples, scan_mask=scan_mask)
        self.adapt_batchnorm(device=device, n_samples=n_samples, scan_mask=scan_mask)

    def _infer_train_batch_output(
        self, ry, rx, *, device, parameters, chunk_size=100, scan_mask=None
    ):
        """Model output ``(2, H, W)`` for the DP at (ry, rx), computed exactly as
        ``find_peaks_model`` does.

        The DP is run inside its train-mode ``find_peaks_model`` chunk, so BatchNorm
        normalizes it with the same ~``chunk_size`` real-DP statistics (the train-mode
        test-time domain adaptation). This reproduces the precomputed detection for that
        position -- unlike the eval + ``adapt_batchnorm`` path, whose global running stats
        differ from the chunk-local stats and over-detect on this OOD scan.

        The chunk is the same slice of ``np.argwhere(scan_mask)`` (row-major) that
        ``find_peaks_model`` would place (ry, rx) in; the resulting output is cached so
        neighbouring cursor positions in the same chunk reuse one forward pass.
        """
        Ry, Rx, _, _ = self.dataset_cartesian.shape
        if scan_mask is None:
            scan_mask = self._scan_mask
        if scan_mask is None:
            scan_mask = np.ones((Ry, Rx), dtype=bool)
        else:
            scan_mask = np.asarray(scan_mask, dtype=bool)
        valid = np.argwhere(scan_mask)  # row-major: matches find_peaks_model's iteration
        match = np.where((valid[:, 0] == ry) & (valid[:, 1] == rx))[0]
        if len(match):
            qi = int(match[0])
            start = (qi // chunk_size) * chunk_size
            chunk_positions = valid[start : start + chunk_size]
            local_i = qi - start
        else:
            # (ry, rx) is outside the ROI -- find_peaks_model never processes it. Still give
            # a faithful readout by running it at the head of a representative ROI chunk.
            start = -1  # never matches a real chunk_start -> not cacheable across positions
            head = valid[: max(0, chunk_size - 1)]
            chunk_positions = (
                np.concatenate([[[ry, rx]], head], axis=0) if len(head) else np.array([[ry, rx]])
            )
            local_i = 0

        cache = self._live_chunk_cache
        if start >= 0 and cache is not None and cache[0] == start and cache[1] == chunk_size:
            return cache[2][local_i]

        chunk = np.array([self.dataset_cartesian[r, c].array for r, c in chunk_positions])
        resized = self.resize_images(chunk, device=device, initial_chunk_size=len(chunk))
        ins = torch.tensor(resized, dtype=torch.float32).to(device)
        ins_batch = self._require_normalization_strategy().transform(
            ins, parameters
        )[:, None, ...]
        self.model.to(device)
        self.model.train()  # per-chunk BatchNorm stats, exactly like find_peaks_model
        with torch.no_grad():
            outs = self.model(ins_batch).detach().cpu().numpy()  # (n, 2, H, W)
        if start >= 0:
            self._live_chunk_cache = (start, chunk_size, outs)
        return outs[local_i]

    def infer_peaks_single(
        self,
        ry: int,
        rx: int,
        *,
        device: str = None,
        sigma_peak_blur: float = 1.0,
        threshold_peak: float = 0.5,
        n_normalize_samples: int = 1000,
        bn_mode: str = "train_batch",
        chunk_size: int = 100,
        scan_mask: ArrayLike = None,
    ):
        """Run the model on the single diffraction pattern at (ry, rx).

        Live counterpart of ``find_peaks_model`` for one scan position: resize ->
        normalize (cached median/iqr) -> model forward -> decode -> rescale to detector
        pixels. Returns a dict with keys ``"y_pixels"``, ``"x_pixels"``, ``"intensities"``
        (empty arrays when no peaks are found), matching the columns/units of
        ``peak_coordinates_cartesian`` / ``peak_intensities``.

        ``bn_mode`` selects the BatchNorm regime:

        - ``"train_batch"`` (default): run the DP inside its train-mode ``find_peaks_model``
          chunk so it gets the same per-chunk domain adaptation. Output **matches the
          precomputed find_peaks_model detection** for that position. Deterministic given
          the ROI + chunk_size.
        - ``"eval_adapt"``: eval mode using dataset-adapted BatchNorm running stats (see
          ``adapt_batchnorm``, lazy + cached). Faster (single-DP forward) but an
          approximation that over-detects on this out-of-distribution scan.
        """
        device = device or self.device
        parameters = self.ensure_normalization_params(
            device=device, n_normalize_samples=n_normalize_samples, scan_mask=scan_mask
        )

        if bn_mode == "train_batch":
            out = self._infer_train_batch_output(
                ry, rx, device=device, parameters=parameters,
                chunk_size=chunk_size, scan_mask=scan_mask,
            )
        elif bn_mode == "eval_adapt":
            # Domain-adapt BatchNorm to this dataset once, then infer in eval mode.
            self.adapt_batchnorm(device=device, n_samples=n_normalize_samples, scan_mask=scan_mask)
            dp = np.asarray(self.dataset_cartesian[ry, rx].array)
            resized = self.resize_images(dp[None], device=device, initial_chunk_size=1)
            ins = torch.tensor(resized, dtype=torch.float32).to(device)
            ins_batch = self._require_normalization_strategy().transform(
                ins, parameters
            )[:, None, ...]
            self.model.to(device)
            self.model.eval()
            with torch.no_grad():
                out = self.model(ins_batch).detach().cpu().numpy()[0]  # (2, H, W)
        else:
            raise ValueError(
                f"bn_mode must be 'train_batch' or 'eval_adapt', got {bn_mode!r}"
            )

        peak_coords, peak_ints = self._postprocess_single(
            out[0], out[1], sigma=sigma_peak_blur, threshold=threshold_peak
        )
        if len(peak_coords) == 0:
            empty = np.array([])
            return {"y_pixels": empty, "x_pixels": empty, "intensities": empty}

        # Rescale from model-input pixels back to original detector pixels (matches
        # the whole-scan rescale in find_peaks_model).
        scale = self.dataset_cartesian.shape[2] / self.final_shape[0]
        coords = np.asarray(peak_coords) * scale  # (N, 2) = [row=y, col=x]
        return {
            "y_pixels": coords[:, 0],
            "x_pixels": coords[:, 1],
            "intensities": np.asarray(peak_ints),
        }

    def find_peaks_model(
        self,
        device: str = "cuda:0",
        scan_mask: ArrayLike = None,
        n_normalize_samples: int = 1000,
        initial_chunk_size: int = 100,
        sigma_peak_blur: float = 1.0,
        threshold_peak: float = 0.5,
        show_plots=False,
        accelerating_voltage_kv: float = None,
    ):
        """Detect peaks throughout the scan with the trained model.

        Parameters
        ----------
        accelerating_voltage_kv
            Electron accelerating voltage used to convert detector sampling from
            mrad to inverse angstroms. For mrad data, the default is 300 kV and an
            explicit warning is emitted. Ignored for reciprocal-length calibration.
        """
        Ry, Rx, Qy, Qx = self.dataset_cartesian.shape
        total_positions = Ry * Rx

        # Resolve this once per run, both to avoid repeated unit parsing and to retain
        # the selected voltage for later polar-coordinate operations.
        sampling_inv_A = self.pixels_to_inv_A(accelerating_voltage_kv)

        # Remember the ROI so later normalization / BN adaptation (and the live widget)
        # restrict to the sample region. Storing the user-provided value (None stays the
        # whole scan); the setter invalidates cached stats only if the mask changed.
        self.scan_mask = scan_mask

        # ============================================
        # Handle scan_mask
        # ============================================
        if scan_mask is None:
            scan_mask = np.ones((Ry, Rx), dtype=bool)
        else:
            scan_mask = np.asarray(scan_mask, dtype=bool)
            if scan_mask.shape != (Ry, Rx):
                raise ValueError(f"scan_mask shape {scan_mask.shape} must match scan shape ({Ry}, {Rx})")
        
        # Get list of valid positions
        valid_positions = np.argwhere(scan_mask)  # Returns array of (ry, rx) pairs
        n_valid = len(valid_positions)
        
        peaks = Vector.from_shape(
            shape=(Ry, Rx),
            fields=["y_pixels", "x_pixels", "y_invA", "x_invA"],
            name="peaks_vector",
            units=["Pixels", "Pixels", "1/Å", "1/Å"],
        )
        intensities = Vector.from_shape(
            shape=(Ry, Rx),
            fields=["intensities", "intensities_sampled_from_dp"],
            name="intensities_vector",
            units=["Normalized", "Normalized"],
        )
        
        # ============================================
        # 1. Compute normalization parameters (only from valid positions)
        # ============================================
        # recompute=True to preserve the original per-call semantics (find_peaks_model
        # always recomputed the sample stats); the cache still serves infer/adapt.
        parameters = self.ensure_normalization_params(
            device=device,
            n_normalize_samples=n_normalize_samples,
            scan_mask=scan_mask,
            recompute=True,
        )

        # Run in TRAIN mode on purpose. The model trains on synthetic data; on the
        # (out-of-distribution) experimental scan, train-mode BatchNorm normalizes each
        # chunk with the experimental data's own statistics — test-time domain adaptation
        # that detects far better than eval mode (which would impose the synthetic-training
        # population stats on real data). Set it explicitly so a prior eval() / adapt_batchnorm
        # (e.g. from the live widget) can't leave the shared model in eval mode. The live
        # single-DP path (infer_peaks_single) instead uses adapt_batchnorm + eval.
        self.model.train()

        # ============================================
        # 2. Process only valid positions with chunking
        # ============================================
        chunk_size = initial_chunk_size
        pos_idx = 0
        
        with tqdm(total=n_valid, desc="Processing patterns") as pbar:
            while pos_idx < n_valid:
                try:
                    # ----------------------------------------
                    # 2a. Determine chunk boundaries
                    # ----------------------------------------
                    end_pos_idx = min(pos_idx + chunk_size, n_valid)
                    actual_chunk_size = end_pos_idx - pos_idx
                    
                    # ----------------------------------------
                    # 2b. Extract chunk data (only valid positions)
                    # ----------------------------------------
                    chunk_data = []
                    chunk_positions = []
                    
                    for i in range(pos_idx, end_pos_idx):
                        ry, rx = valid_positions[i]
                        chunk_data.append(self.dataset_cartesian[ry, rx].array)
                        chunk_positions.append((ry, rx))
                    
                    chunk_array = np.array(chunk_data)
                    
                    # ----------------------------------------
                    # 2c. Resize chunk
                    # ----------------------------------------
                    # self.model.to(device)
                    chunk_resized = self.resize_images(
                        chunk_array, 
                        device=device, 
                        initial_chunk_size=actual_chunk_size
                    )
                    
                    # ----------------------------------------
                    # 2d. Normalize and run model
                    # ----------------------------------------
                    ins = torch.tensor(chunk_resized, dtype=torch.float32).to(device)
                    dps_norm = self._require_normalization_strategy().transform(
                        ins, parameters
                    )
                    ins_batch = dps_norm[:, None, ...]
                    
                    with torch.no_grad():
                        outs = self.model(ins_batch).detach().cpu().numpy()
                    
                    # ----------------------------------------
                    # 2e. Post-process each pattern in chunk
                    # ----------------------------------------
                    for k in range(outs.shape[0]):
                        ry, rx = chunk_positions[k]
                        
                        peak_coords, peak_intensities = self._postprocess_single(
                            outs[k, 0], 
                            outs[k, 1],
                            show=show_plots,
                            sigma=sigma_peak_blur,
                            threshold=threshold_peak,
                        )
                        
                        if len(peak_coords) > 0:
                            peak_intensity_averages = sample_average_from_image(
                                ins_batch[k].squeeze().detach().cpu().numpy(), 
                                peak_coords
                            )
                            peak_intensities_data = np.column_stack([
                                peak_intensities,
                                peak_intensity_averages,
                            ])
                            
                            peak_coords_original = peak_coords * (
                                self.dataset_cartesian.shape[2] / self.final_shape[0]
                            )
                            
                            peak_data = np.column_stack([
                                peak_coords_original,
                                peak_coords_original * sampling_inv_A
                            ])
                            
                            peaks[ry, rx] = peak_data
                            intensities[ry, rx] = peak_intensities_data
                    
                    # ----------------------------------------
                    # 2f. Memory cleanup
                    # ----------------------------------------
                    del ins, dps_norm, ins_batch, outs, chunk_array, chunk_resized
                    if 'cuda' in device:
                        torch.cuda.empty_cache()
                    
                    # ----------------------------------------
                    # 2g. Update progress and move to next chunk
                    # ----------------------------------------
                    pbar.update(actual_chunk_size)
                    pos_idx = end_pos_idx
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        if 'cuda' in device:
                            torch.cuda.empty_cache()
                        
                        chunk_size = max(1, chunk_size // 2)
                        print(f"\nGPU OOM! Reducing chunk size to {chunk_size}")
                        
                        if chunk_size == 1:
                            print("Falling back to CPU processing")
                            device = "cpu"
                    else:
                        raise e
        
        print('Done!')
        self.peak_coordinates_cartesian = peaks
        self.peak_intensities = intensities

    @staticmethod
    def _save_object(filepath, obj):
        # Wrap in a 0-d object array so np.save pickles the WHOLE object. A canon Vector is
        # array-like, so np.save(vector) would otherwise flatten it to an (Ry, Rx) object
        # array of cells that loses the Vector's fields/units and can't be reconstructed
        # (breaks polar_transform_peaks, which needs a Vector). The 0-d wrapper round-trips
        # via the size-1 .item() unwrap in the load_* methods.
        arr = np.empty((), dtype=object)
        arr[()] = obj
        np.save(filepath, arr, allow_pickle=True)

    def save_cartesian_peaks(self, filepath):
        self._save_object(filepath, self.peak_coordinates_cartesian)

    def load_cartesian_peaks(self, filepath):
        peak_coordinates_cartesian = np.load(filepath, allow_pickle=True)
        if isinstance(peak_coordinates_cartesian, np.ndarray) and peak_coordinates_cartesian.dtype == object and peak_coordinates_cartesian.size == 1:
            peak_coordinates_cartesian = peak_coordinates_cartesian.item()
        self.peak_coordinates_cartesian = peak_coordinates_cartesian
    
    def save_polar_peaks(self, filepath):
        self._save_object(filepath, self.polar_peaks)

    def save_polar_data(self, filepath):
        self._save_object(filepath, self.polar_data)

    def load_polar_peaks(self, filepath):
        polar_peaks = np.load(filepath, allow_pickle=True)
        if isinstance(polar_peaks, np.ndarray) and polar_peaks.dtype == object and polar_peaks.size == 1:
            polar_peaks = polar_peaks.item()
        self.polar_peaks = polar_peaks

    def load_polar_data(self, filepath):
        obj = np.load(filepath, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
            obj = obj.item()
        self.polar_data = obj
    
        # Populate attributes expected elsewhere
        r_grid = self.polar_data['r_invA']
        self.max_radius_invA = float(np.max(r_grid))
        self.num_radial_bins = int(r_grid.shape[0])
        self.num_annular_bins = int(r_grid.shape[1])

    def save_peak_intensities(self, filepath):
        self._save_object(filepath, self.peak_intensities)

    def load_peak_intensities(self, filepath):
        peak_intensities = np.load(filepath, allow_pickle=True)
        if isinstance(peak_intensities, np.ndarray) and peak_intensities.dtype == object and peak_intensities.size == 1:
            peak_intensities = peak_intensities.item()
        self.peak_intensities = peak_intensities

    def save_image_centers(self, filepath):
        np.save(filepath, self.image_centers)

    def load_image_centers(self, filepath):
        image_centers = np.load(filepath, allow_pickle=True)
        if isinstance(image_centers, np.ndarray) and image_centers.dtype == object and image_centers.size == 1:
            image_centers = image_centers.item()
        self.image_centers = image_centers
    
    def process_polar(
        self,
        scan_mask: ArrayLike = None,
        two_fold_symmetry: bool = True,
        center_method: str = "descent",
        center_radial_min: float = 4.0,
        center_radial_max: float | None = None,
        center_radial_step: float = 1.0,
        center_num_annular_bins: int = 180,
        center_n_phi: int = 120,
        center_kpow: float = 0.0,
        center_ellipse_params: tuple[float, float, float] | None = None,
        center_device: str | None = None,
        center_batch_size: int = 16,
        center_local_margin: int = 40,
        fallback_to_peaks: bool = True,
    ):
        """Find image centers, then return polar transforms of data and peaks.

        ``center_method`` defaults to Karen Ehrhardt's angular-uniformity descent
        method. Use ``center_method="grid"`` for the slower coarse-to-fine
        search, or ``center_method="peaks"`` to force the previous peak-based
        central-beam heuristic.
        """
        self.image_centers = self.find_central_beams_4d(
            scan_mask=scan_mask,
            center_method=center_method,
            radial_min=center_radial_min,
            radial_max=center_radial_max,
            radial_step=center_radial_step,
            num_annular_bins=center_num_annular_bins,
            n_phi=center_n_phi,
            kpow=center_kpow,
            ellipse_params=center_ellipse_params,
            center_device=center_device,
            center_batch_size=center_batch_size,
            local_margin=center_local_margin,
            fallback_to_peaks=fallback_to_peaks,
        )
        self.polar_peaks = self.polar_transform_peaks(
            cartesian_peaks=self.peak_coordinates_cartesian,
            centers=self.image_centers,
            scan_mask=scan_mask,
            two_fold_symmetry=two_fold_symmetry,
            ellipse_params=center_ellipse_params,
        )
        self.polar_data = self.polar_transform_4d(
            self.dataset_cartesian,
            centers=self.image_centers,
            scan_mask=scan_mask,
            two_fold_symmetry=two_fold_symmetry,
            ellipse_params=center_ellipse_params,
        )

    def find_central_beams_4d(self, **kwargs):
        """Locate the central beam at every scan position.

        Thin wrapper over :func:`quantem.diffraction.polar_transform.find_central_beams_4d`,
        supplying this object's dataset, peaks and device. All keyword arguments are
        forwarded; see that function for the full set.
        """

        return find_central_beams_4d(
            self.dataset_cartesian,
            peaks=getattr(self, "peak_coordinates_cartesian", None),
            default_device=self.device,
            **kwargs,
        )
    
    def polar_transform_peaks(
        self,
        cartesian_peaks,
        centers,
        scan_mask: ArrayLike = None,
        two_fold_symmetry=True,
        ellipse_params: tuple[float, float, float] | None = None,
        use_tqdm: bool=True,
    ):
        """Transform detected Cartesian peak coordinates with Karen's polar convention.

        Peaks are preserved one-to-one. With two-fold symmetry, theta is folded
        modulo pi while partner detections remain separate rows.
        """
        return karen_polar_transform_peaks(
            cartesian_peaks,
            centers,
            scan_mask=scan_mask,
            sampling_conversion_factor=self.pixels_to_inv_A(),
            two_fold_rotation_symmetry=two_fold_symmetry,
            ellipse_params=ellipse_params,
            use_tqdm=use_tqdm,
        )
    
    def polar_transform_4d(
        self,
        data,
        centers,
        scan_mask: ArrayLike = None,
        num_r=None,
        num_theta=360,
        two_fold_symmetry=True,
        ellipse_params: tuple[float, float, float] | None = None,
        device: str | None = None,
        batch_size: int = 128,
        use_tqdm: bool=True,
    ):
        """
        Perform polar transform on the last two axes of a 4D array.
        
        Parameters:
        -----------
        data : ndarray, shape (N, M, H, W)
            4D input array where H, W are the axes to transform
        centers : ndarray, shape (2, N, M)
            Center of each diffraction pattern (usually determined by central beam)
        scan_mask : ArrayLike, optional
            Boolean mask (N, M) indicating which positions to process
        num_r : int, optional
            Number of radial bins. If None, uses max radius across all patterns
        num_theta : int, optional
            Number of angular bins (default: 360)
        two_fold_symmetry : bool, optional
            If True, applies 2-fold symmetry by summing opposite angles (default: True).
            Samples the full [0, 2π] range but folds it to [0, π] by summing
            theta and theta+π positions.
        use_tqdm : bool, optional
            Whether to show progress bar (default: True)
        
        Returns:
        --------
        polar_data : dict
            Dictionary containing polar-transformed data with keys:
            - 'r_pixels': radial coordinates in pixels
            - 'theta': angular coordinates in radians [0, π] if two_fold_symmetry, else [0, 2π]
            - 'r_invA': radial coordinates in 1/Å
            - 'intensity': transformed intensity data
        
        Notes:
        ------
        Also sets the following attributes on self:
        - self.max_radius_pixels : maximum radius in pixels
        - self.max_radius_invA : maximum radius in 1/Å
        - self.num_radial_bins : number of radial bins
        - self.num_annular_bins : number of angular bins (after symmetry folding)
        - self.two_fold_symmetry : whether 2-fold symmetry was used
        """
        N, M, H, W = data.shape
        
        # Handle scan_mask
        if scan_mask is None:
            scan_mask = np.ones((N, M), dtype=bool)
        else:
            scan_mask = np.asarray(scan_mask, dtype=bool)
            if scan_mask.shape != (N, M):
                raise ValueError(f"scan_mask shape {scan_mask.shape} must match {(N, M)}")
        if not np.any(scan_mask):
            raise ValueError("scan_mask must include at least one scan position.")

        centers = np.asarray(centers, dtype=float)
        if centers.shape == (2, N, M):
            centers_karen = np.moveaxis(centers, 0, -1)
            centers_bragg = centers
        elif centers.shape == (N, M, 2):
            centers_karen = centers
            centers_bragg = np.moveaxis(centers, -1, 0)
        else:
            raise ValueError(
                f"centers must have shape {(2, N, M)} or {(N, M, 2)}, got {centers.shape}"
            )
        if two_fold_symmetry and num_theta % 2 != 0:
            raise ValueError("num_theta must be even when two_fold_symmetry=True.")
        
        # Calculate consistent max_radius across entire dataset (only from masked positions)
        valid_centers_0 = centers_bragg[0][scan_mask]
        valid_centers_1 = centers_bragg[1][scan_mask]
        dist_to_origin_sq = (valid_centers_0**2 + valid_centers_1**2).min()
        dist_to_corner_sq = ((H-1 - valid_centers_0)**2 + (W-1 - valid_centers_1)**2).max()
        max_radius_pixels = np.sqrt(max(dist_to_origin_sq, dist_to_corner_sq))
        
        if num_r is None:
            num_r = int(np.ceil(max_radius_pixels))
        num_r = max(1, int(num_r))
        radial_step = max_radius_pixels / num_r if max_radius_pixels > 0 else 1.0
        
        # Calculate maximum radius in inverse angstroms
        max_radius_invA = max_radius_pixels * self.pixels_to_inv_A()
        
        polar_full = karen_polar_transform(
            data,
            origin_array=centers_karen,
            ellipse_params=ellipse_params,
            num_annular_bins=num_theta,
            radial_min=0.0,
            radial_max=max_radius_pixels,
            radial_step=radial_step,
            two_fold_rotation_symmetry=False,
            device=device if device is not None else self.device,
            batch_size=batch_size,
            show_progress=use_tqdm,
        )
        polar_intensity_full = np.asarray(polar_full.array, dtype=np.float32).transpose(0, 1, 3, 2)
        polar_intensity_full[~scan_mask] = 0.0

        # Pre-calculate coordinate arrays in both units using Karen's radial bins.
        num_r_actual = polar_intensity_full.shape[2]
        r_pixels = np.arange(num_r_actual, dtype=float) * radial_step
        theta_full = np.linspace(0, 2*np.pi, polar_intensity_full.shape[-1], endpoint=False)
        r_grid_full, theta_grid_full = np.meshgrid(r_pixels, theta_full, indexing='ij')
        
        # Apply 2-fold symmetry if requested
        if two_fold_symmetry:
            # Fold to [0, π]
            num_theta_folded = polar_intensity_full.shape[-1] // 2
            theta_folded = np.linspace(0, np.pi, num_theta_folded, endpoint=False)

            # Create output arrays
            r_grid, theta_grid = np.meshgrid(r_pixels, theta_folded, indexing='ij')
            r_invA_grid = r_grid * self.pixels_to_inv_A()
            polar_intensity = (
                polar_intensity_full[:, :, :, :num_theta_folded]
                + polar_intensity_full[:, :, :, num_theta_folded:]
            )
            
            num_annular_bins = num_theta_folded
        else:
            # Use full range
            theta_grid = theta_grid_full
            r_grid = r_grid_full
            r_invA_grid = r_grid * self.pixels_to_inv_A()
            polar_intensity = polar_intensity_full
            num_annular_bins = num_theta
        
        # Store metadata
        self.max_radius_pixels = max_radius_pixels
        self.max_radius_invA = max_radius_invA
        self.num_radial_bins = num_r_actual
        self.num_annular_bins = num_annular_bins
        self.two_fold_symmetry = two_fold_symmetry
        
        polar_data = {
            "r_pixels": r_grid,
            "theta": theta_grid,
            "r_invA": r_invA_grid,
            "intensity": polar_intensity,
        }
        
        return polar_data

    def visualize_peak_detection(self, n_images=10, indices=None, images_per_row=5, figsize_per_image=(3.2, 3), vmax_polar=20, vmax_cartesian=None):
        """
        Visualize peak detection results for multiple diffraction patterns.
        
        Parameters:
        -----------
        self : BraggPeaksPolymer
            BraggPeaksPolymer object with processed data
        n_images : int
            Number of images to display (ignored if indices is provided)
        indices : list of tuples, optional
            List of (ind_y, ind_x) coordinates to visualize. If None, random indices are selected.
        images_per_row : int
            Number of images per row (default: 5)
        figsize_per_image : tuple
            Size of each subplot (width, height)
        vmax_polar : float
            Maximum value for polar data colormap
        vmax_cartesian : float
            Maximum value for cartesian data colormap
        
        Returns:
        --------
        fig, axes : matplotlib figure and axes
        """
        
        # Generate or validate indices
        if indices is None:
            Ry, Rx = self.dataset_cartesian.shape[:2]
            # Generate random indices
            flat_indices = np.random.choice(Ry * Rx, size=min(n_images, Ry * Rx), replace=False)
            indices = [(idx // Rx, idx % Rx) for idx in flat_indices]
        else:
            n_images = len(indices)
        
        # Calculate grid dimensions
        n_rows = int(np.ceil(n_images / images_per_row))
        n_cols = 5  # 5 types of visualizations per pattern
        actual_cols = images_per_row * n_cols
        
        # Create figure
        fig_width = figsize_per_image[0] * actual_cols
        fig_height = figsize_per_image[1] * n_rows
        fig, axes = plt.subplots(n_rows, actual_cols, figsize=(fig_width, fig_height))
        
        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Column titles (only for first row)
        col_titles = [
            "Polar Transform",
            "Polar + Peaks",
            "Cartesian + Peaks",
            "Cartesian Original",
            "Cartesian Normalized"
        ]
        
        # Process each image
        for img_idx, (ind_y, ind_x) in enumerate(indices):
            row = img_idx // images_per_row
            col_offset = (img_idx % images_per_row) * n_cols
            
            # Check if peaks exist for this pattern
            has_peaks = (self.peak_coordinates_cartesian[ind_y, ind_x] is not None and 
                         len(self.peak_coordinates_cartesian[ind_y, ind_x]) > 0)
            
            # 1. Polar Transform
            ax = axes[row, col_offset]
            print(self.polar_data["intensity"][ind_y, ind_x].shape)
            im = ax.matshow(self.polar_data["intensity"][ind_y, ind_x], cmap='turbo', vmax=vmax_polar)
            if row == 0:
                ax.set_title(col_titles[0], fontsize=10, pad=10)
            ax.text(0.05, 0.95, f'({ind_y},{ind_x})', transform=ax.transAxes, 
                    fontsize=8, va='top', ha='left', color='white', 
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            ax.set_axis_off()
            
            # 2. Polar Transform with Peaks
            ax = axes[row, col_offset + 1]
            ax.matshow(self.polar_data["intensity"][ind_y, ind_x], cmap='turbo', vmax=vmax_polar)
            if has_peaks and self.polar_peaks[ind_y, ind_x] is not None and len(self.polar_peaks[ind_y, ind_x]) > 0:
                # Convert radial coordinates to bin indices
                r_coords = self.polar_peaks[ind_y, ind_x][:, 0]
                theta_coords = self.polar_peaks[ind_y, ind_x][:, 1]
                
                # Convert theta from radians to angular bins (0 to num_annular_bins)
                theta_period = np.pi if getattr(self, "two_fold_symmetry", False) else 2 * np.pi
                theta_bins = theta_coords * (self.num_annular_bins / theta_period)
                
                ax.scatter(theta_bins, r_coords, c='red', s=15, alpha=0.8, edgecolors='white', linewidths=0.5)
            if row == 0:
                ax.set_title(col_titles[1], fontsize=10, pad=10)
            ax.set_axis_off()
            
            # 3. Cartesian with Peaks and Center
            img = self.dataset_cartesian[ind_y, ind_x].array
            lower_q = 0.01
            upper_q = 0.99
            vmin, vmax = np.quantile(img[np.isfinite(img)], [lower_q, upper_q])
            if vmax_cartesian is None:
                vmax_cartesian = vmax
            ax = axes[row, col_offset + 2]
            ax.matshow(img, cmap="gray", vmin=vmin, vmax=vmax_cartesian)
            # ax.matshow(self.dataset_cartesian[ind_y, ind_x].array, cmap="gray", vmax=vmax_cartesian)
            # ax.matshow(self.resized_cartesian_data[ind_y, ind_x], cmap="gray", vmax=vmax_cartesian)
            if has_peaks:
                ax.scatter(self.peak_coordinates_cartesian[ind_y, ind_x][:, 1], 
                          self.peak_coordinates_cartesian[ind_y, ind_x][:, 0], 
                          c='red', s=15, alpha=0.8, edgecolors='white', linewidths=0.5)
            ax.scatter(self.image_centers[1, ind_y, ind_x], 
                      self.image_centers[0, ind_y, ind_x], 
                      c='red', s=500, marker='x', linewidths=2)
            if row == 0:
                ax.set_title(col_titles[2], fontsize=10, pad=10)
            ax.set_axis_off()
            
            # 4. Original Cartesian
            ax = axes[row, col_offset + 3]
            im = ax.matshow(img, cmap="gray", vmin=vmin, vmax=vmax_cartesian)
            # im = ax.matshow(self.dataset_cartesian[ind_y, ind_x].array, cmap="gray", vmax=vmax_cartesian)
            # im = ax.matshow(self.resized_cartesian_data[ind_y, ind_x], cmap="gray", vmax=vmax_cartesian)
            if row == 0:
                ax.set_title(col_titles[3], fontsize=10, pad=10)
            ax.set_axis_off()
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # # 5. Normalized Cartesian
            # ax = axes[row, col_offset + 4]
            # im = ax.matshow(self.normalized_dps_array[ind_y, ind_x], cmap="gray")
            # if row == 0:
            #     ax.set_title(col_titles[4], fontsize=10, pad=10)
            # ax.set_axis_off()
            # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        total_plots = n_images
        for idx in range(total_plots, n_rows * images_per_row):
            row = idx // images_per_row
            col_offset = (idx % images_per_row) * n_cols
            for col in range(n_cols):
                axes[row, col_offset + col].set_visible(False)
        
        fig.tight_layout()
        return fig, axes

    def estimate_peak_windows(
        self,
        num_bins=200,
        q_min=None,
        q_max=None,
        n_peaks=5,
        height_percentile=10,
        prominence_factor=0.1,
        width_factor=2.0,
        min_width=0.05,
        smoothing_sigma=2.0,
        intensity_field='intensities',
        mode='intensity',
        log_scale=False,
    ):
        """
        Automatically detect the top N most prominent peaks and estimate their windows.
        
        Parameters
        ----------
        num_bins : int
            Number of radial bins
        q_min : float, optional
            Minimum q value for binning
        q_max : float, optional
            Maximum q value for binning
        n_peaks : int
            Number of top peaks to detect
        height_percentile : float
            Percentile threshold for peak height (peaks below this are ignored)
        prominence_factor : float
            Factor of max intensity for minimum peak prominence
        width_factor : float
            Multiplier for estimating peak window width from FWHM
        min_width : float
            Minimum window width in 1/Å
        smoothing_sigma : float
            Gaussian smoothing sigma for noise reduction before peak detection
        mode : {'intensity', 'count'}
            Radial profile to detect peaks on: intensity-weighted histogram of peak q
            ('intensity', default) or the number of detected peaks per bin ('count').
        log_scale : bool
            If True, detect peaks on log1p(profile) so small peaks are not dominated
            by large ones.
            
        Returns
        -------
        peak_centers : array
            q-values for peak centers (shape: n_peaks)
        peak_windows : array
            Window boundaries for each peak (shape: n_peaks, 2)
            Each row is [q_min, q_max] for that peak
        peak_info : dict
            Additional information about detected peaks including:
            - 'heights': peak heights
            - 'prominences': peak prominences
            - 'widths': estimated peak widths (FWHM)
        """
        
        if mode not in ('intensity', 'count'):
            raise ValueError(f"mode must be 'intensity' or 'count', got {mode!r}")

        # Get radial profile (intensity-weighted or peak-count)
        all_r = _vector_field_flat(self.polar_peaks, "r_invA")
        
        if q_min is None:
            q_min = 0
        if q_max is None:
            q_max = np.max(all_r)
        
        r_bins = np.linspace(q_min, q_max, num_bins + 1)
        if mode == 'intensity':
            all_intensity = _vector_field_flat(self.peak_intensities, intensity_field)
            profile, _ = np.histogram(all_r, bins=r_bins, weights=all_intensity)
        else:  # 'count'
            profile, _ = np.histogram(all_r, bins=r_bins)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2

        # Optional log compression so small peaks are not dominated by large ones
        if log_scale:
            profile = np.log1p(profile)

        # Smooth the data to reduce noise
        if smoothing_sigma > 0:
            intensity_smooth = gaussian_filter1d(profile, smoothing_sigma)
        else:
            intensity_smooth = profile
        
        # Calculate thresholds
        height_threshold = np.percentile(intensity_smooth, height_percentile)
        prominence_threshold = prominence_factor * np.max(intensity_smooth)
        
        # Find peaks
        peaks_indices, properties = find_peaks(
            intensity_smooth,
            height=height_threshold,
            prominence=prominence_threshold,
            distance=int(min_width / (r_centers[1] - r_centers[0]))  # Minimum separation
        )
        
        if len(peaks_indices) == 0:
            print("No peaks found with current parameters!")
            return np.array([]), np.array([]).reshape(0, 2), {}
        
        # Sort by prominence and take top N
        prominences = properties['prominences']
        sorted_indices = np.argsort(prominences)[::-1][:n_peaks]
        top_peak_indices = peaks_indices[sorted_indices]
        top_peak_indices = np.sort(top_peak_indices)  # Re-sort by position
        
        # Get peak centers
        peak_centers = r_centers[top_peak_indices]
        
        # Calculate peak widths (FWHM)
        widths_data = peak_widths(intensity_smooth, top_peak_indices, rel_height=0.5)
        fwhm_bins = widths_data[0]  # Width in bins
        fwhm_invA = fwhm_bins * (r_centers[1] - r_centers[0])  # Convert to 1/Å
        
        # Estimate windows: center ± width_factor * FWHM/2, with minimum width
        half_widths = np.maximum(width_factor * fwhm_invA / 2, min_width / 2)
        peak_windows = np.column_stack([
            peak_centers - half_widths,
            peak_centers + half_widths
        ])
        
        # Clip windows to data range
        peak_windows[:, 0] = np.maximum(peak_windows[:, 0], q_min)
        peak_windows[:, 1] = np.minimum(peak_windows[:, 1], q_max)
        
        # Collect additional info
        peak_info = {
            'heights': intensity_smooth[top_peak_indices],
            'prominences': prominences[sorted_indices],
            'widths_fwhm': fwhm_invA,
            'intensity_profile': intensity_smooth,
            'profile': intensity_smooth,
            'r_centers': r_centers,
            'mode': mode,
            'log_scale': log_scale,
        }
        
        # Print summary
        print(f"Detected {len(peak_centers)} peaks:")
        _to_d = lambda q: (1.0 / q if q > 0 else float('inf'))  # d-spacing (Å) = 1 / q (1/Å)
        for i, (center, window, height, prom, width) in enumerate(zip(
            peak_centers, peak_windows, peak_info['heights'],
            peak_info['prominences'], peak_info['widths_fwhm']
        )):
            print(f"  Peak {i+1}: center={center:.3f} 1/Å (d={_to_d(center):.2f} Å), "
                  f"window=[{window[0]:.3f}, {window[1]:.3f}] 1/Å "
                  f"(d=[{_to_d(window[1]):.2f}, {_to_d(window[0]):.2f}] Å), "
                  f"height={height:.1f}, prominence={prom:.1f}, FWHM={width:.3f} 1/Å")
        
        return peak_centers, peak_windows, peak_info

    def peak_radial_intensity_plot(self, *args, **kwargs):
        """Radial peak-intensity profile.

        Forwards to :func:`quantem.diffraction.peak_visualization.peak_radial_intensity_plot`,
        supplying this object's data. See that function for the parameters.
        """

        return _peak_radial_intensity_plot(self.polar_peaks, self.peak_intensities, *args, **kwargs)

    def peak_radial_count_plot(self, *args, **kwargs):
        """Radial peak-count profile.

        Forwards to :func:`quantem.diffraction.peak_visualization.peak_radial_count_plot`,
        supplying this object's data. See that function for the parameters.
        """

        return _peak_radial_count_plot(self.polar_peaks, *args, **kwargs)

    def make_orientation_histogram(
        self,
        radial_ranges: np.ndarray = None,
        orientation_map=None,
        orientation_ind: int = 0,
        orientation_growth_angles: np.array = 0.0,
        orientation_separate_bins: bool = False,
        orientation_flip_sign: bool = False,
        orientation_offset_degrees: float = 0.0,
        upsample_factor: float = 4.0,
        theta_step_deg: float = 1.0,
        sigma_x: float = 1.0,
        sigma_y: float = 1.0,
        sigma_theta: float = 3.0,
        use_peak_sigma: bool = False,
        peak_sigma_samples: int = 6,
        normalize_intensity_image: bool = False,
        normalize_intensity_stack: bool = True,
        progress_bar: bool = True,
        r_field: str = "r_invA",
        theta_field: str = "theta",
        intensity_field: str = "intensities",
        # intensity_field: str = "intensities_sampled_from_dp",
    ):
        """
        Create a 3D or 4D orientation histogram from bragg peaks.
        
        Can generate histograms from either:
        1. Polar peak data with radial ranges
        2. Orientation map with Euler angles (for fiber textures)
    
        Parameters
        ----------
        radial_ranges : np.ndarray, optional
            Size (N x 2) array for N radial bins, or (2,) for a single bin.
        orientation_map : OrientationMap, optional
            Class containing Euler angles to generate a flowline map.
        orientation_ind : int
            Index of the orientation map (default 0)
        orientation_growth_angles : np.array
            Angles to place into histogram, relative to orientation.
        orientation_separate_bins : bool
            Whether to place multiple angles into multiple radial bins.
        orientation_flip_sign : bool
            Flip the direction of theta
        orientation_offset_degrees : float
            Offset for orientation angles in degrees
        upsample_factor : float
            Upsample factor for output histogram
        theta_step_deg : float
            Step size along annular direction in degrees
        sigma_x : float
            Smoothing in x direction before upsample
        sigma_y : float
            Smoothing in y direction before upsample
        sigma_theta : float
            Smoothing in annular direction (units of bins, periodic)
        use_peak_sigma : bool
            Spread signal along annular direction using measured peak width
        peak_sigma_samples : int
            Number of samples for peak sigma spreading
        normalize_intensity_image : bool
            Normalize to max peak intensity = 1, per image
        normalize_intensity_stack : bool
            Normalize to max peak intensity = 1, all images
        progress_bar : bool
            Enable progress bar
        r_field : str
            Name of radial coordinate field
        theta_field : str
            Name of angular coordinate field
        intensity_field : str
            Name of intensity field
    
        Returns
        -------
        orient_hist : np.ndarray
            4D array containing Bragg peak intensity histogram
            [radial_bin, x_probe, y_probe, theta]
        """
        # Coordinates
        theta = np.arange(0, 180, theta_step_deg) * np.pi / 180.0
        dtheta = theta[1] - theta[0]
        dtheta_deg = dtheta * 180 / np.pi
        num_theta_bins = np.size(theta)
        
        # Setup for peak sigma spreading
        if use_peak_sigma:
            v_sigma = np.linspace(-2, 2, 2 * peak_sigma_samples + 1)
            w_sigma = np.exp(-(v_sigma**2) / 2)
    
        if orientation_map is None:
            # Input bins
            radial_ranges = np.array(radial_ranges)
            if radial_ranges.ndim == 1:
                radial_ranges = radial_ranges[None, :]
            radial_ranges_2 = radial_ranges**2
            num_radii = radial_ranges.shape[0]
            size_input = self.polar_peaks.shape
        else:
            orientation_growth_angles = np.atleast_1d(orientation_growth_angles)
            num_angles = orientation_growth_angles.shape[0]
            size_input = [orientation_map.num_x, orientation_map.num_y]
            if orientation_separate_bins is False:
                num_radii = 1
            else:
                num_radii = num_angles
    
        size_output = np.round(
            np.array(size_input).astype("float") * upsample_factor
        ).astype("int")
    
        # Output init
        orient_hist = np.zeros([num_radii, size_output[0], size_output[1], num_theta_bins])
    
        # Loop over all probe positions
        for a0 in range(num_radii):
            t = "Generating histogram " + str(a0)
            for rx, ry in tqdmnd(
                *size_input, desc=t, unit=" probe positions", disable=not progress_bar
            ):
                x = (rx + 0.5) * upsample_factor - 0.5
                y = (ry + 0.5) * upsample_factor - 0.5
                x = np.clip(x, 0, size_output[0] - 2)
                y = np.clip(y, 0, size_output[1] - 2)
                xF = np.floor(x).astype("int")
                yF = np.floor(y).astype("int")
                dx = x - xF
                dy = y - yF
    
                add_data = False
    
                if orientation_map is None:
                    p_r = _vector_field_cell(self.polar_peaks, r_field, rx, ry)
                    p_theta = _vector_field_cell(self.polar_peaks, theta_field, rx, ry)
                    
                    if p_r is not None and len(p_r) > 0:
                        r2 = p_r**2
                        sub = np.logical_and(
                            r2 >= radial_ranges_2[a0, 0], 
                            r2 < radial_ranges_2[a0, 1]
                        )
                        if np.any(sub):
                            intensity_data = _vector_field_cell(
                                self.peak_intensities, intensity_field, rx, ry
                            )
                            if intensity_data is not None and len(intensity_data) > 0:
                                add_data = True
                                intensity = intensity_data[sub]
                                
                                # Get theta values
                                theta_radians = p_theta[sub]
                                if orientation_flip_sign:
                                    theta_radians *= -1
                                # Add offset
                                theta_radians += orientation_offset_degrees * np.pi / 180
                                theta_radians = np.mod(theta_radians, np.pi)
                                t = theta_radians / dtheta
                                
                                # Spread signal using peak sigma if requested
                                if use_peak_sigma:
                                    # Try to get sigma values if available
                                    if 'sigma_theta' in self.polar_peaks.fields:
                                        theta_std = _vector_field_cell(
                                            self.polar_peaks, "sigma_theta", rx, ry
                                        )[sub] / dtheta
                                        t = (t[:, None] + theta_std[:, None] * v_sigma[None, :]).ravel()
                                        intensity = (intensity[:, None] * w_sigma[None, :]).ravel()
                else:
                    if orientation_map.corr[rx, ry, orientation_ind] > 0:
                        if orientation_separate_bins is False:
                            if orientation_flip_sign:
                                t = (
                                    np.array(
                                        [
                                            (
                                                -orientation_map.angles[
                                                    rx, ry, orientation_ind, 0
                                                ]
                                                - orientation_map.angles[
                                                    rx, ry, orientation_ind, 2
                                                ]
                                            )
                                            / dtheta
                                        ]
                                    )
                                    + orientation_growth_angles
                                )
                            else:
                                t = (
                                    np.array(
                                        [
                                            (
                                                orientation_map.angles[
                                                    rx, ry, orientation_ind, 0
                                                ]
                                                + orientation_map.angles[
                                                    rx, ry, orientation_ind, 2
                                                ]
                                            )
                                            / dtheta
                                        ]
                                    )
                                    + orientation_growth_angles
                                )
                            # Add offset
                            t += orientation_offset_degrees / dtheta_deg
                            intensity = (
                                np.ones(num_angles)
                                * orientation_map.corr[rx, ry, orientation_ind]
                            )
                            add_data = True
                        else:
                            if orientation_flip_sign:
                                t = (
                                    np.array(
                                        [
                                            (
                                                -orientation_map.angles[
                                                    rx, ry, orientation_ind, 0
                                                ]
                                                - orientation_map.angles[
                                                    rx, ry, orientation_ind, 2
                                                ]
                                            )
                                            / dtheta
                                        ]
                                    )
                                    + orientation_growth_angles[a0]
                                )
                            else:
                                t = (
                                    np.array(
                                        [
                                            (
                                                orientation_map.angles[
                                                    rx, ry, orientation_ind, 0
                                                ]
                                                + orientation_map.angles[
                                                    rx, ry, orientation_ind, 2
                                                ]
                                            )
                                            / dtheta
                                        ]
                                    )
                                    + orientation_growth_angles[a0]
                                )
                            # Add offset
                            t += orientation_offset_degrees / dtheta_deg
                            intensity = orientation_map.corr[rx, ry, orientation_ind]
                            add_data = True
    
                if add_data:
                    tF = np.floor(t).astype("int")
                    dt = t - tF
    
                    orient_hist[a0, xF, yF, :] = orient_hist[a0, xF, yF, :] + np.bincount(
                        np.mod(tF, num_theta_bins),
                        weights=(1 - dx) * (1 - dy) * (1 - dt) * intensity,
                        minlength=num_theta_bins,
                    )
                    orient_hist[a0, xF, yF, :] = orient_hist[a0, xF, yF, :] + np.bincount(
                        np.mod(tF + 1, num_theta_bins),
                        weights=(1 - dx) * (1 - dy) * (dt) * intensity,
                        minlength=num_theta_bins,
                    )
    
                    orient_hist[a0, xF + 1, yF, :] = orient_hist[
                        a0, xF + 1, yF, :
                    ] + np.bincount(
                        np.mod(tF, num_theta_bins),
                        weights=(dx) * (1 - dy) * (1 - dt) * intensity,
                        minlength=num_theta_bins,
                    )
                    orient_hist[a0, xF + 1, yF, :] = orient_hist[
                        a0, xF + 1, yF, :
                    ] + np.bincount(
                        np.mod(tF + 1, num_theta_bins),
                        weights=(dx) * (1 - dy) * (dt) * intensity,
                        minlength=num_theta_bins,
                    )
    
                    orient_hist[a0, xF, yF + 1, :] = orient_hist[
                        a0, xF, yF + 1, :
                    ] + np.bincount(
                        np.mod(tF, num_theta_bins),
                        weights=(1 - dx) * (dy) * (1 - dt) * intensity,
                        minlength=num_theta_bins,
                    )
                    orient_hist[a0, xF, yF + 1, :] = orient_hist[
                        a0, xF, yF + 1, :
                    ] + np.bincount(
                        np.mod(tF + 1, num_theta_bins),
                        weights=(1 - dx) * (dy) * (dt) * intensity,
                        minlength=num_theta_bins,
                    )
    
                    orient_hist[a0, xF + 1, yF + 1, :] = orient_hist[
                        a0, xF + 1, yF + 1, :
                    ] + np.bincount(
                        np.mod(tF, num_theta_bins),
                        weights=(dx) * (dy) * (1 - dt) * intensity,
                        minlength=num_theta_bins,
                    )
                    orient_hist[a0, xF + 1, yF + 1, :] = orient_hist[
                        a0, xF + 1, yF + 1, :
                    ] + np.bincount(
                        np.mod(tF + 1, num_theta_bins),
                        weights=(dx) * (dy) * (dt) * intensity,
                        minlength=num_theta_bins,
                    )
    
        # Smoothing / interpolation
        if (sigma_x is not None) or (sigma_y is not None) or (sigma_theta is not None):
            if num_radii > 1:
                print("Interpolating orientation matrices ...", end="")
            else:
                print("Interpolating orientation matrix ...", end="")
            if sigma_x is not None and sigma_x > 0:
                orient_hist = gaussian_filter1d(
                    orient_hist,
                    sigma_x * upsample_factor,
                    mode="nearest",
                    axis=1,
                    truncate=3.0,
                )
            if sigma_y is not None and sigma_y > 0:
                orient_hist = gaussian_filter1d(
                    orient_hist,
                    sigma_y * upsample_factor,
                    mode="nearest",
                    axis=2,
                    truncate=3.0,
                )
            if sigma_theta is not None and sigma_theta > 0:
                orient_hist = gaussian_filter1d(
                    orient_hist, sigma_theta / dtheta_deg, mode="wrap", axis=3, truncate=2.0
                )
            print(" done.")
    
        # Normalization
        if normalize_intensity_stack is True:
            stack_max = np.max(orient_hist)
            if stack_max > 0:
                orient_hist = orient_hist / stack_max
        elif normalize_intensity_image is True:
            for a0 in range(num_radii):
                image_max = np.max(orient_hist[a0, :, :, :])
                if image_max > 0:
                    orient_hist[a0, :, :, :] /= image_max
    
        return orient_hist

    def calculate_orientation_correlation(
        self,
        orient_hist,
        radius_max=None,
        pairs="all",
        backend="auto",
        device=None,
        mode_batch_size=None,
        pair_batch_size=None,
        max_memory_fraction=0.6,
        dtype="float32",
        workers=None,
        zero_policy="nan",
        return_numpy=True,
        store_result=True,
        progress_bar=True,
    ):
        """
        Calculate distance-angle correlations from an orientation histogram.

        This method is mathematically equivalent to constructing the full
        ``(dx, dy, relative_theta)`` correlation volume, but processes angular
        Fourier modes in batches and performs the radial integration before the
        angular inverse transform. This substantially reduces peak memory and
        allows the FFT work to run on a GPU.

        Parameters
        ----------
        orient_hist : numpy.ndarray or torch.Tensor
            Histogram with shape ``(radial_bin, scan_x, scan_y, theta)``.
            A three-dimensional ``(scan_x, scan_y, theta)`` input is treated as
            a single radial bin.
        radius_max : int, optional
            Maximum spatial separation in orientation-histogram pixels.
            Defaults to half of the smaller scan dimension.
        pairs : {"all", "autocorrelation"} or sequence of tuple[int, int]
            Radial-bin pairs to correlate. ``"all"`` uses upper-triangular
            ordering; ``"autocorrelation"`` calculates only ``(i, i)``.
        backend : {"auto", "numpy", "torch"}
            ``"auto"`` uses PyTorch when CUDA is available and NumPy otherwise.
        device : str or torch.device, optional
            PyTorch device. Defaults to CUDA when available, otherwise CPU.
        mode_batch_size, pair_batch_size : int, optional
            Angular-frequency and radial-pair batch sizes. CUDA mode batching is
            automatically sized from available memory when omitted.
        max_memory_fraction : float
            Fraction of currently free CUDA memory available to automatic
            batching.
        dtype : {"float32", "float64"}
            Real computation dtype. ``float32`` is recommended for CUDA.
        workers : int, optional
            Number of SciPy FFT workers for the NumPy backend.
        zero_policy : {"nan", "zero", "raise"}
            Handling for radial distances with no normalization signal.
        return_numpy : bool
            Convert PyTorch output to a NumPy array before returning.
        store_result : bool
            Store output in ``self.orient_corr`` and its radial-bin mapping in
            ``self.orient_corr_pairs``.
        progress_bar : bool
            Display progress over angular-mode and radial-pair batches.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Array with shape
            ``(num_pairs, num_theta // 2 + 1, radius_max + 1)`` in multiples of
            a random distribution. A value of 1 indicates random association.

        Notes
        -----
        The full ``pairs="all"`` output uses upper-triangular radial-bin pair
        ordering. Use ``self.orient_corr_pairs`` to label the first output axis.
        """
        orient_corr, pair_indices = _calculate_orientation_correlation(
            orient_hist,
            radius_max=radius_max,
            pairs=pairs,
            backend=backend,
            device=device,
            mode_batch_size=mode_batch_size,
            pair_batch_size=pair_batch_size,
            max_memory_fraction=max_memory_fraction,
            dtype=dtype,
            workers=workers,
            zero_policy=zero_policy,
            return_numpy=return_numpy,
            progress_bar=progress_bar,
        )
        if store_result:
            self.orient_corr = orient_corr
            self.orient_corr_pairs = pair_indices
        return orient_corr

    def plot_orientation_correlation(
        self,
        orient_corr=None,
        *,
        pair_indices=None,
        pixel_size=1.0,
        pixel_units="scan pixels",
        probability_range=(0.5, 2.0),
        cmap="correlation",
        figsize=None,
        show_metrics=True,
        return_metrics=False,
        slope_weight_scale=None,
    ):
        """Plot distance-orientation correlations using Matplotlib.

        The 50% boundary is halfway between the correlation at zero separation
        and the random-association baseline of one. Its intercepts give the
        radial and annular 50% distances. The signed slope is fitted separately
        to the primary correlation-equals-one boundary between positive
        correlation and anticorrelation.

        Parameters
        ----------
        slope_weight_scale : float, optional
            1/e decay length, in ``pixel_units``, of the exponential weighting
            applied to the slope fit. The correlation-equals-one boundary
            saturates with distance, so an unweighted straight line over the
            whole lobe is dominated by the flat tail: it biases the slope low
            and pushes the fitted intercept off the measured boundary at zero
            separation. Weighting towards short distances makes ``slope`` the
            near-origin tangent instead. Defaults to
            ``SLOPE_WEIGHT_FRACTION`` times the fitted distance span. Pass
            ``numpy.inf`` to restore the previous unweighted full-lobe fit.

        Notes
        -----
        Each metrics entry reports ``slope_fit_intercept_degrees`` (compare it
        against the boundary at zero separation to check the fit),
        ``slope_fit_effective_point_count`` (Kish effective sample size, which
        falls as the weighting sharpens), and the resolved
        ``slope_weight_scale``. ``slope_fit_r_squared`` is weighted with the
        same weights, so it is not comparable to an unweighted R-squared.
        """
        from matplotlib.colors import LinearSegmentedColormap, LogNorm
        from matplotlib.lines import Line2D

        def crossing(coordinates, profile, level):
            profile = np.asarray(profile, dtype=float)
            coordinates = np.asarray(coordinates, dtype=float)
            if not np.isfinite(profile[0]):
                return np.nan
            initial_side = profile[0] - level
            if initial_side == 0:
                return float(coordinates[0])
            for point in range(1, len(profile)):
                before, after = profile[point - 1], profile[point]
                if not np.isfinite(before) or not np.isfinite(after):
                    continue
                before_side = before - level
                after_side = after - level
                if before_side == 0:
                    return float(coordinates[point - 1])
                if before_side * after_side <= 0:
                    if before == after:
                        return float(coordinates[point])
                    fraction = -before_side / (after_side - before_side)
                    return float(
                        coordinates[point - 1]
                        + fraction * (coordinates[point] - coordinates[point - 1])
                    )
            return np.nan

        values = self.orient_corr if orient_corr is None else orient_corr
        if values is None:
            raise RuntimeError(
                "No orientation correlation is available. Run "
                "calculate_orientation_correlation() first or pass orient_corr."
            )
        values = np.asarray(values)
        if values.ndim != 3:
            raise ValueError(
                "orient_corr must have shape (pair, relative_angle, distance)."
            )
        if values.shape[0] == 0:
            raise ValueError("orient_corr must contain at least one radial-bin pair.")
        if values.shape[1] < 2 or values.shape[2] < 2:
            raise ValueError(
                "orient_corr requires at least two angle and two distance samples."
            )
        labels = self.orient_corr_pairs if pair_indices is None else pair_indices
        if labels is not None:
            labels = np.asarray(labels)
            if labels.shape != (values.shape[0], 2):
                raise ValueError(
                    f"pair_indices must have shape ({values.shape[0]}, 2)."
                )

        panel_count = values.shape[0]
        column_count = min(3, max(1, panel_count))
        row_count = int(np.ceil(panel_count / column_count))
        if figsize is None:
            figsize = (4.5 * column_count, 3.8 * row_count)
        fig, axes = plt.subplots(
            row_count,
            column_count,
            figsize=figsize,
            squeeze=False,
            constrained_layout=True,
        )
        lower, upper = map(float, probability_range)
        if not 0 < lower < upper:
            raise ValueError("probability_range must satisfy 0 < lower < upper.")
        if cmap == "correlation":
            cmap = LinearSegmentedColormap.from_list(
                "quantem_correlation",
                [
                    (0.00, "#002b9a"),
                    (0.32, "#1769e8"),
                    (0.50, "#b8b8b8"),
                    (0.68, "#f23838"),
                    (1.00, "#9e0015"),
                ],
            )
        distance_max = (values.shape[2] - 1) * float(pixel_size)
        distances = np.arange(values.shape[2], dtype=float) * float(pixel_size)
        angles = np.linspace(0.0, 180.0, values.shape[1])
        image = None
        metrics = []
        for index, ax in enumerate(axes.flat):
            if index >= panel_count:
                ax.set_visible(False)
                continue
            image = ax.imshow(
                values[index],
                origin="lower",
                aspect="auto",
                extent=(0, distance_max, 0, 180),
                norm=LogNorm(vmin=lower, vmax=upper),
                cmap=cmap,
            )
            if labels is None:
                title = f"Ring pair {index}"
                pair = (index, index)
            else:
                pair = tuple(int(value) for value in labels[index])
                title = (
                    f"Autocorrelation of Ring {pair[0]}"
                    if pair[0] == pair[1]
                    else f"Correlation of Rings {pair[0]} and {pair[1]}"
                )
            ax.set(
                title=title,
                xlabel=f"distance ({pixel_units})",
                ylabel="relative orientation (degrees)",
            )
            panel = np.asarray(values[index], dtype=float)
            origin_probability = panel[0, 0]
            half_probability = (
                1.0 + 0.5 * (origin_probability - 1.0)
                if np.isfinite(origin_probability)
                and origin_probability > 0.0
                and not np.isclose(origin_probability, 1.0)
                else np.nan
            )
            radial_distance = np.nan
            annular_distance = np.nan
            slope = np.nan
            slope_fit_r_squared = np.nan
            slope_fit_point_count = 0
            slope_fit_intercept = np.nan
            slope_fit_effective_count = np.nan
            weight_scale = np.nan
            fit_distances = np.array([])
            fit_angles = np.array([])
            if np.isfinite(half_probability):
                radial_distance = crossing(
                    distances, panel[0, :], half_probability
                )
                annular_distance = crossing(
                    angles, panel[:, 0], half_probability
                )
                if np.isfinite(radial_distance):
                    ax.scatter(
                        [radial_distance],
                        [0],
                        marker="o",
                        s=45,
                        facecolor="white",
                        edgecolor="black",
                        linewidth=0.8,
                        zorder=5,
                    )
                if np.isfinite(annular_distance):
                    ax.scatter(
                        [0],
                        [annular_distance],
                        marker="D",
                        s=40,
                        facecolor="white",
                        edgecolor="black",
                        linewidth=0.8,
                        zorder=5,
                    )

            # The slope belongs to the gray probability/random = 1 boundary, not
            # the half-maximum contour used for the two distance intercepts.
            baseline_boundary = np.array(
                [
                    crossing(angles, panel[:, radius], 1.0)
                    for radius in range(panel.shape[1])
                ]
            )
            baseline_radial_intercept = crossing(distances, panel[0, :], 1.0)
            valid = np.isfinite(baseline_boundary)
            if np.isfinite(baseline_radial_intercept):
                valid &= distances <= baseline_radial_intercept + float(pixel_size)

            # Select the earliest contiguous run: this follows the principal
            # red/blue lobe from the angular axis and rejects remote closed loops.
            valid_indices = np.flatnonzero(valid)
            primary_indices = np.array([], dtype=int)
            if valid_indices.size:
                angular_jump = np.abs(
                    np.diff(baseline_boundary[valid_indices])
                )
                maximum_step = max(15.0, 4.0 * (angles[1] - angles[0]))
                split_points = np.flatnonzero(
                    (np.diff(valid_indices) > 1) | (angular_jump > maximum_step)
                ) + 1
                segments = np.split(
                    valid_indices, split_points
                )
                primary_indices = next(
                    (segment for segment in segments if len(segment) >= 2),
                    np.array([], dtype=int),
                )
            if primary_indices.size >= 5:
                # A connected correlation=1 contour can rise away from the
                # origin, turn around at large distance, and return as part of
                # the same loop. Fitting that entire loop can reverse the sign
                # of the visually obvious near-origin boundary. Stop at the
                # first sustained turning point while tolerating isolated
                # pixel-scale contour noise.
                boundary_run = baseline_boundary[primary_indices]
                smoothing_sigma = min(3.0, max(0.75, len(boundary_run) / 100.0))
                boundary_smooth = gaussian_filter1d(
                    boundary_run, sigma=smoothing_sigma, mode="nearest"
                )
                boundary_gradient = np.gradient(boundary_smooth)
                initial_count = min(20, max(3, len(boundary_run) // 10))
                initial_trend = float(
                    np.median(boundary_gradient[:initial_count])
                )
                if not np.isclose(initial_trend, 0.0):
                    reversal = boundary_gradient * np.sign(initial_trend) < 0
                    persistence = min(5, max(2, len(boundary_run) // 20))
                    sustained = np.convolve(
                        reversal.astype(int),
                        np.ones(persistence, dtype=int),
                        mode="valid",
                    )
                    turning_points = np.flatnonzero(sustained == persistence)
                    if turning_points.size and turning_points[0] >= 2:
                        primary_indices = primary_indices[
                            : turning_points[0] + 1
                        ]
            if primary_indices.size:
                fit_distances = distances[primary_indices]
                fit_values = baseline_boundary[primary_indices]

                # The correlation=1 boundary saturates: it climbs steeply near the
                # origin and flattens at large separation. An unweighted straight
                # line over the whole lobe is therefore dominated by the flat tail,
                # which drags the intercept off the measured boundary at d=0 (up to
                # ~7 degrees on real data, visibly landing in the blue region) and
                # biases the slope low. Weight the fit towards short distances so
                # `slope` is the near-origin tangent, which is the physically
                # meaningful quantity.
                fit_span = float(fit_distances.max() - fit_distances.min())
                if slope_weight_scale is None:
                    weight_scale = SLOPE_WEIGHT_FRACTION * fit_span
                else:
                    weight_scale = float(slope_weight_scale)
                if not np.isfinite(weight_scale) or weight_scale <= 0:
                    # np.inf (or a non-positive scale) restores the legacy
                    # unweighted fit over the full lobe.
                    fit_weights = np.ones_like(fit_distances)
                    weight_scale = np.inf
                else:
                    fit_weights = np.exp(
                        -(fit_distances - fit_distances.min()) / weight_scale
                    )

                root_weights = np.sqrt(fit_weights)
                design = (
                    np.vstack([fit_distances, np.ones_like(fit_distances)]).T
                    * root_weights[:, None]
                )
                fit_slope, fit_intercept = np.linalg.lstsq(
                    design, fit_values * root_weights, rcond=None
                )[0]
                fit_angles = fit_intercept + fit_slope * fit_distances
                slope = float(fit_slope)
                slope_fit_intercept = float(fit_intercept)
                slope_fit_point_count = int(fit_distances.size)
                # Kish effective sample size: how many points the weighting really
                # uses, so a too-aggressive scale is visible rather than silent.
                slope_fit_effective_count = float(
                    fit_weights.sum() ** 2 / np.sum(fit_weights**2)
                )
                weight_mean = float(
                    np.average(fit_values, weights=fit_weights)
                )
                residual_sum_squares = float(
                    np.sum(fit_weights * (fit_values - fit_angles) ** 2)
                )
                total_sum_squares = float(
                    np.sum(fit_weights * (fit_values - weight_mean) ** 2)
                )
                slope_fit_r_squared = (
                    1.0 - residual_sum_squares / total_sum_squares
                    if total_sum_squares > 0
                    else np.nan
                )

            if fit_distances.size:
                # Draw only where the weighting actually constrains the line. Past
                # ~3 decay lengths the boundary has saturated away from this
                # tangent, and extending the line there would misrepresent the fit.
                if np.isfinite(weight_scale):
                    drawn = fit_distances <= (
                        fit_distances.min() + 3.0 * weight_scale
                    )
                    if drawn.sum() < 2:
                        drawn = np.zeros_like(fit_distances, dtype=bool)
                        drawn[: min(2, drawn.size)] = True
                else:
                    drawn = np.ones_like(fit_distances, dtype=bool)
                visible_fit = drawn & (fit_angles >= 0) & (fit_angles <= 180)
                ax.plot(
                    fit_distances[visible_fit],
                    fit_angles[visible_fit],
                    color="#ffe600",
                    linestyle="-",
                    linewidth=2.5,
                    zorder=6,
                )
            ax.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        markerfacecolor="white",
                        markeredgecolor="black",
                        label="50% radial intercept",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="D",
                        color="none",
                        markerfacecolor="white",
                        markeredgecolor="black",
                        label="50% annular intercept",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color="#ffe600",
                        linewidth=2.5,
                        label="near-origin baseline fit",
                    ),
                ],
                loc="lower right",
                fontsize=7,
                framealpha=0.82,
            )

            panel_metrics = {
                "pair": pair,
                "title": title,
                "half_probability": float(half_probability),
                "radial_distance": float(radial_distance),
                "annular_distance_degrees": float(annular_distance),
                "slope_degrees_per_unit": float(slope),
                "slope_fit_r_squared": float(slope_fit_r_squared),
                "slope_fit_point_count": slope_fit_point_count,
                "slope_fit_intercept_degrees": float(slope_fit_intercept),
                "slope_fit_effective_point_count": float(
                    slope_fit_effective_count
                ),
                "slope_weight_scale": float(weight_scale),
                "slope_contour_probability": 1.0,
                "distance_units": pixel_units,
            }
            metrics.append(panel_metrics)
            if show_metrics:
                radial_text = (
                    f"{radial_distance:.2f} {pixel_units}"
                    if np.isfinite(radial_distance)
                    else "not resolved"
                )
                annular_text = (
                    f"{annular_distance:.2f} degrees"
                    if np.isfinite(annular_distance)
                    else "not resolved"
                )
                slope_text = (
                    f"{slope:.2f} degrees/{pixel_units}"
                    if np.isfinite(slope)
                    else "not resolved"
                )
                ax.text(
                    0.98,
                    0.98,
                    "50% radial distance = "
                    + radial_text
                    + "\n50% annular distance = "
                    + annular_text
                    + "\nslope = "
                    + slope_text,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox={
                        "boxstyle": "round,pad=0.35",
                        "facecolor": "white",
                        "edgecolor": "black",
                        "alpha": 0.82,
                    },
                )
        if image is not None:
            fig.colorbar(
                image,
                ax=[ax for ax in axes.flat if ax.get_visible()],
                label="probability / random",
            )
        if return_metrics:
            return fig, axes, metrics
        return fig, axes

    def plot_interactive_image_map(self, *args, **kwargs):
        """Interactive diffraction-pattern browser.

        Forwards to :func:`quantem.diffraction.peak_visualization.plot_interactive_image_map`,
        supplying this object's data. See that function for the parameters.
        """

        return _plot_interactive_image_map(self.dataset_cartesian, getattr(self, "polar_data", None), *args, **kwargs)

    def save_diffraction_figures(self, *args, **kwargs):
        """Export the pattern figures at one scan position.

        Forwards to :func:`quantem.diffraction.peak_visualization.save_diffraction_figures`,
        supplying this object's data. See that function for the parameters.
        """

        return _save_diffraction_figures(self.dataset_cartesian, getattr(self, "polar_data", None), *args, **kwargs)
            
    def show_widget(self, **kwargs):
        """Open the interactive polymer 4D-STEM viewer (``quantem.widget``).

        Thin wrapper over ``quantem.widget.show_polymer_4DSTEM``: drag the map to update
        the Current/Lamellar/Backbone/pi-pi DP panels and the polar view, with detected
        peaks overlaid when ``find_peaks_model`` has run. All ``**kwargs`` are forwarded
        to the factory (e.g. ``intensity_map``, ``map_cmap``, ``dp_cmap``, ``show_polar``,
        ``title``).
        """
        from quantem.widget import show_polymer_4DSTEM
        return show_polymer_4DSTEM(self, **kwargs)

    def plot_interactive_peak_map(self, radial_range=None, intensity_map=None,
                                    ry=None, rx=None,
                                    vmax_cartesian=7, vmin_cartesian=0, show_all_peaks=True,
                                    selected_peak_color='red', other_peak_color='gray',
                                    central_beam_color='red',
                                    norm_upper_quantile=None, norm_power=1.0,
                                    peak_intensity_mode='size', peak_size_range=(30, 300),
                                    peak_cmap='hot', peak_vmin=None, peak_vmax=None,
                                    show_polar=True, vmax_polar=None, two_fold_symmetry=True,
                                    map_cmap="viridis", dp_cmap="gray", intensity_field='intensities',
                                    crosshair_color='r', figsize=None, crosshair_width=2, crosshair_size=15,
                                    crosshair_width_peaks=2, crosshair_scaling_peaks=1, crosshair_scaling_central_beam=1,
                                    gaussian_filter_sigma=None, zoom=1):
        """
        Interactive plot for browsing diffraction patterns with peak overlay.
        Central beam (closest to image center) plotted in blue.
        """
        if figsize is None:
            if show_polar:
                figsize = (15, 4)
            else:
                figsize = (12, 5)
        Ry, Rx = self.peak_coordinates_cartesian.shape
        
        if show_polar and not (hasattr(self, 'polar_data') and self.polar_data is not None):
            print("Warning: polar_data not found. Set show_polar=False or run polar_transform_4d first.")
            show_polar = False
        
        # Setup intensity map
        if intensity_map is not None:
            intensity_map, upsample_factor = _resolve_intensity_map(
                self.dataset_cartesian,
                intensity_map,
                (Ry, Rx),
                validate=False,
            )
            map_title = f'Custom Map ({radial_range[0]:.2f}-{radial_range[1]:.2f} 1/Å)' if radial_range else 'Custom Map'
        else:
            intensity_map, upsample_factor = _resolve_intensity_map(
                self.dataset_cartesian,
                intensity_map,
                (Ry, Rx),
                validate=False,
            )
            map_title = f'Peak Map ({radial_range[0]:.2f}-{radial_range[1]:.2f} 1/Å)' if radial_range else 'Peak Map'
        
        is_rgb_map, vmin_intensity_map, vmax_intensity_map = _intensity_display_limits(
            intensity_map
        )
        
        vmax_polar = vmax_polar or vmax_cartesian
        
        # Peak plotting function
        def plot_peaks_on_ax(
            ax,
            peaks_x,
            peaks_y,
            peaks_r_invA,
            peak_intensities,
            central_idx,
            ry_data,
            rx_data,
            center=None,
        ):
            _plot_bragg_peaks_on_ax(
                ax,
                peaks_x,
                peaks_y,
                peaks_r_invA,
                peak_intensities,
                central_idx,
                radial_range=radial_range,
                show_all_peaks=show_all_peaks,
                selected_peak_color=selected_peak_color,
                other_peak_color=other_peak_color,
                central_beam_color=central_beam_color,
                peak_intensity_mode=peak_intensity_mode,
                peak_size_range=peak_size_range,
                peak_cmap=peak_cmap,
                peak_vmin=peak_vmin,
                peak_vmax=peak_vmax,
                crosshair_width_peaks=crosshair_width_peaks,
                crosshair_scaling_peaks=crosshair_scaling_peaks,
                crosshair_scaling_central_beam=crosshair_scaling_central_beam,
                add_colorbar=True,
                center=center,
            )
        
        # Interactive callback
        def show_pattern(ry_slider, rx_slider):
            ry_data = ry_slider // upsample_factor
            rx_data = rx_slider // upsample_factor
            fig, axes = plt.subplots(1, 3 if show_polar else 2, figsize=figsize)
            ax1, ax2 = axes[0], axes[1]
            ax3 = axes[2] if show_polar else None
            
            # Intensity map
            if vmin_intensity_map is None:
                im1 = ax1.imshow(intensity_map, cmap=map_cmap)
            else:
                im1 = ax1.imshow(intensity_map, cmap=map_cmap, vmin=vmin_intensity_map, vmax=vmax_intensity_map)
            ax1.scatter(rx_slider, ry_slider,  facecolor='none', edgecolor=crosshair_color, marker='o', s=crosshair_size, linewidth=crosshair_width)
            ax1.set_title(map_title)
            ax1.set_xlabel('Rx (upsampled)' if upsample_factor > 1 else 'Rx')
            ax1.set_ylabel('Ry (upsampled)' if upsample_factor > 1 else 'Ry')
            if not is_rgb_map:
                plt.colorbar(im1, ax=ax1)
            
            # Create inset axes for the zoomed view
            axins = inset_axes(ax1, width="30%", height="30%", loc='upper right', 
                               borderpad=1.5)
            
            # Calculate 9x9 region with selected pixel at center (4 pixels margin each side)
            margin = 4
            ry_min = max(0, ry_slider - margin)
            ry_max = min(intensity_map.shape[0], ry_slider + margin + 1)
            rx_min = max(0, rx_slider - margin)
            rx_max = min(intensity_map.shape[1], rx_slider + margin + 1)
            
            # Extract and display the zoomed region
            zoomed_region = intensity_map[ry_min:ry_max, rx_min:rx_max]
            
            if vmin_intensity_map is None:
                axins.imshow(zoomed_region, cmap=map_cmap, 
                             extent=[rx_min, rx_max, ry_max, ry_min], 
                             interpolation='nearest')
            else:
                axins.imshow(zoomed_region, cmap=map_cmap, 
                             extent=[rx_min, rx_max, ry_max, ry_min],
                             vmin=vmin_intensity_map, vmax=vmax_intensity_map,
                             interpolation='nearest')
            
            # Draw border around the selected (central) pixel
            pixel_border = Rectangle((rx_slider, ry_slider), 1, 1,
                                     linewidth=2, edgecolor=crosshair_color, 
                                     facecolor='none', zorder=10)
            axins.add_patch(pixel_border)
            
            # Set limits and styling
            axins.set_xlim(rx_min, rx_max)
            axins.set_ylim(ry_max, ry_min)
            axins.set_xticks([])
            axins.set_yticks([])
            axins.set_title('9×9 zoom', fontsize=8, pad=2)
            
            # Optional: Add a rectangle on main plot showing zoomed region
            rect = Rectangle((rx_min, ry_min), rx_max-rx_min, ry_max-ry_min,
                             linewidth=1.5, edgecolor=crosshair_color, 
                             facecolor='none', linestyle='--', alpha=0.7)
            ax1.add_patch(rect)

            
            # Diffraction pattern
            dp_data = _normalized_dp(
                self.dataset_cartesian,
                ry_data,
                rx_data,
                norm_upper_quantile=norm_upper_quantile,
                norm_power=norm_power,
            )
            im_polar_data = self.polar_data['intensity'][ry_data, rx_data].T if show_polar else None
            if gaussian_filter_sigma is not None:
                dp_data = gaussian_filter(dp_data, gaussian_filter_sigma)
                if show_polar:
                    im_polar_data = gaussian_filter(im_polar_data, gaussian_filter_sigma)
            
            peaks_r_invA = _vector_field_cell(self.polar_peaks, "r_invA", ry_data, rx_data)
            peaks_y = _vector_field_cell(
                self.peak_coordinates_cartesian, "y_pixels", ry_data, rx_data
            )
            peaks_x = _vector_field_cell(
                self.peak_coordinates_cartesian, "x_pixels", ry_data, rx_data
            )
            peak_ints = _vector_field_cell(
                self.peak_intensities, intensity_field, ry_data, rx_data
            )
            has_peak_positions = _has_peak_positions(peaks_x, peaks_y)
            center = _display_center(
                getattr(self, "image_centers", None), ry_data, rx_data, dp_data.shape
            )
            central_idx = _central_peak_index(
                peaks_x, peaks_y, peaks_r_invA, center,
                max_dist=_central_beam_max_dist(dp_data.shape),
            )
            (
                dp_data,
                peaks_x,
                peaks_y,
                peaks_r_invA,
                peak_ints,
                central_idx,
                display_center,
            ) = _zoom_peak_overlay(
                dp_data,
                peaks_x,
                peaks_y,
                peaks_r_invA,
                peak_ints,
                central_idx,
                zoom,
                center,
            )

            im2 = ax2.imshow(dp_data, cmap=dp_cmap, vmax=vmax_cartesian, vmin=vmin_cartesian)
            ax2.set_xticks([])
            ax2.set_yticks([])
                
            plot_peaks_on_ax(
                ax2,
                peaks_x,
                peaks_y,
                peaks_r_invA,
                peak_ints,
                central_idx,
                ry_data,
                rx_data,
                center=display_center,
            )
            ax2.set_xlim(-0.5, dp_data.shape[1] - 0.5)
            ax2.set_ylim(dp_data.shape[0] - 0.5, -0.5)
            
            title = f'Diffraction Pattern (Ry={ry_data}, Rx={rx_data})'
            if radial_range:
                title += f'\n{radial_range[0]:.2f}-{radial_range[1]:.2f} 1/Å'
            if not has_peak_positions:
                title += '\nNo peaks at this scan position'
            ax2.set_title(title)
            
            # Polar transform
            if show_polar:
                im3 = ax3.imshow(im_polar_data, 
                                cmap=dp_cmap, vmax=vmax_polar, aspect='auto')
                # ax3.set_aspect('equal', adjustable='box')                
                ax3.set_xlabel('Radius (bins)')
                ax3.set_ylabel('Theta (bins)')
                ax3.set_title(f'Polar (Ry={ry_data}, Rx={rx_data})')
                
                if hasattr(self, 'polar_peaks') and self.polar_peaks is not None:
                    polar_r = _vector_field_cell(
                        self.polar_peaks, "r_invA", ry_data, rx_data
                    )
                    polar_theta = _vector_field_cell(
                        self.polar_peaks, "theta", ry_data, rx_data
                    )
                    if polar_r is not None and len(polar_r) > 0:
                        r_bins, theta_bins = _polar_peak_bins(
                            polar_r,
                            polar_theta,
                            self.max_radius_invA,
                            self.num_radial_bins,
                            self.num_annular_bins,
                            two_fold_symmetry,
                        )
                        plot_peaks_on_ax(ax3, r_bins, theta_bins, polar_r, peak_ints, central_idx, ry_data, rx_data)
            
            plt.tight_layout()
            plt.show()
        
        # Widgets
        if ry is None:
            ry = Ry*upsample_factor//2
        if rx is None:
            rx = Rx*upsample_factor//2
        ry_slider = IntSlider(min=0, max=Ry*upsample_factor-1, value=ry, description='Ry:', continuous_update=False)
        rx_slider = IntSlider(min=0, max=Rx*upsample_factor-1, value=rx, description='Rx:', continuous_update=False)
        interactive_plot = interactive_output(show_pattern, {'ry_slider': ry_slider, 'rx_slider': rx_slider})
        display(VBox([HBox([ry_slider, rx_slider]), interactive_plot]))
        
    def save_peak_figures(self, ry, rx, intensity_map=None,
                         map_title="", prefix='peaks', save_dir='.',
                         vmax_cartesian=7, vmin_cartesian=0,
                         selected_peak_color='red',
                         central_beam_color='red',
                         norm_upper_quantile=None, norm_power=1.0,
                         peak_intensity_mode='size', peak_size_range=(30, 300),
                         peak_cmap='hot', peak_vmin=None, peak_vmax=None,
                         show_polar=True, vmax_polar=None, two_fold_symmetry=True,
                         map_cmap="viridis", dp_cmap="gray", intensity_field='intensities',
                         crosshair_color='r', figsize_individual=None, figsize_combined=None, 
                         crosshair_width=2, crosshair_size=15, crosshair_width_peaks=2,
                         crosshair_scaling_peaks=1, crosshair_scaling_central_beam=1, peak_marker="o",
                         peak_marker_facecolors='none', peak_marker_size=None, gaussian_filter_sigma=None,
                         zoom=1, peak_alpha=1.0, central_linewidth=None,
                         peaks_x=None, peaks_y=None, peak_ints=None, peaks_r_invA=None,
                         central_idx=None, show_central_beam=True,
                         save_intensity_map=True, save_diffraction=True, save_polar=None,
                         dpi=400):
        """
        Save peak-annotated diffraction figures for a specific scan position.
        Central beam (closest to image center) plotted in blue.

        Peaks are read from the precomputed ``peak_coordinates_cartesian`` /
        ``peak_intensities`` / ``polar_peaks`` by default. Pass ``peaks_x`` / ``peaks_y``
        / ``peak_ints`` (and optionally ``peaks_r_invA`` / ``central_idx``) to inject
        peaks directly instead — e.g. from live single-DP inference, where no scan-wide
        peak arrays exist. ``save_intensity_map`` / ``save_diffraction`` / ``save_polar``
        select which figures to write (``save_polar=None`` follows ``show_polar``); this
        lets a caller save the context map once and the DP per panel.
        """

        override_peaks = peaks_x is not None
        if self.peak_coordinates_cartesian is not None:
            Ry, Rx = self.peak_coordinates_cartesian.shape
        else:
            Ry, Rx = int(self.dataset_cartesian.shape[0]), int(self.dataset_cartesian.shape[1])

        if not (0 <= ry < Ry and 0 <= rx < Rx):
            raise ValueError(f"Coordinates ({ry}, {rx}) out of bounds")

        if save_polar is not None:
            show_polar = bool(save_polar)
        if show_polar and not (hasattr(self, 'polar_data') and self.polar_data is not None):
            print("Warning: polar_data not found. Skipping polar save.")
            show_polar = False
        
        intensity_map, upsample_factor = _resolve_intensity_map(
            self.dataset_cartesian,
            intensity_map,
            (Ry, Rx),
            validate=False,
        )
        
        _is_rgb_map, vmin_intensity_map, vmax_intensity_map = _intensity_display_limits(
            intensity_map
        )
        
        vmax_polar = vmax_polar or vmax_cartesian
        
        # Peak plotting function
        def plot_peaks_on_ax(ax, peaks_x, peaks_y, peaks_r_invA, peak_intensities, central_idx, center=None):
            _plot_bragg_peaks_on_ax(
                ax,
                peaks_x,
                peaks_y,
                peaks_r_invA,
                peak_intensities,
                central_idx,
                selected_peak_color=selected_peak_color,
                central_beam_color=central_beam_color,
                peak_intensity_mode=peak_intensity_mode,
                peak_size_range=peak_size_range,
                peak_cmap=peak_cmap,
                peak_vmin=peak_vmin,
                peak_vmax=peak_vmax,
                crosshair_width_peaks=crosshair_width_peaks,
                crosshair_scaling_peaks=crosshair_scaling_peaks,
                crosshair_scaling_central_beam=crosshair_scaling_central_beam,
                peak_marker=peak_marker,
                peak_marker_facecolors=peak_marker_facecolors,
                peak_marker_size=peak_marker_size,
                peak_alpha=peak_alpha,
                central_alpha=peak_alpha,
                central_linewidth=(
                    crosshair_width_peaks if central_linewidth is None else central_linewidth
                ),
                center=center,
                show_central_beam=show_central_beam,
            )
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Get peaks data once (injected overrides win; otherwise read precomputed).
        if override_peaks:
            peaks_x = np.asarray(peaks_x)
            peaks_y = np.asarray(peaks_y)
            peak_ints = None if peak_ints is None else np.asarray(peak_ints)
            peaks_r_invA = None if peaks_r_invA is None else np.asarray(peaks_r_invA)
        else:
            peaks_y = _vector_field_cell(self.peak_coordinates_cartesian, "y_pixels", ry, rx)
            peaks_x = _vector_field_cell(self.peak_coordinates_cartesian, "x_pixels", ry, rx)
            peak_ints = _vector_field_cell(self.peak_intensities, intensity_field, ry, rx)
            peaks_r_invA = (
                _vector_field_cell(self.polar_peaks, "r_invA", ry, rx)
                if getattr(self, 'polar_peaks', None) is not None
                else None
            )
        dp_data = _normalized_dp(
            self.dataset_cartesian,
            ry,
            rx,
            norm_upper_quantile=norm_upper_quantile,
            norm_power=norm_power,
        )
        polar_im_data = self.polar_data['intensity'][ry, rx].T if show_polar else None
        if gaussian_filter_sigma is not None:
            dp_data = gaussian_filter(dp_data, gaussian_filter_sigma)
            if show_polar:
                polar_im_data = gaussian_filter(polar_im_data, gaussian_filter_sigma)

        center = _display_center(getattr(self, "image_centers", None), ry, rx, dp_data.shape)
        if central_idx is None:
            central_idx = _central_peak_index(
                peaks_x, peaks_y, peaks_r_invA, center,
                max_dist=_central_beam_max_dist(dp_data.shape),
            )
        
        (
            dp_data,
            peaks_x,
            peaks_y,
            peaks_r_invA,
            peak_ints,
            central_idx,
            display_center,
        ) = _zoom_peak_overlay(
            dp_data,
            peaks_x,
            peaks_y,
            peaks_r_invA,
            peak_ints,
            central_idx,
            zoom,
            center,
        )
            
        # Save intensity map
        if figsize_individual is None:
            figsize_individual = (6, 6)
        if save_intensity_map:
            fig_map, ax = plt.subplots(figsize=figsize_individual)
            if vmin_intensity_map is None:
                im = ax.imshow(intensity_map, cmap=map_cmap)
            else:
                im = ax.imshow(intensity_map, cmap=map_cmap, vmin=vmin_intensity_map, vmax=vmax_intensity_map)

            ry_slider = ry * upsample_factor
            rx_slider = rx * upsample_factor

            ax.scatter(rx_slider, ry_slider, facecolor='none', edgecolor=crosshair_color, marker='o', s=crosshair_size, linewidth=crosshair_width)

            # Add inset
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            axins = inset_axes(ax, width="30%", height="30%", loc='upper right', borderpad=1.5)

            margin = 4
            ry_min = max(0, ry_slider - margin)
            ry_max = min(intensity_map.shape[0], ry_slider + margin + 1)
            rx_min = max(0, rx_slider - margin)
            rx_max = min(intensity_map.shape[1], rx_slider + margin + 1)

            zoomed_region = intensity_map[ry_min:ry_max, rx_min:rx_max]

            if vmin_intensity_map is None:
                axins.imshow(zoomed_region, cmap=map_cmap, extent=[rx_min, rx_max, ry_max, ry_min], interpolation='nearest')
            else:
                axins.imshow(zoomed_region, cmap=map_cmap, extent=[rx_min, rx_max, ry_max, ry_min],
                             vmin=vmin_intensity_map, vmax=vmax_intensity_map, interpolation='nearest')

            pixel_border = Rectangle((rx_slider, ry_slider), 1, 1, linewidth=2, edgecolor=crosshair_color,
                                     facecolor='none', zorder=10)
            axins.add_patch(pixel_border)

            axins.set_xlim(rx_min, rx_max)
            axins.set_ylim(ry_max, ry_min)
            axins.set_xticks([])
            axins.set_yticks([])
            axins.set_title('9×9 zoom', fontsize=8, pad=2)

            rect = Rectangle((rx_min, ry_min), rx_max-rx_min, ry_max-ry_min,
                             linewidth=1.5, edgecolor=crosshair_color, facecolor='none', linestyle='--', alpha=0.7)
            ax.add_patch(rect)

            ax.set_title(map_title)
            ax.set_xlabel('Rx (upsampled)' if upsample_factor > 1 else 'Rx')
            ax.set_ylabel('Ry (upsampled)' if upsample_factor > 1 else 'Ry')
            fig_map.savefig(save_path / f'{prefix}_ry{ry}_rx{rx}_intensity_map.pdf', format='pdf', bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close(fig_map)
            print(f'✓ Saved: {prefix}_ry{ry}_rx{rx}_intensity_map.pdf')

        # Save diffraction pattern with peaks
        if save_diffraction:
            fig_diff, ax = plt.subplots(figsize=figsize_individual)
            im = ax.imshow(dp_data, cmap=dp_cmap, vmax=vmax_cartesian, vmin=vmin_cartesian)
            ax.set_xticks([])
            ax.set_yticks([])
            if peaks_x is not None:
                plot_peaks_on_ax(ax, peaks_x, peaks_y, peaks_r_invA, peak_ints, central_idx, center=display_center)
            ax.set_xlim(-0.5, dp_data.shape[1] - 0.5)
            ax.set_ylim(dp_data.shape[0] - 0.5, -0.5)
            ax.set_title(f'Diffraction Pattern (Ry={ry}, Rx={rx})')
            fig_diff.savefig(save_path / f'{prefix}_ry{ry}_rx{rx}_diffraction.pdf', format='pdf', bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close(fig_diff)
            print(f'✓ Saved: {prefix}_ry{ry}_rx{rx}_diffraction.pdf')
        
        # Save polar transform with peaks
        if show_polar:
            fig_polar, ax = plt.subplots(figsize=figsize_individual)
            im = ax.imshow(polar_im_data, cmap=dp_cmap, vmax=vmax_polar, aspect='auto')
            ax.set_title(f'Polar (Ry={ry}, Rx={rx})')
            ax.set_xlabel('Radius (bins)')
            ax.set_ylabel('Theta (bins)')
            
            if hasattr(self, 'polar_peaks') and self.polar_peaks is not None:
                polar_r = _vector_field_cell(self.polar_peaks, "r_invA", ry, rx)
                polar_theta = _vector_field_cell(self.polar_peaks, "theta", ry, rx)
                if polar_r is not None and len(polar_r) > 0:
                    r_bins, theta_bins = _polar_peak_bins(
                        polar_r,
                        polar_theta,
                        self.max_radius_invA,
                        self.num_radial_bins,
                        self.num_annular_bins,
                        two_fold_symmetry,
                    )
                    
                    # Find central beam for polar
                    polar_central_idx = np.argmin(polar_r)
                    # Use the full (unzoomed) intensities: polar_r / r_bins / theta_bins are read
                    # from the full polar_peaks, whereas `peak_ints` may have been subset by
                    # zoom > 1 for the Cartesian panel (length mismatch -> IndexError otherwise).
                    polar_peak_ints = _vector_field_cell(
                        self.peak_intensities, intensity_field, ry, rx
                    )
                    if polar_r is not None and len(polar_r) > 0:
                        plot_peaks_on_ax(ax, r_bins, theta_bins, polar_r, polar_peak_ints, polar_central_idx)
            fig_polar.savefig(save_path / f'{prefix}_ry{ry}_rx{rx}_polar.pdf', format='pdf', bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close(fig_polar)
            print(f'✓ Saved: {prefix}_ry{ry}_rx{rx}_polar.pdf')
    
    def save_peak_animation(
        self,
        path,
        *,
        region=None,
        step=1,
        bidirectional=True,
        fps=10,
        intensity_map=None,
        map_title="",
        map_cmap="viridis",
        crosshair_color="r",
        crosshair_size=80,
        crosshair_width=2,
        dp_cmap="gray",
        vmin_cartesian=0,
        vmax_cartesian=7,
        norm_upper_quantile=None,
        norm_power=1.0,
        gaussian_filter_sigma=None,
        zoom=1,
        show_peaks=True,
        selected_peak_color="red",
        central_beam_color="red",
        show_central_beam=True,
        peak_intensity_mode="size",
        peak_size_range=(30, 300),
        peak_marker_size=None,
        crosshair_width_peaks=2,
        crosshair_scaling_central_beam=1,
        peak_alpha=1.0,
        central_linewidth=None,
        intensity_field="intensities",
        live_inference=False,
        infer_device=None,
        sigma_peak_blur=1.0,
        threshold_peak=0.5,
        panels=None,
        figsize=None,
        dpi=100,
        progress=True,
    ):
        """Render a snaking-cursor animation to an animated GIF.

        Walks a boustrophedon (snake) path over the scan and, for each position,
        renders one combined frame: the real-space intensity map with a cursor
        crosshair at the current position (left) beside one or more diffraction-pattern
        panels with detected Bragg peaks overlaid (right). Frames are assembled into a
        looping GIF. This reuses the same rendering primitives as
        :meth:`save_peak_figures` so frames match the per-position saved figures.

        Parameters
        ----------
        path : str | pathlib.Path
            Output ``.gif`` path.
        region : tuple[int, int, int, int] | None
            ``(ry0, ry1, rx0, rx1)`` half-open scan bounds to snake over; ``None``
            covers the whole scan.
        step : int
            Stride between visited positions (>= 1).
        bidirectional : bool
            Snake/boustrophedon path (alternate row direction). ``False`` scans every
            row left->right.
        fps : float
            Playback frames per second.
        intensity_map : np.ndarray | None
            Real-space map to display (computed once). ``None`` uses the mean-intensity
            virtual image. May be scalar ``(H, W)`` or RGB ``(H, W, 3|4)``.
        panels : list[dict] | None
            One dict per diffraction-pattern panel to draw beside the map, each holding
            that panel's display settings (any of: ``title``, ``dp_cmap``,
            ``vmin_cartesian``, ``vmax_cartesian``, ``norm_upper_quantile``,
            ``norm_power``, ``gaussian_filter_sigma``, ``zoom``, ``selected_peak_color``,
            ``central_beam_color``, ``show_central_beam``, ``peak_intensity_mode``,
            ``peak_size_range``, ``peak_marker_size``, ``crosshair_width_peaks``,
            ``crosshair_scaling_central_beam``, ``peak_alpha``, ``central_linewidth``).
            Missing keys fall back to the corresponding top-level argument. ``None``
            (default) draws a single panel from the top-level arguments.
        live_inference : bool
            Run the model per position via :meth:`infer_peaks_single` instead of reading
            precomputed ``peak_coordinates_cartesian`` (slow over large regions).
        figsize : tuple | None
            Figure size. ``None`` auto-sizes to ``(5 * (1 + n_panels), 5)``.

        Returns
        -------
        pathlib.Path
            The written GIF path.
        """
        from PIL import Image

        # A single top-level panel spec unless the caller passes an explicit list.
        if panels is None:
            panels = [dict(
                title=None,
                dp_cmap=dp_cmap,
                vmin_cartesian=vmin_cartesian,
                vmax_cartesian=vmax_cartesian,
                norm_upper_quantile=norm_upper_quantile,
                norm_power=norm_power,
                gaussian_filter_sigma=gaussian_filter_sigma,
                zoom=zoom,
                selected_peak_color=selected_peak_color,
                central_beam_color=central_beam_color,
                show_central_beam=show_central_beam,
                peak_intensity_mode=peak_intensity_mode,
                peak_size_range=peak_size_range,
                peak_marker_size=peak_marker_size,
                crosshair_width_peaks=crosshair_width_peaks,
                crosshair_scaling_central_beam=crosshair_scaling_central_beam,
                peak_alpha=peak_alpha,
                central_linewidth=central_linewidth,
            )]
        n_panels = len(panels)
        if n_panels == 0:
            raise ValueError("panels must contain at least one DP panel spec")

        Ry, Rx = int(self.dataset_cartesian.shape[0]), int(self.dataset_cartesian.shape[1])
        base_shape = (int(self.dataset_cartesian.shape[2]), int(self.dataset_cartesian.shape[3]))

        # Resolve the real-space map ONCE; _mean_intensity_map rescans every DP, so
        # rebuilding it per frame would be quadratic in scan size.
        intensity_map, upsample_factor = _resolve_intensity_map(
            self.dataset_cartesian, intensity_map, (Ry, Rx), validate=False,
        )
        is_rgb_map, map_vmin, map_vmax = _intensity_display_limits(intensity_map)

        # Boustrophedon path over the requested region (mirrors Show4DSTEM.raster).
        if region is None:
            ry0, ry1, rx0, rx1 = 0, Ry, 0, Rx
        else:
            ry0, ry1, rx0, rx1 = region
            ry0, ry1 = max(0, int(ry0)), min(Ry, int(ry1))
            rx0, rx1 = max(0, int(rx0)), min(Rx, int(rx1))
        if ry1 <= ry0 or rx1 <= rx0:
            raise ValueError(f"Empty region {region!r} for scan shape ({Ry}, {Rx})")
        step = max(1, int(step))
        points = []
        for i, ry in enumerate(range(ry0, ry1, step)):
            cols = list(range(rx0, rx1, step))
            if bidirectional and i % 2 == 1:
                cols = cols[::-1]
            points.extend((ry, rx) for rx in cols)

        has_precomputed = (not live_inference) and self.peak_coordinates_cartesian is not None
        has_polar_peaks = getattr(self, "polar_peaks", None) is not None

        if figsize is None:
            figsize = (5 * (1 + n_panels), 5)
        fig, axes = plt.subplots(1, 1 + n_panels, figsize=figsize, dpi=dpi)
        ax_map = axes[0]
        dp_axes = axes[1:]
        frames = []
        try:
            for ry, rx in tqdm(points, desc="Rendering snake", disable=not progress):
                # Peaks + beam center are fetched ONCE per position; each panel then
                # applies its own normalization / zoom crop below.
                peaks_x = peaks_y = peak_ints = peaks_r_invA = None
                if show_peaks:
                    if live_inference:
                        res = self.infer_peaks_single(
                            ry, rx, device=infer_device,
                            sigma_peak_blur=sigma_peak_blur, threshold_peak=threshold_peak,
                        )
                        peaks_x, peaks_y, peak_ints = (
                            res["x_pixels"], res["y_pixels"], res["intensities"],
                        )
                    elif has_precomputed:
                        peaks_y = _vector_field_cell(self.peak_coordinates_cartesian, "y_pixels", ry, rx)
                        peaks_x = _vector_field_cell(self.peak_coordinates_cartesian, "x_pixels", ry, rx)
                        if self.peak_intensities is not None:
                            peak_ints = _vector_field_cell(self.peak_intensities, intensity_field, ry, rx)
                        if has_polar_peaks:
                            peaks_r_invA = _vector_field_cell(self.polar_peaks, "r_invA", ry, rx)

                center = _display_center(getattr(self, "image_centers", None), ry, rx, base_shape)
                # _plot_bragg_peaks_on_ax draws no rings when peaks_r_invA is None. When
                # there is no polar transform, fall back to the pixel radius from center so
                # the rings still render (r_invA is otherwise only used for radial filtering,
                # which this call does not use).
                if peaks_r_invA is None and _has_peak_positions(peaks_x, peaks_y):
                    peaks_r_invA = np.sqrt(
                        (np.asarray(peaks_x) - center[1]) ** 2
                        + (np.asarray(peaks_y) - center[0]) ** 2
                    )
                central_idx = _central_peak_index(
                    peaks_x, peaks_y, peaks_r_invA, center,
                    max_dist=_central_beam_max_dist(base_shape),
                )

                ax_map.clear()
                if is_rgb_map:
                    ax_map.imshow(intensity_map)
                elif map_vmin is None:
                    ax_map.imshow(intensity_map, cmap=map_cmap)
                else:
                    ax_map.imshow(intensity_map, cmap=map_cmap, vmin=map_vmin, vmax=map_vmax)
                ax_map.scatter(
                    rx * upsample_factor, ry * upsample_factor,
                    facecolor="none", edgecolor=crosshair_color, marker="o",
                    s=crosshair_size, linewidth=crosshair_width, zorder=10,
                )
                ax_map.set_title(f"{map_title}  Ry={ry}, Rx={rx}" if map_title else f"Ry={ry}, Rx={rx}")
                ax_map.set_xticks([])
                ax_map.set_yticks([])

                for ax, spec in zip(dp_axes, panels):
                    npow = spec.get("norm_power", norm_power)
                    npow = 1.0 if npow is None else npow
                    sigma = spec.get("gaussian_filter_sigma", gaussian_filter_sigma)
                    dp_p = _normalized_dp(
                        self.dataset_cartesian, ry, rx,
                        norm_upper_quantile=spec.get("norm_upper_quantile", norm_upper_quantile),
                        norm_power=npow,
                    )
                    if sigma is not None:
                        dp_p = gaussian_filter(dp_p, sigma)
                    (
                        dp_p, px, py, r_invA, pint, cidx, disp_center,
                    ) = _zoom_peak_overlay(
                        dp_p, peaks_x, peaks_y, peaks_r_invA, peak_ints,
                        central_idx, spec.get("zoom", zoom), center,
                    )
                    ax.clear()
                    ax.imshow(
                        dp_p, cmap=spec.get("dp_cmap", dp_cmap),
                        vmin=spec.get("vmin_cartesian", vmin_cartesian),
                        vmax=spec.get("vmax_cartesian", vmax_cartesian),
                    )
                    if show_peaks and px is not None:
                        cwp = spec.get("crosshair_width_peaks", crosshair_width_peaks)
                        clw = spec.get("central_linewidth", central_linewidth)
                        if ("marker_size" in spec) or ("central_size" in spec):
                            # Data-proportional circles (radius in detector px) so markers
                            # cover the same fraction of the pattern as the widget canvas
                            # (which scales its px marker radii to the display).
                            _draw_peaks_data_circles(
                                ax, px, py, pint, cidx, disp_center,
                                marker_scaled=spec.get("marker_scaled", True),
                                marker_size=spec.get("marker_size", 8.0),
                                marker_size_min=spec.get("marker_size_min", 4.0),
                                marker_size_max=spec.get("marker_size_max", 16.0),
                                selected_peak_color=spec.get("selected_peak_color", selected_peak_color),
                                central_beam_color=spec.get("central_beam_color", central_beam_color),
                                show_central_beam=spec.get("show_central_beam", show_central_beam),
                                central_size=spec.get("central_size", 5.0),
                                peak_linewidth=cwp,
                                central_linewidth=(cwp if clw is None else clw),
                            )
                        else:
                            palpha = spec.get("peak_alpha", peak_alpha)
                            _plot_bragg_peaks_on_ax(
                                ax, px, py, r_invA, pint, cidx,
                                selected_peak_color=spec.get("selected_peak_color", selected_peak_color),
                                central_beam_color=spec.get("central_beam_color", central_beam_color),
                                peak_intensity_mode=spec.get("peak_intensity_mode", peak_intensity_mode),
                                peak_size_range=spec.get("peak_size_range", peak_size_range),
                                peak_marker_size=spec.get("peak_marker_size", peak_marker_size),
                                crosshair_width_peaks=cwp,
                                crosshair_scaling_central_beam=spec.get(
                                    "crosshair_scaling_central_beam", crosshair_scaling_central_beam
                                ),
                                peak_alpha=palpha,
                                central_alpha=palpha,
                                central_linewidth=(cwp if clw is None else clw),
                                center=disp_center,
                                show_central_beam=spec.get("show_central_beam", show_central_beam),
                            )
                    ax.set_xlim(-0.5, dp_p.shape[1] - 0.5)
                    ax.set_ylim(dp_p.shape[0] - 0.5, -0.5)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(spec.get("title") or "")

                fig.canvas.draw()
                rgba = np.asarray(fig.canvas.buffer_rgba())
                frames.append(Image.fromarray(rgba[..., :3].copy()))
        finally:
            plt.close(fig)

        if not frames:
            raise ValueError("Snake path is empty; check region / step.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = max(10, int(round(1000.0 / max(0.1, fps))))
        frames[0].save(
            str(path), save_all=True, append_images=frames[1:],
            duration=duration_ms, loop=0, optimize=True, disposal=2,
        )
        if progress:
            print(f"✓ Saved {len(frames)}-frame animation: {path.resolve()}")
        return path

    def edit_scan_mask(
        self,
        *,
        initial_x=None,
        initial_y=None,
        initial_radius=None,
        initial_geometry="circle",
        initial_size_x=None,
        initial_size_y=None,
        reference_image=None,
        state_path=None,
        overlay_alpha=0.28,
        crosshair_width=2,
        crosshair_size=12,
        autosave=False,
        display_widget=True,
    ):
        """Create an interactive scan-mask editor.

        X is the horizontal scan-column coordinate and Y is the vertical
        scan-row coordinate. Circle, ellipse, square, and rectangle geometries
        are available. Sizes are radii for round geometries and half-sizes for
        rectangular geometries. A saved ``state_path`` is loaded automatically.
        """
        return ScanMaskEditor(
            self,
            initial_x=initial_x,
            initial_y=initial_y,
            initial_radius=initial_radius,
            initial_geometry=initial_geometry,
            initial_size_x=initial_size_x,
            initial_size_y=initial_size_y,
            reference_image=reference_image,
            state_path=state_path,
            overlay_alpha=overlay_alpha,
            crosshair_width=crosshair_width,
            crosshair_size=crosshair_size,
            autosave=autosave,
            display_widget=display_widget,
        )

    def create_interactive_circular_mask(
        self,
        initial_x0=None,
        initial_y0=None,
        initial_r=None,
        reference_image=None,
        overlay_alpha=0.3,
        crosshair_width=2,
        crosshair_size=15,
        state_path=None,
        autosave=False,
        display_widget=True,
    ):
        """Compatibility wrapper for :meth:`edit_scan_mask`.

        Historically ``initial_x0`` represented the array row and
        ``initial_y0`` represented the array column. New code should use
        ``edit_scan_mask(initial_x=column, initial_y=row, ...)``.
        """
        return self.edit_scan_mask(
            initial_x=initial_y0,
            initial_y=initial_x0,
            initial_radius=initial_r,
            reference_image=reference_image,
            state_path=state_path,
            overlay_alpha=overlay_alpha,
            crosshair_width=crosshair_width,
            crosshair_size=crosshair_size,
            autosave=autosave,
            display_widget=display_widget,
        )

    def plot_peak_histogram_map(self, *args, **kwargs):
        """Per-position peak-intensity histogram map.

        Forwards to :func:`quantem.diffraction.peak_visualization.plot_peak_histogram_map`,
        supplying this object's data. See that function for the parameters.
        """

        return _plot_peak_histogram_map(self.peak_coordinates_cartesian, self.peak_intensities, *args, **kwargs)

    def plot_peak_count_map(self, *args, **kwargs):
        """Per-position peak counts, one map per q window.

        Forwards to :func:`quantem.diffraction.peak_visualization.plot_peak_count_map`,
        supplying this object's data. See that function for the parameters.
        """

        return _plot_peak_count_map(self.peak_coordinates_cartesian, self.polar_peaks, *args, **kwargs)

    def make_flowline_map(
        self,
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
        progress_bar: bool = True,
    ):
        """
        Create an 3D or 4D orientation flowline map - essentially a pixelated "stream map" which represents diffraction data.
    
        Args:
            orient_hist (array):        Histogram of all orientations with coordinates
                                        [radial_bin x_probe y_probe theta]
                                        We assume theta bin ranges from 0 to 180 degrees and is periodic.
            thresh_seed (float):        Threshold for seed generation in histogram.
            thresh_grow (float):        Threshold for flowline growth in histogram.
            thresh_collision (float):   Threshold for termination of flowline growth in histogram.
            sep_seeds (float):          Initial seed separation in bins - set to None to use default value,
                                        which is equal to 0.5*sep_xy.
            sep_xy (float):             Search radius for flowline direction in x and y.
            sep_theta = (float):        Search radius for flowline direction in theta.
            sort_seeds (str):           How to sort the initial seeds for growth:
                                            None - no sorting
                                            'intensity' - sort by histogram intensity
                                            'random' - random order
            linewidth (float):          Thickness of the flowlines in pixels.
            step_size (float):          Step size for flowline growth in pixels.
            min_steps (int):            Minimum number of steps for a flowline to be drawn.
            max_steps (int):            Maximum number of steps for a flowline to be drawn.
            sigma_x (float):            Weighted sigma in x direction for direction update.
            sigma_y (float):            Weighted sigma in y direction for direction update.
            sigma_theta (float):        Weighted sigma in theta for direction update.
            progress_bar (bool):        Enable progress bar
    
        Returns:
            orient_flowlines (array):   4D array containing flowlines
                                        [radial_bin x_probe y_probe theta]
        """
    
        # Ensure sep_xy and sep_theta are arrays
        sep_xy = np.atleast_1d(sep_xy)
        sep_theta = np.atleast_1d(sep_theta)
    
        # number of radial bins
        num_radii = orient_hist.shape[0]
        if num_radii > 1 and len(sep_xy) == 1:
            sep_xy = np.ones(num_radii) * sep_xy
        if num_radii > 1 and len(sep_theta) == 1:
            sep_theta = np.ones(num_radii) * sep_theta
    
        # Default seed separation
        if sep_seeds is None:
            sep_seeds = np.round(np.min(sep_xy) / 2 + 0.5).astype("int")
        else:
            sep_seeds = np.atleast_1d(sep_seeds).astype("int")
            if num_radii > 1 and len(sep_seeds) == 1:
                sep_seeds = (np.ones(num_radii) * sep_seeds).astype("int")
    
        # coordinates
        theta = np.linspace(0, np.pi, orient_hist.shape[3], endpoint=False)
        dtheta = theta[1] - theta[0]
        size_3D = np.array(
            [
                orient_hist.shape[1],
                orient_hist.shape[2],
                orient_hist.shape[3],
            ]
        )
    
        # initialize weighting array
        vx = np.arange(-np.ceil(2 * sigma_x), np.ceil(2 * sigma_x) + 1)
        vy = np.arange(-np.ceil(2 * sigma_y), np.ceil(2 * sigma_y) + 1)
        vt = np.arange(-np.ceil(2 * sigma_theta), np.ceil(2 * sigma_theta) + 1)
        ay, ax, at = np.meshgrid(vy, vx, vt)
        k = (
            np.exp(ax**2 / (-2 * sigma_x**2))
            * np.exp(ay**2 / (-2 * sigma_y**2))
            * np.exp(at**2 / (-2 * sigma_theta**2))
        )
        k = k / np.sum(k)
        vx = vx[:, None, None].astype("int")
        vy = vy[None, :, None].astype("int")
        vt = vt[None, None, :].astype("int")
    
        # initalize flowline array
        orient_flowlines = np.zeros_like(orient_hist)
    
        # initialize output
        xy_t_int = np.zeros((max_steps + 1, 4))
        xy_t_int_rev = np.zeros((max_steps + 1, 4))
    
        # Loop over radial bins
        for a0 in range(num_radii):
            # initialize collision check array
            cr = np.arange(-np.ceil(sep_xy[a0]), np.ceil(sep_xy[a0]) + 1)
            ct = np.arange(-np.ceil(sep_theta[a0]), np.ceil(sep_theta[a0]) + 1)
            ay, ax, at = np.meshgrid(cr, cr, ct)
            c_mask = (
                (ax**2 + ay**2) / sep_xy[a0] ** 2 + at**2 / sep_theta[a0] ** 2
                <= (1 + 1 / sep_xy[a0]) ** 2
            )[None, :, :, :]
            cx = cr[None, :, None, None].astype("int")
            cy = cr[None, None, :, None].astype("int")
            ct = ct[None, None, None, :].astype("int")
    
            # Find all seed locations
            orient = orient_hist[a0, :, :, :]
            sub_seeds = np.logical_and(
                np.logical_and(
                    orient >= np.roll(orient, 1, axis=2),
                    orient >= np.roll(orient, -1, axis=2),
                ),
                orient >= thresh_seed,
            )
    
            # Separate seeds
            if sep_seeds > 0:
                for a1 in range(sep_seeds - 1):
                    sub_seeds[a1::sep_seeds, :, :] = False
                    sub_seeds[:, a1::sep_seeds, :] = False
    
            # Index seeds
            x_inds, y_inds, t_inds = np.where(sub_seeds)
            if sort_seeds is not None:
                if sort_seeds == "intensity":
                    inds_sort = np.argsort(orient[sub_seeds])[::-1]
                elif sort_seeds == "random":
                    inds_sort = np.random.permutation(np.count_nonzero(sub_seeds))
                x_inds = x_inds[inds_sort]
                y_inds = y_inds[inds_sort]
                t_inds = t_inds[inds_sort]
    
            # for a1 in tqdmnd(range(0,40), desc="Drawing flowlines",unit=" seeds", disable=not progress_bar):
            t = "Drawing flowlines " + str(a0)
            for a1 in tqdmnd(
                range(0, x_inds.shape[0]), desc=t, unit=" seeds", disable=not progress_bar
            ):
                # initial coordinate and intensity
                xy0 = np.array((x_inds[a1], y_inds[a1]))
                t0 = theta[t_inds[a1]]
    
                # init theta
                inds_theta = np.mod(
                    np.round(t0 / dtheta).astype("int") + vt, orient.shape[2]
                )
                orient_crop = (
                    k
                    * orient[
                        np.clip(
                            np.round(xy0[0]).astype("int") + vx, 0, orient.shape[0] - 1
                        ),
                        np.clip(
                            np.round(xy0[1]).astype("int") + vy, 0, orient.shape[1] - 1
                        ),
                        inds_theta,
                    ]
                )
                theta_crop = theta[inds_theta]
                t0 = np.sum(orient_crop * theta_crop) / np.sum(orient_crop)
    
                # forward direction
                t = t0
                v0 = np.array((np.cos(t), -np.sin(t)))
                v = v0 * step_size
                xy = xy0
                int_val = self.get_intensity(orient, xy0[0], xy0[1], t0 / dtheta)
                xy_t_int[0, 0:2] = xy0
                xy_t_int[0, 2] = t / dtheta
                xy_t_int[0, 3] = int_val
                # main loop
                grow = True
                count = 0
                while grow is True:
                    count += 1
    
                    # update position and intensity
                    xy = xy + v
                    int_val = self.get_intensity(orient, xy[0], xy[1], t / dtheta)
    
                    # check for collision
                    flow_crop = orient_flowlines[
                        a0,
                        np.clip(np.round(xy[0]).astype("int") + cx, 0, orient.shape[0] - 1),
                        np.clip(np.round(xy[1]).astype("int") + cy, 0, orient.shape[1] - 1),
                        np.mod(np.round(t / dtheta).astype("int") + ct, orient.shape[2]),
                    ]
                    int_flow = np.max(flow_crop[c_mask])
    
                    if (
                        xy[0] < 0
                        or xy[1] < 0
                        or xy[0] > orient.shape[0]
                        or xy[1] > orient.shape[1]
                        or int_val < thresh_grow
                        or int_flow > thresh_collision
                    ):
                        grow = False
                    else:
                        # update direction
                        inds_theta = np.mod(
                            np.round(t / dtheta).astype("int") + vt, orient.shape[2]
                        )
                        orient_crop = (
                            k
                            * orient[
                                np.clip(
                                    np.round(xy[0]).astype("int") + vx,
                                    0,
                                    orient.shape[0] - 1,
                                ),
                                np.clip(
                                    np.round(xy[1]).astype("int") + vy,
                                    0,
                                    orient.shape[1] - 1,
                                ),
                                inds_theta,
                            ]
                        )
                        theta_crop = theta[inds_theta]
                        t = np.sum(orient_crop * theta_crop) / np.sum(orient_crop)
                        # v = np.array((np.cos(t), np.sin(t))) * step_size
                        # v = np.array((np.sin(t), np.cos(t))) * step_size
                        # v = np.array((-np.sin(t), np.cos(t))) * step_size
    
                        xy_t_int[count, 0:2] = xy
                        xy_t_int[count, 2] = t / dtheta
                        xy_t_int[count, 3] = int_val
    
                        if count > max_steps - 1:
                            grow = False
    
                # reverse direction
                t = t0 + np.pi
                v0 = np.array((np.cos(t), -np.sin(t)))
                v = v0 * step_size
                xy = xy0
                int_val = self.get_intensity(orient, xy0[0], xy0[1], t0 / dtheta)
                xy_t_int_rev[0, 0:2] = xy0
                xy_t_int_rev[0, 2] = t / dtheta
                xy_t_int_rev[0, 3] = int_val
                # main loop
                grow = True
                count_rev = 0
                while grow is True:
                    count_rev += 1
    
                    # update position and intensity
                    xy = xy + v
                    int_val = self.get_intensity(orient, xy[0], xy[1], t / dtheta)
    
                    # check for collision
                    flow_crop = orient_flowlines[
                        a0,
                        np.clip(np.round(xy[0]).astype("int") + cx, 0, orient.shape[0] - 1),
                        np.clip(np.round(xy[1]).astype("int") + cy, 0, orient.shape[1] - 1),
                        np.mod(np.round(t / dtheta).astype("int") + ct, orient.shape[2]),
                    ]
                    int_flow = np.max(flow_crop[c_mask])
    
                    if (
                        xy[0] < 0
                        or xy[1] < 0
                        or xy[0] > orient.shape[0]
                        or xy[1] > orient.shape[1]
                        or int_val < thresh_grow
                        or int_flow > thresh_collision
                    ):
                        grow = False
                    else:
                        # update direction
                        inds_theta = np.mod(
                            np.round(t / dtheta).astype("int") + vt, orient.shape[2]
                        )
                        orient_crop = (
                            k
                            * orient[
                                np.clip(
                                    np.round(xy[0]).astype("int") + vx,
                                    0,
                                    orient.shape[0] - 1,
                                ),
                                np.clip(
                                    np.round(xy[1]).astype("int") + vy,
                                    0,
                                    orient.shape[1] - 1,
                                ),
                                inds_theta,
                            ]
                        )
                        theta_crop = theta[inds_theta]
                        t = np.sum(orient_crop * theta_crop) / np.sum(orient_crop) + np.pi
                        v = np.array((np.cos(t), -np.sin(t))) * step_size
                        v = np.array((np.cos(t), -np.sin(t))) * step_size
                        # v = np.array((-np.sin(t), np.cos(t))) * step_size
    
                        xy_t_int_rev[count_rev, 0:2] = xy
                        xy_t_int_rev[count_rev, 2] = t / dtheta
                        xy_t_int_rev[count_rev, 3] = int_val
    
                        if count_rev > max_steps - 1:
                            grow = False
    
                # write into output array
                if count + count_rev > min_steps:
                    if count > 0:
                        orient_flowlines[a0, :, :, :] = self.set_intensity(
                            orient_flowlines[a0, :, :, :], xy_t_int[1:count, :]
                        )
                    if count_rev > 1:
                        orient_flowlines[a0, :, :, :] = self.set_intensity(
                            orient_flowlines[a0, :, :, :], xy_t_int_rev[1:count_rev, :]
                        )
    
        # normalize to step size
        orient_flowlines = orient_flowlines * step_size
    
        # linewidth
        if linewidth > 1.0:
            s = linewidth - 1.0
    
            orient_flowlines = gaussian_filter1d(orient_flowlines, s, axis=1, truncate=3.0)
            orient_flowlines = gaussian_filter1d(orient_flowlines, s, axis=2, truncate=3.0)
            orient_flowlines = orient_flowlines * (s**2)
    
        return orient_flowlines
    
    
    def make_flowline_rainbow_image(
        self,
        orient_flowlines,
        int_range=[0, 0.2],
        sym_rotation_order=2,
        theta_offset=np.pi,
        greyscale=False,
        greyscale_max=True,
        white_background=False,
        power_scaling=1.0,
        sum_radial_bins=False,
        plot_images=True,
        figsize=None,
    ):
        """
        Generate RGB output images from the flowline arrays.
    
        Args:
            orient_flowline (array):    Histogram of all orientations with coordinates [x y radial_bin theta]
                                        We assume theta bin ranges from 0 to 180 degrees and is periodic.
            int_range (float)           2 element array giving the intensity range
            sym_rotation_order (int):   rotational symmety for colouring
            theta_offset (float):       Offset the anglular coloring by this value in radians.
                                        Default pi rotates the hue mapping by 90deg (nematic
                                        sym=2) so color tracks the drawn flowline direction:
                                        cyan for vertical lines, red for horizontal.
            greyscale (bool):           Set to False for color output, True for greyscale output.
            greyscale_max (bool):       If output is greyscale, use max instead of mean for overlapping flowlines.
            white_background (bool):    For either color or greyscale output, switch to white background (from black).
            power_scaling (float):      Power law scaling for flowline intensity output.
            sum_radial_bins (bool):     Sum all radial bins (alternative is to output separate images).
            plot_images (bool):         Plot the outputs for quick visualization.
            figsize (2-tuple):          Size of output figure.
    
        Returns:
            im_flowline (array):        3D or 4D array containing flowline images
        """
    
        # init array
        size_input = orient_flowlines.shape
        size_output = np.array([size_input[0], size_input[1], size_input[2], 3])
        im_flowline = np.zeros(size_output)
        theta_offset = np.atleast_1d(theta_offset)
    
        if greyscale is True:
            for a0 in range(size_input[0]):
                if greyscale_max is True:
                    im = np.max(orient_flowlines[a0, :, :, :], axis=2)
                else:
                    im = np.mean(orient_flowlines[a0, :, :, :], axis=2)
    
                sig = np.clip((im - int_range[0]) / (int_range[1] - int_range[0]), 0, 1)
    
                if power_scaling != 1:
                    sig = sig**power_scaling
    
                if white_background is False:
                    im_flowline[a0, :, :, :] = sig[:, :, None]
                else:
                    im_flowline[a0, :, :, :] = 1 - sig[:, :, None]
    
        else:
            # Color basis
            c0 = np.array([1.0, 0.0, 0.0])
            c1 = np.array([0.0, 0.7, 0.0])
            c2 = np.array([0.0, 0.3, 1.0])
    
            # angles
            theta = np.linspace(0, np.pi, size_input[3], endpoint=False)
            # Negate so the hue handedness matches the drawn flowline direction in the
            # displayed (y-down) map: red horizontal, cyan vertical, "/" yellow, "\" purple.
            theta_color = -theta * sym_rotation_order
    
            if size_input[0] > 1 and len(theta_offset) == 1:
                theta_offset = np.ones(size_input[0]) * theta_offset
    
            for a0 in range(size_input[0]):
                # color projections
                b0 = np.maximum(
                    1
                    - np.abs(
                        np.mod(theta_offset[a0] + theta_color + np.pi, 2 * np.pi) - np.pi
                    )
                    ** 2
                    / (np.pi * 2 / 3) ** 2,
                    0,
                )
                b1 = np.maximum(
                    1
                    - np.abs(
                        np.mod(
                            theta_offset[a0] + theta_color - np.pi * 2 / 3 + np.pi,
                            2 * np.pi,
                        )
                        - np.pi
                    )
                    ** 2
                    / (np.pi * 2 / 3) ** 2,
                    0,
                )
                b2 = np.maximum(
                    1
                    - np.abs(
                        np.mod(
                            theta_offset[a0] + theta_color - np.pi * 4 / 3 + np.pi,
                            2 * np.pi,
                        )
                        - np.pi
                    )
                    ** 2
                    / (np.pi * 2 / 3) ** 2,
                    0,
                )
    
                sig = np.clip(
                    (orient_flowlines[a0, :, :, :] - int_range[0])
                    / (int_range[1] - int_range[0]),
                    0,
                    1,
                )
                if power_scaling != 1:
                    sig = sig**power_scaling
    
                im_flowline[a0, :, :, :] = (
                    np.sum(sig * b0[None, None, :], axis=2)[:, :, None] * c0[None, None, :]
                    + np.sum(sig * b1[None, None, :], axis=2)[:, :, None]
                    * c1[None, None, :]
                    + np.sum(sig * b2[None, None, :], axis=2)[:, :, None]
                    * c2[None, None, :]
                )
    
                # clip limits
                im_flowline[a0, :, :, :] = np.clip(im_flowline[a0, :, :, :], 0, 1)
    
                # contrast flip
                if white_background is True:
                    im = rgb_to_hsv(im_flowline[a0])
                    im_v = im[:, :, 2]
                    im[:, :, 1] = im_v
                    im[:, :, 2] = 1
                    im_flowline[a0] = hsv_to_rgb(im)
    
        if sum_radial_bins is True:
            if white_background is False:
                im_flowline = np.clip(np.sum(im_flowline, axis=0), 0, 1)[None, :, :, :]
            else:
                # im_flowline = np.clip(np.sum(im_flowline,axis=0)+1-im_flowline.shape[0],0,1)[None,:,:,:]
                im_flowline = np.min(im_flowline, axis=0)[None, :, :, :]
    
        if plot_images is True:
            if figsize is None:
                fig, ax = plt.subplots(
                    im_flowline.shape[0], 1, figsize=(10, im_flowline.shape[0] * 10)
                )
            else:
                fig, ax = plt.subplots(im_flowline.shape[0], 1, figsize=figsize)
    
            if im_flowline.shape[0] > 1:
                for a0 in range(im_flowline.shape[0]):
                    ax[a0].imshow(im_flowline[a0])
                    # ax[a0].axis('off')
                plt.subplots_adjust(wspace=0, hspace=0.02)
            else:
                ax.imshow(im_flowline[0])
                # ax.axis('off')
            plt.show()
    
        return im_flowline
    
    
    def make_flowline_rainbow_legend(
        self,
        im_size=np.array([256, 256]),
        sym_rotation_order=2,
        theta_offset_degrees=0.0,
        white_background=False,
        return_image=False,
        radial_range=np.array([0.45, 0.9]),
        plot_legend=True,
        figsize=(4, 4),
    ):
        """
        This function generates a legend for a the rainbow colored flowline maps, and returns it as an RGB image.
    
        Parameters
        ----------
        im_size (np.array):
            Size of legend image in pixels.
        sym_rotation_order (int):
            rotational symmety for colouring
        theta_offset_degrees (float):
            Offset the anglular coloring by this value in degrees.
            Rotation is Q with respect to R, in the positive (counter clockwise) direction.
        white_background (bool):
            For either color or greyscale output, switch to white background (from black).
        return_image (bool):
            Return the image array.
        radial_range (np.array):
            Inner and outer radius for the legend ring.
        plot_legend (bool):
            Plot the generated legend.
        figsize (tuple or list):
            Size of the plotted legend.
    
        Returns
        ----------
    
        im_legend (array):
            Image array for the legend.
        """
    
        # Coordinates
        x = np.linspace(-1, 1, im_size[0])
        y = np.linspace(-1, 1, im_size[1])
        ya, xa = np.meshgrid(y, x)
        # TODO: Can replace with squared term? ra2? Faster
        # ra = np.sqrt(xa**2 + ya**2)
        ra2 = xa**2 + ya**2
        ta = np.arctan2(ya, xa) + np.deg2rad(theta_offset_degrees)
        ta_sym = ta * sym_rotation_order
    
        # mask
        mask = np.logical_and(ra2 > radial_range[0]**2, ra2 < radial_range[1]**2)
        # mask = np.logical_and(ra > radial_range[0], ra < radial_range[1])
    
        # rgb image
        z = mask * np.exp(1j * ta_sym)
        # hue_offset = 0
        amp = np.abs(z)
        vmin = np.min(amp)
        vmax = np.max(amp)
        ph = np.angle(z)  # + hue_offset
        h = np.mod(ph / (2 * np.pi), 1)
        s = 0.85 * np.ones_like(h)
        v = (amp - vmin) / (vmax - vmin)
        im_legend = hsv_to_rgb(np.dstack((h, s, v)))
    
        if white_background is True:
            im_legend[im_legend.sum(2) == 0] = 1
    
        # plotting
        if plot_legend:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.imshow(im_legend)
            ax.invert_yaxis()
            # ax.set_axis_off()
            ax.axis("off")
    
        if return_image:
            return im_legend
    
    
    def make_flowline_combined_image(
        self,
        orient_flowlines,
        int_range=[0, 0.2],
        cvals=np.array(
            [
                [0.0, 0.7, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.7, 1.0],
            ]
        ),
        white_background=False,
        power_scaling=1.0,
        sum_radial_bins=True,
        plot_images=True,
        figsize=None,
    ):
        """
        Generate RGB output images from the flowline arrays.
    
        Args:
            orient_flowline (array):    Histogram of all orientations with coordinates [x y radial_bin theta]
                                        We assume theta bin ranges from 0 to 180 degrees and is periodic.
            int_range (float)           2 element array giving the intensity range
            cvals (array):              Nx3 size array containing RGB colors for different radial ibns.
            white_background (bool):    For either color or greyscale output, switch to white background (from black).
            power_scaling (float):      Power law scaling for flowline intensities.
            sum_radial_bins (bool):     Sum outputs over radial bins.
            plot_images (bool):         Plot the output images for quick visualization.
            figsize (2-tuple):          Size of output figure.
    
        Returns:
            im_flowline (array):        flowline images
        """
    
        # init array
        size_input = orient_flowlines.shape
        size_output = np.array([size_input[0], size_input[1], size_input[2], 3])
        im_flowline = np.zeros(size_output)
        cvals = np.array(cvals)
    
        # Generate all color images
        for a0 in range(size_input[0]):
            sig = np.clip(
                (np.sum(orient_flowlines[a0, :, :, :], axis=2) - int_range[0])
                / (int_range[1] - int_range[0]),
                0,
                1,
            )
            if power_scaling != 1:
                sig = sig**power_scaling
    
            if white_background:
                im_flowline[a0, :, :, :] = 1 - sig[:, :, None] * (
                    1 - cvals[a0, :][None, None, :]
                )
            else:
                im_flowline[a0, :, :, :] = sig[:, :, None] * cvals[a0, :][None, None, :]
    
            # # contrast flip
            # if white_background is True:
            #     im = rgb_to_hsv(im_flowline[a0,:,:,:])
            #     # im_s = im[:,:,1]
            #     im_v = im[:,:,2]
            #     v_range = [np.min(im_v), np.max(im_v)]
            #     print(v_range)
    
            #     im[:,:,1] = im_v
            #     im[:,:,2] = 1
            #     im_flowline[a0,:,:,:] = hsv_to_rgb(im)
    
        if sum_radial_bins is True:
            if white_background is False:
                im_flowline = np.clip(np.sum(im_flowline, axis=0), 0, 1)[None, :, :, :]
            else:
                # im_flowline = np.clip(np.sum(im_flowline,axis=0)+1-im_flowline.shape[0],0,1)[None,:,:,:]
                im_flowline = np.min(im_flowline, axis=0)[None, :, :, :]
    
        if plot_images is True:
            if figsize is None:
                fig, ax = plt.subplots(
                    im_flowline.shape[0], 1, figsize=(10, im_flowline.shape[0] * 10)
                )
            else:
                fig, ax = plt.subplots(im_flowline.shape[0], 1, figsize=figsize)
    
            if im_flowline.shape[0] > 1:
                for a0 in range(im_flowline.shape[0]):
                    ax[a0].imshow(im_flowline[a0])
                    ax[a0].axis("off")
                plt.subplots_adjust(wspace=0, hspace=0.02)
            else:
                ax.imshow(im_flowline[0])
                ax.axis("off")
            plt.show()
    
        return im_flowline

    def get_intensity(
        self, 
        orient, 
        x, 
        y, 
        t
    ):
        # utility function to get histogram intensites
    
        x = np.clip(x, 0, orient.shape[0] - 2)
        y = np.clip(y, 0, orient.shape[1] - 2)
    
        xF = np.floor(x).astype("int")
        yF = np.floor(y).astype("int")
        tF = np.floor(t).astype("int")
        dx = x - xF
        dy = y - yF
        dt = t - tF
        t1 = np.mod(tF, orient.shape[2])
        t2 = np.mod(tF + 1, orient.shape[2])
    
        int_vals = (
            orient[xF, yF, t1] * ((1 - dx) * (1 - dy) * (1 - dt))
            + orient[xF, yF, t2] * ((1 - dx) * (1 - dy) * (dt))
            + orient[xF, yF + 1, t1] * ((1 - dx) * (dy) * (1 - dt))
            + orient[xF, yF + 1, t2] * ((1 - dx) * (dy) * (dt))
            + orient[xF + 1, yF, t1] * ((dx) * (1 - dy) * (1 - dt))
            + orient[xF + 1, yF, t2] * ((dx) * (1 - dy) * (dt))
            + orient[xF + 1, yF + 1, t1] * ((dx) * (dy) * (1 - dt))
            + orient[xF + 1, yF + 1, t2] * ((dx) * (dy) * (dt))
        )
    
        return int_vals
    
    
    def set_intensity(
        self, 
        orient, 
        xy_t_int
    ):
        # utility function to set flowline intensites
    
        xF = np.floor(xy_t_int[:, 0]).astype("int")
        yF = np.floor(xy_t_int[:, 1]).astype("int")
        tF = np.floor(xy_t_int[:, 2]).astype("int")
        dx = xy_t_int[:, 0] - xF
        dy = xy_t_int[:, 1] - yF
        dt = xy_t_int[:, 2] - tF
    
        inds_1D = np.ravel_multi_index(
            [xF, yF, tF], orient.shape[0:3], mode=["clip", "clip", "wrap"]
        )
        orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (1 - dx) * (
            1 - dy
        ) * (1 - dt)
        inds_1D = np.ravel_multi_index(
            [xF, yF, tF + 1], orient.shape[0:3], mode=["clip", "clip", "wrap"]
        )
        orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (1 - dx) * (
            1 - dy
        ) * (dt)
        inds_1D = np.ravel_multi_index(
            [xF, yF + 1, tF], orient.shape[0:3], mode=["clip", "clip", "wrap"]
        )
        orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (1 - dx) * (
            dy
        ) * (1 - dt)
        inds_1D = np.ravel_multi_index(
            [xF, yF + 1, tF + 1], orient.shape[0:3], mode=["clip", "clip", "wrap"]
        )
        orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (1 - dx) * (
            dy
        ) * (dt)
        inds_1D = np.ravel_multi_index(
            [xF + 1, yF, tF], orient.shape[0:3], mode=["clip", "clip", "wrap"]
        )
        orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (dx) * (
            1 - dy
        ) * (1 - dt)
        inds_1D = np.ravel_multi_index(
            [xF + 1, yF, tF + 1], orient.shape[0:3], mode=["clip", "clip", "wrap"]
        )
        orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (dx) * (
            1 - dy
        ) * (dt)
        inds_1D = np.ravel_multi_index(
            [xF + 1, yF + 1, tF], orient.shape[0:3], mode=["clip", "clip", "wrap"]
        )
        orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (dx) * (dy) * (
            1 - dt
        )
        inds_1D = np.ravel_multi_index(
            [xF + 1, yF + 1, tF + 1], orient.shape[0:3], mode=["clip", "clip", "wrap"]
        )
        orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (dx) * (dy) * (
            dt
        )
    
        return orient

    def interactive_probe_selector(self, probe_map=None, figsize=(14, 8), cmap='viridis'):
        """
        Interactive GUI to select probe positions and view diffraction patterns.
        
        Parameters
        ----------
        probe_map : ndarray, optional
            2D array to display as the probe position map. If None, uses mean diffraction intensity.
        figsize : tuple
            Figure size (width, height)
        cmap : str
            Colormap for the probe map
            
        Returns
        -------
        selected_positions : list of tuples
            List of (ry, rx) coordinates of selected positions
        """
        from matplotlib.widgets import Button
        from matplotlib.patches import Circle
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        
        Ry, Rx = self.dataset_cartesian.shape[:2]
        
        # Create default probe map if not provided
        if probe_map is None:
            probe_map = np.mean(self.dataset_cartesian.array, axis=(2, 3))
        
        # Storage for selected positions
        selected_positions = []
        markers = []
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, width_ratios=[2, 2, 1], height_ratios=[1, 1], 
                              hspace=0.3, wspace=0.3)
        
        # Probe map axis
        ax_probe = fig.add_subplot(gs[:, 0])
        im_probe = ax_probe.imshow(probe_map, cmap=cmap, origin='lower', 
                                    interpolation='nearest', aspect='auto')
        ax_probe.set_title('Probe Position Map\n(Click to add point)', fontsize=12)
        ax_probe.set_xlabel('Rx')
        ax_probe.set_ylabel('Ry')
        plt.colorbar(im_probe, ax=ax_probe, label='Intensity')
        
        # Diffraction pattern axes
        ax_dp1 = fig.add_subplot(gs[0, 1])
        ax_dp2 = fig.add_subplot(gs[1, 1])
        ax_dp1.set_title('Diffraction Pattern 1')
        ax_dp2.set_title('Diffraction Pattern 2')
        ax_dp1.axis('off')
        ax_dp2.axis('off')
        
        # Text area for position list
        ax_list = fig.add_subplot(gs[:, 2])
        ax_list.axis('off')
        ax_list.set_title('Selected Positions', fontsize=11, fontweight='bold')
        
        # Add clear all button
        ax_button = plt.axes([0.7, 0.02, 0.1, 0.04])
        btn_clear = Button(ax_button, 'Clear All')
        
        def update_display():
            """Update the position list and diffraction patterns."""
            # Clear position list
            ax_list.clear()
            ax_list.axis('off')
            ax_list.set_title('Selected Positions', fontsize=11, fontweight='bold')
            
            # Display positions
            y_pos = 0.95
            for idx, (ry, rx) in enumerate(selected_positions):
                text = f"{idx+1}. ({ry}, {rx})"
                ax_list.text(0.1, y_pos, text, fontsize=10, transform=ax_list.transAxes,
                            verticalalignment='top')
                y_pos -= 0.08
            
            # Update diffraction patterns
            if len(selected_positions) >= 1:
                ry, rx = selected_positions[-1]
                dp = self.dataset_cartesian[ry, rx].array
                ax_dp1.clear()
                ax_dp1.imshow(dp, cmap='gray')
                ax_dp1.set_title(f'DP at ({ry}, {rx})')
                ax_dp1.axis('off')
            
            if len(selected_positions) >= 2:
                ry, rx = selected_positions[-2]
                dp = self.dataset_cartesian[ry, rx].array
                ax_dp2.clear()
                ax_dp2.imshow(dp, cmap='gray')
                ax_dp2.set_title(f'DP at ({ry}, {rx})')
                ax_dp2.axis('off')
            
            fig.canvas.draw_idle()
        
        def onclick(event):
            """Handle click events on probe map."""
            if event.inaxes == ax_probe and event.button == 1:  # Left click
                rx = int(np.round(event.xdata))
                ry = int(np.round(event.ydata))
                
                # Check bounds
                if 0 <= ry < Ry and 0 <= rx < Rx:
                    selected_positions.append((ry, rx))
                    
                    # Add marker
                    marker = Circle((rx, ry), radius=0.5, color='red', 
                                   fill=True, zorder=10)
                    ax_probe.add_patch(marker)
                    markers.append(marker)
                    
                    # Add label
                    label = ax_probe.text(rx, ry, str(len(selected_positions)), 
                                         color='white', fontsize=8, ha='center', 
                                         va='center', fontweight='bold', zorder=11)
                    markers.append(label)
                    
                    update_display()
        
        def clear_all(event):
            """Clear all selected positions."""
            selected_positions.clear()
            for marker in markers:
                marker.remove()
            markers.clear()
            ax_dp1.clear()
            ax_dp1.axis('off')
            ax_dp2.clear()
            ax_dp2.axis('off')
            update_display()
        
        # Connect events
        fig.canvas.mpl_connect('button_press_event', onclick)
        btn_clear.on_clicked(clear_all)
        
        plt.show()
        
        return selected_positions

    def visualize_selected_patterns(self, *args, **kwargs):
        """Show the diffraction patterns at chosen positions.

        Forwards to :func:`quantem.diffraction.peak_visualization.visualize_selected_patterns`,
        supplying this object's data. See that function for the parameters.
        """

        return _visualize_selected_patterns(self.dataset_cartesian, *args, **kwargs)
    
    
    def interactive_probe_selector_widget(self, probe_map=None, cmap='viridis'):
        """
        Enhanced interactive GUI using ipywidgets for fine-tuning positions.
        
        Parameters
        ----------
        probe_map : ndarray, optional
            2D array to display as the probe position map
        cmap : str
            Colormap for the probe map
            
        Returns
        -------
        selected_positions : list of tuples
            List of (ry, rx) coordinates of selected positions
        """
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        
        Ry, Rx = self.dataset_cartesian.shape[:2]
        
        # Create default probe map if not provided
        if probe_map is None:
            probe_map = np.mean(self.dataset_cartesian.array, axis=(2, 3))
        
        # Storage
        selected_positions = []
        
        # Create output widgets
        output_plot = widgets.Output()
        output_list = widgets.Output()
        
        def update_plot():
            """Update the main plot with markers."""
            with output_plot:
                clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(8, 6))
                
                im = ax.imshow(probe_map, cmap=cmap, origin='lower', 
                              interpolation='nearest', aspect='auto')
                ax.set_title('Probe Position Map (Click to add point)', fontsize=12)
                ax.set_xlabel('Rx')
                ax.set_ylabel('Ry')
                plt.colorbar(im, ax=ax, label='Intensity')
                
                # Add markers
                for idx, (ry, rx) in enumerate(selected_positions):
                    circle = Circle((rx, ry), radius=0.5, color='red', 
                                   fill=True, zorder=10)
                    ax.add_patch(circle)
                    ax.text(rx, ry, str(idx+1), color='white', fontsize=8, 
                           ha='center', va='center', fontweight='bold', zorder=11)
                
                def onclick(event):
                    if event.inaxes == ax and event.button == 1:
                        rx = int(np.round(event.xdata))
                        ry = int(np.round(event.ydata))
                        if 0 <= ry < Ry and 0 <= rx < Rx:
                            selected_positions.append((ry, rx))
                            update_plot()
                            update_list()
                
                fig.canvas.mpl_connect('button_press_event', onclick)
                plt.show()
        
        def update_list():
            """Update the position list with controls."""
            with output_list:
                clear_output(wait=True)
                
                if not selected_positions:
                    print("No positions selected")
                    return
                
                for idx, (ry, rx) in enumerate(selected_positions):
                    print(f"--- Position {idx+1} ---")
                    
                    # Create sliders for fine-tuning
                    ry_slider = widgets.IntSlider(
                        value=ry, min=0, max=Ry-1, step=1,
                        description=f'Ry {idx+1}:', continuous_update=False
                    )
                    rx_slider = widgets.IntSlider(
                        value=rx, min=0, max=Rx-1, step=1,
                        description=f'Rx {idx+1}:', continuous_update=False
                    )
                    
                    def make_update(i):
                        def update_position(change):
                            selected_positions[i] = (ry_slider.value, rx_slider.value)
                            update_plot()
                        return update_position
                    
                    ry_slider.observe(make_update(idx), names='value')
                    rx_slider.observe(make_update(idx), names='value')
                    
                    # Delete button
                    delete_btn = widgets.Button(description=f'Delete {idx+1}', 
                                                button_style='danger')
                    
                    def make_delete(i):
                        def delete_position(b):
                            del selected_positions[i]
                            update_plot()
                            update_list()
                        return delete_position
                    
                    delete_btn.on_click(make_delete(idx))
                    
                    display(widgets.HBox([ry_slider, rx_slider, delete_btn]))
                
                # Clear all button
                clear_btn = widgets.Button(description='Clear All', button_style='warning')
                def clear_all(b):
                    selected_positions.clear()
                    update_plot()
                    update_list()
                clear_btn.on_click(clear_all)
                
                display(clear_btn)
        
        # Layout
        ui = widgets.VBox([
            widgets.HBox([output_plot, output_list])
        ])
        
        display(ui)
        update_plot()
        update_list()
        
        return selected_positions
