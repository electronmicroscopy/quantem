from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from quantem.core.visualization.visualization_utils import ScalebarConfig, add_scalebar_to_ax


def plot_strain_panels(
    e_uu: np.ndarray,
    e_vv: np.ndarray,
    e_uv: np.ndarray,
    rotation: np.ndarray,
    mask: np.ndarray | None,
    u_ref: np.ndarray | None,
    v_ref: np.ndarray | None,
    ds_shape: tuple[int, ...],
    ds_sampling: float = 1.0,
    ds_units: str = "pixels",
    strain_range_percent: tuple[float, float] = (-3.0, 3.0),
    rotation_range_degrees: tuple[float, float] = (-2.0, 2.0),
    mask_range: tuple[float, float] = (0.0, 1.0),
    roi: np.ndarray | None = None,
    plot_rotation: bool = True,
    plot_gvecs: bool = False,
    plot_scalebar: bool = False,
    cmap_strain: str = "RdBu_r",
    cmap_rotation: str = "PiYG",
    layout: str = "horizontal",
    rotate_strain: bool = False,
    rotate_title: bool = False,
    plot_dilation: bool = False,
    figsize: tuple[float, float] | None = None,
    panel_titles: tuple[str, str, str] | None = None,
    **kwargs,
):
    """Render strain (e_uu, e_vv, e_uv) and rotation panels.

    Strain arrays are fractional (multiplied by 100 for display); ``rotation`` is
    in radians (converted to degrees for display). ``panel_titles`` overrides the
    three strain-panel titles (e.g. to label the raw row/col reference frame).

    The mask modulates panel brightness (black where masked out). ``mask_range``
    ``(low, high)`` remaps it linearly before display: mask values ``>= high`` show
    full color, ``<= low`` go black, and values between ramp from black to full.
    The default ``(0.0, 1.0)`` leaves the already-normalized mask unchanged.

    When ``roi`` (a boolean ``(scan_row, scan_col)`` array) is given, positions
    inside it are drawn in color and positions outside it in greyscale (the same
    field, desaturated), so a chosen reference region stands out from its context.
    """
    if mask is None:
        mask = np.ones(ds_shape[:2])

    # remap the mask brightness onto the [low, high] window: <= low -> black,
    # >= high -> full color, linear between. default (0, 1) is a no-op.
    low, high = float(mask_range[0]), float(mask_range[1])
    if high > low:
        mask = np.clip((np.asarray(mask, dtype=float) - low) / (high - low), 0.0, 1.0)
    else:
        mask = (np.asarray(mask, dtype=float) >= high).astype(float)

    if cmap_rotation is None:
        cmap_rotation = cmap_strain

    if layout not in ["horizontal", "vertical"]:
        raise ValueError("layout must be 'horizontal' or 'vertical'")

    ncols = 4 if plot_rotation else 3
    is_horizontal = layout == "horizontal"
    if plot_dilation:
        ncols=3
        plot_rotation = True

    n_strain = 2 if plot_dilation else 3

    if figsize is None:
        figsize = (8, 3) if is_horizontal else (6, 6)

    if is_horizontal:
        fig, ax = plt.subplots(1, ncols, figsize=figsize)
    else:
        fig, ax = plt.subplots(ncols, 1, figsize=figsize)

    cm_strain = plt.get_cmap(cmap_strain).copy()
    cm_strain.set_bad(color="black")
    cm_rot = plt.get_cmap(cmap_rotation).copy()
    cm_rot.set_bad(color="black")

    euu_pct = e_uu * 100
    evv_pct = e_vv * 100
    euv_pct = e_uv * 100
    rot_deg = np.rad2deg(rotation)

    roi_bool = None if roi is None else np.asarray(roi).astype(bool)
    gray_cm = plt.get_cmap("gray").copy()
    gray_cm.set_bad(color="black")

    def _roi_compose(norm_vals, color_cm):
        """Color the field inside the ROI; show it in greyscale outside the ROI."""
        rgb = color_cm(norm_vals)[:, :, :3]
        if roi_bool is None:
            return rgb
        rgb_gray = gray_cm(norm_vals)[:, :, :3]
        return np.where(roi_bool[:, :, np.newaxis], rgb, rgb_gray)

    norm_strain = Normalize(vmin=strain_range_percent[0], vmax=strain_range_percent[1])
    euu_disp = _roi_compose(norm_strain(euu_pct), cm_strain)
    evv_disp = _roi_compose(norm_strain(evv_pct), cm_strain)
    euv_disp = _roi_compose(norm_strain(euv_pct), cm_strain)

    if rotate_strain:
        euu_disp = euu_disp.transpose(1,0,2)
        evv_disp = evv_disp.transpose(1,0,2)
        euv_disp = euv_disp.transpose(1,0,2)
        mask = mask.T
    
    if plot_dilation:
        etot_pct = (e_uu + e_vv) * 100
        etot_disp = _roi_compose(norm_strain(etot_pct), cm_strain)
        if rotate_strain:
            etot_disp = etot_disp.transpose(1,0,2)
        ax[0].imshow(euu_disp * mask[:, :, np.newaxis])
        ax[1].imshow(euv_disp * mask[:, :, np.newaxis])
    else:
        ax[0].imshow(euu_disp * mask[:, :, np.newaxis])
        ax[1].imshow(evv_disp * mask[:, :, np.newaxis])
        ax[2].imshow(euv_disp * mask[:, :, np.newaxis])

    ref_dim = figsize[1] if is_horizontal else figsize[0]
    fs_threshold = 3.0
    fs_scale = min(1.0, max(0.5, ref_dim / fs_threshold))
    title_fs = 16 * fs_scale
    tick_fs = 12 * fs_scale
    title_val = 'vertical' if rotate_title else 'horizontal'
    if panel_titles is None and not plot_dilation:
        panel_titles = (
            r"$\epsilon_{uu}$ $\updownarrow$",
            r"$\epsilon_{vv}$ $\leftrightarrow$",
            r"$\epsilon_{uv}$ $\nwarrow\!\!\!\!\!\!\!\!\searrow$",
        )
        ax[0].set_title(panel_titles[0], fontsize=title_fs, rotation=title_val)
        ax[1].set_title(panel_titles[1], fontsize=title_fs, rotation=title_val)
        ax[2].set_title(panel_titles[2], fontsize=title_fs, rotation=title_val)
    if plot_dilation and panel_titles is None:
        panel_titles = (
            r"$\epsilon_{uu} + \epsilon_{vv}$",
            r"$\epsilon_{uv}$ $\nwarrow\!\!\!\!\!\!\!\!\!\:\searrow$",
            ""
        )
        ax[0].set_title(panel_titles[0], fontsize=title_fs, rotation=title_val)
        ax[1].set_title(panel_titles[1], fontsize=title_fs, rotation=title_val)

    if plot_rotation:
        norm_rot = Normalize(vmin=rotation_range_degrees[0], vmax=rotation_range_degrees[1])
        rot_disp = _roi_compose(norm_rot(rot_deg), cm_rot)
        if rotate_strain: rot_disp = rot_disp.transpose(1,0,2)
        ax[-1].imshow(rot_disp * mask[:, :, np.newaxis])
        ax[-1].set_title(r"Rotation $\circlearrowleft$", fontsize=title_fs, rotation=title_val)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
        a.set_facecolor("black")
        a.set_aspect("equal")

    if plot_scalebar:
        scalebar_kwargs = {}
        for key, value in kwargs.items():
            if key.startswith("scalebar_"):
                scalebar_key = key[len("scalebar_"):]
                scalebar_kwargs[scalebar_key] = value

        scalebar_defaults = {
            "sampling": ds_sampling,
            "units": ds_units,
            "length": None,
            "width_px": 1,
            "pad_px": 0.5,
            "color": "black",
            "loc": "lower left",
            "fontsize": 12,
            "bold": True,
        }
        scalebar_defaults.update(scalebar_kwargs)
        scalebar_config = ScalebarConfig(**scalebar_defaults)
        add_scalebar_to_ax(
            ax[0],
            array_size=int(ds_shape[0]),
            sampling=scalebar_config.sampling,
            length_units=scalebar_config.length,
            units=scalebar_config.units,
            width_px=scalebar_config.width_px,
            pad_px=scalebar_config.pad_px,
            color=scalebar_config.color,
            loc=scalebar_config.loc,
            fontsize=scalebar_config.fontsize,
            bold=scalebar_config.bold,
        )

    cb_size = 0.02
    cb_pad = 0.03
    cb_min_len = 0.16

    def _finalize_layout():
        # set_aspect("equal") only resizes/recenters each panel at draw time, so
        # get_position() before a draw returns stale boxes -- placing the colorbars
        # and g-vector compass off those boxes then spills them off the figure.
        # Settle the layout cheaply first so every box read below is the real one.
        try:
            fig.draw_without_rendering()
        except AttributeError:  # matplotlib < 3.5
            fig.canvas.draw()

    if is_horizontal:
        # Reserve a bottom band wide enough for the colorbar + its tick labels and
        # title (fontsize 16) and a right band for the rotation-panel gap; widen the
        # right band when the g-vector compass is drawn in it. These keep the figure
        # usable when saved "as is" (no bbox_inches='tight').
        right = 0.78 if plot_gvecs else 0.93
        fig.subplots_adjust(left=0.04, right=right, top=0.88, bottom=0.24, wspace=0.05)
        if plot_rotation:
            # nudge the rotation panel right for a visual gap from the strain panels;
            # 0.03 stays inside the reserved right band so nothing is clipped.
            pos3 = ax[-1].get_position()
            ax[-1].set_position([pos3.x0 + 0.03, pos3.y0, pos3.width, pos3.height])
        _finalize_layout()

        cb_orientation = "horizontal"
        b0 = ax[0].get_position()
        b2 = ax[n_strain - 1].get_position()
        cb_y = b2.y0 - cb_pad - cb_size
        strain_cb_pos = [b0.x0, cb_y, b2.x1 - b0.x0, cb_size]

        if plot_rotation:
            b3 = ax[-1].get_position()
            rot_cb_w = max(b3.x1 - b3.x0, cb_min_len)
            rot_cb_cx = 0.5 * (b3.x0 + b3.x1)
            rot_cb_x0 = min(max(rot_cb_cx - 0.5 * rot_cb_w, 0.0), 0.99 - rot_cb_w)
            rot_cb_pos = [rot_cb_x0, cb_y, rot_cb_w, cb_size]
            last_pos = b3
        else:
            rot_cb_pos = None
            last_pos = b2

    else:
        # Top band for the panel titles, right band for the vertical colorbars + labels.
        fig.subplots_adjust(left=0.04, right=0.80, top=0.92, bottom=0.06, hspace=0.15)
        _finalize_layout()

        cb_orientation = "vertical"
        b0 = ax[0].get_position()
        b2 = ax[n_strain - 1].get_position()
        strain_cb_pos = [b0.x1 + cb_pad, b2.y0, cb_size, b0.y1 - b2.y0]

        if plot_rotation:
            b3 = ax[-1].get_position()
            rot_cb_h = max(b3.y1 - b3.y0, cb_min_len)
            rot_cb_cy = 0.5 * (b3.y0 + b3.y1)
            rot_cb_y0 = min(max(rot_cb_cy - 0.5 * rot_cb_h, 0.0), 0.99 - rot_cb_h)
            rot_cb_pos = [b0.x1 + cb_pad, rot_cb_y0, cb_size, rot_cb_h]
            last_pos = b3
        else:
            rot_cb_pos = None
            last_pos = b2

    cax1 = fig.add_axes(strain_cb_pos)
    sm_strain = ScalarMappable(norm=norm_strain, cmap=cm_strain)
    cbar1 = fig.colorbar(sm_strain, cax=cax1, orientation=cb_orientation)
    cbar1.set_label("Strain", fontsize=title_fs)
    cbar1.formatter = FuncFormatter(lambda v, _pos: f"{v:g}%")
    cbar1.update_ticks()
    cbar1.ax.tick_params(labelsize=tick_fs)

    if plot_rotation and rot_cb_pos is not None:
        cax2 = fig.add_axes(rot_cb_pos)
        sm_rot = ScalarMappable(norm=norm_rot, cmap=cm_rot)
        cbar2 = fig.colorbar(sm_rot, cax=cax2, orientation=cb_orientation)
        cbar2.set_label("Rotation", fontsize=title_fs)
        cbar2.formatter = FuncFormatter(lambda v, _pos: f"{v:g}°")
        cbar2.locator = MaxNLocator(nbins=2)
        cbar2.update_ticks()
        cbar2.ax.tick_params(labelsize=tick_fs)

    if plot_gvecs:
        if u_ref is None or v_ref is None:
            print("Warning: u_ref and v_ref not found. Call fit_strain() first.")
            return fig, ax

        # The compass goes in the reserved margin beside the last panel; clamp its
        # right edge to 0.99 so it never spills off the figure when saved "as is".
        if is_horizontal:
            ref_left = last_pos.x1 + 0.005
            ref_width = min(last_pos.width, 0.99 - ref_left)
            ref_ax = fig.add_axes([ref_left, last_pos.y0, ref_width, last_pos.height])
        else:
            ref_left = min(last_pos.x1 + 0.18, 0.74)
            ref_width = min(last_pos.width, 0.99 - ref_left)
            ref_ax = fig.add_axes([ref_left, last_pos.y0, ref_width, last_pos.height])

        ref_ax.set_xlim(-1.5, 1.5)
        ref_ax.set_ylim(-1.5, 1.5)
        ref_ax.set_aspect("equal")
        ref_ax.axis("off")
        u_norm = u_ref / np.linalg.norm(u_ref)
        v_norm = v_ref / np.linalg.norm(v_ref)

        u_row, u_col = u_norm
        v_row, v_col = v_norm
        arrow_props_ref = dict(arrowstyle="->", lw=3, mutation_scale=25)

        u_arrow = FancyArrowPatch(
            (0, 0), (u_col, -u_row),
            color="darkred", **arrow_props_ref
        )
        ref_ax.add_patch(u_arrow)

        v_arrow = FancyArrowPatch(
            (0, 0), (v_col, -v_row),
            color="darkblue", **arrow_props_ref
        )
        ref_ax.add_patch(v_arrow)
        ref_ax.text(u_col * 1.3, -u_row * 1.3, r"$\mathbf{g}_{1}$",
                    fontsize=14, fontweight="bold", color="darkred",
                    ha="center", va="center")

        ref_ax.text(v_col * 1.3, -v_row * 1.3, r"$\mathbf{g}_{2}$",
                    fontsize=14, fontweight="bold", color="darkblue",
                    ha="center", va="center")

    return fig, ax


def plot_strain_precision_histogram(
    edges: np.ndarray,
    counts: np.ndarray,
    precision: dict[str, float],
    component: str,
    unit: str,
    *,
    figsize: tuple[float, float] = (6.0, 4.0),
):
    """Weighted histogram of the local-deviation strain precision.

    ``edges``/``counts`` describe the (mask-weighted, normalized) distribution of the
    chosen ``component`` deviation in display units (``unit``). ``precision`` is the
    weighted-median local deviation per component (used for the annotation box); the
    plotted component's median is marked with a solid line.
    """
    fig, ax = plt.subplots(figsize=figsize)
    edges = np.asarray(edges, dtype=float)
    counts = np.asarray(counts, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    ax.bar(centers, counts, width=widths, align="center",
           color="#4C72B0", edgecolor="white", linewidth=0.3)

    median_value = precision[component]
    if np.isfinite(median_value):
        ax.axvline(median_value, color="crimson", ls="-", lw=2)
        # label the line inline -- a legend box here would sit on top of the info box.
        # Put the text on whichever side of the line keeps it clear of the right box.
        span = float(edges[-1] - edges[0])
        on_right = span > 0 and (median_value - edges[0]) / span > 0.5
        ax.annotate(
            f"median = {median_value:.3g} {unit}",
            xy=(median_value, 0.96), xycoords=("data", "axes fraction"),
            xytext=(-6 if on_right else 6, 0), textcoords="offset points",
            ha="right" if on_right else "left", va="top",
            color="crimson", fontsize=9,
        )

    label = "combined" if component == "combined" else component
    ax.set_xlabel(f"{label} deviation ({unit})", fontsize=12)
    ax.set_ylabel("weighted fraction", fontsize=12)
    ax.set_title("Strain precision  (median local deviation)", fontsize=13)
    ax.tick_params(labelsize=10)

    annotation = "\n".join(
        [
            r"median:",
            rf"  $\epsilon_{{uu}}$: {precision['e_uu']:.3g} %",
            rf"  $\epsilon_{{vv}}$: {precision['e_vv']:.3g} %",
            rf"  $\epsilon_{{uv}}$: {precision['e_uv']:.3g} %",
            rf"  rotation: {precision['rotation']:.3g} °",
            rf"  combined: {precision['combined']:.3g} %",
        ]
    )
    ax.text(0.97, 0.97, annotation, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    fig.tight_layout()
    return fig, ax