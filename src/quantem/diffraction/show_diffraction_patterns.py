"""Interactive tilt-series viewer for forward 4D-STEM diffraction patterns.

Two panels driven by one set of controls:

- **left (3D)**: a lab-frame schematic — the sample box (rotated by the sample
  tilt, back-face edges dashed), the material voxels, the four scan-corner
  rays, and the tiled diffraction image projected onto a detector plane below
  the sample.
- **right (2D)**: the same tiled diffraction image, flat, at full resolution.

Works on either a `SimDiffractionTomography` (showing its measured/ground-truth
patterns) or a `DiffractionTomography` reconstruction (showing patterns
forward-simulated from the recovered volume). Two viewers can be linked by
passing the first call's return value as ``controls=`` to the second, so both
step through tilts and adjust display together::

    controls = show_diffraction_patterns(sim, sim_measurements, title="ground truth")
    show_diffraction_patterns(recon, recon_measurements, controls=controls,
                              title="reconstruction")

Following the `show_2d` pattern this is a plain function returning a controls
handle, rather than a widget subclass.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["show_diffraction_patterns"]


# ------------------------------------------------------------------ geometry --
def _tile_4dstem(array4d: np.ndarray, power: float, rot_k: int) -> np.ndarray:
    """Tile (n_slow, n_fast, dp_y, dp_x) into one fftshifted 2D image."""
    shifted = np.fft.fftshift(np.asarray(array4d), axes=(-2, -1))
    n_slow, n_fast, dp_y, dp_x = shifted.shape
    tiled = shifted.transpose(0, 2, 1, 3).reshape(n_slow * dp_y, n_fast * dp_x)
    return np.rot90(tiled, k=rot_k) ** power


def _block_max(img: np.ndarray, factor: int) -> np.ndarray:
    """Integer max-pool downsample so bright single-pixel features survive."""
    if factor <= 1:
        return img
    H, W = img.shape
    Hc, Wc = (H // factor) * factor, (W // factor) * factor
    return img[:Hc, :Wc].reshape(Hc // factor, factor, Wc // factor, factor).max(axis=(1, 3))


def _shift_corners(corners: list, k: int) -> list:
    k = k % 4
    return corners[k:] + corners[:k]


def _box_corners(nx, ny, nz):
    xmax, ymax, zmax = nx - 1, ny - 1, nz - 1
    return np.array(
        [
            [0, 0, 0], [xmax, 0, 0], [xmax, ymax, 0], [0, ymax, 0],
            [0, 0, zmax], [xmax, 0, zmax], [xmax, ymax, zmax], [0, ymax, zmax],
        ],
        dtype=float,
    )


# corners 0,3,4,7 have x=0 (back face); 1,2,5,6 have x=xmax (front). Tilt is
# about X only, so this front/back split is valid at every tilt.
_BOX_EDGES_SOLID = [
    (1, 2), (5, 6), (1, 5), (2, 6),
    (0, 1), (2, 3), (4, 5), (6, 7),
]
_BOX_EDGES_DASHED = [(3, 0), (7, 4), (0, 4), (3, 7)]


def _rotate_around(points, R, origin):
    return origin + R.apply(points - origin)


def _resolve_tilt(meas, fallback):
    t = meas.metadata.get("tilt_x_deg")
    if t is not None:
        return float(t)
    zxz = meas.metadata.get("zxz_deg")
    if zxz is not None:
        return float(zxz[1])
    return float(fallback)


# --------------------------------------------------------------- main factory --
def show_diffraction_patterns(
    obj,
    datasets,
    tilts_deg=None,
    scan_step=None,
    controls=None,
    title="",
    rot_k=2,
    downsample=None,
    image_plane_scale=2.4,
    l_above_phys=100.0,
    l_below_phys=150.0,
    camera_elev=15.0,
    camera_azim=0.0,
    figsize=(11, 5.5),
    dpi=120,
):
    """Display a linked 3D-schematic + tiled-pattern tilt-series viewer.

    Parameters
    ----------
    obj:
        `DiffractionTomography` / `SimDiffractionTomography` supplying the
        geometry (real-space shape, sampling, material mask). For a
        reconstruction, its `sf_learned` volume is used when present.
    datasets:
        The diffraction patterns to display, one `Dataset4dstem` per tilt —
        e.g. the simulated measurements, or patterns forward-simulated from a
        reconstructed volume. A list (any order) or a {tilt: dataset} dict.
    tilts_deg:
        Tilt angle per dataset (degrees). If None, read from each dataset's
        metadata (`tilt_x_deg`, else `zxz_deg[1]`).
    scan_step:
        Physical scan step (A). If None, taken from each dataset's sampling.
    controls:
        Handle returned by a previous call. When given, this viewer shares that
        call's tilt / power / display / playback controls (for side-by-side
        comparison) instead of creating its own.
    title:
        Label shown above the panels.

    Returns
    -------
    dict — a controls handle; pass as `controls=` to a later call to link.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d')
    import ipywidgets as widgets
    from IPython.display import display

    # --- normalize datasets + tilts to sorted-by-tilt lists ------------------
    if isinstance(datasets, dict):
        items = sorted(datasets.items(), key=lambda kv: kv[0])
        tilt_vals = [float(k) for k, _ in items]
        ds_list = [v for _, v in items]
    else:
        ds_list = list(datasets)
        if tilts_deg is not None:
            tilt_vals = [float(t) for t in tilts_deg]
        else:
            tilt_vals = [_resolve_tilt(m, i) for i, m in enumerate(ds_list)]
        order = np.argsort(tilt_vals)
        tilt_vals = [tilt_vals[i] for i in order]
        ds_list = [ds_list[i] for i in order]

    n_tilts = len(ds_list)
    arrays = [np.asarray(d.array) for d in ds_list]
    intensity_max = [float(a.max()) for a in arrays]

    if scan_step is None:
        scan_step = float(ds_list[0].sampling[0])

    # --- geometry (uses sf_learned volume when present, e.g. a reconstruction)
    real_shape = tuple(int(n) for n in obj.real_shape)
    nx, ny, nz = real_shape
    sampling3 = np.asarray(obj.dataset.sampling[:3], dtype=float)
    material_coords = np.argwhere(np.asarray(obj._get_material_mask())).astype(float)
    scan_origin_idx = (np.asarray(real_shape, dtype=float) - 1) / 2.0

    n_slow, n_fast = int(arrays[0].shape[0]), int(arrays[0].shape[1])
    step_voxel = scan_step / sampling3[0]

    # Downsample the 3D image-plane texture only when the tiled image is large
    # (plot_surface is slow at full res). For small tiles do NOT downsample —
    # a fixed factor would crush single-pixel diffraction spots to nothing.
    _tile_shape = _tile_4dstem(arrays[0], 1.0, rot_k).shape
    ds_factor = downsample if downsample is not None else max(1, round(max(_tile_shape) / 256))

    view_center = (scan_origin_idx[0], scan_origin_idx[1], scan_origin_idx[2] + 5.0)
    view_halfrange = 0.5 * max(nx, ny, nz) + max(l_below_phys / sampling3[2], 6.0) * 0.35

    def show_view(tilt_idx, power, vmin, vmax):
        tilt = tilt_vals[int(tilt_idx)]
        img_full = _tile_4dstem(arrays[int(tilt_idx)], power, rot_k)
        imax = intensity_max[int(tilt_idx)]
        lo, hi = sorted((float(vmin), float(vmax)))
        disp_vmin = (lo * imax) ** power
        disp_vmax = (hi * imax) ** power
        if disp_vmax <= disp_vmin:
            disp_vmax = disp_vmin + 1e-12

        R_sample = Rotation.from_euler("zxz", [0.0, tilt, 0.0], degrees=True)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax2d = fig.add_subplot(1, 2, 2)

        corners_rot = _rotate_around(_box_corners(nx, ny, nz), R_sample, scan_origin_idx)
        for edges, ls in [(_BOX_EDGES_SOLID, "-"), (_BOX_EDGES_DASHED, (0, (4, 3)))]:
            for a, b in edges:
                ax3d.plot(
                    [corners_rot[a, 0], corners_rot[b, 0]],
                    [corners_rot[a, 1], corners_rot[b, 1]],
                    [corners_rot[a, 2], corners_rot[b, 2]],
                    color="0.2", lw=2.0, linestyle=ls,
                )

        if len(material_coords) > 0:
            mat_rot = _rotate_around(material_coords, R_sample, scan_origin_idx)
            ax3d.scatter(
                mat_rot[:, 0], mat_rot[:, 1], mat_rot[:, 2],
                c="darkorange", s=90, alpha=0.55, edgecolors="black",
                linewidths=0.4, marker="o", depthshade=False,
            )

        z_above = scan_origin_idx[2] - l_above_phys / sampling3[2]
        z_below = scan_origin_idx[2] + l_below_phys / sampling3[2]

        corner_indices = [(0, 0), (0, n_fast - 1), (n_slow - 1, n_fast - 1), (n_slow - 1, 0)]
        image_plane_corners = []
        for j, i in corner_indices:
            sv = j - (n_slow - 1) / 2.0
            su = i - (n_fast - 1) / 2.0
            lab_xy = scan_origin_idx[:2] + np.array([su, sv]) * step_voxel
            ax3d.plot(
                [lab_xy[0], lab_xy[0]], [lab_xy[1], lab_xy[1]], [z_above, z_below],
                color="lime", lw=1.6,
            )
            ax3d.scatter(
                [lab_xy[0], lab_xy[0]], [lab_xy[1], lab_xy[1]], [z_above, z_below],
                color="lime", s=22, depthshade=False,
            )
            sv_i = (sv + (0.5 if sv > 0 else -0.5)) * image_plane_scale
            su_i = (su + (0.5 if su > 0 else -0.5)) * image_plane_scale
            img_xy = scan_origin_idx[:2] + np.array([su_i, sv_i]) * step_voxel
            image_plane_corners.append(np.array([img_xy[0], img_xy[1], z_below]))

        P00, P01, P11, P10 = _shift_corners(image_plane_corners, rot_k)
        img_3d = _block_max(img_full, ds_factor)
        H, W = img_3d.shape
        ss = np.linspace(0, 1, H + 1)
        tt = np.linspace(0, 1, W + 1)
        SS, TT = np.meshgrid(ss, tt, indexing="ij")
        positions = (
            (1 - SS)[..., None] * (1 - TT)[..., None] * P00
            + (1 - SS)[..., None] * TT[..., None] * P01
            + SS[..., None] * (1 - TT)[..., None] * P10
            + SS[..., None] * TT[..., None] * P11
        )
        img_norm = np.clip((img_3d - disp_vmin) / (disp_vmax - disp_vmin), 0.0, 1.0)
        colors = plt.get_cmap("inferno")(img_norm)
        ax3d.plot_surface(
            positions[..., 0], positions[..., 1], positions[..., 2],
            facecolors=colors, shade=False, edgecolor="none", linewidth=0,
            antialiased=False, rcount=positions.shape[0] + 1, ccount=positions.shape[1] + 1,
        )

        cx, cy, cz = view_center
        ax3d.set_xlim(cx - view_halfrange, cx + view_halfrange)
        ax3d.set_ylim(cy - view_halfrange, cy + view_halfrange)
        ax3d.set_zlim(cz - view_halfrange, cz + view_halfrange)
        ax3d.invert_zaxis()
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.set_proj_type("persp")
        ax3d.view_init(elev=camera_elev, azim=camera_azim)
        ax3d.set_axis_off()
        ax3d.set_title(f"{title}  (tilt = {tilt:+.1f} deg)" if title else f"tilt = {tilt:+.1f} deg",
                       fontsize=11)

        ax2d.imshow(img_full, cmap="inferno", vmin=disp_vmin, vmax=disp_vmax,
                    interpolation="nearest")
        ax2d.set_title(f"tiled patterns  -  index {int(tilt_idx)} / {n_tilts - 1}", fontsize=11)
        ax2d.axis("off")

        plt.tight_layout()
        plt.show()

    # --- controls (created once, shared via the returned handle) -------------
    if controls is None:
        slider_kw = dict(continuous_update=False, layout=widgets.Layout(width="280px"))
        tilt_slider = widgets.IntSlider(
            min=0, max=n_tilts - 1, value=n_tilts // 2, step=1,
            description="tilt idx", **slider_kw,
        )
        power_slider = widgets.FloatSlider(
            min=0.1, max=1.0, step=0.05, value=0.5, description="power", **slider_kw,
        )
        vmin_slider = widgets.FloatLogSlider(
            base=10, min=-6, max=0, step=0.05, value=1e-6,
            description="vmin/max", readout_format=".4f", **slider_kw,
        )
        vmax_slider = widgets.FloatLogSlider(
            base=10, min=-6, max=0, step=0.05, value=0.02,
            description="vmax/max", readout_format=".4f", **slider_kw,
        )
        play_max = max(2 * (n_tilts - 1) - 1, 0)
        play = widgets.Play(
            value=tilt_slider.value, min=0, max=play_max, step=1, interval=250,
            description="play", repeat=True, show_repeat=False,
            layout=widgets.Layout(width="140px"),
        )
        bounce_toggle = widgets.ToggleButton(
            value=True, description="bounce", icon="retweet",
            layout=widgets.Layout(width="140px"),
        )
        state = {"driving": False}

        def _on_play(change):
            state["driving"] = True
            try:
                k = int(change.new)
                mk = n_tilts - 1
                if mk == 0:
                    idx = 0
                elif bounce_toggle.value:
                    idx = k if k <= mk else 2 * mk - k
                else:
                    idx = k % n_tilts
                if tilt_slider.value != idx:
                    tilt_slider.value = idx
            finally:
                state["driving"] = False

        def _on_slider(change):
            if state["driving"]:
                return
            if play.value != int(change.new):
                play.value = int(change.new)

        play.observe(_on_play, names="value")
        tilt_slider.observe(_on_slider, names="value")

        controls = {
            "tilt": tilt_slider, "power": power_slider,
            "vmin": vmin_slider, "vmax": vmax_slider,
            "play": play, "bounce": bounce_toggle, "n_tilts": n_tilts,
        }
        ui = widgets.HBox([
            widgets.VBox([play, bounce_toggle]),
            widgets.VBox([tilt_slider, power_slider]),
            widgets.VBox([vmin_slider, vmax_slider]),
        ])
        display(ui)

    out = widgets.interactive_output(
        show_view,
        {
            "tilt_idx": controls["tilt"],
            "power": controls["power"],
            "vmin": controls["vmin"],
            "vmax": controls["vmax"],
        },
    )
    display(out)
    return controls
