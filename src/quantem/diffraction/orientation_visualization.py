"""Visualization of orientation maps: IPF maps, pattern overlays, pole figures."""

from __future__ import annotations

import numpy as np
import torch

from quantem.core.visualization.visualization_utils import add_scalebar_to_ax
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.rotations import quat_to_matrix

# one color per candidate phase, used consistently across every plot
DEFAULT_PHASE_COLORS = np.array(
    [
        [1.00, 0.80, 0.25],  # gold
        [0.25, 0.80, 0.90],  # cyan
        [0.45, 0.80, 0.50],  # green
        [0.85, 0.50, 0.80],  # purple
    ]
)
ORIGIN_COLOR = "#2ca02c"
MEASURED_COLOR = "0.15"
IPF_GAMMA = 0.4  # <1 expands the white / mixed-color regions of the wedge
# additive corner colors: full red, green capped to avoid the fluorescent
# look, blue lifted off pure dark blue; pairwise sums give near-max-chroma
# yellow / cyan / violet and the three together give white
IPF_CORNER_COLORS = np.array(
    [
        [1.00, 0.00, 0.00],
        [0.00, 0.70, 0.00],
        [0.00, 0.30, 1.00],
    ]
)


def _bary_to_rgb(w: np.ndarray) -> np.ndarray:
    """Barycentric wedge weights (..., 3) to RGB via the additive anchors."""
    w = np.clip(w, 0, None)
    w = w / np.clip(w.max(axis=-1, keepdims=True), 1e-12, None)
    w = w**IPF_GAMMA
    return np.clip(w @ IPF_CORNER_COLORS, 0, 1)


def _parse_direction(direction) -> torch.Tensor:
    """Lab direction for IPF coloring: 'z' (beam), 'r' (scan row), 'c' (scan
    col), an in-plane angle in degrees (measured from the column axis toward
    the row axis), or an explicit [row, col] / [row, col, z] vector."""
    if isinstance(direction, str):
        return {
            "z": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            "r": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            "c": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            # back-compat aliases
            "x": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            "y": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        }[direction]
    if isinstance(direction, (int, float)):
        th = np.deg2rad(float(direction))
        return torch.tensor([np.sin(th), np.cos(th), 0.0], dtype=torch.float64)
    v = torch.as_tensor(direction, dtype=torch.float64).reshape(-1)
    if v.numel() == 2:
        v = torch.cat([v, torch.zeros(1, dtype=torch.float64)])
    return v / torch.linalg.norm(v)


def _reduce_to_wedge(vectors: torch.Tensor, crystal: Crystal) -> torch.Tensor:
    """Map crystal-frame directions into the fundamental zone-axis wedge.

    Applies all proper symmetry rotations to +/- v and returns, per input
    vector, the orbit member inside the wedge (all barycentric coordinates
    with respect to the wedge corners non-negative).
    """
    corners = crystal.zone_axis_wedge()
    if corners is None:
        # hemisphere fallback: canonicalize to upper hemisphere only
        v = vectors.clone()
        v[v[..., 2] < 0] *= -1
        return v
    Rs = quat_to_matrix(crystal.sym_quats)  # (S, 3, 3)
    v = vectors.reshape(-1, 3)
    orbit = torch.cat(
        [torch.einsum("sij,nj->nsi", Rs, v), torch.einsum("sij,nj->nsi", Rs, -v)],
        dim=1,
    )  # (N, 2S, 3)
    A_inv = torch.linalg.inv(corners.to(vectors.dtype).T)
    w = torch.einsum("ij,nsj->nsi", A_inv, orbit)
    inside = (w > -1e-6).all(dim=-1)
    idx = inside.to(torch.float64).argmax(dim=1)
    out = orbit[torch.arange(v.shape[0]), idx]
    return out.reshape(vectors.shape)


def ipf_color(
    orientations: torch.Tensor,
    crystal: Crystal,
    direction: str | torch.Tensor = "z",
) -> np.ndarray:
    """Inverse pole figure RGB colors for orientations.

    Parameters
    ----------
    orientations : torch.Tensor
        Quaternions (..., 4).
    crystal : Crystal
        Provides symmetry and the fundamental wedge.
    direction : {"x", "y", "z"} | torch.Tensor, default="z"
        Lab direction whose crystal-frame coordinates are colored; "z" is the
        beam direction (zone-axis map).

    Returns
    -------
    np.ndarray
        RGB array (..., 3) in [0, 1].
    """
    direction = _parse_direction(direction)
    R = quat_to_matrix(orientations)  # v_lab = R v_crystal
    v_crystal = torch.einsum("...ji,j->...i", R, direction)
    v = _reduce_to_wedge(v_crystal, crystal)

    corners = crystal.zone_axis_wedge()
    if corners is None:
        # hemisphere: hue from azimuth, saturation from polar angle
        from matplotlib.colors import hsv_to_rgb

        az = (torch.atan2(v[..., 1], v[..., 0]) / (2 * np.pi)) % 1.0
        pol = torch.acos(v[..., 2].clamp(-1, 1)) / (np.pi / 2)
        hsv = torch.stack((az, pol.clamp(0, 1), torch.ones_like(az)), dim=-1)
        return hsv_to_rgb(hsv.numpy())

    A_inv = torch.linalg.inv(corners.to(v.dtype).T)
    w = torch.einsum("ij,...j->...i", A_inv, v)
    return _bary_to_rgb(w.numpy())


def wedge_legend(
    crystal: Crystal,
    ax,
    n: int = 120,
    labels: bool = True,
    orientation: str = "horizontal",
    fontsize: int = 11,
) -> None:
    """Draw the labeled IPF color triangle for the crystal's fundamental wedge.

    Corner direction labels use 4-index Miller-Bravais symbols for hexagonal
    and trigonal crystals. orientation="vertical" rotates the wedge 90
    degrees to fill a tall side panel.
    """
    corners = crystal.zone_axis_wedge()
    if corners is None:
        ax.axis("off")
        return
    c = corners.numpy()
    # vertical: rotate so the [001]/[0001] corner sits at the TOP of the
    # tall panel with the wedge hanging straight down (the rotation aligns
    # the wedge's angular bisector with the downward direction)
    if orientation == "vertical":
        az = [
            np.arctan2(c[k, 1] / (1 + c[k, 2]), c[k, 0] / (1 + c[k, 2]))
            for k in (1, 2)
        ]
        th = -np.pi / 2 - (az[0] + az[1]) / 2
    else:
        th = 0.0
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    cxy = np.stack(
        [c[:, 0] / (1 + c[:, 2]), c[:, 1] / (1 + c[:, 2])], axis=1
    ) @ rot.T
    cx, cy = cxy[:, 0], cxy[:, 1]

    # rasterize the wedge interior: invert the stereographic projection on a
    # pixel grid and alpha-mask outside the wedge, so no color spills past
    # the outline
    m = 8
    x0, x1 = cx.min() - 0.02, cx.max() + 0.02
    y0, y1 = cy.min() - 0.02, cy.max() + 0.02
    X, Y = np.meshgrid(
        np.linspace(x0, x1, n * m), np.linspace(y0, y1, n * m), indexing="xy"
    )
    Xu = np.cos(th) * X + np.sin(th) * Y
    Yu = -np.sin(th) * X + np.cos(th) * Y
    denom = 1 + Xu**2 + Yu**2
    V = np.stack(
        [2 * Xu / denom, 2 * Yu / denom, (1 - Xu**2 - Yu**2) / denom], axis=-1
    )
    A_inv = np.linalg.inv(c.T)
    W = V @ A_inv.T
    inside = (W > -1e-9).all(axis=-1)
    rgba = np.zeros(X.shape + (4,))
    rgba[..., :3] = _bary_to_rgb(W)
    rgba[..., 3] = inside
    ax.imshow(rgba, extent=(x0, x1, y0, y1), origin="lower", interpolation="nearest")
    # black outline along the wedge edges (stereographic great-circle arcs)
    tt = np.linspace(0, 1, 60)[:, None]
    for i0, i1 in ((0, 1), (1, 2), (2, 0)):
        e = c[i0][None, :] * (1 - tt) + c[i1][None, :] * tt
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        exy = np.stack(
            [e[:, 0] / (1 + e[:, 2]), e[:, 1] / (1 + e[:, 2])], axis=1
        ) @ rot.T
        ax.plot(exy[:, 0], exy[:, 1], color="k", lw=1.2)
    if labels:
        names = crystal.zone_axis_wedge_labels() or ["", "", ""]
        center = np.array([cx.mean(), cy.mean()])
        for xi, yi, name in zip(cx, cy, names):
            out = np.array([xi, yi]) - center
            norm = np.linalg.norm(out)
            off = out / norm * 0.08 if norm > 1e-6 else np.array([0, -0.08])
            ha = "left" if off[0] > 0.02 else ("right" if off[0] < -0.02 else "center")
            va = "bottom" if off[1] > 0.02 else ("top" if off[1] < -0.02 else "center")
            ax.text(xi + off[0], yi + off[1], name, fontsize=fontsize, ha=ha, va=va)
    span_x = cx.max() - cx.min()
    span_y = cy.max() - cy.min()
    pad = 0.45 * max(span_x, span_y, 0.2)
    ax.set_xlim(cx.min() - pad, cx.max() + pad)
    ax.set_ylim(cy.min() - pad, cy.max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_orientation_map(
    om,
    direction: str = "z",
    match: int = 0,
    mask: np.ndarray | None = None,
    scalebar: dict | None = None,
    figax=None,
    legend: bool = True,
    axsize: tuple[float, float] = (9.0, 4.5),
):
    """IPF-colored orientation map with the wedge legend in an adjacent panel.

    Parameters
    ----------
    om : OrientationMap
        Matched orientation map.
    direction : {"x", "y", "z"}, default="z"
        Lab direction to color ("z" = zone axis).
    match : int, default=0
        Which match index to plot.
    mask : np.ndarray | None
        Multiplied into the RGB image (e.g. a phase or reliability mask).
    scalebar : dict | None
        Real-space scale bar, e.g. {"sampling": 30, "units": "A"}.
    figax : (fig, (ax_map, ax_legend)) | (fig, ax_map) | None
        Existing axes; with a single axis the legend is skipped.
    """
    import matplotlib.pyplot as plt

    assert om.quats is not None
    rgb = ipf_color(om.quats[..., match, :], om.crystal, direction)
    if mask is not None:
        rgb = rgb * np.asarray(mask, dtype=float)[..., None]

    ax_leg = None
    if figax is None:
        if legend:
            fig, (ax, ax_leg) = plt.subplots(
                1,
                2,
                figsize=(axsize[0] * 1.3, axsize[1]),
                gridspec_kw={"width_ratios": [4, 1]},
            )
        else:
            fig, ax = plt.subplots(figsize=axsize)
    else:
        fig, axs = figax
        if isinstance(axs, (tuple, list, np.ndarray)) and len(axs) == 2:
            ax, ax_leg = axs
        else:
            ax = axs
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if isinstance(direction, str) and direction == "z":
        ax.set_title(f"{om.crystal.name}  out-of-plane orientation")
    else:
        # arrow for the colored in-plane direction lives in the title,
        # like the strain-map axis annotations
        if isinstance(direction, str):
            arrow = {
                "r": r"$\downarrow$",
                "c": r"$\rightarrow$",
                "x": r"$\downarrow$",
                "y": r"$\rightarrow$",
            }.get(direction, "")
            label = {"x": "r", "y": "c"}.get(direction, direction)
            ax.set_title(
                f"{om.crystal.name}  in-plane orientation  {label} {arrow}"
            )
        elif isinstance(direction, (int, float)):
            ax.set_title(
                f"{om.crystal.name}  in-plane orientation  ({direction:g}\u00b0)"
            )
        else:
            ax.set_title(f"{om.crystal.name}  in-plane orientation")
    if scalebar is not None:
        add_scalebar_to_ax(
            ax,
            array_size=rgb.shape[1],
            sampling=scalebar.get("sampling", 1.0),
            length_units=scalebar.get("length", None),
            units=scalebar.get("units", "pixels"),
            width_px=rgb.shape[0] / 40,
            pad_px=rgb.shape[0] / 80,
            color=scalebar.get("color", "white"),
            loc="lower right",
        )
    if legend and ax_leg is not None:
        wedge_legend(om.crystal, ax_leg, orientation="vertical")
    return fig, ax


def plot_pattern_matches(
    orientation_maps,
    positions,
    dataset=None,
    pixel_size: float | None = None,
    origins: np.ndarray | None = None,
    matches=(0, 1),
    colors=None,
    power: float = 0.4,
    q_max_plot: float | None = None,
    scalebar: bool = True,
    show_measured: bool = True,
    marker_scale: float = 250.0,
    axsize: tuple[float, float] = (3.1, 3.1),
):
    """Candidate matches side by side, py4DSTEM style.

    One row per probe position; one column per (crystal, match) candidate,
    so alpha and beta fits sit next to each other for direct comparison.
    Measured peaks are solid gray disks with area proportional to intensity;
    each candidate's simulation is drawn as colored crosses (red, then blue
    by default) also sized by intensity. With `dataset` given, the raw
    pattern is shown behind the crosses instead of the gray disks.

    Parameters
    ----------
    orientation_maps : OrientationMap | list[OrientationMap]
        Matched orientation maps sharing the same peaks.
    positions : list[tuple[int, int]]
        (row, col) probe positions to plot.
    dataset : Dataset4dstem | None
        If given, the diffraction pattern is shown behind the overlay and
        the gray measured disks are omitted.
    pixel_size : float | None
        Reciprocal pixel size (1/Angstroms per pixel); required with dataset.
    origins : np.ndarray | None
        (scan_r, scan_c, 2) fitted origins from measure_origins(); aligns
        the background pattern with the origin-corrected peaks.
    matches : tuple[int, ...], default=(0, 1)
        Match indices per crystal.
    colors : list | None
        One color per crystal; defaults to red, blue, green, purple.
    """
    import matplotlib.pyplot as plt

    oms = (
        list(orientation_maps)
        if isinstance(orientation_maps, (list, tuple))
        else [orientation_maps]
    )
    if colors is None:
        colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    peaks = oms[0].peaks
    fields = peaks.fields
    ix = [fields.index(f) for f in ("qx", "qy", "intensity")]
    # peaks may be rotated into the scan frame; the raw detector image is
    # not, so rotate all overlay coordinates back to the detector frame
    rot_deg = float(peaks.metadata.get("rotation_ccw_deg", 0.0) or 0.0)
    th = np.deg2rad(-rot_deg)
    rot_back = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    panels = [(i, m) for i in range(len(oms)) for m in matches]
    n_r, n_c = len(positions), len(panels)
    fig, axs = plt.subplots(
        n_r,
        n_c,
        figsize=(axsize[0] * n_c, axsize[1] * n_r + 0.2),
        squeeze=False,
    )
    ordinal = ["1st", "2nd", "3rd"] + [f"{k + 1}th" for k in range(3, 9)]
    for pi, (rx, ry) in enumerate(positions):
        data = peaks[rx, ry].array.copy()
        rc = data[:, [ix[0], ix[1]]] @ rot_back.T
        data[:, ix[0]] = rc[:, 0]
        data[:, ix[1]] = rc[:, 1]
        w_meas = data[:, ix[2]].clip(min=0)
        w_meas = w_meas / max(w_meas.max(), 1e-12)
        if q_max_plot is None:
            if dataset is not None and pixel_size is not None:
                q_lim = dataset.shape[-1] / 2 * pixel_size
            else:
                qr = np.hypot(data[:, ix[0]], data[:, ix[1]])
                q_lim = 1.1 * qr.max() if qr.size else 1.0
        else:
            q_lim = q_max_plot
        for ci, (i_om, m) in enumerate(panels):
            om = oms[i_om]
            ax = axs[pi, ci]
            if dataset is not None and pixel_size is not None:
                H, W = dataset.shape[-2], dataset.shape[-1]
                if origins is not None:
                    o_r, o_c = origins[rx, ry]
                else:
                    o_r, o_c = H / 2, W / 2
                # pixel j has center (j - origin) * pixel_size; array edges
                # sit half a pixel beyond the first/last centers
                ax.imshow(
                    np.asarray(dataset.array[rx, ry]) ** power,
                    cmap="gray_r",
                    extent=(
                        (-0.5 - o_c) * pixel_size,
                        (W - 0.5 - o_c) * pixel_size,
                        (H - 0.5 - o_r) * pixel_size,
                        (-0.5 - o_r) * pixel_size,
                    ),
                )
            elif show_measured:
                ax.scatter(
                    data[:, ix[1]],
                    data[:, ix[0]],
                    s=marker_scale * w_meas,
                    color="0.75",
                    lw=0,
                )
            sim = om.generate_pattern(rx, ry, match=m)
            I = sim["intensity"].numpy()
            sim_rc = (
                np.stack([sim["qx"].numpy(), sim["qy"].numpy()], axis=1)
                @ rot_back.T
            )
            if I.size:
                ax.scatter(
                    sim_rc[:, 1],
                    sim_rc[:, 0],
                    s=marker_scale * I / I.max(),
                    marker="+",
                    color=colors[i_om % len(colors)],
                    lw=1.8,
                )
            ax.set_xlim(-q_lim, q_lim)
            ax.set_ylim(q_lim, -q_lim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.set_title(
                "%s %s\ncorr = %.2f"
                % (om.crystal.name, ordinal[m], float(om.corr[rx, ry, m])),
                fontsize=9,
            )
            if scalebar and pi == n_r - 1 and ci == 0:
                add_scalebar_to_ax(
                    ax,
                    array_size=2 * q_lim,
                    sampling=1.0,
                    length_units=0.5,
                    units="A^-1",
                    width_px=q_lim / 45,
                    pad_px=q_lim / 60,
                    color="black",
                    loc="lower right",
                    fontsize=9,
                )
    fig.tight_layout()
    return fig, axs

def plot_cluster_map(
    om,
    clusters: dict,
    colors: np.ndarray | None = None,
    scalebar: dict | None = None,
    figax=None,
):
    """Map of orientation clusters (variants), one color per cluster."""
    import matplotlib.pyplot as plt

    labels = clusters["labels"].numpy()
    if colors is None:
        colors = CLUSTER_COLORS
    K = int(labels.max()) + 1
    rgb = np.zeros(labels.shape + (3,))
    for k in range(K):
        rgb[labels == k] = colors[k % len(colors)]

    if figax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig, ax = figax
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{om.crystal.name} orientation clusters")
    handles = [
        plt.Line2D([0], [0], marker="s", ls="", color=colors[k % len(colors)],
                   label=f"{k + 1}  ({int(clusters['sizes'][k])} px)")
        for k in range(K)
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    if scalebar is not None:
        add_scalebar_to_ax(
            ax,
            array_size=rgb.shape[1],
            sampling=scalebar.get("sampling", 1.0),
            length_units=scalebar.get("length", None),
            units=scalebar.get("units", "pixels"),
            width_px=rgb.shape[0] / 40,
            pad_px=rgb.shape[0] / 80,
            color=scalebar.get("color", "white"),
            loc="lower right",
        )
    return fig, ax


def _pole_points(
    quats: torch.Tensor, crystal: Crystal, pole, mask=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stereographic (x, y, weight) of all symmetry-equivalent poles."""
    q = quats.reshape(-1, 4)
    p = torch.as_tensor(pole, dtype=torch.float64)
    p = p / torch.linalg.norm(p)
    Rs = quat_to_matrix(crystal.sym_quats)
    fam = torch.einsum("sij,j->si", Rs, p)
    fam = torch.unique(torch.round(fam / 1e-6) * 1e-6, dim=0)
    fam = torch.cat([fam, -fam])
    n_fam = fam.shape[0]
    R = quat_to_matrix(q)
    poles_lab = torch.einsum("nij,sj->nsi", R, fam)
    if mask is not None:
        w = torch.as_tensor(np.asarray(mask, dtype=float)).reshape(-1)
    else:
        w = torch.ones(q.shape[0], dtype=torch.float64)
    w_all = w[:, None].expand(-1, n_fam).reshape(-1)
    src_all = (
        torch.arange(q.shape[0])[:, None].expand(-1, n_fam).reshape(-1)
    )
    v = poles_lab.reshape(-1, 3)
    keep = (v[:, 2] > -1e-8) & (w_all > 0)
    v, w_keep, src = v[keep], w_all[keep], src_all[keep]
    x = (v[:, 0] / (1 + v[:, 2])).numpy()
    y = (v[:, 1] / (1 + v[:, 2])).numpy()
    return x, y, w_keep.numpy(), src.numpy()


def _pole_scatter_xy(quats: torch.Tensor, crystal: Crystal, pole) -> np.ndarray:
    """Stereographic (x, y) of all symmetry-equivalent poles for orientations."""
    p = torch.as_tensor(pole, dtype=torch.float64)
    p = p / torch.linalg.norm(p)
    Rs = quat_to_matrix(crystal.sym_quats)
    fam = torch.einsum("sij,j->si", Rs, p)
    fam = torch.unique(torch.round(fam / 1e-6) * 1e-6, dim=0)
    fam = torch.cat([fam, -fam])
    R = quat_to_matrix(torch.atleast_2d(quats))
    v = torch.einsum("nij,sj->nsi", R, fam).reshape(-1, 3)
    v = v[v[:, 2] > -1e-8]
    x = (v[:, 0] / (1 + v[:, 2])).numpy()
    y = (v[:, 1] / (1 + v[:, 2])).numpy()
    return np.stack((x, y), axis=1)


def plot_cluster_pole_figure(
    om,
    clusters: dict,
    pole,
    pole_label: str = "",
    overlay: dict | None = None,
    colors: np.ndarray | None = None,
    figax=None,
):
    """Pole figure of the cluster mean orientations, one color per cluster.

    Parameters
    ----------
    om : OrientationMap
        Provides the crystal symmetry of the clustered phase.
    clusters : dict
        Output of OrientationMap.cluster_orientations().
    pole : array-like
        Crystal-Cartesian pole direction of the plotted family.
    overlay : dict | None
        Second pole family drawn as open markers, e.g.
        {"quats": q_beta_mean, "crystal": ti_beta, "pole": (1, 1, 0),
        "label": "<110> beta"} -- the standard Burgers relationship check.
    """
    import matplotlib.pyplot as plt

    if colors is None:
        colors = CLUSTER_COLORS
    if figax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
    else:
        fig, ax = figax

    theta = np.linspace(0, 2 * np.pi, 361)
    for pol_deg in range(15, 91, 15):
        r = np.tan(np.deg2rad(pol_deg) / 2)
        lw = 1.0 if pol_deg == 90 else 0.4
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="0.75", lw=lw)
    for az in range(0, 180, 15):
        ca, sa = np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az))
        ax.plot([-ca, ca], [-sa, sa], color="0.85", lw=0.4)

    K = clusters["mean_quats"].shape[0]
    for k in range(K):
        xy = _pole_scatter_xy(clusters["mean_quats"][k], om.crystal, pole)
        ax.scatter(
            xy[:, 0], xy[:, 1], s=60, marker="h",
            color=colors[k % len(colors)], edgecolors="k", lw=0.4,
            label=f"{pole_label} {k + 1}",
        )
    if overlay is not None:
        xy = _pole_scatter_xy(
            overlay["quats"], overlay["crystal"], overlay["pole"]
        )
        ax.scatter(
            xy[:, 0], xy[:, 1], s=70, marker="D", facecolors="none",
            edgecolors="k", lw=1.0, label=overlay.get("label", "overlay"),
        )
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.set_title(f"{om.crystal.name} cluster pole figure")
    return fig, ax


def plot_pole_figure(
    om,
    pole: list[float] | torch.Tensor = (0, 0, 1),
    match: int = 0,
    mask: np.ndarray | None = None,
    bins: int = 181,
    color_by: str = "density",
    int_range: tuple[float, float] = (0.0, 1.0),
    smooth_sigma: float = 1.5,
    label: str | None = None,
    grid: bool = True,
    overlay: dict | None = None,
    figax=None,
):
    """Stereographic pole figure of a crystal direction family over the map.

    For every probe position, all symmetry equivalents of `pole` are rotated
    into the lab frame; upper-hemisphere poles are projected
    stereographically and accumulated into a 2D histogram.

    Parameters
    ----------
    om : OrientationMap
        Matched orientation map.
    pole : array-like, default=(0, 0, 1)
        Crystal direction (Cartesian) of the pole family.
    mask : np.ndarray | None
        Per-position weights (e.g. phase mask).
    bins : int, default=181
        Histogram bins across the stereographic disk.
    color_by : {"density", "ipf"}, default="density"
        "density": grayscale-to-color histogram. "ipf": each contribution is
        colored by the IPF (zone axis) color of its probe position, and the
        histogram density sets the brightness, black background.
    int_range : tuple, default=(0.0, 1.0)
        Density display range as fractions of the maximum bin: values below
        the lower limit saturate to black, above the upper limit to full
        brightness.
    label : str | None
        Annotation for the pole family, e.g. "(0001)" or "{110}".
    grid : bool, default=True
        Draw polar-angle circles and azimuth spokes every 30 degrees.
    """
    import matplotlib.pyplot as plt

    assert om.quats is not None
    qmap = om.quats[..., match, :]
    x, y, wk, src = _pole_points(qmap, om.crystal, pole, mask)

    rng = [[-1.05, 1.05], [-1.05, 1.05]]
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=rng, weights=wk)
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter

        H = gaussian_filter(H, smooth_sigma)
    lo, hi = int_range
    # normalize against a high percentile of the occupied bins, not the single
    # hottest bin -- one large uniform grain would otherwise black out the rest
    occupied = H[H > 0]
    h_ref = np.percentile(occupied, 98) if occupied.size else 1.0
    Hn = np.clip((H / max(h_ref, 1e-12) - lo) / max(hi - lo, 1e-12), 0, 1)

    if color_by == "ipf":
        # white background: blend from white toward the per-position IPF
        # color as the histogram density rises
        rgb_pos = ipf_color(om.quats[..., match, :], om.crystal, "z").reshape(-1, 3)
        rgb_all = rgb_pos[src]
        img = np.zeros((bins, bins, 3))
        cnt = np.zeros((bins, bins))
        ii = np.clip(((x - rng[0][0]) / (rng[0][1] - rng[0][0]) * bins).astype(int), 0, bins - 1)
        jj = np.clip(((y - rng[1][0]) / (rng[1][1] - rng[1][0]) * bins).astype(int), 0, bins - 1)
        for k in range(3):
            np.add.at(img[..., k], (ii, jj), rgb_all[:, k] * wk)
        np.add.at(cnt, (ii, jj), wk)
        if smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter

            for k in range(3):
                img[..., k] = gaussian_filter(img[..., k], smooth_sigma)
            cnt = gaussian_filter(cnt, smooth_sigma)
        img = img / np.maximum(cnt[..., None], 1e-12)
        # white background blending toward the IPF color as density rises --
        # keeps dark corner colors (blue) legible
        disp = 1.0 - Hn[..., None] * (1.0 - img)
    else:
        import matplotlib.cm as cm

        # white -> yellow -> red -> black with increasing density
        disp = cm.hot_r(Hn)[..., :3]

    # display in the image frame: horizontal = c (col, rightward), vertical =
    # r (row, downward), matching the orientation maps -- H is indexed
    # [row-bin, col-bin] so no transpose, origin upper
    yy, xx = np.meshgrid(
        0.5 * (ye[:-1] + ye[1:]), 0.5 * (xe[:-1] + xe[1:]), indexing="ij"
    )
    disp = disp.copy()
    disp[(xx**2 + yy**2).T > 1.0] = 1.0

    ax_leg = None
    if figax is None:
        if color_by == "ipf":
            fig, (ax, ax_leg) = plt.subplots(
                1, 2, figsize=(7.2, 5.5), gridspec_kw={"width_ratios": [4, 1]}
            )
        else:
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
    else:
        fig, axs = figax
        if isinstance(axs, (tuple, list, np.ndarray)) and len(np.atleast_1d(axs)) == 2:
            ax, ax_leg = axs
        else:
            ax = axs
    ax.imshow(
        disp,
        extent=(ye[0], ye[-1], xe[-1], xe[0]),
        interpolation="nearest",
    )
    if grid:
        theta = np.linspace(0, 2 * np.pi, 361)
        for pol_deg in (30, 60, 90):
            r = np.tan(np.deg2rad(pol_deg) / 2)
            lw = 1.0 if pol_deg == 90 else 0.5
            ax.plot(r * np.cos(theta), r * np.sin(theta), color="0.65", lw=lw)
            if pol_deg < 90:
                ax.text(
                    r * np.cos(np.deg2rad(45)),
                    r * np.sin(np.deg2rad(45)),
                    f"{pol_deg}°",
                    color="0.45",
                    fontsize=7,
                    ha="center",
                    va="center",
                )
        for az in range(0, 180, 30):
            ca, sa = np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az))
            ax.plot([-ca, ca], [-sa, sa], color="0.85", lw=0.4)
        # compact scan-axes glyph, top-left corner: the pole figure is in
        # the scan (image) frame -- c rightward, r downward
        gx, gy = -1.06, -1.06
        for dx, dy, lbl, ha, va in (
            (0.22, 0.0, "c", "left", "center"),
            (0.0, 0.22, "r", "center", "top"),
        ):
            ax.annotate(
                "", xy=(gx + dx, gy + dy), xytext=(gx, gy),
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.2),
                annotation_clip=False,
            )
            ax.text(
                gx + dx * 1.25, gy + dy * 1.25, lbl,
                fontsize=9, ha=ha, va=va, color="0.3",
            )
        ax.text(
            gx - 0.03, gy - 0.06, "scan axes", fontsize=7, ha="left",
            va="bottom", color="0.45",
        )
    if overlay is not None:
        if "om" in overlay:
            # raw-histogram contour of another map's pole family
            o_om = overlay["om"]
            ox, oy, ow, _ = _pole_points(
                o_om.quats[..., overlay.get("match", 0), :],
                o_om.crystal,
                overlay["pole"],
                overlay.get("mask"),
            )
            Ho, oxe, oye = np.histogram2d(ox, oy, bins=bins, range=rng, weights=ow)
            if (Ho > 0).any():
                from scipy.ndimage import gaussian_filter

                Ho = gaussian_filter(Ho, max(smooth_sigma, 1.0))
                lev = np.percentile(Ho[Ho > 0], 99) * np.array([0.3, 0.7])
                xc = 0.5 * (oxe[:-1] + oxe[1:])
                yc = 0.5 * (oye[:-1] + oye[1:])
                # image frame: horizontal = col bins, vertical = row bins
                ax.contour(
                    yc, xc, Ho, levels=lev, colors="k",
                    linewidths=[0.7, 1.3], alpha=0.85,
                )
                ax.plot([], [], color="k", lw=1.2, label=overlay.get("label", "overlay"))
        else:
            oxy = _pole_scatter_xy(overlay["quats"], overlay["crystal"], overlay["pole"])
            ax.scatter(
                oxy[:, 1], oxy[:, 0], s=80, marker="D", facecolors="none",
                edgecolors="k", lw=1.2, label=overlay.get("label", "overlay"),
            )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(1.2, -1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    title = f"{om.crystal.name} pole figure"
    if label is not None:
        title += f"  {label}"
    ax.set_title(title)
    if ax_leg is not None:
        if color_by == "ipf":
            wedge_legend(om.crystal, ax_leg, orientation="vertical")
        else:
            ax_leg.axis("off")
    return fig, ax
