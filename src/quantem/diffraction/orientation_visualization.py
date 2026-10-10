"""Visualization of orientation maps: IPF maps, pattern overlays, pole figures."""

from __future__ import annotations

import numpy as np
import torch

from quantem.core.visualization.visualization_utils import add_scalebar_to_ax
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.rotations import quat_to_matrix

# one color per candidate phase, used consistently across every plot; index
# with phase_color_cycle() so more crystals than colors cycle through them
DEFAULT_PHASE_COLORS = np.array(
    [
        [1.00, 0.80, 0.25],  # gold
        [0.25, 0.80, 0.90],  # cyan
        [0.45, 0.80, 0.50],  # green
        [0.85, 0.50, 0.80],  # purple
    ]
)
# exponent on the distance from the wedge centre: >1 widens the white centre
# and softens the transition into it, <1 shrinks it (much below 0.5 leaves a
# bright point at the centre)
IPF_SATURATION_POWER = 0.5
# colorfulness relative to the corner colors: 1 keeps them exact, lower values
# wash the whole wedge toward white
IPF_CHROMA = 1.0
# corner colors: full red, green capped to avoid the fluorescent look, blue
# lifted off pure dark blue; pairwise blends give near-max-chroma orange,
# cyan and violet along the edges
IPF_CORNER_COLORS = np.array(
    [
        [1.00, 0.00, 0.00],
        [0.00, 0.70, 0.00],
        [0.00, 0.30, 1.00],
    ]
)
# cluster / grain label colors (tab10 cycle)
CLUSTER_COLORS = [
    (0.122, 0.467, 0.706),
    (1.0, 0.498, 0.055),
    (0.173, 0.627, 0.173),
    (0.839, 0.153, 0.157),
    (0.580, 0.404, 0.741),
    (0.549, 0.337, 0.294),
    (0.890, 0.467, 0.761),
    (0.498, 0.498, 0.498),
    (0.737, 0.741, 0.133),
    (0.090, 0.745, 0.812),
]


def phase_color_cycle(n: int, colors=None) -> np.ndarray:
    """One RGB color per phase, cycling through the palette.

    Parameters
    ----------
    n : int
        Number of phases.
    colors : sequence | None
        Palette of K matplotlib colors (RGB rows or names); None takes
        `DEFAULT_PHASE_COLORS`.

    Returns
    -------
    np.ndarray
        (n, 3) RGB colors in [0, 1]; phase k takes palette entry k modulo K.
    """
    from matplotlib.colors import to_rgb

    palette = DEFAULT_PHASE_COLORS if colors is None else colors
    palette = np.array([to_rgb(c) for c in palette], dtype=float)
    return palette[np.arange(n) % palette.shape[0]]


def _bary_to_rgb(
    w: np.ndarray,
    saturation_power: float | None = None,
    chroma: float | None = None,
) -> np.ndarray:
    """Barycentric wedge weights (..., 3) to RGB.

    The corner colors are mixed additively with the weights scaled by their
    4-norm, a smooth stand-in for dividing by the largest weight: the mix
    stays bright between corners (orange, cyan and violet midway along the
    edges) and the corners keep their own colors. The mix is then faded
    toward white by 1 - 27 w0 w1 w2, which is zero at the wedge centre and
    one on every edge. Both terms are smooth in the weights, so the colors
    change gradually across the whole wedge, with no creases where one
    corner takes over from another, and the mix never leaves the sRGB gamut.

    Parameters
    ----------
    w : np.ndarray
        Barycentric coordinates in the fundamental wedge, (..., 3).
    saturation_power : float | None
        Exponent on the distance from the wedge centre. Above 1 the color
        builds up more slowly away from the centre, widening the white region
        and softening the transition into it; below 1 shrinks it. None takes
        `IPF_SATURATION_POWER`.
    chroma : float | None
        Colorfulness relative to the corner colors: 1 keeps them exact, lower
        washes the wedge toward white. None takes `IPF_CHROMA`.

    Returns
    -------
    np.ndarray
        RGB array (..., 3) in [0, 1].
    """
    saturation_power = IPF_SATURATION_POWER if saturation_power is None else saturation_power
    chroma = IPF_CHROMA if chroma is None else chroma
    w = np.clip(np.asarray(w, dtype=float), 0, None)
    w = w / np.clip(w.sum(axis=-1, keepdims=True), 1e-12, None)
    u = w / np.clip((w**4).sum(axis=-1, keepdims=True) ** 0.25, 1e-12, None)
    rgb = u @ IPF_CORNER_COLORS
    # distance from the centre: 0 there, 1 on the edges, smooth in w
    r = np.clip(1.0 - 27.0 * w[..., 0] * w[..., 1] * w[..., 2], 0, 1) ** saturation_power
    return np.clip(1.0 - np.clip(chroma * r, 0, 1)[..., None] * (1.0 - rgb), 0, 1)


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
    Rs = quat_to_matrix(crystal.sym_quats_matching)  # (S, 3, 3)
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
    saturation_power: float | None = None,
    chroma: float | None = None,
) -> np.ndarray:
    """Inverse pole figure RGB colors for orientations.

    Parameters
    ----------
    orientations : torch.Tensor
        Quaternions (..., 4).
    crystal : Crystal
        Provides symmetry and the fundamental wedge.
    direction : {"z", "r", "c"} | float | array-like, default="z"
        Lab direction whose crystal-frame coordinates are colored: "z" is
        the beam direction (zone-axis map), "r" the scan row axis and "c"
        the scan column axis ("x" and "y" are accepted as aliases of "r"
        and "c"). A number is an in-plane angle in degrees from the column
        axis toward the row axis; a 2 or 3 element vector is an explicit
        (row, col[, z]) direction.
    saturation_power : float | None
        Width of the white centre; see :func:`_bary_to_rgb`.
    chroma : float | None
        Colorfulness relative to the corner colors; see :func:`_bary_to_rgb`.

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
        sat = pol.clamp(0, 1) ** (
            IPF_SATURATION_POWER if saturation_power is None else saturation_power
        )
        hsv = torch.stack((az, sat, torch.ones_like(az)), dim=-1)
        return hsv_to_rgb(hsv.numpy())

    A_inv = torch.linalg.inv(corners.to(v.dtype).T)
    w = torch.einsum("ij,...j->...i", A_inv, v)
    return _bary_to_rgb(w.numpy(), saturation_power, chroma)


def fold_in_plane(quats: torch.Tensor, crystal: Crystal, strict: bool = False) -> torch.Tensor:
    """Fold the in-plane angle of each orientation by its projected symmetry.

    The zero-layer pattern of a zone axis can repeat more often under
    rotation about the beam than the crystal does, and where it does, two
    orientations produce the same measured pattern and the match returns one
    of them arbitrarily. Rotating each orientation about the beam into the
    first such sector makes those two identical, so any map colored from the
    result is continuous across the ambiguity.

    Positions whose pattern is no more symmetric than the crystal itself are
    returned unchanged, unless `strict`, which folds by the projected order
    everywhere.
    """
    from quantem.diffraction.rotations import (
        qmult,
        qnormalize,
        quat_from_axis_angle,
        quat_to_matrix,
    )

    q = torch.as_tensor(quats, dtype=torch.float64)
    shape = q.shape[:-1]
    flat = q.reshape(-1, 4)
    R = quat_to_matrix(flat)
    zone = R[:, 2, :]  # beam direction in crystal coordinates
    # the projected order is piecewise constant in the zone axis: evaluate it
    # once per distinct axis on a coarse grid
    key = torch.round(zone * 200) / 200
    uniq, inv = torch.unique(key, dim=0, return_inverse=True)
    n_proj = torch.as_tensor(
        np.asarray(crystal.projected_rotation_order(uniq.numpy())), dtype=torch.float64
    )[inv]
    if not strict:
        # only fold where the pattern is more symmetric than the crystal is
        # about that same axis, which is where the indexing is degenerate
        n_cryst = _crystal_rotation_order(uniq, crystal)[inv].to(torch.float64)
        n_proj = torch.where(n_proj > n_cryst, n_proj, torch.ones_like(n_proj))
    if bool((n_proj <= 1).all()):
        return q
    a_lab = R[:, :, 0]
    ang = torch.rad2deg(torch.atan2(a_lab[:, 0], a_lab[:, 1]))
    sector = 360.0 / n_proj
    delta = ang - (ang % sector)
    # the in-plane angle is measured from the column axis toward the row
    # axis, which runs opposite to a right-handed rotation about the beam
    beam = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    dq = quat_from_axis_angle(beam, torch.deg2rad(delta))
    return qnormalize(qmult(dq, flat)).reshape(*shape, 4)


def _crystal_rotation_order(axes: torch.Tensor, crystal: Crystal) -> torch.Tensor:
    """Order of the crystal's own rotation axis along each direction, (N,)."""
    from quantem.diffraction.rotations import quat_to_matrix

    Rs = quat_to_matrix(crystal.sym_quats)  # (S, 3, 3)
    u = axes / torch.linalg.norm(axes, dim=1, keepdim=True).clamp_min(1e-12)
    # an operation is a rotation about u when it leaves u fixed
    fixed = torch.einsum("sij,nj->nsi", Rs, u)
    keeps = (fixed - u[:, None, :]).norm(dim=-1) < 1e-6
    return keeps.sum(dim=1)


def wedge_legend(
    crystal: Crystal,
    ax,
    n: int = 120,
    labels: bool = True,
    orientation: str = "horizontal",
    fontsize: int = 11,
    saturation_power: float | None = None,
    chroma: float | None = None,
) -> None:
    """Draw the labeled IPF color triangle for the crystal's fundamental wedge.

    Corner direction labels use 4-index Miller-Bravais symbols for hexagonal
    and trigonal crystals. orientation="vertical" rotates the wedge 90
    degrees to fill a tall side panel.

    `saturation_power` and `chroma` go to :func:`_bary_to_rgb` and must match
    the map being labelled, which the plotting functions ensure.
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
        # bisector from the summed directions, not the mean of two angles,
        # which jumps by pi when a corner sits at +-180 degrees (-0.0 in y)
        d = sum(c[k, :2] / (1 + c[k, 2]) / np.linalg.norm(c[k, :2]) for k in (1, 2))
        th = -np.pi / 2 - np.arctan2(d[1], d[0])
    else:
        th = 0.0
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    cxy = np.stack([c[:, 0] / (1 + c[:, 2]), c[:, 1] / (1 + c[:, 2])], axis=1) @ rot.T
    cx, cy = cxy[:, 0], cxy[:, 1]

    # wedge edges: stereographic great-circle arcs, which can bulge past the
    # corners (the equator arc does once the wedge is rotated upright)
    tt = np.linspace(0, 1, 60)[:, None]
    edges = []
    for i0, i1 in ((0, 1), (1, 2), (2, 0)):
        e = c[i0][None, :] * (1 - tt) + c[i1][None, :] * tt
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        edges.append(np.stack([e[:, 0] / (1 + e[:, 2]), e[:, 1] / (1 + e[:, 2])], axis=1) @ rot.T)
    outline = np.concatenate(edges)

    # rasterize the wedge interior: invert the stereographic projection on a
    # pixel grid covering the whole outline and alpha-mask outside the
    # wedge, so no color spills past it and none is missing inside it
    m = 8
    x0, x1 = outline[:, 0].min() - 0.02, outline[:, 0].max() + 0.02
    y0, y1 = outline[:, 1].min() - 0.02, outline[:, 1].max() + 0.02
    X, Y = np.meshgrid(np.linspace(x0, x1, n * m), np.linspace(y0, y1, n * m), indexing="xy")
    Xu = np.cos(th) * X + np.sin(th) * Y
    Yu = -np.sin(th) * X + np.cos(th) * Y
    denom = 1 + Xu**2 + Yu**2
    V = np.stack([2 * Xu / denom, 2 * Yu / denom, (1 - Xu**2 - Yu**2) / denom], axis=-1)
    A_inv = np.linalg.inv(c.T)
    W = V @ A_inv.T
    inside = (W > -1e-9).all(axis=-1)
    rgba = np.zeros(X.shape + (4,))
    rgba[..., :3] = _bary_to_rgb(W, saturation_power, chroma)
    rgba[..., 3] = inside
    ax.imshow(rgba, extent=(x0, x1, y0, y1), origin="lower", interpolation="nearest")
    for exy in edges:
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
    ox, oy = outline[:, 0], outline[:, 1]
    pad = 0.45 * max(ox.max() - ox.min(), oy.max() - oy.min(), 0.2)
    ax.set_xlim(ox.min() - pad, ox.max() + pad)
    ax.set_ylim(oy.min() - pad, oy.max() + pad)
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
    crop: tuple[int, int, int, int] | None = None,
    title: str | None = None,
    fold: bool | str = "auto",
    smooth: dict | bool | None = None,
    saturation_power: float | None = None,
    chroma: float | None = None,
):
    """IPF-colored orientation map with the wedge legend in an adjacent panel.

    Parameters
    ----------
    om : OrientationMap
        Matched orientation map.
    direction : {"z", "r", "c"} | float | array-like, default="z"
        Lab direction to color: "z" the beam (zone axis), "r" the scan row
        axis, "c" the scan column axis ("x" and "y" are aliases of "r" and
        "c"), a number for an in-plane angle in degrees, or an explicit
        vector; see :func:`ipf_color`.
    match : int, default=0
        Which match index to plot.
    mask : np.ndarray | None
        Multiplied into the RGB image (e.g. a phase or reliability mask).
    figax : (fig, (ax_map, ax_legend)) | (fig, ax_map) | None
        Existing axes; with a single axis the legend is skipped.
    legend : bool, default=True
        Draw the IPF color wedge in a panel beside the map.
    axsize : tuple[float, float], default=(9.0, 4.5)
        Figure size in inches of the map panel; the legend panel widens the
        figure by 30%. Ignored when `figax` is given.
    fold : bool | "auto", default="auto"
        Fold the in-plane part of each orientation by the apparent
        rotational symmetry of its own zero-layer pattern
        (`Crystal.projected_rotation_order`) before coloring. Where that
        symmetry exceeds the crystal's own, as for a cubic crystal near
        <111>, two orientations give the same pattern and the match picks
        between them at random; folding gives them the same color, which
        removes jumps that no refinement can. It changes nothing for
        direction="z", whose color depends only on the zone axis, and
        nothing where the pattern is no more symmetric than the crystal.
        "auto" folds only when some position needs it.
    smooth : dict | bool | None
        Smooth the orientations for display only, leaving the stored ones at
        the fit to their own pattern. The average is bilateral and needs two
        widths, not one: `sigma_px` over probe positions and `sigma_deg` over
        misorientation, with `max_angle_deg` excluding anything further. The
        angular pair is what stops the average at a grain boundary, so pass
        a dict naming the values you want, such as
        {"sigma_px": 1.0, "sigma_deg": 1.0, "max_angle_deg": 5.0}, which is
        also what True uses.
    saturation_power : float | None
        Widens or narrows the white centre of the color wedge. Above 1 covers
        a wider range of orientations near the wedge centre and softens the
        transition into it. None takes `IPF_SATURATION_POWER`.
    chroma : float | None
        Colorfulness relative to the corner colors: 1 keeps them exact, lower
        washes the map toward white. None takes `IPF_CHROMA`.
        The legend is drawn with the same values.
    scalebar : dict | None
        Real-space scale bar, e.g. {"sampling": 30, "units": "A"}.
    crop : (r0, r1, c0, c1) | None
        Show only this window of the map (rows r0:r1, columns c0:c1).
    title : str | None
        Replaces the default title (crystal name and colored direction).

    Returns
    -------
    tuple
        ``(fig, ax)`` with `ax` the map panel.
    """
    import matplotlib.pyplot as plt

    assert om.quats is not None
    quats = om.quats[..., match, :]
    if smooth is not None and smooth is not False:
        if not isinstance(smooth, (dict, bool)):
            raise TypeError(
                "smooth must be a dict of widths or True; a bare number would set the "
                "spatial width and leave the angular tolerance at its default, which "
                "is the argument that keeps the average inside one grain"
            )
        quats = om.smoothed_quats(match=match, **(smooth if isinstance(smooth, dict) else {}))
    if fold and not (isinstance(direction, str) and direction == "z"):
        # the zone-axis color does not depend on the in-plane angle
        quats = fold_in_plane(quats, om.crystal, strict=fold != "auto")
    rgb = ipf_color(quats, om.crystal, direction, saturation_power, chroma)
    if mask is not None:
        rgb = rgb * np.asarray(mask, dtype=float)[..., None]
    if crop is not None:
        r0, r1, c0, c1 = crop
        rgb = rgb[r0:r1, c0:c1]

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
    if title is not None:
        ax.set_title(title)
    elif isinstance(direction, str) and direction == "z":
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
            ax.set_title(f"{om.crystal.name}  in-plane orientation  {label} {arrow}")
        elif isinstance(direction, (int, float)):
            ax.set_title(f"{om.crystal.name}  in-plane orientation  ({direction:g}\u00b0)")
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
        wedge_legend(
            om.crystal,
            ax_leg,
            orientation="vertical",
            saturation_power=saturation_power,
            chroma=chroma,
        )
    return fig, ax


def plot_pattern_matches(
    orientation_maps,
    positions,
    dataset=None,
    pixel_size: float | None = None,
    origins: np.ndarray | None = None,
    matches=(0, 1),
    colors=None,
    norm=None,
    sigma_plot: float | None = 1.0,
    q_max_plot: float | None = None,
    q_max_quantile: float = 0.98,
    scalebar: bool = True,
    show_measured: bool = True,
    marker_scale: float = 250.0,
    measured_scale: float | None = None,
    measured_power: float = 0.5,
    marker: str | None = None,
    transpose_plots: bool = False,
    axsize: tuple[float, float] = (3.1, 3.1),
):
    """Candidate matches side by side, py4DSTEM style.

    One row per probe position; one column per (crystal, match) candidate,
    so alpha and beta fits sit next to each other for direct comparison.
    Measured peaks are solid gray disks with area proportional to intensity;
    each candidate's simulation is drawn as colored markers in its phase
    color, also sized by intensity. With `dataset` given, the raw pattern is
    shown behind the markers instead of the gray disks.

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
        Reciprocal pixel size (1/Angstroms per pixel); required with
        `dataset`.
    origins : np.ndarray | None
        (scan_r, scan_c, 2) fitted origins from measure_origins(); aligns
        the background pattern with the origin-corrected peaks.
    matches : tuple[int, ...], default=(0, 1)
        Match indices per crystal. Indices a map does not hold (the second
        match of a map matched with `num_matches=1`) are skipped.
    norm : dict | str | None
        `norm` of `show_2d`, which draws the recorded pattern, e.g.
        {"power": 0.5, "upper_quantile": 0.98}. The default,
        {"power": 0.4, "upper_quantile": 0.999}, keeps the direct beam from
        flattening the disks.
    sigma_plot : float | None, default=1.0
        Gaussian blur (pixels) of the displayed pattern only, which makes
        the disks easier to see in low-dose data; None shows it raw.
    q_max_plot : float | None
        Half-width of every panel, 1/Angstroms. None fits it to the peaks
        actually plotted, using `q_max_quantile`.
    q_max_quantile : float, default=0.98
        Quantile of the measured peak radii that sets the automatic limit,
        used only when `q_max_plot` is None and no `dataset` is given. A few
        stray high-angle detections would otherwise set the scale for every
        panel and leave the pattern in the middle of empty space, so the
        default trims the furthest 2%. Pass 1.0 to enclose every peak.
    colors : list | None
        One color per crystal; defaults to `DEFAULT_PHASE_COLORS`, the
        palette of the phase map, cycled when there are more crystals.
    marker : str | None
        Matplotlib marker for the simulated peaks. The default is an open
        circle over a diffraction pattern, which leaves the measured disk
        visible inside it, and a plus over the gray measured peaks.
    measured_scale : float | None
        Largest marker area of the gray measured peaks, in points^2. The
        default fits the markers to the patterns shown: the largest disk is
        about 0.6 of the median spacing between neighbouring peaks, so dense
        patterns get small markers and sparse ones large, up to
        ``1.5 * marker_scale``. The direct beam is far brighter than the
        disks, so the areas are compressed by `measured_power` and floored,
        which keeps the weak spots visible.
    measured_power : float, default=0.5
        Compression applied to the measured intensities before sizing.
    scalebar : bool, default=True
        Draw a 0.5 1/Angstrom scale bar in the bottom-left panel.
    show_measured : bool, default=True
        Draw the measured peaks as gray disks. Ignored with `dataset`,
        where the recorded pattern is shown instead.
    marker_scale : float, default=250.0
        Marker area, in points^2, of the strongest simulated reflection;
        the others scale with intensity.
    transpose_plots : bool, default=False
        Panel layout only, nothing in the data is transposed. By default
        rows are probe positions and columns are candidates; True swaps
        them, giving one row per candidate across the positions, which fits
        a few candidates and many positions on a page.
    axsize : tuple[float, float], default=(3.1, 3.1)
        Size of one panel in inches.

    Returns
    -------
    tuple
        ``(fig, axs)`` with `axs` a 2D array of panels.

    Raises
    ------
    ValueError
        If `dataset` is given without `pixel_size`, or none of `matches`
        exists in any map.
    """
    import matplotlib.pyplot as plt

    from quantem.core.visualization import show_2d
    from quantem.diffraction.bragg_vectors_visualization import _blur

    oms = (
        list(orientation_maps)
        if isinstance(orientation_maps, (list, tuple))
        else [orientation_maps]
    )
    if dataset is not None and pixel_size is None:
        raise ValueError(
            "plot_pattern_matches needs pixel_size (1/Angstroms per pixel) to place the "
            "dataset behind the peaks"
        )
    colors = [tuple(c) for c in phase_color_cycle(len(oms), colors)]
    peaks = oms[0].peaks
    fields = peaks.fields
    ix = [fields.index(f) for f in ("qx", "qy", "intensity")]
    # peaks may be rotated into the scan frame; the raw detector image is
    # not, so rotate all overlay coordinates back to the detector frame
    rot_deg = float(peaks.metadata.get("rotation_ccw_deg", 0.0) or 0.0)
    th = np.deg2rad(-rot_deg)
    rot_back = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    # a map matched with fewer matches than requested has no panel for them
    panels = [(i, m) for i, om in enumerate(oms) for m in matches if 0 <= m < om.quats.shape[2]]
    if not panels:
        raise ValueError(f"none of matches={tuple(matches)} exists in the orientation maps")
    n_pos, n_pan = len(positions), len(panels)
    n_r, n_c = (n_pan, n_pos) if transpose_plots else (n_pos, n_pan)
    fig, axs = plt.subplots(
        n_r,
        n_c,
        figsize=(axsize[0] * n_c, axsize[1] * n_r + 0.2),
        squeeze=False,
    )
    over_image = dataset is not None
    if marker is None:
        marker = "o" if over_image else "+"
    ordinal = ["1st", "2nd", "3rd"] + [f"{k + 1}th" for k in range(3, 9)]

    # one limit for every panel, so positions are directly comparable. A
    # position holding only the direct beam has q_max of zero, which would
    # collapse its axes, so the limit is taken over all of them together.
    if q_max_plot is not None:
        q_lim = float(q_max_plot)
    elif over_image:
        q_lim = dataset.shape[-1] / 2 * pixel_size
    else:
        if not 0.0 < q_max_quantile <= 1.0:
            raise ValueError(f"q_max_quantile must be in (0, 1], got {q_max_quantile}")
        q_all = [
            np.hypot(
                peaks[rx, ry].numpy().astype(np.float64)[:, ix[0]],
                peaks[rx, ry].numpy().astype(np.float64)[:, ix[1]],
            )
            for rx, ry in positions
        ]
        # the direct beam is at zero and every position has one, so it is
        # dropped before taking the quantile over the diffracted peaks
        q_flat = np.concatenate([q for q in q_all if q.size]) if q_all else np.empty(0)
        q_flat = q_flat[q_flat > 0.05]
        q_max = float(np.quantile(q_flat, q_max_quantile)) if q_flat.size else 0.0
        q_lim = 1.1 * q_max if q_max > 0 else 1.0

    if measured_scale is None:
        # size the measured disks to the spacing of the peaks on screen, so a
        # dense pattern does not turn into overlapping blobs
        spacings = []
        for rx, ry in positions:
            xy = peaks[rx, ry].numpy().astype(np.float64)[:, [ix[0], ix[1]]]
            xy = xy[(np.abs(xy) <= q_lim).all(axis=1)]
            if xy.shape[0] > 2:
                d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
                np.fill_diagonal(d, np.inf)
                spacings.append(np.median(d.min(axis=1)))
        if spacings:
            spacing_pt = float(np.median(spacings)) / (2 * q_lim) * min(axsize) * 72
            measured_scale = min(1.5 * marker_scale, np.pi / 4 * (0.6 * spacing_pt) ** 2)
        else:
            measured_scale = 1.5 * marker_scale

    for pi, (rx, ry) in enumerate(positions):
        data = peaks[rx, ry].numpy().astype(np.float64)
        rc = data[:, [ix[0], ix[1]]] @ rot_back.T
        data[:, ix[0]] = rc[:, 0]
        data[:, ix[1]] = rc[:, 1]
        # the direct beam outshines every disk, so scaling the areas by the
        # brightest peak shrinks the real spots to nothing; normalize on the
        # diffracted peaks instead, compress, and floor so none vanish
        w_meas = data[:, ix[2]].clip(min=0)
        q_meas = np.hypot(data[:, ix[0]], data[:, ix[1]])
        w_ref = w_meas[q_meas > 0.05]
        hi = float(np.percentile(w_ref, 95)) if w_ref.size else float(w_meas.max(initial=0.0))
        w_meas = np.clip((w_meas / max(hi, 1e-12)) ** measured_power, 0.15, 1.0)
        for ci, (i_om, m) in enumerate(panels):
            om = oms[i_om]
            ax = axs[ci, pi] if transpose_plots else axs[pi, ci]
            if over_image:
                H, W = dataset.shape[-2], dataset.shape[-1]
                if origins is not None:
                    o_r, o_c = origins[rx, ry]
                else:
                    o_r, o_c = H / 2, W / 2
                # the direct beam is orders of magnitude above the disks, so
                # autoscaling to its peak flattens everything else
                img = np.clip(np.asarray(dataset.array[rx, ry], dtype=float), 0, None)
                show_2d(
                    _blur(img, sigma_plot),
                    norm=norm if norm is not None else {"power": 0.4, "upper_quantile": 0.999},
                    cmap="gray_r",
                    figax=(fig, ax),
                    tight_layout=False,
                )
                # pixel j has center (j - origin) * pixel_size; array edges
                # sit half a pixel beyond the first/last centers
                extent = (
                    (-0.5 - o_c) * pixel_size,
                    (W - 0.5 - o_c) * pixel_size,
                    (H - 0.5 - o_r) * pixel_size,
                    (-0.5 - o_r) * pixel_size,
                )
                ax.images[-1].set_extent(extent)
                # the pattern is off-centre by the origin: show exactly the
                # recorded area, so nothing is drawn beyond its edges
                x_lim, y_lim = extent[:2], extent[2:]
            else:
                x_lim, y_lim = (-q_lim, q_lim), (q_lim, -q_lim)
                if show_measured:
                    ax.scatter(
                        data[:, ix[1]],
                        data[:, ix[0]],
                        s=measured_scale * w_meas,
                        color="0.75",
                        lw=0,
                    )
            # a position with too few peaks was never matched, and its stored
            # orientation is still the identity; drawing that [001] pattern
            # would look like a fit where none was attempted
            matched = float(om.corr[rx, ry, m]) > 0
            if om.computed is not None:
                matched = matched and bool(om.computed[rx, ry])
            inten = np.zeros(0)
            if matched:
                sim = om.generate_pattern(rx, ry, match=m)
                inten = sim["intensity"].numpy()
                sim_rc = np.stack([sim["qx"].numpy(), sim["qy"].numpy()], axis=1) @ rot_back.T
            if inten.size:
                inside = (
                    (sim_rc[:, 1] >= min(x_lim))
                    & (sim_rc[:, 1] <= max(x_lim))
                    & (sim_rc[:, 0] >= min(y_lim))
                    & (sim_rc[:, 0] <= max(y_lim))
                )
                size = marker_scale * inten / inten.max()
                sim_rc, size = sim_rc[inside], size[inside]
                color = colors[i_om % len(colors)]
                if marker == "o":
                    # open circles leave the measured disk visible inside
                    ax.scatter(
                        sim_rc[:, 1],
                        sim_rc[:, 0],
                        s=size,
                        marker="o",
                        facecolors="none",
                        edgecolors=color,
                        lw=1.4,
                    )
                else:
                    ax.scatter(
                        sim_rc[:, 1],
                        sim_rc[:, 0],
                        s=size,
                        marker=marker,
                        color=color,
                        lw=1.8,
                    )
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.set_title(
                "%s %s\n(%d, %d)   %s"
                % (
                    om.crystal.name,
                    ordinal[m],
                    rx,
                    ry,
                    ("corr = %.2f" % float(om.corr[rx, ry, m])) if matched else "no match",
                ),
                fontsize=9,
            )
            last_row = (ci == n_r - 1) if transpose_plots else (pi == n_r - 1)
            first_col = (pi == 0) if transpose_plots else (ci == 0)
            if scalebar and last_row and first_col:
                add_scalebar_to_ax(
                    ax,
                    array_size=abs(x_lim[1] - x_lim[0]),
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
    """Map of orientation clusters (variants), one color per cluster.

    Also available as :meth:`OrientationMap.plot_cluster_map`.

    Parameters
    ----------
    om : OrientationMap
        The clustered map; gives the crystal name for the title.
    clusters : dict
        Output of :meth:`OrientationMap.cluster_orientations`.
    colors : sequence | None
        One color per cluster, cycled; defaults to `CLUSTER_COLORS`.
    scalebar : dict | None
        Real-space scale bar, e.g. {"sampling": 30, "units": "A"}.
    figax : (fig, ax) | None
        Existing axes to draw into.

    Returns
    -------
    tuple
        ``(fig, ax)``. Unassigned positions are black.
    """
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
        plt.Line2D(
            [0],
            [0],
            marker="s",
            ls="",
            color=colors[k % len(colors)],
            label=f"{k + 1}  ({int(clusters['sizes'][k])} px)",
        )
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


def _pole_family(crystal: Crystal, pole) -> torch.Tensor:
    """Unit Cartesian vectors (F, 3) of every symmetry equivalent of a pole.

    Parameters
    ----------
    crystal : Crystal
        Gives the lattice and the symmetry rotations.
    pole : array-like
        Crystal direction in Miller indices, [uvw] or [uvtw].

    Returns
    -------
    torch.Tensor
        The distinct symmetry images of the pole and their inverses.
    """
    p = crystal.direction_vector(pole).to(torch.float64)
    Rs = quat_to_matrix(crystal.sym_quats)
    fam = torch.einsum("sij,j->si", Rs, p)
    fam = torch.unique(torch.round(fam / 1e-6) * 1e-6, dim=0)
    return torch.cat([fam, -fam])


def _pole_points(
    quats: torch.Tensor, crystal: Crystal, pole, mask=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stereographic projection of every symmetry-equivalent pole of a map.

    Parameters
    ----------
    quats : torch.Tensor
        (..., 4) orientations.
    crystal : Crystal
        Gives the lattice and the symmetry rotations.
    pole : array-like
        Crystal direction in Miller indices, [uvw] or [uvtw].
    mask : np.ndarray | None
        Per-orientation weights; zero-weight orientations are dropped.

    Returns
    -------
    tuple of np.ndarray
        (x, y, weight, source) of the upper-hemisphere poles, with `source`
        the flat index of the orientation each pole came from.
    """
    q = quats.reshape(-1, 4)
    fam = _pole_family(crystal, pole)
    n_fam = fam.shape[0]
    R = quat_to_matrix(q)
    poles_lab = torch.einsum("nij,sj->nsi", R, fam)
    if mask is not None:
        w = torch.as_tensor(np.asarray(mask, dtype=float)).reshape(-1)
    else:
        w = torch.ones(q.shape[0], dtype=torch.float64)
    w_all = w[:, None].expand(-1, n_fam).reshape(-1)
    src_all = torch.arange(q.shape[0])[:, None].expand(-1, n_fam).reshape(-1)
    v = poles_lab.reshape(-1, 3)
    keep = (v[:, 2] > -1e-8) & (w_all > 0)
    v, w_keep, src = v[keep], w_all[keep], src_all[keep]
    x = (v[:, 0] / (1 + v[:, 2])).numpy()
    y = (v[:, 1] / (1 + v[:, 2])).numpy()
    return x, y, w_keep.numpy(), src.numpy()


def _pole_scatter_xy(quats: torch.Tensor, crystal: Crystal, pole) -> np.ndarray:
    """Stereographic (x, y) of all symmetry-equivalent poles for orientations.

    Parameters
    ----------
    quats : torch.Tensor
        (4,) or (N, 4) orientations.
    crystal : Crystal
        Gives the lattice and the symmetry rotations.
    pole : array-like
        Crystal direction in Miller indices, [uvw] or [uvtw].

    Returns
    -------
    np.ndarray
        (P, 2) projected upper-hemisphere poles.
    """
    fam = _pole_family(crystal, pole)
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

    Also available as :meth:`OrientationMap.plot_cluster_pole_figure`.

    Parameters
    ----------
    om : OrientationMap
        Provides the crystal symmetry of the clustered phase.
    clusters : dict
        Output of :meth:`OrientationMap.cluster_orientations`.
    pole : array-like
        Crystal direction of the plotted family in Miller indices, [uvw] or
        [uvtw].
    pole_label : str, default=""
        Legend label prefix of the pole family, e.g. "[0001]".
    overlay : dict | None
        Second pole family drawn as open markers, e.g.
        {"quats": q_beta_mean, "crystal": ti_beta, "pole": (1, 1, 0),
        "label": "<110> beta"}, the standard Burgers relationship check.
        Its "pole" is in Miller indices of its own crystal.
    colors : sequence | None
        One color per cluster, cycled; defaults to `CLUSTER_COLORS`.
    figax : (fig, ax) | None
        Existing axes to draw into.

    Returns
    -------
    tuple
        ``(fig, ax)``.
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
            xy[:, 0],
            xy[:, 1],
            s=60,
            marker="h",
            color=colors[k % len(colors)],
            edgecolors="k",
            lw=0.4,
            label=f"{pole_label} {k + 1}",
        )
    if overlay is not None:
        xy = _pole_scatter_xy(overlay["quats"], overlay["crystal"], overlay["pole"])
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=70,
            marker="D",
            facecolors="none",
            edgecolors="k",
            lw=1.0,
            label=overlay.get("label", "overlay"),
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
    saturation_power: float | None = None,
    chroma: float | None = None,
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
        Crystal direction of the pole family in Miller indices, [uvw] or
        [uvtw]; converted to Cartesian with the crystal's lattice.
    match : int, default=0
        Which match index to plot.
    mask : np.ndarray | None
        Per-position weights (e.g. phase mask).
    bins : int, default=181
        Histogram bins across the stereographic disk.
    color_by : {"density", "ipf"}, default="density"
        "density": white through yellow and red to black with increasing
        density. "ipf": each contribution is colored by the IPF (zone axis)
        color of its probe position, blended from a white background as the
        density rises, with the color wedge in a panel beside it.
    int_range : tuple, default=(0.0, 1.0)
        Density display range as fractions of the 98th percentile of the
        occupied bins: values below the lower limit show as background,
        above the upper limit at full strength.
    smooth_sigma : float, default=1.5
        Gaussian blur of the histogram, in bins; 0 disables it.
    label : str | None
        Annotation for the pole family, e.g. "(0001)" or "{110}".
    grid : bool, default=True
        Draw polar-angle circles and azimuth spokes every 30 degrees.
    overlay : dict | None
        A second pole family drawn on top. With an "om" key, the density
        of that map's poles is drawn as contours:
        {"om": om_beta, "pole": (1, 1, 0), "match": 0, "mask": mask_beta,
        "label": "<110> beta"}. Otherwise fixed orientations are drawn as
        open markers: {"quats": q, "crystal": xtl, "pole": (1, 1, 0),
        "label": ...}. Poles are Miller indices of the overlay's crystal.
    saturation_power, chroma : float | None
        Color wedge shape, used when `color_by` is "ipf"; see
        :func:`plot_orientation_map`.
    figax : (fig, ax) | (fig, (ax, ax_legend)) | None
        Existing axes; the legend panel is used only with `color_by="ipf"`.

    Returns
    -------
    tuple
        ``(fig, ax)`` with `ax` the pole figure panel.
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
        rgb_pos = ipf_color(
            om.quats[..., match, :], om.crystal, "z", saturation_power, chroma
        ).reshape(-1, 3)
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
    yy, xx = np.meshgrid(0.5 * (ye[:-1] + ye[1:]), 0.5 * (xe[:-1] + xe[1:]), indexing="ij")
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
                "",
                xy=(gx + dx, gy + dy),
                xytext=(gx, gy),
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.2),
                annotation_clip=False,
            )
            ax.text(
                gx + dx * 1.25,
                gy + dy * 1.25,
                lbl,
                fontsize=9,
                ha=ha,
                va=va,
                color="0.3",
            )
        ax.text(
            gx - 0.03,
            gy - 0.06,
            "scan axes",
            fontsize=7,
            ha="left",
            va="bottom",
            color="0.45",
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
                    yc,
                    xc,
                    Ho,
                    levels=lev,
                    colors="k",
                    linewidths=[0.7, 1.3],
                    alpha=0.85,
                )
                ax.plot([], [], color="k", lw=1.2, label=overlay.get("label", "overlay"))
        else:
            oxy = _pole_scatter_xy(overlay["quats"], overlay["crystal"], overlay["pole"])
            ax.scatter(
                oxy[:, 1],
                oxy[:, 0],
                s=80,
                marker="D",
                facecolors="none",
                edgecolors="k",
                lw=1.2,
                label=overlay.get("label", "overlay"),
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
            wedge_legend(
                om.crystal,
                ax_leg,
                orientation="vertical",
                saturation_power=saturation_power,
                chroma=chroma,
            )
        else:
            ax_leg.axis("off")
    return fig, ax
