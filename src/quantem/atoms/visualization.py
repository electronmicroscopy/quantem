"""Static matplotlib plots for :class:`~quantem.atoms.AtomicModel`.

Nothing here modifies model state.  Functions are registered by name so they
can be called as ``model.plot(kind=...)``:

``"pdf"``
    Radial distribution function with the first-peak fit and template shells.
``"histogram"``
    Histogram of a per-site channel.
``"slab"``
    Projection of the sites inside a slab, colored by a channel.
``"slices"``
    Grid of slab projections stepping through the model.
``"template"``
    Neighbor vectors of one site with the fitted template overlaid.
"""

from __future__ import annotations

from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray

PLOT_REGISTRY: dict[str, Callable[..., Any]] = {}

_CATEGORICAL_COLORS = np.array(
    [
        [0.20, 0.45, 0.95],
        [0.95, 0.60, 0.10],
        [0.20, 0.70, 0.30],
        [0.85, 0.20, 0.25],
        [0.60, 0.35, 0.80],
        [0.55, 0.35, 0.20],
        [0.90, 0.45, 0.75],
        [0.50, 0.50, 0.50],
        [0.70, 0.75, 0.10],
        [0.10, 0.75, 0.80],
    ]
)


def _register(name: str):
    def decorator(fn):
        PLOT_REGISTRY[name] = fn
        return fn

    return decorator


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
_AXES = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}


def view_matrix(normal: str | NDArray, up: str | NDArray | None = None) -> NDArray:
    """Orthonormal ``(3, 3)`` matrix whose rows are (right, up, normal).

    Projecting ``(xyz - center) @ view_matrix.T`` gives image columns
    ``(u, v, depth)`` with ``depth`` measured along ``normal`` (toward the
    viewer).

    Parameters
    ----------
    normal : str or array
        Viewing direction as ``"x"``, ``"y"``, ``"z"`` or a 3-vector.
    up : str or array, optional
        Approximate screen-up direction; default picks the axis least aligned
        with ``normal``.
    """
    n = _AXES[normal] if isinstance(normal, str) else np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    if up is None:
        up_v = _AXES["y"] if abs(n[1]) < 0.9 else _AXES["x"]
        if isinstance(normal, str) and normal == "y":
            up_v = _AXES["x"]
    else:
        up_v = _AXES[up] if isinstance(up, str) else np.asarray(up, dtype=float)
    u = up_v - n * (up_v @ n)
    u = u / np.linalg.norm(u)
    r = np.cross(u, n)
    return np.stack([r, u, n], axis=0)


def channel_colors(
    model,
    channel: str,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[NDArray, Any, tuple[float, float], list[str] | None]:
    """Map a channel to RGB colors.

    Returns
    -------
    rgb, colormap, (vmin, vmax), categories
        ``(N, 3)`` colors; categories is a list of labels for categorical
        channels (``-1`` codes are drawn grey), otherwise ``None``.
    """
    values = model.get_channel(channel)
    categories = model.categories.get(channel)
    if categories is not None:
        codes = values.astype(int)
        n_cat = max(len(categories), int(codes.max()) + 1 if codes.size else 1)
        colors = _CATEGORICAL_COLORS[np.arange(n_cat) % len(_CATEGORICAL_COLORS)]
        if channel == "grain" or n_cat > len(_CATEGORICAL_COLORS):
            rng = np.random.default_rng(0)
            colors = rng.uniform(0.15, 0.95, (n_cat, 3))
        rgb = np.full((values.size, 3), 0.6)
        ok = codes >= 0
        rgb[ok] = colors[codes[ok] % n_cat]
        return rgb, ListedColormap(colors), (-0.5, n_cat - 0.5), list(categories)
    finite = np.isfinite(values)
    if vmin is None:
        vmin = float(np.percentile(values[finite], 1)) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.percentile(values[finite], 99)) if finite.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-9
    colormap = plt.get_cmap(cmap or ("turbo" if channel.startswith("score") else "viridis"))
    t = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    rgb = colormap(np.nan_to_num(t))[:, :3]
    rgb[~finite] = 0.6
    return rgb, colormap, (vmin, vmax), None


def draw_spheres(ax, x, y, rgb, marker_size, edgecolor="k", num_layers: int = 6):
    """Draw shaded sphere markers with a stack of offset, brightening discs.

    Parameters
    ----------
    ax : Axes
        Target axes.
    x, y : ndarray
        Marker centers (already sorted back-to-front).
    rgb : ndarray
        ``(N, 3)`` base colors.
    marker_size : float
        Scatter marker area (points^2) of the full sphere.
    edgecolor : str or None
        Outline color of the full disc.
    num_layers : int
        Number of highlight discs stacked toward the upper-left.
    """
    radius_pt = np.sqrt(marker_size) / 2.0
    # convert an offset in points to data units using the axes transform
    fig = ax.figure
    axes_width_pt = ax.get_position().width * fig.get_size_inches()[0] * 72.0
    xlim = ax.get_xlim() if ax.has_data() else (np.min(x), np.max(x))
    data_per_pt = (abs(xlim[1] - xlim[0]) or 1.0) / axes_width_pt
    rgb = np.asarray(rgb)
    ax.scatter(
        x, y, s=marker_size, c=rgb * 0.55, edgecolors=edgecolor, linewidths=0.4 if edgecolor else 0
    )
    for i in range(1, num_layers + 1):
        t = i / num_layers
        scale = 1.0 - 0.75 * t
        shift = 0.28 * t * radius_pt * data_per_pt
        color = np.clip(rgb * (0.55 + 0.6 * t) + 0.35 * t**2, 0, 1)
        ax.scatter(
            x - shift,
            y - shift,
            s=marker_size * scale**2,
            c=color,
            edgecolors="none",
        )


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
@_register("pdf")
def plot_pdf(
    model, show_fit: bool = True, show_templates: bool = True, ax=None, returnfig: bool = False
):
    """Plot the radial distribution function (native units).

    Parameters
    ----------
    show_fit : bool
        Overlay the first-peak fit and first-shell cutoffs.
    show_templates : bool
        Mark the shell radii of matched templates (scaled by the NN distance).
    """
    if model.pdf is None:
        model.compute_pdf()
    pdf, fit = model.pdf, model.nn_fit
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    ax.plot(pdf["r"], pdf["g_smooth"], color="k", lw=1.5, label="RDF")
    if show_fit and fit is not None:
        ax.plot(pdf["r"], fit["fit"], color="r", lw=1.5, label=f"fit  r_nn = {fit['r_nn']:.3f}")
        for c in fit["cutoff"]:
            ax.axvline(c, color="r", ls="--", lw=1)
    if show_templates and model.templates:
        r_nn = model.nn_distance
        for i, (name, template) in enumerate(model.templates.items()):
            col = _CATEGORICAL_COLORS[i % len(_CATEGORICAL_COLORS)]
            for j, s in enumerate(template.shells):
                ax.axvline(s * r_nn, color=col, lw=1, alpha=0.7, label=name if j == 0 else None)
    ax.set_xlabel(f"radius [{model.sites.units[0]}]")
    ax.set_ylabel("g(r)")
    ax.set_xlim(0, pdf["r"].max())
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")
    ax.set_title(model.name)
    return (fig, ax) if returnfig else None


@_register("histogram")
def plot_histogram(
    model, channel: str = "score_max", bins: int = 100, ax=None, returnfig: bool = False, **kwargs
):
    """Histogram of a per-site channel."""
    values = model.get_channel(channel)
    values = values[np.isfinite(values)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.figure
    categories = model.categories.get(channel)
    if categories is not None:
        codes = values.astype(int)
        counts = np.bincount(codes[codes >= 0], minlength=len(categories))
        ax.bar(np.arange(len(categories)), counts, color=_CATEGORICAL_COLORS[: len(categories)])
        ax.set_xticks(np.arange(len(categories)), categories)
        n_none = int((codes < 0).sum())
        ax.set_title(f"{channel}  ({n_none} unclassified)")
    else:
        ax.hist(values, bins=bins, color="0.3", **kwargs)
        ax.set_xlabel(channel)
    ax.set_ylabel("count")
    return (fig, ax) if returnfig else None


def _project(model, normal, up, offset, thickness, hide=None, channel=None):
    v = view_matrix(normal, up)
    xyz = model.positions - model.center[None, :]
    uvd = xyz @ v.T
    keep = np.ones(xyz.shape[0], dtype=bool)
    if thickness is not None:
        keep &= np.abs(uvd[:, 2] - offset) <= thickness / 2.0
    if hide is not None and channel is not None:
        vals = model.get_channel(channel)
        keep &= ~((vals >= hide[0]) & (vals <= hide[1]))
    return uvd, keep, v


@_register("slab")
def plot_slab(
    model,
    channel: str = "structure",
    normal: str | NDArray = "z",
    up: str | NDArray | None = None,
    offset: float = 0.0,
    thickness: float | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    marker_size: float | None = None,
    depth_cue: float = 0.35,
    hide: tuple[float, float] | None = None,
    edgecolor: str | None = "k",
    style: str = "spheres",
    ax=None,
    figsize: tuple[float, float] = (7, 7),
    colorbar: bool = True,
    title: str | None = None,
    returnfig: bool = False,
):
    """Project the sites inside a slab onto the viewing plane.

    Parameters
    ----------
    channel : str
        Channel used for coloring.
    normal : str or array
        Viewing direction (slab normal): ``"x"``, ``"y"``, ``"z"`` or a vector.
    up : str or array, optional
        Screen-up direction.
    offset : float
        Slab center along ``normal`` relative to the model center (calibrated units).
    thickness : float, optional
        Slab thickness (calibrated units); ``None`` shows all sites.
    cmap, vmin, vmax
        Color mapping for continuous channels.
    marker_size : float, optional
        Scatter marker area; default scales with the bond length.
    depth_cue : float
        Darkening of sites far from the viewer (0 = none).
    hide : (lo, hi), optional
        Hide sites whose channel value lies inside this range.
    edgecolor : str or None
        Marker edge color.
    style : {"spheres", "flat"}
        ``"spheres"`` draws shaded spheres (a stack of offset, brightening
        discs per site); ``"flat"`` draws plain scatter markers.
    """
    uvd, keep, _ = _project(model, normal, up, offset, thickness, hide, channel)
    rgb, colormap, (lo, hi), categories = channel_colors(model, channel, cmap, vmin, vmax)
    uvd, rgb = uvd[keep], rgb[keep]
    order = np.argsort(uvd[:, 2])
    uvd, rgb = uvd[order], rgb[order]
    if depth_cue > 0 and uvd.shape[0] > 1:
        d = uvd[:, 2]
        t = (d - d.min()) / max(d.max() - d.min(), 1e-9)
        rgb = rgb * (1 - depth_cue * (1 - t))[:, None]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if marker_size is None:
        # marker diameter ~ 0.9 bond lengths, in points, from the axes width
        all_uv = (model.positions - model.center[None, :]) @ view_matrix(normal, up).T
        extent = max(np.ptp(all_uv[:, 0]), np.ptp(all_uv[:, 1]), 1e-9) * 1.05
        axes_width_pt = ax.get_position().width * fig.get_size_inches()[0] * 72.0
        diameter_pt = 0.9 * model.bond_length * axes_width_pt / extent
        marker_size = max(diameter_pt**2, 1.0)
    if style == "spheres":
        draw_spheres(ax, uvd[:, 1], uvd[:, 0], rgb, marker_size, edgecolor=edgecolor)
    else:
        ax.scatter(
            uvd[:, 1],
            uvd[:, 0],
            s=marker_size,
            c=rgb,
            edgecolors=edgecolor,
            linewidths=0.3 if edgecolor else 0,
        )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel(f"v [{model.units}]")
    ax.set_ylabel(f"u [{model.units}]")
    if title is None:
        n_str = normal if isinstance(normal, str) else np.round(normal, 2)
        title = f"{channel}  |  normal {n_str}"
        if thickness is not None:
            title += f"  |  offset {offset:.1f}, thickness {thickness:.1f}"
    ax.set_title(title)
    if colorbar:
        if categories is not None:
            from matplotlib.lines import Line2D

            handles = [
                Line2D([], [], marker="o", ls="", color=colormap(i), markeredgecolor="k", label=c)
                for i, c in enumerate(categories)
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=8)
        else:
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(lo, hi))
            fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, label=channel)
    return (fig, ax) if returnfig else None


@_register("slices")
def plot_slices(
    model,
    channel: str = "structure",
    normal: str | NDArray = "z",
    num_slices: int = 6,
    thickness: float | None = None,
    start: float | None = None,
    end: float | None = None,
    positions: NDArray | None = None,
    ncols: int = 3,
    figsize_per: float = 4.0,
    returnfig: bool = False,
    **kwargs,
):
    """Grid of slab projections stepping along ``normal``.

    Parameters
    ----------
    num_slices : int
        Number of slabs (ignored when ``positions`` is given).
    thickness : float, optional
        Slab thickness; default is the step between slabs.
    start, end : float, optional
        Range of slab centers along ``normal`` (relative to the model center);
        default spans the model.
    positions : array, optional
        Explicit slab centers along ``normal``, e.g. the atomic layers from
        ``model.layer_positions(normal)``.
    **kwargs
        Forwarded to :func:`plot_slab`.
    """
    v = view_matrix(normal, kwargs.get("up"))
    depth = (model.positions - model.center[None, :]) @ v[2]
    if positions is not None:
        centers = np.asarray(positions, dtype=float)
        num_slices = centers.size
    else:
        if start is None:
            start = float(depth.min()) + 0.05 * np.ptp(depth)
        if end is None:
            end = float(depth.max()) - 0.05 * np.ptp(depth)
        centers = np.linspace(start, end, num_slices)
    if thickness is None:
        if positions is not None and num_slices > 1:
            thickness = 0.8 * float(np.median(np.diff(centers)))
        else:
            thickness = float(centers[1] - centers[0]) if num_slices > 1 else np.ptp(depth)
    nrows = int(np.ceil(num_slices / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize_per * ncols, figsize_per * nrows), squeeze=False
    )
    kwargs.setdefault("colorbar", False)
    kwargs.setdefault("depth_cue", 0.0)
    for i, ax in enumerate(axes.ravel()):
        if i >= num_slices:
            ax.axis("off")
            continue
        plot_slab(
            model,
            channel=channel,
            normal=normal,
            offset=float(centers[i]),
            thickness=thickness,
            ax=ax,
            title=f"offset {centers[i]:.1f} {model.units}",
            **kwargs,
        )
    fig.suptitle(f"{channel} slices along {normal}")
    fig.tight_layout()
    return (fig, axes) if returnfig else None


@_register("template")
def plot_template(
    model, index: int, template: str | None = None, ax=None, returnfig: bool = False
):
    """3D plot of one site's neighbor vectors with the fitted template overlaid.

    Parameters
    ----------
    index : int
        Site index.
    template : str, optional
        Template name; default is the site's classified structure (or the first
        matched template).
    """
    if not model.matches:
        raise RuntimeError("Run match_templates() first.")
    if template is None:
        structure = int(model.get_channel("structure")[index])
        template = model.structure_names[structure] if structure >= 0 else list(model.matches)[0]
    match = model.matches[template]
    tmpl = model.templates[template]
    dxyz, dist = model.neighbor_vectors(normalize=True)
    p = dxyz[index]
    p = p[np.isfinite(p).all(1) & (dist[index] <= 1.15 * tmpl.max_radius * 1.3)]
    t = tmpl.vectors @ match["rotation"][index].T
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=80, c="0.3", edgecolors="k", label="neighbors")
    ax.scatter(t[:, 0], t[:, 1], t[:, 2], s=160, marker="+", c="r", linewidths=2, label=template)
    ax.scatter([0], [0], [0], s=120, c="b", marker="x")
    b = max(tmpl.max_radius, 1.0) * 1.2
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    ax.set_zlim(-b, b)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(
        f"site {index}: {template} score {match['score'][index]:.2f}, matched {match['num_matched'][index]}"
    )
    ax.legend()
    return (fig, ax) if returnfig else None
