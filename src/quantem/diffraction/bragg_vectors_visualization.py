from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_template(
    dp_mean: np.ndarray,
    template: np.ndarray,
    corr_map: np.ndarray,
    position: tuple[int, int],
    *,
    crop: tuple[float, float, float] | None = None,
    figsize: tuple[float, float] = (13, 4),
):
    """Mean diffraction pattern, the (centered) template, and one correlation map.

    Parameters
    ----------
    dp_mean : np.ndarray
        Mean diffraction pattern.
    template : np.ndarray
        The correlation template, centered for display.
    corr_map : np.ndarray
        Correlation map computed at ``position``.
    position : tuple of int
        ``(row, col)`` scan position the correlation map was computed at.
    crop : tuple of float, optional
        ``(center_row, center_col, half_width)`` zoom window (in pixels) applied to
        all three panels. The view spans ``half_width`` either side of the center,
        clamped to each image's bounds, so an over-large ``half_width`` just shows
        the full image. ``None`` (default) shows the full panels.
    figsize : tuple of float, default=(13, 4)
        Figure size in inches.

    Returns
    -------
    tuple
        ``(fig, ax)`` with ``ax`` a length-3 array of axes.
    """
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].imshow(dp_mean, cmap="gray")
    ax[0].set_title("mean diffraction")
    ax[1].imshow(template, cmap="gray")
    ax[1].set_title("template (centered)")
    ax[2].imshow(corr_map, cmap="viridis")
    ax[2].set_title(f"correlation @ {tuple(position)}")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    if crop is not None:
        cr, cc, hw = float(crop[0]), float(crop[1]), float(crop[2])
        for a, img in zip(ax, (dp_mean, template, corr_map)):
            h, w = img.shape[:2]
            a.set_xlim(max(cc - hw, -0.5), min(cc + hw, w - 0.5))
            a.set_ylim(min(cr + hw, h - 0.5), max(cr - hw, -0.5))
    fig.tight_layout()
    return fig, ax


def _mark_positions(ax, positions, *, radius=None, linewidth=0.5):
    """Overlay numbered red markers at each ``(row, col)`` scan position.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw the markers on.
    positions : sequence of tuple of int
        ``(row, col)`` scan positions to mark, numbered in order.
    radius : float, optional
        Marker radius in image pixels. If given, each marker is a ring of that
        radius drawn as a circle patch in data coordinates; if ``None``, a fixed
        screen-size scatter marker is used instead.
    linewidth : float, default=0.5
        Ring stroke width.
    """
    from matplotlib.patches import Circle

    for i, (r, c) in enumerate(positions):
        if radius is None:
            ax.scatter(c, r, s=70, facecolors="none", edgecolors="red", linewidths=linewidth)
        else:
            ax.add_patch(
                Circle((c, r), radius=radius, fill=False, edgecolor="red", linewidth=linewidth)
            )
        ax.annotate(
            str(i),
            (c, r),
            color="red",
            fontsize=11,
            fontweight="bold",
            xytext=(4, 4),
            textcoords="offset points",
        )


def _blur(dp: np.ndarray, sigma: float | None) -> np.ndarray:
    """Gaussian-blur a diffraction pattern for display only (passthrough if ``sigma`` falsy).

    Smoothing the *displayed* pattern (the detection still runs on the raw data)
    makes it easier to judge by eye which correlation peaks sit on real disks and
    which are noise.

    Parameters
    ----------
    dp : np.ndarray
        Diffraction pattern to blur.
    sigma : float or None
        Gaussian blur width in pixels. If falsy (``None``, ``0``, or negative),
        ``dp`` is returned unchanged.

    Returns
    -------
    np.ndarray
        The blurred pattern, or ``dp`` unchanged when ``sigma`` is falsy.
    """
    if not sigma or sigma <= 0:
        return dp
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(np.asarray(dp, dtype=float), float(sigma))


def _grid_axes(n, ncols, *, show_image, axsize=None, figsize=None):
    """Figure with an optional left nav-image axis and a right ``nrows x ncols`` tile grid.

    With ``show_image`` a navigation-image axis is placed to the left of the tile
    grid; otherwise the figure is just the grid.

    Parameters
    ----------
    n : int
        Number of tiles to lay out.
    ncols : int
        Number of tile columns, clamped to ``n``.
    show_image : bool
        If ``True``, add a navigation-image axis to the left of the tile grid.
    axsize : tuple of float, optional
        Per-tile size in inches, ``(w, h)``. Sizes the figure when ``figsize`` is
        not given, so a larger ``axsize`` zooms every tile.
    figsize : tuple of float, optional
        Explicit figure size in inches; overrides the ``axsize``-derived size.

    Returns
    -------
    tuple
        ``(fig, ax_image, dp_axes, ncols)`` where ``ax_image`` is ``None`` when
        ``show_image`` is ``False``, ``dp_axes`` is an ``(nrows, ncols)`` object
        array of axes (trailing unused tiles already turned off), and ``ncols`` is
        clamped to ``n``.
    """
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    tile_w, tile_h = (float(axsize[0]), float(axsize[1])) if axsize is not None else (3.0, 3.2)
    nav_w = (tile_w if axsize is not None else 4.0) if show_image else 0.0
    if figsize is None:
        figsize = (nav_w + tile_w * ncols, max(tile_h, tile_h * nrows))

    fig = plt.figure(figsize=figsize)
    if show_image:
        outer = fig.add_gridspec(1, 2, width_ratios=[nav_w, tile_w * ncols], wspace=0.12)
        ax_image = fig.add_subplot(outer[0, 0])
        grid = outer[0, 1].subgridspec(nrows, ncols, wspace=0.08, hspace=0.2)
    else:
        ax_image = None
        grid = fig.add_gridspec(nrows, ncols, wspace=0.08, hspace=0.2)

    dp_axes = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows * ncols):
        a = fig.add_subplot(grid[i // ncols, i % ncols])
        dp_axes[i // ncols, i % ncols] = a
        if i >= n:
            a.axis("off")
    return fig, ax_image, dp_axes, ncols


def plot_diffraction_grid(
    image: np.ndarray | None,
    dps: list[np.ndarray],
    positions: list[tuple[int, int]],
    *,
    ncols: int = 4,
    image_title: str = "navigation image",
    image_kwargs: dict | None = None,
    marker_radius: float | None = None,
    linewidth: float = 0.5,
    sigma_plot: float | None = None,
    axsize: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    **show_kwargs,
):
    """A tiled grid of the diffraction patterns at ``positions``, optionally beside a nav image.

    When ``image`` is given (e.g. a virtual dark-field image) it is drawn on the
    left with each scan position marked as a numbered red marker at ``(x=col,
    y=row)``; pass ``image=None`` to omit it and show only the grid. Right: the
    diffraction patterns tiled ``ncols`` wide. Both are rendered with
    :func:`~quantem.core.visualization.show_2d`, and the navigation image and the
    diffraction tiles take separate, independent styling (``image_kwargs`` vs
    ``show_kwargs``).

    Parameters
    ----------
    image : np.ndarray or None
        Navigation image (e.g. a virtual dark-field image) drawn at left, with the
        scan positions marked. Pass ``None`` to show only the diffraction grid.
    dps : list of np.ndarray
        Diffraction patterns to tile, one per entry in ``positions``.
    positions : list of tuple of int
        ``(row, col)`` scan positions, used for the tile titles and the nav-image
        markers.
    ncols : int, default=4
        Number of columns in the diffraction-pattern tile grid.
    image_title : str, default="navigation image"
        Title for the navigation image.
    image_kwargs : dict, optional
        Extra keyword arguments (e.g. ``norm``, ``cmap``, ``scalebar``) for the
        navigation image's :func:`show_2d` call, styling it independently of the
        tiles.
    marker_radius : float, optional
        Scan-position marker radius in image pixels; ``None`` uses a fixed
        screen-size marker.
    linewidth : float, default=0.5
        Stroke width of the scan-position markers.
    sigma_plot : float, optional
        Gaussian blur width (pixels) applied to the *displayed* patterns only (the
        data is untouched) to ease judging real features.
    axsize : tuple of float, optional
        Per-tile size in inches; a larger value zooms the tiles.
    figsize : tuple of float, optional
        Explicit figure size in inches.
    **show_kwargs
        Forwarded to :func:`show_2d` for the diffraction-pattern tiles (e.g.
        ``norm``, ``cmap``, ``cbar``).

    Returns
    -------
    tuple
        ``(fig, (ax_image, dp_axes))`` with ``ax_image`` ``None`` when no image is
        shown.
    """
    from quantem.core.visualization import show_2d

    n = len(positions)
    show_image = image is not None
    fig, ax_image, dp_axes, ncols = _grid_axes(
        n, ncols, show_image=show_image, axsize=axsize, figsize=figsize
    )

    if show_image:
        image_show_kwargs = {"cmap": "gray", "title": image_title, **(image_kwargs or {})}
        show_2d(np.asarray(image), figax=(fig, ax_image), **image_show_kwargs)
        _mark_positions(ax_image, positions, radius=marker_radius, linewidth=linewidth)

    user_title = show_kwargs.pop("title", None)
    for i in range(n):
        r, c = positions[i]
        title = user_title if user_title is not None else f"{i}: ({r},{c})"
        show_2d(
            _blur(dps[i], sigma_plot),
            figax=(fig, dp_axes[i // ncols, i % ncols]),
            title=title,
            **show_kwargs,
        )
    return fig, (ax_image, dp_axes)


def plot_detection(
    image: np.ndarray | None,
    dps: list[np.ndarray],
    peaks: list[np.ndarray],
    positions: list[tuple[int, int]],
    *,
    ncols: int = 4,
    peak_radius: float = 6.0,
    marker_radius: float | None = None,
    linewidth: float = 0.5,
    sigma_plot: float | None = None,
    image_title: str = "virtual image",
    image_kwargs: dict | None = None,
    axsize: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    **show_kwargs,
):
    """The diffraction patterns with detected peaks overlaid, optionally beside a nav image.

    When ``image`` is given (e.g. a virtual dark-field image) it is drawn on the
    left with each chosen scan position drawn as a numbered red marker at ``(x=col,
    y=row)``; pass ``image=None`` to omit it and show only the tiles. Right: the
    diffraction patterns tiled ``ncols`` wide, each rendered with
    :func:`~quantem.core.visualization.show_2d`, with the detected peaks overlaid as
    cyan rings that trace the disks rather than obscuring them.

    Parameters
    ----------
    image : np.ndarray or None
        Navigation image (e.g. a virtual dark-field image) drawn at left, with the
        scan positions marked. Pass ``None`` to show only the diffraction tiles.
    dps : list of np.ndarray
        Diffraction patterns to tile, one per entry in ``positions``.
    peaks : list of np.ndarray
        Detected peaks per pattern; ``peaks[i]`` is an ``(M, 3)`` array of
        ``[q_row, q_col, intensity]``. Each peak is drawn as a cyan ring at
        ``(x=q_col, y=q_row)``.
    positions : list of tuple of int
        ``(row, col)`` scan positions, used for the tile titles and the nav-image
        markers.
    ncols : int, default=4
        Number of columns in the diffraction-pattern tile grid.
    peak_radius : float, default=6.0
        Radius of the cyan peak rings, in diffraction pixels.
    marker_radius : float, optional
        Scan-position marker radius in image pixels; ``None`` uses a fixed
        screen-size marker.
    linewidth : float, default=0.5
        Stroke width of both the scan-position markers and the peak rings.
    sigma_plot : float, optional
        Gaussian blur width (pixels) applied to the *displayed* patterns only
        (detection still uses the raw data) to ease telling real disks from false
        positives.
    image_title : str, default="virtual image"
        Title for the navigation image.
    image_kwargs : dict, optional
        Extra keyword arguments for the navigation image's :func:`show_2d` call,
        styling it independently of the tiles.
    axsize : tuple of float, optional
        Per-tile size in inches; a larger value zooms the tiles.
    figsize : tuple of float, optional
        Explicit figure size in inches.
    **show_kwargs
        Forwarded to :func:`show_2d` for the diffraction-pattern tiles (e.g.
        ``norm``, ``cmap``, ``cbar``).

    Returns
    -------
    tuple
        ``(fig, (ax_image, dp_axes))`` with ``ax_image`` ``None`` when no image is
        shown.
    """
    from matplotlib.patches import Circle

    from quantem.core.visualization import show_2d

    n = len(positions)
    show_image = image is not None
    fig, ax_image, dp_axes, ncols = _grid_axes(
        n, ncols, show_image=show_image, axsize=axsize, figsize=figsize
    )

    if show_image:
        image_show_kwargs = {"cmap": "gray", "title": image_title, **(image_kwargs or {})}
        show_2d(np.asarray(image), figax=(fig, ax_image), **image_show_kwargs)
        _mark_positions(ax_image, positions, radius=marker_radius, linewidth=linewidth)

    user_title = show_kwargs.pop("title", None)
    for i in range(n):
        a = dp_axes[i // ncols, i % ncols]
        r, c = positions[i]
        pk = peaks[i]
        title = user_title if user_title is not None else f"{i}: ({r},{c})  n={pk.shape[0]}"
        show_2d(_blur(dps[i], sigma_plot), figax=(fig, a), title=title, **show_kwargs)
        for q in pk:
            a.add_patch(
                Circle(
                    (q[1], q[0]),
                    radius=peak_radius,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=linewidth,
                )
            )

    return fig, (ax_image, dp_axes)


def plot_basis_vectors(
    bvm: np.ndarray,
    cand_rc: np.ndarray,
    cand_int: np.ndarray,
    origin: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    *,
    cmap: str = "gray",
    norm: str | dict = "log_auto",
    zoom: bool = True,
    figsize: tuple[float, float] = (6, 6),
    **show_kwargs,
):
    """The Bragg vector map with the candidate peaks, origin, and basis vectors overlaid.

    Every candidate peak is drawn as a numbered cyan ring; those numbers are the
    indices accepted by
    :meth:`~quantem.diffraction.bragg_vectors.BraggVectors.choose_basis_vectors`
    for overriding ``origin``/``g1``/``g2`` by peak. The chosen origin is a green
    marker and ``g1`` (red) / ``g2`` (blue) are arrows drawn from it, labelled at
    their midpoints so the labels never sit on top of the candidate numbers.

    Parameters
    ----------
    bvm : np.ndarray
        Bragg vector map; rendered through
        :func:`~quantem.core.visualization.show_2d` so its display scaling is
        controlled by ``norm`` and ``show_kwargs``.
    cand_rc : np.ndarray
        ``(N, 2)`` ``[row, col]`` candidate peak positions, brightest first; the
        ring labels are their row indices.
    cand_int : np.ndarray
        ``(N,)`` candidate intensities (unused for drawing; kept for parity with
        the candidate API).
    origin : np.ndarray
        ``(row, col)`` chosen lattice origin.
    g1 : np.ndarray
        First lattice vector as a ``(row, col)`` offset from ``origin``.
    g2 : np.ndarray
        Second lattice vector as a ``(row, col)`` offset from ``origin``.
    cmap : str, default="gray"
        Colormap for the Bragg vector map; gray keeps the colored overlays legible.
    norm : str or dict, default="log_auto"
        Intensity scaling forwarded to :func:`show_2d` (e.g. ``"linear_auto"``,
        ``"log_auto"``, ``"power_sqrt"``, or ``{"power": 0.5}``).
    zoom : bool, default=True
        If ``True``, frame the view to the candidate bounding box (plus a margin)
        so the numbered peaks are large enough to read.
    figsize : tuple of float, default=(6, 6)
        Figure size in inches.
    **show_kwargs
        Extra keyword arguments forwarded to :func:`show_2d` for fine display
        control (e.g. ``vmin``, ``vmax``, ``lower_quantile``, ``upper_quantile``).

    Returns
    -------
    tuple
        ``(fig, ax)``.
    """
    import matplotlib.patheffects as path_effects

    from quantem.core.visualization import show_2d

    stroke = [path_effects.withStroke(linewidth=2.5, foreground="black")]
    origin_color = (0.0, 0.7, 0.0)
    g1_color = (1.0, 0.0, 0.0)
    g2_color = (0.0, 0.7, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    show_2d(np.asarray(bvm), figax=(fig, ax), cmap=cmap, norm=norm, **show_kwargs)

    cand_rc = np.asarray(cand_rc, dtype=float).reshape(-1, 2)
    for i, (r, c) in enumerate(cand_rc):
        ax.scatter(c, r, s=60, facecolors="none", edgecolors="cyan", linewidths=1.0, zorder=3)
        ax.annotate(
            str(i),
            (c, r),
            color="cyan",
            fontsize=9,
            fontweight="bold",
            xytext=(4, 4),
            textcoords="offset points",
            path_effects=stroke,
            zorder=4,
        )

    o = np.asarray(origin, dtype=float).reshape(2)
    ax.scatter(
        o[1],
        o[0],
        s=160,
        marker="P",
        facecolors=[origin_color],
        edgecolors="white",
        linewidths=1.5,
        zorder=6,
    )
    for g, label, color in (
        (np.asarray(g1, float), "g1", g1_color),
        (np.asarray(g2, float), "g2", g2_color),
    ):
        tip = (o[1] + g[1], o[0] + g[0])
        ax.annotate(
            "",
            xy=tip,
            xytext=(o[1], o[0]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.4, shrinkA=0, shrinkB=0),
            zorder=5,
        )
        # Label at the arrow midpoint, nudged perpendicular to the shaft (in screen
        # space) so it clears both the arrow and the candidate numbers on the peaks.
        gnorm = float(np.hypot(g[0], g[1])) + 1e-12
        perp = (g[0] / gnorm * 15.0, g[1] / gnorm * 15.0)
        mid = (o[1] + g[1] / 2.0, o[0] + g[0] / 2.0)
        ax.annotate(
            label,
            mid,
            color=color,
            fontsize=14,
            fontweight="bold",
            xytext=perp,
            textcoords="offset points",
            ha="center",
            va="center",
            path_effects=stroke,
            zorder=7,
        )

    if zoom and cand_rc.shape[0]:
        rmin, cmin = cand_rc.min(axis=0)
        rmax, cmax = cand_rc.max(axis=0)
        margin = 0.12 * max(rmax - rmin, cmax - cmin, 1.0) + 6.0
        h, w = bvm.shape[:2]
        ax.set_xlim(max(cmin - margin, -0.5), min(cmax + margin, w - 0.5))
        ax.set_ylim(min(rmax + margin, h - 0.5), max(rmin - margin, -0.5))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("lattice basis  (origin +,  g1,  g2;  cyan = candidate index)")
    fig.tight_layout()
    return fig, ax


def plot_bvm(
    bvm: np.ndarray,
    counts: np.ndarray,
    *,
    figsize: tuple[float, float] = (10, 4),
):
    """The Bragg vector map (log-scaled) beside the per-position peak count.

    Parameters
    ----------
    bvm : np.ndarray
        Bragg vector map; displayed log-scaled (``log1p``).
    counts : np.ndarray
        Per-position peak count, shape ``(scan_row, scan_col)``.
    figsize : tuple of float, default=(10, 4)
        Figure size in inches.

    Returns
    -------
    tuple
        ``(fig, ax)`` with ``ax`` a length-2 array of axes.
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(np.log1p(bvm), cmap="inferno")
    ax[0].set_title("Bragg vector map (log)")
    im = ax[1].imshow(counts, cmap="viridis")
    ax[1].set_title("peaks per position")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax


def plot_reference_lattice(
    bvm: np.ndarray,
    ref_qpos: np.ndarray,
    ref_ab: np.ndarray,
    origin: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    *,
    cmap: str = "gray",
    norm: str | dict = "log_auto",
    zoom: bool = True,
    figsize: tuple[float, float] = (6, 6),
    **show_kwargs,
):
    """The reference lattice from :meth:`BraggVectors.index_peaks`, drawn over the BVM.

    Each indexed reference site is a ring labelled with its ``(a, b)`` Miller index;
    the chosen origin is a green marker and ``g1`` (red) / ``g2`` (blue) are arrows
    from it. The ring color encodes how far the picked candidate sits from its *ideal*
    lattice site ``origin + a*g1 + b*g2`` (the colorbar reads pixels), scaled to half
    the shorter lattice spacing — the default :meth:`fit_lattice` match radius. A
    mis-picked or duplicate candidate stands out as a ring far from zero offset (bright
    color), sitting off the regular grid, or carrying an index that breaks the pattern.

    Parameters
    ----------
    bvm : np.ndarray
        Bragg vector map; rendered through
        :func:`~quantem.core.visualization.show_2d` so its display scaling is
        controlled by ``norm`` and ``show_kwargs``.
    ref_qpos : np.ndarray
        ``(N, 2)`` ``[row, col]`` reference site positions.
    ref_ab : np.ndarray
        ``(N, 2)`` integer ``[a, b]`` Miller indices for each site.
    origin : np.ndarray
        ``(row, col)`` chosen lattice origin.
    g1 : np.ndarray
        First lattice vector as a ``(row, col)`` offset from ``origin``.
    g2 : np.ndarray
        Second lattice vector as a ``(row, col)`` offset from ``origin``.
    cmap : str, default="gray"
        Colormap for the Bragg vector map; gray keeps the colored overlays legible.
    norm : str or dict, default="log_auto"
        Intensity scaling forwarded to :func:`show_2d` (e.g. ``"linear_auto"``,
        ``"log_auto"``, ``"power_sqrt"``, or ``{"power": 0.5}``).
    zoom : bool, default=True
        If ``True``, frame the view to the reference bounding box (plus a margin)
        so the labelled sites are large enough to read.
    figsize : tuple of float, default=(6, 6)
        Figure size in inches.
    **show_kwargs
        Extra keyword arguments forwarded to :func:`show_2d` for fine display
        control (e.g. ``vmin``, ``vmax``, ``lower_quantile``, ``upper_quantile``).

    Returns
    -------
    tuple
        ``(fig, ax)``.
    """
    import matplotlib.patheffects as path_effects
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from quantem.core.visualization import show_2d

    stroke = [path_effects.withStroke(linewidth=2.5, foreground="black")]
    origin_color = (0.0, 0.7, 0.0)
    g1_color = (1.0, 0.0, 0.0)
    g2_color = (0.0, 0.7, 1.0)

    ref_qpos = np.asarray(ref_qpos, dtype=float).reshape(-1, 2)
    ref_ab = np.asarray(ref_ab, dtype=int).reshape(-1, 2)
    o = np.asarray(origin, dtype=float).reshape(2)
    g1 = np.asarray(g1, dtype=float).reshape(2)
    g2 = np.asarray(g2, dtype=float).reshape(2)

    fig, ax = plt.subplots(figsize=figsize)
    show_2d(np.asarray(bvm), figax=(fig, ax), cmap=cmap, norm=norm, **show_kwargs)

    # offset of each picked candidate from its ideal lattice site origin + a*g1 + b*g2;
    # this is the QC indicator -- a mis-picked or strongly strained candidate rings
    # far from zero, while a clean pick rings near it.
    ideal_qpos = o[None, :] + ref_ab[:, 0:1] * g1[None, :] + ref_ab[:, 1:2] * g2[None, :]
    offset = np.linalg.norm(ref_qpos - ideal_qpos, axis=1)

    # color the rings by that offset, scaled to half the shorter lattice spacing (the
    # default fit_lattice match radius) so the colorbar previews which candidates sit
    # near the inclusion tolerance.
    radius = 0.5 * float(min(np.hypot(*g1), np.hypot(*g2)))
    cmap_offset = plt.get_cmap("plasma")
    norm_offset = Normalize(vmin=0.0, vmax=radius if radius > 0 else 1.0)

    ax.scatter(
        ref_qpos[:, 1],
        ref_qpos[:, 0],
        s=80,
        facecolors="none",
        edgecolors=cmap_offset(norm_offset(offset)),
        linewidths=1.8,
        zorder=3,
    )
    for (r, c), (a, b) in zip(ref_qpos, ref_ab):
        ax.annotate(
            f"{int(a)},{int(b)}",
            (c, r),
            color="white",
            fontsize=9,
            fontweight="bold",
            xytext=(4, 4),
            textcoords="offset points",
            path_effects=stroke,
            zorder=4,
        )

    ax.scatter(
        o[1],
        o[0],
        s=160,
        marker="P",
        facecolors=[origin_color],
        edgecolors="white",
        linewidths=1.5,
        zorder=6,
    )
    for g, label, color in ((g1, "g1", g1_color), (g2, "g2", g2_color)):
        ax.annotate(
            "",
            xy=(o[1] + g[1], o[0] + g[0]),
            xytext=(o[1], o[0]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.4, shrinkA=0, shrinkB=0),
            zorder=5,
        )
        gnorm = float(np.hypot(g[0], g[1])) + 1e-12
        ax.annotate(
            label,
            (o[1] + g[1] / 2.0, o[0] + g[0] / 2.0),
            color=color,
            fontsize=13,
            fontweight="bold",
            xytext=(g[0] / gnorm * 15.0, g[1] / gnorm * 15.0),
            textcoords="offset points",
            ha="center",
            va="center",
            path_effects=stroke,
            zorder=7,
        )

    if zoom and ref_qpos.shape[0]:
        rmin, cmin = ref_qpos.min(axis=0)
        rmax, cmax = ref_qpos.max(axis=0)
        margin = 0.12 * max(rmax - rmin, cmax - cmin, 1.0) + 6.0
        h, w = bvm.shape[:2]
        ax.set_xlim(max(cmin - margin, -0.5), min(cmax + margin, w - 0.5))
        ax.set_ylim(min(rmax + margin, h - 0.5), max(rmin - margin, -0.5))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("reference lattice  (ring color = offset from ideal index;  +origin,  g1,  g2)")

    sm = ScalarMappable(norm=norm_offset, cmap=cmap_offset)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, extend="max")
    cbar.set_label("peak offset from ideal index (px)")

    fig.tight_layout()
    return fig, ax


def plot_lattice_fit(
    mask_weight: np.ndarray,
    fit_error: np.ndarray,
    *,
    figsize: tuple[float, float] = (11, 4.5),
):
    """Per-position diagnostics from :meth:`BraggVectors.fit_lattice`.

    Left: the mask weight — a lattice *order parameter* per position (how well all
    detected intensity snaps to the fitted lattice, intensity-weighted) — so ``0`` is
    a position dominated by off-lattice intensity and ``1`` a clean single crystal.
    This is the weighting handed to the strain reference. Right: the RMS lattice-fit
    residual over the matched peaks, in pixels (low = a clean fit; high = a poorly
    fit, overlapping, or strongly strained position).

    Parameters
    ----------
    mask_weight : np.ndarray
        ``(scan_row, scan_col)`` lattice-order-parameter weight in ``[0, 1]``.
    fit_error : np.ndarray
        ``(scan_row, scan_col)`` RMS fit residual in pixels (``nan`` where no fit
        was made).
    figsize : tuple of float, default=(11, 4.5)
        Figure size in inches.

    Returns
    -------
    tuple
        ``(fig, ax)`` with ``ax`` a length-2 array of axes.
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    im0 = ax[0].imshow(np.asarray(mask_weight), cmap="viridis", vmin=0.0, vmax=1.0)
    ax[0].set_title("mask weight  (lattice order)")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(np.asarray(fit_error), cmap="magma")
    ax[1].set_title("fit RMS error  (px)")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    return fig, ax
