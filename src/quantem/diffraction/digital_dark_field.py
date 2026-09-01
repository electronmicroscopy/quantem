"""Clustering-based digital dark field imaging.

Implements the workflow of MacLaren and co-workers: DBSCAN in the joint
(diffraction, scan) space groups the detected Bragg peaks into single-spot,
single-crystallite clusters (L1); clustering the real-space centers of mass
of those clusters groups the g-vectors of each grain (L2); clustering the
remaining unindexed peaks in diffraction space alone isolates ring-like
nanocrystalline or amorphous components (L3). Each cluster's summed
intensity per probe position is a digital dark field image.

The clustering itself is the generic quantem.core.utils.clustering.dbscan /
cluster_vector; this module holds the diffraction-specific pieces: centers
of mass, DDF image formation, and composite color rendering.
"""

from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb   

from quantem.core.datastructures.vector import Vector
from quantem.core.utils.clustering import cluster_vector, dbscan  # noqa: F401


def _scan_cells(vector) -> np.ndarray:
    """(N, 2) scan (row, col) of every flattened row of a ragged Vector."""
    counts = np.asarray(vector.row_counts(), dtype=int)
    shape = vector.shape[:2]
    cell_r, cell_c = np.divmod(np.arange(counts.size), shape[1])
    return np.stack(
        [np.repeat(cell_r, counts), np.repeat(cell_c, counts)], axis=1
    )


def cluster_coms(
    labeled,
    label_field: str = "cluster",
    intensity_field: str = "intensity",
    weighted: bool = True,
):
    """Real-space center of mass of every cluster.

    Parameters
    ----------
    labeled : Vector
        Vector carrying a cluster label field (from cluster_vector).
    weighted : bool, default=True
        Weight the center of mass by peak intensity.

    Returns
    -------
    coms : np.ndarray
        (K, 2) scan-coordinate centers of mass, ordered by cluster id.
    sizes : np.ndarray
        (K,) number of peaks per cluster.
    """
    fields = labeled.fields
    flat = labeled.flatten()
    labels = flat[:, fields.index(label_field)].astype(int)
    w = flat[:, fields.index(intensity_field)].clip(min=0) if weighted else None
    rc = _scan_cells(labeled).astype(float)

    n = labels.max() + 1
    coms = np.zeros((n, 2))
    sizes = np.zeros(n, dtype=int)
    for k in range(n):
        m = labels == k
        sizes[k] = int(m.sum())
        if sizes[k] == 0:
            coms[k] = np.nan
            continue
        wk = w[m] if w is not None else np.ones(sizes[k])
        wk = wk / max(wk.sum(), 1e-12)
        coms[k] = (rc[m] * wk[:, None]).sum(axis=0)
    return coms, sizes


def ddf_images(
    labeled,
    cluster_ids,
    label_field: str = "cluster",
    intensity_field: str = "intensity",
) -> np.ndarray:
    """Digital dark field images: per-cluster summed intensity per position.

    Returns
    -------
    np.ndarray
        (len(cluster_ids), scan_row, scan_col) images.
    """
    fields = labeled.fields
    flat = labeled.flatten()
    labels = flat[:, fields.index(label_field)].astype(int)
    inten = flat[:, fields.index(intensity_field)].clip(min=0)
    rc = _scan_cells(labeled)
    R, C = labeled.shape[:2]

    out = np.zeros((len(cluster_ids), R, C))
    for i, k in enumerate(np.atleast_1d(cluster_ids)):
        m = labels == k
        np.add.at(out[i], (rc[m, 0], rc[m, 1]), inten[m])
    return out


def composite_ddf(
    images: np.ndarray,
    colors=None,
    gamma: float = 0.33,
    normalize: str = "each",
) -> np.ndarray:
    """Blend a stack of DDF images into one RGB composite.

    Parameters
    ----------
    images : np.ndarray
        (K, R, C) cluster images.
    colors : array-like | None
        (K, 3) RGB color per image; defaults to evenly spaced hues.
    gamma : float, default=0.33
        Power scaling applied to each normalized image before coloring.
    normalize : {"each", "global"}
        Normalize each image to its own maximum, or all to the stack max.

    Returns
    -------
    np.ndarray
        (R, C, 3) RGB image in [0, 1].
    """

    K = images.shape[0]
    if colors is None:
        hues = np.linspace(0, 1, K, endpoint=False)
        colors = hsv_to_rgb(
            np.stack([hues, np.ones(K), np.ones(K)], axis=1)
        )
    colors = np.asarray(colors, dtype=float)

    if normalize == "global":
        norm = np.full(K, max(float(images.max()), 1e-12))
    else:
        norm = np.maximum(images.reshape(K, -1).max(axis=1), 1e-12)
    scaled = (images / norm[:, None, None]) ** gamma
    rgb = np.einsum("krc,kj->rcj", scaled, colors)
    return np.clip(rgb, 0, 1)


def color_wheel(n: int = 256, saturation: float = 1.0) -> np.ndarray:
    """
    (n, n, 4) RGBA hue wheel for labeling composite images.

    Parameters
    n: int
        Size of the output image.
    saturation: float
        Saturation of the hue wheel (0=gray, 1=full color).

    Returns
    -------
    np.ndarray
        (n, n, 4) RGBA image in [0, 1].
    """
     

    y, x = np.mgrid[-1 : 1 : n * 1j, -1 : 1 : n * 1j]
    r = np.hypot(x, y)
    hue = (np.arctan2(y, x) / (2 * np.pi)) % 1.0
    hsv = np.stack(
        [hue, np.full_like(hue, saturation), np.clip(r, 0, 1)], axis=-1
    )
    rgba = np.concatenate(
        [hsv_to_rgb(hsv), (r <= 1.0)[..., None].astype(float)], axis=-1
    )
    return rgba


def plot_cluster_scatter(
    labeled,
    q_fields=("qx", "qy"),
    label_field: str = "cluster",
    specific_cluster: int | None = None,
    max_clusters: int | None = None,
    show_unclustered: bool = True,
    point_size: float = 2.0,
    alpha: float = 0.2,
    figax=None,
):
    """
    All peaks in diffraction space, colored by cluster (unclustered in gray).

    Parameters
    ----------
    labeled : Vector
        Labeled diffraction data.
    q_fields: tuple of str
        Field names for the diffraction-space coordinates to plot.
    label_field: str
        Field name for the cluster label.
    specific_cluster: int | None
        If given, plot only this cluster (unclustered points are not shown).
    max_clusters: int | None
        Maximum number of clusters to plot (for large datasets).
    show_unclustered: bool
        Whether to show unclustered points (label < 0) in gray.
    point_size: float
        Size of the scatter points.
    alpha: float
        Transparency of the scatter points.  Recommended to be << 1.0 for large datasets so only dense
        regions are visible.
    figax: tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) | None
        If provided, plot into this figure and axes instead of creating a new one.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure containing the scatter plot.
    ax: matplotlib.axes.Axes
        The axes containing the scatter plot.
    """

    fields = labeled.fields
    flat = labeled.flatten()
    labels = flat[:, fields.index(label_field)].astype(int)
    qx = flat[:, fields.index(q_fields[0])]
    qy = flat[:, fields.index(q_fields[1])]

    if figax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
    else:
        fig, ax = figax
    q_max = np.max(np.abs([qx, qy]))*1.05
    ax.set_xlim(-q_max, q_max)
    ax.set_ylim(-q_max, q_max)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("$q_y$")
    ax.set_ylabel("$q_x$")

    if specific_cluster is not None:
        m = labels == specific_cluster
        ax.scatter(qy[m], qx[m], s=point_size, color="C0", lw=0, alpha=alpha)
        return fig, ax
    
    if show_unclustered:
        m = labels < 0
        ax.scatter(qy[m], qx[m], s=point_size, color="0.85", lw=0)
    n = labels.max() + 1
    ids = range(n if max_clusters is None else min(n, max_clusters))
    cmap = plt.get_cmap("hsv")
    rng = np.random.default_rng(0)
    hues = rng.permutation(np.linspace(0, 1, len(list(ids)), endpoint=False))
    for k in ids:
        m = labels == k
        ax.scatter(qy[m], qx[m], s=point_size, color=cmap(hues[k]), lw=0, alpha=alpha)
    return fig, ax

def assign_grain_labels(L1, L2_labels,label_field: str = "cluster"):
    """Assign L2 grain labels to L1 clusters based on their centers of mass.

    Parameters
    ----------
    L1 : Vector
        Labeled diffraction data with L1 cluster labels.
    L2_labels : np.ndarray
        (K,) array of L2 grain labels corresponding to each L1 cluster.

    Returns
    -------
    L2 : Vector
        Labeled diffraction data with L2 grain labels assigned.
    L2_labels : np.ndarray
        Updated array of L2 grain labels.
    """
    fields = L1.fields
    flat = L1.flatten()
    L1_labels = flat[:, fields.index(label_field)].astype(int)
    L1_unique = np.unique(L1_labels)

    L1_to_L2 = np.insert(L2_labels,0,-2)
    mapper = dict(zip(L1_unique, L1_to_L2))
    L2_labels_full = np.array([mapper[label] for label in L1_labels])
    
    L2 = L1.copy()
    L2.add_fields("grain_label", values = L2_labels_full)
    
    # Return the new Vector with L2 labels
    return L2