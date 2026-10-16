"""Per-site measurements derived from template matching and neighbor lists.

Functions here operate on plain arrays so they can be tested independently of
:class:`~quantem.atoms.AtomicModel`, which wraps them.  Everything is
vectorized with NumPy / SciPy; nothing loops over atoms in Python.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull

__all__ = [
    "misorientation",
    "segment_grains",
    "fill_labels",
    "strain_from_deformation",
    "fit_plane_intersection",
    "layer_positions",
    "bond_angles",
    "convex_hull_distance",
    "sample_volume",
    "kmeans_1d",
    "gaussian_mixture_1d",
    "rotation_to_quaternion",
]


def misorientation(
    rotation: NDArray,
    neighbor_index: NDArray,
    symmetry: NDArray | None = None,
    chunk_size: int = 2048,
) -> NDArray:
    """Disorientation angle between every site and each of its neighbors.

    Parameters
    ----------
    rotation : ndarray
        ``(N, 3, 3)`` site orientations (lab <- crystal).
    neighbor_index : ndarray
        ``(N, K)`` neighbor indices; ``-1`` marks missing neighbors.
    symmetry : ndarray, optional
        ``(S, 3, 3)`` proper symmetry rotations of the crystal; the minimum
        angle over all symmetry-equivalent descriptions is returned.
    chunk_size : int
        Sites per batch.

    Returns
    -------
    ndarray
        ``(N, K)`` angles in degrees, ``nan`` for missing neighbors.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    n, k = neighbor_index.shape
    if symmetry is None:
        symmetry = np.eye(3)[None]
    out = np.full((n, k), np.nan)
    idx_safe = np.where(neighbor_index >= 0, neighbor_index, 0)
    for start in range(0, n, chunk_size):
        sl = slice(start, min(start + chunk_size, n))
        ra = rotation[sl]  # (n,3,3)
        rb = rotation[idx_safe[sl]]  # (n,k,3,3)
        delta = np.einsum("nji,nkjl->nkil", ra, rb)  # R_a^T R_b
        trace = np.einsum("nkij,sji->nks", delta, symmetry).max(-1)
        ang = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
        out[sl] = ang
    out[neighbor_index < 0] = np.nan
    return out


def segment_grains(
    neighbor_index: NDArray,
    edge_mask: NDArray,
    min_size: int = 1,
) -> NDArray:
    """Label connected components of a neighbor graph.

    Parameters
    ----------
    neighbor_index : ndarray
        ``(N, K)`` neighbor indices (``-1`` = missing).
    edge_mask : ndarray
        ``(N, K)`` boolean; ``True`` where site ``n`` and neighbor ``k`` belong
        to the same grain (e.g. same structure and small misorientation).
    min_size : int
        Components smaller than this are labelled ``-1``.

    Returns
    -------
    ndarray
        ``(N,)`` integer labels ordered by decreasing size (0 = largest),
        ``-1`` for sites in components smaller than ``min_size``.
    """
    n, k = neighbor_index.shape
    rows = np.repeat(np.arange(n), k)
    cols = neighbor_index.ravel()
    keep = edge_mask.ravel() & (cols >= 0)
    graph = coo_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    order = np.argsort(-sizes, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    out = rank[labels]
    out[sizes[labels] < min_size] = -1
    return out


def fill_labels(
    labels: NDArray, neighbor_index: NDArray, edge_mask: NDArray, member: NDArray
) -> NDArray:
    """Assign unlabelled member sites to the majority label of their neighbors.

    Parameters
    ----------
    labels : ndarray
        ``(N,)`` labels, ``-1`` = unassigned.
    neighbor_index : ndarray
        ``(N, K)`` neighbor indices (``-1`` = missing).
    edge_mask : ndarray
        ``(N, K)`` votes are only counted along ``True`` edges.
    member : ndarray
        ``(N,)`` sites eligible for filling.

    Returns
    -------
    ndarray
        Updated copy of ``labels``.
    """
    labels = np.array(labels, copy=True)
    todo = np.where(member & (labels < 0))[0]
    if todo.size == 0:
        return labels
    safe = np.where(neighbor_index >= 0, neighbor_index, 0)
    votes = np.where(edge_mask & (neighbor_index >= 0), labels[safe], -1)[todo]
    n_max = int(labels.max()) + 2
    counts = np.zeros((todo.size, n_max), dtype=int)
    rows = np.repeat(np.arange(todo.size), votes.shape[1])
    valid = votes.ravel() >= 0
    np.add.at(counts, (rows[valid], votes.ravel()[valid]), 1)
    best = counts.argmax(axis=1)
    has_votes = counts.max(axis=1) > 0
    labels[todo[has_votes]] = best[has_votes]
    return labels


def strain_from_deformation(deformation: NDArray, rotation: NDArray, frame: str = "lab") -> dict:
    """Small-strain tensor components from a deformation gradient.

    With ``p ~ F t`` and polar decomposition ``F = V R = R U`` the stretch in
    the lab frame is ``V`` and in the crystal (template) frame ``U``.  The
    strain is ``sym(V) - I`` or ``sym(U) - I``.

    Parameters
    ----------
    deformation : ndarray
        ``(N, 3, 3)`` deformation gradients ``F``.
    rotation : ndarray
        ``(N, 3, 3)`` rotations ``R`` from the rigid fit.
    frame : {"lab", "crystal"}
        Frame in which to express the strain.

    Returns
    -------
    dict of ndarray
        ``e_xx, e_yy, e_zz, e_xy, e_xz, e_yz`` components, ``dilation``
        (mean normal strain) and ``equivalent`` (von Mises deviatoric strain),
        plus the full ``(N, 3, 3)`` tensor under ``"tensor"``.
    """
    f = np.asarray(deformation, dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)
    if frame == "lab":
        stretch = f @ np.transpose(r, (0, 2, 1))
    elif frame == "crystal":
        stretch = np.transpose(r, (0, 2, 1)) @ f
    else:
        raise ValueError("frame must be 'lab' or 'crystal'")
    e = 0.5 * (stretch + np.transpose(stretch, (0, 2, 1))) - np.eye(3)[None]
    dil = np.trace(e, axis1=1, axis2=2) / 3.0
    dev = e - dil[:, None, None] * np.eye(3)[None]
    equivalent = np.sqrt((2.0 / 3.0) * np.einsum("nij,nij->n", dev, dev))
    return {
        "e_xx": e[:, 0, 0],
        "e_yy": e[:, 1, 1],
        "e_zz": e[:, 2, 2],
        "e_xy": e[:, 0, 1],
        "e_xz": e[:, 0, 2],
        "e_yz": e[:, 1, 2],
        "dilation": dil,
        "equivalent": equivalent,
        "tensor": e,
    }


def bond_angles(dxyz: NDArray, valid: NDArray) -> NDArray:
    """All bond angles at each site between pairs of valid neighbor vectors.

    Parameters
    ----------
    dxyz : ndarray
        ``(N, K, 3)`` neighbor vectors.
    valid : ndarray
        ``(N, K)`` mask of first-shell neighbors.

    Returns
    -------
    ndarray
        ``(N, K*(K-1)/2)`` angles in degrees, ``nan`` where either neighbor is
        invalid.
    """
    n, k, _ = dxyz.shape
    unit = dxyz / np.linalg.norm(dxyz, axis=-1, keepdims=True).clip(1e-12)
    iu, ju = np.triu_indices(k, 1)
    cos = np.einsum("nkd,nkd->nk", unit[:, iu], unit[:, ju])
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ang[~(valid[:, iu] & valid[:, ju])] = np.nan
    return ang


def convex_hull_distance(xyz: NDArray) -> NDArray:
    """Signed distance from each point to the convex hull surface (positive inside)."""
    xyz = np.asarray(xyz, dtype=float)
    hull = ConvexHull(xyz)
    eq = hull.equations  # (F, 4): n . x + d = 0, outward normals
    d = -(xyz @ eq[:, :3].T + eq[:, 3][None, :])
    return d.min(axis=1)


def sample_volume(volume: NDArray, xyz: NDArray, radius: float = 1.5) -> NDArray:
    """Mean volume intensity inside a sphere around each site (voxel units).

    Parameters
    ----------
    volume : ndarray
        3D array indexed ``[x, y, z]`` matching the coordinate order of ``xyz``.
    xyz : ndarray
        ``(N, 3)`` site positions in voxel coordinates.
    radius : float
        Integration sphere radius in voxels.

    Returns
    -------
    ndarray
        ``(N,)`` mean intensity; sites whose sphere leaves the volume use only
        the in-bounds voxels.
    """
    volume = np.asarray(volume)
    xyz = np.asarray(xyz, dtype=float)
    r = int(np.ceil(radius))
    rng = np.arange(-r, r + 1)
    off = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    off = off[np.linalg.norm(off, axis=1) <= radius]
    center = np.rint(xyz).astype(int)
    idx = center[:, None, :] + off[None, :, :]  # (N, V, 3)
    shape = np.array(volume.shape)
    inside = np.all((idx >= 0) & (idx < shape), axis=-1)
    idx = np.clip(idx, 0, shape - 1)
    vals = volume[idx[..., 0], idx[..., 1], idx[..., 2]].astype(float)
    vals[~inside] = 0.0
    counts = inside.sum(1).clip(1)
    return vals.sum(1) / counts


def kmeans_1d(
    values: NDArray, num_clusters: int = 2, num_iter: int = 50
) -> tuple[NDArray, NDArray]:
    """Simple 1D k-means with quantile initialization.

    Returns
    -------
    labels, centers : ndarray
        ``(N,)`` labels sorted so that cluster 0 has the smallest center, and
        ``(num_clusters,)`` sorted centers.
    """
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    q = np.linspace(0, 1, num_clusters + 2)[1:-1]
    centers = np.quantile(v[finite], q)
    labels = np.zeros(v.shape, dtype=int)
    for _ in range(num_iter):
        labels = np.argmin(np.abs(v[:, None] - centers[None, :]), axis=1)
        new = np.array(
            [
                v[finite & (labels == c)].mean() if np.any(finite & (labels == c)) else centers[c]
                for c in range(num_clusters)
            ]
        )
        if np.allclose(new, centers):
            break
        centers = new
    order = np.argsort(centers)
    remap = np.empty_like(order)
    remap[order] = np.arange(num_clusters)
    return remap[labels], centers[order]


def gaussian_mixture_1d(
    values: NDArray, num_components: int = 2, num_iter: int = 200, tol: float = 1e-8
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Fit a one-dimensional Gaussian mixture by expectation maximization.

    Components are initialized from k-means and sorted by increasing mean.

    Returns
    -------
    means, sigmas, weights, responsibilities : ndarray
        Component parameters ``(K,)`` and the ``(N, K)`` posterior
        probabilities of each site (rows sum to 1; non-finite values give
        uniform rows).
    """
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    x = v[finite]
    labels, means = kmeans_1d(x, num_components)
    sigmas = np.array(
        [x[labels == k].std() if np.any(labels == k) else x.std() for k in range(num_components)]
    )
    sigmas = np.maximum(sigmas, 1e-6 * (x.std() + 1e-12))
    weights = np.bincount(labels, minlength=num_components) / x.size
    prev = -np.inf
    for _ in range(num_iter):
        log_p = (
            np.log(weights + 1e-300)[None, :]
            - 0.5 * ((x[:, None] - means[None, :]) / sigmas[None, :]) ** 2
            - np.log(sigmas[None, :])
            - 0.5 * np.log(2 * np.pi)
        )
        log_norm = np.logaddexp.reduce(log_p, axis=1)
        resp = np.exp(log_p - log_norm[:, None])
        ll = float(log_norm.sum())
        nk = resp.sum(0) + 1e-12
        means = (resp * x[:, None]).sum(0) / nk
        sigmas = np.sqrt((resp * (x[:, None] - means[None, :]) ** 2).sum(0) / nk)
        sigmas = np.maximum(sigmas, 1e-6 * (x.std() + 1e-12))
        weights = nk / x.size
        if ll - prev < tol * max(1.0, abs(ll)):
            break
        prev = ll
    order = np.argsort(means)
    means, sigmas, weights, resp = means[order], sigmas[order], weights[order], resp[:, order]
    out = np.full((v.size, num_components), 1.0 / num_components)
    out[finite] = resp
    return means, sigmas, weights, out


def rotation_to_quaternion(rotation: NDArray) -> NDArray:
    """Convert ``(N, 3, 3)`` rotation matrices to ``(N, 4)`` unit quaternions (w, x, y, z)."""
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(np.asarray(rotation)).as_quat()  # x, y, z, w
    q = np.roll(q, 1, axis=1)
    q[q[:, 0] < 0] *= -1
    return q


def fit_plane_intersection(points: NDArray, normals: NDArray) -> tuple[NDArray, NDArray]:
    """Least-squares point closest to a set of planes.

    Each plane passes through ``points[i]`` with unit normal ``normals[i]``;
    the returned point minimizes the sum of squared plane distances.

    Returns
    -------
    center, residuals : ndarray
        ``(3,)`` point and ``(N,)`` signed distances of the point to each plane.
    """
    n = np.asarray(normals, dtype=float)
    p = np.asarray(points, dtype=float)
    a = np.einsum("ni,nj->ij", n, n)
    b = np.einsum("ni,ni,nj->j", n, p, n)
    center = np.linalg.solve(a + 1e-9 * np.eye(3), b)
    return center, np.einsum("ni,ni->n", n, center[None, :] - p)


def layer_positions(
    heights: NDArray,
    bin_width: float,
    sigma: float,
    min_fraction: float = 0.25,
) -> tuple[NDArray, NDArray, NDArray]:
    """Peaks of the site density along one direction (atomic layers).

    Parameters
    ----------
    heights : ndarray
        ``(N,)`` coordinates of the sites along the direction.
    bin_width : float
        Histogram bin width.
    sigma : float
        Gaussian smoothing of the histogram (same units as ``heights``).
    min_fraction : float
        Peaks lower than this fraction of the highest peak are ignored.

    Returns
    -------
    positions, centers, density : ndarray
        Peak positions, and the smoothed histogram (bin centers and counts)
        for plotting.
    """
    from scipy.ndimage import gaussian_filter1d

    h = np.asarray(heights, dtype=float)
    edges = np.arange(h.min() - 2 * sigma, h.max() + 2 * sigma + bin_width, bin_width)
    counts = np.histogram(h, edges)[0].astype(float)
    smooth = gaussian_filter1d(counts, sigma / bin_width) if sigma > 0 else counts
    centers = 0.5 * (edges[1:] + edges[:-1])
    inner = smooth[1:-1]
    peaks = (
        np.where(
            (inner > smooth[:-2]) & (inner >= smooth[2:]) & (inner > min_fraction * smooth.max())
        )[0]
        + 1
    )
    # refine each peak with a parabola through its three bins
    pos, height = [], []
    for k in peaks:
        y0, y1, y2 = smooth[k - 1], smooth[k], smooth[k + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        pos.append(centers[k] + delta * bin_width)
        height.append(y1)
    pos, height = np.asarray(pos), np.asarray(height)
    # drop the weaker of any two peaks closer than half the median spacing
    if pos.size > 2:
        spacing = np.median(np.diff(pos))
        keep = np.ones(pos.size, dtype=bool)
        for k in range(1, pos.size):
            if pos[k] - pos[k - 1] < 0.5 * spacing:
                keep[k if height[k] < height[k - 1] else k - 1] = False
        pos = pos[keep]
    return pos, centers, smooth
