"""
Sibson (natural-neighbor) interpolation weights for partitioned-PRISM beamlets.

Computes the interpolation weights of a dense set of beamlet wave vectors with
respect to a sparse set of parent wave vectors, using Sibson's area-stealing
construction on the Delaunay triangulation of the parents. Adapted from MetPy's
natural-neighbor implementation.

All functions are numpy/scipy based and non-differentiable: the weights are pure
geometry, computed once at probe initialization and frozen as buffers.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree  # type: ignore


def triangle_area(pt1, pt2, pt3) -> float:
    a = (
        pt1[0] * pt2[1]
        - pt2[0] * pt1[1]
        + pt2[0] * pt3[1]
        - pt3[0] * pt2[1]
        + pt3[0] * pt1[1]
        - pt1[0] * pt3[1]
    )
    return abs(a) / 2


def circumcircle_radius(pt0, pt1, pt2) -> float:
    a = np.linalg.norm(pt0 - pt1)
    b = np.linalg.norm(pt1 - pt2)
    c = np.linalg.norm(pt2 - pt0)

    t_area = triangle_area(pt0, pt1, pt2)

    if t_area > 0:
        radius = (a * b * c) / (4 * t_area)
    else:
        radius = np.nan

    return radius


def circumcenter(pt0, pt1, pt2, eps: float = 1e-12):
    a_x, a_y = pt0
    b_x, b_y = pt1
    c_x, c_y = pt2

    bc_y_diff = b_y - c_y
    ca_y_diff = c_y - a_y
    ab_y_diff = a_y - b_y
    cb_x_diff = c_x - b_x
    ac_x_diff = a_x - c_x
    ba_x_diff = b_x - a_x

    d_div = a_x * bc_y_diff + b_x * ca_y_diff + c_x * ab_y_diff

    if abs(d_div) < eps:
        # (near-)collinear points have no circumcenter; fall back to centroid
        return np.mean([pt0, pt1, pt2], axis=0)

    d_inv = 0.5 / d_div

    a_mag = a_x * a_x + a_y * a_y
    b_mag = b_x * b_x + b_y * b_y
    c_mag = c_x * c_x + c_y * c_y

    cx = (a_mag * bc_y_diff + b_mag * ca_y_diff + c_mag * ab_y_diff) * d_inv
    cy = (a_mag * cb_x_diff + b_mag * ac_x_diff + c_mag * ba_x_diff) * d_inv
    return cx, cy


def find_natural_neighbors(tri: Delaunay, grid_points: NDArray):
    """For each query point, find the Delaunay simplices whose circumcircle contains it.

    Returns
    -------
    members : dict[int, list[int]]
        Maps query-point index to the list of "natural neighbor" simplex indices.
    circumcenters : NDArray
        Circumcenter of each simplex, shape (num_simplices, 2).
    """
    tree = cKDTree(grid_points)

    triangle_info = []
    members: dict[int, list[int]] = {key: [] for key in range(len(tree.data))}
    for i, indices in enumerate(tri.simplices):
        triangle = tri.points[indices]
        cc = circumcenter(*triangle)
        r = circumcircle_radius(*triangle)
        triangle_info.append(cc)

        for point in tree.query_ball_point(cc, r):
            members[point].append(i)

    return members, np.array(triangle_info)


def find_local_boundary(tri: Delaunay, triangles: list[int]) -> list[tuple[int, int]]:
    """Ordered outer boundary edges of a set of simplices (interior edges cancel)."""
    edges: list[tuple[int, int]] = []

    for triangle in triangles:
        for i in range(3):
            pt1 = tri.simplices[triangle][i]
            pt2 = tri.simplices[triangle][(i + 1) % 3]

            if (pt1, pt2) in edges:
                edges.remove((pt1, pt2))
            elif (pt2, pt1) in edges:
                edges.remove((pt2, pt1))
            else:
                edges.append((pt1, pt2))

    return edges


def polygon_area(poly) -> float:
    a = 0.0
    n = len(poly)

    for i in range(n):
        a += poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]

    return abs(a) / 2.0


def order_edges(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    edge = edges[0]
    edges = edges[1:]

    ordered_edges = [edge]

    num_max = len(edges)
    while len(edges) > 0 and num_max > 0:
        match = edge[1]

        for search_edge in edges:
            vertex = search_edge[0]
            if match == vertex:
                edge = search_edge
                edges.remove(edge)
                ordered_edges.append(search_edge)
                break
        num_max -= 1

    return ordered_edges


def natural_neighbor_weights(
    points: NDArray,
    query_point: NDArray,
    tri: Delaunay,
    neighbors: list[int],
    circumcenters: NDArray,
) -> NDArray:
    """Sibson weights of `query_point` with respect to `points`.

    Weights are non-negative and sum to 1. A query point coinciding with a data
    point gets a one-hot weight. A query point outside every circumcircle (no
    natural neighbors, possible just outside the parents' convex hull) falls
    back to a one-hot weight on the nearest data point.
    """
    weights = np.zeros(len(points))

    overlap = np.isclose(query_point[0], points[:, 0]) * np.isclose(query_point[1], points[:, 1])

    if np.any(overlap):
        weights[np.where(overlap)[0]] = 1.0
        return weights

    if len(neighbors) == 0:
        distances = np.linalg.norm(points - query_point[None], axis=1)
        weights[np.argmin(distances)] = 1.0
        return weights

    edges = find_local_boundary(tri, neighbors)
    edge_vertices = [segment[0] for segment in order_edges(edges)]
    num_vertices = len(edge_vertices)

    p1 = edge_vertices[0]
    p2 = edge_vertices[1]

    c1 = circumcenter(query_point, tri.points[p1], tri.points[p2])
    polygon = [c1]
    for i in range(num_vertices):
        p3 = edge_vertices[(i + 2) % num_vertices]

        c2 = circumcenter(query_point, tri.points[p3], tri.points[p2])
        polygon.append(c2)

        for check_tri in neighbors:
            if p2 in tri.simplices[check_tri]:
                polygon.append(circumcenters[check_tri])

        pts = [polygon[i] for i in ConvexHull(polygon).vertices]
        area = polygon_area(pts)
        weights[(tri.points[p2][0] == points[:, 0]) & (tri.points[p2][1] == points[:, 1])] += area

        polygon = [c2]
        p2 = p3

    return weights / weights.sum()


def pairwise_weights(points: NDArray, query_points: NDArray) -> NDArray:
    """Sibson weights of each query point w.r.t. `points`.

    Returns
    -------
    weights : NDArray
        Shape (len(points), len(query_points)); each column is non-negative
        and sums to 1.
    """
    triangulation = Delaunay(points)
    members, circumcenters = find_natural_neighbors(triangulation, query_points)
    radii = np.array(
        [circumcircle_radius(*triangulation.points[s]) for s in triangulation.simplices]
    )
    jitter = 1e-9 * np.ptp(points, axis=0).max()
    weights = np.zeros((len(points), len(query_points)))
    for i, query_point in enumerate(query_points):
        try:
            weights[:, i] = natural_neighbor_weights(
                points, query_point, triangulation, members[i], circumcenters
            )
        except QhullError:
            # a query exactly on a triangulation edge degenerates its Sibson polygon;
            # the weights are continuous in the query, so nudge it off the edge
            nudged = query_point + jitter
            distances = np.linalg.norm(circumcenters - nudged[None], axis=1)
            neighbors = list(np.where(distances < radii)[0])
            weights[:, i] = natural_neighbor_weights(
                points, nudged, triangulation, neighbors, circumcenters
            )

    return weights


def _wave_vector_indices(
    wave_vectors: NDArray,
    gpts: tuple[int, int] | NDArray,
    sampling: tuple[float, float] | NDArray,
) -> tuple[NDArray, NDArray]:
    """Map wave vectors k = n/extent to their fftfreq-grid pixel indices.

    Wave vectors must lie on the reciprocal grid of (gpts, sampling), i.e.
    k = n / (gpts * sampling) with integer n; raises otherwise.
    """
    extent = np.asarray(gpts) * np.asarray(sampling)
    n = wave_vectors * extent[None]
    n_int = np.rint(n).astype(np.int64)
    if not np.allclose(n, n_int, atol=1e-6):
        raise ValueError("wave_vectors do not lie on the reciprocal grid of (gpts, sampling)")

    rows = np.mod(n_int[:, 0], gpts[0])
    cols = np.mod(n_int[:, 1], gpts[1])
    return rows, cols


def beamlet_weights(
    parent_wave_vectors: NDArray,
    wave_vectors: NDArray,
    gpts: tuple[int, int] | NDArray,
    sampling: tuple[float, float] | NDArray,
) -> NDArray:
    """Scatter Sibson weights of beamlets w.r.t. parent beams onto the reciprocal grid.

    Parameters
    ----------
    parent_wave_vectors : NDArray
        Parent beam wave vectors in inverse Angstroms, shape (num_parent, 2).
    wave_vectors : NDArray
        Dense beamlet wave vectors on the fftfreq grid of (gpts, sampling),
        shape (num_beamlets, 2).
    gpts : tuple[int, int]
        Grid shape (typically the ROI shape).
    sampling : tuple[float, float]
        Real-space sampling in Angstroms.

    Returns
    -------
    weights : NDArray
        Shape (num_parent, *gpts); weights[:, i, j] sums to 1 on beamlet pixels
        and is 0 elsewhere.
    """
    pwv_np = np.asarray(parent_wave_vectors)
    wv_np = np.asarray(wave_vectors)
    point_weights = pairwise_weights(pwv_np, wv_np)

    rows, cols = _wave_vector_indices(wv_np, gpts, sampling)
    weights = np.zeros((len(pwv_np), gpts[0], gpts[1]))
    weights[:, rows, cols] = point_weights

    return weights


def one_hot_beamlet_weights(
    wave_vectors: NDArray,
    gpts: tuple[int, int] | NDArray,
    sampling: tuple[float, float] | NDArray,
) -> NDArray:
    """One-hot weights for the dense (exact-PRISM) limit where parents == beamlets.

    Skips the Delaunay construction entirely; equivalent to `beamlet_weights`
    with parent_wave_vectors == wave_vectors.
    """
    wv_np = np.asarray(wave_vectors)
    rows, cols = _wave_vector_indices(wv_np, gpts, sampling)
    weights = np.zeros((len(wv_np), gpts[0], gpts[1]))
    weights[np.arange(len(wv_np)), rows, cols] = 1.0

    return weights
