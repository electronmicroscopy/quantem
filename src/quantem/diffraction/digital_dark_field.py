"""Digital dark field imaging from detected Bragg peaks.

A digital dark field (DDF) image is the summed intensity of a selected subset
of the detected Bragg peaks at each probe position. These functions select
the peaks in three ways, following MacLaren and co-workers
(https://doi.org/10.1093/mam/ozae104):

- Virtual apertures: peaks within a radius of a set of aperture positions,
  usually a lattice built from two reciprocal lattice vectors
  (refine_lattice_vectors, lattice_distance, aperture_array,
  aperture_array_subtract, aperture_ddf_image).
- Polar selection: peaks within a ring of radius q, optionally restricted to
  a range of azimuthal angles (add_polar_fields, polar_mask,
  radial_ddf_image).
- Clustering: DBSCAN in the joint (diffraction, scan) space groups the peaks
  into single-spot, single-crystallite clusters (L1), clustering either the
  real-space centers of mass or the dark field images of those clusters
  groups the spots of each grain (L2), and clustering the remaining peaks in
  diffraction space alone isolates ring-like nanocrystalline or amorphous
  components (L3) (ddf_images, cluster_coms, cluster_centers,
  group_ddf_images, assign_grain_labels).

The aperture, polar and DDF image functions are ports of Ian MacLaren's
digital dark field functions in py4DSTEM (aperture_array_generator,
aperture_array_subtract, DDFimage, pointlist_to_array with rphi=True,
DDF_radial_image and DDFradialazimuthimage), rewritten to read the peaks
from a Vector.

All functions read the peaks from a Vector with one cell per probe position.
The diffraction coordinates are given by `q_fields`, which defaults to
("qx", "qy") for calibrated peaks or ("q_row", "q_col") for peaks in detector
pixels. Boolean masks are aligned with the flattened rows of the Vector, so
they can be combined with & and | and passed to ddf_image or to
quantem.core.utils.clustering.filter_rows.

Where a function needs the diffraction origin (`center=None`), calibrated
("qx", "qy") peaks are taken to be relative to the direct beam, so the origin
is (0, 0). Peaks in detector pixels ("q_row", "q_col") use the "origin_ref"
stored in the Vector metadata by BraggVectors.correct_peak_origins; without
it, pass `center` explicitly.

The azimuth qphi of the polar functions follows py4DSTEM's DDF functions:
qphi = atan2(-q0, q1) in degrees, measured anticlockwise from the +col
(right) direction as the pattern is displayed with rows increasing
downward, so phi ranges written for py4DSTEM carry over unchanged. This is
not the azimuth used by quantem.diffraction.calibration, atan2(q1, q0),
which is measured from the +row axis toward +col.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection
from matplotlib.colors import hsv_to_rgb

from quantem.core.utils.clustering import cluster_vector, dbscan  # noqa: F401


def _scan_cells(vector) -> np.ndarray:
    """(N, 2) scan (row, col) of every flattened row of a ragged Vector."""
    counts = np.asarray(vector.row_counts(), dtype=int)
    shape = vector.shape[:2]
    cell_r, cell_c = np.divmod(np.arange(counts.size), shape[1])
    return np.stack([np.repeat(cell_r, counts), np.repeat(cell_c, counts)], axis=1)


def _resolve_q_fields(vector, q_fields) -> tuple[str, str]:
    """The two diffraction coordinate fields of a peak Vector."""
    if q_fields is not None:
        return tuple(q_fields)
    for candidate in (("qx", "qy"), ("q_row", "q_col")):
        if all(f in vector.fields for f in candidate):
            return candidate
    raise KeyError(
        f"No diffraction coordinate fields found in {vector.fields}; pass q_fields explicitly."
    )


def _resolve_center(vector, q_fields, center) -> np.ndarray:
    """Diffraction origin: `center` if given, else (0, 0) or the stored origin_ref.

    Calibrated fields are relative to the direct beam already. Detector-pixel
    fields ("q_row", "q_col") need the common origin that
    BraggVectors.correct_peak_origins stores as "origin_ref".
    """
    if center is not None:
        return np.asarray(center, dtype=np.float64).reshape(2)
    if tuple(q_fields) == ("q_row", "q_col"):
        origin_ref = vector.metadata.get("origin_ref")
        if origin_ref is None:
            raise ValueError(
                "Peaks are in detector pixels and carry no 'origin_ref'; pass "
                "center=(row, col) of the direct beam, or correct the origins with "
                "BraggVectors.correct_peak_origins first."
            )
        return np.asarray(origin_ref, dtype=np.float64).reshape(2)
    return np.zeros(2)


def _q_coordinates(vector, q_fields=None, center=(0.0, 0.0)) -> np.ndarray:
    """(N, 2) diffraction coordinates of every flattened row, relative to center."""
    q_fields = _resolve_q_fields(vector, q_fields)
    q = vector.select_fields(*q_fields).numpy().astype(np.float64)
    return q - np.asarray(center, dtype=np.float64)[None, :]


# --------------------------------------------------------------------------- #
# DDF images
# --------------------------------------------------------------------------- #


def ddf_image(
    peaks,
    mask=None,
    intensity_field: str = "intensity",
) -> np.ndarray:
    """Digital dark field image from a subset of the peaks.

    Parameters
    ----------
    peaks : Vector
        Peaks with one cell per probe position.
    mask : array-like of bool, optional
        (N,) selection aligned with the flattened rows of `peaks`, for
        example from aperture_mask or polar_mask. None uses every peak.
    intensity_field : str, default="intensity"
        Field summed at each probe position. Negative values are clipped to 0.

    Returns
    -------
    np.ndarray
        (scan_row, scan_col) image.
    """
    inten = peaks.select_fields(intensity_field).numpy()[:, 0].astype(np.float64).clip(min=0)
    rc = _scan_cells(peaks)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        inten, rc = inten[mask], rc[mask]
    R, C = peaks.shape[:2]
    image = np.zeros((R, C))
    np.add.at(image, (rc[:, 0], rc[:, 1]), inten)
    return image


# --------------------------------------------------------------------------- #
# Virtual apertures
# --------------------------------------------------------------------------- #


def aperture_array(
    g1,
    g2=None,
    mode: str = "array",
    center=(0.0, 0.0),
    shift=(0.0, 0.0),
    n1_range: tuple[int, int] = (-5, 5),
    n2_range: tuple[int, int] = (-5, 5),
    radius_range: tuple[float, float] = (0.0, np.inf),
    shape=None,
    edge: float = 0.0,
) -> np.ndarray:
    """Virtual aperture positions on a lattice of diffraction vectors.

    Each aperture sits at center + (n1 + s1) g1 + (n2 + s2) g2, where (s1, s2)
    is `shift`. We keep the positions whose distance from `center` falls
    inside `radius_range`, which is how the direct beam is usually excluded.
    When the peaks were shifted to a common origin with
    BraggVectors.correct_peak_origins, setting `center` to that origin puts
    the apertures in detector pixels, so they can be drawn over the mean
    pattern or the Bragg vector map.

    Parameters
    ----------
    g1, g2 : array-like of float
        (2,) lattice vectors in the same coordinates as the peaks. g2 is
        required for mode="array", ignored for mode="line", and optional for
        mode="single" (where it is only used with a nonzero s2).
    mode : {"array", "line", "single"}, default="array"
        "array" places a 2D lattice of apertures over n1_range and n2_range,
        "line" places a row of apertures along g1 over n1_range (a systematic
        row, such as a two-beam condition), and "single" places one aperture
        at (s1 g1 + s2 g2).
    center : array-like of float, default=(0, 0)
        (2,) origin of the lattice, normally the direct beam.
    shift : (float, float), default=(0, 0)
        Lattice offset (s1, s2) in multiples of g1 and g2; fractional values
        place apertures between lattice points, for example (0.5, 0.5) for a
        centered superlattice.
    n1_range, n2_range : (int, int), default=(-5, 5)
        Inclusive range of lattice multiples of g1 and g2.
    radius_range : (float, float), default=(0, inf)
        Inner and outer distance from `center` of the apertures kept.
    shape : (int, int), optional
        Detector shape. When given, apertures closer than `edge` to the
        detector boundary are removed, which requires `center` to be in
        detector pixels.
    edge : float, default=0
        Boundary width in pixels, used only with `shape`.

    Returns
    -------
    np.ndarray
        (N, 2) aperture positions.
    """
    if mode == "array" and g2 is None:
        raise ValueError('mode="array" needs both g1 and g2.')
    g1 = np.asarray(g1, dtype=np.float64)
    g2 = np.zeros(2) if g2 is None else np.asarray(g2, dtype=np.float64)
    s1, s2 = (float(v) for v in shift)
    if mode == "single":
        n1 = np.array([0])
        n2 = np.array([0])
    elif mode == "line":
        n1 = np.arange(n1_range[0], n1_range[1] + 1)
        n2 = np.zeros_like(n1)
    elif mode == "array":
        n1, n2 = np.meshgrid(
            np.arange(n1_range[0], n1_range[1] + 1),
            np.arange(n2_range[0], n2_range[1] + 1),
            indexing="ij",
        )
        n1, n2 = n1.ravel(), n2.ravel()
    else:
        raise ValueError(f"mode must be 'array', 'line' or 'single', got {mode!r}.")

    offsets = (n1[:, None] + s1) * g1[None, :] + (n2[:, None] + s2) * g2[None, :]
    r = np.hypot(offsets[:, 0], offsets[:, 1])
    keep = (r >= radius_range[0]) & (r <= radius_range[1])
    positions = offsets + np.asarray(center, dtype=np.float64)[None, :]
    if shape is not None:
        keep &= (
            (positions[:, 0] > edge)
            & (positions[:, 0] < shape[0] - edge)
            & (positions[:, 1] > edge)
            & (positions[:, 1] < shape[1] - edge)
        )
    return positions[keep]


def refine_lattice_vectors(
    peaks,
    g1,
    g2,
    center=None,
    radius: float = 6.0,
    n1_range: tuple[int, int] = (-5, 5),
    n2_range: tuple[int, int] = (-5, 5),
    num_iterations: int = 3,
    q_fields=None,
    intensity_field: str = "intensity",
):
    """Refine one pair of lattice vectors against the peaks of the whole scan.

    This is a single global refinement, used to place virtual apertures; it is
    not the per-position lattice fit of BraggVectors.fit_lattice used for
    strain mapping. For each lattice point n1 g1 + n2 g2 (excluding the
    origin) we take the
    intensity-weighted mean position of all peaks within `radius` of it,
    summed over the scan, then solve for g1 and g2 by weighted least squares
    with each point weighted by its summed intensity. Repeating this a few
    times lets the fit follow lattice points that start near the edge of
    `radius`. The lattice with the most total intensity dominates, so the
    starting vectors should be close to the orientation of interest.

    Parameters
    ----------
    peaks : Vector
        Peaks with one cell per probe position.
    g1, g2 : array-like of float
        (2,) starting lattice vectors, in the units of the peak coordinates.
    center : array-like of float, optional
        (2,) lattice origin, held fixed. None uses (0, 0) for calibrated peaks
        and the stored "origin_ref" for peaks in detector pixels.
    radius : float, default=6.0
        Search radius around each lattice point, in the units of the peak
        coordinates.
    n1_range, n2_range : (int, int), default=(-5, 5)
        Inclusive range of lattice multiples used in the fit.
    num_iterations : int, default=3
        Number of search and fit passes.
    q_fields : (str, str), optional
        Diffraction coordinate fields.
    intensity_field : str, default="intensity"
        Field used as the weight of each peak.

    Returns
    -------
    g1, g2 : np.ndarray
        (2,) refined lattice vectors.
    """
    q_fields = _resolve_q_fields(peaks, q_fields)
    center = _resolve_center(peaks, q_fields, center)
    q = _q_coordinates(peaks, q_fields, center)
    w = peaks.select_fields(intensity_field).numpy()[:, 0].astype(np.float64).clip(min=0)
    n1, n2 = np.meshgrid(
        np.arange(n1_range[0], n1_range[1] + 1),
        np.arange(n2_range[0], n2_range[1] + 1),
        indexing="ij",
    )
    n = np.stack([n1.ravel(), n2.ravel()], axis=1)
    n = n[np.any(n != 0, axis=1)].astype(np.float64)

    g = np.stack([np.asarray(g1, dtype=np.float64), np.asarray(g2, dtype=np.float64)])
    order = np.argsort(q[:, 0])
    q_sorted, w_sorted = q[order], w[order]
    for _ in range(num_iterations):
        targets = n @ g
        means = np.full_like(targets, np.nan)
        weights = np.zeros(targets.shape[0])
        for k, t in enumerate(targets):
            # peaks sorted along the first coordinate, so each search is a slice
            i0, i1 = np.searchsorted(q_sorted[:, 0], [t[0] - radius, t[0] + radius])
            qs, ws = q_sorted[i0:i1], w_sorted[i0:i1]
            m = ((qs - t[None, :]) ** 2).sum(axis=1) <= radius**2
            if ws[m].sum() > 0:
                weights[k] = ws[m].sum()
                means[k] = (qs[m] * ws[m, None]).sum(axis=0) / weights[k]
        ok = weights > 0
        if ok.sum() < 2:
            raise ValueError("Fewer than two lattice points have peaks within radius.")
        sw = np.sqrt(weights[ok])[:, None]
        g, *_ = np.linalg.lstsq(n[ok] * sw, means[ok] * sw, rcond=None)
    return g[0], g[1]


def lattice_distance(
    positions,
    g1,
    g2,
    center=(0.0, 0.0),
) -> np.ndarray:
    """Distance from each position to the nearest point of a 2D lattice.

    Parameters
    ----------
    positions : array-like of float
        (N, 2) diffraction positions, for example from cluster_centers.
    g1, g2 : array-like of float
        (2,) lattice vectors.
    center : array-like of float, default=(0, 0)
        (2,) lattice origin.

    Returns
    -------
    np.ndarray
        (N,) distances, in the units of the positions.
    """
    basis = np.stack([np.asarray(g1, dtype=np.float64), np.asarray(g2, dtype=np.float64)])
    q = np.atleast_2d(np.asarray(positions, dtype=np.float64)) - np.asarray(center)[None, :]
    frac = q @ np.linalg.inv(basis)
    return np.linalg.norm((frac - np.round(frac)) @ basis, axis=1)


def aperture_array_subtract(
    positions,
    positions_remove,
    tol: float = 1.0,
) -> np.ndarray:
    """Remove the apertures that coincide with a second set.

    Subtracting the fundamental lattice from a finer lattice leaves only the
    superlattice positions, for example.

    Parameters
    ----------
    positions : array-like of float
        (N, 2) aperture positions.
    positions_remove : array-like of float
        (M, 2) aperture positions to remove from `positions`.
    tol : float, default=1.0
        Apertures within this distance of any position in
        `positions_remove` are removed.

    Returns
    -------
    np.ndarray
        (N', 2) remaining aperture positions.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    positions_remove = np.atleast_2d(np.asarray(positions_remove, dtype=np.float64))
    if positions_remove.size == 0:
        return positions
    d2 = ((positions[:, None, :] - positions_remove[None, :, :]) ** 2).sum(axis=-1)
    return positions[d2.min(axis=1) > tol**2]


def aperture_mask(
    peaks,
    positions,
    radius: float = 1.0,
    q_fields=None,
) -> np.ndarray:
    """Peaks that fall inside any of a set of virtual apertures.

    Parameters
    ----------
    peaks : Vector
        Peaks with one cell per probe position.
    positions : array-like of float
        (M, 2) aperture positions, in the coordinates of `q_fields`.
    radius : float, default=1.0
        Aperture radius.
    q_fields : (str, str), optional
        Diffraction coordinate fields. Defaults to ("qx", "qy") or
        ("q_row", "q_col"), whichever are present.

    Returns
    -------
    np.ndarray
        (N,) bool mask aligned with the flattened rows of `peaks`. A peak
        inside two overlapping apertures is selected once.
    """
    q = _q_coordinates(peaks, q_fields)
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    mask = np.zeros(q.shape[0], dtype=bool)
    r2 = radius**2
    for p in positions:
        mask |= ((q - p[None, :]) ** 2).sum(axis=1) <= r2
    return mask


def aperture_ddf_image(
    peaks,
    positions,
    radius: float = 1.0,
    q_fields=None,
    intensity_field: str = "intensity",
) -> np.ndarray:
    """Digital dark field image through a set of virtual apertures.

    Equivalent to ddf_image(peaks, aperture_mask(peaks, positions, radius)).

    Parameters
    ----------
    peaks : Vector
        Peaks with one cell per probe position.
    positions : array-like of float
        (M, 2) aperture positions, from aperture_array for example.
    radius : float, default=1.0
        Aperture radius, in the units of the peak coordinates.
    q_fields : (str, str), optional
        Diffraction coordinate fields.
    intensity_field : str, default="intensity"
        Field summed at each probe position.

    Returns
    -------
    np.ndarray
        (scan_row, scan_col) image.
    """
    mask = aperture_mask(peaks, positions, radius=radius, q_fields=q_fields)
    return ddf_image(peaks, mask, intensity_field=intensity_field)


def plot_apertures(
    positions,
    image=None,
    radius: float | None = None,
    positions_removed=None,
    color="tab:green",
    color_removed="tab:red",
    marker_size: float = 60.0,
    figax=None,
    **show_kwargs,
):
    """Aperture positions drawn over a diffraction image.

    Parameters
    ----------
    positions : array-like of float
        (N, 2) aperture positions in (row, col) pixels of `image`.
    image : array-like, optional
        Background image, such as the mean pattern or the Bragg vector map.
    radius : float, optional
        Draw each aperture as a circle of this radius in pixels. None draws
        markers of size `marker_size`.
    positions_removed : array-like of float, optional
        (M, 2) positions drawn in `color_removed`, for example the apertures
        removed by aperture_array_subtract.
    color, color_removed : matplotlib color
        Colors of the kept and removed apertures.
    marker_size : float, default=60
        Marker size in points squared, used when radius is None.
    figax : (Figure, Axes), optional
        Axes to draw into.
    **show_kwargs :
        Passed to quantem.core.visualization.show_2d, for example
        norm={"power": 0.5}.

    Returns
    -------
    fig, ax
    """
    from quantem.core.visualization import show_2d

    if image is not None:
        show_kwargs.setdefault("axsize", (6, 6))
        fig, ax = show_2d(np.asarray(image), figax=figax, **show_kwargs)
    elif figax is not None:
        fig, ax = figax
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")
        ax.invert_yaxis()

    def draw(p, c):
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        if p.size == 0:
            return
        if radius is None:
            ax.scatter(p[:, 1], p[:, 0], s=marker_size, color=c, alpha=0.5, lw=0)
        else:
            ax.add_collection(
                EllipseCollection(
                    widths=2.0 * radius,
                    heights=2.0 * radius,
                    angles=0,
                    units="xy",
                    facecolors=c,
                    alpha=0.4,
                    offsets=p[:, ::-1],
                    offset_transform=ax.transData,
                )
            )

    if positions_removed is not None:
        draw(positions_removed, color_removed)
    draw(positions, color)
    return fig, ax


# --------------------------------------------------------------------------- #
# Polar selection
# --------------------------------------------------------------------------- #


def _polar_coordinates(peaks, q_fields=None, center=None):
    """(qr, qphi) of every flattened row, with qphi = atan2(-q0, q1) in degrees."""
    q_fields = _resolve_q_fields(peaks, q_fields)
    center = _resolve_center(peaks, q_fields, center)
    q = _q_coordinates(peaks, q_fields, center)
    qr = np.hypot(q[:, 0], q[:, 1])
    qphi = np.degrees(np.arctan2(-q[:, 0], q[:, 1]))
    return qr, qphi


def add_polar_fields(
    peaks,
    q_fields=None,
    center=None,
    names: tuple[str, str] = ("qr", "qphi"),
):
    """Copy of the peaks with polar coordinate fields added.

    The radius qr has the units of the diffraction coordinates. The angle
    qphi = atan2(-q0, q1) is in degrees, measured anticlockwise from the +col
    (right) direction as the pattern is displayed with rows increasing
    downward, over the range (-180, 180]. This is py4DSTEM's DDF convention
    (see the module docstring); it differs from the azimuth used in
    quantem.diffraction.calibration.

    Parameters
    ----------
    peaks : Vector
        Peaks with one cell per probe position.
    q_fields : (str, str), optional
        Diffraction coordinate fields.
    center : array-like of float, optional
        (2,) origin of the polar coordinates, normally the direct beam. None
        uses (0, 0) for calibrated peaks and the stored "origin_ref" for peaks
        in detector pixels.
    names : (str, str), default=("qr", "qphi")
        Names of the new fields.

    Returns
    -------
    Vector
    """
    q_fields = _resolve_q_fields(peaks, q_fields)
    qr, qphi = _polar_coordinates(peaks, q_fields, center)
    q_unit = peaks.units[peaks.fields.index(q_fields[0])]
    out = peaks.copy()
    out.add_fields(list(names), values=np.stack([qr, qphi], axis=1), units=[q_unit, "deg"])
    return out


def polar_mask(
    peaks,
    q_radius: float,
    tol: float = 1.0,
    phi_range: tuple[float, float] | None = None,
    q_fields=None,
    center=None,
) -> np.ndarray:
    """Peaks inside a ring, optionally restricted to a range of angles.

    Parameters
    ----------
    peaks : Vector
        Peaks with one cell per probe position.
    q_radius : float
        Ring radius, in the units of the diffraction coordinates.
    tol : float, default=1.0
        Half width of the ring: peaks with |qr - q_radius| <= tol are kept.
    phi_range : (float, float), optional
        Angular range (phi_0, phi_1) in degrees, using the qphi convention of
        add_polar_fields. Peaks with phi_0 <= qphi < phi_1 are kept. When
        phi_0 > phi_1 the range wraps through 180 degrees.
    q_fields : (str, str), optional
        Diffraction coordinate fields.
    center : array-like of float, optional
        (2,) origin of the polar coordinates. None uses (0, 0) for calibrated
        peaks and the stored "origin_ref" for peaks in detector pixels.

    Returns
    -------
    np.ndarray
        (N,) bool mask aligned with the flattened rows of `peaks`.
    """
    qr, qphi = _polar_coordinates(peaks, q_fields, center)
    mask = np.abs(qr - q_radius) <= tol
    if phi_range is not None:
        phi_0, phi_1 = phi_range
        if phi_0 <= phi_1:
            mask &= (qphi >= phi_0) & (qphi < phi_1)
        else:
            mask &= (qphi >= phi_0) | (qphi < phi_1)
    return mask


def radial_ddf_image(
    peaks,
    q_radius: float,
    tol: float = 1.0,
    phi_range: tuple[float, float] | None = None,
    q_fields=None,
    center=None,
    intensity_field: str = "intensity",
) -> np.ndarray:
    """Digital dark field image from a ring of diffraction space.

    Equivalent to ddf_image(peaks, polar_mask(peaks, q_radius, tol,
    phi_range, q_fields, center)).

    Parameters
    ----------
    peaks : Vector
        Peaks with one cell per probe position.
    q_radius : float
        Ring radius, in the units of the diffraction coordinates.
    tol : float, default=1.0
        Half width of the ring, in the same units.
    phi_range : (float, float), optional
        Angular range (phi_0, phi_1) in degrees, as in polar_mask.
    q_fields : (str, str), optional
        Diffraction coordinate fields.
    center : array-like of float, optional
        (2,) origin of the polar coordinates. None uses (0, 0) for calibrated
        peaks and the stored "origin_ref" for peaks in detector pixels.
    intensity_field : str, default="intensity"
        Field summed at each probe position.

    Returns
    -------
    np.ndarray
        (scan_row, scan_col) image.
    """
    mask = polar_mask(
        peaks, q_radius, tol=tol, phi_range=phi_range, q_fields=q_fields, center=center
    )
    return ddf_image(peaks, mask, intensity_field=intensity_field)


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #


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
    label_field : str, default="cluster"
        Field holding the cluster labels; negative labels are ignored.
    intensity_field : str, default="intensity"
        Field used as the weight of each peak when `weighted` is True.
    weighted : bool, default=True
        Weight the center of mass by peak intensity.

    Returns
    -------
    coms : np.ndarray
        (K, 2) centers of mass in scan (row, col) pixels, ordered by cluster
        id. K is the largest label plus one; (0, 2) when no peak is labeled.
    sizes : np.ndarray
        (K,) number of peaks per cluster.
    """
    fields = labeled.fields
    flat = labeled.numpy().astype(np.float64)
    labels = flat[:, fields.index(label_field)].astype(int)
    w = flat[:, fields.index(intensity_field)].clip(min=0) if weighted else None
    rc = _scan_cells(labeled).astype(float)

    n = max(int(labels.max()) + 1, 0) if labels.size else 0
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


def cluster_centers(
    labeled,
    q_fields=None,
    label_field: str = "cluster",
    intensity_field: str = "intensity",
) -> np.ndarray:
    """Intensity-weighted diffraction position of every cluster.

    Parameters
    ----------
    labeled : Vector
        Vector carrying a cluster label field (from cluster_vector).
    q_fields : (str, str), optional
        Diffraction coordinate fields.
    label_field : str, default="cluster"
        Field holding the cluster labels.
    intensity_field : str, default="intensity"
        Field used as the weight of each peak.

    Returns
    -------
    np.ndarray
        (K, 2) mean diffraction positions, ordered by cluster id; (0, 2) when
        no peak is labeled.
    """
    q = _q_coordinates(labeled, q_fields)
    labels = labeled.select_fields(label_field).numpy()[:, 0].astype(int)
    w = labeled.select_fields(intensity_field).numpy()[:, 0].astype(np.float64).clip(min=0)
    m = labels >= 0
    n = int(labels[m].max()) + 1 if m.any() else 0
    if n == 0:
        return np.zeros((0, 2))
    wsum = np.maximum(np.bincount(labels[m], weights=w[m], minlength=n), 1e-12)
    return np.stack(
        [np.bincount(labels[m], weights=w[m] * q[m, k], minlength=n) / wsum for k in range(2)],
        axis=1,
    )


def ddf_images(
    labeled,
    cluster_ids,
    label_field: str = "cluster",
    intensity_field: str = "intensity",
) -> np.ndarray:
    """Digital dark field images: per-cluster summed intensity per position.

    Parameters
    ----------
    labeled : Vector
        Peaks carrying a cluster label field (from cluster_vector).
    cluster_ids : int or array-like of int
        Cluster labels to image, one image each, in this order.
    label_field : str, default="cluster"
        Field holding the cluster labels.
    intensity_field : str, default="intensity"
        Field summed at each probe position. Negative values are clipped to 0.

    Returns
    -------
    np.ndarray
        (len(cluster_ids), scan_row, scan_col) images.
    """
    fields = labeled.fields
    flat = labeled.numpy().astype(np.float64)
    labels = flat[:, fields.index(label_field)].astype(int)
    inten = flat[:, fields.index(intensity_field)].clip(min=0)
    rc = _scan_cells(labeled)
    R, C = labeled.shape[:2]

    cluster_ids = np.atleast_1d(cluster_ids)
    out = np.zeros((len(cluster_ids), R, C))
    for i, k in enumerate(cluster_ids):
        m = labels == k
        np.add.at(out[i], (rc[m, 0], rc[m, 1]), inten[m])
    return out


def assign_grain_labels(
    labeled,
    grain_labels,
    label_field: str = "cluster",
    grain_field: str = "grain_label",
):
    """Copy of the L1-labeled peaks with the L2 grain of every peak added.

    Parameters
    ----------
    labeled : Vector
        Peaks carrying L1 cluster labels (from cluster_vector).
    grain_labels : array-like of int
        (K,) L2 grain label of each L1 cluster, for example
        dbscan(cluster_coms(labeled)[0], ...).
    label_field : str, default="cluster"
        Field holding the L1 cluster labels.
    grain_field : str, default="grain_label"
        Name of the new field.

    Returns
    -------
    Vector
        Copy of `labeled` with `grain_field` added. Peaks outside every L1
        cluster get -2, and peaks whose L1 cluster joined no grain get -1.
    """
    l1 = labeled.select_fields(label_field).numpy()[:, 0].astype(int)
    grain_labels = np.asarray(grain_labels, dtype=int)
    grains = np.where(l1 >= 0, grain_labels[l1.clip(min=0)], -2)

    out = labeled.copy()
    out.add_fields(grain_field, values=grains[:, None], units="index")
    return out


def group_ddf_images(
    images: np.ndarray,
    min_correlation: float = 0.7,
    min_samples: int = 2,
    device: str = "cpu",
) -> np.ndarray:
    """Group DDF images that show the same region of the sample.

    The spots of one grain or lath share the same dark field image, so we
    cluster the images by their cosine similarity. Each image is normalized
    to unit length, and DBSCAN runs with eps = sqrt(2 (1 - min_correlation)),
    so that neighbors have a cosine similarity of at least min_correlation.
    Unlike clustering the centers of mass (cluster_coms), this separates
    grains that extend across the whole field of view, such as a matrix
    phase.

    Parameters
    ----------
    images : np.ndarray
        (K, R, C) DDF images, for example ddf_images(labeled, range(K)).
    min_correlation : float, default=0.7
        Cosine similarity between neighboring images, from 0 to 1.
    min_samples : int, default=2
        DBSCAN min_samples, counting the image itself.
    device : str, default="cpu"
        Torch device for the distance computations.

    Returns
    -------
    np.ndarray
        (K,) group label of each image, -1 for images that joined no group.
        Groups are numbered largest first.
    """
    K = images.shape[0]
    v = images.reshape(K, -1).astype(np.float64)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    eps = float(np.sqrt(2.0 * (1.0 - min_correlation)))
    return dbscan(v, eps=eps, min_samples=min_samples, device=device)


# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #


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
        colors = hsv_to_rgb(np.stack([hues, np.ones(K), np.ones(K)], axis=1))
    colors = np.asarray(colors, dtype=float)

    if normalize == "global":
        norm = np.full(K, max(float(images.max()), 1e-12))
    else:
        norm = np.maximum(images.reshape(K, -1).max(axis=1), 1e-12)
    scaled = (images / norm[:, None, None]) ** gamma
    rgb = np.einsum("krc,kj->rcj", scaled, colors)
    return np.clip(rgb, 0, 1)


def color_wheel(n: int = 256, saturation: float = 1.0) -> np.ndarray:
    """Hue wheel for labeling composite images.

    Parameters
    ----------
    n : int, default=256
        Size of the output image in pixels.
    saturation : float, default=1.0
        Saturation of the hues, from 0 (gray) to 1 (full color).

    Returns
    -------
    np.ndarray
        (n, n, 4) RGBA image in [0, 1], transparent outside the wheel.
    """
    y, x = np.mgrid[-1 : 1 : n * 1j, -1 : 1 : n * 1j]
    r = np.hypot(x, y)
    hue = (np.arctan2(y, x) / (2 * np.pi)) % 1.0
    hsv = np.stack([hue, np.full_like(hue, saturation), np.clip(r, 0, 1)], axis=-1)
    rgba = np.concatenate([hsv_to_rgb(hsv), (r <= 1.0)[..., None].astype(float)], axis=-1)
    return rgba


def plot_cluster_scatter(
    labeled,
    q_fields=None,
    label_field: str = "cluster",
    specific_cluster: int | Sequence[int] | None = None,
    max_clusters: int | None = None,
    show_unclustered: bool = True,
    point_size: float = 2.0,
    alpha: float = 0.2,
    center=None,
    figax=None,
):
    """All peaks in diffraction space, colored by cluster.

    Parameters
    ----------
    labeled : Vector
        Peaks carrying a label field.
    q_fields : (str, str), optional
        Diffraction coordinate fields, drawn as (vertical, horizontal).
    label_field : str, default="cluster"
        Field holding the labels, for example "grain_label" from
        assign_grain_labels.
    specific_cluster : int or sequence of int, optional
        Draw only these clusters, in one color, without the unclustered peaks.
    max_clusters : int, optional
        Draw only the first max_clusters clusters (the largest, since dbscan
        sorts clusters by size).
    show_unclustered : bool, default=True
        Draw the peaks with a negative label in gray.
    point_size : float, default=2.0
        Marker size.
    alpha : float, default=0.2
        Marker opacity. Values well below 1 show the dense regions of large
        datasets.
    center : array-like of float, optional
        (2,) diffraction origin at the middle of the plot. None uses (0, 0)
        for calibrated peaks and the "origin_ref" stored by
        BraggVectors.correct_peak_origins for peaks in detector pixels (or the
        middle of the peak positions when there is none).
    figax : (Figure, Axes), optional
        Axes to draw into.

    Returns
    -------
    fig, ax
    """
    q_fields = _resolve_q_fields(labeled, q_fields)
    q = labeled.select_fields(*q_fields).numpy().astype(np.float64)
    try:
        center = _resolve_center(labeled, q_fields, center)
    except ValueError:
        # pixel peaks without a stored origin: frame the peaks themselves
        center = 0.5 * (q.min(axis=0) + q.max(axis=0)) if q.size else np.zeros(2)
    labels = labeled.select_fields(label_field).numpy()[:, 0].astype(int)
    q0, q1 = q[:, 0], q[:, 1]

    if figax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
    else:
        fig, ax = figax
    q_max = 1.05 * float(np.abs(q - center[None, :]).max()) if q.size else 1.0
    ax.set_xlim(center[1] - q_max, center[1] + q_max)
    ax.set_ylim(center[0] + q_max, center[0] - q_max)
    ax.set_aspect("equal")
    ax.set_xlabel(q_fields[1])
    ax.set_ylabel(q_fields[0])

    if specific_cluster is not None:
        m = np.isin(labels, np.atleast_1d(specific_cluster))
        ax.scatter(q1[m], q0[m], s=point_size, color="C0", lw=0, alpha=alpha)
        return fig, ax

    if show_unclustered:
        m = labels < 0
        ax.scatter(q1[m], q0[m], s=point_size, color="0.85", lw=0)
    n = max(int(labels.max()) + 1, 0) if labels.size else 0
    n_show = n if max_clusters is None else min(n, max_clusters)
    cmap = plt.get_cmap("hsv")
    rng = np.random.default_rng(0)
    hues = rng.permutation(np.linspace(0, 1, n_show, endpoint=False))
    for k in range(n_show):
        m = labels == k
        ax.scatter(q1[m], q0[m], s=point_size, color=cmap(hues[k]), lw=0, alpha=alpha)
    return fig, ax
