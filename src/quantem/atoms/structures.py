"""Ideal nanoparticle structures for comparison with measured atomic models.

The builders return :class:`~quantem.atoms.AtomicModel` objects in physical
units with two channels: ``shell`` (the shell index of each site, 0 at the
center) and ``sector`` (the tetrahedral sector of multiply twinned particles,
or 0 for single crystals).  Distances follow the Mackay convention: sites on
a radial line from the center are spaced by ``bond_length`` and sites within
a shell are 5.15% farther apart, so the 20 tetrahedra of an icosahedron are
slightly distorted FCC.

Structures
----------
``icosahedron``
    Mackay icosahedron with ``num_shells`` shells; 20 FCC tetrahedra sharing
    one center.
``double_icosahedron``
    Two interpenetrating Mackay icosahedra whose centers are one bond apart
    along a common 5-fold axis, related by a mirror through the mid-plane
    between the centers (point group D5h).  Each half keeps the sites on its
    own side of the mid-plane, giving 15 tetrahedral sectors per half plus the
    small shared pentagonal bipyramid between the centers.  This is the
    polyicosahedral motif of the 19-atom double icosahedron, grown shell by
    shell.
``attached_icosahedra``
    Two complete Mackay icosahedra of the same size touching at one vertex on
    a common 5-fold axis, mirror-related (the geometry of oriented attachment
    of two grown particles).
``cuboctahedron``
    FCC cuboctahedron with ``num_shells`` shells (single crystal).
``decahedron``
    Ino decahedron: five FCC tetrahedra sharing a common edge (the 5-fold axis)
    with ``num_shells`` shells and no re-entrant Marks facets.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantem.atoms.atomic_model import AtomicModel

__all__ = [
    "icosahedron",
    "double_icosahedron",
    "cuboctahedron",
    "decahedron",
    "icosahedron_vertices",
]

_MACKAY_TANGENTIAL = 1.0515  # tangential / radial spacing of a Mackay icosahedron


def icosahedron_vertices() -> tuple[NDArray, NDArray]:
    """Unit vertex vectors of an icosahedron and its 20 triangular faces.

    The first vertex points along ``+z`` so that a 5-fold axis is ``z``.

    Returns
    -------
    vertices, faces : ndarray
        ``(12, 3)`` unit vectors and ``(20, 3)`` vertex indices.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            v.append([0.0, s1, s2 * phi])
            v.append([s1, s2 * phi, 0.0])
            v.append([s2 * phi, 0.0, s1])
    v = np.array(v)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    # rotate so that vertex 0 is along +z
    z = v[np.argmax(v[:, 2])]
    axis = np.cross(z, [0, 0, 1.0])
    if np.linalg.norm(axis) > 1e-9:
        axis /= np.linalg.norm(axis)
        ang = np.arccos(np.clip(z @ [0, 0, 1.0], -1, 1))
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rot = np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * k @ k
        v = v @ rot.T
    order = np.argsort(-v[:, 2], kind="stable")
    v = v[order]
    # faces: triples of mutually adjacent vertices (edge length = 1.0515)
    edge = np.linalg.norm(v[0] - v[1])
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d = [np.linalg.norm(v[a] - v[b]) for a, b in ((i, j), (j, k), (i, k))]
                if all(abs(x - edge) < 1e-6 for x in d):
                    faces.append((i, j, k))
    return v, np.array(faces)


def _sector_sites(num_shells: int) -> tuple[NDArray, NDArray, NDArray]:
    """Sites of all 20 tetrahedral sectors of a Mackay icosahedron (unit radial spacing)."""
    v, faces = icosahedron_vertices()
    pts, shell, sector = [np.zeros((1, 3))], [np.zeros(1, int)], [np.full(1, -1)]
    for f, (i, j, k) in enumerate(faces):
        for n in range(1, num_shells + 1):
            for a in range(n + 1):
                for b in range(n + 1 - a):
                    c = n - a - b
                    pts.append((a * v[i] + b * v[j] + c * v[k])[None, :])
                    shell.append(np.array([n]))
                    sector.append(np.array([f]))
    return np.vstack(pts), np.concatenate(shell), np.concatenate(sector)


def _set_sector(model: AtomicModel, sector: NDArray) -> None:
    """Store the sector labels as a categorical channel (``-1`` = center site)."""
    sector = np.asarray(sector, dtype=int)
    num = int(sector.max()) + 1 if sector.size else 1
    model.set_channel("sector", sector, categories=[str(i) for i in range(num)])


def _dedupe(xyz: NDArray, *channels: NDArray, tol: float = 1e-6) -> tuple[NDArray, list[NDArray]]:
    key = np.round(xyz / tol).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    first = np.sort(first)
    return xyz[first], [c[first] for c in channels]


def icosahedron(
    num_shells: int, bond_length: float = 1.0, name: str = "icosahedron"
) -> AtomicModel:
    """Mackay icosahedron.

    Parameters
    ----------
    num_shells : int
        Number of shells around the central site (``K``); the particle has
        ``(10 K^3 + 15 K^2 + 11 K + 3) / 3`` sites.
    bond_length : float
        Radial nearest-neighbor spacing in physical units.
    name : str
        Model name.
    """
    xyz, shell, sector = _sector_sites(num_shells)
    xyz, (shell, sector) = _dedupe(xyz, shell, sector)
    model = AtomicModel.from_array(xyz * bond_length, units="A", name=name)
    model.set_channel("shell", shell)
    _set_sector(model, sector)
    return model


def double_icosahedron(
    num_shells: int,
    bond_length: float = 1.0,
    separation: int = 1,
    name: str = "double icosahedron",
) -> AtomicModel:
    """Two Mackay icosahedra sharing a 5-fold axis, mirror-related (D5h).

    The lower icosahedron is centered at ``(0, 0, -separation / 2)`` (in bond
    lengths) and the upper one at ``(0, 0, +separation / 2)``; each half
    contributes the sites on its own side of the mid-plane ``z = 0``, and
    sites within half a bond of their mirror image are merged onto the plane.
    ``separation = 1`` is the polyicosahedral double icosahedron (19 sites for
    one shell); larger odd separations place the second center on the 5-fold
    axis of the first icosahedron at the position of one of its vertices, so
    that the two halves are twins across the mid-plane.  Sectors ``0-19``
    belong to the lower half and ``20-39`` to the upper half.

    Parameters
    ----------
    num_shells : int
        Shells of each icosahedron.
    bond_length : float
        Radial nearest-neighbor spacing.
    separation : int
        Distance between the two centers in bond lengths.
    name : str
        Model name.
    """
    xyz, shell, sector = _sector_sites(num_shells)
    lower = xyz + np.array([0.0, 0.0, -0.5 * separation])
    keep_lo = lower[:, 2] <= 1e-6
    upper = lower * np.array([1.0, 1.0, -1.0])  # mirror through z = 0
    keep_up = upper[:, 2] >= -1e-6
    pts = np.vstack([lower[keep_lo], upper[keep_up]])
    sh = np.concatenate([shell[keep_lo], shell[keep_up]])
    sec = np.concatenate([sector[keep_lo], sector[keep_up] + 20])
    pts, (sh, sec) = _dedupe(pts, sh, sec)
    model = AtomicModel.from_array(pts * bond_length, units="A", name=name)
    model.set_channel("shell", sh)
    _set_sector(model, sec)
    # Mirror images of sites just below the mid-plane lie 0.11, 0.32, 0.53 or
    # 0.74 bonds from each other (the Mackay heights i + 0.447 j do not fall on
    # the plane); merging every pair closer than 0.75 bonds onto the plane gives
    # the shared mid-plane layer of the D5h structure (19 sites for one shell
    # at separation 1) with all remaining distances >= 1 bond.
    model.merge_close_sites(min_distance=0.75 * bond_length)
    model.metadata["centers"] = [
        [0.0, 0.0, -0.5 * separation * bond_length],
        [0.0, 0.0, 0.5 * separation * bond_length],
    ]
    model.metadata["separation"] = int(separation)
    return model


def cuboctahedron(
    num_shells: int, bond_length: float = 1.0, name: str = "cuboctahedron"
) -> AtomicModel:
    """FCC cuboctahedron with ``num_shells`` shells around a central site."""
    a = bond_length * np.sqrt(2.0)
    n = num_shells
    rng = np.arange(-n, n + 1)
    ijk = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    xyz = ((ijk[:, None, :] + basis[None]).reshape(-1, 3)) * a
    # shell k is bounded by the cube |x_i| <= k a / 2 and the octahedron |x|+|y|+|z| <= k a
    ax = np.abs(xyz) / a
    shell = np.rint(np.maximum(2.0 * ax.max(1), ax.sum(1))).astype(int)
    keep = shell <= n
    model = AtomicModel.from_array(xyz[keep], units="A", name=name)
    model.set_channel("shell", shell[keep])
    _set_sector(model, np.zeros(keep.sum(), dtype=int))
    return model


def decahedron(num_shells: int, bond_length: float = 1.0, name: str = "decahedron") -> AtomicModel:
    """Ino decahedron: five FCC tetrahedra sharing the 5-fold axis ``z``.

    One sector is cut from an FCC crystal as the 70.53 degree wedge between a
    (111) and a (11-1) plane meeting along [1-10]; its azimuth about the axis
    is stretched to 72 degrees and five copies are placed around the axis.
    ``num_shells`` counts the (111) layers from the axis to the surface, and
    sites on the shared twin planes and the axis appear once.

    Parameters
    ----------
    num_shells : int
        Number of (111) layers per sector.
    bond_length : float
        Nearest-neighbor spacing before the 72/70.53 azimuthal stretch.
    name : str
        Model name.
    """
    a = np.sqrt(2.0)  # cubic lattice constant for unit NN spacing
    n = num_shells + 2
    rng = np.arange(-n, n + 1)
    ijk = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    fcc = ((ijk[:, None, :] + basis[None]).reshape(-1, 3)) * a
    n1 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    n2 = np.array([1.0, 1.0, -1.0]) / np.sqrt(3)
    d111 = a / np.sqrt(3)
    l1 = fcc @ n1 / d111
    l2 = -(fcc @ n2) / d111
    eps = 1e-6
    layer = np.rint(l1 + l2).astype(int)
    u = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    keep = (
        (l1 >= -eps) & (l2 >= -eps) & (layer <= num_shells) & (np.abs(fcc @ u) <= num_shells + eps)
    )
    pts = fcc[keep]
    layer = layer[keep]
    # cylindrical coordinates about the wedge axis u = [1,-1,0]; azimuth from the bisector
    bis = n1 - n2
    bis /= np.linalg.norm(bis)  # bisector of the wedge, perpendicular to u
    e2 = np.cross(u, bis)
    z = pts @ u
    x = pts @ bis
    y = pts @ e2
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x) * (72.0 / 70.528779)
    out, sh, sec = [], [], []
    for s_idx in range(5):
        ang = phi + np.radians(72.0 * s_idx)
        out.append(np.stack([rho * np.cos(ang), rho * np.sin(ang), z], 1))
        sh.append(layer)
        sec.append(np.full(layer.size, s_idx))
    xyz = np.vstack(out) * bond_length
    xyz, (sh, sec) = _dedupe(xyz, np.concatenate(sh), np.concatenate(sec), tol=1e-3 * bond_length)
    model = AtomicModel.from_array(xyz, units="A", name=name)
    model.set_channel("shell", sh)
    _set_sector(model, sec)
    return model


def attached_icosahedra(
    num_shells: int, bond_length: float = 1.0, name: str = "attached icosahedra"
) -> AtomicModel:
    """Two complete Mackay icosahedra sharing one vertex site on a 5-fold axis.

    The lower icosahedron is centered at ``z = -num_shells`` bonds and the
    upper one, its mirror image, at ``z = +num_shells``; the vertex at the
    origin is shared.  Sectors ``0-19`` and ``20-39`` label the two particles.
    """
    xyz, shell, sector = _sector_sites(num_shells)
    lower = xyz + np.array([0.0, 0.0, -float(num_shells)])
    upper = lower * np.array([1.0, 1.0, -1.0])
    pts = np.vstack([lower, upper])
    sh = np.concatenate([shell, shell])
    sec = np.concatenate([sector, sector + 20])
    pts, (sh, sec) = _dedupe(pts, sh, sec, tol=1e-6)
    model = AtomicModel.from_array(pts * bond_length, units="A", name=name)
    model.set_channel("shell", sh)
    _set_sector(model, sec)
    model.metadata["centers"] = [
        [0.0, 0.0, -num_shells * bond_length],
        [0.0, 0.0, num_shells * bond_length],
    ]
    return model


def growth_steps(
    model: AtomicModel, origin: NDArray | None = None, name: str = "growth_step"
) -> NDArray:
    """Growth step of every site as its distance from a nucleus in bond lengths.

    Sites are added in order of distance from ``origin`` (default: the first
    entry of ``model.metadata["centers"]``, or the model center), rounded up
    to whole bond lengths, and the result is stored as channel ``name`` so
    that the viewer's growth slider replays an inside-out growth sequence.

    Returns
    -------
    ndarray
        ``(N,)`` integer growth steps.
    """
    if origin is None:
        centers = model.metadata.get("centers")
        origin = np.asarray(centers[0]) if centers else model.center
    r = np.linalg.norm(model.positions - np.asarray(origin, dtype=float)[None, :], axis=1)
    steps = np.ceil(r / model.bond_length - 1e-6).astype(int)
    model.set_channel(name, steps)
    return steps
