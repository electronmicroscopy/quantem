"""Polyhedral templates for local structure classification of 3D atomic models.

A template is the set of neighbor vectors around a reference site in an ideal
crystal, expressed in units of the nearest-neighbor (NN) bond length so that the
first shell has radius 1.  Templates are generated from crystal definitions
(lattice vectors + basis) rather than typed by hand, so any number of neighbor
shells can be requested.

Built-in structures
-------------------
``fcc``
    Face-centered cubic; 12 NN (cuboctahedron), 6 second neighbors at sqrt(2).
``hcp``
    Hexagonal close-packed (ideal c/a); 12 NN (anticuboctahedron).  Atoms on a
    coherent {111} twin plane in an FCC crystal have this local environment,
    so the ``hcp`` score is also the twin-boundary detector for FCC particles.
``bcc``
    Body-centered cubic; 8 NN plus 6 second neighbors at 2/sqrt(3) = 1.155.
    Two shells are used by default because the second shell is so close.
``sc``
    Simple cubic; 6 NN.
``diamond`` (alias ``zincblende``)
    Cubic diamond / zincblende site; 4 NN plus 12 second neighbors at 1.633.
    Two shells are used by default since 4 NN cannot distinguish it from
    ``wurtzite``.
``wurtzite``
    Hexagonal diamond (ideal u = 3/8, c/a = 1.633); 4 NN plus 12 second
    neighbors.  Differs from ``diamond`` only in the second shell (ABAB vs ABC).
``ico``
    Icosahedral (13-atom Mackay) site; 12 NN at the vertices of an icosahedron.

Examples
--------
>>> from quantem.atoms.templates import get_template
>>> fcc = get_template("fcc")
>>> fcc.num_neighbors, fcc.shells
(12, (1.0,))
>>> get_template("fcc", num_shells=2).num_neighbors
18
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PolyhedralTemplate",
    "TEMPLATE_NAMES",
    "get_template",
    "template_from_crystal",
    "template_symmetry_rotations",
]

_SHELL_TOL = 1e-3


@dataclass(frozen=True)
class PolyhedralTemplate:
    """Neighbor vectors of an ideal local environment, in NN bond-length units.

    Parameters
    ----------
    name : str
        Structure name, e.g. ``"fcc"``.
    vectors : ndarray
        ``(M, 3)`` neighbor vectors sorted by radius; first shell radius is 1.
    shells : tuple of float
        Radius of each neighbor shell included in ``vectors``.
    shell_counts : tuple of int
        Number of neighbors in each included shell.
    symmetry : ndarray
        ``(S, 3, 3)`` proper rotations that map the template onto itself.
        Used to reduce orientations to a fundamental zone when computing
        misorientations between neighboring sites.
    """

    name: str
    vectors: NDArray[np.floating]
    shells: tuple[float, ...]
    shell_counts: tuple[int, ...]
    symmetry: NDArray[np.floating] = field(repr=False, default=None)  # type: ignore[assignment]

    @property
    def num_neighbors(self) -> int:
        """Total number of neighbor vectors in the template."""
        return int(self.vectors.shape[0])

    @property
    def max_radius(self) -> float:
        """Radius of the outermost included shell (NN units)."""
        return float(self.shells[-1])

    @property
    def num_symmetry(self) -> int:
        """Number of proper symmetry rotations of the template."""
        return 0 if self.symmetry is None else int(self.symmetry.shape[0])

    def __repr__(self) -> str:
        shells = ", ".join(f"{r:.3f}x{n}" for r, n in zip(self.shells, self.shell_counts))
        return (
            f"PolyhedralTemplate(name={self.name!r}, num_neighbors={self.num_neighbors}, "
            f"shells=[{shells}], num_symmetry={self.num_symmetry})"
        )


# --------------------------------------------------------------------------- #
# Crystal definitions (lattice vectors as rows; basis in fractional coordinates)
# --------------------------------------------------------------------------- #
def _hexagonal_cell(a: float, c: float) -> NDArray:
    return np.array([[a, 0.0, 0.0], [-a / 2, a * np.sqrt(3) / 2, 0.0], [0.0, 0.0, c]])


def _crystal_definition(name: str) -> tuple[NDArray, NDArray, int, int]:
    """Return (cell, basis, site_index, default_shells) with NN distance = 1."""
    if name == "fcc":
        a = np.sqrt(2.0)
        cell = np.eye(3) * a
        basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        return cell, basis, 0, 1
    if name == "bcc":
        a = 2.0 / np.sqrt(3.0)
        cell = np.eye(3) * a
        basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        return cell, basis, 0, 2
    if name == "sc":
        return np.eye(3), np.zeros((1, 3)), 0, 1
    if name == "hcp":
        a = 1.0
        c = a * np.sqrt(8.0 / 3.0)
        cell = _hexagonal_cell(a, c)
        basis = np.array([[0, 0, 0], [1 / 3, 2 / 3, 0.5]])
        return cell, basis, 0, 1
    if name == "diamond":
        a = 4.0 / np.sqrt(3.0)
        cell = np.eye(3) * a
        fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        basis = np.vstack([fcc, fcc + 0.25])
        return cell, basis, 0, 2
    if name == "wurtzite":
        u = 3.0 / 8.0
        c_over_a = np.sqrt(8.0 / 3.0)
        a = 1.0 / (u * c_over_a)  # bond length u*c = 1
        c = a * c_over_a
        cell = _hexagonal_cell(a, c)
        basis = np.array([[0, 0, 0], [1 / 3, 2 / 3, 0.5], [0, 0, u], [1 / 3, 2 / 3, 0.5 + u]])
        return cell, basis, 0, 2
    raise ValueError(f"Unknown crystal structure {name!r}. Choose from {TEMPLATE_NAMES}.")


def template_from_crystal(
    cell: NDArray,
    basis: NDArray,
    site_index: int = 0,
    num_shells: int = 1,
    name: str = "custom",
    max_search: int = 3,
) -> PolyhedralTemplate:
    """Build a template from a crystal definition by collecting neighbor shells.

    Parameters
    ----------
    cell : ndarray
        ``(3, 3)`` lattice vectors as rows.
    basis : ndarray
        ``(B, 3)`` fractional coordinates of the basis atoms.
    site_index : int
        Which basis atom is the reference site.
    num_shells : int
        Number of neighbor shells to include.
    name : str
        Name stored on the template.
    max_search : int
        Half-width (in cells) of the supercell searched for neighbors.

    Returns
    -------
    PolyhedralTemplate
        Neighbor vectors scaled so the first shell has radius 1.
    """
    cell = np.asarray(cell, dtype=float)
    basis = np.asarray(basis, dtype=float)
    rng = np.arange(-max_search, max_search + 1)
    ijk = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    frac = (ijk[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    cart = frac @ cell
    origin = basis[site_index] @ cell
    d = cart - origin[None, :]
    r = np.linalg.norm(d, axis=1)
    keep = r > 1e-8
    d, r = d[keep], r[keep]
    order = np.argsort(r)
    d, r = d[order], r[order]

    # group into shells
    shells: list[float] = []
    counts: list[int] = []
    for radius in r:
        if shells and abs(radius - shells[-1]) < _SHELL_TOL * max(1.0, radius):
            counts[-1] += 1
        else:
            shells.append(float(radius))
            counts.append(1)
    if num_shells > len(shells):
        raise ValueError(f"Only {len(shells)} shells found; increase max_search.")
    n_keep = int(sum(counts[:num_shells]))
    r_nn = shells[0]
    vectors = d[:n_keep] / r_nn
    shells_out = tuple(s / r_nn for s in shells[:num_shells])
    template = PolyhedralTemplate(
        name=name,
        vectors=np.ascontiguousarray(vectors),
        shells=shells_out,
        shell_counts=tuple(counts[:num_shells]),
    )
    return _with_symmetry(template)


def _icosahedron_template(num_shells: int) -> PolyhedralTemplate:
    if num_shells != 1:
        raise ValueError("The 'ico' template only defines a single neighbor shell.")
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    pts = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            pts.append([0.0, s1 * 1.0, s2 * phi])
            pts.append([s1 * 1.0, s2 * phi, 0.0])
            pts.append([s2 * phi, 0.0, s1 * 1.0])
    vectors = np.array(pts)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    template = PolyhedralTemplate(name="ico", vectors=vectors, shells=(1.0,), shell_counts=(12,))
    return _with_symmetry(template)


TEMPLATE_NAMES: tuple[str, ...] = (
    "fcc",
    "hcp",
    "bcc",
    "sc",
    "diamond",
    "zincblende",
    "wurtzite",
    "ico",
)


def get_template(name: str, num_shells: int | None = None) -> PolyhedralTemplate:
    """Return a built-in polyhedral template.

    Parameters
    ----------
    name : str
        One of ``TEMPLATE_NAMES``.  ``"zincblende"`` is an alias of ``"diamond"``.
    num_shells : int, optional
        Number of neighbor shells.  Defaults to 1 for close-packed structures
        and 2 for ``bcc``, ``diamond`` and ``wurtzite``.

    Returns
    -------
    PolyhedralTemplate
    """
    key = name.lower()
    if key == "zincblende":
        key = "diamond"
    if key == "ico":
        return _icosahedron_template(1 if num_shells is None else num_shells)
    cell, basis, site_index, default_shells = _crystal_definition(key)
    return template_from_crystal(
        cell,
        basis,
        site_index=site_index,
        num_shells=default_shells if num_shells is None else int(num_shells),
        name=key,
    )


# --------------------------------------------------------------------------- #
# Symmetry rotations of a template (numerical, template-agnostic)
# --------------------------------------------------------------------------- #
def _frame(u: NDArray, v: NDArray) -> NDArray:
    """Right-handed orthonormal frame with e1 = u/|u| and e2 in the (u, v) plane."""
    e1 = u / np.linalg.norm(u)
    v2 = v - e1 * (v @ e1)
    e2 = v2 / np.linalg.norm(v2)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=1)  # columns


def template_symmetry_rotations(vectors: NDArray, tol: float = 1e-3) -> NDArray:
    """Find all proper rotations mapping a set of vectors onto itself.

    Candidate rotations are built by mapping the frame spanned by two fixed
    template vectors onto the frame spanned by every ordered pair of template
    vectors with the same lengths and angle, then verified against the full set.

    Parameters
    ----------
    vectors : ndarray
        ``(M, 3)`` template vectors.
    tol : float
        Matching tolerance in the same units as ``vectors``.

    Returns
    -------
    ndarray
        ``(S, 3, 3)`` unique proper rotation matrices, identity first.
    """
    v = np.asarray(vectors, dtype=float)
    m = v.shape[0]
    r = np.linalg.norm(v, axis=1)
    # reference pair: vector 0 and its nearest non-collinear partner
    i0 = 0
    cosines = (v @ v[i0]) / (r * r[i0])
    non_collinear = np.where(np.abs(cosines) < 1 - 1e-6)[0]
    j0 = non_collinear[np.argmax(cosines[non_collinear])]
    ref_frame = _frame(v[i0], v[j0])
    ref_angle = np.arccos(np.clip(cosines[j0], -1, 1))

    rotations: list[NDArray] = [np.eye(3)]
    for i in range(m):
        if abs(r[i] - r[i0]) > tol:
            continue
        for j in range(m):
            if i == j or abs(r[j] - r[j0]) > tol:
                continue
            ang = np.arccos(np.clip((v[i] @ v[j]) / (r[i] * r[j]), -1, 1))
            if abs(ang - ref_angle) > 1e-4:
                continue
            rot = _frame(v[i], v[j]) @ ref_frame.T
            if np.linalg.det(rot) < 0:
                continue
            mapped = v @ rot.T
            dist = np.linalg.norm(mapped[:, None, :] - v[None, :, :], axis=2)
            if np.all(dist.min(axis=1) < tol):
                if not any(np.allclose(rot, q, atol=1e-6) for q in rotations):
                    rotations.append(rot)
    return np.stack(rotations, axis=0)


def _with_symmetry(template: PolyhedralTemplate) -> PolyhedralTemplate:
    sym = template_symmetry_rotations(template.vectors)
    return PolyhedralTemplate(
        name=template.name,
        vectors=template.vectors,
        shells=template.shells,
        shell_counts=template.shell_counts,
        symmetry=sym,
    )
