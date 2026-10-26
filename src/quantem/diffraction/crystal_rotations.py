"""Crystal orientations: zone axes, point-group symmetry and fundamental wedges.

Quaternions follow the convention of :mod:`quantem.core.utils.rotations`:
scalar-first (w, x, y, z), specimen (crystal) to lab, beam along -z. The
zone axis is the crystal direction pointing from the specimen back toward
the source, lab +z, in crystal Cartesian coordinates; it is the third row of
R(q)::

    zone_axis = R(q).T @ [0, 0, 1]
"""

from __future__ import annotations

import numpy as np
import torch

from quantem.core.utils.rotations import (
    qmult,
    qnormalize,
    qrotate,
    quat_from_axis_angle,
    quat_from_matrix,
    quat_to_axis_angle,
    quat_to_matrix,
    slerp,
)


def quat_from_zone_axis(
    zone_axis: torch.Tensor,
    in_plane_deg: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """Orientation with the given crystal direction along the beam.

    Parameters
    ----------
    zone_axis : torch.Tensor
        Crystal-frame Cartesian direction(s) (..., 3) to place along lab +z,
        pointing toward the source; need not be normalized.
    in_plane_deg : torch.Tensor | float, default=0.0
        Additional rotation about lab z, degrees.

    Returns
    -------
    torch.Tensor
        Unit quaternions (..., 4), crystal to lab, such that
        quat_to_matrix(q).T @ [0, 0, 1] == zone_axis.
    """
    v = zone_axis / torch.linalg.norm(zone_axis, dim=-1, keepdim=True)
    zhat = torch.zeros_like(v)
    zhat[..., 2] = 1.0
    # minimal rotation taking zone axis to z
    axis = torch.cross(v, zhat, dim=-1)
    sin_t = torch.linalg.norm(axis, dim=-1)
    cos_t = v[..., 2]
    angle = torch.atan2(sin_t, cos_t)
    # antiparallel / parallel cases: rotate about x
    fallback = torch.zeros_like(v)
    fallback[..., 0] = 1.0
    axis = torch.where(sin_t[..., None] < 1e-12, fallback, axis)
    q_tilt = quat_from_axis_angle(axis, angle)
    in_plane = torch.deg2rad(
        torch.as_tensor(in_plane_deg, dtype=v.dtype, device=v.device)
    ).broadcast_to(v.shape[:-1])
    z3 = torch.zeros_like(in_plane)
    q_spin = torch.stack((torch.cos(in_plane / 2), z3, z3, torch.sin(in_plane / 2)), dim=-1)
    return qnormalize(qmult(q_spin, q_tilt))


def zone_axis_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Zone axis of orientations, in crystal Cartesian coordinates.

    Parameters
    ----------
    q : torch.Tensor
        Unit scalar-first quaternions (..., 4), crystal to lab.

    Returns
    -------
    torch.Tensor
        Unit crystal directions (..., 3) along lab +z, toward the source:
        the third row of R(q).
    """
    return quat_to_matrix(q)[..., 2, :]


def sample_zone_axes(
    corners: torch.Tensor,
    step_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Isotropic SLERP grid of unit zone-axis vectors inside a spherical triangle.

    Rows run from corner 0 toward the opposite edge; the number of points
    in each row follows the row's own arc length, so the spacing is close
    to `step_deg` in every direction whatever the apex angle of the wedge
    (a 30 degree hexagonal wedge and a 120 degree trigonal wedge get the
    same density).

    Parameters
    ----------
    corners : torch.Tensor
        (3, 3) rows are the Cartesian corner directions of the fundamental
        zone-axis wedge, e.g. [001], [011], [111] for m-3m.
    step_deg : float
        Angular step between neighboring zone axes, degrees.

    Returns
    -------
    vectors : torch.Tensor
        (N, 3) unit vectors sampling the wedge.
    inds : torch.Tensor
        (N, 2) integer (row, col) indices in the triangular grid.
    """
    c = corners / torch.linalg.norm(corners, dim=-1, keepdim=True)
    a01 = torch.rad2deg(torch.acos((c[0] * c[1]).sum().clamp(-1, 1)))
    a02 = torch.rad2deg(torch.acos((c[0] * c[2]).sum().clamp(-1, 1)))
    n_steps = int(torch.ceil(torch.maximum(a01, a02) / step_deg).item())
    n_steps = max(n_steps, 1)

    vecs, inds = [], []
    for i in range(n_steps + 1):
        t = torch.tensor(i / n_steps, dtype=c.dtype, device=c.device)
        pv = slerp(c[0], c[1], t)
        pw = slerp(c[0], c[2], t)
        if i == 0:
            vecs.append(pv[None])
            inds.append(torch.tensor([[0, 0]]))
            continue
        arc = torch.rad2deg(torch.acos((pv * pw).sum().clamp(-1, 1)))
        n_i = max(1, int(torch.ceil(arc / step_deg).item()))
        s = torch.linspace(0, 1, n_i + 1, dtype=c.dtype, device=c.device)
        row = slerp(pv.expand(n_i + 1, 3), pw.expand(n_i + 1, 3), s)
        row = row / torch.linalg.norm(row, dim=-1, keepdim=True)
        vecs.append(row)
        inds.append(torch.stack((torch.full((n_i + 1,), i), torch.arange(n_i + 1)), dim=-1))
    return torch.cat(vecs), torch.cat(inds).to(torch.long)


def symmetry_axes(sym_quats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Distinct rotation axes of a proper point group and their orders.

    Parameters
    ----------
    sym_quats : torch.Tensor
        Proper rotation quaternions (S, 4) of the group, crystal Cartesian
        frame.

    Returns
    -------
    axes : torch.Tensor
        Unit axes (A, 3), each listed once with the sign that makes it
        point into the upper hemisphere (+z, then +y, then +x on ties).
    orders : torch.Tensor
        Long (A,), the highest rotation order about each axis (a 4-fold
        axis is listed as order 4, not also as 2).
    """
    axis, angle = quat_to_axis_angle(sym_quats)
    keep = angle > 1e-6
    axis, angle = axis[keep], angle[keep]
    order = torch.round(2 * np.pi / angle).to(torch.long)
    # orient each axis to a canonical hemisphere so +-axis merge
    sign = torch.sign(axis[:, 2] + 1e-3 * axis[:, 1] + 1e-6 * axis[:, 0])
    sign[sign == 0] = 1
    axis = axis * sign[:, None]
    out_axes, out_orders = [], []
    for a, n in zip(axis, order):
        for k, b in enumerate(out_axes):
            if float(torch.abs(a @ b)) > 1 - 1e-6:
                out_orders[k] = max(out_orders[k], int(n))
                break
        else:
            out_axes.append(a)
            out_orders.append(int(n))
    if not out_axes:
        return torch.zeros((0, 3), dtype=torch.float64), torch.zeros(0, dtype=torch.long)
    return torch.stack(out_axes), torch.tensor(out_orders, dtype=torch.long)


def _rotate_about(v: torch.Tensor, axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    q = quat_from_axis_angle(axis, torch.tensor(angle_rad, dtype=v.dtype))
    return qrotate(q, v)


def _closest(cands: torch.Tensor, prefer: torch.Tensor) -> torch.Tensor:
    """The candidate (sign chosen freely) closest to the preferred direction,
    with deterministic tie-breaking toward +z, then +y, then +x."""
    signed = torch.cat([cands, -cands])
    key = signed @ prefer + 1e-3 * signed[:, 2] + 1e-6 * signed[:, 1] + 1e-9 * signed[:, 0]
    return signed[int(torch.argmax(key))]


def sample_zone_axis_cap(
    axis: torch.Tensor,
    half_angle_deg: float,
    step_deg: float,
) -> torch.Tensor:
    """Near-uniform sampling of a spherical cap of directions, (N, 3).

    The fiber-texture case: the zone axis is known to lie within
    `half_angle_deg` of `axis` (a fiber axis normal to a 2D material, or a
    textured film), and only that cap needs a library. A half angle of zero
    returns the axis itself, so the match is over the in-plane angle alone.

    Points are placed on a Fibonacci spiral restricted to the cap, which
    gives an equal-area covering; the count follows the cap area divided by
    `step_deg` squared.

    Parameters
    ----------
    axis : torch.Tensor
        Center of the cap (3,), crystal Cartesian; need not be normalized.
    half_angle_deg : float
        Angular radius of the cap, degrees.
    step_deg : float
        Approximate spacing between neighboring directions, degrees.

    Returns
    -------
    torch.Tensor
        Unit directions (N, 3), float64, all within `half_angle_deg` of
        `axis`; (1, 3) holding `axis` itself when `half_angle_deg` is zero.
    """
    axis = torch.as_tensor(axis, dtype=torch.float64)
    axis = axis / torch.linalg.norm(axis).clamp_min(1e-12)
    half = np.deg2rad(float(half_angle_deg))
    if half <= 0:
        return axis[None, :]
    step = np.deg2rad(float(step_deg))
    n = max(1, int(np.ceil(2 * np.pi * (1 - np.cos(half)) / step**2)))
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1.0 - (1.0 - np.cos(half)) * i / n
    r = torch.sqrt((1 - z**2).clamp_min(0))
    phi = i * (np.pi * (3 - np.sqrt(5)))
    pts = torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=1)
    # rotate the +z pole onto `axis`
    zhat = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    v = torch.linalg.cross(zhat, axis)
    c = float(torch.dot(zhat, axis))
    if float(torch.linalg.norm(v)) < 1e-12:
        return pts if c > 0 else -pts
    vx = torch.tensor(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=torch.float64
    )
    R = torch.eye(3, dtype=torch.float64) + vx + vx @ vx / (1 + c)
    return pts @ R.T


def fundamental_zone_axis_wedge(sym_quats: torch.Tensor) -> torch.Tensor | None:
    """Fundamental zone-axis wedge of a Laue group from its proper rotations.

    Zone axes are directions modulo inversion (Friedel), so the wedge is a
    fundamental domain of the Laue group on the projective hemisphere,
    built from the actual symmetry axes in the crystal's Cartesian frame
    rather than from a table keyed on the Laue class. This makes it correct
    for every setting: for -3m the wedge is bounded by the mirror planes
    (perpendicular to the in-plane 2-fold axes), which is 30 degrees away
    from a wedge bounded by the 2-fold axes themselves; for a cell in a
    non-standard Cartesian setting the corners follow the axes wherever
    they point.

    Parameters
    ----------
    sym_quats : torch.Tensor
        Proper rotation quaternions (S, 4) of the group, crystal Cartesian
        frame.

    Returns
    -------
    torch.Tensor | None
        Corner directions as rows (3, 3), or None for Laue classes -1 and
        2/m whose fundamental domain is not a spherical triangle (sample the
        hemisphere instead).
    """
    axes, orders = symmetry_axes(sym_quats)
    n_ops = sym_quats.shape[0]
    x, y, z = (torch.eye(3, dtype=torch.float64)[i] for i in range(3))
    if n_ops <= 2:
        return None

    three = axes[orders == 3]
    if three.shape[0] >= 4:  # cubic
        four = axes[orders == 4]
        two = axes[orders == 2]
        if four.shape[0] > 0:  # m-3m: 4-fold, <110> 2-fold, 3-fold
            c0 = _closest(four, z)
            c2 = _closest(three, c0)
            c1 = _closest(two, c0 + c2)
        else:  # m-3: two cubic 2-fold axes and the 3-fold between them
            c0 = _closest(two, z)
            rest = two[torch.abs(two @ c0) < 0.5]
            c1 = _closest(rest, x)
            c2 = _closest(three, c0 + c1)
        return torch.stack([c0, c1, c2])

    n_max = int(orders.max())
    if n_max in (3, 4, 6):  # uniaxial classes
        c0 = _closest(axes[orders == n_max], z)
        in_plane = axes[(orders == 2) & (torch.abs(axes @ c0) < 1e-6)]
        # a direction perpendicular to the axis, nearest +x
        ref = x - (x @ c0) * c0
        if torch.linalg.norm(ref) < 1e-6:
            ref = y - (y @ c0) * c0
        ref = ref / torch.linalg.norm(ref)
        if in_plane.shape[0] == 0:  # 6/m, 4/m, -3: any 360/n sector
            c2 = ref
            c1 = _rotate_about(c2, c0, 2 * np.pi / n_max)
        elif n_max == 3:  # -3m: the sector between adjacent mirror planes,
            # which are perpendicular to the in-plane 2-fold axes
            a = _closest(in_plane, ref)
            c2 = _rotate_about(a, c0, np.pi / 2)
            c1 = _rotate_about(a, c0, 5 * np.pi / 6)
        else:  # 6/mmm, 4/mmm: between adjacent in-plane 2-fold axes
            c2 = _closest(in_plane, ref)
            c1 = _rotate_about(c2, c0, np.pi / n_max)
        return torch.stack([c0, c1, c2])

    if n_max == 2 and axes.shape[0] == 3:  # mmm: the three 2-fold axes
        c0 = _closest(axes, z)
        rest = axes[torch.abs(axes @ c0) < 0.5]
        c1 = _closest(rest, x)
        c2 = _closest(rest[torch.abs(rest @ c1) < 0.5], y)
        return torch.stack([c0, c1, c2])
    return None


def symmetry_reduced_zone_angles(
    zone_axes: torch.Tensor, sym_quats: torch.Tensor, chunk: int = 8
) -> torch.Tensor:
    """Angular distances between zone axes, minimized over symmetry.

    The minimum is over the symmetry operations and the inversion (zone axes
    are directions modulo sign). Symmetry-equivalent zones are at distance
    zero, so an exclusion ball around a match also excludes its symmetry
    copies.

    Parameters
    ----------
    zone_axes : torch.Tensor
        Unit directions (Z, 3), crystal Cartesian.
    sym_quats : torch.Tensor
        Proper rotation quaternions (S, 4) of the crystal.
    chunk : int, default=8
        Symmetry operations processed at once, which bounds the memory to
        chunk * Z * Z.

    Returns
    -------
    torch.Tensor
        (Z, Z) angles in degrees.
    """
    Rs = quat_to_matrix(sym_quats).to(zone_axes.dtype)
    best = torch.full((zone_axes.shape[0],) * 2, -1.0, dtype=zone_axes.dtype)
    for s0 in range(0, Rs.shape[0], chunk):
        imgs = torch.einsum("sij,zj->szi", Rs[s0 : s0 + chunk], zone_axes)
        dots = torch.einsum("szi,wi->szw", imgs, zone_axes).abs().amax(dim=0)
        best = torch.maximum(best, dots)
    return torch.rad2deg(torch.acos(best.clamp(-1, 1)))


def symmetry_quaternions(
    rotations: np.ndarray,
    lat_real: np.ndarray,
) -> torch.Tensor:
    """Convert spglib integer rotation matrices to Cartesian quaternions.

    Parameters
    ----------
    rotations : np.ndarray
        (S, 3, 3) integer rotation matrices in the lattice basis, as returned
        by spglib (improper operations are discarded).
    lat_real : np.ndarray
        (3, 3) real-space lattice vectors as rows.

    Returns
    -------
    torch.Tensor
        (S', 4) unique proper-rotation quaternions in Cartesian coordinates,
        float64.
    """
    A = torch.as_tensor(lat_real, dtype=torch.float64).T  # columns are a, b, c
    W = torch.as_tensor(np.array(rotations), dtype=torch.float64)
    R_cart = A @ W @ torch.linalg.inv(A)
    proper = torch.linalg.det(R_cart) > 0
    # for a pseudo-symmetry group the lattice is slightly distorted from the
    # ideal one, so A W A^-1 is only approximately orthogonal: take the
    # nearest rotation (polar decomposition) so the operators are exact
    # rotations and the wedge and misorientation math stay consistent
    U, _, Vh = torch.linalg.svd(R_cart[proper])
    q = quat_from_matrix(U @ Vh)
    # deduplicate (q and -q are the same rotation; qnormalize fixed the sign)
    q_unique = torch.unique(torch.round(q / 1e-6) * 1e-6, dim=0)
    return qnormalize(q_unique)


__all__ = [
    "quat_from_zone_axis",
    "zone_axis_from_quat",
    "sample_zone_axes",
    "symmetry_axes",
    "sample_zone_axis_cap",
    "fundamental_zone_axis_wedge",
    "symmetry_reduced_zone_angles",
    "symmetry_quaternions",
]
