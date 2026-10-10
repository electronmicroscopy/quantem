"""Quaternion rotation utilities for orientation mapping.

All orientations in quantem.diffraction are represented as unit quaternions,
stored as torch tensors of shape (..., 4) in scalar-first order (w, x, y, z).
Rotation matrices, Euler angles, and axis-angle forms are provided only as
conversions at the boundaries.

Convention
----------
A quaternion q represents the rotation of crystal-frame vectors into the
laboratory (beam) frame::

    v_lab = R(q) @ v_crystal

The electron beam travels along -z in the lab frame. The zone axis is the
crystal direction that points from the specimen back toward the source,
lab +z, expressed in crystal Cartesian coordinates. It is therefore the
third row of R(q)::

    zone_axis = R(q).T @ [0, 0, 1]

Euler angles use the Z-X-Z convention (Rowenhorst et al., 2015).
"""

from __future__ import annotations

import numpy as np
import torch


def qmult(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a * b of quaternions, broadcasting over leading dims.

    The product applies `b` first, then `a`: R(a * b) = R(a) @ R(b).

    Parameters
    ----------
    a, b : torch.Tensor
        Scalar-first quaternions (..., 4), broadcastable.

    Returns
    -------
    torch.Tensor
        Product quaternions (..., 4), not renormalized.
    """
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def qconj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate (inverse for unit quaternions).

    Parameters
    ----------
    q : torch.Tensor
        Scalar-first quaternions (..., 4).

    Returns
    -------
    torch.Tensor
        (w, -x, -y, -z), shape (..., 4). For a crystal-to-lab orientation
        this is the lab-to-crystal rotation.
    """
    w, x, y, z = q.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)


def qnormalize(q: torch.Tensor) -> torch.Tensor:
    """Normalize to unit length, with w >= 0 canonicalization.

    Parameters
    ----------
    q : torch.Tensor
        Scalar-first quaternions (..., 4), nonzero.

    Returns
    -------
    torch.Tensor
        Unit quaternions (..., 4) with w >= 0; q and -q describe the same
        rotation, so this picks one of the two.
    """
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    return torch.where(q[..., :1] < 0, -q, q)


def qrotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions, v' = R(q) @ v.

    Parameters
    ----------
    q : torch.Tensor
        Unit scalar-first quaternions (..., 4). For an orientation, crystal
        frame vectors are rotated into the lab frame.
    v : torch.Tensor
        Vectors (..., 3), broadcastable against `q`.

    Returns
    -------
    torch.Tensor
        Rotated vectors (..., 3).
    """
    qv = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
    return qmult(qmult(q, qv), qconj(q))[..., 1:]


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to rotation matrices.

    Parameters
    ----------
    q : torch.Tensor
        Unit scalar-first quaternions (..., 4).

    Returns
    -------
    torch.Tensor
        Rotation matrices (..., 3, 3) with v_lab = R @ v_crystal for an
        orientation.
    """
    w, x, y, z = q.unbind(-1)
    two = 2.0
    R = torch.stack(
        (
            1 - two * (y * y + z * z),
            two * (x * y - w * z),
            two * (x * z + w * y),
            two * (x * y + w * z),
            1 - two * (x * x + z * z),
            two * (y * z - w * x),
            two * (x * z - w * y),
            two * (y * z + w * x),
            1 - two * (x * x + y * y),
        ),
        dim=-1,
    )
    return R.reshape(q.shape[:-1] + (3, 3))


def quat_from_matrix(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to unit quaternions.

    Uses the numerically stable branch selection of Shepperd's method,
    vectorized over leading dimensions.

    Parameters
    ----------
    R : torch.Tensor
        Proper rotation matrices (..., 3, 3).

    Returns
    -------
    torch.Tensor
        Unit scalar-first quaternions (..., 4) with w >= 0, the inverse of
        :func:`quat_to_matrix`.
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    # four candidate solutions, one per branch
    q_w = torch.stack((1 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01), dim=-1)
    q_x = torch.stack((m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20), dim=-1)
    q_y = torch.stack((m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21), dim=-1)
    q_z = torch.stack((m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22), dim=-1)
    q_all = torch.stack((q_w, q_x, q_y, q_z), dim=1)  # (N, 4, 4)

    trace_terms = torch.stack(
        (1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22),
        dim=-1,
    )
    branch = trace_terms.argmax(dim=-1)
    q = q_all[torch.arange(R.shape[0], device=R.device), branch]
    return qnormalize(q).reshape(batch_shape + (4,))


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Quaternion for a right-handed rotation about an axis.

    Parameters
    ----------
    axis : torch.Tensor
        Rotation axes (..., 3); need not be normalized.
    angle : torch.Tensor
        Rotation angles (...,), radians.

    Returns
    -------
    torch.Tensor
        Unit scalar-first quaternions (..., 4).
    """
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
    angle = torch.as_tensor(angle, dtype=axis.dtype, device=axis.device)
    half = angle[..., None] / 2
    return torch.cat((torch.cos(half), torch.sin(half) * axis), dim=-1)


def quat_to_axis_angle(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation axis and angle of unit quaternions.

    Parameters
    ----------
    q : torch.Tensor
        Scalar-first quaternions (..., 4).

    Returns
    -------
    axis : torch.Tensor
        Unit rotation axes (..., 3). Undefined (near zero) for the identity.
    angle : torch.Tensor
        Rotation angles (...,) in [0, pi], radians, after w >= 0
        canonicalization.
    """
    q = qnormalize(q)
    angle = 2 * torch.acos(q[..., 0].clamp(-1, 1))
    sin_half = torch.sqrt((1 - q[..., 0] ** 2).clamp_min(1e-24))
    axis = q[..., 1:] / sin_half[..., None]
    return axis, angle


def quat_from_euler_zxz(angles: torch.Tensor) -> torch.Tensor:
    """Quaternion from Z-X-Z Euler angles.

    Parameters
    ----------
    angles : torch.Tensor
        (phi1, Phi, phi2) Euler angles (..., 3), radians, applied as
        R = Rz(phi1) @ Rx(Phi) @ Rz(phi2).

    Returns
    -------
    torch.Tensor
        Scalar-first quaternions (..., 4).
    """
    a, b, c = angles.unbind(-1)
    z = torch.zeros_like(a)
    qa = torch.stack((torch.cos(a / 2), z, z, torch.sin(a / 2)), dim=-1)
    qb = torch.stack((torch.cos(b / 2), torch.sin(b / 2), z, z), dim=-1)
    qc = torch.stack((torch.cos(c / 2), z, z, torch.sin(c / 2)), dim=-1)
    return qmult(qmult(qa, qb), qc)


def quat_to_euler_zxz(q: torch.Tensor) -> torch.Tensor:
    """Z-X-Z Euler angles from unit quaternions.

    Parameters
    ----------
    q : torch.Tensor
        Unit scalar-first quaternions (..., 4).

    Returns
    -------
    torch.Tensor
        (phi1, Phi, phi2) Euler angles (..., 3), radians, the inverse of
        :func:`quat_from_euler_zxz`. In the gimbal-locked case (Phi = 0 or
        pi) the whole rotation about z is put in phi1 and phi2 is 0.
    """
    R = quat_to_matrix(q)
    beta = torch.acos(R[..., 2, 2].clamp(-1, 1))
    alpha = torch.atan2(R[..., 0, 2], -R[..., 1, 2])
    gamma = torch.atan2(R[..., 2, 0], R[..., 2, 1])
    # gimbal-locked cases: fold everything into alpha
    locked = torch.sin(beta).abs() < 1e-8
    alpha_locked = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    alpha = torch.where(locked, alpha_locked, alpha)
    gamma = torch.where(locked, torch.zeros_like(gamma), gamma)
    return torch.stack((alpha, beta, gamma), dim=-1)


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


def misorientation_angle_deg(
    qa: torch.Tensor,
    qb: torch.Tensor,
    sym_ops: torch.Tensor | None = None,
) -> torch.Tensor:
    """Misorientation angle between orientations, minimized over symmetry.

    Parameters
    ----------
    qa, qb : torch.Tensor
        Quaternions (..., 4), broadcastable against each other.
    sym_ops : torch.Tensor | None
        Proper rotation symmetry quaternions (S, 4) of the crystal. If None,
        the raw rotation angle between qa and qb is returned.

    Returns
    -------
    torch.Tensor
        Misorientation angles in degrees (...,).
    """
    dq = qmult(qconj(qa), qb)
    if sym_ops is None:
        w = dq[..., 0].abs().clamp(-1, 1)
    else:
        dq_sym = qmult(dq[..., None, :], sym_ops)  # (..., S, 4)
        w = dq_sym[..., 0].abs().amax(dim=-1).clamp(-1, 1)
    return torch.rad2deg(2 * torch.acos(w))


def misorientation_axis_angle(
    qa: torch.Tensor,
    qb: torch.Tensor,
    sym_ops: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetry-reduced misorientation axis and angle.

    The misorientation dq = conj(qa) * qb takes the crystal frame of `qa`
    onto that of `qb`, R(qb) = R(qa) @ R(dq). Among the equivalent
    dq * s over the symmetry operators s, the one with the smallest angle is
    returned.

    Parameters
    ----------
    qa, qb : torch.Tensor
        Unit scalar-first quaternions (..., 4), crystal to lab,
        broadcastable against each other.
    sym_ops : torch.Tensor | None
        Proper rotation symmetry quaternions (S, 4) of the crystal. If None,
        the raw misorientation is returned.

    Returns
    -------
    axis : torch.Tensor
        Unit rotation axes (..., 3) in the crystal Cartesian frame of `qa`.
        Undefined (near zero) when the angle is zero.
    angle : torch.Tensor
        Misorientation angles (...,) in degrees, in [0, 180]; equal to
        :func:`misorientation_angle_deg` for the same inputs.
    """
    dq = qmult(qconj(qa), qb)
    if sym_ops is not None:
        dq_sym = qmult(dq[..., None, :], sym_ops)  # (..., S, 4)
        best = dq_sym[..., 0].abs().argmax(dim=-1)
        dq = torch.gather(dq_sym, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    dq = qnormalize(dq)
    axis, angle = quat_to_axis_angle(dq)
    return axis, torch.rad2deg(angle)


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between directions.

    Parameters
    ----------
    v0, v1 : torch.Tensor
        End directions (..., 3); normalized internally.
    t : torch.Tensor
        Interpolation fractions (...,), 0 at `v0` and 1 at `v1`.

    Returns
    -------
    torch.Tensor
        Unit vectors (..., 3) along the great circle from `v0` to `v1`.
    """
    v0 = v0 / torch.linalg.norm(v0, dim=-1, keepdim=True)
    v1 = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True)
    omega = torch.acos((v0 * v1).sum(-1, keepdim=True).clamp(-1, 1))
    so = torch.sin(omega)
    t = t[..., None]
    small = so.abs() < 1e-12
    w0 = torch.where(small, 1 - t, torch.sin((1 - t) * omega) / so)
    w1 = torch.where(small, t, torch.sin(t * omega) / so)
    return w0 * v0 + w1 * v1


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


def symmetry_aligned(
    reference: torch.Tensor,
    quats: torch.Tensor,
    sym_quats: torch.Tensor,
) -> torch.Tensor:
    """Symmetry images of `quats` that lie nearest to `reference`, (N, 4).

    Two quaternions can describe the same crystal orientation while being far
    apart as quaternions, so any average over orientations has to bring them
    into a common symmetry branch first. For each input this returns the
    symmetry-equivalent quaternion whose misorientation to the reference is
    smallest, which makes a weighted quaternion mean well defined.

    Parameters
    ----------
    reference : torch.Tensor
        Unit scalar-first quaternion (4,), crystal to lab.
    quats : torch.Tensor
        Unit quaternions, reshaped to (N, 4).
    sym_quats : torch.Tensor
        Proper rotation symmetry quaternions (S, 4) of the crystal, crystal
        Cartesian frame. Each candidate is q * s, the same lab orientation
        of the symmetric crystal.

    Returns
    -------
    torch.Tensor
        (N, 4) float64 quaternions with w >= 0, each describing the same
        orientation as the corresponding input.
    """
    ref = torch.as_tensor(reference, dtype=torch.float64).reshape(4)
    q = torch.as_tensor(quats, dtype=torch.float64).reshape(-1, 4)
    sym = torch.as_tensor(sym_quats, dtype=torch.float64).reshape(-1, 4)
    cand = qmult(q[:, None, :], sym[None, :, :])  # (N, S, 4)
    dots = torch.abs(torch.einsum("nsi,i->ns", cand, ref))
    best = dots.argmax(dim=1)
    return qnormalize(cand[torch.arange(q.shape[0]), best])


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
