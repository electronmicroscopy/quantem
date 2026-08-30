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

The electron beam travels along -z in the lab frame. The zone axis --- the
beam direction expressed in crystal Cartesian coordinates --- is therefore
the third row of R(q)::

    zone_axis = R(q).T @ [0, 0, 1]

Euler angles use the Z-X-Z convention (Rowenhorst et al., 2015).
"""

from __future__ import annotations

import numpy as np
import torch


def qmult(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of quaternions, broadcasting over leading dims."""
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
    """Quaternion conjugate (inverse for unit quaternions)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)


def qnormalize(q: torch.Tensor) -> torch.Tensor:
    """Normalize to unit length, with w >= 0 canonicalization."""
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    return torch.where(q[..., :1] < 0, -q, q)


def qrotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by quaternions q (..., 4)."""
    qv = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
    return qmult(qmult(q, qv), qconj(q))[..., 1:]


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (..., 4) to rotation matrices (..., 3, 3)."""
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
    """Convert rotation matrices (..., 3, 3) to unit quaternions (..., 4).

    Uses the numerically stable branch selection of Shepperd's method,
    vectorized over leading dimensions.
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
    """Quaternion for rotation of `angle` (radians) about `axis` (..., 3)."""
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
    angle = torch.as_tensor(angle, dtype=axis.dtype, device=axis.device)
    half = angle[..., None] / 2
    return torch.cat((torch.cos(half), torch.sin(half) * axis), dim=-1)


def quat_to_axis_angle(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (axis (..., 3), angle (...,)) of unit quaternions."""
    q = qnormalize(q)
    angle = 2 * torch.acos(q[..., 0].clamp(-1, 1))
    sin_half = torch.sqrt((1 - q[..., 0] ** 2).clamp_min(1e-24))
    axis = q[..., 1:] / sin_half[..., None]
    return axis, angle


def quat_from_euler_zxz(angles: torch.Tensor) -> torch.Tensor:
    """Quaternion from Z-X-Z Euler angles (..., 3) in radians."""
    a, b, c = angles.unbind(-1)
    z = torch.zeros_like(a)
    qa = torch.stack((torch.cos(a / 2), z, z, torch.sin(a / 2)), dim=-1)
    qb = torch.stack((torch.cos(b / 2), torch.sin(b / 2), z, z), dim=-1)
    qc = torch.stack((torch.cos(c / 2), z, z, torch.sin(c / 2)), dim=-1)
    return qmult(qmult(qa, qb), qc)


def quat_to_euler_zxz(q: torch.Tensor) -> torch.Tensor:
    """Z-X-Z Euler angles (..., 3) in radians from unit quaternions."""
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
        Crystal-frame Cartesian direction(s) (..., 3) to place along the beam.
    in_plane_deg : torch.Tensor | float, default=0.0
        Additional in-plane rotation of the pattern, degrees.

    Returns
    -------
    torch.Tensor
        Quaternions (..., 4) such that quat_to_matrix(q).T @ [0,0,1] == zone_axis.
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
    q_spin = torch.stack(
        (torch.cos(in_plane / 2), z3, z3, torch.sin(in_plane / 2)), dim=-1
    )
    return qnormalize(qmult(q_spin, q_tilt))


def zone_axis_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Beam direction in crystal Cartesian coordinates (third row of R)."""
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
    """Symmetry-reduced misorientation axis (crystal frame) and angle.

    Returns the rotation axis (..., 3) in crystal Cartesian coordinates and
    the angle (...,) in degrees, minimized over the symmetry operators.
    """
    dq = qmult(qconj(qa), qb)
    if sym_ops is not None:
        dq_sym = qmult(dq[..., None, :], sym_ops)  # (..., S, 4)
        best = dq_sym[..., 0].abs().argmax(dim=-1)
        dq = torch.gather(
            dq_sym, -2, best[..., None, None].expand(*best.shape, 1, 4)
        ).squeeze(-2)
    dq = qnormalize(dq)
    axis, angle = quat_to_axis_angle(dq)
    return axis, torch.rad2deg(angle)


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between unit vectors v0 and v1."""
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
    """Triangular SLERP grid of unit zone-axis vectors inside a spherical triangle.

    Parameters
    ----------
    corners : torch.Tensor
        (3, 3) rows are the Cartesian corner directions of the fundamental
        zone-axis wedge, e.g. [001], [011], [111] for m-3m.
    step_deg : float
        Approximate angular step between neighboring zone axes, degrees.

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
        s = torch.linspace(0, 1, i + 1, dtype=c.dtype, device=c.device)
        row = slerp(pv.expand(i + 1, 3), pw.expand(i + 1, 3), s)
        row = row / torch.linalg.norm(row, dim=-1, keepdim=True)
        vecs.append(row)
        inds.append(torch.stack((torch.full((i + 1,), i), torch.arange(i + 1)), dim=-1))
    return torch.cat(vecs), torch.cat(inds).to(torch.long)


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
    q = quat_from_matrix(R_cart[proper])
    # deduplicate (q and -q are the same rotation; qnormalize fixed the sign)
    q_unique = torch.unique(torch.round(q / 1e-6) * 1e-6, dim=0)
    return qnormalize(q_unique)
