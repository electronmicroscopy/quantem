"""Rotations: quaternions, rotation matrices, Euler angles and misorientation.

This module defines the one rotation convention used across quantem.

Convention
----------
Lab frame
    Right-handed, with x along the image or scan rows, y along the columns
    and z toward the electron source, so the beam travels along -z. Vectors
    are ordered (row, col, z); in reciprocal space (k_row, k_col, k_z).
Rotations
    Active: a matrix rotates vectors, v' = R @ v. An orientation is the
    rotation from specimen (crystal) coordinates to lab coordinates::

        v_lab = R(q) @ v_specimen

    Code that samples a volume or a pattern on a lab grid uses the inverse,
    R.T, which is an implementation detail and not stored.
Quaternions
    Unit quaternions stored as torch tensors (..., 4) in scalar-first order
    (w, x, y, z), with the standard Hamilton product, so that
    R(a * b) = R(a) @ R(b). `qnormalize` picks w >= 0 of the pair q, -q.
Euler angles
    Bunge (phi1, Phi, phi2), intrinsic Z-X-Z, R = Rz(phi1) @ Rx(Phi) @ Rz(phi2),
    the same as scipy ``Rotation.from_euler("ZXZ", ...)`` (upper case, not
    the extrinsic ``"zxz"``). X is the row axis.
Units
    Radians, unless a name or parameter ends in ``_deg``.

Tomography (z1, x, z3) angles are converted to this convention with
:func:`quat_from_tomography_angles`; see its notes for the frame mapping.
"""

from __future__ import annotations

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


def quat_random_uniform(
    shape: tuple[int, ...],
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Uniformly distributed random rotations (Shoemake, 1992).

    Parameters
    ----------
    shape : tuple[int, ...]
        Leading shape of the output.
    generator : torch.Generator | None
        Random number generator, for reproducible draws.
    dtype : torch.dtype, default=torch.float64
        Output dtype.
    device : str | torch.device, default="cpu"
        Output device.

    Returns
    -------
    torch.Tensor
        Unit scalar-first quaternions (*shape, 4), uniform over SO(3).
    """
    u = torch.rand((*shape, 3), generator=generator, dtype=dtype, device=device)
    u1, u2, u3 = u.unbind(-1)
    s1, s2 = torch.sqrt(1 - u1), torch.sqrt(u1)
    t2, t3 = 2 * torch.pi * u2, 2 * torch.pi * u3
    q = torch.stack(
        (s2 * torch.cos(t3), s1 * torch.sin(t2), s1 * torch.cos(t2), s2 * torch.sin(t3)), dim=-1
    )
    return qnormalize(q)


def quat_from_tomography_angles(
    z1_deg: torch.Tensor | float,
    x_deg: torch.Tensor | float,
    z3_deg: torch.Tensor | float,
) -> torch.Tensor:
    """Specimen-to-lab orientation of tomography (z1, x, z3) pose angles.

    Tomography (`quantem.tomography`) and ptycho-tomography use their own
    frame, x along the image columns, y along the rows and z along the beam
    direction, and map lab ray coordinates into the specimen as::

        p_specimen = Rz(-z1) @ Rx(x) @ Rz(-z3) @ p_lab

    with x the tilt angle about the column axis. That frame is the one of
    this module rotated by 180 degrees about the (row + col) diagonal
    (row and col swapped, z reversed), so the pose is a proper rotation here
    as well, with Bunge angles (90 - z3, -x, -90 - z1) degrees. A plain tilt
    series is a rotation by -x about the column axis.

    Parameters
    ----------
    z1_deg, x_deg, z3_deg : torch.Tensor | float
        Tomography pose angles in degrees, broadcastable.

    Returns
    -------
    torch.Tensor
        Unit scalar-first quaternions (..., 4), specimen to lab, float64.
    """
    z1, x, z3 = torch.broadcast_tensors(
        *(torch.as_tensor(a, dtype=torch.float64) for a in (z1_deg, x_deg, z3_deg))
    )
    bunge = torch.deg2rad(torch.stack((90.0 - z3, -x, -90.0 - z1), dim=-1))
    return qnormalize(quat_from_euler_zxz(bunge))


def tomography_angles_from_quat(
    q: torch.Tensor,
    nominal_x_deg: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Tomography (z1, x, z3) pose angles of specimen-to-lab orientations.

    The inverse of :func:`quat_from_tomography_angles`. Every rotation has
    two angle sets, (z1, x, z3) and (z1 + 180, -x, z3 + 180). With
    `nominal_x_deg` the one whose tilt is closer to it is returned,
    otherwise the one with the smaller in-plane angles. At x = 0 or +-180
    only z1 + z3 is defined, and all of it is put in z1.

    Parameters
    ----------
    q : torch.Tensor
        Unit scalar-first quaternions (..., 4), specimen to lab.
    nominal_x_deg : torch.Tensor | float | None
        Expected tilt angle in degrees, broadcastable against q[..., 0].

    Returns
    -------
    torch.Tensor
        (z1, x, z3) in degrees (..., 3), z1 and z3 wrapped to [-180, 180).
    """

    def wrap(a):
        return torch.remainder(a + 180.0, 360.0) - 180.0

    phi1, big_phi, phi2 = torch.rad2deg(quat_to_euler_zxz(q.to(torch.float64))).unbind(-1)
    z1, x, z3 = wrap(-90.0 - phi2), -big_phi, wrap(90.0 - phi1)
    # gimbal lock: quat_to_euler_zxz returns phi2 = 0 and the whole rotation
    # about z in phi1, which is -(z1 + z3) at x = 0 and z1 - z3 + 180 at 180
    locked = torch.sin(torch.deg2rad(big_phi)).abs() < 1e-8
    z1_locked = torch.where(big_phi < 90.0, -phi1, phi1 - 180.0)
    z1 = torch.where(locked, wrap(z1_locked), z1)
    z3 = torch.where(locked, torch.zeros_like(z3), z3)
    # the other branch (z1 + 180, -x, z3 + 180)
    z1_b, x_b, z3_b = wrap(z1 + 180.0), -x, wrap(z3 + 180.0)
    if nominal_x_deg is None:
        use_b = z1_b.abs() + z3_b.abs() < z1.abs() + z3.abs()
    else:
        nominal = torch.as_tensor(nominal_x_deg, dtype=torch.float64)
        use_b = (x_b - nominal).abs() < (x - nominal).abs()
    use_b = use_b & ~locked
    return torch.stack(
        (torch.where(use_b, z1_b, z1), torch.where(use_b, x_b, x), torch.where(use_b, z3_b, z3)),
        dim=-1,
    )


__all__ = [
    "qmult",
    "qconj",
    "qnormalize",
    "qrotate",
    "quat_to_matrix",
    "quat_from_matrix",
    "quat_from_axis_angle",
    "quat_to_axis_angle",
    "quat_from_euler_zxz",
    "quat_to_euler_zxz",
    "misorientation_angle_deg",
    "misorientation_axis_angle",
    "symmetry_aligned",
    "slerp",
    "quat_random_uniform",
    "quat_from_tomography_angles",
    "tomography_angles_from_quat",
]
