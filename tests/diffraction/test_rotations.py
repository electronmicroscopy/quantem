"""Tests for quantem.diffraction.rotations."""

import numpy as np
import pytest
import torch

from quantem.diffraction.rotations import (
    misorientation_angle_deg,
    qconj,
    qmult,
    qnormalize,
    qrotate,
    quat_from_axis_angle,
    quat_from_euler_zxz,
    quat_from_matrix,
    quat_from_zone_axis,
    quat_to_euler_zxz,
    quat_to_matrix,
    sample_zone_axes,
    zone_axis_from_quat,
)


@pytest.fixture
def random_quats():
    torch.manual_seed(0)
    return qnormalize(torch.randn(100, 4, dtype=torch.float64))


def test_matrix_roundtrip(random_quats):
    R = quat_to_matrix(random_quats)
    assert torch.allclose(quat_from_matrix(R), random_quats, atol=1e-10)


def test_euler_roundtrip(random_quats):
    e = quat_to_euler_zxz(random_quats)
    assert torch.allclose(qnormalize(quat_from_euler_zxz(e)), random_quats, atol=1e-8)


def test_rotate_matches_matrix(random_quats):
    v = torch.randn(100, 3, dtype=torch.float64)
    R = quat_to_matrix(random_quats)
    assert torch.allclose(qrotate(random_quats, v), (R @ v[..., None]).squeeze(-1), atol=1e-10)


def test_mult_conj_identity(random_quats):
    q = random_quats
    ident = qmult(q, qconj(q))
    expect = torch.zeros_like(q)
    expect[:, 0] = 1.0
    assert torch.allclose(ident, expect, atol=1e-10)


def test_zone_axis_roundtrip():
    torch.manual_seed(1)
    v = torch.randn(50, 3, dtype=torch.float64)
    v = v / torch.linalg.norm(v, dim=-1, keepdim=True)
    q = quat_from_zone_axis(v, in_plane_deg=25.0)
    assert torch.allclose(zone_axis_from_quat(q), v, atol=1e-10)


def test_axis_angle():
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    q = quat_from_axis_angle(axis, torch.tensor(np.pi / 2, dtype=torch.float64))
    v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(
        qrotate(q, v), torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-9
    )


def test_misorientation_symmetry():
    # 90 degree rotation about z is a cubic symmetry: misorientation 0
    from ase.build import bulk

    from quantem.diffraction.crystal import Crystal

    xtl = Crystal.from_ase(bulk("Au", "fcc", a=4.08, cubic=True))
    qa = torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)
    qb = quat_from_axis_angle(
        torch.tensor([0.0, 0, 1.0], dtype=torch.float64),
        torch.tensor(np.pi / 2, dtype=torch.float64),
    )
    ang = misorientation_angle_deg(qa, qb, xtl.sym_quats)
    assert float(ang) < 1e-4
    ang_nosym = misorientation_angle_deg(qa, qb)
    assert abs(float(ang_nosym) - 90.0) < 1e-6


def test_sample_zone_axes_wedge():
    corners = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float64)
    corners = corners / torch.linalg.norm(corners, dim=-1, keepdim=True)
    v, inds = sample_zone_axes(corners, 2.0)
    assert torch.allclose(torch.linalg.norm(v, dim=-1), torch.ones(v.shape[0], dtype=v.dtype))
    # corners present
    for c in corners:
        assert torch.linalg.norm(v - c, dim=-1).min() < 1e-8


def test_sample_zone_axes_isotropic():
    # the requested step holds in every direction, whatever the apex angle
    for apex_deg in (30.0, 120.0):
        a = np.deg2rad(apex_deg)
        corners = torch.tensor(
            [[0, 0, 1], [1, 0, 0], [np.cos(a), np.sin(a), 0]], dtype=torch.float64
        )
        v, _ = sample_zone_axes(corners, 2.0)
        dots = (v @ v.T).clamp(-1, 1)
        ang = torch.rad2deg(torch.acos(dots))
        ang.fill_diagonal_(1e9)
        nn = ang.min(dim=1).values
        # points on the equator row: spacing within 25% of the step
        eq = v[:, 2].abs() < 1e-9
        assert nn[eq].min() > 1.5 and nn[eq].max() < 2.5


def test_symmetry_reduced_zone_angles():
    from ase.build import bulk

    from quantem.diffraction.crystal import Crystal
    from quantem.diffraction.rotations import symmetry_reduced_zone_angles

    xtl = Crystal.from_ase(bulk("Ti", "bcc", a=3.31, cubic=True), verbose=False)
    za = torch.tensor(
        [[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    za = za / torch.linalg.norm(za, dim=1, keepdim=True)
    ang = symmetry_reduced_zone_angles(za, xtl.sym_quats)
    # cubic axes are symmetry equivalent, as are the two <111> directions
    assert float(ang[0, 1]) < 1e-6 and float(ang[0, 2]) < 1e-6
    assert float(ang[3, 4]) < 1e-6
    assert abs(float(ang[0, 3]) - 54.7356) < 1e-3


def test_misorientation_axis_angle():
    from quantem.diffraction.rotations import misorientation_axis_angle

    axis = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float64) / 3
    qa = qnormalize(torch.tensor([0.9, 0.1, -0.3, 0.2], dtype=torch.float64))
    dq = quat_from_axis_angle(axis, torch.tensor(np.deg2rad(35.0), dtype=torch.float64))
    qb = qmult(qa, dq)  # R(qb) = R(qa) R(dq): dq is in the crystal frame of qa
    ax, ang = misorientation_axis_angle(qa, qb)
    assert abs(float(ang) - 35.0) < 1e-8
    assert torch.allclose(ax, axis, atol=1e-8)

    # with cubic symmetry, a 90 degree turn about [001] plus 10 degrees
    # about the same axis reduces to 10 degrees, axis unchanged up to sign
    sym = _cubic_sym_quats()
    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    qb = qmult(qa, quat_from_axis_angle(z, torch.tensor(np.deg2rad(100.0), dtype=torch.float64)))
    ax, ang = misorientation_axis_angle(qa, qb, sym)
    assert abs(float(ang) - 10.0) < 1e-6
    assert abs(abs(float(ax @ z)) - 1.0) < 1e-6
    assert abs(float(ang) - float(misorientation_angle_deg(qa, qb, sym))) < 1e-6

    # broadcasting over a batch
    qs = qnormalize(torch.randn(7, 4, dtype=torch.float64))
    ax, ang = misorientation_axis_angle(qs, qs[:1], sym)
    assert ax.shape == (7, 3) and ang.shape == (7,)
    assert torch.allclose(ang, misorientation_angle_deg(qs, qs[:1], sym), atol=1e-4)


def _cubic_sym_quats():
    import itertools

    from quantem.diffraction.rotations import symmetry_quaternions

    # the 24 proper rotations of m-3m as signed permutation matrices
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3), dtype=int)
            for i, (j, s) in enumerate(zip(perm, signs)):
                M[i, j] = s
            if round(np.linalg.det(M)) == 1:
                mats.append(M)
    return symmetry_quaternions(np.array(mats), np.eye(3))


def test_symmetry_aligned():
    from quantem.diffraction.rotations import symmetry_aligned

    torch.manual_seed(3)
    sym = _cubic_sym_quats()
    assert sym.shape == (24, 4)
    ref = qnormalize(torch.randn(4, dtype=torch.float64))
    # small perturbations of the reference, each moved to a random symmetry
    # branch: alignment must bring every one back next to the reference
    eps = quat_from_axis_angle(
        torch.randn(10, 3, dtype=torch.float64), torch.full((10,), 0.05, dtype=torch.float64)
    )
    near = qmult(ref.expand(10, 4), eps)
    far = qmult(near, sym[torch.randint(1, 24, (10,))])
    out = symmetry_aligned(ref, far, sym)
    assert out.shape == (10, 4)
    assert torch.allclose(out, qnormalize(near), atol=1e-10)
    # the same orientations as the inputs
    assert torch.allclose(
        misorientation_angle_deg(out, far, sym), torch.zeros(10, dtype=torch.float64), atol=1e-4
    )


def test_sample_zone_axis_cap():
    from quantem.diffraction.rotations import sample_zone_axis_cap

    axis = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float64)
    unit = axis / torch.linalg.norm(axis)
    pts = sample_zone_axis_cap(axis, 10.0, 2.0)
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert torch.allclose(
        torch.linalg.norm(pts, dim=1), torch.ones(pts.shape[0], dtype=torch.float64), atol=1e-12
    )
    ang = torch.rad2deg(torch.acos((pts @ unit).clamp(-1, 1)))
    assert float(ang.max()) <= 10.0 + 1e-9
    # equal-area count: cap area / step^2
    n_expect = 2 * np.pi * (1 - np.cos(np.deg2rad(10))) / np.deg2rad(2) ** 2
    assert abs(pts.shape[0] - np.ceil(n_expect)) <= 1
    # covers the cap: every direction in it has a sample within ~step
    rng = np.random.default_rng(0)
    probe = sample_zone_axis_cap(axis, 9.0, 0.5)[rng.choice(300, 50)]
    d = torch.rad2deg(torch.acos((probe @ pts.T).clamp(-1, 1))).min(dim=1).values
    assert float(d.max()) < 2.0
    # zero half angle: the axis alone; the poles take the short path
    assert torch.allclose(sample_zone_axis_cap(axis, 0.0, 1.0), unit[None])
    down = sample_zone_axis_cap(torch.tensor([0.0, 0.0, -1.0]), 5.0, 1.0)
    assert float(down[:, 2].max()) < -np.cos(np.deg2rad(5.0)) + 1e-9
