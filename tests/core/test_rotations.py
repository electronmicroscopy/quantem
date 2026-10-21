"""The rotation convention of quantem.core.utils.rotations."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from quantem.core.utils.rotations import (
    qmult,
    qnormalize,
    qrotate,
    quat_from_euler_zxz,
    quat_from_matrix,
    quat_from_tomography_angles,
    quat_random_uniform,
    quat_to_euler_zxz,
    quat_to_matrix,
    tomography_angles_from_quat,
)
from quantem.tomography.dataset_models import TomographyINRDataset

# tomography frame (x = col, y = row, z = beam travel) -> lab frame
# (x = row, y = col, z = toward the source)
TOMO_TO_LAB = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float64
)


@pytest.fixture
def quats():
    g = torch.Generator().manual_seed(0)
    return quat_random_uniform((200,), generator=g)


def test_matrix_matches_scipy_active(quats):
    R = quat_to_matrix(quats)
    R_scipy = Rotation.from_quat(quats.numpy(), scalar_first=True).as_matrix()
    np.testing.assert_allclose(R.numpy(), R_scipy, atol=1e-12)
    # 90 degrees about +z takes +x (row) to +y (col)
    q = torch.tensor([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], dtype=torch.float64)
    v = qrotate(q, torch.tensor([1.0, 0, 0], dtype=torch.float64))
    np.testing.assert_allclose(v.numpy(), [0, 1, 0], atol=1e-12)


def test_hamilton_product_composes(quats):
    a, b = quats[:100], quats[100:]
    torch.testing.assert_close(
        quat_to_matrix(qmult(a, b)), quat_to_matrix(a) @ quat_to_matrix(b), atol=1e-12, rtol=0
    )


def test_matrix_roundtrip_and_canonical_sign(quats):
    q = quat_from_matrix(quat_to_matrix(quats))
    assert (q[:, 0] >= 0).all()
    torch.testing.assert_close(q, qnormalize(quats), atol=1e-12, rtol=0)


def test_euler_is_intrinsic_zxz():
    angles = torch.tensor([[0.3, 0.7, 1.1], [-2.0, 2.5, 0.4]], dtype=torch.float64)
    R = quat_to_matrix(quat_from_euler_zxz(angles)).numpy()
    np.testing.assert_allclose(
        R, Rotation.from_euler("ZXZ", angles.numpy()).as_matrix(), atol=1e-12
    )
    # the extrinsic "zxz" is a different rotation for the same numbers
    assert np.abs(R - Rotation.from_euler("zxz", angles.numpy()).as_matrix()).max() > 0.1
    torch.testing.assert_close(quat_to_euler_zxz(quat_from_euler_zxz(angles)), angles)


def test_random_uniform_is_seeded_and_uniform():
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    q1 = quat_random_uniform((20000,), generator=g1)
    torch.testing.assert_close(q1, quat_random_uniform((20000,), generator=g2))
    torch.testing.assert_close(q1.norm(dim=-1), torch.ones(20000, dtype=torch.float64))
    # uniform over SO(3): the mean rotation matrix vanishes
    assert quat_to_matrix(q1).mean(0).abs().max() < 0.02


def test_tomography_angles_match_ray_transform():
    gen = torch.Generator().manual_seed(1)
    T = 16
    z1, x, z3 = (torch.rand(T, generator=gen, dtype=torch.float64) * 360 - 180 for _ in range(3))
    rays = torch.rand(T, 50, 3, generator=gen, dtype=torch.float64) * 2 - 1
    p_spec = TomographyINRDataset.transform_batch_rays(
        rays, z1, x, z3, torch.zeros(T, 2, dtype=torch.float64), N=64, sampling_rate=1.0
    )
    # same mapping through the lab frame: p_spec = R^T p_lab, R specimen -> lab
    R = quat_to_matrix(quat_from_tomography_angles(z1, x, z3))
    P = TOMO_TO_LAB
    expected = torch.einsum("ij,tjk,kl,tnl->tni", P.T, R.transpose(-1, -2), P, rays)
    torch.testing.assert_close(p_spec, expected, atol=1e-12, rtol=0)


def test_tomography_tilt_is_about_the_column_axis():
    q = quat_from_tomography_angles(0.0, 30.0, 0.0)
    col = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    expected = Rotation.from_rotvec(np.deg2rad(-30.0) * col.numpy()).as_matrix()
    np.testing.assert_allclose(quat_to_matrix(q).numpy(), expected, atol=1e-12)


@pytest.mark.parametrize(
    "angles",
    [
        (10.0, 35.0, -20.0),
        (-120.0, -60.0, 45.0),
        (0.0, 70.0, 0.0),
        (25.0, 0.0, 0.0),
        (40.0, 180.0, 0.0),
    ],
)
def test_tomography_angles_roundtrip(angles):
    q = quat_from_tomography_angles(*angles)
    back = tomography_angles_from_quat(q, nominal_x_deg=angles[1])
    torch.testing.assert_close(
        quat_to_matrix(quat_from_tomography_angles(*back.unbind(-1))),
        quat_to_matrix(q),
        atol=1e-10,
        rtol=0,
    )
    # equal up to whole turns (x = 180 and -180 are the same tilt)
    diff = np.remainder(back.numpy() - np.array(angles) + 180.0, 360.0) - 180.0
    np.testing.assert_allclose(diff, 0.0, atol=1e-8)


def test_tomography_angles_default_branch_keeps_small_in_plane_angles():
    q = quat_from_tomography_angles(2.0, 50.0, -3.0)
    np.testing.assert_allclose(
        tomography_angles_from_quat(q).numpy(), [2.0, 50.0, -3.0], atol=1e-8
    )


def test_deprecated_module_reexports():
    with pytest.warns(DeprecationWarning):
        import importlib

        import quantem.diffraction.rotations as old

        importlib.reload(old)
    from quantem.core.utils import rotations as core
    from quantem.diffraction import crystal_rotations

    assert old.quat_to_matrix is core.quat_to_matrix
    assert old.quat_from_zone_axis is crystal_rotations.quat_from_zone_axis
