"""SO3ParamQuat follows the core rotation convention and loads old scalar-last models."""

import io
import math
import pickle

import torch

from quantem.core.ml.models.kplanes import KPlanesTILTED
from quantem.core.ml.models.so3params import SO3ParamQuat, SO3ParamR9SVD
from quantem.core.utils.rotations import qnormalize, quat_random_uniform, quat_to_matrix


def _legacy_shoemake(T):
    """The scalar-last (x, y, z, w) sampler SO3ParamQuat used before 2026-10."""
    u = torch.rand(T, 3)
    a, b = torch.sqrt(1.0 - u[:, 0]), torch.sqrt(u[:, 0])
    t1, t2 = 2 * math.pi * u[:, 1], 2 * math.pi * u[:, 2]
    return torch.stack(
        [a * torch.sin(t1), a * torch.cos(t1), b * torch.sin(t2), b * torch.cos(t2)], -1
    )


def test_as_matrix_is_core_quat_to_matrix():
    so3 = SO3ParamQuat(8)
    q = so3.quats.detach() / so3.quats.detach().norm(dim=-1, keepdim=True)
    torch.testing.assert_close(so3.as_matrix(), quat_to_matrix(q))


def test_from_matrix_roundtrip():
    R = quat_to_matrix(quat_random_uniform((6,), generator=torch.Generator().manual_seed(0)))
    so3 = SO3ParamQuat.from_matrix(R.float())
    torch.testing.assert_close(so3.as_matrix(), R.float(), atol=1e-6, rtol=0)


def test_random_init_gives_the_legacy_rotations():
    torch.manual_seed(5)
    legacy = _legacy_shoemake(10)
    torch.manual_seed(5)
    so3 = SO3ParamQuat(10)
    torch.testing.assert_close(
        so3.as_matrix(), quat_to_matrix(qnormalize(legacy[:, [3, 0, 1, 2]])), atol=1e-6, rtol=0
    )


def _legacy_module(q_wxyz):
    """An SO3ParamQuat as saved before 2026-10: scalar-last, no flag buffer."""
    so3 = SO3ParamQuat(q_wxyz.shape[0], init="identity")
    with torch.no_grad():
        so3.quats.copy_(q_wxyz[:, [1, 2, 3, 0]])
    del so3._buffers["scalar_first"]
    return so3


def test_loads_legacy_state_dict():
    q = quat_random_uniform((5,), generator=torch.Generator().manual_seed(1), dtype=torch.float32)
    state = _legacy_module(q).state_dict()
    assert "scalar_first" not in state
    so3 = SO3ParamQuat(5, init="identity")
    so3.load_state_dict(state)
    torch.testing.assert_close(so3.as_matrix(), quat_to_matrix(q), atol=1e-6, rtol=0)


def test_new_state_dict_is_not_reordered():
    so3 = SO3ParamQuat(5)
    other = SO3ParamQuat(5, init="identity")
    other.load_state_dict(so3.state_dict())
    torch.testing.assert_close(other.as_matrix(), so3.as_matrix())


def test_loads_legacy_pickled_module():
    q = quat_random_uniform((4,), generator=torch.Generator().manual_seed(2), dtype=torch.float32)
    loaded = pickle.loads(pickle.dumps(_legacy_module(q)))
    torch.testing.assert_close(loaded.as_matrix(), quat_to_matrix(q), atol=1e-6, rtol=0)
    buf = io.BytesIO()
    torch.save(_legacy_module(q), buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)
    torch.testing.assert_close(loaded.as_matrix(), quat_to_matrix(q), atol=1e-6, rtol=0)


def test_r9svd_rotation_state_is_the_raw_matrix():
    so3 = SO3ParamR9SVD(3)
    M = so3.rotation_state()
    torch.testing.assert_close(M, so3.M.detach())
    other = SO3ParamR9SVD(3, init="identity")
    other.load_rotation_state(M)
    torch.testing.assert_close(other.M, so3.M)


def test_kplanes_tau_state_roundtrip_for_both_parameterizations():
    for kind in ("r9svd", "quat"):
        kw = dict(resolution=(8, 8, 8), M_features=4, T=3, so3_param_type=kind)
        a, b = KPlanesTILTED(**kw), KPlanesTILTED(**kw, tau_init="identity")
        b.load_tau_state(a.extract_tau_state())
        torch.testing.assert_close(b.so3.as_matrix(), a.so3.as_matrix(), atol=1e-6, rtol=0)
