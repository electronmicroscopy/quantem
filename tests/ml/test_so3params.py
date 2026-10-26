"""SO3ParamQuat follows the core rotation convention."""

import torch

from quantem.core.ml.models.kplanes import KPlanesTILTED
from quantem.core.ml.models.so3params import SO3ParamQuat
from quantem.core.utils.rotations import quat_random_uniform, quat_to_matrix


def test_from_matrix_roundtrip():
    R = quat_to_matrix(quat_random_uniform((6,), generator=torch.Generator().manual_seed(0)))
    so3 = SO3ParamQuat.from_matrix(R.float())
    torch.testing.assert_close(so3.as_matrix(), R.float(), atol=1e-6, rtol=0)


def test_kplanes_tau_state_roundtrip_for_both_parameterizations():
    for kind in ("r9svd", "quat"):
        kw = dict(resolution=(8, 8, 8), M_features=4, T=3, so3_param_type=kind)
        a, b = KPlanesTILTED(**kw), KPlanesTILTED(**kw, tau_init="identity")
        b.load_tau_state(a.extract_tau_state())
        torch.testing.assert_close(
            b.so3.as_matrix(), a.so3.as_matrix(), atol=1e-6, rtol=0, msg=f"so3_param_type={kind}"
        )
