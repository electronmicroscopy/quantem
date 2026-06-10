"""Tests for ``quantem.tomography.object_models``.

The constraint-parsing and ``ObjectPixelated`` tests are pure CPU. The INR / tensor-decomp
construction tests are ``requires_torch`` and follow the ``torch_device`` fixture (they must
be built on CUDA when CUDA is present; see conftest).
"""

import numpy as np
import pytest
import torch

from quantem.tomography.object_models import (
    ObjConstraintParams,
    ObjectBase,
    ObjectINR,
    ObjectPixelated,
    ObjectTensorDecomp,
)
from quantem.tomography.tomography_context import ReconstructionContext

from .conftest import requires_torch


class TestObjConstraintParse:
    def test_parse_pixelated_by_name(self):
        c = ObjConstraintParams.parse_dict({"name": "obj_pixelated", "tv_vol": 0.01})
        assert isinstance(c, ObjConstraintParams.ObjPixelatedConstraints)
        assert c.tv_vol == 0.01

    def test_parse_inr_by_type_key(self):
        c = ObjConstraintParams.parse_dict({"type": "obj_inr", "sparsity": 0.05})
        assert isinstance(c, ObjConstraintParams.ObjINRConstraints)
        assert c.sparsity == 0.05

    def test_parse_tensor_decomp(self):
        c = ObjConstraintParams.parse_dict({"name": "obj_tensor_decomp", "tv_plane": 0.1})
        assert isinstance(c, ObjConstraintParams.ObjTensorDecompConstraints)
        assert c.tv_plane == 0.1

    def test_missing_name_raises(self):
        with pytest.raises(ValueError):
            ObjConstraintParams.parse_dict({"tv_vol": 0.1})

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            ObjConstraintParams.parse_dict({"name": "obj_nope"})

    def test_constraint_key_partitions(self):
        c = ObjConstraintParams.ObjPixelatedConstraints()
        assert "positivity" in c.hard_constraint_keys
        assert "tv_vol" in c.soft_constraint_keys

    def test_constraint_keys_are_real_fields(self):
        """Regression: every soft/hard key must be an attribute, so __str__ never raises.

        ``ObjINRConstraints`` previously listed ``tv_plane`` (a field it does not have), which
        made ``str(constraints)`` blow up with AttributeError via ``Constraints.__str__``.
        """
        for cls in (
            ObjConstraintParams.ObjPixelatedConstraints,
            ObjConstraintParams.ObjINRConstraints,
            ObjConstraintParams.ObjTensorDecompConstraints,
        ):
            c = cls()
            for key in c.soft_constraint_keys + c.hard_constraint_keys:
                assert hasattr(c, key), f"{cls.__name__} lists missing key {key!r}"
            assert isinstance(str(c), str)  # must not raise


class TestObjectPixelatedConstruction:
    def test_from_uniform_is_zeros(self):
        obj = ObjectPixelated.from_uniform(shape=(8, 8, 8), device="cpu")
        assert obj.shape == (8, 8, 8)
        assert torch.allclose(obj.obj, torch.zeros(8, 8, 8))
        assert obj.obj_type == "pixelated"

    def test_from_array_numpy(self):
        arr = np.random.default_rng(0).random((6, 6, 6)).astype(np.float32)
        obj = ObjectPixelated.from_array(arr, device="cpu")
        assert obj.shape == (6, 6, 6)
        assert torch.allclose(obj.obj, torch.from_numpy(arr))
        assert obj.dtype == torch.float32

    def test_from_array_torch_is_copied(self):
        t = torch.ones(4, 4, 4)
        obj = ObjectPixelated.from_array(t, device="cpu")
        t += 5.0
        assert torch.allclose(obj.obj, torch.ones(4, 4, 4))  # original copy untouched

    def test_obj_view_shape(self):
        obj = ObjectPixelated.from_uniform(shape=(5, 6, 7), device="cpu")
        assert obj.obj_view.shape == (1, 5, 6, 7)

    def test_forward_returns_obj(self):
        obj = ObjectPixelated.from_array(torch.full((4, 4, 4), 2.0), device="cpu")
        assert torch.allclose(obj.forward(), obj.obj)


class TestObjectPixelatedConstraints:
    def test_positivity_clamps_negatives(self):
        obj = ObjectPixelated.from_array(torch.full((4, 4, 4), -1.0), device="cpu")
        obj.constraints.positivity = True
        assert torch.all(obj.obj >= 0.0)

    def test_positivity_off_keeps_negatives(self):
        obj = ObjectPixelated.from_array(torch.full((4, 4, 4), -1.0), device="cpu")
        obj.constraints.positivity = False
        assert torch.all(obj.obj < 0.0)

    def test_shrinkage_subtracts_then_floors(self):
        obj = ObjectPixelated.from_array(torch.full((4, 4, 4), 1.0), device="cpu")
        obj.constraints.positivity = False
        obj.constraints.shrinkage = 0.25
        assert torch.allclose(obj.obj, torch.full((4, 4, 4), 0.75))

    def test_tv_loss_scales_with_weight(self):
        # ctx.obj is the 3-D pixelated volume (D, H, W), matching ObjectPixelated._obj.
        ctx = ReconstructionContext(obj=torch.rand(8, 8, 8))
        obj = ObjectPixelated.from_uniform(shape=(8, 8, 8), device="cpu")
        obj.constraints.tv_vol = 1.0
        loss1 = obj.get_tv_loss(ctx)
        obj.constraints.tv_vol = 2.0
        loss2 = obj.get_tv_loss(ctx)
        assert torch.isclose(loss2, 2.0 * loss1)

    @pytest.mark.parametrize(
        "shape",
        [
            (8, 8, 8),  # bare 3-D volume
            (1, 8, 8, 8),  # obj_view layout [C=1, D, H, W]
            (3, 8, 8, 8),  # multimodal [C, D, H, W] (e.g. 3 elemental channels)
        ],
    )
    def test_tv_loss_rank_agnostic_finite_and_positive(self, shape):
        """Regression: get_tv_loss takes TV over the trailing spatial dims for any leading
        channel/batch axes -- not the old 5-D-only indexing. Supports multimodal [C, ...]."""
        ctx = ReconstructionContext(obj=torch.rand(*shape))
        obj = ObjectPixelated.from_uniform(shape=(8, 8, 8), device="cpu")
        obj.constraints.tv_vol = 1.0
        loss = obj.get_tv_loss(ctx)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss > 0.0  # random volume has non-zero total variation

    def test_soft_constraint_zero_when_tv_off(self):
        ctx = ReconstructionContext(obj=torch.rand(8, 8, 8))
        obj = ObjectPixelated.from_uniform(shape=(8, 8, 8), device="cpu")
        obj.constraints.tv_vol = 0.0
        assert float(obj.apply_soft_constraints(ctx).detach()) == 0.0


class TestFactoryGuard:
    def test_objectbase_requires_token(self):
        with pytest.raises(RuntimeError):
            ObjectBase(shape=(4, 4, 4))


@requires_torch
class TestObjectINR:
    def test_from_model_builds(self, torch_device):
        from quantem.core.ml.inr import HSiren

        model = HSiren(in_features=3, out_features=1, hidden_layers=1, hidden_features=8)
        obj = ObjectINR.from_model(model, shape=(16, 16, 16), device=torch_device)
        assert obj.shape == (16, 16, 16)
        assert obj.model is not None

    def test_optimization_parameters_single_group(self, torch_device):
        from quantem.core.ml.inr import HSiren

        model = HSiren(in_features=3, out_features=1, hidden_layers=1, hidden_features=8)
        obj = ObjectINR.from_model(model, shape=(16, 16, 16), device=torch_device)
        groups = obj.get_optimization_parameters()
        assert list(groups.keys()) == ["default"]
        assert len(groups["default"]) > 0


@requires_torch
class TestObjectTensorDecomp:
    def _model(self, n=16):
        from quantem.core.ml.models.kplanes import KPlanesTILTED

        return KPlanesTILTED(
            M_features=2, resolution=(n, n, n), multiscale_res_multipliers=[1], T=2
        )

    def test_pplr_optimization_parameter_keys(self, torch_device):
        obj = ObjectTensorDecomp.from_model(self._model(), shape=(16, 16, 16), device=torch_device)
        keys = set(obj.get_optimization_parameters().keys())
        assert keys == {"grids", "sigma_net", "so3"}

    def test_pretrain_not_implemented(self, torch_device):
        obj = ObjectTensorDecomp.from_model(self._model(), shape=(16, 16, 16), device=torch_device)
        with pytest.raises(NotImplementedError):
            obj.pretrain()


@requires_torch
class TestObjectINRBehaviour:
    def _obj(self, device, n=16):
        from quantem.core.ml.inr import HSiren

        model = HSiren(in_features=3, out_features=1, hidden_layers=1, hidden_features=8)
        return ObjectINR.from_model(model, shape=(n, n, n), device=device)

    def test_forward_masks_out_of_range(self, torch_device):
        obj = self._obj(torch_device)
        # Regression: the mask used to check x and y only, so tilted rays leaving the
        # volume along z still contributed (extrapolated) density to the integral.
        coords = torch.tensor(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            device=torch_device,
        )
        out = obj.forward(coords)
        assert out.shape[0] == 4
        for i in (1, 2, 3):  # each axis outside [-1, 1] -> masked to zero
            assert float(out[i].detach()) == 0.0

    def test_soft_constraints_without_coords(self, torch_device):
        # Regression: soft_loss was created on ctx.coords.device before the None
        # check, raising AttributeError when no constraints are active.
        obj = self._obj(torch_device)
        obj.constraints.tv_vol = 0.0
        obj.constraints.sparsity = 0.0
        loss = obj.apply_soft_constraints(ReconstructionContext())
        assert float(loss.detach()) == 0.0

    def test_apply_hard_constraints_positivity(self, torch_device):
        obj = self._obj(torch_device)
        obj.constraints.positivity = True
        pred = torch.tensor([-1.0, 0.5, 2.0], device=torch_device)
        assert torch.all(obj.apply_hard_constraints(pred) >= 0.0)

    def test_get_tv_loss_scalar(self, torch_device):
        obj = self._obj(torch_device)
        obj.constraints.tv_vol = 1.0
        coords = torch.rand(64, 3, device=torch_device) * 2 - 1
        ctx = ReconstructionContext(coords=coords, pred=torch.rand(64, device=torch_device))
        loss = obj.get_tv_loss(ctx)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


@requires_torch
class TestObjectTensorDecompTV:
    def _obj(self, device, n=16):
        from quantem.core.ml.models.kplanes import KPlanesTILTED

        model = KPlanesTILTED(
            M_features=2, resolution=(n, n, n), multiscale_res_multipliers=[1], T=2
        )
        return ObjectTensorDecomp.from_model(model, shape=(n, n, n), device=device)

    def test_apply_hard_constraints_positivity(self, torch_device):
        obj = self._obj(torch_device)
        obj.constraints.positivity = True
        pred = torch.tensor([-2.0, 0.0, 3.0], device=torch_device)
        assert torch.all(obj.apply_hard_constraints(pred) >= 0.0)

    def test_plane_tv_loss_nonneg_scalar(self, torch_device):
        obj = self._obj(torch_device)
        obj.constraints.tv_plane = 0.1
        loss = obj._get_plane_tv_loss()
        assert loss.ndim == 0
        assert float(loss.detach()) >= 0.0

    def test_volume_tv_loss_scalar(self, torch_device):
        obj = self._obj(torch_device)
        obj.constraints.tv_vol = 0.1
        coords = torch.rand(64, 3, device=torch_device) * 2 - 1
        loss = obj.get_volume_tv_loss(coords)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_normalize_optimizer_params_rejects_non_dict(self, torch_device):
        from quantem.core.ml.optimizer_mixin import OptimizerParams

        obj = self._obj(torch_device)
        with pytest.raises(TypeError):
            obj._normalize_optimizer_params([OptimizerParams.Adam()])

    def test_normalize_optimizer_params_rejects_wrong_keys(self, torch_device):
        from quantem.core.ml.optimizer_mixin import OptimizerParams

        obj = self._obj(torch_device)
        with pytest.raises(ValueError):
            obj._normalize_optimizer_params(
                {"grids": OptimizerParams.Adam(), "wrong": OptimizerParams.Adam()}
            )
