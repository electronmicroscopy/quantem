"""
Tests for the tensor-decomposition (K-Planes) object model, ``ObjectTensorDecomp``.

``ObjectTensorDecomp`` subclasses ``ObjectINR`` and swaps the SIREN for a ``KPlanes``
network, so it reuses every coordinate-query path of the INR and only differs in the
optimizer wiring (per-parameter-group learning rates, PPLR). These tests cover the
shared-core ``KPlanes`` head fix, the object model in isolation (forward/vacuum/PPLR
parameter groups, bad-key rejection), the ptychography PPLR optimizer pass-through, and
an end-to-end reconstruction on the same non-toroidal synthetic object used by
``test_object_inr.py`` (``inr_synthetic_dataset`` in ``conftest.py``). The coordinate-query
paths it inherits (off-object masking, z coordinates, autograd-only backward, the implicit
flag) are tested once, in ``test_object_inr.py``.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from quantem.core import config
from quantem.core.io.serialize import load as autoserialize_load
from quantem.core.ml import OptimizerParams, SchedulerParams
from quantem.core.ml.models.kplanes import CPTilted, KPlanes
from quantem.diffractive_imaging.object_models import (
    ObjectModelType,
    ObjectPixelated,
    ObjectTensorDecomp,
)

from .conftest import inr_best_corr, inr_ptychography

if config.NUM_DEVICES > 0:
    config.set_device("gpu")

# K-Planes recon config (validated against the conftest dataset). grids learn faster than the
# decoder head; both run under a single cosine-annealing schedule.
_LR_GRIDS = 1e-2
_LR_SIGMA = 1e-3
_LOSS_RATIO = 0.3
_CORR_THRESHOLD = 0.6


def _build_kplanes_ptycho(
    num_slices: int = 1,
    slice_thicknesses=None,
    M_features: int = 24,
    resolution: tuple[int, int, int] = (16, 64, 64),
):
    """Build an ObjectTensorDecomp (K-Planes) ptychography with the exact (frozen) probe."""
    obj = ObjectTensorDecomp.from_uniform(
        num_slices=num_slices,
        slice_thicknesses=slice_thicknesses,
        M_features=M_features,
        resolution=resolution,
        multiscale_res_multipliers=(0.25, 0.5, 1.0),
        use_hybrid_mlp=False,
        obj_type="pure_phase",
        rng=0,
    )
    return inr_ptychography(obj)


def _pplr_params() -> dict:
    return {
        "object": {
            "grids": OptimizerParams.Adam(lr=_LR_GRIDS),
            "sigma_net": OptimizerParams.Adam(lr=_LR_SIGMA),
        }
    }


# --------------------------------------------------------------------------- #
# Shared-core KPlanes head fix (use_hybrid_mlp=False must build a decoder head)
# --------------------------------------------------------------------------- #
class TestKPlanesHead:
    def test_head_type_follows_use_hybrid_mlp(self):
        """use_hybrid_mlp=False builds a linear decoder head, True an MLP; both run."""
        for use_hybrid_mlp, head, extra in [
            (False, nn.Linear, {}),
            (True, nn.Sequential, {"hybrid_hidden_dim": 32, "hybrid_num_layers": 2}),
        ]:
            m = KPlanes(
                M_features=8,
                resolution=(16, 16, 16),
                multiscale_res_multipliers=[0.5, 1.0],
                density_activation=nn.Identity(),
                use_hybrid_mlp=use_hybrid_mlp,
                **extra,
            )
            assert isinstance(m.sigma_net, head), use_hybrid_mlp
            assert m(torch.rand(20, 3) * 2 - 1).shape == (20, 1), use_hybrid_mlp
            assert set(m.get_params().keys()) == {"grids", "sigma_net"}, use_hybrid_mlp


# --------------------------------------------------------------------------- #
# ObjectTensorDecomp in isolation
# --------------------------------------------------------------------------- #
class TestObjectTensorDecompUnit:
    def _obj(self, **kw):
        return ObjectTensorDecomp.from_uniform(
            num_slices=kw.pop("num_slices", 1),
            slice_thicknesses=kw.pop("slice_thicknesses", None),
            M_features=8,
            resolution=kw.pop("resolution", (16, 32, 32)),
            multiscale_res_multipliers=(0.5, 1.0),
            rng=0,
            **kw,
        )

    def test_forward_shape_dtype_and_vacuum(self):
        for num_slices, thickness in [(1, None), (3, 2.0)]:
            obj = self._obj(num_slices=num_slices, slice_thicknesses=thickness)
            assert obj.is_implicit is True
            assert obj.name == "ObjectTensorDecomp"
            assert isinstance(obj, ObjectModelType)
            obj._initialize_obj((num_slices, 24, 28))
            coords = torch.rand(5, 8, 8, 2) * 2 - 1
            patches = obj.forward(coords)
            assert patches.shape == (num_slices, 5, 8, 8), num_slices
            assert patches.is_complex(), num_slices
            # vacuum init (zeroed decoder head) -> unit transmission everywhere
            assert torch.allclose(patches, torch.ones_like(patches), atol=1e-6), num_slices

    def test_get_optimization_parameters_keys(self):
        obj = self._obj()
        groups = obj.get_optimization_parameters()
        assert set(groups.keys()) == {"grids", "sigma_net"}
        for tensors in groups.values():
            assert len(tensors) > 0
            assert all(isinstance(t, nn.Parameter) and t.is_leaf for t in tensors)

    def test_pplr_optimizer_construction(self):
        obj = self._obj()
        obj.set_optimizer(
            {
                "grids": OptimizerParams.Adam(lr=_LR_GRIDS),
                "sigma_net": OptimizerParams.Adam(lr=_LR_SIGMA),
            }
        )
        opt = obj.optimizer
        assert isinstance(opt, torch.optim.Adam)
        assert len(opt.param_groups) == 2
        assert sorted(pg["lr"] for pg in opt.param_groups) == [_LR_SIGMA, _LR_GRIDS]

    def test_optimizer_params_rejects_bad_keys(self):
        obj = self._obj()
        # missing a required group
        with pytest.raises(ValueError):
            obj.set_optimizer({"grids": OptimizerParams.Adam(lr=_LR_GRIDS)})
        # single-optimizer shorthand is not a valid PPLR spec here
        with pytest.raises(TypeError):
            obj.set_optimizer({"name": "adam", "lr": _LR_GRIDS})
        # a bare OptimizerParamsType (single optimizer) is also rejected
        with pytest.raises(TypeError):
            obj.set_optimizer(OptimizerParams.Adam(lr=_LR_GRIDS))

    def test_pretrain_without_target_raises(self):
        """pretrain is supported now, but needs a target (use from_pixelated or pass one)."""
        obj = self._obj()
        obj._initialize_obj((1, 24, 28))
        with pytest.raises(ValueError, match="pretrain target"):
            obj.pretrain(num_iters=1, show=False)

    def test_disable_sentinel_accepted(self):
        """The framework's "disabled" optimizer sentinel must pass through (used by reset).

        ``reconstruct(reset=True)`` replays ``reset_optimizer`` with the init default
        ``{"default": NoneOptimizer()}``; that must not be rejected by the PPLR key check.
        """
        obj = self._obj()
        # bare NoneOptimizer and the default-keyed dict both mean "no optimizer"
        obj.set_optimizer(OptimizerParams.NoneOptimizer())
        assert obj.optimizer is None
        obj.set_optimizer({"default": OptimizerParams.NoneOptimizer()})
        assert obj.optimizer is None

    def test_from_model_warns_on_lambda_activation(self):
        # a lambda activation cannot be pickled by AutoSerialize -> from_model must warn
        model = KPlanes(
            M_features=4,
            resolution=(8, 8, 8),
            multiscale_res_multipliers=[1.0],
            use_hybrid_mlp=False,  # default density_activation is a lambda
        )
        with pytest.warns(UserWarning, match="density_activation"):
            ObjectTensorDecomp.from_model(model, num_slices=1, obj_type="pure_phase")

    def test_from_model_rejects_non_kplanes(self):
        with pytest.raises(TypeError):
            ObjectTensorDecomp.from_model(nn.Linear(3, 1), num_slices=1)  # pyright: ignore[reportArgumentType]

    def test_tilted_from_uniform(self):
        """tilted=True builds a KPlanesTILTED with an extra `so3` param group."""
        obj = ObjectTensorDecomp.from_uniform(
            num_slices=1,
            M_features=6,
            resolution=(16, 32, 32),
            multiscale_res_multipliers=(0.5, 1.0),
            tilted=True,
            T=4,
            obj_type="pure_phase",
            rng=0,
        )
        obj._initialize_obj((1, 24, 28))
        assert obj.model.tilted is True
        assert set(obj.get_optimization_parameters().keys()) == {"grids", "sigma_net", "so3"}
        patches = obj.forward(torch.rand(3, 6, 6, 2) * 2 - 1)
        assert patches.shape == (1, 3, 6, 6)
        # vacuum init holds for the tilted decoder too
        assert torch.allclose(patches, torch.ones_like(patches), atol=1e-6)
        obj.set_optimizer({k: OptimizerParams.Adam(lr=1e-2) for k in obj.model.param_keys})
        assert obj.optimizer is not None
        assert len(obj.optimizer.param_groups) == 3

    def test_from_model_accepts_cptilted(self):
        """from_model accepts the CPTilted bottleneck (TensorDecompositionModel, not a KPlanes)."""
        cp = CPTilted(
            C=4,
            resolution=(32, 32, 32),
            multiscale_res_multipliers=[1.0],
            T=4,
            density_activation=nn.Identity(),
        )
        obj = ObjectTensorDecomp.from_model(cp, num_slices=1, obj_type="pure_phase")
        assert set(obj.get_optimization_parameters().keys()) == {"grids", "sigma_net", "so3"}
        assert obj.forward(torch.rand(2, 4, 4, 2) * 2 - 1).shape == (1, 2, 4, 4)

    def test_from_pixelated_pretrain(self):
        """from_pixelated + pretrain warm-starts the K-Planes grid to a pixelated object."""
        h = w = 32
        ys = torch.linspace(-1, 1, h)
        xs = torch.linspace(-1, 1, w)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        phase = (0.5 * torch.sin(3 * gy) * torch.cos(2 * gx)).float()[None]
        pix = ObjectPixelated.from_array(initial_obj=phase, obj_type="pure_phase")
        pix._initialize_obj((1, h, w), sampling=(1.0, 1.0))

        kp = ObjectTensorDecomp.from_pixelated(pix, M_features=16, resolution=(48, 48, 48))
        assert kp.num_slices == pix.num_slices
        assert tuple(kp.pretrain_target.shape) == tuple(pix.obj.shape)

        kp.pretrain(num_iters=120, show=False)  # default PPLR optimizer
        losses = kp.pretrain_losses
        assert losses[-1] < 0.05 * losses[0]

        gt = pix.obj[0].detach().cpu().numpy()
        gt -= gt.mean()

        def _corr_to_pix(arr):
            a = arr - arr.mean()
            return float((a * gt).sum() / np.sqrt((a**2).sum() * (gt**2).sum() + 1e-12))

        assert _corr_to_pix(kp.obj[0].detach().cpu().numpy()) > 0.9
        # pretrained weights are the reset state
        kp.reset()
        assert _corr_to_pix(kp.obj[0].detach().cpu().numpy()) > 0.9

    def test_potential_identity_default_and_positivity_penalty(self):
        """Potential K-Planes uses an identity decoder by default (softplus/relu fit zero-background
        potentials poorly); the inherited soft positivity penalty drives it non-negative instead."""
        obj = self._obj(obj_type="potential", resolution=(16, 16, 16))
        obj._initialize_obj((1, 16, 16))
        assert obj.obj_type == "potential"
        assert isinstance(obj.model.density_activation, nn.Identity)  # identity, not softplus
        # force the whole potential negative via the (zero-weight) decoder bias
        with torch.no_grad():
            obj.model.sigma_net.bias.fill_(-0.5)  # type:ignore
        assert float(obj._materialize_obj().min()) == pytest.approx(-0.5, abs=1e-3)
        obj.constraints = {"positivity_weight": 1.0}
        assert float(obj._sampled_positivity_loss(1.0)) == pytest.approx(0.5, abs=0.05)
        opt = torch.optim.Adam(obj.model.parameters(), lr=2e-2)
        for _ in range(100):
            opt.zero_grad()
            obj.apply_soft_constraints().backward()
            opt.step()
        assert float(obj._materialize_obj().min()) > -1e-2  # driven non-negative


# --------------------------------------------------------------------------- #
# PPLR optimizer pass-through through the ptychography optimizer layer
# --------------------------------------------------------------------------- #
class TestPPLRPassthrough:
    def test_setter_passes_nested_dict_through_unmutated(self):
        ptycho, _ = _build_kplanes_ptycho()
        params = _pplr_params()
        snapshot = {k: dict(v) for k, v in params.items()}  # shallow copy of inner dicts
        ptycho.optimizer_params = params
        ptycho.set_optimizers()
        # the model received the PPLR groups, not a corrupted single-optimizer dict
        stored = ptycho.obj_model.optimizer_params
        assert set(stored.keys()) == {"grids", "sigma_net"}
        # the ptychography setter must not have mutated the caller's nested dict
        assert params["object"] == snapshot["object"]
        assert "name" not in params["object"] and "lr" not in params["object"]
        # optimizer has the two named groups with the requested LRs
        opt = ptycho.obj_model.optimizer
        assert opt is not None
        assert len(opt.param_groups) == 2
        assert sorted(pg["lr"] for pg in opt.param_groups) == [_LR_SIGMA, _LR_GRIDS]


# --------------------------------------------------------------------------- #
# End-to-end reconstruction
# --------------------------------------------------------------------------- #
@pytest.mark.slow
class TestObjectTensorDecompReconstruction:
    def test_loss_decreases_and_recovers_object(self):
        ptycho, gt_phase = _build_kplanes_ptycho()
        ptycho.reconstruct(
            num_iters=150,
            optimizer_params=_pplr_params(),  # probe frozen
            scheduler_params={"object": SchedulerParams.CosineAnnealing()},
            batch_size=200,
        )
        losses = np.array(ptycho._iter_losses)
        assert losses[-1] < _LOSS_RATIO * losses[0]
        assert inr_best_corr(ptycho.obj[0], gt_phase) > _CORR_THRESHOLD

    def test_reconstruct_with_reset_runs(self):
        """reset=True replays reset_optimizer with the disabled sentinel before applying PPLR."""
        ptycho, _ = _build_kplanes_ptycho()
        ptycho.reconstruct(
            num_iters=5,
            reset=True,
            optimizer_params=_pplr_params(),
            batch_size=200,
        )
        losses = np.array(ptycho._iter_losses)
        assert np.isfinite(losses).all()
        # the object optimizer still has its two PPLR groups after the reset cycle
        opt = ptycho.obj_model.optimizer
        assert opt is not None
        assert len(opt.param_groups) == 2

    def test_tilted_reconstruct_runs(self):
        """A tilted K-Planes object reconstructs end-to-end with a 3-group PPLR optimizer."""
        obj = ObjectTensorDecomp.from_uniform(
            num_slices=1,
            M_features=12,
            resolution=(16, 64, 64),
            multiscale_res_multipliers=(0.25, 0.5, 1.0),
            tilted=True,
            T=4,
            obj_type="pure_phase",
            rng=0,
        )
        ptycho, _ = inr_ptychography(obj)
        ptycho.reconstruct(
            num_iters=60,
            optimizer_params={
                "object": {
                    "grids": OptimizerParams.Adam(lr=_LR_GRIDS),
                    "sigma_net": OptimizerParams.Adam(lr=_LR_SIGMA),
                    "so3": OptimizerParams.Adam(lr=1e-3),
                }
            },
            batch_size=200,
        )
        losses = np.array(ptycho._iter_losses)
        assert np.isfinite(losses).all()
        assert losses[-1] < losses[0]

    def test_tv_constraint_reconstruct_runs(self):
        """A reconstruct with the in-plane TV soft constraint runs and stays finite."""
        ptycho, _ = _build_kplanes_ptycho()
        ptycho.reconstruct(
            num_iters=20,
            optimizer_params=_pplr_params(),
            constraints={"object": {"tv_weight_xy": 1e-3}},
            batch_size=200,
        )
        losses = np.array(ptycho._iter_losses)
        assert np.isfinite(losses).all()
        assert ptycho.obj_model.constraints.tv_weight_xy == 1e-3

    def test_scheduler_scales_pplr_groups(self):
        ptycho, _ = _build_kplanes_ptycho()
        ptycho.reconstruct(
            num_iters=10,
            optimizer_params=_pplr_params(),
            scheduler_params={"object": SchedulerParams.CosineAnnealing()},
            batch_size=200,
        )
        # one cosine schedule drives both param groups; both LRs have decayed below their start
        opt = ptycho.obj_model.optimizer
        assert opt is not None
        lrs = sorted(pg["lr"] for pg in opt.param_groups)
        assert lrs[0] < _LR_SIGMA and lrs[1] < _LR_GRIDS

    def test_save_load_roundtrip(self, tmp_path):
        ptycho, _ = _build_kplanes_ptycho()
        ptycho.reconstruct(
            num_iters=20,
            optimizer_params=_pplr_params(),
            batch_size=200,
        )
        obj_before = ptycho.obj.copy()
        path = tmp_path / "kplanes_ptycho.zip"
        ptycho.save(path, mode="o", save_raw_data=True)  # persist dset so loaded.dset works
        loaded = autoserialize_load(path)
        assert loaded.obj_model.is_implicit is True
        assert loaded.dset.implicit_object is True
        assert loaded.obj_model.name == "ObjectTensorDecomp"
        np.testing.assert_allclose(loaded.obj, obj_before, rtol=1e-5, atol=1e-6)
        # continued training still runs after reload
        loaded.reconstruct(
            num_iters=5,
            optimizer_params=_pplr_params(),
            batch_size=200,
        )
