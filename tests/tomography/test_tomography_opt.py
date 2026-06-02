"""Tests for the tomography optimizer / scheduler wiring (``TomographyOpt``).

This is the surface the PPLR ``OptimizerParamsType`` / ``SchedulerParamsType`` refactor
touched. In particular, ``test_set_optimizers_builds_object_and_pose`` and the PPLR test
regression-guard the pose-optimizer path: ``TomographyDatasetBase.get_optimization_parameters``
must return a ``dict[str, list[tensor]]`` (it previously returned a ``list`` and crashed
``set_optimizer`` with ``TypeError: unhashable type: 'dict'``).

Construction only -- no forward passes -- so these run on CPU under CI.
"""

import numpy as np
import pytest

from quantem.core.ml.optimizer_mixin import OptimizerParams, SchedulerParams
from quantem.tomography.dataset_models import TomographyINRDataset
from quantem.tomography.object_models import ObjectINR, ObjectTensorDecomp
from quantem.tomography.tomography import Tomography

from .conftest import requires_torch


def _tilts(nang=5, n=12):
    rng = np.random.default_rng(0)
    angles = np.linspace(-60, 60, nang).astype(np.float32)
    stack = rng.random((nang, n, n)).astype(np.float32)
    return stack, angles


def _inr_tomography(device):
    from quantem.core.ml.inr import HSiren

    model = HSiren(in_features=3, out_features=1, hidden_layers=1, hidden_features=8)
    obj = ObjectINR.from_model(model, shape=(16, 16, 16), device=device)
    stack, angles = _tilts()
    dset = TomographyINRDataset.from_data(stack, angles)
    return Tomography.from_models(dset=dset, obj_model=obj, device=device, verbose=False)


def _td_tomography(device):
    from quantem.core.ml.models.kplanes import KPlanesTILTED

    model = KPlanesTILTED(
        M_features=2, resolution=(16, 16, 16), multiscale_res_multipliers=[1], T=2
    )
    obj = ObjectTensorDecomp.from_model(model, shape=(16, 16, 16), device=device)
    stack, angles = _tilts()
    dset = TomographyINRDataset.from_data(stack, angles)
    return Tomography.from_models(dset=dset, obj_model=obj, device=device, verbose=False)


@requires_torch
class TestOptimizerParams:
    def test_setter_getter_roundtrip(self, torch_device):
        tomo = _inr_tomography(torch_device)
        tomo.optimizer_params = {
            "object": OptimizerParams.Adam(lr=1e-3),
            "pose": OptimizerParams.Adam(lr=1e-2),
        }
        assert set(tomo.optimizer_params.keys()) == {"object", "pose"}

    def test_unknown_key_raises(self, torch_device):
        tomo = _inr_tomography(torch_device)
        with pytest.raises(ValueError):
            tomo.optimizer_params = {"banana": OptimizerParams.Adam(lr=1e-3)}

    def test_set_optimizers_builds_object_and_pose(self, torch_device):
        """Regression guard: the pose path must not raise (see module docstring)."""
        tomo = _inr_tomography(torch_device)
        tomo.optimizer_params = {
            "object": OptimizerParams.Adam(lr=1e-3),
            "pose": OptimizerParams.Adam(lr=1e-2),
        }
        tomo.set_optimizers()
        assert set(tomo.optimizers.keys()) == {"object", "pose"}

    def test_current_lrs(self, torch_device):
        tomo = _inr_tomography(torch_device)
        tomo.optimizer_params = {
            "object": OptimizerParams.Adam(lr=1e-3),
            "pose": OptimizerParams.Adam(lr=1e-2),
        }
        tomo.set_optimizers()
        lrs = tomo.get_current_lrs()
        assert set(lrs.keys()) == {"object", "pose"}
        assert lrs["object"] == pytest.approx(1e-3)
        assert lrs["pose"] == pytest.approx(1e-2)

    def test_remove_optimizer(self, torch_device):
        tomo = _inr_tomography(torch_device)
        tomo.optimizer_params = {
            "object": OptimizerParams.Adam(lr=1e-3),
            "pose": OptimizerParams.Adam(lr=1e-2),
        }
        tomo.set_optimizers()
        tomo.remove_optimizer("object")
        assert "object" not in tomo.optimizers
        assert "pose" in tomo.optimizers

    def test_pplr_object_groups(self, torch_device):
        """Per-parameter LR: object optimizer carries one torch param group per key."""
        tomo = _td_tomography(torch_device)
        tomo.optimizer_params = {
            "object": {
                "grids": OptimizerParams.Adam(lr=1e-2),
                "sigma_net": OptimizerParams.Adam(lr=1e-3),
                "so3": OptimizerParams.Adam(lr=1e-2),
            },
            "pose": OptimizerParams.Adam(lr=1e-2),
        }
        tomo.set_optimizers()
        assert len(tomo.optimizers["object"].param_groups) == 3
        assert "pose" in tomo.optimizers

    def test_step_optimizers_steps_each_once(self, torch_device, monkeypatch):
        """Regression: with object+pose, each optimizer must step exactly once per call.

        ``step_optimizers`` previously looped over both keys and stepped *both* optimizers on
        every pass, so each took two Adam steps per batch.
        """
        tomo = _inr_tomography(torch_device)
        tomo.optimizer_params = {
            "object": OptimizerParams.Adam(lr=1e-3),
            "pose": OptimizerParams.Adam(lr=1e-2),
        }
        tomo.set_optimizers()
        assert set(tomo.optimizers.keys()) == {"object", "pose"}  # both optimizers live

        counts = {"object": 0, "pose": 0}
        monkeypatch.setattr(
            tomo.obj_model,
            "step_optimizer",
            lambda: counts.__setitem__("object", counts["object"] + 1),
        )
        monkeypatch.setattr(
            tomo.dset, "step_optimizer", lambda: counts.__setitem__("pose", counts["pose"] + 1)
        )
        tomo.step_optimizers()
        assert counts == {"object": 1, "pose": 1}

    def test_zero_grad_all_zeros_each_once(self, torch_device, monkeypatch):
        """Companion to the step regression: zero_grad_all must touch each optimizer once."""
        tomo = _inr_tomography(torch_device)
        tomo.optimizer_params = {
            "object": OptimizerParams.Adam(lr=1e-3),
            "pose": OptimizerParams.Adam(lr=1e-2),
        }
        tomo.set_optimizers()
        counts = {"object": 0, "pose": 0}
        monkeypatch.setattr(
            tomo.obj_model,
            "zero_optimizer_grad",
            lambda: counts.__setitem__("object", counts["object"] + 1),
        )
        monkeypatch.setattr(
            tomo.dset,
            "zero_optimizer_grad",
            lambda: counts.__setitem__("pose", counts["pose"] + 1),
        )
        tomo.zero_grad_all()
        assert counts == {"object": 1, "pose": 1}


@requires_torch
class TestSchedulerParams:
    def test_scheduler_setter_getter(self, torch_device):
        tomo = _inr_tomography(torch_device)
        tomo.scheduler_params = {
            "object": SchedulerParams.CosineAnnealing(T_max=10),
            "pose": SchedulerParams.CosineAnnealing(T_max=10),
        }
        assert set(tomo.scheduler_params.keys()) == {"object", "pose"}

    def test_set_schedulers_builds(self, torch_device):
        tomo = _inr_tomography(torch_device)
        tomo.optimizer_params = {
            "object": OptimizerParams.Adam(lr=1e-3),
            "pose": OptimizerParams.Adam(lr=1e-2),
        }
        tomo.set_optimizers()
        tomo.scheduler_params = {
            "object": SchedulerParams.CosineAnnealing(T_max=10),
            "pose": SchedulerParams.CosineAnnealing(T_max=10),
        }
        tomo.set_schedulers(tomo.scheduler_params, num_iter=10)
        assert set(tomo.schedulers.keys()) == {"object", "pose"}

    def test_bad_scheduler_type_raises(self, torch_device):
        tomo = _inr_tomography(torch_device)
        with pytest.raises(TypeError):
            tomo.obj_model.scheduler_params = 123

    def test_setter_does_not_mutate_caller_dict(self, torch_device):
        """Regression: the setter must not inject missing keys into the caller's dict."""
        tomo = _inr_tomography(torch_device)
        d = {"object": SchedulerParams.CosineAnnealing(T_max=10)}
        tomo.scheduler_params = d
        assert set(d.keys()) == {"object"}  # "pose" must not have been added to the input
