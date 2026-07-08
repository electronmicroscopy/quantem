"""Tests for the ``Tomography`` / ``TomographyConventional`` orchestrators and the shared
``TomographyBase`` plumbing, without running a reconstruction.

The ``TomographyConventional`` path uses ``ObjectPixelated`` (no DDP setup), so it builds on
CPU and is always-on -- this exercises the bulk of ``tomography_base.py`` (factory, property
setters/validation, loss accessors). The ``Tomography`` (INR) factory and ``save_volume`` go
through ``setup_distributed`` and so follow the ``torch_device`` fixture under
``requires_torch`` (build on CUDA when present; see conftest).
"""

import numpy as np
import pytest
import torch

from quantem.tomography.dataset_models import TomographyINRDataset, TomographyPixDataset
from quantem.tomography.object_models import ObjConstraintParams, ObjectINR, ObjectPixelated
from quantem.tomography.tomography import Tomography, TomographyConventional
from quantem.tomography.tomography_lite import TomographyLiteINR

from .conftest import requires_torch


def _stack(nang=5, n=12, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((nang, n, n)) * 10).astype(np.float32)


def _conventional(n=12):
    angles = np.linspace(-60, 60, 5).astype(np.float32)
    dset = TomographyPixDataset.from_data(_stack(nang=5, n=n), angles)
    obj = ObjectPixelated.from_uniform(shape=(n, n, n), device="cpu")
    return TomographyConventional.from_models(
        dset=dset, obj_model=obj, device="cpu", verbose=False
    )


class TestConventionalFactory:
    def test_from_models_builds(self):
        tomo = _conventional()
        assert isinstance(tomo, TomographyConventional)
        assert isinstance(tomo.obj_model, ObjectPixelated)
        assert isinstance(tomo.dset, TomographyPixDataset)
        assert tomo.num_epochs == 0

    def test_direct_init_requires_token(self):
        with pytest.raises(RuntimeError):
            TomographyConventional(
                dset=TomographyPixDataset.from_data(
                    _stack(), np.linspace(-60, 60, 5).astype(np.float32)
                ),
                obj_model=ObjectPixelated.from_uniform(shape=(12, 12, 12), device="cpu"),
            )


class TestBaseProperties:
    def test_constraints_setter_dict_and_object(self):
        tomo = _conventional()
        tomo.constraints = {"name": "obj_pixelated", "tv_vol": 0.02}
        assert isinstance(tomo.constraints, ObjConstraintParams.ObjPixelatedConstraints)
        assert tomo.constraints.tv_vol == 0.02
        obj_c = ObjConstraintParams.ObjPixelatedConstraints(positivity=True)
        tomo.constraints = obj_c
        assert tomo.constraints is obj_c

    def test_constraints_setter_none_is_noop(self):
        tomo = _conventional()
        before = tomo.constraints
        tomo.constraints = None
        assert tomo.constraints is before

    def test_constraints_setter_invalid_raises(self):
        tomo = _conventional()
        with pytest.raises(ValueError):
            tomo.constraints = 1.0

    def test_logger_setter_rejects_wrong_type(self):
        tomo = _conventional()
        with pytest.raises(TypeError):
            tomo.logger = "not a logger"

    def test_dset_setter_rejects_wrong_type(self):
        tomo = _conventional()
        with pytest.raises(TypeError):
            tomo.dset = object()

    def test_loss_accessors_start_empty(self):
        tomo = _conventional()
        assert tomo.epoch_losses.shape == (0,)
        assert tomo.consistency_losses.shape == (0,)
        assert tomo.learning_rates == {}

    def test_append_learning_rates_accumulates(self):
        tomo = _conventional()
        tomo.append_learning_rates({"object": 1e-3, "pose": 1e-2})
        tomo.append_learning_rates({"object": 5e-4, "pose": 5e-3})
        assert tomo.learning_rates["object"] == [1e-3, 5e-4]
        assert tomo.learning_rates["pose"] == [1e-2, 5e-3]

    def test_to_updates_device(self):
        tomo = _conventional()
        tomo.to("cpu")
        assert torch.device(tomo.device) == torch.device("cpu")

    def test_plot_losses_runs(self):
        tomo = _conventional()
        tomo._epoch_losses.extend([1.0, 0.5, 0.25])
        tomo.plot_losses()  # Agg backend; plt.show() is a no-op


@requires_torch
class TestInrFactory:
    def _inr_tomo(self, device, n=16):
        from quantem.core.ml.inr import HSiren

        model = HSiren(in_features=3, out_features=1, hidden_layers=1, hidden_features=8)
        obj = ObjectINR.from_model(model, shape=(n, n, n), device=device)
        dset = TomographyINRDataset.from_data(
            _stack(nang=5, n=n), np.linspace(-60, 60, 5).astype(np.float32)
        )
        return Tomography.from_models(dset=dset, obj_model=obj, device=device, verbose=False)

    def test_from_models_builds(self, torch_device):
        tomo = self._inr_tomo(torch_device)
        assert isinstance(tomo, Tomography)
        assert isinstance(tomo.obj_model, ObjectINR)

    def test_plot_losses_runs(self, torch_device):
        tomo = self._inr_tomo(torch_device)
        tomo._epoch_losses.extend([1.0, 0.5])
        tomo._lrs["object"] = [1e-3, 5e-4]
        tomo.plot_losses()

    def test_save_volume_overwrite_guard(self, torch_device, tmp_path):
        tomo = self._inr_tomo(torch_device)
        path = str(tmp_path / "vol.npz")
        tomo.save_volume(path)
        assert (tmp_path / "vol.npz").exists()
        with pytest.raises(FileExistsError):
            tomo.save_volume(path)
        tomo.save_volume(path, overwrite=True)  # must not raise
        with np.load(path) as data:
            assert "volume" in data


@requires_torch
class TestLiteINRReconstructBranch:
    """``TomographyLiteINR.reconstruct`` bundles optimizer/scheduler params only on the first
    epoch and passes ``None`` afterwards. Stub out the heavy ``Tomography.reconstruct`` to
    assert the branch without running a reconstruction."""

    def _lite(self, device, n=12):
        return TomographyLiteINR.from_dataset(
            tilt_series=_stack(nang=5, n=n),
            tilt_angles=np.linspace(-60, 60, 5).astype(np.float32),
            device=device,
        )

    def test_param_bundling_first_then_subsequent(self, torch_device, monkeypatch):
        tomo = self._lite(torch_device)
        captured = {}

        def fake_reconstruct(self, **kwargs):
            captured.clear()
            captured.update(kwargs)
            self._epoch_losses.append(1.0)  # mark an epoch as having run

        monkeypatch.setattr(Tomography, "reconstruct", fake_reconstruct)

        # First call (num_epochs == 0): object + pose params are assembled.
        tomo.reconstruct(num_iter=1, num_workers=0, learn_pose=True)
        assert set(captured["optimizer_params"].keys()) == {"object", "pose"}
        assert set(captured["scheduler_params"].keys()) == {"object", "pose"}

        # Second call (num_epochs > 0): params are passed through as None.
        tomo.reconstruct(num_iter=1, num_workers=0)
        assert captured["optimizer_params"] is None
        assert captured["scheduler_params"] is None
