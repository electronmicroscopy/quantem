"""End-to-end INR / KPlanes (tensor-decomposition) reconstruction.

These exercise the full learned-reconstruction path (model + pose optimisation, autocast,
spawned dataloader workers), so they are gated behind ``requires_gpu`` and ``slow`` and only
run locally with ``--runslow``. Reconstructions are capped at 4 iterations; the assertion is
loss-decreases plus finite output, not convergence quality.

The ``num_workers=2`` is required, not incidental: ``setup_dataloader`` hard-codes
``multiprocessing_context="spawn"``, which is invalid with ``num_workers=0``.
"""

import numpy as np
import pytest
import torch

from quantem.tomography.dataset_models import TomographyINRDataset
from quantem.tomography.object_models import ObjectTensorDecomp
from quantem.tomography.tomography import Tomography
from quantem.tomography.tomography_lite import TomographyLiteConv, TomographyLiteINR

from .conftest import make_tilt_series, requires_gpu

pytestmark = [requires_gpu, pytest.mark.slow]

DEVICE = "cuda:0"


@pytest.fixture(scope="module")
def small_phantom():
    vol = torch.zeros(1, 24, 24, 24)
    vol[0, 6:18, 6:14, 8:16] = 1.0
    angles = np.linspace(-60, 60, 7).astype(np.float32)
    series = make_tilt_series(vol[0], angles)
    return series, angles


class TestLiteINR:
    def test_reconstruct_reduces_loss(self, small_phantom):
        series, angles = small_phantom
        tomo = TomographyLiteINR.from_dataset(
            tilt_series=series, tilt_angles=angles, device=DEVICE
        )
        tomo.reconstruct(num_iter=4, num_workers=2, batch_size=256, learn_pose=True)
        losses = tomo.epoch_losses
        assert len(losses) == 4
        assert losses[-1] < losses[0]
        view = tomo.obj_model.obj_view
        assert view.shape == (1, 24, 24, 24)
        assert np.isfinite(view).all()


class TestLiteConv:
    def test_smoke(self, small_phantom):
        series, angles = small_phantom
        tomo = TomographyLiteConv.from_dataset(
            tilt_series=series, tilt_angles=angles, device=DEVICE
        )
        tomo.reconstruct(num_iter=3, mode="sirt")
        assert tomo.num_epochs == 3
        assert np.isfinite(tomo.obj_model.obj.detach().cpu().numpy()).all()


class TestKPlanes:
    def test_pplr_reconstruct_reduces_loss(self, small_phantom):
        from quantem.core.ml.models.kplanes import KPlanesTILTED
        from quantem.core.ml.optimizer_mixin import OptimizerParams, SchedulerParams

        series, angles = small_phantom
        n = series.shape[1]
        model = KPlanesTILTED(
            M_features=2, resolution=(n, n, n), multiscale_res_multipliers=[1], T=2
        )
        obj = ObjectTensorDecomp.from_model(model, shape=(n, n, n), device=DEVICE)
        dset = TomographyINRDataset.from_data(series, angles)
        tomo = Tomography.from_models(dset=dset, obj_model=obj, device=DEVICE, verbose=False)
        tomo.reconstruct(
            optimizer_params={
                "object": {
                    "grids": OptimizerParams.Adam(lr=1e-2),
                    "sigma_net": OptimizerParams.Adam(lr=1e-3),
                    "so3": OptimizerParams.Adam(lr=1e-2),
                },
                "pose": OptimizerParams.Adam(lr=1e-2),
            },
            scheduler_params={
                "object": SchedulerParams.CosineAnnealing(T_max=4),
                "pose": SchedulerParams.CosineAnnealing(T_max=4),
            },
            num_iter=4,
            batch_size=256,
            num_samples_per_ray=20,
            num_workers=2,
        )
        losses = tomo.epoch_losses
        assert len(losses) == 4
        assert losses[-1] < losses[0]
