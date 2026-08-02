"""End-to-end conventional (SIRT / FBP) reconstruction.

CPU, deterministic, but marked ``slow`` because it runs a (tiny) iterative reconstruction.
Reconstructions are capped at a few iterations: the suite checks behaviour and wiring (loss
decreases, output stays physical) rather than convergence quality, so spatial agreement with
the phantom is only a loose lower bound.
"""

import numpy as np
import pytest

from quantem.tomography.dataset_models import TomographyPixDataset
from quantem.tomography.object_models import ObjConstraintParams, ObjectPixelated
from quantem.tomography.tomography import TomographyConventional

pytestmark = pytest.mark.slow


def _build(tilt_series, tilt_angles, n):
    dset = TomographyPixDataset.from_data(
        tilt_series, tilt_angles, learn_shift=False, learn_tilt_axis=False
    )
    obj = ObjectPixelated.from_uniform(shape=(n, n, n), device="cpu")
    return TomographyConventional.from_models(
        dset=dset, obj_model=obj, device="cpu", verbose=False
    )


class TestSIRT:
    def test_loss_decreases_and_output_physical(self, phantom_volume, tilt_series, tilt_angles):
        n = phantom_volume.shape[0]
        tomo = _build(tilt_series, tilt_angles, n)
        tomo.reconstruct(
            num_iter=4,
            mode="sirt",
            obj_constraints=ObjConstraintParams.ObjPixelatedConstraints(positivity=True),
        )
        losses = tomo.epoch_losses
        assert tomo.num_epochs == 4
        assert losses[-1] < losses[0]
        rec = tomo.obj_model.obj.detach().cpu().numpy()
        assert np.isfinite(rec).all()
        assert rec.min() >= 0.0  # positivity

    def test_recon_correlates_with_phantom(self, phantom_volume, tilt_series, tilt_angles):
        n = phantom_volume.shape[0]
        tomo = _build(tilt_series, tilt_angles, n)
        tomo.reconstruct(
            num_iter=4,
            mode="sirt",
            obj_constraints=ObjConstraintParams.ObjPixelatedConstraints(positivity=True),
        )
        rec = tomo.obj_model.obj.detach().cpu().numpy()
        corr = np.corrcoef(rec.ravel(), phantom_volume.ravel())[0, 1]
        assert corr > 0.15  # loose: only a handful of iterations

    def test_obj_constraints_accepts_dict(self, phantom_volume, tilt_series, tilt_angles):
        n = phantom_volume.shape[0]
        tomo = _build(tilt_series, tilt_angles, n)
        tomo.reconstruct(num_iter=2, mode="sirt", obj_constraints={"name": "obj_pixelated"})
        assert tomo.num_epochs == 2


class TestFBP:
    def test_fbp_runs_single_epoch(self, phantom_volume, tilt_series, tilt_angles):
        n = phantom_volume.shape[0]
        tomo = _build(tilt_series, tilt_angles, n)
        tomo.reconstruct(num_iter=5, mode="fbp")
        # FBP breaks after the first epoch regardless of num_iter.
        assert tomo.num_epochs == 1
        rec = tomo.obj_model.obj.detach().cpu().numpy()
        assert np.isfinite(rec).all()
