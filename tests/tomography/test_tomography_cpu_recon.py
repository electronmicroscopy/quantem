"""CPU INR reconstruction smoke test.

The GPU INR tests (``test_tomography_inr.py``) are gated behind ``requires_gpu``, so on
CPU-only CI the entire learned-reconstruction loop in ``Tomography.reconstruct`` is never
exercised. This runs a tiny INR reconstruction on CPU with ``num_workers=0`` -- valid since
``DDPMixin.setup_dataloader`` now only sets ``multiprocessing_context`` when
``num_workers > 0`` -- so that path is covered in CI.

It is skipped when CUDA is present: object models that go through ``setup_distributed`` must
be built on CUDA when a CUDA device exists (the CPU path is only valid with no CUDA device),
and on GPU machines the ``requires_gpu`` tests already cover this path. Marked ``slow`` so it
runs under ``--runslow``.
"""

import numpy as np
import pytest
import torch

from quantem.tomography.tomography_lite import TomographyLiteINR

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        torch.cuda.is_available(),
        reason="CPU INR path is only valid without CUDA; GPU machines use the requires_gpu tests",
    ),
]


def test_cpu_inr_reconstruct_reduces_loss():
    rng = np.random.default_rng(0)
    n = 12
    series = (rng.random((5, n, n)) * 10).astype(np.float32)
    angles = np.linspace(-60, 60, 5).astype(np.float32)

    tomo = TomographyLiteINR.from_dataset(tilt_series=series, tilt_angles=angles, device="cpu")
    tomo.reconstruct(num_iter=2, num_workers=0, batch_size=64, learn_pose=True)

    losses = tomo.epoch_losses
    assert len(losses) == 2
    assert losses[-1] < losses[0]
    view = tomo.obj_model.obj_view
    assert view.shape == (1, n, n, n)
    assert np.isfinite(view).all()
