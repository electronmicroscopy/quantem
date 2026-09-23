"""Multi-GPU state-management tests for iterative ptychography.

Ported from the standalone verify_multi_gpu.py script. Tests are grouped by
scenario (one test per related cluster of assertions) so that the spawn
overhead is paid once per scenario rather than once per assertion.

All tests are marked ``slow`` and skipped when fewer than 2 CUDA devices are
available. Run with: ``uv run pytest tests/diffractive_imaging/test_multi_gpu.py --runslow``.
"""

import numpy as np
import pytest
import torch

# Helpers must live at module scope so forkserver-spawned DataLoader / DDP workers
# can pickle and re-import them.

N_SCAN = 8
N_DET = 32
PROBE_ENERGY = 80e3
PROBE_SEMIANGLE = 20
PROBE_DEFOCUS = 100
N_ITERS = 4
GPU_IDS = [0, 1]
DEVICE_0 = "cuda:0"


def _make_dataset():
    from quantem.core.datastructures import Dataset4dstem

    rng = np.random.default_rng(42)
    array = rng.random((N_SCAN, N_SCAN, N_DET, N_DET)).astype(np.float32)
    return Dataset4dstem.from_array(
        array,
        name="test",
        sampling=[1.0, 1.0, 0.05, 0.05],
        units=["A", "A", "A^-1", "A^-1"],
    )


def _make_ptycho():
    from quantem.diffractive_imaging import (
        DetectorPixelated,
        ObjectPixelated,
        ProbePixelated,
        Ptychography,
        PtychographyDatasetRaster,
    )

    pdset = PtychographyDatasetRaster.from_dataset4dstem(_make_dataset())
    pdset.preprocess(com_fit_function="constant", plot_rotation=False, plot_com=False)
    obj = ObjectPixelated.from_uniform(obj_type="pure_phase", num_slices=1)
    probe = ProbePixelated.from_params(
        probe_params={
            "energy": PROBE_ENERGY,
            "defocus": PROBE_DEFOCUS,
            "semiangle_cutoff": PROBE_SEMIANGLE,
        }
    )
    ptycho = Ptychography.from_models(
        dset=pdset,
        obj_model=obj,
        probe_model=probe,
        detector_model=DetectorPixelated(),
        verbose=False,
        rng=42,
    )
    ptycho.preprocess(obj_padding_px=(4, 4))
    return ptycho


def _make_dip_ptycho():
    from quantem.core.ml import OptimizerParams
    from quantem.diffractive_imaging import PtychoLite, PtychoLiteDIP

    base = _make_ptycho()
    base.reconstruct(
        num_iters=5,
        reset=True,
        optimizer_params={
            "object": OptimizerParams.Adam(lr=1e-2),
            "probe": OptimizerParams.Adam(lr=1e-2),
        },
        batch_size=16,
        device=0,
    )
    lite = PtychoLite.from_models(
        dset=base.dset,
        obj_model=base.obj_model,
        probe_model=base.probe_model,
        detector_model=base.detector_model,
        verbose=False,
        rng=42,
    )
    lite.preprocess(obj_padding_px=(4, 4))
    return PtychoLiteDIP.from_ptycholite(lite, device="cpu", pretrain_iters=None)


def _opt():
    from quantem.core.ml import OptimizerParams

    return {
        "object": OptimizerParams.Adam(lr=1e-2),
        "probe": OptimizerParams.Adam(lr=1e-2),
    }


# Module-level marks: all tests in this file are slow and require >= 2 GPUs.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="requires >= 2 CUDA devices",
    ),
]


class TestSingleGPUDevicePersistence:
    """device= argument is sticky across follow-up reconstruct() calls."""

    def test_int_device_persists(self):
        p = _make_ptycho()
        p.reconstruct(
            num_iters=N_ITERS, reset=True, optimizer_params=_opt(), batch_size=16, device=0
        )
        assert p.device == DEVICE_0
        assert p.obj_model._obj.device.type == "cuda"

        # follow-up call without device= keeps the previous device
        p.reconstruct(num_iters=N_ITERS, batch_size=16)
        assert p.device == DEVICE_0
        assert p.obj_model._obj.device.type == "cuda"

        # reset=True must not reset device tracking
        p.reconstruct(num_iters=N_ITERS, reset=True, batch_size=16)
        assert p.device == DEVICE_0


class TestMultiGPUDeviceRestoration:
    """device=[…] is stored and restored across spawn boundaries."""

    def test_gpu_list_persists(self):
        p = _make_ptycho()
        p.reconstruct(
            num_iters=N_ITERS, reset=True, optimizer_params=_opt(), batch_size=16, device=GPU_IDS
        )
        assert p.device == GPU_IDS
        assert p.obj_model._obj.device.type == "cuda"

        p.reconstruct(num_iters=N_ITERS, batch_size=16)
        assert p.device == GPU_IDS


class TestMultiGPULossTracking:
    """_iter_losses extends correctly across spawn(reset=False)/reset=True."""

    def test_losses_length_lifecycle(self):
        p = _make_ptycho()
        p.reconstruct(
            num_iters=N_ITERS, reset=True, optimizer_params=_opt(), batch_size=16, device=GPU_IDS
        )
        assert len(p._iter_losses) == N_ITERS

        p.reconstruct(num_iters=N_ITERS, reset=False, batch_size=16, device=GPU_IDS)
        assert len(p._iter_losses) == 2 * N_ITERS, "continuation must not double-count"

        p.reconstruct(num_iters=N_ITERS, reset=True, batch_size=16, device=GPU_IDS)
        assert len(p._iter_losses) == N_ITERS


class TestMultiGPULRTracking:
    """iter_lrs extends correctly across spawn boundaries."""

    def test_iter_lrs_lifecycle(self):
        p = _make_ptycho()
        p.reconstruct(
            num_iters=N_ITERS, reset=True, optimizer_params=_opt(), batch_size=16, device=GPU_IDS
        )
        assert "object" in p.iter_lrs
        assert len(p.iter_lrs["object"]) == N_ITERS

        p.reconstruct(num_iters=N_ITERS, reset=False, batch_size=16, device=GPU_IDS)
        assert len(p.iter_lrs["object"]) == 2 * N_ITERS

        p.reconstruct(num_iters=N_ITERS, reset=True, batch_size=16, device=GPU_IDS)
        assert len(p.iter_lrs["object"]) == N_ITERS


class TestMultiGPUOptimizerState:
    """Adam state survives the save/restore around the spawn worker."""

    def test_adam_state_restored_on_device(self):
        p = _make_ptycho()
        p.reconstruct(
            num_iters=N_ITERS, reset=True, optimizer_params=_opt(), batch_size=16, device=GPU_IDS
        )
        obj_opt = p.optimizers.get("object")
        assert obj_opt is not None
        assert len(obj_opt.state) > 0
        first = next(iter(obj_opt.state.values()))
        assert "exp_avg" in first, "Adam moments missing"
        assert first["exp_avg"].device.type == "cuda"


class TestDIPMultiGPU:
    """DIP path mirrors the pixelated multi-GPU contract."""

    def test_dip_device_and_loss_lifecycle(self):
        d = _make_dip_ptycho()
        d.reconstruct(
            num_iters=N_ITERS,
            reset=True,
            lr_obj=1e-3,
            lr_probe=1e-3,
            batch_size=16,
            device=GPU_IDS,
        )
        assert d.device == GPU_IDS
        assert len(d._iter_losses) == N_ITERS
        assert "object" in d.iter_lrs
        assert len(d.iter_lrs["object"]) == N_ITERS

        d.reconstruct(
            num_iters=N_ITERS,
            reset=False,
            lr_obj=1e-3,
            lr_probe=1e-3,
            batch_size=16,
            device=GPU_IDS,
        )
        assert len(d._iter_losses) == 2 * N_ITERS, "DIP continuation must not double-count"

        d.reconstruct(
            num_iters=N_ITERS,
            reset=True,
            lr_obj=1e-3,
            lr_probe=1e-3,
            batch_size=16,
            device=GPU_IDS,
        )
        assert len(d._iter_losses) == N_ITERS
