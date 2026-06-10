"""Tests for ``DeviceBatchSampler`` and its wiring in ``Tomography.reconstruct``.

The sampler must yield batches identical in content to the per-pixel
DataLoader path (same keys, same index decode as
``TomographyINRDataset.__getitem__``), cover each pixel exactly once per
epoch (minus the dropped tail batch), and respect the train/val index
split. Device-independent, so everything here runs on CPU.
"""

import numpy as np
import torch

from quantem.tomography.dataset_models import DeviceBatchSampler, TomographyINRDataset


def _dset(n_proj=4, n=10, seed=0):
    rng = np.random.default_rng(seed)
    stack = rng.random((n_proj, n, n)).astype(np.float32)
    angles = np.linspace(-60, 60, n_proj).astype(np.float32)
    return TomographyINRDataset.from_data(tilt_stack=stack, tilt_angles=angles)


def test_batches_match_getitem():
    dset = _dset()
    sampler = DeviceBatchSampler(dset, batch_size=37, device="cpu", shuffle=False)
    seen = 0
    for batch in sampler:
        for k in range(len(batch["target_value"])):
            item = dset[seen + k]
            for key in ("projection_idx", "pixel_i", "pixel_j", "phi", "target_value"):
                torch.testing.assert_close(
                    batch[key][k], item[key].to(batch[key].dtype), rtol=0, atol=0
                )
        seen += len(batch["target_value"])


def test_epoch_covers_indices_once_with_drop_last():
    dset = _dset()
    n = len(dset)
    batch_size = 64
    sampler = DeviceBatchSampler(dset, batch_size=batch_size, device="cpu", shuffle=True)
    assert len(sampler) == n // batch_size
    per_proj = dset.tilt_stack.shape[1] * dset.tilt_stack.shape[2]
    flat = []
    for batch in sampler:
        assert len(batch["target_value"]) == batch_size
        flat.append(
            batch["projection_idx"] * per_proj
            + batch["pixel_i"] * dset.tilt_stack.shape[1]
            + batch["pixel_j"]
        )
    flat = torch.cat(flat)
    assert flat.unique().numel() == flat.numel()  # no repeats within an epoch
    assert flat.numel() == len(sampler) * batch_size


def test_shuffle_changes_order_between_epochs():
    dset = _dset()
    sampler = DeviceBatchSampler(dset, batch_size=50, device="cpu", shuffle=True)
    first = next(iter(sampler))["target_value"]
    second = next(iter(sampler))["target_value"]
    assert not torch.equal(first, second)


def test_val_split_is_disjoint():
    dset = _dset()
    n = len(dset)
    perm = torch.randperm(n)
    n_val = n // 10
    train = DeviceBatchSampler(dset, 32, "cpu", indices=perm[n_val:])
    val = DeviceBatchSampler(dset, 32, "cpu", indices=perm[:n_val], shuffle=False)
    assert len(train._indices) + len(val._indices) == n
    assert torch.cat([train._indices, val._indices]).unique().numel() == n


def test_reconstruct_uses_sampler_single_process():
    """Smoke: the single-process reconstruct path builds DeviceBatchSamplers.

    Object models that go through ``setup_distributed`` must be built on CUDA
    when a CUDA device exists, so pick the device accordingly.
    """
    from quantem.core.ml.models.kplanes import KPlanesTILTED
    from quantem.core.ml.optimizer_mixin import OptimizerParams
    from quantem.tomography.object_models import ObjectINR
    from quantem.tomography.tomography import Tomography

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dset = _dset(n_proj=3, n=8)
    model = KPlanesTILTED(M_features=2, resolution=(8, 8, 8), multiscale_res_multipliers=[1], T=1)
    obj = ObjectINR.from_model(model, shape=(8, 8, 8), device=device)
    tomo = Tomography.from_models(dset=dset, obj_model=obj, device=device, verbose=False)
    tomo.reconstruct(
        num_iter=1,
        batch_size=32,
        num_workers=0,
        val_fraction=0.25,
        optimizer_params={
            "object": {"default": OptimizerParams.Adam(lr=1e-3)},
            "pose": OptimizerParams.Adam(lr=1e-3),
        },
    )
    assert isinstance(tomo.dataloader, DeviceBatchSampler)
    assert isinstance(tomo.val_dataloader, DeviceBatchSampler)
    assert tomo.sampler is None
