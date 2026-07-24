"""Tests for ``quantem.tomography.logger_tomography``.

``LoggerTomography`` is a thin tensorboard wrapper that the reconstruction loop only drives
when a ``log_dir`` is passed, so the end-to-end recon tests never exercise it. These CPU,
always-on tests construct a logger against a ``tmp_path`` and drive each method with small
stubs that expose only the attributes the logger reads, asserting the calls run and write
event files. Matplotlib backend is ``Agg`` (set in the root conftest), so figure logging is
headless.
"""

from types import SimpleNamespace

import numpy as np
import torch

from quantem.tomography.logger_tomography import LoggerTomography


def _make_logger(tmp_path) -> LoggerTomography:
    return LoggerTomography(
        log_dir=str(tmp_path),
        run_prefix="test_tomo",
        run_suffix="",
        log_images_every=1,
    )


def test_init_creates_log_dir(tmp_path):
    logger = _make_logger(tmp_path)
    try:
        assert logger.log_dir.exists()
        assert logger.log_dir.name.startswith("test_tomo_")
    finally:
        logger.close()


def test_log_epoch_writes_events(tmp_path):
    logger = _make_logger(tmp_path)
    try:
        logger.log_epoch(epoch=0, loss=1.0, tilt_series_loss=0.8, soft_loss=0.2)
        logger.flush()
        events = list(logger.log_dir.glob("events.out.tfevents.*"))
        assert events, "log_epoch should have written a tensorboard event file"
    finally:
        logger.close()


def test_log_iter_unpacks_learning_rates(tmp_path):
    logger = _make_logger(tmp_path)
    obj_model = SimpleNamespace(_soft_constraint_losses=[0.3])
    try:
        logger.log_iter(
            object_model=obj_model,
            iter=2,
            consistency_loss=0.5,
            total_loss=0.7,
            learning_rates={"object": 1e-3, "pose": 1e-2},
            num_samples_per_ray=16,
            val_loss=0.4,
        )
        logger.flush()
        assert list(logger.log_dir.glob("events.out.tfevents.*"))
    finally:
        logger.close()


def test_log_iter_without_val_loss(tmp_path):
    logger = _make_logger(tmp_path)
    obj_model = SimpleNamespace(_soft_constraint_losses=[0.1])
    try:
        # val_loss defaults to None -> the val branch must be skipped without error.
        logger.log_iter(
            object_model=obj_model,
            iter=0,
            consistency_loss=0.5,
            total_loss=0.6,
            learning_rates={},
            num_samples_per_ray=8,
        )
        logger.flush()
    finally:
        logger.close()


def test_log_iter_images(tmp_path):
    logger = _make_logger(tmp_path)
    n_tilts = 5
    dataset_model = SimpleNamespace(
        z1_params=torch.linspace(-1.0, 1.0, n_tilts),
        z3_params=torch.linspace(1.0, -1.0, n_tilts),
        shifts_params=torch.zeros(n_tilts, 2),
    )
    pred_volume = np.random.default_rng(0).random((2, 6, 6, 6)).astype(np.float32)
    try:
        logger.log_iter_images(
            pred_volume=pred_volume,
            dataset_model=dataset_model,
            iter=1,
        )
        logger.flush()
        assert list(logger.log_dir.glob("events.out.tfevents.*"))
    finally:
        logger.close()
