import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from quantem.tomography.tomography_dataset import (
    PretrainVolumeDataset,
    TomographyDataset,
    TomographyRayDataset,
)


class TomographyDDP:
    """
    Initializing DDP stuff for tomo class.
    """

    def __init__(
        self,
    ):
        self.setup_distributed()

    def setup_distributed(self):
        # Check if in distributed env
        if "RANK" in os.environ:
            # Distributed training
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://"
                )

            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])

            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)

        else:
            # Single GPU/CPU training
            self.world_size = 1
            self.global_rank = 0
            self.local_rank = 0

            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                torch.cuda.set_device(0)
                print("Single GPU training")
            else:
                self.device = torch.device("cpu")
                print("CPU training")

        # Optional performance optimizations (only for CUDA)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def build_model(
        self,
        model: nn.Module,
    ):
        # TODO: Generalized model --> Should be instantiated in the object? Where does `HSIREN` get instantiated?
        model = model.to(self.device)

        if self.world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                broadcast_buffers=True,
                bucket_cap_mb=100,
                gradient_as_bucket_view=True,
            )

            if self.global_rank == 0:
                print("Model wrapped with DDP")

        if self.world_size > 1:
            if self.global_rank == 0:
                print("Model built, distributed, and compiled successfully")

        else:
            print("Model built, compiled successfully")

        return model

    # Setup Tomo DataLoader
    def setup_dataloader(
        self,
        tomo_dataset: TomographyDataset,
        batch_size: int,
        num_workers: int = 0,
        val_fraction: float = 0.0,
    ):
        pin_mem = self.device.type == "cuda"
        persist = num_workers > 0

        # Split dataset if validation fraction > 0
        if val_fraction > 0.0:
            # TODO: Temporary for when only doing validation, current TomographyDataset doesn't work correctly
            train_dataset = TomographyRayDataset(
                tomo_dataset.tilt_series.detach().clone(),
                tomo_dataset.tilt_angles.detach().clone(),
                500,  # TODO: TEMPORARY
                val_ratio=val_fraction,
                mode="train",
                seed=42,
            )
            val_dataset = TomographyRayDataset(
                tomo_dataset.tilt_series.detach().clone(),
                tomo_dataset.tilt_angles.detach().clone(),
                500,  # TODO: TEMPORARY
                val_ratio=val_fraction,
                mode="val",
                seed=42,
            )
        else:
            train_dataset, val_dataset = tomo_dataset, None

        # Samplers for distributed training
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
            )
            if val_dataset:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.global_rank,
                    shuffle=False,
                )
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        # Main dataloader
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
            shuffle=shuffle,
            pin_memory=pin_mem,
            drop_last=True,
            persistent_workers=persist,
        )

        # Validation dataloader if applicable
        if val_dataset:
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size * 4,
                num_workers=num_workers,
                sampler=val_sampler,
                shuffle=False,
                pin_memory=pin_mem,
                drop_last=False,
                persistent_workers=persist,
            )
            self.val_sampler = val_sampler

        self.sampler = train_sampler

        if self.global_rank == 0:
            print("Dataloader setup complete:")
            print(f"  Total projections: {len(tomo_dataset.tilt_angles)}")
            if val_fraction > 0.0:
                print(f"  Total projections (val): {len(val_dataset)}")
            print(f"  Grid size: {tomo_dataset.dims[1]}{tomo_dataset.dims[2]}")
            print(f"  Total pixels: {tomo_dataset.num_pixels:,}")
            if val_fraction > 0.0:
                print(f"  Total pixels (val): {len(val_dataset):,}")
                print(f"  Total pixels (train): {len(train_dataset):,}")
            print(f"  Local batch size (train): {batch_size}")
            print(f"  Global batch size: {batch_size * self.world_size}")
            print(f"  Train batches per GPU per epoch: {len(self.dataloader)}")

    # Setup pretraining dataloader

    def setup_pretraining_dataloader(
        self,
        volume_dataset: PretrainVolumeDataset,
        batch_size: int,
    ):
        # TODO: temp

        num_workers = 0
        if self.world_size > 1:
            sampler = DistributedSampler(
                volume_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
                drop_last=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        self.pretraining_dataloader = DataLoader(
            volume_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            persistent_workers=num_workers > 0,
        )

        self.pretraining_sampler = sampler

        if self.global_rank == 0:
            print("Pretraining dataloader setup complete:")
            print(f"  Total samples: {len(volume_dataset)}")
            print(f"  Grid size: {volume_dataset.N**3}")
            print(f"  Local batch size: {batch_size}")
            print(f"  Global batch size: {batch_size * self.world_size}")
            print(f"  Pretraining batches per GPU per epoch: {len(self.pretraining_dataloader)}")

    def get_scaled_lr(self, base_lr, scaling_rule="sqrt"):
        if scaling_rule == "sqrt":
            return base_lr * np.sqrt(self.world_size)
        elif scaling_rule == "linear":
            return base_lr * self.world_size
        else:
            raise ValueError(f"Invalid scaling rule: {scaling_rule}")

    def scale_lr(
        self,
        optimizer_params: dict,
    ):
        new_optimizer_params = {}
        for key, value in optimizer_params.items():
            if "original_lr" in value:
                new_optimizer_params[key] = {
                    "type": value["type"],
                    "lr": self.get_scaled_lr(value["original_lr"]),
                    "original_lr": value["original_lr"],
                }
            else:
                new_optimizer_params[key] = {
                    "type": value["type"],
                    "lr": self.get_scaled_lr(value["lr"]),
                    "original_lr": value["lr"],
                }

        return new_optimizer_params

    # TODO: Temporary Adaptive L1 Smooth Loss


class AdaptiveSmoothL1Loss(nn.Module):
    def __init__(self, beta_init=None, ema_factor=0.99, eps=1e-8):
        """
        Adaptive smooth L1 loss with EMA-based beta adaptation.

        Args:
            beta_init (float): optional initial β value; if None, starts as 1.0
            ema_factor (float): smoothing factor for EMA (1 - 1/N_b in Eq. 38)
            eps (float): small constant for numerical stability
        """
        super().__init__()
        self.register_buffer("beta2", torch.tensor(beta_init**2 if beta_init else 1.0))
        self.ema_factor = ema_factor
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        abs_diff = diff.abs()

        # compute current batch MSE (Eq. 38)
        mse_batch = torch.mean(diff**2)

        # update β² adaptively using Eq. (39)
        with torch.no_grad():
            self.beta2 = self.ema_factor * self.beta2 + (1 - self.ema_factor) * torch.min(
                self.beta2, mse_batch
            )

        beta = torch.sqrt(self.beta2 + self.eps)

        # Smooth L1 (Eq. 36)
        loss = torch.where(
            abs_diff < beta, 0.5 * (diff**2) / (beta + self.eps), abs_diff - 0.5 * beta
        )
        return loss.mean()


class AdaptiveSmoothL1LossDDP(AdaptiveSmoothL1Loss):
    def forward(self, pred, target):
        diff = pred - target
        abs_diff = diff.abs()
        mse_batch = torch.mean(diff**2)

        # Synchronize β across all GPUs
        if dist.is_initialized():
            mse_batch_all = mse_batch.clone()
            dist.all_reduce(mse_batch_all, op=dist.ReduceOp.AVG)
            mse_batch = mse_batch_all / dist.get_world_size()

        with torch.no_grad():
            self.beta2 = self.ema_factor * self.beta2 + (1 - self.ema_factor) * torch.min(
                self.beta2, mse_batch
            )

        beta = torch.sqrt(self.beta2 + self.eps)
        loss = torch.where(
            abs_diff < beta, 0.5 * (diff**2) / (beta + self.eps), abs_diff - 0.5 * beta
        )
        return loss.mean()
