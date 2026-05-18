import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split

from quantem.tomography.dataset_models import DatasetModelType


def worker_init_fn(worker_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


class DDPMixin:
    """
    Class for setting up all distributed training.

    -
    """

    def setup_distributed(self, device: str | torch.device | None = None):
        """
        Initializes parameters depending if multiple-GPU training, single-GPU training, or CPU training.
        """
        if "RANK" in os.environ:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://"
                )

            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
        else:
            self.world_size = 1
            self.global_rank = 0
            self.local_rank = 0

            if torch.cuda.is_available():
                device = torch.device("cuda:0" if device is None else device)
                torch.cuda.set_device(device.index)
            else:
                device = torch.device("cpu")

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.device = device

    def setup_dataloader(
        self,
        dataset: Dataset | DatasetModelType,
        batch_size: int,
        num_workers: int = 0,
        val_fraction: float = 0.0,
    ):
        pin_mem = self.device.type == "cuda"
        persist = num_workers > 0

        if val_fraction > 0.0:
            train_dataset, val_dataset = random_split(dataset, [1 - val_fraction, val_fraction])  # type: ignore[reportArgumentType] --> dataset inherits from torch Dataset so this is fine.
        else:
            train_dataset = dataset
            val_dataset = None

        if self.world_size > 1:
            shuffle = True
            train_sampler = DistributedSampler(
                train_dataset,  # type: ignore[reportArgumentType] --> Torch datasets do not have a len method, but still works.
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=shuffle,
            )

            if val_dataset:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.global_rank,
                    shuffle=False,
                )
            else:
                val_sampler = None
            shuffle = False

        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        train_dataloader = DataLoader(
            train_dataset,  # type: ignore[reportArgumentType] --> Torch datasets do not have a len method, but still works.
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
            shuffle=shuffle,
            pin_memory=pin_mem,
            drop_last=True,
            persistent_workers=persist,
            multiprocessing_context="spawn",
            worker_init_fn=worker_init_fn,
        )

        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size * 4,
                num_workers=num_workers,
                sampler=val_sampler,
                shuffle=False,
                pin_memory=pin_mem,
                drop_last=False,
                persistent_workers=persist,
                multiprocessing_context="spawn",
                worker_init_fn=worker_init_fn,
            )
            val_dataloader = val_dataloader
        else:
            val_dataloader = None

        if self.global_rank == 0:
            print("Dataloader setup complete:")
            print(f"  Total train samples: {len(train_dataset)}")  # pyright: ignore[reportArgumentType] --> Torch datasets do not have a len method, but still works.
            print(f"  Local batch size: {batch_size}")
            print(f"  Global batch size: {batch_size * self.world_size}")
            print(f"  Train batches per GPU per epoch: {len(train_dataloader)}")

            if val_dataset:
                print(f"  Total val samples: {len(val_dataset)}")
                print(f"  Val batches per GPU per epoch: {len(val_dataloader)}")  # pyright: ignore[reportArgumentType] --> Torch datasets do not have a len method, but still works.

        return train_dataloader, train_sampler, val_dataloader, val_sampler

    def distribute_model(
        self,
        model: nn.Module,
    ) -> nn.Module | nn.parallel.DistributedDataParallel:
        """
        Wraps the model with DistributedDataParallel if mulitple GPUs are available.

        Returns the model.
        """
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
                print("Model wrapped with DDP and compiled")

        if self.world_size > 1:
            if self.global_rank == 0:
                print("Model built, distributed, and compiled successfully")

        else:
            print("Model built, compiled successfully")

        return model

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device | str):
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
