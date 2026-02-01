import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split


class DDPMixin:
    """
    Class for setting up all distributed training.

    -
    """

    def __init__(
        self,
    ):
        self.setup_distributed()

    def setup_distributed(self, device: str | None = None):
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
                print("Single GPU training")
            else:
                device = torch.device("cpu")
                print("CPU training")

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.device = device

    def setup_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
        val_fraction: float = 0.0,
    ):
        pin_mem = self.device.type == "cuda"
        persist = num_workers > 0

        if val_fraction > 0.0:
            train_dataset, val_dataset = random_split(dataset, [1 - val_fraction, val_fraction])
        else:
            train_dataset = dataset
            val_dataset = None

        if self.world_size > 1:
            shuffle = True
            train_sampler = DistributedSampler(
                train_dataset,
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
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
            shuffle=shuffle,
            pin_memory=pin_mem,
            drop_last=True,
            persistent_workers=persist,
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
            )
            val_dataloader = val_dataloader
        else:
            val_dataloader = None

        if self.global_rank == 0:
            print("Dataloader setup complete:")
            print(f"  Total train samples: {len(train_dataset)}")
            print(f"  Local batch size: {batch_size}")
            print(f"  Global batch size: {batch_size * self.world_size}")
            print(f"  Train batches per GPU per epoch: {len(train_dataloader)}")

            if val_dataset:
                print(f"  Total val samples: {len(val_dataset)}")
                print(f"  Val batches per GPU per epoch: {len(val_dataloader)}")

        return train_dataloader, train_sampler, val_dataloader, val_sampler

    def build_model(
        self,
        model: nn.Module,
        pretrained_weights: dict[str, torch.Tensor] | None = None,
    ) -> nn.Module | nn.parallel.DistributedDataParallel:
        """
        Wraps the model with DistributedDataParallel if mulitple GPUs are available.

        Returns the model.
        """

        model = model.to(self.device)
        if pretrained_weights is not None:
            model.load_state_dict(pretrained_weights.copy())

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
