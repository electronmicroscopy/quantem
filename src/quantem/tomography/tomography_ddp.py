import torch.distributed as dist
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.nn as nn
import os

from quantem.tomography.tomography_dataset import TomographyDataset

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
                dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo', init_method='env://') 

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
                self.device = torch.device("cuda:0")
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
        self.model = model.to(self.device)
        
        if self.world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids = [self.local_rank],
                output_device = self.local_rank,
                find_unused_parameters = False,
                broadcast_buffers = True,
                bucket_cap_mb = 100,
                gradient_as_bucket_view = True,
            )
            
            if self.global_rank == 0:
                print("Model wrapped with DDP")
                
        if self.world_size > 1:
            
            if self.global_rank == 0:
                print("Model built, distributed, and compiled successfully")
                
        else:
            print("Model built, compiled successfully")
            
    def setup_dataloader(
        self,
        tomo_dataset: TomographyDataset,
        batch_size: int,
        num_workers: int = 0,
    ):
        
        if self.world_size > 1:
            sampler = DistributedSampler(
                tomo_dataset,
                num_replicas = self.world_size,
                rank = self.global_rank,
                shuffle = True,
            )    
            shuffle = False
            
        else:
            sampler = None
            shuffle = True
            
        self.dataloader = DataLoader(
            tomo_dataset,
            batch_size = batch_size,
            num_workers = num_workers,
            sampler = sampler,
            shuffle = shuffle,
            pin_memory = True if self.device.type == "cuda" else False,
            drop_last = True,
            persistent_workers = False if num_workers == 0 else True,
        )
        
        self.sampler = sampler
        
        if self.global_rank == 0:
            print(f"Dataloader setup complete:")
            print(f"  Total projections: {len(tomo_dataset.tilt_angles)}")
            print(f"  Grid size: {tomo_dataset.dims[1]}{tomo_dataset.dims[2]}")
            print(f"  Total pixels: {tomo_dataset.num_pixels:,}")
            
            print(f"  Local batch size (train): {batch_size}")
            print(f"  Global batch size: {batch_size*self.world_size}")
            print(f"  Train batches per GPU per epoch: {len(self.dataloader)}")
        
    