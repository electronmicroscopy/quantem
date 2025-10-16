from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch
import torch.distributed as dist
import numpy as np

from quantem.tomography.tomography_dataset import TomographyDataset
from quantem.tomography.models import HSiren

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


import os
from pathlib import Path


def get_num_samples_per_ray(epoch):
    """Increase number of samples per ray at specific epochs."""
    schedule = {
        0: 25,
        2: 50,
        4: 100,
        6: 200,
    }

    num_samples = 64
    for epoch_threshold, samples in sorted(schedule.items()):
        if epoch >= epoch_threshold:
            num_samples = samples

    return num_samples

class AuxiliaryParams(torch.nn.Module):
    def __init__(self, num_tilts, device, zero_tilt_idx=None):
        super().__init__()

        if zero_tilt_idx is None:
            # If not provided, assume first projection is reference
            zero_tilt_idx = 0

        self.zero_tilt_idx = zero_tilt_idx
        self.num_tilts = num_tilts

        # Shifts: only parameterize non-reference tilts
        num_param_tilts = num_tilts - 1
        self.shifts_param = torch.nn.Parameter(torch.zeros(num_param_tilts, 2, device=device))

        # Fixed zero shifts for reference
        self.shifts_ref = torch.zeros(1, 2, device=device)

        # Z1 and Z3: parameterize all tilts EXCEPT the reference
        self.z1_param = torch.nn.Parameter(torch.zeros(num_param_tilts, device=device))
        # self.z3_param = torch.nn.Parameter(torch.zeros(num_param_tilts, device=device))

        # Fixed zeros for reference tilt
        self.z1_ref = torch.zeros(1, device=device)
        # self.z3_ref = torch.zeros(1, device=device)

    def forward(self, dummy_input=None):
        # Reconstruct full arrays with zeros at reference position
        before_shifts = self.shifts_param[:self.zero_tilt_idx]
        after_shifts = self.shifts_param[self.zero_tilt_idx:]
        shifts = torch.cat([before_shifts, self.shifts_ref, after_shifts], dim=0)

        before_z1 = self.z1_param[:self.zero_tilt_idx]
        after_z1 = self.z1_param[self.zero_tilt_idx:]
        z1 = torch.cat([before_z1, self.z1_ref, after_z1], dim=0)

        # before_z3 = self.z3_param[:self.zero_tilt_idx]
        # after_z3 = self.z3_param[self.zero_tilt_idx:]
        # z3 = torch.cat([before_z3, self.z3_ref, after_z3], dim=0)

        return shifts, z1, -z1


class TomographyNerf():
    
    def __init__(
        self,
        tomo_dataset: TomographyDataset,
        batch_size: int,
        num_workers: int = 0,
    ):
        
        self.tomo_dataset = tomo_dataset
        self.setup_distributed()
        self.setup_dataloader(
            batch_size = batch_size,
            num_workers = num_workers,
        )
        self.build_model()
        
    # --- Setup ----
    
    def setup_distributed(self):
        """Setup distributed training if available, otherwise use single GPU/CPU."""

        # Check if we're in a distributed environment
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

    def build_model(self):
        
        self.model = HSiren(
            in_features = 3,
            out_features = 1,
            hidden_layers = 4,
            hidden_features = 512,
            first_omega_0 = 30,
            alpha = 1.0,
        ).to(self.device)
        
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
        batch_size: int,
        num_workers: int = 0,
    ):
        
        if self.world_size > 1:
            sampler = DistributedSampler(
                self.tomo_dataset,
                num_replicas = self.world_size,
                rank = self.global_rank,
                shuffle = True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        self.dataloader = DataLoader(
            self.tomo_dataset,
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
            print(f"  Total projections: {len(self.tomo_dataset.tilt_angles)}")
            print(f"  Grid size: {self.tomo_dataset.dims[1]}{self.tomo_dataset.dims[2]}")
            print(f"  Total pixels: {self.tomo_dataset.num_pixels:,}")
            
            print(f"  Local batch size (train): {batch_size}")
            print(f"  Global batch size: {batch_size*self.world_size}")
            print(f"  Train batches per GPU per epoch: {len(self.dataloader)}")


    # --- Creating Volume ---
    def create_volume(self):
        
        N = max(self.tomo_dataset.dims)
        
        with torch.no_grad():
            coords_1d = torch.linspace(-1, 1, N)
            x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
            inputs = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
            
            model = self.model.module if hasattr(self.model, 'module') else self.model
            
            samples_per_gpu = N**3 // self.world_size
            start_idx = self.global_rank * samples_per_gpu
            end_idx = start_idx + samples_per_gpu
            
            inputs_subset = inputs[start_idx:end_idx].to(self.device)
            
            # TODO: torch.nn.functional.softplus here
            # outputs = torch.nn.functional.softplus(model(inputs_subset))
            outputs = model(inputs_subset)
            
            if outputs.dim() > 1:
                outputs = outputs.squeeze(-1)
                
            if self.world_size > 1:
                gathered_outputs = [torch.empty_like(outputs) for _ in range(self.world_size)]
                dist.all_gather(gathered_outputs, outputs.contiguous())
                
                pred_full = torch.cat(gathered_outputs, dim=0).reshape(N, N, N).float()
            else:
                pred_full = outputs.reshape(N, N, N).float()
                
            return pred_full
    
    # --- Scaling LR ---
    def get_scaled_lr(self, base_lr, scaling_rule="sqrt"):
        """Scale learning rate based on world size."""
        if self.world_size == 1:
            return base_lr

        if scaling_rule == "linear":
            # Linear scaling: lr = base_lr * world_size
            return base_lr * self.world_size
        elif scaling_rule == "sqrt":
            # Square root scaling: lr = base_lr * sqrt(world_size)
            return base_lr * np.sqrt(self.world_size)
        else:
            raise ValueError(f"Invalid scaling rule: {scaling_rule}")
        
    # --- Creating Optimizer ---
    def create_optimizer(
        self,
        params,
        lr,
        fused = True,
    ):
        
        return torch.optim.Adam(
            params,
            lr = lr,
            fused = fused,
        )
        
    # Batch Projection Rays ---
    
    def create_batch_projection_rays(self, pixel_i, pixel_j, N, num_samples_per_ray):
        """Create projection rays for entire batch simultaneously."""
        batch_size = len(pixel_i)

        # Convert all pixels to normalized coordinates
        x_coords = (pixel_j / (N - 1)) * 2 - 1
        y_coords = (pixel_i / (N - 1)) * 2 - 1

        # Create z coordinates
        z_coords = torch.linspace(-1, 1, num_samples_per_ray, device=self.device)

        # Create rays for all pixels: [batch_size, num_samples_per_ray, 3]
        rays = torch.zeros(batch_size, num_samples_per_ray, 3, device=self.device)

        # Fill coordinates efficiently
        rays[:, :, 0] = x_coords.unsqueeze(1)  # x constant per ray
        rays[:, :, 1] = y_coords.unsqueeze(1)  # y constant per ray
        rays[:, :, 2] = z_coords.unsqueeze(0)  # z varies along ray

        return rays
    
    @torch.compile(mode="reduce-overhead")    
    def transform_batch_ray_coordinates(self, rays, z1, x, z3, shifts, N, sampling_rate):

        # Step 1: Apply shifts
        shift_x_norm = (shifts[:, 0:1] * sampling_rate * 2) / (N - 1)
        shift_y_norm = (shifts[:, 1:2] * sampling_rate * 2) / (N - 1)

        rays_x = rays[:, :, 0] - shift_x_norm
        rays_y = rays[:, :, 1] - shift_y_norm
        rays_z = rays[:, :, 2]

        # Rotation 1: Z(-z3)
        theta = torch.deg2rad(-z3).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_rot1 = cos_t * rays_x - sin_t * rays_y
        rays_y_rot1 = sin_t * rays_x + cos_t * rays_y
        rays_z_rot1 = rays_z

        # Rotation 2: X(x)
        theta = torch.deg2rad(x).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_rot2 = rays_x_rot1
        rays_y_rot2 = cos_t * rays_y_rot1 - sin_t * rays_z_rot1
        rays_z_rot2 = sin_t * rays_y_rot1 + cos_t * rays_z_rot1

        # Rotation 3: Z(-z1)
        theta = torch.deg2rad(-z1).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_final = cos_t * rays_x_rot2 - sin_t * rays_y_rot2
        rays_y_final = sin_t * rays_x_rot2 + cos_t * rays_y_rot2
        rays_z_final = rays_z_rot2

        # Stack the final result
        transformed_rays = torch.stack([rays_x_final, rays_y_final, rays_z_final], dim=2)

        return transformed_rays
    
    # --- Training ---
    def train(
        self,
        train_lr: float = 1e-5,
        epochs: int = 20,
        warmup_epochs: int = 10,
        use_amp: bool = True,
        tv_weight: float = 0.0,
        viz_freq: int = 1,
        checkpoint_freq: int = 5,
    ):
        log_path = Path("runs/test_1")
        if not log_path.exists():
            log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_path)
        
        # Find index closest to zero degrees
        zero_tilt_idx = torch.argmin(torch.abs(self.tomo_dataset.tilt_angles)).item()
        
        if self.global_rank == 0:
            print(f"Using projection {zero_tilt_idx} (angle={self.tomo_dataset.tilt_angles[zero_tilt_idx]:.2f}°) as reference") 
            

        aux_params = AuxiliaryParams(
            num_tilts = len(self.tomo_dataset.tilt_angles),
            device = self.device,
            zero_tilt_idx = zero_tilt_idx,
        )
        
        if self.world_size > 1:
            aux_params = torch.nn.parallel.DistributedDataParallel(
                aux_params,
                device_ids = [self.local_rank],
                output_device = self.local_rank,
                find_unused_parameters = False,
                broadcast_buffers = True,
            )
            
            if self.global_rank == 0:
                print("Auxiliary parameters wrapped with DDP")
                
        scaled_train_lr = self.get_scaled_lr(train_lr, scaling_rule="sqrt")
        scaled_aux_lr = self.get_scaled_lr(train_lr, scaling_rule="sqrt")
        
        optimizer = self.create_optimizer(
            self.model.parameters(),
            scaled_train_lr,
            fused = True,
        )
        
        aux_optimizer = self.create_optimizer(
            aux_params.parameters(),
            scaled_aux_lr,
            fused = True,
        )
        
        aux_norm = torch.tensor(0.0, device=self.device)
        model_norm = torch.tensor(0.0, device=self.device)
        
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        
        warmup_scheduler_model = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = 0.001,
            total_iters = warmup_epochs,
        )
        cosine_scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = epochs - warmup_epochs,
            eta_min = scaled_train_lr / 100,
        )
        
        scheduler_model = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers = [warmup_scheduler_model, cosine_scheduler_model],
            milestones = [warmup_epochs],
        )
        
        scheduler_aux = torch.optim.lr_scheduler.CosineAnnealingLR(
            aux_optimizer,
            T_max = epochs,
            eta_min = scaled_aux_lr / 100,
        )
        
        N = max(self.tomo_dataset.dims)
        
        device_type = self.device.type
        autocast_dtype = torch.bfloat16 if use_amp else None
        
        for epoch in range(epochs):
            num_samples_per_ray = get_num_samples_per_ray(epoch)
            
            # Log the change if it happens
            if epoch > 0:
                prev_samples = get_num_samples_per_ray(epoch - 1)
                
                if num_samples_per_ray != prev_samples and self.global_rank == 0:
                    print(f"Epoch {epoch}: Changing num_samples_per_ray from {prev_samples} to {num_samples_per_ray}")

            if self.sampler is not None:
                self.sampler.set_epoch(epoch)

            epoch_loss = 0.0
            epoch_mse_loss = 0.0
            epoch_tv_loss = 0.0
            epoch_z1_loss = 0.0

            num_batches = 0
            
            for batch_idx, batch in enumerate(self.dataloader):

                projection_indices = batch['projection_idx']

                pixel_i = batch['pixel_i'].float().to(self.device, non_blocking=True)
                pixel_j = batch['pixel_j'].float().to(self.device, non_blocking=True)
                target_values = batch['target_value'].to(self.device, non_blocking=True)
                phis = batch['phi'].to(self.device, non_blocking=True)
                projection_indices = batch['projection_idx'].to(self.device, non_blocking=True)

                shifts, z1_params, z3_params = aux_params(None)
                batch_shifts = torch.index_select(shifts, 0, projection_indices)
                batch_z1 = torch.index_select(z1_params, 0, projection_indices)
                batch_z3 = torch.index_select(z3_params, 0, projection_indices)
                
                with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_amp):

                    with torch.no_grad():
                        batch_ray_coords = self.create_batch_projection_rays(
                            pixel_i, pixel_j, N, num_samples_per_ray
                        )

                    transformed_rays = self.transform_batch_ray_coordinates(
                        batch_ray_coords,
                        z1=batch_z1,
                        x=phis,
                        z3=batch_z3,
                        shifts=batch_shifts,
                        N=N,
                        sampling_rate=1.0
                    )

                    all_coords = transformed_rays.view(-1, 3)
                    # TODO: torch.nn.functional.softplus here
                    # all_densities = torch.nn.functional.softplus(self.model(all_coords))
                    all_densities = self.model(all_coords)

                    if all_densities.dim() > 1:
                        all_densities = all_densities.squeeze(-1)  # [N, 1] → [N]

                    # Create mask for valid coordinates (within [-1, 1]^3)
                    valid_mask = (
                        (all_coords[:, 0] >= -1) & (all_coords[:, 0] <= 1) &
                        (all_coords[:, 1] >= -1) & (all_coords[:, 1] <= 1) &
                        (all_coords[:, 2] >= -1) & (all_coords[:, 2] <= 1)
                    ).float()
                    
                    all_densities = all_densities * valid_mask
                    
                    if tv_weight > 0:
                        num_tv_samples = min(10000, all_coords.shape[0])
                        tv_indices = torch.randperm(all_coords.shape[0], device=all_coords.device)[:num_tv_samples]

                        # Rerun forward for gradient tracking
                        tv_coords = all_coords[tv_indices].detach().requires_grad_(True)
                        
                        # TODO: torch.nn.functional.softplus here
                        # tv_densities_recomputed = torch.nn.functional.softplus(self.model(tv_coords)) # Get rid of this
                        tv_densities_recomputed = self.model(tv_coords)
                        if tv_densities_recomputed.dim() > 1:
                            tv_densities_recomputed = tv_densities_recomputed.squeeze(-1)

                        # Compute gradients
                        grad_outputs = torch.autograd.grad(
                            outputs=tv_densities_recomputed,
                            inputs=tv_coords,
                            grad_outputs=torch.ones_like(tv_densities_recomputed),
                            create_graph=True
                        )[0]
                        
                        grad_norm = torch.norm(grad_outputs, dim=1)
                        tv_loss = tv_weight * grad_norm.mean()
                    else:
                        tv_loss = torch.tensor(0.0, device=self.device)

                    ray_densities = all_densities.view(len(target_values), num_samples_per_ray) # Reshape rays and integarte
                    step_size = 2.0 / (num_samples_per_ray - 1)

                    predicted_values = ray_densities.sum(dim=1) * step_size

                    mse_loss = torch.nn.functional.mse_loss(predicted_values, target_values)

                    tv_loss_z1 = torch.tensor(0.0, device=self.device)
                    # tv_loss_z1 = tv_loss_1d(z1_params, factor = 1e-3) + tv_loss_1d(z3_params, factor = 1e-3)
                    # tv_loss_z1 = 1e-5*(torch.sum(z1_params**2) + torch.sum(z3_params**2))

                    # Combine losses
                    loss = mse_loss + tv_loss_z1 + tv_loss

                loss.backward()

                epoch_loss += loss.detach()
                epoch_mse_loss += mse_loss.detach()
                epoch_tv_loss += tv_loss.detach()
                epoch_z1_loss += tv_loss_z1.detach()

                num_batches += 1

                if epoch >= warmup_epochs:
                    model_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    aux_norm = torch.nn.utils.clip_grad_norm_(aux_params.parameters(), max_norm = 1)
                    aux_optimizer.step()
                    aux_optimizer.zero_grad()
                else:
                    model_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    aux_optimizer.zero_grad()
                
            scheduler_model.step()
            
            if epoch >= warmup_epochs:
                scheduler_aux.step()

            if epoch % viz_freq == 0 or epoch == epochs - 1 or epoch % checkpoint_freq == 0 :
                with torch.no_grad():
                    pred_full = self.create_volume().cpu()
                    avg_loss = epoch_loss.item() / num_batches
                    avg_mse_loss = epoch_mse_loss.item() / num_batches
                    avg_tv_loss = epoch_tv_loss.item() / num_batches
                    avg_z1_loss = epoch_z1_loss.item() / num_batches
                    
                    metrics = torch.tensor([avg_loss, avg_mse_loss, avg_tv_loss, avg_z1_loss], device=self.device)
                    if self.world_size > 1:
                        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.AVG)
                    avg_loss, avg_mse_loss, avg_tv_loss, avg_z1_loss = metrics.tolist()

            # Logging and visualization
            if self.global_rank == 0 and (epoch % viz_freq == 0 or epoch == epochs - 1):
                with torch.no_grad():
                    current_lr = scheduler_model.get_last_lr()[0]

                    # Log metrics
                    self.writer.add_scalar("train/mse_loss", avg_mse_loss, epoch)

                    self.writer.add_scalar("train/z1_loss", avg_z1_loss, epoch)
                    if tv_weight > 0:
                        self.writer.add_scalar("train/tv_loss", avg_tv_loss, epoch)
                        self.writer.add_scalar("train/total_loss", avg_loss, epoch)
                    self.writer.add_scalar("train/model_grad_norm", model_norm.item(), epoch)
                    self.writer.add_scalar("train/aux_grad_norm", aux_norm.item(), epoch)
                    self.writer.add_scalar("train/lr", current_lr, epoch)
                    self.writer.add_scalar("train/num_samples_per_ray", num_samples_per_ray, epoch)

                    fig, axes = plt.subplots(1, 3, figsize=(36, 12))
                    axes[0].matshow(pred_full.sum(dim=0).cpu().numpy(), cmap='turbo', vmin=0)
                    axes[0].set_title('Sum over Z-axis')

                    axes[1].matshow(pred_full[N//2].cpu().numpy(), cmap='turbo', vmin=0)
                    axes[1].set_title(f'Slice at Z={N//2}')

                    slice_start = max(0, N//2 - 5)
                    slice_end = min(N, N//2 + 6)
                    thick_slice = pred_full[slice_start:slice_end].sum(dim=0).cpu().numpy()
                    axes[2].matshow(thick_slice, cmap='turbo', vmin=0)
                    axes[2].set_title(f'Thick slice sum (Z={slice_start}:{slice_end-1})')

                    self.writer.add_figure("train/viz", fig, epoch, close=True)
                    plt.close(fig)

            if epoch % checkpoint_freq == 0 or epoch == epochs - 1:
                self.save_checkpoint(epoch, aux_params, log_path)

        if self.global_rank == 0:
            print("Training complete.")

    def save_checkpoint(self, epoch, aux_params, log_path):
        """Save model and auxiliary parameters checkpoint (rank 0 only)."""
        if self.global_rank == 0:
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            aux_params_to_save = aux_params.module if hasattr(aux_params, 'module') else aux_params

            checkpoint = {
                "model_state_dict": model_to_save.state_dict(),
                "aux_params_state_dict": aux_params_to_save.state_dict(),
                "epoch": epoch,
            }
            checkpoint_path = os.path.join(log_path, f"checkpoint_epoch_{epoch:04d}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")