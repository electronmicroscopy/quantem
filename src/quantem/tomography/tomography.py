import torch

# from torch_radon.radon import ParallelBeam as Radon
from tqdm.auto import tqdm

from quantem.tomography.object_models import ObjectVoxelwise
from quantem.tomography.tomography_base import TomographyBase
from quantem.tomography.tomography_conv import TomographyConv
from quantem.tomography.tomography_ml import TomographyML
from quantem.tomography.tomography_ddp import TomographyDDP
from quantem.tomography.utils import differentiable_shift_2d, gaussian_kernel_1d, rot_ZXZ

# Temporary imports for TomographyNERF
from quantem.tomography.models import HSiren
from quantem.tomography.object_models import ObjectINN
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist

# Temporary aux class for TomographyNERF
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



class Tomography(TomographyConv, TomographyML, TomographyBase, TomographyDDP):
    """
    Top level class for either using conventional or ML-based reconstruction methods
    for tomography.
    """

    def __init__(
        self,
        dataset,
        volume_obj,
        device,
        _token,
    ):
        super().__init__(dataset, volume_obj, device, _token)
    # --- Reconstruction Method ---

    def sirt_recon(
        self,
        num_iterations: int = 10,
        inline_alignment: bool = False,
        enforce_positivity: bool = True,
        volume_shape: tuple = None,
        reset: bool = True,
        smoothing_sigma: float = None,
        shrinkage: float = None,
        filter_name: str = "hamming",
        circle: bool = True,
        plot_loss: bool = False,
    ):
        num_angles, num_rows, num_cols = self.dataset.tilt_series.shape
        sirt_tilt_series = self.dataset.tilt_series.clone()
        sirt_tilt_series = sirt_tilt_series.permute(2, 0, 1)

        hard_constraints = {
            "positivity": enforce_positivity,
            "shrinkage": shrinkage,
        }
        self.volume_obj.hard_constraints = hard_constraints

        if volume_shape is None:
            volume_shape = (num_rows, num_rows, num_rows)
        else:
            D, H, W = volume_shape

        if reset:
            self.volume_obj.reset()
            self.loss = []

        proj_forward = torch.zeros_like(self.dataset.tilt_series)

        pbar = tqdm(range(num_iterations), desc="SIRT Reconstruction")

        if smoothing_sigma is not None:
            gaussian_kernel = gaussian_kernel_1d(smoothing_sigma).to(self.device)
        else:
            gaussian_kernel = None

        for iter in pbar:
            proj_forward, loss = self._sirt_run_epoch(
                tilt_series=sirt_tilt_series,
                proj_forward=proj_forward,
                angles=self.dataset.tilt_angles,
                inline_alignment=iter > 0 and inline_alignment,
                filter_name=filter_name,
                gaussian_kernel=gaussian_kernel,
                circle=circle,
            )

            pbar.set_description(f"SIRT Reconstruction | Loss: {loss.item():.4f}")

            self.loss.append(loss.item())

        self.sirt_recon_vol = self.volume_obj

        # Permutation due to sinogram ordering.
        self.sirt_recon_vol.obj = self.sirt_recon_vol.obj.permute(1, 2, 0)

        if plot_loss:
            self.plot_loss()
    
    # TODO: ML Recon which has NeRF and AD depending on the object type.
    # TODO: Temporary 

    
    def create_optimizer(self, params, lr, fused = True):

        return torch.optim.Adam(
            params,
            lr = lr,
            fused = fused,
        )
    
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
    
    # --- Creating Volume ---
    def create_volume(self, model):
        
        N = max(self.dataset.dims)
        
        with torch.no_grad():
            coords_1d = torch.linspace(-1, 1, N)
            x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
            inputs = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
            
            model = model.module if hasattr(model, 'module') else model
            
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
    
    
    # TODO: Temp logger
    def setup_logger(self):
        if self.global_rank == 0:
            self.temp_logger = SummaryWriter()
        else:
            self.temp_logger = None
    
    def recon(
        self,
        obj: ObjectINN,
        batch_size: int,
        num_workers: int = 0,
        epochs = 20,
        use_amp = True,
        tv_weight = 0.0,
        viz_freq = 1,
        checkpoint_freq = 5,
        optimizer_params: dict = None,
        scheduler_params: dict = None,
    ):      

        self.setup_distributed()
        self.setup_dataloader(self.dataset, batch_size, num_workers)
        
        obj.model = self.build_model(obj._model)
        
        self.obj = obj
        
        if not hasattr(self, "temp_logger"):
            self.setup_logger()

        zero_tilt_idx = torch.argmin(torch.abs(self.dataset.tilt_angles)).item()
        
        if self.global_rank == 0:
            print(f"Using projection {zero_tilt_idx} (angle={self.dataset.tilt_angles[zero_tilt_idx]:.2f}°) as reference")
        
        # Auxiliary params setup
        self.dataset.setup_auxiliary_params(zero_tilt_idx, self.device)
        aux_params = self.dataset.auxiliary_params
        
        # Scaling learning rates to account for distributed training
        if optimizer_params is not None:
            optimizer_params = self.scale_lr(optimizer_params)
            self.optimizer_params = optimizer_params  
            self.set_optimizers()

        if scheduler_params is not None:
            self.scheduler_params = scheduler_params
            self.set_schedulers(self.scheduler_params, num_iter = epochs)
        
        
        aux_norm = torch.tensor(0.0, device=self.device)
        model_norm = torch.tensor(0.0, device=self.device)
        
        
        for _, opt in self.optimizers.items():
            opt.zero_grad()
        
        N = max(self.dataset.dims)
        
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
                        sampling_rate=1.0,
                    )

                    all_coords = transformed_rays.view(-1, 3)
                    # TODO: torch.nn.functional.softplus here
                    # all_densities = torch.nn.functional.softplus(self.model(all_coords))
                    
                    # TODO: I don't think this work, object needs to be distributed to all devices?
                    all_densities = self.obj.forward(all_coords)
                    # all_densities = self.model(all_coords)

                    # if all_densities.dim() > 1:
                    #     all_densities = all_densities.squeeze(-1)  # [N, 1] → [N]

                    # # Create mask for valid coordinates (within [-1, 1]^3)
                    # valid_mask = (
                    #     (all_coords[:, 0] >= -1) & (all_coords[:, 0] <= 1) &
                    #     (all_coords[:, 1] >= -1) & (all_coords[:, 1] <= 1) &
                    #     (all_coords[:, 2] >= -1) & (all_coords[:, 2] <= 1)
                    # ).float()
                    
                    # all_densities = all_densities * valid_mask
                    # tv_loss = obj.apply_soft_constraints(
                    #     coords = all_coords,
                    #     tv_weight = 1e-5,
                    # )
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

                    loss = mse_loss + tv_loss + tv_loss_z1
                    
                loss.backward()

                epoch_loss += loss.detach()
                epoch_mse_loss += mse_loss.detach()
                epoch_tv_loss += tv_loss.detach()
                epoch_z1_loss += tv_loss_z1.detach()
                num_batches += 1

                for key, opt in self.optimizers.items():
                    
                    if key == "model":
                        model_norm = torch.nn.utils.clip_grad_norm_(self.obj.model.parameters(), max_norm = 1)
                    elif key == "aux_params":
                        aux_norm = torch.nn.utils.clip_grad_norm_(self.dataset.auxiliary_params.parameters(), max_norm = 1)
                    opt.step()
                    opt.zero_grad()
                
            for key, sched in self.schedulers.items():
                sched.step()

                    
            if epoch % viz_freq == 0 or epoch == epochs - 1 or epoch % checkpoint_freq == 0 :
                with torch.no_grad():
                    pred_full = self.create_volume(self.obj.model).cpu()
                    avg_loss = epoch_loss.item() / num_batches
                    avg_mse_loss = epoch_mse_loss.item() / num_batches
                    avg_tv_loss = epoch_tv_loss.item() / num_batches
                    avg_z1_loss = epoch_z1_loss.item() / num_batches
                    
                    metrics = torch.tensor([avg_loss, avg_mse_loss, avg_tv_loss, avg_z1_loss], device=self.device)
                    if self.world_size > 1:
                        dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
                    avg_loss, avg_mse_loss, avg_tv_loss, avg_z1_loss = metrics.tolist()

            if self.global_rank == 0 and (epoch % viz_freq == 0 or epoch == epochs - 1):
                with torch.no_grad():
                    current_lr = self.schedulers["model"].get_last_lr()[0]

                    # Log metrics
                    self.temp_logger.add_scalar("train/mse_loss", avg_mse_loss, epoch)

                    self.temp_logger.add_scalar("train/z1_loss", avg_z1_loss, epoch)
                    if tv_weight > 0:
                        self.temp_logger.add_scalar("train/tv_loss", avg_tv_loss, epoch)
                        self.temp_logger.add_scalar("train/total_loss", avg_loss, epoch)
                    self.temp_logger.add_scalar("train/model_grad_norm", model_norm.item(), epoch)
                    self.temp_logger.add_scalar("train/aux_grad_norm", aux_norm.item(), epoch)
                    self.temp_logger.add_scalar("train/lr", current_lr, epoch)
                    self.temp_logger.add_scalar("train/num_samples_per_ray", num_samples_per_ray, epoch)

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

                    self.temp_logger.add_figure("train/viz", fig, epoch, close=True)
                    plt.close(fig)
                    
        if self.global_rank == 0:
            print("Training complete.")

        # print("Successfully setup DDP and dataloader")
        
    def ad_recon(
        self,
        optimizer_params: dict,
        num_iter: int = 0,
        reset: bool = False,
        scheduler_params: dict | None = None,
        hard_constraints: dict | None = None,
        soft_constraints: dict | None = None,
        # store_iterations: bool | None = None,
        # store_iterations_every: int | None = None,
        # autograd: bool = True,
    ):
        if reset:
            self.reset_recon()

        self.hard_constraints = hard_constraints
        self.soft_constraints = soft_constraints

        # Make sure everything is in the correct device, might be redundant/cleaner way to do this
        self.dataset.to(self.device)
        self.volume_obj.to(self.device)

        # Making optimizable parameters into leaf tensors.
        self.dataset.shifts = self.dataset.shifts.detach().to(self.device).requires_grad_(True)
        self.dataset.z1_angles = (
            self.dataset.z1_angles.detach().to(self.device).requires_grad_(True)
        )
        self.dataset.z3_angles = (
            self.dataset.z3_angles.detach().to(self.device).requires_grad_(True)
        )

        if optimizer_params is not None:
            self.optimizer_params = optimizer_params
            self.set_optimizers()

        if scheduler_params is not None:
            self.scheduler_params = scheduler_params
            self.set_schedulers(self.scheduler_params, num_iter=num_iter)

        if hard_constraints is not None:
            self.volume_obj.hard_constraints = hard_constraints
        if soft_constraints is not None:
            self.volume_obj.soft_constraints = soft_constraints

        pbar = tqdm(range(num_iter), desc="AD Reconstruction")

        for a0 in pbar:
            total_loss = 0.0
            tilt_series_loss = 0.0

            pred_volume = self.volume_obj.forward()

            for i in range(len(self.dataset.tilt_series)):
                forward_projection = self.projection_operator(
                    vol=pred_volume,
                    z1=self.dataset.z1_angles[i],
                    x=self.dataset.tilt_angles[i],
                    z3=self.dataset.z3_angles[i],
                    shift_x=self.dataset.shifts[i, 0],
                    shift_y=self.dataset.shifts[i, 1],
                    device=self.device,
                )

                tilt_series_loss += torch.nn.functional.mse_loss(
                    forward_projection, self.dataset.tilt_series[i]
                )
            tilt_series_loss /= len(self.dataset.tilt_series)

            total_loss = tilt_series_loss + self.volume_obj.soft_loss
            self.loss.append(total_loss.item())

            total_loss.backward()

            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad()

            if self.schedulers is not None:
                for sch in self.schedulers.values():
                    if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        sch.step(total_loss)
                    elif sch is not None:
                        sch.step()

            pbar.set_description(f"AD Reconstruction | Loss: {total_loss:.4f}")

            if self.logger is not None:
                self.logger.log_scalar("loss/total", total_loss.item(), a0)
                self.logger.log_scalar("loss/tilt_series", tilt_series_loss.item(), a0)
                self.logger.log_scalar(
                    "loss/soft constraints", self.volume_obj.soft_loss.item(), a0
                )

                if a0 % self.logger.log_images_every == 0:
                    self.logger.projection_images(
                        volume_obj=self.volume_obj,
                        epoch=a0,
                    )
                    self.logger.tilt_angles_figure(dataset=self.dataset, step=a0)

                self.logger.flush()

        self.ad_recon_vol = self.volume_obj.forward()

        return self

    def reset_recon(self) -> None:
        if isinstance(self.volume_obj, ObjectVoxelwise):
            self.volume_obj.reset()

        self.ad_recon_vol = None

    # --- Projection Operators ----
    def projection_operator(
        self,
        vol,
        z1,
        x,
        z3,
        shift_x,
        shift_y,
        device,
    ):
        projection = (
            rot_ZXZ(
                mags=vol.unsqueeze(0),  # Add batch dimension
                z1=z1,
                x=-x,
                z3=z3,
                device=device,
                mode="bilinear",
            )
            .squeeze()
            .sum(axis=0)
        )

        shifted_projection = differentiable_shift_2d(
            image=projection,
            shift_x=shift_x,
            shift_y=shift_y,
            sampling_rate=1.0,  # Assuming 1 pixel = 1 physical unit
        )

        return shifted_projection
