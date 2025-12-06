from typing import Any


import torch

# from torch_radon.radon import ParallelBeam as Radon
from tqdm.auto import tqdm

from quantem.tomography.object_models import ObjectVoxelwise
from quantem.tomography.tomography_base import TomographyBase
from quantem.tomography.tomography_conv import TomographyConv
from quantem.tomography.tomography_ml import TomographyML
from quantem.tomography.tomography_ddp import PretrainVolumeDataset, TomographyDDP, AdaptiveSmoothL1LossDDP
from quantem.tomography.utils import differentiable_shift_2d, gaussian_kernel_1d, rot_ZXZ

from quantem.core.ml.loss_functions import L1Loss, MSELoss, MSELogMSELoss, LLMSELoss, CharbonnierLoss
from torch.nn import SmoothL1Loss
# Temporary imports for TomographyNERF
from quantem.tomography.models import HSiren
from quantem.tomography.object_models import ObjectINN
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist

# Temporary aux class for TomographyNERF
# TODO: Maybe put this in INN?
def get_num_samples_per_ray(N: int, epoch: int):
    """Increase number of samples per ray at specific epochs."""
    # Exponential schedule

    epochs = np.linspace(0, 10, 5, dtype=int)
    # schedule = np.linspace(20, N, 5, dtype=int)
    # schedule = np.array([20, 100, 150, 200, 200])
    # schedule = np.array([200, 200, 200, 200, 200])
    schedule = np.array([20, 100, 250, 500, 500])
    # schedule = np.array([500, 500, 500, 500, 500])
    # schedule = np.array([300, 300, 300, 300, 300])
    # schedule = np.exp(schedule)
    schedule_warmup = dict[int, int](zip(epochs, schedule))

    for epoch_threshold, samples in schedule_warmup.items():
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
        
        # TODO: More elegant way of doing this.
        self.global_epochs = 0
        self.ddp_instantiated = False
        self.pretraining_instantiated = False
        
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

        print("Devices", sirt_tilt_series.device, proj_forward.device, self.dataset.tilt_angles.device)
        
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
    
    def create_batch_projection_rays(self, pixel_i, pixel_j, N, num_samples_per_ray):
        """Create projection rays for entire batch simultaneously."""
        batch_size = len(pixel_i)

        # Convert all pixels to normalized coordinates
        x_coords = (pixel_j / (N - 1)) * 2 - 1
        y_coords = (pixel_i / (N - 1)) * 2 - 1
        # TODO: maybe pixel_j.device?
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

    # TODO: Temp logger
    def setup_logger(self, log_path):
        if self.global_rank == 0:
            self.temp_logger = SummaryWriter(log_dir=log_path)
        else:
            self.temp_logger = None
            

    def pretrain(
        self,
        volume_dataset: PretrainVolumeDataset,
        obj: ObjectINN,
        batch_size: int,
        soft_constraints: dict = None,
        log_path: str = None,
        optimizer_params: dict = None,
        epochs: int = 100,
        viz_freq: int = 1,
        consistency_criterion: str = 'mse',
    ):
        if not self.pretraining_instantiated:
            self.setup_distributed()
            self.setup_pretraining_dataloader(volume_dataset, batch_size)
            self.pretraining_instantiated = True
            obj.model = self.build_model(obj._model)
            self.obj = obj

        if soft_constraints is not None:
            self.obj.soft_constraints = soft_constraints
        
        if not hasattr(self, "temp_logger"):
            self.setup_logger(log_path=log_path)
        
        if optimizer_params is not None:
            optimizer_params = self.scale_lr(optimizer_params)
            self.optimizer_params = optimizer_params
            self.set_optimizers()
        device_type = self.device.type
        consistency_loss_fn = None
        if consistency_criterion[0].lower() == 'mse':
            consistency_loss_fn = MSELoss()
        elif consistency_criterion[0].lower() == 'l1':
            consistency_loss_fn = L1Loss()
        elif consistency_criterion[0].lower() == 'mse_log':
            consistency_loss_fn = MSELogMSELoss()
        elif consistency_criterion[0].lower() == 'llmse':
            consistency_loss_fn = LLMSELoss()
        elif consistency_criterion[0].lower() == 'smooth_l1':
            consistency_loss_fn = SmoothL1Loss(beta = consistency_criterion[1])
        elif consistency_criterion[0].lower() == 'adaptive_smooth_l1':
            consistency_loss_fn = AdaptiveSmoothL1LossDDP(beta_init = consistency_criterion[1], ema_factor = 0.99, eps = 1e-8)
        elif consistency_criterion[0].lower() == 'charbonnier':
            consistency_loss_fn = CharbonnierLoss(epsilon = 1e-12, reduction = 'mean')
        else:
            raise ValueError(f"Invalid consistency criterion: {consistency_criterion}")

        for epoch in range(epochs):
            if self.pretraining_sampler is not None:
                self.pretraining_sampler.set_epoch(epoch)

            self.obj.model.train()

            epoch_loss = 0.0
            epoch_consistency_loss = 0.0
            epoch_tv_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(self.pretraining_dataloader):
                coords = batch['coords'].to(self.device, non_blocking=True)
                target = batch['target'].to(self.device, non_blocking=True)

                for _, opt in self.optimizers.items():
                    opt.zero_grad()

                with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True):
                    
                    outputs = self.obj.forward(coords)
                    consistency_loss = consistency_loss_fn(outputs, target)
                    tv_loss = self.obj.apply_soft_constraints(coords)
                    
                    loss = consistency_loss + tv_loss

                loss.backward()

                epoch_loss += loss.detach()
                epoch_consistency_loss += consistency_loss.detach()
                epoch_tv_loss += tv_loss.detach()

                torch.nn.utils.clip_grad_norm_(self.obj.model.parameters(), max_norm = 1.0)

                for _, opt in self.optimizers.items():
                    opt.step()
                epoch_loss += loss.detach()
                num_batches +=1
            
            if epoch % viz_freq == 0 or epoch == epochs - 1:
                avg_loss = epoch_loss / num_batches
                with torch.no_grad():
                    self.obj.create_volume(world_size = self.world_size, global_rank = self.global_rank, ray_size = volume_dataset.N)
                    pred_full = self.obj.obj
                    loss_tensor = avg_loss.clone().detach()
                    avg_loss = loss_tensor.item()
                    avg_tv_loss = epoch_tv_loss / num_batches
                    avg_consistency_loss = epoch_consistency_loss / num_batches

                    metrics = torch.tensor([avg_loss, avg_consistency_loss, avg_tv_loss], device=self.device)
                    if self.world_size > 1:
                        dist.all_reduce(metrics, op = dist.ReduceOp.AVG)
                    avg_loss, avg_consistency_loss, avg_tv_loss = metrics.tolist()
                
                if self.global_rank == 0:
                    self.temp_logger.add_scalar("Pretrain/total_loss", avg_loss, epoch)
                    self.temp_logger.add_scalar("Pretrain/consistency_loss", avg_consistency_loss, epoch)
                    self.temp_logger.add_scalar("Pretrain/tv_loss", avg_tv_loss, epoch)
                    
                    # current_lr = self.schedulers["model"].get_last_lr()[0]
                    # self.temp_logger.add_scalar("Pretrain/lr", current_lr, epoch)

                    fig, ax = plt.subplots(ncols = 4, figsize = (36, 12))
                    ax[0].matshow(pred_full.sum(dim=0).cpu().numpy(), cmap='turbo', vmin=0)
                    ax[0].set_title('Sum over Z-axis')
                    ax[1].matshow(pred_full[volume_dataset.N//2].cpu().numpy(), cmap='turbo', vmin=0)
                    ax[1].set_title(f'Slice at Z={volume_dataset.N//2}')
                    ax[2].matshow(pred_full.sum(dim=1).cpu().numpy(), cmap='turbo', vmin=0)
                    ax[2].set_title('Sum over Y-axis')
                    ax[3].matshow(pred_full.sum(dim=2).cpu().numpy(), cmap='turbo', vmin=0)
                    ax[3].set_title('Sum over X-axis')
                    self.temp_logger.add_figure("Pretrain/viz", fig, epoch, close=True)

                    plt.close(fig)
                    print(f"Epoch [{epoch}/{epochs}] Pretrain Total Loss: {avg_loss:.6f}, Consistency Loss: {avg_consistency_loss:.6f}, TV Loss: {avg_tv_loss:.6f}")
            self.global_epochs += 1
        if self.global_rank == 0:

            print(f"Pretraining Completed, saving model weights to {log_path}/model_weights.pth")
            torch.save(self.obj.model.state_dict(), f"{log_path}/model_weights.pth")

    def recon(
        self,
        obj: ObjectINN,
        batch_size: int,
        num_workers: int = 0,
        epochs = 20,
        use_amp = True,
        viz_freq = 1,
        checkpoint_freq = 5,
        optimizer_params: dict = None,
        scheduler_params: dict = None,
        soft_constraints: dict = None,
        vol_save_path: str = None, # TODO: TEMPORARY
        log_path = None,
        val_fraction: float = 0.0,
        learn_shifts: bool = True,
        # l1_loss: bool = False,
        consistency_criterion: str = 'mse',
        model_weights_path: str = None,
    ):      

        if not self.ddp_instantiated:
            if self.pretraining_instantiated == False:
                self.setup_distributed()
                if model_weights_path is not None:
                    print(f"Loading model weights from {model_weights_path}")
                    state_dict = torch.load(model_weights_path, map_location='cpu')

                    # Handle DataParallel/DDP checkpoints
                    if any(k.startswith('module.') for k in state_dict.keys()):
                        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    else:
                        new_state_dict = state_dict

                    obj.model.load_state_dict(new_state_dict)
                    print("Model weights loaded successfully")
                obj.model = self.build_model(obj._model)
                self.obj = obj

            self.setup_dataloader(self.dataset, batch_size, num_workers, val_fraction)
            self.ddp_instantiated = True
        
        
        if soft_constraints is not None:
            self.obj.soft_constraints = soft_constraints
        
        if not hasattr(self, "temp_logger"):
            self.setup_logger(log_path=log_path)

        zero_tilt_idx = torch.argmin(torch.abs(self.dataset.tilt_angles)).item()
        
        if self.global_rank == 0:
            print(f"Using projection {zero_tilt_idx} (angle={self.dataset.tilt_angles[zero_tilt_idx]:.2f}°) as reference")
            print(f"Using consistency criterion: {consistency_criterion}")
        
        # Auxiliary params setup
        self.dataset.setup_auxiliary_params(zero_tilt_idx, self.device, learn_shifts)
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
            num_samples_per_ray = get_num_samples_per_ray(N = N, epoch = self.global_epochs)
            
            # num_samples_per_ray = get_num_samples_per_ray(epoch)
            # Log the change if it happens
            if self.global_rank == 0:
                print(f"Epoch {epoch}: num_samples_per_ray = {num_samples_per_ray}")
                
            
            if self.global_rank == 0 and self.global_epochs > 0:
                prev_samples = get_num_samples_per_ray(N = N, epoch = self.global_epochs - 1)
                
                if num_samples_per_ray != prev_samples:
                    print(f"Epoch {epoch}: Changing num_samples_per_ray from {prev_samples} to {num_samples_per_ray}")
                
                
                
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)

            epoch_loss = 0.0
            epoch_consistency_loss = 0.0
            epoch_tv_loss = 0.0
            epoch_z1_loss = 0.0

            num_batches = 0
            consistency_loss_fn = None
            if consistency_criterion[0].lower() == 'mse':
                consistency_loss_fn = MSELoss()
            elif consistency_criterion[0].lower() == 'l1':
                consistency_loss_fn = L1Loss()
            elif consistency_criterion[0].lower() == 'mse_log':
                consistency_loss_fn = MSELogMSELoss()
            elif consistency_criterion[0].lower() == 'llmse':
                consistency_loss_fn = LLMSELoss()
            elif consistency_criterion[0].lower() == 'smooth_l1':
                consistency_loss_fn = SmoothL1Loss(beta = consistency_criterion[1])
            elif consistency_criterion[0].lower() == 'adaptive_smooth_l1':
                consistency_loss_fn = AdaptiveSmoothL1LossDDP(beta_init = consistency_criterion[1], ema_factor = 0.99, eps = 1e-8)
            elif consistency_criterion[0].lower() == 'charbonnier':
                consistency_loss_fn = CharbonnierLoss(epsilon = 1e-12, reduction = 'mean')
            else:
                raise ValueError(f"Invalid consistency criterion: {consistency_criterion}")

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
                    
                    # TODO: .forward passing z1, x, z3, shifts, N, sampling_rate
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

                    all_densities = self.obj.forward(all_coords)

                    tv_loss = self.obj.apply_soft_constraints(all_coords)
                    ray_densities = all_densities.view(len(target_values), num_samples_per_ray) # Reshape rays and integarte
                    step_size = 2.0 / (num_samples_per_ray - 1)

                    predicted_values = ray_densities.sum(dim=1) * step_size

                    consistency_loss = consistency_loss_fn(predicted_values, target_values)
                    tv_loss_z1 = torch.tensor(0.0, device=self.device)

                    loss = consistency_loss + tv_loss + tv_loss_z1
                    
                loss.backward()

                epoch_loss += loss.detach()
                epoch_consistency_loss += consistency_loss.detach()
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
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(epoch_loss)
                else:   
                    sched.step()

                    
            if epoch % viz_freq == 0 or epoch == epochs - 1 or epoch % checkpoint_freq == 0 :
                with torch.no_grad():
                    
                    self.obj.create_volume(world_size = self.world_size, global_rank = self.global_rank, ray_size = num_samples_per_ray)
                    pred_full = self.obj.obj
                    avg_loss = epoch_loss.item() / num_batches
                    avg_consistency_loss = epoch_consistency_loss.item() / num_batches
                    avg_tv_loss = epoch_tv_loss.item() / num_batches
                    avg_z1_loss = epoch_z1_loss.item() / num_batches
                    shifts, z1, z3 = self.dataset.auxiliary_params.forward()
                    shifts = shifts.detach().cpu()
                    z1 = z1.detach().cpu()
                    z3 = z3.detach().cpu()
                    metrics = torch.tensor([avg_loss, avg_consistency_loss, avg_tv_loss, avg_z1_loss], device=self.device)
                    if self.world_size > 1:
                        dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
                    avg_loss, avg_consistency_loss, avg_tv_loss, avg_z1_loss = metrics.tolist()


            if hasattr(self, "val_dataloader") and self.val_dataloader is not None:
                val_loss = self.validate(aux_params, num_samples_per_ray, device_type, autocast_dtype, use_amp, N)

            if self.global_rank == 0 and (epoch % viz_freq == 0 or epoch == epochs - 1):
                with torch.no_grad():
                    current_lr = self.schedulers["model"].get_last_lr()[0]

                    # Log metrics
                    self.temp_logger.add_scalar(f"train/consistency_loss_{consistency_criterion[0] if isinstance(consistency_criterion, tuple) else consistency_criterion}", avg_consistency_loss, self.global_epochs)
                    if hasattr(self, "val_dataloader") and self.val_dataloader is not None:
                        self.temp_logger.add_scalar(f"val/consistency_loss_{consistency_criterion[0] if isinstance(consistency_criterion, tuple) else consistency_criterion}", val_loss, self.global_epochs)
                    self.temp_logger.add_scalar("train/z1_loss", avg_z1_loss, self.global_epochs)
                    # if tv_weight > 0:
                    self.temp_logger.add_scalar("train/tv_loss", avg_tv_loss, self.global_epochs)
                    self.temp_logger.add_scalar("train/total_loss", avg_loss, self.global_epochs)
                    self.temp_logger.add_scalar("train/model_grad_norm", model_norm.item(), self.global_epochs)
                    self.temp_logger.add_scalar("train/aux_grad_norm", aux_norm.item(), self.global_epochs)
                    self.temp_logger.add_scalar("train/lr", current_lr, self.global_epochs)
                    self.temp_logger.add_scalar("train/num_samples_per_ray", num_samples_per_ray, self.global_epochs)
                    if consistency_criterion[0].lower() == 'adaptive_smooth_l1':
                        self.temp_logger.add_scalar("train/beta_2", consistency_loss_fn.beta2.item(), self.global_epochs)
                    fig, axes = plt.subplots(1, 5, figsize=(36, 12))
                    axes[0].matshow(pred_full.sum(dim=0).cpu().numpy(), cmap='turbo', vmin=0)
                    axes[0].set_title('Sum over Z-axis')

                    axes[1].matshow(pred_full[N//2].cpu().numpy(), cmap='turbo', vmin=0)
                    axes[1].set_title(f'Slice at Z={N//2}')

                    slice_start = max(0, N//2 - 5)
                    slice_end = min(N, N//2 + 6)
                    thick_slice = pred_full[slice_start:slice_end].sum(dim=0).cpu().numpy()
                    axes[2].matshow(thick_slice, cmap='turbo', vmin=0)
                    axes[2].set_title(f'Thick slice sum (Z={slice_start}:{slice_end-1})')
                    
                    axes[3].matshow(pred_full.sum(dim = 1).cpu().numpy(), cmap='turbo', vmin=0)
                    axes[3].set_title('Sum over Y-axis')
                    
                    axes[4].matshow(pred_full.sum(dim = 2).cpu().numpy(), cmap='turbo', vmin=0)
                    axes[4].set_title('Sum over X-axis')

                    self.temp_logger.add_figure("train/viz", fig, self.global_epochs, close=True)

                    fig, axes = plt.subplots(ncols = 3, figsize = (10, 5))
                    axes[0].plot(shifts[:, 0].cpu().numpy(), label = 'Shifts X')
                    axes[0].plot(shifts[:, 1].cpu().numpy(), label = 'Shifts Y')
                    axes[0].legend()
                    axes[1].plot(z1.cpu().numpy(), label = 'Z1')
                    axes[2].plot(z3.cpu().numpy(), label = 'Z3')
                    self.temp_logger.add_figure("train/auxiliary_params", fig, self.global_epochs, close=True)
                    plt.close(fig)
                    
            if epoch % checkpoint_freq == 0 or self.global_epochs == epochs - 1:
                with torch.no_grad():
                    if self.global_rank == 0:
                        save_path = f"{vol_save_path}/volume_epoch_{self.global_epochs:04d}.pt"
                        torch.save(pred_full.cpu(), save_path)
                        
            self.global_epochs += 1

        if self.global_rank == 0:
            print("Training complete.")
        
        torch.save(self.dataset.auxiliary_params.forward(), f"{log_path}/auxiliary_params.pt")
        # print("Successfully setup DDP and dataloader")
    def validate(self, aux_params, num_samples_per_ray, device_type, autocast_dtype, use_amp, N):
        """Validate the model on the validation set."""
        self.obj.model.eval()
        aux_params.eval()

        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
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

                    all_densities = self.obj.forward(all_coords)

                    ray_densities = all_densities.view(len(target_values), num_samples_per_ray) # Reshape rays and integarte
                    step_size = 2.0 / (num_samples_per_ray - 1)

                    predicted_values = ray_densities.sum(dim=1) * step_size

                    mse_loss = torch.nn.functional.mse_loss(predicted_values, target_values)

                    loss = mse_loss

                    val_loss += loss.detach()
                    num_batches += 1

        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0

        if self.world_size > 1:
            val_loss_tensor = avg_val_loss.detach().clone()
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            avg_val_loss = val_loss_tensor.item()

        self.obj.model.train()
        aux_params.train()

        return avg_val_loss

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
