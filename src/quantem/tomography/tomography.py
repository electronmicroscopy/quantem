import torch

# from torch_radon.radon import ParallelBeam as Radon
from tqdm.auto import tqdm

from quantem.tomography.object_models import ObjectVoxelwise
from quantem.tomography.tomography_base import TomographyBase
from quantem.tomography.tomography_conv import TomographyConv
from quantem.tomography.tomography_ml import TomographyML
from quantem.tomography.utils import differentiable_shift_2d, gaussian_kernel_1d, rot_ZXZ


class Tomography(TomographyConv, TomographyML, TomographyBase):
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
