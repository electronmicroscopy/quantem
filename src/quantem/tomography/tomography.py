from typing import List, Literal, Optional, Self, Tuple

import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from quantem.core.ml.ddp import DDPMixin
from quantem.core.ml.profiling import nvtx_range
from quantem.tomography.dataset_models import DatasetModelType
from quantem.tomography.logger_tomography import LoggerTomography
from quantem.tomography.object_models import ObjectModelType
from quantem.tomography.radon.radon import iradon_torch, radon_torch
from quantem.tomography.tomography_base import TomographyBase
from quantem.tomography.tomography_opt import TomographyOpt
from quantem.tomography.utils import (
    gaussian_filter_2d_stack,
    gaussian_kernel_1d,
    torch_phase_cross_correlation,
)


class Tomography(TomographyOpt, TomographyBase, DDPMixin):
    """
    Class for handling all ML tomography reconstruction methods.
    Automatic handling between AD and INR-based tomography.
    """

    @classmethod
    def from_models(
        cls,
        dset: DatasetModelType,
        obj_model: ObjectModelType,
        logger: LoggerTomography | None = None,
        device: str = "cuda",
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        return cls(
            dset=dset,
            obj_model=obj_model,
            logger=logger,
            device=device,
        )

    def reconstruct(
        self,
        num_iter: int = 10,
        batch_size: int = 1024,
        num_workers: int = 32,
        reset: bool = False,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = {},
        constraints: dict = {},  # TODO: What to pass into the constraints?
        loss_func: Tuple[str, Optional[float]] = ("smooth_l1", 0.07),
        num_samples_per_ray: int | List[Tuple[int, int]] = None,
        profiling_mode: bool = False,
        val_fraction: float = 0.0,
        # reset_dset: bool = False,
        reset_dset: DatasetModelType | None = None,
    ):
        """
        This function should be able to handle both AD and INR-based tomography reconstruction methods.
        I.e, auto-detection through the obj model type, while both share the same pose optimization.
        """

        # TODO: Prior to reconstruction, it is assumed that object + dataset are both in the correct devices. Need to implement a way to check this.

        # if self.obj_model.device != self.dset.device:
        #     raise ValueError(
        #         f"Should never happen! obj_model and dset must be on the same device, got {self.obj_model.device} and {self.dset.device}"
        #     )
        if profiling_mode:
            if self.global_rank == 0:
                print("Profiling mode enabled.")

        if reset:
            raise NotImplementedError("Reset is not implemented yet.")

        new_scheduler = reset

        if optimizer_params is not None:
            with nvtx_range(profiling_mode, "Setting Optimizer Params"):
                self.optimizer_params = optimizer_params
                self.set_optimizers()
            new_scheduler = True

        if scheduler_params is not None:
            with nvtx_range(profiling_mode, "Setting Scheduler Params"):
                self.scheduler_params = scheduler_params
            new_scheduler = True

        if constraints is not None:
            with nvtx_range(profiling_mode, "Setting Constraints"):
                self.obj_model.constraints = constraints

        if new_scheduler:
            with nvtx_range(profiling_mode, "Setting Schedulers"):
                self.set_schedulers(scheduler_params, num_iter=num_iter)

        # Setting up DDP
        if not hasattr(self, "dataloader") or reset_dset is not None:
            with nvtx_range(profiling_mode, "Setting Dataloader"):
                if reset_dset is not None:
                    print("Resetting Dataloader")
                    print("Putting in params from previous dataset.")

                    self.dset = reset_dset
                    self.dset.to(self.device)
                    self.optimizer_params = optimizer_params
                    self.set_optimizers()
                self.dataloader, self.sampler, self.val_dataloader, self.val_sampler = (
                    self.setup_dataloader(
                        self.dset,
                        batch_size,
                        num_workers=num_workers,
                        val_fraction=val_fraction,
                    )
                )
        N = max(self.obj_model.shape)

        if num_samples_per_ray is None:
            num_samples_per_ray = max(self.obj_model.shape)
        else:
            if isinstance(num_samples_per_ray, int):
                num_samples_per_ray = num_samples_per_ray
            else:
                print("num_samples_per_ray schedule provided.")

        print(f"N: {N}, num_samples_per_ray: {num_samples_per_ray}")
        for a0 in range(num_iter):
            with nvtx_range(profiling_mode, f"Epoch {a0}"):
                consistency_loss = 0.0
                total_loss = 0.0
                epoch_soft_constraint_loss = 0.0
                self.obj_model.model.train()
                self.dset.train()
                # self._reset_iter_constraints()

                if self.sampler is not None:
                    self.sampler.set_epoch(a0)

                if isinstance(num_samples_per_ray, list):
                    curr_num_samples_per_ray = num_samples_per_ray[a0][1]
                else:
                    curr_num_samples_per_ray = num_samples_per_ray

                if self.global_rank == 0:
                    print(f"curr_num_samples_per_ray: {curr_num_samples_per_ray}")
                for batch_idx, batch in enumerate(self.dataloader):
                    with nvtx_range(profiling_mode, f"batch_{batch_idx}"):
                        self.zero_grad_all()
                        with torch.autocast(
                            device_type=self.device.type,
                            dtype=torch.bfloat16,
                            enabled=True,
                        ):
                            with nvtx_range(profiling_mode, "Getting Coords"):
                                all_coords = self.dset.get_coords(
                                    batch, N, curr_num_samples_per_ray
                                )
                            with nvtx_range(profiling_mode, "Forwarding"):
                                all_densities = self.obj_model.forward(all_coords)

                            with nvtx_range(profiling_mode, "Integrating"):
                                integrated_densities = self.dset.integrate_rays(
                                    all_densities,
                                    curr_num_samples_per_ray,
                                    len(batch["target_value"]),
                                )

                        pred = integrated_densities.float()

                        with nvtx_range(profiling_mode, "Getting Target"):
                            target = (
                                batch["target_value"].to(self.device, non_blocking=True).float()
                            )

                        with nvtx_range(profiling_mode, "Calculating Loss"):
                            batch_consistency_loss = torch.nn.functional.mse_loss(pred, target)

                        with nvtx_range(profiling_mode, "Applying Soft Constraints"):
                            soft_constraints_loss = self.obj_model.apply_soft_constraints(
                                all_coords
                            )

                        with nvtx_range(
                            profiling_mode, "Adding soft constraint loss to epoch loss"
                        ):
                            epoch_soft_constraint_loss += soft_constraints_loss.detach()

                        with nvtx_range(profiling_mode, "Calculating Batch Loss"):
                            batch_loss = (
                                batch_consistency_loss.float() + soft_constraints_loss.detach()
                            )

                        with nvtx_range(profiling_mode, "Backwarding"):
                            batch_loss.backward()
                        with nvtx_range(profiling_mode, "Clipping Gradients"):
                            # Clip gradients
                            torch.nn.utils.clip_grad_norm_(
                                self.obj_model.model.parameters(), max_norm=1.0
                            )
                        with nvtx_range(profiling_mode, "Stepping Optimizers"):
                            self.step_optimizers()
                        with nvtx_range(profiling_mode, "Adding batch loss to total loss"):
                            total_loss += batch_loss.detach()
                        with nvtx_range(
                            profiling_mode, "Adding batch consistency loss to consistency loss"
                        ):
                            consistency_loss += batch_consistency_loss.detach()

                with nvtx_range(profiling_mode, "Stepping Schedulers"):
                    self.step_schedulers(loss=total_loss)
                # TODO: Maybe reorganize the losses so that the order makes sense lol.

                total_loss = total_loss.item() / len(self.dataloader)
                consistency_loss = consistency_loss.item() / len(self.dataloader)
                epoch_soft_constraint_loss = epoch_soft_constraint_loss.item() / len(
                    self.dataloader
                )

                if self.val_dataloader is not None:
                    print("Validating...")
                    self.obj_model.model.eval()
                    self.dset.eval()
                    with torch.no_grad():
                        val_loss = 0.0

                        for batch in self.val_dataloader:
                            with torch.autocast(
                                device_type=self.device.type,
                                dtype=torch.bfloat16,
                                enabled=True,
                            ):
                                with nvtx_range(profiling_mode, "Getting Coords"):
                                    all_coords = self.dset.get_coords(
                                        batch, N, curr_num_samples_per_ray
                                    )

                                with nvtx_range(profiling_mode, "Forwarding"):
                                    all_densities = self.obj_model.forward(all_coords)

                                with nvtx_range(profiling_mode, "Integrating"):
                                    integrated_densities = self.dset.integrate_rays(
                                        all_densities,
                                        curr_num_samples_per_ray,
                                        len(batch["target_value"]),
                                    )

                                with nvtx_range(profiling_mode, "Getting Target"):
                                    target = (
                                        batch["target_value"]
                                        .to(self.device, non_blocking=True)
                                        .float()
                                    )

                                with nvtx_range(profiling_mode, "Calculating Loss"):
                                    batch_val_loss = torch.nn.functional.mse_loss(
                                        integrated_densities, target
                                    )

                                with nvtx_range(profiling_mode, "Adding batch loss to total loss"):
                                    val_loss += (
                                        batch_val_loss.detach() + soft_constraints_loss.detach()
                                    )

                        avg_val_loss = val_loss.item() / len(self.val_dataloader)

                metrics = torch.tensor(
                    [total_loss, consistency_loss, epoch_soft_constraint_loss], device=self.device
                )

                if self.world_size > 1:
                    dist.all_reduce(metrics, dist.ReduceOp.AVG)

                total_loss, consistency_loss, epoch_soft_constraint_loss = metrics.tolist()

                if self.global_rank == 0:
                    print(
                        f"Total Loss: {total_loss:.4f}, Consistency Loss: {consistency_loss:.4f}"
                    )

                    if self.val_dataloader:
                        print(f"Validation loss: {avg_val_loss:4f}")

                self._epoch_losses.append(total_loss)
                self._consistency_losses.append(consistency_loss)
                self.obj_model._soft_constraint_losses.append(epoch_soft_constraint_loss)
                if self.val_dataloader is not None:
                    self._val_losses.append(avg_val_loss)

                with nvtx_range(profiling_mode, "Logging"):
                    if self.logger is not None:
                        if (
                            self.logger.log_images_every > 0
                            and self.num_epochs % self.logger.log_images_every == 0
                        ):
                            print("Creating volume...")
                            pred_full = self.obj_model.create_volume(return_vol=True)

                            if self.global_rank == 0:
                                print("Logging images...")
                                self.logger.log_iter_images(
                                    pred_volume=pred_full,
                                    dataset_model=self.dset,
                                    iter=self.num_epochs,
                                )

                        if self.global_rank == 0:
                            self.logger.log_iter(
                                object_model=self.obj_model,
                                iter=self.num_epochs,
                                consistency_loss=consistency_loss,
                                total_loss=total_loss,
                                learning_rates=self.get_current_lrs(),
                                num_samples_per_ray=curr_num_samples_per_ray,
                                val_loss=avg_val_loss if self.val_dataloader is not None else None,
                            )

                        self.logger.flush()

    # --- Helper Functions ---

    def save_volume(self, path: str = "recon_volume.npz"):
        # TODO: Temporary, need to talk to Arthur what the correct way of saving results is.
        if self.global_rank == 0:
            print(f"Saving volume to {path}")
            np.savez(path, volume=self.obj_model.obj.detach().cpu().numpy())

        if torch.distributed.is_initialized():
            print("Barrier")
            torch.distributed.barrier()


class TomographyConventional(TomographyBase):
    """
    Class for handling all conventional tomography reconstruction methods.
    Will also handle choosing the appropriate dataset model to use.
    """

    @classmethod
    def from_models(
        cls,
        dset: DatasetModelType,
        obj_model: ObjectModelType,
        logger: LoggerTomography | None = None,
        device: str = "cuda",
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        return cls(
            dset=dset,
            obj_model=obj_model,
            logger=logger,
            device=device,
            rng=rng,
        )

    def reconstruct(
        self,
        num_iter: int = 10,
        mode: Literal["sirt", "fbp"] = "sirt",
        reset: bool = False,
        inline_alignment: bool = False,
        smoothing_sigma: float | None = None,
    ):
        pbar = tqdm(range(num_iter), desc=f"{mode} Reconstruction")
        if mode == "sirt" or mode == "fbp":
            proj_forward = torch.zeros_like(self.dset.tilt_stack).permute(2, 0, 1)
        else:
            proj_forward = torch.zeros_like(self.dset.tilt_stack)
        print("proj_forward.shape", proj_forward.shape)
        print("self.dset.tilt_stack.shape", self.dset.tilt_stack.shape)

        if smoothing_sigma is not None:
            gaussian_kernel = gaussian_kernel_1d(smoothing_sigma).to(self.device)
        else:
            gaussian_kernel = None

        for iter in pbar:
            proj_forward, loss = self._reconstruction_epoch(
                inline_alignment=inline_alignment,
                mode=mode,
                proj_forward=proj_forward,
                gaussian_kernel=gaussian_kernel,
            )

            pbar.set_description(f"{mode} Reconstruction | Loss: {loss.item():.4f}")

            self._epoch_losses.append(loss.item())

            if mode == "fbp":
                break

    # --- Conventional reconstruction method ---

    def _reconstruction_epoch(
        self,
        inline_alignment: bool,
        mode: Literal["sirt", "fbp"],
        proj_forward: torch.Tensor,
        gaussian_kernel: torch.Tensor | None = None,
    ):
        loss = 0

        if inline_alignment:
            for ind in range(len(self.dset.tilt_angles)):
                im_proj = proj_forward[:, ind, :]
                im_meas = self.dset.forward(ind).target
                shift = torch_phase_cross_correlation(im_proj, im_meas)
                if torch.linalg.norm(shift) <= 32:
                    shifted = torch.fft.ifft2(
                        torch.fft.fft2(im_meas)
                        * torch.exp(
                            -2j
                            * np.pi
                            * (
                                shift[0]
                                * torch.fft.fftfreq(
                                    im_meas.shape[0], device=im_meas.device
                                ).unsqueeze(1)
                                + shift[1]
                                * torch.fft.fftfreq(im_meas.shape[1], device=im_meas.device)
                            )
                        )
                    ).real

                    proj_forward[:, ind, :] = shifted

        if mode == "sirt" or mode == "fbp":
            proj_forward = radon_torch(
                self.obj_model.obj,
                theta=self.dset.tilt_angles,
                device=self.device,
            )

            error = self.dset.tilt_stack.permute(2, 0, 1) - proj_forward

            correction = iradon_torch(
                error,
                theta=self.dset.tilt_angles,
                device=self.device,
                filter_name="ramp",
                circle=True,
            )

            normalization = iradon_torch(
                torch.ones_like(error),
                theta=self.dset.tilt_angles,
                device=self.device,
                filter_name=None,
                circle=True,
            )

            normalization[normalization == 0] = 1e-6

            correction /= normalization

            self.obj_model.obj += correction

            if gaussian_kernel is not None:
                self.obj_model.obj = gaussian_filter_2d_stack(self.obj_model.obj, gaussian_kernel)

        loss = torch.mean(torch.abs(error))

        return proj_forward, loss
