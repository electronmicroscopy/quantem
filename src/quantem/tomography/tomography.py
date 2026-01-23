from typing import Literal, Optional, Self, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from quantem.core.ml.ddp import DDPMixin
from quantem.tomography.dataset_models import DatasetModelType
from quantem.tomography.logger_tomography import LoggerTomography
from quantem.tomography.object_models import ObjectModelType
from quantem.tomography.radon.radon import iradon_torch, radon_torch
from quantem.tomography.tomography_base import TomographyBase
from quantem.tomography.tomography_opt import TomographyOpt
from quantem.tomography.utils import torch_phase_cross_correlation
from quantem.tomography_old.utils import gaussian_filter_2d_stack, gaussian_kernel_1d


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
        # obj_model: ObjectModelType,
        # dset: DatasetModelType,
        num_iter: int = 10,
        batch_size: int = 1024,
        num_workers: int = 32,
        reset: bool = False,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        constraints: dict = {},  # TODO: What to pass into the constraints?
        loss_func: Tuple[str, Optional[float]] = ("smooth_l1", 0.07),
    ):
        """
        This function should be able to handle both AD and INR-based tomography reconstruction methods.
        I.e, auto-detection through the obj model type, while both share the same pose optimization.
        """

        # TODO: Prior to reconstruction, it is assumed that object + dataset are both in the correct devices. Need to implement a way to check this.

        if self.obj_model.device != self.dset.device:
            raise ValueError(
                f"Should never happen! obj_model and dset must be on the same device, got {self.obj_model.device} and {self.dset.device}"
            )

        if reset:
            raise NotImplementedError("Reset is not implemented yet.")

        if optimizer_params is not None:
            self.optimizer_params = optimizer_params
            self.set_optimizers()

        if scheduler_params is not None:
            self.scheduler_params = scheduler_params
            self.set_schedulers()

        new_scheduler = reset
        if new_scheduler:
            raise NotImplementedError("New schedulers are not implemented yet.")

        # Setting up DDP
        if not hasattr(self, "dataloader"):
            self.dataloader, self.sampler = self.setup_dataloader(
                self.dset, batch_size, num_workers=num_workers
            )

        self.obj_model.model.train()

        N = max(self.obj_model.obj.shape)
        num_samples_per_ray = max(self.obj_model.obj.shape)
        print(f"N: {N}, num_samples_per_ray: {num_samples_per_ray}")

        for a0 in range(num_iter):
            consistency_loss = 0.0
            total_loss = 0.0
            # self._reset_iter_constraints()

            if self.sampler is not None:
                self.sampler.set_epoch(a0)

            for batch_idx, batch in enumerate(self.dataloader):
                self.zero_grad_all()
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=True,
                ):
                    all_coords = self.dset.get_coords(batch, N, num_samples_per_ray)

                    all_densities = self.obj_model.forward(all_coords)

                    integrated_densities = self.dset.integrate_rays(
                        all_densities, num_samples_per_ray
                    )

                    # batch_consistency_loss = loss_func(integrated_densities, batch["target_value"])
                    batch_consistency_loss = torch.nn.functional.mse_loss(
                        integrated_densities, batch["target_value"]
                    )

                    # soft_constraints_loss = self._soft_constraints()
                    batch_loss = batch_consistency_loss  # + soft_constraints_loss
                batch_loss.backward()
                self.step_optimizers()
                total_loss += batch_loss.item()
                consistency_loss += batch_consistency_loss.item()

            total_loss = total_loss / len(self.dataloader)
            consistency_loss = consistency_loss / len(self.dataloader)
            print(f"Total Loss: {total_loss:.4f}, Consistency Loss: {consistency_loss:.4f}")


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
                print(self.obj_model.obj.shape)
                self.obj_model.obj = gaussian_filter_2d_stack(self.obj_model.obj, gaussian_kernel)

        loss = torch.mean(torch.abs(error))

        return proj_forward, loss
