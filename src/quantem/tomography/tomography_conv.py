import numpy as np
import torch

# from torch_radon.radon import ParallelBeam as Radon
from quantem.tomography.radon.radon import iradon_torch, radon_torch
from quantem.tomography.tomography_base import TomographyBase
from quantem.tomography.utils import gaussian_filter_2d_stack, torch_phase_cross_correlation


class TomographyConv(TomographyBase):
    """
    Class for handling conventional reconstruction methods of tomography data.
    """

    # --- Reconstruction Methods ---

    def _sirt_run_epoch(
        self,
        tilt_series: torch.Tensor,
        proj_forward: torch.Tensor,
        angles: torch.Tensor,
        inline_alignment: bool,
        filter_name: str,
        circle: bool,
        gaussian_kernel: torch.Tensor | None,
    ):
        loss = 0

        if inline_alignment:
            for ind in range(len(self.dataset.tilt_angles)):
                im_proj = proj_forward[ind]
                im_meas = tilt_series[ind]

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

                    proj_forward[ind] = shifted

        # Forward projection

        sinogram_est = radon_torch(self.volume_obj.obj, theta=angles, device=self.device)
        # proj_forward = sinogram_est.permute(1, 2, 0)
        # error = (tilt_series - proj_forward).permute(2, 0, 1)
        proj_forward = sinogram_est
        error = tilt_series - proj_forward

        correction = iradon_torch(
            error, theta=angles, device=self.device, filter_name=filter_name, circle=circle
        )

        normalization = iradon_torch(
            torch.ones_like(error),
            theta=angles,
            device=self.device,
            filter_name=None,
            circle=circle,
        )

        normalization[normalization == 0] = 1e-6

        correction /= normalization

        self.volume_obj._obj += correction

        loss = torch.mean(torch.abs(error))

        # for z in tqdm(range(self.volume_obj.obj.shape[0]), desc="SIRT Reconstruction"):
        #     slice_estimate = self.volume_obj.obj[z]
        #     sinogram_est = radon_torch(slice_estimate, theta=angles, device=self.device)

        #     sinogram_true = tilt_series[:, :, z]

        #     error = sinogram_true - sinogram_est

        #     correction = iradon_torch(
        #         error, theta=angles, device=self.device, filter_name=filter_name, circle=circle
        #     )

        #     # I'm pretty sure this implementation of normalization is wrong
        #     normalization = iradon_torch(
        #         torch.ones_like(error),
        #         theta=angles,
        #         device=self.device,
        #         filter_name=None,
        #         circle=circle,
        #     )
        #     normalization[normalization == 0] = 1e-6

        #     correction /= normalization

        #     self.volume_obj._obj[z] += correction

        #     proj_forward[:, :, z] = sinogram_est

        #     loss += torch.mean(torch.abs(error))

        # loss /= self.volume_obj._obj.shape[0]

        if gaussian_kernel is not None:
            self.volume_obj.obj = gaussian_filter_2d_stack(self.volume_obj.obj, gaussian_kernel)

        return proj_forward, loss

    # Deprecated torch_radon implementations
    # def _sirt_run_epoch(
    #     self,
    #     radon: Radon,
    #     stack_recon: torch.Tensor,
    #     stack_torch: torch.Tensor,
    #     proj_forward: torch.Tensor,
    #     step_size: float = 0.25,
    #     gaussian_kernel: torch.Tensor = None,
    #     inline_alignment=True,
    #     enforce_positivity=True,
    #     shrinkage: float = None,
    # ):
    #     loss = 0

    #     if inline_alignment:
    #         for ind in range(len(self.tilt_series.tilt_angles)):
    #             im_proj = proj_forward[:, ind, :]
    #             im_meas = stack_torch[:, ind, :]

    #             shift = torch_phase_cross_correlation(im_proj, im_meas)
    #             if torch.linalg.norm(shift) <= 32:
    #                 shifted = torch.fft.ifft2(
    #                     torch.fft.fft2(im_meas)
    #                     * torch.exp(
    #                         -2j
    #                         * np.pi
    #                         * (
    #                             shift[0]
    #                             * torch.fft.fftfreq(
    #                                 im_meas.shape[0], device=im_meas.device
    #                             ).unsqueeze(1)
    #                             + shift[1]
    #                             * torch.fft.fftfreq(im_meas.shape[1], device=im_meas.device)
    #                         )
    #                     )
    #                 ).real

    #                 stack_torch[:, ind, :] = shifted

    #     proj_forward = radon.forward(stack_recon)

    #     proj_diff = stack_torch - proj_forward

    #     loss = torch.mean(torch.abs(proj_diff))

    #     recon_slice_update = radon.backward(
    #         radon.filter_sinogram(
    #             proj_diff,
    #         )
    #     )

    #     stack_recon += step_size * recon_slice_update
    #     if enforce_positivity:
    #         stack_recon = torch.clamp(stack_recon, min=0)

    #     if gaussian_kernel is not None:
    #         stack_recon = gaussian_filter_2d_stack(
    #             stack_recon,
    #             gaussian_kernel,
    #         )

    #     if shrinkage is not None:
    #         stack_recon = torch.max(
    #             stack_recon - shrinkage,
    #             torch.zeros_like(stack_recon),
    #         )

    #     return stack_recon, loss

    # def _sirt_serial_run_epoch(
    #     self,
    #     radon: Radon,
    #     stack_recon: torch.Tensor,
    #     stack_torch: torch.Tensor,
    #     proj_forward: torch.Tensor,
    #     step_size: float = 0.25,
    #     gaussian_kernel: torch.Tensor = None,
    #     inline_alignment=True,
    #     enforce_positivity=True,
    # ):
    #     recon_slice_update = torch.zeros_like(stack_recon).to(self.device)

    #     loss = 0

    #     for i in range(stack_recon.shape[0]):
    #         proj_forward[i] = radon.forward(stack_recon[i])

    #     proj_diff = stack_torch - proj_forward

    #     loss = torch.mean(torch.abs(proj_diff))

    #     for i in range(stack_recon.shape[0]):
    #         recon_slice_update[i] = radon.backward(
    #             radon.filter_sinogram(
    #                 proj_diff[i],
    #             )
    #         )

    #     stack_recon += step_size * recon_slice_update

    #     if enforce_positivity:
    #         stack_recon = torch.clamp(stack_recon, min=0)

    #     return stack_recon, loss

    # --- Properties ---
    # @property
    # def reconstruction_method(self) -> str:
    #     """Get the reconstruction method."""
    #     return self._reconstruction_method
    # @reconstruction_method.setter
    # def reconstruction_method(self, value: str):
    #     """Set the reconstruction method."""
    #     if value not in ["SIRT", "FBP"]:
    #         raise ValueError("Invalid reconstruction method. Choose 'SIRT' or 'FBP'.")
    #     self._reconstruction_method = value
