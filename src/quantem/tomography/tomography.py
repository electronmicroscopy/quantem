import torch
from torch_radon.radon import ParallelBeam as Radon
from tqdm.auto import tqdm

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.tomography.tomography_base import TomographyBase
from quantem.tomography.tomography_conv import TomographyConv
from quantem.tomography.tomography_ml import TomographyML
from quantem.tomography.utils import gaussian_kernel_1d


class Tomography(TomographyConv, TomographyML, TomographyBase):
    """
    Top level class for either using conventional or ML-based reconstruction methods
    for tomography.
    """

    def __init__(
        self,
        tilt_series,
        recon_volume,
        device,
        _token,
    ):
        super().__init__(tilt_series, recon_volume, device, _token)

    # --- Reconstruction Method ---

    def sirt_recon(
        self,
        num_iterations: int = 10,
        step_size: float = 0.25,
        inline_alignment: bool = False,
        enforce_positivity: bool = True,
        smoothing_sigma: float = None,
        reset: bool = True,
        mode: str = "batch",
        shrinkage: float = None,
    ):
        num_slices, num_angles, num_rows = self.tilt_series.array.shape

        if reset:
            self.loss = []
            stack_recon = torch.zeros(
                (num_slices, num_rows, num_rows),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            stack_recon = torch.tensor(
                self.recon_volume.array,
                device=self.device,
                dtype=torch.float32,
            )

        stack_torch = torch.tensor(
            self.tilt_series.array,
            device=self.device,
            dtype=torch.float32,
        )

        torch_angles = torch.tensor(
            self.tilt_series.tilt_angles_rad,
            device=self.device,
            dtype=torch.float32,
        )

        radon = Radon(
            det_count=num_rows,
            angles=torch_angles,
        )

        proj_forward = torch.zeros(
            (num_slices, num_angles, num_rows),
            dtype=torch.float32,
            device="cuda",
        )

        if smoothing_sigma is not None:
            kernel_1d = gaussian_kernel_1d(smoothing_sigma).to(self.device)
        else:
            kernel_1d = None

        pbar = tqdm(range(num_iterations), desc="SIRT Iterations")
        if mode == "batch":
            for a0 in pbar:
                if inline_alignment and a0 > 2:
                    stack_recon, loss = self._sirt_run_epoch(
                        radon=radon,
                        stack_recon=stack_recon,
                        stack_torch=stack_torch,
                        proj_forward=proj_forward,
                        step_size=step_size,
                        gaussian_kernel=kernel_1d,
                        inline_alignment=True,
                        enforce_positivity=enforce_positivity,
                        shrinkage=shrinkage,
                    )
                else:
                    stack_recon, loss = self._sirt_run_epoch(
                        radon=radon,
                        stack_recon=stack_recon,
                        stack_torch=stack_torch,
                        proj_forward=proj_forward,
                        step_size=step_size,
                        gaussian_kernel=kernel_1d,
                        inline_alignment=False,
                        enforce_positivity=enforce_positivity,
                        shrinkage=shrinkage,
                    )

                self.loss.append(loss.item())

        elif mode == "serial":
            for a0 in pbar:
                stack_recon, loss = self._sirt_serial_run_epoch(
                    radon=radon,
                    stack_recon=stack_recon,
                    stack_torch=stack_torch,
                    proj_forward=proj_forward,
                    step_size=step_size,
                    gaussian_kernel=kernel_1d,
                    inline_alignment=inline_alignment,
                    enforce_positivity=enforce_positivity,
                )

                self.loss.append(loss.item())

        self.recon_volume = Dataset3d.from_array(
            array=stack_recon.cpu().numpy(),
            name=self.tilt_series.name,
            origin=self.tilt_series.origin,
            sampling=self.tilt_series.sampling,
            units=self.tilt_series.units,
            signal_units=self.tilt_series.signal_units,
        )

    def recon_ML(
        self,
    ):
        raise NotImplementedError(
            "ML-based reconstruction is not implemented yet. Please use the SIRT method for now."
        )
