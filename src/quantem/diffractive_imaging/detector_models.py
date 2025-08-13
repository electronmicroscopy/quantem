from abc import abstractmethod

import torch

from quantem.core.io.serialize import AutoSerialize


class DetectorBase(AutoSerialize):
    @abstractmethod
    def forward(self, exit_waves: torch.Tensor) -> torch.Tensor:
        """
        Exit waves to measured intensities
        """
        raise NotImplementedError


class DetectorPixelated(DetectorBase):
    """
    A detector model that simulates pixelated detectors.
    """

    def forward(self, exit_waves: torch.Tensor) -> torch.Tensor:
        """
        Exit waves to measured intensities
        """
        # exit_waves shape: (nprobes, batch_size, roi_shape[0], roi_shape[1])
        # incoherent sum of all probe components
        exit_fft = torch.fft.fft2(exit_waves)
        intensities = torch.sum(torch.abs(exit_fft) ** 2, dim=0)
        return torch.fft.fftshift(intensities, dim=(-2, -1))  # detector centering


DetectorModelType = DetectorPixelated  # | DetectorPixelatedDIP
