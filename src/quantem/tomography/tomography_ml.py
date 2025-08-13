from typing import Optional

from quantem.tomography.tomography_base import TomographyBase


class TomographyML(TomographyBase):
    """
    Class for handling conventional reconstruction methods of tomography data.
    """

    # def __init__(
    #     self,
    #     optimizers: dict,
    #     schedulers: Optional[dict] = None,
    #     training_steps: Optional[dict] = None,
    # ):
    #     super().__init__()

    # All in reconstruction

    # --- Reconstruction Methods ---

    def reconstruct(self):
        pass

    """
    TODO
    Reconstruction using ML-AD
    1. Voxel-Wise Reconstruction
    2. 3D-UNet Reconstruction 
    """

    # --- Properties ---
    @property
    def optimizers(self) -> dict:
        """Get the optimizers."""
        return self._optimizers

    @optimizers.setter
    def optimizers(self, value: dict):
        """Set the optimizers."""
        self._optimizers = value

    @property
    def schedulers(self) -> Optional[dict]:
        """Get the schedulers."""
        return self._schedulers

    @schedulers.setter
    def schedulers(self, value: Optional[dict]):
        """Set the schedulers."""
        self._schedulers = value

    @property
    def training_steps(self) -> Optional[dict]:
        """Get the training steps."""
        return self._training_steps

    @training_steps.setter
    def training_steps(self, value: Optional[dict]):
        """Set the training steps."""
        self._training_steps = value
