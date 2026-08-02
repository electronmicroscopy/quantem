from dataclasses import dataclass
from typing import Optional
from quantem.core.ml.constraints import BaseContext

import torch


@dataclass
class ReconstructionContext(BaseContext):
    """
    Handles all reconstruction parameters to be passed into object models.

    Subclasses will pick whatever parameter they need
        - Pixelated reads ".volume"
        - INR reads ".coords" and recomputes via the model.
        - TensorDecomp reads ".coords" and ".pred" (and ".all densities")

    Variable descriptions:
    - volume: Reconstructed object (volume).
    - coords: Used for INR reconstructions to provide the coordinates to the model.
    - pred: Predicted values per coordinate position from the model.
    - all_densities: Integrated densities per ray from the model.
    - obj: Object model (INR, TensorDecomp, etc.).
    """

    volume: Optional[torch.Tensor] = None
    coords: Optional[torch.Tensor] = None
    pred: Optional[torch.Tensor] = None
    all_densities: Optional[torch.Tensor] = None
    obj: Optional[torch.Tensor] = None
