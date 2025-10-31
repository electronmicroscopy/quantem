import torch

from quantem.core.ml.splat.splat_base import SplatBase
from quantem.core.ml.splat.splat_constraints import SplatConstraints


class SplatTomo(SplatConstraints, SplatBase):
    """
    3D splatting for tomography -- includes rendering to tilts
    This will probably eventually move to the tomography module in the end, but for now
    lets keep it here and see how it goes.
    """

    pass

    def __init__(self):
        SplatBase.__init__(self)

    def forward(self, tilts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
