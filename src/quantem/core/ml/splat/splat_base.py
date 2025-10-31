import torch

from quantem.core.io.serialize import AutoSerialize


class SplatBase(AutoSerialize):
    """
    2/3D Splatting class for rendering point clouds.
    """

    pass

    def __init__(self):
        pass

    def render_volume(self, grid: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
