import torch

from quantem.core.ml.splat.splat_base import SplatBase


class SplatConstraints(SplatBase):
    """
    Mixins for applying constraints to splatting. Not entirely sure how we'll do this of course.
    """

    def apply_hard_constraints(self):
        with torch.no_grad():
            raise NotImplementedError()

    def apply_soft_constraints(self) -> torch.Tensor:
        raise NotImplementedError()
