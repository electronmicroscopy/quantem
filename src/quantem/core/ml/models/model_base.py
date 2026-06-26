from abc import ABC, abstractmethod
from typing import Dict

import torch.nn as nn


class PPLR(ABC):
    """
    Abstract base class for models that require multi-parameter optimization.
    """

    @abstractmethod
    def get_params(self) -> Dict[str, list[nn.Parameter]]:
        """
        Return a dictionary of parameters grouped by key.

        For example if your nn.Module has multiple optimizable parameter groups,
        you can return a dictionary with the keys "grids" and "sigma_net"
        (KPlanes example).
        """
        pass

    @property
    @abstractmethod
    def param_keys(self) -> list[str]:
        """List of available parameter-group keys."""
        pass


class TensorDecompositionModel(nn.Module, ABC):
    """
    Base class for factored tensor-decomposition models.

    Subclasses must set ``td_type`` as a normal attribute in ``__init__``.
    """

    td_type: str


class PlanarDecompositionModel(TensorDecompositionModel, PPLR):
    """
    Planar factored-grid models: K-Planes, K-Planes-TILTED, HexPlane, tri-planes.

    Subclasses must set ``grids``, ``tilted``, and ``resolution`` as normal
    attributes in ``__init__``.
    """

    grids: nn.ParameterList
    tilted: bool
    resolution: list[int]
