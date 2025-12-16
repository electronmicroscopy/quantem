from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import Dataset

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.optimizer_mixin import OptimizerMixin


@dataclass
class DatasetValue:
    """
    Class for storing the forward call for both PixDataset and INRDataset.
    """

    target: torch.Tensor
    tilt_angle: int | float
    pixel_loc: tuple[int, int] | None = None  # Only for INRDataset


class TomographyDatasetBase(AutoSerialize, OptimizerMixin, nn.Module):
    """
    Base tomography dataset class for all tomography datasets to inherit from.
    """

    _token = object()

    DEFAULT_LRS = {
        "pose_lr": 5e-2,
    }

    def __init__(
        self,
        tilt_stack: Dataset3d,
        tilt_angles: NDArray | torch.Tensor,
        learn_pose: bool = True,
        _token: object | None = None,
    ):
        AutoSerialize.__init__(self)
        OptimizerMixin.__init__(self)
        nn.Module.__init__(self)

        if _token is not self._token:
            raise RuntimeError("Use TomographyDatasetBase.from_* to instantiate this class.")

        if not (
            tilt_stack.shape[0] < tilt_stack.shape[1] or tilt_stack.shape[0] < tilt_stack.shape[2]
        ):
            raise ValueError(
                "The number of tilt projections should be in the first dimension of the dataset."
            )

        self.tilt_stack = tilt_stack
        self.tilt_angles = tilt_angles
        self.learn_pose = learn_pose

        self._reference_tilt_angle_idx = torch.argmin(torch.abs(self.tilt_angles))

    @abstractmethod
    def forward(
        self,
        dummy_input: Any = None,  # Note all nn.Modules require some input.
    ):
        """
        Forward pass should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


class TomographyPixDataset(
    TomographyDatasetBase
):  # Dataset): TODO: Does this need to be a dataset?
    """
    Dataset class for pixel-based tomography, i.e AD, SIRT, WBP, etc...

    TODO:
     - What should the forward pass return? In AD, it should be both the tilt image and the pose.
       In SIRT, it's only the tilt image.
    """

    pass

    def __init__(
        self,
    ):
        pass

    def forward(
        self,
        proj_idx: int,
    ):
        return DatasetValue(
            target=self.tilt_stack[proj_idx],
            tilt_angle=self.tilt_angles[proj_idx],
            pixel_loc=None,
        )


class TomographyINRDataset(TomographyDatasetBase, Dataset):
    """
    Dataset class for INR-based tomography.

    The two main methods here are that the `forward` call will return the relative pose parameters,
    while `__getitem__` will actually return the pixel values of the tilt stack.
    """

    def __init__(
        self,
    ):
        pass

    def forward(self, dummy_input: Any = None):
        """
        Forward pass for INR-based tomography. In the forward pass, the only parameters that
        are passed will be the shifts, z1 and z3 Euler angles.
        """
        pass

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, Any]:
        """
        Gets the item for INR i.e, the project index, pixel value at (i, j), and the tilt angle.
        """
        pass

        actual_idx = idx

        projection_idx = actual_idx // (self.tilt_stack.shape[1] * self.tilt_stack.shape[2])
        remaining = actual_idx % (self.tilt_stack.shape[1] * self.tilt_stack.shape[2])

        pixel_i = remaining // self.tilt_stack.shape[1]
        pixel_j = remaining % self.tilt_stack.shape[1]

        return DatasetValue(
            target=self.tilt_stack[projection_idx, pixel_i, pixel_j],
            tilt_angle=self.tilt_angles[projection_idx],
            pixel_loc=(pixel_i, pixel_j),
        )

    def __len__(
        self,
    ):
        """
        Returns the number of pixels in the tilt stack.
        """
        pass


DatasetModelType = TomographyINRDataset
