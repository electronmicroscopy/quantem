from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
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
    pose: tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter] | None = (
        None  # If there is pose optimization.
    )


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
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        learn_pose: bool = True,
        _token: object | None = None,
    ):
        AutoSerialize.__init__(self)
        OptimizerMixin.__init__(self)
        nn.Module.__init__(self)
        # if _token is not self._token: TODO: Idk why this isn't working.
        #     raise RuntimeError("Use TomographyPixDataset.from_* to instantiate this class.")

        if not (
            tilt_stack.shape[0] < tilt_stack.shape[1] or tilt_stack.shape[0] < tilt_stack.shape[2]
        ):
            raise ValueError(
                "The number of tilt projections should be in the first dimension of the dataset."
            )

        self.volume_size = (int(max(tilt_stack.shape)), tilt_stack.shape[1], tilt_stack.shape[2])

        # TODO: Maybe have the validation in here too.
        max_val = np.quantile(tilt_stack, 0.95)
        if type(tilt_stack) is not torch.Tensor:
            tilt_stack = torch.from_numpy(tilt_stack)
        if type(tilt_angles) is not torch.Tensor:
            tilt_angles = torch.from_numpy(tilt_angles)

        # Tilt stack normalization
        tilt_stack = tilt_stack / max_val

        self.tilt_stack = tilt_stack
        self.tilt_angles = tilt_angles
        self.learn_pose = learn_pose

        # The reference tilt angle is the one with the smallest absolute tilt angle.
        # I.e, the pose will not be optimized for the reference tilt angle.
        self._reference_tilt_angle_idx = torch.argmin(torch.abs(self.tilt_angles))
        # TODO: Implement AuxParams from old tomography_dataset.py here.

        self.z1_param = torch

    @abstractmethod
    def forward(
        self,
        dummy_input: Any = None,  # Note all nn.Modules require some input.
    ):
        """
        Forward pass should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    # --- Properties ---
    @property
    def reference_tilt_idx(self) -> int:
        return self._reference_tilt_angle_idx

    @reference_tilt_idx.setter
    def reference_tilt_idx(self, reference_tilt_idx: int):
        self._reference_tilt_angle_idx = reference_tilt_idx

    @property
    def learnable_tilts(self) -> int:
        return self.tilt_angles.shape[0] - 1

    @learnable_tilts.setter
    def learnable_tilts(self, learnable_tilts: int):
        self._learnable_tilts = learnable_tilts

    @property
    def params(self) -> dict[str, torch.nn.Parameter]:
        """
        Returns the parameters that should be optimized for this dataset.

        Should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    # --- Helper Functions ---
    def to(self, device: str):
        self.tilt_stack = self.tilt_stack.to(device)
        self.tilt_angles = self.tilt_angles.to(device)


class TomographyPixDataset(
    TomographyDatasetBase
):  # Dataset): TODO: Does this need to be a dataset?
    """
    Dataset class for pixel-based tomography, i.e AD, SIRT, WBP, etc...

    TODO:
     - What should the forward pass return? In AD, it should be both the tilt image and the pose.
       In SIRT, it's only the tilt image.
    """

    def __init__(
        self,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        learn_pose: bool = False,
        _token: object | None = None,
    ):
        super().__init__(
            tilt_stack=tilt_stack, tilt_angles=tilt_angles, learn_pose=learn_pose, _token=_token
        )

    def forward(
        self,
        proj_idx: int,
    ) -> DatasetValue:
        """
        Forward pass for pixel-based tomography.
        Returns the full tilt image for the given projection index, and the tilt angle.
        """

        return DatasetValue(
            target=self.tilt_stack[proj_idx],
            tilt_angle=self.tilt_angles[proj_idx],
            pixel_loc=None,
        )

    @classmethod
    def from_data(
        cls,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        learn_pose: bool = False,
    ):
        return cls(tilt_stack=tilt_stack, tilt_angles=tilt_angles, learn_pose=learn_pose)


class TomographyINRDataset(TomographyDatasetBase, Dataset):
    """
    Dataset class for INR-based tomography.

    The two main methods here are that the `forward` call will return the relative pose parameters,
    while `__getitem__` will actually return the pixel values of the tilt stack.

    TODO: I think TomographyINRDataset shouldn't handle the train/val split and will be handled later? Yea this is handled in setup_dataloader in DDP
    """

    def __init__(
        self,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        learn_pose: bool = True,
        seed: int = 42,
        token: object | None = None,
    ):
        super().__init__(tilt_stack, tilt_angles, learn_pose, token)

        self.z1_param = torch.nn.Parameter(torch.zeros(self.learnable_tilts))

    def forward(self, dummy_input: Any = None):
        """
        Forward pass for INR-based tomography. In the forward pass, the only parameters that
        are passed will be the shifts, z1 and z3 Euler angles.
        """
        pass

    def __getitem__(
        self,
        idx: int,
    ) -> DatasetValue:
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

        return self._total_pixels

    @property
    def mode(self) -> Literal["train", "val"]:
        """
        Returns the mode of the dataset.
        """
        return self._mode


class TomographyINRPretrainDataset(Dataset):
    """
    Dataset class for pretraining INR models.
    """

    def __init__(
        self,
        pretrain_target: torch.Tensor,
    ):
        data = pretrain_target.float()

        total_elements = data.numel()
        if total_elements > 1e6:
            sample_size = min(int(1e6), total_elements)
            flat_data = data.flatten()
            indices = torch.randperm(total_elements)[:sample_size]
            sampled_data = flat_data[indices]
            data_quantile = torch.quantile(sampled_data, 0.95)
        else:
            data_quantile = torch.quantile(data, 0.95)

        data = data / data_quantile
        data = torch.permute(data, (2, 1, 0))
        # data = torch.flip(data, dims=(2,))

        self.volume = data.cpu()
        self.N = pretrain_target.shape[0]  # Assumes cubic volume.
        self.total_samples = pretrain_target.shape[0] ** 3

        coords_1d = torch.linspace(-1, 1, self.N)
        x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
        self.coords = torch.stack([x, y, z], dim=-1).reshape(-1, 3).cpu()
        self.targets = self.volume.reshape(-1).cpu()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return {"coords": self.coords[idx], "target": self.targets[idx]}


DatasetModelType = TomographyINRDataset | TomographyPixDataset
