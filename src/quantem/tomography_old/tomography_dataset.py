from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch._tensor import Tensor
from torch.utils.data import Dataset

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import (
    validate_array,
    validate_tensor,
)


class AuxiliaryParams(torch.nn.Module):
    def __init__(self, num_tilts, device, zero_tilt_idx=None, learn_shifts: bool = True):
        super().__init__()

        self.learn_shifts = learn_shifts

        if zero_tilt_idx is None:
            # If not provided, assume first projection is reference
            zero_tilt_idx = 0

        self.zero_tilt_idx = zero_tilt_idx
        self.num_tilts = num_tilts

        # Shifts: only parameterize non-reference tilts
        num_param_tilts = num_tilts - 1
        self.shifts_param = torch.nn.Parameter(
            torch.zeros(num_param_tilts, 2, device=device), requires_grad=learn_shifts
        )

        # Fixed zero shifts for reference
        self.shifts_ref = torch.zeros(1, 2, device=device)

        # Z1 and Z3: parameterize all tilts EXCEPT the reference
        self.z1_param = torch.nn.Parameter(torch.zeros(num_param_tilts, device=device))
        self.z3_param = torch.nn.Parameter(torch.zeros(num_param_tilts, device=device))

        # Fixed zeros for reference tilt
        self.z1_ref = torch.zeros(1, device=device)
        self.z3_ref = torch.zeros(1, device=device)

    def forward(self, dummy_input=None):
        # Reconstruct full arrays with zeros at reference position
        before_shifts = self.shifts_param[: self.zero_tilt_idx]
        after_shifts = self.shifts_param[self.zero_tilt_idx :]
        shifts = torch.cat([before_shifts, self.shifts_ref, after_shifts], dim=0)

        before_z1 = self.z1_param[: self.zero_tilt_idx]
        after_z1 = self.z1_param[self.zero_tilt_idx :]
        z1 = torch.cat([before_z1, self.z1_ref, after_z1], dim=0)

        before_z3 = self.z3_param[: self.zero_tilt_idx]
        after_z3 = self.z3_param[self.zero_tilt_idx :]
        z3 = torch.cat([before_z3, self.z3_ref, after_z3], dim=0)

        return shifts, z1, z3


class TomographyDataset(AutoSerialize, Dataset):
    _token = object()

    """
    A tomography dataset which contains the tilt series, and also instatiates the 
    z1, z3, and shifts of the tilt series.
    
    Idea for this dataset is so that we can avoid moving things around as a torch tensor,
    since the SIRT reconstruction algorthim, and AD reconstruction we have are all torch based.
    """

    def __init__(
        self,
        tilt_series: Tensor,
        tilt_angles: Tensor,
        z1_angles: Tensor,
        z3_angles: Tensor,
        shifts: Tensor,
        clamp: bool,
    ):
        self._tilt_series = tilt_series
        # Enforce Positivity

        self._tilt_angles = tilt_angles
        self._z1_angles = z1_angles
        self._z3_angles = z3_angles
        self._shifts = shifts

        self._initial_tilt_angles = tilt_angles.clone()
        self._initial_z1_angles = z1_angles.clone()
        self._initial_z3_angles = z3_angles.clone()
        self._initial_shifts = shifts.clone()

        # Move everything to CPU
        self.to("cpu")

        # Number of indices (?)

        # Enforce normalization of tilt series
        try:
            tilt_percentile = torch.quantile(self._tilt_series, 0.95)
        except (AttributeError, RuntimeError):
            tilt_percentile = np.quantile(self._tilt_series, 0.95)

        self._tilt_series = self._tilt_series / tilt_percentile
        if clamp:
            self._tilt_series = torch.clamp(self._tilt_series, min=0)

    # --- Class Methods ---
    @classmethod
    def from_data(
        cls,
        tilt_series: Dataset3d | NDArray | Tensor,
        tilt_angles: NDArray | Tensor,
        z1_angles: NDArray | Tensor | None = None,
        z3_angles: NDArray | Tensor | None = None,
        shifts: NDArray | Tensor | None = None,
        clamp: bool = True,
    ):
        """
        tilt_series: (N, H, W)
        tilt_angles: (N,) - In units of degrees, the alpha tilt angle.
        z1_angles: (N,) - In units of degrees, beta tilt angle.
        z3_angles: (N,) - In units of degrees, negative beta tilt angle.
        shifts: (N, 2)

        - The convention we use for projecting down is ZXZ Euler Angles.
        - In theory, Z1 and Z3 should be the same value, except Z3 would be the
        negative value of Z1. However, in some cases this is not the case and
        there could be some merit in also optimizing both angles. However, the
        downside is the rotation becomes less interpretable.
        - The tilt angle can also be optimized.
        """
        validated_tilt_series = torch.tensor(validate_array(tilt_series, "tilt_series"))
        validated_tilt_angles = torch.tensor(validate_array(tilt_angles, "tilt_angles"))

        if z1_angles is not None:
            validated_z1_angles = torch.tensor(validate_array(z1_angles, "z1_angles"))
        else:
            validated_z1_angles = torch.zeros(len(validated_tilt_angles))

        if z3_angles is not None:
            validated_z3_angles = torch.tensor(validate_array(z3_angles, "z3_angles"))
        else:
            validated_z3_angles = torch.zeros(len(validated_tilt_angles))

        if shifts is not None:
            validated_shifts = torch.tensor(validate_array(shifts, "shifts"))
        else:
            validated_shifts = torch.zeros(len(validated_tilt_angles), 2)

        return cls(
            tilt_series=validated_tilt_series,
            tilt_angles=validated_tilt_angles,
            z1_angles=validated_z1_angles,
            z3_angles=validated_z3_angles,
            shifts=validated_shifts,
            clamp=clamp,
            #    name=name,
            #    origin=origin,
            #    sampling=sampling,
            #    units=units,
            #    signal_units=signal_units,
        )

    def to(self, device: str):
        self._tilt_series = self._tilt_series.to(device)
        self._tilt_angles = self._tilt_angles.to(device)
        self._z1_angles = self._z1_angles.to(device)
        self._z3_angles = self._z3_angles.to(device)
        self._shifts = self._shifts.to(device)

    # --- Properties ---

    @property
    def tilt_series(self) -> Tensor:
        return self._tilt_series

    @property
    def tilt_angles(self) -> Tensor:
        return self._tilt_angles

    @property
    def z1_angles(self) -> Tensor:
        return self._z1_angles

    @property
    def z3_angles(self) -> Tensor:
        return self._z3_angles

    @property
    def shifts(self) -> Tensor:
        return self._shifts

    @property
    def initial_tilt_angles(self) -> Tensor:
        return self._initial_tilt_angles

    @property
    def initial_z1_angles(self) -> Tensor:
        return self._initial_z1_angles

    @property
    def initial_z3_angles(self) -> Tensor:
        return self._initial_z3_angles

    @property
    def initial_shifts(self) -> Tensor:
        return self._initial_shifts

    @property
    def num_projections(self) -> int:
        return self._tilt_series.shape[0]

    @property
    def num_pixels(self) -> int:
        return self._tilt_series.shape[0] * self._tilt_series.shape[1] * self._tilt_series.shape[2]

    @property
    def dims(self) -> tuple[int, int, int]:
        return self._tilt_series.shape[0], self._tilt_series.shape[1], self._tilt_series.shape[2]

    def __len__(self) -> int:
        return self.num_pixels

    def __getitem__(self, idx):
        actual_idx = idx

        projection_idx = actual_idx // (self.dims[1] * self.dims[2])
        remaining = actual_idx % (self.dims[1] * self.dims[2])

        # TODO: What if non-square tilt images?
        if self.dims[1] != self.dims[2]:
            raise NotImplementedError("Non-square tilt images are not supported yet.")

        # TODO: row, column
        pixel_i = remaining // self.dims[1]
        pixel_j = remaining % self.dims[1]

        target_value = self._tilt_series[projection_idx, pixel_i, pixel_j]
        phi = self._tilt_angles[projection_idx]

        return {
            "projection_idx": projection_idx,
            "pixel_i": pixel_i,
            "pixel_j": pixel_j,
            "target_value": target_value,
            "phi": phi,
        }

    # --- Setters ---

    @tilt_series.setter
    def tilt_series(self, tilt_series: NDArray | Tensor | Dataset3d) -> None:
        if isinstance(tilt_series, Dataset3d):
            validated_tilt_series = torch.tensor(tilt_series.array)
        elif isinstance(tilt_series, Tensor):
            validated_tilt_series = validate_tensor(tilt_series, "tilt_series")
        else:
            validated_tilt_series = torch.tensor(validate_array(tilt_series, "tilt_series"))

        self._tilt_series = validated_tilt_series

    @tilt_angles.setter
    def tilt_angles(self, tilt_angles: NDArray | Tensor) -> None:
        if tilt_angles.shape[0] != self.tilt_series.shape[0]:
            raise ValueError("Tilt angles must match the number of projections.")

        validated_tilt_angles = torch.tensor(validate_array(tilt_angles, "tilt_angles"))
        self._tilt_angles = validated_tilt_angles

    @z1_angles.setter
    def z1_angles(self, z1_angles: NDArray | Tensor) -> None:
        if z1_angles.shape[0] != self.tilt_series.shape[0]:
            raise ValueError("Z1 angles must match the number of projections.")

        if isinstance(z1_angles, Tensor):
            validated_z1_angles = validate_tensor(z1_angles, "z1_angles")
        else:
            validated_z1_angles = torch.tensor(validate_array(z1_angles, "z1_angles"))
        self._z1_angles = validated_z1_angles

    @z3_angles.setter
    def z3_angles(self, z3_angles: NDArray | Tensor) -> None:
        if z3_angles.shape[0] != self.tilt_series.shape[0]:
            raise ValueError("Z3 angles must match the number of projections.")

        if isinstance(z3_angles, Tensor):
            validated_z3_angles = validate_tensor(z3_angles, "z3_angles")
        else:
            validated_z3_angles = torch.tensor(validate_array(z3_angles, "z3_angles"))
        self._z3_angles = validated_z3_angles

    @shifts.setter
    def shifts(self, shifts: NDArray | Tensor) -> None:
        if shifts.shape[0] != self.tilt_series.shape[0]:
            raise ValueError("Shifts must match the number of projections.")

        if isinstance(shifts, Tensor):
            validated_shifts = validate_tensor(shifts, "shifts")
        else:
            validated_shifts = torch.tensor(validate_array(shifts, "shifts"))

        self._shifts = validated_shifts

    @initial_tilt_angles.setter
    def initial_tilt_angles(self, tilt_angles: NDArray | Tensor) -> None:
        self._initial_tilt_angles = tilt_angles

    @initial_z1_angles.setter
    def initial_z1_angles(self, z1_angles: NDArray | Tensor) -> None:
        self._initial_z1_angles = z1_angles

    @initial_z3_angles.setter
    def initial_z3_angles(self, z3_angles: NDArray | Tensor) -> None:
        self._initial_z3_angles = z3_angles

    @initial_shifts.setter
    def initial_shifts(self, shifts: NDArray | Tensor) -> None:
        self._initial_shifts = shifts

    # TODO: Temp auxiliary params

    def setup_auxiliary_params(
        self, zero_tilt_idx: int = None, device: str = "cpu", learn_shifts: bool = True
    ) -> None:
        if not hasattr(self, "_auxiliary_params"):
            self._auxiliary_params = AuxiliaryParams(
                num_tilts=len(self.tilt_angles),
                device=device,
                zero_tilt_idx=zero_tilt_idx,
                learn_shifts=learn_shifts,
            )

        else:
            print("Auxiliary params already set")

    @property
    def auxiliary_params(self) -> AuxiliaryParams:
        return self._auxiliary_params

    # --- RESET ---

    def reset(self) -> None:
        self._tilt_angles = self._initial_tilt_angles.clone()
        self._z1_angles = self._initial_z1_angles.clone()
        self._z3_angles = self._initial_z3_angles.clone()
        self._shifts = self._initial_shifts.clone()


# TODO: Temporary for use when only doing validation
class TomographyRayDataset(Dataset):
    """
    Dataset for ray-based tomography training.
    Each sample contains: target pixel value, pixel coordinates, phi angle, and auxiliary info.
    """

    def __init__(self, tilt_series, phis, N, val_ratio=0.0, mode="train", seed=42):
        """
        Args:
            tilt_series: [M, N, N] tensor of projections
            phis: [M] tensor of tilt angles
            N: Grid size
        """
        self.tilt_series = tilt_series.cpu()

        self.tilt_series = torch.clamp(self.tilt_series, min=0.0)

        # tilt_percentile = torch.quantile(self.tilt_series, .95)
        tilt_percentile = np.quantile(self.tilt_series.numpy(), 0.95)

        self.tilt_series = self.tilt_series / tilt_percentile

        self.phis = phis.cpu()
        self.N = N
        self.M = tilt_series.shape[0]
        self.total_pixels = self.M * self.N * self.N

        # Create train/val split
        torch.manual_seed(seed)
        all_indices = torch.randperm(self.total_pixels)

        num_val = int(self.total_pixels * val_ratio)
        if mode == "val":
            self.indices = all_indices[:num_val]
        else:  # train
            self.indices = all_indices[num_val:]

        self.mode = mode

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx].item()

        projection_idx = actual_idx // (self.N * self.N)
        remaining = actual_idx % (self.N * self.N)
        pixel_i = remaining // self.N
        pixel_j = remaining % self.N

        target_value = self.tilt_series[projection_idx, pixel_i, pixel_j]
        phi = self.phis[projection_idx]

        return {
            "projection_idx": projection_idx,
            "pixel_i": pixel_i,
            "pixel_j": pixel_j,
            "target_value": target_value,
            "phi": phi,
        }


# Pretrain Volume Dataset
class PretrainVolumeDataset(Dataset):
    def __init__(
        self,
        volume_path: Path | str,
        N: int,
    ):
        self._N = N
        if str(volume_path).endswith(".npy"):
            data = torch.from_numpy(np.load(volume_path)).float()
        elif str(volume_path).endswith(".pt"):
            data = torch.load(volume_path).float()
        else:
            raise ValueError(f"Unsupported file format: {volume_path}")

        # Normalize data
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
        self.N = N
        self.total_samples = N**3

        coords_1d = torch.linspace(-1, 1, N)
        x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
        self.coords = torch.stack([x, y, z], dim=-1).reshape(-1, 3).cpu()
        self.targets = self.volume.reshape(-1).cpu()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return {"coords": self.coords[idx], "target": self.targets[idx]}

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N: int):
        self._N = N
