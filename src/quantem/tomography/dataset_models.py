from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

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
    projection_idx: int | None = None  # Only for INRDataset
    pose: tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter] | None = (
        None  # If there is pose optimization.  # Pose is tuple (shifts, z1, z3)
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
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
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
        self.learn_shift = learn_shift
        self.learn_tilt_axis = learn_tilt_axis

        # The reference tilt angle is the one with the smallest absolute tilt angle.
        # I.e, the pose will not be optimized for the reference tilt angle.
        self._reference_tilt_angle_idx = torch.argmin(torch.abs(self.tilt_angles))
        # TODO: Implement AuxParams from old tomography_dataset.py here.

        # TODO: The parameters won't be initialized unless .to(device) is called.
        self._z1_angles = torch.zeros(self.learnable_tilts)
        self._z3_angles = torch.zeros(self.learnable_tilts)
        self._shifts = torch.zeros(self.learnable_tilts, 2)

        # Fixed zeros for reference tilt
        self._z1_ref = torch.zeros(1)
        self._z3_ref = torch.zeros(1)
        self._shifts_ref = torch.zeros(1, 2)

    # --- Class methods ---
    @classmethod
    def from_data(
        cls,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
    ):
        return cls(
            tilt_stack=tilt_stack,
            tilt_angles=tilt_angles,
            learn_shift=learn_shift,
            learn_tilt_axis=learn_tilt_axis,
        )

    # --- Optimization Parameters ---
    # @property
    # def params(self):
    #     # TODO: Need to double check if this is correct way, also need to implement get_optimization_parameters @Arthur!!
    #     raise NotImplementedError("This method should be implemented in subclasses.")

    # --- Forward pass ---
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
    def tilt_stack(self) -> torch.Tensor:
        return self._tilt_stack

    @tilt_stack.setter
    def tilt_stack(self, tilt_stack: torch.Tensor):
        if type(tilt_stack) is not torch.Tensor:
            print("Converting tilt stack to torch.Tensor")
            tilt_stack = torch.from_numpy(tilt_stack)

        self._tilt_stack = tilt_stack

    @property
    def tilt_angles(self) -> torch.Tensor:
        return self._tilt_angles

    @tilt_angles.setter
    def tilt_angles(self, tilt_angles: torch.Tensor):
        if type(tilt_angles) is not torch.Tensor:
            print("Converting tilt angles to torch.Tensor")
            tilt_angles = torch.from_numpy(tilt_angles)

        self._tilt_angles = tilt_angles

    @property
    def learn_shift(self) -> bool:
        return self._learn_shift

    @learn_shift.setter
    def learn_shift(self, learn_shift: bool):
        self._learn_shift = learn_shift

    @property
    def learn_tilt_axis(self) -> bool:
        return self._learn_tilt_axis

    @learn_tilt_axis.setter
    def learn_tilt_axis(self, learn_tilt_axis: bool):
        self._learn_tilt_axis = learn_tilt_axis

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

    @property
    def z1_params(self) -> torch.nn.Parameter:
        return self._z1_params

    @z1_params.setter
    def z1_params(self, z1_angles: torch.Tensor, device: str):
        self._z1_params = nn.Parameter(z1_angles.to(device))

    @property
    def z3_params(self) -> torch.nn.Parameter:
        return self._z3_params

    @z3_params.setter
    def z3_params(self, z3_angles: torch.Tensor, device: str):
        self._z3_params = nn.Parameter(z3_angles.to(device))

    @property
    def shifts_params(self) -> torch.nn.Parameter:
        return self._shifts_params

    @shifts_params.setter
    def shifts_params(self, shifts: torch.Tensor, device: str):
        self._shifts_params = nn.Parameter(shifts.to(device))

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device: str):
        self._device = device

    # --- Helper Functions ---
    def to(self, device: str):
        """
        Moves the dataset to the device, and also insantiates the aux params to the device.
        """
        self.tilt_stack = self.tilt_stack.to(device)
        self.tilt_angles = self.tilt_angles.to(device)

        self._z1_params = nn.Parameter(self._z1_angles.to(device))
        self._z3_params = nn.Parameter(self._z3_angles.to(device))
        self._shifts_params = nn.Parameter(self._shifts.to(device))

        self._z1_ref = self._z1_ref.to(device)
        self._z3_ref = self._z3_ref.to(device)
        self._shifts_ref = self._shifts_ref.to(device)

        self.device = device

    def get_optimization_parameters(self) -> list[nn.Parameter]:
        """
        Get the parameters that should be optimized for this model.
        """
        return list[nn.Parameter](self.params)


class TomographyPixDataset(TomographyDatasetBase):
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
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
        _token: object | None = None,
    ):
        super().__init__(
            tilt_stack=tilt_stack,
            tilt_angles=-tilt_angles,  # TODO: Flip the tilt angles to be negative to match the convention of INR.
            learn_shift=learn_shift,
            learn_tilt_axis=learn_tilt_axis,
            _token=_token,
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
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
        seed: int = 42,
        token: object | None = None,
    ):
        super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, token)

    # --- Forward Pass w/ Params Method for OptimizerMixin ---
    def forward(self, dummy_input: Any = None):
        """
        Forward pass for INR-based tomography. In the forward pass, the only parameters that
        are passed will be the shifts, z1 and z3 Euler angles.
        """

        first_half_shifts = self.shifts_params[: self.reference_tilt_idx]
        second_half_shifts = self.shifts_params[self.reference_tilt_idx :]
        shifts = torch.cat([first_half_shifts, self._shifts_ref, second_half_shifts], dim=0)

        first_half_z1 = self.z1_params[: self.reference_tilt_idx]
        second_half_z1 = self.z1_params[self.reference_tilt_idx :]
        z1 = torch.cat([first_half_z1, self._z1_ref, second_half_z1], dim=0)

        first_half_z3 = self.z3_params[: self.reference_tilt_idx]
        second_half_z3 = self.z3_params[self.reference_tilt_idx :]
        z3 = torch.cat([first_half_z3, self._z3_ref, second_half_z3], dim=0)

        if self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3
        elif self.learn_shift:
            return shifts, torch.zeros_like(z1), torch.zeros_like(z3)
        elif self.learn_tilt_axis:
            return torch.zeros_like(shifts), z1, z3
        elif self.learn_shift and self.learn_tilt_axis:
            return shifts, z1, z3
        else:
            return torch.zeros_like(shifts), torch.zeros_like(z1), torch.zeros_like(z3)

    @property
    def params(self):
        return self.parameters()

    def get_coords(
        self, batch: dict[str, torch.Tensor], N: int, num_samples_per_ray: int
    ) -> torch.Tensor:
        pixel_i = batch["pixel_i"].float().to(self.device, non_blocking=True)
        pixel_j = batch["pixel_j"].float().to(self.device, non_blocking=True)
        # target_values = batch["target_value"].to(self.device, non_blocking=True)
        phis = batch["phi"].to(self.device, non_blocking=True)
        projection_indices = batch["projection_idx"].to(self.device, non_blocking=True)
        with torch.no_grad():
            batch_ray_coords = self.create_batch_rays(pixel_i, pixel_j, N, num_samples_per_ray)

        shifts, z1_params, z3_params = self.forward(None)
        batch_shifts = torch.index_select(shifts, 0, projection_indices)
        batch_z1 = torch.index_select(z1_params, 0, projection_indices)
        batch_z3 = torch.index_select(z3_params, 0, projection_indices)

        transformed_rays = self.transform_batch_rays(
            batch_ray_coords,
            z1=batch_z1,
            x=phis,
            z3=batch_z3,
            shifts=batch_shifts,
            N=N,
            sampling_rate=1.0,
        )
        all_coords = transformed_rays.view(-1, 3)

        all_coords = all_coords.to(self.device, dtype=torch.float32, non_blocking=True)
        return all_coords

    @staticmethod
    @torch.compile(mode="reduce-overhead")
    def create_batch_rays(
        pixel_i: torch.Tensor, pixel_j: torch.Tensor, N: int, num_samples_per_ray: int
    ) -> torch.Tensor:
        batch_size = len(pixel_i)
        x_coords = (pixel_j / (N - 1)) * 2 - 1
        y_coords = (pixel_i / (N - 1)) * 2 - 1
        z_coords = torch.linspace(-1, 1, num_samples_per_ray, device=pixel_i.device)

        rays = torch.zeros(batch_size, num_samples_per_ray, 3, device=pixel_i.device)

        rays[:, :, 0] = x_coords.unsqueeze(1)
        rays[:, :, 1] = y_coords.unsqueeze(1)
        rays[:, :, 2] = z_coords.unsqueeze(0)

        return rays

    @staticmethod
    def transform_batch_rays(
        rays: torch.Tensor,
        z1: torch.Tensor,
        x: torch.Tensor,
        z3: torch.Tensor,
        shifts: torch.Tensor,
        N: int,
        sampling_rate: float,
    ) -> torch.Tensor:
        shift_x_norm = (shifts[:, 0:1] * sampling_rate * 2) / (N - 1)
        shift_y_norm = (shifts[:, 1:2] * sampling_rate * 2) / (N - 1)

        rays_x = rays[:, :, 0] - shift_x_norm
        rays_y = rays[:, :, 1] - shift_y_norm
        rays_z = rays[:, :, 2]

        theta = torch.deg2rad(-z3).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_rot1 = cos_t * rays_x - sin_t * rays_y
        rays_y_rot1 = sin_t * rays_x + cos_t * rays_y
        rays_z_rot1 = rays_z

        theta = torch.deg2rad(x).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_rot2 = rays_x_rot1
        rays_y_rot2 = cos_t * rays_y_rot1 - sin_t * rays_z_rot1
        rays_z_rot2 = sin_t * rays_y_rot1 + cos_t * rays_z_rot1

        theta = torch.deg2rad(-z1).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_final = cos_t * rays_x_rot2 - sin_t * rays_y_rot2
        rays_y_final = sin_t * rays_x_rot2 + cos_t * rays_y_rot2
        rays_z_final = rays_z_rot2

        transformed_rays = torch.stack([rays_x_final, rays_y_final, rays_z_final], dim=2)

        return transformed_rays

    @staticmethod
    @torch.compile(mode="reduce-overhead")
    def integrate_rays(
        rays: torch.Tensor, num_samples_per_ray: int, target_values_len: int
    ) -> torch.Tensor:
        ray_densities = rays.view(
            target_values_len,
            num_samples_per_ray,
        )
        step_size = 2.0 / (num_samples_per_ray - 1)

        predicted_values = ray_densities.sum(dim=1) * step_size

        return predicted_values

    # --- Torch Dataset Methods ---
    def __getitem__(
        self,
        idx: int,
    ) -> DatasetValue:
        """
        Gets the item for INR i.e, the project index, pixel value at (i, j), and the tilt angle.
        """

        actual_idx = idx

        projection_idx = actual_idx // (self.tilt_stack.shape[1] * self.tilt_stack.shape[2])
        remaining = actual_idx % (self.tilt_stack.shape[1] * self.tilt_stack.shape[2])

        pixel_i = remaining // self.tilt_stack.shape[1]
        pixel_j = remaining % self.tilt_stack.shape[1]

        # return DatasetValue(
        #     target=self.tilt_stack[projection_idx, pixel_i, pixel_j],
        #     tilt_angle=self.tilt_angles[projection_idx],
        #     projection_idx=projection_idx,
        #     pixel_loc=(pixel_i, pixel_j),
        # )
        return {
            "projection_idx": torch.tensor(projection_idx),
            "pixel_i": torch.tensor(pixel_i),
            "pixel_j": torch.tensor(pixel_j),
            "phi": self.tilt_angles[projection_idx],  # tensor
            "target_value": self.tilt_stack[projection_idx, pixel_i, pixel_j],  # tensor
        }

    def __len__(
        self,
    ):
        """
        Returns the number of pixels in the tilt stack.
        """
        N = max(self.tilt_stack.shape)
        return self.tilt_stack.shape[0] * N * N

    def to(self, device: str):
        self._z1_params = nn.Parameter(self._z1_angles.to(device))
        self._z3_params = nn.Parameter(self._z3_angles.to(device))
        self._shifts_params = nn.Parameter(self._shifts.to(device))

        self._z1_ref = self._z1_ref.to(device)
        self._z3_ref = self._z3_ref.to(device)
        self._shifts_ref = self._shifts_ref.to(device)

        self.device = device
        self.reconnect_optimizer_to_parameters()


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

class TomographyThroughFocalINRDataset(TomographyINRDataset):
    """
    Dataset class for INR-based tomography that assumes non-parallel illumination condition.
    Inherits from TomographyINRDataset.
    """

    def __init__(
        self,
        tilt_stack: Dataset3d | NDArray | torch.Tensor,
        tilt_angles: NDArray | torch.Tensor,
        ## Through focal parameters:
        convergence_angle: float,
        z_focus: float | torch.Tensor = 0.0,
        num_rays: int = 1,
        random_rays: bool = False,
        learn_z_focus: bool = False,
        ## optional parameters
        learn_shift: bool = True,
        learn_tilt_axis: bool = True,
        seed: int = 42,
        token: object | None = None,
    ):
        super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, token)
        self.num_rays = int(num_rays)

        self._convergence_angle = convergence_angle
        self._random_rays = random_rays

        if not isinstance(z_focus, torch.Tensor):
            z_focus = torch.tensor(z_focus, dtype=torch.float32)

        if learn_z_focus:
            self._z_focus = torch.nn.Parameter(z_focus)
        else:
            self._z_focus = z_focus

    @property
    def convergence_angle(self) -> float:
        return self._convergence_angle

    @property
    def random_rays(self) -> bool:
        return self._random_rays

    @property
    def z_focus(self) -> torch.Tensor:
        return self._z_focus

    def get_coords(
        self, batch: dict[str, torch.Tensor], N: int, num_samples_per_ray: int,
    ) -> torch.Tensor:
        num_rays = self.num_rays
        convergence_angle = self._convergence_angle
        random_rays = self._random_rays
        pixel_i = batch["pixel_i"].float().to(self.device, non_blocking=True)
        pixel_j = batch["pixel_j"].float().to(self.device, non_blocking=True)
        # target_values = batch["target_value"].to(self.device, non_blocking=True)
        phis = batch["phi"].to(self.device, non_blocking=True)
        projection_indices = batch["projection_idx"].to(self.device, non_blocking=True)
        with torch.no_grad():
            batch_ray_coords = self.create_batch_rays(
                pixel_i,
                pixel_j,
                N,
                num_samples_per_ray,
                num_rays,
                convergence_angle,
                random_rays,
                )

        shifts, z1_params, z3_params = self.forward(None)
        batch_shifts = torch.index_select(shifts, 0, projection_indices)
        batch_z1 = torch.index_select(z1_params, 0, projection_indices)
        batch_z3 = torch.index_select(z3_params, 0, projection_indices)

        transformed_rays = self.transform_batch_rays(
            batch_ray_coords,
            z1=batch_z1,
            x=phis,
            z3=batch_z3,
            shifts=batch_shifts,
            N=N,
            sampling_rate=1.0,
        )
        all_coords = transformed_rays.view(-1, 3)

        all_coords = all_coords.to(self.device, dtype=torch.float32, non_blocking=True)
        return all_coords

    # @staticmethod
    @torch.compile(mode="reduce-overhead")
    def create_batch_rays(
        self,
        pixel_i: torch.Tensor, pixel_j: torch.Tensor, N: int, num_samples_per_ray: int,
        num_rays: int, convergence_angle: float, random_rays: bool,
    ) -> torch.Tensor:
        batch_size = len(pixel_i)
        x_coords_0 = (pixel_j / (N - 1)) * 2 - 1
        y_coords_0 = (pixel_i / (N - 1)) * 2 - 1
        z_coords = torch.linspace(-1, 1, num_samples_per_ray, device=pixel_i.device)

        def get_theta_phi(
            convergence_angle: float,
            num_rays: int,
            device: torch.device | None = None,
            random_rays: bool = False,
        ):
            if device is None:
                device = torch.device("cpu")

            num_rays = int(num_rays)

            # random_rays sampling method
            if random_rays:
                theta = torch.rand(num_rays, device = device) * 2 * torch.pi
                phi = torch.rand(num_rays, device = device) * convergence_angle

            else:
                # 1 ray: single central ray
                if num_rays == 1:
                    theta = torch.zeros(1, device=device)
                    phi = torch.zeros(1, device=device)

                # 2 or 3 rays: single ring
                elif num_rays == 2 or num_rays == 3:
                    rand_offset = torch.rand(1, device=device) * 2 * torch.pi
                    theta = (
                        torch.linspace(0, 2 * torch.pi, num_rays + 1, device=device)[:-1]
                        + rand_offset
                    )
                    phi = torch.full((num_rays,), 0.5 * convergence_angle, device=device) # for the case where there is no center ray, use half of the convergence angle.

                # 4–8 rays: central + ring
                elif 3 < num_rays < 9:
                    rand_offset = torch.rand(1, device=device) * 2 * torch.pi
                    theta0 = torch.zeros(1, device=device)
                    phi0 = torch.zeros(1, device=device)
                    theta_ring = (
                        torch.linspace(0, 2 * torch.pi, num_rays, device=device)[:-1]
                        + rand_offset
                    )
                    phi_ring = torch.full(
                        (num_rays - 1,), 1 * convergence_angle, device=device
                    )
                    theta = torch.cat((theta0, theta_ring))
                    phi = torch.cat((phi0, phi_ring))

                # 9 rays: two rings. Rays at 1/3 and 2/3 * convergence angle
                elif num_rays == 9:
                    rand_offset = torch.rand(1, device=device) * 2 * torch.pi
                    theta_inner = (
                        torch.linspace(0, 2 * torch.pi, 4, device=device)[:-1]
                        + rand_offset
                    )
                    phi_inner = torch.full((3,), convergence_angle / 3, device=device)
                    theta_outer = (
                        torch.linspace(0, 2 * torch.pi, 7, device=device)[:-1]
                        + rand_offset
                    )
                    phi_outer = torch.full((6,), 2 * convergence_angle / 3, device=device)
                    theta = torch.cat((theta_inner, theta_outer))
                    phi = torch.cat((phi_inner, phi_outer))

                # 10 rays: two rings, one central ray. Rays at 0, 1/2 and 1 * convergence angle
                elif num_rays == 10:
                    rand_offset = torch.rand(1, device=device) * 2 * torch.pi
                    theta0 = torch.zeros(1, device=device)
                    phi0 = torch.zeros(1, device=device)

                    theta_inner = (
                        torch.linspace(0, 2 * torch.pi, 4, device=device)[:-1]
                        + rand_offset
                    )
                    phi_inner = torch.full((3,), convergence_angle / 3, device=device)
                    theta_outer = (
                        torch.linspace(0, 2 * torch.pi, 7, device=device)[:-1]
                        + rand_offset
                    )
                    phi_outer = torch.full((6,), 2 * convergence_angle / 3, device=device)
                    theta = torch.cat((theta0, theta_inner, theta_outer))
                    phi = torch.cat((phi0, phi_inner, phi_outer))

                # 19 rays: central + two rings. Rays at 0, 1/2 and 1 * convergence angle
                elif num_rays == 19:
                    rand_offset = torch.rand(1, device=device) * 2 * torch.pi
                    theta0 = torch.zeros(1, device=device)
                    phi0 = torch.zeros(1, device=device)
                    theta_inner = (
                        torch.linspace(0, 2 * torch.pi, 7, device=device)[:-1]
                        + rand_offset
                    )
                    phi_inner = torch.full((6,), convergence_angle / 2, device=device)
                    theta_outer = (
                        torch.linspace(0, 2 * torch.pi, 13, device=device)[:-1]
                        + rand_offset
                    )
                    phi_outer = torch.full((12,), 1 * convergence_angle, device=device)
                    theta = torch.cat((theta0, theta_inner, theta_outer))
                    phi = torch.cat((phi0, phi_inner, phi_outer))

                # Unsupported ray counts
                else:
                    raise ValueError(
                        f"Unsupported num_rays={num_rays}. "
                        "Supported values are: 1, 2, 3, 4–8, 9, 10, 19."
                    )

            return theta, phi

        theta, phi = get_theta_phi(
            convergence_angle = convergence_angle,
            num_rays = num_rays,
            device = pixel_i.device,
            random_rays = random_rays,
        )

        dz = z_coords[None, :] - self._z_focus
        dx = torch.tan(phi)[:, None] * torch.cos(theta)[:, None] * dz
        dy = torch.tan(phi)[:, None] * torch.sin(theta)[:, None] * dz

        x_coords = x_coords_0[:, None, None] + dx[None, :, :]
        y_coords = y_coords_0[:, None, None] + dy[None, :, :]
        z_coords = z_coords[None, None, :].expand(
            batch_size, num_rays, num_samples_per_ray
        )

        x_coords = x_coords.reshape(batch_size, num_rays * num_samples_per_ray)
        y_coords = y_coords.reshape(batch_size, num_rays * num_samples_per_ray)
        z_coords = z_coords.reshape(batch_size, num_rays * num_samples_per_ray)

        rays = torch.zeros(
            batch_size,
            num_rays * num_samples_per_ray,
            3,
            device=x_coords.device,
        )

        rays[:, :, 0] = x_coords
        rays[:, :, 1] = y_coords
        rays[:, :, 2] = z_coords

        return rays

    # @staticmethod
    @torch.compile(mode="reduce-overhead")
    def integrate_rays(
        self,
        rays: torch.Tensor, num_samples_per_ray: int, target_values_len: int,
    ) -> torch.Tensor:
        num_rays = self.num_rays
        ray_densities = rays.view(
            target_values_len,
            # num_samples_per_ray,
            num_samples_per_ray * num_rays,
        )
        step_size = 2.0 / (num_samples_per_ray - 1)

        predicted_values = ray_densities.sum(dim=1) * step_size

        return predicted_values

    @staticmethod
    def transform_batch_rays(
        rays: torch.Tensor,
        z1: torch.Tensor,
        x: torch.Tensor,
        z3: torch.Tensor,
        shifts: torch.Tensor,
        N: int,
        sampling_rate: float,
    ) -> torch.Tensor:

        shift_x_norm = (shifts[:, 0:1] * sampling_rate * 2) / (N - 1)
        shift_y_norm = (shifts[:, 1:2] * sampling_rate * 2) / (N - 1)

        shift_x_norm = shift_x_norm.expand(-1, rays.shape[1])
        shift_y_norm = shift_y_norm.expand(-1, rays.shape[1])

        rays_x = rays[:, :, 0] - shift_x_norm
        rays_y = rays[:, :, 1] - shift_y_norm
        rays_z = rays[:, :, 2]

        theta = torch.deg2rad(-z3).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_rot1 = cos_t * rays_x - sin_t * rays_y
        rays_y_rot1 = sin_t * rays_x + cos_t * rays_y
        rays_z_rot1 = rays_z

        theta = torch.deg2rad(x).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_rot2 = rays_x_rot1
        rays_y_rot2 = cos_t * rays_y_rot1 - sin_t * rays_z_rot1
        rays_z_rot2 = sin_t * rays_y_rot1 + cos_t * rays_z_rot1

        theta = torch.deg2rad(-z1).view(-1, 1)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        rays_x_final = cos_t * rays_x_rot2 - sin_t * rays_y_rot2
        rays_y_final = sin_t * rays_x_rot2 + cos_t * rays_y_rot2
        rays_z_final = rays_z_rot2

        transformed_rays = torch.stack([rays_x_final, rays_y_final, rays_z_final], dim=2)

        return transformed_rays


DatasetModelType = TomographyINRDataset | TomographyPixDataset | TomographyThroughFocalINRDataset