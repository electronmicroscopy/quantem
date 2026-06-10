from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import Dataset

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.constraints import BaseConstraints, Constraints
from quantem.core.ml.optimizer_mixin import OptimizerMixin
from quantem.tomography.utils import tv_loss_1d

# --- Constraints ---


class DatasetConstraintParams:
    """
    Namespace class for dataset constraint parameter dataclasses and parsing utilities.

    Contains constraint definitions for different tomography dataset types and a
    factory method for instantiating the appropriate constraint class from a dict.

    Supported constraint types
    --------------------------
    BaseTomographyDatasetConstraints
        Base soft constraints for z-position and lateral shift regularization.
    ThroughFocalDatasetConstraints
        Inherits base constraints; not yet implemented.

    Examples
    --------
    >>> DatasetConstraintParams.parse_dict({"name": "base_tomography_dataset", "tv_zs": 0.1})
    BaseTomographyDatasetConstraints(tv_zs=0.1, tv_shifts=0.0)
    >>> DatasetConstraintParams.parse_dict({"type": "base_tomography_dataset"})
    BaseTomographyDatasetConstraints(tv_zs=0.0, tv_shifts=0.0)
    """

    @dataclass
    class BaseTomographyDatasetConstraints(Constraints):
        """
        Soft constraints for a base tomography dataset.

        Attributes
        ----------
        tv_zs : float
            Total variation regularization weight for Z1 and Z3 Euler angles.
        tv_shifts : float
            Total variation regularization weight for X and Y shifts.
        soft_constraint_keys : list[str]
            Constraint fields penalized softly during optimization.
        hard_constraint_keys : list[str]
            Constraint fields enforced strictly (none for this class).
        """

        tv_zs: float = 0.0
        tv_shifts: float = 0.0
        _name: str = "base_tomography_dataset"

        soft_constraint_keys = ["tv_zs", "tv_shifts"]
        hard_constraint_keys = []

    @dataclass
    class ThroughFocalDatasetConstraints(BaseTomographyDatasetConstraints):
        """
        Constraints for a through-focal tomography dataset.

        Inherits all constraints from ``BaseTomographyDatasetConstraints``.
        Currently not implemented — instantiation will raise ``NotImplementedError``.
        """

        pass

    @classmethod
    def parse_dict(
        cls, d: dict
    ) -> "DatasetConstraintParams.BaseTomographyDatasetConstraints | DatasetConstraintParams.ThroughFocalDatasetConstraints":
        """
        Instantiate a dataset constraint dataclass from a configuration dictionary.

        The dictionary must contain a ``'name'`` or ``'type'`` key identifying
        which constraint class to construct. All remaining keys are forwarded as
        keyword arguments to the selected dataclass.

        Parameters
        ----------
        d : dict
            Configuration dictionary. Must include ``'name'`` or ``'type'``
            with one of the following values (case-insensitive):

            - ``'base_tomography_dataset'`` → :class:`BaseTomographyDatasetConstraints`
            - ``'through_focal_dataset'`` → :class:`ThroughFocalDatasetConstraints`
              *(not yet implemented)*

            The value may also be a class ``type`` object, in which case its
            ``__name__`` is used after lower-casing.

        Returns
        -------
        BaseTomographyDatasetConstraints or ThroughFocalDatasetConstraints
            An instance of the appropriate constraint dataclass.

        Raises
        ------
        ValueError
            If neither ``'name'`` nor ``'type'`` is present, if the value is not
            a string or type, or if the name does not match any known dataset
            constraint type.
        NotImplementedError
            If ``'through_focal_dataset'`` is requested, as it is not yet implemented.
        """
        d = dict(d)
        name = d.pop("name", None)
        type_ = d.pop("type", None)
        name = name or type_
        if name is None:
            raise ValueError("Must provide either 'name' or 'type' key")
        if isinstance(name, type):
            name = name.__name__.lower()
        elif isinstance(name, str):
            name = name.lower()
        else:
            raise ValueError(f"Unknown dataset constraint type: {name}")
        if name == "base_tomography_dataset":
            return DatasetConstraintParams.BaseTomographyDatasetConstraints(**d)
        elif name == "through_focal_dataset":
            raise NotImplementedError("Through focal dataset constraints are not implemented yet.")
        else:
            raise ValueError(f"Unknown dataset constraint type: {name.lower()}")


DatasetConstraintsType = (
    DatasetConstraintParams.BaseTomographyDatasetConstraints
    | DatasetConstraintParams.ThroughFocalDatasetConstraints
)


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
        if _token is not self._token:
            raise RuntimeError("Use TomographyPixDataset.from_* to instantiate this class.")

        if not (
            tilt_stack.shape[0] < tilt_stack.shape[1] or tilt_stack.shape[0] < tilt_stack.shape[2]
        ):
            raise ValueError(
                "The number of tilt projections should be in the first dimension of the dataset."
            )

        if type(tilt_stack) is not torch.Tensor:
            tilt_stack = torch.from_numpy(tilt_stack)
        if type(tilt_angles) is not torch.Tensor:
            tilt_angles = torch.from_numpy(tilt_angles)
        max_val = torch.quantile(tilt_stack, 0.95)

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
            _token=cls._token,
        )

    # --- Optimization Parameters ---

    def get_optimization_parameters(self) -> dict[str, list[torch.Tensor]]:
        """Single param group keyed by DEFAULT_OPTIMIZER_KEY.

        Hyperparameters are baked by ``set_optimizer``, not here — return only the tensors,
        matching the ``dict[str, list[tensor]]`` contract the object models use.
        """
        return {self.DEFAULT_OPTIMIZER_KEY: list(self.parameters())}

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
        return int(self._reference_tilt_angle_idx)

    @reference_tilt_idx.setter
    def reference_tilt_idx(self, reference_tilt_idx: int):
        self._reference_tilt_angle_idx = reference_tilt_idx

    @property
    def learnable_tilts(self) -> int:
        # Derived from the tilt series (all tilts minus the fixed reference); there is
        # deliberately no setter -- the old one wrote a private attribute this getter
        # never read, so assignments appeared to succeed while doing nothing.
        return self.tilt_angles.shape[0] - 1

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
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device | str):
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device

    # --- Helper Functions ---
    @abstractmethod
    def to(self, device: torch.device | str):  # type: ignore
        """
        Moves the dataset to the device, and also insantiates the aux params to the device.
        """

        raise NotImplementedError("This method should be implemented in subclasses.")


class TomographyDatasetConstraints(BaseConstraints, TomographyDatasetBase):
    DEFAULT_CONSTRAINTS = DatasetConstraintParams.BaseTomographyDatasetConstraints()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraints: DatasetConstraintParams.BaseTomographyDatasetConstraints = (
            self.DEFAULT_CONSTRAINTS.copy()
        )

    def apply_soft_constraints(self) -> torch.Tensor:
        soft_loss = torch.tensor(0.0, device=self.z1_params.device)
        if self.constraints.tv_zs > 0:
            tv_loss_zs = tv_loss_1d(self.z1_params)
            tv_loss_zs += tv_loss_1d(self.z3_params)
            tv_loss_zs = self.constraints.tv_zs * tv_loss_zs
            soft_loss += tv_loss_zs

        if self.constraints.tv_shifts > 0:
            # Shift params is of shape (N, 2)
            tv_loss_shifts = tv_loss_1d(self.shifts_params[:, 0])
            tv_loss_shifts += tv_loss_1d(self.shifts_params[:, 1])
            tv_loss_shifts = self.constraints.tv_shifts * tv_loss_shifts
            soft_loss += tv_loss_shifts
        return soft_loss

    def apply_hard_constraints(self) -> torch.Tensor:
        """
        No hard constraints have been implemented yet.
        """
        return torch.tensor(0.0)


class TomographyPixDataset(TomographyDatasetConstraints):
    """
    Dataset class for pixel-based tomography, i.e AD, SIRT, WBP, etc...

    These algorithms only require the tilt image in the forward call.
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

    def forward(  # type:ignore
        self,
        proj_idx: int,
    ) -> DatasetValue:
        """
        Forward pass for pixel-based tomography.
        Returns the full tilt image for the given projection index, and the tilt angle.
        """

        return DatasetValue(
            target=self.tilt_stack[proj_idx],
            tilt_angle=self.tilt_angles[proj_idx].item(),
            pixel_loc=None,
        )

    def to(self, device: str | torch.device):
        """
        Moves the tilt stack and tilt_angles to the device, along with other nn.Parameters to the device.
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


class TomographyINRDataset(TomographyDatasetConstraints, Dataset):
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
        _token: object | None = None,
    ):
        super().__init__(tilt_stack, tilt_angles, learn_shift, learn_tilt_axis, _token=_token)

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
        else:
            return torch.zeros_like(shifts), torch.zeros_like(z1), torch.zeros_like(z3)

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
    ) -> dict:
        """
        Gets the item for INR i.e, the project index, pixel value at (i, j), and the tilt angle.
        """

        actual_idx = idx

        projection_idx = actual_idx // (self.tilt_stack.shape[1] * self.tilt_stack.shape[2])
        remaining = actual_idx % (self.tilt_stack.shape[1] * self.tilt_stack.shape[2])

        pixel_i = remaining // self.tilt_stack.shape[1]
        pixel_j = remaining % self.tilt_stack.shape[1]

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

    def to(self, device: torch.device | str):
        self._z1_params = nn.Parameter(self._z1_angles.to(device))
        self._z3_params = nn.Parameter(self._z3_angles.to(device))
        self._shifts_params = nn.Parameter(self._shifts.to(device))

        self._z1_ref = self._z1_ref.to(device)
        self._z3_ref = self._z3_ref.to(device)
        self._shifts_ref = self._shifts_ref.to(device)

        self.device = device
        self.reconnect_optimizer_to_parameters()

    # --- Save learned parameters ---

    def save_parameters(self, path: str):
        """
        Saves the learned parameters to a file.
        """
        torch.save(
            {
                "z1": self._z1_params.detach().cpu(),
                "z3": self._z3_params.detach().cpu(),
                "shifts": self._shifts_params.detach().cpu(),
            },
            path,
        )

    def load_parameters(self, path: str):
        """
        Loads the learned parameters from a file.
        """
        data = torch.load(path)
        self._z1_params = nn.Parameter(data["z1"]).to(self.device)
        self._z3_params = nn.Parameter(data["z3"]).to(self.device)
        self._shifts_params = nn.Parameter(data["shifts"]).to(self.device)
        if self.optimizer is not None:
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
        data = torch.permute(data, (0, 3, 2, 1))
        # data = torch.flip(data, dims=(2,))

        self.volume = data.cpu()
        self.N = pretrain_target.shape[1]  # Assumes cubic volume.
        self.total_samples = pretrain_target.shape[1] ** 3

        coords_1d = torch.linspace(-1, 1, self.N)
        x, y, z = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
        self.coords = torch.stack([x, y, z], dim=-1).reshape(-1, 3).cpu()
        self.targets = self.volume.reshape(-1).cpu()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return {"coords": self.coords[idx], "target": self.targets[idx]}


DatasetModelType = TomographyINRDataset | TomographyPixDataset
