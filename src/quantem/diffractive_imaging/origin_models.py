import math
from typing import TYPE_CHECKING, Self, Tuple

from quantem.core import config

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch

from numpy.typing import NDArray
from torch.nn import functional as F

from quantem.core.datastructures import Dataset
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import (
    validate_tensor,
)
from quantem.diffractive_imaging.ptycho_utils import SimpleBatcher


class CenterOfMassOriginModel(AutoSerialize):
    """ """

    _token = object()

    def __init__(self, dataset: Dataset, device: str | int, _token: object | None = None):
        """ """
        if _token is not self._token:
            raise RuntimeError(
                "Use CenterOfMassOriginModel.from_dataset() to instantiate this class."
            )

        self.device = device
        self.dataset = dataset
        self.num_dps = math.prod(self.dataset.shape[:-2])
        self._tensor = validate_tensor(self.dataset.array, "tensor", dtype=torch.float).to(
            self.device
        )

        # defaults
        self._origin_measured = None
        self._origin_fitted = None
        self._shifted_tensor = None

    @classmethod
    def from_dataset(cls, dataset: Dataset, device: str | int = "cpu") -> Self:
        """ """
        return cls(
            dataset=dataset,
            device=device,
            _token=cls._token,
        )

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @dataset.setter
    def dataset(self, value: Dataset):
        if isinstance(value, Dataset):
            self._dataset = value
        else:
            raise TypeError("dataset must be a valid Dataset.")

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @tensor.setter
    def tensor(self, value: torch.Tensor):
        self._tensor = validate_tensor(value, "tensor", dtype=torch.float).to(self.device)
        self._dataset.array = self._tensor.detach().numpy()

    @property
    def device(self) -> str:
        """This should be of form 'cuda:X' or 'cpu', as defined by quantem.config"""
        if hasattr(self, "_device"):
            return self._device
        else:
            return config.get("device")

    @device.setter
    def device(self, device: str | int | None):
        # allow setting gpu/cpu, but not changing the device from the config gpu device
        if device is not None:
            dev, _id = config.validate_device(device)
            self._device = dev
            try:
                self.to(dev)
            except AttributeError:
                pass

    def calculate_origin(
        self,
        max_batch_size: int | None = None,
    ):
        """ """
        nqx, nqy = self.dataset.shape[-2:]
        tensor_3d = self.tensor.view((-1, nqx, nqy))

        qx = torch.arange(nqx, dtype=torch.float, device=self.device)
        qy = torch.arange(nqy, dtype=torch.float, device=self.device)
        qxa, qya = torch.meshgrid(qx, qy, indexing="ij")

        if max_batch_size is None:
            max_batch_size = self.num_dps

        batcher = SimpleBatcher(self.num_dps, batch_size=max_batch_size, shuffle=False)

        com_measured = torch.empty((self.num_dps, 2), dtype=torch.float, device=self.device)

        for batch_idx in batcher:
            intensities = tensor_3d[batch_idx]
            summed_intensities = torch.sum(intensities, dim=(-2, -1))
            com_measured[batch_idx, 0] = (
                torch.sum(intensities * qxa[None, :, :], dim=(-2, -1)) / summed_intensities
            )
            com_measured[batch_idx, 1] = (
                torch.sum(intensities * qya[None, :, :], dim=(-2, -1)) / summed_intensities
            )

        self.origin_measured = com_measured
        return self

    @property
    def origin_measured(self) -> torch.Tensor:
        return self._origin_measured

    @origin_measured.setter
    def origin_measured(self, value: torch.Tensor):
        self._origin_measured = (
            validate_tensor(value, "measured origin", dtype=torch.float)
            .to(self.device)
            .view((-1, 2))
            .expand((self.num_dps, 2))
        )

    def fit_origin_background(
        self,
        probe_positions: torch.Tensor | NDArray | None = None,
        fit_method: str = "plane",
    ):
        """ """

        if self._origin_measured is None:
            raise ValueError("measured origins not detected. Use self.calculate_origin() first.")

        if probe_positions is None:
            if self.dataset.ndim != 4:
                raise ValueError(
                    "probe positions could not be inferred from dataset, please pass them explicitly."
                )

            nx, ny = self.dataset.shape[:2]

            x = torch.arange(nx, dtype=torch.float, device=self.device)
            y = torch.arange(ny, dtype=torch.float, device=self.device)
            xa, ya = torch.meshgrid(x, y, indexing="ij")
            probe_positions = torch.stack([xa, ya], -1).view((-1, 2))
        else:
            probe_positions = validate_tensor(
                probe_positions, "probe positions", dtype=torch.float
            ).view((-1, 2))
            if probe_positions.shape != self.origin_measured.shape:
                raise ValueError("probe positions shape must match the measured origins.")

        if fit_method == "plane":

            def fit_linear_plane(points: torch.Tensor):
                """ """
                # Covariance matrix
                centroid = points.mean(0)
                centered_points = points - centroid
                covariance_matrix = torch.cov(centered_points.T)

                # Fall back to CPU (to support MPS)
                eigenvectors = torch.linalg.eigh(covariance_matrix.cpu())[1].to(points.device)

                # The normal vector to the plane is the eigenvector corresponding to the smallest eigenvalue
                normal_vector = eigenvectors[:, 0]
                a, b, c = normal_vector

                # Calculate d using the centroid: d = -(ax_c + by_c + cz_c)
                d = -torch.dot(normal_vector, centroid)
                return a, b, c, d

            com_x_pts = torch.concatenate((probe_positions, self.origin_measured[:, 0, None]), 1)
            com_y_pts = torch.concatenate((probe_positions, self.origin_measured[:, 1, None]), 1)

            ax, bx, cx, dx = fit_linear_plane(com_x_pts)
            ay, by, cy, dy = fit_linear_plane(com_y_pts)

            com_fitted_x = (
                probe_positions @ torch.tensor([-ax, -bx], device=self.device) - dx
            ) / cx
            com_fitted_y = (
                probe_positions @ torch.tensor([-ay, -by], device=self.device) - dy
            ) / cy
            com_fitted = torch.stack([com_fitted_x, com_fitted_y], -1)

        elif fit_method == "constant":
            com_fitted = self.origin_measured.mean(0)

        else:
            raise NotImplementedError(
                "only fit_method='plane' and 'constant' are implemented for now."
            )

        self.origin_fitted = com_fitted
        return self

    @property
    def origin_fitted(self) -> torch.Tensor:
        return self._origin_fitted

    @origin_fitted.setter
    def origin_fitted(self, value: torch.Tensor):
        self._origin_fitted = (
            validate_tensor(value, "fitted origin", dtype=torch.float)
            .to(self.device)
            .view((-1, 2))
            .expand((self.num_dps, 2))
        )

    def shift_origin_to(
        self,
        origin_coordinate: Tuple[int | float, int | float] = (0, 0),
        max_batch_size: int | None = None,
        mode: str = "bilinear",
    ):
        if self._origin_fitted is None:
            raise ValueError(
                "fitted origins not detected. Use self.fit_origin_background() first."
            )

        origin_fitted = self.origin_fitted
        H, W = self.dataset.shape[-2:]

        tensor_3d = self.tensor.view((-1, 1, H, W))
        shifted_tensor_3d = torch.empty_like(tensor_3d)
        coordinate = torch.as_tensor(origin_coordinate, dtype=torch.float, device=self.device)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing="ij"
        )
        base_grid = torch.stack((grid_y, grid_x), dim=-1).float()

        if max_batch_size is None:
            max_batch_size = self.num_dps

        batcher = SimpleBatcher(self.num_dps, batch_size=max_batch_size, shuffle=False)

        size_tensor = torch.tensor([H, W], dtype=torch.float, device=self.device)

        for batch_idx in batcher:
            intensities = tensor_3d[batch_idx]

            shift_yx = origin_fitted[batch_idx] - coordinate
            shift_tensor = shift_yx.view(-1, 1, 1, 2)

            shifted_grid = (base_grid[None, ...] + shift_tensor) % size_tensor

            grid_x_norm = 2 * shifted_grid[..., 1] / (W - 1) - 1
            grid_y_norm = 2 * shifted_grid[..., 0] / (H - 1) - 1
            grid = torch.stack((grid_x_norm, grid_y_norm), dim=-1)

            shifted_tensor_3d[batch_idx] = F.grid_sample(
                intensities,
                grid,
                mode=mode,
                padding_mode="zeros",
                align_corners=True,
            )

        self.shifted_tensor = shifted_tensor_3d.view(self.tensor.shape)
        return self

    @property
    def shifted_tensor(self) -> torch.Tensor:
        return self._shifted_tensor

    @shifted_tensor.setter
    def shifted_tensor(self, value: torch.Tensor):
        self._shifted_tensor = validate_tensor(value, "shifted tensor", dtype=torch.float).to(
            self.device
        )

    @staticmethod
    def _estimate_detector_rotation(
        com_normalized: torch.Tensor, rotation_angles_rad: torch.Tensor
    ):
        """ """

        com_measured_x = (
            torch.cos(rotation_angles_rad) * com_normalized[None, :, :, 0]
            - torch.sin(rotation_angles_rad) * com_normalized[None, :, :, 1]
        )
        com_measured_y = (
            torch.sin(rotation_angles_rad) * com_normalized[None, :, :, 0]
            + torch.cos(rotation_angles_rad) * com_normalized[None, :, :, 1]
        )

        com_grad_x_y = com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
        com_grad_y_x = com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
        rotation_curl = torch.mean(torch.abs(com_grad_y_x - com_grad_x_y), dim=(-2, -1))

        return rotation_curl

    def estimate_detector_rotation(
        self, rotation_angles_deg: torch.Tensor | NDArray | None = None
    ):
        """ """
        if rotation_angles_deg is None:
            rotation_angles_deg = torch.arange(-89, 90, 1, device=self.device, dtype=torch.float)

        rotation_angles_deg = torch.as_tensor(
            rotation_angles_deg, dtype=torch.float, device=self.device
        )
        rotation_angles_rad = torch.deg2rad(rotation_angles_deg)[:, None, None]

        com_measured = self.origin_measured.reshape((self.tensor.shape[:2]) + (2,))
        com_fitted = self.origin_fitted.reshape((self.tensor.shape[:2]) + (2,))
        com_normalized = com_measured - com_fitted

        curl_no_transpose = self._estimate_detector_rotation(
            com_normalized,
            rotation_angles_rad,
        )

        curl_transpose = self._estimate_detector_rotation(
            com_normalized.flip([-1]),
            rotation_angles_rad,
        )

        if curl_no_transpose.min() < curl_transpose.min():
            self._detector_transpose = False
            ind_min = torch.argmin(curl_no_transpose)
        else:
            self._detector_transpose = True
            ind_min = torch.argmin(curl_transpose)

        self._detector_rotation_deg = rotation_angles_deg[ind_min].item()

        return self

    @property
    def detector_rotation_deg(self) -> float:
        return self._detector_rotation_deg

    @detector_rotation_deg.setter
    def detector_rotation_deg(self, value: float):
        self._detector_rotation_deg = float(value)

    @property
    def detector_transpose(self) -> float:
        return self._detector_transpose

    @detector_transpose.setter
    def detector_transpose(self, value: float):
        self._detector_transpose = bool(value)

    def forward(
        self,
        max_batch_size: int | None = None,
        fit_origin_bkg: bool = True,
        probe_positions: torch.Tensor | NDArray | None = None,
        fit_method: str = "plane",
        estimate_detector_orientation: bool = True,
        rotation_angles_deg: torch.Tensor | NDArray | None = None,
        shift_to_origin: bool = True,
        origin_coordinate: Tuple[int | float, int | float] = (0, 0),
        mode: str = "bilinear",
    ):
        """
        Runs the full Center-of-Mass origin alignment workflow.

        Args:
            max_batch_size: Maximum batch size to use during CoM calculation and shifting.
            fit_origin_bkg: Whether to fit a smooth background model to the measured origins.
            probe_positions: Probe scan positions (if not inferable from dataset).
            fit_method: Method to fit the origin background ("plane" supported).
            shift_to_origin: Whether to shift all patterns to a common origin.
            origin_coordinate: Target origin position in (qx, qy) coordinates.
            mode: Interpolation mode for shifting ("bicubic", "bilinear", etc.).
            padding_mode: How to handle boundaries during shifting.

        Returns:
            self
        """
        self.calculate_origin(max_batch_size)

        if fit_origin_bkg:
            self.fit_origin_background(probe_positions, fit_method)
            if estimate_detector_orientation:
                self.estimate_detector_rotation(rotation_angles_deg)
            if shift_to_origin:
                self.shift_origin_to(origin_coordinate, max_batch_size, mode)

        return self
