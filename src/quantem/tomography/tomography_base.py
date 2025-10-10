import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch._tensor import Tensor
from tqdm.auto import tqdm

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.visualization.visualization import show_2d
from quantem.imaging.drift import cross_correlation_shift
from quantem.tomography.object_models import ObjectModelType, ObjectVoxelwise
from quantem.tomography.tomography_dataset import TomographyDataset
from quantem.tomography.tomography_logger import LoggerTomography


class TomographyBase(AutoSerialize):
    _token = object()

    DEFAULT_HARD_CONSTRAINTS = {
        "positivity": False,
        "shrinkage": False,
        "circular_mask": False,
        "fourier_filter": False,
    }

    DEFAULT_SOFT_CONSTRAINTS = {
        "tv_vol": 0.0,
    }

    def __init__(
        self,
        dataset: TomographyDataset,
        volume_obj: ObjectModelType,  # ObjectDIP?
        device: str = "cuda",
        # ABF/HAADF property
        logger: LoggerTomography | None = None,
        _token: object | None = None,
    ):
        """Initialize a Tomography object.

        Parameters
        ----------
        array : NDArray | Any
            The underlying 3D array data
        name : str
            A descriptive name for the dataset
        """

        # if _token is not self._token:
        #     raise RuntimeError(
        #         "This class is not meant to be instantiated directly. Use the from_data method."
        #     )

        self._device = device
        self._dataset = dataset
        self._volume_obj = volume_obj
        self._loss = []
        self._mode = []

        self._hard_constraints = self.DEFAULT_HARD_CONSTRAINTS.copy()
        self._soft_constraints = self.DEFAULT_SOFT_CONSTRAINTS.copy()

        self._logger = None

    @classmethod
    def from_data(
        cls,
        tilt_series: Dataset3d | NDArray | Tensor,
        tilt_angles: NDArray | Tensor,
        z1_angles: NDArray | Tensor | None = None,
        z3_angles: NDArray | Tensor | None = None,
        shifts: NDArray | Tensor | None = None,
        volume_obj: NDArray | Dataset3d | ObjectModelType | None = None,
        device: str = "cpu",
    ):
        device = device.lower()

        dataset = TomographyDataset.from_data(
            tilt_series=tilt_series,
            tilt_angles=tilt_angles,
            z1_angles=z1_angles,
            z3_angles=z3_angles,
            shifts=shifts,
        )

        dataset.to(device)

        if volume_obj is None:
            max_shape = max(dataset.tilt_series.shape)
            volume_obj = ObjectVoxelwise(
                volume_shape=(max_shape, max_shape, max_shape),
                device=device,
            )
        elif isinstance(volume_obj, Dataset3d):
            volume = torch.from_numpy(volume_obj.array)
            volume_obj = ObjectVoxelwise(
                volume_shape=volume_obj.shape, device=device, initial_volume=volume
            )
            volume_obj.obj = volume
        elif isinstance(volume_obj, np.ndarray):
            volume = torch.from_numpy(volume_obj)
            volume_obj = ObjectVoxelwise(
                volume_shape=volume_obj.shape, device=device, initial_volume=volume
            )
            volume_obj.obj = volume
        else:
            raise ValueError("volume_obj must be a Dataset3d, NDArray ObjectModelType")

        return cls(
            dataset=dataset,
            volume_obj=volume_obj,
            device=device,
            _token=cls._token,
        )

    # --- Properties ---
    @property
    def dataset(self) -> TomographyDataset:
        """Tomography dataset."""

        return self._dataset

    @dataset.setter
    def dataset(
        self,
        tilt_series: Dataset3d | NDArray | TomographyDataset,
        tilt_angles: NDArray | Tensor,
        z1_angles: NDArray | Tensor | None = None,
        z3_angles: NDArray | Tensor | None = None,
        shifts: NDArray | Tensor | None = None,
        # name: str | None = None,
        # origin: NDArray | tuple | list | float | int | None = None,
        # sampling: NDArray | tuple | list | float | int | None = None,
        # units: list[str] | tuple | list | None = None,
        # signal_units: str = "arb. units",
    ):
        """Set the tilt series dataset."""

        if not isinstance(tilt_series, TomographyDataset):
            dataset = TomographyDataset.from_array(
                array=tilt_series,
                tilt_angles=tilt_angles,
                z1_angles=z1_angles,
                z3_angles=z3_angles,
                shifts=shifts,
            )

        self._dataset = dataset

    @property
    def volume_obj(self) -> Dataset3d | ObjectModelType | None:
        """Reconstruction volume dataset."""

        return self._volume_obj

    @volume_obj.setter
    # TODO: add support for ObjectModelType
    def volume_obj(self, volume_obj: Dataset3d | NDArray):
        """Set the reconstruction volume dataset."""
        if isinstance(volume_obj, ObjectModelType):
            self._volume_obj = volume_obj
        elif not isinstance(volume_obj, Dataset3d):
            volume_obj = Dataset3d.from_array(
                array=volume_obj,
                # name=self._tilt_series.name,
                # origin=self._tilt_series.origin,
                # sampling=self._tilt_series.sampling,
                # units=self._tilt_series.units,
                # signal_units=self._tilt_series.signal_units,
            )
        elif isinstance(volume_obj, Dataset3d):
            self._volume_obj = volume_obj
        else:
            raise ValueError("volume_obj must be a Dataset3d or ObjectModelType")

    @property
    def device(self) -> str:
        """Computation device."""

        return self._device

    @device.setter
    def device(self, device: str):
        """Set the computation device."""

        # if "cuda" not in device or "gpu" not in device:
        #     raise NotImplementedError("Tomography not currently supported on CPU.")

        self._device = device

    @property
    def loss(self) -> list:
        """List of loss values during reconstruction."""

        return self._loss

    @loss.setter
    def loss(self, loss: list):
        """Set the loss values during reconstruction."""

        if not isinstance(loss, list):
            raise TypeError("Loss must be a list.")

        self._loss = loss

    @property
    def mode(self) -> list:
        """List of modes used during reconstruction."""

        return self._mode

    @property
    def epochs(self) -> int:
        """Number of epochs used during reconstruction."""
        return len(self.loss)

    @property
    def logger(self) -> LoggerTomography:
        return self._logger

    @logger.setter
    def logger(self, logger: LoggerTomography):
        if not isinstance(logger, LoggerTomography):
            raise TypeError("Logger must be a LoggerTomography")

        self._logger = logger

    # --- Constraints ---

    @property
    def hard_constraints(self) -> dict:
        """Hard constraints for the reconstruction."""
        return self._hard_constraints

    @hard_constraints.setter
    def hard_constraints(self, hard_constraints: dict):
        """Set the hard constraints for the reconstruction."""

        gkeys = self.DEFAULT_HARD_CONSTRAINTS.keys()
        for key, value in hard_constraints.items():
            if key not in gkeys:
                raise KeyError(f"Invalid object constraint key '{key}', allowed keys are {gkeys}")
            self._hard_constraints[key] = value

        self._hard_constraints = hard_constraints

    @property
    def soft_constraints(self) -> dict:
        """Soft constraints for the reconstruction."""
        return self._soft_constraints

    @soft_constraints.setter
    def soft_constraints(self, soft_constraints: dict):
        """Set the soft constraints for the reconstruction."""

        gkeys = self.DEFAULT_SOFT_CONSTRAINTS.keys()
        for key, value in soft_constraints.items():
            if key not in gkeys:
                raise KeyError(f"Invalid object constraint key '{key}', allowed keys are {gkeys}")
            self._soft_constraints[key] = value

        self._soft_constraints = soft_constraints

    # --- RESET ---

    def reset_recon(self) -> None:
        self.volume_obj.reset()
        self.dataset.reset()
        self.loss = []
        self.hard_constraints = self.DEFAULT_HARD_CONSTRAINTS.copy()
        self.soft_constraints = self.DEFAULT_SOFT_CONSTRAINTS.copy()

        self._optimizers = {}
        self._schedulers = {}

    # --- Preprocessing ---

    """
    TODO
    1. Implement tilt series cross-correlation alignment
    2. Background subtraction (for ABF)
    3. COM Alignment
    4. Masking
    5. Drift Correction
    """

    def cross_corr_alignment(
        self,
        upsample_factor: int = 1,
        overwrite: bool = False,
    ):
        # TODO: This needs to be able to work with torch tensors.

        placeholder_tilt_series = self.dataset.tilt_series.clone().detach().cpu().numpy()

        aligned_tilt_series = np.zeros_like(placeholder_tilt_series)
        aligned_tilt_series[0] = placeholder_tilt_series[0]
        shifts = []
        num_imgs = placeholder_tilt_series.shape[1]

        pbar = tqdm(range(num_imgs - 1), desc="Cross-correlation alignment")

        for i in pbar:
            shift, aligned_img = cross_correlation_shift(
                placeholder_tilt_series[i],
                placeholder_tilt_series[i + 1],
                upsample_factor=upsample_factor,
                return_shifted_image=True,
            )

            aligned_tilt_series[i + 1] = aligned_img
            shifts.append(shift)

        if overwrite:
            # TODO: Check this overwrite idea, maybe also need to save the relative shifts?
            self.dataset.tilt_series = np.array(aligned_tilt_series)

        return np.array(aligned_tilt_series), np.array(shifts)

    # --- Postprocessing ---

    """
    TODO
    1. Apply circular mask
    """

    def circular_mask(self, shape, radius, center=None, dtype=torch.float32, device="cpu"):
        """Generate a 2D circular mask of given shape and radius."""
        H, W = shape

        if center is None:
            center = (H // 2, W // 2)
        y = torch.arange(H, dtype=dtype, device=device).view(-1, 1)
        x = torch.arange(W, dtype=dtype, device=device).view(1, -1)
        dist_sq = (x - center[1]) ** 2 + (y - center[0]) ** 2
        return (dist_sq <= radius**2).to(dtype)

    def recon_vol_circular_mask(self, radii):
        """
        Apply 2D circular masks along all three axes of a 3D volume.

        Args:
            volume (torch.Tensor): 3D tensor of shape (H, W, D)
            radii (tuple): (r0, r1, r2) for axes 0, 1, 2
        Returns:
            masked_volume: tensor with all masks applied
        """
        H, W, D = self.volume_obj.array.shape
        device = self.device
        dtype = torch.float32
        volume_obj = torch.tensor(
            self.volume_obj.array,
            device=self.device,
            dtype=dtype,
        )
        # Masks for each axis
        mask0 = self.circular_mask((W, D), radii[0], dtype=dtype, device=device).unsqueeze(
            0
        )  # shape (1, W, D)
        mask1 = self.circular_mask((H, D), radii[1], dtype=dtype, device=device).unsqueeze(
            1
        )  # shape (H, 1, D)
        mask2 = self.circular_mask((H, W), radii[2], dtype=dtype, device=device).unsqueeze(
            2
        )  # shape (H, W, 1)

        # Broadcast and multiply all masks together
        total_mask = mask0 * mask1 * mask2  # shape (H, W, D)

        volume_obj = volume_obj * total_mask
        volume_obj = volume_obj.detach().cpu().numpy()
        self.volume_obj = Dataset3d.from_array(
            array=volume_obj,
            # name=self.volume_obj.name,
            # origin=self.volume_obj.origin,
            # sampling=self.volume_obj.sampling,
            # units=self.volume_obj.units,
            # signal_units=self.volume_obj.signal_units,
        )

    # --- Visualizations ---

    def plot_projections(
        self,
        cmap: str = "turbo",
        fft: bool = False,
        norm: str = "log_auto",
        figax: tuple[plt.Figure, plt.Axes] | None = None,
        **kwargs,
    ):
        """
        Plots the projections of the volume object.
        Note that the volume object is in the order of (z, y, x).
        Parameters
        ----------
        cmap : str
            The colormap to use for the projections.
        fft : bool
        """

        volume_obj_np = self.volume_obj.obj.detach().cpu().numpy()

        if figax is None:
            fig, ax = plt.subplots(ncols=3, figsize=(20, 8))
        else:
            fig, ax = figax

        show_2d(
            volume_obj_np.sum(axis=0),
            figax=(fig, ax[0]),
            cmap=cmap,
            title="Y-X Projection",
        )
        show_2d(
            volume_obj_np.sum(axis=1),
            figax=(fig, ax[1]),
            cmap=cmap,
            title="Z-X Projection",
        )
        show_2d(
            volume_obj_np.sum(axis=2),
            figax=(fig, ax[2]),
            cmap=cmap,
            title="Z-Y Projection",
        )

        if fft:
            fig, ax = plt.subplots(ncols=3, figsize=(25, 8))

            show_2d(
                np.abs(np.fft.fftshift(np.fft.fftn(volume_obj_np.sum(axis=0)))),
                figax=(fig, ax[0]),
                cmap=cmap,
                title="Y-X Projection FFT",
                norm=norm,
            )

            show_2d(
                np.abs(np.fft.fftshift(np.fft.fftn(volume_obj_np.sum(axis=1)))),
                figax=(fig, ax[1]),
                cmap=cmap,
                title="Z-X Projection FFT",
                norm=norm,
            )
            show_2d(
                np.abs(np.fft.fftshift(np.fft.fftn(volume_obj_np.sum(axis=2)))),
                figax=(fig, ax[2]),
                cmap=cmap,
                title="Z-Y Projection FFT",
                norm=norm,
            )

    def plot_slice(
        self,
        cmap="turbo",
        slice_index: int = 0,
        vmin: float = 0,
        figax: tuple[plt.Figure, plt.Axes] | None = None,
    ):
        if figax is None:
            fig, ax = plt.subplots(figsize=(15, 8), ncols=3)
        else:
            fig, ax = figax

        show_2d(
            self.volume_obj.obj[slice_index, :, :],
            figax=(fig, ax[0]),
            cmap=cmap,
            vmin=vmin,
        )
        show_2d(
            self.volume_obj.obj[:, slice_index, :],
            figax=(fig, ax[1]),
            cmap=cmap,
            vmin=vmin,
        )

        show_2d(
            self.volume_obj.obj[:, :, slice_index],
            figax=(fig, ax[2]),
            cmap=cmap,
            vmin=vmin,
        )

    def plot_loss(
        self,
        figsize: tuple = (8, 4),
        figax: tuple[plt.Figure, plt.Axes] | None = None,
    ):
        if figax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = figax

        ax.semilogy(
            self.loss,
            label="Loss",
        )
