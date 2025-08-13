from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from tqdm.auto import tqdm

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.visualization.visualization import show_2d
from quantem.imaging.drift import cross_correlation_shift
from quantem.tomography.tilt_series_dataset import TiltSeries


class TomographyBase(AutoSerialize):
    _token = object()

    def __init__(
        self,
        tilt_series: Dataset3d,
        recon_volume: Dataset3d | None,
        device: str = "cuda",
        # ABF/HAADF property
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

        if _token is not self._token:
            raise RuntimeError(
                "This class is not meant to be instantiated directly. Use the from_data method."
            )

        self._device = device
        self._tilt_series = tilt_series
        self._recon_volume = recon_volume
        self._loss = []
        self._mode = []

    @classmethod
    def from_tilt_series(
        cls,
        device: str,
        tilt_series: NDArray | Dataset3d | Any,
        tilt_angles: list | NDArray = None,
        recon_volume: NDArray | Dataset3d | None = None,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ):
        device = device.lower()

        tilt_series = TiltSeries.from_array(
            array=tilt_series,
            tilt_angles=tilt_angles,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
        )

        if recon_volume is not None:
            if not isinstance(recon_volume, Dataset3d):
                recon_volume = Dataset3d.from_array(
                    array=recon_volume,
                    name=name,
                    origin=origin,
                    sampling=sampling,
                    units=units,
                    signal_units=signal_units,
                )
        else:
            empty_recon_vol = np.zeros(
                (
                    tilt_series.array.shape[2],
                    tilt_series.array.shape[2],
                    tilt_series.array.shape[2],
                ),
                dtype=tilt_series.array.dtype,
            )

            recon_volume = Dataset3d.from_array(
                array=empty_recon_vol,
                name=name,
                origin=origin,
                sampling=sampling,
                units=units,
                signal_units=signal_units,
            )

        return cls(
            tilt_series=tilt_series,
            recon_volume=recon_volume,
            device=device,
            _token=cls._token,
        )

    # --- Properties ---
    @property
    def tilt_series(self) -> TiltSeries:
        """Tilt series dataset."""

        return self._tilt_series

    @tilt_series.setter
    def tilt_series(
        self,
        tilt_series: Dataset3d | NDArray | TiltSeries,
        tilt_angles: list | NDArray = None,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ):
        """Set the tilt series dataset."""

        if not isinstance(tilt_series, TiltSeries):
            tilt_series = TiltSeries.from_array(
                array=tilt_series,
                tilt_angles=tilt_angles,
                name=name,
                origin=origin,
                sampling=sampling,
                units=units,
                signal_units=signal_units,
            )

        self._tilt_series = tilt_series

    @property
    def recon_volume(self) -> Dataset3d | None:
        """Reconstruction volume dataset."""

        return self._recon_volume

    @recon_volume.setter
    def recon_volume(self, recon_volume: Dataset3d | NDArray):
        """Set the reconstruction volume dataset."""

        if not isinstance(recon_volume, Dataset3d):
            recon_volume = Dataset3d.from_array(
                array=recon_volume,
                name=self._tilt_series.name,
                origin=self._tilt_series.origin,
                sampling=self._tilt_series.sampling,
                units=self._tilt_series.units,
                signal_units=self._tilt_series.signal_units,
            )

        self._recon_volume = recon_volume

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
        aligned_tilt_series = np.zeros_like(self.tilt_series.array)
        aligned_tilt_series[:, 0, :] = self.tilt_series.array[:, 0, :]
        shifts = []
        num_imgs = self.tilt_series.array.shape[1]

        pbar = tqdm(range(num_imgs - 1), desc="Cross-correlation alignment")

        for i in pbar:
            shift, aligned_img = cross_correlation_shift(
                self.tilt_series.array[:, i, :],
                self.tilt_series.array[:, i + 1, :],
                upsample_factor=upsample_factor,
                return_shifted_image=True,
            )

            aligned_tilt_series[:, i + 1, :] = aligned_img
            shifts.append(shift)

        if overwrite:
            # TODO: Check this overwrite idea, maybe also need to save the relative shifts?
            self.tilt_series.array = aligned_tilt_series

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
        H, W, D = self.recon_volume.array.shape
        device = self.device
        dtype = torch.float32
        recon_volume = torch.tensor(
            self.recon_volume.array,
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

        recon_volume = recon_volume * total_mask
        recon_volume = recon_volume.detach().cpu().numpy()
        self.recon_volume = Dataset3d.from_array(
            array=recon_volume,
            name=self.recon_volume.name,
            origin=self.recon_volume.origin,
            sampling=self.recon_volume.sampling,
            units=self.recon_volume.units,
            signal_units=self.recon_volume.signal_units,
        )

    # --- Visualizations ---

    def plot_projections(
        self,
        cmap: str = "turbo",
        loss: bool = False,
        fft: bool = False,
    ):
        if loss:
            fig, ax = plt.subplots(ncols=4, figsize=(25, 8))
            ax[3].semilogy(
                self.loss,
            )
            ax[3].set_title("Loss")
        else:
            fig, ax = plt.subplots(ncols=3, figsize=(20, 8))

        show_2d(
            self.recon_volume.array.sum(axis=0),
            figax=(fig, ax[0]),
            cmap=cmap,
            title="Z-X Projection",
        )
        show_2d(
            self.recon_volume.array.sum(axis=1),
            figax=(fig, ax[1]),
            cmap=cmap,
            title="Y-X Projection",
        )
        show_2d(
            self.recon_volume.array.sum(axis=2),
            figax=(fig, ax[2]),
            cmap=cmap,
            title="Y-Z Projection",
        )

        if fft:
            fig, ax = plt.subplots(ncols=3, figsize=(25, 8))

            show_2d(
                np.abs(np.log(np.fft.fftshift(np.fft.fftn(self.recon_volume.array.sum(axis=0))))),
                figax=(fig, ax[0]),
                cmap=cmap,
                title="Z-X Projection FFT",
            )

            show_2d(
                np.abs(np.log(np.fft.fftshift(np.fft.fftn(self.recon_volume.array.sum(axis=1))))),
                figax=(fig, ax[1]),
                cmap=cmap,
                title="Y-X Projection FFT",
            )
            show_2d(
                np.abs(np.log(np.fft.fftshift(np.fft.fftn(self.recon_volume.array.sum(axis=2))))),
                figax=(fig, ax[2]),
                cmap=cmap,
                title="Y-Z Projection FFT",
            )

    def plot_slice(
        self,
        cmap="turbo",
        slice_index: int = 0,
        vmin: float = 0,
    ):
        fig, ax = plt.subplots(figsize=(15, 8), ncols=3)

        # show_2d(
        #     self.recon_volume.array[slice_index, :, :],
        #     figax = (fig, ax[0]),
        #     cmap = cmap,
        #     title = f"Z-X Slice {slice_index}",
        #     norm = norm,
        #     cbar = True,
        # )
        # show_2d(
        #     self.recon_volume.array[:, slice_index, :],
        #     figax = (fig, ax[1]),
        #     cmap = cmap,
        #     title = f"Y-X Sliec {slice_index}",
        #     norm = norm,
        # )
        # show_2d(
        #     self.recon_volume.array[:, :, slice_index],
        #     figax = (fig, ax[2]),
        #     cmap = cmap,
        #     title = f"Y-Z Slice {slice_index}",
        #     norm = norm,
        # )

        ax[0].matshow(
            self.recon_volume.array[slice_index, :, :],
            cmap=cmap,
            vmin=vmin,
        )

        ax[1].matshow(
            self.recon_volume.array[:, slice_index, :],
            cmap=cmap,
            vmin=vmin,
        )

        ax[2].matshow(
            self.recon_volume.array[:, :, slice_index],
            cmap=cmap,
            vmin=vmin,
        )

    def plot_loss(
        self,
        figsize: tuple = (8, 8),
    ):
        fig, ax = plt.subplots(figsize=figsize)

        ax.semilogy(
            self.loss,
            label="Loss",
        )
