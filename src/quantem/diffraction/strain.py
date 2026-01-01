from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.imaging_utils import rotate_image
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import ScalebarConfig, show_2d


class StrainMap(AutoSerialize):
    """
    Nanobeam strain mapping
    """

    _token = object()

    def __init__(
        self,
        dataset: Dataset4dstem,
        input_data: Any | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use StrainMap.from_data() to instantiate this class.")
        super().__init__()
        self.dataset = dataset
        self.input_data = input_data
        self.strain = None
        self.metadata: dict[str, Any] = {}
        self.transform: Dataset2d | None = None
        self.transform_rotated: Dataset2d | None = None

    @classmethod
    def from_data(cls, data: NDArray | Dataset4dstem, *, name: str = "strain_map") -> StrainMap:
        if isinstance(data, Dataset4dstem):
            return cls(dataset=data, input_data=data, _token=cls._token)

        arr = ensure_valid_array(data)
        if arr.ndim != 4:
            raise ValueError(
                "StrainMap.from_data expects a 4D array with shape (scan_r, scan_c, dp_r, dp_c)."
            )

        ds4 = Dataset4dstem.from_array(arr, name=name)
        return cls(dataset=ds4, input_data=data, _token=cls._token)

    def preprocess(
        self,
        mode: str = "linear",
        plot_transform: bool = True,
        cropping_factor: float = 0.5,
        **plot_kwargs: Any,
    ) -> StrainMap:
        if self.dataset.units[2] == "A":
            qrow_sampling_ang = float(self.dataset.sampling[2])
        elif self.dataset.units[2] == "mrad":
            wavelength = float(electron_wavelength_angstrom(float(self.dataset.metadata["energy"])))
            qrow_sampling_ang = float(self.dataset.sampling[2]) / 1000.0 / wavelength
        else:
            raise ValueError(f"unrecognized diffraction-space unit for axis 2: {self.dataset.units[2]}")

        if self.dataset.units[3] == "A":
            qcol_sampling_ang = float(self.dataset.sampling[3])
        elif self.dataset.units[3] == "mrad":
            wavelength = float(electron_wavelength_angstrom(float(self.dataset.metadata["energy"])))
            qcol_sampling_ang = float(self.dataset.sampling[3]) / 1000.0 / wavelength
        else:
            raise ValueError(f"unrecognized diffraction-space unit for axis 3: {self.dataset.units[3]}")

        self.metadata["sampling_real"] = np.array(
            (
                1.0 / (qrow_sampling_ang * float(self.dataset.shape[2])),
                1.0 / (qcol_sampling_ang * float(self.dataset.shape[3])),
            ),
            dtype=float,
        )

        if mode == "linear":
            im = np.mean(np.abs(np.fft.fft2(self.dataset.array)), axis=(0, 1))
        elif mode == "log":
            im = np.mean(np.abs(np.fft.fft2(np.log(self.dataset.array + 1.0))), axis=(0, 1))
        else:
            raise ValueError("mode must be 'linear' or 'log'")

        self.transform = Dataset2d.from_array(
            np.fft.fftshift(im),
            origin=(im.shape[0] // 2, im.shape[1] // 2),
            sampling=(1.0, 1.0),
            units=("A", "A"),
            signal_units="intensity",
        )

        self.transform_rotated = Dataset2d.from_array(
            rotate_image(
                self.transform.array,
                float(self.dataset.metadata["q_to_r_rotation_cw_deg"]),
                clockwise=True,
            ),
            origin=(im.shape[0] // 2, im.shape[1] // 2),
            sampling=(1.0, 1.0),
            units=("A", "A"),
            signal_units="intensity",
        )

        if plot_transform:
            self.plot_transform(cropping_factor=cropping_factor, **plot_kwargs)

        return self









    def plot_transform(
        self, 
        cropping_factor: float = 0.25, 
        **plot_kwargs: Any
    ):
        if self.transform is None or self.transform_rotated is None:
            raise ValueError("Run preprocess() first to compute transform images.")

        defaults = dict(
            vmax=1.0,
            title=("Original Transform", "Rotated Transform"),
            scalebar=ScalebarConfig(
                sampling=self.metadata["sampling_real"],
                units=r"$\mathrm{\AA}$",
                length=2,
            ),
        )
        defaults.update(plot_kwargs)

        fig, ax = show_2d([self.transform, self.transform_rotated], **defaults)

        axes = np.atleast_1d(ax)
        for a in axes:
            _apply_center_crop_limits(a, self.transform.shape, cropping_factor)

        return fig, ax







def _apply_center_crop_limits(ax, shape: tuple[int, int], cropping_factor: float) -> None:
    cf = float(cropping_factor)
    if cf >= 1.0:
        return
    if not (0.0 < cf <= 1.0):
        raise ValueError("cropping_factor must be in (0, 1].")

    H, W = int(shape[0]), int(shape[1])
    r0 = (H - 1) / 2.0
    c0 = (W - 1) / 2.0
    half_h = 0.5 * cf * H
    half_w = 0.5 * cf * W

    ax.set_xlim(c0 - half_w, c0 + half_w)

    y0, y1 = ax.get_ylim()
    if y0 > y1:
        ax.set_ylim(r0 + half_h, r0 - half_h)
    else:
        ax.set_ylim(r0 - half_h, r0 + half_h)


