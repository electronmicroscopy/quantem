from typing import Any, List, Self

import numpy as np
from numpy.typing import NDArray

from quantem.core.datastructures.dataset3d import Dataset2d, Dataset3d
from quantem.core.utils.validators import ensure_valid_array

# from quantem.tomography.alignment import tilt_series_cross_cor_align, compute_com_tilt_series


class TiltSeries(Dataset3d):
    def __init__(
        self,
        array: NDArray | Any,  # Assumes a input tilt series [phis, x, y]
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        tilt_angles: list | NDArray,
        signal_units: str = "arb. units",
        _token: object | None = None,
    ):
        super().__init__(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            _token=_token,
        )

        self._tilt_angles = tilt_angles

    @classmethod
    def from_array(
        cls,
        array: NDArray | List[Dataset2d] | Any,
        tilt_angles: list | NDArray = None,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        if tilt_angles is not None:
            validated_tilt_angles = ensure_valid_array(tilt_angles, ndim=1)
        else:
            validated_tilt_angles = None

        array = np.transpose(array, axes=(2, 0, 1))

        return cls(
            array=array,
            tilt_angles=validated_tilt_angles
            if validated_tilt_angles is not None
            else ["duck" for _ in range(array.shape[0])],
            name=name if name is not None else "Tilt Series Dataset",
            origin=origin if origin is not None else np.zeros(3),
            sampling=sampling if sampling is not None else np.ones(3),
            units=units if units is not None else ["index", "pixels", "pixels"],
            signal_units=signal_units,
            _token=cls._token,
        )

    # --- Properties ---

    @property
    def tilt_angles(self) -> NDArray:
        """Get the tilt angles of the dataset."""
        return self._tilt_angles

    @property
    def tilt_angles_rad(self) -> NDArray:
        """Get the tilt angles of the dataset in radians."""
        return np.deg2rad(self._tilt_angles)

    @tilt_angles.setter
    def tilt_angles(self, angles: NDArray | list) -> None:
        """Set the tilt angles of the dataset."""
        if len(angles) != self.shape[0]:
            raise ValueError("Tilt angles must match the number of projections.")

        # Convert to numpy array if not already
        if isinstance(self._tilt_angles, NDArray):
            self._tilt_angles = np.array(angles)
        else:
            self._tilt_angles = angles
