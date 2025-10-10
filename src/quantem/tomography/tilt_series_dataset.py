from typing import Any, List, Self

import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core.datastructures.dataset3d import Dataset2d, Dataset3d
from quantem.core.utils.validators import ensure_valid_array

# from quantem.tomography.alignment import tilt_series_cross_cor_align, compute_com_tilt_series


# DEPRECATED: Use TomographyDataset instead.
class TiltSeries(Dataset3d):
    def __init__(
        self,
        array: NDArray | Any,  # Assumes a input tilt series [phis, x, y]
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        tilt_angles: list | NDArray,
        z1_angles: list | NDArray,
        z3_angles: list | NDArray,
        shifts: list[tuple[float, float]] | NDArray,
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
        self._z1_angles = z1_angles
        self._z3_angles = z3_angles
        self._shifts = shifts

    @classmethod
    def from_array(
        cls,
        array: NDArray | List[Dataset2d] | Any,
        tilt_angles: list | NDArray = None,
        z1_angles: list | NDArray = None,
        z3_angles: list | NDArray = None,
        shifts: list[tuple[float, float]] | NDArray = None,
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

        # array = np.transpose(array, axes=(2, 0, 1))

        if z1_angles is not None:
            validated_z1_angles = ensure_valid_array(z1_angles, ndim=1)
        else:
            validated_z1_angles = torch.zeros(len(validated_tilt_angles))

        if z3_angles is not None:
            validated_z3_angles = ensure_valid_array(z3_angles, ndim=1)
        else:
            validated_z3_angles = torch.zeros(len(validated_tilt_angles))

        if shifts is not None:
            validated_shifts = ensure_valid_array(shifts, ndim=2)
        else:
            validated_shifts = torch.zeros((len(validated_tilt_angles), 2))

        array = torch.from_numpy(array)

        return cls(
            array=array,
            tilt_angles=validated_tilt_angles
            if validated_tilt_angles is not None
            else ["duck" for _ in range(array.shape[0])],
            z1_angles=validated_z1_angles,
            z3_angles=validated_z3_angles,
            shifts=validated_shifts,
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

    @property
    def z1_angles(self) -> NDArray:
        """Get the z1 angles of the dataset."""
        return self._z1_angles

    @z1_angles.setter
    def z1_angles(self, angles: NDArray | list) -> None:
        """Set the z1 angles of the dataset."""
        if len(angles) != self.shape[0]:
            raise ValueError("Z1 angles must match the number of projections.")
        self._z1_angles = angles

    @property
    def z3_angles(self) -> NDArray:
        """Get the z3 angles of the dataset."""
        return self._z3_angles

    @z3_angles.setter
    def z3_angles(self, angles: NDArray | list) -> None:
        """Set the z3 angles of the dataset."""
        if len(angles) != self.shape[0]:
            raise ValueError("Z3 angles must match the number of projections.")
        self._z3_angles = angles

    @property
    def shifts(self) -> NDArray:
        """Get the shifts of the dataset."""
        return self._shifts

    @shifts.setter
    def shifts(self, shifts: list[tuple[float, float]] | NDArray) -> None:
        """Set the shifts of the dataset."""
        self._shifts = shifts
