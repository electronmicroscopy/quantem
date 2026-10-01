from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core.datastructures.dataset4d import Dataset4d


class Polar4dstem(Dataset4d):
    """4D-STEM dataset in polar coordinates (scan_row, scan_col, phi, r_pix)."""

    def __init__(
        self,
        array: NDArray | None = None,
        tensor: torch.Tensor | None = None,
        name: str = "",
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
        metadata: dict | None = None,
        origin_array: NDArray | None = None,
        _token: object | None = None,
    ):
        if metadata is None:
            metadata = {}
        mdata_keys_polar = [
            "polar_radial_min",
            "polar_radial_max",
            "polar_radial_step",
            "polar_num_annular_bins",
            "polar_two_fold_rotation_symmetry",
            "polar_ellipticity",
        ]
        for k in mdata_keys_polar:
            if k not in metadata:
                metadata[k] = None
        super().__init__(
            array=array,
            tensor=tensor,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            metadata=metadata,
            _token=_token,
        )
        self.origin_array = origin_array

    @classmethod
    def from_array(
        cls,
        array: NDArray,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
        metadata: dict | None = None,
    ) -> Self:
        if isinstance(array, torch.Tensor):
            raise TypeError(
                f"Polar4dstem.from_array requires a numpy array, got {type(array).__name__}. "
                "Use Polar4dstem.from_tensor for torch tensors."
            )
        array = np.asarray(array)
        if array.ndim != 4:
            raise ValueError(
                f"Found array with shape: {array.shape}. "
                "Polar4dstem.from_array expects a 4D array."
            )
        if origin is None:
            origin = np.zeros(4, dtype=float)
        if sampling is None:
            sampling = np.ones(4, dtype=float)
        if units is None:
            units = ["pixels", "pixels", "deg", "pixels"]
        if metadata is None:
            metadata = {}
        return cls(
            array=array,
            name=name if name is not None else "Polar 4D-STEM dataset",
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            metadata=metadata,
            _token=cls._token,
        )

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
        metadata: dict | None = None,
    ) -> Self:
        """Create a Polar4dstem from a torch tensor."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Polar4dstem.from_tensor requires torch.Tensor, got {type(tensor).__name__}."
            )
        if tensor.ndim != 4:
            raise ValueError(
                f"Found tensor with shape: {tuple(tensor.shape)}. "
                "Polar4dstem.from_tensor expects a 4D tensor."
            )
        if origin is None:
            origin = np.zeros(4, dtype=float)
        if sampling is None:
            sampling = np.ones(4, dtype=float)
        if units is None:
            units = ["pixels", "pixels", "deg", "pixels"]
        if metadata is None:
            metadata = {}
        return cls(
            tensor=tensor,
            name=name if name is not None else "Polar 4D-STEM dataset",
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            metadata=metadata,
            _token=cls._token,
        )

    @property
    def n_phi(self) -> int:
        return int(self.shape[2])

    @property
    def n_r(self) -> int:
        return int(self.shape[3])
