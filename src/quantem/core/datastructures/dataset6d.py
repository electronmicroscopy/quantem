from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from quantem.core.datastructures.dataset import Dataset
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization.visualization_utils import ScalebarConfig


@Dataset.register_dimension(6)
class Dataset6d(Dataset):
    """6D dataset class that inherits from Dataset."""

    def __init__(
        self,
        array: NDArray | Any,
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        signal_units: str = "arb. units",
        metadata: dict | None = None,
        _token: object | None = None,
    ):
        """Initialize a 6D dataset."""
        super().__init__(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            metadata={} if metadata is None else metadata,
            _token=_token,
        )

    def __len__(self):
        return int(self.array.shape)

    @classmethod
    def from_array(
        cls,
        array: NDArray | Any,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
        metadata: dict | None = None,
    ) -> Self:
        array = ensure_valid_array(array, ndim=6)

        return cls(
            array=array,
            name=name if name is not None else "6D dataset",
            origin=origin if origin is not None else np.zeros(6),
            sampling=sampling if sampling is not None else np.ones(6),
            units=(
                units
                if units is not None
                else ["index", "index", "index", "pixels", "pixels", "pixels"]
            ),
            signal_units=signal_units,
            metadata=metadata,
            _token=cls._token,
        )

    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, int, int, int, int, int],
        name: str = "constant 6D dataset",
        fill_value: float = 0.0,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
        metadata: dict | None = None,
    ) -> Self:
        """Create a new Dataset6d filled with a constant value."""
        array = np.full(shape, fill_value, dtype=np.float32)
        return cls.from_array(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            metadata=metadata,
        )
    
