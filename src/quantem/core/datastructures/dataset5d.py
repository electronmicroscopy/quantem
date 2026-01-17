from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from quantem.core.datastructures.dataset import Dataset
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization.visualization_utils import ScalebarConfig


@Dataset.register_dimension(5)
class Dataset5d(Dataset):
    """5D dataset class that inherits from Dataset.

    This class represents 5D stacks of data, such as time-series or tilt-series
    of 4D-STEM experiments.

    The data consists of a 5D array with dimensions (stack, scan_row, scan_col, k_row, k_col).
    The first dimension represents the stack axis (time, tilt, defocus, etc.),
    dimensions 1-2 represent real space scanning positions, and dimensions 3-4
    represent reciprocal space diffraction patterns.

    Attributes
    ----------
    None beyond base Dataset.
    """

    def __init__(
        self,
        array: NDArray | Any,
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        signal_units: str = "arb. units",
        metadata: dict = {},
        _token: object | None = None,
    ):
        """Initialize a 5D dataset.

        Parameters
        ----------
        array : NDArray | Any
            The underlying 5D array data
        name : str
            A descriptive name for the dataset
        origin : NDArray | tuple | list | float | int
            The origin coordinates for each dimension
        sampling : NDArray | tuple | list | float | int
            The sampling rate/spacing for each dimension
        units : list[str] | tuple | list
            Units for each dimension
        signal_units : str, optional
            Units for the array values, by default "arb. units"
        metadata : dict, optional
            Additional metadata, by default {}
        _token : object | None, optional
            Token to prevent direct instantiation, by default None
        """
        super().__init__(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            metadata=metadata,
            _token=_token,
        )

    @classmethod
    def from_array(
        cls,
        array: NDArray | Any,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        """Create a new Dataset5d from an array.

        Parameters
        ----------
        array : NDArray | Any
            The underlying 5D array data
        name : str | None, optional
            A descriptive name for the dataset. If None, defaults to "5D dataset"
        origin : NDArray | tuple | list | float | int | None, optional
            The origin coordinates for each dimension. If None, defaults to zeros
        sampling : NDArray | tuple | list | float | int | None, optional
            The sampling rate/spacing for each dimension. If None, defaults to ones
        units : list[str] | tuple | list | None, optional
            Units for each dimension. If None, defaults to ["index", "pixels", "pixels", "pixels", "pixels"]
        signal_units : str, optional
            Units for the array values, by default "arb. units"

        Returns
        -------
        Dataset5d
            A new Dataset5d instance
        """
        array = ensure_valid_array(array, ndim=5)
        return cls(
            array=array,
            name=name if name is not None else "5D dataset",
            origin=origin if origin is not None else np.zeros(5),
            sampling=sampling if sampling is not None else np.ones(5),
            units=units if units is not None else ["index", "pixels", "pixels", "pixels", "pixels"],
            signal_units=signal_units,
            _token=cls._token,
        )

    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, int, int, int, int],
        name: str = "constant 5D dataset",
        fill_value: float = 0.0,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        """Create a new Dataset5d filled with a constant value.

        Parameters
        ----------
        shape : tuple[int, int, int, int, int]
            Shape of the 5D array
        name : str, optional
            Name for the dataset, by default "constant 5D dataset"
        fill_value : float, optional
            Value to fill the array with, by default 0.0
        origin : NDArray | tuple | list | float | int | None, optional
            Origin coordinates for each dimension
        sampling : NDArray | tuple | list | float | int | None, optional
            Sampling rate for each dimension
        units : list[str] | tuple | list | None, optional
            Units for each dimension
        signal_units : str, optional
            Units for the array values, by default "arb. units"

        Returns
        -------
        Dataset5d
            A new Dataset5d instance filled with the specified value
        """
        array = np.full(shape, fill_value, dtype=np.float32)
        return cls.from_array(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
        )

    def show(
        self,
        index: tuple[int, int, int] = (0, 0, 0),
        scalebar: ScalebarConfig | bool = True,
        title: str | None = None,
        **kwargs,
    ):
        """
        Display a 2D slice of the 5D dataset.

        Parameters
        ----------
        index : tuple[int, int, int]
            3D index of the 2D slice to display (along axes (0, 1, 2)).
        scalebar : ScalebarConfig or bool
            If True, displays scalebar
        title : str
            Title of Dataset
        **kwargs : dict
            Keyword arguments for show_2d
        """
        return self[index].show(scalebar=scalebar, title=title, **kwargs)
