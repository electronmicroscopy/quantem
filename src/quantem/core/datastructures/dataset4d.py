from typing import Any, Self, Union

import numpy as np
from numpy.typing import NDArray

from quantem.core.datastructures.dataset import Dataset
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization.visualization_utils import ScalebarConfig


class Dataset4d(Dataset):
    """4D dataset class that inherits from Dataset.

    This class represents 4D stacks of 2D datasets, such as 4D-STEM experiments.

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
        _token: object | None = None,
    ):
        """Initialize a 4D dataset.

        Parameters
        ----------
        array : NDArray | Any
            The underlying 3D array data
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
        array = ensure_valid_array(array, ndim=4)
        return cls(
            array=array,
            name=name if name is not None else "4D dataset",
            origin=origin if origin is not None else np.zeros(4),
            sampling=sampling if sampling is not None else np.ones(4),
            units=units if units is not None else ["index", "index", "pixels", "pixels"],
            signal_units=signal_units,
            _token=cls._token,
        )

    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, int, int, int],
        name: str = "constant 4D dataset",
        fill_value: float = 0.0,
        origin: Union[NDArray, tuple, list, float, int] | None = None,
        sampling: Union[NDArray, tuple, list, float, int] | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        """Create a new Dataset4d filled with a constant value."""
        array = np.full(shape, fill_value, dtype=np.float32)
        return cls.from_array(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
        )

    def __getitem__(self, index) -> Dataset2d | Dataset3d:
        """
        Simple indexing function to return Dataset2d or Dataset3d view.

        Parameters
        ----------
        index : tuple
            Index to access a subset of the dataset

        Returns
        -------
        dataset
            A new Dataset2d or Dataset3D instance containing the indexed data
        """
        array_view = self.array[index]
        ndim = array_view.ndim
        calibrated_origin = self.origin.ndim == self.ndim

        if ndim == 2:
            cls = Dataset2d
        elif ndim == 3:
            cls = Dataset3d
        else:
            raise ValueError("only 2D and 3D slices are supported.")

        return cls.from_array(
            array=array_view,
            name=self.name + str(index),
            origin=self.origin[index] if calibrated_origin else self.origin[-ndim:],
            sampling=self.sampling[-ndim:],
            units=self.units[-ndim:],
            signal_units=self.signal_units,
        )

    def show(
        self,
        index: tuple[int, int] = (0, 0),
        scalebar: ScalebarConfig | bool = True,
        title: str | None = None,
        **kwargs,
    ):
        """
        Display a 2D slice of the 4D dataset.

        Parameters
        ----------
        index : tuple[int,int]
            2D index of the 2D slice to display (along axes (0,1)).
        scalebar: ScalebarConfig or bool
            If True, displays scalebar
        title: str
            Title of Dataset
        **kwargs: dict
            Keyword arguments for show_2d
        """

        return self[index].show(scalebar=scalebar, title=title, **kwargs)
