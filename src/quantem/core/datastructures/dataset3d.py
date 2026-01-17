from typing import Any, Self, Union

import numpy as np
from numpy.typing import NDArray

from quantem.core.datastructures.dataset import Dataset
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization.visualization_utils import ScalebarConfig


@Dataset.register_dimension(3)
class Dataset3d(Dataset):
    """3D dataset class that inherits from Dataset.

    This class represents 3D stacks of 2D datasets, such as image sequences.

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
        """Initialize a 3D dataset.

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
        """Create a Dataset3d from a 3D array.

        Parameters
        ----------
        array : NDArray | Any
            3D array with shape (n_frames, height, width)
        name : str | None
            Dataset name. Default: "3D dataset"
        origin : NDArray | tuple | list | float | int | None
            Origin for each dimension. Default: [0, 0, 0]
        sampling : NDArray | tuple | list | float | int | None
            Sampling for each dimension. Default: [1, 1, 1]
        units : list[str] | tuple | list | None
            Units for each dimension. Default: ["index", "pixels", "pixels"]
        signal_units : str
            Units for array values. Default: "arb. units"

        Returns
        -------
        Dataset3d

        Examples
        --------
        >>> import numpy as np
        >>> from quantem.core.datastructures import Dataset3d
        >>> arr = np.random.rand(10, 64, 64)
        >>> data = Dataset3d.from_array(arr)
        >>> data.shape
        (10, 64, 64)

        With calibration:

        >>> data = Dataset3d.from_array(
        ...     arr,
        ...     sampling=[1, 0.1, 0.1],
        ...     units=["frame", "nm", "nm"],
        ... )

        Visualize:

        >>> data.show()              # all frames in grid
        >>> data.show(index=0)       # single frame
        >>> data.show(ncols=2)       # 2 columns
        """
        array = ensure_valid_array(array, ndim=3)
        return cls(
            array=array,
            name=name if name is not None else "3D dataset",
            origin=origin if origin is not None else np.zeros(3),
            sampling=sampling if sampling is not None else np.ones(3),
            units=units if units is not None else ["index", "pixels", "pixels"],
            signal_units=signal_units,
            _token=cls._token,
        )

    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, int, int],
        name: str = "constant 3D dataset",
        fill_value: float = 0.0,
        origin: Union[NDArray, tuple, list, float, int] | None = None,
        sampling: Union[NDArray, tuple, list, float, int] | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        """Create a Dataset3d filled with a constant value.

        Parameters
        ----------
        shape : tuple[int, int, int]
            Shape (n_frames, height, width)
        name : str
            Dataset name. Default: "constant 3D dataset"
        fill_value : float
            Value to fill array with. Default: 0.0
        origin : NDArray | tuple | list | float | int | None
            Origin for each dimension
        sampling : NDArray | tuple | list | float | int | None
            Sampling for each dimension
        units : list[str] | tuple | list | None
            Units for each dimension
        signal_units : str
            Units for array values

        Returns
        -------
        Dataset3d

        Examples
        --------
        >>> data = Dataset3d.from_shape((10, 64, 64))
        >>> data.shape
        (10, 64, 64)
        >>> data.array.max()
        0.0
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

    def to_dataset2d(self):
        """ """
        return [self[i] for i in range(self.shape[0])]

    def show(
        self,
        index: int | None = None,
        scalebar: ScalebarConfig | bool = True,
        title: str | None = None,
        suptitle: str | None = None,
        ncols: int = 4,
        returnfig: bool = False,
        **kwargs,
    ):
        """
        Display 2D slices of the 3D dataset.

        Parameters
        ----------
        index : int | None
            Index of the 2D slice to display. If None, shows all slices in a grid.
        scalebar : ScalebarConfig or bool
            If True, displays scalebar.
        title : str | None
            Title for the plot. If None, uses "Frame 0", "Frame 1", etc.
        suptitle : str | None
            Figure super title displayed above all subplots.
        ncols : int
            Maximum columns when showing all slices. Default: 4.
        returnfig : bool
            If True, returns (fig, axes). Default: False.
        **kwargs : dict
            Keyword arguments for show_2d (cmap, cbar, norm, etc.).

        Examples
        --------
        >>> data.show()           # show all frames in grid
        >>> data.show(index=0)    # show single frame
        >>> data.show(ncols=3)    # 3 columns
        >>> data.show(suptitle="Diffraction patterns")  # with super title
        >>> fig, axes = data.show(returnfig=True)  # get figure for customization
        """
        from quantem.core.visualization import show_2d

        if index is not None:
            # Handle negative index
            actual_index = index if index >= 0 else self.shape[0] + index
            default_title = title if title is not None else f"Frame {actual_index}"
            result = self[index].show(scalebar=scalebar, title=default_title, **kwargs)
            return result if returnfig else None

        # Show all frames in a grid
        n = self.shape[0]
        nrows = (n + ncols - 1) // ncols
        arrays = []
        titles = []
        for row in range(nrows):
            row_arrays = []
            row_titles = []
            for col in range(ncols):
                i = row * ncols + col
                if i < n:
                    row_arrays.append(self.array[i])
                    row_titles.append(f"Frame {i}" if title is None else f"{title} {i}")
                else:
                    row_arrays.append(np.zeros_like(self.array[0]))
                    row_titles.append("")
            arrays.append(row_arrays)
            titles.append(row_titles)

        fig, axes = show_2d(arrays, scalebar=scalebar, title=titles, **kwargs)
        if suptitle is not None:
            fig.suptitle(suptitle, fontsize=14)
            fig.subplots_adjust(top=0.92)
        return (fig, axes) if returnfig else None
