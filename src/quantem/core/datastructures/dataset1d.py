from typing import Self

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from quantem.core.datastructures.dataset import Dataset
from quantem.core.utils.validators import ensure_valid_array


@Dataset.register_dimension(1)
class Dataset1d(Dataset):
    """1D dataset class that inherits from Dataset.

    This class represents a 1D dataset, such as spectra from XEDS or EELS.

    Attributes
    ----------
    None beyond base Dataset.
    """

    def __init__(
        self,
        array: NDArray,
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
        metadata: dict = {},
        _token: object | None = None,
    ):
        """Initialize a 1D dataset.

        Parameters
        ----------
        array : NDArray
            The underlying 1D array data
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
            metadata=metadata,
            _token=_token,
        )

    @classmethod
    def from_array(
        cls,
        array: NDArray,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        """Create a Dataset1d from a 1D array.

        Parameters
        ----------
        array : NDArray
            1D array with shape (length)
        name : str | None
            Dataset name. Default: "1D dataset"
        origin : NDArray | tuple | list | float | int | None
            Origin for each dimension. Default: [0]
        sampling : NDArray | tuple | list | float | int | None
            Sampling for each dimension. Default: [1]
        units : list[str] | tuple | list | None
            Units for each dimension. Default: ["pixels"]
        signal_units : str
            Units for array values. Default: "arb. units"

        Returns
        -------
        Dataset1d

        Examples
        --------
        >>> import numpy as np
        >>> from quantem.core.datastructures import Dataset1d
        >>> arr = np.random.rand(10)
        >>> data = Dataset1d.from_array(arr)
        >>> data.shape
        (10,)

        With calibration:

        >>> data = Dataset1d.from_array(
        ...     arr,
        ...     sampling=[0.15],
        ...     units=["eV"],
        ... )

        Visualize:

        >>> data.show()              # all frames in grid
        >>> data.show(index=0)       # single frame
        >>> data.show(ncols=2)       # 2 columns
        """
        array = ensure_valid_array(array, ndim=1)
        return cls(
            array=array,
            name=name if name is not None else "1D dataset",
            origin=origin if origin is not None else np.zeros(1),
            sampling=sampling if sampling is not None else np.ones(1),
            units=units if units is not None else ["pixels"],
            signal_units=signal_units,
            _token=cls._token,
        )

    @classmethod
    def from_shape(
        cls,
        shape: tuple[int],
        name: str = "constant 1D dataset",
        fill_value: float = 0.0,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        """Create a Dataset1d filled with a constant value.

        Parameters
        ----------
        shape : tuple[int]
            Shape (length)
        name : str
            Dataset name. Default: "constant 1D dataset"
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
        Dataset1d

        Examples
        --------
        >>> data = Dataset1d.from_shape((10000))
        >>> data.shape
        (10000,)
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

    def show(
        self,
        title: str | None = None,
        returnfig: bool = False,
    ):
        """
        Plots 1D dataset

        Parameters
        ----------
        title: str
            Title of Dataset
        returnfig: bool
            Option to include figure as return value for method
        """

        if title is None:
            title = self.name

        fig, (ax) = plt.subplots(1, 1, figsize=(4, 4))

        ax.plot(
            float(self.origin[0]) + float(self.sampling[0]) * np.arange(self.shape[0]),
            self.array,
            linewidth=1.5,
        )
        ax.set_xlabel(self.units[0])
        ax.set_ylabel(self.signal_units)
        ax.set_title(title)

        fig.tight_layout()
        plt.show()

        return (fig, ax) if returnfig else None
