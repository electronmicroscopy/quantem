from typing import Self

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from quantem.core.datastructures.dataset import Dataset
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import show_2d
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
        array: NDArray,
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
        array : NDArray
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
        array: NDArray,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        """Create a Dataset3d from a 3D array.

        Parameters
        ----------
        array : NDArray
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
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
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
        start: int = 0,
        end: int | None = None,
        step: int = 1,
        max: int | None = 20,
        ncols: int = 4,
        scalebar: ScalebarConfig | bool = False,
        title_prefix: str | None = None,
        suptitle: str | None = None,
        returnfig: bool = False,
        **kwargs,
    ) -> tuple[Figure, Axes] | None:
        """
        Display 2D slices of the 3D dataset.

        Parameters
        ----------
        start : int, default 0
            First frame index. Supports negative indexing.
        end : int or None, optional
            End frame index (exclusive). If None, determined by max.
        step : int, default 1
            Step between frames. Negative step shows frames in reverse order.
        max : int or None, default 20
            Maximum number of frames to show. Prevents memory issues.
            Set to None to show all frames.
        ncols : int, default 4
            Number of columns in grid.
        scalebar : ScalebarConfig or bool, default False
            If True, displays scalebar on each frame.
        title_prefix : str or None, optional
            Prefix for frame titles. If None, uses "Frame 0", "Frame 1", etc.
            If set to "DP", generates "DP 0", "DP 1", etc.
        suptitle : str or None, optional
            Figure super title displayed above all subplots.
        returnfig : bool, default False
            If True, returns (fig, axes).
        **kwargs : dict
            Keyword arguments for show_2d (cmap, cbar, vmin, vmax, norm, etc.).

        Returns
        -------
        tuple[Figure, Axes] or None
            If returnfig=True, returns (fig, axes) tuple for further customization.
            If returnfig=False, returns None.

        Raises
        ------
        ValueError
            If step is zero, ncols < 1, max < 1, start is out of bounds,
            or the specified range has no frames to display.

        Examples
        --------
        Basic usage:

        >>> data.show()                    # first 20 frames
        >>> data.show(max=None)            # all frames (use with caution)

        Single frame:

        >>> data.show(start=5, max=1)      # frame 5
        >>> data.show(start=-1, max=1)     # last frame

        Frame range:

        >>> data.show(start=10, end=50)    # frames 10-49
        >>> data.show(step=5)              # frames 0,5,10,15,...
        >>> data.show(start=0, end=100, step=10)  # every 10th frame
        >>> data.show(start=9, step=-1)    # reverse order: 9,8,7,...,0

        Grid layout:

        >>> data.show(ncols=2)             # 2 columns
        >>> data.show(ncols=5, max=10)     # 5x2 grid

        Titles:

        >>> data.show(title_prefix="DP")           # "DP 0", "DP 1", ...
        >>> data.show(suptitle="Time Series")      # figure title
        >>> data.show(title_prefix="T", suptitle="Tomography Slices")

        Customization:

        >>> data.show(cmap="viridis")              # colormap
        >>> data.show(scalebar=True)               # show scalebar
        >>> fig, axes = data.show(returnfig=True)  # get figure for further customization
        """
        total_frames = self.shape[0]

        # Validate inputs
        if step == 0:
            raise ValueError("Step cannot be zero.")
        if ncols < 1:
            raise ValueError(f"ncols must be >= 1, got {ncols}.")
        if max is not None and max < 1:
            raise ValueError(f"max must be >= 1 or None, got {max}.")
        if start < 0:
            start = total_frames + start
        if start < 0 or start >= total_frames:
            raise ValueError(
                f"Start {start} is out of bounds for dataset with {total_frames} frames. "
                f"Please use start in range [0, {total_frames - 1}]."
            )

        # Compute frame indices
        # Default end: go forward to total_frames, or backward to -1 (before index 0)
        if end is not None:
            end_idx = end
        else:
            end_idx = total_frames if step > 0 else -1

        # Clamp to valid range (only needed for positive step)
        if step > 0:
            end_idx = min(end_idx, total_frames)

        # Apply max limit to avoid creating huge list
        if max is not None:
            max_end = start + max * step
            if step > 0:
                end_idx = min(end_idx, max_end)
            elif max_end > end_idx:
                end_idx = max_end

        frame_idx = list(range(start, end_idx, step))
        if len(frame_idx) == 0:
            raise ValueError(
                f"No frames to display with start={start}, end={end}, step={step}. "
                "Please check your range parameters."
            )
        # Build grid
        n_frames = len(frame_idx)
        ncols = min(ncols, n_frames)  # Don't create more columns than frames
        images = [self.array[i] for i in frame_idx]
        labels = [
            f"Frame {i}" if title_prefix is None else f"{title_prefix} {i}"
            for i in frame_idx
        ]
        # Pad last row to complete the grid (show_2d requires rectangular input)
        remainder = n_frames % ncols
        pad_count = 0 if remainder == 0 else ncols - remainder
        if pad_count > 0:
            images.extend([np.zeros_like(self.array[0])] * pad_count)
            labels.extend([""] * pad_count)
        image_grid = [images[i : i + ncols] for i in range(0, len(images), ncols)]
        label_grid = [labels[i : i + ncols] for i in range(0, len(labels), ncols)]
        fig, axes = show_2d(image_grid, scalebar=scalebar, title=label_grid, **kwargs)
        if pad_count > 0:
            for ax in np.array(axes).flat[-pad_count:]:
                ax.clear()
                ax.axis("off")
        if suptitle:
            # Reserve fixed space for suptitle (absolute, not proportional)
            suptitle_margin = 0.7
            fig_height = fig.get_figheight()
            fig.subplots_adjust(top=1 - suptitle_margin / fig_height)
            fig.suptitle(suptitle, fontsize=14, y=1 - 0.2 / fig_height)
        return (fig, axes) if returnfig else None
