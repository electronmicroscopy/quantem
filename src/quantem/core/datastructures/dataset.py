import os
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Self, cast, overload

import numpy as np
from numpy.typing import DTypeLike, NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.utils import get_array_module
from quantem.core.utils.validators import (
    ensure_valid_array,
    validate_ndinfo,
    validate_pathlike,
    validate_units,
)


class Dataset(AutoSerialize):
    """
    A class representing a multi-dimensional dataset with metadata.
    Uses standard properties and validation within __init__ for type safety.

    Attributes (Properties):
        array (NDArray | Any): The underlying n-dimensional array data (Any for CuPy).
        name (str): A descriptive name for the dataset.
        origin (NDArray): The origin coordinates for each dimension (1D array).
        sampling (NDArray): The sampling rate/spacing for each dimension (1D array).
        units (list[str]): Units for each dimension.
        signal_units (str): Units for the array values.
    """

    _token = object()

    def __init__(
        self,
        array: Any,  # Input can be array-like
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        signal_units: str = "arb. units",
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use Dataset.from_array() to instantiate this class.")
        super().__init__()
        self._array = ensure_valid_array(array)
        self.name = name
        self.origin = origin
        self.sampling = sampling
        self.units = units
        self.signal_units = signal_units
        self._file_path = None

    @classmethod
    def from_array(
        cls,
        array: Any,  # Input can be array-like
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple | list | None = None,
        signal_units: str = "arb. units",
    ) -> Self:
        """
        Validates and creates a Dataset from an array.

        Parameters
        ----------
        array: Any
            The array to validate and create a Dataset from.
        name: str | None
            The name of the Dataset.
        origin: NDArray | tuple | list | float | int | None
            The origin of the Dataset.
        sampling: NDArray | tuple | list | float | int | None
            The sampling of the Dataset.
        units: list[str] | tuple | list | None
            The units of the Dataset.
        signal_units: str
            The units of the signal.

        Returns
        -------
        Dataset
            A Dataset object with the validated array and metadata.
        """
        validated_array = ensure_valid_array(array)
        _ndim = validated_array.ndim

        # Set defaults if None
        _name = name if name is not None else f"{_ndim}d dataset"
        _origin = origin if origin is not None else np.zeros(_ndim)
        _sampling = sampling if sampling is not None else np.ones(_ndim)
        _units = units if units is not None else ["pixels"] * _ndim

        return cls(
            array=validated_array,
            name=_name,
            origin=_origin,
            sampling=_sampling,
            units=_units,
            signal_units=signal_units,
            _token=cls._token,
        )

    # --- Properties ---
    @property
    def array(self) -> NDArray:
        """The underlying n-dimensional array data. Can be a np.ndarray or cp.ndarray."""
        return self._array

    @array.setter
    def array(self, value: NDArray) -> None:
        self._array = ensure_valid_array(value, ndim=self.ndim)  # want to allow changing dtype
        # self._array = ensure_valid_array(value, dtype=self.dtype, ndim=self.ndim)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def origin(self) -> NDArray:
        return self._origin

    @origin.setter
    def origin(self, value: NDArray | tuple | list | float | int) -> None:
        self._origin = validate_ndinfo(value, self.ndim, "origin")

    @property
    def sampling(self) -> NDArray:
        return self._sampling

    @sampling.setter
    def sampling(self, value: NDArray | tuple | list | float | int) -> None:
        self._sampling = validate_ndinfo(value, self.ndim, "sampling")

    @property
    def units(self) -> list[str]:
        return self._units

    @units.setter
    def units(self, value: list[str] | tuple[str, ...] | list) -> None:
        self._units = validate_units(value, self.ndim)

    @property
    def signal_units(self) -> str:
        return self._signal_units

    @signal_units.setter
    def signal_units(self, value: str) -> None:
        self._signal_units = str(value)

    @property
    def file_path(self) -> Path | None:
        return self._file_path

    @file_path.setter
    def file_path(self, value: os.PathLike | str | None) -> None:
        self._file_path = validate_pathlike(value)

    # --- Derived Properties ---
    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def dtype(self) -> DTypeLike:
        return self.array.dtype

    @property
    def _xp(self) -> ModuleType:
        return get_array_module(self.array)

    @property
    def device(self) -> str:
        """
        Outputting a string is likely temporary -- once we have our use cases we can
        figure out a more permanent device solution that enables easier translation between
        numpy <-> cupy <-> torch <-> numpy
        """
        return str(self.array.device)  # type:ignore

    # --- Summaries ---
    def __repr__(self) -> str:
        description = [
            f"Dataset(shape={self.shape}, dtype={self.dtype}, name='{self.name}')",
            f"  sampling: {self.sampling}",
            f"  units: {self.units}",
            f"  signal units: '{self.signal_units}'",
        ]
        return "\n".join(description)

    def __str__(self) -> str:
        description = [
            f"quantem Dataset named '{self.name}'",
            f"  shape: {self.shape}",
            f"  dtype: {self.dtype}",
            f"  device: {self.device}",
            f"  origin: {self.origin}",
            f"  sampling: {self.sampling}",
            f"  units: {self.units}",
            f"  signal units: '{self.signal_units}'",
        ]
        return "\n".join(description)

    # --- Methods ---
    def copy(self, copy_custom_attributes: bool = True) -> Self:
        """
        Copies Dataset.

        Parameters
        ----------
        copy_custom_attributes: bool, optional
            If True, copies non-standard attributes. Standard attributes (array, metadata)
            are always deep-copied. Default is True.
        """
        # Metadata arrays (origin, sampling) are numpy, use copy()
        # Units list is copied by slicing
        new_dataset = type(self).from_array(
            array=self.array.copy(),
            name=self.name,
            origin=self.origin.copy(),
            sampling=self.sampling.copy(),
            units=self.units[:],
            signal_units=self.signal_units,
        )

        # Copy custom attributes if requested
        if copy_custom_attributes:
            self._copy_custom_attributes(new_dataset)

        return new_dataset

    def _copy_custom_attributes(self, new_dataset: Self) -> None:
        """
        Copy custom attributes from self to new_dataset.
        This method can be overridden by subclasses to handle specific custom attributes.

        Parameters
        ----------
        new_dataset : Self
            The new dataset instance to copy attributes to
        """
        # Standard attributes that should not be copied
        standard_attrs = {
            "_array",
            "_name",
            "_origin",
            "_sampling",
            "_units",
            "_signal_units",
            "_token",
            "__dict__",
            "__class__",
            "__weakref__",
        }

        # Copy all non-standard attributes (but not properties)
        for attr_name in dir(self):
            if not attr_name.startswith("__") and attr_name not in standard_attrs:
                # Skip properties first - check the class, not the instance
                if not isinstance(getattr(type(self), attr_name, None), property):
                    if hasattr(self, attr_name) and not callable(getattr(self, attr_name)):
                        try:
                            attr_value = getattr(self, attr_name)
                            # Try to copy the attribute if it has a copy method
                            if hasattr(attr_value, "copy"):
                                setattr(new_dataset, attr_name, attr_value.copy())
                            else:
                                setattr(new_dataset, attr_name, attr_value)
                        except (AttributeError, TypeError):
                            # Skip attributes that can't be copied
                            pass

    def mean(self, axes: int | tuple[int, ...] | None = None) -> Any:
        """
        Computes and returns mean of the data array.

        Parameters
        ----------
        axes: int or tuple of ints, optional
            Axes over which to compute mean. If None specified, mean of all elements is computed.

        Returns
        --------
        mean: scalar or array (np.ndarray or cp.ndarray)
            Mean of the data.
        """
        return self.array.mean(axis=axes)

    def max(self, axes: int | tuple[int, ...] | None = None) -> Any:
        """
        Computes and returns max of the data array.

        Parameters
        ----------
        axes: int or tuple of ints, optional
            Axes over which to compute max. If None specified, max of all elements is computed.

        Returns
        --------
        maximum: scalar or array (np.ndarray or cp.ndarray)
            Maximum of the data.
        """
        return self.array.max(axis=axes)

    def min(self, axes: int | tuple[int, ...] | None = None) -> Any:
        """
        Computes and returns min of the data array.

        Parameters
        ----------
        axes: int or tuple of ints, optional
            Axes over which to compute min. If None specified, min of all elements is computed.

        Returns
        --------
        minimum: scalar or array (np.ndarray or cp.ndarray)
            Minimum of the data.
        """
        return self.array.min(axis=axes)

    @overload
    def pad(
        self,
        pad_width: int | tuple[int, int] | tuple[tuple[int, int], ...] | None,
        output_shape: tuple[int, ...] | None,
        modify_in_place: Literal[True],
        **kwargs: Any,
    ) -> None: ...

    @overload
    def pad(
        self,
        pad_width: int | tuple[int, int] | tuple[tuple[int, int], ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        modify_in_place: Literal[False] = False,
        **kwargs: Any,
    ) -> "Dataset": ...

    def pad(
        self,
        pad_width: int | tuple[int, int] | tuple[tuple[int, int], ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        modify_in_place: bool = False,
        **kwargs: Any,
    ) -> "Dataset | None":
        """
        Pads Dataset data array using numpy.pad or cupy.pad.
        Metadata (origin, sampling) is not modified.

        Parameters
        ----------
        pad_width: int, tuple
            Number of values padded to the edges of each axis. See numpy.pad documentation.
        modify_in_place: bool
            If True, modifies this dataset's array directly. If False, returns a new Dataset.
        kwargs: dict
            Additional keyword arguments passed to numpy.pad or cupy.pad.

        Returns
        --------
        Dataset or None
            Padded Dataset if modify_in_place is False, otherwise None.
        """
        if pad_width is not None:
            if output_shape is not None:
                raise ValueError("pad_width and output_shape cannot both be specified.")
            padded_array = np.pad(self.array, pad_width=pad_width, **kwargs)
        elif output_shape is not None:
            if len(output_shape) != self.ndim:
                raise ValueError("output_shape must be a tuple of length ndim.")
            padded_array = np.pad(
                self.array,
                pad_width=[
                    (
                        max(0, int(np.floor((output_shape[i] - self.shape[i]) / 2))),
                        max(0, int(np.ceil((output_shape[i] - self.shape[i]) / 2))),
                    )
                    for i in range(self.ndim)
                ],
                **kwargs,
            )
        else:
            raise ValueError("pad_width or output_shape must be specified.")

        if modify_in_place:
            self._array = padded_array
            return None
        else:
            new_dataset = self.copy()
            new_dataset.array = padded_array
            new_dataset.name = self.name + " (padded)"
            return new_dataset

    @overload
    def crop(
        self,
        crop_widths: tuple[tuple[int, int], ...],
        axes: tuple | None,
        modify_in_place: Literal[True],
    ) -> None: ...

    @overload
    def crop(
        self,
        crop_widths: tuple[tuple[int, int], ...],
        axes: tuple | None = None,
        modify_in_place: Literal[False] = False,
    ) -> Self: ...

    def crop(
        self,
        crop_widths: tuple[tuple[int, int], ...],
        axes: tuple | None = None,
        modify_in_place: bool = False,
    ) -> Self | None:
        """
        Crops Dataset

        Parameters
        ----------
        crop_widths:tuple
            Min and max for cropping each axis specified as a tuple
        axes:
            Axes over which to crop. If None specified, all are cropped.
        modify_in_place: bool
            If True, modifies dataset

        Returns
        --------
        Dataset (cropped) only if modify_in_place is False
        """
        if axes is None:
            if len(crop_widths) != self.ndim:
                raise ValueError("crop_widths must match number of dimensions when axes is None.")
            axes = tuple(range(self.ndim))
        elif np.isscalar(axes):
            axes = (axes,)
            crop_widths = (crop_widths[0],)  # Take first crop_width for single axis
        else:
            axes = tuple(axes)

        if len(crop_widths) != len(axes):
            raise ValueError("Length of crop_widths must match length of axes.")

        full_slices = []
        crop_dict = dict(zip(axes, crop_widths))
        for axis, dim in enumerate(self.shape):
            if axis in crop_dict:
                before, after = crop_dict[axis]
                start = before
                stop = after if after != 0 else None
                full_slices.append(slice(start, stop))
            else:
                full_slices.append(slice(None))
        if modify_in_place is False:
            dataset = self.copy()
            dataset.array = dataset.array[tuple(full_slices)]
            return dataset
        else:
            self.array = self.array[tuple(full_slices)]
            return None

    @overload
    def bin(
        self,
        bin_factors,
        axes,
        modify_in_place: Literal[True],
    ) -> None: ...

    @overload
    def bin(
        self,
        bin_factors,
        axes=None,
        modify_in_place: Literal[False] = False,
    ) -> Self: ...

    def bin(
        self,
        bin_factors,
        axes=None,
        modify_in_place: bool = False,
    ) -> Self | None:
        """
        Bins Dataset

        Parameters
        ----------
        bin_factors:tuple or int
            bin factors for each axis
        axes:
            Axis over which to bin. If None is specified, all axes are binned.
        modify_in_place: bool
            If True, modifies dataset

        Returns
        --------
        Dataset (binned) only if modify_in_place is False
        """
        if axes is None:
            axes = tuple(range(self.ndim))
        elif np.isscalar(axes):
            axes = (axes,)

        if isinstance(bin_factors, int):
            bin_factors = tuple([bin_factors] * len(axes))
        elif isinstance(bin_factors, (list, tuple)):
            if len(bin_factors) != len(axes):
                raise ValueError("bin_factors and axes must have the same length.")
            bin_factors = tuple(bin_factors)
        else:
            raise TypeError("bin_factors must be an int or tuple of ints.")

        axis_to_factor = dict(zip(axes, bin_factors))

        slices = []
        new_shape = []
        for axis in range(self.ndim):
            if axis in axis_to_factor:
                factor = axis_to_factor[axis]
                length = self.shape[axis] - (self.shape[axis] % factor)
                slices.append(slice(0, length))
                new_shape.extend([length // factor, factor])
            else:
                slices.append(slice(None))
                new_shape.append(self.shape[axis])

        reshape_dims = []
        reduce_axes = []
        current_axis = 0

        for axis in range(self.ndim):
            if axis in axis_to_factor:
                reshape_dims.extend([new_shape[current_axis], axis_to_factor[axis]])
                reduce_axes.append(len(reshape_dims) - 1)
                current_axis += 2
            else:
                reshape_dims.append(new_shape[current_axis])
                current_axis += 1

        if modify_in_place is False:
            dataset = self.copy()
            dataset.array = np.sum(
                dataset.array[tuple(slices)].reshape(reshape_dims),
                axis=tuple(reduce_axes),
            )
            # Update sampling for binned axes # TODO improve this implementation
            for axis, factor in axis_to_factor.items():
                axis = cast(int, axis)
                if axis < len(dataset.sampling):
                    dataset.sampling[axis] *= factor
            return dataset
        else:
            self.array = np.sum(
                self.array[tuple(slices)].reshape(reshape_dims), axis=tuple(reduce_axes)
            )
            # Update sampling for binned axes
            for axis, factor in axis_to_factor.items():
                axis = cast(int, axis)
                if axis < len(self.sampling):
                    self.sampling[axis] *= factor
            return None
