from typing import Any, Optional, Self, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.utils import get_array_module
from quantem.core.utils.validators import (
    ensure_valid_array,
    validate_ndinfo,
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
        origin: Union[NDArray, tuple, list, float, int],
        sampling: Union[NDArray, tuple, list, float, int],
        units: Union[list[str], tuple, list],
        signal_units: str = "arb. units",
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use Dataset.from_array() to instantiate this class.")

        self._array = ensure_valid_array(array)
        self.name = name
        self.origin = origin
        self.sampling = sampling
        self.units = units
        self.signal_units = signal_units

    @classmethod
    def from_array(
        cls,
        array: Any,  # Input can be array-like
        name: str | None = None,
        origin: Union[NDArray, tuple, list, float, int] | None = None,
        sampling: Union[NDArray, tuple, list, float, int] | None = None,
        units: Union[list[str], tuple, list] | None = None,
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
        origin: Union[NDArray, tuple, list, float, int] | None
            The origin of the Dataset.
        sampling: Union[NDArray, tuple, list, float, int] | None
            The sampling of the Dataset.
        units: Union[list[str], tuple, list] | None
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
        self._array = ensure_valid_array(value, dtype=self.dtype, ndim=self.ndim)

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
    def origin(self, value: Union[NDArray, tuple, list, float, int]) -> None:
        self._origin = validate_ndinfo(value, self.ndim, "origin")

    @property
    def sampling(self) -> NDArray:
        return self._sampling

    @sampling.setter
    def sampling(self, value: Union[NDArray, tuple, list, float, int]) -> None:
        self._sampling = validate_ndinfo(value, self.ndim, "sampling")

    @property
    def units(self) -> list[str]:
        return self._units

    @units.setter
    def units(self, value: Union[list[str], tuple, list]) -> None:
        self._units = validate_units(value, self.ndim)

    @property
    def signal_units(self) -> str:
        return self._signal_units

    @signal_units.setter
    def signal_units(self, value: str) -> None:
        self._signal_units = str(value)

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
    def _xp(self):
        return get_array_module(self.array)

    @property
    def device(self) -> str:
        """
        Outputting a string is likely temporary -- once we have our use cases we can
        figure out a more permanent device solution that enables easier translation between
        numpy <-> cupy <-> torch <-> numpy
        """
        return str(self.array.device)

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
    def copy(self) -> Self:
        """
        Copies Dataset.

        Parameters
        ----------
        copy_attributes: bool
            If True, copies non-standard attributes. Standard attributes (array, metadata)
            are always deep-copied.
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

        return new_dataset

    def mean(self, axes: Optional[tuple[int, ...]] = None) -> Any:
        """
        Computes and returns mean of the data array.

        Parameters
        ----------
        axes: tuple, optional
            Axes over which to compute mean. If None specified, mean of all elements is computed.

        Returns
        --------
        mean: scalar or array (np.ndarray or cp.ndarray)
            Mean of the data.
        """
        return self.array.mean(axis=axes)

    def max(self, axes: Optional[tuple[int, ...]] = None) -> Any:
        """
        Computes and returns max of the data array.

        Parameters
        ----------
        axes: tuple, optional
            Axes over which to compute max. If None specified, max of all elements is computed.

        Returns
        --------
        maximum: scalar or array (np.ndarray or cp.ndarray)
            Maximum of the data.
        """
        return self.array.max(axis=axes)

    def min(self, axes: Optional[tuple[int, ...]] = None) -> Any:
        """
        Computes and returns min of the data array.

        Parameters
        ----------
        axes: tuple, optional
            Axes over which to compute min. If None specified, min of all elements is computed.

        Returns
        --------
        minimum: scalar or array (np.ndarray or cp.ndarray)
            Minimum of the data.
        """
        return self.array.min(axis=axes)

    def pad(
        self,
        pad_width: Union[int, tuple[int, int], tuple[tuple[int, int], ...]] | None = None,
        output_shape: tuple[int, ...] | None = None,
        modify_in_place: bool = False,
        **kwargs: Any,
    ) -> Optional["Dataset"]:
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

    def crop(self, crop_widths, axes=None, modify_in_place=False):
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
            crop_widths = (crop_widths,)
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
                stop = dim - after if after != 0 else None
                full_slices.append(slice(start, stop))
            else:
                full_slices.append(slice(None))
        if modify_in_place is False:
            dataset = self.copy()
            dataset.array = dataset.array[tuple(full_slices)]
            return dataset
        else:
            self.array = self.array[tuple(full_slices)]

    def bin(
        self,
        bin_factors,
        axes=None,
        modify_in_place=False,
    ):
        """
        Bin the Dataset by integer factors along selected axes using block summation.

        Notes
        -----
        - Sampling is multiplied by the bin factor on each binned axis.
        - Origin is shifted to the center of the first block on each binned axis:
          origin_new = origin_old + 0.5 * (factor - 1) * sampling_old
        - Any trailing remainder (shape % factor) is dropped.

        Parameters
        ----------
        bin_factors : int | tuple[int, ...]
            Bin factors per specified axis (must be positive integers).
        axes : int | tuple[int, ...] | None
            Axes to bin. If None, all axes are binned.
        modify_in_place : bool
            If True, modifies this dataset; otherwise returns a new Dataset.

        Returns
        -------
        Dataset or None
        """
        xp = self._xp

        # Normalize axes
        if axes is None:
            axes = tuple(range(self.ndim))
        elif np.isscalar(axes):
            axes = (int(axes),)
        else:
            axes = tuple(int(ax) for ax in axes)

        # Normalize factors to per-axis tuple
        if isinstance(bin_factors, int):
            bin_factors = tuple([int(bin_factors)] * len(axes))
        elif isinstance(bin_factors, (list, tuple)):
            if len(bin_factors) != len(axes):
                raise ValueError("bin_factors and axes must have the same length.")
            bin_factors = tuple(int(fac) for fac in bin_factors)
        else:
            raise TypeError("bin_factors must be an int or tuple of ints.")

        if any(fac <= 0 for fac in bin_factors):
            raise ValueError("All bin factors must be positive integers.")

        axis_to_factor = dict(zip(axes, bin_factors))

        # Compute effective slices that drop any remainder at the end
        slices = []
        effective_lengths = []
        for a0 in range(self.ndim):
            if a0 in axis_to_factor:
                fac = axis_to_factor[a0]
                length_eff = (self.shape[a0] // fac) * fac
                slices.append(slice(0, length_eff))
                effective_lengths.append(length_eff)
            else:
                slices.append(slice(None))
                effective_lengths.append(self.shape[a0])

        # Build reshape dims and which axes to reduce (sum over block dimension)
        reshape_dims = []
        reduce_axes = []
        running_axis = 0
        for a1 in range(self.ndim):
            if a1 in axis_to_factor:
                fac = axis_to_factor[a1]
                nblocks = effective_lengths[a1] // fac
                reshape_dims.extend([nblocks, fac])
                # The 'fac' dimension is immediately after the 'nblocks' dim
                reduce_axes.append(running_axis + 1)
                running_axis += 2
            else:
                reshape_dims.append(effective_lengths[a1])
                running_axis += 1

        # Perform block summation
        array_view = self.array[tuple(slices)]
        array_binned = xp.sum(array_view.reshape(tuple(reshape_dims)), axis=tuple(reduce_axes))

        # Update metadata
        new_sampling = self.sampling.copy()
        new_origin = self.origin.copy()
        for ax_binned, fac_binned in axis_to_factor.items():
            # sampling increases by factor
            old_sampling = new_sampling[ax_binned]
            new_sampling[ax_binned] = old_sampling * fac_binned
            # shift origin to the center of the first block
            new_origin[ax_binned] = new_origin[ax_binned] + 0.5 * (fac_binned - 1) * old_sampling

        if modify_in_place:
            self._array = array_binned
            self._sampling = new_sampling
            self._origin = new_origin
            return None

        dataset = self.copy()
        dataset.array = array_binned
        dataset.sampling = new_sampling
        dataset.origin = new_origin
        # Optional: annotate the name with factors (3 sig figs) for the binned axes
        factors_str = " ".join(
            f"{axis_to_factor[a2]:.3g}" if a2 in axis_to_factor else "1" for a2 in range(self.ndim)
        )
        dataset.name = f"{self.name} (binned factors {factors_str})"
        return dataset

    def fourier_resample(
        self,
        out_shape: Optional[tuple[int, ...]] = None,
        factors: Optional[Union[float, tuple[float, ...]]] = None,
        axes: Optional[tuple[int, ...]] = None,
        modify_in_place: bool = False,
    ) -> Optional["Dataset"]:
        """
        Resample the dataset using Fourier cropping/zero-padding.

        Parameters
        ----------
        out_shape : tuple of int, optional
            Desired output shape along specified axes.
        factors : float or tuple of float, optional
            Downsample/upsample factors per axis (e.g., 0.5 halves size, 2 doubles size).
            If scalar, same factor applied to all specified axes.
        axes : tuple of int, optional
            Axes to resample. If None, all axes are used.
        modify_in_place : bool, default False
            If True, modify this dataset in place. Otherwise return a new Dataset.

        Returns
        -------
        Dataset or None
            Resampled dataset if modify_in_place is False, otherwise None.
        """
        xp = self._xp
        if axes is None:
            axes = tuple(range(self.ndim))

        # Determine output shape
        if out_shape is not None and factors is not None:
            raise ValueError("Specify either out_shape or factors, not both.")

        if out_shape is None and factors is None:
            raise ValueError("Must specify either out_shape or factors.")

        if factors is not None:
            if np.isscalar(factors):
                factors = (float(factors),) * len(axes)
            elif len(factors) != len(axes):
                raise ValueError("factors length must match number of axes.")
            out_shape = tuple(int(round(self.shape[ax] * fac)) for ax, fac in zip(axes, factors))
        else:
            if len(out_shape) != len(axes):
                raise ValueError("out_shape length must match number of axes.")
            factors = tuple(new_len / self.shape[ax] for ax, new_len in zip(axes, out_shape))

        # Forward FFT
        F = xp.fft.fftn(self.array, axes=axes, norm="ortho")
        F = xp.fft.fftshift(F, axes=axes)

        # Slice or pad Fourier domain
        slices = []
        pad_widths = []
        for ax, new_len in zip(axes, out_shape):
            old_len = self.shape[ax]
            if new_len < old_len:  # crop
                start = (old_len - new_len) // 2
                end = start + new_len
                slices.append(slice(start, end))
                pad_widths.append(None)
            elif new_len > old_len:  # pad
                slices.append(slice(None))
                before = (new_len - old_len) // 2
                after = new_len - old_len - before
                pad_widths.append((before, after))
            else:  # unchanged
                slices.append(slice(None))
                pad_widths.append(None)

        # Apply slices
        full_slices = []
        ax_map = dict(zip(axes, slices))
        for a0 in range(self.ndim):
            if a0 in ax_map:
                full_slices.append(ax_map[a0])
            else:
                full_slices.append(slice(None))
        F_resampled = F[tuple(full_slices)]

        # Apply padding
        for ax_index, pad in zip(axes, pad_widths):
            if pad is not None:
                pad_spec = [(0, 0)] * self.ndim
                pad_spec[ax_index] = pad
                F_resampled = xp.pad(F_resampled, pad_spec, mode="constant")

        # Inverse FFT
        F_resampled = xp.fft.ifftshift(F_resampled, axes=axes)
        array_resampled = xp.fft.ifftn(F_resampled, axes=axes, norm="ortho")

        # If input was real, discard small imaginary part
        if xp.isrealobj(self.array):
            array_resampled = array_resampled.real

        # Scale to preserve mean intensity
        in_size = np.prod([self.shape[ax] for ax in axes])
        out_size = np.prod([out_shape[idx] for idx in range(len(axes))])
        scale = in_size / out_size
        array_resampled *= scale

        # Update sampling (inverse of factor)
        new_sampling = self.sampling.copy()
        for ax_resampled, fac_resampled in zip(axes, factors):
            new_sampling[ax_resampled] /= fac_resampled

        # Update origin to keep the array CENTER fixed in world coordinates
        new_origin = self.origin.copy()
        for ax_resampled, fac_resampled in zip(axes, factors):
            old_len = self.shape[ax_resampled]
            new_len = out_shape[axes.index(ax_resampled)]
            old_center_idx = (old_len - 1) / 2.0
            new_center_idx = (new_len - 1) / 2.0
            old_sampling = self.sampling[ax_resampled]
            # place new center at the same physical coordinate as old center
            new_origin[ax_resampled] = (
                self.origin[ax_resampled]
                + old_center_idx * old_sampling
                - new_center_idx * new_sampling[ax_resampled]
            )

        # Prepare output
        factors_str = " ".join(f"{fac:.3g}" for fac in factors)
        if modify_in_place:
            self._array = array_resampled
            self._sampling = new_sampling
            self._origin = new_origin
            self.name = self.name + f" (resampled factors {factors_str})"
            return None
        else:
            dataset = self.copy()
            dataset.array = array_resampled
            dataset.sampling = new_sampling
            dataset.origin = new_origin
            dataset.name = self.name + f" (resampled factors {factors_str})"
            return dataset

    def transpose(
        self,
        order: Optional[tuple[int, ...]] = None,
        modify_in_place: bool = False,
    ) -> Optional["Dataset"]:
        """
        Transpose (permute) axes of the dataset and reorder metadata accordingly.

        Parameters
        ----------
        order : tuple[int, ...], optional
            A permutation of range(self.ndim). If None, axes are reversed (NumPy's default).
        modify_in_place : bool, default False
            If True, modify this dataset in place. Otherwise return a new Dataset.

        Returns
        -------
        Dataset or None
            Transposed dataset if modify_in_place is False, otherwise None.
        """
        if order is None:
            order = tuple(range(self.ndim - 1, -1, -1))

        if len(order) != self.ndim or set(order) != set(range(self.ndim)):
            raise ValueError(f"'order' must be a permutation of 0..{self.ndim - 1}; got {order!r}")

        array_t = self.array.transpose(order)

        # Reorder metadata to match new axis order
        new_origin = self.origin[list(order)].copy()
        new_sampling = self.sampling[list(order)].copy()
        new_units = [self.units[ax] for ax in order]

        if modify_in_place:
            # Use private attrs to avoid dtype/ndim enforcement in the setter
            self._array = array_t
            self._origin = new_origin
            self._sampling = new_sampling
            self._units = new_units
            return None

        # Create a new Dataset without extra array copies
        return type(self).from_array(
            array=array_t,
            name=self.name,  # keep name unchanged for now
            origin=new_origin,
            sampling=new_sampling,
            units=new_units,
            signal_units=self.signal_units,
        )

    def astype(
        self,
        dtype: DTypeLike,
        copy: bool = True,
        modify_in_place: bool = False,
    ) -> Optional["Dataset"]:
        """
        Cast the array to a new dtype. Metadata is unchanged.

        Parameters
        ----------
        dtype : DTypeLike
            Target dtype (e.g., np.float32, "complex64", etc.).
        copy : bool, default True
            If False and no cast is needed, a view may be returned by the backend.
        modify_in_place : bool, default False
            If True, modify this dataset in place. Otherwise return a new Dataset.

        Returns
        -------
        Dataset or None
            Dtype-cast dataset if modify_in_place is False, otherwise None.
        """
        array_cast = self.array.astype(dtype, copy=copy)

        if modify_in_place:
            # Bypass the array setter so we can actually change dtype
            self._array = array_cast
            return None

        # Build a new Dataset with identical metadata
        return type(self).from_array(
            array=array_cast,
            name=self.name,
            origin=self.origin.copy(),
            sampling=self.sampling.copy(),
            units=self.units[:],
            signal_units=self.signal_units,
        )
