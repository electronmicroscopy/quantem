import os
import numbers
from pathlib import Path
from typing import Any, Literal, Optional, Self, Union, overload

import numpy as np
from numpy.typing import DTypeLike, NDArray

from quantem.core.io.serialize import AutoSerialize
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
        array (NDArray): The underlying n-dimensional NumPy array data.
        name (str): A descriptive name for the dataset.
        origin (NDArray): The origin coordinates for each dimension (1D array).
        sampling (NDArray): The sampling rate/spacing for each dimension (1D array).
        units (list[str]): Units for each dimension.
        signal_units (str): Units for the array values.

    Notes
    -----
    This branch is NumPy-only. CuPy arrays are explicitly rejected.
    """

    _token = object()
    _registry: dict[int, type] = {}

    def __init__(
        self,
        array: Any,  # Input can be array-like
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        signal_units: str = "arb. units",
        metadata: Optional[dict] = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use Dataset.from_array() to instantiate this class.")
        super().__init__()
        arr = ensure_valid_array(array)
        if not isinstance(arr, np.ndarray):
            raise TypeError("Dataset requires a NumPy array (CuPy is not supported on this branch).")
        self._array = arr
        self.name = name
        self.origin = origin
        self.sampling = sampling
        self.units = units
        self.signal_units = signal_units
        self._file_path = None
        self._metadata = {} if metadata is None else dict(metadata)

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
        if not isinstance(validated_array, np.ndarray):
            raise TypeError("Dataset requires a NumPy array (CuPy is not supported on this branch).")
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
        """The underlying n-dimensional NumPy array data."""
        return self._array

    @array.setter
    def array(self, value: NDArray) -> None:
        arr = ensure_valid_array(value, ndim=self.ndim)  # want to allow changing dtype
        if not isinstance(arr, np.ndarray):
            raise TypeError("Dataset requires a NumPy array (CuPy is not supported on this branch).")
        self._array = arr
        # self._array = ensure_valid_array(value, dtype=self.dtype, ndim=self.ndim)

    @property
    def metadata(self) -> dict:
        return self._metadata

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
    def device(self) -> str:
        """
        Outputting a string is likely temporary -- once we have our use cases we can
        figure out a more permanent device solution that enables easier translation between
        numpy <-> torch <-> numpy, etc.

        For NumPy-only datasets, this is always "cpu".
        """
        return "cpu"

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
            "_registry",
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
        mean: scalar or array (np.ndarray)
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
        maximum: scalar or array (np.ndarray)
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
        minimum: scalar or array (np.ndarray)
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
        Pads Dataset data array using numpy.pad.
        Metadata (origin, sampling) is not modified.

        Parameters
        ----------
        pad_width: int, tuple
            Number of values padded to the edges of each axis. See numpy.pad documentation.
        output_shape: tuple of int, optional
            Convenience option to pad to a desired output shape by symmetric padding.
        modify_in_place: bool
            If True, modifies this dataset's array directly. If False, returns a new Dataset.
        kwargs: dict
            Additional keyword arguments passed to numpy.pad.

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
            axes = (int(axes),)
            crop_widths = (crop_widths[0],)  # Take first crop_width for single axis
        else:
            axes = tuple(int(a) for a in axes)

        if len(crop_widths) != len(axes):
            raise ValueError("Length of crop_widths must match length of axes.")

        full_slices = []
        crop_dict = dict(zip(axes, crop_widths))
        for axis, _ in enumerate(self.shape):
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

        self.array = self.array[tuple(full_slices)]
        return None

    @overload
    def bin(
        self,
        bin_factors,
        axes,
        modify_in_place: Literal[True],
        reducer: str = "sum",
    ) -> None: ...

    @overload
    def bin(
        self,
        bin_factors,
        axes=None,
        modify_in_place: Literal[False] = False,
        reducer: str = "sum",
    ) -> Self: ...

    def bin(
        self,
        bin_factors,
        axes=None,
        modify_in_place: bool = False,
        reducer: str = "sum",
    ) -> Self | None:
        """
        Bin the Dataset by integer factors along selected axes using block reduction.

        Parameters
        ----------
        bin_factors : int | tuple[int, ...]
            Bin factors per specified axis (positive integers).
        axes : int | tuple[int, ...] | None
            Axes to bin. If None, all axes are binned.
        modify_in_place : bool
            If True, modifies this dataset; otherwise returns a new Dataset.
        reducer : {"sum","mean"}
            Reduction applied within each block. "sum" (default) preserves counts;
            "mean" averages over each block (block volume = product of factors).

        Notes
        -----
        - Any remainder (shape % factor) is dropped on each binned axis.
        - Sampling is multiplied by the factor on each binned axis.
        - Origin is shifted to the center of the first block:
            origin_new = origin_old + 0.5 * (factor - 1) * sampling_old
        """
        reducer_norm = str(reducer).lower()
        if reducer_norm not in ("sum", "mean"):
            raise ValueError("reducer must be 'sum' or 'mean'")

        if axes is None:
            axes = tuple(range(self.ndim))
        elif np.isscalar(axes):
            axes = (int(axes),)
        else:
            axes = tuple(int(ax) for ax in axes)

        if isinstance(bin_factors, numbers.Integral):
            bin_factors = (int(bin_factors),) * len(axes)
        elif isinstance(bin_factors, (list, tuple)):
            if len(bin_factors) != len(axes):
                raise ValueError("bin_factors and axes must have the same length.")
            for fac in bin_factors:
                if not isinstance(fac, numbers.Integral):
                    raise TypeError(f"Each bin factor must be an integer, got {fac!r}")
            bin_factors = tuple(int(fac) for fac in bin_factors)
        else:
            raise TypeError("bin_factors must be an int or tuple of ints.")

        if any(fac <= 0 for fac in bin_factors):
            raise ValueError("All bin factors must be positive integers.")

        axis_to_factor = dict(zip(axes, bin_factors))

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

        reshape_dims = []
        reduce_axes = []
        running_axis = 0
        for a1 in range(self.ndim):
            if a1 in axis_to_factor:
                fac = axis_to_factor[a1]
                nblocks = effective_lengths[a1] // fac
                reshape_dims.extend([nblocks, fac])
                reduce_axes.append(running_axis + 1)
                running_axis += 2
            else:
                reshape_dims.append(effective_lengths[a1])
                running_axis += 1

        array_view = self.array[tuple(slices)].reshape(tuple(reshape_dims))
        array_binned = np.sum(array_view, axis=tuple(reduce_axes))
        if reducer_norm == "mean":
            block_volume = 1
            for fac_b in axis_to_factor.values():
                block_volume *= fac_b
            array_binned = array_binned / block_volume

        new_sampling = self.sampling.astype(float).copy()
        new_origin = self.origin.astype(float).copy()
        for ax_binned, fac_binned in axis_to_factor.items():
            old_sampling = new_sampling[ax_binned]
            new_sampling[ax_binned] = old_sampling * fac_binned
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

        factors_str = " ".join(
            f"{axis_to_factor[a2]:.3g}" if a2 in axis_to_factor else "1" for a2 in range(self.ndim)
        )
        suffix = f"(binned factors {factors_str}" + (", mean)" if reducer_norm == "mean" else ")")
        dataset.name = f"{self.name} {suffix}"
        return dataset

    def fourier_resample(
        self,
        out_shape: Optional[tuple[int, ...]] = None,
        factors: Optional[Union[float, tuple[float, ...]]] = None,
        axes: Optional[tuple[int, ...]] = None,
        modify_in_place: bool = False,
    ) -> Optional["Dataset"]:
        """
        Fourier resample the dataset by centered cropping (downsample) or zero padding (upsample).
        The operation is performed in the Fourier domain using fftshift alignment and default FFT
        normalization. The physical center is preserved and the mean intensity is kept constant.

        Parameters
        ----------
        out_shape : tuple of int, optional
            Output lengths for the selected axes. Must have the same length as `axes`.
            Use this when specifying the exact output shape.
        factors : float or tuple of float, optional
            Multiplicative resampling factors for each axis. A scalar factor is applied
            to all axes. Use this when specifying scaling rather than absolute size.
            Exactly one of `out_shape` or `factors` must be provided.
        axes : tuple of int, optional
            Axes to resample. Defaults to all axes. A scalar is interpreted as a single axis.
        modify_in_place : bool
            If True, update the dataset in place and return None.
            If False, return a new Dataset with the resampled array and updated metadata.

        Returns
        -------
        Dataset or None
            A new resampled dataset if `modify_in_place` is False, otherwise None.
        """
        if axes is None:
            axes = tuple(range(self.ndim))
        elif np.isscalar(axes):
            axes = (int(axes),)
        else:
            axes = tuple(int(a0) for a0 in axes)

        if (out_shape is None) == (factors is None):
            raise ValueError("Specify exactly one of out_shape or factors.")

        # Resolve out_shape & factors
        if factors is not None:
            if np.isscalar(factors):
                factors = (float(factors),) * len(axes)
            else:
                factors = tuple(float(f) for f in factors)
                if len(factors) != len(axes):
                    raise ValueError("factors length must match number of axes.")
            out_shape = tuple(max(1, int(round(self.shape[a1] * f))) for a1, f in zip(axes, factors))
        else:
            if len(out_shape) != len(axes):
                raise ValueError("out_shape length must match number of axes.")
            out_shape = tuple(int(nl) for nl in out_shape)
            factors = tuple(out_len / self.shape[a2] for a2, out_len in zip(axes, out_shape))

        if any(nl < 1 for nl in out_shape):
            raise ValueError("All output lengths must be >= 1.")

        def _shift_center_index(n: int) -> int:
            # index of DC after fftshift: n//2 for even, (n-1)//2 for odd
            return n // 2 if (n % 2 == 0) else (n - 1) // 2

        # Forward FFT (default normalization: forward unscaled, inverse 1/N)
        F = np.fft.fftn(self.array, axes=axes)
        F = np.fft.fftshift(F, axes=axes)

        # Center-aligned crop/pad per axis (so DC stays centered)
        axis_to_outlen = dict(zip(axes, out_shape))
        slices: list[slice] = []
        pad_specs: list[tuple[int, int]] = []
        for a3 in range(self.ndim):
            if a3 in axis_to_outlen:
                old_len = self.shape[a3]
                new_len = axis_to_outlen[a3]
                oc = _shift_center_index(old_len)
                nc = _shift_center_index(new_len)

                if new_len < old_len:
                    start = oc - nc
                    end = start + new_len
                    slices.append(slice(start, end))
                    pad_specs.append((0, 0))
                elif new_len > old_len:
                    slices.append(slice(None))
                    before = nc - oc
                    after = new_len - old_len - before
                    pad_specs.append((before, after))
                else:
                    slices.append(slice(None))
                    pad_specs.append((0, 0))
            else:
                slices.append(slice(None))
                pad_specs.append((0, 0))

        F_rs = F[tuple(slices)]
        if any(pw != (0, 0) for pw in pad_specs):
            F_rs = np.pad(F_rs, pad_specs, mode="constant")

        # Inverse FFT
        F_rs = np.fft.ifftshift(F_rs, axes=axes)
        array_resampled = np.fft.ifftn(F_rs, axes=axes)

        if np.isrealobj(self.array):
            array_resampled = array_resampled.real

        # Mean preservation with default FFTs:
        # ones -> F(0)=N_in, IFFT size N_out -> constant N_in/N_out; multiply by N_out/N_in.
        N_in = int(np.prod([self.shape[a4] for a4 in axes]))
        N_out = int(np.prod([axis_to_outlen[a5] for a5 in axes]))
        if N_in > 0 and N_out > 0:
            array_resampled *= N_out / N_in

        # Metadata (ensure float arrays to avoid truncation)
        new_sampling = self.sampling.astype(float).copy()
        for a6, out_len in zip(axes, out_shape):
            fac_actual = out_len / self.shape[a6]
            new_sampling[a6] = new_sampling[a6] / fac_actual

        new_origin = self.origin.astype(float).copy()
        for a7, out_len in zip(axes, out_shape):
            old_len = self.shape[a7]
            old_center_idx = (old_len - 1) / 2.0
            new_center_idx = (out_len - 1) / 2.0
            old_sampling = self.sampling[a7]
            new_origin[a7] = (
                self.origin[a7] + old_center_idx * old_sampling - new_center_idx * new_sampling[a7]
            )

        if modify_in_place:
            self._array = array_resampled
            self._sampling = new_sampling
            self._origin = new_origin
            return None

        ds = self.copy()
        ds.array = array_resampled
        ds.sampling = new_sampling
        ds.origin = new_origin
        return ds

    def __getitem__(self, index) -> Self:
        """
        General indexing method for Dataset objects.

        Returns a new Dataset (or subclass) corresponding to the indexed data.
        Metadata (origin, sampling, units) is sliced or reduced accordingly.
        Handles step slicing (e.g., [::2]) by multiplying sampling accordingly.

        Parameters
        ----------
        index : int | slice | tuple | Ellipsis
            Indexing expression applied to the underlying array.

        Returns
        -------
        Dataset
            A new Dataset instance with appropriately adjusted metadata.
        """
        array_view = self.array[index]

        # Normalize index into tuple form
        if not isinstance(index, tuple):
            index = (index,)

        # Expand Ellipsis
        if Ellipsis in index:
            ellipsis_pos = index.index(Ellipsis)
            num_missing = self.ndim - (len(index) - 1)
            index = index[:ellipsis_pos] + (slice(None),) * num_missing + index[ellipsis_pos + 1 :]

        # Pad with slices if index shorter than ndim
        if len(index) < self.ndim:
            index = index + (slice(None),) * (self.ndim - len(index))

        # Compute which dimensions are kept
        kept_axes = [i for i, idx in enumerate(index) if not isinstance(idx, (int, np.integer))]

        # Slice/reduce metadata accordingly
        new_origin = np.asarray(self.origin)[kept_axes] if np.ndim(self.origin) > 0 else self.origin
        new_sampling = (
            np.asarray(self.sampling)[kept_axes] if np.ndim(self.sampling) > 0 else self.sampling
        )
        new_units = [self.units[i] for i in kept_axes] if len(self.units) > 0 else self.units

        # Adjust sampling for slice steps (e.g. [::2] doubles spacing)
        for i, idx in enumerate(index):
            if isinstance(idx, slice) and idx.step not in (None, 1):
                if i in kept_axes:
                    j = kept_axes.index(i)
                    new_sampling[j] *= idx.step

        out_ndim = array_view.ndim

        if out_ndim == self.ndim:
            cls = type(self)
        else:
            try:
                cls = self._registry[out_ndim]
            except KeyError:
                cls = Dataset

        # Construct new dataset
        return cls.from_array(  # type: ignore ## would be nice to properly type slicing, but hard
            array=array_view,
            name=f"{self.name}{index}",
            origin=new_origin,
            sampling=new_sampling,
            units=new_units,
            signal_units=self.signal_units,
        )

    @classmethod
    def register_dimension(cls, ndim: int):
        """Decorator for registering subclasses for a specific dimensionality."""

        def decorator(subclass):
            cls._registry[ndim] = subclass
            return subclass

        return decorator
