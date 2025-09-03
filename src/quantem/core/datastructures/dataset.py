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
        Updates sampling and origin (origin -> center of first block).
        """
        xp = self._xp

        # Normalize axes
        if axes is None:
            axes = tuple(range(self.ndim))
        elif np.isscalar(axes):
            axes = (int(axes),)
        else:
            axes = tuple(int(ax) for ax in axes)

        # Normalize factors
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

        # Drop remainders
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

        # Reshape to blocks and sum
        reshape_dims = []
        reduce_axes = []
        running_axis = 0
        for a1 in range(self.ndim):
            if a1 in axis_to_factor:
                fac = axis_to_factor[a1]
                nblocks = effective_lengths[a1] // fac
                reshape_dims.extend([nblocks, fac])
                reduce_axes.append(running_axis + 1)  # the 'fac' dim
                running_axis += 2
            else:
                reshape_dims.append(effective_lengths[a1])
                running_axis += 1

        array_view = self.array[tuple(slices)]
        array_binned = xp.sum(array_view.reshape(tuple(reshape_dims)), axis=tuple(reduce_axes))

        # ---- Metadata (ensure float to avoid truncation) ----
        new_sampling = self.sampling.astype(float).copy()
        new_origin = self.origin.astype(float).copy()
        for ax_binned, fac_binned in axis_to_factor.items():
            old_sampling = new_sampling[ax_binned]
            new_sampling[ax_binned] = old_sampling * fac_binned
            # shift origin to center of the first block
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
        Fourier resample via centered crop (down) / zero-pad (up), using default FFT norms.
        Preserves mean and keeps the physical center fixed.
        """
        xp = self._xp
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
            out_shape = tuple(
                max(1, int(round(self.shape[a1] * f))) for a1, f in zip(axes, factors)
            )
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
        F = xp.fft.fftn(self.array, axes=axes)
        F = xp.fft.fftshift(F, axes=axes)

        # Center-aligned crop/pad per axis (so DC stays centered)
        axis_to_outlen = dict(zip(axes, out_shape))
        slices = []
        pad_specs = []
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
            F_rs = xp.pad(F_rs, pad_specs, mode="constant")

        # Inverse FFT
        F_rs = xp.fft.ifftshift(F_rs, axes=axes)
        array_resampled = xp.fft.ifftn(F_rs, axes=axes)

        if xp.isrealobj(self.array):
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

        # Name suffix
        factors_map = {axk: (axis_to_outlen[axk] / self.shape[axk]) for axk in axes}
        factors_list = [f"{factors_map.get(a8, 1.0):.3g}" for a8 in range(self.ndim)]
        suffix = " ".join(factors_list)

        if modify_in_place:
            self._array = array_resampled
            self._sampling = new_sampling
            self._origin = new_origin
            self.name = self.name + f" (resampled factors {suffix})"
            return None

        ds = self.copy()
        ds.array = array_resampled
        ds.sampling = new_sampling
        ds.origin = new_origin
        ds.name = self.name + f" (resampled factors {suffix})"
        return ds

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
