"""5D-STEM dataset class for time series, tilt series, and other stacked 4D-STEM data."""

from typing import Iterator, Self

import numpy as np
from numpy.typing import NDArray

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.dataset5d import Dataset5d
from quantem.core.utils.masks import create_annular_mask, create_circle_mask
from quantem.core.utils.validators import ensure_valid_array

STACK_TYPES = ("time", "tilt", "energy", "dose", "focus", "generic")


class Dataset5dstem(Dataset5d):
    """5D-STEM dataset with dimensions (stack, scan_row, scan_col, k_row, k_col).

    The stack axis represents time frames, tilt angles, defocus values, etc.
    Dimensions 1-2 are real-space scan positions; dimensions 3-4 are reciprocal-space
    diffraction patterns.

    Parameters
    ----------
    stack_type : str
        Type of stack: "time", "tilt", "energy", "dose", "focus", or "generic".
    stack_values : NDArray | None
        Explicit values for the stack dimension (e.g., timestamps, tilt angles).

    Examples
    --------
    >>> data = read_5dstem("path/to/file.h5")
    >>> len(data)                    # number of frames
    10
    >>> frame = data[0]              # get first frame as Dataset4dstem
    >>> mean_4d = data.stack_mean()  # average over stack -> Dataset4dstem
    >>> for frame in data:           # iterate over frames
    ...     process(frame)
    """

    def __init__(
        self,
        array: NDArray,
        name: str,
        origin: NDArray,
        sampling: NDArray,
        units: list[str],
        signal_units: str = "arb. units",
        metadata: dict | None = None,
        stack_type: str = "generic",
        stack_values: NDArray | None = None,
        _token: object | None = None,
    ):
        metadata = metadata or {}
        for key in ("r_to_q_rotation_cw_deg", "ellipticity"):
            metadata.setdefault(key, None)

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

        if stack_type not in STACK_TYPES:
            raise ValueError(f"stack_type must be one of {STACK_TYPES}, got '{stack_type}'")

        self._stack_type = stack_type

        if stack_values is not None:
            stack_values = np.asarray(stack_values)
            if len(stack_values) != self.shape[0]:
                raise ValueError(
                    f"stack_values length ({len(stack_values)}) must match "
                    f"number of frames ({self.shape[0]})"
                )
        self._stack_values = stack_values
        self._virtual_images: dict[str, Dataset3d] = {}
        self._virtual_detectors: dict[str, dict] = {}

    def __repr__(self) -> str:
        return (
            f"Dataset5dstem(shape={self.shape}, dtype={self.array.dtype}, "
            f"stack_type='{self._stack_type}')"
        )

    def __str__(self) -> str:
        return (
            f"Dataset5dstem '{self.name}'\n"
            f"  shape: {self.shape} ({len(self)} frames)\n"
            f"  stack_type: '{self._stack_type}'\n"
            f"  scan sampling: {self.sampling[1:3]} {self.units[1:3]}\n"
            f"  k sampling: {self.sampling[3:]} {self.units[3:]}"
        )

    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self) -> Iterator[Dataset4dstem]:
        for i in range(len(self)):
            yield self._get_frame(i)

    def __getitem__(self, idx) -> "Dataset4dstem | Dataset5dstem":
        if isinstance(idx, int):
            return self._get_frame(idx)

        # Handle tuple where first element is int (e.g., data[0, ...])
        if isinstance(idx, tuple) and len(idx) > 0 and isinstance(idx[0], int):
            return self._get_frame(idx[0])[idx[1:]]

        # Slicing returns Dataset5dstem with preserved stack_type
        if isinstance(idx, slice):
            sliced_array = self.array[idx]
            sliced_values = self._stack_values[idx] if self._stack_values is not None else None
            return self.from_array(
                array=sliced_array,
                name=self.name,
                origin=self.origin,
                sampling=self.sampling,
                units=self.units,
                signal_units=self.signal_units,
                stack_type=self._stack_type,
                stack_values=sliced_values,
            )

        return super().__getitem__(idx)

    @property
    def stack_type(self) -> str:
        """Type of stack dimension: 'time', 'tilt', 'energy', 'dose', 'focus', or 'generic'."""
        return self._stack_type

    @property
    def stack_values(self) -> NDArray | None:
        """Explicit values for the stack dimension, or None if using indices."""
        return self._stack_values

    @property
    def virtual_images(self) -> dict[str, Dataset3d]:
        """Cached virtual image stacks, keyed by name."""
        return self._virtual_images

    @property
    def virtual_detectors(self) -> dict[str, dict]:
        """Virtual detector configurations for regenerating images."""
        return self._virtual_detectors

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    @classmethod
    def from_array(
        cls,
        array: NDArray,
        name: str | None = None,
        origin: NDArray | tuple | list | None = None,
        sampling: NDArray | tuple | list | None = None,
        units: list[str] | None = None,
        signal_units: str = "arb. units",
        stack_type: str = "generic",
        stack_values: NDArray | None = None,
    ) -> Self:
        """Create Dataset5dstem from a 5D array.

        Parameters
        ----------
        array : NDArray
            5D array with shape (stack, scan_row, scan_col, k_row, k_col).
        name : str, optional
            Dataset name. Default: "5D-STEM dataset".
        origin : array-like, optional
            Origin for each dimension (4 or 5 elements). Default: zeros.
        sampling : array-like, optional
            Sampling for each dimension (4 or 5 elements). Default: ones.
        units : list[str], optional
            Units for each dimension (4 or 5 elements). Default: ["pixels", ...].
        signal_units : str, optional
            Units for intensity values. Default: "arb. units".
        stack_type : str, optional
            Type of stack dimension. Default: "generic".
        stack_values : NDArray, optional
            Explicit values for stack positions (e.g., times, angles).

        Returns
        -------
        Dataset5dstem
        """
        array = ensure_valid_array(array, ndim=5)

        # Accept 4-element inputs (scan + k dims); prepend stack defaults
        def expand_to_5d(arr, default):
            if arr is None:
                return default
            arr = np.asarray(arr)
            if arr.size == 4:
                return np.concatenate([[default[0]], arr])
            return arr

        origin_5d = expand_to_5d(origin, np.zeros(5))
        sampling_5d = expand_to_5d(sampling, np.ones(5))

        if units is None:
            units_5d = ["pixels"] * 5
        elif len(units) == 4:
            units_5d = ["index"] + list(units)
        else:
            units_5d = list(units)

        return cls(
            array=array,
            name=name or "5D-STEM dataset",
            origin=origin_5d,
            sampling=sampling_5d,
            units=units_5d,
            signal_units=signal_units,
            stack_type=stack_type,
            stack_values=stack_values,
            _token=cls._token,
        )

    @classmethod
    def from_4dstem(
        cls,
        datasets: list[Dataset4dstem],
        stack_type: str = "generic",
        stack_values: NDArray | None = None,
        name: str | None = None,
    ) -> Self:
        """Create Dataset5dstem by stacking multiple Dataset4dstem objects.

        Parameters
        ----------
        datasets : list[Dataset4dstem]
            List of 4D-STEM datasets to stack. Must have identical shapes.
        stack_type : str, optional
            Type of stack dimension. Default: "generic".
        stack_values : NDArray, optional
            Explicit values for stack positions.
        name : str, optional
            Dataset name.

        Returns
        -------
        Dataset5dstem
        """
        if not datasets:
            raise ValueError("datasets list cannot be empty")

        first = datasets[0]

        # Validate consistency across all datasets
        for i, ds in enumerate(datasets[1:], start=1):
            if ds.shape != first.shape:
                raise ValueError(
                    f"Dataset {i} shape {ds.shape} doesn't match first dataset shape {first.shape}"
                )
            if not np.allclose(ds.sampling, first.sampling):
                raise ValueError(
                    f"Dataset {i} sampling {ds.sampling} doesn't match first dataset"
                )
            if ds.units != first.units:
                raise ValueError(
                    f"Dataset {i} units {ds.units} doesn't match first dataset"
                )

        stacked = np.stack([d.array for d in datasets], axis=0)

        return cls.from_array(
            array=stacked,
            name=name or "5D-STEM dataset",
            origin=np.concatenate([[0], first.origin]),
            sampling=np.concatenate([[1], first.sampling]),
            units=["index"] + list(first.units),
            signal_units=first.signal_units,
            stack_type=stack_type,
            stack_values=stack_values,
        )

    # -------------------------------------------------------------------------
    # Stack operations
    # -------------------------------------------------------------------------

    def stack_mean(self) -> Dataset4dstem:
        """Average over the stack axis. Returns Dataset4dstem."""
        return self._reduce_stack(np.mean, "mean")

    def stack_sum(self) -> Dataset4dstem:
        """Sum over the stack axis. Returns Dataset4dstem."""
        return self._reduce_stack(np.sum, "sum")

    def stack_max(self) -> Dataset4dstem:
        """Maximum over the stack axis. Returns Dataset4dstem."""
        return self._reduce_stack(np.max, "max")

    def stack_min(self) -> Dataset4dstem:
        """Minimum over the stack axis. Returns Dataset4dstem."""
        return self._reduce_stack(np.min, "min")

    def _reduce_stack(self, func, suffix: str) -> Dataset4dstem:
        """Apply reduction function over stack axis."""
        return Dataset4dstem.from_array(
            array=func(self.array, axis=0),
            name=f"{self.name}_{suffix}",
            origin=self.origin[1:],
            sampling=self.sampling[1:],
            units=self.units[1:],
            signal_units=self.signal_units,
        )

    def _get_frame(self, idx: int) -> Dataset4dstem:
        """Extract a single 4D frame from the stack."""
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Frame index {idx} out of range for {len(self)} frames")

        frame = Dataset4dstem.from_array(
            array=self.array[idx],
            name=f"{self.name}_frame{idx}",
            origin=self.origin[1:],
            sampling=self.sampling[1:],
            units=self.units[1:],
            signal_units=self.signal_units,
        )

        # Inherit STEM metadata
        frame.metadata["r_to_q_rotation_cw_deg"] = self.metadata.get("r_to_q_rotation_cw_deg")
        frame.metadata["ellipticity"] = self.metadata.get("ellipticity")

        # Inherit virtual detector definitions
        for name, info in self._virtual_detectors.items():
            frame._virtual_detectors[name] = {
                "mask": None,
                "mode": info["mode"],
                "geometry": info["geometry"],
            }

        return frame

    # -------------------------------------------------------------------------
    # Virtual imaging
    # -------------------------------------------------------------------------

    def get_virtual_image(
        self,
        mask: np.ndarray | None = None,
        mode: str | None = None,
        geometry: tuple | None = None,
        name: str = "virtual_image",
        attach: bool = True,
    ) -> Dataset3d:
        """Compute virtual image stack for all frames.

        Parameters
        ----------
        mask : np.ndarray, optional
            Custom mask matching diffraction pattern shape.
        mode : str, optional
            Mask mode: "circle" or "annular".
        geometry : tuple, optional
            For "circle": ((cy, cx), radius).
            For "annular": ((cy, cx), (r_inner, r_outer)).
        name : str, optional
            Name for the virtual image. Default: "virtual_image".
        attach : bool, optional
            Store result in virtual_images dict. Default: True.

        Returns
        -------
        Dataset3d
            Virtual image stack with shape (n_frames, scan_row, scan_col).
        """
        dp_shape = self.array.shape[-2:]

        if mask is not None:
            if mask.shape != dp_shape:
                raise ValueError(f"Mask shape {mask.shape} != diffraction pattern shape {dp_shape}")
            final_mask = mask
        elif mode and geometry:
            if mode == "circle":
                center, radius = geometry
                final_mask = create_circle_mask(dp_shape, center, radius)
            elif mode == "annular":
                center, radii = geometry
                final_mask = create_annular_mask(dp_shape, center, radii)
            else:
                raise ValueError(f"Unknown mode '{mode}'. Use 'circle' or 'annular'.")
        else:
            raise ValueError("Provide either mask or both mode and geometry")

        virtual_stack = np.sum(self.array * final_mask, axis=(-1, -2))

        vi = Dataset3d.from_array(
            array=virtual_stack,
            name=name,
            origin=self.origin[:3],
            sampling=self.sampling[:3],
            units=self.units[:3],
            signal_units=self.signal_units,
        )

        if attach:
            self._virtual_images[name] = vi
            self._virtual_detectors[name] = {
                "mask": final_mask.copy() if mask is not None else None,
                "mode": mode,
                "geometry": geometry,
            }

        return vi

    # -------------------------------------------------------------------------
    # Copy
    # -------------------------------------------------------------------------

    def _copy_custom_attributes(self, new_dataset) -> None:
        """Copy Dataset5dstem-specific attributes."""
        super()._copy_custom_attributes(new_dataset)
        new_dataset._stack_type = self._stack_type
        new_dataset._stack_values = self._stack_values.copy() if self._stack_values is not None else None
        new_dataset._virtual_images = {}
        new_dataset._virtual_detectors = {
            name: {"mask": None, "mode": info["mode"], "geometry": info["geometry"]}
            for name, info in self._virtual_detectors.items()
        }
