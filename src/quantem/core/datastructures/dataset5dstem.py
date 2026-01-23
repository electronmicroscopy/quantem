"""5D-STEM dataset class for time series, tilt series, and other stacked 4D-STEM data."""

from typing import Iterator, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.dataset5d import Dataset5d
from quantem.core.utils.diffractive_imaging_utils import fit_probe_circle
from quantem.core.utils.masks import create_annular_mask, create_circle_mask
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import show_2d

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
    >>> data = read_5dstem("path/to/file.h5")  # shape (4, 6, 7, 3, 5)
    >>> len(data)                              # number of frames -> 4

    Indexing (integer removes dimension, slice keeps it):

    >>> data[2]           # -> Dataset4dstem (6, 7, 3, 5) one frame
    >>> data[1:3]         # -> Dataset5dstem (2, 6, 7, 3, 5) substack
    >>> data[:, 4, 1]     # -> Dataset3d (4, 3, 5) one scan position, all frames
    >>> data[:, 1:5, 2:6] # -> Dataset5dstem (4, 4, 4, 3, 5) scan region crop
    >>> data[..., 0:2, 1:4]  # -> Dataset5dstem (4, 6, 7, 2, 3) k-space crop

    Stack operations (reduce over stack axis):

    >>> data.stack_mean()  # -> Dataset4dstem
    >>> data.stack_std()   # -> Dataset4dstem

    Virtual imaging:

    >>> vi = data.get_virtual_image(mode="circle", geometry=((1, 2), 1))
    >>> vi.shape  # -> (4, 6, 7) = (frames, scan_row, scan_col)
    """

    def __init__(
        self,
        array: ArrayLike,
        name: str,
        origin: ArrayLike,
        sampling: ArrayLike,
        units: list[str],
        signal_units: str = "arb. units",
        metadata: dict | None = None,
        stack_type: str = "generic",
        stack_values: ArrayLike | None = None,
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
        # Handle integer indexing (including numpy integers)
        if isinstance(idx, (int, np.integer)):
            return self._get_frame(int(idx))
        # Handle tuple where first element is int (e.g., data[0, ...])
        if isinstance(idx, tuple) and len(idx) > 0 and isinstance(idx[0], (int, np.integer)):
            return self._get_frame(int(idx[0]))[idx[1:]]
        # Reject advanced indexing on stack axis (lists, arrays, boolean masks)
        if isinstance(idx, (list, np.ndarray)):
            raise TypeError(
                "Advanced indexing with lists/arrays on stack axis is not supported. "
                "Use integer indexing or slices instead."
            )
        if isinstance(idx, tuple) and len(idx) > 0 and isinstance(idx[0], (list, np.ndarray)):
            raise TypeError(
                "Advanced indexing with lists/arrays on stack axis is not supported. "
                "Use integer indexing or slices instead."
            )
        # Get result from base class slicing
        result = super().__getitem__(idx)
        # If result is still 5D, wrap back into Dataset5dstem with preserved metadata
        if result.array.ndim == 5:
            return self.from_array(
                array=result.array,
                name=result.name,
                origin=result.origin,
                sampling=result.sampling,
                units=result.units,
                signal_units=result.signal_units,
                stack_type=self._stack_type,
                stack_values=None,  # Don't try to slice stack_values
            )

        return result

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

    @classmethod
    def from_file(cls, file_path: str, file_type: str | None = None, **kwargs) -> "Dataset5dstem":
        """Load Dataset5dstem from a file.

        Parameters
        ----------
        file_path : str
            Path to data file.
        file_type : str | None
            File type hint. If None, auto-detect from extension.
        **kwargs
            Additional arguments passed to read_5dstem (e.g., stack_type).

        Returns
        -------
        Dataset5dstem

        Examples
        --------
        >>> data = Dataset5dstem.from_file("path/to/data.h5")
        >>> data = Dataset5dstem.from_file("path/to/data.h5", stack_type="tilt")
        """
        from quantem.core.io.file_readers import read_5dstem

        return read_5dstem(file_path, file_type=file_type, **kwargs)

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    @classmethod
    def from_array(
        cls,
        array: ArrayLike,
        name: str | None = None,
        origin: ArrayLike | None = None,
        sampling: ArrayLike | None = None,
        units: list[str] | None = None,
        signal_units: str = "arb. units",
        stack_type: str = "generic",
        stack_values: ArrayLike | None = None,
    ) -> Self:
        """Create Dataset5dstem from a 5D array.

        Parameters
        ----------
        array : array-like
            5D array with shape (stack, scan_row, scan_col, k_row, k_col).
        name : str, optional
            Dataset name. Default: "5D-STEM dataset".
        origin : array-like, optional
            Origin for each dimension (5 elements). Default: zeros.
        sampling : array-like, optional
            Sampling for each dimension (5 elements). Default: ones.
        units : list[str], optional
            Units for each dimension (5 elements). Default: ["pixels", ...].
        signal_units : str, optional
            Units for intensity values. Default: "arb. units".
        stack_type : str, optional
            Type of stack dimension. Default: "generic".
        stack_values : array-like, optional
            Explicit values for stack positions (e.g., times, angles).

        Returns
        -------
        Dataset5dstem

        Examples
        --------
        Basic usage:

        >>> import numpy as np
        >>> arr = np.random.rand(10, 256, 256, 64, 64)
        >>> data = Dataset5dstem.from_array(arr, stack_type="time")
        >>> data.shape
        (10, 256, 256, 64, 64)

        With calibrations:

        >>> data = Dataset5dstem.from_array(
        ...     arr,
        ...     stack_type="tilt",
        ...     stack_values=np.linspace(-60, 60, 10),  # tilt angles in degrees
        ...     sampling=[1, 0.5, 0.5, 0.01, 0.01],
        ...     units=["deg", "nm", "nm", "1/nm", "1/nm"],
        ... )
        """
        array = ensure_valid_array(array, ndim=5)

        origin_5d = np.zeros(5) if origin is None else np.asarray(origin)
        sampling_5d = np.ones(5) if sampling is None else np.asarray(sampling)
        units_5d = ["pixels"] * 5 if units is None else list(units)

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

        Examples
        --------
        Stack multiple 4D-STEM datasets into a tilt series:

        >>> from quantem.core.io import read_4dstem
        >>> frames = [read_4dstem(f"tilt_{i:02d}.h5") for i in range(10)]
        >>> tilt_series = Dataset5dstem.from_4dstem(
        ...     frames,
        ...     stack_type="tilt",
        ...     stack_values=np.linspace(-60, 60, 10),
        ... )
        >>> tilt_series.shape
        (10, 256, 256, 128, 128)

        Stack synthetic data:

        >>> datasets = [Dataset4dstem.from_array(np.random.rand(64, 64, 32, 32)) for _ in range(5)]
        >>> data = Dataset5dstem.from_4dstem(datasets, stack_type="time")
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

    def stack_std(self) -> Dataset4dstem:
        """Standard deviation over the stack axis. Returns Dataset4dstem."""
        return self._reduce_stack(np.std, "std")

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

        return frame

    # -------------------------------------------------------------------------
    # Virtual imaging
    # -------------------------------------------------------------------------
    def get_virtual_image(
        self,
        mask: ArrayLike | None = None,
        mode: str | None = None,
        geometry: tuple | list | None = None,
        name: str = "virtual_image",
        attach: bool = True,
    ) -> Dataset3d:
        """Compute virtual image stack for all frames.

        Parameters
        ----------
        mask : np.ndarray, optional
            Custom mask matching diffraction pattern shape (k_row, k_col).
        mode : str, optional
            Mask mode: "circle" or "annular".
        geometry : tuple or list, optional
            Detector geometry in pixels. Format depends on mode:

            For "circle" mode:
                - ``(None, radius)`` : Auto-fit center per frame.
                - ``((cy, cx), radius)`` : Fixed center for all frames.
                - ``[((cy0, cx0), r0), ...]`` : Per-frame geometry list.

            For "annular" mode:
                - ``(None, (r_inner, r_outer))`` : Auto-fit center per frame.
                - ``((cy, cx), (r_inner, r_outer))`` : Fixed center.
                - ``[((cy0, cx0), (r0_in, r0_out)), ...]`` : Per-frame list.

        name : str, optional
            Name for the virtual image. Default: "virtual_image".
        attach : bool, optional
            Store result in virtual_images dict. Default: True.

        Returns
        -------
        Dataset3d
            Virtual image stack with shape (n_frames, scan_row, scan_col).

        Notes
        -----
        All geometry values are in pixels. To convert from mrad:

            >>> radius_px = radius_mrad / data.sampling[-1]

        Examples
        --------
        Auto-fit center per frame:

        >>> bf = data.get_virtual_image(mode="circle", geometry=(None, 20))
        >>> adf = data.get_virtual_image(mode="annular", geometry=(None, (30, 80)))

        Fixed center for all frames:

        >>> k_center = (data.shape[-2] // 2, data.shape[-1] // 2)
        >>> bf = data.get_virtual_image(mode="circle", geometry=(k_center, 20))

        Per-frame geometry:

        >>> geometries = [(center, 20) for center in fitted_centers]
        >>> bf = data.get_virtual_image(mode="circle", geometry=geometries)
        """
        dp_shape = self.array.shape[-2:]
        n_frames = len(self)

        if mask is not None:
            if mask.shape != dp_shape:
                raise ValueError(f"Mask shape {mask.shape} != diffraction pattern shape {dp_shape}")
            virtual_stack = np.sum(self.array * mask, axis=(-1, -2))
        elif mode is not None:
            # Full auto-detection: geometry=None
            if geometry is None:
                geometries = self._auto_fit_centers(mode, None)
                virtual_stack = self._compute_per_frame_virtual(mode, geometries, dp_shape)
                geometry = geometries
            # Per-frame geometry list
            elif isinstance(geometry, list):
                if len(geometry) != n_frames:
                    raise ValueError(
                        f"geometry list length ({len(geometry)}) must match "
                        f"number of frames ({n_frames})"
                    )
                virtual_stack = self._compute_per_frame_virtual(mode, geometry, dp_shape)
            else:
                # Single geometry tuple: (center_or_none, radius_or_radii)
                center, radius_or_radii = geometry
                if center is None:
                    # Auto-fit center per frame (radius may also be None for auto)
                    geometries = self._auto_fit_centers(mode, radius_or_radii)
                    virtual_stack = self._compute_per_frame_virtual(mode, geometries, dp_shape)
                    geometry = geometries  # Store fitted geometries
                else:
                    # Fixed center for all frames
                    if mode == "circle":
                        final_mask = create_circle_mask(dp_shape, center, radius_or_radii)
                    elif mode == "annular":
                        final_mask = create_annular_mask(dp_shape, center, radius_or_radii)
                    else:
                        raise ValueError(f"Unknown mode '{mode}'. Use 'circle' or 'annular'.")
                    virtual_stack = np.sum(self.array * final_mask, axis=(-1, -2))
        else:
            raise ValueError("Provide either mask or both mode and geometry")

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
                "mask": mask.copy() if mask is not None else None,
                "mode": mode,
                "geometry": geometry,
            }

        return vi

    def _auto_fit_centers(self, mode: str, radius_or_radii) -> list:
        """Fit probe center for each frame and return list of geometries.

        If radius_or_radii is None, also auto-detect the radius.
        """
        geometries = []
        for i in range(len(self)):
            dp_mean = np.mean(self.array[i], axis=(0, 1))
            cy, cx, detected_radius = fit_probe_circle(dp_mean, show=False)
            radius = detected_radius if radius_or_radii is None else radius_or_radii
            geometries.append(((cy, cx), radius))
        return geometries

    def _compute_per_frame_virtual(self, mode: str, geometries: list, dp_shape: tuple) -> np.ndarray:
        """Compute virtual images with per-frame geometry."""
        virtual_stack = np.zeros((len(self), self.shape[1], self.shape[2]), dtype=self.array.dtype)
        for i, geom in enumerate(geometries):
            center, radius_or_radii = geom
            if mode == "circle":
                mask = create_circle_mask(dp_shape, center, radius_or_radii)
            elif mode == "annular":
                mask = create_annular_mask(dp_shape, center, radius_or_radii)
            else:
                raise ValueError(f"Unknown mode '{mode}'")
            virtual_stack[i] = np.sum(self.array[i] * mask, axis=(-1, -2))
        return virtual_stack

    def show_virtual_images(self, figsize: tuple[int, int] | None = None, **kwargs) -> tuple:
        """Display all virtual images stored in the dataset.

        Parameters
        ----------
        figsize : tuple[int, int] | None
            Figure size. If None, auto-calculated.
        **kwargs
            Arguments passed to show_2d (cmap, norm, cbar, etc.)

        Returns
        -------
        tuple
            (fig, axs) from matplotlib.
        """
        if not self.virtual_images:
            print("No virtual images. Create with get_virtual_image().")
            return None, None

        # Each virtual image is Dataset3d - show first frame of each
        arrays = [vi.array[0] for vi in self.virtual_images.values()]
        titles = [f"{name} (frame 0)" for name in self.virtual_images.keys()]

        n = len(arrays)
        if figsize is None:
            figsize = (4 * min(n, 4), 4 * ((n + 3) // 4))

        return show_2d(arrays, title=titles, figax_size=figsize, **kwargs)

    def regenerate_virtual_images(self) -> None:
        """Regenerate virtual images from stored detector information."""
        if not self._virtual_detectors:
            return

        self._virtual_images.clear()

        for name, info in self._virtual_detectors.items():
            try:
                if info["mode"] is not None and info["geometry"] is not None:
                    self.get_virtual_image(
                        mode=info["mode"],
                        geometry=info["geometry"],
                        name=name,
                        attach=True,
                    )
                else:
                    print(f"Warning: Cannot regenerate '{name}' - insufficient detector info.")
            except Exception as e:
                print(f"Warning: Failed to regenerate '{name}': {e}")

    def update_virtual_detector(
        self,
        name: str,
        mask: np.ndarray | None = None,
        mode: str | None = None,
        geometry: tuple | None = None,
    ) -> None:
        """Update virtual detector and regenerate the corresponding image.

        Parameters
        ----------
        name : str
            Name of virtual detector to update.
        mask : np.ndarray | None
            New mask (must match DP dimensions).
        mode : str | None
            New mode ("circle" or "annular").
        geometry : tuple | None
            New geometry.
        """
        if name not in self._virtual_detectors:
            raise ValueError(f"Detector '{name}' not found. Available: {list(self._virtual_detectors.keys())}")

        self._virtual_detectors[name]["mask"] = mask.copy() if mask is not None else None
        self._virtual_detectors[name]["mode"] = mode
        self._virtual_detectors[name]["geometry"] = geometry

        self.get_virtual_image(mask=mask, mode=mode, geometry=geometry, name=name, attach=True)

    def clear_virtual_images(self) -> None:
        """Clear virtual images while keeping detector information."""
        self._virtual_images.clear()

    def clear_all_virtual_data(self) -> None:
        """Clear both virtual images and detector information."""
        self._virtual_images.clear()
        self._virtual_detectors.clear()

    def show(self, *args, **kwargs):
        """Not implemented for 5D data.

        Raises
        ------
        NotImplementedError
            Always raised. Use alternative methods for visualization.

        See Also
        --------
        show_virtual_images : Display virtual images.
        data[i].show() : Show a single 4D frame.
        data[i].dp_mean.show() : Show mean diffraction pattern.
        """
        raise NotImplementedError(
            "show() is not meaningful for 5D data. Use:\n"
            "  - data[i].show() for a single frame\n"
            "  - data[i].dp_mean.show() for mean DP\n"
            "  - data.show_virtual_images() for virtual images\n"
            "  - data.get_virtual_image(...).show() for virtual image stack"
        )

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
