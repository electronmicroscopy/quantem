from typing import Any, Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge
from numpy.typing import NDArray

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4d import Dataset4d
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import show_2d
from quantem.core.visualization.visualization_utils import ScalebarConfig


class Dataset4dstem(Dataset4d):
    """A 4D-STEM dataset class that inherits from Dataset4d.

    This class represents a 4D scanning transmission electron microscopy (STEM) dataset,
    where the data consists of a 4D array with dimensions (scan_y, scan_x, dp_y, dp_x).
    The first two dimensions represent real space scanning positions, while the latter
    two dimensions represent reciprocal space diffraction patterns.

    Attributes
    ----------
    virtual_images : dict[str, Dataset2d]
        Dictionary storing virtual images generated from the 4D-STEM dataset.
        Keys are image names and values are Dataset2d objects containing the images.
    virtual_detectors : dict[str, dict]
        Dictionary storing virtual detector information (masks, modes, geometry)
        for regenerating virtual images after dataset operations.

    Notes
    -----
    Virtual images are automatically regenerated after dataset operations like cropping,
    padding, and binning. The detector information (mode and geometry) is preserved and
    used to create new virtual images that match the transformed dataset dimensions.

    For virtual images created with custom masks, regeneration is not automatic due to
    the complexity of mask transformation. These will need to be recreated manually.
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
        """Initialize a 4D-STEM dataset.

        Parameters
        ----------
        array : NDArray | Any
            The underlying 4D array data
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
        self._virtual_images = {}
        self._virtual_detectors = {}  # Store detector information for regeneration

    @classmethod
    def from_file(cls, file_path: str, file_type: str) -> "Dataset4dstem":
        """
        Create a new Dataset4dstem from a file.

        Parameters
        ----------
        file_path : str
            Path to the data file
        file_type : str
            The type of file reader needed. See rosettasciio for supported formats
            https://hyperspy.org/rosettasciio/supported_formats/index.html

        Returns
        -------
        Dataset4dstem
            A new Dataset4dstem instance loaded from the file
        """
        # Import here to avoid circular imports
        from quantem.core.io.file_readers import read_4dstem

        return read_4dstem(file_path, file_type)

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
        """
        Create a new Dataset4dstem from an array.

        Parameters
        ----------
        array : NDArray | Any
            The underlying 4D array data
        name : str | None, optional
            A descriptive name for the dataset. If None, defaults to "4D-STEM dataset"
        origin : NDArray | tuple | list | float | int | None, optional
            The origin coordinates for each dimension. If None, defaults to zeros
        sampling : NDArray | tuple | list | float | int | None, optional
            The sampling rate/spacing for each dimension. If None, defaults to ones
        units : list[str] | tuple | list | None, optional
            Units for each dimension. If None, defaults to ["pixels"] * 4
        signal_units : str, optional
            Units for the array values, by default "arb. units"

        Returns
        -------
        Dataset4dstem
            A new Dataset4dstem instance
        """
        array = ensure_valid_array(array, ndim=4)
        return cls(
            array=array,
            name=name if name is not None else "4D-STEM dataset",
            origin=origin if origin is not None else np.zeros(4),
            sampling=sampling if sampling is not None else np.ones(4),
            units=units if units is not None else ["pixels"] * 4,
            signal_units=signal_units,
            _token=cls._token,
        )

    @property
    def virtual_images(self) -> dict[str, Dataset2d]:
        """
        Dictionary storing virtual images generated from the 4D-STEM dataset.

        Returns
        -------
        dict[str, Dataset2d]
            Dictionary with image names as keys and Dataset2d objects as values
        """
        return self._virtual_images

    @property
    def virtual_detectors(self) -> dict[str, dict]:
        """
        Dictionary storing virtual detector information for regenerating virtual images.

        Returns
        -------
        dict[str, dict]
            Dictionary with detector names as keys and detector info dictionaries as values.
            Each detector info dict contains 'mask', 'mode', and 'geometry' keys.
        """
        return self._virtual_detectors

    @property
    def dp_mean(self) -> Dataset2d:
        """
        Dataset containing the mean diffraction pattern.

        Returns
        -------
        Dataset
            A Dataset containing the mean diffraction pattern
        """
        if hasattr(self, "_dp_mean"):
            return self._dp_mean
        else:
            return self.get_dp_mean(attach=False)

    def get_dp_mean(self, attach: bool = True) -> Dataset2d:
        """
        Get mean diffraction pattern.

        Parameters
        ----------
        attach : bool, optional
            If True, attaches mean diffraction pattern to self, by default True

        Returns
        -------
        Dataset
            A new Dataset with the mean diffraction pattern
        """
        dp_mean = self.mean((0, 1))

        dp_mean_dataset = Dataset2d.from_array(
            array=dp_mean,
            name=self.name + "_dp_mean",
            origin=self.origin[-2:],
            sampling=self.sampling[-2:],
            units=self.units[-2:],
            signal_units=self.signal_units,
        )

        if attach is True:
            self._dp_mean = dp_mean_dataset

        return dp_mean_dataset

    @property
    def dp_max(self) -> Dataset2d:
        """
        Dataset containing the max diffraction pattern.

        Returns
        -------
        Dataset
            A Dataset containing the max diffraction pattern
        """
        if hasattr(self, "_dp_max"):
            return self._dp_max
        else:
            return self.get_dp_max(attach=False)

    def get_dp_max(self, attach: bool = True) -> Dataset2d:
        """
        Get max diffraction pattern.

        Parameters
        ----------
        attach : bool, optional
            If True, attaches max diffraction pattern to dataset, by default True

        Returns
        -------
        Dataset
            A new Dataset with the max diffraction pattern
        """
        dp_max = self.max((0, 1))

        dp_max_dataset = Dataset2d.from_array(
            array=dp_max,
            name=self.name + "_dp_max",
            origin=self.origin[-2:],
            sampling=self.sampling[-2:],
            units=self.units[-2:],
            signal_units=self.signal_units,
        )

        if attach is True:
            self._dp_max = dp_max_dataset

        return dp_max_dataset

    @property
    def dp_median(self) -> Dataset2d:
        """
        Dataset containing the median diffraction pattern.

        Returns
        -------
        Dataset
            A Dataset containing the median diffraction pattern
        """
        if hasattr(self, "_dp_median"):
            return self._dp_median
        else:
            return self.get_dp_median(attach=False)

    def get_dp_median(self, attach: bool = True) -> Dataset2d:
        """
        Get median diffraction pattern.

        Parameters
        ----------
        attach : bool, optional
            If True, attaches median diffraction pattern to dataset, by default True

        Returns
        -------
        Dataset
            A new Dataset with the median diffraction pattern
        """
        dp_median = np.median(self.array, axis=(0, 1))

        dp_median_dataset = Dataset2d.from_array(
            array=dp_median,
            name=self.name + "_dp_median",
            origin=self.origin[-2:],
            sampling=self.sampling[-2:],
            units=self.units[-2:],
            signal_units=self.signal_units,
        )

        if attach is True:
            self._dp_median = dp_median_dataset

        return dp_median_dataset

    def get_virtual_image(
        self,
        mask: np.ndarray | None = None,
        mode: str | None = None,
        geometry: tuple | None = None,
        name: str = "virtual_image",
        attach: bool = True,
        show: bool = False,
    ) -> Dataset2d:
        """
        Get virtual image using either a provided mask or by creating a mask from mode and geometry.

        Parameters
        ----------
        mask : np.ndarray | None, optional
            Mask for forming virtual images from 4D-STEM data. The mask should be the same
            shape as the datacube Kx and Ky. If provided, mode and geometry are ignored.
        mode : str | None, optional
            Mode for automatic mask creation. Options are "circle" or "annular".
            Required if mask is not provided.
        geometry : tuple | None, optional
            Geometry parameters for automatic mask creation:
            - For "circle" mode: ((cy, cx), r) where (cy, cx) is center and r is radius
            - For "annular" mode: ((cy, cx), (r_inner, r_outer)) where (cy, cx) is center
              and (r_inner, r_outer) are inner and outer radii
            Required if mask is not provided.
        name : str, optional
            Name of virtual image, by default "virtual_image"
        attach : bool, optional
            If True, attaches virtual image to dataset, by default True
        show : bool, optional
            If True, displays the virtual image overlaid with the mean diffraction pattern
            and detector geometry using matplotlib.

        Returns
        -------
        Dataset2d
            A new Dataset2d with the virtual image
        """
        if mask is not None:
            # Use provided mask
            if mask.shape != self.array.shape[-2:]:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match diffraction pattern shape {self.array.shape[-2:]}"
                )
            final_mask = mask
        elif mode is not None and geometry is not None:
            # Create mask from mode and geometry
            if mode == "circle":
                if (
                    len(geometry) != 2
                    or len(geometry[0]) != 2
                    or not isinstance(geometry[1], (int, float))
                ):
                    raise ValueError("For circle mode, geometry must be ((cy, cx), r)")
                center, radius = geometry
                final_mask = self._create_circle_mask(center, radius)
            elif mode == "annular":
                if len(geometry) != 2 or len(geometry[0]) != 2 or len(geometry[1]) != 2:
                    raise ValueError(
                        "For annular mode, geometry must be ((cy, cx), (r_inner, r_outer))"
                    )
                center, radii = geometry
                final_mask = self._create_annular_mask(center, radii)
            else:
                raise ValueError(
                    f"Unknown mode '{mode}'. Supported modes are 'circle' and 'annular'"
                )
        else:
            raise ValueError("Either mask or both mode and geometry must be provided")

        virtual_image = np.sum(self.array * final_mask, axis=(-1, -2))

        virtual_image_dataset = Dataset2d.from_array(
            array=virtual_image,
            name=name,
            origin=self.origin[0:2],
            sampling=self.sampling[0:2],
            units=self.units[0:2],
            signal_units=self.signal_units,
        )

        if attach is True:
            self._virtual_images[name] = virtual_image_dataset
            # Store detector information for regeneration after operations
            self._virtual_detectors[name] = {
                "mask": final_mask.copy() if mask is not None else None,
                "mode": mode,
                "geometry": geometry,
            }

        if show:
            dp_mean_dataset = self.get_dp_mean(attach=False)
            _fig, ax = show_2d(dp_mean_dataset.array, title=f"Mean DP with {name} detector")

            if mask is not None:
                # For custom masks, show the mask contour
                ax.contour(final_mask, colors="red", linewidths=2, alpha=0.8)
                ax.text(
                    0.02,
                    0.98,
                    "Custom Mask",
                    transform=ax.transAxes,
                    color="red",
                    fontsize=12,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
            elif mode == "circle" and geometry is not None:
                center, radius = geometry
                cy, cx = center
                circle = Circle((cx, cy), radius, color="red", fill=True, alpha=0.2)
                ax.add_patch(circle)
                circle = Circle(
                    (cx, cy),
                    radius,
                    color="red",
                    fill=False,
                    alpha=0.8,
                    linewidth=2,
                )
                ax.add_patch(circle)
            elif mode == "annular" and geometry is not None:
                center, radii = geometry
                cy, cx = center
                r_inner, r_outer = radii
                annulus_filled = Wedge(
                    (cx, cy),
                    r_outer,
                    0,
                    360,
                    width=r_outer - r_inner,
                    color="red",
                    fill=True,
                    alpha=0.2,
                )
                ax.add_patch(annulus_filled)
                circle_inner = Circle(
                    (cx, cy), r_inner, color="red", fill=False, linewidth=2, alpha=0.8
                )
                circle_outer = Circle(
                    (cx, cy), r_outer, color="red", fill=False, linewidth=2, alpha=0.8
                )
                ax.add_patch(circle_inner)
                ax.add_patch(circle_outer)
            plt.show()

        return virtual_image_dataset

    def _create_circle_mask(self, center: tuple[float, float], radius: float) -> np.ndarray:
        """
        Create a circular mask for virtual image formation.

        Parameters
        ----------
        center : tuple[float, float]
            Center coordinates (cy, cx) of the circle
        radius : float
            Radius of the circle

        Returns
        -------
        np.ndarray
            Boolean mask with True inside the circle
        """
        cy, cx = center
        dp_shape = self.array.shape[-2:]  # Get diffraction pattern dimensions
        y, x = np.ogrid[: dp_shape[0], : dp_shape[1]]

        # Calculate distance from center
        distance = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

        return distance <= radius

    def _create_annular_mask(
        self, center: tuple[float, float], radii: tuple[float, float]
    ) -> np.ndarray:
        """
        Create an annular (ring-shaped) mask for virtual image formation.

        Parameters
        ----------
        center : tuple[float, float]
            Center coordinates (cy, cx) of the annulus
        radii : tuple[float, float]
            Inner and outer radii (r_inner, r_outer) of the annulus

        Returns
        -------
        np.ndarray
            Boolean mask with True inside the annular region
        """
        cy, cx = center
        r_inner, r_outer = radii
        dp_shape = self.array.shape[-2:]  # Get diffraction pattern dimensions
        y, x = np.ogrid[: dp_shape[0], : dp_shape[1]]

        # Calculate distance from center
        distance = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

        return (distance >= r_inner) & (distance <= r_outer)

    def show_virtual_images(self, figsize: tuple[int, int] | None = None, **kwargs) -> tuple:
        """
        Display all virtual images stored in the dataset using show_2d.

        Parameters
        ----------
        figsize : tuple[int, int] | None, optional
            Figure size in inches. If None, automatically calculated based on number of images
        **kwargs
            Additional keyword arguments passed to show_2d (e.g., cmap, norm, cbar, etc.)

        Returns
        -------
        tuple
            (fig, axs) tuple from matplotlib/show_2d
        """
        if not self.virtual_images:
            print("No virtual images to display. Create virtual images with get_virtual_image().")
            return None, None

        arrays = [vi.array for vi in self.virtual_images.values()]
        titles = list(self.virtual_images.keys())

        n_images = len(arrays)
        if n_images <= 4:
            arrays_organized = arrays
            titles_organized = titles
        else:
            max_cols = 4
            arrays_organized = []
            titles_organized = []

            for i in range(0, n_images, max_cols):
                row_arrays = arrays[i : i + max_cols]
                row_titles = titles[i : i + max_cols]
                arrays_organized.append(row_arrays)
                titles_organized.append(row_titles)

        if figsize is None:
            if n_images == 1:
                figsize = (6, 6)
            elif n_images <= 4:
                figsize = (4 * min(n_images, 2), 4 * ((n_images + 1) // 2))
            else:
                n_rows = (n_images + 3) // 4
                n_cols = min(n_images, 4)
                figsize = (4 * n_cols, 4 * n_rows)

        # Add scalebar to first image if available
        if arrays_organized and hasattr(self, "sampling") and len(self.sampling) >= 2:
            scalebar = ScalebarConfig(
                sampling=self.sampling[0],
                units=self.units[0],
            )
            kwargs.setdefault("scalebar", [scalebar] + [False] * (len(arrays) - 1))
        fig, axs = show_2d(arrays_organized, title=titles_organized, figsize=figsize, **kwargs)

        return fig, axs

    def regenerate_virtual_images(self) -> None:
        """
        Regenerate virtual images from stored detector information.
        This is called after operations like crop, bin, or pad to update virtual images
        to match the new dataset dimensions.
        """
        if not self._virtual_detectors:
            return

        self._virtual_images.clear()

        # Regenerate each virtual image
        for name, detector_info in self._virtual_detectors.items():
            try:
                if detector_info["mode"] is not None and detector_info["geometry"] is not None:
                    # Regenerate from mode and geometry
                    self.get_virtual_image(
                        mode=detector_info["mode"],
                        geometry=detector_info["geometry"],
                        name=name,
                        attach=True,
                        show=False,
                    )
                else:
                    print(
                        f"Warning: Cannot regenerate virtual image '{name}' - insufficient detector information."
                    )
            except Exception as e:
                print(f"Warning: Failed to regenerate virtual image '{name}': {e}")

    def update_virtual_detector(
        self,
        name: str,
        mask: np.ndarray | None = None,
        mode: str | None = None,
        geometry: tuple | None = None,
    ) -> None:
        """
        Update virtual detector information and regenerate the corresponding virtual image.

        This is useful for updating custom masks after dataset operations or changing
        detector parameters.

        Parameters
        ----------
        name : str
            Name of the virtual detector to update
        mask : np.ndarray | None, optional
            New mask for the detector. Should match current diffraction pattern dimensions.
        mode : str | None, optional
            New mode for the detector ("circle" or "annular")
        geometry : tuple | None, optional
            New geometry for the detector
        """
        if name not in self._virtual_detectors:
            raise ValueError(
                f"Virtual detector '{name}' not found. Available detectors: {list(self._virtual_detectors.keys())}"
            )

        # Update detector information
        self._virtual_detectors[name]["mask"] = mask.copy() if mask is not None else None
        self._virtual_detectors[name]["mode"] = mode
        self._virtual_detectors[name]["geometry"] = geometry

        # Regenerate the virtual image
        self.get_virtual_image(
            mask=mask, mode=mode, geometry=geometry, name=name, attach=True, show=False
        )

    def clear_virtual_images(self) -> None:
        """
        Clear virtual images while keeping detector information for regeneration.
        """
        self._virtual_images.clear()

    def clear_all_virtual_data(self) -> None:
        """
        Clear both virtual images and detector information.
        """
        self._virtual_images.clear()
        self._virtual_detectors.clear()

    def copy(self, copy_custom_attributes: bool = True) -> Self:
        """
        Copies Dataset4dstem including virtual images and other custom attributes.

        Parameters
        ----------
        copy_custom_attributes : bool, optional
            If True, copies non-standard attributes. Default is True.

        Returns
        -------
        Dataset4dstem
            A new Dataset4dstem instance with all attributes copied
        """
        # Call parent copy method which will handle custom attributes
        new_dataset = super().copy(copy_custom_attributes)

        return new_dataset

    def _copy_custom_attributes(self, new_dataset) -> None:
        """
        Copy custom attributes specific to Dataset4dstem.

        Parameters
        ----------
        new_dataset
            The new dataset instance to copy attributes to
        """
        # First call parent's generic attribute copying
        super()._copy_custom_attributes(new_dataset)

        # Handle Dataset4dstem-specific attributes that need special treatment
        # Override virtual detector information with custom logic
        new_dataset._virtual_detectors = {}
        for name, detector_info in self._virtual_detectors.items():
            new_dataset._virtual_detectors[name] = {
                "mask": None,  # Custom masks can't be regenerated after operations
                "mode": detector_info["mode"],
                "geometry": detector_info["geometry"],
            }

        # Initialize empty virtual images dict (will be populated by regenerate_virtual_images)
        new_dataset._virtual_images = {}

    def median_filter_masked_pixels(self, mask: np.ndarray, kernel_width: int = 3):
        """
        This function fixes a datacube where the same pixels are consistently
        bad. It requires a mask that identifies all the bad pixels in the dataset.
        Then for each diffraction pattern, a median kernel is applied around each
        bad pixel with the specified width.

        Parameters
        ----------
        mask:
            a boolean mask that specifies the bad pixels in the datacube
        kernel_width (optional):
            specifies the width of the median kernel

        """
        if kernel_width % 2 == 0:
            width_max = kernel_width // 2
            width_min = kernel_width // 2

        else:
            width_max = int(np.ceil(kernel_width / 2))
            width_min = int(np.floor(kernel_width / 2))

        num_bad_pixels_indicies = np.array(np.where(mask))
        for a0 in range(num_bad_pixels_indicies.shape[1]):
            index_x = num_bad_pixels_indicies[0, a0]
            index_y = num_bad_pixels_indicies[1, a0]

            x_min = index_x - width_min
            y_min = index_y - width_min

            x_max = index_x + width_max
            y_max = index_y + width_max

            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0

            if x_max > self.shape[2]:
                x_max = self.shape[2]
            if y_max > self.shape[3]:
                y_max = self.shape[3]

            self.array[:, :, index_x, index_y] = np.median(
                self.array[:, :, x_min:x_max, y_min:y_max], axis=(2, 3)
            )
