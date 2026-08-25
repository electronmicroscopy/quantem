"""Image and canvas preparation for drift alignment."""

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

import quantem.imaging.drift.plot as drift_plot
import quantem.imaging.drift.report as report
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.utils.compound_validators import (
    validate_list_of_dataset2d,
    validate_pad_value,
)
from quantem.core.utils.validators import ensure_valid_array
from quantem.imaging.drift.core import knots as drift_knots


def input_array(value):
    """Return the array carried by a drift-correction input."""
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value
    array = getattr(value, "array", None)
    if isinstance(array, np.ndarray):
        return array
    raise TypeError(
        f"DriftCorrection accepts ndarray, torch.Tensor, or Dataset "
        f"objects; got {type(value).__name__}. To load from disk, call "
        f"Dataset2d.from_file(path) (or Dataset4d.from_file) first."
    )


def prepare_image_collection(
    correction,
    arrays,
    scan_direction_degrees,
    source_datasets=None,
):
    """Preserve scan calibration while preparing 2-D alignment images."""
    correction.imgs = validate_list_of_dataset2d(arrays)
    if source_datasets is not None:
        for image, source in zip(correction.imgs, source_datasets):
            for name in ("origin", "sampling", "units"):
                value = getattr(source, name, None)
                if value is not None:
                    setattr(image, name, value[:2])
            signal_units = getattr(source, "signal_units", None)
            if signal_units is not None:
                image.signal_units = signal_units
            metadata = getattr(source, "metadata", None)
            if isinstance(metadata, dict):
                image.metadata.update(metadata)
    correction.scan_direction_degrees = ensure_valid_array(
        scan_direction_degrees, ndim=1
    )


def prepare_inputs(
    correction,
    datasets,
    scan_direction_degrees,
    alignment_image,
):
    """Prepare image, reference, or 4D-STEM inputs for one correction state."""
    if len(datasets) < 2:
        raise TypeError(
            f"DriftCorrection requires at least 2 datasets, got {len(datasets)}"
        )

    if scan_direction_degrees is None:
        found = [
            getattr(dataset, "metadata", {}).get("scan_rotation_deg")
            if isinstance(getattr(dataset, "metadata", None), dict)
            else None
            for dataset in datasets
        ]
        if not all(angle is not None for angle in found):
            raise TypeError(
                "scan_direction_degrees is required: the inputs carry no "
                "scan-angle metadata. Load Velox files with "
                "em.imaging.read_emd(path) (which stamps "
                "metadata['scan_rotation_deg']) or pass the angles "
                "explicitly, e.g. scan_direction_degrees=(0, 90)."
            )
        scan_direction_degrees = found
    if np.isscalar(scan_direction_degrees):
        angles = [float(scan_direction_degrees)] * len(datasets)
    else:
        angles = [float(angle) for angle in scan_direction_degrees]
        if len(angles) != len(datasets):
            raise ValueError(
                f"scan_direction_degrees length ({len(angles)}) must match "
                f"number of datasets ({len(datasets)})"
            )

    arrays = [input_array(dataset) for dataset in datasets]
    dimensions = [array.ndim for array in arrays]
    for index, ndim in enumerate(dimensions):
        if ndim < 2:
            raise TypeError(f"dataset {index} must be ≥2-D, got ndim={ndim}")

    if len(arrays) == 2:
        first, second = arrays
        first_ndim, second_ndim = dimensions
        if first_ndim == 2 and (second_ndim >= 3 or angles[0] == angles[1]):
            if first.shape != second.shape[:2]:
                raise ValueError(
                    f"reference shape {first.shape} must match the leading "
                    f"two axes of drifted (got {second.shape[:2]})"
                )
            if alignment_image is not None:
                moving = np.asarray(alignment_image)
            elif second_ndim == 2:
                moving = second
            else:
                from quantem.imaging.drift.fourdstem import integrate_virtual_detector

                moving = integrate_virtual_detector(second)
            prepare_image_collection(correction, [first, moving], angles)
            correction._reference_mode = True
            correction._datasets = [None, second]
            correction._built_from_datasets = True
            return

        if first_ndim >= 4 and second_ndim >= 4:
            from quantem.imaging.drift.fourdstem import integrate_virtual_detector

            prepare_image_collection(
                correction,
                [
                    integrate_virtual_detector(first),
                    integrate_virtual_detector(second),
                ],
                angles,
                datasets,
            )
            correction._datasets = [first, second]
            correction._built_from_datasets = True
            return

        sources = datasets if first_ndim == second_ndim == 2 else None
        prepare_image_collection(correction, arrays, angles, sources)
        return

    prepare_image_collection(correction, arrays, angles, datasets)


def validate_downsample(value: int) -> int:
    """Validate a computational downsampling factor."""
    factor = int(value)
    if factor < 1:
        raise ValueError(f"downsample must be >= 1, got {value!r}.")
    return factor


def average_downsample_2d(array: NDArray, factor: int) -> NDArray:
    """Average-pool a two-dimensional image by an integer factor."""
    image = np.asarray(array)
    if factor == 1:
        return np.ascontiguousarray(image)
    if image.ndim != 2:
        raise ValueError(
            f"downsample currently supports 2-D images, got ndim={image.ndim}."
        )
    rows, cols = image.shape
    if rows % factor or cols % factor:
        raise ValueError(
            f"downsample={factor} requires image dimensions divisible by "
            f"{factor}, got {image.shape}."
        )
    downsampled = image.reshape(
        rows // factor,
        factor,
        cols // factor,
        factor,
    ).mean(axis=(1, 3))
    dtype = (
        image.dtype
        if np.issubdtype(image.dtype, np.floating)
        else np.float32
    )
    return np.ascontiguousarray(downsampled.astype(dtype, copy=False))


def match_scan_shapes(first, second, *, verbose: bool = False):
    """Mean-bin an integer-resolution pair to the same physical scan grid.

    Orthogonal scans of the same field can be saved at different integer
    sampling densities. Mean-binning the finer scan preserves the measured
    field and gives the correction one shared pixel grid; non-integer shape
    mismatches cannot make that physical guarantee and remain an error.
    """
    if first.shape == second.shape:
        return first, second

    if np.prod(first.shape) >= np.prod(second.shape):
        fine_index, fine, coarse = 0, first, second
    else:
        fine_index, fine, coarse = 1, second, first
    factors = tuple(
        fine_size // coarse_size
        for fine_size, coarse_size in zip(fine.shape, coarse.shape)
    )
    if not all(
        factor >= 2 and fine_size == factor * coarse_size
        for factor, fine_size, coarse_size in zip(
            factors, fine.shape, coarse.shape
        )
    ) or factors[0] != factors[1]:
        raise ValueError(
            "shape mismatch cannot be reconciled by equal integer mean "
            f"binning: {tuple(first.shape)} != {tuple(second.shape)}. "
            "Confirm that both files cover the same field of view."
        )

    factor = factors[0]
    rows, columns = coarse.shape
    array = np.asarray(fine.array).reshape(
        rows, factor, columns, factor
    ).mean(axis=(1, 3))
    binned = type(fine).from_array(
        np.asarray(array, dtype=np.float32),
        name=fine.name,
        origin=fine.origin,
        sampling=np.asarray(fine.sampling) * factor,
        units=fine.units,
        signal_units=fine.signal_units,
    )
    binned.metadata.update(fine.metadata)
    if hasattr(fine, "file_path"):
        binned.file_path = fine.file_path
    if verbose:
        print(
            f"mean-binned scan {fine_index} by {factor}x: "
            f"{tuple(fine.shape)} -> {tuple(binned.shape)}"
        )
    return (binned, second) if fine_index == 0 else (first, binned)


def reference_downsample(
    reference_scan_shape: tuple[int, int],
    target_scan_shape: tuple[int, int],
    *,
    reference_sampling: NDArray | None = None,
    target_sampling: NDArray | None = None,
) -> int:
    """Return the averaging needed to put a reference on the target grid.

    Sampling metadata decides when available. Without it, an exact isotropic
    integer shape ratio is treated as a coarser acquisition grid, as in a
    2048-pixel HAADF reference paired with a 1024-pixel spectrum image.
    """
    source_rows, source_columns = map(int, reference_scan_shape)
    target_rows, target_columns = map(int, target_scan_shape)
    exact_integer_scale = (
        source_rows % target_rows == 0
        and source_columns % target_columns == 0
        and source_rows // target_rows == source_columns // target_columns
    )
    downsample = source_rows // target_rows if exact_integer_scale else 1
    if downsample > 1 and reference_sampling is not None and target_sampling is not None:
        reference_step = np.asarray(reference_sampling, dtype=float)[:2]
        target_step = np.asarray(target_sampling, dtype=float)[:2]
        if np.all(np.isfinite(reference_step)) and np.all(np.isfinite(target_step)):
            same_grid = np.allclose(
                target_step,
                reference_step,
                rtol=0.05,
                atol=0.0,
            )
            coarser_grid = np.allclose(
                target_step,
                reference_step * downsample,
                rtol=0.05,
                atol=0.0,
            )
            if same_grid:
                return 1
            if not coarser_grid:
                return 1
    return downsample


def match_reference_image(
    image: NDArray,
    reference_scan_shape: tuple[int, int],
    target_scan_shape: tuple[int, int],
) -> NDArray:
    """Remove solver padding and center-crop a reference to a target scan."""
    array = np.asarray(image)
    source_rows, source_columns = map(int, reference_scan_shape)
    target_rows, target_columns = map(int, target_scan_shape)
    if array.shape[0] < source_rows or array.shape[1] < source_columns:
        raise ValueError(
            "corrected reference is smaller than its acquired scan shape: "
            f"corrected={array.shape[:2]}, acquired={reference_scan_shape}."
        )

    if target_rows > source_rows or target_columns > source_columns:
        raise ValueError(
            "target scan is larger than the solved reference grid: "
            f"reference={reference_scan_shape}, target={target_scan_shape}."
        )
    native_row = (array.shape[0] - source_rows) // 2
    native_column = (array.shape[1] - source_columns) // 2
    native = array[
        native_row : native_row + source_rows,
        native_column : native_column + source_columns,
    ]
    row_start = (native.shape[0] - target_rows) // 2
    column_start = (native.shape[1] - target_columns) // 2
    return np.ascontiguousarray(native[
        row_start : row_start + target_rows,
        column_start : column_start + target_columns,
    ])


def automatic_alignment_normalization(
    images: Sequence[Dataset2d],
) -> tuple[bool, str]:
    """Choose min-max conditioning for images with a large DC background."""
    background_to_contrast = []
    for image in images:
        array = np.asarray(image.array, dtype=np.float32)
        contrast = float(np.std(array))
        background = abs(float(np.mean(array)))
        background_to_contrast.append(
            background / max(contrast, np.finfo(np.float32).eps)
        )
    ratio = max(background_to_contrast)
    enabled = ratio >= 5.0
    reason = (
        "large_dc_background"
        if enabled
        else "same_detector_well_conditioned"
    )
    return enabled, reason


def scale_coordinate_metadata(
    values,
    factor: int,
    *,
    offset: NDArray | None = None,
):
    """Scale the first two coordinate entries while preserving length."""
    scaled = np.asarray(values, dtype=float).copy()
    if scaled.size < 2:
        scaled = np.resize(scaled, 2)
    if offset is not None:
        scaled[:2] += offset[:2]
    else:
        scaled[:2] *= factor
    return scaled


def automatic_downsample_factor(shape: tuple[int, int]) -> int:
    """Return the largest safe average-downsampling factor, capped at eight."""
    height, width = shape
    for factor in (8, 4, 2):
        if height % factor == 0 and width % factor == 0:
            return factor
    return 1


def resolve_downsample(
    value: int | str,
    shape: tuple[int, int],
) -> int:
    """Resolve the automatic affine-search downsampling factor."""
    if value == "auto":
        return automatic_downsample_factor(shape)
    if isinstance(value, str):
        raise ValueError(
            "downsample must be 'auto' or a positive integer, "
            f"got {value!r}."
        )
    factor = validate_downsample(value)
    if any(size % factor for size in shape):
        raise ValueError(
            f"downsample={factor} requires both image dimensions "
            f"to be divisible by {factor}, got {shape}. Choose a divisor "
            "such as 1, 2, 4, or 8."
        )
    return factor


def minimum_affine_padding_fraction(
    image_shape: tuple[int, int],
    scan_direction_degrees: Sequence[float],
    max_drift_rate: float = 0.25,
    translation_margin: float = 0.0,
) -> float:
    """Return the smallest canvas fraction covering the affine envelope."""
    rows, cols = (int(value) for value in image_shape)
    if rows < 1 or cols < 1:
        raise ValueError(f"image_shape must be positive, got {image_shape}.")
    rate = abs(float(max_drift_rate))
    translation = float(translation_margin)
    if not np.isfinite(rate):
        raise ValueError(
            f"max_drift_rate must be finite, got {max_drift_rate!r}."
        )
    if not np.isfinite(translation) or translation < 0:
        raise ValueError(
            "translation_margin must be finite and non-negative, "
            f"got {translation_margin!r}."
        )

    half_rows = (rows - 1) / 2.0
    half_cols = (cols - 1) / 2.0
    max_row_extent = 0.0
    max_col_extent = 0.0
    for angle_degrees in scan_direction_degrees:
        angle = np.deg2rad(float(angle_degrees))
        fast = np.asarray((np.sin(angle), np.cos(angle)))
        slow = np.asarray((np.cos(angle), -np.sin(angle)))
        extent = np.abs(slow) * half_rows + np.abs(fast) * half_cols
        max_row_extent = max(max_row_extent, float(extent[0]))
        max_col_extent = max(max_col_extent, float(extent[1]))

    affine_margin = rate * half_rows

    def even_ceiling(value: float) -> int:
        integer = int(np.ceil(value))
        return integer if integer % 2 == 0 else integer + 1

    canvas_rows = even_ceiling(
        2.0 * (max_row_extent + affine_margin + translation) + 1.0
    )
    canvas_cols = even_ceiling(
        2.0 * (max_col_extent + affine_margin + translation) + 1.0
    )
    return max(
        0.0,
        canvas_rows / rows - 1.0,
        canvas_cols / cols - 1.0,
    )


def apply_downsample(correction, factor: int) -> None:
    """Average-downsample images and update their coordinate metadata."""
    original_records = []
    downsampled_images = []
    for image_index, image in enumerate(correction.imgs):
        original_array = np.asarray(image.array)
        original_sampling = np.asarray(image.sampling, dtype=float).copy()
        original_origin = np.asarray(image.origin, dtype=float).copy()
        original_units = list(image.units)
        original_shape = tuple(int(value) for value in original_array.shape[:2])
        downsampled = average_downsample_2d(original_array, factor)
        sampling = scale_coordinate_metadata(original_sampling, factor)
        origin_offset = (factor - 1) * original_sampling[:2] / 2.0
        origin = scale_coordinate_metadata(
            original_origin,
            factor,
            offset=origin_offset,
        )
        metadata = dict(getattr(image, "metadata", {}))
        metadata.update(
            {
                "downsample": factor,
                "downsample_method": "average",
                "downsample_original_shape": list(original_shape),
                "downsample_original_sampling": original_sampling.tolist(),
                "downsample_original_origin": original_origin.tolist(),
                "downsampled_shape": list(downsampled.shape[:2]),
                "sampling_is_downsampled": True,
            }
        )
        image_dataset = Dataset2d.from_array(
            downsampled,
            name=f"{image.name} ({factor}x downsample)",
            origin=origin,
            sampling=sampling,
            units=original_units,
            signal_units=image.signal_units,
        )
        image_dataset.metadata.update(metadata)
        downsampled_images.append(image_dataset)
        original_records.append(
            {
                "image_index": image_index,
                "shape": list(original_shape),
                "sampling": original_sampling.tolist(),
                "origin": original_origin.tolist(),
                "units": original_units,
            }
        )
    correction.imgs = downsampled_images
    correction.downsample = factor
    correction.downsample_method = "average"
    correction.downsample_metadata = {
        "factor": factor,
        "method": "average",
        "original_images": original_records,
        "downsampled_shape": list(correction.imgs[0].shape[:2]),
    }


def preprocess(
    self,
    *,
    padding_fraction: float | str = "auto",
    padding_value: float | str | list[float] = "median",
    smoothing_sigma: float = 0.5,
    num_knots: int = 1,
    normalize: bool = False,
    downsample: int = 1,
    verbose: bool = True,
    show_combined: bool = False,
    show_scans: bool = False,
    show_knots: bool = True,
    show_knot_plot: bool = False,
):
    """Build the shared scanline canvas before drift correction.

    Affine correction calls this automatically. Use it directly only when a
    publication requires a fixed padding, normalization, or knot layout.

    Parameters
    ----------
    padding_fraction : float or "auto", default "auto"
        Fractional canvas expansion. ``"auto"`` covers the scan rotations and
        affine search envelope.
    padding_value : float, str, or list of float, default "median"
        Intensity outside each measured scan footprint.
    smoothing_sigma : float, default 0.5
        Gaussian smoothing in pixels after scanline interpolation.
    num_knots : int, default 1
        Knots per scanline for a fixed publication setup. One knot represents
        affine drift. For routine non-rigid correction, prefer
        ``correct_nonrigid(num_knots=...)`` so preprocessing stays automatic.
    normalize : bool, default False
        Scale each scan to ``[0, 1]`` for mixed-detector comparisons.
    downsample : int, default 1
        Computational downsampling for the 2D alignment images.

    Returns
    -------
    object
        The same correction object for method chaining.

    Examples
    --------
    >>> drift.preprocess(padding_fraction=0.25, show_combined=True)
    """
    downsample = validate_downsample(downsample)
    if downsample > 1:
        if getattr(self, "_datasets", None) is not None:
            raise NotImplementedError(
                "preprocess(downsample>1) currently supports 2-D "
                "image-collection solves only. Reference, EDS/EELS, and "
                "4D-STEM dataset correction need full-resolution fields."
            )
        if hasattr(self, "_initial_knots"):
            raise RuntimeError(
                "downsample changes the image grid. Create a "
                "new DriftCorrection object before changing it."
            )
        apply_downsample(self, downsample)
        if verbose:
            original = self.downsample_metadata["original_images"][0]
            units = ", ".join(str(unit) for unit in original["units"][:2])
            print(
                "preprocess: downsample="
                f"{downsample} applies computational average downsampling "
                f"{tuple(original['shape'])} -> "
                f"{tuple(self.imgs[0].shape[:2])}; "
                f"sampling {original['sampling'][:2]} -> "
                f"{np.asarray(self.imgs[0].sampling, dtype=float)[:2].tolist()} "
                f"{units}. This is not acquisition binning; scale metadata "
                "was updated for display and scale bars."
            )
    else:
        self.downsample = 1
        self.downsample_method = "none"
        self.downsample_metadata = {
            "factor": 1,
            "method": "none",
            "original_images": [
                {
                    "image_index": image_index,
                    "shape": list(image.shape[:2]),
                    "sampling": np.asarray(
                        image.sampling, dtype=float
                    ).tolist(),
                    "origin": np.asarray(
                        image.origin, dtype=float
                    ).tolist(),
                    "units": list(image.units),
                }
                for image_index, image in enumerate(self.imgs)
            ],
            "downsampled_shape": list(self.imgs[0].shape[:2]),
        }

    self._normalized = bool(normalize)
    if normalize:
        for image in self.imgs:
            array = image.array.astype(np.float32)
            low, high = array.min(), array.max()
            image.array = (array - low) / (high - low + 1e-8)
    self.pad_value = validate_pad_value(padding_value, self.imgs)
    self.kde_sigma = float(smoothing_sigma)
    number_knots = int(num_knots)
    if number_knots < 1:
        raise ValueError(f"num_knots must be >= 1 (got {num_knots}).")
    self.number_knots = number_knots

    unique_directions = {
        round(direction, 6) for direction in self.scan_direction_degrees
    }
    if len(unique_directions) > 1:
        for image_index, image in enumerate(self.imgs):
            if image.shape[0] != image.shape[1]:
                raise ValueError(
                    "Multi-direction scan collection require square images, "
                    f"but image {image_index} is {image.shape}. Either crop "
                    "to square or use a single scan direction."
                )
    self.scan_direction = np.deg2rad(self.scan_direction_degrees)
    self.scan_fast = np.stack(
        [np.sin(self.scan_direction), np.cos(self.scan_direction)], axis=1
    )
    self.scan_slow = np.stack(
        [np.cos(self.scan_direction), -np.sin(self.scan_direction)], axis=1
    )

    translation_margin = 0.0
    if padding_fraction == "auto":
        translation_margin = (
            min(self.imgs[0].shape[:2]) * 0.125
            if self._built_from_datasets and not self._reference_mode
            else 0.0
        )
        self.pad_fraction = minimum_affine_padding_fraction(
            tuple(int(value) for value in self.imgs[0].shape[:2]),
            self.scan_direction_degrees,
            translation_margin=translation_margin,
        )
        padding_mode = "auto"
    elif isinstance(padding_fraction, str):
        raise ValueError(
            "padding_fraction must be 'auto' or a non-negative float, "
            f"got {padding_fraction!r}."
        )
    else:
        self.pad_fraction = float(padding_fraction)
        if not np.isfinite(self.pad_fraction) or self.pad_fraction < 0:
            raise ValueError(
                "padding_fraction must be a finite, non-negative value, "
                f"got {padding_fraction!r}."
            )
        padding_mode = "explicit"
    self.shape = (
        len(self.imgs),
        int(
            np.round(
                self.imgs[0].shape[0] * (1 + self.pad_fraction) / 2
            )
            * 2
        ),
        int(
            np.round(
                self.imgs[0].shape[1] * (1 + self.pad_fraction) / 2
            )
            * 2
        ),
    )
    self.preprocess_info = {
        "padding_mode": padding_mode,
        "padding_fraction": self.pad_fraction,
        "canvas_shape": list(self.shape[1:]),
        "normalize": self._normalized,
        "smoothing_sigma": self.kde_sigma,
        "num_knots": self.number_knots,
        "translation_margin": (
            translation_margin if padding_mode == "auto" else None
        ),
    }
    self.knots = [
        torch.tensor(
            drift_knots.initialize_scanline_knots(
                input_shape=self.imgs[image_index].shape,
                output_shape=self.shape[1:],
                scan_fast=self.scan_fast[image_index],
                scan_slow=self.scan_slow[image_index],
                number_knots=self.number_knots,
            ),
            dtype=self._dtype,
            device=self._device,
        )
        for image_index in range(self.shape[0])
    ]
    self.u_per_image = [
        np.linspace(0, 1, self.imgs[index].shape[1])
        for index in range(self.shape[0])
    ]
    device = self._device
    dtype = self._dtype
    self.imgs_t = [
        torch.tensor(self.imgs[index].array, dtype=dtype, device=device)
        for index in range(self.shape[0])
    ]
    self.scan_fast_t = [
        torch.tensor(self.scan_fast[index], dtype=dtype, device=device)
        for index in range(self.shape[0])
    ]
    self.scan_slow_t = [
        torch.tensor(self.scan_slow[index], dtype=dtype, device=device)
        for index in range(self.shape[0])
    ]
    self.imgs_warped = Dataset3d.from_shape(self.shape)
    canvas_shape = (self.shape[1], self.shape[2])
    warped_t = torch.zeros(
        self.shape[0], *canvas_shape, dtype=dtype, device=device
    )
    for image_index in range(self.shape[0]):
        warped, _ = drift_knots.interpolator(self, image_index).warp_to_canvas(
            self.imgs_t[image_index],
            canvas_shape,
            self.kde_sigma,
            self.pad_value[image_index],
        )
        warped_t[image_index] = warped
        self.imgs_warped.array[image_index] = warped.cpu().numpy()
    self._initial_knots = [knot.clone() for knot in self.knots]
    report.record_error(self, 0, warped_t)
    drift_plot.show_after_step(
        self,
        "initial",
        show_combined=show_combined,
        show_scans=show_scans,
        show_knots=show_knots,
    )
    if show_knot_plot:
        self.plot_knots()
    return self
