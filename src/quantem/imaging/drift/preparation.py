
import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.utils.compound_validators import (
    validate_pad_value,
)
from quantem.core.utils.imaging_utils import (
    cross_correlation_shift,
)
from quantem.imaging.drift import diagnostics
from quantem.imaging.drift.core.knots import (
    DriftInterpolator,
    bilinear_kde_batch,
    transform_coordinates_single_knot,
)


def validate_downsample(value: int) -> int:
    """Validate one integer computational downsampling factor."""
    factor = int(value)
    if factor < 1 or factor != value:
        raise ValueError(f"downsample must be a positive integer, got {value!r}.")
    return factor


def average_downsample_2d(array: NDArray, factor: int) -> NDArray:
    """Average-pool a two-dimensional image by an exact integer factor."""
    image = np.asarray(array)
    factor = validate_downsample(factor)
    if factor == 1:
        return np.ascontiguousarray(image)
    if image.ndim != 2:
        raise ValueError(
            f"downsample currently supports 2-D images, got ndim={image.ndim}."
        )
    rows, columns = image.shape
    if rows % factor or columns % factor:
        raise ValueError(
            f"downsample={factor} requires image dimensions divisible by "
            f"{factor}, got {image.shape}."
        )
    downsampled = image.reshape(
        rows // factor,
        factor,
        columns // factor,
        factor,
    ).mean(axis=(1, 3))
    dtype = image.dtype if np.issubdtype(image.dtype, np.floating) else np.float32
    return np.ascontiguousarray(downsampled.astype(dtype, copy=False))


def automatic_downsample_factor(shape: tuple[int, int]) -> int:
    """Return the largest exact average-pooling factor, capped at eight."""
    rows, columns = shape
    for factor in (8, 4, 2):
        if rows % factor == 0 and columns % factor == 0:
            return factor
    return 1


def resolve_downsample(value: int | str, shape: tuple[int, int]) -> int:
    """Resolve ``"auto"`` or validate one explicit exact divisor."""
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
            f"downsample={factor} requires both image dimensions to be "
            f"divisible by {factor}, got {shape}."
        )
    return factor


def _scaled_coordinate_metadata(
    values,
    factor: int,
    *,
    offset: NDArray | None = None,
) -> NDArray:
    """Scale the first two coordinate entries while preserving their length."""
    scaled = np.asarray(values, dtype=float).copy()
    if scaled.size < 2:
        scaled = np.resize(scaled, 2)
    if offset is None:
        scaled[:2] *= factor
    else:
        scaled[:2] += offset[:2]
    return scaled


def apply_downsample(correction, factor: int) -> None:
    """Average-downsample alignment images and preserve physical coordinates."""
    factor = resolve_downsample(factor, tuple(correction.images[0].shape))
    original_records = []
    downsampled_images = []
    for image_index, image in enumerate(correction.images):
        original_array = np.asarray(image.array)
        original_sampling = np.asarray(image.sampling, dtype=float).copy()
        original_origin = np.asarray(image.origin, dtype=float).copy()
        original_shape = tuple(int(value) for value in original_array.shape)
        downsampled = average_downsample_2d(original_array, factor)
        sampling = _scaled_coordinate_metadata(original_sampling, factor)
        origin_offset = (factor - 1) * original_sampling[:2] / 2.0
        origin = _scaled_coordinate_metadata(
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
                "downsampled_shape": list(downsampled.shape),
                "sampling_is_downsampled": True,
            }
        )
        downsampled_image = Dataset2d.from_array(
            downsampled,
            name=f"{image.name} ({factor}x downsample)",
            origin=origin,
            sampling=sampling,
            units=list(image.units),
            signal_units=image.signal_units,
        )
        downsampled_image.metadata.update(metadata)
        downsampled_images.append(downsampled_image)
        original_records.append(
            {
                "image_index": image_index,
                "shape": list(original_shape),
                "sampling": original_sampling.tolist(),
                "origin": original_origin.tolist(),
                "units": list(image.units),
            }
        )
    correction.images = downsampled_images
    correction.downsample = factor
    correction.downsample_method = "average"
    correction.downsample_metadata = {
        "factor": factor,
        "method": "average",
        "original_images": original_records,
        "downsampled_shape": list(correction.images[0].shape),
    }


def preprocess(
    self,
    pad_fraction: float = 0.25,
    pad_value: float | str | list[float] = "median",
    kde_sigma: float = 0.5,
    number_knots: int = 1,
    downsample: int | str = 1,
    show_merged: bool = False,
    show_images: bool = False,
    show_knots: bool = True,
    **kwargs,
):
    """Prepare images for drift correction by building the scanline model.

    Computes scan direction vectors, initializes Bezier knots that map
    each scanline onto a padded canvas, and generates the initial warped
    images. This must be called before any alignment step.

    Without preprocessing, there is no spatial model connecting the raw
    images to the shared canvas - alignment methods would have no
    coordinates to optimize.

    Parameters
    ----------
    pad_fraction : float
        Fraction of the image size to add as padding around the canvas.
        Larger values give more room for drift but use more memory.
        ``pad_fraction=0.25`` adds 25% on each side.
    pad_value : float, str, or list[float]
        Fill value for pixels outside the image footprint. Can be
        ``'median'``, ``'mean'``, ``'min'``, ``'max'``, a quantile
        (e.g. ``0.25``), or a per-image list of floats.
    kde_sigma : float
        Gaussian smoothing sigma (in pixels) applied after bilinear
        scatter. Smooths the warped images to reduce scatter noise.
    number_knots : int
        Number of Bezier knots per scanline. Use ``1`` (recommended)
        for linear drift correction. Higher values allow per-scanline
        curvature but are slower and rarely needed.
    downsample : int or "auto", default 1
        Exact average-pooling factor for the 2-D alignment images. Sampling
        and pixel-center origin are updated so physical coordinates remain
        calibrated. This is computational downsampling, not detector binning.
    show_merged : bool
        Display the merged (averaged) warped images after preprocessing.
    show_images : bool
        Display each individual warped image after preprocessing.
    show_knots : bool
        Overlay knot positions on displayed images.
    **kwargs
        Additional keyword arguments passed to plotting functions.

    Returns
    -------
    Self
        For method chaining: ``drift.preprocess().align_affine()``.

    Examples
    --------
    >>> drift = DriftCorrection.from_data(
    ...     images=[im0, im1], scan_direction_degrees=[0, 90])
    >>> drift.preprocess(pad_fraction=0.25, kde_sigma=0.5, number_knots=1)
    """
    factor = resolve_downsample(downsample, tuple(self.images[0].shape))
    if factor > 1:
        if getattr(self, "_datasets", None) is not None:
            raise ValueError(
                "preprocess(downsample>1) is only for 2-D alignment images. "
                "Keep multidimensional data native and use the affine search "
                "pyramid instead."
            )
        if hasattr(self, "_initial_knots"):
            raise RuntimeError(
                "downsample changes the alignment grid. Create a new "
                "DriftCorrection before choosing a different factor."
            )
        apply_downsample(self, factor)
    else:
        self.downsample = 1
        self.downsample_method = "none"
        self.downsample_metadata = {
            "factor": 1,
            "method": "none",
            "original_images": [
                {
                    "image_index": index,
                    "shape": list(image.shape),
                    "sampling": np.asarray(image.sampling, dtype=float).tolist(),
                    "origin": np.asarray(image.origin, dtype=float).tolist(),
                    "units": list(image.units),
                }
                for index, image in enumerate(self.images)
            ],
            "downsampled_shape": list(self.images[0].shape),
        }
    self.pad_fraction = float(pad_fraction)
    self.pad_value = validate_pad_value(pad_value, self.images)
    self.kde_sigma = float(kde_sigma)
    self.number_knots = int(number_knots)
    self.scan_direction = np.deg2rad(self.scan_direction_degrees)
    self.scan_fast = np.stack(
        [np.sin(-self.scan_direction), np.cos(-self.scan_direction)], axis=1)
    self.scan_slow = np.stack(
        [np.cos(-self.scan_direction), -np.sin(-self.scan_direction)], axis=1)
    self.shape = (
        len(self.images),
        int(np.round(self.images[0].shape[0] * (1 + self.pad_fraction) / 2) * 2),
        int(np.round(self.images[1].shape[1] * (1 + self.pad_fraction) / 2) * 2),
    )
    # Initialize knots - each image's scanlines mapped to the padded canvas
    self.knots = []
    for img_idx in range(self.shape[0]):
        shape = self.images[img_idx].shape
        v_slow = np.linspace(-(shape[0] - 1) / 2, (shape[0] - 1) / 2, shape[0])
        u_fast = np.linspace(-(shape[1] - 1) / 2, (shape[1] - 1) / 2, self.number_knots)
        row_knots = ((self.shape[1] - 1) / 2
                     + u_fast[None, :] * self.scan_fast[img_idx, 0]
                     + v_slow[:, None] * self.scan_slow[img_idx, 0])
        col_knots = ((self.shape[2] - 1) / 2
                     + u_fast[None, :] * self.scan_fast[img_idx, 1]
                     + v_slow[:, None] * self.scan_slow[img_idx, 1])
        self.knots.append(np.stack([row_knots, col_knots], axis=0))
    self._initial_knots = [knots.copy() for knots in self.knots]
    self.interpolator = [
        DriftInterpolator(
            input_shape=self.images[i].shape,
            output_shape=self.shape[1:],
            scan_fast=self.scan_fast[i],
            scan_slow=self.scan_slow[i],
            pad_value=self.pad_value[i],
            kde_sigma=self.kde_sigma,
        )
        for i in range(self.shape[0])
    ]
    # Cache source data on GPU and generate initial warped images
    device = self._device
    dtype = self._dtype
    self.images_t = [
        torch.tensor(self.images[i].array, dtype=dtype, device=device)
        for i in range(self.shape[0])
    ]
    self.scan_fast_t = [
        torch.tensor(self.scan_fast[i], dtype=dtype, device=device)
        for i in range(self.shape[0])
    ]
    self.images_warped = Dataset3d.from_shape(self.shape)
    self.weights_warped = Dataset3d.from_shape(self.shape)
    canvas_shape = (self.shape[1], self.shape[2])
    warped_t = torch.zeros(self.shape[0], *canvas_shape, dtype=dtype, device=device)
    for img_idx in range(self.shape[0]):
        knots_t = torch.tensor(self.knots[img_idx], dtype=dtype, device=device)
        row_t, col_t = transform_coordinates_single_knot(
            knots_t, self.scan_fast_t[img_idx], self.images[img_idx].shape)
        warped, weights = bilinear_kde_batch(
            row_t[None], col_t[None], self.images_t[img_idx], canvas_shape,
            self.kde_sigma, self.pad_value[img_idx])
        warped_t[img_idx] = warped[0]
        self.images_warped.array[img_idx] = warped[0].cpu().numpy()
        self.weights_warped.array[img_idx] = weights[0].cpu().numpy()
    diagnostics._record_stage(self, "initial")
    self.calculate_error(0, _warped_t=warped_t)
    kwargs.pop("title", None)
    if show_merged:
        self.plot_merged_images(show_knots=show_knots, title="Merged: initial", **kwargs)
    if show_images:
        self.plot_transformed_images(
            show_knots=show_knots,
            title=[f"Image {i}: initial" for i in range(self.shape[0])],
            **kwargs,
        )
    return self


def align_translation(
    self,
    upsample_factor: int = 8,
    min_image_shift: float | None = None,
    max_image_shift: float = 32,
    show_merged: bool = True,
    show_images: bool = False,
    show_knots: bool = True,
    **kwargs,
):
    """
    Solve for the translation between all images in DriftCorrection.images_warped
    """
    dxy = np.zeros((self.shape[0], 2))
    F_ref = np.fft.fft2(self.images_warped.array[0])
    for ind in range(1, self.shape[0]):
        shifts, image_shift = cross_correlation_shift(
            F_ref,
            np.fft.fft2(self.images_warped.array[ind]),
            upsample_factor=upsample_factor,
            max_shift=max_image_shift,
            fft_input=True,
            fft_output=True,
            return_shifted_image=True,
        )
        dxy[ind, :] = shifts
        F_ref = F_ref * ind / (ind + 1) + image_shift / (ind + 1)
    dxy -= np.mean(dxy, axis=0)
    if min_image_shift is not None:
        if np.linalg.norm(dxy[ind]) < min_image_shift:
            dxy[ind] = 0.0
    for ind in range(self.shape[0]):
        self.knots[ind][0] += dxy[ind, 0]
        self.knots[ind][1] += dxy[ind, 1]
    for ind in range(self.shape[0]):
        self.images_warped.array[ind], self.weights_warped.array[ind] = self.interpolator[
            ind
        ].warp_image(
            self.images[ind].array,
            self.knots[ind],
        )
    diagnostics._record_stage(self, "translation")
    kwargs.pop("title", None)
    if show_merged:
        self.plot_merged_images(show_knots=show_knots, title="Merged: translation", **kwargs)
    if show_images:
        self.plot_transformed_images(
            show_knots=show_knots,
            title=[f"Image {i}: translation" for i in range(self.shape[0])],
            **kwargs,
        )
    return self
