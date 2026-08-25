
import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.utils.imaging_utils import (
    fourier_cropping,
)
from quantem.core.visualization import show_2d
from quantem.imaging.drift.core.knots import (
    bilinear_kde_batch,
    transform_coordinates_single_knot,
)


@torch.inference_mode()
def apply_correction(self, spectrum_image: Dataset3d) -> Dataset3d:
    """Apply one learned spatial field to every spectrum-image channel.

    Parameters
    ----------
    spectrum_image : Dataset3d
        Spectrum image with axes ``(scan_row, scan_col, energy)``.

    Returns
    -------
    Dataset3d
        Corrected spectrum image with unchanged calibration, units, metadata,
        and spectral-axis ordering.

    Examples
    --------
    >>> corrected = drift.apply_correction(spectrum_image)
    >>> corrected.shape == spectrum_image.shape
    True
    """
    if not getattr(self, "_reference_mode", False):
        raise RuntimeError(
            "apply_correction requires DriftCorrection.from_reference(...)."
        )
    if not hasattr(self, "_initial_knots"):
        raise RuntimeError(
            "No drift field found. Call preprocess() and align_affine() first."
        )
    if "affine" not in getattr(self, "_diagnostic_knots", {}):
        raise RuntimeError(
            "No fitted reference field found. Call align_affine() before "
            "applying the correction."
        )
    if not isinstance(spectrum_image, Dataset3d):
        raise TypeError("spectrum_image must be a Dataset3d.")
    dataset = spectrum_image
    scan_rows, scan_cols, num_channels = dataset.shape
    if (scan_rows, scan_cols) != tuple(self.images[1].shape):
        raise ValueError(
            "The spectrum-image scan axes must match the fitted alignment image; "
            f"got {(scan_rows, scan_cols)} and {self.images[1].shape}."
        )
    if self.knots[1].shape[-1] != 1:
        raise ValueError(
            "Spectrum-image correction currently requires number_knots=1."
        )

    device = self._device
    delta_canvas_t = torch.as_tensor(
        self.knots[1] - self._initial_knots[1],
        dtype=torch.float32,
        device=device,
    )[:, :, 0]
    scan_fast_t = torch.as_tensor(
        self.scan_fast[1], dtype=torch.float32, device=device
    )
    scan_slow_t = torch.as_tensor(
        self.scan_slow[1], dtype=torch.float32, device=device
    )
    aspect = float(scan_rows - 1) / float(scan_cols - 1) if scan_cols > 1 else 1.0
    determinant = (
        scan_slow_t[0] * scan_fast_t[1]
        - scan_fast_t[0] * aspect * scan_slow_t[1]
    )
    drift_row_t = (
        scan_fast_t[1] * delta_canvas_t[0]
        - scan_fast_t[0] * aspect * delta_canvas_t[1]
    ) / determinant
    drift_col_t = (
        -scan_slow_t[1] * delta_canvas_t[0]
        + scan_slow_t[0] * delta_canvas_t[1]
    ) / determinant
    row_t = torch.arange(scan_rows, dtype=torch.float32, device=device)
    col_t = torch.arange(scan_cols, dtype=torch.float32, device=device)
    sample_row_t = row_t[:, None] - drift_row_t[:, None]
    sample_col_t = col_t[None, :] - drift_col_t[:, None]
    grid_t = torch.stack(
        (
            2.0 * sample_col_t.expand(scan_rows, scan_cols) / (scan_cols - 1) - 1.0,
            2.0 * sample_row_t.expand(scan_rows, scan_cols) / (scan_rows - 1) - 1.0,
        ),
        dim=-1,
    )[None]

    corrected_array = np.empty(dataset.shape, dtype=np.float32)
    chunk_size = min(num_channels, 64)
    for start in range(0, num_channels, chunk_size):
        stop = min(start + chunk_size, num_channels)
        channels_t = torch.as_tensor(
            np.ascontiguousarray(dataset.array[:, :, start:stop]),
            dtype=torch.float32,
            device=device,
        ).permute(2, 0, 1)[None]
        corrected_t = torch.nn.functional.grid_sample(
            channels_t,
            grid_t,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0].permute(1, 2, 0)
        corrected_array[:, :, start:stop] = corrected_t.cpu().numpy()

    corrected = Dataset3d.from_array(
        corrected_array,
        name=f"drift-corrected {dataset.name}",
        origin=dataset.origin.copy(),
        sampling=dataset.sampling.copy(),
        units=list(dataset.units),
        signal_units=dataset.signal_units,
    )
    corrected.metadata.update(copy.deepcopy(dataset.metadata))
    return corrected


@torch.inference_mode()
def generate_corrected(
    self,
    upsample_factor: int = 2,
    output_original_shape: bool = True,
    strip_padding: bool = False,
    mask_output: bool = True,
    mask_edge_blend: float = 8.0,
    fourier_filter: bool = True,
    filter_midpoint: float = 0.5,
    kde_sigma: float | None = 0.5,
    weight_thresh: float = 0.1,
    show_merged: bool = True,
    **kwargs,
):
    """Generate the final drift-corrected image on GPU using torch.

    Parameters
    ----------
    upsample_factor : int, default 2
        Factor to upsample the output image for enhanced interpolation accuracy.
    output_original_shape : bool, default True
        If True, crop the output image back to the original input dimensions.
    strip_padding : bool, default False
        If True (and output_original_shape is True), also strip the scan padding
        to return only the original scan-area pixels.
    mask_output : bool, default True
        If True, mask the output using the probe position weights.
    mask_edge_blend : float, default 8.0
        Pixels over which the mask edge is blended.
    fourier_filter : bool, default True
        Whether to apply Fourier-based directional filtering to merge corrected images.
    filter_midpoint : float, default 0.5
        Midpoint for the sigmoid-based Fourier weighting filter.
    kde_sigma : float or None, default 0.5
        Standard deviation for kernel density estimation. Uses object's kde_sigma if None.
    weight_thresh : float, default 0.1
        Threshold for masking outputs.
    show_merged : bool, default True
        Whether to display the final corrected image.
    **kwargs
        Additional keyword arguments passed to the plotting function.

    Returns
    -------
    image_corr : Dataset2d
        The final drift-corrected output image.
    """
    if not hasattr(self, "knots"):
        raise RuntimeError(
            "No knots found. Call .preprocess() before generating the corrected image."
        )

    device = self._device
    dtype = self._dtype
    up_h = round(self.shape[1] * upsample_factor)
    up_w = round(self.shape[2] * upsample_factor)
    canvas_shape = (up_h, up_w)

    if kde_sigma is None:
        kde_sigma = self.kde_sigma

    stack_corr = torch.zeros(self.shape[0], up_h, up_w, dtype=dtype, device=device)
    weight_corr = torch.zeros_like(stack_corr)
    for img_idx in range(self.shape[0]):
        knots_t = torch.as_tensor(self.knots[img_idx], dtype=dtype, device=device)
        row_t, col_t = transform_coordinates_single_knot(
            knots_t,
            self.scan_fast_t[img_idx],
            self.images[img_idx].shape,
        )
        warped, weights = bilinear_kde_batch(
            row_t[None] * upsample_factor,
            col_t[None] * upsample_factor,
            self.images_t[img_idx],
            canvas_shape,
            kde_sigma * upsample_factor,
            self.pad_value[img_idx],
        )
        stack_corr[img_idx] = warped[0]
        weight_corr[img_idx] = weights[0]

    if fourier_filter:
        freq_row = torch.fft.fftfreq(up_h, dtype=dtype, device=device)[:, None]
        freq_col = torch.fft.fftfreq(up_w, dtype=dtype, device=device)[None, :]
        freq_angle = torch.atan2(freq_col, freq_row)
        stack_fft = torch.fft.fft2(stack_corr)
        weights = torch.zeros_like(stack_corr)
        for img_idx in range(self.shape[0]):
            weights[img_idx] = torch.abs(
                torch.remainder(
                    (freq_angle - self.scan_direction[img_idx]) / np.pi + 0.5,
                    1.0,
                ) - 0.5
            ) / 0.5
            weights[img_idx, 0, 0] = 1.0
            weights[img_idx] = _bounded_sine_sigmoid_torch(
                weights[img_idx],
                midpoint=filter_midpoint,
            )
            stack_fft[img_idx] *= weights[img_idx]
        weights_sum = weights.sum(0)
        fft_sum = stack_fft.sum(0)
        image_corr_fft = torch.where(
            weights_sum > 0.0,
            fft_sum / weights_sum.clamp(min=1e-8),
            torch.zeros_like(fft_sum),
        )
    else:
        image_corr_fft = torch.fft.fft2(stack_corr.mean(0))

    if mask_output:
        weight_np = weight_corr.cpu().numpy()
        mask_edge = np.prod(weight_np >= (weight_thresh / upsample_factor**2), axis=0)
        mask_edge[:, 0] = False
        mask_edge[:, -1] = False
        mask_edge[0, :] = False
        mask_edge[-1, :] = False
        mask_inner = distance_transform_edt(mask_edge) <= mask_edge_blend
        mask_np = (
            np.cos(
                (np.pi / 2)
                * np.clip(distance_transform_edt(mask_inner) / mask_edge_blend, 0.0, 1.0)
            )
            ** 2
        )
        mask_t = torch.as_tensor(mask_np, dtype=dtype, device=device)
        pad_value_mean = float(np.mean(self.pad_value))
        image_corr_fft = torch.fft.fft2(
            torch.fft.ifft2(image_corr_fft).real * mask_t + pad_value_mean * (1 - mask_t)
        )

    if output_original_shape:
        image_corr_fft = _fourier_crop_torch(image_corr_fft, self.shape[-2:]) / upsample_factor**2

    corr_np = torch.fft.ifft2(image_corr_fft).real.cpu().numpy()
    if strip_padding and output_original_shape:
        scan_h, scan_w = self.images[0].shape[:2]
        canvas_h, canvas_w = corr_np.shape[:2]
        pad_h = (canvas_h - scan_h) // 2
        pad_w = (canvas_w - scan_w) // 2
        corr_np = corr_np[pad_h:pad_h + scan_h, pad_w:pad_w + scan_w]

    image_corr = Dataset2d.from_array(
        corr_np,
        name="drift corrected image",
        origin=self.images[0].origin,
        sampling=self.images[0].sampling,
        units=self.images[0].units,
    )
    if show_merged:
        show_2d(image_corr.array, **kwargs)
        plt.show()
    return image_corr


def generate_corrected_image(
    self,
    upsample_factor: int = 2,
    output_original_shape: bool = True,
    mask_output: bool = True,
    mask_edge_blend: float = 8.0,
    fourier_filter: bool = True,
    filter_midpoint: float = 0.5,
    kde_sigma: float = 0.5,
    weight_thresh=0.1,
    show_image: bool = True,
    **kwargs,
):
    """
    Generate the final drift-corrected image after aligning a stack of input images.

    Parameters
    ----------
    upsample_factor : int, default 2
        Factor to upsample the output image for enhanced interpolation accuracy.
    output_original_shape : bool, default True
        If True, crop the output image back to the original input dimensions after processing.
    mask_output : bool, default True
        If true, mask the output using the probe position weights
    mask_edge_blend : float, default 8.0
        Value in pixels to blend from the edge of the mask (where we have data)
    fourier_filter : bool, default True
        Whether to apply Fourier-based directional filtering to merge corrected images.
    filter_midpoint : float, default 0.5
        Midpoint for the sigmoid-based Fourier weighting filter, determining transition smoothness.
        Setting this to a low value close to 0 will include more signal but also more slow scan artifacts.
        If using 2 images at 0 and 90 degrees scan angles, any value >0.75 will be unstable.
        Only use larger values (close to 1.0) if multiple images covering many scan angles are used.
    kde_sigma : float, default 0.5
        Standard deviation for kernel density estimation used during image interpolation. Defaults
        to the object's stored kde_sigma if set to None.
    weight_thresh: float, default 0.1
        This value sets the threshold for masking the outputs.
        For very large jitter artifacts this value can be lowered.
    show_image : bool, default True
        Whether to display the final corrected image after processing.
    **kwargs : dict
        Additional keyword arguments passed to the plotting function when displaying the image.

    Returns
    -------
    image_corr : Dataset2d
        The final drift-corrected output image encapsulated in a Dataset2d object.

    Notes
    -----
    - The function applies per-frame warping using knot-based interpolation and optionally
      performs directional Fourier filtering to blend multiple warped images.
    - The Fourier filter suppresses directional artifacts by weighting image contributions based
      on their scan angles, utilizing a bounded sine sigmoid for smooth transition.
    - Upsampling enhances interpolation precision but may increase computational cost.
    """

    # init
    stack_corr = np.zeros(
        (
            self.shape[0],
            np.round(self.shape[1] * upsample_factor).astype("int"),
            np.round(self.shape[2] * upsample_factor).astype("int"),
        )
    )
    weight_corr = np.zeros(
        (
            self.shape[0],
            np.round(self.shape[1] * upsample_factor).astype("int"),
            np.round(self.shape[2] * upsample_factor).astype("int"),
        )
    )

    if kde_sigma is None:
        kde_sigma = self.kde_sigma

    # Update images
    for ind in range(self.shape[0]):
        stack_corr[ind], weight_corr[ind] = self.interpolator[ind].warp_image(
            self.images[ind].array,
            self.knots[ind],
            kde_sigma=kde_sigma,
            upsample_factor=upsample_factor,
        )

    if fourier_filter:
        # Apply fourier filtering
        kx = np.fft.fftfreq(stack_corr.shape[1])[:, None]
        ky = np.fft.fftfreq(stack_corr.shape[2])[None, :]
        kt = np.arctan2(ky, kx)

        stack_fft = np.fft.fft2(stack_corr)
        weights = np.zeros_like(stack_corr)

        for ind in range(stack_corr.shape[0]):
            # Calculate weights as a function of angle
            weights[ind] = np.abs(
                np.mod((kt - self.scan_direction[ind]) / np.pi + 0.5, 1.0) - 0.5
            ) / (1 / 2)
            weights[ind][0, 0] = 1.0

            # Apply sigmoid to weighting function
            weights[ind] = bounded_sine_sigmoid(
                weights[ind],
                midpoint=filter_midpoint,
            )

            # Weight the fourier transformed images
            stack_fft[ind] *= weights[ind]

        weights_sum = np.sum(weights, axis=0)
        image_corr_fft = np.zeros_like(weights_sum, dtype=complex)
        np.divide(
            np.sum(stack_fft, axis=0),
            weights_sum,
            where=weights_sum > 0.0,
            out=image_corr_fft,
        )

    else:
        image_corr_fft = np.fft.fft2(np.mean(stack_corr, axis=0))

    if mask_output:
        # Note that we compute 2 boolean masks to round off the corners of image blending

        # calculate mask from product of individual image masks
        # scale weights by upsample factor to normalize to mean value of 1.0
        mask_edge = np.prod(weight_corr >= (weight_thresh / upsample_factor**2), axis=0)
        # Set outermost pixels to False to define the boundary for edge blending
        mask_edge[:, 0] = False
        mask_edge[:, -1] = False
        mask_edge[0, :] = False
        mask_edge[-1, :] = False
        # Find inner boundary mask
        mask_inner = distance_transform_edt(mask_edge) <= mask_edge_blend
        # compute mask using edge blending value
        mask = (
            np.cos(
                (np.pi / 2)
                * np.clip(distance_transform_edt(mask_inner) / mask_edge_blend, 0.0, 1.0)
            )
            ** 2
        )
        # Mean pad value
        pad_value_mean = np.mean([ind.pad_value for ind in self.interpolator])
        # apply mask
        image_corr_fft = np.fft.fft2(
            np.fft.ifft2(image_corr_fft) * mask + pad_value_mean * (1 - mask)
        )

    if output_original_shape:
        image_corr_fft = fourier_cropping(image_corr_fft, self.shape[-2:]) / upsample_factor**2

    # TODO - adjust origin / sampling if output sampling is different from input
    # i.e. if output_original_shape is False, and upsample_factor > 1
    image_corr = Dataset2d.from_array(
        np.real(np.fft.ifft2(image_corr_fft)),
        name="drift corrected image",
        origin=self.images[0].origin,
        sampling=self.images[0].sampling,
        units=self.images[0].units,
    )

    if show_image:
        fig, ax = show_2d(image_corr.array, **kwargs)
        # Force a render whether we're drawing into a provided Axes or a fresh Figure
        ax_to_draw = kwargs.get("ax", ax)
        try:
            ax_to_draw.figure.canvas.draw_idle()
            # If we're not drawing into a caller-provided Axes, also pop the window
            if "ax" not in kwargs:
                plt.show()
        except Exception:
            # Fallback: if backend is odd, try a blocking show
            plt.show()
    return image_corr


def calculate_error(
    self,
    mode: int,
    _warped_t: torch.Tensor | None = None,
):
    """Compute per-image MAE against the mean and append to error history.

    Measures how well the warped images agree by computing the mean
    absolute difference of each image from the stack mean. Without
    error tracking, there is no way to verify that alignment steps
    are actually improving the result.

    Parameters
    ----------
    mode : int
        Stage identifier (0=preprocess, 1=affine, 2=nonrigid).
    _warped_t : torch.Tensor or None
        If provided, compute error from this tensor directly,
        avoiding a GPU-to-CPU round-trip.
    """
    if _warped_t is not None:
        images_mean = _warped_t.mean(dim=0)
        sig_diff = torch.mean(
            torch.abs(_warped_t - images_mean[None]), dim=(1, 2)
        ).cpu().numpy()
    else:
        # Lazy refresh: align_nonrigid defers the warped→numpy sync until
        # someone reads it, so calculate_error must trigger the refresh.
        self._ensure_warped_images()
        images_mean = np.mean(self.images_warped.array, axis=0)
        sig_diff = np.mean(
            np.abs(self.images_warped.array - images_mean[None, :, :]), axis=(1, 2)
        )

    # Error vector
    error_current = np.hstack((mode, np.mean(sig_diff), sig_diff))

    # Initialize or append to error tracking array
    if not hasattr(self, "error_track"):
        self.error_track = error_current[None, :]  # initialize with first row
    else:
        self.error_track = np.vstack((self.error_track, error_current))


def bounded_sine_sigmoid(x, midpoint=0.5, width=1.0):
    """
    Piecewise bounded sigmoid: zero, raised sine squared, one.

    Parameters
    ----------
    x : array-like, shape (...,)
        Input values in [0, 1].
    midpoint : float
        Center of the sigmoid transition.
    width : float
        Width of the sigmoid (range over which it ramps from 0 to 1).
    Returns
    -------
    y : array-like
        Output in [0, 1], same shape as x.
    """
    x = np.asarray(x)
    # Truncate width if midpoint too close to edge
    left_max = midpoint - width / 2
    right_min = midpoint + width / 2
    if left_max < 0:
        warnings.warn(
            f"width={width} is too large for midpoint={midpoint}, "
            f"clamping width to {2 * midpoint}.",
            RuntimeWarning,
        )
        width = 2 * midpoint

    if right_min > 1:
        warnings.warn(
            f"width={width} is too large for midpoint={midpoint}, "
            f"clamping width to {2 * (1 - midpoint)}.",
            RuntimeWarning,
        )
        width = 2 * (1 - midpoint)
    # Recalculate edges
    left = midpoint - width / 2
    right = midpoint + width / 2

    y = np.zeros_like(x, dtype=float)
    in_band = (x >= left) & (x <= right)
    # Map [left, right] to [0, pi/2]
    t = (x[in_band] - left) / width  # goes from 0 to 1
    y[in_band] = np.sin(t * np.pi / 2) ** 2
    y[x > right] = 1.0
    return y


def _bounded_sine_sigmoid_torch(
    x: torch.Tensor,
    midpoint: float = 0.5,
    width: float = 1.0,
) -> torch.Tensor:
    width = min(width, 2 * midpoint, 2 * (1 - midpoint))
    left = midpoint - width / 2
    right = midpoint + width / 2
    t = ((x - left) / width).clamp(0.0, 1.0)
    return torch.where(x > right, torch.ones_like(x), torch.sin(t * (np.pi / 2)) ** 2)


def _fourier_crop_torch(
    fft_array: torch.Tensor,
    crop_shape: tuple[int, int],
) -> torch.Tensor:
    crop_h, crop_w = crop_shape
    h1 = crop_h // 2
    h2 = crop_h - h1
    w1 = crop_w // 2
    w2 = crop_w - w1
    result = torch.zeros(crop_shape, dtype=fft_array.dtype, device=fft_array.device)
    result[:h1, :w1] = fft_array[:h1, :w1]
    result[:h1, -w2:] = fft_array[:h1, -w2:]
    result[-h2:, :w1] = fft_array[-h2:, :w1]
    result[-h2:, -w2:] = fft_array[-h2:, -w2:]
    return result
