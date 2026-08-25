import warnings

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from quantem.core.utils.imaging_utils import (
    bilinear_kde,
)


class DriftInterpolator:
    def __init__(
        self,
        input_shape,
        output_shape,
        scan_fast,
        scan_slow,
        pad_value,
        kde_sigma,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.scan_fast = scan_fast
        self.scan_slow = scan_slow
        self.pad_value = pad_value
        self.kde_sigma = kde_sigma

        self.rows_input = np.arange(input_shape[0])
        self.cols_input = np.arange(input_shape[1])
        self.u = np.linspace(0, 1, input_shape[1])

    def transform_rows(
        self,
        knots_row: NDArray,
    ):
        num_knots = knots_row.shape[-1]
        basis = np.linspace(0, 1, num_knots)

        if num_knots == 1:
            xa = knots_row[0] + self.u[None, :] * self.scan_fast[0] * (self.input_shape[0] - 1)
            ya = knots_row[1] + self.u[None, :] * self.scan_fast[1] * (self.input_shape[1] - 1)
        elif num_knots == 2:
            xa = interp1d(basis, knots_row[0], kind="linear", assume_sorted=True)(self.u)
            ya = interp1d(basis, knots_row[1], kind="linear", assume_sorted=True)(self.u)
        else:
            kind = "quadratic" if num_knots == 3 else "cubic"
            xa = interp1d(
                basis,
                knots_row[0],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self.u)
            ya = interp1d(
                basis,
                knots_row[1],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self.u)

        return xa, ya

    def transform_coordinates(
        self,
        knots: NDArray,
    ):
        num_knots = knots.shape[-1]

        if num_knots == 1:
            # vectorized version for speed
            xa, ya = self.transform_rows(knots)
        else:
            xa = np.zeros(self.input_shape)
            ya = np.zeros(self.input_shape)
            for i in range(self.input_shape[0]):
                xa[i], ya[i] = self.transform_rows(knots[:, i])

        return xa, ya

    def warp_image(
        self,
        image: NDArray,
        knots: NDArray,  # shape: (2, rows, num_knots)
        kde_sigma=None,
        output_shape=None,
        pad_value=None,
        upsample_factor=None,
    ) -> NDArray:
        xa, ya = self.transform_coordinates(
            knots,
        )

        if kde_sigma is None:
            kde_sigma = self.kde_sigma

        if output_shape is None:
            output_shape = self.output_shape

        if pad_value is None:
            pad_value = self.pad_value

        if upsample_factor is None:
            upsample_factor = 1.0

        image_interp, weight_interp = bilinear_kde(
            xa=xa * upsample_factor,  # rows
            ya=ya * upsample_factor,  # cols
            values=image,
            output_shape=np.round(np.array(output_shape) * upsample_factor).astype("int"),
            kde_sigma=kde_sigma * upsample_factor,
            pad_value=pad_value,
            return_pix_count=True,
        )

        return image_interp, weight_interp


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
