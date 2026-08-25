"""Knot interpolation and batched scan-coordinate projection."""

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from quantem.core.utils.imaging_utils import bilinear_kde


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


def transform_coordinates_single_knot(
    knots: torch.Tensor,
    scan_fast: torch.Tensor,
    input_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-knot fast path: map source pixels to canvas coordinates.

    **Single-knot only.** Each scanline has exactly one (row, col) anchor;
    the fast-scan-direction position is filled in by linear interpolation
    along the scanline. Multi-knot Bezier interpolation is intentionally
    not supported here - that's the scipy backend's job. The pytorch path
    optimizes for the common single-knot case (≥95% of real STEM workflows).

    Called by ``preprocess``, ``_affine_grid_search_batch``, and
    ``_warp_and_translate_torch`` to map source image pixels onto the
    padded output canvas. Without this, the warped images would have
    no spatial mapping and the grid search couldn't score test drifts.

    Each input row maps to a line on the canvas:
    ``row = knot_row + fraction * scan_fast[0] * (num_rows - 1)``
    ``col = knot_col + fraction * scan_fast[1] * (num_cols - 1)``
    where row and col dimensions scale independently for non-square images.

    Parameters
    ----------
    knots : torch.Tensor
        Knot positions, shape ``(2, num_rows, 1)``. First dim is (row, col).
        The trailing 1 is the single-knot dimension; multi-knot inputs are
        rejected by the caller before reaching this function.
    scan_fast : torch.Tensor
        Fast scan direction vector, shape ``(2,)``.
    input_shape : tuple[int, int]
        Original image shape ``(num_rows, num_cols)``.

    Returns
    -------
    row_coords : torch.Tensor
        Row coordinates on canvas, shape ``(num_rows, num_cols)``.
    col_coords : torch.Tensor
        Column coordinates on canvas, shape ``(num_rows, num_cols)``.

    Examples
    --------
    >>> knots = torch.zeros(2, 64, 1)
    >>> scan_fast = torch.tensor([0.0, 1.0])
    >>> r, c = transform_coordinates_single_knot(knots, scan_fast, (64, 64))
    >>> r.shape
    torch.Size([64, 64])
    """
    num_rows, num_cols = input_shape
    fast_fraction = torch.linspace(0, 1, num_cols, dtype=knots.dtype, device=knots.device)
    row_coords = knots[0, :, 0:1] + fast_fraction[None, :] * scan_fast[0] * (num_rows - 1)
    col_coords = knots[1, :, 0:1] + fast_fraction[None, :] * scan_fast[1] * (num_cols - 1)
    return row_coords, col_coords


def bilinear_kde_batch(
    row_coords: torch.Tensor,
    col_coords: torch.Tensor,
    source_image: torch.Tensor,
    output_shape: tuple[int, int],
    kde_sigma: float,
    pad_value: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched bilinear KDE: scatter N source images onto an output canvas.

    Each pixel scatters its value to its 4 nearest grid neighbors with
    bilinear weights ``(1-dr)·(1-dc)``, ``dr·(1-dc)``, ``(1-dr)·dc``,
    ``dr·dc`` where ``dr, dc`` are fractional row/col distances.
    Accumulated counts and values are Gaussian-smoothed, then normalized:
    ``output = pad_value·(1-coverage) + coverage·(values/counts)``.

    Used by both the affine grid search (N = candidate drift vectors,
    single source image broadcast across drifts) and the nonrigid loop
    (N = stacked source images, one per drift).

    Parameters
    ----------
    row_coords : torch.Tensor
        Row coordinates of input pixels, shape ``(N, rows, cols)``.
    col_coords : torch.Tensor
        Column coordinates of input pixels, shape ``(N, rows, cols)``.
    source_image : torch.Tensor
        Pixel values to scatter. Either ``(rows, cols)`` (same image used
        for all N drifts - affine grid search) or ``(N, rows, cols)``
        (different image per drift - multi-image batched warping).
    output_shape : tuple[int, int]
        Canvas size ``(num_rows, num_cols)`` for the output images.
    kde_sigma : float
        Gaussian smoothing sigma in pixels.
    pad_value : float or torch.Tensor
        Fill value where pixel coverage is below threshold. If a tensor of
        shape ``(N,)``, applies a different pad value per drift.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(warped_images, sum_weights)`` - warped images and smoothed pixel coverage,
        both shape ``(N, num_rows, num_cols)``.
    """
    num_test_drifts = row_coords.shape[0]
    num_rows, num_cols = output_shape
    coverage_threshold = 1e-3
    # Flatten spatial dims - scatter_add_ works on 1D buffers
    row_flat = row_coords.flatten(1)
    col_flat = col_coords.flatten(1)
    # Stay in float for fractional distance, convert to int only for scatter indices
    row_floor = row_flat.floor()
    col_floor = col_flat.floor()
    frac_row = row_flat - row_floor
    frac_col = col_flat - col_floor
    row_floor = row_floor.int()
    col_floor = col_floor.int()
    if source_image.dim() == 3:
        # Per-drift source images: each drift scatters its own pixel values
        source_values_flat = source_image.flatten()
    else:
        source_values_flat = source_image.flatten().repeat(num_test_drifts)
    # All N batch entries scatter into one flat buffer - offset separates them
    batch_offsets = (
        torch.arange(num_test_drifts, device=row_coords.device, dtype=torch.int32)
        * num_rows * num_cols
    )[:, None]
    # Float32 accumulators - scatter_add_ requires source dtype to match,
    # so all input tensors must be float32 (raises on float64).
    sum_weights = torch.zeros(
        num_test_drifts * num_rows * num_cols, dtype=torch.float32, device=row_coords.device
    )
    sum_values = torch.zeros_like(sum_weights)
    # Periodic wrapping so pixels near edges scatter to the opposite side
    row_wrapped = row_floor % num_rows
    col_wrapped = col_floor % num_cols
    row_next = (row_wrapped + 1) % num_rows
    col_next = (col_wrapped + 1) % num_cols
    # Each pixel distributes its value to the 4 nearest grid neighbors
    # weighted by bilinear distance: (1-dr)(1-dc), dr(1-dc), (1-dr)dc, dr·dc
    for corner_row, corner_col, corner_weight in [
        (row_wrapped, col_wrapped, ((1 - frac_row) * (1 - frac_col)).flatten()),
        (row_next, col_wrapped, (frac_row * (1 - frac_col)).flatten()),
        (row_wrapped, col_next, ((1 - frac_row) * frac_col).flatten()),
        (row_next, col_next, (frac_row * frac_col).flatten()),
    ]:
        flat_indices = (corner_row * num_cols + corner_col + batch_offsets).flatten()
        sum_weights.scatter_add_(0, flat_indices, corner_weight)
        sum_values.scatter_add_(0, flat_indices, corner_weight * source_values_flat)
    sum_weights = sum_weights.reshape(num_test_drifts, num_rows, num_cols)
    sum_values = sum_values.reshape(num_test_drifts, num_rows, num_cols)
    # Smooth the scattered counts and values to fill gaps between pixels
    sum_weights = gaussian_smooth_batch(sum_weights, kde_sigma)
    sum_values = gaussian_smooth_batch(sum_values, kde_sigma)
    # Blend between pad_value (uncovered) and normalized values (covered),
    # ramping linearly with coverage to avoid hard edges at the boundary
    coverage_weight = torch.clamp(sum_weights / coverage_threshold, max=1.0)
    if isinstance(pad_value, torch.Tensor) and pad_value.dim() == 1:
        # Per-drift pad value: reshape (N,) → (N, 1, 1) for broadcasting
        pad_value = pad_value[:, None, None]
    warped_images = pad_value * (1 - coverage_weight) + coverage_weight * (
        sum_values / torch.clamp(sum_weights, min=1e-8)
    )
    return warped_images, sum_weights


def gaussian_smooth_batch(
    field_stack: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Batched 2D Gaussian smoothing matching ``scipy.ndimage.gaussian_filter``.

    Used by ``bilinear_kde_batch`` to smooth scattered counts and
    values before normalization. Without smoothing, the warped images
    have salt-and-pepper artifacts from the scatter step.

    Parameters
    ----------
    field_stack : torch.Tensor
        Input tensor of shape ``(N, num_rows, num_cols)``.
    sigma : float
        Standard deviation of the Gaussian kernel in pixels.

    Returns
    -------
    torch.Tensor
        Smoothed tensor of shape ``(N, num_rows, num_cols)``.

    """
    kernel, radius = _gaussian_kernel_1d(sigma, field_stack.dtype, field_stack.device)
    # Separable kernel: column pass then row pass to halve FLOPs vs full 2D conv
    kernel_col = kernel[None, None, None, :]
    kernel_row = kernel[None, None, :, None]
    field_stack = field_stack[:, None]
    field_stack = torch.nn.functional.conv2d(_symmetric_pad(field_stack, 0, radius), kernel_col)
    field_stack = torch.nn.functional.conv2d(_symmetric_pad(field_stack, radius, 0), kernel_row)
    return field_stack[:, 0]


def gaussian_smooth_1d(
    signal: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """1D Gaussian smoothing matching ``scipy.ndimage.gaussian_filter``.

    Smooths each row of the input independently using a separable 1D kernel.
    Used for regularizing knot displacement vectors in the nonrigid loop,
    where the signal is 1D (one value per scan line).

    Parameters
    ----------
    signal : torch.Tensor
        Input tensor of shape ``(N, L)`` - N channels, L samples.
    sigma : float
        Standard deviation of the Gaussian kernel in pixels.

    Returns
    -------
    torch.Tensor
        Smoothed tensor of shape ``(N, L)``.
    """
    kernel, radius = _gaussian_kernel_1d(sigma, signal.dtype, signal.device)
    signal_padded = _symmetric_pad_1d(signal[:, None], radius)
    return torch.nn.functional.conv1d(signal_padded, kernel[None, None, :])[:, 0]


def _symmetric_pad_1d(signal: torch.Tensor, pad: int) -> torch.Tensor:
    """Symmetric 1D padding matching scipy's reflect mode.

    Same edge-repeat semantics as ``_symmetric_pad`` but for 1D signals.
    Used by ``gaussian_smooth_1d`` for regularization of knot vectors.
    """
    left = signal[:, :, :pad].flip(-1)
    right = signal[:, :, -pad:].flip(-1)
    return torch.cat([left, signal, right], dim=-1)


def _symmetric_pad(
    field_stack: torch.Tensor,
    pad_rows: int,
    pad_cols: int,
) -> torch.Tensor:
    """Symmetric padding matching scipy's reflect mode for parity.

    Without this, the torch and numpy Gaussian smoothing paths produce
    different results near canvas edges, breaking numerical parity.

    Scipy's ``mode='reflect'`` repeats the edge pixel
    (``[1,2,3]`` → ``[2,1,1,2,3,3,2]``), but PyTorch's
    ``F.pad(mode='reflect')`` does not (``[1,2,3]`` → ``[3,2,1,2,3,2,1]``).

    Parameters
    ----------
    field_stack : torch.Tensor
        Input tensor of shape ``(N, C, num_rows, num_cols)``.
    pad_rows : int
        Number of rows to pad on top and bottom.
    pad_cols : int
        Number of columns to pad on left and right.

    Returns
    -------
    torch.Tensor
        Padded tensor.

    Examples
    --------
    >>> t = torch.tensor([[[[1., 2., 3.]]]])
    >>> _symmetric_pad(t, 0, 2)
    tensor([[[[2., 1., 1., 2., 3., 3., 2.]]]])
    """
    if pad_cols > 0:
        left = field_stack[:, :, :, :pad_cols].flip(-1)
        right = field_stack[:, :, :, -pad_cols:].flip(-1)
        field_stack = torch.cat([left, field_stack, right], dim=-1)
    if pad_rows > 0:
        top = field_stack[:, :, :pad_rows, :].flip(-2)
        bottom = field_stack[:, :, -pad_rows:, :].flip(-2)
        field_stack = torch.cat([top, field_stack, bottom], dim=-2)
    return field_stack


def _gaussian_kernel_1d(sigma, dtype, device, _cache={}):
    """Normalized 1D Gaussian ``exp(-0.5*(x/sigma)^2)``, radius ``4*sigma``.

    Cached via mutable default arg - the grid search calls this ~800 times
    with the same sigma, saving ~44ms of redundant kernel construction.
    """
    key = (sigma, dtype, device)
    if key not in _cache:
        radius = int(4 * sigma + 0.5)
        offsets = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
        kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        _cache[key] = (kernel / kernel.sum(), radius)
    return _cache[key]
