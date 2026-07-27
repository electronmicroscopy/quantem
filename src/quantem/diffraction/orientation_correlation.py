"""Memory-efficient distance-angle correlations for orientation histograms."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
from scipy import fft as scipy_fft
from tqdm.auto import tqdm


def _validate_and_shape_input(orient_hist):
    is_torch = isinstance(orient_hist, torch.Tensor)
    if orient_hist.ndim == 3:
        orient_hist = orient_hist[None]
    elif orient_hist.ndim != 4:
        raise ValueError(
            "orient_hist must have shape (x, y, theta) or "
            "(radial_bin, x, y, theta)"
        )

    num_radii, size_x, size_y, num_theta = orient_hist.shape
    if min(num_radii, size_x, size_y) < 1 or num_theta < 2:
        raise ValueError(
            "orient_hist must contain at least one radial bin and spatial pixel, "
            "and at least two theta bins"
        )
    return orient_hist, is_torch


def _resolve_pairs(pairs, num_radii):
    if isinstance(pairs, str):
        if pairs == "all":
            pair_list = [
                (first, second)
                for first in range(num_radii)
                for second in range(first, num_radii)
            ]
        elif pairs == "autocorrelation":
            pair_list = [(index, index) for index in range(num_radii)]
        else:
            raise ValueError(
                "pairs must be 'all', 'autocorrelation', or a sequence of pairs"
            )
    else:
        pair_list = []
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("each entry in pairs must contain two indices")
            first, second = int(pair[0]), int(pair[1])
            if not (
                0 <= first < num_radii and 0 <= second < num_radii
            ):
                raise ValueError(
                    f"radial-bin pair {(first, second)} is outside "
                    f"[0, {num_radii})"
                )
            pair_list.append((first, second))

        if not pair_list:
            raise ValueError("pairs must contain at least one radial-bin pair")

    return np.asarray(pair_list, dtype=np.int64)


def _radial_geometry(size_x, size_y, radius_max):
    """Build two-point linear interpolation from spatial pixels to radial bins."""
    padded_x = max(2 * size_x, 2 * radius_max)
    padded_y = max(2 * size_y, 2 * radius_max)

    x = np.mod(np.arange(padded_x) + padded_x / 2, padded_x) - padded_x / 2
    y = np.mod(np.arange(padded_y) + padded_y / 2, padded_y) - padded_y / 2
    yy, xx = np.meshgrid(y, x)
    radius = np.sqrt(xx**2 + yy**2)

    lower_mask = radius <= radius_max
    upper_mask = radius <= radius_max - 1
    lower_floor = np.floor(radius[lower_mask]).astype(np.int64)
    upper_floor = np.floor(radius[upper_mask]).astype(np.int64)

    return {
        "padded_shape": (padded_x, padded_y),
        "point_indices": (
            np.flatnonzero(lower_mask),
            np.flatnonzero(upper_mask),
        ),
        "radial_bins": (lower_floor, upper_floor + 1),
        "radial_weights": (
            1.0 - (radius[lower_mask] - lower_floor),
            radius[upper_mask] - upper_floor,
        ),
    }


def _normalize_correlation(
    radial_correlation,
    correlation_spectrum,
    num_modes,
    num_theta,
    zero_policy,
):
    denominator = correlation_spectrum[:, 0, :].real / num_theta

    if isinstance(radial_correlation, torch.Tensor):
        dtype = radial_correlation.dtype
        maximum = torch.max(torch.abs(denominator))
        threshold = torch.finfo(dtype).eps * torch.clamp(
            maximum, min=torch.finfo(dtype).tiny
        )
        valid = torch.abs(denominator) > threshold
        if zero_policy == "raise" and not bool(torch.all(valid).item()):
            raise ZeroDivisionError(
                "orientation correlation has radial distances with zero "
                "normalization signal"
            )
        safe_denominator = torch.where(
            valid, denominator, torch.ones_like(denominator)
        )
        output = (
            radial_correlation[:, :num_modes, :]
            / safe_denominator[:, None, :]
        )
        fill_value = float("nan") if zero_policy == "nan" else 0.0
        return output.masked_fill(~valid[:, None, :], fill_value)

    dtype = radial_correlation.dtype
    maximum = float(np.max(np.abs(denominator), initial=0.0))
    threshold = np.finfo(dtype).eps * max(maximum, np.finfo(dtype).tiny)
    valid = np.abs(denominator) > threshold
    if zero_policy == "raise" and not np.all(valid):
        raise ZeroDivisionError(
            "orientation correlation has radial distances with zero "
            "normalization signal"
        )
    fill_value = np.nan if zero_policy == "nan" else 0.0
    output = np.full(
        (radial_correlation.shape[0], num_modes, radial_correlation.shape[2]),
        fill_value,
        dtype=dtype,
    )
    np.divide(
        radial_correlation[:, :num_modes, :],
        denominator[:, None, :],
        out=output,
        where=valid[:, None, :],
    )
    return output


def _calculate_numpy(
    orient_hist,
    pair_indices,
    geometry,
    num_theta,
    radius_max,
    *,
    dtype,
    mode_batch_size,
    pair_batch_size,
    workers,
    zero_policy,
    progress_bar,
):
    real_dtype = np.float32 if dtype == "float32" else np.float64
    complex_dtype = np.complex64 if dtype == "float32" else np.complex128
    histogram = np.asarray(orient_hist, dtype=real_dtype)
    if not np.all(np.isfinite(histogram)):
        raise ValueError("orient_hist contains NaN or infinite values")

    num_pairs = len(pair_indices)
    num_modes = num_theta // 2 + 1
    num_distances = radius_max + 1
    mode_batch_size = min(mode_batch_size or 1, num_modes)
    pair_batch_size = min(pair_batch_size or 4, num_pairs)
    if mode_batch_size < 1 or pair_batch_size < 1:
        raise ValueError("mode_batch_size and pair_batch_size must be at least 1")

    theta_spectrum = scipy_fft.rfft(histogram, axis=-1, workers=workers)
    correlation_spectrum = np.empty(
        (num_pairs, num_modes, num_distances), dtype=complex_dtype
    )
    point_indices = geometry["point_indices"]
    radial_bins = geometry["radial_bins"]
    radial_weights = geometry["radial_weights"]

    total = (
        int(np.ceil(num_modes / mode_batch_size))
        * int(np.ceil(num_pairs / pair_batch_size))
    )
    progress = tqdm(
        total=total,
        desc="Calculate orientation correlations (CPU)",
        unit="batch",
        disable=not progress_bar,
    )
    try:
        for mode_start in range(0, num_modes, mode_batch_size):
            mode_stop = min(mode_start + mode_batch_size, num_modes)
            spatial_spectrum = scipy_fft.fft2(
                np.moveaxis(
                    theta_spectrum[..., mode_start:mode_stop], -1, 1
                ),
                s=geometry["padded_shape"],
                axes=(-2, -1),
                workers=workers,
            )

            for pair_start in range(0, num_pairs, pair_batch_size):
                pair_stop = min(pair_start + pair_batch_size, num_pairs)
                pair_batch = pair_indices[pair_start:pair_stop]
                cross_spectrum = (
                    spatial_spectrum[pair_batch[:, 0]]
                    * np.conj(spatial_spectrum[pair_batch[:, 1]])
                )
                spatial_correlation = scipy_fft.ifft2(
                    cross_spectrum, axes=(-2, -1), workers=workers
                ).reshape(len(pair_batch), mode_stop - mode_start, -1)
                radial_correlation = np.zeros(
                    (
                        len(pair_batch),
                        mode_stop - mode_start,
                        num_distances,
                    ),
                    dtype=complex_dtype,
                )

                # NumPy does not provide batched bincount. Only the small
                # pair/mode dimensions are looped; spatial work stays vectorized.
                for pair_index in range(len(pair_batch)):
                    for mode_index in range(mode_stop - mode_start):
                        output = radial_correlation[pair_index, mode_index]
                        for points, bins, weights in zip(
                            point_indices, radial_bins, radial_weights
                        ):
                            values = (
                                spatial_correlation[
                                    pair_index, mode_index, points
                                ]
                                * weights
                            )
                            output += np.bincount(
                                bins,
                                weights=values.real,
                                minlength=num_distances,
                            )
                            output += 1j * np.bincount(
                                bins,
                                weights=values.imag,
                                minlength=num_distances,
                            )

                correlation_spectrum[
                    pair_start:pair_stop, mode_start:mode_stop
                ] = radial_correlation
                progress.update()
    finally:
        progress.close()

    radial_correlation = scipy_fft.irfft(
        correlation_spectrum, n=num_theta, axis=1, workers=workers
    )
    output = _normalize_correlation(
        radial_correlation,
        correlation_spectrum,
        num_modes,
        num_theta,
        zero_policy,
    )
    return output.astype(real_dtype, copy=False)


def _calculate_torch(
    orient_hist,
    pair_indices,
    geometry,
    num_theta,
    radius_max,
    *,
    device,
    dtype,
    mode_batch_size,
    pair_batch_size,
    max_memory_fraction,
    zero_policy,
    progress_bar,
):
    device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device} was requested, but CUDA is not available"
        )

    real_dtype = torch.float32 if dtype == "float32" else torch.float64
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )
    histogram = torch.as_tensor(
        orient_hist, dtype=real_dtype, device=device
    )
    if not bool(torch.all(torch.isfinite(histogram)).item()):
        raise ValueError("orient_hist contains NaN or infinite values")

    num_radii = histogram.shape[0]
    num_pairs = len(pair_indices)
    num_modes = num_theta // 2 + 1
    num_distances = radius_max + 1
    pair_batch_was_requested = pair_batch_size is not None
    pair_batch_size = min(pair_batch_size or 4, num_pairs)
    if pair_batch_size < 1:
        raise ValueError("pair_batch_size must be at least 1")

    if mode_batch_size is None:
        if device.type == "cuda":
            free_memory, total_memory = torch.cuda.mem_get_info(device)
            # cudaMemGetInfo reports memory free on the *device*, which ignores
            # any per-process cap from torch.cuda.set_per_process_memory_fraction.
            # Budgeting off the device figure under a cap sizes batches for
            # headroom this process may not allocate, and the fft2 below then
            # raises OutOfMemoryError while the device still looks mostly free.
            # Take whichever allowance is smaller.
            try:
                process_fraction = torch.cuda.get_per_process_memory_fraction(device)
            except (AttributeError, RuntimeError, TypeError):
                process_fraction = 1.0
            complex_bytes = 8 if complex_dtype == torch.complex64 else 16
            padded_pixels = int(np.prod(geometry["padded_shape"]))
            # theta_spectrum is allocated after this estimate, so reserve room for
            # it up front rather than discovering the shortfall mid-loop.
            spectrum_bytes = (
                histogram.numel() // num_theta * (num_theta // 2 + 1)
            ) * complex_bytes

            def available_bytes():
                budget = free_memory
                if 0.0 < process_fraction < 1.0:
                    # Headroom is measured against *allocated*, not *reserved*:
                    # blocks the caching allocator holds but no tensor is using
                    # are reusable, and after a failed run they can account for
                    # most of the cap. Counting them as spent wrongly reports a
                    # zero budget.
                    remaining = (
                        process_fraction * total_memory
                        - torch.cuda.memory_allocated(device)
                    )
                    budget = min(budget, max(0, int(remaining)))
                return max(0, budget - spectrum_bytes)

            memory_budget = int(available_bytes() * max_memory_fraction)

            def estimate_bytes_per_mode(batch_size):
                return (
                    padded_pixels
                    * complex_bytes
                    * (num_radii + 3 * batch_size)
                )

            bytes_per_mode = estimate_bytes_per_mode(pair_batch_size)
            if not pair_batch_was_requested:
                while pair_batch_size > 1 and bytes_per_mode > memory_budget:
                    pair_batch_size = max(1, pair_batch_size // 2)
                    bytes_per_mode = estimate_bytes_per_mode(pair_batch_size)
            if bytes_per_mode > memory_budget:
                # A previous failure can leave the cap saturated with cached
                # blocks. Return them and re-measure before giving up.
                torch.cuda.empty_cache()
                free_memory, total_memory = torch.cuda.mem_get_info(device)
                memory_budget = int(available_bytes() * max_memory_fraction)
                if not pair_batch_was_requested:
                    while pair_batch_size > 1 and bytes_per_mode > memory_budget:
                        pair_batch_size = max(1, pair_batch_size // 2)
                        bytes_per_mode = estimate_bytes_per_mode(pair_batch_size)
            if bytes_per_mode > memory_budget:
                allocated_gib = torch.cuda.memory_allocated(device) / 1024**3
                cap_text = (
                    f"{process_fraction * total_memory / 1024**3:.2f} GiB "
                    "(torch.cuda.set_per_process_memory_fraction)"
                    if 0.0 < process_fraction < 1.0
                    else f"{total_memory / 1024**3:.2f} GiB (device total, no cap)"
                )
                raise MemoryError(
                    "A single angular-mode batch is estimated to require "
                    f"{bytes_per_mode / 1024**3:.2f} GiB, but the CUDA memory "
                    f"budget is only {memory_budget / 1024**3:.2f} GiB "
                    f"(max_memory_fraction={max_memory_fraction}). This process "
                    f"is capped at {cap_text}, currently holds "
                    f"{allocated_gib:.2f} GiB live, and must also reserve "
                    f"{spectrum_bytes / 1024**3:.2f} GiB for the angular "
                    f"spectrum; {free_memory / 1024**3:.2f} GiB is free on the "
                    "device. Raise the per-process cap or max_memory_fraction, "
                    "or reduce radius_max or the orientation-histogram upsampling."
                )
            mode_batch_size = max(
                1,
                memory_budget // max(bytes_per_mode, 1),
            )
        else:
            mode_batch_size = 1
    mode_batch_size = min(int(mode_batch_size), num_modes)
    if mode_batch_size < 1:
        raise ValueError("mode_batch_size must be at least 1")

    point_indices = [
        torch.as_tensor(values, dtype=torch.long, device=device)
        for values in geometry["point_indices"]
    ]
    radial_bins = [
        torch.as_tensor(values, dtype=torch.long, device=device)
        for values in geometry["radial_bins"]
    ]
    radial_weights = [
        torch.as_tensor(values, dtype=real_dtype, device=device)
        for values in geometry["radial_weights"]
    ]
    pair_indices = torch.as_tensor(
        pair_indices, dtype=torch.long, device=device
    )

    theta_spectrum = torch.fft.rfft(histogram, dim=-1)
    correlation_spectrum = torch.empty(
        (num_pairs, num_modes, num_distances),
        dtype=complex_dtype,
        device=device,
    )
    total = (
        int(np.ceil(num_modes / mode_batch_size))
        * int(np.ceil(num_pairs / pair_batch_size))
    )
    progress = tqdm(
        total=total,
        desc=f"Calculate orientation correlations ({device})",
        unit="batch",
        disable=not progress_bar,
    )
    try:
        for mode_start in range(0, num_modes, mode_batch_size):
            mode_stop = min(mode_start + mode_batch_size, num_modes)
            spatial_spectrum = torch.fft.fft2(
                theta_spectrum[..., mode_start:mode_stop].movedim(-1, 1),
                s=geometry["padded_shape"],
                dim=(-2, -1),
            )

            for pair_start in range(0, num_pairs, pair_batch_size):
                pair_stop = min(pair_start + pair_batch_size, num_pairs)
                pair_batch = pair_indices[pair_start:pair_stop]
                cross_spectrum = (
                    spatial_spectrum.index_select(0, pair_batch[:, 0])
                    * torch.conj(
                        spatial_spectrum.index_select(0, pair_batch[:, 1])
                    )
                )
                spatial_correlation = torch.fft.ifft2(
                    cross_spectrum, dim=(-2, -1)
                ).flatten(-2)
                radial_correlation = torch.zeros(
                    (
                        len(pair_batch),
                        mode_stop - mode_start,
                        num_distances,
                    ),
                    dtype=complex_dtype,
                    device=device,
                )
                for points, bins, weights in zip(
                    point_indices, radial_bins, radial_weights
                ):
                    radial_correlation.index_add_(
                        -1,
                        bins,
                        spatial_correlation.index_select(-1, points) * weights,
                    )

                correlation_spectrum[
                    pair_start:pair_stop, mode_start:mode_stop
                ] = radial_correlation
                progress.update()
    finally:
        progress.close()

    radial_correlation = torch.fft.irfft(
        correlation_spectrum, n=num_theta, dim=1
    )
    return _normalize_correlation(
        radial_correlation,
        correlation_spectrum,
        num_modes,
        num_theta,
        zero_policy,
    )


def calculate_orientation_correlation(
    orient_hist,
    radius_max: int | None = None,
    pairs: str | Sequence[tuple[int, int]] = "all",
    backend: str = "auto",
    device=None,
    mode_batch_size: int | None = None,
    pair_batch_size: int | None = None,
    max_memory_fraction: float = 0.6,
    dtype: str = "float32",
    workers: int | None = None,
    zero_policy: str = "nan",
    return_numpy: bool = True,
    progress_bar: bool = True,
):
    """
    Compute spatial-distance versus relative-angle correlations.

    The angular Fourier modes are streamed through batched 2D spatial
    correlations and radially integrated before the angular inverse transform.
    This is equivalent to constructing a full 3D correlation volume, while
    requiring substantially less peak memory.

    Returns
    -------
    orient_corr, pair_indices
        Correlation values have shape
        ``(num_pairs, num_theta // 2 + 1, radius_max + 1)`` and are normalized
        in multiples of a random distribution. ``pair_indices`` maps the first
        axis back to radial-bin pairs.
    """
    if backend not in {"auto", "numpy", "torch"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")
    if zero_policy not in {"nan", "zero", "raise"}:
        raise ValueError("zero_policy must be 'nan', 'zero', or 'raise'")
    if not 0 < max_memory_fraction <= 1:
        raise ValueError("max_memory_fraction must be in the interval (0, 1]")

    orient_hist, is_torch_input = _validate_and_shape_input(orient_hist)
    num_radii, size_x, size_y, num_theta = orient_hist.shape
    if radius_max is None:
        radius_max = int(np.ceil(min(size_x, size_y) / 2))
    elif not isinstance(radius_max, (int, np.integer)):
        raise TypeError("radius_max must be an integer or None")
    radius_max = int(radius_max)
    if radius_max < 0:
        raise ValueError("radius_max must be non-negative")

    pair_indices = _resolve_pairs(pairs, num_radii)
    geometry = _radial_geometry(size_x, size_y, radius_max)
    if backend == "auto":
        wants_cuda = device is None or str(device).startswith("cuda")
        backend = (
            "torch"
            if torch.cuda.is_available() and wants_cuda
            else "numpy"
        )

    if backend == "numpy":
        histogram = (
            orient_hist.detach().cpu().numpy()
            if is_torch_input
            else orient_hist
        )
        output = _calculate_numpy(
            histogram,
            pair_indices,
            geometry,
            num_theta,
            radius_max,
            dtype=dtype,
            mode_batch_size=mode_batch_size,
            pair_batch_size=pair_batch_size,
            workers=workers,
            zero_policy=zero_policy,
            progress_bar=progress_bar,
        )
    else:
        output = _calculate_torch(
            orient_hist,
            pair_indices,
            geometry,
            num_theta,
            radius_max,
            device=device,
            dtype=dtype,
            mode_batch_size=mode_batch_size,
            pair_batch_size=pair_batch_size,
            max_memory_fraction=max_memory_fraction,
            zero_policy=zero_policy,
            progress_bar=progress_bar,
        )
        if return_numpy:
            output = output.detach().cpu().numpy()

    return output, pair_indices


# Default 1/e decay length for the orientation-correlation slope fit, as a
# fraction of the fitted lobe's distance span. Chosen on real scan data
# (pg3T2 07-03-2024/61): it holds the fitted intercept within ~1.7 degrees of
# the measured boundary at zero separation across all six ring pairs, versus
# up to 7 degrees for an unweighted fit, while keeping enough effective points
# for a stable slope. See plot_orientation_correlation(slope_weight_scale=...).
SLOPE_WEIGHT_FRACTION = 0.10


def plot_orientation_correlation(
    orient_corr,
    orient_corr_pairs=None,
    *,
    pair_indices=None,
    pixel_size=1.0,
    pixel_units="scan pixels",
    probability_range=(0.5, 2.0),
    cmap="correlation",
    figsize=None,
    show_metrics=True,
    return_metrics=False,
    slope_weight_scale=None,
):
    """Plot distance-orientation correlations using Matplotlib.

    The 50% boundary is halfway between the correlation at zero separation
    and the random-association baseline of one. Its intercepts give the
    radial and annular 50% distances. The signed slope is fitted separately
    to the primary correlation-equals-one boundary between positive
    correlation and anticorrelation.

    Parameters
    ----------
    slope_weight_scale : float, optional
        1/e decay length, in ``pixel_units``, of the exponential weighting
        applied to the slope fit. The correlation-equals-one boundary
        saturates with distance, so an unweighted straight line over the
        whole lobe is dominated by the flat tail: it biases the slope low
        and pushes the fitted intercept off the measured boundary at zero
        separation. Weighting towards short distances makes ``slope`` the
        near-origin tangent instead. Defaults to
        ``SLOPE_WEIGHT_FRACTION`` times the fitted distance span. Pass
        ``numpy.inf`` to restore the previous unweighted full-lobe fit.

    Notes
    -----
    Each metrics entry reports ``slope_fit_intercept_degrees`` (compare it
    against the boundary at zero separation to check the fit),
    ``slope_fit_effective_point_count`` (Kish effective sample size, which
    falls as the weighting sharpens), and the resolved
    ``slope_weight_scale``. ``slope_fit_r_squared`` is weighted with the
    same weights, so it is not comparable to an unweighted R-squared.
    """
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.lines import Line2D

    def crossing(coordinates, profile, level):
        profile = np.asarray(profile, dtype=float)
        coordinates = np.asarray(coordinates, dtype=float)
        if not np.isfinite(profile[0]):
            return np.nan
        initial_side = profile[0] - level
        if initial_side == 0:
            return float(coordinates[0])
        for point in range(1, len(profile)):
            before, after = profile[point - 1], profile[point]
            if not np.isfinite(before) or not np.isfinite(after):
                continue
            before_side = before - level
            after_side = after - level
            if before_side == 0:
                return float(coordinates[point - 1])
            if before_side * after_side <= 0:
                if before == after:
                    return float(coordinates[point])
                fraction = -before_side / (after_side - before_side)
                return float(
                    coordinates[point - 1]
                    + fraction * (coordinates[point] - coordinates[point - 1])
                )
        return np.nan

    values = orient_corr
    if values is None:
        raise RuntimeError(
            "No orientation correlation is available. Run "
            "calculate_orientation_correlation() first or pass orient_corr."
        )
    values = np.asarray(values)
    if values.ndim != 3:
        raise ValueError(
            "orient_corr must have shape (pair, relative_angle, distance)."
        )
    if values.shape[0] == 0:
        raise ValueError("orient_corr must contain at least one radial-bin pair.")
    if values.shape[1] < 2 or values.shape[2] < 2:
        raise ValueError(
            "orient_corr requires at least two angle and two distance samples."
        )
    labels = orient_corr_pairs if pair_indices is None else pair_indices
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (values.shape[0], 2):
            raise ValueError(
                f"pair_indices must have shape ({values.shape[0]}, 2)."
            )

    panel_count = values.shape[0]
    column_count = min(3, max(1, panel_count))
    row_count = int(np.ceil(panel_count / column_count))
    if figsize is None:
        figsize = (4.5 * column_count, 3.8 * row_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    lower, upper = map(float, probability_range)
    if not 0 < lower < upper:
        raise ValueError("probability_range must satisfy 0 < lower < upper.")
    if cmap == "correlation":
        cmap = LinearSegmentedColormap.from_list(
            "quantem_correlation",
            [
                (0.00, "#002b9a"),
                (0.32, "#1769e8"),
                (0.50, "#b8b8b8"),
                (0.68, "#f23838"),
                (1.00, "#9e0015"),
            ],
        )
    distance_max = (values.shape[2] - 1) * float(pixel_size)
    distances = np.arange(values.shape[2], dtype=float) * float(pixel_size)
    angles = np.linspace(0.0, 180.0, values.shape[1])
    image = None
    metrics = []
    for index, ax in enumerate(axes.flat):
        if index >= panel_count:
            ax.set_visible(False)
            continue
        image = ax.imshow(
            values[index],
            origin="lower",
            aspect="auto",
            extent=(0, distance_max, 0, 180),
            norm=LogNorm(vmin=lower, vmax=upper),
            cmap=cmap,
        )
        if labels is None:
            title = f"Ring pair {index}"
            pair = (index, index)
        else:
            pair = tuple(int(value) for value in labels[index])
            title = (
                f"Autocorrelation of Ring {pair[0]}"
                if pair[0] == pair[1]
                else f"Correlation of Rings {pair[0]} and {pair[1]}"
            )
        ax.set(
            title=title,
            xlabel=f"distance ({pixel_units})",
            ylabel="relative orientation (degrees)",
        )
        panel = np.asarray(values[index], dtype=float)
        origin_probability = panel[0, 0]
        half_probability = (
            1.0 + 0.5 * (origin_probability - 1.0)
            if np.isfinite(origin_probability)
            and origin_probability > 0.0
            and not np.isclose(origin_probability, 1.0)
            else np.nan
        )
        radial_distance = np.nan
        annular_distance = np.nan
        slope = np.nan
        slope_fit_r_squared = np.nan
        slope_fit_point_count = 0
        slope_fit_intercept = np.nan
        slope_fit_effective_count = np.nan
        weight_scale = np.nan
        fit_distances = np.array([])
        fit_angles = np.array([])
        if np.isfinite(half_probability):
            radial_distance = crossing(
                distances, panel[0, :], half_probability
            )
            annular_distance = crossing(
                angles, panel[:, 0], half_probability
            )
            if np.isfinite(radial_distance):
                ax.scatter(
                    [radial_distance],
                    [0],
                    marker="o",
                    s=45,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=5,
                )
            if np.isfinite(annular_distance):
                ax.scatter(
                    [0],
                    [annular_distance],
                    marker="D",
                    s=40,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=5,
                )

        # The slope belongs to the gray probability/random = 1 boundary, not
        # the half-maximum contour used for the two distance intercepts.
        baseline_boundary = np.array(
            [
                crossing(angles, panel[:, radius], 1.0)
                for radius in range(panel.shape[1])
            ]
        )
        baseline_radial_intercept = crossing(distances, panel[0, :], 1.0)
        valid = np.isfinite(baseline_boundary)
        if np.isfinite(baseline_radial_intercept):
            valid &= distances <= baseline_radial_intercept + float(pixel_size)

        # Select the earliest contiguous run: this follows the principal
        # red/blue lobe from the angular axis and rejects remote closed loops.
        valid_indices = np.flatnonzero(valid)
        primary_indices = np.array([], dtype=int)
        if valid_indices.size:
            angular_jump = np.abs(
                np.diff(baseline_boundary[valid_indices])
            )
            maximum_step = max(15.0, 4.0 * (angles[1] - angles[0]))
            split_points = np.flatnonzero(
                (np.diff(valid_indices) > 1) | (angular_jump > maximum_step)
            ) + 1
            segments = np.split(
                valid_indices, split_points
            )
            primary_indices = next(
                (segment for segment in segments if len(segment) >= 2),
                np.array([], dtype=int),
            )
        if primary_indices.size >= 5:
            # A connected correlation=1 contour can rise away from the
            # origin, turn around at large distance, and return as part of
            # the same loop. Fitting that entire loop can reverse the sign
            # of the visually obvious near-origin boundary. Stop at the
            # first sustained turning point while tolerating isolated
            # pixel-scale contour noise.
            boundary_run = baseline_boundary[primary_indices]
            smoothing_sigma = min(3.0, max(0.75, len(boundary_run) / 100.0))
            boundary_smooth = gaussian_filter1d(
                boundary_run, sigma=smoothing_sigma, mode="nearest"
            )
            boundary_gradient = np.gradient(boundary_smooth)
            initial_count = min(20, max(3, len(boundary_run) // 10))
            initial_trend = float(
                np.median(boundary_gradient[:initial_count])
            )
            if not np.isclose(initial_trend, 0.0):
                reversal = boundary_gradient * np.sign(initial_trend) < 0
                persistence = min(5, max(2, len(boundary_run) // 20))
                sustained = np.convolve(
                    reversal.astype(int),
                    np.ones(persistence, dtype=int),
                    mode="valid",
                )
                turning_points = np.flatnonzero(sustained == persistence)
                if turning_points.size and turning_points[0] >= 2:
                    primary_indices = primary_indices[
                        : turning_points[0] + 1
                    ]
        if primary_indices.size:
            fit_distances = distances[primary_indices]
            fit_values = baseline_boundary[primary_indices]

            # The correlation=1 boundary saturates: it climbs steeply near the
            # origin and flattens at large separation. An unweighted straight
            # line over the whole lobe is therefore dominated by the flat tail,
            # which drags the intercept off the measured boundary at d=0 (up to
            # ~7 degrees on real data, visibly landing in the blue region) and
            # biases the slope low. Weight the fit towards short distances so
            # `slope` is the near-origin tangent, which is the physically
            # meaningful quantity.
            fit_span = float(fit_distances.max() - fit_distances.min())
            if slope_weight_scale is None:
                weight_scale = SLOPE_WEIGHT_FRACTION * fit_span
            else:
                weight_scale = float(slope_weight_scale)
            if not np.isfinite(weight_scale) or weight_scale <= 0:
                # np.inf (or a non-positive scale) restores the legacy
                # unweighted fit over the full lobe.
                fit_weights = np.ones_like(fit_distances)
                weight_scale = np.inf
            else:
                fit_weights = np.exp(
                    -(fit_distances - fit_distances.min()) / weight_scale
                )

            root_weights = np.sqrt(fit_weights)
            design = (
                np.vstack([fit_distances, np.ones_like(fit_distances)]).T
                * root_weights[:, None]
            )
            fit_slope, fit_intercept = np.linalg.lstsq(
                design, fit_values * root_weights, rcond=None
            )[0]
            fit_angles = fit_intercept + fit_slope * fit_distances
            slope = float(fit_slope)
            slope_fit_intercept = float(fit_intercept)
            slope_fit_point_count = int(fit_distances.size)
            # Kish effective sample size: how many points the weighting really
            # uses, so a too-aggressive scale is visible rather than silent.
            slope_fit_effective_count = float(
                fit_weights.sum() ** 2 / np.sum(fit_weights**2)
            )
            weight_mean = float(
                np.average(fit_values, weights=fit_weights)
            )
            residual_sum_squares = float(
                np.sum(fit_weights * (fit_values - fit_angles) ** 2)
            )
            total_sum_squares = float(
                np.sum(fit_weights * (fit_values - weight_mean) ** 2)
            )
            slope_fit_r_squared = (
                1.0 - residual_sum_squares / total_sum_squares
                if total_sum_squares > 0
                else np.nan
            )

        if fit_distances.size:
            # Draw only where the weighting actually constrains the line. Past
            # ~3 decay lengths the boundary has saturated away from this
            # tangent, and extending the line there would misrepresent the fit.
            if np.isfinite(weight_scale):
                drawn = fit_distances <= (
                    fit_distances.min() + 3.0 * weight_scale
                )
                if drawn.sum() < 2:
                    drawn = np.zeros_like(fit_distances, dtype=bool)
                    drawn[: min(2, drawn.size)] = True
            else:
                drawn = np.ones_like(fit_distances, dtype=bool)
            visible_fit = drawn & (fit_angles >= 0) & (fit_angles <= 180)
            ax.plot(
                fit_distances[visible_fit],
                fit_angles[visible_fit],
                color="#ffe600",
                linestyle="-",
                linewidth=2.5,
                zorder=6,
            )
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    label="50% radial intercept",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    color="none",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    label="50% annular intercept",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#ffe600",
                    linewidth=2.5,
                    label="near-origin baseline fit",
                ),
            ],
            loc="lower right",
            fontsize=7,
            framealpha=0.82,
        )

        panel_metrics = {
            "pair": pair,
            "title": title,
            "half_probability": float(half_probability),
            "radial_distance": float(radial_distance),
            "annular_distance_degrees": float(annular_distance),
            "slope_degrees_per_unit": float(slope),
            "slope_fit_r_squared": float(slope_fit_r_squared),
            "slope_fit_point_count": slope_fit_point_count,
            "slope_fit_intercept_degrees": float(slope_fit_intercept),
            "slope_fit_effective_point_count": float(
                slope_fit_effective_count
            ),
            "slope_weight_scale": float(weight_scale),
            "slope_contour_probability": 1.0,
            "distance_units": pixel_units,
        }
        metrics.append(panel_metrics)
        if show_metrics:
            radial_text = (
                f"{radial_distance:.2f} {pixel_units}"
                if np.isfinite(radial_distance)
                else "not resolved"
            )
            annular_text = (
                f"{annular_distance:.2f} degrees"
                if np.isfinite(annular_distance)
                else "not resolved"
            )
            slope_text = (
                f"{slope:.2f} degrees/{pixel_units}"
                if np.isfinite(slope)
                else "not resolved"
            )
            ax.text(
                0.98,
                0.98,
                "50% radial distance = "
                + radial_text
                + "\n50% annular distance = "
                + annular_text
                + "\nslope = "
                + slope_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "alpha": 0.82,
                },
            )
    if image is not None:
        fig.colorbar(
            image,
            ax=[ax for ax in axes.flat if ax.get_visible()],
            label="probability / random",
        )
    if return_metrics:
        return fig, axes, metrics
    return fig, axes
