"""Memory-efficient distance-angle correlations for orientation histograms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
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
            free_memory, _ = torch.cuda.mem_get_info(device)
            complex_bytes = 8 if complex_dtype == torch.complex64 else 16
            padded_pixels = int(np.prod(geometry["padded_shape"]))
            memory_budget = int(free_memory * max_memory_fraction)

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
                required_gib = bytes_per_mode / 1024**3
                budget_gib = memory_budget / 1024**3
                raise MemoryError(
                    "A single angular-mode batch is estimated to require "
                    f"{required_gib:.2f} GiB, but the configured CUDA memory "
                    f"budget is {budget_gib:.2f} GiB. Reduce radius_max, the "
                    "orientation-histogram upsampling, or pair_batch_size."
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
