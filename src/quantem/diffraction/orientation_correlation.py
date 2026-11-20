"""Distance-angle correlations of orientation histograms."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

# 1/e decay length of the slope-fit weights, as a fraction of the fitted distance span.
SLOPE_WEIGHT_FRACTION = 0.10
# Orientations are pi-periodic, so the num_theta // 2 + 1 retained lags span 0 to 90 degrees.
MAX_RELATIVE_ANGLE_DEGREES = 90.0

_FIT_KEYS = (
    "slope_degrees_per_unit",
    "slope_fit_r_squared",
    "slope_fit_intercept_degrees",
    "slope_fit_effective_point_count",
    "slope_weight_scale",
    "slope_fit_start_distance",
    "slope_fit_stop_distance",
)
_FIT_COLOR = "#ffe600"
_CMAP_NODES = (0.0, 0.32, 0.5, 0.68, 1.0)
_CMAP_COLORS = ("#002b9a", "#1769e8", "#b8b8b8", "#f23838", "#9e0015")
_TEXT_STYLE = dict(
    ha="right",
    va="top",
    fontsize=8,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", alpha=0.82),
)
_MARKER_STYLE = {"facecolor": "white", "edgecolor": "black", "linewidth": 0.8, "zorder": 5}


def _resolve_pairs(pairs, num_radii):
    if isinstance(pairs, str):
        if pairs not in ("all", "autocorrelation"):
            raise ValueError("pairs must be 'all', 'autocorrelation', or a sequence of pairs")
        auto = pairs == "autocorrelation"
        pairs = [
            (i, j) for i in range(num_radii) for j in range(i, num_radii) if i == j or not auto
        ]
    pair_indices = np.array([[int(index) for index in pair] for pair in pairs], dtype=np.int64)
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2 or len(pair_indices) == 0:
        raise ValueError("pairs must be a non-empty sequence of (int, int) radial-bin pairs")
    if pair_indices.min() < 0 or pair_indices.max() >= num_radii:
        raise ValueError(f"radial-bin pairs must lie in [0, {num_radii})")
    return pair_indices


def _radial_geometry(size_x, size_y, radius_max, device, dtype):
    """Padded shape, and the points, bins and weights of two-point linear radial binning."""
    shape = (max(2 * size_x, 2 * radius_max), max(2 * size_y, 2 * radius_max))
    x, y = ((torch.arange(n) + n // 2) % n - n // 2 for n in shape)
    radius = torch.sqrt((x[:, None] ** 2 + y[None, :] ** 2).double()).flatten()
    lower = torch.nonzero(radius <= radius_max).flatten()
    upper = torch.nonzero(radius <= radius_max - 1).flatten()
    lower_floor, upper_floor = radius[lower].floor(), radius[upper].floor()
    points = torch.cat((lower, upper))
    bins = torch.cat((lower_floor, upper_floor + 1)).long()
    weights = torch.cat((1.0 - (radius[lower] - lower_floor), radius[upper] - upper_floor))
    return shape, points.to(device), bins.to(device), weights.to(device=device, dtype=dtype)


def calculate_orientation_correlation(
    orient_hist,
    radius_max: int | None = None,
    pairs: str | Sequence[tuple[int, int]] = "all",
    device=None,
    mode_batch_size: int = 8,
    pair_batch_size: int = 4,
    dtype: str = "float32",
    zero_policy: str = "nan",
    return_numpy: bool = True,
    progress_bar: bool = True,
):
    """Spatial-distance versus relative-angle correlation of an orientation histogram.

    Angular Fourier modes are streamed in batches through padded 2D spatial
    correlations and binned radially before the angular inverse transform, which
    equals the full 3D correlation volume at a fraction of its memory.

    Parameters
    ----------
    orient_hist : numpy.ndarray or torch.Tensor
        Shape ``(radial_bin, x, y, theta)``, or ``(x, y, theta)`` for one radial
        bin. Theta spans 0-180 degrees and is periodic.
    radius_max : int, optional
        Largest spatial separation in pixels. Defaults to half the smaller scan size.
    pairs : {"all", "autocorrelation"} or sequence of (int, int)
        Radial-bin pairs; ``"all"`` uses upper-triangular ordering.
    device : str or torch.device, optional
        Defaults to the input tensor's device, else CPU.
    mode_batch_size, pair_batch_size : int
        Angular modes and radial-bin pairs per batch.
    dtype : {"float32", "float64"}
    zero_policy : {"nan", "zero", "raise"}
        Handling of distances with no normalization signal.
    return_numpy, progress_bar : bool

    Returns
    -------
    orient_corr : numpy.ndarray or torch.Tensor
        Shape ``(num_pairs, num_theta // 2 + 1, radius_max + 1)``, in multiples
        of a random distribution (1 is random association).
    pair_indices : numpy.ndarray
        Shape ``(num_pairs, 2)``, the radial bins of each pair.
    """
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    if zero_policy not in ("nan", "zero", "raise"):
        raise ValueError("zero_policy must be 'nan', 'zero', or 'raise'")
    if mode_batch_size < 1 or pair_batch_size < 1:
        raise ValueError("mode_batch_size and pair_batch_size must be at least 1")
    if device is None:
        device = orient_hist.device if isinstance(orient_hist, torch.Tensor) else "cpu"
    real_dtype = getattr(torch, dtype)
    histogram = torch.as_tensor(orient_hist, dtype=real_dtype, device=device)
    device = histogram.device
    if histogram.ndim == 3:
        histogram = histogram[None]
    if histogram.ndim != 4:
        raise ValueError("orient_hist must have shape (x, y, theta) or (radial_bin, x, y, theta)")
    num_radii, size_x, size_y, num_theta = histogram.shape
    if min(num_radii, size_x, size_y) < 1 or num_theta < 2:
        raise ValueError("orient_hist needs a radial bin, a spatial pixel and two theta bins")
    if not torch.isfinite(histogram).all():
        raise ValueError("orient_hist contains NaN or infinite values")
    if radius_max is None:
        radius_max = int(np.ceil(min(size_x, size_y) / 2))
    elif not isinstance(radius_max, (int, np.integer)):
        raise TypeError("radius_max must be an integer or None")
    radius_max = int(radius_max)
    if radius_max < 0:
        raise ValueError("radius_max must be non-negative")

    pair_indices = _resolve_pairs(pairs, num_radii)
    pair_tensor = torch.as_tensor(pair_indices, device=device)
    shape, points, bins, weights = _radial_geometry(size_x, size_y, radius_max, device, real_dtype)
    num_pairs, num_modes = len(pair_indices), num_theta // 2 + 1
    theta_spectrum = torch.fft.rfft(histogram, dim=-1)
    spectrum = theta_spectrum.new_zeros((num_pairs, num_modes, radius_max + 1))
    num_batches = -(-num_modes // mode_batch_size) * -(-num_pairs // pair_batch_size)
    desc = f"Calculate orientation correlations ({device})"
    with tqdm(total=num_batches, desc=desc, unit="batch", disable=not progress_bar) as progress:
        for mode_start in range(0, num_modes, mode_batch_size):
            modes = slice(mode_start, mode_start + mode_batch_size)
            spatial = torch.fft.fft2(theta_spectrum[..., modes].movedim(-1, 1), s=shape)
            for pair_start in range(0, num_pairs, pair_batch_size):
                batch = pair_tensor[pair_start : pair_start + pair_batch_size]
                cross = spatial[batch[:, 0]] * spatial[batch[:, 1]].conj()
                values = torch.fft.ifft2(cross).flatten(-2)[..., points] * weights
                spectrum[pair_start : pair_start + len(batch), modes].index_add_(-1, bins, values)
                progress.update()

    correlation = torch.fft.irfft(spectrum, n=num_theta, dim=1)[:, :num_modes]
    denominator = spectrum[:, 0].real / num_theta
    finfo = torch.finfo(real_dtype)
    threshold = finfo.eps * torch.clamp(denominator.abs().max(), min=finfo.tiny)
    valid = denominator.abs() > threshold
    if zero_policy == "raise" and not valid.all():
        raise ZeroDivisionError("orientation correlation has distances with zero normalization")
    correlation = correlation / torch.where(valid, denominator, 1.0)[:, None]
    correlation = correlation.masked_fill(~valid[:, None], np.nan if zero_policy == "nan" else 0)
    if return_numpy:
        correlation = correlation.cpu().numpy()
    return correlation, pair_indices


def _crossing(coordinates, profile, level):
    """First linearly interpolated coordinate where ``profile`` crosses ``level``."""
    side = np.asarray(profile, dtype=float) - level
    if not np.isfinite(side[0]):
        return np.nan
    if side[0] == 0:
        return float(coordinates[0])
    before, after = side[:-1], side[1:]
    crosses = np.isfinite(before) & np.isfinite(after) & (np.sign(before) * np.sign(after) <= 0)
    hits = np.flatnonzero(crosses)
    if not hits.size:
        return np.nan
    i = hits[0]
    if before[i] == 0:
        return float(coordinates[i])
    fraction = before[i] / (before[i] - after[i])
    return float(coordinates[i] + fraction * (coordinates[i + 1] - coordinates[i]))


def _turning_point(boundary):
    """Number of leading points before the boundary's first sustained reversal."""
    count = len(boundary)
    sigma = min(3.0, max(0.75, count / 100.0))
    radius = int(4.0 * sigma + 0.5)  # as scipy gaussian_filter1d with mode="nearest"
    kernel = np.exp(-0.5 / sigma**2 * np.arange(-radius, radius + 1) ** 2)
    smooth = np.convolve(np.pad(boundary, radius, mode="edge"), kernel / kernel.sum(), "valid")
    gradient = np.gradient(smooth)
    trend = float(np.median(gradient[: min(20, max(3, count // 10))]))
    if np.isclose(trend, 0.0):
        return count
    persistence = min(5, max(2, count // 20))
    reversal = (gradient * np.sign(trend) < 0).astype(int)
    sustained = np.convolve(reversal, np.ones(persistence, dtype=int), mode="valid")
    turns = np.flatnonzero(sustained == persistence)
    return int(turns[0]) + 1 if turns.size and turns[0] >= 2 else count


def _slope_fit(panel, angles, distances, pixel_size, weight_scale):
    """Weighted line fit to the earliest run of the correlation = 1 boundary."""
    boundary = np.array([_crossing(angles, column, 1.0) for column in panel.T])
    valid = np.isfinite(boundary)
    radial_one = _crossing(distances, panel[0], 1.0)
    if np.isfinite(radial_one):
        valid &= distances <= radial_one + pixel_size
    run = np.flatnonzero(valid)
    if run.size:
        jumps = np.abs(np.diff(boundary[run])) > max(15.0, 4.0 * (angles[1] - angles[0]))
        segments = np.split(run, np.flatnonzero((np.diff(run) > 1) | jumps) + 1)
        run = next((segment for segment in segments if len(segment) >= 2), run[:0])
    if run.size >= 5:
        run = run[: _turning_point(boundary[run])]
    if not run.size:
        return {key: np.nan for key in _FIT_KEYS} | {"slope_fit_point_count": 0}

    d, b = distances[run], boundary[run]
    scale = SLOPE_WEIGHT_FRACTION * (d.max() - d.min()) if weight_scale is None else weight_scale
    if np.isfinite(scale) and scale > 0:
        w = np.exp(-(d - d.min()) / scale)
    else:
        w, scale = np.ones_like(d), np.inf
    root = np.sqrt(w)
    design = np.stack([d, np.ones_like(d)], axis=1) * root[:, None]
    slope, intercept = np.linalg.lstsq(design, b * root, rcond=None)[0]
    residual = np.sum(w * (b - (intercept + slope * d)) ** 2)
    total = np.sum(w * (b - np.average(b, weights=w)) ** 2)
    r_squared = 1.0 - residual / total if total > 0 else np.nan
    values = (slope, r_squared, intercept, w.sum() ** 2 / np.sum(w**2), scale, d[0], d[-1])
    return dict(zip(_FIT_KEYS, map(float, values))) | {"slope_fit_point_count": int(d.size)}


def orientation_correlation_metrics(
    orient_corr,
    pair_indices=None,
    pixel_size=1.0,
    pixel_units="scan pixels",
    slope_weight_scale=None,
) -> list[dict]:
    """50% intercepts and correlation = 1 boundary slope of each correlation panel.

    The lobe is whichever end of the angle axis (0 or 90 degrees) is larger at
    zero separation, and the 50% level is halfway from it to the random baseline
    of 1. The radial distance is where the lobe row crosses that level, and the
    annular distance is the angular offset from the lobe where the
    zero-separation column crosses it. An intercept outside the window is NaN,
    with ``*_censored`` set and ``*_lower_bound*`` at the window edge. The slope
    is an exponentially weighted line fit to the earliest contiguous run of the
    correlation = 1 boundary, stopped at its first sustained turning point.

    Parameters
    ----------
    orient_corr : array_like
        Shape ``(pair, relative_angle, distance)``, the angle axis spanning 0-90 degrees.
    pair_indices : array_like, optional
        Shape ``(pair, 2)``, used for the titles.
    pixel_size : float
        Distance per sample, in ``pixel_units``.
    pixel_units : str
    slope_weight_scale : float, optional
        1/e length of the slope-fit weights in ``pixel_units``. Defaults to
        ``SLOPE_WEIGHT_FRACTION`` of the fitted span; ``numpy.inf`` fits unweighted.

    Returns
    -------
    list of dict
        One per pair. ``slope_fit_r_squared`` is weighted and
        ``slope_fit_effective_point_count`` is the Kish effective sample size.
    """
    if orient_corr is None:
        raise RuntimeError("No orientation correlation; run calculate_orientation_correlation().")
    values = np.asarray(orient_corr, dtype=float)
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError("orient_corr must have shape (pair, relative_angle, distance).")
    if values.shape[1] < 2 or values.shape[2] < 2:
        raise ValueError("orient_corr requires at least two angle and two distance samples.")
    if pair_indices is not None and np.shape(pair_indices) != (values.shape[0], 2):
        raise ValueError(f"pair_indices must have shape ({values.shape[0]}, 2).")
    pixel_size = float(pixel_size)
    distances = np.arange(values.shape[2], dtype=float) * pixel_size
    angles = np.linspace(0.0, MAX_RELATIVE_ANGLE_DEGREES, values.shape[1])
    weight_scale = None if slope_weight_scale is None else float(slope_weight_scale)
    metrics = []
    for index, panel in enumerate(values):
        pair = (index, index) if pair_indices is None else tuple(map(int, pair_indices[index]))
        if pair_indices is None:
            title = f"Ring pair {index}"
        elif pair[0] == pair[1]:
            title = f"Autocorrelation of Ring {pair[0]}"
        else:
            title = f"Correlation of Rings {pair[0]} and {pair[1]}"
        zero = panel[:, 0]
        last_wins = np.isfinite(zero[-1]) and (not np.isfinite(zero[0]) or zero[-1] > zero[0])
        lobe = len(zero) - 1 if last_wins else 0
        origin = panel[lobe, 0]
        resolved = np.isfinite(origin) and origin > 0.0 and not np.isclose(origin, 1.0)
        half = 1.0 + 0.5 * (origin - 1.0) if resolved else np.nan
        radial = annular = np.nan
        if resolved:
            radial = _crossing(distances, panel[lobe], half)
            offsets = np.arange(len(zero), dtype=float) * float(angles[1] - angles[0])
            annular = _crossing(offsets, zero if lobe == 0 else zero[::-1], half)
        radial_censored = bool(resolved and not np.isfinite(radial))
        annular_censored = bool(resolved and not np.isfinite(annular))
        metrics.append(
            {
                "pair": pair,
                "title": title,
                "half_probability": float(half),
                "lobe_angle_degrees": float(angles[lobe]),
                "radial_distance": float(radial),
                "radial_distance_censored": radial_censored,
                "radial_distance_lower_bound": float(distances[-1]) if radial_censored else np.nan,
                "annular_distance_degrees": float(annular),
                "annular_distance_censored": annular_censored,
                "annular_distance_lower_bound_degrees": (
                    MAX_RELATIVE_ANGLE_DEGREES if annular_censored else np.nan
                ),
                "max_relative_angle_degrees": MAX_RELATIVE_ANGLE_DEGREES,
                **_slope_fit(panel, angles, distances, pixel_size, weight_scale),
                "slope_contour_probability": 1.0,
                "distance_units": pixel_units,
            }
        )
    return metrics


def _describe(value, lower_bound, units):
    if np.isfinite(value):
        return f"{value:.2f} {units}"
    return f"> {lower_bound:.2f} {units}" if np.isfinite(lower_bound) else "not resolved"


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
    """Plot distance-orientation correlations, marking the metrics of each panel.

    Circles mark the 50% radial intercepts, diamonds the 50% annular intercepts,
    and the yellow line the slope fit out to three weight lengths.

    Parameters
    ----------
    orient_corr, pixel_size, pixel_units, slope_weight_scale
        As in :func:`orientation_correlation_metrics`.
    orient_corr_pairs, pair_indices : array_like, optional
        Shape ``(pair, 2)``; ``pair_indices`` takes precedence.
    probability_range : (float, float)
        Limits of the logarithmic color scale.
    cmap : str or matplotlib.colors.Colormap
        ``"correlation"`` is a blue-gray-red diverging map centered on 1.
    figsize : (float, float), optional
    show_metrics, return_metrics : bool
        Write the metrics in each panel; also return them.

    Returns
    -------
    fig, axes[, metrics]
    """
    labels = orient_corr_pairs if pair_indices is None else pair_indices
    metrics = orientation_correlation_metrics(
        orient_corr, labels, pixel_size, pixel_units, slope_weight_scale
    )
    values = np.asarray(orient_corr, dtype=float)
    lower, upper = map(float, probability_range)
    if not 0 < lower < upper:
        raise ValueError("probability_range must satisfy 0 < lower < upper.")
    if cmap == "correlation":
        cmap = LinearSegmentedColormap.from_list(
            "quantem_correlation", list(zip(_CMAP_NODES, _CMAP_COLORS))
        )
    cols = min(3, len(metrics))
    rows = -(-len(metrics) // cols)
    figsize = figsize or (4.5 * cols, 3.8 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, constrained_layout=True)
    distances = np.arange(values.shape[2], dtype=float) * float(pixel_size)
    extent = (0, distances[-1], 0, MAX_RELATIVE_ANGLE_DEGREES)
    image_style = {"origin": "lower", "aspect": "auto", "extent": extent, "cmap": cmap}
    handle = {"color": "none", "markerfacecolor": "white", "markeredgecolor": "black"}
    legend = [
        Line2D([0], [0], marker="o", label="50% radial intercept", **handle),
        Line2D([0], [0], marker="D", label="50% annular intercept", **handle),
        Line2D([0], [0], color=_FIT_COLOR, linewidth=2.5, label="near-origin baseline fit"),
    ]
    for ax in axes.flat[len(metrics) :]:
        ax.set_visible(False)
    for ax, panel, m in zip(axes.flat, values, metrics):
        image = ax.imshow(panel, norm=LogNorm(vmin=lower, vmax=upper), **image_style)
        ax.set(title=m["title"], xlabel=f"distance ({pixel_units})")
        ax.set_ylabel("relative orientation (degrees)")
        lobe = m["lobe_angle_degrees"]
        if np.isfinite(m["radial_distance"]):
            ax.scatter([m["radial_distance"]], [lobe], marker="o", s=45, **_MARKER_STYLE)
        if np.isfinite(m["annular_distance_degrees"]):
            annular_angle = abs(lobe - m["annular_distance_degrees"])
            ax.scatter([0], [annular_angle], marker="D", s=40, **_MARKER_STYLE)
        start, stop = m["slope_fit_start_distance"], m["slope_fit_stop_distance"]
        if np.isfinite(start):
            d = distances[(distances >= start) & (distances <= stop)]
            fit = m["slope_fit_intercept_degrees"] + m["slope_degrees_per_unit"] * d
            drawn = d <= start + 3.0 * m["slope_weight_scale"]
            if drawn.sum() < 2:
                drawn = np.arange(d.size) < 2
            shown = drawn & (fit >= 0) & (fit <= MAX_RELATIVE_ANGLE_DEGREES)
            ax.plot(d[shown], fit[shown], color=_FIT_COLOR, linewidth=2.5, zorder=6)
        ax.legend(handles=legend, loc="lower right", fontsize=7, framealpha=0.82)
        if show_metrics:
            radial = _describe(m["radial_distance"], m["radial_distance_lower_bound"], pixel_units)
            annular = _describe(
                m["annular_distance_degrees"], m["annular_distance_lower_bound_degrees"], "degrees"
            )
            slope = _describe(m["slope_degrees_per_unit"], np.nan, f"degrees/{pixel_units}")
            text = f"50% radial distance = {radial}\n50% annular distance = {annular}"
            ax.text(0.98, 0.98, f"{text}\nslope = {slope}", transform=ax.transAxes, **_TEXT_STYLE)
    fig.colorbar(image, ax=list(axes.flat[: len(metrics)]), label="probability / random")
    if return_metrics:
        return fig, axes, metrics
    return fig, axes
