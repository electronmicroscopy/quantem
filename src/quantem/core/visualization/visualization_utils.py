from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import matplotlib as mpl
import numpy as np
from colorspacious import cspace_convert
from matplotlib import cm, colors, legend, ticker
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from numpy.typing import NDArray
from scipy.stats import binned_statistic_2d

from quantem.core import config
from quantem.core.visualization.custom_normalizations import CustomNormalization


def array_to_rgba(
    scaled_amplitude: NDArray,
    scaled_angle: Optional[NDArray] = None,
    *,
    cmap: Union[str, colors.Colormap] = "gray",
    chroma_boost: float = 1,
) -> NDArray:
    """Convert amplitude and angle arrays to an RGBA color array.

    This function creates a color representation of data using either a simple colormap
    or a perceptually-uniform color space based on amplitude and angle information.

    Parameters
    ----------
    scaled_amplitude : np.ndarray
        Array of amplitude values, typically normalized to [0, 1].
    scaled_angle : np.ndarray, optional
        Array of angle values in radians. If provided, creates a color representation
        using the JCh color space where amplitude controls lightness and angle controls hue.
    cmap : str or mpl.colors.Colormap, default="gray"
        Colormap to use when scaled_angle is None.
    chroma_boost : float, default=1
        Factor to boost color saturation when using angle-based coloring.

    Returns
    -------
    np.ndarray
        RGBA array with shape (height, width, 4) where the last dimension contains
        (red, green, blue, alpha) values in the range [0, 1].

    Raises
    ------
    ValueError
        If scaled_angle is provided but has a different shape than scaled_amplitude.
    """
    cmap_obj = cmap if isinstance(cmap, colors.Colormap) else mpl.colormaps.get_cmap(cmap)
    if scaled_angle is None:
        rgba = cmap_obj(scaled_amplitude)
    else:
        if scaled_angle.shape != scaled_amplitude.shape:
            raise ValueError()

        J = scaled_amplitude * 61.5
        C = np.minimum(chroma_boost * 98 * J / 123, 110)
        h = np.rad2deg(scaled_angle) + 180

        JCh = np.stack((J, C, h), axis=-1)
        with np.errstate(invalid="ignore"):
            rgb = cspace_convert(JCh, "JCh", "sRGB1").clip(0, 1)

        alpha = np.ones_like(scaled_amplitude)
        rgba = np.dstack((rgb, alpha))

    return rgba


def list_of_arrays_to_rgba(
    list_of_arrays: List[NDArray],
    *,
    norm: CustomNormalization = CustomNormalization(),
    chroma_boost: float = 1,
) -> NDArray:
    """Converts a list of arrays to a perceptually-uniform RGB array.

    This function takes multiple arrays and creates a color representation where each
    array is assigned a unique hue angle, and the amplitude of each array determines
    the contribution to the final color. The result is a perceptually-uniform color
    representation that can effectively visualize multiple data sources simultaneously.

    Parameters
    ----------
    list_of_arrays : list of np.ndarray
        List of arrays to be converted to a color representation. All arrays must have
        the same shape.
    norm : CustomNormalization, default=CustomNormalization()
        Normalization to apply to each array before processing.
    chroma_boost : float, default=1
        Factor to boost color saturation in the final output.

    Returns
    -------
    np.ndarray
        RGBA array with shape (height, width, 4) representing the combined data.
    """
    list_of_arrays = [norm(array) for array in list_of_arrays]
    bins = np.asarray(list_of_arrays)
    n = bins.shape[0]

    # circular encoding
    hue_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    hue_angles += np.linspace(0.0, 0.5, n) * (2 * np.pi / n / 2)  # jitter to avoid cancellation
    complex_weights = np.exp(1j * hue_angles)[:, None, None] * bins

    # weighted average direction (w/ normalization)
    complex_sum = complex_weights.sum(0)
    scaled_amplitude = np.clip(np.abs(complex_sum), 0, 1)
    scaled_angle = np.angle(complex_sum)

    return array_to_rgba(scaled_amplitude, scaled_angle, chroma_boost=chroma_boost)


@dataclass
class ScalebarConfig:
    """Configuration for adding a scale bar to a plot.

    Attributes
    ----------
    sampling : float, default=1.0
        Physical units per pixel.
    units : str, default="pixels"
        Units to display on the scale bar.
    length : float, optional
        Length of the scale bar in physical units. If None, an appropriate length
        will be estimated.
    width_px : float, default=1
        Width of the scale bar in pixels.
    pad_px : float, default=0.5
        Padding around the scale bar in pixels.
    color : str, default="white"
        Color of the scale bar.
    loc : str or int, default="lower right"
        Location of the scale bar on the plot. Can be a string like "lower right"
        or an integer location code.
    """

    sampling: float = 1.0
    units: str = "pixels"
    length: Optional[float] = None
    width_px: float = 1
    pad_px: float = 0.5
    color: str = "white"
    loc: Union[str, int] = "lower right"


SCALEBAR_KWARGS = [
    "sampling",
    "units",
]


def _resolve_scalebar(cfg: Any, **kwargs) -> Optional[ScalebarConfig]:
    """Resolve various input types to a ScalebarConfig object.

    Parameters
    ----------
    cfg : None, bool, dict, or ScalebarConfig
        Configuration for the scale bar.

    Returns
    -------
    ScalebarConfig or None
        Resolved configuration object or None if cfg is None or False.

    Raises
    ------
    TypeError
        If cfg is not one of the supported types.
    """
    if cfg is None:
        scalebar_kwargs = {k: kwargs[k] for k in SCALEBAR_KWARGS if k in kwargs}
        if scalebar_kwargs:
            if "sampling" in scalebar_kwargs and "units" not in scalebar_kwargs:
                scalebar_kwargs["units"] = config.get("viz.real_space_units")
            return ScalebarConfig(**scalebar_kwargs)
        else:
            return None
    elif cfg is False:
        return None
    elif cfg is True:
        return ScalebarConfig()
    elif isinstance(cfg, dict):
        return ScalebarConfig(**cfg)
    elif isinstance(cfg, ScalebarConfig):
        return cfg
    else:
        raise TypeError("scalebar must be None, dict, bool, or ScalebarConfig")


def estimate_scalebar_length(length: float, sampling: float) -> Tuple[float, float]:
    """Estimate an appropriate scale bar length based on data dimensions.

    This function calculates a "nice" scale bar length that is a multiple of
    0.5, 1.0, or 2.0 times a power of 10, depending on the data range.

    Parameters
    ----------
    length : float
        Total length of the data in physical units.
    sampling : float
        Physical units per pixel.

    Returns
    -------
    tuple
        (length_units, length_pixels) where length_units is the estimated
        scale bar length in physical units and length_pixels is the equivalent
        in pixels.
    """
    if isinstance(sampling, (tuple, list, np.ndarray)):
        if not np.allclose(sampling, sampling[0], atol=1e-2):
            raise ValueError("Sampling must be a single value or uniform across dimensions.")
        sampling = sampling[0]

    d = length * sampling / 2
    exp = np.floor(np.log10(d))
    base = d / (10**exp)
    if base >= 1 and base < 2.1:
        spacing = 0.5
    elif base >= 2.1 and base < 4.6:
        spacing = 1.0
    elif base >= 4.6 and base <= 10:
        spacing = 2.0
    else:
        spacing = 1.0  # default case
    spacing = spacing * 10**exp
    return spacing, spacing / sampling


def _normalize_length_units(length_units: float, units: str) -> tuple[float, str]:
    """
    pick intelligent units for the scalebar length
    """
    if units in ["A", "Å", "angstrom", "Angstrom"]:
        length_A = length_units
    elif units in ["nm", "nanometer", "nanometre"]:
        length_A = length_units * 10
    elif units in ["um", "μm", "micrometer", "micrometre"]:
        length_A = length_units * 1e4
    elif units in ["mm", "millimeter", "millimetre"]:
        length_A = length_units * 1e7
    elif units in ["cm", "centimeter", "centimetre"]:
        length_A = length_units * 1e8
    else:
        return length_units, units

    if length_A < 0.1:
        return length_A * 100, "pm"
    elif length_A < 10:
        return length_A, "Å"
    elif length_A < 3e4:
        return length_A / 10, "nm"
    elif length_A < 1e7:
        return length_A / 1e4, "μm"
    else:
        return length_A / 1e7, "mm"


def add_scalebar_to_ax(
    ax: Axes,
    array_size: float,
    sampling: float,
    length_units: Optional[float],
    units: str,
    width_px: float,
    pad_px: float,
    color: str,
    loc: Union[str, int],
) -> None:
    """Add a scale bar to a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the scale bar to.
    array_size : float
        Size of the data array in pixels.
    sampling : float
        Physical units per pixel.
    length_units : float, optional
        Length of the scale bar in physical units. If None, an appropriate length
        will be estimated.
    units : str
        Units to display on the scale bar.
    width_px : float
        Width of the scale bar in pixels.
    pad_px : float
        Padding around the scale bar in pixels.
    color : str
        Color of the scale bar.
    loc : str or int
        Location of the scale bar on the plot.
    """
    if length_units is None:
        length_units, length_px = estimate_scalebar_length(array_size, sampling)
    else:
        length_px = length_units / sampling

    length_units, units = _normalize_length_units(length_units, units)

    if length_units % 1 == 0.0:
        label = f"{length_units:.0f} {units}"
    else:
        label = f"{length_units:.2f} {units}"

    if isinstance(loc, int):
        loc_codes = legend.Legend.codes
        loc_strings = {v: k for k, v in loc_codes.items()}
        loc = loc_strings[loc]

    bar = AnchoredSizeBar(
        ax.transData,
        length_px,
        label,
        loc,
        pad=pad_px,
        color=color,
        frameon=False,
        label_top=loc[:3] == "low",
        size_vertical=int(width_px),  # Convert to int as required by AnchoredSizeBar
    )
    ax.add_artist(bar)


def add_cbar_to_ax(
    fig: Figure,
    cax: Axes,
    norm: colors.Normalize,
    cmap: colors.Colormap,
    eps: float = 1e-8,
) -> Colorbar:
    """Add a colorbar to a matplotlib figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to add the colorbar to.
    cax : matplotlib.axes.Axes
        The axes to place the colorbar in.
    norm : matplotlib.colors.Normalize
        The normalization for the colormap.
    cmap : matplotlib.colors.Colormap
        The colormap to use.
    eps : float, default=1e-8
        Small value to avoid floating point errors when filtering ticks.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The created colorbar object.
    """
    tick_locator = ticker.AutoLocator()
    vmin = cast(float, norm.vmin)  # Cast to float since we know it can't be None
    vmax = cast(float, norm.vmax)  # Cast to float since we know it can't be None
    ticks = tick_locator.tick_values(vmin, vmax)
    # Convert to numpy array for boolean indexing
    ticks_arr = np.asarray(ticks)
    mask = (ticks_arr >= vmin - eps) & (ticks_arr <= vmax + eps)
    ticks = ticks_arr[mask]

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, ticks=ticks, format=formatter)
    # set tick positions, fixes bug of gap between image and bbox
    for label in cb.ax.get_yticklabels():
        label.set_verticalalignment("center")
        label.set_horizontalalignment("left")

    return cb


def add_arg_cbar_to_ax(
    fig: Figure,
    cax: Axes,
    chroma_boost: float = 1,
) -> Colorbar:
    """Add a colorbar for phase values to a matplotlib figure.

    This function creates a colorbar suitable for displaying phase values.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to add the colorbar to.
    cax : matplotlib.axes.Axes
        The axes to place the colorbar in.
    chroma_boost : float, default=1
        Factor to boost color saturation.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The created colorbar object.
    """
    h = np.linspace(0, 360, 256, endpoint=False)
    J = np.full_like(h, 61.5)
    C = np.full_like(h, np.minimum(49 * chroma_boost, 110))
    JCh = np.stack((J, C, h), axis=-1)
    rgb_vals = cspace_convert(JCh, "JCh", "sRGB1").clip(0, 1)

    angle_cmap = colors.ListedColormap(rgb_vals)
    angle_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    sm = cm.ScalarMappable(norm=angle_norm, cmap=angle_cmap)
    cb_angle = fig.colorbar(sm, cax=cax)

    cb_angle.set_label("arg", rotation=0, ha="center", va="bottom")
    cb_angle.ax.yaxis.set_label_coords(0.5, -0.05)
    cb_angle.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb_angle.set_ticklabels(
        [r"$-\pi$", r"$-\dfrac{\pi}{2}$", "$0$", r"$\dfrac{\pi}{2}$", r"$\pi$"]
    )

    return cb_angle


def turbo_black(num_colors: int = 256, fade_len: Optional[int] = None) -> colors.ListedColormap:
    """Create a modified version of the 'turbo' colormap that fades to black.

    This function creates a colormap based on the 'turbo' colormap but with
    the beginning portion fading to black, which can be useful for visualizing
    data with a clear zero point.

    Parameters
    ----------
    num_colors : int, default=256
        Number of colors in the colormap.
    fade_len : int, optional
        Number of colors to fade to black at the beginning. If None, defaults
        to num_colors // 8.

    Returns
    -------
    matplotlib.colors.ListedColormap
        The modified colormap.
    """
    if fade_len is None:
        fade_len = num_colors // 8
    turbo = mpl.colormaps.get_cmap("turbo").resampled(num_colors)
    colors_array = turbo(np.linspace(0, 1, num_colors))
    fade = np.linspace(0, 1, fade_len)[:, None]
    colors_array[:fade_len, :3] *= fade
    return colors.ListedColormap(colors_array)


_turbo_black = turbo_black()
try:
    mpl.colormaps.register(_turbo_black, name="turbo_black")
    mpl.colormaps.register(_turbo_black.reversed(), name="turbo_black_r")
except ValueError:
    # If the colormap is already registered, we can ignore the error.
    pass


def bilinear_histogram_2d(
    shape: Tuple[int, int],
    x: NDArray,
    y: NDArray,
    weight: NDArray,
    origin: Tuple[float, float] = (0.0, 0.0),
    sampling: Tuple[float, float] = (1.0, 1.0),
    statistic: str = "sum",
) -> NDArray:
    """Create a 2D histogram with bilinear binning.

    This function creates a 2D histogram where data points are distributed
    across bins according to their position relative to bin centers, allowing
    for smoother visualizations than standard histograms.

    Parameters
    ----------
    shape : tuple
        (Nx, Ny) shape of the output histogram.
    x : array-like
        x-coordinates of the data points.
    y : array-like
        y-coordinates of the data points.
    weight : array-like
        Weights for each data point.
    origin : tuple, default=(0.0, 0.0)
        (x0, y0) origin of the histogram grid.
    sampling : tuple, default=(1.0, 1.0)
        (dx, dy) sampling intervals.
    statistic : str, default="sum"
        Statistic to compute for each bin. Options include "sum", "mean", "count", etc.

    Returns
    -------
    np.ndarray
        2D histogram array with shape (Nx, Ny).
    """
    Nx, Ny = shape
    dx, dy = sampling
    x0, y0 = origin
    x1, y1 = x0 + Nx * dx, y0 + Ny * dy

    # Convert shape tuple to list for binned_statistic_2d
    bins: Sequence[int] = [Nx, Ny]
    hist, _, _, _ = binned_statistic_2d(
        x,
        y,
        values=weight,
        statistic=statistic,
        bins=bins,  # type: ignore[arg-type]  # scipy's type hints are incorrect
        range=[[x0, x1], [y0, y1]],  # [[x_min, x_max], [y_min, y_max]]
    )

    return hist  # shape = (Nx, Ny), i.e., array[x, y]
