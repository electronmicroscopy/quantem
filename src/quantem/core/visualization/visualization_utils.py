from dataclasses import dataclass
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from colorspacious import cspace_convert
from matplotlib import cm, colors, legend, patheffects, ticker
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
    scaled_angle: NDArray | None = None,
    *,
    cmap: str | colors.Colormap = "gray",
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


def combine_arrays_to_rgba(
    list_of_arrays: list[NDArray],
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
        Units of ``sampling`` (e.g. "nm", "Å", "1/nm").
    length : float, optional
        Length of the scale bar in physical units (in ``display_units`` if
        given). If None, an appropriate length will be estimated.
    display_units : str, optional
        Units to show the scale bar in, e.g. "nm" for data calibrated in μm.
        Converts between pm, Å, nm, μm, mm, cm and m, or between their
        reciprocals (e.g. "1/nm" to "1/Å"). If None, units are chosen
        automatically from the bar length.
    width_px : float, default=1
        Width of the scale bar in pixels.
    pad_px : float, default=0.5
        Padding around the scale bar in pixels.
    color : str, default="white"
        Color of the scale bar.
    loc : str or int, default="lower right"
        Location of the scale bar on the plot. Can be a string like "lower right"
        or an integer location code.
    fontsize : int, default=12
        Font size of the scale bar label in points.
    bold : bool, default=True
        Whether to render the scale bar label in bold.
    box : bool, default=False
        Draw a translucent box behind the bar and label.
    box_color : str, default="black"
        Fill color of the box.
    box_alpha : float, default=0.5
        Opacity of the box.
    outline_color : str, optional
        Color of an outline drawn around the bar and label (via matplotlib
        path effects) so the scale bar stays readable on any background.
        If None, no outline is drawn.
    outline_width : float, default=2.0
        Width of the outline in points.
    """

    sampling: float = 1.0
    units: str = "pixels"
    length: float | None = None
    display_units: str | None = None
    width_px: float = 1
    pad_px: float = 0.5
    color: str = "white"
    loc: str | int = "lower right"
    fontsize: int = 12
    bold: bool = True
    box: bool = False
    box_color: str = "black"
    box_alpha: float = 0.5
    outline_color: str | None = None
    outline_width: float = 2.0


SCALEBAR_KWARGS = [
    "sampling",
    "units",
]


def _resolve_scalebar(cfg: Any, **kwargs) -> ScalebarConfig | None:
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
    elif hasattr(cfg, "to_config"):
        return cfg.to_config()
    else:
        raise TypeError(
            "scalebar must be None, dict, bool, ScalebarConfig, or ShowParams.Scalebar"
        )


def _fill_scalebar_calibration(cfg: Any, sampling: float, units: str) -> Any:
    """Fill in a scale bar's sampling and units from a dataset's calibration.

    Used when a dataset (e.g. ``Dataset2d``) is shown, so the scale bar picks up
    the dataset's calibration unless the user set it explicitly.

    - ``True`` becomes a ``ScalebarConfig`` with the dataset's sampling and units.
    - A dict gets ``sampling`` and ``units`` added if they are missing.
    - A ``ScalebarConfig`` or ``ShowParams.Scalebar`` still on the defaults
      (``sampling=1.0`` and ``units="pixels"``) gets the dataset's values.
    - ``None`` and ``False`` (no scale bar) are returned unchanged.
    """
    from dataclasses import replace

    if cfg is None or cfg is False:
        return cfg
    if cfg is True:
        return ScalebarConfig(sampling=sampling, units=units)
    if isinstance(cfg, dict):
        if "sampling" in cfg or "units" in cfg:
            return cfg
        return {**cfg, "sampling": sampling, "units": units}
    resolved = _resolve_scalebar(cfg)
    if resolved is not None and resolved.sampling == 1.0 and resolved.units == "pixels":
        return replace(resolved, sampling=sampling, units=units)
    return resolved


def estimate_scalebar_length(length: float, sampling: float) -> tuple[float, float]:
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


# Length units the scale bar knows how to convert, as multiples of 1 Å.
_LENGTH_UNITS_IN_ANGSTROM = {
    "pm": 1e-2,
    "Å": 1.0,
    "nm": 1e1,
    "μm": 1e4,
    "mm": 1e7,
    "cm": 1e8,
    "m": 1e10,
}

# Accepted spellings (compared in lower case) for each unit above.
_LENGTH_UNIT_ALIASES = {
    **dict.fromkeys(["pm", "picometer", "picometre", "picometers", "picometres"], "pm"),
    **dict.fromkeys(
        ["a", "å", "Å", "ang", "angstrom", "angstroms", "ångström", "ångströms"], "Å"
    ),
    **dict.fromkeys(["nm", "nanometer", "nanometre", "nanometers", "nanometres"], "nm"),
    **dict.fromkeys(
        [
            "um",
            "μm",  # Greek mu
            "µm",  # micro sign
            "micrometer",
            "micrometre",
            "micrometers",
            "micrometres",
            "micron",
            "microns",
        ],
        "μm",
    ),
    **dict.fromkeys(["mm", "millimeter", "millimetre", "millimeters", "millimetres"], "mm"),
    **dict.fromkeys(["cm", "centimeter", "centimetre", "centimeters", "centimetres"], "cm"),
    **dict.fromkeys(["m", "meter", "metre", "meters", "metres"], "m"),
}

_RECIPROCAL_SUFFIXES = ("^-1", "**-1", "⁻¹", "-1")


def _parse_length_units(units: Any) -> tuple[str, int] | None:
    """Parse a length unit string into (canonical unit, power).

    Power is 1 for real-space lengths (e.g. ``"nm"``) and -1 for reciprocal
    lengths (e.g. ``"1/nm"``, ``"nm^-1"``, ``"nm⁻¹"``, ``"$\\mathrm{nm}^{-1}$"``).
    Returns None if the units are not a recognised length (e.g. ``"pixels"``).
    """
    if not isinstance(units, str):
        return None
    u = units.strip().replace("$", "").replace("\\mathrm", "").replace(" ", "")
    u = u.replace("{", "").replace("}", "")
    power = 1
    if u.startswith("1/"):
        u, power = u[2:], -1
    else:
        for suffix in _RECIPROCAL_SUFFIXES:
            if u.endswith(suffix):
                u, power = u[: -len(suffix)], -1
                break
    base = _LENGTH_UNIT_ALIASES.get(u.lower())
    return (base, power) if base is not None else None


def _length_unit_label(base: str, power: int) -> str:
    """Display label for a parsed unit, e.g. ("nm", -1) -> nm⁻¹ in mathtext."""
    return base if power == 1 else f"$\\mathrm{{{base}}}^{{-1}}$"


def _length_unit_factor(from_units: str, to_units: str) -> float:
    """Factor that converts a value in ``from_units`` to ``to_units``.

    Works for real-space lengths (pm, Å, nm, μm, mm, cm, m) and their
    reciprocals; e.g. ``_length_unit_factor("μm", "nm") == 1000`` and
    ``_length_unit_factor("1/nm", "1/Å") == 0.1``.

    Raises
    ------
    ValueError
        If either unit is not a recognised length, or one is a real-space
        length and the other a reciprocal length.
    """
    src = _parse_length_units(from_units)
    dst = _parse_length_units(to_units)
    known = ", ".join(_LENGTH_UNITS_IN_ANGSTROM)
    if dst is None:
        raise ValueError(
            f"Unknown display units {to_units!r}. Use one of {known}, or a reciprocal "
            "such as '1/nm' or 'Å^-1'."
        )
    if src is None:
        raise ValueError(
            f"Cannot convert scale bar units {from_units!r} to {to_units!r}: "
            f"{from_units!r} is not a recognised length unit ({known})."
        )
    if src[1] != dst[1]:
        kinds = {1: "a real-space length", -1: "a reciprocal length"}
        raise ValueError(
            f"Cannot convert {from_units!r} ({kinds[src[1]]}) to {to_units!r} "
            f"({kinds[dst[1]]})."
        )
    ratio = _LENGTH_UNITS_IN_ANGSTROM[src[0]] / _LENGTH_UNITS_IN_ANGSTROM[dst[0]]
    return ratio if src[1] == 1 else 1 / ratio


def _normalize_length_units(length_units: float, units: str) -> tuple[float, str]:
    """
    Pick intelligent units for the scalebar length. Handles both direct and inverse units.
    """
    parsed = _parse_length_units(units)
    if parsed is None:
        # unknown unit (e.g. "pixels"): return as is
        return length_units, units
    base, power = parsed

    if power == -1:
        length_invA = length_units / _LENGTH_UNITS_IN_ANGSTROM[base]
        if length_invA < 0.1:
            return length_invA * 10, _length_unit_label("nm", -1)
        elif length_invA < 100:
            return length_invA, _length_unit_label("Å", -1)
        else:
            return length_invA / 100, _length_unit_label("pm", -1)

    length_A = length_units * _LENGTH_UNITS_IN_ANGSTROM[base]
    if length_A <= 0.1:
        return length_A * 100, "pm"
    elif length_A < 10:
        return length_A, "Å"
    elif length_A < 1e4:
        return length_A / 10, "nm"
    elif length_A < 1e7:
        return length_A / 1e4, "μm"
    elif length_A < 1e10:
        return length_A / 1e7, "mm"
    else:
        return length_A / 1e10, "m"


def add_scalebar_to_ax(
    ax: Axes,
    array_size: float,
    sampling: float,
    length_units: float | None,
    units: str,
    width_px: float,
    pad_px: float,
    color: str,
    loc: str | int,
    fontsize: int = 12,
    bold: bool = True,
    box: bool = False,
    box_color: str = "black",
    box_alpha: float = 0.5,
    outline_color: str | None = None,
    outline_width: float = 2.0,
    display_units: str | None = None,
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
    fontsize : int
        Font size of the scale bar label in points.
    bold : bool
        Whether to render the scale bar label in bold.
    box : bool, default=False
        Draw a translucent box behind the bar and label so it stays
        readable on any image (e.g. white bar on a black box, or black bar
        on a white box). Padding follows ``pad_px``.
    box_color : str, default="black"
        Fill color of the box.
    box_alpha : float, default=0.5
        Opacity of the box.
    outline_color : str, optional
        Color of an outline drawn around the bar and label using
        matplotlib path effects. If None, no outline is drawn.
    outline_width : float, default=2.0
        Width of the outline in points.
    display_units : str, optional
        Units to show the scale bar in (e.g. "nm" when ``units`` is "μm", or
        "1/Å" when ``units`` is "1/nm"). ``sampling`` is converted to these
        units and ``length_units``, if given, is taken to be in them. If None,
        units are chosen automatically from the bar length.
    """
    from matplotlib.font_manager import FontProperties

    if display_units is not None:
        if isinstance(sampling, (tuple, list)):
            sampling = np.asarray(sampling, dtype=float)
        sampling = sampling * _length_unit_factor(units, display_units)
        units = _length_unit_label(*cast(tuple, _parse_length_units(display_units)))

    if length_units is None:
        length_units, length_px = estimate_scalebar_length(array_size, sampling)
    else:
        length_px = length_units / sampling

    if display_units is None:
        length_units, units = _normalize_length_units(length_units, units)

    if length_units % 1 == 0.0:
        label = f"{length_units:.0f} {units}"
    elif length_units >= 0.1:
        label = f"{length_units:.2f} {units}"
    else:  # e.g. 0.005 would round to "0.01" with .2f
        label = f"{length_units:g} {units}"

    if isinstance(loc, int):
        loc_codes = legend.Legend.codes
        loc_strings = {v: k for k, v in loc_codes.items()}
        loc = loc_strings[loc]

    fontprops = FontProperties(size=fontsize, weight="bold" if bold else "normal")

    label_top = loc[:3] == "low"
    bar = AnchoredSizeBar(
        ax.transData,
        length_px,
        label,
        loc,
        pad=pad_px,
        color=color,
        frameon=box,
        label_top=label_top,
        size_vertical=int(width_px),
        fontproperties=fontprops,
        sep=2 if label_top else int(round(0.3 * fontsize)),
    )
    if box:
        bar.patch.set(facecolor=box_color, edgecolor="none", alpha=box_alpha)
    if outline_color is not None:
        effects = [
            patheffects.withStroke(linewidth=outline_width, foreground=outline_color)
        ]
        for artist in bar.size_bar.get_children():
            artist.set_path_effects(effects)
        for artist in bar.txt_label.get_children():
            artist.set_path_effects(effects)
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
    # only fall back to a shared exponent for genuinely extreme ranges: a
    # correlation running 0 to 0.8 should read 0.0, 0.2, ... not 0, 2 x 10^-1
    formatter.set_powerlimits((-3, 4))

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
    cb_angle.ax.yaxis.set_label_coords(0.5, -0.07)
    cb_angle.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb_angle.set_ticklabels(
        [r"$-\pi$", r"$-\dfrac{\pi}{2}$", "$0$", r"$\dfrac{\pi}{2}$", r"$\pi$"]
    )

    return cb_angle


def turbo_black(num_colors: int = 256, fade_len: int | None = None) -> colors.ListedColormap:
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
    shape: tuple[int, int],
    x: NDArray,
    y: NDArray,
    weight: NDArray,
    origin: tuple[float, float] = (0.0, 0.0),
    sampling: tuple[float, float] = (1.0, 1.0),
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

    x = _as_histogram_vector(x, "x")
    y = _as_histogram_vector(y, "y")
    weight = _as_histogram_vector(weight, "weight")
    if not (x.shape == y.shape == weight.shape):
        raise ValueError(
            f"x, y, and weight must have matching shapes after coercion, got {x.shape}, {y.shape}, {weight.shape}"
        )

    # Convert shape tuple to list for binned_statistic_2d
    bins: list[int] = [Nx, Ny]
    hist, _, _, _ = binned_statistic_2d(
        x,
        y,
        values=weight,
        statistic=statistic,
        bins=bins,  # type: ignore[arg-type]  # scipy's type hints are incorrect
        range=[[x0, x1], [y0, y1]],  # [[x_min, x_max], [y_min, y_max]]
    )

    return hist  # shape = (Nx, Ny), i.e., array[x, y]


def _as_histogram_vector(value: NDArray, name: str) -> NDArray:
    array = np.asarray(value)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and 1 in array.shape:
        return array.reshape(-1)
    raise ValueError(f"{name} must be 1D or shape (N, 1)/(1, N), got {array.shape}")


def axes_with_inset(
    axsize=(4, 4),
    ax_size_embed=None,  # None -> 0.25 of main axes in each dimension (fractional)
    loc="upper right",  # "upper left"|"upper right"|"lower left"|"lower right" or "ul|ur|ll|lr"
    borderpad=0,  # keep 0 for perfectly flush
    inset_facecolor="black",  # only the inset background is black to hide any seam
    hide_inset_spines=True,  # remove inset spines to avoid edge lines
):
    """
    Create a figure with one main axes and a flush inset axes overlaid on it.
    - Fractional inset by default (relative to main axes size).
    - Only the inset axes background is set to black (main axes stays default).
    """
    fig, ax_main = plt.subplots(1, 1, figsize=axsize)

    # lazy import here (some environments need it this way)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # normalize location string
    loc_map = {"ul": "upper left", "ur": "upper right", "ll": "lower left", "lr": "lower right"}
    loc_norm = loc_map.get(str(loc).lower(), loc).lower()
    if loc_norm not in {"upper left", "upper right", "lower left", "lower right"}:
        raise ValueError(
            "loc must be one of: upper left|upper right|lower left|lower right|ul|ur|ll|lr"
        )

    # resolve inset size: percent strings (fractions) or inches (float > 1)
    def _size_spec(embed):
        if embed is None:
            return "25%", "25%"
        if isinstance(embed, (int, float)):
            v = float(embed)
            return (f"{v * 100:.6g}%", f"{v * 100:.6g}%") if v <= 1.0 else (v, v)
        if len(embed) != 2:
            raise ValueError("ax_size_embed must be None, a scalar, or a (w, h) tuple.")
        w, h = float(embed[0]), float(embed[1])
        w_spec = f"{w * 100:.6g}%" if w <= 1.0 else w
        h_spec = f"{h * 100:.6g}%" if h <= 1.0 else h
        return w_spec, h_spec

    w_spec, h_spec = _size_spec(ax_size_embed)

    # inset, flush to chosen corner of the main axes (no figure background changes)
    ax_inset = inset_axes(
        ax_main,
        width=w_spec,
        height=h_spec,
        loc=loc_norm,
        borderpad=borderpad,  # 0 => perfectly flush
        bbox_to_anchor=(0, 0, 1, 1),  # anchor in main-axes coords
        bbox_transform=ax_main.transAxes,
    )
    ax_inset.set_zorder(ax_main.get_zorder() + 1)
    ax_inset.set_facecolor(inset_facecolor)
    ax_inset.patch.set_alpha(1.0)

    if hide_inset_spines:
        for spine in ax_inset.spines.values():
            spine.set_visible(False)

    return fig, [ax_main, ax_inset]
