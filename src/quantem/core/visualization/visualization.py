# from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, Union, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray

from quantem.core import config
from quantem.core.utils.utils import to_cpu
from quantem.core.visualization.custom_normalizations import (
    CustomNormalization,
    NormalizationConfig,
    _resolve_normalization,
)
from quantem.core.visualization.show_params import ShowParams
from quantem.core.visualization.visualization_utils import (
    ScalebarConfig,
    _resolve_scalebar,
    add_arg_cbar_to_ax,
    add_cbar_to_ax,
    add_scalebar_to_ax,
    array_to_rgba,
    combine_arrays_to_rgba,
)

if TYPE_CHECKING:
    from quantem.core.datastructures import Dataset2d

ArrayLike: TypeAlias = Union[NDArray, torch.Tensor, "Dataset2d"]  # union required here

# --- show_2d / grid broadcast inputs ---
# There might be a cleaner way to do this, but better to have it here than in the functions
NormInputCell: TypeAlias = NormalizationConfig | ShowParams.Norm | dict | str
Show2dNormInput: TypeAlias = (
    NormInputCell | None | Sequence[NormInputCell] | Sequence[Sequence[NormInputCell]]
)

ScalebarInputCell: TypeAlias = ScalebarConfig | ShowParams.Scalebar | dict | bool | None
Show2dScalebarInput: TypeAlias = (
    ScalebarConfig
    | ShowParams.Scalebar
    | dict
    | bool
    | None
    | Sequence[ScalebarInputCell]
    | Sequence[Sequence[ScalebarInputCell]]
)

CmapType: TypeAlias = str | colors.Colormap


def _show_2d_array(
    array: NDArray,
    *,
    norm: NormInputCell | None = None,
    scalebar: ScalebarInputCell = None,
    cmap: str | colors.Colormap = "gray",
    chroma_boost: float = 1.0,
    cbar: bool = False,
    title: str | None = None,
    figax: tuple[Any, Any] | None = None,
    figsize: tuple[int, int] = (8, 8),
    show_ticks: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Render a single 2D array (real or complex) with optional colorbar/scalebar.

    Complex data uses amplitude + phase in a perceptually uniform RGB encoding.

    Parameters
    ----------
    array : ndarray
        The 2D array to visualize. Can be real or complex.
    norm : NormalizationConfig or dict or str, optional
        Configuration for normalizing the data before visualization.
    scalebar : ScalebarConfig or dict or bool, optional
        Configuration for adding a scale bar to the plot.
    cmap : str or Colormap, default="gray"
        Colormap to use for real data or amplitude of complex data.
    chroma_boost : float, default=1.0
        Factor to boost color saturation when displaying complex data.
    cbar : bool, default=False
        Whether to add a colorbar to the plot.
    title : str, optional
        Title for the plot.
    figax : tuple, optional
        (fig, ax) tuple to use for plotting. If None, a new figure and axes are created.
    figsize : tuple, default=(8, 8)
        Figure size in inches, used only if figax is None.
    show_ticks : bool, default=False
        Whether to show axis ticks and labels.

    **kwargs : dict
        Additional keyword arguments passed to the plotting functions.
        vmin, vmax,


    Returns
    -------
    fig : Figure
        The matplotlib figure object.
    ax : Axes
        The matplotlib axes object.
    """
    is_complex = np.iscomplexobj(array)
    if is_complex:
        amplitude = np.abs(array)
        angle = np.angle(array)
    else:
        amplitude = array
        angle = None

    if array.dtype == "bool":
        array = np.array(array, dtype="float")

    norm_config = _resolve_normalization(norm, **kwargs)
    scalebar_config = _resolve_scalebar(scalebar, **kwargs)

    norm_obj = CustomNormalization(
        interval_type=norm_config.interval_type,
        stretch_type=norm_config.stretch_type,
        lower_quantile=norm_config.lower_quantile,
        upper_quantile=norm_config.upper_quantile,
        vmin=norm_config.vmin,
        vmax=norm_config.vmax,
        vcenter=norm_config.vcenter,
        half_range=norm_config.half_range,
        power=norm_config.power,
        logarithmic_index=norm_config.logarithmic_index,
        asinh_linear_range=norm_config.asinh_linear_range,
        data=amplitude,
    )

    scaled_amplitude = norm_obj(amplitude)
    rgba = array_to_rgba(scaled_amplitude, angle, cmap=cmap, chroma_boost=chroma_boost)

    if figax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    ax.imshow(rgba, interpolation=config.get("viz.interpolation"))

    if title is not None:
        ax.set_title(title, fontsize=kwargs.get("title_fontsize", 12))
    if not show_ticks:
        ax.set(xticks=[], yticks=[])

    if cbar:
        divider = make_axes_locatable(ax)
        ax_cb_abs = divider.append_axes("right", size="5%", pad="2.5%")
        # Convert cmap to Colormap if it's a string
        cmap_obj = mpl.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
        cb_abs = add_cbar_to_ax(fig, ax_cb_abs, norm_obj, cmap_obj)

        if is_complex:
            ax_cb_angle = divider.append_axes("right", size="5%", pad="15%")
            add_arg_cbar_to_ax(fig, ax_cb_angle, chroma_boost=chroma_boost)
            cb_abs.set_label("abs", rotation=0, ha="center", va="bottom")
            cb_abs.ax.yaxis.set_label_coords(0.5, -0.07)

    if scalebar_config is not None:
        add_scalebar_to_ax(
            ax,
            rgba.shape[1],
            scalebar_config.sampling,
            scalebar_config.length,
            scalebar_config.units,
            scalebar_config.width_px,
            scalebar_config.pad_px,
            scalebar_config.color,
            scalebar_config.loc,
            scalebar_config.fontsize,
            scalebar_config.bold,
        )

    for spine in ax.spines.values():  # fixes asymmetry of bbox for some reason
        spine.set_linewidth(kwargs.get("spine_linewidth", 1))

    return fig, ax


# TODO this should call _show_2d_array
def _show_2d_combined(
    list_of_arrays: Sequence[NDArray],
    *,
    norm: NormInputCell | None = None,
    scalebar: ScalebarInputCell = None,
    chroma_boost: float = 1.0,
    cbar: bool = False,
    figax: tuple[Any, Any] | None = None,
    figsize: tuple[int, int] = (8, 8),
    title: str | None = None,
    show_ticks: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Fuse multiple 2D arrays into one RGB panel (``show_2d(..., combine_images=True)``).

    This function takes a list of 2D arrays and creates a single visualization
    where each array is assigned a unique color, and their amplitudes determine
    the contribution to the final color. This is useful for comparing multiple
    related datasets.

    Parameters
    ----------
    list_of_arrays : sequence of ndarray
        Sequence of 2D arrays to combine into a single visualization.
    norm : NormalizationConfig or dict or str, optional
        Configuration for normalizing the data before visualization.
    scalebar : ScalebarConfig or dict or bool, optional
        Configuration for adding a scale bar to the plot.
    cmap : str or Colormap, default="gray"
        Base colormap to use (though each array will get a unique color).
    chroma_boost : float, default=1.0
        Factor to boost color saturation.
    cbar : bool, default=False
        Whether to add a colorbar to the plot (not yet implemented).
    figax : tuple, optional
        (fig, ax) tuple to use for plotting. If None, a new figure and axes are created.
    figsize : tuple, default=(8, 8)
        Figure size in inches, used only if figax is None.
    title : str, optional
        Title for the plot.
    show_ticks : bool, default=False
        Whether to show axis ticks and labels.

    Returns
    -------
    fig : Figure
        The matplotlib figure object.
    ax : Axes
        The matplotlib axes object.

    Raises
    ------
    NotImplementedError
        If cbar is True (colorbar for combined visualization not yet implemented).
    """
    norm_config = _resolve_normalization(norm, **kwargs)
    scalebar_config = _resolve_scalebar(scalebar)

    norm_obj = CustomNormalization(
        interval_type=norm_config.interval_type,
        stretch_type=norm_config.stretch_type,
        lower_quantile=norm_config.lower_quantile,
        upper_quantile=norm_config.upper_quantile,
        vmin=norm_config.vmin,
        vmax=norm_config.vmax,
        vcenter=norm_config.vcenter,
        half_range=norm_config.half_range,
        power=norm_config.power,
        logarithmic_index=norm_config.logarithmic_index,
        asinh_linear_range=norm_config.asinh_linear_range,
    )

    # Convert Sequence to list for list_of_arrays_to_rgba
    list_of_arrays_list = list(list_of_arrays)
    rgba = combine_arrays_to_rgba(
        list_of_arrays_list,
        norm=norm_obj,
        chroma_boost=chroma_boost,
    )

    if figax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    ax.imshow(rgba, interpolation=config.get("viz.interpolation"))

    if title is not None:
        ax.set_title(title, fontsize=kwargs.get("title_fontsize", 12))
    if not show_ticks:
        ax.set(xticks=[], yticks=[])

    if cbar:
        raise NotImplementedError()

    if scalebar_config is not None:
        add_scalebar_to_ax(
            ax,
            rgba.shape[1],
            scalebar_config.sampling,
            scalebar_config.length,
            scalebar_config.units,
            scalebar_config.width_px,
            scalebar_config.pad_px,
            scalebar_config.color,
            scalebar_config.loc,
            scalebar_config.fontsize,
            scalebar_config.bold,
        )

    return fig, ax


def _normalize_show_input_to_grid(
    arrays: Any,  # NDArray | Sequence[NDArray] | Sequence[Sequence[NDArray]]
) -> list[list[NDArray]]:
    """Convert various input formats to a consistent grid format for visualization.

    This helper function normalizes different input formats to a consistent
    grid format that can be used by the visualization functions.

    Parameters
    ----------
    arrays : ndarray or sequence of ndarray or sequence of sequences of ndarray
        Input arrays in various formats.

    Returns
    -------
    list of lists of ndarray
        Normalized grid format where each inner list represents a row of arrays.
    """
    if isinstance(arrays, np.ndarray):
        if not np.iscomplexobj(arrays):
            arrays = arrays.astype(np.float32)  # int/bool arrays can cause issues with norm
        if arrays.ndim == 2:
            return [[arrays]]
        elif arrays.ndim == 3:
            if arrays.shape[0] == 1:
                return [[arrays[0]]]
            elif arrays.shape[2] == 1:
                return [[arrays[:, :, 0]]]
        raise ValueError(f"Input array must be 2D, got shape {arrays.shape}")
    if isinstance(arrays, Sequence) and not isinstance(arrays[0], Sequence):
        # Convert sequence to list and ensure each element is an NDArray
        return [[cast(NDArray, arr) for arr in arrays]]
    # Convert outer and inner sequences to lists, ensuring proper types
    return [[cast(NDArray, arr) for arr in row] for row in arrays]


def _norm_show_args(
    args: Any,
    shape: tuple[int, int],
) -> list[list[Any]]:
    """Normalize the arguments for visualization to a grid format.

    This helper function ensures that the arguments passed to the visualization
    functions are in a consistent grid format.

    Parameters
    ----------
    args : Any
        The arguments to normalize. Can be a single value, a list of values,
        or a list of lists of values.
    shape : tuple of int, default=(1, 1)
        The desired grid shape (nrows, ncols).

    Returns
    -------
    list of lists of (str | bool)
        Normalized grid format where each inner list represents a row of arguments.

    Raises
    ------
    ValueError
        If args cannot be broadcast to the given shape.
    """
    nrows, ncols = shape

    if not args:
        return [[None for _ in range(ncols)] for _ in range(nrows)]

    if not isinstance(args, (list, tuple)):
        return [[args for _ in range(ncols)] for _ in range(nrows)]

    if not isinstance(args[0], (list, tuple)):
        # 1D sequence
        args_list = list(args)
        if len(args_list) == ncols and nrows == 1:
            return [[cast(str | bool | float, arg) for arg in args_list]]
        elif len(args_list) == nrows and ncols == 1:
            return [[cast(str | bool | float, arg)] for arg in args_list]
        elif len(args_list) == 1:
            return [
                [cast(str | bool | float, args_list[0]) for _ in range(ncols)]
                for _ in range(nrows)
            ]
        elif len(args_list) == ncols and nrows > 1:
            return [[cast(str | bool | float, arg) for arg in args_list] for _ in range(nrows)]
        elif len(args_list) == nrows and ncols > 1:
            return [[cast(str | bool | float, arg)] * ncols for arg in args_list]
        else:
            raise ValueError(
                f"Cannot broadcast 1D args of length {len(args_list)} to shape {shape}"
            )

    # 2D sequence
    args_grid = [list(row) for row in args]
    if len(args_grid) == nrows and all(len(row) == ncols for row in args_grid):
        return [[cast(str | bool | float, arg) for arg in row] for row in args_grid]
    elif len(args_grid) == 1 and len(args_grid[0]) == 1:
        return [
            [cast(str | bool | float, args_grid[0][0]) for _ in range(ncols)] for _ in range(nrows)
        ]
    elif len(args_grid) == 1 and len(args_grid[0]) == ncols:
        return [[cast(str | bool | float, arg) for arg in args_grid[0]] for _ in range(nrows)]
    elif len(args_grid) == nrows and all(len(row) == 1 for row in args_grid):
        return [[cast(str | bool | float, row[0]) for _ in range(ncols)] for row in args_grid]
    else:
        # Fill out the last row with None values if needed
        result = []
        for row in args_grid:
            row_casted = [cast(str | bool | float, arg) for arg in row]
            if len(row_casted) < ncols:
                row_casted += [None] * (ncols - len(row_casted))
            result.append(row_casted)
        return result


def _normalize_show_args_to_grid(
    shape: tuple[int, int],
    norm: Show2dNormInput = None,
    scalebar: Show2dScalebarInput = None,
    cmap: CmapType | Sequence[CmapType] | Sequence[Sequence[CmapType]] = "gray",
    cbar: bool | Sequence[bool] | Sequence[Sequence[bool]] = False,
    title: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
    chroma_boost: float | Sequence[float] = 1.0,
    show_ticks: bool | Sequence[bool] | Sequence[Sequence[bool]] = False,
) -> list[list[dict]]:
    """Normalize all show arguments to grid format and return as list of dicts."""
    norms = _norm_show_args(norm, shape)
    scalebars = _norm_show_args(scalebar, shape)
    cmaps = _norm_show_args(cmap, shape)
    chroma_boosts = _norm_show_args(chroma_boost, shape)
    cbars = _norm_show_args(cbar, shape)
    titles = _norm_show_args(title, shape)
    show_ticks_list = _norm_show_args(show_ticks, shape)

    args = [
        [
            {
                "norm": norms[i][j],
                "scalebar": scalebars[i][j],
                "cmap": cmaps[i][j],
                "chroma_boost": chroma_boosts[i][j],
                "cbar": cbars[i][j],
                "title": titles[i][j],
                "show_ticks": show_ticks_list[i][j],
            }
            for j in range(shape[1])
        ]
        for i in range(shape[0])
    ]
    return args


def show_2d(
    arrays: ArrayLike | Sequence[ArrayLike] | Sequence[Sequence[ArrayLike]],
    *,
    norm: Show2dNormInput = None,
    scalebar: Show2dScalebarInput = None,
    cmap: CmapType | Sequence[CmapType] | Sequence[Sequence[CmapType]] = "gray",
    cbar: bool | Sequence[bool] | Sequence[Sequence[bool]] = False,
    title: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
    figax: tuple[Any, Any] | None = None,
    axsize: tuple[int, int] = (4, 4),
    save: os.PathLike | str | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Display one or more 2D arrays in a grid layout.

    ``norm``, ``scalebar``, ``cmap``, ``cbar``, and ``title`` may be scalars or nested
    sequences broadcast to the panel grid.

    Parameters
    ----------
    arrays : ndarray or sequence of ndarray or sequence of sequences of ndarray
        The arrays to visualize. Can be a single array, a sequence of arrays,
        or a nested sequence representing a grid of arrays.
    norm : NormalizationConfig or dict or str, optional
        Configuration for normalizing the data before visualization. This can be a string,
        dictionary, or a NormalizationConfig object. Strings for preset normalization types
        include: ``"linear_auto"`` (quantile), ``"log_auto"``, ``"power_sqrt"``.
        Dict form example: ``{"power": 0.5}``.
    scalebar : ScalebarConfig or dict or bool, optional
        Configuration for adding a scale bar to the plot.
        Dict form example: ``{"sampling": 0.5, "units": "Å"}`` or
        ``{"sampling": 0.02, "units": "1/Å"}``.
    cmap : str or Colormap, default="gray"
        Colormap to use for real data or amplitude of complex data.
        Common choices: ``"gray"``, ``"viridis"``, ``"turbo"``, ``"inferno"``, ``"magma"``.
    cbar : bool, default=False
        Whether to add a colorbar to the plot.
    title : str, optional
        Title for the plot.
    figax : tuple, optional
        (fig, axs) tuple to use for plotting. If None, a new figure and axes are created.
    axsize : tuple, default=(4, 4)
        Size of each subplot in inches.
    save : os.PathLike or str, optional
        Path to save the figure to.

    **kwargs : dict
        Additional keyword arguments passed to _show_2d_array or _show_2d_combined:
        chroma_boost: float or sequence of float, default=1.0
            Factor to boost color saturation when displaying complex data.
        tight_layout: bool, default=True
            Whether to apply tight_layout to the figure.
        combine_images: bool, default=False
            If True, combine all arrays into a single visualization using color encoding.
            Only works for a single row of arrays.
        figsize: tuple, default None
            Size of the figure in inches. If None, calculated based on axsize and grid shape.
        show_ticks : bool, default=False
            Whether to show axis ticks and labels.
        vmin: float, optional
            Minimum value for the color scale.
        vmax: float, optional
            Maximum value for the color scale.
        lower_quantile: float, optional
            Lower quantile for the color scale.
        upper_quantile: float, optional
            Upper quantile for the color scale.

    Returns
    -------
    fig : Figure
        The matplotlib figure object.
    axs : Axes | ndarray of Axes
        - If a single image is shown, a single Axes instance.
        - If a single row or column of images is shown, a 1D ndarray of Axes.
        - If a grid of images is shown, a 2D ndarray of Axes.

    Raises
    ------
    ValueError
        If combine_images is True but arrays contains multiple rows, or if
        figax is provided but the axes shape doesn't match the grid shape.

    Examples
    --------
    Typed normalization and scale bar (same information as str/dict forms):

    >>> import numpy as np
    >>> from quantem.core.visualization import show_2d, ShowParams
    >>> img = np.random.rand(128, 128)
    >>> fig, ax = show_2d(
    ...     img,
    ...     norm=ShowParams.Norm.log_auto(),
    ...     scalebar=ShowParams.Scalebar(sampling=0.5, units="Å"),
    ...     cbar=True,
    ... )

    Preset strings and dicts:

    >>> fig, axs = show_2d(
    ...     [np.random.rand(64, 64) ** 3 for _ in range(3)],
    ...     norm="log_auto",
    ...     cmap="turbo",
    ...     title=["a", "b", "c"],
    ...     scalebar={"sampling": 0.02, "units": "1/Å"},
    ... )

    Multi-row grid with titles:

    >>> rows = [[np.random.rand(48, 48) for _ in range(3)] for _ in range(2)]
    >>> labels = [["BF", "ADF", "ABF"], ["HAADF", "DPC", "SSB"]]
    >>> fig, axs = show_2d(rows, title=labels, cmap="gray", scalebar={"sampling": 0.5, "units": "Å"})
    """
    arrays = to_cpu(arrays)
    grid = _normalize_show_input_to_grid(arrays)
    nrows = len(grid)
    ncols = max(len(row) for row in grid)

    if kwargs.pop("combine_images", False):
        if nrows > 1:
            raise ValueError()
        if isinstance(norm, Sequence):  # flatten norm
            norm = norm[0] if not isinstance(norm[0], Sequence) else norm[0][0]
        if isinstance(scalebar, Sequence):  # flatten scalebar
            scalebar = scalebar[0] if not isinstance(scalebar[0], Sequence) else scalebar[0][0]
        if isinstance(title, Sequence) and not isinstance(title, str):  # flatten title
            title = title[0] if isinstance(title[0], str) else title[0][0]

        fig, axs = _show_2d_combined(
            grid[0],
            norm=norm,
            scalebar=scalebar,
            figax=figax,
            title=title,
            figsize=kwargs.pop("figsize", axsize),
            **kwargs,
        )
    else:
        normalized_args = _normalize_show_args_to_grid(
            shape=(nrows, ncols),
            norm=norm,
            scalebar=scalebar,
            cmap=cmap,
            cbar=cbar,
            title=kwargs.pop("titles", None) if title is None else title,
            chroma_boost=kwargs.pop("chroma_boost", 1.0),
            show_ticks=kwargs.pop("show_ticks", False),
        )

        if figax is not None:
            fig, axs = figax
            if not isinstance(axs, np.ndarray):
                axs = np.array([[axs]])
            elif axs.ndim == 1:
                axs = axs.reshape(1, -1)
            if axs.shape != (nrows, ncols):
                raise ValueError()
        else:
            figsize = kwargs.pop("figsize", (axsize[0] * ncols, axsize[1] * nrows))
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

        for i, row in enumerate(grid):
            for j, array in enumerate(row):
                figax_local = (fig, axs[i][j])
                _show_2d_array(
                    array,
                    figax=figax_local,
                    **normalized_args[i][j],
                    **kwargs,
                )

        for i, row in enumerate(grid):
            for j in range(len(row), ncols):
                axs[i][j].axis("off")  # type: ignore

        # Safe layout handling: only adjust layout if we created the figure
        tight_layout = kwargs.get("tight_layout", True)
        if figax is None and tight_layout:
            only_subplots = all(
                getattr(ax, "get_subplotspec", lambda: None)() is not None for ax in fig.axes
            )
            if only_subplots:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fig.tight_layout()
            else:
                fig.subplots_adjust(
                    wspace=kwargs.get("wspace", 0.25),
                    hspace=kwargs.get("hspace", 0.25),
                )

    if kwargs.get("force_show", False):
        plt.show()

    if save is not None:
        print(f"Saving figure to {save}")
        fig.savefig(
            save,
            bbox_inches="tight",
            pad_inches=kwargs.get("pad_inches", 0),
            dpi=kwargs.get("dpi", 300),
        )

    # Normalize axs return shape for easier downstream use
    if isinstance(axs, np.ndarray):
        if axs.size == 1:
            axs_out: Any = axs[0, 0]
        elif axs.ndim == 2 and (axs.shape[0] == 1 or axs.shape[1] == 1):
            axs_out = axs.ravel()
        else:
            axs_out = axs
    else:
        axs_out = axs

    return fig, axs_out
