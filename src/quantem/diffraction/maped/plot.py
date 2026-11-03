"""Every MAPED picture.

Each function takes the MAPED object and draws one figure with ``show_2d``;
none computes anything MAPED keeps. Diffraction patterns use a square-root
scale (``power_sqrt``) so the weak spots show next to the direct beam.
"""

from collections.abc import Sequence

import numpy as np
from torch import stack

from quantem.core.visualization import show_2d
from quantem.diffraction.maped.core.align import blend_shifted_images, shift_images


def tilts(
    maped,
    plot_scale: float | Sequence[float] | None = None,
    show_detector_center: bool = True,
    bright_field_cmap: str = "gray",
    bright_field_contrast="linear_auto",
    pattern_cmap: str = "inferno",
    pattern_contrast="power_sqrt",
    **plot_kwargs,
):
    """
    Show each tilt: its bright-field image and its mean diffraction pattern.

    Parameters
    ----------
    plot_scale : float or sequence of float, optional
        Divides the bright-field images in this picture only, one value for all
        tilts or one per tilt, so tilts of different dose compare by eye.
    show_detector_center : bool
        Mark the detector center with a red cross, so each beam's offset from
        it shows at a glance.
    bright_field_cmap, pattern_cmap : str
        Matplotlib colormaps of the two columns.
    bright_field_contrast, pattern_contrast : str or dict
        Contrast as ``show_2d`` takes it: ``"linear_auto"``, ``"log_auto"``,
        ``"power_sqrt"`` (2nd to 98th percentile), or a dict such as
        ``{"lower_quantile": 0.01, "upper_quantile": 0.999, "power": 0.5}``.
    **plot_kwargs
        Passed to ``show_2d``.

    Returns
    -------
    MAPED
        The same object, so the call chains like the stages.
    """
    names = maped.tilt_names
    num_tilts = len(names)
    scales = plot_scale if isinstance(plot_scale, Sequence) else [plot_scale or 1] * num_tilts
    bright_fields = [image / scale for image, scale in zip(maped.im_bf, scales)]
    _, axes = show_2d(
        [[image, pattern] for image, pattern in zip(bright_fields, maped.dp_mean)],
        title=[[f"{name}: bright field", f"{name}: mean diffraction pattern"] for name in names],
        cmap=[[bright_field_cmap, pattern_cmap]] * num_tilts,
        norm=[[bright_field_contrast, pattern_contrast]] * num_tilts,
        returnfig=True,
        **plot_kwargs,
    )
    if show_detector_center:
        detector_rows, detector_cols = maped.dp_mean[0].shape
        for axis in np.asarray(axes, dtype=object).reshape(num_tilts, 2)[:, 1]:
            _cross(axis, detector_rows / 2, detector_cols / 2, size=14)
    return maped


def beam_centers(maped, **plot_kwargs) -> None:
    """Each mean diffraction pattern with its beam center as a red cross, four per row.

    Four per row keeps each pattern large enough to read; the title only names
    the tilt, since the cross is the result.
    """
    per_row = 4
    patterns, names = maped.dp_mean, maped.tilt_names
    _, axes = show_2d(
        [patterns[start : start + per_row] for start in range(0, len(patterns), per_row)],
        title=[names[start : start + per_row] for start in range(0, len(names), per_row)],
        cmap="inferno",
        norm="power_sqrt",
        returnfig=True,
        **plot_kwargs,
    )
    axes = np.ravel(np.asarray(axes, dtype=object))
    for axis, (row, col) in zip(axes, maped.beam_centers.tolist()):
        _cross(axis, row, col, size=16)


def aligned_patterns(maped, **plot_kwargs) -> None:
    """Mean of the patterns, each moved by its detector shift: sharp spots mean a good alignment."""
    aligned = shift_images(stack(maped.dp_mean), maped.diffraction_shifts).mean(0)
    title = f"Mean diffraction pattern, {len(maped.dp_mean)} tilts aligned on the detector"
    show_2d(aligned, title=title, cmap="inferno", norm="power_sqrt", **plot_kwargs)


def aligned_images(maped, images, edge_width: float, **plot_kwargs) -> None:
    """Blend the alignment images after applying their measured scan shifts."""
    blended = blend_shifted_images(stack(images), maped.real_space_shifts, edge_width)
    title = f"Images, {len(images)} tilts aligned in the scan"
    show_2d(blended, title=title, **plot_kwargs)


def merged(maped, **plot_kwargs) -> None:
    """The merged bright field and merged mean diffraction pattern, side by side."""
    show_2d(
        [[maped.im_bf_merged, maped.dp_mean_merged]],
        title=[["Merged bright field", "Merged mean diffraction pattern"]],
        cmap=[["gray", "inferno"]],
        norm=[[None, "power_sqrt"]],
        **plot_kwargs,
    )


def _cross(axis, row: float, col: float, size: int) -> None:
    """Why: a red cross reads at a glance on the inferno colormap, where a dot would vanish in the beam."""
    axis.plot([col], [row], marker="+", color="red", markersize=size, markeredgewidth=2)


def positions(
    maped,
    positions: Sequence[tuple[int, int]],
    *,
    tilt_indices: Sequence[int] | None = None,
):
    """Compare aligned patterns at matching specimen positions."""
    from matplotlib import pyplot as plt

    from quantem.diffraction.maped.core.merge import MergedRegions
    from quantem.diffraction.maped.io import read_region

    if getattr(maped, "merged", None) is None or maped.im_bf_merged is None:
        raise RuntimeError("Run merge() before inspecting positions.")
    positions = list(positions)
    rows, columns = maped.merged.shape[:2]
    if not positions or any(
        not (0 <= row < rows and 0 <= col < columns)
        or not isinstance(row, (int, np.integer))
        or not isinstance(col, (int, np.integer))
        for row, col in positions
    ):
        raise ValueError(f"Choose integer (row, col) positions inside {(rows, columns)}.")
    indices = list(range(len(maped.tilts))) if tilt_indices is None else list(tilt_indices)
    if any(
        not isinstance(index, (int, np.integer)) or not 0 <= index < len(maped.tilts)
        for index in indices
    ):
        raise ValueError(f"Choose tilt_indices between 0 and {len(maped.tilts) - 1}.")
    record = maped.merged.metadata["maped_merge"]
    region = record.get("scan_region", record["scan_box"])
    offset_row, _, offset_col, _ = region
    grid, markers, names = [], [], []
    for index in indices:
        source = MergedRegions(
            [maped.tilts[index]],
            maped.real_space_shifts[index : index + 1],
            maped.diffraction_shifts[index : index + 1],
            compile_merge=False,
        )
        panels = [maped.im_bf[index].detach().cpu().numpy()]
        for row, col in positions:
            scan_row, scan_col = row + offset_row, col + offset_col
            block = next(source.blocks((scan_row, scan_row + 1, scan_col, scan_col + 1)))
            panels.append(block[0].detach().cpu().numpy().copy())
        shift_row, shift_col = maped.real_space_shifts[index].detach().cpu().tolist()
        markers.append(
            [
                (row + offset_row - shift_row, col + offset_col - shift_col)
                for row, col in positions
            ]
        )
        grid.append(panels)
        names.append(maped.tilt_names[index])
    panels = [maped.im_bf_merged.detach().cpu().numpy()]
    for row, col in positions:
        block = read_region(maped.merged, (row, row + 1, col, col + 1))
        panels.append(block[0, 0].detach().cpu().numpy().copy())
    grid.append(panels)
    markers.append(positions)
    names.append("Merged")
    labels = [str(index + 1) for index in range(len(positions))]
    pattern_max = float(np.quantile(np.stack([p for row in grid for p in row[1:]]), 0.999))
    scan_limits = np.quantile(np.concatenate([row[0].ravel() for row in grid]), [0.02, 0.98])
    norms = [
        {"vmin": float(scan_limits[0]), "vmax": float(scan_limits[1])},
        *[{"vmin": 0, "vmax": pattern_max, "power": 0.5}] * len(positions),
    ]
    titles = [
        [f"{name}: scan image"]
        + [f"{name}: {label} ({row}, {col})" for label, (row, col) in zip(labels, positions)]
        for name in names
    ]
    figure, axes = show_2d(
        grid,
        title=titles,
        cmap=[["gray"] + ["inferno"] * len(positions)] * len(grid),
        norm=[norms] * len(grid),
        axsize=(3, 3),
    )
    axes = np.asarray(axes, dtype=object).reshape(len(grid), -1)
    for axis, points in zip(axes[:, 0], markers, strict=True):
        for label, (row, col) in zip(labels, points, strict=True):
            axis.plot(col, row, "+", color="cyan", markersize=8)
            axis.annotate(
                label,
                (col, row),
                xytext=(5, 5),
                textcoords="offset points",
                color="cyan",
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 1},
            )
    plt.close(figure)
    return figure
