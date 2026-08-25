"""Standalone plot functions for :class:`~quantem.imaging.drift.DriftCorrection`.

Public functions take ``self`` as their first argument because the class binds
them as methods, for example
``DriftCorrection.plot_combined = plot.plot_combined``, so orchestration stays
in ``correction.py`` and visualization stays here.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

import quantem.imaging.drift.apply as drift_apply
import quantem.imaging.drift.fourdstem as fourdstem
from quantem.core.visualization import show_2d
from quantem.imaging.drift.core import knots as drift_knots
from quantem.imaging.drift.core.warping import ensure_warped_images


def show_after_step(
    correction,
    label: str,
    *,
    show_combined: bool,
    show_scans: bool,
    show_knots: bool,
):
    """Show the requested quality-control views after one correction stage."""
    stage = _stage_description(label)
    if show_combined:
        correction.plot_combined(
            show_knots=show_knots,
            rgb=True,
            title=f"Combined: {stage}",
        )
    if show_scans:
        titles = []
        for image_index in range(len(correction.scan_direction_degrees)):
            scan_name = _scan_name(correction, image_index)
            if correction._reference_mode and image_index == 0:
                titles.append("Reference")
            elif stage == "raw":
                titles.append(f"{scan_name}: raw")
            else:
                titles.append(f"{scan_name} {stage}")
        correction.plot_warped_images(show_knots=show_knots, title=titles)


def _quantile_to_uint8(
    tensor: torch.Tensor, low: float = 1.0, high: float = 99.0, *, per_image: bool = False
) -> torch.Tensor:
    """Percentile-stretch a torch tensor to ``uint8`` in [0, 255] on its device.

    Maps the ``[low, high]`` percentile range to [0, 255] with clamping, the
    shared normalization step behind every device-rendered drift display.
    ``per_image`` stretches each leading-axis image with its own percentiles;
    otherwise one percentile pair scales the whole tensor. Centralizing this
    keeps display contrast consistent.
    """
    # torch.quantile errors above ~2**24 elements; stride-subsample large inputs
    # (percentiles are stable under uniform decimation, and it stays deterministic).
    def _sub(flat_1d: torch.Tensor) -> torch.Tensor:
        cap = 1 << 24
        if flat_1d.shape[-1] > cap:
            stride = flat_1d.shape[-1] // (cap // 2)
            return flat_1d[..., ::stride]
        return flat_1d
    if per_image:
        flat = _sub(tensor.reshape(tensor.shape[0], -1))
        lo = torch.quantile(flat, low / 100.0, dim=1)
        hi = torch.quantile(flat, high / 100.0, dim=1)
        scale = torch.clamp(hi - lo, min=1e-9)
        norm = ((tensor - lo[:, None, None]) / scale[:, None, None]).clamp_(0, 1)
    else:
        flat = _sub(tensor.flatten())
        lo = torch.quantile(flat, low / 100.0)
        hi = torch.quantile(flat, high / 100.0)
        scale = torch.clamp(hi - lo, min=1e-9)
        norm = ((tensor - lo) / scale).clamp_(0, 1)
    return (norm * 255).to(torch.uint8)


def overlay_pair(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    mode: str = "rgb",
    output_dtype: str | np.dtype | None = None,
) -> np.ndarray:
    """Color overlay of two images for checking alignment.

    Default ``mode='rgb'`` is the classic red-green: ``reference`` -> red, ``moving`` -> green,
    aligned regions -> yellow (reads at a glance: yellow everywhere = registered).
    ``mode='green-magenta'`` is the colorblind-safe alternative (reference -> magenta,
    moving -> green, aligned -> white) for published figures. Both images share one percentile
    scale so the two colors match in brightness (per-image scaling would unbalance them).

    ``output_dtype="uint8"`` returns the same display colors quantized to
    0--255. Use it for large review stacks; the default remains float32 in
    0--1 for numerical plotting compatibility.
    """
    ref = np.asarray(reference, dtype=np.float32)
    mov = np.asarray(moving, dtype=np.float32)
    lo = min(float(np.percentile(ref, 1)), float(np.percentile(mov, 1)))
    hi = max(float(np.percentile(ref, 99)), float(np.percentile(mov, 99)))
    def _norm(arr):
        return np.clip((arr - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    ref, mov = _norm(ref), _norm(mov)
    if mode == "rgb":
        overlay = np.stack([ref, mov, np.zeros_like(ref)], -1)
    else:
        overlay = np.stack([ref, mov, ref], -1)
    if output_dtype is None or np.dtype(output_dtype) == np.dtype(np.float32):
        return overlay
    if np.dtype(output_dtype) == np.dtype(np.uint8):
        return np.rint(overlay * 255).astype(np.uint8)
    raise ValueError(
        f"output_dtype must be None, 'float32', or 'uint8', got {output_dtype!r}"
    )


# ---------------------------------------------------------------------------
# Public plotting API
# ---------------------------------------------------------------------------

def plot_probe_positions(
    self,
    *,
    image_index: int = 0,
    stride: int = 16,
    strip_padding: bool = True,
    axsize: tuple[float, float] = (5.0, 4.8),
    cmap: str = "viridis",
) -> tuple[Figure, np.ndarray]:
    """Plot nominal and drift-updated probe positions for one scan image.

    The positions remain indexed like the raw scan: point ``(r, c)`` belongs
    to the raw diffraction pattern at ``dataset[r, c]``. The corrected
    positions are plotted in the shared drift-corrected coordinate frame, so
    image 0 and image 1 position maps can be passed to ptychography without
    interpolating diffraction patterns.

    Parameters
    ----------
    image_index : int, default 0
        Scan whose probe positions are shown.
    stride : int, default 16
        Plot every nth position along each scan axis.
    strip_padding : bool, default True
        Express positions in the original image frame.
    axsize : tuple of float, default (5.0, 4.8)
        Size of each panel in inches.
    cmap : str, default "viridis"
        Colormap for displacement magnitude.

    Returns
    -------
    Figure
        Position comparison figure.
    numpy.ndarray
        Matplotlib axes for the two panels.

    Examples
    --------
    >>> figure, axes = drift.plot_probe_positions(image_index=1, stride=8)
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    nominal = self.probe_positions(
        image_index=image_index,
        corrected=False,
        strip_padding=strip_padding,
        plot=False,
    )
    corrected = self.probe_positions(
        image_index=image_index,
        corrected=True,
        strip_padding=strip_padding,
        plot=False,
    )
    disp = corrected - nominal
    mag = np.linalg.norm(disp, axis=-1)
    s = (slice(None, None, stride), slice(None, None, stride))
    nominal_s = nominal[s]
    corrected_s = corrected[s]
    disp_s = disp[s]
    mag_s = mag[s]

    fig, axes = plt.subplots(1, 2, figsize=(axsize[0] * 2, axsize[1]))
    ax0, ax1 = axes
    ax0.scatter(
        nominal_s[..., 1].ravel(),
        nominal_s[..., 0].ravel(),
        s=8,
        c="0.75",
        label="nominal",
        linewidths=0,
    )
    sc0 = ax0.scatter(
        corrected_s[..., 1].ravel(),
        corrected_s[..., 0].ravel(),
        s=10,
        c=mag_s.ravel(),
        cmap=cmap,
        label="drift-updated",
        linewidths=0,
    )
    ax0.set_title(f"image {image_index} probe positions\nnominal vs drift-updated")
    ax0.set_xlabel("col position (px)")
    ax0.set_ylabel("row position (px)")
    ax0.legend(loc="best", frameon=False)
    fig.colorbar(sc0, ax=ax0, fraction=0.046, pad=0.04, label="displacement (px)")

    q = ax1.quiver(
        nominal_s[..., 1],
        nominal_s[..., 0],
        disp_s[..., 1],
        disp_s[..., 0],
        mag_s,
        angles="xy",
        scale_units="xy",
        scale=1,
        cmap=cmap,
        width=0.003,
    )
    ax1.set_title(f"image {image_index} drift displacement\nnominal -> drift-updated")
    ax1.set_xlabel("col position (px)")
    ax1.set_ylabel("row position (px)")
    fig.colorbar(q, ax=ax1, fraction=0.046, pad=0.04, label="displacement (px)")
    for ax in axes:
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    return fig, axes


def plot_warped_images(
    self,
    *,
    show_knots: bool = True,
    knot_colors: tuple[str, ...] = ("#c2185b", "#0097a7"),
    axsize: tuple[int, int] = (8, 8),
    max_display_px: int | None = None,
    ax: Axes | None = None,
    img_idx: int | None = None,
    stage: str | None = None,
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    """Inspect each scan separately to reveal scan-specific residual artifacts.

    A combined image can hide an artifact confined to one acquisition.
    ``stage`` compares a saved initial, affine, strip, or non-rigid checkpoint
    without changing the solved correction object. ``max_display_px=None``
    renders every native pixel; set a limit to opt into display downsampling.

    Returns
    -------
    Figure
        Figure containing one panel per scan.
    numpy.ndarray
        Matplotlib axes for the scan panels.

    Examples
    --------
    >>> figure, axes = drift.plot_warped_images(stage="affine")
    """
    titles = kwargs.pop("title", None)
    arr_np = drift_apply.warped_stack(self, stage)
    arr_t = torch.as_tensor(arr_np, device=self._device, dtype=self._dtype)
    h, w = arr_t.shape[-2:]
    if max_display_px and max(h, w) > max_display_px:
        factor = int(max(h, w) // max_display_px)
        if factor > 1:
            print(f"display downsampled {factor}x (max_display_px={max_display_px}); pass max_display_px=None for native pixels")
            arr_t = torch.nn.functional.avg_pool2d(arr_t.unsqueeze(0), factor).squeeze(0)
    u8 = _quantile_to_uint8(arr_t, low=1.0, high=99.0, per_image=True).cpu().numpy()
    stage_knots = drift_knots.stage_knots(self, stage)
    n = u8.shape[0]
    # single-panel mode: draw one warped image (with its origin) into a provided ax
    if ax is not None:
        assert img_idx is not None, "pass img_idx when drawing into a single ax"
        ax.imshow(u8[img_idx], cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        if titles is not None:
            ax.set_title(titles[img_idx] if isinstance(titles, (list, tuple)) else titles)
        if show_knots:
            row_scale = u8.shape[-2] / self.shape[1]
            column_scale = u8.shape[-1] / self.shape[2]
            kn = stage_knots[img_idx].cpu().numpy()
            ax.plot(
                kn[1] * column_scale,
                kn[0] * row_scale,
                color=knot_colors[img_idx % len(knot_colors)],
            )
        return ax.figure, ax
    fig, ax = plt.subplots(1, n, figsize=(axsize[0] * n, axsize[1]))
    if n == 1:
        ax = np.array([ax])
    for i in range(n):
        ax[i].imshow(u8[i], cmap="gray", vmin=0, vmax=255)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        if titles is not None:
            ax[i].set_title(titles[i] if isinstance(titles, (list, tuple)) else titles)
    if show_knots:
        row_scale = u8.shape[-2] / self.shape[1]
        column_scale = u8.shape[-1] / self.shape[2]
        for img_idx in range(self.shape[0]):
            knots_np = stage_knots[img_idx].cpu().numpy()
            # per-scan attribution: magenta = first scan, cyan = second (CMYK-safe,
            # colorblind-safe, and distinct from the RGB comparison colors)
            ax[img_idx].plot(
                knots_np[1] * column_scale,
                knots_np[0] * row_scale,
                color=knot_colors[img_idx % len(knot_colors)],
            )
    return fig, ax


def plot_convergence(
    self,
    *,
    figsize: tuple[float, float] = (7, 4.2),
    log_scale: bool = True,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot convergence of the iterative non-rigid refinement.

    The logarithmic loss scale is useful for judging whether later iterations
    still provide a meaningful reduction. Cycle zero is the final affine value;
    subsequent points are non-rigid refinement cycles. Affine alignment is a
    candidate-grid search rather than an iterative optimizer, so it is not
    presented as a convergence curve.

    Parameters
    ----------
    figsize : tuple, default (7, 4.2)
    log_scale : bool, default True
        Display the mean disagreement on a logarithmic y-axis.
    **kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    fig : Figure
    ax : Axes

    Examples
    --------
    >>> figure, axis = drift.plot_convergence()
    """
    track = np.asarray(self.error_track)
    is_nonrigid = track[:, 0] == 2
    error_percent = 100 * track[:, 1]
    affine_error = error_percent[~is_nonrigid]
    nonrigid_error = error_percent[is_nonrigid]
    if nonrigid_error.size == 0:
        raise RuntimeError(
            "plot_convergence() requires a completed correct_nonrigid() call."
        )
    if affine_error.size:
        nonrigid_error = np.concatenate((affine_error[-1:], nonrigid_error))

    fig, ax = plt.subplots(figsize=figsize)
    color = "#0072B2"
    plot_kwargs = {"color": color, "linewidth": 2.0, **kwargs}
    cycles = np.arange(nonrigid_error.size)
    ax.plot(cycles, nonrigid_error, **plot_kwargs)
    ax.scatter(
        cycles[[0, -1]], nonrigid_error[[0, -1]],
        s=28, color=color, edgecolor="white", linewidth=0.7, zorder=3,
    )
    ax.set_title("Non-rigid convergence", fontsize=11, fontweight="semibold")
    ax.set_xlabel("Refinement cycle")
    ax.set_ylabel("Mean disagreement (%)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, which="major", color="#d0d0d0", linewidth=0.7)
    ax.grid(True, which="minor", color="#e8e8e8", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Building blocks - used internally by the public API functions above
# ---------------------------------------------------------------------------


def _render_uint8(
    dc,
    rgb: bool,
    low: float = 1.0,
    high: float = 99.0,
    max_display_px: int | None = None,
    stack: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize a corrected stack for fast grayscale or RGB display."""
    arr_np = drift_apply.warped_stack(dc) if stack is None else stack
    arr_t = torch.as_tensor(arr_np, device=dc._device, dtype=dc._dtype)
    h, w = arr_t.shape[-2:]
    if max_display_px and max(h, w) > max_display_px:
        factor = int(max(h, w) // max_display_px)
        if factor > 1:
            arr_t = torch.nn.functional.avg_pool2d(arr_t.unsqueeze(0), factor).squeeze(0)
    if rgb and arr_t.shape[0] <= 3:
        H, W = arr_t.shape[-2:]
        rgb_t = torch.zeros(H, W, 3, device=arr_t.device, dtype=torch.uint8)
        for i in range(arr_t.shape[0]):
            rgb_t[..., i] = _quantile_to_uint8(arr_t[i], low=low, high=high)
        return rgb_t.cpu().numpy()
    merged = arr_t.mean(0)
    return _quantile_to_uint8(merged, low=low, high=high).cpu().numpy()


def _single_image_u8(
    dc,
    idx: int,
    max_display_px: int | None,
    stack: np.ndarray | None = None,
) -> np.ndarray:
    """One warped image, quantile-stretched to uint8 (same normalize as the merged view)."""
    arr_np = drift_apply.warped_stack(dc) if stack is None else stack
    arr_t = torch.as_tensor(arr_np[idx], device=dc._device, dtype=dc._dtype)
    h, w = arr_t.shape[-2:]
    if max_display_px and max(h, w) > max_display_px and int(max(h, w) // max_display_px) > 1:
        arr_t = torch.nn.functional.avg_pool2d(arr_t[None, None], int(max(h, w) // max_display_px))[0, 0]
    return _quantile_to_uint8(arr_t).cpu().numpy()


def _stage_description(stage: str | None) -> str:
    """Return a concise, natural-language description of a correction stage."""

    return {
        None: "current correction",
        "initial": "raw",
        "raw": "raw",
        "affine": "after affine correction",
        "strip": "after strip correction",
        "nonrigid": "after non-rigid correction",
        "non-rigid": "after non-rigid correction",
    }.get(stage, str(stage))


def _format_angle_degrees(angle: float) -> str:
    """Format a scan angle with a mathematical minus and degree sign."""

    value = float(angle)
    sign = "−" if value < 0 else ""
    return f"{sign}{abs(value):g}°"


def _scan_name(dc, image_index: int) -> str:
    """Return the scientist-facing role or angle for one scan."""

    if getattr(dc, "_reference_mode", False):
        return "Reference" if image_index == 0 else "Moving scan"
    angle = _format_angle_degrees(dc.scan_direction_degrees[image_index])
    return f"{angle} scan"


def plot_combined(
    self,
    *,
    show_knots: bool = True,
    rgb: bool = True,
    axsize: tuple[int, int] = (16, 16),
    max_display_px: int | None = None,
    ax: Axes | None = None,
    stage: str | tuple[str, ...] | list[str] | None = None,
    show_scans: bool = False,
    interactive: bool = False,
    width: int | None = 1000,
    mode: str = "rgb",
    zoom: float = 1.0,
    center: tuple[float, float] | None = None,
    display_bin: int = 1,
    **kwargs,
) -> tuple[Figure, Axes] | object:
    """Plot the combined scan (RGB registration overlay) with optional knot curves.

    Bound on :class:`~quantem.imaging.drift.DriftCorrection` as
    ``DriftCorrection.plot_combined``. This is the primary registration QA plot
    in tutorials (yellow means aligned in the red/green overlay).

    Parameters
    ----------
    show_knots : bool, default True
        Overlay knot trajectories on static non-rigid panels. Initial and
        affine panels stay clean because multiple knots add freedom only to
        the non-rigid model. Use :meth:`plot_knots` to inspect affine geometry.
    rgb : bool, default True
        If True (2 or 3 scans), RGB comparison (image 0 red, 1 green, 2 blue).
        Misalignment shows as color fringes. If False, grayscale mean.
    axsize : tuple of int, default (16, 16)
        Matplotlib figure size scale for the static path.
    max_display_px : int or None
        Cap longest display edge for static rendering. None keeps native size.
    ax : matplotlib.axes.Axes or None
        Optional axis to draw into (static path).
    stage : str, tuple of str, or None
        Knot checkpoint to render. Examples: ``"affine"``, ``"nonrigid"``,
        ``("initial", "affine")``, ``("initial", "affine", "nonrigid")``.
        A tuple draws stages side by side in one call.
    show_scans : bool, default False
        Static path only: also show each warped scan beside the combined panel.
        Ignored when ``ax`` is set.
    interactive : bool, default False
        If True, return a :class:`quantem.widget.Show2D` gallery instead of a
        static figure.
    width : int or None, default 1000
        Interactive panel width in pixels. Ignored on the static path.
    mode : str, default "rgb"
        Overlay mode for interactive RGB (e.g. ``"rgb"``, green-magenta variants).
    zoom : float, default 1.0
        Static path: axis-limit zoom (not array crop). Interactive path: Show2D zoom.
    center : tuple of float or None
        Static path: fractional ``(row, col)`` pan in ``[-0.5, 0.5]``. Interactive
        path: absolute pixel center when provided.
    display_bin : int, default 1
        Interactive Show2D display binning only. Default 1 (no silent reduction).
        Pass 2/4/8 for speed, or ``"auto"`` for Show2D budget behavior.
    **kwargs
        Forwarded to the static renderer or Show2D (for example ``show_fft``).

    Returns
    -------
    (Figure, Axes) or Show2D
        Static matplotlib objects, or an interactive widget when
        ``interactive=True``.

    Notes
    -----
    Interactive RGB bakes contrast into the overlay. Do not pass
    ``auto_contrast`` / ``link_contrast`` for that mode; they are ignored.
    ``rgb=False`` still forwards those Show2D contrast controls.

    Examples
    --------
    >>> dc.plot_combined(stage=("initial", "affine"), interactive=True, width=620)
    >>> dc.plot_combined(
    ...     stage=("affine", "nonrigid"),
    ...     interactive=False,
    ...     show_knots=True,
    ... )
    """
    if interactive:
        from quantem.widget import Show2D  # lazy: quantem.widget is an optional extra

        stage_names = list(stage) if isinstance(stage, (tuple, list)) else [stage]
        if rgb and self.shape[0] != 2:
            raise ValueError(
                f"interactive rgb=True needs exactly 2 images for the green-magenta overlay, got {self.shape[0]}"
            )
        images, labels = [], []
        for stage_name in stage_names:
            stage_display = _stage_description(stage_name)
            warped = drift_apply.warped_stack(self, stage_name)
            # One panel per stage: RGB comparison or grayscale mean.
            if rgb:
                images.append(overlay_pair(warped[0], warped[1], mode=mode))
                labels.append(f"Combined: {stage_display} (RGB)")
            else:
                images.append(warped.mean(axis=0))
                labels.append(f"Combined: {stage_display}")
        sampling = float(np.asarray(self.imgs[0].sampling, dtype=float)[0])
        unit = self.imgs[0].units[0] if getattr(self.imgs[0], "units", None) else "px"
        show_kwargs = dict(kwargs)
        # A combined registration comparison is a clean scientific result,
        # not a general-purpose image-editing workspace. Callers can still opt
        # back into the full interface with ui_mode="interactive".
        show_kwargs.setdefault("ui_mode", "report")
        if rgb:
            # Overlay is already percentile-normalized on a shared scale.
            # Contrast knobs would only confuse an RGB registration view.
            show_kwargs.pop("auto_contrast", None)
            show_kwargs.pop("link_contrast", None)
            show_kwargs.pop("cmap", None)
            show_kwargs.setdefault("show_fft", False)
        return Show2D(
            images,
            labels=labels,
            sampling=sampling,
            units=unit,
            ncols=len(stage_names),
            size=int(width) if width else 0,
            display_bin=display_bin,
            zoom=zoom,
            center=center,
            **show_kwargs,
        )
    if isinstance(stage, (tuple, list)):
        # One call draws every requested stage side by side. Pass ax= as a
        # sequence of Axes (one per stage) to embed into a larger gallery;
        # otherwise a fresh 1 x n figure is created.
        stage_names = list(stage)
        if ax is None:
            fig, axes = plt.subplots(
                1,
                len(stage_names),
                figsize=(axsize[0] * len(stage_names) / 2, axsize[1] / 2),
                squeeze=False,
            )
            stage_axes = list(axes[0])
        else:
            stage_axes = list(np.atleast_1d(ax).ravel())
            if len(stage_axes) != len(stage_names):
                raise ValueError(
                    f"stage has {len(stage_names)} entries but ax has "
                    f"{len(stage_axes)} axes; provide one Axes per stage."
                )
            fig = stage_axes[0].figure
        for stage_name, stage_ax in zip(stage_names, stage_axes, strict=True):
            plot_combined(
                self,
                show_knots=show_knots,
                rgb=rgb,
                max_display_px=max_display_px,
                ax=stage_ax,
                stage=stage_name,
                zoom=zoom,
                center=center,
                **dict(kwargs),
            )
        return fig, np.asarray(stage_axes, dtype=object)
    stage_display = _stage_description(stage)
    display_stack = drift_apply.warped_stack(self, stage)
    merged_u8 = _render_uint8(
        self,
        rgb=rgb and self.shape[0] <= 3,
        max_display_px=max_display_px,
        stack=display_stack,
    )
    merged_title = kwargs.pop(
        "title",
        f"Combined: {stage_display}" + (" (RGB)" if rgb else ""),
    )
    kwargs.pop("cmap", None)
    panels = []
    if show_scans and ax is None:
        panels = [
            (
                _single_image_u8(self, i, max_display_px, stack=display_stack),
                f"{_scan_name(self, i)} {stage_display}"
                if stage_display != "raw"
                else f"{_scan_name(self, i)}: raw",
            )
            for i in range(self.shape[0])
        ]
    panels.append((merged_u8, merged_title))
    if ax is None:
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(axsize[0] * len(panels) / 2, axsize[1] / 2),
            squeeze=False,
        )
        axes = list(axes[0])
    else:
        fig, axes = ax.figure, [ax]
    for panel_ax, (u8, panel_title) in zip(axes, panels):
        if u8.ndim == 3:
            panel_ax.imshow(u8)
        else:
            panel_ax.imshow(u8, cmap="gray", vmin=0, vmax=255)
        panel_ax.set_title(panel_title)
        panel_ax.set_xticks([])
        panel_ax.set_yticks([])
    merged_ax = axes[-1]
    nonrigid_stage = stage in ("nonrigid", "non-rigid")
    if stage is None:
        error_track = np.asarray(getattr(self, "error_track", []))
        nonrigid_stage = bool(
            error_track.ndim == 2
            and error_track.shape[1] > 0
            and np.any(error_track[:, 0] == 2)
        )
    if show_knots and nonrigid_stage:
        stage_knots = drift_knots.stage_knots(self, stage)
        row_scale = merged_u8.shape[0] / self.shape[1]
        column_scale = merged_u8.shape[1] / self.shape[2]
        for knots in stage_knots:
            knots_np = knots.cpu().numpy()
            merged_ax.plot(
                knots_np[1] * column_scale,
                knots_np[0] * row_scale,
            )
    if zoom > 1.0:
        # Axis limits preserve the overlay-to-pixel coordinate mapping.
        panel_h, panel_w = merged_u8.shape[:2]
        win = min(panel_h, panel_w) / float(zoom)
        offset_row, offset_col = center if center is not None else (0.0, 0.0)
        cy = min(max(panel_h / 2 + offset_row * panel_h, win / 2), panel_h - win / 2)
        cx = min(max(panel_w / 2 + offset_col * panel_w, win / 2), panel_w - win / 2)
        for panel_ax in axes:
            panel_ax.set_xlim(cx - win / 2, cx + win / 2)
            panel_ax.set_ylim(cy + win / 2, cy - win / 2)
    return fig, (merged_ax if len(axes) == 1 else np.array(axes))


def plot_knots(
    self,
    *,
    figsize: tuple[int, int] | None = None,
    stage: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot knot trajectories before and after correction plus the per-scanline delta field.

    Two panels per image:
    - Top: mean warped image with initial knots (dashed) and corrected knots (solid)
      overlaid. A third dotted line shows the affine-only state when available, so
      the affine vs. nonrigid contributions are visible side by side.
    - Bottom: per-scanline correction delta (row and col components) in pixels.
      A smooth curve means the correction field is physically reasonable. Rapid
      oscillations indicate ``knot_smoothing_sigma`` is too small and the optimizer
      is fitting noise rather than real drift.

    Parameters
    ----------
    figsize : (width, height), optional
        Figure size in inches. Defaults to ``(7 * num_images, 11)`` so each
        square image occupies the same column width as its correction chart.
    stage : {"affine", "strip", "nonrigid", None}, optional
        Show only one stage's contribution. ``"nonrigid"`` plots just the
        nonrigid delta on its OWN y-scale - the refinement is often sub-pixel
        and disappears next to affine's tens-of-pixels ramp, yet it is the
        parameter that matters when judging the nonrigid solve. ``"strip"``
        plots the piecewise-rigid contribution relative to affine,
        ``"affine"`` plots just the affine ramp, and ``None`` overlays every
        completed stage.

    Returns
    -------
    fig : Figure
    axes : np.ndarray of Axes, shape (2, num_images)

    Examples
    --------
    >>> figure, axes = drift.plot_knots(stage="nonrigid")
    """
    ensure_warped_images(self)
    num_images = self.shape[0]
    warped = self.imgs_warped.array   # each panel shows its own corrected image, not the merge
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if figsize is None:
        figsize = (7 * num_images, 11)
    fig, axes = plt.subplots(
        2, num_images, figsize=figsize,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    if num_images == 1:
        axes = axes[:, None]
    scanlines = np.arange(self.knots[0].shape[1])
    for img_idx in range(num_images):
        color = colors[img_idx % len(colors)]
        initial = self._initial_knots[img_idx].cpu().numpy()
        current = self.knots[img_idx].cpu().numpy()
        affine_np = (
            self._knots_after_affine[img_idx].cpu().numpy()
            if hasattr(self, "_knots_after_affine")
            else None
        )
        strip_np = (
            self._knots_after_strip[img_idx].cpu().numpy()
            if hasattr(self, "_knots_after_strip")
            else None
        )
        has_nonrigid = bool(np.any(np.asarray(self.error_track)[:, 0] == 2))
        ax_img = axes[0, img_idx]
        bg = warped[img_idx]
        ax_img.imshow(bg, cmap="gray", origin="upper", aspect="equal",
                      vmin=float(bg.min()), vmax=float(bg.max()))
        ax_img.plot(initial[1, :, 0], initial[0, :, 0],
                    "--", color=color, lw=1.2, alpha=0.7, label="before alignment")
        if affine_np is not None and stage in (None, "affine"):
            ax_img.plot(affine_np[1, :, 0], affine_np[0, :, 0],
                        ":", color=color, lw=1.0, alpha=0.6,
                        label="after affine correction")
        if strip_np is not None and stage in (None, "strip"):
            ax_img.plot(strip_np[1, :, 0], strip_np[0, :, 0],
                        "-", color=color, lw=1.5,
                        label="after strip correction")
        if has_nonrigid and stage in (None, "nonrigid"):
            ax_img.plot(current[1, :, 0], current[0, :, 0],
                        "-", color=color, lw=1.5,
                        label="after non-rigid correction")
        ax_img.legend(fontsize=8, loc="upper right")
        scan_name = _scan_name(self, img_idx)
        ax_img.set_title(f"{scan_name}: knot trajectory", fontsize=10)
        ax_img.axis("off")
        ax_delta = axes[1, img_idx]
        if affine_np is not None and stage in (None, "affine"):
            aff_delta = affine_np - initial
            ax_delta.plot(scanlines, aff_delta[0, :, 0],
                          ":", lw=1.0, alpha=0.6, label="row \u0394 after affine")
            ax_delta.plot(scanlines, aff_delta[1, :, 0],
                          ":", lw=1.0, alpha=0.6, label="col \u0394 after affine")
        # Each residual stage is measured from its immediate predecessor so
        # the smaller strip/non-rigid contribution remains readable.
        if strip_np is not None and stage in (None, "strip"):
            strip_base = affine_np if affine_np is not None else initial
            strip_delta = strip_np - strip_base
            ax_delta.plot(scanlines, strip_delta[0, :, 0], lw=1.2,
                          label="row Δ from strip correction")
            ax_delta.plot(scanlines, strip_delta[1, :, 0], lw=1.2,
                          label="column Δ from strip correction")
        if has_nonrigid and stage in (None, "nonrigid"):
            nonrigid_base = strip_np if strip_np is not None else affine_np
            if nonrigid_base is None:
                nonrigid_base = initial
            nonrigid_delta = current - nonrigid_base
            ax_delta.plot(scanlines, nonrigid_delta[0, :, 0], lw=1.2,
                          label="row Δ from non-rigid correction")
            ax_delta.plot(scanlines, nonrigid_delta[1, :, 0], lw=1.2,
                          label="column Δ from non-rigid correction")
        ax_delta.axhline(0, color="k", lw=0.5, ls="--")
        ax_delta.set_xlabel("scan line")
        ax_delta.set_ylabel("correction (pixels)")
        ax_delta.set_title(f"{scan_name}: correction field", fontsize=10)
        handles, labels = ax_delta.get_legend_handles_labels()
        if handles:
            ax_delta.legend(handles, labels, fontsize=8)
        ax_delta.grid(alpha=0.3)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Complete workflow views
# ---------------------------------------------------------------------------

def static_comparison(correction, zoom: float, size: int, **kwargs):
    """Compare raw and corrected scans in a publication-ready figure.

    A shared contrast range makes intensity differences visible across scans;
    independent auto contrast is useful when acquisition brightness differs.
    Zoom changes only the displayed field of view, never the corrected data.
    """
    panels = drift_apply.comparison_panels(correction)
    cmap = kwargs.pop("cmap", "inferno")
    auto_contrast = bool(kwargs.pop("auto_contrast", False))
    link_contrast = bool(kwargs.pop("link_contrast", True))
    smooth = bool(kwargs.pop("smooth", False))
    kwargs.pop("show_fft", None)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    offset_row, offset_col = kwargs.pop("center", (0.0, 0.0)) or (0.0, 0.0)

    images = [np.asarray(image) for image in panels["images"]]
    ncols = int(panels["ncols"])
    image_grid = [images[index : index + ncols] for index in range(0, len(images), ncols)]
    label_grid = [
        panels["labels"][index : index + ncols]
        for index in range(0, len(images), ncols)
    ]

    norm = None
    if link_contrast:
        values = np.concatenate([image.ravel()[::16] for image in images])
        values = values[np.isfinite(values)]
        low, high = (
            np.percentile(values, (1.0, 99.0))
            if auto_contrast
            else (values.min(), values.max())
        )
        norm = {
            "vmin": float(low if vmin is None else vmin),
            "vmax": float(high if vmax is None else vmax),
        }
    elif auto_contrast:
        norm = "linear_auto"
    elif vmin is not None or vmax is not None:
        norm = {"vmin": vmin, "vmax": vmax}

    figure, axes = show_2d(
        image_grid,
        title=label_grid,
        cmap=cmap,
        norm=norm,
        scalebar={"sampling": panels["pixel_size"], "units": panels["pixel_unit"]},
        axsize=(max(float(size), 200.0) / 100.0,) * 2,
        **kwargs,
    )
    axes_array = np.asarray(axes, dtype=object).reshape(len(image_grid), ncols)
    interpolation = "bilinear" if smooth else "nearest"
    for axis, image in zip(axes_array.flat, images, strict=True):
        if axis.images:
            axis.images[0].set_interpolation(interpolation)
        height, width = image.shape[:2]
        visible_height, visible_width = height / zoom, width / zoom
        center_row = np.clip(
            (height - 1) / 2 + offset_row * height,
            visible_height / 2.0,
            height - visible_height / 2.0,
        )
        center_column = np.clip(
            (width - 1) / 2 + offset_col * width,
            visible_width / 2.0,
            width - visible_width / 2.0,
        )
        axis.set_xlim(
            center_column - visible_width / 2,
            center_column + visible_width / 2,
        )
        axis.set_ylim(
            center_row + visible_height / 2,
            center_row - visible_height / 2,
        )
    return figure


def show(
    self,
    *,
    zoom: float = 1.0,
    size: int = 400,
    mode: str = "interactive",
    **kwargs,
):
    """Show raw and corrected scans in their common acquisition frame.

    Interactive modes return a linked Show2D comparison for inspecting local
    alignment. ``mode="static"`` returns the same panels as a Matplotlib figure
    for papers and saved reports.

    Parameters
    ----------
    zoom : float, default 1.0
        Initial magnification of the common scan field.
    size : int, default 400
        Interactive panel size in pixels.
    mode : str, default "interactive"
        Use ``"static"`` for Matplotlib or a Show2D UI mode for exploration.

    Returns
    -------
    object
        Show2D viewer or Matplotlib figure selected by ``mode``.

    Examples
    --------
    >>> viewer = drift.show(zoom=2)
    >>> figure = drift.show(mode="static")
    """
    if mode == "static":
        return static_comparison(self, zoom, size, **kwargs)

    from quantem.widget import Show2D

    panels = drift_apply.comparison_panels(self)
    return Show2D(
        panels["images"],
        ncols=panels["ncols"],
        gallery_gap_px=0,
        size=size,
        pixel_size=panels["pixel_size"],
        pixel_unit=panels["pixel_unit"],
        labels=panels["labels"],
        zoom=zoom,
        ui_mode=mode,
        **kwargs,
    )


def show_4dstem(
    self,
    *,
    det_bin: int = 1,
    view_mode: str = "multiple",
    **kwargs,
):
    """Show raw, corrected, and merged diffraction datasets together.

    The three synchronized stages make it possible to inspect whether scan
    correction improves the virtual image without changing diffraction detail.

    Parameters
    ----------
    det_bin : int, default 1
        Detector-axis binning used only for interactive display.
    view_mode : str, default "multiple"
        Show4DSTEM layout used for the processing stages.

    Returns
    -------
    Show4DSTEM
        Interactive comparison viewer.

    Examples
    --------
    >>> viewer = drift.show_4dstem(det_bin=2)
    """
    from quantem.widget import Show4DSTEM

    sampling = np.asarray(self.imgs[0].sampling, dtype=float)
    units = list(self.imgs[0].units)
    kwargs.setdefault(
        "frame_labels",
        ["0° raw", "0° affine-corrected", "0°/90° affine-corrected + combined"],
    )
    kwargs.setdefault("title", "4D-STEM affine drift correction")
    kwargs.setdefault("frame_dim_label", "Processing stage")
    kwargs.setdefault("compare_cols", 3)
    kwargs.setdefault("compare_max_panels", 3)
    kwargs.setdefault("compare_dp_mode", "selected")
    kwargs.setdefault(
        "sampling",
        (float(sampling[0]), float(sampling[1]), float(det_bin), float(det_bin)),
    )
    kwargs.setdefault("units", [units[0], units[1], "pixels", "pixels"])

    viewer = Show4DSTEM(
        fourdstem.corrected_4dstem_views(self, det_bin=det_bin),
        view_mode=view_mode,
        **kwargs,
    )
    viewer.frame_idx = 2
    return viewer
