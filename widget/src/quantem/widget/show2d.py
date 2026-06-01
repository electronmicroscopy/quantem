"""
show2d: Static 2D image viewer with optional FFT and histogram analysis.

For displaying a single image or a static gallery of multiple images.
Unlike Show3D (interactive), Show2D focuses on static visualization.
"""

import base64
import io
import json
import math
import os
import pathlib
import warnings
from enum import StrEnum
from typing import Self

import anywidget
import matplotlib
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import traitlets
from quantem.widget.array_utils import _resize_image, to_numpy
from quantem.widget.state import resolve_widget_version, save_state_file, unwrap_state_payload

from quantem.core.datastructures import Dataset2d, Dataset3d


def _reject_unknown_kwargs(cls, kwargs: dict) -> None:
    """Raise TypeError if kwargs contains any key that isn't a declared trait.

    anywidget/traitlets silently accept unknown keys, which let stale notebooks
    pass obsolete params like ``pixel_size_angstrom=0.5`` with no warning.  This
    helper catches typos and renamed-trait references at construction time.
    """
    traits = set(cls.class_trait_names())
    unknown = [k for k in kwargs if k not in traits]
    if unknown:
        key = sorted(unknown)[0]
        raise TypeError(
            f"{cls.__name__}() got unexpected keyword argument {key!r}. "
            f"Check for typos or a renamed parameter (e.g. canvas_size → size, "
            f"image_width_px → size, pixel_size_angstrom → pixel_size)."
        )


def _round_to_nice(value: float) -> float:
    """Round a physical length to a 'nice' value (1, 2, 5, 10, 20, 50, ...)."""
    if value <= 0:
        return 1.0
    exp = math.floor(math.log10(value))
    base = 10 ** exp
    mantissa = value / base
    if mantissa < 1.5:
        return base
    elif mantissa < 3.5:
        return 2 * base
    elif mantissa < 7.5:
        return 5 * base
    else:
        return 10 * base


class Colormap(StrEnum):
    INFERNO = "inferno"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    PLASMA = "plasma"
    GRAY = "gray"


class Show2D(anywidget.AnyWidget):
    """
    Static 2D image viewer with optional FFT and histogram analysis.

    Display a single image or multiple images in a gallery layout.
    For interactive stack viewing with playback, use Show3D instead.

    Parameters
    ----------
    data : array_like
        2D array (height, width) for single image, or
        3D array (N, height, width) for multiple images displayed as gallery.
    labels : list of str, optional
        Labels for each image in gallery mode.
    title : str, optional
        Title to display above the image(s).
    cmap : str, default "inferno"
        Colormap name ("magma", "viridis", "gray", "inferno", "plasma").
    sampling : float or tuple of float, optional
        Pixel size per axis ``(row, col)``. Scalar broadcasts to both axes.
        Used for scale bar display. Defaults to ``(1, 1)``.
    units : str or list of str, optional
        Unit string per axis. Scalar broadcasts to both. Common: ``"A"``,
        ``"nm"``, ``"pixels"``. Defaults to ``["pixels", "pixels"]``.
    show_fft : bool, default False
        Show FFT and histogram panels.
    show_stats : bool, default True
        Show statistics (mean, min, max, std).
    log_scale : bool, default False
        Use log scale for intensity mapping.
    auto_contrast : bool, default False
        Use percentile-based contrast.
    vmin : float, optional
        Absolute minimum intensity for color mapping. When both vmin and vmax
        are set, all gallery images share the same intensity scale: essential
        for A/B visual comparison.
    vmax : float, optional
        Absolute maximum intensity for color mapping.
    ncols : int, default 3
        Number of columns in gallery mode.
    size : int, default 0
        Canvas rendering size in CSS pixels (the on-screen width of each image).
        ``0`` uses the frontend default: 500 px for a single image, 300 px per
        image in gallery mode.  Pass e.g. ``size=800`` to enlarge for a
        presentation, or ``size=200`` to compress alongside a control panel.
        This controls **display only**: the underlying image resolution is
        never resampled; zooming into a 4K image preserves every pixel.
    Attributes
    ----------
    render_total_ms : int or None
        End-to-end wall clock from constructor start to first browser paint,
        populated by a JS→Python round-trip after the first canvas render.
        ``None`` until the browser has actually painted; also printed to stdout
        when it fires.  Use to triage "is it Python, wire, or the browser?"
        during live acquisitions.
    render_python_build_ms : int or None
        Subset of ``render_total_ms`` covering Python ``__init__`` only.
    render_wire_js_ms : int or None
        Subset covering everything after Python returns: Comm transfer, JS
        decode, colormap, and canvas paint.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Show2D

    Single 2D NumPy array:

    >>> Show2D(np.random.rand(512, 512))

    PyTorch tensor (CPU or GPU, any dtype):

    >>> import torch
    >>> Show2D(torch.rand(512, 512))

    3D NumPy stack ``(N, H, W)`` rendered as a gallery:

    >>> Show2D(np.random.rand(6, 256, 256), ncols=3)

    List of arrays with different shapes (center-padded to a common canvas):

    >>> Show2D([np.random.rand(256, 256), np.random.rand(300, 400)])

    quantem ``Dataset2d``: title, sampling, units auto-extracted:

    >>> from quantem.core.datastructures import Dataset2d
    >>> ds = Dataset2d.from_array(np.random.rand(512, 512))
    >>> Show2D(ds)

    quantem ``Dataset3d``: gallery view of N frames with calibration:

    >>> from quantem.core.datastructures import Dataset3d
    >>> ds = Dataset3d.from_array(np.random.rand(6, 256, 256))
    >>> Show2D(ds, ncols=3)

    A/B comparison with shared contrast and linked zoom/pan:

    >>> a, b = np.random.rand(512, 512), np.random.rand(512, 512)
    >>> Show2D([a, b], vmin=0, vmax=1, link_zoom=True, link_pan=True)

    Per-image absolute contrast (one ``vmin``/``vmax`` per image):

    >>> Show2D([a, b], vmin=[0.0, 0.2], vmax=[1.0, 0.8])

    Drift comparison: diff mode adds a ``A - B`` panel alongside the originals
    (gallery becomes ``[A, B, A - B]``):

    >>> Show2D([a, b], diff_mode=True, link_zoom=True, link_pan=True)

    Large image: display-only canvas size (full resolution preserved):

    >>> Show2D(np.random.rand(4096, 4096), size=800)

    Static export to PDF or PNG (vector PDF for publication figures):

    >>> w = Show2D(np.random.rand(512, 512), sampling=0.5, units="nm")
    >>> w.save_image("figure.pdf", dpi=150)
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show2d.js"

    # =========================================================================
    # Core State
    # GPU memory budget for display buffers (MB). Each 4K image needs ~192 MB.
    # 12×4K = 2304 MB fits. 16+ triggers auto-bin.
    _GPU_DISPLAY_BUDGET_MB = 2500

    # =========================================================================
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    n_images = traitlets.Int(1).tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    _display_bin_factor = traitlets.Int(1).tag(sync=True)  # 1 = full-res, 2/4/8 = binned
    _gpu_max_buffer_mb = traitlets.Int(0).tag(sync=True)  # GPU reports maxBufferSize (JS→Python)
    # Flipped True by JS after the first colormap pass has painted to canvas.
    # Used by the Python-side truthful timing print (end-to-end wall clock, not just __init__).
    _js_rendered = traitlets.Bool(False).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    # Offline mode: stack quantized to uint8 against global (min, max). 4x
    # smaller than float32 — drops standalone HTML from ~200 MB to ~110 MB.
    # JS dequantizes on read. Eye can't tell uint8 from float32 after colormap
    # reduces to 256 levels anyway.
    offline = traitlets.Bool(False).tag(sync=True)
    # True only on a clone written by export_html: forces the standalone HTML to
    # render on a light/white background regardless of the viewer's OS theme.
    _export_light = traitlets.Bool(False).tag(sync=True)
    _offline_min = traitlets.Float(0.0).tag(sync=True)
    _offline_max = traitlets.Float(1.0).tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    ncols = traitlets.Int(3).tag(sync=True)

    # =========================================================================
    # Display Options
    # =========================================================================
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmins = traitlets.List(trait=traitlets.Float(allow_none=True), allow_none=True, default_value=None).tag(sync=True)
    vmaxs = traitlets.List(trait=traitlets.Float(allow_none=True), allow_none=True, default_value=None).tag(sync=True)

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    pixel_unit = traitlets.Unicode("pixels").tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    size = traitlets.Int(0).tag(sync=True)  # Canvas rendering size in CSS pixels; 0 = frontend default
    smooth = traitlets.Bool(False).tag(sync=True)
    initial_zoom = traitlets.Float(1.0).tag(sync=True)
    zoom_row = traitlets.Float(None, allow_none=True).tag(sync=True)
    zoom_col = traitlets.Float(None, allow_none=True).tag(sync=True)
    link_zoom = traitlets.Bool(False).tag(sync=True)
    link_pan = traitlets.Bool(False).tag(sync=True)
    link_contrast = traitlets.Bool(True).tag(sync=True)
    diff_mode = traitlets.Bool(False).tag(sync=True)
    diff_reference = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # UI Visibility
    # =========================================================================
    show_controls = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    stats_mean = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_min = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_max = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_std = traitlets.List(traitlets.Float()).tag(sync=True)

    # =========================================================================
    # Analysis Panels (FFT + Histogram shown together)
    # =========================================================================
    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_window = traitlets.Bool(True).tag(sync=True)

    # =========================================================================
    # Selected Image (for single-image analysis display)
    # =========================================================================
    selected_idx = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # ROI Selection
    # =========================================================================
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_list = traitlets.List([]).tag(sync=True)
    roi_selected_idx = traitlets.Int(-1).tag(sync=True)

    # =========================================================================
    # Line Profile
    # =========================================================================
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)

    # =========================================================================
    # Per-Image Rotation
    # =========================================================================
    image_rotations = traitlets.List(traitlets.Int(), []).tag(sync=True)

    def __init__(
        self,
        data: np.ndarray | list[np.ndarray],
        labels: list[str | None] = None,
        title: str = "",
        cmap: str | Colormap = Colormap.INFERNO,
        sampling: float | tuple[float, float] | list[float] | None = None,
        units: str | list[str] | None = None,
        scale_bar_visible: bool = True,
        show_fft: bool = False,
        fft_window: bool = True,
        show_controls: bool = True,
        show_stats: bool = True,
        verbose: bool = True,
        log_scale: bool = False,
        auto_contrast: bool = False,
        offline: bool = False,
        vmin: float | list | None = None,
        vmax: float | list | None = None,
        ncols: int = 3,
        size: int = 0,
        smooth: bool = False,
        zoom: float = 1.0,
        zoom_row: float | None = None,
        zoom_col: float | None = None,
        link_zoom: bool | None = None,
        link_pan: bool | None = None,
        link_contrast: bool = True,
        diff_mode: bool = False,
        view_box: tuple | list | None = None,
        display_bin: int | str = "auto",
        state=None,
        **kwargs,
    ):
        import time as _time
        _t0 = _time.perf_counter()
        # Reject typos and stale kwargs (e.g. image_width_px, pixel_size_angstrom).
        # anywidget/traitlets silently ignores unknown keys, which hid the
        # pixel_size_angstrom bug in show2d_all_features.ipynb for months.
        _reject_unknown_kwargs(type(self), kwargs)
        super().__init__(**kwargs)
        # hold_sync() batches ALL traitlet assignments into a single comm message
        # sent when the context manager exits.  Without this, each self.x = y
        # fires a separate round-trip over the ZMQ/websocket channel, which
        # can add 20+ seconds for a 30-image gallery in VS Code Jupyter.
        with self.hold_sync():
            self._init_sync(
                data=data, labels=labels, title=title, cmap=cmap,
                sampling=sampling, units=units, scale_bar_visible=scale_bar_visible,
                show_fft=show_fft, fft_window=fft_window,
                show_controls=show_controls, show_stats=show_stats,
                log_scale=log_scale, auto_contrast=auto_contrast, offline=offline,
                vmin=vmin, vmax=vmax,
                ncols=ncols, size=size, smooth=smooth, zoom=zoom,
                zoom_row=zoom_row, zoom_col=zoom_col,
                link_zoom=link_zoom, link_pan=link_pan, link_contrast=link_contrast,
                diff_mode=diff_mode, view_box=view_box,
                display_bin=display_bin, verbose=verbose, state=state, _t0=_t0)

    def _init_sync(self, *, data, labels, title, cmap, sampling, units,
                   scale_bar_visible, show_fft, fft_window,
                   show_controls, show_stats, log_scale, auto_contrast, offline,
                   vmin, vmax,
                   ncols, size, smooth, zoom, zoom_row, zoom_col,
                   link_zoom, link_pan, link_contrast, diff_mode, view_box,
                   display_bin, verbose, state, _t0):
        import time as _time
        self._verbose = verbose
        self.widget_version = resolve_widget_version()
        self._display_data = None  # initialized after data setup
        self._display_bin = 1

        # First-class support for quantem Dataset2d / Dataset3d:
        # auto-extract array + sampling + units from the dataset object.
        if isinstance(data, (Dataset2d, Dataset3d)) or (
            hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling")
        ):
            if not title and data.name:
                title = data.name
            if sampling is None:
                sampling = tuple(float(s) for s in data.sampling[-2:])
            if units is None and hasattr(data, "units"):
                units = list(data.units[-2:])
            data = data.array
        # Same auto-extract for list/tuple of Dataset2d (gallery from per-file load).
        elif isinstance(data, (list, tuple)) and len(data) > 0 and (
            isinstance(data[0], (Dataset2d, Dataset3d)) or
            (hasattr(data[0], "array") and hasattr(data[0], "sampling"))
        ):
            first = data[0]
            if sampling is None:
                sampling = tuple(float(s) for s in first.sampling[-2:])
            if units is None and hasattr(first, "units"):
                units = list(first.units[-2:])
            data = [d.array for d in data]

        # Convert NumPy / PyTorch / list inputs to a NumPy array.
        if isinstance(data, list):
            images = [to_numpy(d) for d in data]

            # Check if all images have the same shape
            shapes = [img.shape for img in images]
            if len(set(shapes)) > 1:
                # Different sizes - resize all to the largest
                max_h = max(s[0] for s in shapes)
                max_w = max(s[1] for s in shapes)
                images = [_resize_image(img, max_h, max_w) for img in images]

            data = np.stack(images)
        else:
            data = to_numpy(data)

        # Ensure 3D shape (N, H, W)
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        # Avoid redundant copy: np.asarray is a no-op when already float32 + contiguous
        if data.dtype == np.float32:
            self._data = np.array(data, dtype=np.float32, copy=True)
        else:
            self._data = np.asarray(data, dtype=np.float32)
        # Store originals for rotation reset: views into _data (no copy).
        # Only materialized as independent copies when a rotation is applied.
        self._data_original = [self._data[i] for i in range(self._data.shape[0])]
        self._originals_are_views = True
        self.n_images = int(data.shape[0])
        self.height = int(data.shape[1])
        self.width = int(data.shape[2])
        self.image_rotations = [0] * self.n_images

        # Labels
        if labels is None:
            self.labels = [f"Image {i+1}" for i in range(self.n_images)]
        else:
            self.labels = list(labels)

        # Options
        self.title = title
        self.cmap = cmap
        # Resolve sampling + units to scalar pixel_size + pixel_unit (column axis).
        # Scalar shorthand: sampling=0.5 → (0.5, 0.5). units="nm" → ["nm", "nm"].
        if sampling is None:
            self.pixel_size = 0.0
        elif isinstance(sampling, (int, float)):
            self.pixel_size = float(sampling)
        else:
            self.pixel_size = float(sampling[-1])
        if units is None:
            self.pixel_unit = "pixels"
        elif isinstance(units, str):
            self.pixel_unit = units
        else:
            self.pixel_unit = str(units[-1])
        self.scale_bar_visible = scale_bar_visible
        self.size = size
        self.smooth = smooth
        # view_box sugar: sets zoom + zoom_row/col to center on box
        if view_box is not None:
            r0, r1, c0, c1 = [float(v) for v in view_box]
            box_h = max(1.0, r1 - r0)
            box_w = max(1.0, c1 - c0)
            zoom = float(min(self.height / box_h, self.width / box_w))
            zoom_row = (r0 + r1) / 2
            zoom_col = (c0 + c1) / 2
        self.initial_zoom = zoom
        self.zoom_row = zoom_row
        self.zoom_col = zoom_col
        # Auto-link zoom + pan in gallery (n_images >= 2) so dragging one panel
        # follows the other — typical compare/diff workflow. Single image: no-op.
        self.link_zoom = (self.n_images >= 2) if link_zoom is None else link_zoom
        self.link_pan = (self.n_images >= 2) if link_pan is None else link_pan
        self.link_contrast = link_contrast
        self.diff_mode = diff_mode if self.n_images >= 2 else False
        if show_fft and self.height * self.width > 2048 * 2048:
            warnings.warn(
                f"FFT on {self.height}×{self.width} image ({self.height * self.width / 1e6:.1f}M pixels) "
                f"may be slow. Consider using ROI FFT for a sub-region.",
                stacklevel=2,
            )
        self.show_fft = show_fft
        self.fft_window = fft_window
        self.show_controls = show_controls
        self.show_stats = show_stats
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.offline = offline
        # Accept scalar OR list for vmin/vmax. List → per-image (vmins/vmaxs).
        if isinstance(vmin, (list, tuple)) or isinstance(vmax, (list, tuple)):
            n = self.n_images
            def _expand(v):
                if v is None:
                    return [None] * n
                if isinstance(v, (list, tuple)):
                    if len(v) != n:
                        raise ValueError(f"vmin/vmax list has length {len(v)} but n_images is {n}. Pass a list of length {n} or a scalar to apply uniformly.")
                    return [None if x is None else float(x) for x in v]
                return [float(v)] * n
            self.vmins = _expand(vmin)
            self.vmaxs = _expand(vmax)
            self.vmin = None
            self.vmax = None
        else:
            self.vmin = vmin
            self.vmax = vmax
        self.ncols = ncols

        # Auto-bin for display: keep full-res in _data, send binned to JS.
        # GPU memory budget: ~2 GB for display buffers (128 MB per image at 4K).
        # At 4K: max ~16 full-res. Beyond that, auto-downsample.
        if display_bin == "auto":
            # Each 4K image needs ~192 MB GPU buffers (float32 + RGBA + read)
            # Tested: 12×4K (2.3 GB) works, 24×4K (4.6 GB) OOMs
            # Budget: 2.5 GB allows 12×4K full-res, bins above that
            gpu_budget_mb = self._GPU_DISPLAY_BUDGET_MB
            per_image_mb = (self.height * self.width * 4 * 3) / (1024 * 1024)  # 3 buffers
            total_mb = self.n_images * per_image_mb
            if total_mb > gpu_budget_mb:
                # Find minimum bin factor to fit
                for bf in [2, 4, 8]:
                    binned_mb = self.n_images * per_image_mb / (bf * bf)
                    if binned_mb <= gpu_budget_mb:
                        self._display_bin = bf
                        break
                else:
                    self._display_bin = 8
        elif isinstance(display_bin, int) and display_bin > 1:
            self._display_bin = display_bin

        if self._display_bin > 1:
            from quantem.widget.array_utils import bin2d
            orig_h, orig_w = self._data.shape[1], self._data.shape[2]
            self._display_data = bin2d(self._data, factor=self._display_bin, mode="mean")
            self.height = int(self._display_data.shape[1])
            self.width = int(self._display_data.shape[2])
            if self.pixel_size > 0:
                self.pixel_size = self.pixel_size * self._display_bin
            self._display_bin_factor = self._display_bin
            if verbose:
                print(f"  Display bin {self._display_bin}×: {orig_h}×{orig_w} → {self.height}×{self.width} ({self._display_data.nbytes // 1024 // 1024} MB)")
        else:
            self._display_data = self._data
            self._display_bin_factor = 1

        # Compute initial stats (from full-res data)
        self._compute_all_stats()

        # Send display data to JS (possibly binned)
        self._update_all_frames()

        self.selected_idx = 0

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

        # Stash wall-clock start on the instance; the observer below prints the
        # TRUE end-to-end time after JS signals first paint.  The Python-only
        # __init__ number is misleading for widget UX: a widget is not "done"
        # until the browser has painted its first frame.
        self._init_t0 = _t0
        self._init_py_elapsed_ms = (_time.perf_counter() - _t0) * 1000
        self.observe(self._on_first_render, names=["_js_rendered"])

    def _on_first_render(self, change):
        import time as _time
        if not change.get("new"):
            return
        total_ms = (_time.perf_counter() - self._init_t0) * 1000
        py_ms = self._init_py_elapsed_ms
        shape = (f"{self.n_images}×{self.height}×{self.width}"
                 if self.n_images > 1 else f"{self.height}×{self.width}")
        mem = self._data.nbytes
        mem_str = f"{mem / (1 << 20):.0f} MB" if mem >= 1 << 20 else f"{mem / (1 << 10):.0f} KB"
        # Expose as attributes so tests and notebooks can assert on them.
        # These are the ground truth for "did JS actually paint": if they're
        # None, the JS side never signaled first render.
        self.render_total_ms = int(total_ms)
        self.render_python_build_ms = int(py_ms)
        self.render_wire_js_ms = int(total_ms - py_ms)
        if not getattr(self, "_verbose", True):
            return
        print(
            f"Show2D: {shape} {mem_str}: "
            f"rendered in {total_ms:.0f} ms (Python build {py_ms:.0f} ms, "
            f"wire+JS {total_ms - py_ms:.0f} ms)",
            flush=True,
        )
        # Detach observer: one-shot, we only care about the first paint.
        try:
            self.unobserve(self._on_first_render, names=["_js_rendered"])
        except (ValueError, KeyError):
            pass

    def __repr__(self) -> str:
        if self.n_images > 1:
            shape = f"{self.n_images}×{self.height}×{self.width}"
            return f"Show2D({shape}, idx={self.selected_idx}, cmap={self.cmap})"
        return f"Show2D({self.height}×{self.width}, cmap={self.cmap})"

    def _repr_mimebundle_(self, **kwargs):
        """Return widget view + (optionally) static PNG fallback.

        Live Jupyter renders the interactive widget; the PNG fallback is only
        consumed by nbsphinx / GitHub / nbviewer when the widget view cannot be
        rendered.  Building the fallback runs matplotlib over every gallery image
        (~1.7 s for a 30×512² stack) and that cost pays off only in static builds.
        Gate it behind ``QUANTEM_WIDGET_STATIC_FALLBACK=1`` so interactive sessions
        return immediately.
        """
        bundle = super()._repr_mimebundle_(**kwargs)
        if not os.environ.get("QUANTEM_WIDGET_STATIC_FALLBACK"):
            return bundle
        data_dict = bundle[0] if isinstance(bundle, tuple) else bundle
        n = self.n_images
        ncols = min(self.ncols, n)
        nrows = math.ceil(n / ncols)
        cell = 4
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(cell * ncols, cell * nrows),
            squeeze=False,
        )
        max_preview = 256
        for i in range(nrows * ncols):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            if i < n:
                img = self._data[i]
                h, w = img.shape
                if h > max_preview or w > max_preview:
                    step = max(h // max_preview, w // max_preview, 1)
                    img = img[::step, ::step]
                ax.imshow(img, cmap=self.cmap, origin="upper")
                ax.set_title(self.labels[i], fontsize=10)
            ax.axis("off")
        if self.title:
            fig.suptitle(self.title, fontsize=12)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        data_dict["image/png"] = base64.b64encode(buf.getvalue()).decode("ascii")
        if isinstance(bundle, tuple):
            return (data_dict, bundle[1])
        return data_dict

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.log_scale:
            frame = np.log1p(np.maximum(frame, 0))
        if self.vmin is not None and self.vmax is not None:
            vmin = float(self.vmin)
            vmax = float(self.vmax)
            if self.log_scale:
                vmin = float(np.log1p(max(vmin, 0)))
                vmax = float(np.log1p(max(vmax, 0)))
        elif self.auto_contrast:
            vmin = float(np.percentile(frame, 2))
            vmax = float(np.percentile(frame, 98))
        else:
            vmin = float(frame.min())
            vmax = float(frame.max())
        if vmax > vmin:
            normalized = np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255)
            return normalized.astype(np.uint8)
        return np.zeros(frame.shape, dtype=np.uint8)

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        idx: int | None = None,
        format: str | None = None,
        dpi: int = 150,
        title: bool | str = False,
        colorbar: bool = False,
        scalebar: bool = False,
    ) -> pathlib.Path:
        """Save current image as PNG, PDF, or TIFF.

        When ``title``, ``colorbar``, or ``scalebar`` are enabled, the output
        is a publication-quality figure rendered via matplotlib. Otherwise a
        raw colormapped image is saved directly (faster, exact pixel output).

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        idx : int, optional
            Image index in gallery mode. Defaults to current selected_idx.
        format : str, optional
            'png', 'pdf', or 'tiff'. If omitted, inferred from file extension.
        dpi : int, default 150
            Output DPI.
        title : bool or str, default False
            ``True`` uses the widget title, a string sets a custom title.
        colorbar : bool, default False
            Include a colorbar showing the intensity mapping.
        scalebar : bool, default False
            Include a scale bar (requires ``pixel_size > 0``).

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        from matplotlib import colormaps
        from PIL import Image

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf", "tiff", "tif"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'.")

        i = idx if idx is not None else self.selected_idx
        if i < 0 or i >= self.n_images:
            raise IndexError(f"Image index {i} out of range [0, {self.n_images})")

        frame = self._data[i]
        normalized = self._normalize_frame(frame)
        cmap_fn = colormaps.get_cmap(self.cmap)
        path.parent.mkdir(parents=True, exist_ok=True)

        use_figure = title or colorbar or scalebar
        if not use_figure:
            rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)
            img = Image.fromarray(rgba)
            if fmt == "pdf":
                Image.init()
                img = img.convert("RGB")
            img.save(str(path), dpi=(dpi, dpi))
            return path

        # Publication-quality figure via matplotlib
        h, w = frame.shape
        aspect = h / w
        fig_w = 6
        fig, ax = plt.subplots(figsize=(fig_w, fig_w * aspect))
        im = ax.imshow(normalized, cmap=cmap_fn, vmin=0, vmax=255, origin="upper")
        ax.axis("off")

        if title:
            label = title if isinstance(title, str) else self.title
            if label:
                ax.set_title(label, fontsize=14, fontweight="bold", pad=8)

        if colorbar:
            # Map 0–255 back to data-space values for tick labels
            if self.log_scale:
                frame_proc = np.log1p(np.maximum(frame, 0))
            else:
                frame_proc = frame
            if self.vmin is not None and self.vmax is not None:
                dmin = float(self.vmin)
                dmax = float(self.vmax)
                if self.log_scale:
                    dmin = float(np.log1p(max(dmin, 0)))
                    dmax = float(np.log1p(max(dmax, 0)))
            elif self.auto_contrast:
                dmin = float(np.percentile(frame_proc, 2))
                dmax = float(np.percentile(frame_proc, 98))
            else:
                dmin = float(frame_proc.min())
                dmax = float(frame_proc.max())
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            n_ticks = 5
            tick_positions = np.linspace(0, 255, n_ticks)
            tick_labels = [f"{dmin + (dmax - dmin) * t / 255:.4g}" for t in tick_positions]
            cb.set_ticks(tick_positions)
            cb.set_ticklabels(tick_labels)

        if scalebar and self.pixel_size > 0:
            # Compute a nice scale bar length
            target_frac = 0.2  # ~20% of image width
            raw_length_px = target_frac * w
            raw_length_phys = raw_length_px * self.pixel_size  # in Å
            nice = _round_to_nice(raw_length_phys)
            bar_px = nice / self.pixel_size
            if nice >= 10:
                label_text = f"{nice / 10:.4g} nm"
            else:
                label_text = f"{nice:.4g} Å"
            margin = 0.03
            bar_y = h * (1 - margin) - 2
            bar_x = w * (1 - margin) - bar_px
            ax.plot([bar_x, bar_x + bar_px], [bar_y, bar_y],
                    color="white", linewidth=3, solid_capstyle="butt")
            ax.plot([bar_x, bar_x + bar_px], [bar_y, bar_y],
                    color="black", linewidth=1, solid_capstyle="butt")
            ax.text(bar_x + bar_px / 2, bar_y - h * 0.02, label_text,
                    color="white", fontsize=10, fontweight="bold",
                    ha="center", va="bottom",
                    path_effects=[
                        matplotlib.patheffects.withStroke(linewidth=2, foreground="black")
                    ])

        fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                    facecolor="white", pad_inches=0.1)
        plt.close(fig)
        return path

    def state_dict(self):
        return {
            "title": self.title,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "show_stats": self.show_stats,
            "show_fft": self.show_fft,
            "fft_window": self.fft_window,
            "show_controls": self.show_controls,
            "pixel_size": self.pixel_size,
            "pixel_unit": self.pixel_unit,
            "scale_bar_visible": self.scale_bar_visible,
            "size": self.size,
            "smooth": self.smooth,
            "initial_zoom": self.initial_zoom,
            "vmins": self.vmins,
            "vmaxs": self.vmaxs,
            "link_zoom": self.link_zoom,
            "link_pan": self.link_pan,
            "link_contrast": self.link_contrast,
            "zoom_row": self.zoom_row,
            "zoom_col": self.zoom_col,
            "diff_mode": self.diff_mode,
            "ncols": self.ncols,
            "selected_idx": self.selected_idx,
            "roi_active": self.roi_active,
            "roi_list": self.roi_list,
            "roi_selected_idx": self.roi_selected_idx,
            "profile_line": self.profile_line,
            "image_rotations": list(self.image_rotations),
            "display_bin": self._display_bin,
        }

    def save(self, path: str):
        save_state_file(path, "Show2D", self.state_dict())

    def load_state_dict(self, state):
        for key, val in state.items():
            # Silent migrations for renamed keys in older saved state files.
            if key == "pixel_size_angstrom":
                key = "pixel_size"
            elif key == "canvas_size":
                key = "size"
            if key == "display_bin":
                self._display_bin = val
                continue
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self):
        """Print a human-readable snapshot of the widget's current state.

        Reports image dimensions and pixel size, data min/max/mean, display
        settings (colormap, contrast, scale, FFT), active ROIs and profile
        line, per-image rotations, and the most recent render timings.
        """
        lines = [self.title or "Show2D", "═" * 32]
        if self.n_images > 1:
            lines.append(f"Image:    {self.n_images}×{self.height}×{self.width} ({self.ncols} cols)")
        else:
            lines.append(f"Image:    {self.height}×{self.width}")
        if self.pixel_size > 0:
            ps = self.pixel_size
            if ps >= 10:
                lines[-1] += f" ({ps / 10:.2f} nm/px)"
            else:
                lines[-1] += f" ({ps:.2f} Å/px)"
        if hasattr(self, "_data") and self._data is not None:
            arr = self._data
            lines.append(f"Data:     min={float(arr.min()):.4g}  max={float(arr.max()):.4g}  mean={float(arr.mean()):.4g}")
        cmap = self.cmap
        scale = "log" if self.log_scale else "linear"
        if self.vmin is not None and self.vmax is not None:
            contrast = f"vmin={self.vmin:.4g}, vmax={self.vmax:.4g}"
        elif self.auto_contrast:
            contrast = "auto contrast"
        else:
            contrast = "manual contrast"
        display = f"{cmap} | {contrast} | {scale}"
        if self.show_fft:
            display += " | FFT"
            if not self.fft_window:
                display += " (no window)"
        lines.append(f"Display:  {display}")
        if self.roi_active and self.roi_list:
            lines.append(f"ROI:      {len(self.roi_list)} region(s)")
        if self.profile_line:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            lines.append(f"Profile:  ({p0['row']:.0f}, {p0['col']:.0f}) → ({p1['row']:.0f}, {p1['col']:.0f})")
        non_zero = [(i, r * 90) for i, r in enumerate(self.image_rotations) if r % 4 != 0]
        if non_zero:
            parts = [f"#{i}={deg}°" for i, deg in non_zero]
            lines.append(f"Rotated:  {', '.join(parts)}")
        rt = getattr(self, "render_total_ms", None)
        if rt is not None:
            pb = getattr(self, "render_python_build_ms", 0)
            wj = getattr(self, "render_wire_js_ms", 0)
            lines.append(f"Rendered: {rt} ms total (Python build {pb} ms, wire+JS {wj} ms)")
        else:
            lines.append("Rendered: (pending first browser paint)")
        print("\n".join(lines))

    def _compute_all_stats(self):
        """Compute statistics for all images (vectorized over all frames)."""
        # Vectorized reduction over (H, W) is faster than per-image loops
        # for large galleries (e.g. 12×4096×4096: 164ms vs 191ms).
        axes = (1, 2) if self._data.ndim == 3 else None
        self.stats_mean = np.mean(self._data, axis=axes).ravel().tolist()
        self.stats_min = np.min(self._data, axis=axes).ravel().tolist()
        self.stats_max = np.max(self._data, axis=axes).ravel().tolist()
        self.stats_std = np.std(self._data, axis=axes).ravel().tolist()

    def _update_all_frames(self):
        """Send display data to JS (possibly binned for large galleries)."""
        data = self._display_data if self._display_data is not None else self._data
        if self.offline:
            # Quantize to uint8 against global (min, max). Drops standalone-HTML
            # bytes by 4x. Display-only mode — colormap reduces to 256 levels
            # anyway, eye can't tell the difference.
            arr = np.ascontiguousarray(data, dtype=np.float32)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                lo, hi = 0.0, 1.0
            else:
                lo = float(finite.min())
                hi = float(finite.max())
            rng = hi - lo if hi > lo else 1.0
            quantized = np.clip((arr - lo) * (255.0 / rng), 0, 255).astype(np.uint8)
            self._offline_min = lo
            self._offline_max = hi
            self.frame_bytes = quantized.tobytes()
        else:
            self.frame_bytes = data.tobytes()

    def _apply_rotations(self):
        """Re-rotate each displayed image from its original by ``image_rotations[i] * 90°``.

        This is purely a display-time reorientation of each 2D image via
        ``np.rot90`` — it is NOT scan rotation (which would rotate the
        scan grid in a 4D-STEM dataset). Originals are kept in
        ``_data_original`` so successive rotations compose from the
        unrotated source rather than accumulating interpolation error.
        Mixed shapes after rotation are center-padded to a common size.
        """
        # Materialize originals as independent copies only when a non-zero
        # rotation exists (they start as views into _data to avoid 800MB copy at init)
        has_rotation = any(
            (self.image_rotations[i] if i < len(self.image_rotations) else 0) % 4 != 0
            for i in range(len(self._data_original))
        )
        # No-rotation fast path: skip 30+ MB of redundant tobytes + stats recomputation
        # on every widget init.  The observer fires once when image_rotations = [0]*n
        # is assigned in __init__; without this guard that triggered a full frame
        # rebuild + stats recompute for a no-op.
        if not has_rotation and self._originals_are_views:
            return
        if self._originals_are_views and has_rotation:
            self._data_original = [img.copy() for img in self._data_original]
            self._originals_are_views = False
        rotated = []
        for i, orig in enumerate(self._data_original):
            k = self.image_rotations[i] if i < len(self.image_rotations) else 0
            k = k % 4
            if k == 0:
                rotated.append(orig)
            else:
                rotated.append(np.rot90(orig, k=k))
        # If shapes differ after rotation, center-pad all to max dims
        shapes = [img.shape for img in rotated]
        if len(set(shapes)) > 1:
            max_h = max(s[0] for s in shapes)
            max_w = max(s[1] for s in shapes)
            padded = []
            for img in rotated:
                h, w = img.shape
                pad_top = (max_h - h) // 2
                pad_bot = max_h - h - pad_top
                pad_left = (max_w - w) // 2
                pad_right = max_w - w - pad_left
                padded.append(np.pad(img, ((pad_top, pad_bot), (pad_left, pad_right)), mode="constant", constant_values=0))
            rotated = padded
        self._data = np.stack(rotated).astype(np.float32)
        # Recompute display data if binning is active
        if self._display_bin > 1:
            from quantem.widget.array_utils import bin2d
            self._display_data = bin2d(self._data, factor=self._display_bin, mode="mean")
        else:
            self._display_data = self._data
        display = self._display_data if self._display_data is not None else self._data
        self.height = int(display.shape[1])
        self.width = int(display.shape[2])
        self._compute_all_stats()
        self._update_all_frames()

    @traitlets.observe("image_rotations")
    def _on_image_rotations_changed(self, change):
        if hasattr(self, "_data_original"):
            self._apply_rotations()

    def rotate(self, idx: int, angle: int) -> Self:
        """Rotate image ``idx`` by ``angle`` degrees (CCW-positive, matches np.rot90).

        Rotation convention follows ``np.rot90``::

            angle | image_rotations | np.rot90 k | direction
            ------+-----------------+------------+----------
              90  |        1        |     1      | 90° CCW
             180  |        2        |     2      | 180°
             -90  |        3        |     3      | 90° CW
             360  |        0        |     0      | identity

        Parameters
        ----------
        idx : int
            Image index in the gallery (0-based).
        angle : int
            Rotation angle in degrees (must be a multiple of 90).
            Positive = counter-clockwise, negative = clockwise.

        Returns
        -------
        Self
        """
        if angle % 90 != 0:
            raise ValueError(f"Rotation angle must be a multiple of 90 (got {angle}). Use 0, 90, 180, 270, or -90, -180, -270.")
        if idx < 0 or idx >= self.n_images:
            raise IndexError(f"Image index {idx} out of range [0, {self.n_images})")
        k = (angle // 90) % 4
        rots = list(self.image_rotations)
        while len(rots) < self.n_images:
            rots.append(0)
        rots[idx] = (rots[idx] + k) % 4
        self.image_rotations = rots
        return self

    def _sample_profile(self, row0, col0, row1, col1):
        img = self._data[self.selected_idx]
        h, w = img.shape
        dc, dr = col1 - col0, row1 - row0
        length = (dc**2 + dr**2) ** 0.5
        n = max(2, int(np.ceil(length)))
        t = np.linspace(0, 1, n)
        cs = col0 + t * dc
        rs = row0 + t * dr
        ci = np.floor(cs).astype(int)
        ri = np.floor(rs).astype(int)
        cf = cs - ci
        rf = rs - ri
        c0c = np.clip(ci, 0, w - 1)
        c1c = np.clip(ci + 1, 0, w - 1)
        r0c = np.clip(ri, 0, h - 1)
        r1c = np.clip(ri + 1, 0, h - 1)
        return (img[r0c, c0c] * (1 - cf) * (1 - rf) +
                img[r0c, c1c] * cf * (1 - rf) +
                img[r1c, c0c] * (1 - cf) * rf +
                img[r1c, c1c] * cf * rf).astype(np.float32)

    def set_profile(self, start: tuple, end: tuple):
        """Set a line profile between two points (image pixel coordinates).

        Parameters
        ----------
        start : tuple of (row, col)
            Start point in pixel coordinates.
        end : tuple of (row, col)
            End point in pixel coordinates.
        """
        row0, col0 = start
        row1, col1 = end
        self.profile_line = [
            {"row": float(row0), "col": float(col0)},
            {"row": float(row1), "col": float(col1)},
        ]

    def clear_profile(self):
        """Clear the current line profile."""
        self.profile_line = []

    def _upsert_selected_roi(self, updates: dict):
        rois = list(self.roi_list)
        color_cycle = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"]
        defaults = {
            "shape": "square",
            "row": int(self.height // 2),
            "col": int(self.width // 2),
            "radius": 10,
            "radius_inner": 5,
            "width": 20,
            "height": 20,
            "line_width": 2,
            "highlight": False,
            "visible": True,
            "locked": False,
        }
        if self.roi_selected_idx >= 0 and self.roi_selected_idx < len(rois):
            current = {**defaults, **rois[self.roi_selected_idx]}
            if not current.get("color"):
                current["color"] = color_cycle[self.roi_selected_idx % len(color_cycle)]
            rois[self.roi_selected_idx] = {**current, **updates}
        else:
            rois.append({**defaults, "color": color_cycle[len(rois) % len(color_cycle)], **updates})
            self.roi_selected_idx = len(rois) - 1
        self.roi_list = rois
        self.roi_active = True

    def add_roi(self, row: int | None = None, col: int | None = None, shape: str = "square") -> Self:
        with self.hold_sync():
            self.roi_selected_idx = -1
            self._upsert_selected_roi({
                "shape": shape,
                "row": int(self.height // 2 if row is None else row),
                "col": int(self.width // 2 if col is None else col),
            })
        return self

    def clear_rois(self) -> Self:
        with self.hold_sync():
            self.roi_list = []
            self.roi_selected_idx = -1
            self.roi_active = False
        return self

    def delete_selected_roi(self) -> Self:
        idx = int(self.roi_selected_idx)
        if idx < 0 or idx >= len(self.roi_list):
            return self
        with self.hold_sync():
            rois = [roi for i, roi in enumerate(self.roi_list) if i != idx]
            self.roi_list = rois
            self.roi_selected_idx = min(idx, len(rois) - 1) if rois else -1
            if not rois:
                self.roi_active = False
        return self

    def set_roi(self, row: int, col: int, radius: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "circle", "row": int(row), "col": int(col), "radius": int(radius)})
        return self

    def roi_circle(self, radius: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "circle", "radius": int(radius)})
        return self

    def roi_square(self, half_size: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "square", "radius": int(half_size)})
        return self

    def roi_rectangle(self, width: int = 20, height: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "rectangle", "width": int(width), "height": int(height)})
        return self

    def roi_annular(self, inner: int = 5, outer: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "annular", "radius_inner": int(inner), "radius": int(outer)})
        return self

    @property
    def profile(self):
        """Get profile line endpoints as [(row0, col0), (row1, col1)] or [].

        Returns
        -------
        list of tuple
            Line endpoints in pixel coordinates, or empty list if no profile.
        """
        return [(p["row"], p["col"]) for p in self.profile_line]

    @property
    def profile_values(self):
        """Get intensity values along the profile line as a numpy array.

        Returns
        -------
        np.ndarray or None
            Float32 array of sampled intensities, or None if no profile.
        """
        if len(self.profile_line) < 2:
            return None
        p0, p1 = self.profile_line
        return self._sample_profile(p0["row"], p0["col"], p1["row"], p1["col"])

    @property
    def profile_distance(self):
        """Get total distance of the profile line in calibrated units.

        Returns
        -------
        float or None
            Distance in angstroms (if pixel_size > 0) or pixels.
            None if no profile line is set.
        """
        if len(self.profile_line) < 2:
            return None
        p0, p1 = self.profile_line
        dc = p1["col"] - p0["col"]
        dr = p1["row"] - p0["row"]
        dist_px = (dc**2 + dr**2) ** 0.5
        if self.pixel_size > 0:
            return dist_px * self.pixel_size
        return dist_px

