"""
show3d: Interactive 3D stack viewer widget with advanced features.

For viewing a stack of 2D images (e.g., defocus sweep, time series, z-stack, movies).
Includes playback controls, statistics, ROI selection, FFT, and more.
"""

import gc
import http.server
import json
import math
import pathlib
import secrets
import sys
import threading
import time
import tempfile
import urllib.parse
import warnings
import weakref
from enum import Enum
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.config_utils import (
    _config_float,
    _load_quantem_config,
    _normalize_rotation_deg,
    _pixel_size_from_quantem_config,
    _post_crop_from_quantem_config,
    _rotate_stack_inplane,
)
from quantem.widget.show2d import _reject_unknown_kwargs
from quantem.widget.state import (
    resolve_widget_version,
    save_state_file,
    unwrap_state_payload,
)

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _all_finite(arr: np.ndarray, *, chunk_size: int = 1_000_000) -> bool:
    """Chunked full finite scan without allocating one giant boolean array."""
    flat = np.ravel(arr)
    for start in range(0, flat.size, chunk_size):
        if not np.isfinite(flat[start : start + chunk_size]).all():
            return False
    return True


def _normalize_padding(padding: object) -> tuple[int, int]:
    """Normalize constructor spatial padding to (rows, cols)."""
    if isinstance(padding, bool):
        raise ValueError("padding must be a non-negative int or (rows, cols), not bool")
    if isinstance(padding, (int, np.integer)):
        pad = (int(padding), int(padding))
    else:
        try:
            vals = tuple(padding)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("padding must be a non-negative int or (rows, cols)") from exc
        if len(vals) != 2:
            raise ValueError("padding tuple must contain exactly two values: (rows, cols)")
        if any(isinstance(v, bool) for v in vals):
            raise ValueError("padding values must be non-negative integers, not bool")
        pad = (int(vals[0]), int(vals[1]))
    if pad[0] < 0 or pad[1] < 0:
        raise ValueError(f"padding must be non-negative, got {pad}")
    return pad


def _pad_stack(data: np.ndarray, padding: tuple[int, int],
               mode: str = "median") -> np.ndarray:
    """Pad a (N, H, W) stack with a spatial border.

    ``mode="median"`` (default) fills with the stack median so the border is
    contrast-neutral - it sits at the histogram centre and does not shift the
    per-frame percentile clip. ``mode="constant"`` fills with 0 (black), which
    drags the low percentile down and darkens the real image.

    Speed: ``np.pad`` has its own ``mode="median"`` but it recomputes a median
    per pad-vector and is slow on a large stack. We compute ONE scalar median
    (sampled to ~1e6 elements - the border value does not need an exact median)
    and do a constant fill, so the pad itself is a fast memset.
    """
    pad_y, pad_x = padding
    if pad_y == 0 and pad_x == 0:
        return data
    if mode == "constant":
        fill = 0.0
    elif mode == "median":
        flat = data.ravel()
        sample = flat[:: max(1, flat.size // 1_000_000)]
        fill = float(np.median(sample))
    else:
        raise ValueError(f"pad_mode must be 'median' or 'constant', got {mode!r}")
    return np.pad(data, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)),
                  mode="constant", constant_values=fill)


def _normalize_crop(crop: object) -> tuple[int, int, int, int]:
    """Normalize constructor spatial crop to (top, bottom, left, right)."""
    if isinstance(crop, bool):
        raise ValueError("crop must be a non-negative int, (rows, cols), or (top, bottom, left, right), not bool")
    if isinstance(crop, (int, np.integer)):
        c = int(crop)
        vals = (c, c, c, c)
    else:
        try:
            raw = tuple(crop)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("crop must be a non-negative int, (rows, cols), or (top, bottom, left, right)") from exc
        if len(raw) == 2:
            if any(isinstance(v, bool) for v in raw):
                raise ValueError("crop values must be non-negative integers, not bool")
            rows, cols = int(raw[0]), int(raw[1])
            vals = (rows, rows, cols, cols)
        elif len(raw) == 4:
            if any(isinstance(v, bool) for v in raw):
                raise ValueError("crop values must be non-negative integers, not bool")
            vals = tuple(int(v) for v in raw)
        else:
            raise ValueError("crop tuple must contain 2 values (rows, cols) or 4 values (top, bottom, left, right)")
    if any(v < 0 for v in vals):
        raise ValueError(f"crop must be non-negative, got {vals}")
    return vals


def _crop_stack(data: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a (N, H, W) stack by (top, bottom, left, right) pixels."""
    top, bottom, left, right = crop
    if top == bottom == left == right == 0:
        return data
    h, w = int(data.shape[1]), int(data.shape[2])
    if top + bottom >= h or left + right >= w:
        raise ValueError(
            f"crop {crop} removes the entire image: input spatial shape is {(h, w)}"
        )
    row_stop = h - bottom if bottom else h
    col_stop = w - right if right else w
    return data[:, top:row_stop, left:col_stop]


class Colormap(str, Enum):
    """Available colormaps for image display."""

    INFERNO = "inferno"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    MAGMA = "magma"
    HOT = "hot"
    GRAY = "gray"
    HSV = "hsv"
    TURBO = "turbo"
    CIVIDIS = "cividis"
    RDBU = "RdBu"
    RDBU_R = "RdBu_r"
    SEISMIC = "seismic"
    TWILIGHT = "twilight"
    TWILIGHT_SHIFTED = "twilight_shifted"

    def __str__(self) -> str:
        return self.value


# Names that the JS bundle's GPUColormapEngine knows about. Keep in sync with
# js/colormaps.ts COLORMAPS table.
_VALID_CMAPS = frozenset({
    "inferno", "viridis", "plasma", "magma", "hot", "gray", "hsv", "turbo",
    "cividis", "RdBu", "RdBu_r", "seismic", "twilight", "twilight_shifted",
})

# Keep a single synced playback chunk small enough for the Jupyter/AnyWidget
# Comm path to survive. The data is still exact float32; large stacks move as
# sliding windows instead of browser-hostile 256 MB+ messages.
_MAX_PLAYBACK_CHUNK_BYTES = 128 * 1024 * 1024
_MAX_PLAYBACK_FPS = 30.0


class _Show3DFrameHTTPServer(http.server.ThreadingHTTPServer):
    """Tiny localhost server for exact float32 frame reads."""

    allow_reuse_address = True
    daemon_threads = True


class Show3D(anywidget.AnyWidget):
    """
    Interactive 3D stack viewer for sequential 2D images.

    Renders an (N, H, W) stack through a WebGPU canvas (CPU fallback when WebGPU
    is unavailable) with a sliding prefetch buffer for smooth playback. The full
    stack is held once on the Python side; scrubbing and playback ship individual
    frames or chunks over the Jupyter Comm channel. Common use cases: defocus
    sweep, time series, depth stack, in-situ movie, side-by-side trial comparison.

    Features
    --------
    - Interactive scrubber + ``play`` / ``pause`` / ``stop`` / ``goto`` with bookmark and loop range
    - Per-frame statistics, log scale, percentile auto-contrast, manual vmin/vmax
    - Diff mode (vs first frame or vs previous frame) for delta visualization
    - ROI tools (circle / square / rectangle / annular) with per-frame timeseries
    - Line profiles sampled across the full stack
    - FFT panel with optional Hann window
    - Side-by-side multi-panel mode with linked or independent zoom / pan / contrast
    - Frame hiding (``hide``, ``show``, ``set_hidden``, ``show_all``) without rebuilding
    - Python-side PNG / PDF / TIFF single-frame save via ``save_image``
    - JSON state save/load via ``state_dict`` / ``load_state_dict`` / ``save``
    - Explicit ``free`` to release VRAM/RAM held by traitlets observers

    Parameters
    ----------
    data : array_like
        3D array of shape (N, height, width) where N is the stack dimension.
        Also accepts a 2D array (treated as a single-frame stack), a torch
        tensor (CPU or GPU, any dtype), or a quantem ``Dataset3d``. Complex
        input is rejected; cast to magnitude or phase first.
    labels : list[str] | None, optional
        Labels for each slice (e.g., ``["C10=-500nm", "C10=-400nm", ...]``).
        If ``None``, uses string slice indices.
    title : str, optional
        Title to display above the image.
    cmap : str or Colormap, default Colormap.MAGMA
        Colormap name. Use Colormap enum (Colormap.MAGMA, Colormap.VIRIDIS, etc.)
        or string ("magma", "viridis", "gray", "inferno", "plasma").
    vmin : float, optional
        Minimum value for colormap. If None, uses data min.
    vmax : float, optional
        Maximum value for colormap. If None, uses data max.
    pixel_size : float, optional
        Pixel size in Å for scale bar display.
    log_scale : bool, default False
        Use log scale for intensity mapping.
    auto_contrast : bool, default True
        Use percentile-based contrast (ignores vmin/vmax).
    percentile_low : float, default 0.5
        Lower percentile for auto-contrast.
    percentile_high : float, default 99.5
        Upper percentile for auto-contrast.
    fps : float, default 30.0
        Frames per second for playback, capped at 30.
    timestamps : list of float, optional
        Timestamps for each frame (e.g., seconds or dose values).
    timestamp_unit : str, default "s"
        Unit for timestamps (e.g., "s", "ms", "e/A2").
    size : int, default 0
        Canvas rendering size in CSS pixels (the on-screen width of the main
        viewport).  ``0`` uses the frontend default (500 px).  Pass e.g.
        ``size=800`` to enlarge for a presentation, or ``size=300`` to compress
        alongside a control panel.  This controls **display only** - the
        underlying stack resolution is never resampled; scrubbing and zoom
        still see every pixel of the full-resolution frame.
    padding : int or tuple[int, int], default 0
        Zero-valued spatial border added to every frame before display. Use
        this when aligned movies need extra field of view so shifted frames do
        not crop at the canvas edge. An int pads rows and columns equally;
        ``(rows, cols)`` pads each axis independently.
    crop : int or tuple[int, int] or tuple[int, int, int, int], default 0
        Spatial crop applied before padding. Use an int to crop all sides,
        ``(rows, cols)`` for symmetric row/column cropping, or
        ``(top, bottom, left, right)`` for side-specific cropping.
    max_cols : int, default 4
        Multi-panel grid wrap.  ``0`` = single row (no wrap), ``N>0`` = wrap into
        rows of at most ``N`` panels.  Default ``4`` is a good fit for a
        13"–16" laptop screen; bump to ``6`` on wide monitors or drop to ``3``
        for a portrait split layout.  Empty trailing cells in a partial last
        row are not rendered (transparent, non-interactive).
    panel_gap : int, default 10
        Gap in CSS pixels between adjacent panels.  ``0`` = flush (panels share
        an edge - useful for tiled montages), ``20`` = roomy (clear separation
        for slides).  Single-panel widgets ignore this.
    panel_title_font_size : int, default 11
        Font size in CSS pixels for the per-panel title drawn at the top of
        each multi-panel slot.  Bump to ``14–16`` for slide-projection clarity;
        drop to ``9`` to fit titles inside narrow panels on a small screen.
    show_panel_titles : bool, default True
        Draw the top-center per-panel title and frame counter on multi-panel
        canvases. Set ``False`` for clean GIF/video exports.
    show_resize_handles : bool, default True
        Render the bottom-right corner triangle on every real panel.  Dragging
        any handle resizes the entire multi-panel canvas (linked).  Set
        ``False`` to declutter a screenshot or printed figure where the
        operator already has the layout they want.
    show_zoom_indicator : bool, default True
        Draw the ``1.0×`` zoom readout at the bottom-left of every panel.
        Set ``False`` for clean static layouts or when the scale bar alone
        is enough to communicate scale.
    show_scale_bar : bool, default True
        Draw the bottom-right scale bar. When ``pixel_size`` is provided the
        label uses physical units; otherwise it shows pixel units. Set
        ``False`` for GIF/video exports or uncluttered figure captures.
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

    Example
    -------
    Instantiate Show3D with a stack of 2D images, drive playback, and persist
    the display state to disk:

    >>> import numpy as np
    >>> from quantem.widget import Show3D
    >>> stack = np.random.rand(12, 256, 256).astype(np.float32)
    >>> labels = [f"C10={c:.0f}nm" for c in np.linspace(-500, -200, 12)]
    >>> w = Show3D(stack, labels=labels, title="Defocus Sweep", pixel_size=0.5)
    >>> w.play()  # doctest: +SKIP
    >>> w.goto(3)  # doctest: +SKIP
    >>> w.save("show3d_state.json")  # doctest: +SKIP

    Notes
    -----
    - The stack is loaded once into ``self._data`` (``float32``); ``set_image``
      replaces the data without rebuilding the widget.
    - When the stack is large (>32 MB per frame), an internal display copy is
      binned for faster scrubbing; full-resolution data is kept for stats,
      ROIs, FFT, profiles, and direct image saving.
    - Multi-panel mode is auto-detected from input shape and configured via
      ``n_panels``, ``panel_titles``, and ``link_panels``.
    - Call ``free()`` before discarding the widget; ``del`` alone will not
      release VRAM because traitlets observers pin the refcount.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show3d.js"

    # =========================================================================
    # Core State
    # =========================================================================
    # GPU memory budget for display buffers (same as Show2D)
    _GPU_DISPLAY_BUDGET_MB = 2500

    slice_idx = traitlets.CInt(0).tag(sync=True)
    n_slices = traitlets.Int(1).tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    # Monotonic counter incremented each time frame_bytes is written. Defensive
    # against the case where traitlets.Bytes identity-compares to suppress the
    # trait change event when JS sees the same DataView wrapper - JS subscribes
    # to this counter as a guaranteed-changing dep so render effects always
    # re-fire on slice scrubs / playback ticks.
    frame_seq = traitlets.Int(0).tag(sync=True)
    # Offline/export mode packs the full display stack so JS can slice
    # client-side without a Python kernel. _offline_stack is the compact
    # uint8 path; _offline_float_stack is exact float32 for precision-preserving
    # standalone HTML exports.
    offline = traitlets.Bool(False).tag(sync=True)
    # True only on a clone written by export_html: forces the standalone HTML to
    # render on a light/white background regardless of the viewer's OS theme.
    # Decoupled from `offline` (which selects uint8 vs float data packing).
    _export_light = traitlets.Bool(False).tag(sync=True)
    _offline_stack = traitlets.Bytes(b"").tag(sync=True)
    _offline_float_stack = traitlets.Bytes(b"").tag(sync=True)
    _offline_min = traitlets.Float(0.0).tag(sync=True)
    _offline_max = traitlets.Float(1.0).tag(sync=True)
    # Frontend-triggered standalone HTML export. The request is JSON so repeated
    # exports of the same mode can include a unique id and still sync.
    export_request = traitlets.Unicode("").tag(sync=True)
    export_status = traitlets.Unicode("").tag(sync=True)
    export_enabled = traitlets.Bool(True).tag(sync=True)
    export_payload = traitlets.Bytes(b"").tag(sync=True)
    export_payload_id = traitlets.Unicode("").tag(sync=True)
    export_filename = traitlets.Unicode("").tag(sync=True)
    # Flipped True by JS after the first colormap pass has painted to canvas.
    # Drives the truthful timing print (end-to-end, not __init__-only).
    _js_rendered = traitlets.Bool(False).tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("plasma").tag(sync=True)
    dim_label = traitlets.Unicode("Frame").tag(sync=True)
    dim_sampling = traitlets.Float(1.0).tag(sync=True)
    dim_unit = traitlets.Unicode("").tag(sync=True)

    # Multi-Panel (side-by-side stacks, independent zoom by default with optional link)
    n_panels = traitlets.Int(1).tag(sync=True)
    panel_titles = traitlets.List(traitlets.Unicode()).tag(sync=True)
    panel_width_px = traitlets.Int(0).tag(sync=True)
    shared_panel_source = traitlets.Bool(False).tag(sync=True)
    separate_panel_frames = traitlets.Bool(False).tag(sync=True)
    # Per-panel "best frame" marker. One int per panel; -1 = unset. Used to flag
    # the user's preferred iteration / trial / focal slice without losing the
    # full stack. JS draws a gold star top-right of each panel when set.
    starred = traitlets.List(traitlets.Int()).tag(sync=True)
    # Real frame count per panel for stack comparison: stacks of different
    # lengths get auto-padded to the longest; this trait lets JS mark
    # "end-of-stack" frames (frame idx >= real[panel]). Empty = all real.
    panel_real_frames = traitlets.List(traitlets.Int()).tag(sync=True)
    # Single Link toggle controls both zoom and pan (independent axes proved confusing).
    link_panels = traitlets.Bool(True).tag(sync=True)
    link_contrast = traitlets.Bool(True).tag(sync=True)  # share vmin/vmax across panels
    # 0 = single row (no wrap). N > 0 = wrap into rows of at most N panels.
    max_cols = traitlets.Int(4).tag(sync=True)
    # Per-widget customization for multi-panel display.
    show_resize_handles = traitlets.Bool(True).tag(sync=True)
    show_zoom_indicator = traitlets.Bool(True).tag(sync=True)
    show_panel_titles = traitlets.Bool(True).tag(sync=True)
    panel_title_font_size = traitlets.Int(11).tag(sync=True)
    panel_gap = traitlets.Int(10).tag(sync=True)
    # Hover-x hide feature: enables UI to drop frames from scrubber without
    # rebuilding the widget. hidden_indices is the live state; visible_indices
    # is derived (read-only).
    hideable = traitlets.Bool(False).tag(sync=True)
    hidden_indices = traitlets.List(traitlets.Int()).tag(sync=True)

    # =========================================================================
    # Playback Controls
    # =========================================================================
    playing = traitlets.Bool(False).tag(sync=True)
    reverse = traitlets.Bool(False).tag(sync=True)  # Play in reverse direction
    boomerang = traitlets.Bool(True).tag(sync=True)  # Ping-pong playback
    fps = traitlets.Float(30.0).tag(sync=True)
    # Moving-window time average (temporal binning for noisy data). 1 = off.
    # The displayed frame is the mean of `avg_window` consecutive frames.
    # Edge behavior: the window slides inward to stay FULL-WIDTH rather than
    # shrinking - so frame 0 with window 5 averages frames [0..4], and the last
    # frame averages the final 5. This keeps denoising strength constant across
    # the stack, at the cost of the average not being centered on the displayed
    # index near the ends (frame 0 shows the mean centered ~2 frames in). Even
    # windows are front-biased (window 4 = [idx-2 .. idx+1]).
    avg_window = traitlets.Int(1).tag(sync=True)
    loop = traitlets.Bool(True).tag(sync=True)
    loop_start = traitlets.Int(0).tag(sync=True)  # Start frame for loop range
    loop_end = traitlets.Int(-1).tag(sync=True)  # End frame for loop (-1 = last)
    bookmarked_frames = traitlets.List(traitlets.Int()).tag(sync=True)
    playback_path = traitlets.List(traitlets.Int()).tag(sync=True)

    # =========================================================================
    # Statistics Panel
    # =========================================================================
    show_controls = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(False).tag(sync=True)
    stats_mean = traitlets.Float(0.0).tag(sync=True)
    stats_min = traitlets.Float(0.0).tag(sync=True)
    stats_max = traitlets.Float(0.0).tag(sync=True)
    stats_std = traitlets.Float(0.0).tag(sync=True)
    # =========================================================================
    # Display Options
    # =========================================================================
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(True).tag(sync=True)
    percentile_low = traitlets.Float(0.5).tag(sync=True)
    percentile_high = traitlets.Float(99.5).tag(sync=True)
    image_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    image_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmin_per_panel = traitlets.List(traitlets.Float(None, allow_none=True), default_value=[]).tag(sync=True)
    vmax_per_panel = traitlets.List(traitlets.Float(None, allow_none=True), default_value=[]).tag(sync=True)
    data_min = traitlets.Float(0.0).tag(sync=True)
    data_max = traitlets.Float(0.0).tag(sync=True)
    auto_vmins = traitlets.List(traitlets.Float()).tag(sync=True)
    auto_vmaxs = traitlets.List(traitlets.Float()).tag(sync=True)

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(0.0).tag(sync=True)  # 0 = no scale bar
    pixel_unit = traitlets.Unicode("A").tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    # Canvas smoothing: False = nearest-neighbor (sharp atoms); True = bilinear.
    smooth = traitlets.Bool(True).tag(sync=True)
    # Whole-stack rotation as k * 90 deg (k = 0..3). Applied in Python by rotating
    # _data and re-broadcasting frame_bytes; cheap for typical EM stacks.
    image_rotation = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # Timestamps / Dose
    # =========================================================================
    timestamps = traitlets.List(traitlets.Float()).tag(sync=True)
    timestamp_unit = traitlets.Unicode("s").tag(sync=True)

    # =========================================================================
    # ROI Selection
    # =========================================================================
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_list = traitlets.List([]).tag(sync=True)
    roi_selected_idx = traitlets.Int(-1).tag(sync=True)
    # roi_stats: Python-only readout. JS computes its own ROI stats locally
    # (allRoiStats useMemo in index.tsx), so don't ship over Comm every ROI move.
    roi_stats = traitlets.Dict({})
    roi_plot_data = traitlets.Bytes(b"").tag(sync=True)
    # =========================================================================
    # Sizing
    # =========================================================================
    size = traitlets.Int(0).tag(sync=True)  # Canvas rendering size in CSS pixels; 0 = frontend default

    # =========================================================================
    # Diff Mode
    # =========================================================================
    diff_mode = traitlets.Unicode("off").tag(sync=True)

    # =========================================================================
    # Analysis Panels (FFT + Histogram shown together)
    # =========================================================================
    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_window = traitlets.Bool(True).tag(sync=True)
    widget_version = traitlets.Unicode("unknown")  # Python-only: telemetry readout
    # =========================================================================
    # Line Profile
    # =========================================================================
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)
    profile_width = traitlets.Int(1).tag(sync=True)
    # Kymograph (space-time) side panel: when on with a drawn profile line, JS
    # samples that line on every frame from _offline_stack into a static
    # (n_slices, line_length) image - distance along the line on X, frame/time
    # on Y. Single-panel only; JS hides the toggle when n_panels > 1 or the
    # offline stack is absent (it needs every frame client-side to build it).
    show_kymograph = traitlets.Bool(False).tag(sync=True)

    # =========================================================================
    # Playback Buffer (sliding prefetch)
    # =========================================================================
    _buffer_bytes = traitlets.Bytes(b"").tag(sync=True)
    _buffer_start = traitlets.Int(0).tag(sync=True)
    _buffer_count = traitlets.Int(0).tag(sync=True)
    _prefetch_request = traitlets.Int(-1).tag(sync=True)
    frame_server_url = traitlets.Unicode("").tag(sync=True)
    frame_server_version = traitlets.Int(0).tag(sync=True)

    # Browser-side benchmark hook. Setting benchmark_request from Python starts
    # an in-widget measurement in the visible frontend; benchmark_result is
    # written back by JS when sampling finishes.
    benchmark_request = traitlets.Dict({}).tag(sync=True)
    benchmark_result = traitlets.Dict({}).tag(sync=True)

    # Render-time telemetry (set after first browser paint; docstring promises these).
    render_total_ms = traitlets.Int(allow_none=True, default_value=None)
    render_python_build_ms = traitlets.Int(allow_none=True, default_value=None)
    render_wire_js_ms = traitlets.Int(allow_none=True, default_value=None)

    _VALID_DIFF_MODES = {"off", "previous", "first"}

    @traitlets.validate("diff_mode")
    def _validate_diff_mode(self, proposal: dict) -> str:
        """Reject unknown diff modes; only off/previous/first are renderable."""
        val = proposal["value"]
        if val not in self._VALID_DIFF_MODES:
            raise traitlets.TraitError(
                f"Invalid diff_mode '{val}'. Must be one of: {sorted(self._VALID_DIFF_MODES)}"
            )
        return val

    @traitlets.validate("avg_window")
    def _validate_avg_window(self, proposal: dict) -> int:
        val = int(proposal["value"])
        if val < 1:
            raise traitlets.TraitError(f"avg_window must be >= 1, got {val}")
        if val > 15:
            raise traitlets.TraitError(f"avg_window must be <= 15, got {val}")
        return val

    @traitlets.validate("image_vmin_pct", "image_vmax_pct")
    def _validate_image_clip_pct(self, proposal: dict) -> float:
        val = float(proposal["value"])
        if not math.isfinite(val):
            raise traitlets.TraitError(f"{proposal['trait'].name} must be finite, got {val}")
        if val < 0 or val > 100:
            raise traitlets.TraitError(f"{proposal['trait'].name} must be in [0, 100], got {val}")
        return val

    @traitlets.validate("playback_path")
    def _validate_playback_path(self, proposal: dict) -> list:
        """Wrap indices into [0, n_slices) to keep JS in bounds."""
        # Wrap indices to [0, n_slices) so JS never indexes out of bounds.
        val = list(proposal["value"])
        n = max(1, int(self.n_slices))
        return [int(i) % n for i in val]

    @traitlets.validate("vmax")
    def _validate_vmax_ge_vmin(self, proposal: dict) -> float | None:
        """Reject non-finite vmax and enforce vmax >= vmin."""
        new_vmax = proposal["value"]
        if new_vmax is not None:
            if not math.isfinite(new_vmax):
                raise traitlets.TraitError(f"vmax must be finite, got {new_vmax}")
            if self.vmin is not None and new_vmax < self.vmin:
                raise traitlets.TraitError(
                    f"vmax ({new_vmax}) must be >= vmin ({self.vmin})"
                )
        return new_vmax

    @traitlets.validate("vmin")
    def _validate_vmin_le_vmax(self, proposal: dict) -> float | None:
        """Reject non-finite vmin and enforce vmin <= vmax."""
        new_vmin = proposal["value"]
        if new_vmin is not None:
            if not math.isfinite(new_vmin):
                raise traitlets.TraitError(f"vmin must be finite, got {new_vmin}")
            if self.vmax is not None and new_vmin > self.vmax:
                raise traitlets.TraitError(
                    f"vmin ({new_vmin}) must be <= vmax ({self.vmax})"
                )
        return new_vmin

    @traitlets.validate("cmap")
    def _validate_cmap(self, proposal: dict) -> str:
        """Reject unknown colormap names."""
        val = str(proposal["value"])
        if val not in _VALID_CMAPS:
            raise traitlets.TraitError(
                f"Unknown cmap {val!r}. Valid: {sorted(_VALID_CMAPS)}"
            )
        return val

    @traitlets.validate("bookmarked_frames")
    def _validate_bookmarks(self, proposal: dict) -> list:
        """Drop bookmark indices outside [0, n_slices) so markers stay onscreen."""
        # Drop indices outside [0, n_slices). JS draws bookmark markers and
        # raw out-of-range values caused offscreen / negative positions.
        n = max(1, int(self.n_slices))
        return [int(i) for i in proposal["value"] if 0 <= int(i) < n]

    @traitlets.validate("loop_end")
    def _validate_loop_end(self, proposal: dict) -> int:
        """Clamp to [0, n_slices) and enforce loop_end >= loop_start (-1 = last)."""
        # -1 sentinel = "last frame". Otherwise must be >= loop_start.
        val = int(proposal["value"])
        if val < 0:
            return val
        n = max(1, int(self.n_slices))
        val = min(val, n - 1)
        if val < int(self.loop_start):
            raise traitlets.TraitError(
                f"loop_end ({val}) must be >= loop_start ({self.loop_start})"
            )
        return val

    @traitlets.validate("loop_start")
    def _validate_loop_start(self, proposal: dict) -> int:
        """Clamp to [0, n_slices) and enforce loop_start <= loop_end."""
        val = int(proposal["value"])
        n = max(1, int(self.n_slices))
        val = max(0, min(val, n - 1))
        end = int(self.loop_end)
        if end >= 0 and val > end:
            raise traitlets.TraitError(
                f"loop_start ({val}) must be <= loop_end ({end})"
            )
        return val

    @traitlets.validate("pixel_size")
    def _validate_pixel_size(self, proposal: dict) -> float:
        """Reject non-finite or negative pixel sizes."""
        val = float(proposal["value"])
        if math.isnan(val) or math.isinf(val):
            raise traitlets.TraitError(f"pixel_size must be finite, got {val}")
        if val < 0:
            raise traitlets.TraitError(f"pixel_size must be >= 0, got {val}")
        return val

    @traitlets.validate("labels")
    def _validate_labels(self, proposal: dict) -> list:
        """Require labels length to match n_slices or be empty."""
        # Length must match n_slices for label lookups.
        val = list(proposal["value"])
        if val and len(val) != int(self.n_slices):
            raise traitlets.TraitError(
                f"labels length ({len(val)}) must equal n_slices ({self.n_slices}) or be empty"
            )
        return val

    @traitlets.validate("timestamps")
    def _validate_timestamps(self, proposal: dict) -> list:
        """Require timestamps length to match n_slices or be empty."""
        # Empty list = no timestamps. Otherwise length must match n_slices.
        val = list(proposal["value"])
        if val and len(val) != int(self.n_slices):
            raise traitlets.TraitError(
                f"timestamps length ({len(val)}) must equal n_slices ({self.n_slices}) or be empty"
            )
        return val

    @traitlets.validate("panel_titles")
    def _validate_panel_titles(self, proposal: dict) -> list:
        """Require panel_titles length to match n_panels or be empty."""
        # Empty list = default per-panel labels. Otherwise must match n_panels.
        val = list(proposal["value"])
        if val and len(val) != int(self.n_panels):
            raise traitlets.TraitError(
                f"panel_titles length ({len(val)}) must equal n_panels ({self.n_panels}) or be empty"
            )
        return val

    @traitlets.validate("starred")
    def _validate_starred(self, proposal: dict) -> list:
        """Require `starred` length to match n_panels. Each entry is a frame
        index in [-1, n_slices); -1 means no star on that panel."""
        val = list(proposal["value"])
        n_pan = int(self.n_panels)
        if not val:
            return [-1] * n_pan
        if len(val) != n_pan:
            raise traitlets.TraitError(
                f"starred length ({len(val)}) must equal n_panels ({n_pan})"
            )
        n_sl = int(self.n_slices)
        for i, v in enumerate(val):
            if v != -1 and not (0 <= v < n_sl):
                raise traitlets.TraitError(
                    f"starred[{i}] = {v} out of range [-1, {n_sl})"
                )
        return val

    def _validate_panel_list_length(self, name: str, value: list) -> list:
        """Require per-panel list traits to have exactly one entry per panel."""
        n_pan = int(self.n_panels)
        if len(value) != n_pan:
            raise traitlets.TraitError(
                f"{name} length ({len(value)}) must equal n_panels ({n_pan})"
            )
        return value

    @traitlets.validate("vmin_per_panel", "vmax_per_panel")
    def _validate_panel_bounds(self, proposal: dict) -> list:
        """Require per-panel vmin/vmax lists to match n_panels and be finite."""
        name = proposal["trait"].name
        val = self._validate_panel_list_length(name, list(proposal["value"]))
        for i, bound in enumerate(val):
            if bound is not None and not math.isfinite(float(bound)):
                raise traitlets.TraitError(f"{name}[{i}] must be finite or None, got {bound}")
        other_name = "vmax_per_panel" if name == "vmin_per_panel" else "vmin_per_panel"
        other = list(getattr(self, other_name))
        if len(other) == len(val):
            for i, bound in enumerate(val):
                other_bound = other[i]
                if bound is None or other_bound is None:
                    continue
                if name == "vmin_per_panel" and float(bound) > float(other_bound):
                    raise traitlets.TraitError(
                        f"vmin_per_panel[{i}] ({bound}) must be <= vmax_per_panel[{i}] ({other_bound})"
                    )
                if name == "vmax_per_panel" and float(bound) < float(other_bound):
                    raise traitlets.TraitError(
                        f"vmax_per_panel[{i}] ({bound}) must be >= vmin_per_panel[{i}] ({other_bound})"
                    )
        return val

    def _reset_panel_contrast_traits(self) -> None:
        """Initialize per-panel contrast traits from the shared contrast defaults."""
        n_pan = int(self.n_panels)
        self.vmin_per_panel = [None] * n_pan
        self.vmax_per_panel = [None] * n_pan

    @traitlets.validate("fps")
    def _validate_fps(self, proposal: dict) -> float:
        """Reject invalid fps and cap playback at the browser budget."""
        val = float(proposal["value"])
        if not math.isfinite(val):
            raise traitlets.TraitError(f"fps must be finite, got {val}")
        if val <= 0:
            raise traitlets.TraitError(f"fps must be > 0, got {val}")
        return min(val, _MAX_PLAYBACK_FPS)

    @traitlets.validate("slice_idx")
    def _validate_slice_idx(self, proposal: dict) -> int:
        """Clamp to [0, n_slices) so stale indices don't IndexError on reload."""
        # Clamp to [0, n_slices). State load with stale index used to crash
        # _update_all → _data[idx] with IndexError on a smaller new stack.
        val = int(proposal["value"])
        n = max(1, int(self.n_slices))
        return max(0, min(val, n - 1))

    @traitlets.validate("profile_width")
    def _validate_profile_width(self, proposal: dict) -> int:
        """Reject profile widths < 1."""
        val = int(proposal["value"])
        if val < 1:
            raise traitlets.TraitError(f"profile_width must be >= 1, got {val}")
        return val

    @traitlets.validate("percentile_low")
    def _validate_percentile_low(self, proposal: dict) -> float:
        """Require percentile_low in [0, 100] and strictly < percentile_high."""
        val = float(proposal["value"])
        if not 0 <= val <= 100:
            raise traitlets.TraitError(f"percentile_low must be in [0, 100], got {val}")
        if val >= float(self.percentile_high):
            raise traitlets.TraitError(
                f"percentile_low ({val}) must be < percentile_high ({self.percentile_high})"
            )
        return val

    @traitlets.validate("percentile_high")
    def _validate_percentile_high(self, proposal: dict) -> float:
        """Require percentile_high in [0, 100] and strictly > percentile_low."""
        val = float(proposal["value"])
        if not 0 <= val <= 100:
            raise traitlets.TraitError(f"percentile_high must be in [0, 100], got {val}")
        if val <= float(self.percentile_low):
            raise traitlets.TraitError(
                f"percentile_high ({val}) must be > percentile_low ({self.percentile_low})"
            )
        return val

    @traitlets.validate("roi_selected_idx")
    def _validate_roi_selected_idx(self, proposal: dict) -> int:
        """Clamp to [0, len(roi_list)) or -1 for nothing selected."""
        # -1 = nothing selected. Otherwise clamp to [0, len(roi_list))
        # so JS doesn't index OOB and Python stats don't silently return {}.
        val = int(proposal["value"])
        if val < 0:
            return -1
        return min(val, max(0, len(self.roi_list) - 1))

    _VALID_ROI_SHAPES = {"circle", "square", "rectangle", "annular"}

    @traitlets.validate("roi_list")
    def _validate_roi_list(self, proposal: dict) -> list:
        """Reject unknown ROI shapes and non-numeric or negative geometry fields."""
        # Reject unknown shapes (used to silently fall back to circle).
        # Clamp negative radii / dims to 1 so stats reflect what user sees.
        val = list(proposal["value"])
        for i, r in enumerate(val):
            shape = r.get("shape", "circle")
            if shape not in self._VALID_ROI_SHAPES:
                raise traitlets.TraitError(
                    f"ROI {i}: unknown shape {shape!r}. "
                    f"Valid: {sorted(self._VALID_ROI_SHAPES)}"
                )
            for k in ("radius", "radius_inner", "width", "height"):
                if k in r and r[k] is not None:
                    try:
                        v = float(r[k])
                    except (TypeError, ValueError):
                        raise traitlets.TraitError(
                            f"ROI {i}: {k}={r[k]!r} is not a number"
                        )
                    if v < 0:
                        raise traitlets.TraitError(
                            f"ROI {i}: {k} must be >= 0, got {r[k]}"
                        )
        return val

    # === Construction ===

    def __init__(
        self,
        *data_args,
        labels: list[str] | None = None,
        panel_titles: list[str] | None = None,
        panel_real_frames: list[int] | None = None,
        title: str = "",
        cmap: str | Colormap = Colormap.PLASMA,
        vmin: float | None = None,
        vmax: float | None = None,
        pixel_size: float = 0.0,
        pixel_unit: str | None = None,
        smooth: bool = True,
        image_rotation: int = 0,
        log_scale: bool = False,
        auto_contrast: bool = True,
        image_vmin_pct: float = 0.0,
        image_vmax_pct: float = 100.0,
        percentile_low: float = 0.5,
        percentile_high: float = 99.5,
        fps: float = 30.0,
        avg_window: int = 1,
        timestamps: list[float] | None = None,
        timestamp_unit: str = "s",
        show_fft: bool = False,
        fft_window: bool = True,
        show_stats: bool | None = None,
        show_controls: bool = True,
        size: int = 0,
        crop: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        padding: int | tuple[int, int] = 0,
        pad_mode: str = "median",
        config: "Mapping | str | pathlib.Path | None" = None,
        rotation_deg: float | None = None,
        post_crop: int | tuple[int, int] | tuple[int, int, int, int] | None = None,
        apply_config_transforms: bool = True,
        diff_mode: str = "off",
        buffer_size: int = 64,
        dim_label: str = "Frame",
        use_torch: bool | None = None,
        device: str | None = None,
        display_bin: int | str = "auto",
        hideable: bool = False,
        offline: bool | None = None,
        state=None,
        max_cols: int | None = None,
        panel_gap: int | None = None,
        panel_title_font_size: int | None = None,
        show_panel_titles: bool | None = None,
        show_resize_handles: bool | None = None,
        show_zoom_indicator: bool | None = None,
        show_scale_bar: bool | None = None,
        dedupe_identical_panels: bool = False,
        **kwargs,
    ):
        if hideable:
            kwargs["hideable"] = True
        if max_cols is not None:
            kwargs["max_cols"] = int(max_cols)
        if panel_gap is not None:
            kwargs["panel_gap"] = int(panel_gap)
        if panel_title_font_size is not None:
            kwargs["panel_title_font_size"] = int(panel_title_font_size)
        if show_panel_titles is not None:
            kwargs["show_panel_titles"] = bool(show_panel_titles)
        if show_resize_handles is not None:
            kwargs["show_resize_handles"] = bool(show_resize_handles)
        if show_zoom_indicator is not None:
            kwargs["show_zoom_indicator"] = bool(show_zoom_indicator)
        if show_scale_bar is not None:
            kwargs["scale_bar_visible"] = bool(show_scale_bar)
        kwargs.pop("show_playback", None)
        legacy_link_zoom = kwargs.pop("link_zoom", None)
        legacy_link_pan = kwargs.pop("link_pan", None)
        if "link_panels" not in kwargs:
            if legacy_link_zoom is not None and legacy_link_pan is not None:
                kwargs["link_panels"] = bool(legacy_link_zoom) and bool(legacy_link_pan)
            elif legacy_link_zoom is not None:
                kwargs["link_panels"] = bool(legacy_link_zoom)
            elif legacy_link_pan is not None:
                kwargs["link_panels"] = bool(legacy_link_pan)
        _t0 = time.perf_counter()
        # Reject unknown kwargs so typos raise instead of being silently ignored.
        _reject_unknown_kwargs(type(self), kwargs)
        super().__init__(**kwargs)
        # hold_sync() batches ALL traitlet assignments into a single comm message
        # sent when the context manager exits.  Without this, each self.x = y
        # fires a separate round-trip over the ZMQ/websocket channel, which
        # can add 30+ seconds for a large stack in VS Code Jupyter.
        if panel_real_frames is not None:
            self.panel_real_frames = list(panel_real_frames)
        with self.hold_sync():
            self._init_sync(data_args, labels=labels, panel_titles=panel_titles,
                            title=title, cmap=cmap, vmin=vmin, vmax=vmax,
                            pixel_size=pixel_size, pixel_unit=pixel_unit,
                            smooth=smooth, image_rotation=image_rotation,
                            log_scale=log_scale,
                            auto_contrast=auto_contrast,
                            image_vmin_pct=image_vmin_pct,
                            image_vmax_pct=image_vmax_pct,
                            percentile_low=percentile_low,
                            percentile_high=percentile_high, fps=fps, avg_window=avg_window,
                            timestamps=timestamps,
                            timestamp_unit=timestamp_unit, show_fft=show_fft,
                            fft_window=fft_window,
                            show_stats=show_stats, show_controls=show_controls,
                            size=size, crop=crop, padding=padding, pad_mode=pad_mode,
                            config=config, rotation_deg=rotation_deg,
                            post_crop=post_crop,
                            apply_config_transforms=apply_config_transforms,
                            diff_mode=diff_mode, buffer_size=buffer_size,
                            dim_label=dim_label, use_torch=use_torch, device=device,
                            display_bin=display_bin, offline=offline,
                            state=state, dedupe_identical_panels=dedupe_identical_panels,
                            _t0=_t0)

    def _init_sync(self, data_args: tuple, *, labels: list[str] | None,
                   panel_titles: list[str] | None, title: str,
                   cmap: str | Colormap, vmin: float | None, vmax: float | None,
                   pixel_size: float, pixel_unit: str | None, smooth: bool, image_rotation: int,
                   log_scale: bool, auto_contrast: bool,
                   image_vmin_pct: float, image_vmax_pct: float,
                   percentile_low: float, percentile_high: float,
                   fps: float, avg_window: int,
                   timestamps: list[float] | None,
                   timestamp_unit: str, show_fft: bool, fft_window: bool,
                   show_stats: bool | None, show_controls: bool,
                   size: int, crop: int | tuple[int, int] | tuple[int, int, int, int],
                   padding: int | tuple[int, int], pad_mode: str,
                   config, rotation_deg, post_crop, apply_config_transforms: bool,
                   diff_mode: str, buffer_size: int, dim_label: str,
                   use_torch: bool | None, device: str | None,
                   display_bin: int | str, offline: bool | None,
                   state: dict | str | pathlib.Path | None,
                   dedupe_identical_panels: bool, _t0: float) -> None:
        """Heavy setup called synchronously by `__init__` inside `hold_sync()`.
        Validates panels, allocates frame_bytes, wires observers, and applies
        optional `state`. Split out from `__init__` so the construction surface
        reads as a clean kwargs list while the heavy work has its own scope."""
        self.widget_version = resolve_widget_version()
        self._frame_server = None
        self._frame_server_thread = None
        self._frame_server_token = secrets.token_urlsafe(18)
        self._separate_panel_data = None
        self._crop = _normalize_crop(crop)
        self._padding = _normalize_padding(padding)
        if pad_mode not in ("median", "constant"):
            raise ValueError(f"pad_mode must be 'median' or 'constant', got {pad_mode!r}")
        self._pad_mode = pad_mode

        # ── QuantEM config convenience (mirrors Show3DSlices) ──
        # config supplies in-plane rotation + post-crop alignment and pixel size
        # so a ptycho z-stack calibrates with no manual array math. Explicit
        # rotation_deg / post_crop / pixel_size always win over config values.
        config_data = _load_quantem_config(config)
        post_crop_was_set = post_crop is not None
        if rotation_deg is None and config_data is not None and apply_config_transforms:
            rotation_deg = _config_float(config_data, "data", "rotation_deg") or 0.0
        config_rotation = _normalize_rotation_deg(rotation_deg) if rotation_deg is not None else 0.0

        def _apply_config_transform(arr):
            """Rotate then post-crop a (N, H, W) stack per config / explicit args."""
            if config_rotation:
                arr = _rotate_stack_inplane(arr, config_rotation)
            if post_crop_was_set:
                crop_spec = post_crop
            elif config_data is not None and apply_config_transforms:
                crop_spec = _post_crop_from_quantem_config(arr.shape, config_data)
            else:
                crop_spec = 0
            return _crop_stack(arr, _normalize_crop(crop_spec))

        # Optional torch acceleration. Do not move NumPy/Dataset input to GPU
        # merely because CUDA/MPS exists: real multi-panel ptycho stacks can be
        # many GB, and an implicit copy can OOM before the widget renders.
        self._use_torch = False
        self._device = None
        self._data_torch = None
        self._display_torch = None
        first_tensor_device = None
        if use_torch is None:
            use_torch = False
            if _HAS_TORCH:
                for d in data_args:
                    if isinstance(d, torch.Tensor):
                        use_torch = True
                        first_tensor_device = d.device
                        break
        if use_torch:
            if not _HAS_TORCH:
                raise ImportError(
                    "use_torch=True requires PyTorch. Install it with: pip install torch"
                )
            self._use_torch = True
            if device is not None:
                self._device = torch.device(device)
            elif first_tensor_device is not None:
                self._device = first_tensor_device
            else:
                self._device = torch.device(
                    "mps" if torch.backends.mps.is_available()
                    else "cuda" if torch.cuda.is_available()
                    else "cpu"
                )

        # ── Parse data args: single or multi-panel ──
        # Show3D(data) → single panel
        # Show3D(data1, data2, ...) → multi-panel (side-by-side, synced)
        if len(data_args) == 0:
            raise TypeError("Show3D requires at least one data argument")

        # Flatten: Show3D([data1, data2]) also works for multi-panel
        if (len(data_args) == 1 and isinstance(data_args[0], (list, tuple))
                and len(data_args[0]) > 0 and not isinstance(data_args[0][0], (int, float))):
            # Check if it's a list of arrays vs a single array
            first = data_args[0][0]
            if hasattr(first, 'ndim') or isinstance(first, np.ndarray):
                data_args = tuple(data_args[0])

        data = data_args[0]

        # Check if data is a Dataset3d and extract metadata
        _extracted_title = None
        _extracted_pixel_size = None
        _extracted_pixel_unit = None
        _extracted_dim_sampling = None
        _extracted_dim_unit = None
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            _extracted_title = data.name if data.name else None
            if hasattr(data, "sampling") and len(data.sampling) >= 3:
                sampling_val = float(data.sampling[1])
                _extracted_dim_sampling = float(data.sampling[0])
                if hasattr(data, "units"):
                    units = list(data.units)
                    if len(units) >= 1:
                        _extracted_dim_unit = str(units[0])
                    if len(units) >= 2:
                        _extracted_pixel_unit = str(units[1])
                _extracted_pixel_size = sampling_val
            data = data.array

        # Convert first panel to NumPy
        data = to_numpy(data)
        if data.ndim == 2:
            data = data[None, ...]
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array, got {data.ndim}D")
        if 0 in data.shape:
            raise ValueError(f"Empty stack: shape {data.shape}. All dims must be >= 1.")
        if np.iscomplexobj(data):
            raise TypeError(
                "Show3D does not accept complex data. Convert first: "
                "np.abs(arr) for magnitude or np.angle(arr) for phase."
            )
        if not _all_finite(data):
            raise ValueError(
                "Data contains NaN or inf. Clean first: "
                "np.nan_to_num(arr, nan=0, posinf=0, neginf=0)."
            )
        data = _pad_stack(_crop_stack(data, self._crop), self._padding, self._pad_mode)
        data = _apply_config_transform(data)

        # Multi-panel: convert remaining args, validate shapes, concatenate.
        # If the caller repeats the same stack object across panels, keep the
        # data once and let JS draw that exact frame into multiple panel slots.
        # This is the "36 full-res frames across 9 panels" stress case: no
        # pre-binning, no 9x Python copy, no 604 MB synthetic browser frame.
        if len(data_args) > 1:
            def _raw_array_obj(obj):
                return obj.array if hasattr(obj, "array") else obj

            first_source_obj = _raw_array_obj(data_args[0])
            shared_source = all(_raw_array_obj(extra) is first_source_obj for extra in data_args[1:])

            def _sample_is_finite(arr: np.ndarray) -> bool:
                return _all_finite(arr)

            def _as_valid_panel(arr: np.ndarray, panel_name: str) -> np.ndarray:
                if not _sample_is_finite(arr):
                    raise ValueError(
                        f"{panel_name} contains NaN or inf. Clean first: "
                        "np.nan_to_num(arr, nan=0, posinf=0, neginf=0)."
                    )
                with np.errstate(over="ignore", invalid="ignore"):
                    arr32 = arr.astype(np.float32, copy=False)
                if not _sample_is_finite(arr32):
                    raise ValueError(
                        f"{panel_name} exceeds float32 range (|value| > 3.4e38) "
                        "after cast; rescale first."
                    )
                return arr32

            self.n_panels = len(data_args)
            if panel_titles is not None:
                self.panel_titles = list(panel_titles)
            else:
                self.panel_titles = [f"Panel {i+1}" for i in range(self.n_panels)]
            self.starred = [-1] * self.n_panels

            if shared_source:
                _as_valid_panel(data, "Panel 0")
                self.shared_panel_source = True
                self.panel_width_px = int(data.shape[2])
                self._panel_width = self.panel_width_px
                self._multi_panel_bin = 0
                if not self.panel_real_frames:
                    self.panel_real_frames = [int(data.shape[0])] * self.n_panels
            else:
                self.shared_panel_source = False
                # copy=False avoids a redundant 120 MB+ allocation per panel when
                # the user already passed float32 (the common case for ptycho recons).
                panels = [_as_valid_panel(data, "Panel 0")]
                for i, extra in enumerate(data_args[1:], 1):
                    if hasattr(extra, "array"):
                        extra = extra.array
                    arr = to_numpy(extra)
                    if arr.ndim == 2:
                        arr = arr[None, ...]
                    if arr.ndim != 3:
                        raise ValueError(f"Panel {i}: expected 3D array, got {arr.ndim}D")
                    if 0 in arr.shape:
                        raise ValueError(f"Panel {i}: empty stack shape {arr.shape}. All dims must be >= 1.")
                    if np.iscomplexobj(arr):
                        raise TypeError(
                            f"Panel {i}: complex data not accepted. Convert first: "
                            "np.abs(arr) for magnitude or np.angle(arr) for phase."
                        )
                    arr = _pad_stack(_crop_stack(arr, self._crop), self._padding, self._pad_mode)
                    arr = _apply_config_transform(arr)
                    # Image (H,W) must match across panels - viewer cannot composite
                    # different image sizes into one canvas.
                    if arr.shape[1:] != panels[0].shape[1:]:
                        raise ValueError(
                            f"Panel {i} image shape {arr.shape[1:]} must match panel 0 image shape {panels[0].shape[1:]}."
                        )
                    # Slice counts can differ - caller compares trials with different
                    # iteration counts. We auto-pad shorter stacks below.
                    panels.append(_as_valid_panel(arr, f"Panel {i}"))
                self.n_panels = len(panels)
                # Auto-pad short stacks to longest, auto-fill panel_real_frames so
                # JS marks end-of-stack frames. Pad by repeating each panel's last
                # frame - visually obvious vs zeros and keeps colormap range stable.
                real_n = [p.shape[0] for p in panels]
                max_n = max(real_n)
                if any(n != max_n for n in real_n):
                    padded = []
                    for p, n in zip(panels, real_n):
                        if n == max_n:
                            padded.append(p)
                        else:
                            last = p[-1:]
                            pad = np.broadcast_to(last, (max_n - n, *p.shape[1:]))
                            padded.append(np.concatenate([p, pad], axis=0))
                    panels = padded
                    if not self.panel_real_frames:
                        self.panel_real_frames = real_n
                orig_h = panels[0].shape[1]
                orig_w = panels[0].shape[2]
                identical_panels = False
                if dedupe_identical_panels and len(panels) > 1:
                    identical_panels = all(
                        p.shape == panels[0].shape and np.array_equal(p, panels[0])
                        for p in panels[1:]
                    )

                if identical_panels:
                    data = panels[0]
                    self.shared_panel_source = True
                    self.panel_width_px = int(orig_w)
                    self._panel_width = self.panel_width_px
                    self._multi_panel_bin = 0
                    if not self.panel_real_frames:
                        self.panel_real_frames = real_n
                    print(
                        "  Exact duplicate panels deduped for display: "
                        f"{self.n_panels} panels share one {orig_h}x{orig_w} source frame"
                    )
                else:
                    # For large multi-panel stress tests, apply an explicit display_bin
                    # before concatenating panels. Binning after concat would first
                    # materialize a 4096 x (4096 * panels) slab; nine 4k panels is a
                    # 604 MB frame. The full source panels remain referenced here.
                    panel_bin = display_bin if isinstance(display_bin, int) and display_bin > 1 else 1
                    if panel_bin > 1:
                        from quantem.widget.array_utils import bin2d

                        self._source_panels = panels
                        panels = [
                            np.asarray(bin2d(p, factor=panel_bin, mode="mean"), dtype=np.float32)
                            for p in panels
                        ]
                        if pixel_size > 0:
                            pixel_size = pixel_size * panel_bin
                        display_bin = 1
                        print(
                            f"  Multi-panel display bin {panel_bin}x before concat: "
                            f"{orig_h}x{orig_w} -> {panels[0].shape[1]}x{panels[0].shape[2]} per panel"
                        )
                    if panel_bin == 1:
                        # Keep non-identical 4k panels separate. Concatenating
                        # nine panels makes each playback frame 4096 x 36864
                        # (~604 MB) and exceeds practical WebGPU storage-buffer
                        # binding limits. The frame server exposes exact
                        # float32 panel frames, and the frontend renders one
                        # GPU buffer per panel without binning or quantization.
                        self.separate_panel_frames = True
                        self._separate_panel_data = panels
                        data = panels[0]
                    else:
                        # Explicitly binned panels are small enough for the
                        # legacy concatenated-frame path.
                        data = np.concatenate(panels, axis=2)
                    self._panel_width = panels[0].shape[2]
                    self.panel_width_px = self._panel_width
                    self._multi_panel_bin = panel_bin
        else:
            self.n_panels = 1
            self.shared_panel_source = False
            self._multi_panel_bin = 0
            if panel_titles is not None:
                self.panel_titles = list(panel_titles)
            self.starred = [-1]

        # Reject complex input - silently dropping the imaginary part on
        # ptychography probes was a real data-loss footgun. User should
        # pass np.abs(probe) for magnitude or np.angle(probe) for phase.
        if np.iscomplexobj(data):
            raise TypeError(
                "Show3D does not accept complex data. Convert first: "
                "np.abs(arr) for magnitude or np.angle(arr) for phase."
            )
        # Store data as float32 numpy array. The pre-cast NaN/inf check runs on
        # the original dtype; values that fit in float64 but exceed float32 range
        # (~3.4e38) silently overflow to inf and contaminate stats / display.
        # Sample-check the cast output and reject early if so.
        with np.errstate(over="ignore", invalid="ignore"):
            self._data = data.astype(np.float32, copy=False)
        if not _all_finite(self._data):
            raise ValueError(
                "Data exceeds float32 range (|value| > 3.4e38) after cast; "
                "values silently overflowed to inf. Rescale first: "
                "arr = arr / np.max(np.abs(arr)) or use np.log1p(np.abs(arr))."
            )

        # Create GPU copy if torch acceleration enabled
        if self._use_torch:
            self._data_torch = torch.from_numpy(self._data).to(self._device)

        # Dimensions
        self.n_slices = int(self._data.shape[0])
        orig_h = int(self._data.shape[1])
        orig_w = int(self._data.shape[2])

        # NEVER BIN (CLAUDE.md rule). Display data is always source-pixel-exact.
        # Honor explicit display_bin=N>1 only if caller asks; "auto" stays 1.
        self._display_bin = 1
        if isinstance(display_bin, int) and display_bin > 1:
            self._display_bin = display_bin

        if self.separate_panel_frames:
            self._display_data = self._data
            self.height = orig_h
            self.width = int(self.panel_width_px) * int(self.n_panels)
        elif self._display_bin > 1:
            from quantem.widget.array_utils import bin2d
            self._display_data = bin2d(self._data, factor=self._display_bin, mode="mean")
            self.height = int(self._display_data.shape[1])
            self.width = int(self._display_data.shape[2])
            if pixel_size > 0:
                pixel_size = pixel_size * self._display_bin
            print(f"  Display bin {self._display_bin}× (explicit): {orig_h}×{orig_w} → {self.height}×{self.width}")
        else:
            self._display_data = self._data
            self.height = orig_h
            self.width = orig_w
        if self.shared_panel_source:
            self.panel_width_px = self.width

        # Color range (global across all frames)
        self._vmin_user = vmin
        self._vmax_user = vmax
        # Compute global min/max ONCE - each scan is bandwidth-bound (~150 ms
        # per pass on 1.34 GB float32), so eliminating the duplicate scans
        # saves ~300 ms on a 20x4k stack.
        if self.separate_panel_frames and self._separate_panel_data is not None:
            stack_min = min(float(p.min()) for p in self._separate_panel_data)
            stack_max = max(float(p.max()) for p in self._separate_panel_data)
        elif self._use_torch:
            stack_min = float(self._data_torch.min().item())
            stack_max = float(self._data_torch.max().item())
        else:
            stack_min = float(self._data.min())
            stack_max = float(self._data.max())
        self._vmin = vmin if vmin is not None else stack_min
        self._vmax = vmax if vmax is not None else stack_max
        self.data_min = stack_min
        self.data_max = stack_max
        # Cache the diff_mode='off' range so toggling Off→Previous→Off restores exact value.
        self._data_min_off = self.data_min
        self._data_max_off = self.data_max

        # Labels
        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = [str(i) for i in range(self.n_slices)]

        # Title and colormap - use extracted title if not explicitly provided
        self.title = title if title else (_extracted_title or "")
        self.cmap = str(cmap)  # Convert Colormap enum to string

        # Config sampling wins over a Dataset3d's default [1,1,1] sampling when
        # the user did not pass pixel_size explicitly: a ptycho config carries
        # the true Å/px (lateral) and slice thickness (depth) calibration.
        if config_data is not None and pixel_size == 0.0:
            config_pixel_size = _pixel_size_from_quantem_config(config_data)
            if config_pixel_size is not None:
                _extracted_pixel_size = config_pixel_size[1]
                _extracted_pixel_unit = "A"
                _extracted_dim_sampling = config_pixel_size[0]
                _extracted_dim_unit = _extracted_dim_unit or "A"
        # Use extracted pixel_size if not explicitly provided
        if pixel_size == 0.0 and _extracted_pixel_size is not None:
            pixel_size = _extracted_pixel_size
            if (
                (pixel_unit is None or pixel_unit in ("A", "Å", "angstrom", "Angstrom"))
                and _extracted_pixel_unit in ("nm", "nanometer")
            ):
                pixel_size *= 10
        if pixel_unit is None and _extracted_pixel_unit in ("nm", "nanometer"):
            pixel_unit = "A"
        elif pixel_unit is None:
            pixel_unit = _extracted_pixel_unit or "A"

        # Display options
        self.pixel_size = pixel_size
        self.pixel_unit = pixel_unit
        self.smooth = smooth
        # image_rotation: pure display-side (JS canvas transform), no data copy.
        self.image_rotation = image_rotation % 4
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.image_vmin_pct = image_vmin_pct
        self.image_vmax_pct = image_vmax_pct
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.vmin = vmin
        self.vmax = vmax
        self._reset_panel_contrast_traits()
        self.fps = fps
        self.avg_window = avg_window

        # Timestamps
        if timestamps is not None:
            self.timestamps = [float(t) for t in timestamps]
        elif _extracted_dim_sampling is not None:
            self.timestamps = [float(i * _extracted_dim_sampling) for i in range(self.n_slices)]
        else:
            self.timestamps = []
        self.timestamp_unit = _extracted_dim_unit or timestamp_unit
        self.dim_label = dim_label
        self.dim_sampling = _extracted_dim_sampling or 1.0
        self.dim_unit = _extracted_dim_unit or ""
        self.diff_mode = diff_mode
        self._refresh_auto_contrast_ranges()
        self.show_fft = show_fft
        self.fft_window = fft_window
        # Stats panel on by default for single-panel (cheap, useful readout);
        # off for multi-panel where N stat blocks would crowd the layout.
        # Explicit True/False from the caller always wins.
        self.show_stats = (int(self.n_panels) == 1) if show_stats is None else show_stats
        self.show_controls = show_controls
        self.size = size
        frame_bytes = self.height * self.width * 4  # float32
        # Exact float32 sliding window. Do not ship the whole stack when it
        # would cross the browser/Jupyter ~2 GB Comm cliff (36×4k×4k is 2.4 GB).
        max_frames = max(1, _MAX_PLAYBACK_CHUNK_BYTES // max(1, frame_bytes))
        self._buffer_size = min(buffer_size, self.n_slices, max_frames)

        # Initial position at middle
        self.slice_idx = int(self.n_slices // 2)
        self._roi_plot_timer = None

        # Offline mode stores the quantized uint8 stack for kernel-free scrub.
        # Exact standalone exports use _offline_float_stack on an export clone.
        # Default (offline=None): auto-enable uint8 when the pack fits a 1 GB
        # budget; opt out with offline=False for absurdly huge stacks.
        # Pick the right stack source for offline packing. separate_panel_frames
        # keeps each panel as its own array in self._separate_panel_data; the
        # offline path needs every panel, concatenated horizontally so JS
        # indexes by (sliceIdx * width * height) where `width` is the trait
        # value (= total concat width). Otherwise `_display_data` already holds
        # the concatenated or single-panel stack.
        if self.separate_panel_frames and self._separate_panel_data is not None:
            offline_source = np.concatenate(self._separate_panel_data, axis=2)
        else:
            offline_source = self._display_data
        stack_bytes = int(np.prod(offline_source.shape))  # uint8 = 1 B/px
        if offline is None:
            offline = stack_bytes <= 1 * 1024 * 1024 * 1024
        if offline:
            self.offline = True
            arr = np.ascontiguousarray(offline_source, dtype=np.float32)
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
            self._offline_stack = quantized.tobytes()

        # Observers
        self.observe(self._on_slice_change, names=["slice_idx"])
        self.observe(
            self._on_roi_change,
            names=["roi_active", "roi_list", "roi_selected_idx"],
        )
        self.observe(self._on_playing_change, names=["playing"])
        self.observe(self._on_prefetch, names=["_prefetch_request"])
        self.observe(self._on_diff_mode_change, names=["diff_mode"])
        self.observe(self._on_export_request_change, names=["export_request"])

        self._start_frame_server()

        # Initial update
        self._update_all()

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                    expected_widget="Show3D",
                )
            else:
                state = unwrap_state_payload(state, expected_widget="Show3D")
            self.load_state_dict(state)

        # Stash wall-clock start on the instance; observer below prints the
        # TRUE end-to-end time after JS signals first paint.  The Python-only
        # __init__ number is misleading for widget UX.
        self._init_t0 = _t0
        self._init_py_elapsed_ms = (time.perf_counter() - _t0) * 1000
        self.observe(self._on_first_render, names=["_js_rendered"])


    # === Public API ===

    def set_image(self, data, labels: list[str] | None = None) -> None:
        """Replace the stack data in place without rebuilding the widget.

        Swaps in a new stack while preserving display settings (cmap, contrast,
        log scale, pixel size, FFT toggle, playback config) so the operator can
        cycle through datasets in one widget cell. Resets state that is tied to
        the previous frame dimensions: ROIs and the line profile are cleared if
        ``(height, width)`` changes; bookmarks and the loop range are clamped
        to the new ``n_slices``; the playback prefetch buffer is invalidated;
        cached display data and any in-flight ROI debounce timer are dropped.

        Parameters
        ----------
        data : array_like
            3D stack ``(N, H, W)``, a 2D image (promoted to a single-frame
            stack), a torch tensor, or a quantem ``Dataset3d``. Complex data
            is rejected (cast to magnitude or phase first); non-finite data
            is rejected with a hint to ``np.nan_to_num``.
        labels : list[str] | None, optional
            New per-frame labels. If ``None``, string indices ``"0"..."N-1"``
            are used.

        Returns
        -------
        None
            Mutates the widget in place. The browser canvas re-renders
            automatically via traitlet sync.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack_a, title="A")  # doctest: +SKIP
        >>> w.set_image(stack_b, labels=[f"frame {i}" for i in range(len(stack_b))])  # doctest: +SKIP

        Notes
        -----
        - Prefer ``set_image`` over constructing a new ``Show3D`` when iterating
          through datasets in the same cell: it avoids re-creating the canvas
          and preserves the operator's contrast / zoom / cmap state.
        - The new stack is cast to ``float32``. If float64 input contains values
          outside ``float32`` range, an error is raised (silent overflow to
          ``inf`` would corrupt stats).
        """
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            data = data.array
        data = to_numpy(data)
        if data.ndim == 2:
            data = data[None, ...]
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array, got {data.ndim}D")
        if 0 in data.shape:
            raise ValueError(f"Empty stack: shape {data.shape}. All dims must be >= 1.")
        if not _all_finite(data):
            raise ValueError(
                "Data contains NaN or inf. Clean first: "
                "np.nan_to_num(arr, nan=0, posinf=0, neginf=0)."
            )
        data = _pad_stack(
            _crop_stack(data, getattr(self, "_crop", (0, 0, 0, 0))),
            getattr(self, "_padding", (0, 0)),
            getattr(self, "_pad_mode", "median"),
        )
        # Stop playback so JS doesn't keep painting from the stale _buffer_bytes
        # while we swap data. Invalidate cached torch view of display data.
        # Cancel any pending ROI plot timer so it doesn't fire mid-swap with
        # stale _display_data dims (race observed by audit).
        if getattr(self, "_roi_plot_timer", None) is not None:
            self._roi_plot_timer.cancel()
            self._roi_plot_timer = None
        self.playing = False
        self._display_torch = None
        # Clear ROIs / profile / bookmarks: their pixel coords are tied to the
        # previous (height, width) and may land out of bounds in the new stack.
        prev_h, prev_w = int(self.height), int(self.width)
        if np.iscomplexobj(data):
            raise TypeError(
                "Show3D does not accept complex data. Convert first: "
                "np.abs(arr) for magnitude or np.angle(arr) for phase."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            self._data = data.astype(np.float32, copy=False)
        # Pre-cast check ran on the source dtype; if float64 values exceed float32
        # range they silently become inf on cast and contaminate stats.
        if not _all_finite(self._data):
            raise ValueError(
                "Data exceeds float32 range (|value| > 3.4e38) after cast; "
                "values silently overflowed to inf. Rescale first: "
                "arr = arr / np.max(np.abs(arr)) or use np.log1p(np.abs(arr))."
            )
        if self._use_torch:
            self._data_torch = torch.from_numpy(self._data).to(self._device)
        self.n_panels = 1
        self.panel_titles = []
        self.starred = [-1]
        self._reset_panel_contrast_traits()
        self._multi_panel_bin = 0
        self._panel_width = int(data.shape[2])
        self.n_slices = int(data.shape[0])

        # Keep display dims = source dims. Constructor's auto-bin path was
        # silently down-sampling large frames + leaving pixel_size unscaled,
        # which gave wrong scale bars + ROI coords after set_image(). The
        # caller can re-instantiate with explicit display_bin if they need
        # the lower-memory rendering path.
        orig_h, orig_w = data.shape[1], data.shape[2]
        self._display_bin = 1
        self._display_data = self._data
        self.height = orig_h
        self.width = orig_w

        if self._use_torch:
            self.data_min = float(self._data_torch.min().item())
            self.data_max = float(self._data_torch.max().item())
        else:
            self.data_min = float(self._data.min())
            self.data_max = float(self._data.max())
        self._data_min_off = self.data_min
        self._data_max_off = self.data_max
        self._vmin = self._vmin_user if self._vmin_user is not None else self.data_min
        self._vmax = self._vmax_user if self._vmax_user is not None else self.data_max
        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = [str(i) for i in range(self.n_slices)]
        self.slice_idx = min(self.slice_idx, self.n_slices - 1)
        # Clear pixel-coord overlays if dims changed (stale ROIs / profile would
        # land out of bounds → empty stats or wrong-pixel sampling).
        if (self.height, self.width) != (prev_h, prev_w):
            self.roi_list = []
            self.roi_selected_idx = -1
            self.profile_line = []
            # profile_active is the JS-side toggle (no Python trait); skipping
        # Re-run bookmark validator with new n_slices (out-of-range markers
        # silently survived the swap otherwise).
        self.bookmarked_frames = list(self.bookmarked_frames)
        # Clamp loop range to new bounds.
        if self.loop_start >= self.n_slices:
            self.loop_start = 0
        if self.loop_end >= self.n_slices:
            self.loop_end = -1
        # Recompute buffer_size against new frame size, then invalidate JS-side
        # buffer (otherwise JS would slice the new H×W out of the old buffer).
        frame_bytes_n = self.height * self.width * 4
        max_frames = max(1, _MAX_PLAYBACK_CHUNK_BYTES // max(1, frame_bytes_n))
        self._buffer_size = min(self._buffer_size, self.n_slices, max_frames)
        self._buffer_bytes = b""
        self._bump_frame_server_version()
        self._refresh_auto_contrast_ranges()
        self._update_all()

    def __repr__(self) -> str:
        parts = f"Show3D({self.n_slices}×{self.height}×{self.width}, frame={self.slice_idx}, cmap={self.cmap}"
        if self.diff_mode != "off":
            parts += f", diff={self.diff_mode}"
        parts += ")"
        return parts

    def state_dict(self) -> dict:
        """Return a JSON-serializable snapshot of every user-tunable trait.

        Captures display config (cmap, contrast, log scale, FFT, diff mode),
        playback config (fps, loop, range, bookmarks, playback path), per-frame
        labels and timestamps, scale-bar settings, and any active ROIs and line
        profile. The raw stack data is NOT included; pair the snapshot with the
        original ``data`` argument (or re-attach via ``set_image``) on restore.

        Key order in the returned dict is deliberate: cross-validating pairs
        (``percentile_high`` before ``percentile_low``, ``loop_end`` before
        ``loop_start``) are emitted in the order the validators expect so a
        round-trip through ``load_state_dict`` cannot wedge in an intermediate
        state.

        Returns
        -------
        dict
            Mapping of trait name -> serializable value. Suitable for
            ``json.dump`` or use with ``save`` / ``load_state_dict``.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> state = w.state_dict()  # doctest: +SKIP
        >>> w2 = Show3D(stack)  # doctest: +SKIP
        >>> w2.load_state_dict(state)  # doctest: +SKIP

        Notes
        -----
        - Schema versioning is handled inside ``save``; the dict returned here
          is the unversioned inner payload.
        - To save directly to disk in one step, use ``save(path)`` instead.
        """
        return {
            # Widget discriminator: lets load_state_dict reject cross-widget
            # loads (e.g. Show3DSlices.load_state_dict(show3d.state_dict()))
            # cleanly instead of partially applying overlapping keys.
            "_widget": "Show3D",
            "title": self.title,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            # percentile_high before percentile_low so cross-validator doesn't reject mid-load.
            "percentile_high": self.percentile_high,
            "percentile_low": self.percentile_low,
            "image_vmin_pct": self.image_vmin_pct,
            "image_vmax_pct": self.image_vmax_pct,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "link_contrast": self.link_contrast,
            "link_panels": self.link_panels,
            "vmin_per_panel": list(self.vmin_per_panel),
            "vmax_per_panel": list(self.vmax_per_panel),
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "show_fft": self.show_fft,
            "show_kymograph": self.show_kymograph,
            "fft_window": self.fft_window,
            "pixel_size": self.pixel_size,
            "pixel_unit": self.pixel_unit,
            "smooth": self.smooth,
            "image_rotation": self.image_rotation,
            "scale_bar_visible": self.scale_bar_visible,
            "size": self.size,
            "fps": self.fps,
            "avg_window": self.avg_window,
            "loop": self.loop,
            "reverse": self.reverse,
            "boomerang": self.boomerang,
            # IMPORTANT: loop_end MUST precede loop_start in dict order.
            # Validators cross-check; loop_start with -1 sentinel skips check,
            # but loop_end=5 then loop_start=10 would otherwise raise during load.
            "loop_end": self.loop_end,
            "loop_start": self.loop_start,
            "bookmarked_frames": self.bookmarked_frames,
            "starred": list(self.starred),
            "playback_path": self.playback_path,
            "slice_idx": self.slice_idx,
            "roi_active": self.roi_active,
            "roi_list": self.roi_list,
            "roi_selected_idx": self.roi_selected_idx,
            "profile_line": self.profile_line,
            "profile_width": self.profile_width,
            "diff_mode": self.diff_mode,
            "dim_label": self.dim_label,
            "dim_sampling": self.dim_sampling,
            "dim_unit": self.dim_unit,
            "labels": list(self.labels),
            "panel_titles": list(self.panel_titles),
            "timestamps": list(self.timestamps),
            "timestamp_unit": self.timestamp_unit,
        }

    def save(self, path: str):
        """Write the current widget state to a versioned JSON file.

        Wraps ``state_dict`` in a small envelope that records the widget type
        (``"Show3D"``) and a schema version so ``load_state_dict`` can refuse
        states that belong to a different widget. The raw stack data is NOT
        written; only the display / playback / ROI / profile configuration.

        Parameters
        ----------
        path : str
            Destination JSON file path. Parent directories must already exist.

        Returns
        -------
        None

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack, cmap="viridis", pixel_size=0.5)  # doctest: +SKIP
        >>> w.save("show3d_state.json")  # doctest: +SKIP

        Notes
        -----
        - To restore, instantiate ``Show3D`` with the (possibly different) data
          and call ``w.load_state_dict(json.loads(open(path).read()))``.
        - Use ``state_dict`` directly if you want to embed widget state in a
          larger document instead of a standalone file.
        """
        save_state_file(path, "Show3D", self.state_dict())

    def export_html(
        self,
        path: str | pathlib.Path | None = None,
        *,
        quantized: bool = False,
        title: str | None = None,
    ) -> pathlib.Path:
        """Write a standalone HTML viewer for sharing.

        The exact export embeds the current float32 stack bytes and preserves
        numerical precision. The quantized export writes the existing offline
        uint8 representation plus global min/max metadata, making a smaller
        single-file report for visual sharing.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination HTML path. Defaults to the current working directory
            with the widget title, stack shape, and export mode in the name.
        quantized : bool, default False
            If True, write the uint8 offline pack. If False, write exact
            float32 bytes.
        title : str, optional
            Browser page title. Defaults to the widget title or class name.

        Returns
        -------
        pathlib.Path
            The written HTML file.
        """
        if self._data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")

        export_path = pathlib.Path(path) if path is not None else self._default_html_export_path(quantized)
        self._write_html_export(export_path, quantized=quantized, title=title)
        size_mb = export_path.stat().st_size / (1024 * 1024)
        mode = "quantized" if quantized else "exact float32"
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, {mode})"
        return export_path

    def load_state_dict(self, state: dict) -> None:
        """Apply a saved ``state_dict`` snapshot to this widget.

        Restores display, playback, ROI, and profile configuration from a dict
        produced by ``state_dict``. Unknown keys (typically from a newer widget
        version or a different widget type) are dropped with a ``UserWarning``
        instead of raising, so partial / forward-compatible loads succeed.
        Cross-validated trait pairs are applied atomically:

        - ``percentile_low`` / ``percentile_high`` are validated together and
          applied in whichever order keeps the invariant ``low < high``.
        - ``vmin`` / ``vmax`` are cleared first so either bound can be set
          regardless of the current contrast limits.
        - ``loop_start`` / ``loop_end`` are clamped to ``[0, n_slices)`` and
          applied in safe order via the ``-1`` sentinel.

        Parameters
        ----------
        state : dict
            Mapping previously returned by ``state_dict`` (or its on-disk
            equivalent). Old ``canvas_size`` aliases are migrated to ``size``;
            the constructor-derived ``display_bin`` key is ignored.

        Returns
        -------
        None
            Mutates the widget in place.

        Example
        -------
        >>> import json
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.load_state_dict(json.load(open("show3d_state.json")))  # doctest: +SKIP

        Notes
        -----
        - Unknown keys raise ``warnings.warn`` rather than an exception so
          forward-compatible files from a future widget still load.
        - The new contrast (``vmin`` / ``vmax``) is also stored on the private
          user-override slots so subsequent ``set_image`` calls keep the
          loaded contrast pinned.
        """
        state = dict(state)
        # Reject cross-widget loads up front. Without this check, loading a
        # Show3DSlices state_dict() into Show3D would partially apply the
        # overlapping keys (cmap, log_scale, etc.) and leave the widget in a
        # plausible-but-wrong state.
        marker = state.pop("_widget", None)
        if marker is not None and marker != "Show3D":
            raise ValueError(
                f"load_state_dict: state was saved from {marker!r}, not Show3D. "
                f"Use the matching widget class to load it."
            )
        # Derive allowed keys from state_dict() so the two stay in lockstep -
        # adding a trait to state_dict() automatically lets load_state_dict()
        # accept it. Matches the Show4DSTEM pattern.
        allowed = set(self.state_dict().keys())
        unknown = []
        if "canvas_size" in state:
            state["size"] = state.pop("canvas_size")
        # `display_bin` is constructor/data dependent. Loading only the private
        # integer leaves display_data/height/width stale, so ignore saved values.
        state.pop("display_bin", None)
        # Removed no-op trait from older saved states.
        state.pop("show_playback", None)
        # Drop length-coupled traits when stack size differs from saved state.
        # Otherwise the labels/timestamps validators raise, breaking the common
        # workflow of saving a state from one trial and loading into another.
        n_cur = int(self.n_slices)
        for key in ("labels", "timestamps"):
            if key in state and isinstance(state[key], list) and 0 < len(state[key]) != n_cur:
                state.pop(key)
        # Drop `starred` if its length doesn't match current n_panels (e.g.
        # saved from a 4-panel widget, loading into a single-panel one).
        if "starred" in state and isinstance(state["starred"], list) and len(state["starred"]) != int(self.n_panels):
            state.pop("starred")
        # These were briefly present in the development branch. Contrast auto,
        # percentile, and log scale now stay global to match Show2D.
        for key in (
            "auto_contrast_per_panel",
            "percentile_low_per_panel",
            "percentile_high_per_panel",
            "log_scale_per_panel",
        ):
            state.pop(key, None)
        panel_len_keys = ("vmin_per_panel", "vmax_per_panel")
        for key in panel_len_keys:
            if key in state and isinstance(state[key], list) and len(state[key]) != int(self.n_panels):
                state.pop(key)
        # Apply per-panel vmin/vmax as a pair, pre-cleared, so the cross-validator
        # (vmin_per_panel[i] <= vmax_per_panel[i]) never trips mid-load on the
        # generic setattr loop when the incoming vmin exceeds the CURRENT vmax.
        pp_vmin = state.pop("vmin_per_panel", None)
        pp_vmax = state.pop("vmax_per_panel", None)
        if pp_vmin is not None or pp_vmax is not None:
            n_pan = int(self.n_panels)
            self.vmin_per_panel = [None] * n_pan
            self.vmax_per_panel = [None] * n_pan
            if pp_vmax is not None:
                self.vmax_per_panel = list(pp_vmax)
            if pp_vmin is not None:
                self.vmin_per_panel = list(pp_vmin)
        if int(self.n_panels) > 1:
            for key in ("roi_active", "roi_list", "roi_selected_idx"):
                state.pop(key, None)
        for key in list(state):
            if key not in allowed:
                unknown.append(key)
                state.pop(key)

        pct_low_marker = object()
        pct_high_marker = object()
        pct_low = state.pop("percentile_low", pct_low_marker)
        pct_high = state.pop("percentile_high", pct_high_marker)
        if pct_low is not pct_low_marker or pct_high is not pct_high_marker:
            low = float(self.percentile_low if pct_low is pct_low_marker else pct_low)
            high = float(self.percentile_high if pct_high is pct_high_marker else pct_high)
            if not (0 <= low <= 100 and 0 <= high <= 100 and low < high):
                raise traitlets.TraitError(
                    f"percentile_low ({low}) must be < percentile_high ({high}) and both in [0, 100]"
                )
            if high <= float(self.percentile_low):
                self.percentile_low = low
                self.percentile_high = high
            else:
                self.percentile_high = high
                self.percentile_low = low

        vmin_marker = object()
        vmax_marker = object()
        vmin = state.pop("vmin", vmin_marker)
        vmax = state.pop("vmax", vmax_marker)
        if vmin is not vmin_marker or vmax is not vmax_marker:
            new_vmin = self.vmin if vmin is vmin_marker else vmin
            new_vmax = self.vmax if vmax is vmax_marker else vmax
            if new_vmin is not None and new_vmax is not None and float(new_vmin) > float(new_vmax):
                raise traitlets.TraitError(f"vmin ({new_vmin}) must be <= vmax ({new_vmax})")
            self.vmin = None
            self.vmax = None
            if new_vmin is not None:
                self.vmin = float(new_vmin)
            if new_vmax is not None:
                self.vmax = float(new_vmax)

        loop_start_marker = object()
        loop_end_marker = object()
        loop_start = state.pop("loop_start", loop_start_marker)
        loop_end = state.pop("loop_end", loop_end_marker)
        if loop_start is not loop_start_marker or loop_end is not loop_end_marker:
            n = max(1, int(self.n_slices))
            start = int(self.loop_start if loop_start is loop_start_marker else loop_start)
            end = int(self.loop_end if loop_end is loop_end_marker else loop_end)
            start = max(0, min(start, n - 1))
            if end >= 0:
                end = min(end, n - 1)
                if start > end:
                    raise traitlets.TraitError(f"loop_start ({start}) must be <= loop_end ({end})")
            else:
                end = -1
            self.loop_start = 0
            self.loop_end = end
            self.loop_start = start

        for key, val in state.items():
            setattr(self, key, val)
        self._vmin_user = self.vmin
        self._vmax_user = self.vmax
        self._vmin = self.vmin if self.vmin is not None else self.data_min
        self._vmax = self.vmax if self.vmax is not None else self.data_max
        if unknown:
            warnings.warn(
                f"load_state_dict ignored unknown keys: {unknown}. "
                "These may be from a newer widget version or a different widget type.",
                stacklevel=2,
            )

    def summary(self) -> None:
        """Print a one-screen status report for the current widget.

        Sections include: title and stack shape (with pixel-size readout when
        set), current frame index and label, raw data min/max/mean, display
        config (cmap, contrast, log/linear, FFT, diff mode), playback config
        (fps, loop, reverse, boomerang) and loop range, active ROI count, line
        profile endpoints if set, and render timing once the first browser
        paint has fired. Useful for notebook reproducibility and bug reports.

        Returns
        -------
        None
            Prints to stdout.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack, title="defocus")  # doctest: +SKIP
        >>> w.summary()  # doctest: +SKIP

        Notes
        -----
        - ``Rendered:`` reads ``(pending first browser paint)`` until the JS
          side has round-tripped its first paint timestamp.
        """
        lines = [self.title or "Show3D", "═" * 32]
        lines.append(f"Stack:    {self.n_slices}×{self.height}×{self.width}")
        if self.pixel_size > 0:
            ps = self.pixel_size
            if ps >= 10:
                lines[-1] += f" ({ps / 10:.2f} nm/px)"
            else:
                lines[-1] += f" ({ps:.2f} Å/px)"
        lines.append(f"Frame:    {self.slice_idx}/{self.n_slices - 1}")
        if self.labels and self.slice_idx < len(self.labels):
            lines[-1] += f" [{self.labels[self.slice_idx]}]"
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
        if self.diff_mode != "off":
            display += f" | diff={self.diff_mode}"
        lines.append(f"Display:  {display}")
        lines.append(f"Playback: {self.fps} fps | loop={'on' if self.loop else 'off'} | reverse={'on' if self.reverse else 'off'} | boomerang={'on' if self.boomerang else 'off'}")
        if self.loop_start > 0 or self.loop_end >= 0:
            end = self.loop_end if self.loop_end >= 0 else self.n_slices - 1
            lines.append(f"Range:    {self.loop_start}–{end}")
        if self.roi_active and self.roi_list:
            lines.append(f"ROI:      {len(self.roi_list)} region(s)")
        if len(self.profile_line) >= 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            lines.append(f"Profile:  ({p0['row']:.0f}, {p0['col']:.0f}) → ({p1['row']:.0f}, {p1['col']:.0f}) width={self.profile_width}")
        rt = getattr(self, "render_total_ms", None)
        if rt is not None:
            pb = getattr(self, "render_python_build_ms", 0)
            wj = getattr(self, "render_wire_js_ms", 0)
            lines.append(f"Rendered: {rt} ms total (Python build {pb} ms, wire+JS {wj} ms)")
        else:
            lines.append("Rendered: (pending first browser paint)")
        print("\n".join(lines))

    def play(self) -> Self:
        """Start playback from the current frame.

        Sets the ``playing`` trait to ``True``, which triggers the JS
        playback loop and the Python-side sliding-prefetch buffer that
        ships chunks of frames ahead of the scrubber position.

        Returns
        -------
        Self
            The widget, for chaining (``w.play().goto(0)``).

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack, fps=10)  # doctest: +SKIP
        >>> w.play()  # doctest: +SKIP
        """
        self.playing = True
        return self

    def pause(self) -> Self:
        """Pause playback at the current frame.

        Sets ``playing`` to ``False`` without resetting ``slice_idx``.
        Per-frame statistics are refreshed for the current frame on pause.

        Returns
        -------
        Self
            The widget, for chaining.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.play().pause()  # doctest: +SKIP
        """
        self.playing = False
        return self

    def stop(self) -> Self:
        """Stop playback and jump back to frame 0.

        Returns
        -------
        Self
            The widget, for chaining.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.play().stop()  # doctest: +SKIP

        Notes
        -----
        - Unlike ``pause``, ``stop`` resets ``slice_idx`` to 0. Use ``pause``
          to keep the current frame visible.
        """
        self.playing = False
        self.slice_idx = 0
        return self

    def goto(self, index: int) -> Self:
        """Jump to a specific frame index.

        The index is taken modulo ``n_slices``, so negative or out-of-range
        values wrap rather than raise.

        Parameters
        ----------
        index : int
            Target frame index. Wrapped into ``[0, n_slices)``.

        Returns
        -------
        Self
            The widget, for chaining.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.goto(5)  # doctest: +SKIP
        >>> w.goto(-1)  # last frame  # doctest: +SKIP
        """
        self.slice_idx = int(index) % self.n_slices
        return self

    def star_panel(self, panel: int = 0, frame: int | None = None) -> Self:
        """Mark a "best frame" star on a panel.

        Parameters
        ----------
        panel : int, default 0
            Panel index (0-based). For single-panel widgets, panel=0 is the
            only valid value.
        frame : int | None, default None
            Frame index to star. ``None`` stars the currently displayed frame
            (``slice_idx``).

        Returns
        -------
        Self
            The widget, for chaining.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack_a, stack_b)  # doctest: +SKIP
        >>> w.goto(50)  # doctest: +SKIP
        >>> w.star_panel(0)            # mark frame 50 as best on panel 0
        >>> w.star_panel(1, frame=80)  # mark frame 80 as best on panel 1
        >>> w.starred_frames           # → {0: 50, 1: 80}  # doctest: +SKIP
        """
        if not (0 <= panel < self.n_panels):
            raise ValueError(f"panel {panel} out of range [0, {self.n_panels})")
        if frame is None:
            frame = int(self.slice_idx)
        starred = list(self.starred)
        if len(starred) != self.n_panels:
            starred = [-1] * self.n_panels
        starred[panel] = int(frame)
        self.starred = starred
        return self

    def unstar_panel(self, panel: int) -> Self:
        """Clear the star on a panel (sets `starred[panel] = -1`)."""
        if not (0 <= panel < self.n_panels):
            raise ValueError(f"panel {panel} out of range [0, {self.n_panels})")
        starred = list(self.starred)
        if len(starred) != self.n_panels:
            starred = [-1] * self.n_panels
        starred[panel] = -1
        self.starred = starred
        return self

    @property
    def starred_frames(self) -> dict[int, int]:
        """Mapping of panel index → starred frame index, only for panels that
        have a star set. Returns ``{}`` if no panel is starred. Useful for
        downstream code like ``best_iters = {trial: w.starred_frames.get(i)}``."""
        return {i: f for i, f in enumerate(self.starred) if f >= 0}

    @property
    def visible_indices(self) -> list[int]:
        """Live list of frame indices NOT in hidden_indices. Read-only;
        mutate via set_hidden() / show_all() / hide()."""
        hidden = set(self.hidden_indices)
        return [i for i in range(self.n_slices) if i not in hidden]

    def hide(self, *indices: int) -> Self:
        """Mark one or more frames as hidden from the scrubber.

        Hidden frames are excluded from the scrubber UI and from playback but
        kept in memory; restore them with ``show`` or ``show_all``. Idempotent:
        hiding an already-hidden index is a no-op. At least one frame always
        stays visible, so a call that would hide every frame is silently
        rejected.

        Parameters
        ----------
        *indices : int
            One or more frame indices to hide. Values outside ``[0, n_slices)``
            are accepted but never become visible-or-hidden in the UI.

        Returns
        -------
        Show3D
            The widget, for chaining (``w.hide(0, 1).hide(5)``).

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.hide(0, 1, 2)  # doctest: +SKIP

        Notes
        -----
        - The ``hideable`` trait must be ``True`` for the JS scrubber to show
          the hide overlay; the underlying ``hidden_indices`` trait is always
          honored regardless.
        """
        keep = set(self.hidden_indices) | {int(i) for i in indices}
        # Always keep at least one frame visible.
        if len(keep) >= self.n_slices:
            return self
        self.hidden_indices = sorted(keep)
        return self

    def show(self, *indices: int) -> Self:
        """Restore one or more previously hidden frames.

        Idempotent: indices that are not currently hidden are silently ignored.

        Parameters
        ----------
        *indices : int
            Frame indices to make visible again.

        Returns
        -------
        Show3D
            The widget, for chaining.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.hide(2).show(2)  # doctest: +SKIP
        """
        drop = {int(i) for i in indices}
        self.hidden_indices = sorted(set(self.hidden_indices) - drop)
        return self

    def set_hidden(self, indices: list[int]) -> Self:
        """Replace the hidden set wholesale.

        Discards any current ``hidden_indices`` and installs ``indices``
        clamped to ``[0, n_slices)``. At least one frame is always visible;
        if the input would hide every frame, the largest index is dropped.

        Parameters
        ----------
        indices : list[int]
            Full replacement set of hidden frame indices. Order and duplicates
            are normalized internally.

        Returns
        -------
        Show3D
            The widget, for chaining.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.set_hidden([0, 1, 2, 7])  # doctest: +SKIP
        """
        clean = sorted({int(i) for i in indices if 0 <= int(i) < self.n_slices})
        # Always keep at least one frame visible.
        if len(clean) >= self.n_slices:
            clean = clean[:-1]
        self.hidden_indices = clean
        return self

    def show_all(self) -> Self:
        """Clear the hidden set so every frame is visible.

        Returns
        -------
        Show3D
            The widget, for chaining.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.hide(0, 1, 2).show_all()  # doctest: +SKIP
        """
        self.hidden_indices = []
        return self

    @property
    def roi(self) -> dict:
        """The selected ROI dict (or the first ROI if none is selected)."""
        idx = self.roi_selected_idx
        if 0 <= idx < len(self.roi_list):
            return self.roi_list[idx]
        if self.roi_list:
            return self.roi_list[0]
        return {}

    def add_roi(self, row: int | None = None, col: int | None = None, shape: str = "square") -> Self:
        with self.hold_sync():
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
        """Delete the currently selected ROI."""
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
        """Set selected ROI position and size (creates one if needed)."""
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "circle", "row": int(row), "col": int(col), "radius": int(radius)})
        return self

    def roi_circle(self, radius: int = 10) -> Self:
        """Set selected ROI shape to circle."""
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "circle", "radius": int(radius)})
        return self

    def roi_square(self, half_size: int = 10) -> Self:
        """Set selected ROI shape to square."""
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "square", "radius": int(half_size)})
        return self

    def roi_rectangle(self, width: int = 20, height: int = 10) -> Self:
        """Set selected ROI shape to rectangle."""
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "rectangle", "width": int(width), "height": int(height)})
        return self

    def roi_annular(self, inner: int = 5, outer: int = 10) -> Self:
        """Set selected ROI shape to annular (donut)."""
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "annular", "radius_inner": int(inner), "radius": int(outer)})
        return self

    @property
    def profile(self):
        """Get profile line endpoints as [(row0, col0), (row1, col1)] or []."""
        return [(p["row"], p["col"]) for p in self.profile_line]

    @property
    def profile_values(self):
        """Get intensity values along the profile line for the current frame."""
        if len(self.profile_line) < 2:
            return None
        p0, p1 = self.profile_line
        return self._sample_profile(p0["row"], p0["col"], p1["row"], p1["col"])

    @property
    def profile_distance(self):
        """Get total distance of the profile line in calibrated units (Å or px)."""
        if len(self.profile_line) < 2:
            return None
        p0, p1 = self.profile_line
        dc = p1["col"] - p0["col"]
        dr = p1["row"] - p0["row"]
        dist_px = (dc**2 + dr**2) ** 0.5
        if self.pixel_size > 0:
            return dist_px * self.pixel_size
        return dist_px

    def set_profile(self, start: tuple[float, float], end: tuple[float, float]) -> Self:
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
        return self

    def clear_profile(self) -> Self:
        """Clear the current line profile."""
        self.profile_line = []
        return self

    def profile_all_frames(self, start: tuple[float, float] | None = None, end: tuple[float, float] | None = None) -> np.ndarray:
        """Extract the line profile from every frame, returning (n_slices, n_points).

        Uses the current profile_line unless start/end are provided.
        Always samples raw data (ignores diff_mode).

        Parameters
        ----------
        start : tuple of (row, col), optional
            Start point. Overrides current profile_line.
        end : tuple of (row, col), optional
            End point. Overrides current profile_line.

        Returns
        -------
        np.ndarray
            Shape (n_slices, n_points) float32 array.
        """
        if start is not None and end is not None:
            row0, col0 = float(start[0]), float(start[1])
            row1, col1 = float(end[0]), float(end[1])
        elif len(self.profile_line) >= 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            row0, col0 = p0["row"], p0["col"]
            row1, col1 = p1["row"], p1["col"]
        else:
            raise ValueError(
                "No profile line set. Call set_profile() first or pass start/end."
            )
        rows = []
        for i in range(self.n_slices):
            rows.append(self._sample_profile_on(self._data[i], row0, col0, row1, col1))
        return np.stack(rows)

    def save_image(self, path: str | pathlib.Path, *, frame_idx: int | None = None,
                   format: str | None = None, dpi: int = 150) -> pathlib.Path:
        """Save a single frame as a PNG, PDF, or TIFF file.

        The saved image is colorized with the current ``cmap`` and contrast
        (``vmin`` / ``vmax`` or percentile auto-contrast), so the saved file
        matches what the browser shows for that frame. ``diff_mode`` is
        respected: ``"previous"`` saves ``frame - frame[idx-1]`` and ``"first"``
        saves ``frame - frame[0]``.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path. Parent directories are created if needed.
        frame_idx : int | None, optional
            Frame index to save. Defaults to the current ``slice_idx``.
        format : str | None, optional
            One of ``"png"``, ``"pdf"``, ``"tiff"``. If omitted, inferred
            from the file extension; defaults to ``"png"`` if no extension.
        dpi : int, default 150
            DPI metadata written into the file.

        Returns
        -------
        pathlib.Path
            The written file path.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack, cmap="viridis")  # doctest: +SKIP
        >>> w.save_image("frame_5.png", frame_idx=5)  # doctest: +SKIP
        >>> w.save_image("frame.pdf", dpi=300)  # doctest: +SKIP

        Notes
        -----
        - PDF output is converted to RGB internally (no alpha channel).
        - Frame indices outside ``[0, n_slices)`` raise ``IndexError``;
          unsupported extensions raise ``ValueError``.
        - For multi-frame output, call ``save_image`` in a loop.
        """
        from matplotlib import colormaps
        from PIL import Image

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf", "tiff", "tif"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'.")

        idx = frame_idx if frame_idx is not None else self.slice_idx
        if idx < 0 or idx >= self.n_slices:
            raise IndexError(f"Frame index {idx} out of range [0, {self.n_slices})")

        # Respect diff_mode so saved frame matches what user sees.
        frame = self._data[idx]
        if self.diff_mode == "previous":
            frame = frame - self._data[idx - 1] if idx > 0 else np.zeros_like(frame)
        elif self.diff_mode == "first":
            frame = frame - self._data[0]
        normalized = self._normalize_frame(frame)
        cmap_fn = colormaps.get_cmap(self.cmap)
        rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)

        img = Image.fromarray(rgba)
        if fmt == "pdf":
            Image.init()
            img = img.convert("RGB")
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), dpi=(dpi, dpi))
        return path

    def save_gif(self, path: str | pathlib.Path, *, quality: str = "high",
                 fps: float | None = None) -> pathlib.Path:
        """Save the z-stack as an animated GIF matching the live view.

        Each frame is colorized with the current ``cmap`` and contrast
        (``vmin`` / ``vmax`` or percentile auto-contrast, ``log_scale`` honored),
        carries a burnt-in scale bar when ``pixel_size`` is set, and plays at the
        widget's ``fps`` so the shared file looks like what the operator scrubs.

        Parameters
        ----------
        path : str or pathlib.Path
            Output ``.gif`` path. Parent directories are created.
        quality : {"high", "medium", "low"}, default "high"
            Spatial resolution tier (1.0 / 0.6 / 0.35). GIF is always a 256-color
            palette, so quality trades resolution for file size.
        fps : float, optional
            Playback rate. Defaults to the widget's ``fps``.

        Returns
        -------
        pathlib.Path
            The written GIF path.

        Notes
        -----
        Single-panel Show3D only; multi-panel raises. Browser zoom/pan is a
        view-only transform and is not reflected (the full frame is exported).
        """
        from quantem.widget import gif_utils
        if getattr(self, "n_panels", 1) > 1:
            raise ValueError("save_gif supports single-panel Show3D only.")
        if quality not in gif_utils.QUALITY_SCALE:
            raise ValueError(f"quality must be one of {list(gif_utils.QUALITY_SCALE)}, got {quality!r}.")
        fps = float(self.fps) if fps is None else float(fps)
        unit = self.pixel_unit or "A"
        rendered = []
        for idx in range(self.n_slices):
            frame = self._data[idx]
            if self.diff_mode == "previous":
                frame = frame - self._data[idx - 1] if idx > 0 else np.zeros_like(frame)
            elif self.diff_mode == "first":
                frame = frame - self._data[0]
            img = gif_utils.colorize(self._normalize_frame(frame), self.cmap)
            rendered.append(gif_utils.finalize_frame(img, quality, self.pixel_size, unit))
        # Boomerang from the middle: open on the centre slice, sweep up to the
        # top, down to the bottom, back to the centre - so the GIF starts and
        # ends on the same middle frame the live widget opens on.
        mid = self.n_slices // 2
        order = (list(range(mid, self.n_slices))
                 + list(range(self.n_slices - 2, -1, -1))
                 + list(range(1, mid + 1)))
        frames = [rendered[i] for i in order]
        return gif_utils.write_gif(frames, path, fps)

    def free(self) -> None:
        """Release VRAM and RAM held by this widget.

        Drops the numpy stack, the torch view (if any), the binned display
        copy, synced frame / ROI / prefetch bytes, and cancels any in-flight
        ROI debounce timer.
        When applicable, also flushes the CuPy memory pool and the FFT plan
        cache (in case a torch tensor was a view into CuPy memory), and
        releases the active torch device cache (``mps`` or ``cuda``).

        ``del widget`` alone does NOT free memory: traitlets installs strong
        observer references that pin the widget's refcount until ``free`` is
        called.

        Returns
        -------
        None
            Mutates the widget in place. After this call, frame data is gone
            and rendering will be blank; rebuild a new widget for further use.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.free()  # doctest: +SKIP

        Notes
        -----
        - Idempotent: calling ``free`` twice is a no-op.
        - After ``free``, ``set_image`` cannot restore the widget (the
          observers are still pinned but ``_data`` is ``None``); construct a
          new ``Show3D`` if you need to re-display something.
        """
        self._stop_frame_server()
        if self._data is None:
            return
        # Stop frontend playback FIRST so JS rAF loop tears down before we
        # null out the byte buffers it's reading from. Without this, free()
        # mid-play leaves the JS rAF rendering from a stale Float32Array
        # view into freed memory.
        self.playing = False
        # Cancel pending ROI debounce so its callback can't fire post-free.
        if self._roi_plot_timer is not None:
            self._roi_plot_timer.cancel()
            self._roi_plot_timer = None
        device = str(self._device) if self._device is not None else ""
        self._data = None
        self._data_torch = None
        self._display_data = None
        self._separate_panel_data = None
        # Offline/export stacks are synced Bytes traits; without clearing them
        # free() leaves that RAM pinned by traitlets after _data is dropped.
        for trait in (
            "frame_bytes",
            "roi_plot_data",
            "_buffer_bytes",
            "_offline_stack",
            "_offline_float_stack",
            "export_payload",
        ):
            setattr(self, trait, b"")
        gc.collect()
        # Flush cupy pool: _data may have been a torch view into cupy memory.
        if "cupy" in sys.modules:
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.fft.config.get_plan_cache().clear()
        if device == "mps":
            torch.mps.empty_cache()
        elif device.startswith("cuda"):
            torch.cuda.empty_cache()


    # === Observers ===

    def _on_first_render(self, change):
        """Observer for `_js_rendered=True`: prints true end-to-end construction
        timing telemetry (Python + comm + JS paint) and unobserves itself."""
        if not change.get("new"):
            return
        total_ms = (time.perf_counter() - self._init_t0) * 1000
        py_ms = self._init_py_elapsed_ms
        shape = f"{self.n_slices}×{self.height}×{self.width}"
        if self.separate_panel_frames and getattr(self, "_separate_panel_data", None) is not None:
            mem = sum(int(p.nbytes) for p in self._separate_panel_data)
        else:
            mem = self._data.nbytes
        mem_str = f"{mem / (1 << 20):.0f} MB" if mem >= 1 << 20 else f"{mem / (1 << 10):.0f} KB"
        self.render_total_ms = int(total_ms)
        self.render_python_build_ms = int(py_ms)
        self.render_wire_js_ms = int(total_ms - py_ms)
        print(
            f"Show3D: {shape} {mem_str} - "
            f"rendered in {total_ms:.0f} ms (Python build {py_ms:.0f} ms, "
            f"wire+JS {total_ms - py_ms:.0f} ms)",
            flush=True,
        )
        try:
            self.unobserve(self._on_first_render, names=["_js_rendered"])
        except (ValueError, KeyError):
            pass  # observer already removed

    def _on_diff_mode_change(self, change: dict | None = None) -> None:
        """Observer: `diff_mode` flipped. Recomputes the symmetric-around-zero
        data range so the colormap pins black at 0 (positive vs negative diff
        signal stays balanced)."""
        data = self._display_data
        if self.diff_mode == "off":
            # Restore the constructor's full-resolution data range so toggling
            # Off→Previous→Off is idempotent (computing from binned data drifts).
            self.data_min = float(getattr(self, "_data_min_off", data.min()))
            self.data_max = float(getattr(self, "_data_max_off", data.max()))
        elif self.diff_mode == "previous":
            # Vectorized diff: data[1:] - data[:-1]
            # Symmetric clamp around 0 so the all-zero baseline frame at idx=0
            # stays inside the displayed range whether diffs are positive or negative.
            if self.n_slices < 2:
                self.data_min = 0.0
                self.data_max = 0.0
            else:
                diffs = data[1:] - data[:-1]
                self.data_min = min(0.0, float(diffs.min()))
                self.data_max = max(0.0, float(diffs.max()))
        elif self.diff_mode == "first":
            if self.n_slices < 2:
                self.data_min = 0.0
                self.data_max = 0.0
            else:
                diffs = data[1:] - data[0:1]
                self.data_min = min(0.0, float(diffs.min()))
                self.data_max = max(0.0, float(diffs.max()))
        else:
            self.data_min = float(data.min())
            self.data_max = float(data.max())
        self._bump_frame_server_version()
        self._refresh_auto_contrast_ranges()
        self._update_all()

    def _on_slice_change(self, change: dict | None = None) -> None:
        """Observer: `slice_idx` changed via scrub. Skipped during playback
        (playback drives frames from the JS-side prefetch buffer). Stats are
        computed JS-side from frame_bytes; Python only ships the raw bytes."""
        if self.playing:
            return
        self._update_all()

    def _on_playing_change(self, change: dict | None = None) -> None:
        """Observer: `playing` toggled. Starts the JS animation via a sliding
        buffer chunk, or refreshes stats on the held frame on stop."""
        if self.playing:
            if self.frame_server_url:
                with self.hold_sync():
                    self._buffer_start = int(self.slice_idx)
                    self._buffer_count = 0
                    self._buffer_bytes = b""
            else:
                self._send_buffer(self.slice_idx)
        else:
            self._update_all()

    def _on_prefetch(self, change: dict | None = None) -> None:
        """Observer: JS requested the next playback chunk. Re-sends the sliding
        window so playback never starves."""
        if self.frame_server_url:
            return
        if self._prefetch_request >= 0 and self.playing:
            self._send_buffer(self._prefetch_request % self.n_slices)

    def _on_roi_change(self, change: dict | None = None) -> None:
        """Handle ROI change. Stats for current frame are instant.
        Full-stack ROI plot is debounced (500ms) to avoid UI freeze during drag."""
        if int(self.n_panels) > 1:
            self.roi_stats = {}
            self.roi_plot_data = b""
            return
        # Auto-select first ROI if the user added one programmatically and
        # roi_selected_idx is still -1 (otherwise stats stay empty silently).
        if self.roi_active and self.roi_list and self.roi_selected_idx < 0:
            self.roi_selected_idx = 0
        if self.roi_active:
            self._update_roi_stats(self._get_display_frame())
            # Debounce the expensive all-frame ROI plot
            if self._roi_plot_timer is not None:
                self._roi_plot_timer.cancel()
            self._roi_plot_timer = threading.Timer(0.5, self._compute_roi_plot)
            self._roi_plot_timer.start()
        else:
            self.roi_stats = {}
            self.roi_plot_data = b""

    def _on_export_request_change(self, change: dict) -> None:
        """Handle toolbar export requests from the live notebook frontend."""
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
            mode = str(payload.get("mode", "exact"))
            if mode == "clear":
                self.export_payload = b""
                self.export_payload_id = ""
                self.export_filename = ""
                return
            if mode not in ("exact", "quantized"):
                raise ValueError(f"unknown export mode {mode!r}")
            quantized = mode == "quantized"
            if payload.get("download"):
                filename = str(payload.get("filename") or self._default_html_export_path(quantized).name)
                request_id = str(payload.get("id") or "")
                self.export_status = f"Preparing {filename}..."
                html = self._html_export_bytes(quantized=quantized)
                self.export_filename = filename
                self.export_payload = html
                self.export_payload_id = request_id
                size_mb = len(html) / (1024 * 1024)
                label = "quantized" if quantized else "exact float32"
                self.export_status = f"Ready {filename} ({size_mb:.1f} MB, {label})"
            else:
                self.export_status = f"Exporting {mode} HTML..."
                self.export_html(quantized=quantized)
        except Exception as exc:
            self.export_status = f"Export failed: {exc}"

    # === Internal primitives ===

    def _default_html_export_path(self, quantized: bool) -> pathlib.Path:
        """Build a stable, human-readable export filename in the kernel cwd."""
        label = self.title.strip() or "show3d"
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        if not slug:
            slug = "show3d"
        mode = "quantized" if quantized else "exact"
        return pathlib.Path.cwd() / f"{slug}_{self.n_slices}x{self.height}x{self.width}_{mode}.html"

    def _write_html_export(
        self,
        path: str | pathlib.Path,
        *,
        quantized: bool,
        title: str | None = None,
    ) -> pathlib.Path:
        """Write a standalone HTML export without updating toolbar status."""
        from ipywidgets.embed import dependency_state, embed_minimal_html

        export_path = pathlib.Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        page_title = title or self.title or "Show3D"
        export_widget = self._clone_for_html_export(quantized=quantized)
        try:
            state = dependency_state([export_widget], drop_defaults=False)
            embed_minimal_html(
                str(export_path),
                views=[export_widget],
                title=page_title,
                drop_defaults=False,
                state=state,
            )
        finally:
            export_widget.free()
        return export_path

    def _html_export_bytes(self, *, quantized: bool) -> bytes:
        """Build a standalone HTML export in a temp directory and return bytes."""
        with tempfile.TemporaryDirectory(prefix="show3d-export-") as tmp:
            path = pathlib.Path(tmp) / self._default_html_export_path(quantized).name
            self._write_html_export(path, quantized=quantized)
            return path.read_bytes()

    def _export_data_args(self) -> tuple[np.ndarray, ...]:
        """Return display-shaped data args so exported HTML matches the widget."""
        if self._display_data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        n_panels = int(self.n_panels)
        if self.shared_panel_source and n_panels > 1:
            src = np.ascontiguousarray(self._display_data, dtype=np.float32)
            return tuple(src for _ in range(n_panels))
        if self._separate_panel_data is not None:
            return tuple(
                np.ascontiguousarray(panel, dtype=np.float32)
                for panel in self._separate_panel_data
            )
        if n_panels > 1 and int(self.panel_width_px) > 0:
            panel_w = int(self.panel_width_px)
            if int(self.width) == panel_w * n_panels:
                return tuple(
                    np.ascontiguousarray(
                        self._display_data[:, :, i * panel_w : (i + 1) * panel_w],
                        dtype=np.float32,
                    )
                    for i in range(n_panels)
                )
        return (np.ascontiguousarray(self._display_data, dtype=np.float32),)

    def _offline_stack_source(self) -> np.ndarray:
        """Return the stack shape that the offline frontend indexes per frame."""
        if self._display_data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        if self.separate_panel_frames and self._separate_panel_data is not None:
            return np.concatenate(self._separate_panel_data, axis=2)
        return np.ascontiguousarray(self._display_data, dtype=np.float32)

    def _pack_exact_offline_stack(self) -> None:
        """Embed the full display stack as float32 for exact standalone HTML."""
        arr = np.ascontiguousarray(self._offline_stack_source(), dtype=np.float32)
        if arr.size:
            lo = float(arr.min())
            hi = float(arr.max())
        else:
            lo, hi = 0.0, 1.0
        self.offline = True
        self._offline_min = lo
        self._offline_max = hi
        self._offline_stack = b""
        self._offline_float_stack = arr.tobytes()
        self.frame_bytes = b""
        self._buffer_bytes = b""

    def _clone_for_html_export(self, *, quantized: bool) -> Self:
        """Create an export-only widget with current state and requested packing."""
        clone = type(self)(
            *self._export_data_args(),
            labels=list(self.labels) if self.labels else None,
            panel_titles=list(self.panel_titles) if self.panel_titles else None,
            panel_real_frames=list(self.panel_real_frames) if self.panel_real_frames else None,
            title=self.title,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            pixel_size=self.pixel_size,
            pixel_unit=self.pixel_unit,
            smooth=self.smooth,
            image_rotation=self.image_rotation,
            log_scale=self.log_scale,
            auto_contrast=self.auto_contrast,
            image_vmin_pct=self.image_vmin_pct,
            image_vmax_pct=self.image_vmax_pct,
            percentile_low=self.percentile_low,
            percentile_high=self.percentile_high,
            fps=self.fps,
            avg_window=self.avg_window,
            timestamps=list(self.timestamps) if self.timestamps else None,
            timestamp_unit=self.timestamp_unit,
            show_fft=self.show_fft,
            fft_window=self.fft_window,
            show_stats=self.show_stats,
            show_controls=self.show_controls,
            size=self.size,
            diff_mode=self.diff_mode,
            buffer_size=getattr(self, "_buffer_size", 64),
            dim_label=self.dim_label,
            use_torch=False,
            display_bin=1,
            offline=quantized,
            max_cols=self.max_cols,
            panel_gap=self.panel_gap,
            panel_title_font_size=self.panel_title_font_size,
            show_panel_titles=self.show_panel_titles,
            show_resize_handles=self.show_resize_handles,
            show_zoom_indicator=self.show_zoom_indicator,
            show_scale_bar=self.scale_bar_visible,
        )
        clone.load_state_dict(self.state_dict())
        if quantized:
            clone._offline_float_stack = b""
        else:
            clone._pack_exact_offline_stack()
        clone.playing = False
        clone.export_enabled = False
        clone.export_status = ""
        clone.export_payload = b""
        clone.export_payload_id = ""
        clone.export_filename = ""
        clone._stop_frame_server()
        clone.frame_server_url = ""
        clone._buffer_bytes = b""
        clone._export_light = True
        return clone

    def _start_frame_server(self) -> None:
        """Start the localhost exact-frame endpoint used by browser playback."""
        if getattr(self, "_frame_server", None) is not None:
            return

        widget_ref = weakref.ref(self)
        token = self._frame_server_token

        class FrameHandler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args) -> None:  # noqa: D401
                return

            def _cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
                self.send_header("Cache-Control", "no-store")

            def _text(self, status: int, message: str) -> None:
                body = message.encode("utf-8")
                self.send_response(status)
                self._cors()
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def do_OPTIONS(self) -> None:
                self.send_response(204)
                self._cors()
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:
                parsed = urllib.parse.urlsplit(self.path)
                if parsed.path != "/frame":
                    self._text(404, "not found")
                    return
                params = urllib.parse.parse_qs(parsed.query)
                if params.get("token", [""])[0] != token:
                    self._text(403, "forbidden")
                    return
                try:
                    idx = int(params.get("idx", [""])[0])
                except ValueError:
                    self._text(400, "idx must be an integer")
                    return
                panel_param = params.get("panel", [None])[0]
                try:
                    panel = int(panel_param) if panel_param is not None else None
                except ValueError:
                    self._text(400, "panel must be an integer")
                    return
                version_param = params.get("version", [None])[0]
                try:
                    version = int(version_param) if version_param is not None else None
                except ValueError:
                    self._text(400, "version must be an integer")
                    return

                widget = widget_ref()
                if widget is None:
                    self._text(410, "widget is gone")
                    return
                status, frame_or_message = widget._frame_for_http(idx, version, panel)
                if status != 200:
                    self._text(status, str(frame_or_message))
                    return

                frame = frame_or_message
                view = memoryview(frame).cast("B")
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(view)))
                self.send_header("X-Frame-Index", str(idx))
                self.send_header("X-Frame-Shape", f"{frame.shape[0]},{frame.shape[1]}")
                self.end_headers()
                try:
                    self.wfile.write(view)
                except (BrokenPipeError, ConnectionResetError):
                    return

        try:
            server = _Show3DFrameHTTPServer(("127.0.0.1", 0), FrameHandler)
        except OSError as exc:
            warnings.warn(f"Show3D frame server disabled: {exc}", RuntimeWarning, stacklevel=2)
            self.frame_server_url = ""
            return

        thread = threading.Thread(
            target=server.serve_forever,
            name=f"Show3DFrameServer-{id(self):x}",
            daemon=True,
        )
        thread.start()
        self._frame_server = server
        self._frame_server_thread = thread
        host, port = server.server_address[:2]
        quoted_token = urllib.parse.quote(self._frame_server_token, safe="")
        self.frame_server_url = f"http://{host}:{port}/frame?token={quoted_token}"
        self._bump_frame_server_version()

    def _stop_frame_server(self) -> None:
        """Stop the localhost frame endpoint."""
        server = getattr(self, "_frame_server", None)
        thread = getattr(self, "_frame_server_thread", None)
        self._frame_server = None
        self._frame_server_thread = None
        if self.frame_server_url:
            self.frame_server_url = ""
            self._bump_frame_server_version()
        if server is None:
            return
        try:
            server.shutdown()
        finally:
            server.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _bump_frame_server_version(self) -> None:
        self.frame_server_version = int(self.frame_server_version) + 1

    def _frame_for_http(self, idx: int, version: int | None, panel: int | None = None) -> tuple[int, np.ndarray | str]:
        if version is not None and version != self.frame_server_version:
            return 409, "stale frame server version"
        data = self._display_data
        if data is None:
            return 410, "frame data has been released"
        if idx < 0 or idx >= int(self.n_slices):
            return 416, f"frame index {idx} out of range [0, {self.n_slices})"
        if panel is not None:
            if not self.separate_panel_frames:
                return 400, "panel query is only valid for separate-panel frames"
            if panel < 0 or panel >= int(self.n_panels):
                return 416, f"panel index {panel} out of range [0, {self.n_panels})"
            frame = np.asarray(self._get_display_panel_frame(panel, idx), dtype=np.float32)
        else:
            frame = np.asarray(self._get_display_frame(idx), dtype=np.float32)
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        return 200, frame

    def _get_display_panel_frame(self, panel: int, idx: int) -> np.ndarray:
        panels = getattr(self, "_separate_panel_data", None)
        if not self.separate_panel_frames or panels is None:
            frame = self._get_display_frame(idx)
            pw = int(self.panel_width_px) or frame.shape[1] // max(1, int(self.n_panels))
            return frame[:, panel * pw : (panel + 1) * pw]
        frame = panels[panel][idx]
        if self.diff_mode == "previous":
            if idx == 0:
                return np.zeros_like(frame)
            return frame - panels[panel][idx - 1]
        if self.diff_mode == "first":
            return frame - panels[panel][0]
        return frame

    def _get_display_frame(self, idx: int | None = None) -> np.ndarray:
        """Return the (binned, possibly diff-mode) frame for display. idx=None uses
        current slice_idx. Applies `diff_mode` subtraction so the FFT / stats
        observers see the same data the browser renders."""
        if idx is None:
            idx = self.slice_idx
        if self.separate_panel_frames and getattr(self, "_separate_panel_data", None) is not None:
            return np.concatenate(
                [self._get_display_panel_frame(panel, idx) for panel in range(int(self.n_panels))],
                axis=1,
            )
        data = self._display_data
        frame = data[idx]
        if self.diff_mode == "previous":
            if idx == 0:
                return np.zeros_like(frame)
            return frame - data[idx - 1]
        if self.diff_mode == "first":
            return frame - data[0]
        return frame

    def _refresh_auto_contrast_ranges(self) -> None:
        """Precompute one stack-level auto-contrast range for JS playback.

        Show3D is a scrubber, so Auto should give a stable intensity mapping
        across frames and panels. The synced lists still have one entry per
        slice for the existing JS cache contract, but every entry carries the
        same stack percentile range.
        """
        if self.n_slices <= 0:
            self.auto_vmins = []
            self.auto_vmaxs = []
            return
        if not self.auto_contrast:
            self.auto_vmins = []
            self.auto_vmaxs = []
            return

        low_target = self.percentile_low / 100.0
        high_target = self.percentile_high / 100.0
        bins = 1024
        denom = bins - 1
        separate_panels = self.separate_panel_frames and getattr(self, "_separate_panel_data", None) is not None
        mn = float("inf")
        mx = float("-inf")
        total_size = 0
        for i in range(self.n_slices):
            if separate_panels:
                panel_frames = [self._get_display_panel_frame(panel, i) for panel in range(int(self.n_panels))]
                mn = min(mn, min(float(np.min(frame)) for frame in panel_frames))
                mx = max(mx, max(float(np.max(frame)) for frame in panel_frames))
                total_size += sum(int(frame.size) for frame in panel_frames)
            else:
                frame = self._get_display_frame(i)
                mn = min(mn, float(np.min(frame)))
                mx = max(mx, float(np.max(frame)))
                total_size += int(frame.size)
        if total_size <= 0 or not math.isfinite(mn) or not math.isfinite(mx):
            self.auto_vmins = []
            self.auto_vmaxs = []
            return
        if mn == mx:
            self.auto_vmins = [mn] * int(self.n_slices)
            self.auto_vmaxs = [mx] * int(self.n_slices)
            return

        hist = np.zeros(bins, dtype=np.int64)
        for i in range(self.n_slices):
            if separate_panels:
                for panel in range(int(self.n_panels)):
                    frame = self._get_display_panel_frame(panel, i)
                    panel_hist, _ = np.histogram(frame, bins=bins, range=(mn, mx))
                    hist += panel_hist
            else:
                frame = self._get_display_frame(i)
                frame_hist, _ = np.histogram(frame, bins=bins, range=(mn, mx))
                hist += frame_hist

        csum = np.cumsum(hist)
        low_count = int(total_size * low_target)
        high_count = int(np.ceil(total_size * high_target))
        lo = int(np.searchsorted(csum, low_count, side="left"))
        hi = int(np.searchsorted(csum, high_count, side="left"))
        lo = max(0, min(denom, lo))
        hi = max(0, min(denom, hi))
        span = mx - mn
        vmin = float(mn + (lo / denom) * span)
        vmax = float(mn + (hi / denom) * span)
        vmins = [vmin] * int(self.n_slices)
        vmaxs = [vmax] * int(self.n_slices)

        with self.hold_sync():
            self.auto_vmins = vmins
            self.auto_vmaxs = vmaxs

    def _get_color_range(self, frame: np.ndarray) -> tuple[float, float]:
        """Get vmin/vmax based on current settings."""
        if self.vmin is not None or self.vmax is not None:
            vmin = float(self.vmin if self.vmin is not None else self._vmin)
            vmax = float(self.vmax if self.vmax is not None else self._vmax)
            if self.log_scale:
                # Signed log so negative vmin (e.g. diff_mode) doesn't collapse to 0.
                vmin = float(np.sign(vmin) * np.log1p(abs(vmin)))
                vmax = float(np.sign(vmax) * np.log1p(abs(vmax)))
        elif self.auto_contrast:
            vmin = float(np.percentile(frame, self.percentile_low))
            vmax = float(np.percentile(frame, self.percentile_high))
        else:
            vmin = self._vmin
            vmax = self._vmax
        return vmin, vmax

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame to uint8 with current display settings."""
        # Signed log so negatives don't collapse to zero. Matches JS `slog` for
        # signed data (diff_mode, phase, residuals, anything that can go negative).
        if self.log_scale:
            frame = np.sign(frame) * np.log1p(np.abs(frame))

        vmin, vmax = self._get_color_range(frame)

        if vmax > vmin:
            normalized = np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255)
            return normalized.astype(np.uint8)
        return np.zeros(frame.shape, dtype=np.uint8)

    def _update_all(self) -> None:
        """Ship frame_bytes to JS. Stats (mean/min/max/std/histogram) are
        recomputed entirely on the JS side from the float32 bytes - Python
        doing the same reductions wastes ~50 ms per scrub at 4k full-res.
        ROI stats stay on Python because the mask logic + per-ROI bookkeeping
        lives here. In offline mode JS slices from _offline_stack directly,
        so frame_bytes is dead weight in the HTML state - skip the write."""
        display_frame = None
        if self.roi_active or not self.separate_panel_frames:
            display_frame = self._get_display_frame()
        with self.hold_sync():
            if self.roi_active and display_frame is not None:
                self._update_roi_stats(display_frame)
            else:
                self.roi_stats = {}
            if not self.offline and not self.separate_panel_frames:
                self.frame_bytes = display_frame.tobytes()
            self.frame_seq = self.frame_seq + 1

    def _update_roi_stats(self, frame: np.ndarray) -> None:
        """Compute mean/min/max/std/area of the currently-selected ROI on `frame`
        and write to the `roi_stats` trait so the JS panel can render. Skipped
        when no ROI selected (cheap fast-path for the common scrub case)."""
        idx = self.roi_selected_idx
        if idx < 0 or idx >= len(self.roi_list):
            self.roi_stats = {}
            return
        roi = self.roi_list[idx]
        mask = self._roi_mask(roi)
        # Mask is built at display (binned) dims, matching `frame`. The torch path
        # used to index raw _data_torch[slice_idx] which is full-res → shape mismatch.
        # Stats on 16 MB binned numpy frame are <5 ms; no torch round-trip needed.
        region = frame[mask]
        if region.size > 0:
            self.roi_stats = {
                "mean": float(region.mean()),
                "min": float(region.min()),
                "max": float(region.max()),
                "std": float(region.std()),
            }
        else:
            self.roi_stats = {}

    def _compute_roi_plot(self) -> None:
        """Compute selected ROI mean for all frames. Uses display data (binned) for speed."""
        idx = self.roi_selected_idx
        if idx < 0 or idx >= len(self.roi_list):
            self.roi_plot_data = b""
            return
        mask = self._roi_mask(self.roi_list[idx])
        if mask.sum() == 0:
            self.roi_plot_data = b""
            return
        # Use _display_data (binned) - 4-16× less data than _data, same ROI result.
        # Cache torch view of _display_data on the instance so every drag doesn't
        # reallocate VRAM (was leaking ~4 GB/drag on large stacks).
        # Apply diff_mode so plot matches what the stats panel shows.
        data = self._display_data
        if self.diff_mode == "previous":
            diff = np.zeros_like(data)
            diff[1:] = data[1:] - data[:-1]
            data = diff
        elif self.diff_mode == "first":
            data = data - data[0:1]
        if self._use_torch and self.diff_mode == "off":
            if getattr(self, "_display_torch", None) is None:
                self._display_torch = torch.from_numpy(self._display_data).to(self._device)
            mask_t = torch.from_numpy(mask).to(self._device)
            masked = self._display_torch[:, mask_t]
            means = masked.mean(dim=1).cpu().numpy().astype(np.float32)
        else:
            means = np.array([float(data[i][mask].mean()) for i in range(self.n_slices)], dtype=np.float32)
        self.roi_plot_data = means.tobytes()

    def _roi_mask(self, roi: dict) -> np.ndarray:
        """Return a boolean mask of the ROI over the current frame. Supports
        circle / square / rectangle / annular shapes. Strict-< edge match the
        JS strokeRect drawing so Python stats agree with what the user sees."""
        r, c = np.ogrid[0 : self.height, 0 : self.width]
        shape = roi.get("shape", "circle")
        row = float(roi.get("row", 0))
        col = float(roi.get("col", 0))
        radius = max(1.0, float(roi.get("radius", 10)))
        if shape == "circle":
            return (c - col) ** 2 + (r - row) ** 2 <= radius**2
        if shape == "square":
            # Strict < to match JS strokeRect width = 2*radius (exclusive).
            return (np.abs(c - col) < radius) & (np.abs(r - row) < radius)
        if shape == "rectangle":
            half_w = max(1.0, float(roi.get("width", 20)) / 2.0)
            half_h = max(1.0, float(roi.get("height", 20)) / 2.0)
            # Strict < to match JS strokeRect (width=width, exclusive at edge).
            return (np.abs(c - col) < half_w) & (np.abs(r - row) < half_h)
        if shape == "annular":
            inner = max(0.0, float(roi.get("radius_inner", 5)))
            dist2 = (c - col) ** 2 + (r - row) ** 2
            return (dist2 >= inner**2) & (dist2 <= radius**2)
        return (c - col) ** 2 + (r - row) ** 2 <= radius**2

    def _upsert_selected_roi(self, updates: dict) -> None:
        """Merge `updates` into the currently-selected ROI, or append a new ROI
        if none is selected. Fills defaults + cyclic color so single-field edits
        (e.g. `radius=20`) don't strip unspecified fields."""
        if int(self.n_panels) > 1:
            raise ValueError("Show3D ROI tools are only available for single-panel widgets.")
        rois = list(self.roi_list)
        color_cycle = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"]
        defaults = {
            "shape": "circle",
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

    def _send_buffer(self, start_idx: int) -> None:
        end_idx = start_idx + self._buffer_size
        if self.diff_mode == "off":
            data = self._display_data
            if end_idx <= self.n_slices:
                chunk = data[start_idx:end_idx]
            else:
                chunk = np.concatenate(
                    [data[start_idx:], data[: end_idx - self.n_slices]]
                )
        else:
            frames = []
            for j in range(self._buffer_size):
                idx = (start_idx + j) % self.n_slices
                frames.append(self._get_display_frame(idx))
            chunk = np.stack(frames)
        with self.hold_sync():
            self._buffer_start = int(start_idx)
            self._buffer_count = int(chunk.shape[0])
            self._buffer_bytes = chunk.tobytes()

    def _sample_line(self, img: np.ndarray, row0: float, col0: float, row1: float, col1: float) -> np.ndarray:
        """Sample one pixel line between two endpoints via bilinear interpolation.
        N samples = ceil(line length). Used as the building block for line
        profiles; without bilinear the profile aliases on diagonal lines."""
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
                img[r1c, c1c] * cf * rf)

    def _sample_profile_on(self, img: np.ndarray, row0: float, col0: float, row1: float, col1: float) -> np.ndarray:
        """Sample a width-`profile_width` strip averaged perpendicular to the line.
        Averaging across the strip reduces shot noise on thin line profiles -
        a single-pixel sample is too noisy for atomic-resolution data."""
        pw = self.profile_width
        if pw <= 1:
            return self._sample_line(img, row0, col0, row1, col1).astype(np.float32)
        dc, dr = col1 - col0, row1 - row0
        length = (dc**2 + dr**2) ** 0.5
        if length < 1e-8:
            return self._sample_line(img, row0, col0, row1, col1).astype(np.float32)
        perp_r, perp_c = -dc / length, dr / length
        half = (pw - 1) / 2.0
        offsets = np.linspace(-half, half, pw)
        accumulated = None
        for off in offsets:
            vals = self._sample_line(img, row0 + off * perp_r, col0 + off * perp_c,
                                     row1 + off * perp_r, col1 + off * perp_c)
            if accumulated is None:
                accumulated = vals.copy()
            else:
                accumulated += vals
        return (accumulated / pw).astype(np.float32)

    def _sample_profile(self, row0: float, col0: float, row1: float, col1: float) -> np.ndarray:
        """Sample the line profile on the current display frame (binned, diff-aware)
        so the returned profile matches what the user sees."""
        return self._sample_profile_on(self._get_display_frame(), row0, col0, row1, col1)
