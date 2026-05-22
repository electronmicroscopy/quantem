"""
show3d: Interactive 3D stack viewer widget with advanced features.

For viewing a stack of 2D images (e.g., defocus sweep, time series, z-stack, movies).
Includes playback controls, statistics, ROI selection, FFT, and more.
"""

import gc
import json
import pathlib
import sys
from enum import Enum
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.show2d import _reject_unknown_kwargs
from quantem.widget.state import (
    build_json_header,
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


class Show3D(anywidget.AnyWidget):
    """
    Interactive 3D stack viewer with advanced features for electron microscopy.

    View a stack of 2D images along a specific dimension (e.g., defocus sweep,
    time series, depth stack, in-situ movies). Includes playback controls,
    statistics panel, ROI selection, FFT view, and more.

    Parameters
    ----------
    data : array_like
        3D array of shape (N, height, width) where N is the stack dimension.
    labels : list of str, optional
        Labels for each slice (e.g., ["C10=-500nm", "C10=-400nm", ...]).
        If None, uses slice indices.
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
    fps : float, default 5.0
        Frames per second for playback.
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
    show_resize_handles : bool, default True
        Render the bottom-right corner triangle on every real panel.  Dragging
        any handle resizes the entire multi-panel canvas (linked).  Set
        ``False`` to declutter a screenshot or printed figure where the
        operator already has the layout they want.
    show_zoom_indicator : bool, default True
        Draw the ``1.0×`` zoom readout at the bottom-left of every panel.
        Set ``False`` for clean static layouts or when the scale bar alone
        is enough to communicate scale.

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
    >>> from quantem.widget import Show3D
    >>>
    >>> # View defocus sweep
    >>> labels = [f"C10={c10:.0f}nm" for c10 in np.linspace(-500, -200, 12)]
    >>> Show3D(stack, labels=labels, title="Defocus Sweep")
    >>>
    >>> # View in-situ movie with timestamps
    >>> times = np.arange(100) * 0.1  # 100 frames at 10 fps
    >>> Show3D(movie, timestamps=times, timestamp_unit="s", fps=30)
    >>>
    >>> # With scale bar
    >>> Show3D(data, pixel_size=0.5, title="HRTEM")
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
    _display_bin_factor = traitlets.Int(1)  # Python-only: JS doesn't read
    # Flipped True by JS after the first colormap pass has painted to canvas.
    # Drives the truthful timing print (end-to-end, not __init__-only).
    _js_rendered = traitlets.Bool(False).tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("magma").tag(sync=True)
    dim_label = traitlets.Unicode("Frame").tag(sync=True)

    # Multi-Panel (side-by-side stacks, independent zoom by default with optional link)
    n_panels = traitlets.Int(1).tag(sync=True)
    panel_titles = traitlets.List(traitlets.Unicode()).tag(sync=True)
    panel_width_px = traitlets.Int(0)  # Python-only: JS infers panel width from frame width / n_panels
    # Real frame count per panel for stack comparison: stacks of different
    # lengths get auto-padded to the longest; this trait lets JS mark
    # "end-of-stack" frames (frame idx >= real[panel]). Empty = all real.
    panel_real_frames = traitlets.List(traitlets.Int()).tag(sync=True)
    # Single Link toggle controls both zoom AND pan (independent axes proved confusing).
    link_zoom = traitlets.Bool(True)  # Python-only: JS uses link_panels for linked zoom+pan
    link_pan = traitlets.Bool(True)   # Python-only: JS uses link_panels for linked zoom+pan
    link_panels = traitlets.Bool(True).tag(sync=True)
    link_contrast = traitlets.Bool(True).tag(sync=True)  # share vmin/vmax across panels
    # 0 = single row (no wrap). N > 0 = wrap into rows of at most N panels.
    max_cols = traitlets.Int(4).tag(sync=True)
    # Per-widget customization for multi-panel display.
    show_resize_handles = traitlets.Bool(True).tag(sync=True)
    show_zoom_indicator = traitlets.Bool(True).tag(sync=True)
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
    boomerang = traitlets.Bool(False).tag(sync=True)  # Ping-pong playback
    fps = traitlets.Float(5.0).tag(sync=True)  # Default 5 FPS for easier control
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
    # Per-panel stats (length = n_panels). Empty for single-panel.
    # Per-panel stats: JS computes its own locally (`localPanelStats`), so these
    # are Python-only readouts. Don't sync - saves shipping 4 List[Float] per scrub.
    stats_mean_per_panel = traitlets.List(traitlets.Float())
    stats_min_per_panel = traitlets.List(traitlets.Float())
    stats_max_per_panel = traitlets.List(traitlets.Float())
    stats_std_per_panel = traitlets.List(traitlets.Float())

    # =========================================================================
    # Display Options
    # =========================================================================
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(True).tag(sync=True)
    percentile_low = traitlets.Float(0.5).tag(sync=True)
    percentile_high = traitlets.Float(99.5).tag(sync=True)
    vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    data_min = traitlets.Float(0.0).tag(sync=True)
    data_max = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(0.0).tag(sync=True)  # 0 = no scale bar
    pixel_unit = traitlets.Unicode("A").tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    # Canvas smoothing: False = nearest-neighbor (sharp atoms); True = bilinear.
    smooth = traitlets.Bool(False).tag(sync=True)
    # Whole-stack rotation as k * 90 deg (k = 0..3). Applied in Python by rotating
    # _data and re-broadcasting frame_bytes; cheap for typical EM stacks.
    image_rotation = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # Timestamps / Dose
    # =========================================================================
    timestamps = traitlets.List(traitlets.Float()).tag(sync=True)
    timestamp_unit = traitlets.Unicode("s").tag(sync=True)
    current_timestamp = traitlets.Float(0.0)  # Python-only: JS reads timestamps[slice_idx] directly

    # =========================================================================
    # ROI Selection
    # =========================================================================
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_list = traitlets.List([]).tag(sync=True)
    roi_selected_idx = traitlets.Int(-1).tag(sync=True)
    roi_stats = traitlets.Dict({}).tag(sync=True)
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
    show_playback = traitlets.Bool(False)         # Python-only: not consumed in JS
    widget_version = traitlets.Unicode("unknown")  # Python-only: telemetry readout
    # =========================================================================
    # Line Profile
    # =========================================================================
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)
    profile_width = traitlets.Int(1).tag(sync=True)

    # =========================================================================
    # Export (GIF / ZIP of PNGs)
    # =========================================================================
    _gif_export_requested = traitlets.Bool(False).tag(sync=True)
    _gif_data = traitlets.Bytes(b"").tag(sync=True)
    _gif_metadata_json = traitlets.Unicode("").tag(sync=True)
    _zip_export_requested = traitlets.Bool(False).tag(sync=True)
    _zip_data = traitlets.Bytes(b"").tag(sync=True)
    _bundle_export_requested = traitlets.Bool(False).tag(sync=True)
    _bundle_data = traitlets.Bytes(b"").tag(sync=True)

    # =========================================================================
    # Playback Buffer (sliding prefetch)
    # =========================================================================
    _buffer_bytes = traitlets.Bytes(b"").tag(sync=True)
    _buffer_start = traitlets.Int(0).tag(sync=True)
    _buffer_count = traitlets.Int(0).tag(sync=True)
    _prefetch_request = traitlets.Int(-1).tag(sync=True)

    # Render-time telemetry (set after first browser paint; docstring promises these).
    render_total_ms = traitlets.Int(allow_none=True, default_value=None)
    render_python_build_ms = traitlets.Int(allow_none=True, default_value=None)
    render_wire_js_ms = traitlets.Int(allow_none=True, default_value=None)

    _VALID_DIFF_MODES = {"off", "previous", "first"}

    @traitlets.validate("diff_mode")
    def _validate_diff_mode(self, proposal):
        val = proposal["value"]
        if val not in self._VALID_DIFF_MODES:
            raise traitlets.TraitError(
                f"Invalid diff_mode '{val}'. Must be one of: {sorted(self._VALID_DIFF_MODES)}"
            )
        return val

    @traitlets.validate("playback_path")
    def _validate_playback_path(self, proposal):
        # Wrap indices to [0, n_slices) so JS never indexes out of bounds.
        # Direct trait assignment was raw; wrap here so JS never indexes OOB.
        val = list(proposal["value"])
        n = max(1, int(self.n_slices))
        return [int(i) % n for i in val]

    @traitlets.validate("vmax")
    def _validate_vmax_ge_vmin(self, proposal):
        import math
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
    def _validate_vmin_le_vmax(self, proposal):
        import math
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
    def _validate_cmap(self, proposal):
        val = str(proposal["value"])
        if val not in _VALID_CMAPS:
            raise traitlets.TraitError(
                f"Unknown cmap {val!r}. Valid: {sorted(_VALID_CMAPS)}"
            )
        return val

    @traitlets.validate("bookmarked_frames")
    def _validate_bookmarks(self, proposal):
        # Drop indices outside [0, n_slices). JS draws bookmark markers and
        # raw out-of-range values caused offscreen / negative positions.
        n = max(1, int(self.n_slices))
        return [int(i) for i in proposal["value"] if 0 <= int(i) < n]

    @traitlets.validate("loop_end")
    def _validate_loop_end(self, proposal):
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
    def _validate_loop_start(self, proposal):
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
    def _validate_pixel_size(self, proposal):
        val = float(proposal["value"])
        import math
        if math.isnan(val) or math.isinf(val):
            raise traitlets.TraitError(f"pixel_size must be finite, got {val}")
        if val < 0:
            raise traitlets.TraitError(f"pixel_size must be >= 0, got {val}")
        return val

    @traitlets.validate("labels")
    def _validate_labels(self, proposal):
        # Length must match n_slices; mismatch caused IndexError in ZIP export.
        val = list(proposal["value"])
        if val and len(val) != int(self.n_slices):
            raise traitlets.TraitError(
                f"labels length ({len(val)}) must equal n_slices ({self.n_slices}) or be empty"
            )
        return val

    @traitlets.validate("timestamps")
    def _validate_timestamps(self, proposal):
        # Empty list = no timestamps. Otherwise length must match n_slices.
        val = list(proposal["value"])
        if val and len(val) != int(self.n_slices):
            raise traitlets.TraitError(
                f"timestamps length ({len(val)}) must equal n_slices ({self.n_slices}) or be empty"
            )
        return val

    @traitlets.validate("panel_titles")
    def _validate_panel_titles(self, proposal):
        # Empty list = default per-panel labels. Otherwise must match n_panels.
        val = list(proposal["value"])
        if val and len(val) != int(self.n_panels):
            raise traitlets.TraitError(
                f"panel_titles length ({len(val)}) must equal n_panels ({self.n_panels}) or be empty"
            )
        return val

    @traitlets.validate("fps")
    def _validate_fps(self, proposal):
        import math
        val = float(proposal["value"])
        if not math.isfinite(val):
            raise traitlets.TraitError(f"fps must be finite, got {val}")
        if val <= 0:
            raise traitlets.TraitError(f"fps must be > 0, got {val}")
        return val

    @traitlets.validate("slice_idx")
    def _validate_slice_idx(self, proposal):
        # Clamp to [0, n_slices). State load with stale index used to crash
        # _update_all → _data[idx] with IndexError on a smaller new stack.
        val = int(proposal["value"])
        n = max(1, int(self.n_slices))
        return max(0, min(val, n - 1))

    @traitlets.validate("profile_width")
    def _validate_profile_width(self, proposal):
        val = int(proposal["value"])
        if val < 1:
            raise traitlets.TraitError(f"profile_width must be >= 1, got {val}")
        return val

    @traitlets.validate("percentile_low")
    def _validate_percentile_low(self, proposal):
        val = float(proposal["value"])
        if not 0 <= val <= 100:
            raise traitlets.TraitError(f"percentile_low must be in [0, 100], got {val}")
        if val >= float(self.percentile_high):
            raise traitlets.TraitError(
                f"percentile_low ({val}) must be < percentile_high ({self.percentile_high})"
            )
        return val

    @traitlets.validate("percentile_high")
    def _validate_percentile_high(self, proposal):
        val = float(proposal["value"])
        if not 0 <= val <= 100:
            raise traitlets.TraitError(f"percentile_high must be in [0, 100], got {val}")
        if val <= float(self.percentile_low):
            raise traitlets.TraitError(
                f"percentile_high ({val}) must be > percentile_low ({self.percentile_low})"
            )
        return val

    @traitlets.validate("roi_selected_idx")
    def _validate_roi_selected_idx(self, proposal):
        # -1 = nothing selected. Otherwise clamp to [0, len(roi_list))
        # so JS doesn't index OOB and Python stats don't silently return {}.
        val = int(proposal["value"])
        if val < 0:
            return -1
        return min(val, max(0, len(self.roi_list) - 1))

    _VALID_ROI_SHAPES = {"circle", "square", "rectangle", "annular"}

    @traitlets.validate("roi_list")
    def _validate_roi_list(self, proposal):
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
                        if float(r[k]) < 0:
                            raise traitlets.TraitError(
                                f"ROI {i}: {k} must be >= 0, got {r[k]}"
                            )
                    except (TypeError, ValueError):
                        pass
        return val

    def __init__(
        self,
        *data_args,
        labels: list[str] | None = None,
        panel_titles: list[str] | None = None,
        panel_real_frames: list[int] | None = None,
        title: str = "",
        cmap: str | Colormap = Colormap.MAGMA,
        vmin: float | None = None,
        vmax: float | None = None,
        pixel_size: float = 0.0,
        pixel_unit: str = "A",
        smooth: bool = False,
        image_rotation: int = 0,
        log_scale: bool = False,
        auto_contrast: bool = True,
        percentile_low: float = 0.5,
        percentile_high: float = 99.5,
        fps: float = 5.0,
        timestamps: list[float] | None = None,
        timestamp_unit: str = "s",
        show_fft: bool = False,
        fft_window: bool = True,
        show_playback: bool = False,
        show_stats: bool = False,
        show_controls: bool = True,
        size: int = 0,
        diff_mode: str = "off",
        buffer_size: int = 64,
        dim_label: str = "Frame",
        use_torch: bool | None = None,
        device: str | None = None,
        display_bin: int | str = "auto",
        hideable: bool = False,
        state=None,
        max_cols: int | None = None,
        panel_gap: int | None = None,
        panel_title_font_size: int | None = None,
        show_resize_handles: bool | None = None,
        show_zoom_indicator: bool | None = None,
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
        if show_resize_handles is not None:
            kwargs["show_resize_handles"] = bool(show_resize_handles)
        if show_zoom_indicator is not None:
            kwargs["show_zoom_indicator"] = bool(show_zoom_indicator)
        import time
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
                            auto_contrast=auto_contrast, percentile_low=percentile_low,
                            percentile_high=percentile_high, fps=fps, timestamps=timestamps,
                            timestamp_unit=timestamp_unit, show_fft=show_fft,
                            fft_window=fft_window, show_playback=show_playback,
                            show_stats=show_stats, show_controls=show_controls,
                            size=size,
                            diff_mode=diff_mode, buffer_size=buffer_size,
                            dim_label=dim_label, use_torch=use_torch, device=device,
                            display_bin=display_bin, state=state, _t0=_t0)

    def _init_sync(self, data_args, *, labels, panel_titles, title, cmap, vmin, vmax,
                   pixel_size, pixel_unit, smooth, image_rotation,
                   log_scale, auto_contrast, percentile_low, percentile_high,
                   fps, timestamps, timestamp_unit, show_fft, fft_window, show_playback,
                   show_stats, show_controls, size,
                   diff_mode, buffer_size, dim_label, use_torch, device, display_bin,
                   state, _t0):
        import time
        self.widget_version = resolve_widget_version()

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
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            _extracted_title = data.name if data.name else None
            if hasattr(data, "sampling") and len(data.sampling) >= 3:
                sampling_val = float(data.sampling[1])
                if hasattr(data, "units"):
                    units = list(data.units)
                    if units[1] in ("nm", "nanometer"):
                        sampling_val = sampling_val * 10  # nm → Å
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

        # Multi-panel: convert remaining args, validate shapes, concatenate
        if len(data_args) > 1:
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
            if panel_titles is not None:
                self.panel_titles = list(panel_titles)
            else:
                self.panel_titles = [f"Panel {i+1}" for i in range(len(panels))]
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
            # NEVER BIN (CLAUDE.md rule). Operator wants full source resolution
            # on every multi-panel surface - pixel-exact for microscopy.
            # Memory: bumped JS-side buffer cap (see _buffer_size logic) so
            # 10 panels × 1366² × 4B fits.
            panel_bin = 1
            orig_h = panels[0].shape[1]
            normalized = []
            for p in panels:
                if p.size > 10_000_000:
                    sample = p.flat[::max(1, p.size // 1_000_000)]
                    p2, p98 = np.percentile(sample, [2, 98])
                else:
                    p2, p98 = np.percentile(p, [2, 98])
                rng_inv = np.float32(1.0 / max(p98 - p2, 1e-10))
                normalized.append((p - np.float32(p2)) * rng_inv)

            # Concatenate panels back-to-back. JS paints clean bg-color gaps
            # between them at render time (so the gap matches the page theme
            # instead of becoming colormap[0]).
            data = np.concatenate(normalized, axis=2)
            self._panel_width = panels[0].shape[2]
            self.panel_width_px = self._panel_width
            self._multi_panel_bin = panel_bin
        else:
            self.n_panels = 1
            self._multi_panel_bin = 0
            if panel_titles is not None:
                self.panel_titles = list(panel_titles)

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

        if self._display_bin > 1:
            from quantem.widget.array_utils import bin2d
            self._display_data = bin2d(self._data, factor=self._display_bin, mode="mean")
            self.height = int(self._display_data.shape[1])
            self.width = int(self._display_data.shape[2])
            self._display_bin_factor = self._display_bin
            if pixel_size > 0:
                pixel_size = pixel_size * self._display_bin
            print(f"  Display bin {self._display_bin}× (explicit): {orig_h}×{orig_w} → {self.height}×{self.width}")
        else:
            self._display_data = self._data
            self.height = orig_h
            self.width = orig_w
            self._display_bin_factor = 1

        # Color range (global across all frames)
        self._vmin_user = vmin
        self._vmax_user = vmax
        if self._use_torch:
            self._vmin = vmin if vmin is not None else float(self._data_torch.min().item())
            self._vmax = vmax if vmax is not None else float(self._data_torch.max().item())
            self.data_min = float(self._data_torch.min().item())
            self.data_max = float(self._data_torch.max().item())
        else:
            self._vmin = vmin if vmin is not None else float(self._data.min())
            self._vmax = vmax if vmax is not None else float(self._data.max())
            self.data_min = float(self._data.min())
            self.data_max = float(self._data.max())
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

        # Use extracted pixel_size if not explicitly provided
        if pixel_size == 0.0 and _extracted_pixel_size is not None:
            pixel_size = _extracted_pixel_size

        # Display options
        self.pixel_size = pixel_size
        self.pixel_unit = pixel_unit
        self.smooth = smooth
        # image_rotation: pure display-side (JS canvas transform), no data copy.
        self.image_rotation = image_rotation % 4
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.vmin = vmin
        self.vmax = vmax
        self.fps = fps

        # Timestamps
        if timestamps is not None:
            self.timestamps = [float(t) for t in timestamps]
        else:
            self.timestamps = []
        self.timestamp_unit = timestamp_unit
        self.dim_label = dim_label
        self.diff_mode = diff_mode
        self.show_fft = show_fft
        self.fft_window = fft_window
        self.show_playback = show_playback
        self.show_stats = show_stats
        self.show_controls = show_controls
        self.size = size
        frame_bytes = self.height * self.width * 4  # float32
        # 256 MB buffer cap. Holds 16 frames at 16 MB/frame (4K binned) so
        # scrubbing within window is paint-limited (no Comm round-trip).
        # Sliding window prefetches outside the cached zone.
        max_buffer_bytes = 4 * 1024 * 1024 * 1024  # 4 GB cap, NEVER BIN
        min_buffer_frames = 8
        max_frames = max(min_buffer_frames, max_buffer_bytes // frame_bytes)
        self._buffer_size = min(buffer_size, self.n_slices, max_frames)

        # Initial position at middle
        self.slice_idx = int(self.n_slices // 2)
        self._roi_plot_timer = None

        # Observers
        self.observe(self._on_slice_change, names=["slice_idx"])
        self.observe(
            self._on_roi_change,
            names=["roi_active", "roi_list", "roi_selected_idx"],
        )
        self.observe(self._on_gif_export, names=["_gif_export_requested"])
        self.observe(self._on_zip_export, names=["_zip_export_requested"])
        self.observe(self._on_bundle_export, names=["_bundle_export_requested"])
        self.observe(self._on_playing_change, names=["playing"])
        self.observe(self._on_prefetch, names=["_prefetch_request"])
        self.observe(self._on_diff_mode_change, names=["diff_mode"])

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

    def _on_first_render(self, change):
        import time
        if not change.get("new"):
            return
        total_ms = (time.perf_counter() - self._init_t0) * 1000
        py_ms = self._init_py_elapsed_ms
        shape = f"{self.n_slices}×{self.height}×{self.width}"
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

    def set_image(self, data, labels=None):
        """Replace the stack data. Preserves all display settings."""
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
        # Stop playback so JS doesn't keep painting from the stale _buffer_bytes
        # while we swap data. Invalidate cached torch view of display data.
        # Cancel any pending ROI plot timer so it doesn't fire mid-swap with
        # stale _display_data dims (race observed by audit).
        if getattr(self, "_roi_plot_timer", None) is not None:
            # threading.Timer.cancel() is documented not to raise; guard kept
            # for defensive paranoia only.
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
        self._multi_panel_bin = 0
        self._panel_width = int(data.shape[2])
        self.n_slices = int(data.shape[0])

        # Auto-bin display data
        orig_h, orig_w = data.shape[1], data.shape[2]
        frame_mb = orig_h * orig_w * 4 / (1024 * 1024)
        self._display_bin = 1
        if frame_mb > 32:
            for bf in [2, 4, 8]:
                if frame_mb / (bf * bf) <= 32:
                    self._display_bin = bf
                    break
            else:
                self._display_bin = 8

        if self._display_bin > 1:
            from quantem.widget.array_utils import bin2d
            self._display_data = bin2d(self._data, factor=self._display_bin, mode="mean")
            self.height = int(self._display_data.shape[1])
            self.width = int(self._display_data.shape[2])
            self._display_bin_factor = self._display_bin
        else:
            self._display_data = self._data
            self.height = orig_h
            self.width = orig_w
            self._display_bin_factor = 1

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
        max_buffer_bytes = 4 * 1024 * 1024 * 1024  # 4 GB cap, NEVER BIN
        max_frames = max(8, max_buffer_bytes // max(1, frame_bytes_n))
        self._buffer_size = min(self._buffer_size, self.n_slices, max_frames)
        with self.hold_sync():
            self._buffer_bytes = b""
            self._buffer_count = 0
            self._buffer_start = 0
        self._update_all()

    def __repr__(self) -> str:
        parts = f"Show3D({self.n_slices}×{self.height}×{self.width}, frame={self.slice_idx}, cmap={self.cmap}"
        if self.diff_mode != "off":
            parts += f", diff={self.diff_mode}"
        parts += ")"
        return parts

    def state_dict(self):
        return {
            "title": self.title,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            # percentile_high before percentile_low so cross-validator doesn't reject mid-load.
            "percentile_high": self.percentile_high,
            "percentile_low": self.percentile_low,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "show_fft": self.show_fft,
            "fft_window": self.fft_window,
            "show_playback": self.show_playback,
            "pixel_size": self.pixel_size,
            "pixel_unit": self.pixel_unit,
            "smooth": self.smooth,
            "image_rotation": self.image_rotation,
            "scale_bar_visible": self.scale_bar_visible,
            "size": self.size,
            "fps": self.fps,
            "loop": self.loop,
            "reverse": self.reverse,
            "boomerang": self.boomerang,
            # IMPORTANT: loop_end MUST precede loop_start in dict order.
            # Validators cross-check; loop_start with -1 sentinel skips check,
            # but loop_end=5 then loop_start=10 would otherwise raise during load.
            "loop_end": self.loop_end,
            "loop_start": self.loop_start,
            "bookmarked_frames": self.bookmarked_frames,
            "playback_path": self.playback_path,
            "slice_idx": self.slice_idx,
            "roi_active": self.roi_active,
            "roi_list": self.roi_list,
            "roi_selected_idx": self.roi_selected_idx,
            "profile_line": self.profile_line,
            "profile_width": self.profile_width,
            "diff_mode": self.diff_mode,
            "dim_label": self.dim_label,
            "labels": list(self.labels),
            "panel_titles": list(self.panel_titles),
            "timestamps": list(self.timestamps),
            "timestamp_unit": self.timestamp_unit,
        }

    def save(self, path: str):
        save_state_file(path, "Show3D", self.state_dict())

    def load_state_dict(self, state):
        import warnings
        state = dict(state)
        allowed = {
            "title", "cmap", "log_scale", "auto_contrast",
            "percentile_high", "percentile_low", "vmin", "vmax",
            "show_stats", "show_controls", "show_fft", "fft_window",
            "show_playback", "pixel_size", "pixel_unit", "smooth",
            "image_rotation", "scale_bar_visible", "size", "fps",
            "loop", "reverse", "boomerang", "loop_end", "loop_start",
            "bookmarked_frames", "playback_path", "slice_idx",
            "roi_active", "roi_list", "roi_selected_idx", "profile_line",
            "profile_width", "diff_mode", "dim_label", "labels",
            "panel_titles", "timestamps", "timestamp_unit",
        }
        unknown = []
        if "canvas_size" in state:
            state["size"] = state.pop("canvas_size")
        # `display_bin` is constructor/data dependent. Loading only the private
        # integer leaves display_data/height/width stale, so ignore saved values.
        state.pop("display_bin", None)
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

    def free(self):
        """Release VRAM and RAM held by this widget. `del widget` won't
        free memory because traitlets observers pin the refcount."""
        if self._data is None:
            return
        # Cancel pending ROI debounce so its callback can't fire post-free.
        if self._roi_plot_timer is not None:
            self._roi_plot_timer.cancel()
            self._roi_plot_timer = None
        device = str(self._device) if self._device is not None else ""
        self._data = None
        self._data_torch = None
        self._display_data = None
        for trait in ("frame_bytes", "roi_plot_data", "_gif_data", "_zip_data", "_bundle_data", "_buffer_bytes"):
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

    @property
    def visible_indices(self) -> list[int]:
        """Live list of frame indices NOT in hidden_indices. Read-only;
        mutate via set_hidden() / show_all() / hide()."""
        hidden = set(self.hidden_indices)
        return [i for i in range(self.n_slices) if i not in hidden]

    def hide(self, *indices: int) -> "Show3D":
        """Hide one or more frames from the scrubber. Idempotent."""
        keep = set(self.hidden_indices) | {int(i) for i in indices}
        # Always keep at least one frame visible.
        if len(keep) >= self.n_slices:
            return self
        self.hidden_indices = sorted(keep)
        return self

    def show(self, *indices: int) -> "Show3D":
        """Restore frames previously hidden. Idempotent."""
        drop = {int(i) for i in indices}
        self.hidden_indices = sorted(set(self.hidden_indices) - drop)
        return self

    def set_hidden(self, indices: list[int]) -> "Show3D":
        """Replace the hidden set wholesale."""
        clean = sorted({int(i) for i in indices if 0 <= int(i) < self.n_slices})
        # Always keep at least one frame visible.
        if len(clean) >= self.n_slices:
            clean = clean[:-1]
        self.hidden_indices = clean
        return self

    def show_all(self) -> "Show3D":
        """Restore every frame."""
        self.hidden_indices = []
        return self

    def summary(self):
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
        # Signed log so negatives don't collapse to zero. Matches JS `slog` so
        # GIF/PNG exports look identical to live render for signed data
        # (diff_mode, phase, residuals, anything that can go negative).
        if self.log_scale:
            frame = np.sign(frame) * np.log1p(np.abs(frame))

        vmin, vmax = self._get_color_range(frame)

        if vmax > vmin:
            normalized = np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255)
            return normalized.astype(np.uint8)
        return np.zeros(frame.shape, dtype=np.uint8)

    def _get_display_frame(self, idx=None):
        if idx is None:
            idx = self.slice_idx
        data = self._display_data
        frame = data[idx]
        if self.diff_mode == "previous":
            if idx == 0:
                return np.zeros_like(frame)
            return frame - data[idx - 1]
        if self.diff_mode == "first":
            return frame - data[0]
        return frame

    def _on_diff_mode_change(self, change=None):
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
        self._update_all()

    def _update_all(self):
        """Update frame, stats, and all derived data. Uses hold_sync for batched transfer."""
        display_frame = self._get_display_frame()
        with self.hold_sync():
            # Stats from display_frame (binned). Full-res stats on 4K cost ~50ms/scrub.
            # Binned stats on 2K cost ~12ms = scrub FPS jumps 17 → 60+.
            self.stats_mean = float(display_frame.mean())
            self.stats_min = float(display_frame.min())
            self.stats_max = float(display_frame.max())
            self.stats_std = float(display_frame.std())
            # Per-panel stats so multi-panel widgets show each panel's range
            # separately rather than a misleading global aggregate.
            if self.n_panels > 1 and self._panel_width > 0:
                pw = self._panel_width
                means, mins, maxs, stds = [], [], [], []
                for i in range(self.n_panels):
                    sl = display_frame[:, i * pw:(i + 1) * pw]
                    means.append(float(sl.mean()))
                    mins.append(float(sl.min()))
                    maxs.append(float(sl.max()))
                    stds.append(float(sl.std()))
                self.stats_mean_per_panel = means
                self.stats_min_per_panel = mins
                self.stats_max_per_panel = maxs
                self.stats_std_per_panel = stds
            if self.timestamps and self.slice_idx < len(self.timestamps):
                self.current_timestamp = self.timestamps[self.slice_idx]
            if self.roi_active:
                self._update_roi_stats(display_frame)
            else:
                self.roi_stats = {}
            self.frame_bytes = display_frame.tobytes()
            self.frame_seq = self.frame_seq + 1

    def _roi_mask(self, roi: dict):
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

    def _update_roi_stats(self, frame: np.ndarray):
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

    def _send_buffer(self, start_idx: int):
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

    def _on_playing_change(self, change=None):
        if self.playing:
            self._send_buffer(self.slice_idx)
        else:
            # Playback stopped - refresh stats for the current frame
            self._update_all()

    def _on_prefetch(self, change=None):
        if self._prefetch_request >= 0 and self.playing:
            self._send_buffer(self._prefetch_request % self.n_slices)

    def _on_slice_change(self, change=None):
        if self.playing:
            return
        self._update_all()

    def _on_roi_change(self, change=None):
        """Handle ROI change. Stats for current frame are instant.
        Full-stack ROI plot is debounced (500ms) to avoid UI freeze during drag."""
        # Auto-select first ROI if the user added one programmatically and
        # roi_selected_idx is still -1 (otherwise stats stay empty silently).
        if self.roi_active and self.roi_list and self.roi_selected_idx < 0:
            self.roi_selected_idx = 0
        if self.roi_active:
            self._update_roi_stats(self._get_display_frame())
            # Debounce the expensive all-frame ROI plot
            if self._roi_plot_timer is not None:
                self._roi_plot_timer.cancel()
            import threading
            self._roi_plot_timer = threading.Timer(0.5, self._compute_roi_plot)
            self._roi_plot_timer.start()
        else:
            self.roi_stats = {}
            self.roi_plot_data = b""

    def _compute_roi_plot(self):
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

    # =========================================================================
    # Public Methods
    # =========================================================================

    def play(self) -> Self:
        """Start playback."""
        self.playing = True
        return self

    def pause(self) -> Self:
        """Pause playback."""
        self.playing = False
        return self

    def stop(self) -> Self:
        """Stop playback and reset to beginning."""
        self.playing = False
        self.slice_idx = 0
        return self

    def goto(self, index: int) -> Self:
        """Jump to a specific frame index."""
        self.slice_idx = int(index) % self.n_slices
        return self

    def profile_all_frames(self, start: tuple | None = None, end: tuple | None = None) -> np.ndarray:
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

    def _upsert_selected_roi(self, updates: dict):
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

    def _sample_line(self, img, row0, col0, row1, col1):
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

    def _sample_profile_on(self, img, row0, col0, row1, col1):
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

    def _sample_profile(self, row0, col0, row1, col1):
        return self._sample_profile_on(self._get_display_frame(), row0, col0, row1, col1)

    def set_profile(self, start: tuple, end: tuple) -> Self:
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

    def _on_gif_export(self, change=None):
        if not self._gif_export_requested:
            return
        self._gif_export_requested = False
        try:
            self._generate_gif()
        except (RuntimeError, OSError, ValueError, MemoryError, ImportError) as e:
            # On error: clear _gif_data + bump frame_seq so JS observer fires
            # and resets exporting=False. Without this the UI shows "..." forever.
            import warnings
            warnings.warn(f"GIF export failed: {type(e).__name__}: {e}")
            self._gif_data = b""

    def _normalize_frames_torch(self, start: int, end: int) -> np.ndarray:
        """Batch-normalize frames [start, end] on GPU. Returns (N, H, W) uint8 numpy."""
        frames = self._data_torch[start : end + 1].clone()
        # Signed log so negatives don't collapse to 0 (matches _normalize_frame
        # + JS slog). Without this, GIF export of phase/diff data looks wrong.
        if self.log_scale:
            frames = torch.sign(frames) * torch.log1p(torch.abs(frames))
        if self.auto_contrast:
            flat = frames.reshape(-1).float()
            # torch.quantile fails with "input tensor is too large" above 2^24 ≈ 16.7M
            # elements (e.g. 16 × 1370² = 30M). Subsample for percentile estimation -
            # 1M samples is more than enough for the 2/98 percentile to within ~0.01%.
            if flat.numel() > 16_000_000:
                stride = flat.numel() // 1_000_000
                flat_sub = flat[::stride]
            else:
                flat_sub = flat
            vmin = float(torch.quantile(flat_sub, self.percentile_low / 100.0).item())
            vmax = float(torch.quantile(flat_sub, self.percentile_high / 100.0).item())
        else:
            vmin = self._vmin
            vmax = self._vmax
            if self.log_scale:
                vmin = float(np.sign(vmin) * np.log1p(abs(vmin)))
                vmax = float(np.sign(vmax) * np.log1p(abs(vmax)))
        if vmax > vmin:
            normalized = torch.clamp((frames - vmin) / (vmax - vmin) * 255.0, 0, 255).to(torch.uint8)
        else:
            normalized = torch.zeros_like(frames, dtype=torch.uint8)
        return normalized.cpu().numpy()

    def _generate_gif(self):
        import io

        from matplotlib import colormaps
        from PIL import Image

        start = max(0, self.loop_start)
        end = self.loop_end if self.loop_end >= 0 else self.n_slices - 1
        end = min(end, self.n_slices - 1)

        cmap_fn = colormaps.get_cmap(self.cmap)
        duration_ms = int(1000 / max(0.1, self.fps))

        pil_frames = []
        if self._use_torch:
            normalized_all = self._normalize_frames_torch(start, end)
            for i in range(normalized_all.shape[0]):
                rgba = cmap_fn(normalized_all[i] / 255.0)
                rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                pil_frames.append(Image.fromarray(rgb))
        else:
            for i in range(start, end + 1):
                frame = self._data[i]
                normalized = self._normalize_frame(frame)
                rgba = cmap_fn(normalized / 255.0)
                rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                pil_frames.append(Image.fromarray(rgb))

        if not pil_frames:
            with self.hold_sync():
                self._gif_data = b""
                self._gif_metadata_json = ""
            return

        buf = io.BytesIO()
        # Use shared adaptive palette so frames don't quantize independently
        # (default PIL per-frame palette caused colormap flicker between slices).
        pil_p = [f.convert("P", palette=Image.ADAPTIVE, colors=256) for f in pil_frames]
        pil_p[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=pil_p[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
        )
        metadata = {
            **build_json_header("Show3D"),
            "format": "gif",
            "export_kind": "animated_frames",
            "frame_range": {"start": int(start), "end": int(end)},
            "n_frames": int(len(pil_frames)),
            "duration_ms": int(duration_ms),
            "display": {
                "cmap": self.cmap,
                "log_scale": bool(self.log_scale),
                "auto_contrast": bool(self.auto_contrast),
                "percentile_low": float(self.percentile_low),
                "percentile_high": float(self.percentile_high),
            },
        }
        gif_bytes = buf.getvalue()
        with self.hold_sync():
            self._gif_metadata_json = ""
            self._gif_data = b""
        with self.hold_sync():
            self._gif_metadata_json = json.dumps(metadata, indent=2)
            self._gif_data = gif_bytes

    def _on_zip_export(self, change=None):
        if not self._zip_export_requested:
            return
        self._zip_export_requested = False
        try:
            self._generate_zip()
        except (RuntimeError, OSError, ValueError, MemoryError, ImportError) as e:
            import warnings
            warnings.warn(f"ZIP export failed: {type(e).__name__}: {e}")
            self._zip_data = b""

    def _generate_zip(self):
        import io
        import zipfile

        from matplotlib import colormaps
        from PIL import Image

        start = max(0, self.loop_start)
        end = self.loop_end if self.loop_end >= 0 else self.n_slices - 1
        end = min(end, self.n_slices - 1)

        cmap_fn = colormaps.get_cmap(self.cmap)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            metadata = {
                **build_json_header("Show3D"),
                "format": "zip",
                "export_kind": "png_frames",
                "frame_range": {"start": int(start), "end": int(end)},
                "n_frames": int(end - start + 1),
                "display": {"cmap": self.cmap, "log_scale": bool(self.log_scale)},
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
            if self._use_torch:
                normalized_all = self._normalize_frames_torch(start, end)
                for j in range(normalized_all.shape[0]):
                    i = start + j
                    rgba = cmap_fn(normalized_all[j] / 255.0)
                    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                    img = Image.fromarray(rgb)
                    img_buf = io.BytesIO()
                    img.save(img_buf, format="PNG")
                    label = self.labels[i] if self.labels else str(i).zfill(4)
                    zf.writestr(f"frame_{label}.png", img_buf.getvalue())
            else:
                for i in range(start, end + 1):
                    frame = self._data[i]
                    normalized = self._normalize_frame(frame)
                    rgba = cmap_fn(normalized / 255.0)
                    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                    img = Image.fromarray(rgb)
                    img_buf = io.BytesIO()
                    img.save(img_buf, format="PNG")
                    label = self.labels[i] if self.labels else str(i).zfill(4)
                    zf.writestr(f"frame_{label}.png", img_buf.getvalue())
        zip_bytes = buf.getvalue()
        self._zip_data = b""
        self._zip_data = zip_bytes

    def _on_bundle_export(self, change=None):
        if not self._bundle_export_requested:
            return
        self._bundle_export_requested = False
        self._generate_bundle()

    def _roi_timeseries_csv(self) -> str:
        import csv
        import io

        rois = list(self.roi_list)
        masks = [self._roi_mask(roi) for roi in rois]
        out = io.StringIO()
        writer = csv.writer(out)
        header = ["frame_index", "label"]
        if self.timestamps and len(self.timestamps) >= self.n_slices:
            header.append(f"timestamp_{self.timestamp_unit or 'value'}")
        header.extend([f"roi_{i + 1}_mean" for i in range(len(rois))])
        writer.writerow(header)

        if self._use_torch:
            # Vectorized per-ROI means across all frames
            masks_t = [torch.from_numpy(m).to(self._device) for m in masks]
            roi_means = []
            for mask_t in masks_t:
                masked = self._data_torch[:, mask_t]  # (n_slices, n_pixels)
                if masked.shape[1] > 0:
                    roi_means.append(masked.mean(dim=1).cpu().numpy())
                else:
                    roi_means.append(np.full(self.n_slices, np.nan))
            for i in range(self.n_slices):
                row = [i, self.labels[i] if i < len(self.labels) else str(i)]
                if self.timestamps and len(self.timestamps) >= self.n_slices:
                    row.append(float(self.timestamps[i]))
                for rm in roi_means:
                    val = rm[i]
                    row.append(float(val) if not np.isnan(val) else "")
                writer.writerow(row)
        else:
            for i in range(self.n_slices):
                row = [i, self.labels[i] if i < len(self.labels) else str(i)]
                if self.timestamps and len(self.timestamps) >= self.n_slices:
                    row.append(float(self.timestamps[i]))
                frame = self._data[i]
                for mask in masks:
                    region = frame[mask]
                    row.append(float(region.mean()) if region.size > 0 else "")
                writer.writerow(row)
        return out.getvalue()

    def _generate_bundle(self):
        import io
        import zipfile

        from matplotlib import colormaps
        from PIL import Image

        idx = int(np.clip(self.slice_idx, 0, self.n_slices - 1))
        cmap_fn = colormaps.get_cmap(self.cmap)
        # Respect diff_mode so saved frame matches what user sees.
        frame = self._data[idx]
        if self.diff_mode == "previous":
            frame = frame - self._data[idx - 1] if idx > 0 else np.zeros_like(frame)
        elif self.diff_mode == "first":
            frame = frame - self._data[0]
        normalized = self._normalize_frame(frame)
        rgba = cmap_fn(normalized / 255.0)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb)
        img_buf = io.BytesIO()
        img.save(img_buf, format="PNG")

        state_payload = {**build_json_header("Show3D"), "state": self.state_dict()}
        csv_text = self._roi_timeseries_csv()
        label = self.labels[idx] if idx < len(self.labels) else str(idx)
        safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(label)).strip("_") or str(idx)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"frame_{safe_label}.png", img_buf.getvalue())
            zf.writestr("roi_timeseries.csv", csv_text)
            zf.writestr("state.json", json.dumps(state_payload, indent=2))
        bundle_bytes = buf.getvalue()
        self._bundle_data = b""
        self._bundle_data = bundle_bytes


    def save_image(self, path: str | pathlib.Path, *, frame_idx: int | None = None,
                   format: str | None = None, dpi: int = 150) -> pathlib.Path:
        """Save a single frame as PNG, PDF, or TIFF.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        frame_idx : int, optional
            Frame index to export. Defaults to current slice_idx.
        format : str, optional
            'png', 'pdf', or 'tiff'. If omitted, inferred from file extension.
        dpi : int, default 150
            Output DPI metadata.

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
