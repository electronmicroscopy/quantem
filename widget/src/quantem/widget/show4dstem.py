"""
show4dstem: Fast interactive 4D-STEM viewer widget.

Single chunked-torch path on every device (CUDA / MPS / CPU). Reductions cast
uint16 → float32 in scan-row chunks bounded by _CHUNK_BYTE_BUDGET, so transient
memory stays the same regardless of total dataset size.

To reduce data size, bin k-space at the dataset level before viewing:

    dataset = dataset.bin(2, axes=(2, 3))  # 2x2 k-space binning
    widget = Show4DSTEM(dataset)
"""

import json
import math
import pathlib
import time
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from quantem.core.datastructures import Dataset4dstem

import anywidget
import numpy as np
import torch
import traitlets
from quantem.widget.array_utils import to_numpy
from quantem.widget.state import (
    build_json_header,
    resolve_widget_version,
    save_state_file,
    unwrap_state_payload,
)

from quantem.core.config import validate_device

# Cap transient chunk memory at ~600 MB regardless of detector size.
_CHUNK_BYTE_BUDGET = 600 * 1024 * 1024


def _format_memory(nbytes: int) -> str:
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.0f} MB"
    if nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.0f} KB"
    return f"{nbytes} B"


# ============================================================================
# Constants
# ============================================================================
DEFAULT_BF_RATIO = 0.125  # BF disk radius as fraction of detector size (1/8)
MIN_LOG_VALUE = 1e-10  # Minimum value for log scale to avoid log(0)
DEFAULT_VI_ROI_RATIO = 0.15  # Default VI ROI size as fraction of scan dimension

class Show4DSTEM(anywidget.AnyWidget):
    """
    Fast interactive 4D-STEM viewer with advanced features.

    Optimized for speed with binary transfer and pre-normalization.
    Works with NumPy and PyTorch arrays.

    Parameters
    ----------
    data : Dataset4dstem or array_like
        Dataset4dstem object (calibration auto-extracted), 4D array
        of shape (scan_rows, scan_cols, det_rows, det_cols), or 5D array
        of shape (n_frames, scan_rows, scan_cols, det_rows, det_cols)
        for time-series or tilt-series data.
    scan_shape : tuple, optional
        If data is flattened (N, det_rows, det_cols), provide scan dimensions.
    sampling : tuple of 4 floats, optional
        Pixel size per axis ``(scan_row, scan_col, k_row, k_col)``. Scalar
        broadcasts to all four axes. Defaults to ``(1, 1, 1, 1)``.
        Auto-extracted from ``Dataset4dstem`` if not provided.
    units : list of 4 str, optional
        Unit string per axis. Common: ``["A", "A", "mrad", "mrad"]``.
        Defaults to ``["pixels"] * 4``. Auto-extracted from
        ``Dataset4dstem`` if not provided.
    center : tuple[float, float], optional
        (center_row, center_col) of the diffraction pattern in pixels.
        If not provided, defaults to detector center.
    bf_radius : float, optional
        Bright field disk radius in pixels. If not provided, estimated as 1/8 of detector size.
    precompute_virtual_images : bool, default True
        Precompute BF/ABF/LAADF/HAADF virtual images for preset switching.
    frame_dim_label : str, optional
        Label for the frame dimension when 5D data is provided.
        Defaults to "Frame". Common values: "Tilt", "Time", "Focus".
    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Show4DSTEM

    4D NumPy array ``(scan_rows, scan_cols, det_rows, det_cols)``:

    >>> Show4DSTEM(np.random.rand(64, 64, 128, 128))

    PyTorch tensor (CPU or GPU):

    >>> import torch
    >>> Show4DSTEM(torch.rand(64, 64, 128, 128))

    With explicit calibration (real-space Å, k-space mrad):

    >>> Show4DSTEM(np.random.rand(64, 64, 128, 128),
    ...            sampling=(2.39, 2.39, 0.46, 0.46),
    ...            units=["A", "A", "mrad", "mrad"])

    quantem ``Dataset4dstem`` — calibration + units auto-extracted:

    >>> from quantem.core.datastructures import Dataset4dstem
    >>> ds = Dataset4dstem.from_array(np.random.rand(64, 64, 128, 128))
    >>> Show4DSTEM(ds)

    Flattened scan ``(N, det_rows, det_cols)`` with explicit scan shape:

    >>> Show4DSTEM(np.random.rand(4096, 128, 128), scan_shape=(64, 64))

    Custom BF disk center and radius (overrides auto-detection):

    >>> Show4DSTEM(np.random.rand(64, 64, 128, 128),
    ...            center=(64, 64), bf_radius=12)

    5D time-series or tilt-series ``(n_frames, scan_r, scan_c, det_r, det_c)``:

    >>> Show4DSTEM(np.random.rand(20, 64, 64, 128, 128), frame_dim_label="Tilt")

    Raster animation (scan path through 4D dataset):

    >>> w = Show4DSTEM(np.random.rand(64, 64, 128, 128))
    >>> w.raster(step=2, interval_ms=50)

    Static export to PDF or PNG (single panel or all four):

    >>> w = Show4DSTEM(np.random.rand(64, 64, 128, 128))
    >>> w.save_image("dp.pdf", view="diffraction")
    >>> w.save_image("all.pdf", view="all")
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show4dstem.js"

    # Position in scan space
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    pos_row = traitlets.Int(0).tag(sync=True)
    pos_col = traitlets.Int(0).tag(sync=True)

    # Shape of scan space (for slider bounds)
    shape_rows = traitlets.Int(1).tag(sync=True)
    shape_cols = traitlets.Int(1).tag(sync=True)

    # Detector shape for frontend
    det_rows = traitlets.Int(1).tag(sync=True)
    det_cols = traitlets.Int(1).tag(sync=True)

    # Raw float32 frame as bytes (JS handles scale/colormap for real-time interactivity)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Global min/max for DP normalization (computed once from sampled frames)
    dp_global_min = traitlets.Float(0.0).tag(sync=True)
    dp_global_max = traitlets.Float(1.0).tag(sync=True)

    # =========================================================================
    # Detector Calibration (for presets and scale bar)
    # =========================================================================
    center_col = traitlets.Float(0.0).tag(sync=True)  # Detector center col
    center_row = traitlets.Float(0.0).tag(sync=True)  # Detector center row
    bf_radius = traitlets.Float(0.0).tag(sync=True)  # BF disk radius (pixels)

    # =========================================================================
    # ROI Drawing (for virtual imaging)
    # roi_radius is multi-purpose by mode:
    #   - circle: radius of circle
    #   - square: half-size (distance from center to edge)
    #   - annular: outer radius (roi_radius_inner = inner radius)
    #   - rect: uses roi_width/roi_height instead
    # =========================================================================
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_mode = traitlets.Unicode("point").tag(sync=True)
    roi_center_col = traitlets.Float(0.0).tag(sync=True)
    roi_center_row = traitlets.Float(0.0).tag(sync=True)
    # Compound trait for batched row+col updates (JS sends both at once, 1 observer fires)
    roi_center = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
    roi_radius = traitlets.Float(10.0).tag(sync=True)
    roi_radius_inner = traitlets.Float(5.0).tag(sync=True)
    roi_width = traitlets.Float(20.0).tag(sync=True)
    roi_height = traitlets.Float(10.0).tag(sync=True)

    # =========================================================================
    # Virtual Image (ROI-based, updates as you drag ROI on DP)
    # =========================================================================
    virtual_image_bytes = traitlets.Bytes(b"").tag(sync=True)  # Raw float32 (JS computes stats + range)

    # =========================================================================
    # VI ROI (real-space region selection for summed DP)
    # =========================================================================
    vi_roi_mode = traitlets.Unicode("off").tag(sync=True)  # "off", "circle", "rect"
    vi_roi_center_row = traitlets.Float(0.0).tag(sync=True)
    vi_roi_center_col = traitlets.Float(0.0).tag(sync=True)
    # Compound (row, col) trait — JS sets in one call; one observer fires; bytes
    # never compute against split-trait state (old col + new row, or vice versa).
    vi_roi_center = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
    vi_roi_radius = traitlets.Float(5.0).tag(sync=True)
    vi_roi_width = traitlets.Float(10.0).tag(sync=True)
    vi_roi_height = traitlets.Float(10.0).tag(sync=True)
    # Reduction over scan positions inside vi_roi: mean is default (size-invariant DP),
    # sum scales with area (quantitative counts), max picks brightest position per detector pixel.
    vi_roi_reduce = traitlets.Unicode("mean").tag(sync=True)
    vi_roi_dp_bytes = traitlets.Bytes(b"").tag(sync=True)  # Reduced DP from VI ROI

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(1.0).tag(sync=True)  # real-space pixel size (col axis)
    pixel_unit = traitlets.Unicode("pixels").tag(sync=True)
    k_pixel_size = traitlets.Float(1.0).tag(sync=True)  # k-space pixel size (col axis)
    k_pixel_unit = traitlets.Unicode("pixels").tag(sync=True)
    k_calibrated = traitlets.Bool(False).tag(sync=True)  # True if k-space has real units

    # =========================================================================
    # Path Animation (programmatic crosshair control)
    # =========================================================================
    path_playing = traitlets.Bool(False).tag(sync=True)
    path_index = traitlets.Int(0).tag(sync=True)
    path_length = traitlets.Int(0).tag(sync=True)
    path_interval_ms = traitlets.Int(100).tag(sync=True)  # ms between frames
    path_loop = traitlets.Bool(True).tag(sync=True)  # loop when reaching end

    # =========================================================================
    # Auto-detection trigger (frontend sets to True, backend resets to False)
    # =========================================================================

    # =========================================================================
    # Statistics for display (mean, min, max, std)
    # =========================================================================
    # dp_stats and vi_stats are computed JS-side from frame_bytes / virtual_image_bytes.
    # Keeping them out of Python traits eliminates a 4-message comm race that produced
    # mismatched bytes/min/max on rapid preset/ROI changes.

    # =========================================================================
    # Display settings (synced for programmatic export parity)
    # =========================================================================
    dp_colormap = traitlets.Unicode("inferno").tag(sync=True)
    vi_colormap = traitlets.Unicode("inferno").tag(sync=True)
    fft_colormap = traitlets.Unicode("inferno").tag(sync=True)

    dp_scale_mode = traitlets.Unicode("linear").tag(sync=True)  # "linear" | "log"
    vi_scale_mode = traitlets.Unicode("linear").tag(sync=True)  # "linear" | "log"
    fft_scale_mode = traitlets.Unicode("linear").tag(sync=True)  # "linear" | "log"

    dp_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    dp_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    vi_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    vi_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    fft_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    fft_vmax_pct = traitlets.Float(100.0).tag(sync=True)

    # Absolute intensity bounds (override percentile sliders when both set)
    dp_vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    dp_vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    vi_vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vi_vmax = traitlets.Float(None, allow_none=True).tag(sync=True)

    fft_auto = traitlets.Bool(True).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    # Single-trait preset request: JS sets to "bf"/"abf"/"adf"/"haadf" → Python
    # observer calls apply_preset() which batches the 5 ROI trait writes
    # atomically. Avoids the JS-side ordering race where individual roi_mode/
    # radius/center traits would commit in separate comm messages.
    _preset_request = traitlets.Unicode("").tag(sync=True)
    fft_window = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    dp_show_colorbar = traitlets.Bool(False).tag(sync=True)
    # VI panel auto-contrast (1st/99th percentile clip) and CSS smoothing.
    # DP panel doesn't need either — Bragg spots are best read with nearest-
    # neighbor + the slider's percentile range.
    vi_auto_contrast = traitlets.Bool(False).tag(sync=True)
    vi_smooth = traitlets.Bool(False).tag(sync=True)

    # =========================================================================
    # Frame Animation (5D time/tilt series)
    # =========================================================================
    frame_idx = traitlets.Int(0).tag(sync=True)
    n_frames = traitlets.Int(1).tag(sync=True)
    frame_dim_label = traitlets.Unicode("Frame").tag(sync=True)
    frame_labels = traitlets.List(traitlets.Unicode(), []).tag(sync=True)
    frame_playing = traitlets.Bool(False).tag(sync=True)
    frame_loop = traitlets.Bool(True).tag(sync=True)
    frame_fps = traitlets.Float(5.0).tag(sync=True)
    frame_reverse = traitlets.Bool(False).tag(sync=True)
    frame_boomerang = traitlets.Bool(False).tag(sync=True)

    # Export (GIF)
    _gif_export_requested = traitlets.Bool(False).tag(sync=True)
    _gif_data = traitlets.Bytes(b"").tag(sync=True)
    _gif_metadata_json = traitlets.Unicode("").tag(sync=True)

    # Line Profile (for DP panel)
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)
    profile_width = traitlets.Int(1).tag(sync=True)

    # =========================================================================
    def __init__(
        self,
        data: "Dataset4dstem | np.ndarray",
        scan_shape: tuple[int, int] | None = None,
        sampling: tuple[float, ...] | list[float] | None = None,
        units: list[str] | tuple[str, ...] | None = None,
        center: tuple[float, float] | None = None,
        bf_radius: float | None = None,
        precompute_virtual_images: bool = True,
        frame_dim_label: str | None = None,
        frame_labels: list[str] | None = None,
        title: str = "",
        show_fft: bool = False,
        fft_window: bool = True,
        show_controls: bool = True,
        dp_vmin: float | None = None,
        dp_vmax: float | None = None,
        vi_vmin: float | None = None,
        vi_vmax: float | None = None,
        verbose: bool = True,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()
        _t0 = time.perf_counter()
        _verbose = verbose

        _io_labels = None

        # Extract underlying array / tensor + auto-calibrate from Dataset input
        # (duck-typed via the dual-slot private attributes _tensor / _array).
        tensor = getattr(data, "_tensor", None)
        array = getattr(data, "_array", None)
        if tensor is not None or array is not None:
            if not title and getattr(data, "name", ""):
                title = str(data.name)
            if sampling is None:
                sampling = tuple(float(s) for s in data.sampling)
            if units is None:
                units = list(data.units)
            data = tensor if tensor is not None else array

        # Resolve sampling + units (4 axes for 4D-STEM):
        # [scan_row, scan_col, k_row, k_col]. Scalar/None broadcast to (1, 1, 1, 1).
        if sampling is None:
            sampling = (1.0, 1.0, 1.0, 1.0)
        elif isinstance(sampling, (int, float)):
            sampling = (float(sampling),) * 4
        else:
            sampling = tuple(float(s) for s in sampling)
        if units is None:
            units = ["pixels"] * 4
        elif isinstance(units, str):
            units = [units] * 4
        else:
            units = [str(u) for u in units]

        self.title = title
        self.pixel_size = sampling[1]   # scan_col axis (horizontal scale bar)
        self.pixel_unit = units[1] if len(units) > 1 else "pixels"
        self.k_pixel_size = sampling[3] if len(sampling) > 3 else 1.0
        self.k_pixel_unit = units[3] if len(units) > 3 else "pixels"
        # k-space considered calibrated when its unit is real (mrad, 1/Å, etc.).
        self.k_calibrated = self.k_pixel_unit not in ("pixels", "")
        self.show_fft = show_fft
        self.fft_window = fft_window
        self.show_controls = show_controls
        self.dp_vmin = dp_vmin
        self.dp_vmax = dp_vmax
        self.vi_vmin = vi_vmin
        self.vi_vmax = vi_vmax
        # Path animation (configured via set_path() or raster())
        self._path_points: list[tuple[int, int]] = []
        # Suppress per-trait recompute during apply_preset batch writes
        self._suppress_roi_recompute = False
        # Torch tensor input keeps its device (lets user pin a specific GPU via
        # `data.cuda(1)`). NumPy / Dataset input gets default-validated device.
        if isinstance(data, torch.Tensor) or getattr(data, "_is_gpu_frames", False):
            # `_is_gpu_frames` lets a duck-typed GPU array (e.g. a chunk-backed
            # no-bin stack that can't be one tensor) take the GPU path without a
            # numpy round-trip. It must expose .shape/.dtype/.ndim/.device and
            # single-frame integer indexing.
            self._device = data.device
            self._data_pre = data
            data_np = None
        else:
            device_str, _ = validate_device(None)
            self._device = torch.device(device_str)
            data_np = to_numpy(data)
            self._data_pre = None
            self._saturation_value = (
                65535 if data_np.dtype == np.uint16
                else 255 if data_np.dtype == np.uint8
                else None
            )
        # Handle dimensionality — 5D loads eagerly for instant frame switching
        # Resolve shape from whichever input path we took
        shape = tuple(self._data_pre.shape) if self._data_pre is not None else data_np.shape
        ndim = len(shape)
        _tc = time.perf_counter()
        if ndim == 5:
            self.n_frames = shape[0]
            self._scan_shape = (shape[1], shape[2])
            self._det_shape = (shape[3], shape[4])
        elif ndim == 3:
            self.n_frames = 1
            if scan_shape is not None:
                self._scan_shape = scan_shape
            else:
                n = shape[0]
                side = int(n ** 0.5)
                if side * side != n:
                    raise ValueError(
                        f"Cannot infer square scan_shape from N={n}. "
                        f"Provide scan_shape explicitly."
                    )
                self._scan_shape = (side, side)
            self._det_shape = (shape[1], shape[2])
        elif ndim == 4:
            self.n_frames = 1
            self._scan_shape = (shape[0], shape[1])
            self._det_shape = (shape[2], shape[3])
        else:
            raise ValueError(f"Show4DSTEM expects a 3D ((N, det_h, det_w) flat-scan), 4D ((scan_h, scan_w, det_h, det_w)), or 5D ((n_frames, scan_h, scan_w, det_h, det_w)) array. Got {ndim}D.")
        if self._data_pre is not None:
            self._data = self._data_pre if self._data_pre.device == self._device else self._data_pre.to(self._device)
            del self._data_pre
        else:
            self._data = torch.from_numpy(data_np).to(self._device)
            # Saturation filter: zero detector pixels at full-scale (65535 / 255).
            # PyTorch lacks unsigned int comparison kernels, but uint16 viewed
            # as int16 has identical bytes (65535 → -1) and int16 comparison
            # works on every device. Apply in scan-row chunks so the transient
            # bool mask stays bounded (≤600 MB) and fits constrained-VRAM
            # devices (Mac 24 GB unified, etc.). View-write keeps native dtype.
            sat = getattr(self, "_saturation_value", None)
            view_dtype = (
                torch.int16 if sat is not None and self._data.dtype == torch.uint16
                else torch.int8 if sat is not None and self._data.dtype == torch.uint8
                else None
            )
            if view_dtype is not None:
                view = self._data.view(view_dtype).reshape(-1, *self._det_shape)
                rows = view.shape[0]
                # Bool mask transient = positions × det_h × det_w bytes; cap at budget.
                pos_per_chunk = max(1, _CHUNK_BYTE_BUDGET // max(1, self._det_shape[0] * self._det_shape[1]))
                for i in range(0, rows, pos_per_chunk):
                    chunk = view[i:i + pos_per_chunk]
                    chunk.masked_fill_(chunk == -1, 0)
        # Keep native dtype (uint8/uint16) to bound memory at ~ data_size.
        # Reductions cast in chunks (bounded transient).
        if _verbose:
            if str(self._device) == "mps":
                torch.mps.synchronize()
            n_bytes = self._data.element_size() * self._data.numel()
            print(f"  to {self._device}: {time.perf_counter() - _tc:.2f}s ({n_bytes / 1e9:.1f} GB)")

        self.shape_rows = self._scan_shape[0]
        self.shape_cols = self._scan_shape[1]
        self.det_rows = self._det_shape[0]
        self.det_cols = self._det_shape[1]
        # Initial position at center
        self.pos_row = self.shape_rows // 2
        self.pos_col = self.shape_cols // 2
        # Frame dimension label (for 5D time/tilt series UI)
        self.frame_dim_label = frame_dim_label if frame_dim_label is not None else "Frame"
        # Per-frame labels: explicit param > inferred > empty
        resolved_labels = frame_labels or _io_labels or []
        self._frame_labels = resolved_labels
        if resolved_labels:
            self.frame_labels = list(resolved_labels)
        # Histogram axis range — first frame is enough (JS does per-frame percentile clipping).
        # Cast to float for min/max reductions: PyTorch CUDA lacks integer min/max kernels,
        # and the first slice is tiny (144 KB at 192×192) so the cast is free.
        first_frame = self._data[0] if self._data.ndim == 5 else self._data
        first_frame_sample = first_frame[0] if first_frame.ndim >= 3 else first_frame
        if not torch.is_floating_point(first_frame_sample):
            first_frame_sample = first_frame_sample.float()
        self.dp_global_min = max(float(first_frame_sample.min()), MIN_LOG_VALUE)
        self.dp_global_max = float(first_frame_sample.max())
        # Cache coordinate tensors for mask creation (avoid repeated torch.arange)
        self._det_row_coords = torch.arange(self.det_rows, device=self._device, dtype=torch.float32)[:, None]
        self._det_col_coords = torch.arange(self.det_cols, device=self._device, dtype=torch.float32)[None, :]
        self._scan_row_coords = torch.arange(self.shape_rows, device=self._device, dtype=torch.float32)[:, None]
        self._scan_col_coords = torch.arange(self.shape_cols, device=self._device, dtype=torch.float32)[None, :]
        # Setup center and BF radius
        det_size = min(self.det_rows, self.det_cols)
        if center is not None and bf_radius is not None:
            self.center_row = float(center[0])
            self.center_col = float(center[1])
            self.bf_radius = float(bf_radius)
        elif center is not None:
            self.center_row = float(center[0])
            self.center_col = float(center[1])
            self.bf_radius = det_size * DEFAULT_BF_RATIO
        elif bf_radius is not None:
            self.center_col = float(self.det_cols / 2)
            self.center_row = float(self.det_rows / 2)
            self.bf_radius = float(bf_radius)
        else:
            # Neither provided - auto-detect from data
            # Set defaults first (will be overwritten by auto-detect)
            self.center_col = float(self.det_cols / 2)
            self.center_row = float(self.det_rows / 2)
            self.bf_radius = det_size * DEFAULT_BF_RATIO
            # Auto-detect center and bf_radius from the data
            _tc = time.perf_counter()
            self.auto_detect_center(update_roi=False)
            if _verbose:
                print(f"  auto_detect_center: {time.perf_counter() - _tc:.2f}s")

        # Pre-compute and cache common virtual images (BF, ABF, ADF)
        # Each cache stores (bytes, stats) tuple
        self._cached_bf_virtual = None
        self._cached_abf_virtual = None
        self._cached_adf_virtual = None
        self._cached_haadf_virtual = None
        if precompute_virtual_images and self.n_frames == 1:
            self._precompute_common_virtual_images()

        # Update frame when position changes (scale/colormap handled in JS)
        self.observe(self._update_frame, names=["pos_row", "pos_col"])
        # Observe individual ROI params
        self.observe(self._on_roi_change, names=[
            "roi_center_col", "roi_center_row", "roi_radius", "roi_radius_inner",
            "roi_active", "roi_mode", "roi_width", "roi_height"
        ])
        # Observe compound roi_center for batched updates from JS
        self.observe(self._on_roi_center_change, names=["roi_center"])
        # Invalidate precomputed virtual image caches when calibration changes
        self.observe(self._on_calibration_change, names=["center_row", "center_col", "bf_radius"])

        # Initialize default ROI at BF center — batch to avoid redundant observer callbacks
        with self.hold_trait_notifications():
            self.roi_center_col = self.center_col
            self.roi_center_row = self.center_row
            self.roi_center = [self.center_row, self.center_col]
            self.roi_radius = self.bf_radius * 0.5  # Start with half BF radius
            self.roi_active = True

        # Compute initial virtual image and frame (once, after all ROI traits are set)
        _tc = time.perf_counter()
        self._compute_virtual_image_from_roi()
        self._update_frame()
        if _verbose:
            print(f"  virtual image + frame: {time.perf_counter() - _tc:.2f}s")
        
        # Path animation: observe index changes from frontend
        self.observe(self._on_path_index_change, names=["path_index"])
        self.observe(self._on_gif_export, names=["_gif_export_requested"])

        # Frame animation (5D): observe frame_idx changes from frontend
        self.observe(self._on_frame_idx_change, names=["frame_idx"])
        self.observe(self._on_preset_request, names=["_preset_request"])

        # Auto-detect trigger: observe changes from frontend

        # VI ROI: observe changes for summed DP computation
        # Initialize VI ROI center to scan center with reasonable default sizes
        self.vi_roi_center_row = float(self.shape_rows / 2)
        self.vi_roi_center_col = float(self.shape_cols / 2)
        # Set initial ROI size based on scan dimension
        default_roi_size = max(3, min(self.shape_rows, self.shape_cols) * DEFAULT_VI_ROI_RATIO)
        self.vi_roi_radius = float(default_roi_size)
        self.vi_roi_width = float(default_roi_size * 2)
        self.vi_roi_height = float(default_roi_size)
        self.observe(self._on_vi_roi_change, names=[
            "vi_roi_mode", "vi_roi_center_row", "vi_roi_center_col",
            "vi_roi_radius", "vi_roi_width", "vi_roi_height", "vi_roi_reduce"
        ])
        self.observe(self._on_vi_roi_center_change, names=["vi_roi_center"])

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

        if _verbose:
            shape = "x".join(str(s) for s in self._data.shape)
            print(f"Show4DSTEM: {shape} {self._device}, {time.perf_counter() - _t0:.2f}s total")

    def __repr__(self) -> str:
        shape = (
            f"({self.n_frames}, {self.shape_rows}, {self.shape_cols}, {self.det_rows}, {self.det_cols})"
            if self.n_frames > 1
            else f"({self.shape_rows}, {self.shape_cols}, {self.det_rows}, {self.det_cols})"
        )
        frame_info = f", {self.frame_dim_label.lower()}={self.frame_idx}" if self.n_frames > 1 else ""
        title_info = f", title='{self.title}'" if self.title else ""
        return (
            f"Show4DSTEM(shape={shape}, "
            f"sampling=({self.pixel_size} {self.pixel_unit}, {self.k_pixel_size} {self.k_pixel_unit}), "
            f"pos=({self.pos_row}, {self.pos_col}){frame_info}{title_info})"
        )

    def state_dict(self):
        return {
            "title": self.title,
            "pos_row": self.pos_row,
            "pos_col": self.pos_col,
            "pixel_size": self.pixel_size,
            "pixel_unit": self.pixel_unit,
            "k_pixel_size": self.k_pixel_size,
            "k_pixel_unit": self.k_pixel_unit,
            "k_calibrated": self.k_calibrated,
            "center_row": self.center_row,
            "center_col": self.center_col,
            "bf_radius": self.bf_radius,
            "roi_active": self.roi_active,
            "roi_mode": self.roi_mode,
            "roi_center_row": self.roi_center_row,
            "roi_center_col": self.roi_center_col,
            "roi_radius": self.roi_radius,
            "roi_radius_inner": self.roi_radius_inner,
            "roi_width": self.roi_width,
            "roi_height": self.roi_height,
            "vi_roi_mode": self.vi_roi_mode,
            "vi_roi_center_row": self.vi_roi_center_row,
            "vi_roi_center_col": self.vi_roi_center_col,
            "vi_roi_radius": self.vi_roi_radius,
            "vi_roi_width": self.vi_roi_width,
            "vi_roi_height": self.vi_roi_height,
            "vi_roi_reduce": self.vi_roi_reduce,
            "dp_colormap": self.dp_colormap,
            "vi_colormap": self.vi_colormap,
            "fft_colormap": self.fft_colormap,
            "dp_scale_mode": self.dp_scale_mode,
            "vi_scale_mode": self.vi_scale_mode,
            "fft_scale_mode": self.fft_scale_mode,
            "dp_vmin_pct": self.dp_vmin_pct,
            "dp_vmax_pct": self.dp_vmax_pct,
            "vi_vmin_pct": self.vi_vmin_pct,
            "vi_vmax_pct": self.vi_vmax_pct,
            "fft_vmin_pct": self.fft_vmin_pct,
            "fft_vmax_pct": self.fft_vmax_pct,
            "dp_vmin": self.dp_vmin,
            "dp_vmax": self.dp_vmax,
            "vi_vmin": self.vi_vmin,
            "vi_vmax": self.vi_vmax,
            "fft_auto": self.fft_auto,
            "show_fft": self.show_fft,
            "fft_window": self.fft_window,
            "show_controls": self.show_controls,
            "dp_show_colorbar": self.dp_show_colorbar,
            "vi_auto_contrast": self.vi_auto_contrast,
            "vi_smooth": self.vi_smooth,
            "path_interval_ms": self.path_interval_ms,
            "path_loop": self.path_loop,
            "profile_line": self.profile_line,
            "profile_width": self.profile_width,
            "frame_idx": self.frame_idx,
            "frame_dim_label": self.frame_dim_label,
            "frame_labels": list(self.frame_labels),
            "frame_loop": self.frame_loop,
            "frame_fps": self.frame_fps,
            "frame_reverse": self.frame_reverse,
            "frame_boomerang": self.frame_boomerang,
        }

    def save(self, path: str):
        save_state_file(path, "Show4DSTEM", self.state_dict())

    def load_state_dict(self, state):
        allowed_keys = set(self.state_dict().keys())
        pending_pos_row = state.get("pos_row", None)
        pending_pos_col = state.get("pos_col", None)
        pending_frame_idx = state.get("frame_idx", None)
        for key, val in state.items():
            if key in {"pos_row", "pos_col", "frame_idx"}:
                continue
            if key in allowed_keys:
                setattr(self, key, val)
        if pending_frame_idx is not None:
            self.frame_idx = int(max(0, min(int(pending_frame_idx), self.n_frames - 1)))
        if pending_pos_row is not None or pending_pos_col is not None:
            row = int(self.pos_row if pending_pos_row is None else pending_pos_row)
            col = int(self.pos_col if pending_pos_col is None else pending_pos_col)
            self.pos_row = int(max(0, min(row, self.shape_rows - 1)))
            self.pos_col = int(max(0, min(col, self.shape_cols - 1)))

    def free(self):
        """Free GPU memory held by this widget.

        Deletes the internal data tensor, runs garbage collection, and
        flushes the MPS allocator cache. Call this before loading a new
        dataset to avoid running out of GPU memory.

        Examples
        --------
        >>> w.free()          # release ~9 GB of MPS memory
        >>> del result        # free the source numpy array
        """
        import gc

        device = str(self._device) if hasattr(self, "_device") else ""
        nbytes = self._data.nbytes if hasattr(self._data, "nbytes") else 0
        self._data = None
        gc.collect()
        if device == "mps":
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass
        elif device.startswith("cuda"):
            torch.cuda.empty_cache()
        if nbytes > 0:
            print(f"freed {_format_memory(nbytes)} ({device})")

    def summary(self):
        name = self.title if self.title else "Show4DSTEM"
        lines = [name, "═" * 32]
        if self.n_frames > 1:
            parts = [f"{self.n_frames} ({self.frame_dim_label}), current: {self.frame_idx}"]
            parts.append(f"{self.frame_fps} fps")
            if self.frame_loop:
                parts.append("loop")
            if self.frame_reverse:
                parts.append("reverse")
            if self.frame_boomerang:
                parts.append("bounce")
            lines.append(f"Frames:   {' | '.join(parts)}")
            if self._frame_labels:
                if len(self._frame_labels) <= 4:
                    lines.append(f"Labels:   {self._frame_labels}")
                else:
                    lines.append(f"Labels:   {self._frame_labels[:3]} ... ({len(self._frame_labels)} total)")
        lines.append(f"Scan:     {self.shape_rows}×{self.shape_cols} ({self.pixel_size:.2f} {self.pixel_unit}/px)")
        lines.append(f"Detector: {self.det_rows}×{self.det_cols} ({self.k_pixel_size:.4f} {self.k_pixel_unit}/px)")
        lines.append(f"Position: ({self.pos_row}, {self.pos_col})")
        lines.append(f"Center:   ({self.center_row:.1f}, {self.center_col:.1f})  BF r={self.bf_radius:.1f} px")
        if self.roi_active:
            lines.append(f"ROI:      {self.roi_mode} at ({self.roi_center_row:.1f}, {self.roi_center_col:.1f}) r={self.roi_radius:.1f}")
        if self.vi_roi_mode != "off":
            lines.append(f"VI ROI:   {self.vi_roi_mode} at ({self.vi_roi_center_row:.1f}, {self.vi_roi_center_col:.1f}) r={self.vi_roi_radius:.1f}")
        dp_contrast = f"{self.dp_vmin_pct:.1f}-{self.dp_vmax_pct:.1f}%"
        if self.dp_vmin is not None and self.dp_vmax is not None:
            dp_contrast += f", dp_vmin={self.dp_vmin:.4g}, dp_vmax={self.dp_vmax:.4g}"
        lines.append(
            f"DP view:  {self.dp_colormap}, {self.dp_scale_mode}, {dp_contrast}"
        )
        vi_contrast = f"{self.vi_vmin_pct:.1f}-{self.vi_vmax_pct:.1f}%"
        if self.vi_vmin is not None and self.vi_vmax is not None:
            vi_contrast += f", vi_vmin={self.vi_vmin:.4g}, vi_vmax={self.vi_vmax:.4g}"
        lines.append(
            f"VI view:  {self.vi_colormap}, {self.vi_scale_mode}, {vi_contrast}"
        )
        if self.show_fft:
            fft_parts = [f"{self.fft_colormap}, {self.fft_scale_mode}, {self.fft_vmin_pct:.1f}-{self.fft_vmax_pct:.1f}%, auto={self.fft_auto}"]
            if not self.fft_window:
                fft_parts.append("no window")
            lines.append(f"FFT view: {', '.join(fft_parts)}")
        if self.profile_line and len(self.profile_line) == 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            lines.append(f"Profile:  ({p0['row']:.0f}, {p0['col']:.0f}) -> ({p1['row']:.0f}, {p1['col']:.0f}) width={self.profile_width}")
        print("\n".join(lines))

    # =========================================================================
    # Convenience Properties
    # =========================================================================

    @property
    def position(self) -> tuple[int, int]:
        """Current scan position as (row, col) tuple."""
        return (self.pos_row, self.pos_col)

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        """Set scan position from (row, col) tuple."""
        self.pos_row, self.pos_col = value

    @property
    def scan_shape(self) -> tuple[int, int]:
        """Scan dimensions as (rows, cols) tuple."""
        return (self.shape_rows, self.shape_cols)

    @property
    def detector_shape(self) -> tuple[int, int]:
        """Detector dimensions as (rows, cols) tuple."""
        return (self.det_rows, self.det_cols)

    @property
    def _frame_data(self) -> torch.Tensor:
        """Per-frame data (4D or 3D flattened), accounting for 5D time/tilt series."""
        if self.n_frames > 1:
            return self._data[self.frame_idx]
        return self._data

    # =========================================================================
    # Line Profile
    # =========================================================================

    def set_profile(self, start: tuple, end: tuple) -> Self:
        row0, col0 = start
        row1, col1 = end
        self.profile_line = [
            {"row": float(row0), "col": float(col0)},
            {"row": float(row1), "col": float(col1)},
        ]
        return self

    def clear_profile(self) -> Self:
        self.profile_line = []
        return self

    @property
    def profile(self) -> list[tuple[float, float]]:
        if len(self.profile_line) == 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            return [(p0["row"], p0["col"]), (p1["row"], p1["col"])]
        return []

    @property
    def profile_values(self):
        if len(self.profile_line) != 2:
            return None
        p0, p1 = self.profile_line[0], self.profile_line[1]
        frame = self._get_frame(self.pos_row, self.pos_col)
        return self._sample_line(frame, p0["row"], p0["col"], p1["row"], p1["col"])

    @property
    def profile_distance(self) -> float:
        if len(self.profile_line) != 2:
            return 0.0
        p0, p1 = self.profile_line[0], self.profile_line[1]
        dist_px = np.sqrt((p1["row"] - p0["row"]) ** 2 + (p1["col"] - p0["col"]) ** 2)
        if self.k_calibrated:
            return float(dist_px * self.k_pixel_size)
        return float(dist_px)

    def _sample_line(self, frame, row0, col0, row1, col1):
        h, w = frame.shape[:2]
        dc = col1 - col0
        dr = row1 - row0
        length = np.sqrt(dc * dc + dr * dr)
        n = max(2, int(np.ceil(length)))
        t = np.linspace(0.0, 1.0, n)
        c = col0 + t * dc
        r = row0 + t * dr
        ci = np.floor(c).astype(np.intp)
        ri = np.floor(r).astype(np.intp)
        cf = c - ci
        rf = r - ri
        c0 = np.clip(ci, 0, w - 1)
        c1 = np.clip(ci + 1, 0, w - 1)
        r0 = np.clip(ri, 0, h - 1)
        r1 = np.clip(ri + 1, 0, h - 1)
        return (
            frame[r0, c0] * (1 - cf) * (1 - rf)
            + frame[r0, c1] * cf * (1 - rf)
            + frame[r1, c0] * (1 - cf) * rf
            + frame[r1, c1] * cf * rf
        ).astype(np.float32)

    # =========================================================================
    # Path Animation Methods
    # =========================================================================
    
    def set_path(
        self,
        points: list[tuple[int, int]],
        interval_ms: int = 100,
        loop: bool = True,
        autoplay: bool = True,
    ) -> Self:
        """
        Set a custom path of scan positions to animate through.

        Parameters
        ----------
        points : list[tuple[int, int]]
            List of (row, col) scan positions to visit.
        interval_ms : int, default 100
            Time between frames in milliseconds.
        loop : bool, default True
            Whether to loop when reaching end.
        autoplay : bool, default True
            Start playing immediately.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.

        Examples
        --------
        >>> widget.set_path([(0, 0), (10, 10), (20, 20), (30, 30)])
        >>> widget.set_path([(i, i) for i in range(48)], interval_ms=50)
        """
        self._path_points = list(points)
        self.path_length = len(self._path_points)
        self.path_index = 0
        self.path_interval_ms = interval_ms
        self.path_loop = loop
        if autoplay and self.path_length > 0:
            self.path_playing = True
        return self
    
    def play(self) -> Self:
        """Start playing the path animation."""
        if self.path_length > 0:
            self.path_playing = True
        return self
    
    def pause(self) -> Self:
        """Pause the path animation."""
        self.path_playing = False
        return self
    
    def stop(self) -> Self:
        """Stop and reset path animation to beginning."""
        self.path_playing = False
        self.path_index = 0
        return self
    
    def goto(self, index: int) -> Self:
        """Jump to a specific index in the path."""
        if 0 <= index < self.path_length:
            self.path_index = index
        return self
    
    def _on_path_index_change(self, change):
        """Called when path_index changes (from frontend timer)."""
        idx = change["new"]
        if 0 <= idx < len(self._path_points):
            row, col = self._path_points[idx]
            # Clamp to valid range
            self.pos_row = max(0, min(self.shape_rows - 1, row))
            self.pos_col = max(0, min(self.shape_cols - 1, col))

    def _on_preset_request(self, change):
        """JS preset shortcut → atomic apply_preset (no per-trait race)."""
        name = (change.get("new") or "").strip().lower()
        if name in ("bf", "abf", "adf", "haadf"):
            self.apply_preset(name)
            self._preset_request = ""  # consume trigger

    def _on_frame_idx_change(self, change=None):
        """Called when frame_idx changes (5D time/tilt series).

        Recomputes virtual image and diffraction pattern for the new frame.
        Invalidates precomputed caches since they are per-frame.
        """
        if self.n_frames <= 1:
            return
        # Invalidate precomputed caches (they were for a different frame)
        self._cached_bf_virtual = None
        self._cached_abf_virtual = None
        self._cached_adf_virtual = None
        self._cached_haadf_virtual = None
        # Recompute virtual image and displayed frame
        self._compute_virtual_image_from_roi()
        self._update_frame()
        # Recompute reduced DP if VI ROI is active
        if self.vi_roi_mode != "off":
            self._compute_vi_roi_dp()

    # =========================================================================
    # Path Animation Patterns
    # =========================================================================

    def raster(
        self,
        step: int = 1,
        bidirectional: bool = False,
        interval_ms: int = 100,
        loop: bool = True,
    ) -> Self:
        """
        Play a raster scan path (row by row, left to right).

        This mimics real STEM scanning: left→right, step down, left→right, etc.

        Parameters
        ----------
        step : int, default 1
            Step size between positions.
        bidirectional : bool, default False
            If True, use snake/boustrophedon pattern (alternating direction).
            If False (default), always scan left→right like real STEM.
        interval_ms : int, default 100
            Time between frames in milliseconds.
        loop : bool, default True
            Whether to loop when reaching the end.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.
        """
        points = []
        for r in range(0, self.shape_rows, step):
            cols = list(range(0, self.shape_cols, step))
            if bidirectional and (r // step % 2 == 1):
                cols = cols[::-1]  # Alternate direction for snake pattern
            for c in cols:
                points.append((r, c))
        return self.set_path(points=points, interval_ms=interval_ms, loop=loop)
    
    # =========================================================================
    # ROI Mode Methods
    # =========================================================================
    
    def roi_circle(self, radius: float | None = None) -> Self:
        """
        Switch to circle ROI mode for virtual imaging.
        
        In circle mode, the virtual image integrates over a circular region
        centered at the current ROI position (like a virtual bright field detector).
        
        Parameters
        ----------
        radius : float, optional
            Radius of the circle in pixels. If not provided, uses current value
            or defaults to half the BF radius.
            
        Returns
        -------
        Show4DSTEM
            Self for method chaining.
            
        Examples
        --------
        >>> widget.roi_circle(20)  # 20px radius circle
        >>> widget.roi_circle()    # Use default radius
        """
        self.roi_mode = "circle"
        if radius is not None:
            self.roi_radius = float(radius)
        return self
    
    def roi_point(self) -> Self:
        """
        Switch to point ROI mode (single-pixel indexing).
        
        In point mode, the virtual image shows intensity at the exact ROI position.
        This is the default mode.
        
        Returns
        -------
        Show4DSTEM
            Self for method chaining.
        """
        self.roi_mode = "point"
        return self

    def roi_square(self, half_size: float | None = None) -> Self:
        """
        Switch to square ROI mode for virtual imaging.

        In square mode, the virtual image integrates over a square region
        centered at the current ROI position.

        Parameters
        ----------
        half_size : float, optional
            Half-size of the square in pixels (distance from center to edge).
            A half_size of 15 creates a 30x30 pixel square.
            If not provided, uses current roi_radius value.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.

        Examples
        --------
        >>> widget.roi_square(15)  # 30x30 pixel square (half_size=15)
        >>> widget.roi_square()    # Use default size
        """
        self.roi_mode = "square"
        if half_size is not None:
            self.roi_radius = float(half_size)
        return self

    def roi_annular(
        self, inner_radius: float | None = None, outer_radius: float | None = None
    ) -> Self:
        """
        Set ROI mode to annular (donut-shaped) for ADF/HAADF imaging.
        
        Parameters
        ----------
        inner_radius : float, optional
            Inner radius in pixels. If not provided, uses current roi_radius_inner.
        outer_radius : float, optional
            Outer radius in pixels. If not provided, uses current roi_radius.
            
        Returns
        -------
        Show4DSTEM
            Self for method chaining.
            
        Examples
        --------
        >>> widget.roi_annular(20, 50)  # ADF: inner=20px, outer=50px
        >>> widget.roi_annular(30, 80)  # HAADF: larger angles
        """
        self.roi_mode = "annular"
        if inner_radius is not None:
            self.roi_radius_inner = float(inner_radius)
        if outer_radius is not None:
            self.roi_radius = float(outer_radius)
        return self

    def roi_rect(
        self, width: float | None = None, height: float | None = None
    ) -> Self:
        """
        Set ROI mode to rectangular.
        
        Parameters
        ----------
        width : float, optional
            Width in pixels. If not provided, uses current roi_width.
        height : float, optional
            Height in pixels. If not provided, uses current roi_height.
            
        Returns
        -------
        Show4DSTEM
            Self for method chaining.
            
        Examples
        --------
        >>> widget.roi_rect(30, 20)  # 30px wide, 20px tall
        >>> widget.roi_rect(40, 40)  # 40x40 rectangle
        """
        self.roi_mode = "rect"
        if width is not None:
            self.roi_width = float(width)
        if height is not None:
            self.roi_height = float(height)
        return self

    def auto_detect_center(self, update_roi: bool = True) -> Self:
        """
        Automatically detect BF disk center and radius using centroid.

        This method analyzes the summed diffraction pattern to find the
        bright field disk center and estimate its radius. The detected
        values are applied to the widget's calibration (center_row, center_col,
        bf_radius).

        Parameters
        ----------
        update_roi : bool, default True
            If True, also update ROI center and recompute cached virtual images.
            Set to False during __init__ when ROI is not yet initialized.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.

        Examples
        --------
        >>> widget = Show4DSTEM(data)
        >>> widget.auto_detect_center()  # Auto-detect and apply
        """
        # Sum diffraction patterns over scan positions to find BF disk centroid.
        # Chunked torch INTEGER path: works identically on CUDA / MPS / CPU.
        # int64 accumulator, no float32 cast of the stack — keeps the data in its
        # native dtype (a float32 cast doubles memory and is lossy above 2^24),
        # is bit-exact, and avoids the MPS "tensor dims larger than INT_MAX"
        # error a single full-stack reduce hits once positions*det > 2^31 (a bin2
        # 512x512x96x96 stack = 2.42e9 elements). The chunk cap keeps each op's
        # element count well under 2^31; the (det, det) accumulator is tiny.
        data_flat = self._data.reshape(-1, *self._det_shape)
        n_pos = data_flat.shape[0]
        mean_dp = torch.zeros(self._det_shape, dtype=torch.int64, device=self._device)
        pos_per_chunk = max(1, (1 << 30) // max(1, self._det_shape[0] * self._det_shape[1]))
        for i in range(0, n_pos, pos_per_chunk):
            mean_dp += data_flat[i:i + pos_per_chunk].sum(dim=0, dtype=torch.int64)

        # float only on the tiny (det, det) summed image, never on the full stack
        mean_dp = mean_dp.float()
        threshold = mean_dp.mean() + mean_dp.std()
        mask = mean_dp > threshold

        total = mask.sum()
        if total == 0:
            return self

        cx = float((self._det_col_coords * mask).sum() / total)
        cy = float((self._det_row_coords * mask).sum() / total)
        radius = float(torch.sqrt(total / torch.pi))

        # Apply detected values
        self.center_col = cx
        self.center_row = cy
        self.bf_radius = radius

        if update_roi:
            # Also update ROI to center
            self.roi_center_col = cx
            self.roi_center_row = cy
            # Recompute cached virtual images with new calibration
            self._precompute_common_virtual_images()

        return self

    def _get_frame(self, row: int, col: int) -> np.ndarray:
        """Get single diffraction frame at position (row, col) as numpy array."""
        if self._data is None:
            return np.zeros((self.det_rows, self.det_cols), dtype=np.float32)
        data = self._frame_data
        if data.ndim == 3:
            idx = row * self.shape_cols + col
            return data[idx].cpu().numpy()
        else:
            return data[row, col].cpu().numpy()

    def _apply_scale_mode(self, data: np.ndarray, mode: str) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float32)
        if mode == "log":
            return np.log1p(np.maximum(arr, 0.0)).astype(np.float32)
        return arr.astype(np.float32)

    def _slider_range(
        self,
        data_min: float,
        data_max: float,
        vmin_pct: float,
        vmax_pct: float,
    ) -> tuple[float, float]:
        v0 = float(max(0.0, min(100.0, vmin_pct)))
        v1 = float(max(0.0, min(100.0, vmax_pct)))
        if v1 < v0:
            v0, v1 = v1, v0
        rng = float(data_max - data_min)
        return (
            float(data_min + (v0 / 100.0) * rng),
            float(data_min + (v1 / 100.0) * rng),
        )

    def _render_colormap_rgb(
        self,
        data: np.ndarray,
        cmap_name: str,
        vmin: float,
        vmax: float,
    ) -> np.ndarray:
        from matplotlib import colormaps

        arr = np.asarray(data, dtype=np.float32)
        if vmax <= vmin:
            normalized = np.zeros_like(arr, dtype=np.float32)
        else:
            normalized = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        rgba = colormaps.get_cmap(cmap_name)(normalized)
        return (rgba[..., :3] * 255).astype(np.uint8)

    def _get_virtual_image_array(self) -> np.ndarray:
        if not self.virtual_image_bytes:
            return np.zeros((self.shape_rows, self.shape_cols), dtype=np.float32)
        arr = np.frombuffer(self.virtual_image_bytes, dtype=np.float32)
        expected = self.shape_rows * self.shape_cols
        if arr.size != expected:
            return np.zeros((self.shape_rows, self.shape_cols), dtype=np.float32)
        return arr.reshape(self.shape_rows, self.shape_cols).copy()

    def _get_vi_roi_dp_array(self) -> np.ndarray | None:
        if self.vi_roi_mode == "off":
            return None
        self._compute_vi_roi_dp()
        if not self.vi_roi_dp_bytes:
            return None
        arr = np.frombuffer(self.vi_roi_dp_bytes, dtype=np.float32)
        expected = self.det_rows * self.det_cols
        if arr.size != expected:
            return None
        return arr.reshape(self.det_rows, self.det_cols).copy()

    def _fft_enhanced_range(self, mag: np.ndarray) -> tuple[float, float]:
        arr = np.asarray(mag, dtype=np.float32).copy()
        if arr.size == 0:
            return 0.0, 0.0
        center_row = arr.shape[0] // 2
        center_col = arr.shape[1] // 2
        neighbors = []
        if center_col - 1 >= 0:
            neighbors.append(arr[center_row, center_col - 1])
        if center_col + 1 < arr.shape[1]:
            neighbors.append(arr[center_row, center_col + 1])
        if center_row - 1 >= 0:
            neighbors.append(arr[center_row - 1, center_col])
        if center_row + 1 < arr.shape[0]:
            neighbors.append(arr[center_row + 1, center_col])
        if neighbors:
            arr[center_row, center_col] = float(np.mean(neighbors))
        dmin = float(arr.min())
        dmax = float(arr.max())
        if dmax <= dmin:
            return dmin, dmax
        pmax = float(np.percentile(arr, 99.9))
        if pmax <= dmin:
            pmax = dmax
        return dmin, pmax

    def _render_dp_rgb(self) -> tuple[np.ndarray, dict]:
        vi_roi_arr = self._get_vi_roi_dp_array()
        if vi_roi_arr is not None:
            raw = vi_roi_arr
            source = "vi_roi_dp"
        else:
            raw = self._get_frame(self.pos_row, self.pos_col).astype(np.float32)
            source = "single_frame"

        scale_mode = self.dp_scale_mode
        scaled = self._apply_scale_mode(raw, scale_mode)
        data_min = float(scaled.min()) if scaled.size else 0.0
        data_max = float(scaled.max()) if scaled.size else 0.0
        if self.dp_vmin is not None and self.dp_vmax is not None:
            vmin = float(self._apply_scale_mode(
                np.array([max(self.dp_vmin, 0)], dtype=np.float32), scale_mode
            )[0])
            vmax = float(self._apply_scale_mode(
                np.array([max(self.dp_vmax, 0)], dtype=np.float32), scale_mode
            )[0])
        else:
            vmin, vmax = self._slider_range(data_min, data_max, self.dp_vmin_pct, self.dp_vmax_pct)
        rgb = self._render_colormap_rgb(scaled, self.dp_colormap, vmin, vmax)
        metadata = {
            "source": source,
            "colormap": self.dp_colormap,
            "scale_mode": scale_mode,
            "vmin_pct": float(self.dp_vmin_pct),
            "vmax_pct": float(self.dp_vmax_pct),
            "vmin": float(vmin),
            "vmax": float(vmax),
        }
        return rgb, metadata

    def _render_virtual_rgb(self) -> tuple[np.ndarray, dict]:
        raw = self._get_virtual_image_array()
        scaled = self._apply_scale_mode(raw, self.vi_scale_mode)
        data_min = float(scaled.min()) if scaled.size else 0.0
        data_max = float(scaled.max()) if scaled.size else 0.0
        if self.vi_vmin is not None and self.vi_vmax is not None:
            vmin = float(self._apply_scale_mode(
                np.array([max(self.vi_vmin, 0)], dtype=np.float32), self.vi_scale_mode
            )[0])
            vmax = float(self._apply_scale_mode(
                np.array([max(self.vi_vmax, 0)], dtype=np.float32), self.vi_scale_mode
            )[0])
        else:
            vmin, vmax = self._slider_range(data_min, data_max, self.vi_vmin_pct, self.vi_vmax_pct)
        rgb = self._render_colormap_rgb(scaled, self.vi_colormap, vmin, vmax)
        metadata = {
            "colormap": self.vi_colormap,
            "scale_mode": self.vi_scale_mode,
            "vmin_pct": float(self.vi_vmin_pct),
            "vmax_pct": float(self.vi_vmax_pct),
            "vmin": float(vmin),
            "vmax": float(vmax),
        }
        return rgb, metadata

    def _render_fft_rgb(self) -> tuple[np.ndarray, dict]:
        virtual_raw = self._get_virtual_image_array()
        fft = np.fft.fftshift(np.fft.fft2(virtual_raw))
        mag = np.abs(fft).astype(np.float32)
        scaled = self._apply_scale_mode(mag, self.fft_scale_mode)
        if self.fft_auto:
            display_min, display_max = self._fft_enhanced_range(scaled)
        else:
            display_min = float(scaled.min()) if scaled.size else 0.0
            display_max = float(scaled.max()) if scaled.size else 0.0
        vmin, vmax = self._slider_range(display_min, display_max, self.fft_vmin_pct, self.fft_vmax_pct)
        rgb = self._render_colormap_rgb(scaled, self.fft_colormap, vmin, vmax)
        metadata = {
            "colormap": self.fft_colormap,
            "scale_mode": self.fft_scale_mode,
            "auto": bool(self.fft_auto),
            "vmin_pct": float(self.fft_vmin_pct),
            "vmax_pct": float(self.fft_vmax_pct),
            "vmin": float(vmin),
            "vmax": float(vmax),
        }
        return rgb, metadata

    _EXPORT_VIEWS = ("diffraction", "virtual", "fft", "all")
    _EXPORT_FORMATS = ("png", "pdf")

    def _validate_export_view(self, view: str | None) -> str:
        view_key = (view or "all").strip().lower()
        if view_key not in self._EXPORT_VIEWS:
            raise ValueError(f"Unsupported view '{view}'. Supported: {', '.join(self._EXPORT_VIEWS)}")
        return view_key

    def _validate_frame_idx(self, frame_idx: int | None) -> int:
        if frame_idx is None:
            return int(self.frame_idx)
        idx = int(frame_idx)
        if idx < 0 or idx >= self.n_frames:
            raise ValueError(
                f"frame_idx={idx} is out of range [0, {self.n_frames - 1}]"
            )
        return idx

    def _validate_position(self, position: tuple[int, int] | None) -> tuple[int, int]:
        if position is None:
            return int(self.pos_row), int(self.pos_col)
        if len(position) != 2:
            raise ValueError(
                "position must be a (row, col) tuple with exactly two values"
            )
        row = int(position[0])
        col = int(position[1])
        if row < 0 or row >= self.shape_rows or col < 0 or col >= self.shape_cols:
            raise ValueError(
                f"position=({row}, {col}) is out of range for "
                f"scan_shape=({self.shape_rows}, {self.shape_cols})"
            )
        return row, col

    def _resolve_export_format(self, path: pathlib.Path, fmt: str | None) -> str:
        resolved = (fmt or path.suffix.lstrip(".") or "png").strip().lower()
        if resolved not in self._EXPORT_FORMATS:
            raise ValueError(f"Unsupported format '{resolved}'. Supported: {', '.join(self._EXPORT_FORMATS)}")
        return resolved

    @staticmethod
    def _round_to_nice_value(value: float) -> float:
        if value <= 0:
            return 1.0
        magnitude = 10 ** math.floor(math.log10(value))
        normalized = value / magnitude
        if normalized < 1.5:
            return float(magnitude)
        if normalized < 3.5:
            return float(2 * magnitude)
        if normalized < 7.5:
            return float(5 * magnitude)
        return float(10 * magnitude)

    def _format_scale_label(self, value: float, unit: str) -> str:
        nice = self._round_to_nice_value(value)
        if unit == "Å":
            if nice >= 10:
                return f"{int(round(nice / 10))} nm"
            if nice >= 1:
                return f"{int(round(nice))} Å"
            return f"{nice:.2f} Å"
        if unit == "mrad":
            if nice >= 1000:
                return f"{int(round(nice / 1000))} rad"
            if nice >= 1:
                return f"{int(round(nice))} mrad"
            return f"{nice:.2f} mrad"
        if nice >= 1:
            return f"{int(round(nice))} px"
        return f"{nice:.1f} px"

    @staticmethod
    def _draw_crosshair(draw, x: float, y: float, size: float, color, width: int) -> None:
        draw.line([(x - size, y), (x + size, y)], fill=color, width=width)
        draw.line([(x, y - size), (x, y + size)], fill=color, width=width)

    def _draw_scalebar_overlay(self, image, pixel_size: float, unit: str) -> None:
        from PIL import ImageDraw, ImageFont

        if pixel_size <= 0:
            return

        draw = ImageDraw.Draw(image, mode="RGBA")
        font = ImageFont.load_default()
        width, height = image.size
        margin = max(8, int(min(width, height) * 0.04))
        thickness = max(2, int(height * 0.01))
        target_bar_px = max(36, int(width * 0.15))
        target_physical = float(target_bar_px) * float(pixel_size)
        nice_physical = self._round_to_nice_value(target_physical)
        bar_px = max(12, int(round(nice_physical / float(pixel_size))))
        bar_px = min(bar_px, max(12, int(width * 0.8)))

        x1 = width - margin
        x0 = x1 - bar_px
        y1 = height - margin
        y0 = y1 - thickness

        draw.rectangle([(x0 + 1, y0 + 1), (x1 + 1, y1 + 1)], fill=(0, 0, 0, 180))
        draw.rectangle([(x0, y0), (x1, y1)], fill=(255, 255, 255, 255))

        label = self._format_scale_label(nice_physical, unit)
        label_bbox = draw.textbbox((0, 0), label, font=font)
        label_w = label_bbox[2] - label_bbox[0]
        label_h = label_bbox[3] - label_bbox[1]
        tx = x0 + (bar_px - label_w) / 2
        ty = y0 - label_h - 4
        draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0, 220), font=font)
        draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)

        zoom_label = "1.0x"
        zoom_bbox = draw.textbbox((0, 0), zoom_label, font=font)
        zoom_h = zoom_bbox[3] - zoom_bbox[1]
        zx = margin
        zy = height - margin - zoom_h
        draw.text((zx + 1, zy + 1), zoom_label, fill=(0, 0, 0, 220), font=font)
        draw.text((zx, zy), zoom_label, fill=(255, 255, 255, 255), font=font)

    def _draw_dp_overlays(self, image) -> None:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(image, mode="RGBA")
        width, height = image.size
        scale_x = float(width) / float(max(1, self.det_cols))
        scale_y = float(height) / float(max(1, self.det_rows))
        cx = float(self.roi_center_col) * scale_x
        cy = float(self.roi_center_row) * scale_y

        if self.roi_active and self.roi_mode != "point":
            stroke = (0, 220, 0, 240)
            fill = (0, 220, 0, 45)
            if self.roi_mode == "circle":
                rx = float(self.roi_radius) * scale_x
                ry = float(self.roi_radius) * scale_y
                draw.ellipse([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=stroke, fill=fill, width=2)
            elif self.roi_mode == "square":
                rx = float(self.roi_radius) * scale_x
                ry = float(self.roi_radius) * scale_y
                draw.rectangle([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=stroke, fill=fill, width=2)
            elif self.roi_mode == "rect":
                rx = (float(self.roi_width) / 2.0) * scale_x
                ry = (float(self.roi_height) / 2.0) * scale_y
                draw.rectangle([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=stroke, fill=fill, width=2)
            elif self.roi_mode == "annular":
                outer_rx = float(self.roi_radius) * scale_x
                outer_ry = float(self.roi_radius) * scale_y
                inner_rx = float(self.roi_radius_inner) * scale_x
                inner_ry = float(self.roi_radius_inner) * scale_y
                draw.ellipse(
                    [(cx - outer_rx, cy - outer_ry), (cx + outer_rx, cy + outer_ry)],
                    outline=stroke,
                    fill=fill,
                    width=2,
                )
                draw.ellipse(
                    [(cx - inner_rx, cy - inner_ry), (cx + inner_rx, cy + inner_ry)],
                    outline=stroke,
                    fill=(0, 0, 0, 0),
                    width=2,
                )

        marker_color = (0, 220, 0, 255) if self.roi_active else (255, 100, 100, 255)
        self._draw_crosshair(draw, cx, cy, size=max(6, int(min(width, height) * 0.03)), color=marker_color, width=2)

        if len(self.profile_line) == 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            x0 = float(p0["col"]) * scale_x
            y0 = float(p0["row"]) * scale_y
            x1 = float(p1["col"]) * scale_x
            y1 = float(p1["row"]) * scale_y
            draw.line([(x0, y0), (x1, y1)], fill=(0, 200, 255, 240), width=max(1, int(self.profile_width)))
            r = 3
            draw.ellipse([(x0 - r, y0 - r), (x0 + r, y0 + r)], fill=(0, 200, 255, 255))
            draw.ellipse([(x1 - r, y1 - r), (x1 + r, y1 + r)], fill=(0, 200, 255, 255))

    def _draw_vi_overlays(self, image) -> None:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(image, mode="RGBA")
        width, height = image.size
        scale_x = float(width) / float(max(1, self.shape_cols))
        scale_y = float(height) / float(max(1, self.shape_rows))

        px = float(self.pos_col) * scale_x
        py = float(self.pos_row) * scale_y
        self._draw_crosshair(
            draw,
            px,
            py,
            size=max(6, int(min(width, height) * 0.03)),
            color=(255, 100, 100, 240),
            width=2,
        )

        if self.vi_roi_mode == "off":
            return

        cx = float(self.vi_roi_center_col) * scale_x
        cy = float(self.vi_roi_center_row) * scale_y
        stroke = (180, 80, 255, 240)
        fill = (180, 80, 255, 45)
        if self.vi_roi_mode == "circle":
            rx = float(self.vi_roi_radius) * scale_x
            ry = float(self.vi_roi_radius) * scale_y
            draw.ellipse([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=stroke, fill=fill, width=2)
        elif self.vi_roi_mode == "square":
            rx = float(self.vi_roi_radius) * scale_x
            ry = float(self.vi_roi_radius) * scale_y
            draw.rectangle([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=stroke, fill=fill, width=2)
        elif self.vi_roi_mode == "rect":
            rx = (float(self.vi_roi_width) / 2.0) * scale_x
            ry = (float(self.vi_roi_height) / 2.0) * scale_y
            draw.rectangle([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=stroke, fill=fill, width=2)

        self._draw_crosshair(
            draw,
            cx,
            cy,
            size=max(6, int(min(width, height) * 0.03)),
            color=(180, 80, 255, 240),
            width=2,
        )

    def _decorate_panel(
        self,
        image,
        panel_key: str,
        include_overlays: bool,
        include_scalebar: bool,
    ):
        out = image.copy()
        if include_overlays:
            if panel_key == "diffraction":
                self._draw_dp_overlays(out)
            elif panel_key == "virtual":
                self._draw_vi_overlays(out)
        if include_scalebar:
            if panel_key == "diffraction":
                unit = "mrad" if self.k_calibrated else "px"
                self._draw_scalebar_overlay(out, float(self.k_pixel_size), unit)
            elif panel_key == "virtual":
                self._draw_scalebar_overlay(out, float(self.pixel_size), "Å")
        return out

    def _render_panel_image(
        self,
        panel_key: str,
        include_overlays: bool,
        include_scalebar: bool,
    ) -> tuple[Any, dict[str, Any]]:
        from PIL import Image

        if panel_key == "diffraction":
            rgb, render_meta = self._render_dp_rgb()
        elif panel_key == "virtual":
            rgb, render_meta = self._render_virtual_rgb()
        elif panel_key == "fft":
            rgb, render_meta = self._render_fft_rgb()
        else:
            raise ValueError(f"Unsupported panel {panel_key!r}. Valid options: 'diffraction', 'virtual', 'fft', 'all'.")

        panel = Image.fromarray(rgb, mode="RGB")
        panel = self._decorate_panel(panel, panel_key, include_overlays, include_scalebar)
        return panel, render_meta

    def _compose_horizontal(self, panels: list[Any]):
        from PIL import Image

        height = max(panel.height for panel in panels)
        width = sum(panel.width for panel in panels)
        composite = Image.new("RGB", (width, height), color=(0, 0, 0))
        x0 = 0
        for panel in panels:
            composite.paste(panel, (x0, 0))
            x0 += panel.width
        return composite

    def _calibration_metadata(self) -> dict[str, Any]:
        return {
            "pixel_size_angstrom": float(self.pixel_size),
            "pixel_size_unit": "Å/px",
            "k_pixel_size": float(self.k_pixel_size),
            "k_pixel_size_unit": "mrad/px" if self.k_calibrated else "px/px",
            "k_calibrated": bool(self.k_calibrated),
            "center_row": float(self.center_row),
            "center_col": float(self.center_col),
            "bf_radius": float(self.bf_radius),
        }

    def _roi_metadata(self) -> dict[str, Any]:
        return {
            "active": bool(self.roi_active),
            "mode": self.roi_mode,
            "center_row": float(self.roi_center_row),
            "center_col": float(self.roi_center_col),
            "radius": float(self.roi_radius),
            "radius_inner": float(self.roi_radius_inner),
            "width": float(self.roi_width),
            "height": float(self.roi_height),
        }

    def _vi_roi_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.vi_roi_mode,
            "center_row": float(self.vi_roi_center_row),
            "center_col": float(self.vi_roi_center_col),
            "radius": float(self.vi_roi_radius),
            "width": float(self.vi_roi_width),
            "height": float(self.vi_roi_height),
        }

    def _build_image_export_metadata(
        self,
        export_path: pathlib.Path,
        view_key: str,
        fmt: str,
        render_meta: dict[str, Any],
        include_overlays: bool,
        include_scalebar: bool,
        export_kind: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            **build_json_header("Show4DSTEM"),
            "view": view_key,
            "format": fmt,
            "export_kind": export_kind,
            "path": str(export_path),
            "position": {"row": int(self.pos_row), "col": int(self.pos_col)},
            "frame_idx": int(self.frame_idx),
            "n_frames": int(self.n_frames),
            "scan_shape": {"rows": int(self.shape_rows), "cols": int(self.shape_cols)},
            "detector_shape": {"rows": int(self.det_rows), "cols": int(self.det_cols)},
            "roi": self._roi_metadata(),
            "vi_roi": self._vi_roi_metadata(),
            "calibration": self._calibration_metadata(),
            "display": render_meta,
            "include_overlays": bool(include_overlays),
            "include_scalebar": bool(include_scalebar),
        }
        if extra:
            metadata.update(extra)
        return metadata

    def save_image(
        self,
        path: str | pathlib.Path,
        view: str | None = None,
        position: tuple[int, int] | None = None,
        frame_idx: int | None = None,
        format: str | None = None,
        include_metadata: bool = True,
        metadata_path: str | pathlib.Path | None = None,
        include_overlays: bool | None = None,
        include_scalebar: bool | None = None,
        restore_state: bool = True,
        dpi: int | None = None,
    ) -> pathlib.Path:
        """
        Save the current visualization as PNG or PDF.

        Parameters
        ----------
        path : str or pathlib.Path
            Output image path.
        view : str, optional
            One of: "diffraction", "virtual", "fft", "all".
        position : tuple[int, int], optional
            Temporary scan position override as (row, col) for this export.
        frame_idx : int, optional
            Temporary frame index override for 5D data.
        format : str, optional
            "png" or "pdf". If omitted, inferred from file extension.
        include_metadata : bool, default True
            If True, writes JSON metadata next to the image.
        metadata_path : str or pathlib.Path, optional
            Override metadata JSON path.
        include_overlays : bool, default True
            Draw ROI/profile/crosshair overlays on exported panels.
        include_scalebar : bool, default True
            Draw panel scale bars on exported panels.
        restore_state : bool, default True
            If True, temporary position/frame overrides are reverted after export.
        dpi : int, optional
            Export DPI metadata.

        Returns
        -------
        pathlib.Path
            The written image path.
        """
        from PIL import Image

        export_path = pathlib.Path(path)
        view_key = self._validate_export_view(view)
        fmt = self._resolve_export_format(export_path, format)
        dpi_value = 300 if dpi is None else int(dpi)
        overlays_enabled = True if include_overlays is None else bool(include_overlays)
        scalebar_enabled = True if include_scalebar is None else bool(include_scalebar)
        if dpi_value <= 0:
            raise ValueError(f"dpi must be > 0, got {dpi_value}")

        export_path.parent.mkdir(parents=True, exist_ok=True)

        prev_row, prev_col = self.pos_row, self.pos_col
        prev_frame = self.frame_idx
        meta_path: pathlib.Path | None = None

        try:
            if frame_idx is not None:
                self.frame_idx = self._validate_frame_idx(frame_idx)
            if position is not None:
                row, col = self._validate_position(position)
                self.pos_row = row
                self.pos_col = col

            if view_key == "diffraction":
                image, dp_meta = self._render_panel_image(
                    "diffraction", overlays_enabled, scalebar_enabled
                )
                render_meta = {"diffraction": dp_meta}
            elif view_key == "virtual":
                image, vi_meta = self._render_panel_image(
                    "virtual", overlays_enabled, scalebar_enabled
                )
                render_meta = {"virtual": vi_meta}
            elif view_key == "fft":
                image, fft_meta = self._render_panel_image(
                    "fft", overlays_enabled, scalebar_enabled
                )
                render_meta = {"fft": fft_meta}
            else:
                panel_images = []
                render_meta = {}
                dp_img, dp_meta = self._render_panel_image(
                    "diffraction", overlays_enabled, scalebar_enabled
                )
                vi_img, vi_meta = self._render_panel_image(
                    "virtual", overlays_enabled, scalebar_enabled
                )
                panel_images.extend([dp_img, vi_img])
                render_meta = {"diffraction": dp_meta, "virtual": vi_meta}
                if self.show_fft:
                    fft_img, fft_meta = self._render_panel_image(
                        "fft", overlays_enabled, scalebar_enabled
                    )
                    panel_images.append(fft_img)
                    render_meta["fft"] = fft_meta
                image = self._compose_horizontal(panel_images)

            if fmt == "pdf":
                Image.init()
                image = image.convert("RGB")
                image.save(export_path, format="PDF", resolution=dpi_value)
            else:
                image.save(export_path, format="PNG", dpi=(dpi_value, dpi_value))

            if include_metadata:
                meta_path = (
                    pathlib.Path(metadata_path)
                    if metadata_path is not None
                    else export_path.with_suffix(".json")
                )
                metadata = self._build_image_export_metadata(
                    export_path=export_path,
                    view_key=view_key,
                    fmt=fmt,
                    render_meta=render_meta,
                    include_overlays=overlays_enabled,
                    include_scalebar=scalebar_enabled,
                    export_kind="single_view_image",
                    extra={"dpi": int(dpi_value)},
                )
                meta_path.write_text(json.dumps(metadata, indent=2))
        finally:
            if restore_state:
                self.frame_idx = prev_frame
                self.pos_row = prev_row
                self.pos_col = prev_col

        return export_path

    def apply_preset(self, name: str) -> Self:
        preset_name = str(name).strip().lower()
        # Batch all trait writes atomically. Without this, each individual
        # trait change fires _on_roi_change, and intermediate states (e.g. mode
        # just switched to "annular" but radius_inner still stale from the
        # previous preset) compute a wrong mask -> black VI flashes before the
        # final correct frame. hold_trait_notifications defers observers until
        # all 5 traits have committed.
        bf = self.bf_radius
        center_row = float(self.center_row)
        center_col = float(self.center_col)
        self._suppress_roi_recompute = True
        try:
            if preset_name == "bf":
                with self.hold_trait_notifications():
                    self.roi_active = True
                    self.roi_mode = "circle"
                    self.roi_center_row = center_row
                    self.roi_center_col = center_col
                    self.roi_radius = float(max(1.0, bf))
            elif preset_name == "abf":
                with self.hold_trait_notifications():
                    self.roi_active = True
                    self.roi_mode = "annular"
                    self.roi_center_row = center_row
                    self.roi_center_col = center_col
                    self.roi_radius_inner = float(max(0.5, bf * 0.5))
                    self.roi_radius = float(max(1.0, bf))
            elif preset_name == "adf":
                with self.hold_trait_notifications():
                    self.roi_active = True
                    self.roi_mode = "annular"
                    self.roi_center_row = center_row
                    self.roi_center_col = center_col
                    self.roi_radius_inner = float(max(1.0, bf))
                    self.roi_radius = float(max(bf + 1.0, bf * 2.0))
            elif preset_name == "haadf":
                with self.hold_trait_notifications():
                    self.roi_active = True
                    self.roi_mode = "annular"
                    self.roi_center_row = center_row
                    self.roi_center_col = center_col
                    self.roi_radius_inner = float(max(1.0, bf * 2.0))
                    self.roi_radius = float(max(bf * 2.0 + 1.0, bf * 4.0))
            else:
                raise ValueError(
                    f"Unknown preset {name!r}. Choices: 'bf', 'abf', 'adf', 'haadf'."
                )
        finally:
            self._suppress_roi_recompute = False
        # Single recompute with final, consistent state.
        self._compute_virtual_image_from_roi()
        return self


    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        mode = self.dp_scale_mode
        scaled = self._apply_scale_mode(frame, mode)
        if self.dp_vmin is not None and self.dp_vmax is not None:
            fmin = float(self._apply_scale_mode(
                np.array([max(self.dp_vmin, 0)], dtype=np.float32), mode
            )[0])
            fmax = float(self._apply_scale_mode(
                np.array([max(self.dp_vmax, 0)], dtype=np.float32), mode
            )[0])
        else:
            fmin = float(scaled.min())
            fmax = float(scaled.max())
            fmin, fmax = self._slider_range(fmin, fmax, self.dp_vmin_pct, self.dp_vmax_pct)
        if fmax > fmin:
            return np.clip((scaled - fmin) / (fmax - fmin) * 255, 0, 255).astype(np.uint8)
        return np.zeros(frame.shape, dtype=np.uint8)

    def _on_gif_export(self, change=None):
        if not self._gif_export_requested:
            return
        self._gif_export_requested = False
        self._generate_gif()

    def _generate_gif(self):
        import io

        from matplotlib import colormaps
        from PIL import Image

        if not self._path_points:
            with self.hold_sync():
                self._gif_data = b""
                self._gif_metadata_json = ""
            return

        cmap_fn = colormaps.get_cmap(self.dp_colormap)
        duration_ms = max(10, self.path_interval_ms)

        pil_frames = []
        for row, col in self._path_points:
            row = max(0, min(self.shape_rows - 1, row))
            col = max(0, min(self.shape_cols - 1, col))
            frame = self._get_frame(row, col).astype(np.float32)
            normalized = self._normalize_frame(frame)
            rgba = cmap_fn(normalized / 255.0)
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(rgb))

        if not pil_frames:
            return

        buf = io.BytesIO()
        pil_frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        metadata = {
            **build_json_header("Show4DSTEM"),
            "view": "diffraction",
            "format": "gif",
            "export_kind": "path_animation",
            "n_frames": int(len(pil_frames)),
            "duration_ms": int(duration_ms),
            "path_loop": bool(self.path_loop),
            "path_points": [{"row": int(row), "col": int(col)} for row, col in self._path_points],
            "frame_idx": int(self.frame_idx),
            "n_frames_total": int(self.n_frames),
            "scan_shape": {"rows": int(self.shape_rows), "cols": int(self.shape_cols)},
            "detector_shape": {"rows": int(self.det_rows), "cols": int(self.det_cols)},
            "calibration": self._calibration_metadata(),
            "display": {
                "diffraction": {
                    "colormap": self.dp_colormap,
                    "scale_mode": self.dp_scale_mode,
                    "vmin_pct": float(self.dp_vmin_pct),
                    "vmax_pct": float(self.dp_vmax_pct),
                }
            },
        }
        with self.hold_sync():
            self._gif_metadata_json = json.dumps(metadata, indent=2)
            self._gif_data = buf.getvalue()

    def _update_frame(self, change=None):
        """Send raw float32 frame to frontend (JS handles scale/colormap)."""
        if self._data is None:
            return
        # Get frame as tensor (stays on device)
        data = self._frame_data
        if data.ndim == 3:
            idx = self.pos_row * self.shape_cols + self.pos_col
            frame = data[idx]
        else:
            frame = data[self.pos_row, self.pos_col]

        # Cast small frame to float32 for stats and JS transfer. Bulk data
        # stays in native dtype; only this single 192×192 (~144 KB) frame
        # gets promoted.
        if frame.dtype != torch.float32:
            frame = frame.float()
        # Stats compute moved to JS (frontend has frame_bytes; computeStats() in
        # js/stats.ts does mean/min/max/std on the Float32Array directly,
        # avoiding 4 sync trait round-trips per scan-position click).
        self.frame_bytes = frame.cpu().numpy().tobytes()

    def _on_roi_change(self, change=None):
        """Recompute virtual image when individual ROI params change.

        High-frequency drag updates use the compound roi_center trait instead.
        """
        if not self.roi_active:
            return
        if getattr(self, "_suppress_roi_recompute", False):
            return
        self._compute_virtual_image_from_roi()

    def _on_roi_center_change(self, change=None):
        """Handle batched roi_center updates from JS (single observer for row+col).

        This is the fast path for drag operations. JS sends [row, col] as a single
        compound trait, so only one observer fires per mouse move.
        """
        if not self.roi_active:
            return
        if getattr(self, "_suppress_roi_recompute", False):
            return
        if change and "new" in change:
            row, col = change["new"]
            # Sync to individual traits (without triggering _on_roi_change observers)
            self.unobserve(self._on_roi_change, names=["roi_center_col", "roi_center_row"])
            self.roi_center_row = row
            self.roi_center_col = col
            self.observe(self._on_roi_change, names=["roi_center_col", "roi_center_row"])
        self._compute_virtual_image_from_roi()

    def _on_vi_roi_center_change(self, change=None):
        """Apply compound (row, col) update atomically (avoids split-trait race)."""
        if change and "new" in change:
            row, col = change["new"]
            self.unobserve(self._on_vi_roi_change, names=["vi_roi_center_row", "vi_roi_center_col"])
            self.vi_roi_center_row = float(row)
            self.vi_roi_center_col = float(col)
            self.observe(self._on_vi_roi_change, names=["vi_roi_center_row", "vi_roi_center_col"])
        if self.vi_roi_mode == "off":
            self.vi_roi_dp_bytes = b""
            return
        self._compute_vi_roi_dp()

    def _on_vi_roi_change(self, change=None):
        """Recompute reduced DP when VI ROI or reduction changes."""
        if self.vi_roi_mode == "off":
            self.vi_roi_dp_bytes = b""
            return
        self._compute_vi_roi_dp()

    def _compute_vi_roi_dp(self):
        """Reduce diffraction patterns over scan positions inside VI ROI.

        Reduction selected by `vi_roi_reduce`:
        - "mean": average DP (size-invariant, default for region-of-interest analysis)
        - "sum": total counts (scales with ROI area; use for quantitative integration)
        - "max": brightest pixel per detector position across the region
        """
        if self._data is None:
            return
        if self.vi_roi_mode == "circle":
            mask = (self._scan_row_coords - self.vi_roi_center_row) ** 2 + (self._scan_col_coords - self.vi_roi_center_col) ** 2 <= self.vi_roi_radius ** 2
        elif self.vi_roi_mode == "square":
            half_size = self.vi_roi_radius
            mask = (torch.abs(self._scan_row_coords - self.vi_roi_center_row) <= half_size) & (torch.abs(self._scan_col_coords - self.vi_roi_center_col) <= half_size)
        elif self.vi_roi_mode == "rect":
            half_w = self.vi_roi_width / 2
            half_h = self.vi_roi_height / 2
            mask = (torch.abs(self._scan_row_coords - self.vi_roi_center_row) <= half_h) & (torch.abs(self._scan_col_coords - self.vi_roi_center_col) <= half_w)
        else:
            return

        n_positions = int(mask.sum())
        if n_positions == 0:
            self.vi_roi_dp_bytes = b""
            return

        reduce = self.vi_roi_reduce
        data = self._frame_data
        # Single chunked torch path. For each scan-row chunk: cast to float32 and
        # broadcast-multiply by the mask (no `chunk[row_mask]` slab, which would
        # roughly duplicate the chunk in memory when the mask is dense). Sum/mean
        # use einsum over scan dims; max masks zero rows then takes amax.
        data_4d = data if data.ndim == 4 else data.reshape(self._scan_shape[0], self._scan_shape[1], *self._det_shape)
        rows_per_chunk = self._chunk_rows()
        if reduce == "sum" or reduce == "mean":
            dp = torch.zeros(self._det_shape, dtype=torch.float32, device=self._device)
        else:  # max
            dp = torch.full(self._det_shape, -float("inf"), dtype=torch.float32, device=self._device)
        for i in range(0, self._scan_shape[0], rows_per_chunk):
            row_mask = mask[i:i + rows_per_chunk]
            if not bool(row_mask.any()):
                continue
            chunk = data_4d[i:i + rows_per_chunk]
            if not torch.is_floating_point(chunk):
                chunk = chunk.float()
            row_mask_f = row_mask.float()
            if reduce == "max":
                # Outside-mask positions become 0; doesn't affect amax provided
                # the data has any non-negative pixels (true for detector counts).
                dp = torch.maximum(dp, (chunk * row_mask_f[..., None, None]).amax(dim=(0, 1)))
            else:
                dp += torch.einsum("rcij,rc->ij", chunk, row_mask_f)
        if reduce == "mean":
            dp /= float(n_positions)

        self.vi_roi_dp_bytes = dp.cpu().numpy().tobytes()

    def _create_circular_mask(self, cx: float, cy: float, radius: float):
        """Create circular mask (boolean tensor on device)."""
        mask = (self._det_col_coords - cx) ** 2 + (self._det_row_coords - cy) ** 2 <= radius ** 2
        return mask

    def _create_square_mask(self, cx: float, cy: float, half_size: float):
        """Create square mask (boolean tensor on device)."""
        mask = (torch.abs(self._det_col_coords - cx) <= half_size) & (torch.abs(self._det_row_coords - cy) <= half_size)
        return mask

    def _create_annular_mask(
        self, cx: float, cy: float, inner: float, outer: float
    ):
        """Create annular (donut) mask (boolean tensor on device)."""
        dist_sq = (self._det_col_coords - cx) ** 2 + (self._det_row_coords - cy) ** 2
        mask = (dist_sq >= inner ** 2) & (dist_sq <= outer ** 2)
        return mask

    def _create_rect_mask(self, cx: float, cy: float, half_width: float, half_height: float):
        """Create rectangular mask (boolean tensor on device)."""
        mask = (torch.abs(self._det_col_coords - cx) <= half_width) & (torch.abs(self._det_row_coords - cy) <= half_height)
        return mask

    def _on_calibration_change(self, change=None):
        self._cached_bf_virtual = None
        self._cached_abf_virtual = None
        self._cached_adf_virtual = None
        self._cached_haadf_virtual = None

    def _precompute_common_virtual_images(self):
        """Pre-compute BF/ABF/ADF/HAADF virtual image bytes. Annular ranges match
        apply_preset() so the cache always hits on preset clicks."""
        cx, cy, bf = self.center_col, self.center_row, self.bf_radius
        self._cached_bf_virtual = self._to_float32_bytes(
            self._fast_masked_sum(self._create_circular_mask(cx, cy, bf))
        )
        self._cached_abf_virtual = self._to_float32_bytes(
            self._fast_masked_sum(self._create_annular_mask(cx, cy, bf * 0.5, bf))
        )
        self._cached_adf_virtual = self._to_float32_bytes(
            self._fast_masked_sum(self._create_annular_mask(cx, cy, bf, bf * 2.0))
        )
        self._cached_haadf_virtual = self._to_float32_bytes(
            self._fast_masked_sum(self._create_annular_mask(cx, cy, bf * 2.0, bf * 4.0))
        )

    def _get_cached_preset(self) -> bytes | None:
        """Return cached preset bytes if current ROI matches BF/ABF/ADF preset shape."""
        # Must be centered on detector center
        if abs(self.roi_center_col - self.center_col) >= 1 or abs(self.roi_center_row - self.center_row) >= 1:
            return None

        bf = self.bf_radius

        # BF: circle at bf_radius
        if (self.roi_mode == "circle" and abs(self.roi_radius - bf) < 1):
            return self._cached_bf_virtual

        # ABF: annular at 0.5*bf to bf
        if (self.roi_mode == "annular" and
            abs(self.roi_radius_inner - bf * 0.5) < 1 and
            abs(self.roi_radius - bf) < 1):
            return self._cached_abf_virtual

        # ADF: annular at bf to 2*bf
        if (self.roi_mode == "annular" and
            abs(self.roi_radius_inner - bf) < 1 and
            abs(self.roi_radius - bf * 2.0) < 1):
            return self._cached_adf_virtual

        # HAADF: annular at 2*bf to 4*bf
        if (self.roi_mode == "annular" and
            abs(self.roi_radius_inner - bf * 2.0) < 1 and
            abs(self.roi_radius - bf * 4.0) < 1):
            return self._cached_haadf_virtual

        return None

    def _virtual_image_for_frame(self, frame_idx: int) -> np.ndarray:
        """Compute virtual image for a specific 5D frame without mutating traits.

        Single chunked-torch path matching _fast_masked_sum.
        """
        data = self._data[frame_idx] if self.n_frames > 1 else self._data
        cx, cy = self.roi_center_col, self.roi_center_row
        if self.roi_mode == "circle" and self.roi_radius > 0:
            mask = self._create_circular_mask(cx, cy, self.roi_radius)
        elif self.roi_mode == "square" and self.roi_radius > 0:
            mask = self._create_square_mask(cx, cy, self.roi_radius)
        elif self.roi_mode == "annular" and self.roi_radius > 0:
            mask = self._create_annular_mask(cx, cy, self.roi_radius_inner, self.roi_radius)
        elif self.roi_mode == "rect" and self.roi_width > 0 and self.roi_height > 0:
            mask = self._create_rect_mask(cx, cy, self.roi_width / 2, self.roi_height / 2)
        else:
            row = int(max(0, min(round(cy), self._det_shape[0] - 1)))
            col = int(max(0, min(round(cx), self._det_shape[1] - 1)))
            if data.ndim == 4:
                vi = data[:, :, row, col]
            else:
                vi = data[:, row, col].reshape(self._scan_shape)
            return vi.cpu().numpy().astype(np.float32, copy=False)
        data_4d = data if data.ndim == 4 else data.reshape(self._scan_shape[0], self._scan_shape[1], *self._det_shape)
        mask_f = mask.float()
        rows_per_chunk = self._chunk_rows()
        out = torch.zeros(self._scan_shape, dtype=torch.float32, device=self._device)
        for i in range(0, data_4d.shape[0], rows_per_chunk):
            chunk = data_4d[i:i + rows_per_chunk]
            if not torch.is_floating_point(chunk):
                chunk = chunk.float()
            out[i:i + rows_per_chunk] = torch.tensordot(chunk, mask_f, dims=([2, 3], [0, 1]))
        return out.cpu().numpy().astype(np.float32, copy=False)

    def _chunk_rows(self) -> int:
        """Pick rows-per-chunk so float32 transient stays under _CHUNK_BYTE_BUDGET.

        Float32 cast of one chunk = rows × scan_cols × det_h × det_w × 4 bytes.
        Selected slabs (e.g. vi_roi reduce) inherit the same per-row budget.
        """
        per_row = self._scan_shape[1] * self._det_shape[0] * self._det_shape[1] * 4
        return max(1, _CHUNK_BYTE_BUDGET // max(1, per_row))

    def _fast_masked_sum(self, mask: torch.Tensor) -> torch.Tensor:
        """Sum data over scan positions weighted by detector mask.

        Chunked tensordot. Per-chunk float32 cast bounded by _CHUNK_BYTE_BUDGET.
        Identical math on CUDA / MPS / CPU.
        """
        data = self._frame_data
        if data.ndim == 3:
            data_4d = data.reshape(self._scan_shape[0], self._scan_shape[1], *self._det_shape)
        else:
            data_4d = data
        # Single chunked torch path. Per scan-row chunk: cast to float32, contract
        # with mask via tensordot. Transient memory bounded by chunk size. Same
        # code on CUDA / MPS / CPU. Identical results regardless of device.
        mask_f = mask.float()
        n_rows = data_4d.shape[0]
        out = torch.zeros(self._scan_shape, dtype=torch.float32, device=self._device)
        # Convert positions chunk size to row chunks based on scan width.
        rows_per_chunk = self._chunk_rows()
        for i in range(0, n_rows, rows_per_chunk):
            chunk = data_4d[i:i + rows_per_chunk]
            if not torch.is_floating_point(chunk):
                chunk = chunk.float()
            out[i:i + rows_per_chunk] = torch.tensordot(chunk, mask_f, dims=([2, 3], [0, 1]))
        return out

    def _to_float32_bytes(self, arr: torch.Tensor) -> bytes:
        """Convert tensor (any numeric dtype) to float32 bytes for JS transfer.

        Cast to float32 only at the small output. Integer reductions (uint16 sums,
        int64 accumulators) get promoted here so the multi-GB raw data never gets
        copied to float. Stats (min/max/mean/std) are computed JS-side from the
        same Float32Array — keeping them out of separate traits avoids a
        comm-message ordering race where bytes from click N arrive with stats
        from click N-1, producing a wrong colormap normalization (uniform white
        flash on rapid preset switching).
        """
        if arr.dtype != torch.float32:
            arr = arr.float()
        return arr.cpu().numpy().tobytes()

    def _compute_virtual_image_from_roi(self):
        """Compute virtual image based on ROI mode."""
        if self._data is None:
            return
        cached = self._get_cached_preset()
        if cached is not None:
            self.virtual_image_bytes = cached
            return

        cx, cy = self.roi_center_col, self.roi_center_row

        if self.roi_mode == "circle" and self.roi_radius > 0:
            mask = self._create_circular_mask(cx, cy, self.roi_radius)
        elif self.roi_mode == "square" and self.roi_radius > 0:
            mask = self._create_square_mask(cx, cy, self.roi_radius)
        elif self.roi_mode == "annular" and self.roi_radius > 0:
            mask = self._create_annular_mask(cx, cy, self.roi_radius_inner, self.roi_radius)
        elif self.roi_mode == "rect" and self.roi_width > 0 and self.roi_height > 0:
            mask = self._create_rect_mask(cx, cy, self.roi_width / 2, self.roi_height / 2)
        else:
            # Point mode: single-pixel indexing
            row = int(max(0, min(round(cy), self._det_shape[0] - 1)))
            col = int(max(0, min(round(cx), self._det_shape[1] - 1)))
            data = self._frame_data
            if data.ndim == 4:
                virtual_image = data[:, :, row, col]
            else:
                virtual_image = data[:, row, col].reshape(self._scan_shape)
            self.virtual_image_bytes = self._to_float32_bytes(virtual_image)
            return

        self.virtual_image_bytes = self._to_float32_bytes(self._fast_masked_sum(mask))

