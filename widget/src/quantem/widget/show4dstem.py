"""
show4d: Fast interactive 4D-STEM viewer widget with advanced features.

Features:
- Binary transfer (no base64 overhead)
- ROI drawing tools
- Path animation (raster scan, custom paths)
"""

import pathlib

import anywidget
import numpy as np
import torch
import traitlets

from quantem.core.config import validate_device
from quantem.widget.array_utils import to_numpy


# ============================================================================
# Constants
# ============================================================================
DEFAULT_BF_RATIO = 0.125  # BF disk radius as fraction of detector size (1/8)
SPARSE_MASK_THRESHOLD = 0.2  # Use sparse indexing below this mask coverage
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
        Dataset4dstem object (calibration auto-extracted) or 4D array
        of shape (scan_x, scan_y, det_x, det_y).
    scan_shape : tuple, optional
        If data is flattened (N, det_x, det_y), provide scan dimensions.
    pixel_size : float, optional
        Pixel size in Å (real-space). Used for scale bar.
        Auto-extracted from Dataset4dstem if not provided.
    k_pixel_size : float, optional
        Detector pixel size in mrad (k-space). Used for scale bar.
        Auto-extracted from Dataset4dstem if not provided.
    center : tuple[float, float], optional
        (center_x, center_y) of the diffraction pattern in pixels.
        If not provided, defaults to detector center.
    bf_radius : float, optional
        Bright field disk radius in pixels. If not provided, estimated as 1/8 of detector size.
    precompute_virtual_images : bool, default True
        Precompute BF/ABF/LAADF/HAADF virtual images for preset switching.
    log_scale : bool, default False
        Use log scale for better dynamic range visualization.

    Examples
    --------
    >>> # From Dataset4dstem (calibration auto-extracted)
    >>> from quantem.core.io.file_readers import read_emdfile_to_4dstem
    >>> dataset = read_emdfile_to_4dstem("data.h5")
    >>> Show4DSTEM(dataset)

    >>> # From raw array with manual calibration
    >>> import numpy as np
    >>> data = np.random.rand(64, 64, 128, 128)
    >>> Show4DSTEM(data, pixel_size=2.39, k_pixel_size=0.46)

    >>> # With raster animation
    >>> widget = Show4DSTEM(dataset)
    >>> widget.raster(step=2, interval_ms=50)
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show4dstem.js"
    _css = pathlib.Path(__file__).parent / "static" / "show4dstem.css"

    # Position in scan space
    pos_x = traitlets.Int(0).tag(sync=True)
    pos_y = traitlets.Int(0).tag(sync=True)

    # Shape of scan space (for slider bounds)
    shape_x = traitlets.Int(1).tag(sync=True)
    shape_y = traitlets.Int(1).tag(sync=True)

    # Detector shape for frontend
    det_x = traitlets.Int(1).tag(sync=True)
    det_y = traitlets.Int(1).tag(sync=True)

    # Raw float32 frame as bytes (JS handles scale/colormap for real-time interactivity)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Global min/max for DP normalization (computed once from sampled frames)
    dp_global_min = traitlets.Float(0.0).tag(sync=True)
    dp_global_max = traitlets.Float(1.0).tag(sync=True)

    # =========================================================================
    # Detector Calibration (for presets and scale bar)
    # =========================================================================
    center_x = traitlets.Float(0.0).tag(sync=True)  # Detector center X
    center_y = traitlets.Float(0.0).tag(sync=True)  # Detector center Y
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
    roi_center_x = traitlets.Float(0.0).tag(sync=True)
    roi_center_y = traitlets.Float(0.0).tag(sync=True)
    # Compound trait for batched X+Y updates (JS sends both at once, 1 observer fires)
    roi_center = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
    roi_radius = traitlets.Float(10.0).tag(sync=True)
    roi_radius_inner = traitlets.Float(5.0).tag(sync=True)
    roi_width = traitlets.Float(20.0).tag(sync=True)
    roi_height = traitlets.Float(10.0).tag(sync=True)

    # =========================================================================
    # Virtual Image (ROI-based, updates as you drag ROI on DP)
    # =========================================================================
    virtual_image_bytes = traitlets.Bytes(b"").tag(sync=True)  # Raw float32
    vi_data_min = traitlets.Float(0.0).tag(sync=True)  # Min of current VI for normalization
    vi_data_max = traitlets.Float(1.0).tag(sync=True)  # Max of current VI for normalization

    # =========================================================================
    # VI ROI (real-space region selection for summed DP)
    # =========================================================================
    vi_roi_mode = traitlets.Unicode("off").tag(sync=True)  # "off", "circle", "rect"
    vi_roi_center_x = traitlets.Float(0.0).tag(sync=True)
    vi_roi_center_y = traitlets.Float(0.0).tag(sync=True)
    vi_roi_radius = traitlets.Float(5.0).tag(sync=True)
    vi_roi_width = traitlets.Float(10.0).tag(sync=True)
    vi_roi_height = traitlets.Float(10.0).tag(sync=True)
    summed_dp_bytes = traitlets.Bytes(b"").tag(sync=True)  # Summed DP from VI ROI
    summed_dp_count = traitlets.Int(0).tag(sync=True)  # Number of positions summed

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(1.0).tag(sync=True)  # Å per pixel (real-space)
    k_pixel_size = traitlets.Float(1.0).tag(sync=True)  # mrad per pixel (k-space)
    k_calibrated = traitlets.Bool(False).tag(sync=True)  # True if k-space has mrad calibration

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
    auto_detect_trigger = traitlets.Bool(False).tag(sync=True)

    # =========================================================================
    # Statistics for display (mean, min, max, std)
    # =========================================================================
    dp_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(sync=True)
    vi_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(sync=True)
    mask_dc = traitlets.Bool(True).tag(sync=True)  # Mask center pixel for DP stats

    def __init__(
        self,
        data: "Dataset4dstem | np.ndarray",
        scan_shape: tuple[int, int] | None = None,
        pixel_size: float | None = None,
        k_pixel_size: float | None = None,
        center: tuple[float, float] | None = None,
        bf_radius: float | None = None,
        precompute_virtual_images: bool = False,
        log_scale: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.log_scale = log_scale

        # Extract calibration from Dataset4dstem if provided
        k_calibrated = False
        if hasattr(data, "sampling") and hasattr(data, "array"):
            # Dataset4dstem: extract calibration and array
            # sampling = [scan_x, scan_y, det_x, det_y]
            units = getattr(data, "units", ["pixels"] * 4)
            if pixel_size is None and units[0] in ("Å", "angstrom", "A", "nm"):
                pixel_size = float(data.sampling[0])
                if units[0] == "nm":
                    pixel_size *= 10  # Convert nm to Å
            if k_pixel_size is None and units[2] in ("mrad", "1/Å", "1/A"):
                k_pixel_size = float(data.sampling[2])
                k_calibrated = True
            data = data.array

        # Store calibration values (default to 1.0 if not provided)
        self.pixel_size = pixel_size if pixel_size is not None else 1.0
        self.k_pixel_size = k_pixel_size if k_pixel_size is not None else 1.0
        self.k_calibrated = k_calibrated or (k_pixel_size is not None)
        # Path animation (configured via set_path() or raster())
        self._path_points: list[tuple[int, int]] = []
        # Convert to NumPy then PyTorch tensor using quantem device config
        data_np = to_numpy(data)
        device_str, _ = validate_device(None)  # Get device from quantem config
        self._device = torch.device(device_str)
        self._data = torch.from_numpy(data_np.astype(np.float32)).to(self._device)
        # Remove saturated hot pixels (65535 for uint16, 255 for uint8)
        saturated_value = 65535.0 if data_np.dtype == np.uint16 else 255.0 if data_np.dtype == np.uint8 else None
        if saturated_value is not None:
            self._data[self._data >= saturated_value] = 0
        # Handle flattened data
        if data.ndim == 3:
            if scan_shape is not None:
                self._scan_shape = scan_shape
            else:
                # Infer square scan shape from N
                n = data.shape[0]
                side = int(n ** 0.5)
                if side * side != n:
                    raise ValueError(
                        f"Cannot infer square scan_shape from N={n}. "
                        f"Provide scan_shape explicitly."
                    )
                self._scan_shape = (side, side)
            self._det_shape = (data.shape[1], data.shape[2])
        elif data.ndim == 4:
            self._scan_shape = (data.shape[0], data.shape[1])
            self._det_shape = (data.shape[2], data.shape[3])
        else:
            raise ValueError(f"Expected 3D or 4D array, got {data.ndim}D")

        self.shape_x = self._scan_shape[0]
        self.shape_y = self._scan_shape[1]
        self.det_x = self._det_shape[0]
        self.det_y = self._det_shape[1]
        # Initial position at center
        self.pos_x = self.shape_x // 2
        self.pos_y = self.shape_y // 2
        # Precompute global range for consistent scaling (hot pixels already removed)
        self.dp_global_min = max(float(self._data.min()), MIN_LOG_VALUE)
        self.dp_global_max = float(self._data.max())
        # Cache coordinate tensors for mask creation (avoid repeated torch.arange)
        self._det_row_coords = torch.arange(self.det_x, device=self._device, dtype=torch.float32)[:, None]
        self._det_col_coords = torch.arange(self.det_y, device=self._device, dtype=torch.float32)[None, :]
        self._scan_row_coords = torch.arange(self.shape_x, device=self._device, dtype=torch.float32)[:, None]
        self._scan_col_coords = torch.arange(self.shape_y, device=self._device, dtype=torch.float32)[None, :]
        # Setup center and BF radius
        # If user provides explicit values, use them
        # Otherwise, auto-detect from the data for accurate presets
        det_size = min(self.det_x, self.det_y)
        if center is not None and bf_radius is not None:
            # User provided both - use explicit values
            self.center_x = float(center[0])
            self.center_y = float(center[1])
            self.bf_radius = float(bf_radius)
        elif center is not None:
            # User provided center only - use it with default bf_radius
            self.center_x = float(center[0])
            self.center_y = float(center[1])
            self.bf_radius = det_size * DEFAULT_BF_RATIO
        elif bf_radius is not None:
            # User provided bf_radius only - use detector center
            self.center_x = float(self.det_y / 2)
            self.center_y = float(self.det_x / 2)
            self.bf_radius = float(bf_radius)
        else:
            # Neither provided - auto-detect from data
            # Set defaults first (will be overwritten by auto-detect)
            self.center_x = float(self.det_y / 2)
            self.center_y = float(self.det_x / 2)
            self.bf_radius = det_size * DEFAULT_BF_RATIO
            # Auto-detect center and bf_radius from the data
            self.auto_detect_center(update_roi=False)

        # Pre-compute and cache common virtual images (BF, ABF, ADF)
        # Each cache stores (bytes, stats) tuple
        self._cached_bf_virtual = None
        self._cached_abf_virtual = None
        self._cached_adf_virtual = None
        if precompute_virtual_images:
            self._precompute_common_virtual_images()

        # Update frame when position changes (scale/colormap handled in JS)
        self.observe(self._update_frame, names=["pos_x", "pos_y"])
        # Observe individual ROI params (for backward compatibility)
        self.observe(self._on_roi_change, names=[
            "roi_center_x", "roi_center_y", "roi_radius", "roi_radius_inner",
            "roi_active", "roi_mode", "roi_width", "roi_height"
        ])
        # Observe compound roi_center for batched updates from JS
        self.observe(self._on_roi_center_change, names=["roi_center"])

        # Initialize default ROI at BF center
        self.roi_center_x = self.center_x
        self.roi_center_y = self.center_y
        self.roi_center = [self.center_x, self.center_y]
        self.roi_radius = self.bf_radius * 0.5  # Start with half BF radius
        self.roi_active = True
        
        # Compute initial virtual image and frame
        self._compute_virtual_image_from_roi()
        self._update_frame()
        
        # Path animation: observe index changes from frontend
        self.observe(self._on_path_index_change, names=["path_index"])

        # Auto-detect trigger: observe changes from frontend
        self.observe(self._on_auto_detect_trigger, names=["auto_detect_trigger"])

        # VI ROI: observe changes for summed DP computation
        # Initialize VI ROI center to scan center with reasonable default sizes
        self.vi_roi_center_x = float(self.shape_x / 2)
        self.vi_roi_center_y = float(self.shape_y / 2)
        # Set initial ROI size based on scan dimension
        default_roi_size = max(3, min(self.shape_x, self.shape_y) * DEFAULT_VI_ROI_RATIO)
        self.vi_roi_radius = float(default_roi_size)
        self.vi_roi_width = float(default_roi_size * 2)
        self.vi_roi_height = float(default_roi_size)
        self.observe(self._on_vi_roi_change, names=[
            "vi_roi_mode", "vi_roi_center_x", "vi_roi_center_y",
            "vi_roi_radius", "vi_roi_width", "vi_roi_height"
        ])

    def __repr__(self) -> str:
        k_unit = "mrad" if self.k_calibrated else "px"
        return (
            f"Show4DSTEM(shape=({self.shape_x}, {self.shape_y}, {self.det_x}, {self.det_y}), "
            f"sampling=({self.pixel_size} Å, {self.k_pixel_size} {k_unit}), "
            f"pos=({self.pos_x}, {self.pos_y}))"
        )

    # =========================================================================
    # Convenience Properties
    # =========================================================================

    @property
    def position(self) -> tuple[int, int]:
        """Current scan position as (x, y) tuple."""
        return (self.pos_x, self.pos_y)

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        """Set scan position from (x, y) tuple."""
        self.pos_x, self.pos_y = value

    @property
    def scan_shape(self) -> tuple[int, int]:
        """Scan dimensions as (shape_x, shape_y) tuple."""
        return (self.shape_x, self.shape_y)

    @property
    def detector_shape(self) -> tuple[int, int]:
        """Detector dimensions as (det_x, det_y) tuple."""
        return (self.det_x, self.det_y)

    # =========================================================================
    # Path Animation Methods
    # =========================================================================
    
    def set_path(
        self,
        points: list[tuple[int, int]],
        interval_ms: int = 100,
        loop: bool = True,
        autoplay: bool = True,
    ) -> "Show4DSTEM":
        """
        Set a custom path of scan positions to animate through.

        Parameters
        ----------
        points : list[tuple[int, int]]
            List of (x, y) scan positions to visit.
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
    
    def play(self) -> "Show4DSTEM":
        """Start playing the path animation."""
        if self.path_length > 0:
            self.path_playing = True
        return self
    
    def pause(self) -> "Show4DSTEM":
        """Pause the path animation."""
        self.path_playing = False
        return self
    
    def stop(self) -> "Show4DSTEM":
        """Stop and reset path animation to beginning."""
        self.path_playing = False
        self.path_index = 0
        return self
    
    def goto(self, index: int) -> "Show4DSTEM":
        """Jump to a specific index in the path."""
        if 0 <= index < self.path_length:
            self.path_index = index
        return self
    
    def _on_path_index_change(self, change):
        """Called when path_index changes (from frontend timer)."""
        idx = change["new"]
        if 0 <= idx < len(self._path_points):
            x, y = self._path_points[idx]
            # Clamp to valid range
            self.pos_x = max(0, min(self.shape_x - 1, x))
            self.pos_y = max(0, min(self.shape_y - 1, y))

    def _on_auto_detect_trigger(self, change):
        """Called when auto_detect_trigger is set to True from frontend."""
        if change["new"]:
            self.auto_detect_center()
            # Reset trigger to allow re-triggering
            self.auto_detect_trigger = False

    # =========================================================================
    # Path Animation Patterns
    # =========================================================================

    def raster(
        self,
        step: int = 1,
        bidirectional: bool = False,
        interval_ms: int = 100,
        loop: bool = True,
    ) -> "Show4DSTEM":
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
        for x in range(0, self.shape_x, step):
            row = list(range(0, self.shape_y, step))
            if bidirectional and (x // step % 2 == 1):
                row = row[::-1]  # Alternate direction for snake pattern
            for y in row:
                points.append((x, y))
        return self.set_path(points=points, interval_ms=interval_ms, loop=loop)
    
    # =========================================================================
    # ROI Mode Methods
    # =========================================================================
    
    def roi_circle(self, radius: float | None = None) -> "Show4DSTEM":
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
    
    def roi_point(self) -> "Show4DSTEM":
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

    def roi_square(self, half_size: float | None = None) -> "Show4DSTEM":
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
    ) -> "Show4DSTEM":
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
    ) -> "Show4DSTEM":
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

    def auto_detect_center(self, update_roi: bool = True) -> "Show4DSTEM":
        """
        Automatically detect BF disk center and radius using centroid.

        This method analyzes the summed diffraction pattern to find the
        bright field disk center and estimate its radius. The detected
        values are applied to the widget's calibration (center_x, center_y,
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
        # Sum all diffraction patterns to get average (PyTorch)
        if self._data.ndim == 4:
            summed_dp = self._data.sum(dim=(0, 1))
        else:
            summed_dp = self._data.sum(dim=0)

        # Threshold at mean + std to isolate BF disk
        threshold = summed_dp.mean() + summed_dp.std()
        mask = summed_dp > threshold

        # Avoid division by zero
        total = mask.sum()
        if total == 0:
            return self

        # Calculate centroid using cached coordinate grids
        cx = float((self._det_col_coords * mask).sum() / total)
        cy = float((self._det_row_coords * mask).sum() / total)

        # Estimate radius from mask area (A = pi*r^2)
        radius = float(torch.sqrt(total / torch.pi))

        # Apply detected values
        self.center_x = cx
        self.center_y = cy
        self.bf_radius = radius

        if update_roi:
            # Also update ROI to center
            self.roi_center_x = cx
            self.roi_center_y = cy
            # Recompute cached virtual images with new calibration
            self._precompute_common_virtual_images()

        return self

    def _get_frame(self, x: int, y: int) -> np.ndarray:
        """Get single diffraction frame at position (x, y) as numpy array."""
        if self._data.ndim == 3:
            idx = x * self.shape_y + y
            return self._data[idx].cpu().numpy()
        else:
            return self._data[x, y].cpu().numpy()

    def _update_frame(self, change=None):
        """Send raw float32 frame to frontend (JS handles scale/colormap)."""
        # Get frame as tensor (stays on device)
        if self._data.ndim == 3:
            idx = self.pos_x * self.shape_y + self.pos_y
            frame = self._data[idx]
        else:
            frame = self._data[self.pos_x, self.pos_y]

        # Apply log scale if enabled
        if self.log_scale:
            frame = torch.log1p(frame)

        # Compute stats from frame (optionally mask DC component)
        if self.mask_dc and self.det_x > 3 and self.det_y > 3:
            # Mask center 3x3 region for stats (only for detectors > 3x3)
            cx, cy = self.det_x // 2, self.det_y // 2
            mask = torch.ones_like(frame, dtype=torch.bool)
            mask[max(0, cx-1):cx+2, max(0, cy-1):cy+2] = False
            masked_vals = frame[mask]
            self.dp_stats = [
                float(masked_vals.mean()),
                float(masked_vals.min()),
                float(masked_vals.max()),
                float(masked_vals.std()),
            ]
        else:
            self.dp_stats = [
                float(frame.mean()),
                float(frame.min()),
                float(frame.max()),
                float(frame.std()),
            ]

        # Convert to numpy only for sending bytes to frontend
        self.frame_bytes = frame.cpu().numpy().astype(np.float32).tobytes()

    def _on_roi_change(self, change=None):
        """Recompute virtual image when individual ROI params change.

        This handles legacy setters (setRoiCenterX/Y) from button handlers.
        High-frequency updates use the compound roi_center trait instead.
        """
        if not self.roi_active:
            return
        self._compute_virtual_image_from_roi()

    def _on_roi_center_change(self, change=None):
        """Handle batched roi_center updates from JS (single observer for X+Y).

        This is the fast path for drag operations. JS sends [x, y] as a single
        compound trait, so only one observer fires per mouse move.
        """
        if not self.roi_active:
            return
        if change and "new" in change:
            x, y = change["new"]
            # Sync to individual traits (without triggering _on_roi_change observers)
            self.unobserve(self._on_roi_change, names=["roi_center_x", "roi_center_y"])
            self.roi_center_x = x
            self.roi_center_y = y
            self.observe(self._on_roi_change, names=["roi_center_x", "roi_center_y"])
        self._compute_virtual_image_from_roi()

    def _on_vi_roi_change(self, change=None):
        """Compute summed DP when VI ROI changes."""
        if self.vi_roi_mode == "off":
            self.summed_dp_bytes = b""
            self.summed_dp_count = 0
            return
        self._compute_summed_dp_from_vi_roi()

    def _compute_summed_dp_from_vi_roi(self):
        """Sum diffraction patterns from positions inside VI ROI (PyTorch)."""
        # Create mask in scan space using cached coordinates
        if self.vi_roi_mode == "circle":
            mask = (self._scan_row_coords - self.vi_roi_center_x) ** 2 + (self._scan_col_coords - self.vi_roi_center_y) ** 2 <= self.vi_roi_radius ** 2
        elif self.vi_roi_mode == "square":
            half_size = self.vi_roi_radius
            mask = (torch.abs(self._scan_row_coords - self.vi_roi_center_x) <= half_size) & (torch.abs(self._scan_col_coords - self.vi_roi_center_y) <= half_size)
        elif self.vi_roi_mode == "rect":
            half_w = self.vi_roi_width / 2
            half_h = self.vi_roi_height / 2
            mask = (torch.abs(self._scan_row_coords - self.vi_roi_center_x) <= half_h) & (torch.abs(self._scan_col_coords - self.vi_roi_center_y) <= half_w)
        else:
            return

        # Count positions in mask
        n_positions = int(mask.sum())
        if n_positions == 0:
            self.summed_dp_bytes = b""
            self.summed_dp_count = 0
            return

        self.summed_dp_count = n_positions

        # Compute average DP using masked sum (vectorized)
        if self._data.ndim == 4:
            # (scan_x, scan_y, det_x, det_y) - sum over masked scan positions
            avg_dp = self._data[mask].mean(dim=0)
        else:
            # Flattened: (N, det_x, det_y) - need to convert mask indices
            flat_indices = torch.nonzero(mask.flatten(), as_tuple=True)[0]
            avg_dp = self._data[flat_indices].mean(dim=0)

        # Normalize to 0-255 for display
        vmin, vmax = float(avg_dp.min()), float(avg_dp.max())
        if vmax > vmin:
            normalized = torch.clamp((avg_dp - vmin) / (vmax - vmin) * 255, 0, 255)
            normalized = normalized.cpu().numpy().astype(np.uint8)
        else:
            normalized = np.zeros((self.det_x, self.det_y), dtype=np.uint8)

        self.summed_dp_bytes = normalized.tobytes()

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

    def _precompute_common_virtual_images(self):
        """Pre-compute BF/ABF/ADF virtual images for instant preset switching."""
        cx, cy, bf = self.center_x, self.center_y, self.bf_radius
        # Cache (bytes, stats, min, max) for each preset
        bf_arr = self._fast_masked_sum(self._create_circular_mask(cx, cy, bf))
        abf_arr = self._fast_masked_sum(self._create_annular_mask(cx, cy, bf * 0.5, bf))
        adf_arr = self._fast_masked_sum(self._create_annular_mask(cx, cy, bf, bf * 4.0))

        self._cached_bf_virtual = (
            self._to_float32_bytes(bf_arr, update_vi_stats=False),
            [float(bf_arr.mean()), float(bf_arr.min()), float(bf_arr.max()), float(bf_arr.std())],
            float(bf_arr.min()), float(bf_arr.max())
        )
        self._cached_abf_virtual = (
            self._to_float32_bytes(abf_arr, update_vi_stats=False),
            [float(abf_arr.mean()), float(abf_arr.min()), float(abf_arr.max()), float(abf_arr.std())],
            float(abf_arr.min()), float(abf_arr.max())
        )
        self._cached_adf_virtual = (
            self._to_float32_bytes(adf_arr, update_vi_stats=False),
            [float(adf_arr.mean()), float(adf_arr.min()), float(adf_arr.max()), float(adf_arr.std())],
            float(adf_arr.min()), float(adf_arr.max())
        )

    def _get_cached_preset(self) -> tuple[bytes, list[float], float, float] | None:
        """Check if current ROI matches a cached preset and return (bytes, stats, min, max) tuple."""
        # Must be centered on detector center
        if abs(self.roi_center_x - self.center_x) >= 1 or abs(self.roi_center_y - self.center_y) >= 1:
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

        # ADF: annular at bf to 4*bf (combines LAADF + HAADF)
        if (self.roi_mode == "annular" and
            abs(self.roi_radius_inner - bf) < 1 and
            abs(self.roi_radius - bf * 4.0) < 1):
            return self._cached_adf_virtual

        return None

    def _fast_masked_sum(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked sum using PyTorch.

        Uses sparse indexing for small masks (<20% coverage) which is faster
        because it only processes non-zero pixels:
        - r=10 (1%): ~0.8ms (sparse) vs ~13ms (full)
        - r=30 (8%): ~4ms (sparse) vs ~13ms (full)

        For large masks (≥20%), uses full tensordot which has constant ~13ms.
        """
        mask_float = mask.float()
        n_det = self._det_shape[0] * self._det_shape[1]
        n_nonzero = int(mask.sum())
        coverage = n_nonzero / n_det

        if coverage < SPARSE_MASK_THRESHOLD:
            # Sparse: faster for small masks
            indices = torch.nonzero(mask_float.flatten(), as_tuple=True)[0]
            n_scan = self._scan_shape[0] * self._scan_shape[1]
            data_flat = self._data.reshape(n_scan, n_det)
            result = data_flat[:, indices].sum(dim=1).reshape(self._scan_shape)
        else:
            # Tensordot: faster for large masks
            result = torch.tensordot(self._data, mask_float, dims=([2, 3], [0, 1]))

        return result

    def _to_float32_bytes(self, arr: torch.Tensor, update_vi_stats: bool = True) -> bytes:
        """Convert tensor to float32 bytes."""
        # Compute min/max (fast on GPU)
        vmin = float(arr.min())
        vmax = float(arr.max())
        self.vi_data_min = vmin
        self.vi_data_max = vmax

        # Compute full stats if requested
        if update_vi_stats:
            self.vi_stats = [float(arr.mean()), vmin, vmax, float(arr.std())]

        return arr.cpu().numpy().astype(np.float32).tobytes()

    def _compute_virtual_image_from_roi(self):
        """Compute virtual image based on ROI mode."""
        cached = self._get_cached_preset()
        if cached is not None:
            # Cached preset returns (bytes, stats, min, max) tuple
            vi_bytes, vi_stats, vi_min, vi_max = cached
            self.virtual_image_bytes = vi_bytes
            self.vi_stats = vi_stats
            self.vi_data_min = vi_min
            self.vi_data_max = vi_max
            return

        cx, cy = self.roi_center_x, self.roi_center_y

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
            if self._data.ndim == 4:
                virtual_image = self._data[:, :, row, col]
            else:
                virtual_image = self._data[:, row, col].reshape(self._scan_shape)
            self.virtual_image_bytes = self._to_float32_bytes(virtual_image)
            return

        self.virtual_image_bytes = self._to_float32_bytes(self._fast_masked_sum(mask))
