"""
show4d: Fast interactive 4D-STEM viewer widget with advanced features.

Features:
- Binary transfer (no base64 overhead)
- Live statistics panel (mean/max/min)
- Virtual detector overlays (BF/ADF circles)
- Linked scan view (side-by-side)
- Auto-range with percentile scaling
- ROI drawing tools
"""

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy

if TYPE_CHECKING:
    from quantem.detector import Detector


# Detector geometry constants (ratios of detector size)
DEFAULT_BF_RATIO = 0.125  # 1/8 of detector size
DEFAULT_ADF_INNER_RATIO = 0.1875  # 1.5 * BF = 3/16 of detector
DEFAULT_ADF_OUTER_RATIO = 0.375  # 3 * BF = 3/8 of detector


class Show4DSTEM(anywidget.AnyWidget):
    """
    Fast interactive 4D-STEM viewer with advanced features.

    Optimized for speed with binary transfer and pre-normalization.
    Works with NumPy and PyTorch arrays.

    Parameters
    ----------
    data : array_like
        4D array of shape (scan_x, scan_y, det_x, det_y).
        Supports NumPy and PyTorch arrays.
    scan_shape : tuple, optional
        If data is flattened (N, det_x, det_y), provide scan dimensions.
    detector : Detector, optional
        Detector object from quantem.detector for automatic calibration.
        If provided, center and bf_radius are extracted from the detector.
    pixel_size : float, default 1.0
        Pixel size in nm (real-space). Used for scale bar.
    det_pixel_size : float, default 1.0
        Detector pixel size in mrad (k-space). Used for scale bar.
    center : tuple[float, float], optional
        (center_x, center_y) of the diffraction pattern in pixels.
        If not provided, defaults to detector center.
    bf_radius : float, optional
        Bright field disk radius in pixels. If not provided, estimated as 1/8 of detector size.
    log_scale : bool, default True
        Use log scale for better dynamic range visualization.
    auto_range : bool, default False
        Use percentile-based scaling instead of global min/max.
    percentile_low : float, default 1.0
        Lower percentile for auto-range (0-100).
    percentile_high : float, default 99.0
        Upper percentile for auto-range (0-100).
    path_points : list[tuple[int, int]], optional
        List of (x, y) scan positions for programmatic animation.
        Use with play(), pause(), stop() methods.
    path_interval_ms : int, default 100
        Time between frames in path animation (milliseconds).
    path_loop : bool, default True
        Whether to loop when path animation reaches the end.

    Examples
    --------
    >>> from quantem.widget import Show4DSTEM
    >>> import numpy as np
    >>> data = np.random.rand(64, 64, 128, 128)
    >>> Show4DSTEM(data)

    >>> # With manual calibration
    >>> Show4DSTEM(data, pixel_size=0.5, det_pixel_size=0.1, bf_radius=20)

    >>> # With Detector object (optional dependency)
    >>> from quantem.detector import Detector
    >>> det = Detector("data.h5")
    >>> Show4DSTEM(det.data, detector=det)  # Uses det.center, det.bf_radius
    
    >>> # With path animation
    >>> path = [(i, i) for i in range(64)]  # Diagonal path
    >>> widget = Show4DSTEM(data, path_points=path, path_interval_ms=50)
    >>> widget.play()  # Start animation
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

    # Pre-normalized uint8 frame as bytes (no base64!)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Log scale toggle
    log_scale = traitlets.Bool(True).tag(sync=True)

    # =========================================================================
    # Stats Panel
    # =========================================================================
    stats_mean = traitlets.Float(0.0).tag(sync=True)
    stats_max = traitlets.Float(0.0).tag(sync=True)
    stats_min = traitlets.Float(0.0).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)

    # =========================================================================
    # Detector Integration (BF/ADF overlays)
    # =========================================================================
    has_detector = traitlets.Bool(False).tag(sync=True)
    center_x = traitlets.Float(0.0).tag(sync=True)
    center_y = traitlets.Float(0.0).tag(sync=True)
    bf_radius = traitlets.Float(0.0).tag(sync=True)
    show_bf_overlay = traitlets.Bool(True).tag(sync=True)
    show_adf_overlay = traitlets.Bool(False).tag(sync=True)
    adf_inner_radius = traitlets.Float(0.0).tag(sync=True)
    adf_outer_radius = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # Linked Scan View
    # =========================================================================
    show_scan_view = traitlets.Bool(False).tag(sync=True)
    scan_mode = traitlets.Unicode("bf").tag(sync=True)  # 'bf', 'adf', 'custom'
    scan_image_bytes = traitlets.Bytes(b"").tag(sync=True)

    # =========================================================================
    # Auto-Range (percentile scaling)
    # =========================================================================
    auto_range = traitlets.Bool(False).tag(sync=True)
    percentile_low = traitlets.Float(1.0).tag(sync=True)
    percentile_high = traitlets.Float(99.0).tag(sync=True)

    # =========================================================================
    # ROI Drawing (for virtual imaging)
    # =========================================================================
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_mode = traitlets.Unicode("point").tag(sync=True)  # 'point', 'circle', 'square', 'rect', or 'annular'
    roi_center_x = traitlets.Float(0.0).tag(sync=True)
    roi_center_y = traitlets.Float(0.0).tag(sync=True)
    roi_radius = traitlets.Float(10.0).tag(sync=True)  # Outer radius for circle/annular, half-width for square
    roi_radius_inner = traitlets.Float(5.0).tag(sync=True)  # Inner radius for annular mode
    roi_width = traitlets.Float(20.0).tag(sync=True)  # Width for rectangular mode
    roi_height = traitlets.Float(10.0).tag(sync=True)  # Height for rectangular mode
    roi_integrated_value = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # Mean Diffraction Pattern
    # =========================================================================
    mean_dp_bytes = traitlets.Bytes(b"").tag(sync=True)
    show_mean_dp = traitlets.Bool(True).tag(sync=True)

    # =========================================================================
    # BF Image (Bright Field integrated image)
    # =========================================================================
    bf_image_bytes = traitlets.Bytes(b"").tag(sync=True)
    show_bf_image = traitlets.Bool(True).tag(sync=True)

    # =========================================================================
    # Virtual Image (ROI-based, updates as you drag ROI on DP)
    # =========================================================================
    virtual_image_bytes = traitlets.Bytes(b"").tag(sync=True)

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(1.0).tag(sync=True)  # nm per pixel (real-space)
    det_pixel_size = traitlets.Float(1.0).tag(sync=True)  # mrad per pixel (k-space)

    # =========================================================================
    # Path Animation (programmatic crosshair control)
    # =========================================================================
    path_playing = traitlets.Bool(False).tag(sync=True)
    path_index = traitlets.Int(0).tag(sync=True)
    path_length = traitlets.Int(0).tag(sync=True)
    path_interval_ms = traitlets.Int(100).tag(sync=True)  # ms between frames
    path_loop = traitlets.Bool(True).tag(sync=True)  # loop when reaching end

    def __init__(
        self,
        data,
        scan_shape: tuple[int, int] | None = None,
        detector: "Detector | None" = None,
        pixel_size: float = 1.0,
        det_pixel_size: float = 1.0,
        center: tuple[float, float] | None = None,
        bf_radius: float | None = None,
        log_scale: bool = True,
        auto_range: bool = False,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        path_points: list[tuple[int, int]] | None = None,
        path_interval_ms: int = 100,
        path_loop: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._log_scale = log_scale
        self.log_scale = log_scale
        self.auto_range = auto_range
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        
        # Store calibration values
        self.pixel_size = pixel_size
        self.det_pixel_size = det_pixel_size
        
        # Path animation settings
        self._path_points: list[tuple[int, int]] = path_points or []
        self.path_length = len(self._path_points)
        self.path_interval_ms = path_interval_ms
        self.path_loop = path_loop

        # Convert to NumPy
        self._data = to_numpy(data)

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

        # Precompute global range for consistent scaling
        self._compute_global_range()

        # Setup center and BF radius: detector > user params > defaults
        if detector is not None:
            # Use Detector object for calibration
            det_center = detector.center
            self.center_x = float(det_center[0])
            self.center_y = float(det_center[1])
            self.bf_radius = float(detector.bf_radius)
        elif center is not None:
            # Use user-provided center
            self.center_x = float(center[0])
            self.center_y = float(center[1])
            det_size = min(self.det_x, self.det_y)
            self.bf_radius = float(bf_radius) if bf_radius is not None else det_size * DEFAULT_BF_RATIO
        else:
            # Default to detector center
            self.center_x = float(self.det_y / 2)
            self.center_y = float(self.det_x / 2)
            det_size = min(self.det_x, self.det_y)
            self.bf_radius = float(bf_radius) if bf_radius is not None else det_size * DEFAULT_BF_RATIO
        
        # Set ADF defaults based on detector size
        det_size = min(self.det_x, self.det_y)
        self.adf_inner_radius = det_size * DEFAULT_ADF_INNER_RATIO
        self.adf_outer_radius = det_size * DEFAULT_ADF_OUTER_RATIO
        self.has_detector = True

        # Compute mean DP and BF image (sent once on init)
        self._compute_mean_dp()
        self._compute_bf_image()
        
        # Pre-compute and cache common virtual images (BF, ABF, LAADF, HAADF)
        self._cached_bf_virtual = None
        self._cached_abf_virtual = None
        self._cached_laadf_virtual = None
        self._cached_haadf_virtual = None
        self._precompute_common_virtual_images()
        
        # Update frame when position or settings change
        self.observe(self._update_frame, names=[
            "pos_x", "pos_y", "log_scale", "auto_range",
            "percentile_low", "percentile_high"
        ])
        self.observe(self._on_roi_change, names=[
            "roi_center_x", "roi_center_y", "roi_radius", "roi_radius_inner", 
            "roi_active", "roi_mode", "roi_width", "roi_height"
        ])
        self.observe(self._on_scan_mode_change, names=["scan_mode", "show_scan_view"])
        
        # Initialize default ROI at BF center
        self.roi_center_x = self.center_x
        self.roi_center_y = self.center_y
        self.roi_radius = self.bf_radius * 0.5  # Start with half BF radius
        self.roi_active = True
        
        # Compute initial virtual image
        try:
            self._compute_virtual_image_from_roi()
        except Exception:
            pass
        
        self._update_frame()
        
        # Path animation: observe index changes from frontend
        self.observe(self._on_path_index_change, names=["path_index"])

    # =========================================================================
    # Array Utilities
    # =========================================================================
    
    @staticmethod
    def _to_cpu(arr):
        """Convert array to CPU (NumPy)."""
        return np.asarray(arr)
    
    @staticmethod
    def _to_scalar(val):
        """Convert scalar value to Python float."""
        return float(val)

    # =========================================================================
    # Path Animation Methods
    # =========================================================================
    
    def set_path(
        self,
        points: list[tuple[int, int]] | None = None,
        generator: "Callable[[int, int, int], tuple[int, int]] | None" = None,
        n_frames: int | None = None,
        interval_ms: int | None = None,
        loop: bool | None = None,
        autoplay: bool = True,
    ) -> "Show4DSTEM":
        """
        Set a path of scan positions to animate through.
        
        You can provide either a list of points OR a generator function.
        
        Parameters
        ----------
        points : list[tuple[int, int]], optional
            List of (x, y) scan positions to visit.
        generator : callable, optional
            Custom function with signature `f(index, shape_x, shape_y) -> (x, y)`.
            Called for each frame to get the next position.
        n_frames : int, optional
            Number of frames when using generator. Required if using generator.
        interval_ms : int, optional
            Time between frames in milliseconds. Default 100ms.
        loop : bool, optional
            Whether to loop when reaching end. Default True.
        autoplay : bool, default True
            Start playing immediately.
            
        Returns
        -------
        Show4DSTEM
            Self for method chaining.
            
        Examples
        --------
        >>> # Option 1: List of points
        >>> path = [(0, 0), (10, 10), (20, 20), (30, 30)]
        >>> widget.set_path(points=path)
        
        >>> # Option 2: Custom generator function
        >>> def my_path(i, sx, sy):
        ...     # Random walk
        ...     import random
        ...     return (random.randint(0, sx-1), random.randint(0, sy-1))
        >>> widget.set_path(generator=my_path, n_frames=100)
        
        >>> # Option 3: Lambda for quick patterns
        >>> widget.set_path(
        ...     generator=lambda i, sx, sy: (i % sx, (i * 3) % sy),
        ...     n_frames=200
        ... )
        """
        if generator is not None:
            # Use generator function to create points
            if n_frames is None:
                n_frames = 100  # Default
            self._path_points = [
                generator(i, self.shape_x, self.shape_y) 
                for i in range(n_frames)
            ]
        elif points is not None:
            self._path_points = list(points)
        else:
            raise ValueError("Must provide either 'points' or 'generator'")
            
        self.path_length = len(self._path_points)
        self.path_index = 0
        
        if interval_ms is not None:
            self.path_interval_ms = interval_ms
        if loop is not None:
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

    # =========================================================================
    # Path Animation Patterns
    # =========================================================================
    
    def play_raster(self, step: int = 1, bidirectional: bool = False) -> "Show4DSTEM":
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
        return self.set_path(points=points)
    
    # =========================================================================
    # ROI Mode Methods
    # =========================================================================
    
    def set_roi_circle(self, radius: float | None = None) -> "Show4DSTEM":
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
        >>> widget.set_roi_circle(20)  # 20px radius circle
        >>> widget.set_roi_circle()    # Use default radius
        """
        self.roi_mode = "circle"
        if radius is not None:
            self.roi_radius = float(radius)
        return self
    
    def set_roi_point(self) -> "Show4DSTEM":
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

    def set_roi_square(self, size: float | None = None) -> "Show4DSTEM":
        """
        Switch to square ROI mode for virtual imaging.
        
        In square mode, the virtual image integrates over a square region
        centered at the current ROI position.
        
        Parameters
        ----------
        size : float, optional
            Half-size of the square in pixels (distance from center to edge).
            If not provided, uses current roi_radius value.
            
        Returns
        -------
        Show4DSTEM
            Self for method chaining.
            
        Examples
        --------
        >>> widget.set_roi_square(15)  # 30x30 pixel square
        >>> widget.set_roi_square()    # Use default size
        """
        self.roi_mode = "square"
        if size is not None:
            self.roi_radius = float(size)
        return self

    def set_roi_annular(
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
        >>> widget.set_roi_annular(20, 50)  # ADF: inner=20px, outer=50px
        >>> widget.set_roi_annular(30, 80)  # HAADF: larger angles
        """
        self.roi_mode = "annular"
        if inner_radius is not None:
            self.roi_radius_inner = float(inner_radius)
        if outer_radius is not None:
            self.roi_radius = float(outer_radius)
        return self

    def set_roi_rect(
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
        >>> widget.set_roi_rect(30, 20)  # 30px wide, 20px tall
        >>> widget.set_roi_rect(40, 40)  # 40x40 rectangle
        """
        self.roi_mode = "rect"
        if width is not None:
            self.roi_width = float(width)
        if height is not None:
            self.roi_height = float(height)
        return self

    def _compute_global_range(self):
        """Compute global min/max from sampled frames for consistent scaling."""
        
        # Sample corners and center
        samples = [
            (0, 0),
            (0, self.shape_y - 1),
            (self.shape_x - 1, 0),
            (self.shape_x - 1, self.shape_y - 1),
            (self.shape_x // 2, self.shape_y // 2),
        ]
        
        all_min, all_max = float("inf"), float("-inf")
        all_values = []
        for x, y in samples:
            frame = self._get_frame(x, y)
            fmin = float(frame.min())
            fmax = float(frame.max())
            all_min = min(all_min, fmin)
            all_max = max(all_max, fmax)
            
            # Sample values for percentile estimation
            all_values.append(self._to_cpu(frame).flatten()[::100])
        
        self._global_min = max(all_min, 1e-10)
        self._global_max = all_max
        
        # Precompute log range
        self._log_min = np.log1p(self._global_min)
        self._log_max = np.log1p(self._global_max)
        
        # Store sampled values for percentile computation
        self._sampled_values = np.concatenate(all_values)

    def _get_frame(self, x: int, y: int):
        """Get single diffraction frame at position (x, y)."""
        if self._data.ndim == 3:
            idx = x * self.shape_y + y
            return self._data[idx]
        else:
            return self._data[x, y]

    def _compute_percentile_range(self, frame):
        """Compute percentile-based range for a frame."""
        
        # Use NumPy for percentile (faster for small arrays)
        frame_np = self._to_cpu(frame).flatten()
        
        vmin = float(np.percentile(frame_np, self.percentile_low))
        vmax = float(np.percentile(frame_np, self.percentile_high))
        return max(vmin, 1e-10), vmax

    def _update_frame(self, change=None):
        """Send pre-normalized uint8 frame to frontend with stats."""
        frame = self._get_frame(self.pos_x, self.pos_y)

        # Compute stats
        self.stats_mean = self._to_scalar(frame.mean())
        self.stats_max = self._to_scalar(frame.max())
        self.stats_min = self._to_scalar(frame.min())

        # Determine value range
        if self.auto_range:
            vmin, vmax = self._compute_percentile_range(frame)
            if self.log_scale:
                vmin = np.log1p(vmin)
                vmax = np.log1p(vmax)
        else:
            if self.log_scale:
                vmin, vmax = self._log_min, self._log_max
            else:
                vmin, vmax = self._global_min, self._global_max

        # Apply log scale if enabled
        if self.log_scale:
            frame = np.log1p(frame.astype(np.float32))
        else:
            frame = frame.astype(np.float32)

        # Normalize to 0-255
        if vmax > vmin:
            normalized = np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255)
            normalized = normalized.astype(np.uint8)
        else:
            normalized = np.zeros(frame.shape, dtype=np.uint8)

        # Send as raw bytes (no base64 encoding!)
        self.frame_bytes = normalized.tobytes()

    def _on_roi_change(self, change=None):
        """Compute integrated value when ROI changes."""
        # Skip if ROI is not active or has no valid size
        if not self.roi_active:
            self.roi_integrated_value = 0.0
            return
        
        # For circle/square/annular modes, need positive radius
        if self.roi_mode in ("circle", "square", "annular") and self.roi_radius <= 0:
            self.roi_integrated_value = 0.0
            return
        
        frame = self._get_frame(self.pos_x, self.pos_y)
        
        # Create mask based on ROI mode
        if self.roi_mode == "circle" and self.roi_radius > 0:
            mask = self._create_circular_mask(
                self.roi_center_x, self.roi_center_y, self.roi_radius
            )
        elif self.roi_mode == "square" and self.roi_radius > 0:
            mask = self._create_square_mask(
                self.roi_center_x, self.roi_center_y, self.roi_radius
            )
        elif self.roi_mode == "annular" and self.roi_radius > 0:
            mask = self._create_annular_mask(
                self.roi_center_x, self.roi_center_y,
                self.roi_radius_inner, self.roi_radius
            )
        elif self.roi_mode == "rect" and self.roi_width > 0 and self.roi_height > 0:
            mask = self._create_rect_mask(
                self.roi_center_x, self.roi_center_y,
                self.roi_width / 2, self.roi_height / 2
            )
        else:
            # Point mode: no mask, just single pixel
            self.roi_integrated_value = 0.0
            self._compute_virtual_image_from_roi()
            return
        
        # Compute integrated value (use multiplication to avoid indexing issues)
        integrated = self._to_scalar((frame * mask).sum())
        self.roi_integrated_value = integrated
        
        # Fast path: check if we can use cached preset
        cached = self._get_cached_preset()
        if cached is not None:
            self.virtual_image_bytes = cached
            return
        
        # Real-time update using fast masked sum
        self._compute_virtual_image_from_roi()

    def _on_scan_mode_change(self, change=None):
        """Recompute scan image when mode changes."""
        if self.show_scan_view and self.has_detector:
            self._compute_scan_image()

    def _compute_scan_image(self):
        """Compute virtual detector image (BF or ADF)."""
        
        # Get appropriate mask
        if self.scan_mode == "bf":
            mask = self._create_circular_mask(
                self.center_x, self.center_y, self.bf_radius
            )
        elif self.scan_mode == "adf":
            mask = self._create_annular_mask(
                self.center_x, self.center_y,
                self.adf_inner_radius, self.adf_outer_radius
            )
        else:
            # Custom ROI mask
            if self.roi_active and self.roi_radius > 0:
                mask = self._create_circular_mask(
                    self.roi_center_x, self.roi_center_y, self.roi_radius
                )
            else:
                return
        
        # Compute integrated image
        if self._data.ndim == 4:
            # (Rx, Ry, Qx, Qy) -> Apply mask and sum
            scan_image = (self._data * mask).sum(axis=(-2, -1))
        else:
            # (N, Qx, Qy) -> reshape and sum
            scan_image = (self._data * mask).sum(axis=(-2, -1))
            scan_image = scan_image.reshape(self._scan_shape)
        
        # Normalize to uint8
        smin = self._to_scalar(scan_image.min())
        smax = self._to_scalar(scan_image.max())
        
        if smax > smin:
            normalized = np.clip((scan_image - smin) / (smax - smin) * 255, 0, 255)
            normalized = normalized.astype(np.uint8)
        else:
            normalized = np.zeros(scan_image.shape, dtype=np.uint8)
        
        normalized = self._to_cpu(normalized)
        
        self.scan_image_bytes = normalized.tobytes()

    def _create_circular_mask(self, cx: float, cy: float, radius: float):
        """Create circular mask (boolean)."""
        y, x = np.ogrid[:self.det_x, :self.det_y]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        return mask

    def _create_square_mask(self, cx: float, cy: float, half_size: float):
        """Create square mask (boolean)."""
        y, x = np.ogrid[:self.det_x, :self.det_y]
        mask = (np.abs(x - cx) <= half_size) & (np.abs(y - cy) <= half_size)
        return mask

    def _create_annular_mask(
        self, cx: float, cy: float, inner: float, outer: float
    ):
        """Create annular (donut) mask (boolean)."""
        y, x = np.ogrid[:self.det_x, :self.det_y]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        mask = (dist_sq >= inner ** 2) & (dist_sq <= outer ** 2)
        return mask

    def _create_rect_mask(self, cx: float, cy: float, half_width: float, half_height: float):
        """Create rectangular mask (boolean)."""
        y, x = np.ogrid[:self.det_x, :self.det_y]
        mask = (np.abs(x - cx) <= half_width) & (np.abs(y - cy) <= half_height)
        return mask

    def _compute_mean_dp(self):
        """Compute and send mean diffraction pattern."""
        if self._data.ndim == 4:
            mean_dp = self._data.mean(axis=(0, 1))
        else:
            mean_dp = self._data.mean(axis=0)
        # Log scale
        mean_dp = np.log1p(mean_dp)
        # Normalize to uint8
        mean_dp_cpu = self._to_cpu(mean_dp)
        vmin, vmax = float(mean_dp_cpu.min()), float(mean_dp_cpu.max())
        if vmax > vmin:
            normalized = np.clip((mean_dp_cpu - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        else:
            normalized = np.zeros(mean_dp_cpu.shape, dtype=np.uint8)
        self.mean_dp_bytes = normalized.tobytes()

    def _compute_bf_image(self):
        """Compute BF integrated image using detected probe."""
        
        # Create BF mask
        mask = self._create_circular_mask(self.center_x, self.center_y, self.bf_radius)
        
        # Compute integrated BF image
        if self._data.ndim == 4:
            bf_image = (self._data * mask).sum(axis=(-2, -1))
        else:
            bf_image = (self._data * mask).sum(axis=(-2, -1))
            bf_image = bf_image.reshape(self._scan_shape)
        
        # Normalize to uint8
        vmin = self._to_scalar(bf_image.min())
        vmax = self._to_scalar(bf_image.max())
        
        if vmax > vmin:
            normalized = np.clip((bf_image - vmin) / (vmax - vmin) * 255, 0, 255)
            normalized = normalized.astype(np.uint8)
        else:
            normalized = np.zeros(bf_image.shape, dtype=np.uint8)
        
        normalized = self._to_cpu(normalized)
        
        self.bf_image_bytes = normalized.tobytes()

    def _precompute_common_virtual_images(self):
        """Pre-compute BF/ABF/LAADF/HAADF virtual images for instant mode switching."""
        
        def _compute_and_normalize(mask):
            if self._data.ndim == 4:
                img = (self._data * mask).sum(axis=(-2, -1))
            else:
                img = (self._data * mask).sum(axis=(-2, -1)).reshape(self._scan_shape)
            vmin, vmax = self._to_scalar(img.min()), self._to_scalar(img.max())
            if vmax > vmin:
                norm = np.clip((img - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
            else:
                norm = np.zeros(img.shape, dtype=np.uint8)
            return self._to_cpu(norm).tobytes()
        
        # BF: circle at bf_radius
        bf_mask = self._create_circular_mask(self.center_x, self.center_y, self.bf_radius)
        self._cached_bf_virtual = _compute_and_normalize(bf_mask)
        
        # ABF: annular at 0.5*bf to bf (matches JS button)
        abf_mask = self._create_annular_mask(
            self.center_x, self.center_y, self.bf_radius * 0.5, self.bf_radius
        )
        self._cached_abf_virtual = _compute_and_normalize(abf_mask)
        
        # LAADF: annular at bf to 2*bf (matches JS button)
        laadf_mask = self._create_annular_mask(
            self.center_x, self.center_y, self.bf_radius, self.bf_radius * 2.0
        )
        self._cached_laadf_virtual = _compute_and_normalize(laadf_mask)
        
        # HAADF: annular at 2*bf to 4*bf (matches JS button)
        haadf_mask = self._create_annular_mask(
            self.center_x, self.center_y, self.bf_radius * 2.0, self.bf_radius * 4.0
        )
        self._cached_haadf_virtual = _compute_and_normalize(haadf_mask)

    def _get_cached_preset(self) -> bytes | None:
        """Check if current ROI matches a cached preset and return it."""
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
        
        # LAADF: annular at bf to 2*bf
        if (self.roi_mode == "annular" and 
            abs(self.roi_radius_inner - bf) < 1 and 
            abs(self.roi_radius - bf * 2.0) < 1):
            return self._cached_laadf_virtual
        
        # HAADF: annular at 2*bf to 4*bf
        if (self.roi_mode == "annular" and 
            abs(self.roi_radius_inner - bf * 2.0) < 1 and 
            abs(self.roi_radius - bf * 4.0) < 1):
            return self._cached_haadf_virtual
        
        return None

    def _fast_masked_sum(self, mask) -> 'np.ndarray':
        """Fast masked sum using element-wise multiply (memory efficient)."""
        # Handle both 3D and 4D data
        if self._data.ndim == 4:
            # (scan_x, scan_y, det_x, det_y) -> sum over detector dims
            virtual_image = (self._data.astype(np.float32) * mask).sum(axis=(2, 3))
        else:
            # (N, det_x, det_y) -> sum over detector dims then reshape
            virtual_image = (self._data.astype(np.float32) * mask).sum(axis=(1, 2))
            virtual_image = virtual_image.reshape(self._scan_shape)
        return virtual_image

    def _compute_virtual_image_from_roi(self):
        """Compute virtual image based on ROI mode (point, circle, square, or annular)."""
        
        # Fast path: use cached images for presets (BF/ABF/LAADF/HAADF)
        cached = self._get_cached_preset()
        if cached is not None:
            self.virtual_image_bytes = cached
            return
        
        if self.roi_mode == "circle" and self.roi_radius > 0:
            mask = self._create_circular_mask(
                self.roi_center_x, self.roi_center_y, self.roi_radius
            )
            virtual_image = self._fast_masked_sum(mask)
        elif self.roi_mode == "square" and self.roi_radius > 0:
            mask = self._create_square_mask(
                self.roi_center_x, self.roi_center_y, self.roi_radius
            )
            virtual_image = self._fast_masked_sum(mask)
        elif self.roi_mode == "annular" and self.roi_radius > 0 and self.roi_radius_inner >= 0:
            mask = self._create_annular_mask(
                self.roi_center_x, self.roi_center_y, 
                self.roi_radius_inner, self.roi_radius
            )
            virtual_image = self._fast_masked_sum(mask)
        elif self.roi_mode == "rect" and self.roi_width > 0 and self.roi_height > 0:
            mask = self._create_rect_mask(
                self.roi_center_x, self.roi_center_y,
                self.roi_width / 2, self.roi_height / 2
            )
            virtual_image = self._fast_masked_sum(mask)
        else:
            # Point mode: single-pixel indexing - O(1) on GPU!
            kx = int(max(0, min(self._det_shape[0] - 1, round(self.roi_center_y))))
            ky = int(max(0, min(self._det_shape[1] - 1, round(self.roi_center_x))))
            
            if self._data.ndim == 4:
                virtual_image = self._data[:, :, kx, ky]
            else:
                virtual_image = self._data[:, kx, ky].reshape(self._scan_shape)
        
        # Normalize to uint8
        vmin = self._to_scalar(virtual_image.min())
        vmax = self._to_scalar(virtual_image.max())
        
        if vmax > vmin:
            normalized = np.clip((virtual_image - vmin) / (vmax - vmin) * 255, 0, 255)
            normalized = normalized.astype(np.uint8)
        else:
            normalized = np.zeros(virtual_image.shape, dtype=np.uint8)
        
        normalized_cpu = self._to_cpu(normalized)
        
        self.virtual_image_bytes = normalized_cpu.tobytes()
