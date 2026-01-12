"""
show4d: Fast interactive 4D-STEM viewer widget with advanced features.

Features:
- Binary transfer (no base64 overhead)
- Live statistics panel (mean/max/min)
- Virtual detector overlays (BF/ADF circles)
- Linked scan view (side-by-side)
- ROI drawing tools
- Path animation (raster scan, custom paths)
"""

import pathlib

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.core.datastructures import Dataset4dstem


# Detector geometry constant
DEFAULT_BF_RATIO = 0.125  # 1/8 of detector size


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

    # Pre-normalized uint8 frame as bytes (no base64!)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Log scale toggle
    log_scale = traitlets.Bool(False).tag(sync=True)

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
    roi_radius = traitlets.Float(10.0).tag(sync=True)
    roi_radius_inner = traitlets.Float(5.0).tag(sync=True)
    roi_width = traitlets.Float(20.0).tag(sync=True)
    roi_height = traitlets.Float(10.0).tag(sync=True)

    # =========================================================================
    # Virtual Image (ROI-based, updates as you drag ROI on DP)
    # =========================================================================
    virtual_image_bytes = traitlets.Bytes(b"").tag(sync=True)

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(1.0).tag(sync=True)  # Å per pixel (real-space)
    k_pixel_size = traitlets.Float(1.0).tag(sync=True)  # mrad per pixel (k-space)

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
        data: "Dataset4dstem | np.ndarray",
        scan_shape: tuple[int, int] | None = None,
        pixel_size: float | None = None,
        k_pixel_size: float | None = None,
        center: tuple[float, float] | None = None,
        bf_radius: float | None = None,
        log_scale: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.log_scale = log_scale

        # Extract calibration from Dataset4dstem if provided
        if hasattr(data, "sampling") and hasattr(data, "array"):
            # Dataset4dstem: extract calibration and array
            # sampling = [scan_x, scan_y, det_x, det_y]
            if pixel_size is None:
                pixel_size = float(data.sampling[0])
            if k_pixel_size is None:
                k_pixel_size = float(data.sampling[2])
            data = data.array

        # Store calibration values (default to 1.0 if not provided)
        self.pixel_size = pixel_size if pixel_size is not None else 1.0
        self.k_pixel_size = k_pixel_size if k_pixel_size is not None else 1.0
        # Path animation (configured via set_path() or raster())
        self._path_points: list[tuple[int, int]] = []
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
        # Setup center and BF/ADF radii based on detector size
        det_size = min(self.det_x, self.det_y)
        if center is not None:
            self.center_x = float(center[0])
            self.center_y = float(center[1])
        else:
            self.center_x = float(self.det_y / 2)
            self.center_y = float(self.det_x / 2)
        self.bf_radius = float(bf_radius) if bf_radius is not None else det_size * DEFAULT_BF_RATIO

        # Pre-compute and cache common virtual images (BF, ABF, LAADF, HAADF)
        self._cached_bf_virtual = None
        self._cached_abf_virtual = None
        self._cached_laadf_virtual = None
        self._cached_haadf_virtual = None
        self._precompute_common_virtual_images()

        # Update frame when position or settings change
        self.observe(self._update_frame, names=["pos_x", "pos_y", "log_scale"])
        self.observe(self._on_roi_change, names=[
            "roi_center_x", "roi_center_y", "roi_radius", "roi_radius_inner",
            "roi_active", "roi_mode", "roi_width", "roi_height"
        ])
        
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

    def __repr__(self) -> str:
        return (
            f"Show4DSTEM(shape=({self.shape_x}, {self.shape_y}, {self.det_x}, {self.det_y}), "
            f"sampling=({self.pixel_size} Å, {self.k_pixel_size} mrad), "
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
        for x, y in samples:
            frame = self._get_frame(x, y)
            fmin = float(frame.min())
            fmax = float(frame.max())
            all_min = min(all_min, fmin)
            all_max = max(all_max, fmax)

        self._global_min = max(all_min, 1e-10)
        self._global_max = all_max

        # Precompute log range
        self._log_min = np.log1p(self._global_min)
        self._log_max = np.log1p(self._global_max)

    def _get_frame(self, x: int, y: int):
        """Get single diffraction frame at position (x, y)."""
        if self._data.ndim == 3:
            idx = x * self.shape_y + y
            return self._data[idx]
        else:
            return self._data[x, y]

    def _update_frame(self, change=None):
        """Send pre-normalized uint8 frame to frontend."""
        frame = self._get_frame(self.pos_x, self.pos_y)

        # Determine value range
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
        """Recompute virtual image when ROI changes."""
        if not self.roi_active:
            return
        self._compute_virtual_image_from_roi()

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

    def _precompute_common_virtual_images(self):
        """Pre-compute BF/ABF/LAADF/HAADF virtual images for instant mode switching."""
        def _compute_and_normalize(mask):
            if self._data.ndim == 4:
                img = (self._data * mask).sum(axis=(-2, -1))
            else:
                img = (self._data * mask).sum(axis=(-2, -1)).reshape(self._scan_shape)
            vmin, vmax = float(img.min()), float(img.max())
            if vmax > vmin:
                norm = np.clip((img - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
            else:
                norm = np.zeros(img.shape, dtype=np.uint8)
            return norm.tobytes()
        
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
            # Point mode: single-pixel indexing
            # Array indexing: [row, col] where row=det_y, col=det_x in image coords
            row = int(max(0, min(self._det_shape[0] - 1, round(self.roi_center_y))))
            col = int(max(0, min(self._det_shape[1] - 1, round(self.roi_center_x))))
            if self._data.ndim == 4:
                virtual_image = self._data[:, :, row, col]
            else:
                virtual_image = self._data[:, row, col].reshape(self._scan_shape)

        # Normalize to uint8
        vmin, vmax = float(virtual_image.min()), float(virtual_image.max())
        if vmax > vmin:
            normalized = np.clip((virtual_image - vmin) / (vmax - vmin) * 255, 0, 255)
            normalized = normalized.astype(np.uint8)
        else:
            normalized = np.zeros(virtual_image.shape, dtype=np.uint8)
        self.virtual_image_bytes = normalized.tobytes()
