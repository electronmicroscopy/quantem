import numpy as np
import quantem.widget
from quantem.widget import Show4DSTEM


def test_version_exists():
    assert hasattr(quantem.widget, "__version__")


def test_version_is_string():
    assert isinstance(quantem.widget.__version__, str)


def test_show4dstem_loads():
    """Widget can be created from mock 4D data."""
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget is not None


def test_show4dstem_flattened_scan_shape_mapping():
    """Test flattened 3D data with explicit scan shape."""
    data = np.zeros((6, 2, 2), dtype=np.float32)
    for idx in range(data.shape[0]):
        data[idx] = idx
    widget = Show4DSTEM(data, scan_shape=(2, 3))
    assert (widget.shape_x, widget.shape_y) == (2, 3)
    assert (widget.det_x, widget.det_y) == (2, 2)
    frame = widget._get_frame(1, 2)
    assert np.array_equal(frame, np.full((2, 2), 5, dtype=np.float32))


def test_show4dstem_log_scale():
    """Test that log scale changes frame bytes."""
    data = np.random.rand(2, 2, 8, 8).astype(np.float32) * 100 + 1
    widget = Show4DSTEM(data, log_scale=True)
    log_bytes = bytes(widget.frame_bytes)
    widget.log_scale = False
    widget._update_frame()
    linear_bytes = bytes(widget.frame_bytes)
    assert log_bytes != linear_bytes


def test_show4dstem_auto_detect_center():
    """Test automatic center spot detection using centroid."""
    data = np.zeros((2, 2, 7, 7), dtype=np.float32)
    for i in range(7):
        for j in range(7):
            dist = np.sqrt((i - 3) ** 2 + (j - 3) ** 2)
            if dist <= 1.5:
                data[:, :, i, j] = 100.0
    widget = Show4DSTEM(data, precompute_virtual_images=False)
    widget.auto_detect_center()
    assert abs(widget.center_x - 3.0) < 0.5
    assert abs(widget.center_y - 3.0) < 0.5
    assert widget.bf_radius > 0


def test_show4dstem_adf_preset_cache():
    """Test that ADF preset cache works when precompute is enabled."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data, center=(8, 8), bf_radius=2, precompute_virtual_images=True)
    assert widget._cached_adf_virtual is not None
    widget.roi_mode = "annular"
    widget.roi_center_x = 8
    widget.roi_center_y = 8
    widget.roi_radius_inner = 2
    widget.roi_radius = 8
    cached = widget._get_cached_preset()
    assert cached == widget._cached_adf_virtual


def test_show4dstem_rectangular_scan_shape():
    """Test that rectangular (non-square) scans work correctly."""
    data = np.random.rand(4, 8, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget.shape_x == 4
    assert widget.shape_y == 8
    assert widget.det_x == 16
    assert widget.det_y == 16
    frame_00 = widget._get_frame(0, 0)
    frame_37 = widget._get_frame(3, 7)
    assert frame_00.shape == (16, 16)
    assert frame_37.shape == (16, 16)


def test_show4dstem_hot_pixel_removal_uint16():
    """Test that saturated uint16 hot pixels are removed at init."""
    data = np.zeros((4, 4, 8, 8), dtype=np.uint16)
    data[:, :, :, :] = 100
    data[:, :, 3, 5] = 65535
    data[:, :, 1, 2] = 65535
    widget = Show4DSTEM(data)
    assert widget.dp_global_max < 65535
    assert widget.dp_global_max == 100.0
    frame = widget._get_frame(0, 0)
    assert frame[3, 5] == 0
    assert frame[1, 2] == 0
    assert frame[0, 0] == 100


def test_show4dstem_hot_pixel_removal_uint8():
    """Test that saturated uint8 hot pixels are removed at init."""
    data = np.zeros((4, 4, 8, 8), dtype=np.uint8)
    data[:, :, :, :] = 50
    data[:, :, 2, 3] = 255
    widget = Show4DSTEM(data)
    assert widget.dp_global_max == 50.0
    frame = widget._get_frame(0, 0)
    assert frame[2, 3] == 0


def test_show4dstem_no_hot_pixel_removal_float32():
    """Test that float32 data is not modified (no saturated value)."""
    data = np.ones((4, 4, 8, 8), dtype=np.float32) * 1000
    widget = Show4DSTEM(data)
    assert widget.dp_global_max == 1000.0


def test_show4dstem_roi_modes():
    """Test all ROI modes compute virtual images correctly."""
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data, center=(8, 8), bf_radius=3)
    for mode in ["point", "circle", "square", "annular", "rect"]:
        widget.roi_mode = mode
        widget.roi_active = True
        assert len(widget.vi_stats) == 4
        assert widget.vi_stats[2] >= widget.vi_stats[1]


def test_show4dstem_virtual_image_excludes_hot_pixels():
    """Test that virtual images don't include hot pixel contributions."""
    data = np.ones((4, 4, 8, 8), dtype=np.uint16) * 10
    data[:, :, 4, 4] = 65535
    widget = Show4DSTEM(data, center=(4, 4), bf_radius=2)
    widget.roi_mode = "circle"
    widget.roi_center_x = 4
    widget.roi_center_y = 4
    widget.roi_radius = 3
    assert widget.vi_stats[2] < 1000
