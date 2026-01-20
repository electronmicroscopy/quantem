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
    data = np.zeros((6, 2, 2), dtype=np.float32)
    for idx in range(data.shape[0]):
        data[idx] = idx

    widget = Show4DSTEM(data, scan_shape=(2, 3))
    assert (widget.shape_x, widget.shape_y) == (2, 3)
    assert (widget.det_x, widget.det_y) == (2, 2)
    frame = widget._get_frame(1, 2)
    assert np.array_equal(frame, np.full((2, 2), 5, dtype=np.float32))


def test_log_scale_changes_frame_bytes():
    data = np.array([[[[0, 1], [3, 7]]]], dtype=np.float32)
    widget = Show4DSTEM(data, log_scale=True)
    log_bytes = bytes(widget.frame_bytes)

    widget.log_scale = False
    widget._update_frame()
    linear_bytes = bytes(widget.frame_bytes)

    assert log_bytes != linear_bytes


def test_auto_detect_center():
    """Test automatic center spot detection using centroid."""
    # Create data with a bright spot at (3, 3) in a 7x7 detector
    data = np.zeros((2, 2, 7, 7), dtype=np.float32)
    # Add a bright circular spot centered at (3, 3)
    for i in range(7):
        for j in range(7):
            dist = np.sqrt((i - 3) ** 2 + (j - 3) ** 2)
            if dist <= 1.5:
                data[:, :, i, j] = 100.0

    widget = Show4DSTEM(data, precompute_virtual_images=False)
    # Initial center should be at detector center (3.5, 3.5)
    assert widget.center_x == 3.5
    assert widget.center_y == 3.5

    # Run auto-detection
    widget.auto_detect_center()

    # Center should be detected near (3, 3)
    assert abs(widget.center_x - 3.0) < 0.5
    assert abs(widget.center_y - 3.0) < 0.5
    # BF radius should be approximately sqrt(pi*r^2 / pi) = r ~ 1.5
    assert widget.bf_radius > 0


def test_adf_preset_cache():
    """Test that ADF preset uses combined bf to 4*bf range."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data, center=(8, 8), bf_radius=2)

    # Check that ADF cache exists (replaced LAADF/HAADF)
    assert widget._cached_adf_virtual is not None
    assert not hasattr(widget, "_cached_laadf_virtual")
    assert not hasattr(widget, "_cached_haadf_virtual")

    # Set ROI to match ADF range
    widget.roi_mode = "annular"
    widget.roi_center_x = 8
    widget.roi_center_y = 8
    widget.roi_radius_inner = 2  # bf
    widget.roi_radius = 8  # 4*bf

    # Should return cached value
    cached = widget._get_cached_preset()
    assert cached == widget._cached_adf_virtual


def test_rectangular_scan_shape():
    """Test that rectangular (non-square) scans work correctly."""
    # Non-square scan: 4 rows x 8 columns
    data = np.random.rand(4, 8, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)

    assert widget.shape_x == 4
    assert widget.shape_y == 8
    assert widget.det_x == 16
    assert widget.det_y == 16

    # Verify frame retrieval works at corners
    frame_00 = widget._get_frame(0, 0)
    frame_37 = widget._get_frame(3, 7)
    assert frame_00.shape == (16, 16)
    assert frame_37.shape == (16, 16)
