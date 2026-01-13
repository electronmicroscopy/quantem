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


def test_roi_circle_integrated_value():
    data = np.zeros((1, 1, 5, 5), dtype=np.float32)
    rows = np.arange(5, dtype=np.float32)[:, None]
    cols = np.arange(5, dtype=np.float32)[None, :]
    data[0, 0] = rows * 10 + cols
    widget = Show4DSTEM(data, center=(2, 2), bf_radius=1, log_scale=False)
    widget.roi_mode = "circle"
    widget.roi_center_x = 2
    widget.roi_center_y = 2
    widget.roi_radius = 1
    widget.roi_active = True
    widget._on_roi_change()
    assert np.isclose(widget.roi_integrated_value, 110.0)


def test_scan_image_bf_mode():
    base = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=np.float32,
    )
    data = np.zeros((2, 2, 3, 3), dtype=np.float32)
    for x in range(2):
        for y in range(2):
            data[x, y] = base + x * 100 + y * 10

    widget = Show4DSTEM(data, center=(1, 1), bf_radius=1, log_scale=False)
    widget.scan_mode = "bf"
    widget.show_scan_view = True
    widget._compute_scan_image()

    actual = np.frombuffer(widget.scan_image_bytes, dtype=np.uint8).reshape(2, 2)
    scan_image = np.array([[25, 75], [525, 575]], dtype=np.float32)
    expected = np.clip(
        (scan_image - scan_image.min()) / (scan_image.max() - scan_image.min()) * 255,
        0,
        255,
    ).astype(np.uint8)
    assert np.array_equal(actual, expected)


def test_log_scale_changes_frame_bytes():
    data = np.array([[[[0, 1], [3, 7]]]], dtype=np.float32)
    widget = Show4DSTEM(data, log_scale=True)
    log_bytes = bytes(widget.frame_bytes)

    widget.log_scale = False
    widget._update_frame()
    linear_bytes = bytes(widget.frame_bytes)

    assert log_bytes != linear_bytes
