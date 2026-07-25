from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from quantem.diffraction import BraggPeaksPolymer, ScanMaskEditor


def _analysis(scan_shape=(7, 9)):
    rows, columns = scan_shape
    analysis = object.__new__(BraggPeaksPolymer)
    analysis.dataset_cartesian = SimpleNamespace(
        shape=(rows, columns, 3, 4),
        array=np.arange(rows * columns * 12, dtype=float).reshape(
            rows, columns, 3, 4
        ),
        sampling=(0.5, 0.5, 1.0, 1.0),
        units=("nm", "nm", "1/Å", "1/Å"),
        virtual_images={},
    )
    analysis._scan_mask = None
    return analysis


def test_controls_follow_probe_directions_and_apply_is_explicit():
    analysis = _analysis()
    editor = analysis.edit_scan_mask(
        initial_x=4,
        initial_y=3,
        initial_radius=2,
        display_widget=False,
    )
    try:
        assert isinstance(editor, ScanMaskEditor)
        initially_applied = analysis.scan_mask.copy()
        assert editor.widget.layout.width == "500px"
        assert editor.output.layout.width == "438px"
        assert tuple(editor.figure.get_size_inches()) == pytest.approx((4.0, 3.25))
        assert editor.circle_artist.get_linewidth() == pytest.approx(1.05)
        assert editor.circle_artist.get_linestyle() != "-"
        assert editor.figure.number not in plt.get_fignums()
        # Radius is the row immediately above the image/Y-slider row.
        assert editor.radius_slider in editor.widget.children[3].children
        assert editor.output in editor.widget.children[5].children

        editor.x_slider.value += 1
        assert editor.x == 5
        assert editor.circle_artist.center == (5, 3)

        # Increasing the vertical slider moves its thumb and the probe upward.
        editor.y_slider.value += 1
        assert editor.y == 2
        assert editor.circle_artist.center == (5, 2)

        assert not np.array_equal(editor.mask, initially_applied)
        np.testing.assert_array_equal(analysis.scan_mask, initially_applied)
        # This is the exact compatibility path used by older notebook cells.
        mask_arr = editor
        assert mask_arr.sum() == initially_applied.sum()
        np.testing.assert_array_equal(np.asarray(mask_arr), initially_applied)
        editor.apply()
        np.testing.assert_array_equal(analysis.scan_mask, editor.mask)
        assert mask_arr.sum() == editor.mask.sum()
    finally:
        editor.close()


def test_saved_mask_is_loaded_and_applied(tmp_path):
    path = tmp_path / "scan_mask.npz"
    first = _analysis()
    editor = first.edit_scan_mask(
        initial_x=2,
        initial_y=3,
        initial_radius=2,
        state_path=path,
        display_widget=False,
    )
    try:
        editor.set_mask(
            x=6,
            y=1,
            geometry="rectangle",
            size_x=2,
            size_y=1,
        )
        expected = editor.mask
        assert editor.save() == path
    finally:
        editor.close()

    second = _analysis()
    loaded = second.edit_scan_mask(
        initial_x=0,
        initial_y=0,
        initial_radius=1,
        state_path=path,
        display_widget=False,
    )
    try:
        assert (loaded.x, loaded.y) == (6, 1)
        assert (loaded.geometry, loaded.size_x, loaded.size_y) == (
            "rectangle",
            2,
            1,
        )
        np.testing.assert_array_equal(loaded.mask, expected)
        np.testing.assert_array_equal(second.scan_mask, expected)
        with np.load(path, allow_pickle=False) as state:
            assert int(state["schema_version"]) == ScanMaskEditor.SCHEMA_VERSION
            assert tuple(state["scan_shape"]) == (7, 9)
    finally:
        loaded.close()


def test_saved_mask_rejects_a_different_scan_shape(tmp_path):
    path = tmp_path / "scan_mask.npz"
    editor = _analysis().edit_scan_mask(
        state_path=path, display_widget=False
    )
    try:
        editor.save()
    finally:
        editor.close()

    with pytest.raises(ValueError, match="does not match current scan shape"):
        _analysis((8, 9)).edit_scan_mask(
            state_path=path, display_widget=False
        )
    plt.close("all")


def test_version_one_circle_state_remains_loadable(tmp_path):
    path = tmp_path / "scan_mask_v1.npz"
    yy, xx = np.ogrid[:7, :9]
    mask = (yy - 3) ** 2 + (xx - 4) ** 2 <= 2**2
    np.savez_compressed(
        path,
        schema_version=np.asarray(1),
        scan_shape=np.asarray((7, 9)),
        mask=mask,
        center_row=np.asarray(3),
        center_column=np.asarray(4),
        radius=np.asarray(2),
    )
    editor = _analysis().edit_scan_mask(
        state_path=path, display_widget=False
    )
    try:
        assert (editor.geometry, editor.size_x, editor.size_y) == (
            "circle",
            2,
            2,
        )
        np.testing.assert_array_equal(editor.mask, mask)
    finally:
        editor.close()


def test_legacy_wrapper_preserves_historical_row_column_arguments():
    analysis = _analysis()
    editor = analysis.create_interactive_circular_mask(
        initial_x0=2,
        initial_y0=6,
        initial_r=3,
        display_widget=False,
    )
    try:
        assert (editor.x, editor.y, editor.radius) == (6, 2, 3)
        assert editor["x0"] == 2
        assert editor["y0"] == 6
    finally:
        editor.close()


@pytest.mark.parametrize(
    ("geometry", "size_x", "size_y", "expected_count"),
    [
        ("circle", 2, 2, 13),
        ("ellipse", 3, 1, 9),
        ("square", 2, 2, 25),
        ("rectangle", 2, 1, 15),
    ],
)
def test_geometry_selector_builds_expected_masks(
    geometry, size_x, size_y, expected_count
):
    editor = _analysis().edit_scan_mask(
        initial_x=4,
        initial_y=3,
        display_widget=False,
    )
    try:
        editor.set_mask(
            geometry=geometry,
            size_x=size_x,
            size_y=size_y,
        )
        assert editor.geometry == geometry
        assert editor.mask.sum() == expected_count
        assert editor.circle_artist.get_linewidth() == pytest.approx(1.05)
        if geometry in {"circle", "square"}:
            assert editor.size_y == editor.size_x
            assert editor.size_y_row.layout.display == "none"
        else:
            assert editor.size_y_row.layout.display != "none"
    finally:
        editor.close()
