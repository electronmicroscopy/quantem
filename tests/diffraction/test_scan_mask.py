from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.path import Path

from quantem.diffraction import BraggPeaksPolymer, ScanMaskShape
from quantem.diffraction.scan_mask import SCHEMA_VERSION, save_scan_mask


def _analysis(scan_shape=(7, 9)):
    rows, columns = scan_shape
    analysis = object.__new__(BraggPeaksPolymer)
    analysis.dataset_cartesian = SimpleNamespace(
        shape=(rows, columns, 3, 4),
        sampling=(0.5, 0.5, 1.0, 1.0),
        units=("nm", "nm", "1/Å", "1/Å"),
    )
    analysis._dataset_cartesian = analysis.dataset_cartesian
    analysis._scan_mask = None
    analysis._scan_mask_shape = None
    return analysis


def _shape(geometry, size_x, size_y=None, angle=0.0):
    """A shape centred on a 41 x 41 scan, for geometry checks."""
    return ScanMaskShape(x=20, y=20, geometry=geometry, size_x=size_x, size_y=size_y, angle=angle)


def test_geometries_follow_scan_directions_and_sizes():
    # x is the column and y the row, on a non-square scan so a swap would show.
    analysis = _analysis()
    analysis.set_scan_mask("circle", x=6, y=0, size_x=1)
    assert analysis.scan_mask[0, 6] and not analysis.scan_mask[6, 0]
    assert analysis.scan_mask.sum() == 4  # the circle is clipped at the top edge

    # Circle and square tie the second size to the first.
    cases = [
        ("circle", 2, 5, 13),
        ("ellipse", 3, 1, 9),
        ("square", 2, 5, 25),
        ("rectangle", 2, 1, 15),
    ]
    for geometry, size_x, size_y, expected_count in cases:
        mask = analysis.set_scan_mask(geometry, x=4, y=3, size_x=size_x, size_y=size_y)
        assert mask.sum() == expected_count, geometry
        assert analysis.scan_mask_shape.geometry == geometry, geometry

    with pytest.raises(ValueError, match="outside scan shape"):
        analysis.set_scan_mask("circle", x=9, y=0, size_x=1)
    with pytest.raises(ValueError, match="Unknown mask geometry"):
        analysis.set_scan_mask("triangle")


def test_rotation_moves_the_mask():
    for geometry, size_x, size_y, angle in (
        ("ellipse", 12, 4, 90),
        ("rectangle", 10, 4, 45),
        ("square", 10, 4, 45),
    ):
        upright = _shape(geometry, size_x, size_y).mask((41, 41))
        turned = _shape(geometry, size_x, size_y, angle).mask((41, 41))
        assert not np.array_equal(upright, turned), geometry
        if geometry == "ellipse":
            # A 90 degree turn swaps the long axis, preserving area to within
            # the pixellation of the turn.
            assert upright[20, :].sum() > upright[:, 20].sum()
            assert turned[20, :].sum() < turned[:, 20].sum()
            assert turned.sum() == pytest.approx(upright.sum(), rel=0.15)
        if geometry == "square":
            # A rotated square still reaches its corners on the diagonal.
            assert turned[20, 20 + 13] and not upright[20, 20 + 13]


def test_a_diagonal_ellipse_is_neither_axis_aligned():
    mask = _shape("ellipse", 14, 3, 45).mask((41, 41))
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    # A 45 degree ellipse spans both directions nearly equally...
    assert abs(len(rows) - len(columns)) <= 2
    # ...and its extent exceeds the short semi-axis in both, unlike an upright one.
    assert len(rows) > 2 * 3 and len(columns) > 2 * 3
    # The long axis runs along increasing row and column together; the outline
    # test below checks that the drawn shape agrees.
    assert mask[20 + 6, 20 + 6] and not mask[20 - 6, 20 + 6]


def test_outline_matches_the_mask_at_every_angle():
    """The drawn outline must agree with what will actually be selected.

    A rotation applied to the mask but not the outline (or with the opposite
    sign) would silently mislead. Rectangles get a looser bound: the mask is
    inclusive on both edges while the outline is drawn a half pixel outside.
    """
    yy, xx = np.mgrid[0:41, 0:41]
    points = np.column_stack([xx.ravel(), yy.ravel()])
    for geometry, size_x, size_y in (("ellipse", 14, 3), ("rectangle", 12, 4), ("square", 9, 9)):
        for angle in (0, 30, 45, 120):
            shape = _shape(geometry, size_x, size_y, angle)
            inside = Path(shape.outline()).contains_points(points).reshape(41, 41)
            bound = 0.99 if geometry == "ellipse" else 0.96
            assert (inside == shape.mask((41, 41))).mean() > bound, f"{geometry} at {angle} deg"


def test_angle_is_normalized_per_geometry():
    """A circle ignores rotation; other angles wrap at 180."""
    circle = _shape("circle", 8, angle=37)
    assert circle.angle == 0.0
    np.testing.assert_array_equal(circle.mask((41, 41)), _shape("circle", 8).mask((41, 41)))
    assert _shape("rectangle", 10, 4, angle=200).angle == pytest.approx(20.0)


def test_saved_mask_round_trips_with_its_shape_and_painting(tmp_path):
    first = _analysis()
    first.set_scan_mask("ellipse", x=4, y=3, size_x=3, size_y=1, angle=30)
    shaped = first.scan_mask.copy()
    painted = shaped.copy()
    painted[0, 0] = True  # a pixel added by hand after the shape was drawn
    unshaped = _analysis()
    unshaped.scan_mask = painted

    for label, analysis, expected_mask, expected_shape, expected_painted in (
        ("shape", first, shaped, first.scan_mask_shape, False),
        ("painted", None, painted, first.scan_mask_shape, True),
        ("no shape", unshaped, painted, None, True),
    ):
        if analysis is None:
            first.scan_mask = painted
            analysis = first
        path = analysis.save_scan_mask(tmp_path / f"{label}.npz")
        with np.load(path, allow_pickle=False) as state:
            assert int(state["schema_version"]) == SCHEMA_VERSION, label
            assert bool(state["painted"]) == expected_painted, label

        loaded = _analysis()
        np.testing.assert_array_equal(loaded.load_scan_mask(path), expected_mask, label)
        assert loaded.scan_mask_shape == expected_shape, label

    with pytest.raises(ValueError, match="does not match current scan shape"):
        _analysis((8, 9)).load_scan_mask(path)


def test_older_state_files_remain_loadable(tmp_path):
    yy, xx = np.ogrid[:7, :9]
    circle = (yy - 3) ** 2 + (xx - 4) ** 2 <= 2**2
    common = dict(
        scan_shape=np.asarray((7, 9)),
        mask=circle,
        center_row=np.asarray(3),
        center_column=np.asarray(4),
    )
    for version, extra, expected in (
        (1, dict(radius=np.asarray(2)), ScanMaskShape(x=4, y=3, size_x=2)),
        (
            3,
            dict(
                geometry=np.asarray("rectangle"),
                size_x=np.asarray(2),
                size_y=np.asarray(1),
                angle=np.asarray(30.0),
            ),
            ScanMaskShape(x=4, y=3, geometry="rectangle", size_x=2, size_y=1, angle=30.0),
        ),
    ):
        path = tmp_path / f"v{version}.npz"
        np.savez_compressed(path, schema_version=np.asarray(version), **common, **extra)
        analysis = _analysis()
        np.testing.assert_array_equal(analysis.load_scan_mask(path), circle, f"v{version}")
        assert analysis.scan_mask_shape == expected, f"v{version}"

    # A mask written without a shape reloads with none.
    path = save_scan_mask(tmp_path / "bare.npz", circle)
    analysis = _analysis()
    analysis.load_scan_mask(path)
    assert analysis.scan_mask_shape is None
