import zipfile

import numpy as np
import pytest

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import load


def make_line_vector() -> Vector:
    v = Vector.from_shape(
        shape=(4,),
        fields=["intensity", "kx", "ky"],
        units=["a.u.", "px", "px"],
        name="line",
    )
    v[0] = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]])
    v[1] = np.array([[3.0, 30.0, 300.0]])
    v[2] = np.array([[4.0, 40.0, 400.0], [5.0, 50.0, 500.0]])
    v[3] = np.array([[6.0, 60.0, 600.0]])
    return v


def make_grid_vector() -> Vector:
    v = Vector.from_shape(shape=(3, 2), fields=["intensity", "kx", "ky"])
    for i in range(3):
        for j in range(2):
            base = float(i * 10 + j)
            v[i, j] = np.array([[base, base + 100.0, base + 200.0]])
    return v


class TestVector:
    def test_initialization_and_len(self):
        v1 = Vector.from_shape(shape=(2, 3), fields=["a", "b", "c"])
        assert v1.shape == (2, 3)
        assert len(v1) == 2
        assert v1.num_cells == 6
        assert v1.num_fields == 3
        assert v1.dtype == np.dtype(float)
        assert v1.fields == ["a", "b", "c"]
        assert v1.units == ["none", "none", "none"]
        assert v1.name == "2d ragged array"
        assert v1[0, 0].array.shape == (0, 3)
        np.testing.assert_array_equal(v1[0, 0].flatten(), v1[0, 0].array)

        v2 = Vector.from_shape(shape=(2, 3), num_fields=2)
        assert v2.fields == ["field_0", "field_1"]

        with pytest.raises(TypeError):
            len(v1[0, 0])

        with pytest.raises(ValueError, match="Must specify either 'fields' or 'num_fields'."):
            Vector.from_shape(shape=(2, 3))

        with pytest.raises(ValueError, match="does not match length of fields"):
            Vector.from_shape(shape=(2, 3), num_fields=2, fields=["a", "b", "c"])

        with pytest.raises(ValueError, match="Duplicate field names"):
            Vector.from_shape(shape=(2, 3), fields=["a", "a"])

        assert str(v1) == (
            "quantem.Vector, shape=(2, 3), name=2d ragged array\n"
            "  fields = ['a', 'b', 'c']\n"
            "  units: ['none', 'none', 'none']"
        )

    def test_indexing_and_array_contract(self):
        v = make_grid_vector()

        assert isinstance(v[:2, 1], Vector)
        assert v[:2, 1].shape == (2,)
        assert v[1].shape == (2,)
        assert v[1, 1].shape == ()
        np.testing.assert_array_equal(v[-1, -1].array, np.array([[21.0, 121.0, 221.0]]))

        with pytest.raises(ValueError):
            _ = v[:, 1].array

        result = v[[-1, 0], 1]
        assert result.shape == (2,)
        assert result.num_cells == 2
        np.testing.assert_array_equal(result[0].array, np.array([[21.0, 121.0, 221.0]]))
        np.testing.assert_array_equal(result[1].array, np.array([[1.0, 101.0, 201.0]]))

    def test_select_fields_and_chaining_equivalence(self):
        v = make_line_vector()

        selected = v.select_fields("kx")
        assert selected.fields == ["kx"]
        assert selected.units == ["px"]
        assert selected.shape == v.shape

        np.testing.assert_array_equal(
            v.select_fields("kx")[2].array,
            v[2].select_fields("kx").array,
        )

        with pytest.raises(KeyError):
            v.select_fields("missing")

        with pytest.raises(TypeError):
            _ = v["kx"]

        with pytest.raises(TypeError):
            _ = v[1, "kx"]

        multi = v.select_fields("intensity", "kx")
        assert multi.fields == ["intensity", "kx"]
        assert multi.dtype == np.dtype(float)
        assert multi.total_rows == 6
        assert multi.row_counts() == [2, 1, 2, 1]

    def test_reductions_over_all_rows(self):
        v = make_line_vector()

        np.testing.assert_allclose(v.sum(), np.array([21.0, 210.0, 2100.0]))
        np.testing.assert_allclose(v.mean(), np.array([3.5, 35.0, 350.0]))
        np.testing.assert_allclose(v.min(), np.array([1.0, 10.0, 100.0]))
        np.testing.assert_allclose(v.max(), np.array([6.0, 60.0, 600.0]))
        np.testing.assert_allclose(v.std(), np.std(v.flatten(), axis=0))
        assert v.count() == 6

        # Field and fixed-grid selections narrow what is reduced
        np.testing.assert_allclose(v.select_fields("kx").mean(), np.array([35.0]))
        np.testing.assert_allclose(v[:2].sum(), np.array([6.0, 60.0, 600.0]))
        assert v[:2].count() == 3

    def test_reductions_per_cell(self):
        v = make_line_vector()

        np.testing.assert_allclose(
            v.sum(per_cell=True),
            np.array(
                [[3.0, 30.0, 300.0], [3.0, 30.0, 300.0], [9.0, 90.0, 900.0], [6.0, 60.0, 600.0]]
            ),
        )
        np.testing.assert_allclose(
            v.mean(per_cell=True),
            np.array(
                [[1.5, 15.0, 150.0], [3.0, 30.0, 300.0], [4.5, 45.0, 450.0], [6.0, 60.0, 600.0]]
            ),
        )
        np.testing.assert_allclose(v.min(per_cell=True)[0], np.array([1.0, 10.0, 100.0]))
        np.testing.assert_allclose(v.max(per_cell=True)[0], np.array([2.0, 20.0, 200.0]))
        np.testing.assert_allclose(
            v.select_fields("intensity").std(per_cell=True)[:, 0],
            np.array([0.5, 0.0, 0.5, 0.0]),
        )
        np.testing.assert_array_equal(v.count(per_cell=True), np.array([2, 1, 2, 1]))

        # Per-cell results keep the fixed-grid shape plus a trailing field axis
        grid = make_grid_vector()
        assert grid.sum(per_cell=True).shape == (3, 2, 3)
        assert grid.count(per_cell=True).shape == (3, 2)
        np.testing.assert_allclose(grid.max(per_cell=True)[2, 1], np.array([21.0, 121.0, 221.0]))

    def test_reductions_handle_empty_cells_and_selections(self):
        v = Vector.from_shape(shape=(3,), fields=["intensity"])
        v[0] = np.array([[2.0], [4.0]])
        v[2] = np.array([[9.0]])

        np.testing.assert_allclose(v.sum(per_cell=True)[:, 0], np.array([6.0, 0.0, 9.0]))
        per_cell_mean = v.mean(per_cell=True)[:, 0]
        np.testing.assert_allclose(per_cell_mean[[0, 2]], np.array([3.0, 9.0]))
        assert np.isnan(per_cell_mean[1])
        assert np.isnan(v.min(per_cell=True)[1, 0])
        assert np.isnan(v.max(per_cell=True)[1, 0])
        assert np.isnan(v.std(per_cell=True)[1, 0])
        np.testing.assert_array_equal(v.count(per_cell=True), np.array([2, 0, 1]))

        # A selection with no rows at all
        empty = v[1]
        np.testing.assert_allclose(empty.sum(), np.array([0.0]))
        assert np.isnan(empty.mean()).all()
        assert empty.count() == 0

    def test_reductions_as_dataset(self):
        v = make_grid_vector()

        image = v.select_fields("intensity").max(per_cell=True, as_dataset=True)
        assert isinstance(image, Dataset2d)
        assert image.shape == (3, 2)
        assert image.signal_units == "none"
        assert "max" in image.name
        np.testing.assert_allclose(image.array, np.array([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]]))

        counts = v.count(per_cell=True, as_dataset=True)
        assert isinstance(counts, Dataset2d)
        assert counts.signal_units == "counts"
        np.testing.assert_array_equal(counts.array, np.ones((3, 2)))

        line = make_line_vector()
        line_sum = line.select_fields("kx").sum(per_cell=True, as_dataset=True)
        assert line_sum.shape == (4,)
        assert line_sum.signal_units == "px"

        with pytest.raises(ValueError, match="exactly one selected field"):
            v.max(per_cell=True, as_dataset=True)

        with pytest.raises(ValueError, match="requires per_cell=True"):
            v.select_fields("intensity").max(as_dataset=True)

        with pytest.raises(ValueError, match="requires per_cell=True"):
            v.count(as_dataset=True)

        with pytest.raises(ValueError, match="at least one fixed-grid axis"):
            v[0, 0].select_fields("intensity").max(per_cell=True, as_dataset=True)

    def test_array_mutation_writes_through_for_single_field(self):
        v = make_line_vector()
        cell = v.select_fields("kx")[1].array
        cell[0, 0] = 99.0
        assert v[1].array[0, 1] == 99.0

    def test_set_flattened_updates_rowwise(self):
        v = make_line_vector()
        kx = v.select_fields("kx")

        flat_kx = kx.flatten()
        mask = flat_kx >= 30.0
        flat_kx[mask[:, 0], 0] = -1.0
        kx.set_flattened(flat_kx)

        np.testing.assert_array_equal(
            kx.flatten(),
            np.array([[10.0], [20.0], [-1.0], [-1.0], [-1.0], [-1.0]]),
        )

    def test_field_arithmetic_with_scalar_and_ndarray(self):
        v = make_line_vector()

        kx = v.select_fields("kx")
        kx += 10
        np.testing.assert_array_equal(
            v.select_fields("kx").flatten(),
            np.array([[20.0], [30.0], [40.0], [50.0], [60.0], [70.0]]),
        )

        v.select_fields("kx")[...] += np.arange(6)
        np.testing.assert_array_equal(
            v.select_fields("kx").flatten(),
            np.array([[20.0], [31.0], [42.0], [53.0], [64.0], [75.0]]),
        )

        summed = v.select_fields("intensity") + v.select_fields("ky")
        np.testing.assert_array_equal(
            summed.flatten(),
            np.array([[101.0], [202.0], [303.0], [404.0], [505.0], [606.0]]),
        )

    def test_power_operations(self):
        v = make_line_vector()

        squared = v.select_fields("intensity") ** 2
        np.testing.assert_array_equal(
            squared.flatten(),
            np.array([[1.0], [4.0], [9.0], [16.0], [25.0], [36.0]]),
        )

        intensity = v.select_fields("intensity")
        intensity **= 2
        np.testing.assert_array_equal(
            intensity.flatten(),
            np.array([[1.0], [4.0], [9.0], [16.0], [25.0], [36.0]]),
        )

        reverse = 2 ** v.select_fields("intensity")
        np.testing.assert_array_equal(
            reverse.flatten(),
            np.array([[2.0], [16.0], [512.0], [65536.0], [33554432.0], [68719476736.0]]),
        )

    def test_unary_mod_and_floor_division_operations(self):
        v = make_line_vector()

        negative = -v.select_fields("intensity")
        np.testing.assert_array_equal(
            negative.flatten(),
            np.array([[-1.0], [-2.0], [-3.0], [-4.0], [-5.0], [-6.0]]),
        )

        absolute = abs(negative)
        np.testing.assert_array_equal(
            absolute.flatten(),
            np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        )

        floored = v.select_fields("ky") // 150
        np.testing.assert_array_equal(
            floored.flatten(),
            np.array([[0.0], [1.0], [2.0], [2.0], [3.0], [4.0]]),
        )

        modded = v.select_fields("ky") % 150
        np.testing.assert_array_equal(
            modded.flatten(),
            np.array([[100.0], [50.0], [0.0], [100.0], [50.0], [0.0]]),
        )

        ky = v.select_fields("ky")
        ky //= 150
        np.testing.assert_array_equal(
            ky.flatten(),
            np.array([[0.0], [1.0], [2.0], [2.0], [3.0], [4.0]]),
        )

        intensity = v.select_fields("intensity")
        intensity %= 2
        np.testing.assert_array_equal(
            intensity.flatten(),
            np.array([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0]]),
        )

    def test_numpy_ufunc_support(self):
        v = make_line_vector()

        sine = np.sin(v.select_fields("kx"))
        np.testing.assert_allclose(
            sine.flatten(),
            np.sin(v.select_fields("kx").flatten()),
        )

        maximum = np.maximum(v.select_fields("intensity"), 3.0)  # type: ignore[arg-type]
        np.testing.assert_array_equal(
            maximum.flatten(),
            np.array([[3.0], [3.0], [3.0], [4.0], [5.0], [6.0]]),
        )

        frac, whole = np.modf(v.select_fields("intensity") / 2.0)
        np.testing.assert_allclose(
            frac.flatten(),
            np.array([[0.5], [0.0], [0.5], [0.0], [0.5], [0.0]]),
        )
        np.testing.assert_allclose(
            whole.flatten(),
            np.array([[0.0], [1.0], [1.0], [2.0], [2.0], [3.0]]),
        )

    def test_field_assignment_from_vector_expression(self):
        v = make_line_vector()
        scale = 2.5

        v[:2].select_fields("intensity")[...] = v[2:4].select_fields("intensity") * scale
        np.testing.assert_array_equal(
            v[:2].select_fields("intensity").flatten(),
            np.array([[10.0], [12.5], [15.0]]),
        )

    def test_field_assignment_requires_matching_per_cell_row_counts(self):
        v = make_line_vector()
        with pytest.raises(ValueError, match="Per-cell row counts must match"):
            v[:2].select_fields("intensity")[...] = v[1:3].select_fields("intensity")

    def test_full_cell_assignment_allows_row_count_changes(self):
        v = make_line_vector()

        v[1] = v[0]
        assert v[1].array.shape == (2, 3)
        np.testing.assert_array_equal(v[1].array, v[0].array)

        v[0:2] = v[1:3]
        assert v[0].array.shape == (2, 3)
        assert v[1].array.shape == (2, 3)

        broadcast_cell = np.array([[9.0, 8.0, 7.0]])
        v[[0, 3]] = broadcast_cell
        np.testing.assert_array_equal(v[0].array, broadcast_cell)
        np.testing.assert_array_equal(v[3].array, broadcast_cell)

    def test_append_rows_and_compact(self):
        v = make_line_vector()

        v.append_rows(1, np.array([[7.0, 70.0, 700.0]]))
        np.testing.assert_array_equal(
            v[1].array,
            np.array([[3.0, 30.0, 300.0], [7.0, 70.0, 700.0]]),
        )

        v[1] = np.array([[8.0, 80.0, 800.0]])
        assert v._state["data"].shape[0] > v.total_rows

        v.compact()
        assert v._state["data"].shape[0] == v.total_rows

        with pytest.raises(ValueError, match="exactly one cell"):
            v.append_rows(slice(None), np.array([[1.0, 2.0, 3.0]]))

    def test_boolean_indexing_is_axis_wise(self):
        v = make_grid_vector()

        rows = np.array([True, False, True])
        cols = np.array([False, True])
        selected = v[rows, cols]

        assert selected.shape == (2, 1)
        np.testing.assert_array_equal(selected[0, 0].array, np.array([[1.0, 101.0, 201.0]]))
        np.testing.assert_array_equal(selected[1, 0].array, np.array([[21.0, 121.0, 221.0]]))

        with pytest.raises(IndexError):
            _ = v[np.array([[True, False], [False, True]])]

    def test_empty_selection_is_valid_and_no_op_for_scalar_math(self):
        v = make_grid_vector()
        before = v.copy().flatten()

        empty = v[[], :]
        assert empty.shape == (0, 2)
        assert empty.flatten().shape == (0, 3)
        assert empty.copy().shape == (0, 2)

        empty.select_fields("kx")[...] += 1
        np.testing.assert_array_equal(v.flatten(), before)

    def test_add_fields_defaults_expression_and_multiple_values(self):
        v = make_line_vector()

        v.add_fields(("h", "k"))
        assert v.fields == ["intensity", "kx", "ky", "h", "k"]
        assert np.isnan(v[0].array[:, 3:5]).all()

        v.add_fields("field_out", v.select_fields("kx") + v.select_fields("ky"))
        np.testing.assert_array_equal(
            v.select_fields("field_out").flatten(),
            np.array([[110.0], [220.0], [330.0], [440.0], [550.0], [660.0]]),
        )

        v2 = make_line_vector()
        v2.add_fields(("h", "k"), (1.0, np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])))
        np.testing.assert_array_equal(v2.select_fields("h").flatten(), np.ones((6, 1)))
        np.testing.assert_array_equal(
            v2.select_fields("k").flatten(),
            np.array([[5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]),
        )

        with pytest.raises(ValueError, match="all fields are selected"):
            v2.select_fields("kx").add_fields("bad")

    def test_rename_fields(self):
        v = make_line_vector()
        kx_data = v.select_fields("kx").flatten().copy()

        v.rename_fields({"kx": "qx", "ky": "qy"})
        assert v.fields == ["intensity", "qx", "qy"]
        np.testing.assert_array_equal(v.select_fields("qx").flatten(), kx_data)

        # Renaming through a field-selected view updates that view's selected names
        view = v.select_fields("qx")
        assert view.fields == ["qx"]
        view.rename_fields({"qx": "px"})
        assert view.fields == ["px"]
        assert v.fields == ["intensity", "px", "qy"]

        with pytest.raises(KeyError, match="Unknown field"):
            v.rename_fields({"nonexistent": "x"})

        with pytest.raises(ValueError, match="already exist"):
            v.rename_fields({"px": "intensity"})

    def test_remove_fields_preserves_remaining_data(self):
        v = make_line_vector()
        v.add_fields("extra", 1.0)
        v.remove_fields(("kx", "extra"))

        assert v.fields == ["intensity", "ky"]
        np.testing.assert_array_equal(
            v[0].array,
            np.array([[1.0, 100.0], [2.0, 200.0]]),
        )

    def test_mask_empties_deselected_cells(self):
        v = make_grid_vector()

        grid_mask = np.array([[True, False], [False, True], [True, True]])
        masked = v.mask(grid_mask)

        assert isinstance(masked, Vector)
        assert masked.shape == v.shape
        assert masked.fields == v.fields
        assert masked.units == v.units
        assert masked.name == v.name
        assert masked.row_counts() == [1, 0, 0, 1, 1, 1]
        np.testing.assert_array_equal(masked[0, 0].array, v[0, 0].array)
        assert masked[0, 1].array.shape == (0, 3)
        np.testing.assert_array_equal(masked[1, 1].array, v[1, 1].array)

        # The source Vector is untouched
        assert v.row_counts() == [1] * 6

    def test_mask_accepts_flat_and_integer_masks(self):
        v = make_grid_vector()
        grid_mask = np.array([[True, False], [False, True], [True, True]])

        # A flat mask in row-major cell order matches the grid-shaped mask
        np.testing.assert_array_equal(
            v.mask(grid_mask.reshape(-1)).flatten(),
            v.mask(grid_mask).flatten(),
        )

        # Integer masks are read as nonzero-means-keep
        np.testing.assert_array_equal(
            v.mask(grid_mask.astype(int)).flatten(),
            v.mask(grid_mask).flatten(),
        )

    def test_mask_over_fixed_grid_dimensions(self):
        # 1D
        line = make_line_vector()
        line_masked = line.mask(np.array([False, True, False, True]))
        assert line_masked.shape == (4,)
        assert line_masked.row_counts() == [0, 1, 0, 1]
        np.testing.assert_array_equal(
            line_masked.flatten(),
            np.array([[3.0, 30.0, 300.0], [6.0, 60.0, 600.0]]),
        )

        # 0D, where the mask is a single boolean
        assert line[0].mask(np.True_).array.shape == (2, 3)
        assert line[0].mask(np.False_).array.shape == (0, 3)

        # 3D
        cube = Vector.from_shape(shape=(2, 2, 2), fields=["kx", "ky"])
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    cube[i, j, k] = np.array([[float(i), float(j + k)]])
        cube_mask = np.zeros((2, 2, 2), dtype=bool)
        cube_mask[1, 0, 1] = True
        cube_masked = cube.mask(cube_mask)
        assert cube_masked.shape == (2, 2, 2)
        assert cube_masked.total_rows == 1
        np.testing.assert_array_equal(cube_masked[1, 0, 1].array, np.array([[1.0, 1.0]]))

    def test_mask_on_field_and_grid_selections(self):
        v = make_grid_vector()

        # Masking a field-selected view keeps only that field, like copy()
        kx_masked = v.select_fields("kx").mask(np.array([[True, False]] * 3))
        assert kx_masked.fields == ["kx"]
        np.testing.assert_array_equal(kx_masked.flatten(), np.array([[100.0], [110.0], [120.0]]))

        # Masking a fixed-grid selection is relative to that selection's shape
        sub = v[:2]
        sub_masked = sub.mask(np.array([[True, True], [False, False]]))
        assert sub_masked.shape == (2, 2)
        assert sub_masked.row_counts() == [1, 1, 0, 0]

    def test_mask_in_place_empties_cells_across_all_fields(self):
        v = make_grid_vector()

        assert v.mask(np.array([[True, False], [True, False], [True, False]])) is not None
        assert (
            v.mask(np.array([[True, False], [True, False], [True, False]]), modify_in_place=True)
            is None
        )
        assert v.shape == (3, 2)
        assert v.row_counts() == [1, 0, 1, 0, 1, 0]
        np.testing.assert_array_equal(
            v.flatten(),
            np.array([[0.0, 100.0, 200.0], [10.0, 110.0, 210.0], [20.0, 120.0, 220.0]]),
        )

        # Cells are emptied across every field, even through a field-selected view
        v2 = make_grid_vector()
        v2.select_fields("kx").mask(np.zeros((3, 2), dtype=bool), modify_in_place=True)
        assert v2.fields == ["intensity", "kx", "ky"]
        assert v2.row_counts() == [0] * 6

        # In-place masking of a grid selection leaves unselected cells alone
        v3 = make_grid_vector()
        v3[0].mask(np.array([False, True]), modify_in_place=True)
        assert v3.row_counts() == [0, 1, 1, 1, 1, 1]

    def test_filter_rows_keeps_selected_rows(self):
        v = make_line_vector()

        intensity = v.select_fields("intensity").flatten()
        filtered = v.filter_rows(intensity > 3.0)

        assert isinstance(filtered, Vector)
        assert filtered.shape == v.shape
        assert filtered.fields == v.fields
        assert filtered.units == v.units
        assert filtered.row_counts() == [0, 0, 2, 1]
        np.testing.assert_array_equal(
            filtered.flatten(),
            np.array([[4.0, 40.0, 400.0], [5.0, 50.0, 500.0], [6.0, 60.0, 600.0]]),
        )
        # The source Vector is untouched
        assert v.row_counts() == [2, 1, 2, 1]

        # (n_rows, 1) and 1D masks are equivalent, as are integer masks
        np.testing.assert_array_equal(
            v.filter_rows((intensity > 3.0)[:, 0]).flatten(), filtered.flatten()
        )
        np.testing.assert_array_equal(
            v.filter_rows(np.array([0, 0, 0, 1, 1, 1])).flatten(), filtered.flatten()
        )

        # A single-field Vector mask works too
        np.testing.assert_array_equal(
            v.filter_rows(np.greater(v.select_fields("intensity"), 3.0)).flatten(),
            filtered.flatten(),
        )

    def test_filter_rows_in_place_and_on_selections(self):
        v = make_line_vector()

        kr = v.select_fields("ky").flatten()[:, 0]
        assert v.filter_rows((kr > 150.0) & (kr < 550.0), modify_in_place=True) is None
        assert v.row_counts() == [1, 1, 2, 0]
        np.testing.assert_array_equal(v[0].array, np.array([[2.0, 20.0, 200.0]]))

        # Rows drop across all fields even when the mask came from a field view
        v2 = make_line_vector()
        kx = v2.select_fields("kx")
        kx.filter_rows(kx.flatten() < 45.0, modify_in_place=True)
        assert v2.fields == ["intensity", "kx", "ky"]
        assert v2.row_counts() == [2, 1, 1, 0]
        np.testing.assert_array_equal(v2[2].array, np.array([[4.0, 40.0, 400.0]]))

        # Filtering a field-selected view returns only that field, like copy()
        kx_only = make_line_vector().select_fields("kx")
        kx_filtered = kx_only.filter_rows(kx_only.flatten() >= 40.0)
        assert kx_filtered.fields == ["kx"]
        np.testing.assert_array_equal(kx_filtered.flatten(), np.array([[40.0], [50.0], [60.0]]))

        # A fixed-grid selection only sees its own rows, and leaves the rest alone
        v3 = make_line_vector()
        v3[:2].filter_rows(np.array([False, True, True]), modify_in_place=True)
        assert v3.row_counts() == [1, 1, 2, 1]
        np.testing.assert_array_equal(v3[0].array, np.array([[2.0, 20.0, 200.0]]))

    def test_filter_rows_edge_cases_and_validation(self):
        v = make_line_vector()

        np.testing.assert_array_equal(v.filter_rows(np.ones(6, dtype=bool)).flatten(), v.flatten())

        drop_all = v.filter_rows(np.zeros(6, dtype=bool))
        assert drop_all.row_counts() == [0, 0, 0, 0]
        assert drop_all.flatten().shape == (0, 3)

        empty = v[[]]
        assert empty.filter_rows(np.array([], dtype=bool)).flatten().shape == (0, 3)

        with pytest.raises(ValueError, match="expected 6 rows"):
            v.filter_rows(np.ones(5, dtype=bool))

        with pytest.raises(TypeError, match="boolean or integer"):
            v.filter_rows(np.ones(6, dtype=float))

        with pytest.raises(ValueError, match="Reduce multi-column masks"):
            v.filter_rows(np.ones((6, 3), dtype=bool))

        with pytest.raises(ValueError, match="exactly one field"):
            v.filter_rows(np.greater(v.select_fields("intensity", "kx"), 3.0))

        with pytest.raises(ValueError, match="matching per-cell row counts"):
            v.filter_rows(np.greater(v[:2].select_fields("intensity"), 3.0))

    def test_mask_edge_cases_and_validation(self):
        v = make_grid_vector()

        keep_all = v.mask(np.ones((3, 2), dtype=bool))
        np.testing.assert_array_equal(keep_all.flatten(), v.flatten())

        drop_all = v.mask(np.zeros((3, 2), dtype=bool))
        assert drop_all.row_counts() == [0] * 6
        assert drop_all.flatten().shape == (0, 3)

        empty = v[[], :]
        assert empty.mask(np.zeros((0, 2), dtype=bool)).flatten().shape == (0, 3)

        with pytest.raises(ValueError, match=r"expected \(3, 2\)"):
            v.mask(np.ones((2, 3), dtype=bool))

        with pytest.raises(ValueError, match="flat mask with 6 entries"):
            v.mask(np.ones(5, dtype=bool))

        with pytest.raises(TypeError, match="boolean or integer"):
            v.mask(np.ones((3, 2), dtype=float))

    def test_copy_is_deep(self):
        v = make_line_vector()
        v_copy = v.select_fields(["intensity", "kx"]).copy()

        v_copy[0].array[0, 0] = -1.0
        assert v[0].array[0, 0] == 1.0
        assert v_copy.fields == ["intensity", "kx"]
        assert v_copy.shape == (4,)

    def test_from_data_supports_nested_fixed_grid(self):
        data = [
            [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0], [5.0, 6.0]])],
            [np.array([[7.0, 8.0]]), np.array([[9.0, 10.0]])],
        ]
        v = Vector.from_data(data=data, fields=["a", "b"], units=["u1", "u2"], name="nested")

        assert v.shape == (2, 2)
        assert v.fields == ["a", "b"]
        assert v.units == ["u1", "u2"]
        assert v.name == "nested"
        np.testing.assert_array_equal(v[0, 1].array, np.array([[3.0, 4.0], [5.0, 6.0]]))

        tuple_cells = [
            ([1.0, 2.0], [3.0, 4.0]),
            ([5.0, 6.0], [7.0, 8.0], [9.0, 10.0]),
        ]
        tuple_vector = Vector.from_data(data=tuple_cells, fields=["a", "b"])
        assert tuple_vector.shape == (2,)
        np.testing.assert_array_equal(tuple_vector[0].array, np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_array_equal(
            tuple_vector[1].array,
            np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        )

        tuple_data = (np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]))
        tuple_outer = Vector.from_data(data=tuple_data, fields=["a", "b"])
        assert tuple_outer.shape == (2,)

        with pytest.raises(TypeError, match="Data must be a list or tuple"):
            Vector.from_data(data=np.array([1, 2, 3]))  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="same number of fields"):
            Vector.from_data(data=[np.array([[1.0, 2.0]]), np.array([[1.0, 2.0, 3.0]])])

    def test_save_and_load_round_trip(self, tmp_path):
        v = make_grid_vector()
        v.add_fields("extra", v.select_fields("intensity") + 1.0)

        path = tmp_path / "vector_test.zip"
        v.save(path, mode="o", compression_level=4)

        with zipfile.ZipFile(path) as zf:
            names = [info.filename for info in zf.infolist()]
        assert len(names) < 30
        assert "_state/data/zarr.json" in names
        assert all(not name.startswith("_selection_coords/") for name in names)

        loaded = load(path)
        assert isinstance(loaded, Vector)
        assert loaded.shape == v.shape
        assert loaded.fields == v.fields
        assert loaded.units == v.units
        np.testing.assert_array_equal(loaded[2, 1].array, v[2, 1].array)
