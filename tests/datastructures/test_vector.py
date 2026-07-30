import zipfile

import numpy as np
import pytest
import torch

from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import load

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def assert_rows(actual: torch.Tensor, expected: list[list[float]]) -> None:
    """Compare a rowwise tensor result against literal expected values."""
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=actual.dtype))


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
        assert v1.dtype == torch.float32
        assert v1.device == "cpu"
        assert v1.fields == ["a", "b", "c"]
        assert v1.units == ["none", "none", "none"]
        assert v1.name == "2d ragged array"
        assert v1[0, 0].tensor.shape == (0, 3)
        torch.testing.assert_close(v1[0, 0].flatten(), v1[0, 0].tensor)

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
            "  units: ['none', 'none', 'none']\n"
            "  dtype: torch.float32, device: cpu"
        )

    def test_indexing_and_tensor_contract(self):
        v = make_grid_vector()

        assert isinstance(v[:2, 1], Vector)
        assert v[:2, 1].shape == (2,)
        assert v[1].shape == (2,)
        assert v[1, 1].shape == ()
        assert_rows(v[-1, -1].tensor, [[21.0, 121.0, 221.0]])

        with pytest.raises(ValueError):
            _ = v[:, 1].tensor

        result = v[[-1, 0], 1]
        assert result.shape == (2,)
        assert result.num_cells == 2
        assert_rows(result[0].tensor, [[21.0, 121.0, 221.0]])
        assert_rows(result[1].tensor, [[1.0, 101.0, 201.0]])

        # Torch tensors work as fancy indices alongside lists and ndarrays.
        assert v[torch.tensor([0, 2])].shape == (2, 2)

    def test_select_fields_and_chaining_equivalence(self):
        v = make_line_vector()

        selected = v.select_fields("kx")
        assert selected.fields == ["kx"]
        assert selected.units == ["px"]
        assert selected.shape == v.shape

        torch.testing.assert_close(
            v.select_fields("kx")[2].tensor,
            v[2].select_fields("kx").tensor,
        )

        with pytest.raises(KeyError):
            v.select_fields("missing")

        with pytest.raises(TypeError):
            _ = v["kx"]

        with pytest.raises(TypeError):
            _ = v[1, "kx"]

        multi = v.select_fields("intensity", "kx")
        assert multi.fields == ["intensity", "kx"]
        assert multi.dtype == torch.float32
        assert multi.total_rows == 6
        assert multi.row_counts() == [2, 1, 2, 1]

    def test_select_fields_respects_requested_order(self):
        v = Vector.from_shape(shape=(1,), fields=["a", "b", "c"])
        v[0] = np.array([[1.0, 2.0, 3.0]])

        # Data must come back in the order asked for, matching .fields --
        # including the full-width reorder, which used to fall through a
        # fast path that returned storage order.
        reordered = v.select_fields("c", "b", "a")
        assert reordered.fields == ["c", "b", "a"]
        assert_rows(reordered[0].tensor, [[3.0, 2.0, 1.0]])
        assert_rows(reordered.flatten(), [[3.0, 2.0, 1.0]])

        subset = v.select_fields("c", "a")
        assert subset.fields == ["c", "a"]
        assert_rows(subset[0].tensor, [[3.0, 1.0]])

        # Ascending contiguous selections keep the writable-view contract.
        contiguous = v.select_fields("a", "b")
        contiguous[0].tensor[0, 0] = -1.0
        assert v[0].tensor[0, 0] == -1.0

    def test_tensor_mutation_writes_through_for_single_field(self):
        v = make_line_vector()
        cell = v.select_fields("kx")[1].tensor
        cell[0, 0] = 99.0
        assert v[1].tensor[0, 1] == 99.0

    def test_set_flattened_updates_rowwise(self):
        v = make_line_vector()
        kx = v.select_fields("kx")

        flat_kx = kx.flatten()
        mask = flat_kx >= 30.0
        flat_kx[mask[:, 0], 0] = -1.0
        kx.set_flattened(flat_kx)

        assert_rows(kx.flatten(), [[10.0], [20.0], [-1.0], [-1.0], [-1.0], [-1.0]])

    def test_field_arithmetic_with_scalar_and_array(self):
        v = make_line_vector()

        kx = v.select_fields("kx")
        kx += 10
        assert_rows(
            v.select_fields("kx").flatten(),
            [[20.0], [30.0], [40.0], [50.0], [60.0], [70.0]],
        )

        # NumPy arrays are still accepted as input and converted at the boundary.
        v.select_fields("kx")[...] += np.arange(6)
        assert_rows(
            v.select_fields("kx").flatten(),
            [[20.0], [31.0], [42.0], [53.0], [64.0], [75.0]],
        )

        # ... as are torch tensors.
        v.select_fields("kx")[...] += torch.ones(6)
        assert_rows(
            v.select_fields("kx").flatten(),
            [[21.0], [32.0], [43.0], [54.0], [65.0], [76.0]],
        )

        summed = v.select_fields("intensity") + v.select_fields("ky")
        assert_rows(
            summed.flatten(),
            [[101.0], [202.0], [303.0], [404.0], [505.0], [606.0]],
        )

    def test_power_operations(self):
        v = make_line_vector()

        squared = v.select_fields("intensity") ** 2
        assert_rows(squared.flatten(), [[1.0], [4.0], [9.0], [16.0], [25.0], [36.0]])

        intensity = v.select_fields("intensity")
        intensity **= 2
        assert_rows(intensity.flatten(), [[1.0], [4.0], [9.0], [16.0], [25.0], [36.0]])

        reverse = 2 ** v.select_fields("intensity")
        assert_rows(
            reverse.flatten(),
            [[2.0], [16.0], [512.0], [65536.0], [33554432.0], [68719476736.0]],
        )

    def test_unary_mod_and_floor_division_operations(self):
        v = make_line_vector()

        negative = -v.select_fields("intensity")
        assert_rows(negative.flatten(), [[-1.0], [-2.0], [-3.0], [-4.0], [-5.0], [-6.0]])

        absolute = abs(negative)
        assert_rows(absolute.flatten(), [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])

        floored = v.select_fields("ky") // 150
        assert_rows(floored.flatten(), [[0.0], [1.0], [2.0], [2.0], [3.0], [4.0]])

        modded = v.select_fields("ky") % 150
        assert_rows(modded.flatten(), [[100.0], [50.0], [0.0], [100.0], [50.0], [0.0]])

        ky = v.select_fields("ky")
        ky //= 150
        assert_rows(ky.flatten(), [[0.0], [1.0], [2.0], [2.0], [3.0], [4.0]])

        intensity = v.select_fields("intensity")
        intensity %= 2
        assert_rows(intensity.flatten(), [[1.0], [0.0], [1.0], [0.0], [1.0], [0.0]])

    def test_torch_function_support(self):
        v = make_line_vector()
        kx = v.select_fields("kx")

        sine = torch.sin(kx)
        assert isinstance(sine, Vector)
        torch.testing.assert_close(sine.flatten(), torch.sin(kx.flatten()))

        maximum = torch.maximum(kx, torch.tensor(35.0))
        assert_rows(maximum.flatten(), [[35.0], [35.0], [35.0], [40.0], [50.0], [60.0]])

        # Multi-output functions return a tuple of Vectors.
        mantissa, exponent = torch.frexp(v.select_fields("intensity"))
        assert isinstance(mantissa, Vector)
        assert_rows(mantissa.flatten(), [[0.5], [0.5], [0.75], [0.5], [0.625], [0.75]])

        # Reductions do not have rowwise shape, so they pass through unwrapped.
        total = torch.sum(kx)
        assert isinstance(total, torch.Tensor)
        assert not isinstance(total, Vector)
        torch.testing.assert_close(total, torch.tensor(210.0))

        # Mismatched ragged structure is rejected.
        with pytest.raises(ValueError, match="matching per-cell row counts"):
            torch.add(v[:2].select_fields("kx"), v[1:3].select_fields("kx"))

    def test_numpy_interop_and_ufunc_optout(self):
        v = make_line_vector()
        kx = v.select_fields("kx")

        # numpy() is the NumPy counterpart of flatten()
        flat = kx.numpy()
        assert isinstance(flat, np.ndarray)
        np.testing.assert_array_equal(
            flat, np.array([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]], dtype=np.float32)
        )

        # NumPy ufuncs are explicitly disabled; use the torch equivalent.
        with pytest.raises(TypeError):
            np.sin(kx)

        # ndarray-on-the-left arithmetic still defers to Vector.__radd__
        result = np.float64(1.0) + kx
        assert isinstance(result, Vector)
        assert_rows(result.flatten(), [[11.0], [21.0], [31.0], [41.0], [51.0], [61.0]])

    def test_dtype_and_device_options(self):
        default = Vector.from_shape(shape=(2,), fields=["a"])
        assert default.dtype == torch.float32

        doubled = Vector.from_shape(shape=(2,), fields=["a"], dtype=torch.float64)
        doubled[0] = np.array([[1.0], [2.0]])
        assert doubled.dtype == torch.float64
        assert doubled[0].tensor.dtype == torch.float64

        # The buffer dtype is authoritative: float64 input does not widen a
        # float32 Vector.
        narrow = Vector.from_shape(shape=(2,), fields=["a"])
        narrow[0] = np.array([[1.0], [2.0]], dtype=np.float64)
        assert narrow.dtype == torch.float32

        from_data = Vector.from_data(
            data=[np.array([[1.0, 2.0]])], fields=["a", "b"], dtype=torch.float64
        )
        assert from_data.dtype == torch.float64

    def test_out_of_place_arithmetic_uses_torch_dtype_promotion(self):
        integer = Vector.from_shape(shape=(1,), fields=["a"], dtype=torch.int64)
        integer[0] = [[3]]
        divided = integer / 2
        assert divided.dtype == torch.float32
        assert_rows(divided.flatten(), [[1.5]])

        real = Vector.from_shape(shape=(1,), fields=["a"])
        real[0] = [[1.0]]

        doubled = real + torch.tensor([[2.0]], dtype=torch.float64)
        assert doubled.dtype == torch.float64
        assert_rows(doubled.flatten(), [[3.0]])

        complex_result = real + torch.tensor([[2.0j]], dtype=torch.complex64)
        assert complex_result.dtype == torch.complex64
        torch.testing.assert_close(
            complex_result.flatten(), torch.tensor([[1.0 + 2.0j]], dtype=torch.complex64)
        )

    def test_unknown_shape_preserving_torch_function_returns_tensor(self):
        v = make_line_vector()

        flipped = torch.flip(v.select_fields("kx"), dims=(0,))

        assert isinstance(flipped, torch.Tensor)
        assert not isinstance(flipped, Vector)
        torch.testing.assert_close(flipped, torch.flip(v.select_fields("kx").flatten(), dims=(0,)))

    def test_numpy_detaches_tensor(self):
        v = Vector.from_shape(shape=(1,), fields=["a"])
        v._state["data"] = torch.tensor([[1.0]], requires_grad=True)
        v._state["cell_starts"][0] = 0
        v._state["cell_lengths"][0] = 1

        array = v.numpy()

        np.testing.assert_array_equal(array, np.array([[1.0]], dtype=np.float32))
        assert not array.flags.writeable

    def test_device_property_and_to_cpu(self):
        v = make_line_vector()
        assert v.device == "cpu"
        assert v.to("cpu") is v
        assert v.device == "cpu"

    @requires_cuda
    def test_to_cuda_moves_shared_storage(self):
        v = make_grid_vector()
        view = v.select_fields("kx")

        v.to("cuda")
        assert v.device.startswith("cuda")
        # Views share _state, so the move is visible through them too.
        assert view.device.startswith("cuda")
        assert v[1, 1].tensor.device.type == "cuda"
        # Offset bookkeeping deliberately stays on the CPU.
        assert v._state["cell_starts"].device.type == "cpu"

        view += 1.0
        assert view.flatten().device.type == "cuda"
        assert isinstance(v.numpy(), np.ndarray)

    @requires_cuda
    def test_cuda_vector_saves_and_loads_to_cpu(self, tmp_path):
        v = make_grid_vector().to("cuda")
        path = tmp_path / "cuda_vector.zip"
        v.save(path, mode="o")

        # save() must leave the live object untouched on its original device.
        assert v.device.startswith("cuda")

        loaded = load(path)
        assert loaded.device == "cpu"
        torch.testing.assert_close(loaded[2, 1].tensor, v[2, 1].tensor.cpu())

    def test_field_assignment_from_vector_expression(self):
        v = make_line_vector()
        scale = 2.5

        v[:2].select_fields("intensity")[...] = v[2:4].select_fields("intensity") * scale
        assert_rows(v[:2].select_fields("intensity").flatten(), [[10.0], [12.5], [15.0]])

    def test_field_assignment_requires_matching_per_cell_row_counts(self):
        v = make_line_vector()
        with pytest.raises(ValueError, match="Per-cell row counts must match"):
            v[:2].select_fields("intensity")[...] = v[1:3].select_fields("intensity")

    def test_full_cell_assignment_allows_row_count_changes(self):
        v = make_line_vector()

        v[1] = v[0]
        assert v[1].tensor.shape == (2, 3)
        torch.testing.assert_close(v[1].tensor, v[0].tensor)

        v[0:2] = v[1:3]
        assert v[0].tensor.shape == (2, 3)
        assert v[1].tensor.shape == (2, 3)

        broadcast_cell = np.array([[9.0, 8.0, 7.0]])
        v[[0, 3]] = broadcast_cell
        assert_rows(v[0].tensor, [[9.0, 8.0, 7.0]])
        assert_rows(v[3].tensor, [[9.0, 8.0, 7.0]])

    def test_append_rows_and_compact(self):
        v = make_line_vector()

        v.append_rows(1, np.array([[7.0, 70.0, 700.0]]))
        assert_rows(v[1].tensor, [[3.0, 30.0, 300.0], [7.0, 70.0, 700.0]])

        v[1] = torch.tensor([[8.0, 80.0, 800.0]])
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
        assert_rows(selected[0, 0].tensor, [[1.0, 101.0, 201.0]])
        assert_rows(selected[1, 0].tensor, [[21.0, 121.0, 221.0]])

        # Torch bool masks work the same way.
        assert v[torch.tensor([True, False, True]), cols].shape == (2, 1)

        with pytest.raises(IndexError):
            _ = v[np.array([[True, False], [False, True]])]

    def test_repeated_fancy_indices_are_readable_but_not_writable(self):
        v = make_line_vector()
        repeated = v[[0, 0]].select_fields("intensity")

        assert repeated.shape == (2,)
        assert_rows(repeated.flatten(), [[1.0], [2.0], [1.0], [2.0]])
        assert_rows((repeated + 1).flatten(), [[2.0], [3.0], [2.0], [3.0]])

        before = v.flatten()
        with pytest.raises(ValueError, match="repeated cell indices"):
            repeated += 1
        torch.testing.assert_close(v.flatten(), before)

        with pytest.raises(ValueError, match="repeated cell indices"):
            repeated.set_flattened(torch.zeros((4, 1)))
        torch.testing.assert_close(v.flatten(), before)

        with pytest.raises(ValueError, match="repeated cell indices"):
            v[[0, 0]] = torch.tensor([[9.0, 9.0, 9.0]])
        torch.testing.assert_close(v.flatten(), before)

    def test_empty_selection_is_valid_and_no_op_for_scalar_math(self):
        v = make_grid_vector()
        before = v.copy().flatten()

        empty = v[[], :]
        assert empty.shape == (0, 2)
        assert empty.flatten().shape == (0, 3)

        empty.select_fields("kx")[...] += 1
        torch.testing.assert_close(v.flatten(), before)

    def test_add_fields_defaults_expression_and_multiple_values(self):
        v = make_line_vector()

        v.add_fields(("h", "k"))
        assert v.fields == ["intensity", "kx", "ky", "h", "k"]
        assert torch.isnan(v[0].tensor[:, 3:5]).all()

        v.add_fields("field_out", v.select_fields("kx") + v.select_fields("ky"))
        assert_rows(
            v.select_fields("field_out").flatten(),
            [[110.0], [220.0], [330.0], [440.0], [550.0], [660.0]],
        )

        v2 = make_line_vector()
        v2.add_fields(("h", "k"), (1.0, np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])))
        assert_rows(v2.select_fields("h").flatten(), [[1.0]] * 6)
        assert_rows(
            v2.select_fields("k").flatten(),
            [[5.0], [6.0], [7.0], [8.0], [9.0], [10.0]],
        )

        with pytest.raises(ValueError, match="all fields are selected"):
            v2.select_fields("kx").add_fields("bad")

    def test_rename_fields(self):
        v = make_line_vector()
        kx_data = v.select_fields("kx").flatten().clone()

        v.rename_fields({"kx": "qx", "ky": "qy"})
        assert v.fields == ["intensity", "qx", "qy"]
        torch.testing.assert_close(v.select_fields("qx").flatten(), kx_data)

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
        assert_rows(v[0].tensor, [[1.0, 100.0], [2.0, 200.0]])

    def test_copy_is_deep(self):
        v = make_line_vector()
        v_copy = v.select_fields(["intensity", "kx"]).copy()

        v_copy[0].tensor[0, 0] = -1.0
        assert v[0].tensor[0, 0] == 1.0
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
        assert_rows(v[0, 1].tensor, [[3.0, 4.0], [5.0, 6.0]])

        tuple_cells = [
            ([1.0, 2.0], [3.0, 4.0]),
            ([5.0, 6.0], [7.0, 8.0], [9.0, 10.0]),
        ]
        tuple_vector = Vector.from_data(data=tuple_cells, fields=["a", "b"])
        assert tuple_vector.shape == (2,)
        assert_rows(tuple_vector[0].tensor, [[1.0, 2.0], [3.0, 4.0]])
        assert_rows(tuple_vector[1].tensor, [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])

        tuple_data = (np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]))
        tuple_outer = Vector.from_data(data=tuple_data, fields=["a", "b"])
        assert tuple_outer.shape == (2,)

        # Torch tensors are accepted as cell payloads too.
        tensor_vector = Vector.from_data(
            data=[torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])], fields=["a", "b"]
        )
        assert tensor_vector.shape == (2,)
        assert_rows(tensor_vector[1].tensor, [[3.0, 4.0]])

        with pytest.raises(TypeError, match="Data must be a list or tuple"):
            Vector.from_data(data=np.array([1, 2, 3]))  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="same number of fields"):
            Vector.from_data(data=[np.array([[1.0, 2.0]]), np.array([[1.0, 2.0, 3.0]])])

    def test_to_polars_line_vector(self):
        pytest.importorskip("polars")
        v = make_line_vector()

        df = v.to_polars()
        assert df.columns == ["dim_0", "intensity", "kx", "ky"]
        assert df.height == v.total_rows == 6
        assert df["dim_0"].to_list() == [0, 0, 1, 2, 2, 3]
        np.testing.assert_array_equal(df.select(v.fields).to_numpy(), v.numpy())

    def test_to_polars_grid_and_dim_names(self):
        pytest.importorskip("polars")
        v = make_grid_vector()

        df = v.to_polars()
        assert df.columns == ["dim_0", "dim_1", "intensity", "kx", "ky"]
        assert df.height == 6
        assert list(zip(df["dim_0"], df["dim_1"])) == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
        ]

        renamed = v.to_polars(dim_names=("rx", "ry"))
        assert renamed.columns == ["rx", "ry", "intensity", "kx", "ky"]

    def test_to_polars_respects_current_selection(self):
        pytest.importorskip("polars")
        v = make_grid_vector()

        # Field selection narrows the value columns, grid columns are kept.
        field_view = v.select_fields("kx").to_polars()
        assert field_view.columns == ["dim_0", "dim_1", "kx"]

        # Fixed-grid selection reports *root* grid coordinates, not local ones.
        cell_view = v[2].to_polars()
        assert cell_view.height == 2
        assert cell_view["dim_0"].to_list() == [2, 2]
        assert cell_view["dim_1"].to_list() == [0, 1]

        scalar_view = v[1, 1].select_fields("ky").to_polars()
        assert scalar_view.columns == ["dim_0", "dim_1", "ky"]
        assert scalar_view.to_dicts() == [{"dim_0": 1, "dim_1": 1, "ky": 211.0}]

    def test_to_polars_handles_empty_cells_and_zero_dim_grid(self):
        pytest.importorskip("polars")

        sparse = Vector.from_shape(shape=(3,), fields=["a"])
        sparse[0] = np.array([[1.0], [2.0]])
        sparse[2] = np.array([[9.0]])
        df = sparse.to_polars()
        assert df["dim_0"].to_list() == [0, 0, 2]
        assert df["a"].to_list() == [1.0, 2.0, 9.0]

        empty = Vector.from_shape(shape=(2,), fields=["a", "b"]).to_polars()
        assert empty.height == 0
        assert empty.columns == ["dim_0", "a", "b"]

        # A 0D fixed grid has no grid axes, so no index columns are emitted.
        scalar_grid = Vector.from_shape(shape=(), fields=["a", "b"])
        scalar_grid[...] = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert scalar_grid.to_polars().columns == ["a", "b"]

    def test_to_polars_rejects_colliding_and_mismatched_names(self):
        pytest.importorskip("polars")

        with pytest.raises(ValueError, match="collide with field name"):
            Vector.from_shape(shape=(2,), fields=["dim_0", "b"]).to_polars()

        v = make_grid_vector()
        with pytest.raises(ValueError, match="Expected 2 dim_names, got 1"):
            v.to_polars(dim_names=("only_one",))

        with pytest.raises(ValueError, match="Duplicate dim_names"):
            v.to_polars(dim_names=("same", "same"))

    def test_save_and_load_round_trip(self, tmp_path):
        v = make_grid_vector()
        v.add_fields("extra", v.select_fields("intensity") + 1.0)

        path = tmp_path / "vector_test.zip"
        v.save(path, mode="o", compression_level=4)

        with zipfile.ZipFile(path) as zf:
            names = [info.filename for info in zf.infolist()]
        assert len(names) < 30
        # The row buffer is written as a compressed Zarr array, not a torch blob.
        assert "_state/data/zarr.json" in names
        assert all(not name.startswith("_selection_coords/") for name in names)

        # save() must not leave the live object holding NumPy state.
        assert isinstance(v._state["data"], torch.Tensor)

        loaded = load(path)
        assert isinstance(loaded, Vector)
        assert loaded.shape == v.shape
        assert loaded.fields == v.fields
        assert loaded.units == v.units
        assert loaded.dtype == torch.float32
        assert loaded.device == "cpu"
        assert isinstance(loaded._state["cell_starts"], torch.Tensor)
        torch.testing.assert_close(loaded[2, 1].tensor, v[2, 1].tensor)

    def test_post_load_rehydrates_numpy_state(self):
        """Vectors saved before the torch migration hold NumPy in _state."""
        v = make_grid_vector()

        # Simulate a NumPy-era file: every buffer restored as an ndarray, with
        # float64 data and a list-valued shape.
        v._state["data"] = v._state["data"].numpy().astype(np.float64)
        v._state["cell_starts"] = v._state["cell_starts"].numpy()
        v._state["cell_lengths"] = v._state["cell_lengths"].numpy()
        v._state["shape"] = [3, 2]
        v._selection_shape = [3, 2]
        v._selection_indices = np.arange(6, dtype=np.int64)

        v._post_load()

        assert isinstance(v._state["data"], torch.Tensor)
        assert isinstance(v._state["cell_starts"], torch.Tensor)
        assert isinstance(v._selection_indices, torch.Tensor)
        assert v._state["shape"] == (3, 2)
        assert v._selection_shape == (3, 2)
        # The stored precision is preserved rather than coerced to the default.
        assert v.dtype == torch.float64
        assert_rows(v[2, 1].tensor, [[21.0, 121.0, 221.0]])
