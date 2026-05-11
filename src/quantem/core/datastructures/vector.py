from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import (
    validate_fields,
    validate_num_fields,
    validate_shape,
    validate_vector_units,
)


class Vector(AutoSerialize):
    """Ragged cell data on a fixed grid.

    A ``Vector`` has two independent axes of structure:
    - fixed-grid dimensions given by ``shape``
    - ragged rows stored inside each fixed-grid cell

    Each ragged row has one value per named field, so each cell behaves like a
    small 2D array with shape ``(n_rows, num_fields)``, where ``n_rows`` may
    vary from cell to cell.

    Parameters
    ----------
    shape : tuple of int
        Fixed-grid shape.
    fields : sequence of str
        Field names in column order.
    units : sequence of str, optional
        Units corresponding to ``fields``. If omitted, units default to
        ``"none"`` for all fields.
    name : str, optional
        Descriptive name for the Vector.
    metadata : dict, optional
        Additional user metadata.

    Notes
    -----
    The public API keeps fixed-grid indexing and field selection separate:
    - use ``[]`` for fixed-grid indexing
    - use ``select_fields(...)`` for field selection

    Fixed-grid indexing always returns a ``Vector``. A 0D selection exposes its
    underlying cell array through ``.array``. Multi-cell selections can be
    concatenated with ``flatten()``.

    The internal representation is compact:
    - ``_state["data"]`` stores all ragged rows in one numeric 2D array
    - ``_state["cell_starts"]`` stores the start offset for each cell
    - ``_state["cell_lengths"]`` stores the row count for each cell

    A ``Vector`` selection is a write-through view over shared storage. Views
    track only the selected fixed-grid shape, selected cell indices, and selected
    field names.

    Examples
    --------
    Create a Vector and assign one cell:

    >>> import numpy as np
    >>> v = Vector.from_shape((2, 2), fields=("kx", "ky", "intensity"))
    >>> v[0, 0] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> v[0, 0].array.shape
    (2, 3)

    Select fields and apply in-place arithmetic:

    >>> kx = v.select_fields("kx")
    >>> kx += 16
    >>> kx.flatten().shape
    (2, 1)

    Apply a rowwise transform with ``flatten()`` and ``set_flattened()``:

    >>> kx = v.select_fields("kx")
    >>> ky = v.select_fields("ky")
    >>> kx.set_flattened(
    ...     np.where(
    ...         ((kx.flatten() - 16) ** 2 + (ky.flatten() - 16) ** 2) < 12,
    ...         10,
    ...         kx.flatten(),
    ...     )
    ... )
    """

    __array_priority__ = 1000
    _token = object()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        shape: tuple[int, ...],
        fields: Sequence[str],
        units: Sequence[str] | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not self._token:
            raise RuntimeError(
                "Use Vector.from_shape() or Vector.from_data() to instantiate this class."
            )
        root_shape = validate_shape(shape)
        root_fields = validate_fields(list(fields))
        root_units = validate_vector_units(
            list(units) if units is not None else None,
            len(root_fields),
        )

        self._state = {
            "shape": root_shape,
            "fields": list(root_fields),
            "units": list(root_units),
            "name": name or f"{len(root_shape)}d ragged array",
            "metadata": dict(metadata or {}),
            "data": np.empty((0, len(root_fields)), dtype=float),
            "cell_starts": np.zeros(_cell_count(root_shape), dtype=np.int64),
            "cell_lengths": np.zeros(_cell_count(root_shape), dtype=np.int64),
        }
        self._selection_shape = root_shape
        self._selection_indices: NDArray[np.int64] | None = None
        self._selected_fields: tuple[str, ...] | None = None

    @classmethod
    def _from_view(
        cls,
        state: dict[str, Any],
        selection_shape: tuple[int, ...],
        selection_indices: NDArray[np.int64] | None,
        selected_fields: tuple[str, ...] | None,
    ) -> "Vector":
        """Build a view that shares backing storage with another Vector."""
        obj = cls.__new__(cls)
        obj._state = state
        obj._selection_shape = selection_shape
        obj._selection_indices = (
            None if selection_indices is None else selection_indices.astype(np.int64, copy=False)
        )
        obj._selected_fields = selected_fields
        return obj

    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, ...],
        num_fields: int | None = None,
        fields: Sequence[str] | None = None,
        units: Sequence[str] | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Vector":
        """Create an empty Vector with the given fixed-grid shape and fields."""
        fields = _resolve_fields(fields, num_fields, None)
        return cls(
            shape=shape,
            fields=fields,
            units=units,
            name=name,
            metadata=metadata,
            _token=cls._token,
        )

    @classmethod
    def from_data(
        cls,
        data: Sequence[Any],
        num_fields: int | None = None,
        fields: Sequence[str] | None = None,
        units: Sequence[str] | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Vector":
        """Create a Vector from nested fixed-grid data.

        The outer nesting defines the fixed-grid shape. Each leaf must coerce to a
        2D cell array with consistent field count across all cells.
        """
        if not isinstance(data, (list, tuple)):
            raise TypeError(f"Data must be a list or tuple, got {type(data)}")
        root_shape, cell_arrays = _flatten_fixed_grid(data) if len(data) > 0 else ((0,), [])
        inferred_counts = {array.shape[1] for array in cell_arrays}
        if len(inferred_counts) > 1:
            raise ValueError("All cell arrays must have the same number of fields.")
        inferred_fields = cell_arrays[0].shape[1] if cell_arrays else 0

        vector = cls(
            shape=root_shape,
            fields=_resolve_fields(fields, num_fields, inferred_fields),
            units=units,
            name=name,
            metadata=metadata,
            _token=cls._token,
        )
        vector._replace_cells(np.arange(len(cell_arrays), dtype=np.int64), cell_arrays)
        return vector

    # ------------------------------------------------------------------ #
    # Identity properties
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        """Human-readable Vector name."""
        return self._state["name"]

    @name.setter
    def name(self, value: str) -> None:
        self._state["name"] = str(value)

    @property
    def metadata(self) -> dict[str, Any]:
        """Mutable metadata dictionary shared by all views."""
        return self._state["metadata"]

    # ------------------------------------------------------------------ #
    # Shape & structure properties
    # ------------------------------------------------------------------ #

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the fixed-grid shape of this selection."""
        return self._selection_shape

    @property
    def fields(self) -> list[str]:
        """Return selected field names in column order."""
        if self._selected_fields is None:
            return list(self._state["fields"])
        return list(self._selected_fields)

    @property
    def units(self) -> list[str]:
        """Return units for the selected fields."""
        lookup = dict(zip(self._state["fields"], self._state["units"]))
        return [lookup[field] for field in self.fields]

    @property
    def num_fields(self) -> int:
        """Return the number of selected fields."""
        return len(self.fields)

    @property
    def num_cells(self) -> int:
        """Return the number of fixed-grid cells in the current selection."""
        return int(self._selected_cell_indices().size)

    @property
    def total_rows(self) -> int:
        """Return the total ragged-row count in the current selection."""
        return int(self._state["cell_lengths"][self._selected_cell_indices()].sum())

    @property
    def dtype(self) -> np.dtype[Any]:
        """Return the NumPy dtype of the backing row buffer."""
        return self._state["data"].dtype

    # ------------------------------------------------------------------ #
    # Data access
    # ------------------------------------------------------------------ #

    @property
    def array(self) -> NDArray[Any]:
        """Return the selected cell as a NumPy array.

        This is only valid for 0D selections. Single-field and contiguous
        multi-field selections return writable views into the backing storage.
        Non-contiguous multi-field selections return a copy because NumPy cannot
        expose a writable column-subset view for that layout.
        """
        if self.shape != ():
            raise ValueError(".array is only valid when the selection contains exactly one cell.")
        cell = self._cell_matrix(self._selected_cell_indices()[0])
        cols = self._field_indices()
        if cols.size == self._full_num_fields:
            return cell
        if cols.size == 1:
            col = int(cols[0])
            return cell[:, col : col + 1]
        if _is_contiguous(cols):
            return cell[:, int(cols[0]) : int(cols[-1]) + 1]
        return cell[:, cols].copy()

    def flatten(self) -> NDArray[Any]:
        """Concatenate selected cells in row-major order.

        Returns a 2D array with shape ``(total_rows, num_fields)`` even for
        single-field selections.
        """
        arrays = [
            self._selected_cell_matrix(index)
            for index in self._selected_cell_indices()
            if self._cell_row_count(index) > 0
        ]
        if arrays:
            return np.vstack(arrays)

        dtype = self._state["data"].dtype if self._state["data"].ndim == 2 else float
        return np.empty((0, self.num_fields), dtype=dtype)

    def row_counts(self) -> list[int]:
        """Return per-cell row counts in the current selection order."""
        return [self._cell_row_count(int(index)) for index in self._selected_cell_indices()]

    # ------------------------------------------------------------------ #
    # Field management
    # ------------------------------------------------------------------ #

    def select_fields(self, *field_names: str | Sequence[str]) -> "Vector":
        """Return a view containing only the requested fields.

        Accepted forms:
        - ``select_fields("kx")``
        - ``select_fields("kx", "ky")``
        - ``select_fields(["kx", "ky"])``
        """
        if not field_names:
            raise ValueError("At least one field name is required.")
        if len(field_names) == 1 and not isinstance(field_names[0], str):
            selected = _normalize_field_names(field_names[0])
        elif not all(isinstance(n, str) for n in field_names):
            raise TypeError(
                "select_fields(...) expects field names as strings or one sequence of strings."
            )
        else:
            selected = _normalize_field_names(field_names)  # type: ignore[arg-type]
        available = set(self.fields)
        missing = [field for field in selected if field not in available]
        if missing:
            raise KeyError(f"Unknown field(s): {missing}")

        selected_fields = None if selected == tuple(self._state["fields"]) else selected
        return self._from_view(
            self._state,
            self.shape,
            self._selection_indices,
            selected_fields,
        )

    def add_fields(
        self,
        names: str | Sequence[str],
        values: Any | None = None,
        units: str | Sequence[str] | None = None,
    ) -> None:
        """Add one or more new fields to the full Vector schema."""
        self._require_full_field_view("add_fields")
        new_fields = _normalize_field_names(names)
        if any(field in self._state["fields"] for field in new_fields):
            raise ValueError("One or more new field names already exist.")

        new_units = _normalize_units(units, len(new_fields))
        old_fields = list(self._state["fields"])
        self._state["fields"].extend(new_fields)
        self._state["units"].extend(new_units)
        self._expand_storage(len(new_fields))

        if values is None:
            return

        target = self.select_fields(*new_fields)
        if (
            len(new_fields) > 1
            and isinstance(values, (list, tuple))
            and len(values) == len(new_fields)
        ):
            for field, value in zip(new_fields, values):
                target.select_fields(field)[...] = value
        else:
            target[...] = values

        if self._selected_fields is not None and tuple(old_fields) == self._selected_fields:
            self._selected_fields = None

    def rename_fields(self, mapping: dict[str, str]) -> None:
        """Rename one or more fields in-place.

        Parameters
        ----------
        mapping : dict
            Maps each old field name to its new name, e.g.
            ``{"kx": "qx", "ky": "qy"}``.
        """
        old_field_set = set(self._state["fields"])
        missing = [old for old in mapping if old not in old_field_set]
        if missing:
            raise KeyError(f"Unknown field(s): {missing}")
        new_names = list(mapping.values())
        conflicts = [n for n in new_names if n in old_field_set and n not in mapping]
        if conflicts:
            raise ValueError(f"New field name(s) already exist: {conflicts}")
        validate_fields(new_names)

        rename = {old: new for old, new in mapping.items()}
        self._state["fields"] = [rename.get(f, f) for f in self._state["fields"]]
        if self._selected_fields is not None:
            self._selected_fields = tuple(rename.get(f, f) for f in self._selected_fields)

    def remove_fields(self, names: str | Sequence[str]) -> None:
        """Remove one or more fields from the full Vector schema."""
        self._require_full_field_view("remove_fields")
        to_remove = set(_normalize_field_names(names))
        old_fields = self._state["fields"]
        old_units = self._state["units"]

        missing = [field for field in to_remove if field not in old_fields]
        if missing:
            raise KeyError(f"Unknown field(s): {missing}")
        if len(to_remove) == len(old_fields):
            raise ValueError("Cannot remove all fields from a Vector.")

        keep = [i for i, field in enumerate(old_fields) if field not in to_remove]
        self._state["fields"] = [old_fields[i] for i in keep]
        self._state["units"] = [old_units[i] for i in keep]
        self._state["data"] = self._state["data"][:, keep]

        if self._selected_fields is not None:
            self._selected_fields = tuple(
                field for field in self._selected_fields if field in self._state["fields"]
            )
            if len(self._selected_fields) == len(self._state["fields"]):
                self._selected_fields = None

    # ------------------------------------------------------------------ #
    # Cell / row mutation
    # ------------------------------------------------------------------ #

    def append_rows(self, idx: Any, rows: Any) -> None:
        """Append one or more rows to a single selected cell.

        ``idx`` is interpreted with the same fixed-grid indexing rules as
        ``__getitem__`` and must resolve to exactly one cell. Appending rows is a
        full-cell operation, so all fields must be selected.
        """
        target = self[idx]
        if target.shape != ():
            raise ValueError("append_rows requires an index that selects exactly one cell.")
        target._require_full_field_view("append_rows")

        new_rows = _coerce_cell_array(rows, target.num_fields)
        if new_rows.shape[0] == 0:
            return

        cell_index = int(target._selected_cell_indices()[0])
        existing = target._cell_matrix(cell_index)
        combined = np.vstack((existing, new_rows)) if existing.shape[0] > 0 else new_rows.copy()
        target._replace_cells(np.array([cell_index], dtype=np.int64), [combined])

    def set_flattened(self, values: Any) -> None:
        """Write values back in flattened row-major order.

        This updates existing rows without changing per-cell row counts. It is
        the rowwise companion to ``flatten()`` and is especially useful for
        NumPy-based transforms that operate on all selected rows at once.
        """
        field_indices = self._field_indices()
        targets = self._selected_cell_indices()
        row_counts = self.row_counts()
        total_rows = sum(row_counts)

        if isinstance(values, Vector):
            if values.num_fields != self.num_fields:
                raise ValueError(f"Expected {self.num_fields} fields, got {values.num_fields}")
            flat_values = values.flatten()
            if flat_values.shape[0] != total_rows:
                raise ValueError(f"Expected {total_rows} rows, got {flat_values.shape[0]}")
        else:
            flat_values = _broadcast_field_values(values, total_rows, self.num_fields)

        cursor = 0
        for target, rows in zip(targets, row_counts):
            cell = self._cell_matrix(int(target))
            if rows > 0:
                cell[:, field_indices] = flat_values[cursor : cursor + rows]
            cursor += rows

    def compact(self) -> None:
        """Repack the backing row buffer to remove dead rows.

        Whole-cell replacement appends new rows and leaves previous rows unused
        until compaction. Calling ``compact()`` makes memory usage and save size
        predictable at the cost of reallocating the backing buffer.
        """
        data = self._state["data"]
        used_rows = int(self._state["cell_lengths"].sum())
        if used_rows == 0:
            self._state["data"] = np.empty((0, self._full_num_fields), dtype=data.dtype)
            self._state["cell_starts"].fill(0)
            return

        compacted = np.empty((used_rows, self._full_num_fields), dtype=data.dtype)
        starts = np.zeros_like(self._state["cell_starts"])
        cursor = 0
        for linear_index in range(_cell_count(self._state["shape"])):
            length = self._cell_row_count(linear_index)
            starts[linear_index] = cursor
            if length > 0:
                cell = self._cell_matrix(linear_index)
                compacted[cursor : cursor + length] = cell
                cursor += length
        self._state["data"] = compacted
        self._state["cell_starts"] = starts

    # ------------------------------------------------------------------ #
    # Python data model
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Return ``shape[0]`` for non-scalar selections."""
        if self.shape == ():
            raise TypeError("len() of unsized 0D Vector")
        return self.shape[0]

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"quantem.Vector, shape={self.shape}, name={self.name}",
                f"  fields = {self.fields}",
                f"  units: {self.units}",
            ]
        )

    __str__ = __repr__

    def copy(self) -> "Vector":
        """Return a deep copy of the current selection."""
        copied = self.__class__(
            shape=self.shape,
            fields=self.fields,
            units=self.units,
            name=self.name,
            metadata=copy.deepcopy(self.metadata),
            _token=self.__class__._token,
        )
        target_cells = copied._selected_cell_indices()
        source_arrays = [
            self._selected_cell_matrix(index).copy() for index in self._selected_cell_indices()
        ]
        copied._replace_cells(target_cells, source_arrays)
        return copied

    def __getitem__(self, idx: Any) -> "Vector":
        """Return a fixed-grid selection as another Vector view."""
        if _looks_like_field_selector(idx):
            raise TypeError("Use select_fields(...) for field selection.")
        if idx is Ellipsis:
            return self

        selection_shape, selection_indices = _select_linear_indices(
            self.shape,
            self._selected_cell_indices(),
            idx,
        )
        return self._from_view(
            self._state,
            selection_shape,
            selection_indices,
            self._selected_fields,
        )

    def __setitem__(self, idx: Any, value: Any) -> None:
        """Assign to a fixed-grid selection."""
        if idx is Ellipsis:
            target = self
        else:
            target = self[idx]
        target._assign(value)

    # ------------------------------------------------------------------ #
    # Arithmetic operators
    # ------------------------------------------------------------------ #

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """Apply supported NumPy ufuncs elementwise.

        Supported operations are limited to elementwise ``__call__`` ufuncs. The
        result preserves the current selection shape and fields.
        """
        if method != "__call__":
            return NotImplemented

        out = kwargs.get("out")
        if out is not None:
            return NotImplemented

        vector_inputs = [value for value in inputs if isinstance(value, Vector)]
        if not vector_inputs:
            return NotImplemented

        template = vector_inputs[0]
        row_counts = template.row_counts()
        total_rows = sum(row_counts)

        for other in vector_inputs[1:]:
            if other.shape != template.shape:
                raise ValueError("Vector ufunc inputs must have matching fixed-grid shapes.")
            if other.num_fields != template.num_fields:
                raise ValueError("Vector ufunc inputs must have matching field counts.")
            if other.row_counts() != row_counts:
                raise ValueError("Vector ufunc inputs must have matching per-cell row counts.")

        flat_inputs = [
            _normalize_ufunc_input(value, total_rows, template.num_fields) for value in inputs
        ]
        result = ufunc(*flat_inputs, **kwargs)
        if isinstance(result, tuple):
            return tuple(_vector_from_flat_result(template, item, row_counts) for item in result)
        return _vector_from_flat_result(template, result, row_counts)

    def __add__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.add)

    def __sub__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.subtract)

    def __mul__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.multiply)

    def __truediv__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.divide)

    def __floordiv__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.floor_divide)

    def __mod__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.mod)

    def __pow__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.power)

    def __radd__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.add, reverse=True)

    def __rmul__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.multiply, reverse=True)

    def __rsub__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.subtract, reverse=True)

    def __rtruediv__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.divide, reverse=True)

    def __rfloordiv__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.floor_divide, reverse=True)

    def __rmod__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.mod, reverse=True)

    def __rpow__(self, other: Any) -> "Vector":
        return self._binary_op(other, np.power, reverse=True)

    def __iadd__(self, other: Any) -> "Vector":
        self._inplace_op(other, np.add)
        return self

    def __isub__(self, other: Any) -> "Vector":
        self._inplace_op(other, np.subtract)
        return self

    def __imul__(self, other: Any) -> "Vector":
        self._inplace_op(other, np.multiply)
        return self

    def __itruediv__(self, other: Any) -> "Vector":
        self._inplace_op(other, np.divide)
        return self

    def __ifloordiv__(self, other: Any) -> "Vector":
        self._inplace_op(other, np.floor_divide)
        return self

    def __imod__(self, other: Any) -> "Vector":
        self._inplace_op(other, np.mod)
        return self

    def __ipow__(self, other: Any) -> "Vector":
        self._inplace_op(other, np.power)
        return self

    def __neg__(self) -> "Vector":
        return self._binary_op(-1, np.multiply)

    def __pos__(self) -> "Vector":
        return self.copy()

    def __abs__(self) -> "Vector":
        result = self.copy()
        result._inplace_unary(np.abs)
        return result

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #

    def save(
        self,
        path: str | Path,
        mode: Literal["w", "o"] = "w",
        store: Literal["auto", "zip", "dir"] = "auto",
        skip: str | type | Sequence[str | type] = (),
        compression_level: int | None = 4,
    ) -> None:
        """
        Save the Vector object to disk using Zarr serialization. self.compact() is called before
        saving to reduce file size if possible.

        Parameters
        ----------
        path : str or Path
            Target file path. Use '.zip' extension for zip format, otherwise a directory.
        mode : {'w', 'o'}
            'w' = write only if file doesn't exist, 'o' = overwrite if it does.
        store : {'auto', 'zip', 'dir'}
            Storage format. 'auto' infers from file extension.
        skip : str, type, or list of (str or type)
            Attribute names/types to skip (by name or type) during serialization.
        compression_level : int or None
            If set (0–9), applies Zstandard compression with Blosc backend at that level.
            Level 0 disables compression. Raises ValueError if > 9.

        Notes
        -----
        Skipped attribute names and types are also stored in the file metadata for correct
        round-trip skipping during load().
        """
        self.compact()
        super().save(
            path,
            mode=mode,
            store=store,
            skip=skip,
            compression_level=compression_level,
        )

    # ------------------------------------------------------------------ #
    # Private helpers — backing-store access
    # ------------------------------------------------------------------ #

    @property
    def _full_num_fields(self) -> int:
        return len(self._state["fields"])

    def _field_indices(self) -> NDArray[np.int64]:
        """Map selected field names to column indices in the backing buffer."""
        if self._selected_fields is None:
            return np.arange(self._full_num_fields, dtype=np.int64)

        lookup = {field: i for i, field in enumerate(self._state["fields"])}
        try:
            return np.array([lookup[field] for field in self._selected_fields], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"Unknown field(s): {[str(exc.args[0])]}") from exc

    def _require_full_field_view(self, operation: str) -> None:
        """Raise if a schema-changing/full-row operation is attempted on a field view."""
        if self._selected_fields is not None:
            raise ValueError(f"{operation} is only allowed when all fields are selected.")

    def _selected_cell_indices(self) -> NDArray[np.int64]:
        """Return linear cell indices for the current fixed-grid selection."""
        if self._selection_indices is None:
            return np.arange(_cell_count(self._state["shape"]), dtype=np.int64)
        return self._selection_indices

    def _cell_row_count(self, linear_index: int) -> int:
        """Return the row count for one cell in the backing buffer."""
        return int(self._state["cell_lengths"][linear_index])

    def _cell_matrix(self, linear_index: int) -> NDArray[Any]:
        """Return the full backing matrix for one cell."""
        start = int(self._state["cell_starts"][linear_index])
        length = int(self._state["cell_lengths"][linear_index])
        return self._state["data"][start : start + length]

    def _selected_cell_matrix(self, linear_index: int) -> NDArray[Any]:
        """Return one cell with the current field selection applied."""
        cell = self._cell_matrix(linear_index)
        cols = self._field_indices()
        if cols.size == self._full_num_fields:
            return cell
        if cols.size == 1:
            col = int(cols[0])
            return cell[:, col : col + 1]
        if _is_contiguous(cols):
            return cell[:, int(cols[0]) : int(cols[-1]) + 1]
        return cell[:, cols].copy()

    def _replace_cells(self, targets: NDArray[np.int64], arrays: Sequence[NDArray[Any]]) -> None:
        """Replace complete cells in the compact row buffer.

        Whole-cell replacement is implemented by appending the new payload rows to
        the end of the backing buffer and then updating ``cell_starts`` /
        ``cell_lengths`` for the targeted cells. This keeps the operation simple
        and makes overlapping assignment semantics easy to reason about, but it
        leaves the previous rows unreachable until compaction removes them.
        """
        if len(targets) != len(arrays):
            raise ValueError("Target cell count does not match source cell count.")
        if len(targets) == 0:
            return

        normalized = [_coerce_cell_array(array, self._full_num_fields) for array in arrays]
        payloads = [array for array in normalized if array.shape[0] > 0]
        if payloads:
            appended = np.vstack(payloads)
            self._state["data"] = np.concatenate((self._state["data"], appended), axis=0)

        cursor = self._state["data"].shape[0] - sum(array.shape[0] for array in normalized)
        for target, array in zip(targets, normalized):
            self._state["cell_starts"][target] = cursor
            self._state["cell_lengths"][target] = array.shape[0]
            cursor += array.shape[0]

        self._maybe_compact_storage()

    def _expand_storage(self, num_new_fields: int) -> None:
        """Append new ``np.nan``-initialized columns for added fields."""
        data = self._state["data"]
        dtype = np.result_type(data.dtype, float)
        if data.shape[0] == 0:
            self._state["data"] = np.empty((0, data.shape[1] + num_new_fields), dtype=dtype)
            return

        filler = np.full((data.shape[0], num_new_fields), np.nan, dtype=dtype)
        self._state["data"] = np.concatenate((data.astype(dtype, copy=False), filler), axis=1)

    def _maybe_compact_storage(self) -> None:
        """Compact automatically once dead rows become materially larger than live rows."""
        data = self._state["data"]
        used_rows = int(self._state["cell_lengths"].sum())
        if data.shape[0] <= used_rows + 1024 or data.shape[0] <= 2 * used_rows:
            return
        self.compact()

    # ------------------------------------------------------------------ #
    # Private helpers — assignment
    # ------------------------------------------------------------------ #

    def _assign(self, value: Any) -> None:
        """Dispatch assignment based on whether all fields or a subset are selected."""
        if self._selected_fields is None:
            self._assign_full_cells(value)
        else:
            self._assign_selected_fields(value)

    def _assign_full_cells(self, value: Any) -> None:
        """Replace full cell payloads.

        Full-cell assignment may change the ragged row count of each targeted
        cell, because the existing cell matrix is replaced as a whole.
        """
        targets = self._selected_cell_indices()
        if isinstance(value, Vector):
            source_cells = value._selected_cell_indices()
            if len(targets) != len(source_cells):
                raise ValueError(f"Expected {len(targets)} cells, got {len(source_cells)}")
            if value.num_fields != self.num_fields:
                raise ValueError(f"Expected {self.num_fields} fields, got {value.num_fields}")
            arrays = [value._selected_cell_matrix(index).copy() for index in source_cells]
            self._replace_cells(targets, arrays)
            return

        array = _coerce_cell_array(value, self.num_fields)
        self._replace_cells(targets, [array] * len(targets))

    def _assign_selected_fields(self, value: Any) -> None:
        """Update only the selected columns while preserving row counts.

        This is the in-place path for assignments such as
        ``vector.select_fields("kx")[...] = rhs``. The target cell structure is
        preserved, so each target cell keeps its existing row count and only the
        selected columns are overwritten.
        """
        targets = self._selected_cell_indices()
        field_indices = self._field_indices()
        row_counts = [self._cell_row_count(index) for index in targets]
        total_rows = sum(row_counts)

        if isinstance(value, Vector):
            source_cells = value._selected_cell_indices()
            if len(targets) != len(source_cells):
                raise ValueError(f"Expected {len(targets)} cells, got {len(source_cells)}")
            if value.num_fields != self.num_fields:
                raise ValueError(f"Expected {self.num_fields} fields, got {value.num_fields}")
            source_counts = [value._cell_row_count(index) for index in source_cells]
            if row_counts != source_counts:
                raise ValueError("Per-cell row counts must match for field-selected assignment.")
            snapshots = [value._selected_cell_matrix(index).copy() for index in source_cells]
            for target, array in zip(targets, snapshots):
                cell = self._cell_matrix(int(target))
                if array.shape[0] > 0:
                    cell[:, field_indices] = array
            return

        if np.isscalar(value):
            for target in targets:
                cell = self._cell_matrix(int(target))
                if cell.shape[0] > 0:
                    cell[:, field_indices] = value
            return

        broadcast = _broadcast_field_values(value, total_rows, self.num_fields)
        cursor = 0
        for target, rows in zip(targets, row_counts):
            chunk = broadcast[cursor : cursor + rows]
            cell = self._cell_matrix(int(target))
            if rows > 0:
                cell[:, field_indices] = chunk
            cursor += rows

    # ------------------------------------------------------------------ #
    # Private helpers — arithmetic
    # ------------------------------------------------------------------ #

    def _binary_op(self, other: Any, op: Any, reverse: bool = False) -> "Vector":
        """Return a new Vector produced by elementwise arithmetic."""
        result = self.copy()
        result._inplace_op(other, op, reverse=reverse)
        return result

    def _inplace_unary(self, op: Any) -> None:
        """Apply a unary elementwise operation in-place to the selected fields."""
        targets = self._selected_cell_indices()
        field_indices = self._field_indices()
        for target in targets:
            cell = self._cell_matrix(int(target))
            lhs = cell[:, field_indices]
            if lhs.shape[0] > 0:
                cell[:, field_indices] = op(lhs)

    def _inplace_op(self, other: Any, op: Any, reverse: bool = False) -> None:
        """Apply elementwise arithmetic in-place to the selected fields."""
        targets = self._selected_cell_indices()
        field_indices = self._field_indices()
        row_counts = [self._cell_row_count(index) for index in targets]
        total_rows = sum(row_counts)

        if isinstance(other, Vector):
            source_cells = other._selected_cell_indices()
            if len(targets) != len(source_cells):
                raise ValueError(f"Expected {len(targets)} cells, got {len(source_cells)}")
            if other.num_fields != self.num_fields:
                raise ValueError(f"Expected {self.num_fields} fields, got {other.num_fields}")
            source_counts = [other._cell_row_count(index) for index in source_cells]
            if row_counts != source_counts:
                raise ValueError("Per-cell row counts must match for Vector arithmetic.")
            snapshots = [other._selected_cell_matrix(index).copy() for index in source_cells]
            for target, rhs in zip(targets, snapshots):
                cell = self._cell_matrix(int(target))
                lhs = cell[:, field_indices]
                cell[:, field_indices] = op(rhs, lhs) if reverse else op(lhs, rhs)
            return

        if np.isscalar(other):
            for target in targets:
                cell = self._cell_matrix(int(target))
                lhs = cell[:, field_indices]
                if lhs.shape[0] > 0:
                    cell[:, field_indices] = op(other, lhs) if reverse else op(lhs, other)
            return

        broadcast = _broadcast_field_values(other, total_rows, self.num_fields)
        cursor = 0
        for target, rows in zip(targets, row_counts):
            chunk = broadcast[cursor : cursor + rows]
            cell = self._cell_matrix(int(target))
            lhs = cell[:, field_indices]
            if rows > 0:
                cell[:, field_indices] = op(chunk, lhs) if reverse else op(lhs, chunk)
            cursor += rows


def _resolve_fields(
    fields: Sequence[str] | None,
    num_fields: int | None,
    inferred: int | None,
) -> list[str]:
    """Resolve field names from constructor arguments.

    ``inferred`` is the field count inferred from data; pass ``None`` when there
    is no data source and explicit fields/num_fields are required.
    """
    if fields is not None:
        root_fields = validate_fields(list(fields))
        count = len(root_fields)
        if num_fields is not None and count != num_fields:
            raise ValueError(
                f"num_fields ({num_fields}) does not match length of fields ({count})"
            )
        if inferred is not None and count != inferred:
            raise ValueError(f"num_fields ({inferred}) does not match length of fields ({count})")
        return root_fields
    if num_fields is not None:
        count = validate_num_fields(num_fields)
        if inferred is not None and count != inferred:
            raise ValueError(
                f"Provided num_fields ({count}) does not match inferred ({inferred})."
            )
        return [f"field_{i}" for i in range(count)]
    if inferred is not None:
        return [f"field_{i}" for i in range(inferred)]
    raise ValueError("Must specify either 'fields' or 'num_fields'.")


def _cell_count(shape: tuple[int, ...]) -> int:
    """Return the number of fixed-grid cells in a shape."""
    return int(np.prod(shape, dtype=np.int64)) if shape else 1


def _normalize_field_names(field_names: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize one-or-many field names into a validated tuple."""
    if isinstance(field_names, str):
        normalized = (field_names,)
    else:
        normalized = tuple(field_names)
    if not normalized:
        raise ValueError("At least one field name is required.")
    validate_fields(list(normalized))
    return normalized


def _normalize_units(units: str | Sequence[str] | None, count: int) -> list[str]:
    """Normalize field units to a list matching ``count``."""
    if units is None:
        return ["none"] * count
    if isinstance(units, str):
        if count != 1:
            raise ValueError("A single unit can only be provided for a single field.")
        return [units]
    normalized = list(units)
    if len(normalized) != count:
        raise ValueError(f"Expected {count} units, got {len(normalized)}")
    return normalized


def _looks_like_field_selector(idx: Any) -> bool:
    """Return True for indices that look like field selection by mistake."""
    if isinstance(idx, str):
        return True
    if isinstance(idx, tuple) and any(_looks_like_field_selector(item) for item in idx):
        return True
    if isinstance(idx, list) and idx and all(isinstance(item, str) for item in idx):
        return True
    return False


def _coerce_cell_array(value: Any, num_fields: int) -> NDArray[Any]:
    """Normalize a single-cell payload to shape ``(n_rows, num_fields)``."""
    if isinstance(value, Vector):
        if value.shape != ():
            raise ValueError("Expected a 0D Vector for single-cell assignment.")
        array = value.array.copy()
    else:
        array = np.asarray(value)

    if array.ndim == 0:
        raise ValueError("Cell assignment requires a 2D array.")
    if array.ndim == 1:
        if array.size == 0:
            array = np.empty((0, num_fields), dtype=float)
        elif num_fields == 1:
            array = array.reshape(-1, 1)
        else:
            array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("Cell assignment requires a 2D array.")
    if array.shape[1] != num_fields:
        raise ValueError(f"Expected {num_fields} fields, got {array.shape[1]}")
    return array


def _flatten_fixed_grid(node: Any) -> tuple[tuple[int, ...], list[NDArray[Any]]]:
    """Recursively flatten nested fixed-grid input into row-major cell order."""
    if isinstance(node, np.ndarray):
        return (), [_coerce_inferred_cell_array(node)]
    if not isinstance(node, (list, tuple)):
        raise TypeError("Data must be a nested list/tuple of cell arrays or row sequences.")
    if _looks_like_cell_rows(node):
        return (), [_coerce_inferred_cell_array(node)]
    if len(node) == 0:
        return (0,), []

    child_shape: tuple[int, ...] | None = None
    cells: list[NDArray[Any]] = []
    for child in node:
        shape, child_cells = _flatten_fixed_grid(child)
        if child_shape is None:
            child_shape = shape
        elif child_shape != shape:
            raise ValueError("All nested fixed-grid branches must have matching shapes.")
        cells.extend(child_cells)

    assert child_shape is not None
    return (len(node),) + child_shape, cells


def _looks_like_cell_rows(node: Sequence[Any]) -> bool:
    """Return True when a sequence should be interpreted as cell rows, not grid nesting."""
    if len(node) == 0:
        return True
    return all(_is_row_like(item) for item in node)


def _is_row_like(item: Any) -> bool:
    """Return True for a single row of scalar values."""
    if isinstance(item, np.ndarray):
        return item.ndim == 1
    if not isinstance(item, (list, tuple)):
        return False
    return all(np.isscalar(value) for value in item)


def _coerce_inferred_cell_array(value: Any) -> NDArray[Any]:
    """Infer a 2D cell array from row-like input during ``from_data``."""
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError("Cell data must be 1D or 2D.")
    if array.ndim == 1:
        if array.size == 0:
            return np.empty((0, 0), dtype=float)
        return array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("Cell data must be 1D or 2D.")
    return array


def _select_linear_indices(
    shape: tuple[int, ...],
    current_indices: NDArray[np.int64],
    idx: Any,
) -> tuple[tuple[int, ...], NDArray[np.int64]]:
    """Apply fixed-grid indexing to a flattened cell-index view.

    ``current_indices`` stores the linear cell indices represented by the current
    selection. This helper reshapes those indices to the current selection shape,
    applies NumPy-like indexing on the fixed-grid axes, and then returns:
    - the output fixed-grid shape
    - the flattened linear indices of the selected cells, in row-major order
    """
    if shape == ():
        if idx in ((), Ellipsis):
            return (), np.array([int(current_indices[0])], dtype=np.int64)
        raise IndexError("Too many indices for 0D Vector")

    index_tuple = _normalize_index_tuple(idx, len(shape))
    current_grid = current_indices.reshape(shape)

    axis_positions: list[NDArray[np.int64]] = []
    out_shape: list[int] = []
    scalar_axes: list[bool] = []
    for axis, axis_index in enumerate(index_tuple):
        positions, is_scalar = _positions_for_axis(axis_index, shape[axis])
        axis_positions.append(positions)
        scalar_axes.append(is_scalar)
        if not is_scalar:
            out_shape.append(len(positions))

    if all(scalar_axes):
        scalar_key = tuple(int(positions[0]) for positions in axis_positions)
        value = int(current_grid[scalar_key])
        return (), np.array([value], dtype=np.int64)

    mesh_inputs = [
        positions if not is_scalar else positions[:1]
        for positions, is_scalar in zip(axis_positions, scalar_axes)
    ]
    grids = np.meshgrid(*mesh_inputs, indexing="ij")
    selected = np.asarray(current_grid[tuple(grids)], dtype=np.int64).reshape(-1)
    return tuple(out_shape), selected


def _normalize_index_tuple(idx: Any, ndim: int) -> tuple[Any, ...]:
    """Normalize fixed-grid indexing to a full ``ndim``-length tuple."""
    if idx is Ellipsis:
        return (slice(None),) * ndim
    if not isinstance(idx, tuple):
        idx = (idx,)

    ellipsis_count = sum(item is Ellipsis for item in idx)
    if ellipsis_count > 1:
        raise IndexError("An index can only have a single ellipsis.")
    if ellipsis_count == 1:
        ellipsis_pos = idx.index(Ellipsis)
        fill = ndim - (len(idx) - 1)
        idx = idx[:ellipsis_pos] + (slice(None),) * fill + idx[ellipsis_pos + 1 :]
    if len(idx) > ndim:
        raise IndexError(f"Too many indices for Vector: expected {ndim}, got {len(idx)}")
    if len(idx) < ndim:
        idx = idx + (slice(None),) * (ndim - len(idx))
    return idx


def _positions_for_axis(axis_index: Any, size: int) -> tuple[NDArray[np.int64], bool]:
    """Resolve one axis index into concrete positions and scalar-vs-vector shape behavior."""
    if isinstance(axis_index, (bool, np.bool_)):
        raise TypeError("Boolean scalars are not valid Vector indices.")

    if isinstance(axis_index, (int, np.integer)):
        index = int(axis_index)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError("Vector index out of range")
        return np.array([index], dtype=np.int64), True

    if isinstance(axis_index, slice):
        return np.arange(size, dtype=np.int64)[axis_index], False

    array = np.asarray(axis_index)
    if array.ndim == 0:
        if np.issubdtype(array.dtype, np.integer):
            return _positions_for_axis(int(array.item()), size)
        raise TypeError(f"Unsupported index type: {type(axis_index)!r}")

    if array.dtype == bool or np.issubdtype(array.dtype, np.bool_):
        if array.ndim != 1:
            raise IndexError("Full-grid boolean masks are not supported.")
        if array.shape[0] != size:
            raise IndexError(
                f"Boolean mask length {array.shape[0]} does not match axis length {size}"
            )
        return np.flatnonzero(array).astype(np.int64, copy=False), False

    if array.ndim != 1:
        raise IndexError("Fancy indexing arrays must be one-dimensional.")
    if array.size == 0:
        return np.array([], dtype=np.int64), False
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("Fancy indices must be integers or booleans.")

    positions = array.astype(np.int64, copy=True)
    positions[positions < 0] += size
    if np.any((positions < 0) | (positions >= size)):
        raise IndexError("Vector index out of range")
    return positions, False


def _broadcast_field_values(value: Any, total_rows: int, num_fields: int) -> NDArray[Any]:
    """Broadcast array-like input to flattened rowwise assignment shape."""
    array = np.asarray(value)
    if array.ndim == 0:
        return np.broadcast_to(array.reshape(1, 1), (total_rows, num_fields))
    if num_fields == 1 and array.ndim == 1:
        if total_rows == 0 and array.shape[0] == 0:
            return array.reshape(0, 1)
        if array.shape[0] != total_rows:
            raise ValueError(f"Expected {total_rows} values, got {array.shape[0]}")
        return array.reshape(total_rows, 1)
    try:
        return np.broadcast_to(array, (total_rows, num_fields))
    except ValueError as exc:
        raise ValueError(
            f"Cannot broadcast value with shape {array.shape} to ({total_rows}, {num_fields})"
        ) from exc


def _normalize_ufunc_input(value: Any, total_rows: int, num_fields: int) -> Any:
    """Normalize one ufunc input to flattened Vector-compatible form."""
    if isinstance(value, Vector):
        return value.flatten()
    if np.isscalar(value):
        return value
    return _broadcast_field_values(value, total_rows, num_fields)


def _vector_from_flat_result(
    template: Vector,
    values: Any,
    row_counts: list[int],
) -> Vector:
    """Build a Vector from flattened rowwise result data."""
    total_rows = sum(row_counts)
    flat_values = _broadcast_field_values(values, total_rows, template.num_fields)

    result = Vector.from_shape(
        shape=template.shape,
        fields=template.fields,
        units=template.units,
        name=template.name,
    )
    result._state["metadata"] = copy.deepcopy(template.metadata)

    if total_rows == 0:
        result._state["data"] = np.empty((0, template.num_fields), dtype=flat_values.dtype)
        return result

    cursor = 0
    cells: list[NDArray[Any]] = []
    for rows in row_counts:
        cells.append(flat_values[cursor : cursor + rows].copy())
        cursor += rows

    result._replace_cells(result._selected_cell_indices(), cells)
    return result


def _is_contiguous(indices: NDArray[np.int64]) -> bool:
    """Return True when integer column indices form one contiguous slice."""
    if indices.size <= 1:
        return True
    return bool(np.all(indices[1:] - indices[:-1] == 1))
