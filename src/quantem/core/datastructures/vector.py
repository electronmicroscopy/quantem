from __future__ import annotations

import copy
import functools
import math
import numbers
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core import config
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import (
    validate_fields,
    validate_num_fields,
    validate_shape,
    validate_vector_units,
)

if TYPE_CHECKING:
    import polars as pl

DEFAULT_DTYPE = torch.float32

# Only functions whose output preserves the meaning of each individual row may
# be rebuilt as a Vector. Other torch functions still receive flattened tensor
# inputs, but their results stay as ordinary tensors.
_SAFE_ELEMENTWISE_TORCH_FUNCTIONS = frozenset(
    getattr(torch, name)
    for name in """
        abs absolute acos acosh add asin asinh atan atan2 atanh
        ceil clamp clip cos cosh divide erf erfc exp floor frexp
        log log10 log1p log2 maximum minimum multiply neg pow
        remainder round rsqrt sigmoid sin sinh sqrt square subtract
        tan tanh trunc
    """.split()
    if hasattr(torch, name)
)

# Tensor methods dispatched by operators such as ``tensor + vector``. Treating
# them as elementwise makes ``tensor + vector`` return a Vector, matching
# ``vector + tensor``.
_SAFE_ELEMENTWISE_TORCH_FUNCTIONS |= frozenset(
    getattr(torch.Tensor, name)
    for name in """
        add sub mul div remainder __pow__ __floordiv__
        __radd__ __rsub__ __rmul__ __rtruediv__ __rfloordiv__ __rmod__ __rpow__
    """.split()
    if hasattr(torch.Tensor, name)
)


class Vector(AutoSerialize):
    """Ragged cell data on a fixed grid, backed by torch.

    A ``Vector`` stores a variable number of rows in each cell of a fixed grid
    with dimensions ``shape``, such as the Bragg peaks measured at each probe
    position of a 4D-STEM scan. Each row has one value per named field, so each
    cell is a 2D tensor with shape ``(n_rows, num_fields)``, where ``n_rows``
    can vary from cell to cell.

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
    dtype : torch.dtype, optional
        Row-buffer dtype. Defaults to ``torch.float32``.
    device : str or torch.device, optional
        Device for the row buffer. Defaults to ``"cpu"``.

    Notes
    -----
    ``tensor``, ``flatten()`` and all arithmetic results are ``torch.Tensor``
    values on ``device``. NumPy arrays, Python sequences and scalars are
    accepted as inputs and converted to tensors, and ``numpy()`` returns a
    NumPy copy of the flattened rows.

    Fixed-grid indexing uses ``[]`` and always returns a ``Vector``, while
    field selection uses ``select_fields(...)``. A 0D selection exposes its
    cell through ``.tensor``, and ``flatten()`` concatenates the rows of a
    multi-cell selection.

    All rows are stored in a single 2D tensor ``_state["data"]``, with the
    start offset and row count of each cell in ``_state["cell_starts"]`` and
    ``_state["cell_lengths"]``. These offsets stay on the CPU when the row
    buffer is on a GPU, because they are read one scalar at a time and a GPU
    copy would synchronize on every cell access.

    Selections are write-through views of the shared storage, and store only
    their fixed-grid shape, cell indices and field names. Because the storage
    is shared, ``to(device)`` moves every view of the same Vector.

    Examples
    --------
    Create a Vector and assign one cell:

    >>> import torch
    >>> v = Vector.from_shape((2, 2), fields=("kx", "ky", "intensity"))
    >>> v[0, 0] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> v[0, 0].tensor.shape
    torch.Size([2, 3])

    Select fields and apply in-place arithmetic:

    >>> kx = v.select_fields("kx")
    >>> kx += 16
    >>> kx.flatten().shape
    torch.Size([2, 1])

    Apply a rowwise transform with ``flatten()`` and ``set_flattened()``:

    >>> kx = v.select_fields("kx")
    >>> ky = v.select_fields("ky")
    >>> kx.set_flattened(
    ...     torch.where(
    ...         ((kx.flatten() - 16) ** 2 + (ky.flatten() - 16) ** 2) < 12,
    ...         10.0,
    ...         kx.flatten(),
    ...     )
    ... )
    """

    # Opt out of NumPy's ufunc machinery entirely. This makes ``np.sin(vector)``
    # raise a clear TypeError instead of building an object array, and makes
    # ``ndarray + vector`` defer to ``Vector.__radd__``.
    __array_ufunc__ = None
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
        dtype: torch.dtype | None = None,
        device: str | int | torch.device | None = None,
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
        root_dtype = DEFAULT_DTYPE if dtype is None else dtype
        root_device = _resolve_device(device)

        self._state = {
            "shape": root_shape,
            "fields": list(root_fields),
            "units": list(root_units),
            "name": name or f"{len(root_shape)}d ragged array",
            "metadata": dict(metadata or {}),
            "data": torch.empty((0, len(root_fields)), dtype=root_dtype, device=root_device),
            "cell_starts": torch.zeros(_cell_count(root_shape), dtype=torch.int64),
            "cell_lengths": torch.zeros(_cell_count(root_shape), dtype=torch.int64),
        }
        self._selection_shape = root_shape
        self._selection_indices: torch.Tensor | None = None
        self._selected_fields: tuple[str, ...] | None = None

    @classmethod
    def _from_view(
        cls,
        state: dict[str, Any],
        selection_shape: tuple[int, ...],
        selection_indices: torch.Tensor | None,
        selected_fields: tuple[str, ...] | None,
    ) -> "Vector":
        """Build a view that shares backing storage with another Vector."""
        obj = cls.__new__(cls)
        obj._state = state
        obj._selection_indices = (
            None if selection_indices is None else selection_indices.to(torch.int64)
        )
        obj._selection_shape = selection_shape
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
        dtype: torch.dtype | None = None,
        device: str | int | torch.device | None = None,
    ) -> "Vector":
        """Create a Vector with zero rows in every cell.

        Parameters
        ----------
        shape : tuple of int
            Fixed-grid shape. Zero-length axes are allowed.
        num_fields : int, optional
            Number of fields, named ``field_0``, ``field_1``, ... when ``fields``
            is not given.
        fields : sequence of str, optional
            Field names in column order. One of ``fields`` or ``num_fields`` is
            required.
        units : sequence of str, optional
            Units of each field. Defaults to ``"none"``.
        name : str, optional
            Descriptive name.
        metadata : dict, optional
            Additional user metadata.
        dtype : torch.dtype, optional
            Row-buffer dtype. Defaults to ``torch.float32``.
        device : str, int or torch.device, optional
            Row-buffer device. Defaults to ``"cpu"``.

        Returns
        -------
        Vector
        """
        fields = _resolve_fields(fields, num_fields, None)
        return cls(
            shape=shape,
            fields=fields,
            units=units,
            name=name,
            metadata=metadata,
            dtype=dtype,
            device=device,
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
        dtype: torch.dtype | None = None,
        device: str | int | torch.device | None = None,
    ) -> "Vector":
        """Create a Vector from nested fixed-grid data.

        The nesting depth of ``data`` sets the fixed-grid shape, and each leaf is
        one cell. Leaves can be tensors, NumPy arrays or nested sequences with
        shape ``(n_rows, num_fields)``, and an empty list ``[]`` gives a cell
        with zero rows. All leaves are cast to ``dtype``.

        Parameters
        ----------
        data : list or tuple
            Nested fixed-grid data, e.g. ``data[i][j]`` is the cell at ``(i, j)``.
        num_fields : int, optional
            Number of fields, checked against the data when given.
        fields : sequence of str, optional
            Field names in column order. Required when every cell is empty.
        units : sequence of str, optional
            Units of each field. Defaults to ``"none"``.
        name : str, optional
            Descriptive name.
        metadata : dict, optional
            Additional user metadata.
        dtype : torch.dtype, optional
            Row-buffer dtype. Defaults to ``torch.float32``, so pass
            ``torch.float64`` to keep double-precision input.
        device : str, int or torch.device, optional
            Row-buffer device. Defaults to ``"cpu"``.

        Returns
        -------
        Vector
        """
        if not isinstance(data, (list, tuple)):
            raise TypeError(f"Data must be a list or tuple, got {type(data)}")
        root_shape, cell_arrays = _flatten_fixed_grid(data) if len(data) > 0 else ((0,), [])
        # An empty list ([]) carries no field count, so it is skipped here and
        # padded to (0, num_fields) on assignment.
        inferred_counts = {array.shape[1] for array in cell_arrays if array.shape != (0, 0)}
        if len(inferred_counts) > 1:
            raise ValueError("All cell arrays must have the same number of fields.")
        if inferred_counts:
            inferred_fields: int | None = inferred_counts.pop()
        elif fields is None and num_fields is None:
            inferred_fields = 0
        else:
            inferred_fields = None

        vector = cls(
            shape=root_shape,
            fields=_resolve_fields(fields, num_fields, inferred_fields),
            units=units,
            name=name,
            metadata=metadata,
            dtype=dtype,
            device=device,
            _token=cls._token,
        )
        num_fields = vector._full_num_fields
        lengths = torch.tensor([array.shape[0] for array in cell_arrays], dtype=torch.int64)
        rows = [array.reshape(-1, num_fields) for array in cell_arrays if array.shape[0] > 0]
        if rows and all(isinstance(array, np.ndarray) for array in rows):
            vector._state["data"] = vector._to_buffer(_as_tensor(np.concatenate(rows)))
        elif rows:
            # Cast each cell first, so tensors on different devices can be joined.
            vector._state["data"] = torch.cat(
                [vector._to_buffer(_as_tensor(array)) for array in rows], dim=0
            )
        vector._state["cell_lengths"] = lengths
        vector._state["cell_starts"] = torch.cumsum(lengths, 0) - lengths
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
        if self._selection_indices is None:
            return _cell_count(self._state["shape"])
        return int(self._selection_indices.numel())

    @property
    def total_rows(self) -> int:
        """Return the total ragged-row count in the current selection."""
        return int(self._selected_cell_lengths().sum())

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the backing row buffer."""
        return self._state["data"].dtype

    @property
    def device(self) -> str:
        """Return the device string of the backing row buffer."""
        return str(self._state["data"].device)

    # ------------------------------------------------------------------ #
    # Data access
    # ------------------------------------------------------------------ #

    @property
    def tensor(self) -> torch.Tensor:
        """Rows of a single selected cell, with shape ``(n_rows, num_fields)``.

        This property is only defined for 0D selections such as ``v[i, j]``,
        which differs from ``Dataset.tensor`` (the full array). Use
        :meth:`flatten` for the rows of a multi-cell selection.

        When the selected fields are contiguous and in storage order, the
        result is a writable view of the backing storage. Reordered or
        non-contiguous field selections return a copy, so writes to it do not
        reach the Vector; use ``v[i, j] = ...`` or :meth:`set_flattened` instead.
        """
        if self.shape != ():
            raise ValueError(".tensor is only valid when the selection contains exactly one cell.")
        return self._selected_cell_matrix(int(self._selected_cell_indices()[0]))

    def flatten(self) -> torch.Tensor:
        """Concatenate the rows of all selected cells in row-major cell order.

        Returns
        -------
        torch.Tensor
            Copy of the selected rows with shape ``(total_rows, num_fields)``,
            which stays 2D for single-field selections. Use :meth:`set_flattened`
            to write modified rows back.
        """
        data = self._state["data"]
        gather = self._row_gather_index(self._selected_cell_indices())
        if gather.numel() == 0:
            return torch.empty((0, self.num_fields), dtype=data.dtype, device=data.device)
        return _select_columns(data.index_select(0, gather.to(data.device)), self._field_indices())

    def numpy(self) -> NDArray[Any]:
        """Return :meth:`flatten` as a read-only NumPy array on the CPU.

        The result covers every selected cell, unlike :attr:`tensor`. As for
        ``Dataset.numpy()``, the array is read-only so that in-place edits raise
        an error rather than modifying a copy. Use ``numpy().copy()`` for a
        writable array, and :meth:`set_flattened` to write values back.
        """
        array = self.flatten().detach().cpu().numpy()
        array.flags.writeable = False
        return array

    def row_counts(self) -> list[int]:
        """Return per-cell row counts in the current selection order."""
        return self._selected_cell_lengths().tolist()

    def to(self, device: str | int | torch.device) -> "Vector":
        """Move the row buffer to ``device`` in place and return ``self``.

        ``device`` is normalized with :func:`quantem.core.config.validate_device`,
        so ``"cuda"``, ``0``, ``"cuda:0"`` and ``torch.device("cuda:0")`` give the
        same device. All views share the same storage, so this moves every view
        of the Vector. The cell offsets stay on the CPU.
        """
        self._state["data"] = self._state["data"].to(_resolve_device(device))
        return self

    # ------------------------------------------------------------------ #
    # Field management
    # ------------------------------------------------------------------ #

    def select_fields(self, *field_names: str | Sequence[str]) -> "Vector":
        """Return a write-through view of the requested fields, in the requested order.

        Fields can be given as separate arguments, ``select_fields("kx", "ky")``,
        or as one sequence, ``select_fields(["kx", "ky"])``.
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
        """Add fields to the Vector in place.

        New columns are filled with NaN, so an integer row buffer is promoted to
        ``torch.float32``. This method requires a view with all fields selected.

        Parameters
        ----------
        names : str or sequence of str
            Names of the new fields.
        values : Vector, tensor, array, scalar or sequence, optional
            Initial values, broadcast to ``(total_rows, len(names))``. When
            several fields are added, a sequence with one entry per field sets
            each field separately.
        units : str or sequence of str, optional
            Units of the new fields. Defaults to ``"none"``.
        """
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

        Notes
        -----
        Views created before the rename keep the old field names and raise
        ``KeyError``; create new views with ``select_fields`` after renaming.
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
        """Remove fields from the Vector in place.

        This method requires a view with all fields selected, and at least one
        field must remain.
        """
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
        """Append rows to one cell.

        Parameters
        ----------
        idx : int, tuple or index
            Fixed-grid index, using the same rules as ``v[idx]``, which must
            select exactly one cell.
        rows : tensor, array or sequence
            New rows with shape ``(n_rows, num_fields)``, or ``(num_fields,)``
            for a single row. All fields must be selected.
        """
        target = self[idx]
        if target.shape != ():
            raise ValueError("append_rows requires an index that selects exactly one cell.")
        target._require_full_field_view("append_rows")

        new_rows = target._coerce_cell(rows, target.num_fields)
        if new_rows.shape[0] == 0:
            return

        cell_index = int(target._selected_cell_indices()[0])
        combined = torch.cat((target._cell_matrix(cell_index), new_rows), dim=0)
        target._replace_cells(torch.tensor([cell_index], dtype=torch.int64), [combined])

    def set_flattened(self, values: Any) -> None:
        """Overwrite the selected rows and fields, in the order given by :meth:`flatten`.

        The per-cell row counts do not change. The typical use is a transform
        applied to all rows at once, ``v.set_flattened(f(v.flatten()))``.

        Parameters
        ----------
        values : Vector, tensor, array or scalar
            New values, broadcast to shape ``(total_rows, num_fields)``.
        """
        self._require_unique_cell_targets("set_flattened")
        total_rows = self.total_rows

        if isinstance(values, Vector):
            if values.num_fields != self.num_fields:
                raise ValueError(f"Expected {self.num_fields} fields, got {values.num_fields}")
            flat_values = values.flatten()
            if flat_values.shape[0] != total_rows:
                raise ValueError(f"Expected {total_rows} rows, got {flat_values.shape[0]}")
            flat_values = self._to_buffer(flat_values)
        else:
            flat_values = self._broadcast_values(values, total_rows, self.num_fields)

        self._write_selected_rows(flat_values)

    def compact(self) -> None:
        """Repack the backing row buffer to remove dead rows.

        Whole-cell replacement appends new rows and leaves previous rows unused
        until compaction. Calling ``compact()`` makes memory usage and save size
        predictable at the cost of reallocating the backing buffer.
        """
        data = self._state["data"]
        lengths = self._state["cell_lengths"]
        used_rows = int(lengths.sum())
        if data.shape[0] == used_rows:
            return  # already dense, nothing to reclaim

        all_cells = torch.arange(_cell_count(self._state["shape"]), dtype=torch.int64)
        gather = self._row_gather_index(all_cells)
        if gather.numel() == 0:
            self._state["data"] = torch.empty(
                (0, self._full_num_fields), dtype=data.dtype, device=data.device
            )
        else:
            self._state["data"] = data.index_select(0, gather.to(data.device))
        self._state["cell_starts"] = torch.cumsum(lengths, 0) - lengths

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
                f"  dtype: {self.dtype}, device: {self.device}",
            ]
        )

    __str__ = __repr__

    def copy(self) -> "Vector":
        """Return a deep copy of the current selection."""
        return _vector_from_rows(self, self.flatten(), self.row_counts())

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

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Apply torch functions to the flattened rows of Vector arguments.

        Each ``Vector`` argument is replaced by ``flatten()`` before calling
        ``func``. The result is returned as a ``Vector`` with the same shape and
        fields when ``func`` is in :data:`_SAFE_ELEMENTWISE_TORCH_FUNCTIONS` and
        the result has shape ``(total_rows, num_fields)``. All other results,
        including reductions (``torch.sum``), predicates (``torch.allclose``) and
        shape-changing functions (``torch.t``), are returned as plain tensors.

        The allowlist is required because the shape test alone is ambiguous:
        ``torch.t`` applied to a Vector with equal row and field counts returns a
        tensor of the same shape, with the rows permuted.
        """
        kwargs = {} if kwargs is None else kwargs
        if kwargs.get("out") is not None:
            return NotImplemented

        all_args = list(args) + list(kwargs.values())
        vector_inputs = [value for value in all_args if isinstance(value, Vector)]
        if not vector_inputs:
            # Functions like torch.cat/torch.stack take a *sequence* of tensors,
            # so the Vectors never appear as arguments in their own right. Say so,
            # rather than letting torch report an opaque dispatch failure.
            if any(
                isinstance(item, Vector)
                for value in all_args
                if isinstance(value, (list, tuple))
                for item in value
            ):
                raise TypeError(
                    f"{func.__name__} takes a sequence of tensors, which cannot hold Vectors: "
                    "ragged rows have no single shape to combine along. Pass flatten() "
                    "results instead, e.g. torch.cat([a.flatten(), b.flatten()])."
                )
            return NotImplemented

        template = vector_inputs[0]
        row_counts = template.row_counts()

        for other in vector_inputs[1:]:
            if other.shape != template.shape:
                raise ValueError("Vector inputs must have matching fixed-grid shapes.")
            if other.num_fields != template.num_fields:
                raise ValueError("Vector inputs must have matching field counts.")
            if other.row_counts() != row_counts:
                raise ValueError("Vector inputs must have matching per-cell row counts.")

        # For elementwise functions on single-field Vectors, a 1D tensor is one
        # value per row, as in v + x. Other functions (e.g. index_select) keep
        # their own meaning for 1D arguments.
        per_row = (
            sum(row_counts)
            if template.num_fields == 1 and func in _SAFE_ELEMENTWISE_TORCH_FUNCTIONS
            else None
        )
        flat_args = tuple(_flatten_torch_input(value, per_row) for value in args)
        flat_kwargs = {key: _flatten_torch_input(value, per_row) for key, value in kwargs.items()}
        result = func(*flat_args, **flat_kwargs)

        if func not in _SAFE_ELEMENTWISE_TORCH_FUNCTIONS:
            return result
        if isinstance(result, tuple):
            return tuple(_maybe_wrap_result(template, item, row_counts) for item in result)
        return _maybe_wrap_result(template, result, row_counts)

    def __add__(self, other: Any) -> "Vector":
        return self._binary_op(other, torch.add)

    def __sub__(self, other: Any) -> "Vector":
        return self._binary_op(other, torch.subtract)

    def __mul__(self, other: Any) -> "Vector":
        return self._binary_op(other, torch.multiply)

    def __truediv__(self, other: Any) -> "Vector":
        return self._binary_op(other, torch.divide)

    def __floordiv__(self, other: Any) -> "Vector":
        return self._binary_op(other, torch.floor_divide)

    def __mod__(self, other: Any) -> "Vector":
        return self._binary_op(other, torch.remainder)

    def __pow__(self, other: Any) -> "Vector":
        return self._binary_op(other, torch.pow)

    def __radd__(self, other: Any) -> "Vector":
        return self._binary_op(other, torch.add, reverse=True)

    def __rmul__(self, other: Any) -> "Vector":
        return self._binary_op(other, torch.multiply, reverse=True)

    def __rsub__(self, other: Any) -> "Vector":
        return self._binary_op(other, torch.subtract, reverse=True)

    def __rtruediv__(self, other: Any) -> "Vector":
        return self._binary_op(other, torch.divide, reverse=True)

    def __rfloordiv__(self, other: Any) -> "Vector":
        return self._binary_op(other, torch.floor_divide, reverse=True)

    def __rmod__(self, other: Any) -> "Vector":
        return self._binary_op(other, torch.remainder, reverse=True)

    def __rpow__(self, other: Any) -> "Vector":
        return self._binary_op(other, torch.pow, reverse=True)

    def __iadd__(self, other: Any) -> "Vector":
        self._inplace_op(other, torch.add)
        return self

    def __isub__(self, other: Any) -> "Vector":
        self._inplace_op(other, torch.subtract)
        return self

    def __imul__(self, other: Any) -> "Vector":
        self._inplace_op(other, torch.multiply)
        return self

    def __itruediv__(self, other: Any) -> "Vector":
        self._inplace_op(other, torch.divide)
        return self

    def __ifloordiv__(self, other: Any) -> "Vector":
        self._inplace_op(other, torch.floor_divide)
        return self

    def __imod__(self, other: Any) -> "Vector":
        self._inplace_op(other, torch.remainder)
        return self

    def __ipow__(self, other: Any) -> "Vector":
        self._inplace_op(other, torch.pow)
        return self

    def __neg__(self) -> "Vector":
        return self._binary_op(-1, torch.multiply)

    def __pos__(self) -> "Vector":
        return self.copy()

    def __abs__(self) -> "Vector":
        result = self.copy()
        result._inplace_unary(torch.abs)
        return result

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #

    def to_polars(self, dim_names: Sequence[str] | None = None) -> "pl.DataFrame":
        """Export the current selection to a polars DataFrame.

        Each row of the Vector becomes one DataFrame row. The leading integer
        columns give the fixed-grid index of the row's cell, one column per grid
        dimension, followed by one column per selected field. Grid indices refer
        to the full Vector rather than the selection, so a row from ``v[1, :]``
        has ``dim_0 == 1``, and ``v[1].select_fields("kx").to_polars()`` returns
        only the ``kx`` values of the cells in ``v[1]``.

        Requires the optional dependency polars, installed with
        ``pip install "quantem[dataframe]"``.

        Parameters
        ----------
        dim_names : sequence of str, optional
            Names for the fixed-grid index columns. Defaults to
            ``("dim_0", ..., "dim_{ndim-1}")``. Must have one entry per
            fixed-grid dimension.

        Returns
        -------
        polars.DataFrame
            Shape ``(total_rows, grid_ndim + num_fields)``.

        Examples
        --------
        >>> v = Vector.from_shape((3, 2), fields=("kx", "ky"))
        >>> v.to_polars().columns
        ['dim_0', 'dim_1', 'kx', 'ky']
        """
        try:
            import polars as pl
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Vector.to_polars() requires polars, which is an optional dependency. "
                'Install it with: pip install polars   (or: pip install "quantem[dataframe]")'
            ) from exc

        root_shape = self._state["shape"]
        index_names = _resolve_dim_names(dim_names, len(root_shape))

        collisions = [name for name in index_names if name in self.fields]
        if collisions:
            raise ValueError(
                f"Fixed-grid column name(s) {collisions} collide with field name(s). "
                "Pass dim_names=... to to_polars() to rename the index columns."
            )

        columns: dict[str, Any] = {}
        if index_names:
            counts = np.asarray(self.row_counts(), dtype=np.int64)
            cells = np.asarray(self._selected_cell_indices().tolist(), dtype=np.int64)
            coords = np.unravel_index(cells, root_shape)
            for name, axis_coords in zip(index_names, coords):
                columns[name] = pl.Series(
                    name, np.repeat(axis_coords, counts).astype(np.int64, copy=False)
                )

        values = self.numpy()
        for column, field in enumerate(self.fields):
            columns[field] = pl.Series(field, values[:, column])

        return pl.DataFrame(columns)

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
            If set (0-9), applies Zstandard compression with Blosc backend at that level.
            Level 0 disables compression. Raises ValueError if > 9.

        Notes
        -----
        Skipped attribute names and types are also stored in the file metadata for correct
        round-trip skipping during load().

        The row buffer and cell offsets are written as compressed Zarr arrays,
        and are loaded as CPU tensors, so a Vector saved from a GPU can be loaded
        without CUDA. A Vector saved as an attribute of another object is
        written by that object's ``save()``, and its buffers are stored
        uncompressed.
        """
        # Nested Vectors go through AutoSerialize._recursive_save, not this
        # method. Storing plain non-grad tensors as Zarr arrays in
        # AutoSerialize._serialize_value would fix that, and reduce this
        # override to compact().
        self.compact()
        buffer_keys = ("data", "cell_starts", "cell_lengths")
        saved_state = {key: self._state[key] for key in buffer_keys}
        saved_indices = self._selection_indices
        try:
            for key, tensor in saved_state.items():
                self._state[key] = tensor.detach().cpu().numpy()
            if saved_indices is not None:
                self._selection_indices = saved_indices.detach().cpu().numpy()  # type: ignore[assignment]
            super().save(
                path,
                mode=mode,
                store=store,
                skip=skip,
                compression_level=compression_level,
            )
        finally:
            for key, tensor in saved_state.items():
                self._state[key] = tensor
            self._selection_indices = saved_indices

    def _post_load(self) -> None:
        """Rehydrate NumPy-backed state into CPU tensors after deserialization.

        Called by ``AutoSerialize._recursive_load``. This handles both files
        written by :meth:`save` and older files written when Vector was
        NumPy-backed; in both cases the stored dtype is preserved rather than
        being coerced to the current default.
        """
        state = getattr(self, "_state", None)
        if isinstance(state, dict):
            if "data" in state:
                state["data"] = _as_tensor(state["data"])
            for key in ("cell_starts", "cell_lengths"):
                if key in state:
                    state[key] = _as_tensor(state[key], dtype=torch.int64)
            if "shape" in state:
                state["shape"] = tuple(int(dim) for dim in state["shape"])

        selection_shape = getattr(self, "_selection_shape", None)
        if selection_shape is not None:
            self._selection_shape = tuple(int(dim) for dim in selection_shape)

        indices = getattr(self, "_selection_indices", None)
        if indices is not None:
            self._selection_indices = _as_tensor(indices, dtype=torch.int64)

        selected_fields = getattr(self, "_selected_fields", None)
        if selected_fields is not None:
            self._selected_fields = tuple(selected_fields)

    # ------------------------------------------------------------------ #
    # Private helpers — backing-store access
    # ------------------------------------------------------------------ #

    @property
    def _full_num_fields(self) -> int:
        return len(self._state["fields"])

    def _field_indices(self) -> list[int]:
        """Map selected field names to column indices in the backing buffer.

        Returned as a plain list so it can index a tensor on any device without
        needing a matching index tensor there.
        """
        if self._selected_fields is None:
            return list(range(self._full_num_fields))

        lookup = {field: i for i, field in enumerate(self._state["fields"])}
        try:
            return [lookup[field] for field in self._selected_fields]
        except KeyError as exc:
            raise KeyError(f"Unknown field(s): {[str(exc.args[0])]}") from exc

    def _require_full_field_view(self, operation: str) -> None:
        """Raise if a schema-changing/full-row operation is attempted on a field view."""
        if self._selected_fields is not None:
            raise ValueError(f"{operation} is only allowed when all fields are selected.")

    def _selected_cell_indices(self) -> torch.Tensor:
        """Return linear cell indices for the current fixed-grid selection."""
        if self._selection_indices is None:
            return _cell_arange(_cell_count(self._state["shape"]))
        return self._selection_indices

    def _selected_cell_lengths(self) -> torch.Tensor:
        """Return per-cell row counts for the current selection, in order."""
        lengths = self._state["cell_lengths"]
        if self._selection_indices is None:
            return lengths
        return lengths[self._selection_indices]

    def _row_gather_index(self, cells: torch.Tensor) -> torch.Tensor:
        """Buffer row indices for ``cells``, concatenated in row-major order.

        This is the vectorized replacement for walking cells one at a time: the
        result indexes ``_state["data"]`` directly, so gathering a whole
        selection is a single ``index_select`` instead of one slice per cell.
        """
        lengths = self._state["cell_lengths"][cells]
        total = int(lengths.sum())
        if total == 0:
            return torch.empty(0, dtype=torch.int64)
        starts = self._state["cell_starts"][cells]
        # Row r of output cell k comes from buffer row (start_k - out_start_k) + r.
        offsets = starts - (torch.cumsum(lengths, 0) - lengths)
        return torch.repeat_interleave(offsets, lengths) + torch.arange(total, dtype=torch.int64)

    def _cell_row_count(self, linear_index: int) -> int:
        """Return the row count for one cell in the backing buffer."""
        return int(self._state["cell_lengths"][linear_index])

    def _cell_matrix(self, linear_index: int) -> torch.Tensor:
        """Return the full backing matrix for one cell."""
        start = int(self._state["cell_starts"][linear_index])
        length = int(self._state["cell_lengths"][linear_index])
        return self._state["data"][start : start + length]

    def _selected_cell_matrix(self, linear_index: int) -> torch.Tensor:
        """Return one cell with the current field selection applied."""
        return _select_columns(self._cell_matrix(linear_index), self._field_indices())

    def _to_buffer(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast a tensor to the backing buffer's dtype and device."""
        data = self._state["data"]
        return tensor.to(dtype=data.dtype, device=data.device)

    def _write_selected_rows(self, values: torch.Tensor) -> None:
        """Write a ``(total_rows, num_fields)`` block into the selected rows and fields.

        This is the scatter counterpart of :meth:`flatten`: every selected row is
        written with one ``index_put_``, so the cost does not scale with the cell
        count. Values that cannot be cast to the buffer dtype without changing
        kind (e.g. float into an integer buffer) raise, matching torch in-place
        semantics.
        """
        data = self._state["data"]
        rows = self._row_gather_index(self._selected_cell_indices()).to(data.device)
        if rows.numel() == 0:
            return
        if not torch.can_cast(values.dtype, data.dtype):
            raise TypeError(f"Cannot write {values.dtype} values into a {data.dtype} Vector.")
        cols = torch.tensor(self._field_indices(), dtype=torch.int64, device=data.device)
        data[rows[:, None], cols[None, :]] = values.to(dtype=data.dtype, device=data.device)

    def _coerce_cell(self, value: Any, num_fields: int) -> torch.Tensor:
        """Normalize a single-cell payload onto this Vector's dtype/device."""
        data = self._state["data"]
        return _coerce_cell_array(value, num_fields, data.dtype, data.device)

    def _broadcast_values(self, value: Any, total_rows: int, num_fields: int) -> torch.Tensor:
        """Broadcast array-like input onto this Vector's dtype/device."""
        data = self._state["data"]
        return _broadcast_field_values(value, total_rows, num_fields, data.dtype, data.device)

    def _replace_cells(self, targets: torch.Tensor, arrays: Sequence[Any]) -> None:
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

        normalized = [self._coerce_cell(array, self._full_num_fields) for array in arrays]
        payloads = [array for array in normalized if array.shape[0] > 0]
        if payloads:
            appended = torch.cat(payloads, dim=0)
            self._state["data"] = torch.cat((self._state["data"], appended), dim=0)

        lengths = torch.tensor([array.shape[0] for array in normalized], dtype=torch.int64)
        cursor = self._state["data"].shape[0] - int(lengths.sum())
        self._state["cell_starts"][targets] = cursor + torch.cumsum(lengths, 0) - lengths
        self._state["cell_lengths"][targets] = lengths

        self._maybe_compact_storage()

    def _expand_storage(self, num_new_fields: int) -> None:
        """Append new NaN-initialized columns for added fields."""
        data = self._state["data"]
        # Promote to a float dtype first: torch.full(..., nan) rejects integer dtypes.
        # This is a "smallest float that holds NaN" rule, independent of the
        # new-Vector default in DEFAULT_DTYPE.
        dtype = torch.promote_types(data.dtype, torch.float32)
        filler = torch.full(
            (data.shape[0], num_new_fields), float("nan"), dtype=dtype, device=data.device
        )
        self._state["data"] = torch.cat((data.to(dtype), filler), dim=1)

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
        self._require_unique_cell_targets("Assignment")
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
            arrays = [
                value._selected_cell_matrix(index).clone() for index in source_cells.tolist()
            ]
            self._replace_cells(targets, arrays)
            return

        array = self._coerce_cell(value, self.num_fields)
        self._replace_cells(targets, [array] * len(targets))

    def _assign_selected_fields(self, value: Any) -> None:
        """Update only the selected columns while preserving row counts.

        This is the in-place path for assignments such as
        ``vector.select_fields("kx")[...] = rhs``. The target cell structure is
        preserved, so each target cell keeps its existing row count and only the
        selected columns are overwritten.
        """
        row_counts = self.row_counts()

        if isinstance(value, Vector):
            if value.num_cells != self.num_cells:
                raise ValueError(f"Expected {self.num_cells} cells, got {value.num_cells}")
            if value.num_fields != self.num_fields:
                raise ValueError(f"Expected {self.num_fields} fields, got {value.num_fields}")
            if value.row_counts() != row_counts:
                raise ValueError("Per-cell row counts must match for field-selected assignment.")
            # flatten() gathers into a new tensor, so overlapping source and
            # target selections are read before anything is written.
            self._write_selected_rows(self._to_buffer(value.flatten()))
            return

        self._write_selected_rows(
            self._broadcast_values(_scalar_value(value), sum(row_counts), self.num_fields)
        )

    # ------------------------------------------------------------------ #
    # Private helpers — arithmetic
    # ------------------------------------------------------------------ #

    def _binary_op(self, other: Any, op: Any, reverse: bool = False) -> "Vector":
        """Return a new Vector produced by elementwise arithmetic."""
        row_counts = self.row_counts()
        lhs = self.flatten()

        if isinstance(other, Vector):
            if other.num_cells != self.num_cells:
                raise ValueError(f"Expected {self.num_cells} cells, got {other.num_cells}")
            if other.num_fields != self.num_fields:
                raise ValueError(f"Expected {self.num_fields} fields, got {other.num_fields}")
            if other.row_counts() != row_counts:
                raise ValueError("Per-cell row counts must match for Vector arithmetic.")
            rhs: Any = other.flatten()
        elif _is_scalar(other):
            rhs = _scalar_value(other)
        else:
            if not isinstance(other, torch.Tensor):
                # NumPy and list operands promote like Python scalars: float64
                # input keeps a float32 Vector float32, while an integer Vector
                # times float values still gives a float result.
                other = _as_tensor(other)
                other = other.to(torch.result_type(lhs, torch.zeros((), dtype=other.dtype)))
            rhs = _broadcast_field_values(
                other,
                sum(row_counts),
                self.num_fields,
                dtype=None,
                device=lhs.device,
            )

        rows = op(rhs, lhs) if reverse else op(lhs, rhs)
        return _vector_from_rows(self, rows, row_counts)

    def _inplace_unary(self, op: Any) -> None:
        """Apply a unary elementwise operation in-place to the selected fields."""
        self._require_unique_cell_targets("In-place arithmetic")
        self._write_selected_rows(op(self.flatten()))

    def _inplace_op(self, other: Any, op: Any, reverse: bool = False) -> None:
        """Apply elementwise arithmetic in-place to the selected fields."""
        self._require_unique_cell_targets("In-place arithmetic")
        row_counts = self.row_counts()
        lhs = self.flatten()

        if isinstance(other, Vector):
            if other.num_cells != self.num_cells:
                raise ValueError(f"Expected {self.num_cells} cells, got {other.num_cells}")
            if other.num_fields != self.num_fields:
                raise ValueError(f"Expected {self.num_fields} fields, got {other.num_fields}")
            if other.row_counts() != row_counts:
                raise ValueError("Per-cell row counts must match for Vector arithmetic.")
            rhs: Any = self._to_buffer(other.flatten())
        elif _is_scalar(other):
            rhs = _scalar_value(other)
        else:
            rhs = self._broadcast_values(other, sum(row_counts), self.num_fields)

        self._write_selected_rows(op(rhs, lhs) if reverse else op(lhs, rhs))

    def _require_unique_cell_targets(self, operation: str) -> None:
        """Reject ambiguous write-through operations on repeated cell indices."""
        indices = self._selection_indices
        if indices is None:
            return
        if indices.numel() != torch.unique(indices).numel():
            raise ValueError(
                f"{operation} does not support repeated cell indices in a write selection."
            )


def _resolve_device(device: str | int | torch.device | None) -> torch.device:
    """Normalize a device specifier.

    Note that ``None`` means CPU here, whereas ``config.validate_device(None)``
    resolves to whatever accelerator is available. A data container that
    silently lands on a GPU is surprising, so the default is explicit and
    ``to()`` is how you move one.
    """
    if device is None:
        return torch.device("cpu")
    resolved, _ = config.validate_device(device)
    return torch.device(resolved)


def _as_tensor(
    value: Any,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Coerce array-like input (tensor, ndarray, sequence, scalar) to a tensor.

    Incoming tensors are detached: the row buffer is written in place, which is
    not allowed on a tensor that requires grad.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        if isinstance(value, np.ndarray) and any(stride < 0 for stride in value.strides):
            # torch cannot wrap negatively-strided memory (e.g. arr[::-1]).
            value = np.ascontiguousarray(value)
        tensor = torch.as_tensor(value)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if device is not None and tensor.device != torch.device(device):
        tensor = tensor.to(device)
    return tensor


def _is_scalar(value: Any) -> bool:
    """Return True for values that broadcast as a single number."""
    if isinstance(value, torch.Tensor):
        return value.ndim == 0
    return isinstance(value, (numbers.Number, np.generic))


def _scalar_value(value: Any) -> Any:
    """Unwrap a scalar-like value into something torch ops accept directly."""
    # torch ops do not reliably accept numpy scalar types; python/tensor pass through.
    return value.item() if isinstance(value, np.generic) else value


def _flatten_torch_input(value: Any, per_row: int | None = None) -> Any:
    """Prepare one argument for torch dispatch.

    A Vector is replaced by its flattened rows. When ``per_row`` is given, a 1D
    tensor of that length becomes a column, so for single-field Vectors ``x + v``
    and ``v + x`` both apply one value per row.
    """
    if isinstance(value, Vector):
        return value.flatten()
    if isinstance(value, torch.Tensor) and value.ndim == 1 and value.shape[0] == per_row:
        return value.reshape(-1, 1)
    return value


def _maybe_wrap_result(template: "Vector", value: Any, row_counts: list[int]) -> Any:
    """Rebuild a Vector from a shape-compatible result of an approved rowwise operation."""
    if isinstance(value, torch.Tensor) and tuple(value.shape) == (
        sum(row_counts),
        template.num_fields,
    ):
        return _vector_from_rows(template, value, row_counts)
    return value


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


@functools.lru_cache(maxsize=8)
def _cell_arange(num_cells: int) -> torch.Tensor:
    """Cached ``arange(num_cells)``, shared read-only by every root Vector of that size."""
    return torch.arange(num_cells, dtype=torch.int64)


def _cell_count(shape: tuple[int, ...]) -> int:
    """Return the number of fixed-grid cells in a shape."""
    return math.prod(shape) if shape else 1


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


def _resolve_dim_names(dim_names: Sequence[str] | None, ndim: int) -> list[str]:
    """Resolve fixed-grid index column names for DataFrame export."""
    if dim_names is None:
        return [f"dim_{i}" for i in range(ndim)]
    resolved = [str(name) for name in dim_names]
    if len(resolved) != ndim:
        raise ValueError(f"Expected {ndim} dim_names, got {len(resolved)}")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate dim_names are not allowed.")
    return resolved


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


def _coerce_cell_array(
    value: Any,
    num_fields: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Normalize a single-cell payload to shape ``(n_rows, num_fields)``."""
    if isinstance(value, Vector):
        if value.shape != ():
            raise ValueError("Expected a 0D Vector for single-cell assignment.")
        array = value.tensor.clone()
    else:
        array = _as_tensor(value)

    if array.ndim == 0:
        raise ValueError("Cell assignment requires a 2D array.")
    if array.ndim == 1 or array.shape == (0, 0):
        if array.numel() == 0:
            array = torch.empty((0, num_fields), dtype=dtype, device=device)
        elif num_fields == 1:
            array = array.reshape(-1, 1)
        else:
            array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("Cell assignment requires a 2D array.")
    if array.shape[1] != num_fields:
        raise ValueError(f"Expected {num_fields} fields, got {array.shape[1]}")
    return array.to(dtype=dtype, device=device)


def _flatten_fixed_grid(node: Any) -> tuple[tuple[int, ...], list[torch.Tensor]]:
    """Recursively flatten nested fixed-grid input into row-major cell order."""
    if isinstance(node, (np.ndarray, torch.Tensor)):
        return (), [_coerce_inferred_cell_array(node)]
    if not isinstance(node, (list, tuple)):
        raise TypeError("Data must be a nested list/tuple of cell arrays or row sequences.")
    if _looks_like_cell_rows(node):
        return (), [_coerce_inferred_cell_array(node)]
    if len(node) == 0:
        return (0,), []

    child_shape: tuple[int, ...] | None = None
    cells: list[torch.Tensor] = []
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
    """Return True for a single row of scalar values.

    An empty list is not a row, so ``[[], []]`` is read as two empty cells
    rather than one cell with two zero-length rows.
    """
    if isinstance(item, (np.ndarray, torch.Tensor)):
        return item.ndim == 1 and item.shape[0] > 0
    if not isinstance(item, (list, tuple)):
        return False
    return len(item) > 0 and all(_is_scalar(value) for value in item)


def _coerce_inferred_cell_array(value: Any) -> torch.Tensor | NDArray[Any]:
    """Infer a 2D cell from row-like input during ``from_data``.

    NumPy input stays NumPy, so that ``from_data`` can join all cells with one
    ``np.concatenate`` and convert to torch once.
    """
    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            return value.reshape(0, 0) if value.size == 0 else value.reshape(1, -1)
        if value.ndim != 2:
            raise ValueError("Cell data must be 1D or 2D.")
        return value
    array = _as_tensor(value)
    if array.ndim == 0:
        raise ValueError("Cell data must be 1D or 2D.")
    if array.ndim == 1:
        if array.numel() == 0:
            return torch.empty((0, 0), dtype=array.dtype)
        return array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("Cell data must be 1D or 2D.")
    return array


def _select_linear_indices(
    shape: tuple[int, ...],
    current_indices: torch.Tensor,
    idx: Any,
) -> tuple[tuple[int, ...], torch.Tensor]:
    """Apply fixed-grid indexing to a flattened cell-index view.

    ``current_indices`` stores the linear cell indices represented by the current
    selection. This helper reshapes those indices to the current selection shape,
    applies NumPy-like indexing on the fixed-grid axes, and then returns:
    - the output fixed-grid shape
    - the flattened linear indices of the selected cells, in row-major order
    """
    key = idx if isinstance(idx, tuple) else (idx,)
    if len(key) == len(shape) and all(type(i) is int for i in key):
        linear = 0
        for i, size in zip(key, shape):
            if not -size <= i < size:
                raise IndexError("Vector index out of range")
            linear = linear * size + (i % size)
        return (), current_indices[linear : linear + 1].clone()

    if shape == ():
        if idx in ((), Ellipsis):
            return (), torch.tensor([int(current_indices[0])], dtype=torch.int64)
        raise IndexError("Too many indices for 0D Vector")

    index_tuple = _normalize_index_tuple(idx, len(shape))
    current_grid = current_indices.reshape(shape)

    axis_positions: list[torch.Tensor] = []
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
        return (), torch.tensor([value], dtype=torch.int64)

    mesh_inputs = [
        positions if not is_scalar else positions[:1]
        for positions, is_scalar in zip(axis_positions, scalar_axes)
    ]
    grids = torch.meshgrid(*mesh_inputs, indexing="ij")
    selected = current_grid[tuple(grids)].reshape(-1).to(torch.int64)
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


def _positions_for_axis(axis_index: Any, size: int) -> tuple[torch.Tensor, bool]:
    """Resolve one axis index into concrete positions and scalar-vs-vector shape behavior."""
    if isinstance(axis_index, (bool, np.bool_)):
        raise TypeError("Boolean scalars are not valid Vector indices.")

    if isinstance(axis_index, (int, np.integer)):
        index = int(axis_index)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError("Vector index out of range")
        return torch.tensor([index], dtype=torch.int64), True

    if isinstance(axis_index, slice):
        return torch.arange(size, dtype=torch.int64)[axis_index], False

    array = _as_index_tensor(axis_index)
    if array.ndim == 0:
        if _is_integer_dtype(array.dtype):
            return _positions_for_axis(int(array.item()), size)
        raise TypeError(f"Unsupported index type: {type(axis_index)!r}")

    if array.dtype == torch.bool:
        if array.ndim != 1:
            raise IndexError("Full-grid boolean masks are not supported.")
        if array.shape[0] != size:
            raise IndexError(
                f"Boolean mask length {array.shape[0]} does not match axis length {size}"
            )
        return array.nonzero(as_tuple=False).reshape(-1).to(torch.int64), False

    if array.ndim != 1:
        raise IndexError("Fancy indexing arrays must be one-dimensional.")
    if array.numel() == 0:
        return torch.empty(0, dtype=torch.int64), False
    if not _is_integer_dtype(array.dtype):
        raise TypeError("Fancy indices must be integers or booleans.")

    positions = array.to(torch.int64).clone()
    positions[positions < 0] += size
    if bool(((positions < 0) | (positions >= size)).any()):
        raise IndexError("Vector index out of range")
    return positions, False


def _as_index_tensor(value: Any) -> torch.Tensor:
    """Coerce an index-like value into a CPU tensor without forcing a dtype."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return _as_tensor(np.asarray(value))


def _is_integer_dtype(dtype: torch.dtype) -> bool:
    """Return True for signed/unsigned integer dtypes (excluding bool)."""
    return not dtype.is_floating_point and not dtype.is_complex and dtype != torch.bool


def _broadcast_field_values(
    value: Any,
    total_rows: int,
    num_fields: int,
    dtype: torch.dtype | None,
    device: torch.device,
) -> torch.Tensor:
    """Broadcast array-like input to flattened rowwise assignment shape."""
    array = _as_tensor(value, dtype=dtype, device=device)
    if array.ndim == 0:
        return array.reshape(1, 1).expand(total_rows, num_fields)
    if num_fields == 1 and array.ndim == 1:
        if total_rows == 0 and array.shape[0] == 0:
            return array.reshape(0, 1)
        if array.shape[0] != total_rows:
            raise ValueError(f"Expected {total_rows} values, got {array.shape[0]}")
        return array.reshape(total_rows, 1)
    try:
        return torch.broadcast_to(array, (total_rows, num_fields))
    except RuntimeError as exc:
        raise ValueError(
            f"Cannot broadcast value with shape {tuple(array.shape)} "
            f"to ({total_rows}, {num_fields})"
        ) from exc


def _vector_from_rows(
    template: Vector,
    rows: torch.Tensor,
    row_counts: list[int],
) -> Vector:
    """Build a Vector from an already row-major block of rows.

    ``rows`` is used directly as the new row buffer and the offsets are
    computed from ``row_counts``, which avoids splitting and re-concatenating
    the cells.
    """
    source = _as_tensor(rows)
    result = Vector.from_shape(
        shape=template.shape,
        fields=template.fields,
        units=template.units,
        name=template.name,
        dtype=source.dtype,
        device=source.device,
    )
    result._state["metadata"] = copy.deepcopy(template.metadata)

    lengths = torch.tensor(row_counts, dtype=torch.int64)
    result._state["data"] = source.contiguous()
    result._state["cell_lengths"] = lengths
    result._state["cell_starts"] = torch.cumsum(lengths, 0) - lengths
    return result


def _is_contiguous(indices: Sequence[int]) -> bool:
    """Return True when integer column indices form one ascending contiguous slice."""
    if len(indices) <= 1:
        return True
    return all(after - before == 1 for before, after in zip(indices, indices[1:]))


def _select_columns(rows: torch.Tensor, cols: list[int]) -> torch.Tensor:
    """Apply a field selection to a 2D row block.

    A contiguous ascending column run is returned as a writable view; any other
    selection goes through advanced indexing, which copies. Reordered selections
    must take the copying path so the columns come back in the requested order
    rather than storage order.
    """
    if _is_contiguous(cols):
        if not cols:
            return rows[:, :0]
        return rows[:, cols[0] : cols[-1] + 1]
    return rows[:, cols]
