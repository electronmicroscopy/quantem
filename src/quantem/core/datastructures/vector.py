from __future__ import annotations

import copy
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


class Vector(AutoSerialize):
    """Ragged cell data on a fixed grid, backed by torch.

    A ``Vector`` has two independent axes of structure:
    - fixed-grid dimensions given by ``shape``
    - ragged rows stored inside each fixed-grid cell

    Each ragged row has one value per named field, so each cell behaves like a
    small 2D tensor with shape ``(n_rows, num_fields)``, where ``n_rows`` may
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
    dtype : torch.dtype, optional
        Row-buffer dtype. Defaults to ``torch.float32``.
    device : str or torch.device, optional
        Device for the row buffer. Defaults to ``"cpu"``.

    Notes
    -----
    This class is torch-native: ``tensor``, ``flatten()`` and every arithmetic
    result are ``torch.Tensor`` values living on ``device``. NumPy arrays,
    Python sequences and scalars are accepted as *input* anywhere a payload is
    taken and converted at the boundary; call ``numpy()`` to get NumPy back.

    The public API keeps fixed-grid indexing and field selection separate:
    - use ``[]`` for fixed-grid indexing
    - use ``select_fields(...)`` for field selection

    Fixed-grid indexing always returns a ``Vector``. A 0D selection exposes its
    underlying cell tensor through ``.tensor``. Multi-cell selections can be
    concatenated with ``flatten()``.

    The internal representation is compact:
    - ``_state["data"]`` stores all ragged rows in one numeric 2D tensor
    - ``_state["cell_starts"]`` stores the start offset for each cell
    - ``_state["cell_lengths"]`` stores the row count for each cell

    The offset bookkeeping is deliberately kept on the CPU even when the row
    buffer lives on a GPU: it is read one scalar at a time, so keeping it on
    the device would force a synchronization on every cell access.

    A ``Vector`` selection is a write-through view over shared storage. Views
    track only the selected fixed-grid shape, selected cell indices, and selected
    field names. Because ``_state`` is shared, ``to(device)`` moves every view
    of the same Vector.

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
        """Create an empty Vector with the given fixed-grid shape and fields."""
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

        The outer nesting defines the fixed-grid shape. Each leaf must coerce to a
        2D cell tensor with consistent field count across all cells. Leaves may be
        tensors, NumPy arrays or nested sequences; they are cast to ``dtype``
        (``torch.float32`` by default), so pass ``dtype=torch.float64`` to keep
        double precision.
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
            dtype=dtype,
            device=device,
            _token=cls._token,
        )
        vector._replace_cells(torch.arange(len(cell_arrays), dtype=torch.int64), cell_arrays)
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
        """Return the selected cell as a torch tensor.

        Unlike ``Dataset.tensor``, which is the whole payload, this is *one
        cell*: it is only valid for 0D selections and raises otherwise. Use
        :meth:`flatten` to get every row of a multi-cell selection.

        Contiguous field selections return writable views into the backing
        storage. Reordered or non-contiguous selections return a copy, because
        torch cannot expose a writable column-subset view for that layout.
        """
        if self.shape != ():
            raise ValueError(".tensor is only valid when the selection contains exactly one cell.")
        return self._selected_cell_matrix(int(self._selected_cell_indices()[0]))

    def flatten(self) -> torch.Tensor:
        """Concatenate selected cells in row-major order.

        Returns a 2D tensor with shape ``(total_rows, num_fields)`` even for
        single-field selections.
        """
        data = self._state["data"]
        gather = self._row_gather_index(self._selected_cell_indices())
        if gather.numel() == 0:
            return torch.empty((0, self.num_fields), dtype=data.dtype, device=data.device)
        return _select_columns(data.index_select(0, gather.to(data.device)), self._field_indices())

    def numpy(self) -> NDArray[Any]:
        """Return the flattened selection as a NumPy array.

        This is the NumPy counterpart of :meth:`flatten`, **not** of
        :attr:`tensor` -- it covers the whole selection, not one cell. The
        result is a detached CPU copy, and like ``Dataset.numpy()`` it is marked
        read-only so accidental in-place writes raise instead of silently going
        nowhere; use :meth:`set_flattened` to write values back.
        """
        array = self.flatten().detach().cpu().numpy()
        array.flags.writeable = False
        return array

    def row_counts(self) -> list[int]:
        """Return per-cell row counts in the current selection order."""
        return self._selected_cell_lengths().tolist()

    def to(self, device: str | int | torch.device) -> "Vector":
        """Move the backing row buffer to ``device`` and return ``self``.

        ``device`` is normalized via :func:`quantem.core.config.validate_device`
        so ``"cuda"``, ``0``, ``"cuda:0"`` and ``torch.device("cuda:0")`` all
        resolve to the same canonical device.

        Because all views share one ``_state``, this moves every view of the same
        Vector, not just this one. The offset bookkeeping stays on the CPU by
        design.
        """
        self._state["data"] = self._state["data"].to(_resolve_device(device))
        return self

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

        new_rows = target._coerce_cell(rows, target.num_fields)
        if new_rows.shape[0] == 0:
            return

        cell_index = int(target._selected_cell_indices()[0])
        combined = torch.cat((target._cell_matrix(cell_index), new_rows), dim=0)
        target._replace_cells(torch.tensor([cell_index], dtype=torch.int64), [combined])

    def set_flattened(self, values: Any) -> None:
        """Write values back in flattened row-major order.

        This updates existing rows without changing per-cell row counts. It is
        the rowwise companion to ``flatten()`` and is especially useful for
        tensor-based transforms that operate on all selected rows at once.
        """
        self._require_unique_cell_targets("set_flattened")
        field_indices = self._field_indices()
        targets = self._selected_cell_indices().tolist()
        row_counts = self.row_counts()
        total_rows = sum(row_counts)

        if isinstance(values, Vector):
            if values.num_fields != self.num_fields:
                raise ValueError(f"Expected {self.num_fields} fields, got {values.num_fields}")
            flat_values = values.flatten()
            if flat_values.shape[0] != total_rows:
                raise ValueError(f"Expected {total_rows} rows, got {flat_values.shape[0]}")
            flat_values = self._to_buffer(flat_values)
        else:
            flat_values = self._broadcast_values(values, total_rows, self.num_fields)

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
        """Apply torch functions over the ragged rows.

        Every ``Vector`` argument is replaced by its flattened rows and ``func``
        is applied to those tensors. The result is rebuilt into a ``Vector``
        -- preserving the selection shape and fields -- only when both hold:

        - ``func`` is in :data:`_SAFE_ELEMENTWISE_TORCH_FUNCTIONS`, i.e. it maps
          each row to a row and so keeps the ragged structure meaningful
        - the result is a tensor shaped ``(total_rows, num_fields)``

        Anything else is returned exactly as torch produced it. That covers
        reductions (``torch.sum``), predicates (``torch.allclose``), and
        shape-changing ops (``torch.t``) -- all of which still *work*, they just
        hand back plain tensors rather than Vectors.

        The allowlist is what makes this safe: shape alone is not a reliable
        test, since ``torch.t`` on a Vector whose row and field counts happen to
        be equal returns a same-shaped tensor that would otherwise be rewrapped
        with its rows silently permuted.
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

        flat_args = tuple(_flatten_torch_input(value) for value in args)
        flat_kwargs = {key: _flatten_torch_input(value) for key, value in kwargs.items()}
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

        Every ragged row becomes one DataFrame row. The fixed-grid location of
        that row is carried in leading integer columns -- one per fixed-grid
        dimension -- followed by one column per selected field.

        Grid coordinates are always reported in the *root* grid, not the
        selection's local coordinates, so a view such as ``v[1, :]`` still
        reports ``dim_0 == 1``. Both fixed-grid indexing and ``select_fields``
        are honored, so ``v[1].select_fields("kx").to_polars()`` returns only
        the ``kx`` rows belonging to cell 1.

        Requires polars, which is an optional dependency:
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

        The row buffer and offset arrays are written as NumPy arrays rather than as
        torch tensors, so they land in chunked, compressed Zarr arrays instead of
        uncompressed ``torch.save`` blobs. :meth:`_post_load` converts them back to
        CPU tensors on load, which also makes a GPU-saved Vector loadable without CUDA.

        Note this only applies when the Vector is the *root* of the save.
        ``AutoSerialize`` walks nested objects with ``_recursive_save`` rather than
        calling their ``save()``, so a Vector held as an attribute of another
        serializable object still round-trips correctly but writes its buffers as
        uncompressed blobs. Fixing that properly means teaching
        ``AutoSerialize._serialize_value`` to store plain non-grad tensors as Zarr
        arrays, at which point this whole override collapses to ``compact()``.
        """
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
            return torch.arange(_cell_count(self._state["shape"]), dtype=torch.int64)
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
        # new-Vector default in DEFAULT_DTYPE -- keep them separate.
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
        targets = self._selected_cell_indices().tolist()
        field_indices = self._field_indices()
        row_counts = self.row_counts()
        total_rows = sum(row_counts)

        if isinstance(value, Vector):
            source_cells = value._selected_cell_indices().tolist()
            if len(targets) != len(source_cells):
                raise ValueError(f"Expected {len(targets)} cells, got {len(source_cells)}")
            if value.num_fields != self.num_fields:
                raise ValueError(f"Expected {self.num_fields} fields, got {value.num_fields}")
            source_counts = value.row_counts()
            if row_counts != source_counts:
                raise ValueError("Per-cell row counts must match for field-selected assignment.")
            snapshots = [
                self._to_buffer(value._selected_cell_matrix(index)).clone()
                for index in source_cells
            ]
            for target, array in zip(targets, snapshots):
                cell = self._cell_matrix(int(target))
                if array.shape[0] > 0:
                    cell[:, field_indices] = array
            return

        if _is_scalar(value):
            scalar = _scalar_value(value)
            for target in targets:
                cell = self._cell_matrix(int(target))
                if cell.shape[0] > 0:
                    cell[:, field_indices] = scalar
            return

        broadcast = self._broadcast_values(value, total_rows, self.num_fields)
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
        targets = self._selected_cell_indices().tolist()
        field_indices = self._field_indices()
        for target in targets:
            cell = self._cell_matrix(int(target))
            lhs = cell[:, field_indices]
            if lhs.shape[0] > 0:
                cell[:, field_indices] = op(lhs)

    def _inplace_op(self, other: Any, op: Any, reverse: bool = False) -> None:
        """Apply elementwise arithmetic in-place to the selected fields."""
        self._require_unique_cell_targets("In-place arithmetic")
        targets = self._selected_cell_indices().tolist()
        field_indices = self._field_indices()
        row_counts = self.row_counts()
        total_rows = sum(row_counts)

        if isinstance(other, Vector):
            source_cells = other._selected_cell_indices().tolist()
            if len(targets) != len(source_cells):
                raise ValueError(f"Expected {len(targets)} cells, got {len(source_cells)}")
            if other.num_fields != self.num_fields:
                raise ValueError(f"Expected {self.num_fields} fields, got {other.num_fields}")
            source_counts = other.row_counts()
            if row_counts != source_counts:
                raise ValueError("Per-cell row counts must match for Vector arithmetic.")
            snapshots = [
                self._to_buffer(other._selected_cell_matrix(index)).clone()
                for index in source_cells
            ]
            for target, rhs in zip(targets, snapshots):
                cell = self._cell_matrix(int(target))
                lhs = cell[:, field_indices]
                cell[:, field_indices] = op(rhs, lhs) if reverse else op(lhs, rhs)
            return

        if _is_scalar(other):
            scalar = _scalar_value(other)
            for target in targets:
                cell = self._cell_matrix(int(target))
                lhs = cell[:, field_indices]
                if lhs.shape[0] > 0:
                    cell[:, field_indices] = op(scalar, lhs) if reverse else op(lhs, scalar)
            return

        broadcast = self._broadcast_values(other, total_rows, self.num_fields)
        cursor = 0
        for target, rows in zip(targets, row_counts):
            chunk = broadcast[cursor : cursor + rows]
            cell = self._cell_matrix(int(target))
            lhs = cell[:, field_indices]
            if rows > 0:
                cell[:, field_indices] = op(chunk, lhs) if reverse else op(lhs, chunk)
            cursor += rows

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


def _flatten_torch_input(value: Any) -> Any:
    """Replace a Vector argument with its flattened rows for torch dispatch."""
    return value.flatten() if isinstance(value, Vector) else value


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
    if array.ndim == 1:
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
    """Return True for a single row of scalar values."""
    if isinstance(item, (np.ndarray, torch.Tensor)):
        return item.ndim == 1
    if not isinstance(item, (list, tuple)):
        return False
    return all(_is_scalar(value) for value in item)


def _coerce_inferred_cell_array(value: Any) -> torch.Tensor:
    """Infer a 2D cell tensor from row-like input during ``from_data``."""
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

    ``rows`` is exactly the buffer the result needs, so it is adopted directly
    and the offsets are derived from ``row_counts`` -- no per-cell split and
    re-concatenation.
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
