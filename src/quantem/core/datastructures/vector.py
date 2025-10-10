from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import (
    validate_fields,
    validate_num_fields,
    validate_shape,
    validate_vector_data,
    validate_vector_data_for_inference,
    validate_vector_units,
)


class Vector(AutoSerialize):
    """
    A class for holding vector data with ragged array lengths. This class supports any number of fixed dimensions
    (indexed first) followed by a ragged numpy array that can have any number of entries (rows) and columns (fields).
    Inherits from AutoSerialize for serialization support.

    Basic Usage:
    -----------
    # Create a 2D vector with shape=(4, 3) and 3 named fields
    v = Vector.from_shape(shape=(4, 3), fields=['field0', 'field1', 'field2'])

    # Alternative creation with num_fields instead of fields
    v = Vector.from_shape(shape=(4, 3), num_fields=3)  # Fields will be named field_0, field_1, field_2

    # Create with custom name and units
    v = Vector.from_shape(
        shape=(4, 3),
        fields=['field0', 'field1', 'field2'],
        name='my_vector',
        units=['unit0', 'unit1', 'unit2'],
    )

    # Access data at specific indices
    data = v[0, 1]  # Returns numpy array at position (0,1)

    # Set data at specific indices
    v[0, 1] = np.array([[1.0, 2.0, 3.0]])  # Must match num_fields

    # Create a deep copy
    v_copy = v.copy()

    Example usage of from_data:
    -----------------------------------
    data = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8], [9, 10]])
    ]
    v = Vector.from_data(
        data,
        fields=['x', 'y'],
        name='my_ragged_vector',
        units=['m', 'm']
    )

    # Or using lists instead of numpy arrays:
    data = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8], [9, 10]],
    ]
    v = Vector.from_data(
        data,
        fields=['x', 'y'],
        name='my_ragged_vector',
        units=['m', 'm']
    )

    Field Operations:
    ----------------
    # Access a specific field
    field_data = v['field0']  # Returns a FieldView object

    # Perform operations on a field
    v['field0'] += 16  # Add 16 to all field0 values

    # Apply a function to a field
    v['field2'] = lambda x: x * 2  # Double all field2 values

    # Get flattened field data
    field_flat = v['field0'].flatten()  # Returns 1D numpy array

    # Set field data from flattened array
    v['field2'].set_flattened(new_values)  # Must match total length

    Advanced Operations:
    -------------------
    # Complex field calculations
    scale = v['field0'].flatten() / (v['field0'].flatten()**2 + v['field1'].flatten()**2)
    v['field2'].set_flattened(v['field2'].flatten() * scale)

    # Slicing and assignment
    v[2:4, 1] = v[1:3, 1]  # Copy data from one region to another

    # Boolean indexing
    mask = v['field0'].flatten() > 0
    v['field2'].set_flattened(v['field2'].flatten() * mask)

    # Field management
    v.add_fields(('field3', 'field4', 'field5'))  # Add new fields
    v.remove_fields(('field3', 'field4', 'field5'))  # Remove fields

    Direct Data Access:
    ------------------
    # Get data with integer indexing
    data = v.get_data(0, 1)  # Returns numpy array at (0,1)

    # Get data with slice indexing
    data = v.get_data(slice(0, 2), 1)  # Returns list of arrays for rows 0-1 at column 1

    # Set data with integer indexing
    v.set_data(np.array([[1.0, 2.0, 3.0]]), 0, 1)  # Set data at (0,1)

    # Set data with slice indexing
    v.set_data([np.array([[1.0, 2.0, 3.0]]), np.array([[4.0, 5.0, 6.0]])],
               slice(0, 2), 1)  # Set data for rows 0-1 at column 1

    Notes:
    -----
    - All numpy arrays stored in the vector must have the same number of columns (fields)
    - Field names must be unique
    - Slicing operations return new Vector instances
    - Field operations are performed in-place
    - Units are stored for each field and can be accessed via the units attribute
    - The name attribute can be used to identify the vector in a larger context
    """

    _token = object()

    def __init__(
        self,
        shape: Tuple[int, ...],
        fields: List[str],
        units: List[str],
        name: str,
        _token: object | None = None,
    ) -> None:
        if _token is not self._token:
            raise RuntimeError("Use Vector.from_shape() or Vector.from_data() to instantiate.")

        self.shape = shape
        self.fields = fields
        self.units = units
        self.name = name
        self._data = nested_list(self.shape, fill=None)

    @classmethod
    def from_shape(
        cls,
        shape: Tuple[int, ...],
        num_fields: Optional[int] = None,
        fields: Optional[List[str]] = None,
        units: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> "Vector":
        """
        Factory method to create a Vector with the specified shape and fields.

        Parameters
        ----------
        shape : Tuple[int, ...]
            The shape of the vector (dimensions)
        num_fields : Optional[int]
            Number of fields in the vector
        name : Optional[str]
            Name of the vector
        fields : Optional[List[str]]
            List of field names
        units : Optional[List[str]]
            List of units for each field

        Returns
        -------
        Vector
            A new Vector instance
        """
        validated_shape = validate_shape(shape)
        ndim = len(validated_shape)

        if fields is not None:
            validated_fields = validate_fields(fields)
            validated_num_fields = len(validated_fields)
            if num_fields is not None and validated_num_fields != num_fields:
                raise ValueError(
                    f"num_fields ({num_fields}) does not match length of fields ({validated_num_fields})"
                )
        elif num_fields is not None:
            validated_num_fields = validate_num_fields(num_fields)
            validated_fields = [f"field_{i}" for i in range(validated_num_fields)]
        else:
            raise ValueError("Must specify either 'fields' or 'num_fields'.")

        validated_units = validate_vector_units(units, validated_num_fields)
        name = name or f"{ndim}d ragged array"

        return cls(
            shape=validated_shape,
            fields=validated_fields,
            units=validated_units,
            name=name,
            _token=cls._token,
        )

    @classmethod
    def from_data(
        cls,
        data: List[Any],
        num_fields: Optional[int] = None,
        fields: Optional[List[str]] = None,
        units: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> "Vector":
        """
        Factory method to create a Vector from a list of
        ragged lists or ragged numpy arrays.

        Parameters
        ----------
        data : List[Any]
            A list of ragged lists containing the vector data.
            Each element should be a numpy array with shape (n, num_fields).
        num_fields : Optional[int]
            Number of fields in the vector. If not provided, it will be inferred from the data.
        fields : Optional[List[str]]
            List of field names
        units : Optional[List[str]]
            List of units for each field
        name : Optional[str]
            Name of the vector

        Returns
        -------
        Vector
            A new Vector instance with the provided data

        Raises
        ------
        ValueError
            If the data structure is invalid or inconsistent
        TypeError
            If the data contains invalid types
        """
        inferred_shape, inferred_num_fields = validate_vector_data_for_inference(data)

        final_num_fields = num_fields or inferred_num_fields
        if num_fields is not None and num_fields != inferred_num_fields:
            raise ValueError(
                f"Provided num_fields ({num_fields}) does not match inferred ({inferred_num_fields})."
            )

        vector = cls.from_shape(
            shape=inferred_shape,
            num_fields=final_num_fields,
            fields=fields,
            units=units,
            name=name,
        )

        # Now fully validate and set the data
        vector.data = data
        return vector

    def get_data(
        self, *indices: Union[int, slice, List[int], np.ndarray[Any, np.dtype[Any]]]
    ) -> Union[NDArray, List[NDArray]]:
        """
        Get data at specified indices.

        Parameters:
        -----------
        *indices : Union[int, slice, List[int], np.ndarray]
            Indices to access. Must match the number of dimensions in the vector.
            Supports fancy indexing with lists or numpy arrays.

        Returns:
        --------
        numpy.ndarray or list
            The data at the specified indices.

        Raises:
        -------
        IndexError
            If indices are out of bounds.
        ValueError
            If the number of indices does not match the vector dimensions.
        """
        if len(indices) != len(self._shape):
            raise ValueError(f"Expected {len(self._shape)} indices, got {len(indices)}")

        # Handle fancy indexing and slicing
        def get_indices(dim_idx: Any, dim_size: int) -> np.ndarray:
            if isinstance(dim_idx, slice):
                start, stop, step = dim_idx.indices(dim_size)
                return np.arange(start, stop, step)
            elif isinstance(dim_idx, (np.ndarray, list)):
                idx = np.asarray(dim_idx)
                if np.any((idx < 0) | (idx >= dim_size)):
                    raise IndexError(f"Index out of bounds for axis with size {dim_size}")
                return idx
            elif isinstance(dim_idx, (int, np.integer)):
                if dim_idx < 0 or dim_idx >= dim_size:
                    raise IndexError(
                        f"Index {dim_idx} out of bounds for axis with size {dim_size}"
                    )
                return np.array([dim_idx])
            return np.arange(dim_size)

        # Get indices for each dimension
        indices_arrays = [get_indices(i, s) for i, s in zip(indices, self._shape)]

        # If all indices are single integers, return a single array
        if all(len(i) == 1 for i in indices_arrays):
            ref = self._data
            for idx in (i[0] for i in indices_arrays):
                ref = ref[idx]
            return ref

        # Create result structure for fancy indexing
        result = []
        for idx in np.ndindex(*[len(i) for i in indices_arrays]):
            src_idx = tuple(ind[i] for ind, i in zip(indices_arrays, idx))
            result.append(self._data[src_idx[0]][src_idx[1]])

        return result

    def set_data(
        self,
        value: Union[NDArray, List[NDArray]],
        *indices: Union[int, slice, List[int], np.ndarray[Any, np.dtype[Any]]],
    ) -> None:
        """
        Set data at specified indices.

        Parameters
        ----------
        value : Union[NDArray, List[NDArray]]
            The numpy array(s) to set at the specified indices. Must have shape (_, num_fields).
            For fancy indexing, can be a list of arrays.
        *indices : Union[int, slice, List[int], np.ndarray]
            Indices to set data at. Must match the number of dimensions in the vector.
            Supports fancy indexing with lists or numpy arrays.

        Raises
        ------
        IndexError
            If indices are out of bounds.
        ValueError
            If the number of indices does not match the vector dimensions,
            or if the value shape doesn't match the expected shape.
        TypeError
            If the value is not a numpy array or list of numpy arrays.
        """
        if len(indices) != len(self._shape):
            raise ValueError(f"Expected {len(self._shape)} indices, got {len(indices)}")

        # Handle fancy indexing and slicing
        def get_indices(dim_idx: Any, dim_size: int) -> np.ndarray:
            if isinstance(dim_idx, slice):
                start, stop, step = dim_idx.indices(dim_size)
                return np.arange(start, stop, step)
            elif isinstance(dim_idx, (np.ndarray, list)):
                idx = np.asarray(dim_idx)
                if np.any((idx < 0) | (idx >= dim_size)):
                    raise IndexError(f"Index out of bounds for axis with size {dim_size}")
                return idx
            elif isinstance(dim_idx, (int, np.integer)):
                if dim_idx < 0 or dim_idx >= dim_size:
                    raise IndexError(
                        f"Index {dim_idx} out of bounds for axis with size {dim_size}"
                    )
                return np.array([dim_idx])
            return np.arange(dim_size)

        # Get indices for each dimension
        indices_arrays = [get_indices(i, s) for i, s in zip(indices, self._shape)]

        # If all indices are single integers, handle as single value
        if all(len(i) == 1 for i in indices_arrays):
            if not isinstance(value, np.ndarray):
                raise TypeError(f"Value must be a numpy array, got {type(value).__name__}")
            if value.ndim != 2 or value.shape[1] != self.num_fields:
                raise ValueError(
                    f"Expected a numpy array with shape (_, {self.num_fields}), got {value.shape}"
                )
            ref = self._data
            for idx in (i[0] for i in indices_arrays[:-1]):
                ref = ref[idx]
            ref[indices_arrays[-1][0]] = value
            return

        # Handle fancy indexing
        if not isinstance(value, list):
            raise TypeError("For fancy indexing, value must be a list of numpy arrays")

        # Validate and set values
        for idx in np.ndindex(*[len(i) for i in indices_arrays]):
            src_idx = tuple(ind[i] for ind, i in zip(indices_arrays, idx))
            if not isinstance(value[idx[0]], np.ndarray):
                raise TypeError(f"Expected numpy array, got {type(value[idx[0]]).__name__}")
            if value[idx[0]].ndim != 2 or value[idx[0]].shape[1] != self.num_fields:
                raise ValueError(
                    f"Expected array with shape (_, {self.num_fields}), got {value[idx[0]].shape}"
                )
            ref = self._data
            for i in src_idx[:-1]:
                ref = ref[i]
            ref[src_idx[-1]] = value[idx[0]]

    @overload
    def __getitem__(self, idx: str) -> "_FieldView": ...
    @overload
    def __getitem__(
        self,
        idx: Union[Tuple[Union[int, slice, List[int]], ...], int, slice, List[int]],
    ) -> Union[NDArray, "Vector"]: ...

    def __getitem__(
        self,
        idx: Union[str, Tuple[Union[int, slice, List[int]], ...], int, slice, List[int]],
    ) -> Union["_FieldView", NDArray, "Vector"]:
        """Get data or a view of the vector at specified indices."""
        if isinstance(idx, str):
            if idx not in self._fields:
                raise KeyError(f"Field '{idx}' not found.")
            return _FieldView(self, idx)

        # Normalize idx to tuple
        normalized: Tuple[Any, ...] = (idx,) if not isinstance(idx, tuple) else idx

        # Convert lists/arrays to ndarray
        idx_converted: Tuple[Union[int, slice, np.ndarray[Any, np.dtype[Any]]], ...] = tuple(
            np.asarray(i) if isinstance(i, (list, np.ndarray)) else i for i in normalized
        )

        # Check if we should return a numpy array (all indices are integers)
        return_np = all(isinstance(i, (int, np.integer)) for i in idx_converted[: len(self.shape)])
        if len(idx_converted) < len(self.shape):
            return_np = False

        if return_np:
            view = self._data
            for i in idx_converted:
                view = view[i]
            return cast(NDArray[Any], view)

        # Handle fancy indexing and slicing
        def get_indices(dim_idx: Any, dim_size: int) -> np.ndarray:
            if isinstance(dim_idx, slice):
                start, stop, step = dim_idx.indices(dim_size)
                return np.arange(start, stop, step)
            elif isinstance(dim_idx, (np.ndarray, list)):
                return np.asarray(dim_idx)
            elif isinstance(dim_idx, (int, np.integer)):
                return np.array([dim_idx])
            return np.arange(dim_size)

        # Get indices for each dimension
        full_idx = list(idx_converted) + [slice(None)] * (len(self.shape) - len(idx_converted))
        indices = [get_indices(i, s) for i, s in zip(full_idx, self.shape)]

        # Create new shape and data
        new_shape = [len(i) for i in indices]
        new_data = [[None] * new_shape[-1] for _ in range(new_shape[0])]

        # Fill the new data structure
        for out_idx in np.ndindex(*new_shape):
            src_idx = tuple(ind[i] for ind, i in zip(indices, out_idx))
            new_data[out_idx[0]][out_idx[1]] = self._data[src_idx[0]][src_idx[1]]

        # Create new Vector
        vector_new = Vector.from_shape(
            shape=tuple(new_shape),
            num_fields=self.num_fields,
            name=self.name + "[view]",
            fields=self.fields,
            units=self.units,
        )
        vector_new._data = new_data
        return vector_new

    def __setitem__(
        self,
        idx: Union[Tuple[Union[int, slice, List[int]], ...], int, slice, List[int], str],
        value: Union[NDArray, List[NDArray]],
    ) -> None:
        """Set data at specified indices."""
        if isinstance(idx, str):
            if idx not in self._fields:
                raise KeyError(f"Field '{idx}' not found.")
            field_view = _FieldView(self, idx)
            field_view.set_flattened(value)
            return

        # Normalize idx to tuple
        normalized: Tuple[Any, ...] = (idx,) if not isinstance(idx, tuple) else idx

        # Convert lists/arrays to ndarray
        idx_converted: Tuple[Union[int, slice, np.ndarray[Any, np.dtype[Any]]], ...] = tuple(
            np.asarray(i) if isinstance(i, (list, np.ndarray)) else i for i in normalized
        )

        # Check if we're doing slice‐ or array‐based (multi‐cell) indexing
        has_fancy = any(
            isinstance(i, slice) or (isinstance(i, np.ndarray) and i.size > 1)
            for i in idx_converted[: len(self.shape)]
        )

        if has_fancy:
            # If user passed a Vector, extract its cell arrays
            if isinstance(value, Vector):

                def _flatten_cells(data):
                    if isinstance(data, np.ndarray):
                        return [data]
                    out = []
                    for sub in data:
                        out.extend(_flatten_cells(sub))
                    return out

                value = _flatten_cells(value._data)

            # For fancy indexing, value should be a list of arrays
            if not isinstance(value, list):
                raise TypeError(
                    "For fancy/slice indexing, value must be a list of numpy arrays or a Vector"
                )

            # Get indices for each dimension
            def get_indices(dim_idx: Any, dim_size: int) -> np.ndarray:
                if isinstance(dim_idx, slice):
                    start, stop, step = dim_idx.indices(dim_size)
                    return np.arange(start, stop, step)
                elif isinstance(dim_idx, (np.ndarray, list)):
                    idx = np.asarray(dim_idx)
                    if np.any((idx < 0) | (idx >= dim_size)):
                        raise IndexError(f"Index out of bounds for axis with size {dim_size}")
                    return idx
                elif isinstance(dim_idx, (int, np.integer)):
                    if dim_idx < 0 or dim_idx >= dim_size:
                        raise IndexError(f"Index out of bounds for axis with size {dim_size}")
                    return np.array([dim_idx])
                return np.arange(dim_size)

            indices_arrays = [get_indices(i, s) for i, s in zip(idx_converted, self._shape)]
            total_indices = np.prod([len(i) for i in indices_arrays])

            if len(value) != total_indices:
                raise ValueError(f"Expected {total_indices} arrays, got {len(value)}")

            # Validate and set values
            for array_idx, idx in enumerate(np.ndindex(*[len(i) for i in indices_arrays])):
                src_idx = tuple(ind[i] for ind, i in zip(indices_arrays, idx))
                if not isinstance(value[array_idx], np.ndarray):
                    raise TypeError(f"Expected numpy array, got {type(value[array_idx]).__name__}")
                if value[array_idx].ndim != 2 or value[array_idx].shape[1] != self.num_fields:
                    raise ValueError(
                        f"Expected array with shape (_, {self.num_fields}), got {value[array_idx].shape}"
                    )
                ref = self._data
                for i in src_idx[:-1]:
                    ref = ref[i]
                ref[src_idx[-1]] = value[array_idx]
        else:
            # For single value assignment
            if not isinstance(value, np.ndarray):
                raise TypeError(f"Value must be a numpy array, got {type(value).__name__}")
            if value.ndim != 2 or value.shape[1] != self.num_fields:
                raise ValueError(
                    f"Expected a numpy array with shape (_, {self.num_fields}), got {value.shape}"
                )
            ref = self._data
            for i in idx_converted[:-1]:
                ref = ref[i]
            ref[idx_converted[-1]] = value

    def add_fields(self, new_fields: Union[str, List[str]]) -> None:
        """
        Add new fields to the vector.

        Parameters
        ----------
        new_fields : Union[str, List[str]]
            Field name(s) to add. Must be unique and not already present.

        Raises
        ------
        ValueError
            If any field name already exists or if there are duplicates
        """
        if isinstance(new_fields, str):
            new_fields = [new_fields]
        else:
            new_fields = list(new_fields)

        if any(name in self._fields for name in new_fields):
            raise ValueError("One or more new field names already exist.")

        if len(set(new_fields)) != len(new_fields):
            raise ValueError("Duplicate field names in input are not allowed.")

        self._fields = list(self._fields) + list(new_fields)
        self._units = list(self._units) + ["none"] * len(new_fields)

        def expand_array(arr: Any) -> Any:
            if isinstance(arr, np.ndarray):
                if arr.shape[1] != self.num_fields - len(new_fields):
                    raise ValueError(
                        f"Expected arrays with {self.num_fields - len(new_fields)} fields, got {arr.shape[1]}"
                    )
                pad = np.zeros((arr.shape[0], len(new_fields)))
                return np.hstack([arr, pad])
            elif isinstance(arr, list):
                return [expand_array(sub) for sub in arr]
            else:
                return arr

        self._data = expand_array(self._data)

    def remove_fields(self, fields_to_remove: Union[str, List[str]]) -> None:
        """
        Remove fields from the vector.

        Parameters
        ----------
        fields_to_remove : Union[str, List[str]]
            Field name(s) to remove. Must exist in the vector.

        Raises
        ------
        ValueError
            If any field doesn't exist
        """
        if isinstance(fields_to_remove, str):
            fields_to_remove = [fields_to_remove]
        else:
            fields_to_remove = list(fields_to_remove)

        field_to_index = {name: i for i, name in enumerate(self._fields)}
        indices_to_remove = []
        for field in fields_to_remove:
            if field not in field_to_index:
                print(f"Warning: field '{field}' not found.")
            else:
                indices_to_remove.append(field_to_index[field])

        if not indices_to_remove:
            return

        indices_to_remove = sorted(set(indices_to_remove))
        keep_indices = [i for i in range(self.num_fields) if i not in indices_to_remove]

        # Update metadata
        self._fields = [self._fields[i] for i in keep_indices]
        self._units = [self._units[i] for i in keep_indices]

        def prune_array(arr: Any) -> Any:
            if isinstance(arr, np.ndarray):
                if arr.shape[1] < max(indices_to_remove) + 1:
                    raise ValueError(
                        f"Cannot remove field index {max(indices_to_remove)} from array with shape {arr.shape}"
                    )
                return arr[:, keep_indices]
            elif isinstance(arr, list):
                return [prune_array(sub) for sub in arr]
            else:
                return arr

        self._data = prune_array(self._data)

    def copy(self) -> "Vector":
        """
        Create a deep copy of the vector.

        Returns
        -------
        Vector
            A new Vector instance with the same data, shape, fields, and units.
        """
        import copy

        vector_copy = Vector.from_shape(
            shape=self.shape,
            name=self.name,
            fields=self.fields,
            units=self.units,
        )
        vector_copy._data = copy.deepcopy(self._data)
        return vector_copy

    def flatten(self) -> NDArray:
        """
        Flatten the vector into a 2D numpy array.

        Returns
        -------
        NDArray
            A 2D numpy array containing all data, with shape (total_rows, num_fields).
        """

        def collect_arrays(data: Any) -> List[NDArray]:
            if isinstance(data, np.ndarray):
                return [data]
            elif isinstance(data, list):
                arrays = []
                for item in data:
                    arrays.extend(collect_arrays(item))
                return arrays
            else:
                return []

        arrays = collect_arrays(self._data)
        if not arrays:
            return np.empty((0, self.num_fields))
        return np.vstack(arrays)

    def __repr__(self) -> str:
        description = [
            f"quantem.Vector, shape={self._shape}, name={self._name}",
            f"  fields = {self._fields}",
            f"  units: {self._units}",
        ]
        return "\n".join(description)

    def __str__(self) -> str:
        description = [
            f"quantem.Vector, shape={self._shape}, name={self._name}",
            f"  fields = {self._fields}",
            f"  units: {self._units}",
        ]
        return "\n".join(description)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the vector.

        Returns
        -------
        Tuple[int, ...]
            The dimensions of the vector.
        """
        return self._shape

    @shape.setter
    def shape(self, value: Tuple[int, ...]) -> None:
        """
        Set the shape of the vector.

        Parameters
        ----------
        value : Tuple[int, ...]
            The new shape. All dimensions must be positive.

        Raises
        ------
        ValueError
            If any dimension is not positive.
        TypeError
            If value is not a tuple or contains non-integer values.
        """
        self._shape = validate_shape(value)

    @property
    def num_fields(self) -> int:
        """
        Get the number of fields in the vector.

        Returns
        -------
        int
            The number of fields.
        """
        return len(self._fields)

    @property
    def name(self) -> str:
        """
        Get the name of the vector.

        Returns
        -------
        str
            The name of the vector
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """
        Set the name of the vector.

        Parameters
        ----------
        value : str
            The new name of the vector
        """
        self._name = str(value)

    @property
    def fields(self) -> List[str]:
        """
        Get the field names of the vector.

        Returns
        -------
        List[str]
            The list of field names.
        """
        return self._fields

    @fields.setter
    def fields(self, value: List[str]) -> None:
        """
        Set the field names of the vector.

        Parameters
        ----------
        value : List[str]
            The new field names. Must match num_fields and be unique.

        Raises
        ------
        ValueError
            If length doesn't match num_fields or if there are duplicates.
        TypeError
            If value is not a list or contains non-string values.
        """
        self._fields = validate_fields(value)

    @property
    def units(self) -> List[str]:
        """
        Get the units of the vector's fields.

        Returns
        -------
        List[str]
            The list of units, one per field.
        """
        return self._units

    @units.setter
    def units(self, value: List[str]) -> None:
        """
        Set the units of the vector's fields.

        Parameters
        ----------
        value : List[str]
            The new units. Must match num_fields.

        Raises
        ------
        ValueError
            If length doesn't match num_fields.
        TypeError
            If value is not a list or contains non-string values.
        """
        self._units = validate_vector_units(value, self.num_fields)

    @property
    def data(self) -> List[Any]:
        """
        Get the raw data of the vector.

        Returns
        -------
        List[Any]
            The nested list structure containing the vector's data.
        """
        return self._data

    @data.setter
    def data(self, value: List[Any]) -> None:
        """
        Set the raw data of the vector.

        Parameters
        ----------
        value : List[Any]
            The new data structure. Must match the vector's shape and num_fields.

        Raises
        ------
        ValueError
            If the data structure doesn't match shape or num_fields.
        TypeError
            If value is not a list or contains invalid data types.
        """
        self._data = validate_vector_data(value, self.shape, self.num_fields)


# Helper function for nesting lists
def nested_list(shape: Tuple[int, ...], fill: Any = None) -> Any:
    if len(shape) == 0:
        return fill
    return [nested_list(shape[1:], fill) for _ in range(shape[0])]


# Helper class for numerical field operations
class _FieldView:
    def __init__(self, vector: Vector, field_name: str) -> None:
        self.vector = vector
        self.field_name = field_name
        self.field_index = vector._fields.index(field_name)

    def _apply_op(self, op: Any) -> None:
        def apply(arr: Any) -> None:
            if isinstance(arr, np.ndarray):
                arr[:, self.field_index] = op(arr[:, self.field_index])
            elif isinstance(arr, list):
                for sub in arr:
                    apply(sub)

        apply(self.vector._data)

    def __iadd__(self, other: Union[float, int, np.ndarray]) -> "_FieldView":
        """Handle in-place addition (+=)."""
        self._apply_op(lambda x: x + other)
        return self

    def __isub__(self, other: Union[float, int, np.ndarray]) -> "_FieldView":
        """Handle in-place subtraction (-=)."""
        self._apply_op(lambda x: x - other)
        return self

    def __imul__(self, other: Union[float, int, np.ndarray]) -> "_FieldView":
        """Handle in-place multiplication (*=)."""
        self._apply_op(lambda x: x * other)
        return self

    def __itruediv__(self, other: Union[float, int, np.ndarray]) -> "_FieldView":
        """Handle in-place division (/=)."""
        self._apply_op(lambda x: x / other)
        return self

    def __ifloordiv__(self, other: Union[float, int, np.ndarray]) -> "_FieldView":
        """Handle in-place floor division (//=)."""
        self._apply_op(lambda x: x // other)
        return self

    def __imod__(self, other: Union[float, int, np.ndarray]) -> "_FieldView":
        """Handle in-place modulo (%=)."""
        self._apply_op(lambda x: x % other)
        return self

    def __ipow__(self, other: Union[float, int, np.ndarray]) -> "_FieldView":
        """Handle in-place power (**=)."""
        self._apply_op(lambda x: x**other)
        return self

    def flatten(self) -> NDArray:
        def collect(arr: Any) -> List[NDArray]:
            if isinstance(arr, np.ndarray):
                return [arr[:, self.field_index]]
            elif isinstance(arr, list):
                result = []
                for sub in arr:
                    result.extend(collect(sub))
                return result
            else:
                return []

        arrays = collect(self.vector._data)
        if not arrays:
            return np.empty((0,), dtype=float)
        return np.concatenate(arrays, axis=0)

    def set_flattened(self, values: ArrayLike) -> None:
        """
        Set the field values across the entire Vector from a 1D flattened array.
        """

        def fill(arr: Any, values: NDArray, cursor: int) -> int:
            if isinstance(arr, np.ndarray):
                n = arr.shape[0]
                arr[:, self.field_index] = values[cursor : cursor + n]
                return cursor + n
            elif isinstance(arr, list):
                for sub in arr:
                    cursor = fill(sub, values, cursor)
                return cursor
            return cursor

        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError("Input to set_flattened must be a 1D array.")

        expected = self.flatten().shape[0]
        if values.shape[0] != expected:
            raise ValueError(f"Expected {expected} values, got {values.shape[0]}")

        fill(self.vector._data, values, cursor=0)

    def __getitem__(
        self, idx: Union[Tuple[Union[int, slice], ...], int, slice]
    ) -> Union[NDArray, "_FieldView"]:
        # Optionally allow v['field0'][0, 1] to get subregion, or v['field0'][...] slice
        sub = self.vector[idx]
        if isinstance(sub, Vector):
            return sub[self.field_name]
        elif isinstance(sub, np.ndarray):
            return sub[:, self.field_index]
        return cast(NDArray, None)

    def __array__(self) -> np.ndarray:
        """Convert to numpy array when needed."""
        return self.flatten()
