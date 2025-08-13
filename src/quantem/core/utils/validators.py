from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from quantem.core import config
from quantem.core.utils import array_funcs as arr

if TYPE_CHECKING:
    import cupy as cp
    import torch
else:
    if config.get("has_torch"):
        import torch
    if config.get("has_cupy"):
        import cupy as cp


# --- Dataset Validation Functions ---
def ensure_valid_array(
    array: "np.ndarray | cp.ndarray", dtype: DTypeLike = None, ndim: int | None = None
) -> Union[np.ndarray, "cp.ndarray"]:
    """
    Ensure input is a numpy array or cupy array (if available), converting if necessary.

    Parameters
    ----------
    array : Union[NDArray, Any]
        The input array to validate and convert
    dtype : DTypeLike, optional
        The desired data type for the array
    ndim : int, optional
        The expected number of dimensions for the array

    Returns
    -------
    Union[NDArray, Any]
        The validated array with the specified dtype and ndim

    Raises
    ------
    ValueError
        If the array is not at least 1D, doesn't contain numeric values,
        or has a different number of dimensions than expected
    TypeError
        If the input could not be converted to a NumPy array
    """
    is_cupy = False
    if config.get("has_cupy"):
        if isinstance(array, cp.ndarray):
            is_cupy = True
    if isinstance(array, np.ndarray) or is_cupy:
        if dtype is not None:
            validated_array = array.astype(dtype)  # copies the array
        else:
            validated_array = array
    else:  # default to numpy
        try:
            validated_array = np.array(array, dtype=dtype)
            if validated_array.ndim < 1:
                raise ValueError("Array must be at least 1D")
            elif not np.issubdtype(validated_array.dtype, np.number):
                raise ValueError("Array must contain numeric values")
        except Exception as e:
            raise TypeError(f"Input could not be converted to a NumPy array: {e}")
    if ndim is not None:
        val_ndim = validated_array.ndim
        if val_ndim < ndim:
            for _ in range(ndim - val_ndim):
                validated_array = np.expand_dims(validated_array, axis=0)
            warn(f"Array ndim {val_ndim} is being padded to {ndim}")
        elif val_ndim > ndim:
            raise ValueError(f"Array ndim {val_ndim} > expected ndim {ndim}")
    return validated_array


def validate_ndinfo(
    value: Union[np.ndarray, "cp.ndarray", tuple, list, float, int],
    ndim: int,
    name: str,
    dtype=None,
) -> "np.ndarray | cp.ndarray":
    """
    Validate and convert origin/sampling to a 1D numpy array of type dtype and correct length.

    Parameters
    ----------
    value : Union[NDArray, tuple, list, float, int]
        The value to validate and convert
    ndim : int
        The expected number of dimensions
    name : str
        The name of the parameter being validated (for error messages)
    dtype : type, optional
        The desired data type for the array

    Returns
    -------
    NDArray
        A 1D numpy array with the specified dtype and length

    Raises
    ------
    ValueError
        If the array doesn't contain numeric values or has the wrong length
    TypeError
        If the value is not a numpy array, tuple, list, or scalar,
        or if it could not be converted to a 1D numeric NumPy array
    """
    if np.isscalar(value):
        arr = np.full(ndim, value, dtype=dtype)
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"{name} must contain numeric values")
        return arr
    elif not isinstance(value, (np.ndarray, tuple, list)):
        raise TypeError(f"{name} must be a numpy array, tuple, list, or scalar, got {type(value)}")

    try:
        arr = np.array(value, dtype=dtype).flatten()
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert {name} to a 1D numeric NumPy array: {e}")

    if len(arr) != ndim:
        raise ValueError(f"Length of {name} ({len(arr)}) must match data ndim ({ndim})")

    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values")

    return arr


def validate_units(value: Union[List[str], tuple, list, str], ndim: int) -> List[str]:
    """
    Validate and convert units to a list of strings of correct length.

    Parameters
    ----------
    value : Union[List[str], tuple, list, str]
        The units to validate and convert
    ndim : int
        The expected number of dimensions

    Returns
    -------
    List[str]
        A list of strings representing the units

    Raises
    ------
    ValueError
        If the length of units doesn't match the expected number of dimensions
    TypeError
        If units is not a list, tuple, or string
    """
    if isinstance(value, str):
        return [value] * ndim
    elif not isinstance(value, (list, tuple)):
        raise TypeError(f"Units must be a list, tuple, or string, got {type(value)}")
    elif len(value) != ndim:
        raise ValueError(f"Length of units ({len(value)}) must match data ndim ({ndim})")

    return [str(unit) for unit in value]


# --- Vector Validation Functions ---
def validate_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Validate and convert shape to a tuple of integers.

    Parameters
    ----------
    shape : Tuple[int, ...]
        The shape to validate

    Returns
    -------
    Tuple[int, ...]
        The validated shape

    Raises
    ------
    ValueError
        If shape contains non-positive integers
    TypeError
        If shape is not a tuple or contains non-integer values
    """
    if not isinstance(shape, tuple):
        raise TypeError(f"Shape must be a tuple, got {type(shape)}")

    validated = []
    for dim in shape:
        if not isinstance(dim, int):
            raise TypeError(f"Shape dimensions must be integers, got {type(dim)}")
        if dim <= 0:
            raise ValueError(f"Shape dimensions must be positive, got {dim}")
        validated.append(dim)

    return tuple(validated)


def validate_fields(fields: List[str]) -> List[str]:
    if not isinstance(fields, (list, tuple)):
        raise TypeError(f"fields must be a list or tuple, got {type(fields)}")
    if len(set(fields)) != len(fields):
        raise ValueError("Duplicate field names are not allowed.")
    return [str(field) for field in fields]


def validate_num_fields(num_fields: int, fields: Optional[List[str]] = None) -> int:
    """
    Validate number of fields.

    Parameters
    ----------
    num_fields : int
        The number of fields
    fields : Optional[List[str]]
        List of field names

    Returns
    -------
    int
        The validated number of fields

    Raises
    ------
    ValueError
        If num_fields is not positive or doesn't match fields length
    """
    if not isinstance(num_fields, int):
        raise TypeError(f"num_fields must be an integer, got {type(num_fields)}")
    if num_fields <= 0:
        raise ValueError(f"num_fields must be positive, got {num_fields}")
    if fields is not None and len(fields) != num_fields:
        raise ValueError(
            f"num_fields ({num_fields}) does not match length of fields ({len(fields)})"
        )
    return num_fields


def validate_vector_units(units: Optional[List[str]], num_fields: int) -> List[str]:
    if units is None:
        return ["none"] * num_fields
    if not isinstance(units, (list, tuple)):
        raise TypeError(f"units must be a list or tuple, got {type(units)}")
    if len(units) != num_fields:
        raise ValueError(f"Length of units ({len(units)}) must match num_fields ({num_fields})")
    return [str(unit) for unit in units]


def validate_vector_data_for_inference(data: List[Any]) -> Tuple[Tuple[int, ...], int]:
    if not isinstance(data, list):
        raise TypeError("Data must be a list.")
    if len(data) == 0:
        raise ValueError("Data list cannot be empty.")

    first_item = data[0]
    if isinstance(first_item, list):
        first_item = np.array(first_item)
    if not isinstance(first_item, np.ndarray):
        raise TypeError("Data elements must be numpy arrays or convertible.")

    inferred_num_fields = first_item.shape[1]

    for item in data:
        if isinstance(item, list):
            item = np.array(item)
        if not isinstance(item, np.ndarray) or item.shape[1] != inferred_num_fields:
            raise ValueError("All data arrays must have same number of fields.")

    shape = (len(data),)
    return shape, inferred_num_fields


def validate_vector_data(data: List[Any], shape: Tuple[int, ...], num_fields: int) -> List[Any]:
    """
    Validate that the data structure matches the expected shape and number of fields.

    Parameters
    ----------
    data : List[Any]
        The nested list structure containing the vector's data
    shape : Tuple[int, ...]
        The expected shape of the vector
    num_fields : int
        The expected number of fields

    Returns
    -------
    List[Any]
        The validated data structure

    Raises
    ------
    ValueError
        If the data structure doesn't match the expected shape,
        or if any array doesn't have the correct number of fields
    TypeError
        If data is not a list or contains invalid data types
    """
    # Check if data is a list
    if not isinstance(data, list):
        raise TypeError("Data must be a list")

    # Check if the length of data matches the expected shape
    if len(data) != shape[0]:
        raise ValueError(f"Expected {shape[0]} items in data, got {len(data)}")

    validated_data = []

    for idx, item in enumerate(data):
        # Convert item to numpy array if it's a list
        if isinstance(item, list):
            item = np.array(item)

        # Check if the item is a numpy array
        if not isinstance(item, np.ndarray):
            raise TypeError(
                f"Data element at index {idx} must be a numpy array or convertible to one"
            )

        # Check if the number of fields matches
        if item.shape[1] != num_fields:
            raise ValueError(
                f"Data element at index {idx} must have {num_fields} fields, got {item.shape[1]}"
            )

        validated_data.append(item)

    return validated_data


# --- Miscellaneous Validation Functions ---
### for now just adding all the validator cases, can combine later as necessary
### currently doing a mix of converting and validating, probably should make that explicit
def validate_gt(
    value: float | int, cutoff: float | int, name: str, geq: bool = False
) -> float | int:
    if geq:  # greater than or equal to
        if value < cutoff:
            raise ValueError(f"{name} must be greater than or equal to {cutoff}, got {value}")
    elif value <= cutoff:
        raise ValueError(f"{name} must be greater than {cutoff}, got {value}")
    return value


def validate_int(value: int | float, name: str) -> int:
    try:
        return int(round(value))
    except (ValueError, TypeError):
        raise TypeError(f"{name} must be an integer, got {type(value)}")


def validate_float(value: float, name: str) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        raise TypeError(f"{name} must be a float, got {type(value)}")


def validate_arraylike(
    value: "ArrayLike | torch.Tensor", name: str
) -> "np.ndarray | cp.ndarray | torch.Tensor":
    if isinstance(value, np.ndarray):
        return value
    if config.get("has_cupy"):
        if isinstance(value, cp.ndarray):
            return value
    if config.get("has_torch"):
        if isinstance(value, torch.Tensor):
            return value
    try:
        return np.array(value)
    except Exception as e:
        raise TypeError(f"{name} must be array or tensorlike, got type {type(value)}: {e}")


def validate_array_or_tensor(
    value: "ArrayLike | torch.Tensor",
    name: str,
    dtype: "DTypeLike | torch.dtype | str | None" = None,
    ndim: int | None = None,
    shape: np.ndarray | tuple | None = None,
    expand_dims: bool = False,
    allow_tensor: bool = True,
) -> "np.ndarray | cp.ndarray":
    value = validate_arraylike(value, name)
    if config.get("has_torch"):
        if isinstance(value, torch.Tensor):
            if not allow_tensor:
                warn(
                    f"{name} is a torch tensor on device {value.device}. Converting to CPU numpy array."
                )
                value = value.cpu().detach().numpy()
    if dtype is not None:
        val_dtype_str = canonical_dtype_str(value.dtype)
        req_dtype_str = canonical_dtype_str(dtype)
        if val_dtype_str != req_dtype_str:
            value = arr.as_type(value, req_dtype_str)
    if ndim is not None and value.ndim != ndim:
        if expand_dims and ndim > value.ndim:
            for _ in range(ndim - value.ndim):
                value = arr.expand_dims(value, axis=0)
        else:
            raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}")
    if shape is not None and not np.array_equal(value.shape, shape):
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    return value


def validate_xplike(value: ArrayLike, name: str) -> "np.ndarray|cp.ndarray":
    """for np.ndarray and cp.ndarray, returns the array as is. For a torch.Tensor or any other type
    (including list, tuple, etc.) tries to convert and return a np.ndarray."""
    if isinstance(value, np.ndarray):
        return value
    if config.get("has_cupy"):
        if isinstance(value, cp.ndarray):
            return value
    if config.get("has_torch"):
        if isinstance(value, torch.Tensor):
            return value.cpu().detach().numpy()
    try:
        return np.array(value)
    except Exception as e:
        raise TypeError(f"{name} must be array like, got type {type(value)}: {e}")


def canonical_dtype_str(dtype: "DTypeLike | torch.dtype"):
    """
    Converts a dtype (NumPy, CuPy, torch, Python, or string) to a canonical string for comparison.
    """
    if isinstance(dtype, str):
        return dtype.lower().replace("torch.", "").replace("numpy.", "").replace("cp.", "")
    if isinstance(dtype, type):
        return dtype.__name__.lower()
    elif hasattr(dtype, "name"):
        return str(getattr(dtype, "name")).lower()
    elif config.get("has_torch"):
        if isinstance(dtype, torch.dtype):
            return str(dtype).split(".")[-1].lower()
    return str(dtype).lower()


def validate_array(
    value: "ArrayLike | torch.Tensor",
    name: str,
    dtype: "DTypeLike | torch.dtype | str | None" = None,
    ndim: int | None = None,
    shape: np.ndarray | tuple | None = None,
    expand_dims: bool = False,
) -> "np.ndarray | cp.ndarray":
    value = validate_array_or_tensor(
        value, name, dtype, ndim, shape, expand_dims, allow_tensor=False
    )
    return value


def validate_tensor(
    value: "ArrayLike | torch.Tensor",
    name: str,
    dtype: "DTypeLike | torch.dtype | str | None" = None,
    ndim: int | None = None,
    shape: np.ndarray | tuple | None = None,
    expand_dims: bool = False,
) -> "torch.Tensor":
    if not config.get("has_torch"):
        raise ImportError("Torch is not available, cannot validate torch tensors.")
    value = validate_array_or_tensor(
        value, name, dtype, ndim, shape, expand_dims, allow_tensor=True
    )
    if isinstance(value, torch.Tensor):
        return value
    else:
        return torch.tensor(value)


def validate_np_len(value: ArrayLike, length: int, name: str = "") -> np.ndarray:
    if isinstance(value, np.ndarray):
        pass
    elif isinstance(value, (float, int)):
        value = np.array([value] * length)
    elif isinstance(value, (list, tuple)):
        value = np.array(value)
    else:
        raise TypeError(f"{name} must be a numpy array or convertible to one")
    if value.shape[0] != length:
        raise ValueError(f"{name} must have {length} elements, got {value.shape[0]}")
    return value


def validate_arr_gt(value: "np.ndarray", cutoff: float | int, name: str) -> "np.ndarray":
    if np.any(value <= cutoff):
        raise ValueError(f"All elements of {name} must be greater than {cutoff}")
    return value


def validate_tens_shape(
    value: "np.ndarray | cp.ndarray | torch.Tensor",
    shape: np.ndarray | tuple,
    name: str,
) -> "np.ndarray | cp.ndarray | torch.Tensor":
    if not isinstance(value, (np.ndarray, cp.ndarray, torch.Tensor)):
        raise TypeError(f"{name} must be a numpy array, cupy array, or torch tensor")
    if not np.array_equal(value.shape, shape):
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    return value


def validate_dict_keys(dict: dict, valid_keys: list):
    keys = list(dict.keys())
    invalid_keys = list(set(keys) - set(valid_keys))
    if invalid_keys:
        raise ValueError(f"Invalid keys: {invalid_keys}")


# def validate_dtype(value: Any, dtype: DTypeLike, name: str) -> DTypeLike:
#     try:
#         return
