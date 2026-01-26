"""
Array utilities for handling NumPy, CuPy, and PyTorch arrays uniformly.

This module provides utilities to convert arrays from different backends
into NumPy arrays for widget processing.
"""

from typing import Any, Literal
import numpy as np


ArrayBackend = Literal["numpy", "cupy", "torch", "unknown"]


def get_array_backend(data: Any) -> ArrayBackend:
    """
    Detect the array backend of the input data.

    Parameters
    ----------
    data : array-like
        Input array (NumPy, CuPy, PyTorch, or other).

    Returns
    -------
    str
        One of: "numpy", "cupy", "torch", "unknown"
    """
    # Check PyTorch first (has both .numpy and .detach methods)
    if hasattr(data, "detach") and hasattr(data, "numpy"):
        return "torch"
    # Check CuPy (has .get() or __cuda_array_interface__)
    if hasattr(data, "__cuda_array_interface__"):
        return "cupy"
    if hasattr(data, "get") and hasattr(data, "__array__"):
        # CuPy arrays have .get() to transfer to CPU
        type_name = type(data).__module__
        if "cupy" in type_name:
            return "cupy"
    # Check NumPy
    if isinstance(data, np.ndarray):
        return "numpy"
    return "unknown"


def to_numpy(data: Any, dtype: np.dtype | None = None) -> np.ndarray:
    """
    Convert any array-like (NumPy, CuPy, PyTorch) to a NumPy array.

    Parameters
    ----------
    data : array-like
        Input array from any supported backend.
    dtype : np.dtype, optional
        Target dtype for the output array. If None, preserves original dtype.

    Returns
    -------
    np.ndarray
        NumPy array with the same data.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget.array_utils import to_numpy
    >>>
    >>> # NumPy passthrough
    >>> arr = np.random.rand(10, 10)
    >>> result = to_numpy(arr)
    >>>
    >>> # CuPy conversion (if available)
    >>> import cupy as cp
    >>> gpu_arr = cp.random.rand(10, 10)
    >>> cpu_arr = to_numpy(gpu_arr)
    >>>
    >>> # PyTorch conversion (if available)
    >>> import torch
    >>> tensor = torch.rand(10, 10)
    >>> arr = to_numpy(tensor)
    """
    backend = get_array_backend(data)

    if backend == "torch":
        # PyTorch tensor: detach from graph, move to CPU, convert to numpy
        result = data.detach().cpu().numpy()

    elif backend == "cupy":
        # CuPy array: use .get() to transfer to CPU
        if hasattr(data, "get"):
            result = data.get()
        else:
            # Fallback for __cuda_array_interface__
            import cupy as cp

            result = cp.asnumpy(data)

    elif backend == "numpy":
        # NumPy array: passthrough (may copy if dtype changes)
        result = data

    else:
        # Unknown backend: try np.asarray as fallback
        result = np.asarray(data)

    # Apply dtype conversion if specified
    if dtype is not None:
        result = np.asarray(result, dtype=dtype)

    return result
