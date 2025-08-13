import numbers
from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from quantem.core import config

if config.get("has_cupy"):
    pass  # type: ignore
else:
    pass

from quantem.core.datastructures import Dataset2d, Dataset3d
from quantem.core.utils.validators import ensure_valid_array

"""
Compound validators which involve Dataset and Vector objects. In a separate
.py file than pure python/numpy validators to avoid circular imports.
"""


def validate_list_of_dataset2d(
    images: Union[List[Dataset2d], List[NDArray], Dataset3d, NDArray],
) -> List[Dataset2d]:
    """
    Validate that the passes images is a list of Dataset2d objects.

    Parameters
    ----------
    images : Union[List[Dataset2d], List[NDArray], Dataset3d, NDArray]
        The images list to validate

    Returns
    -------
    List[Dataset2d]
        The validated data structure

    Raises
    ------
    ValueError
        If the data structure doesn't match the expected shape,
        or if any array doesn't have the correct number of fields
    TypeError
        If data is not a list or contains invalid data types
    """

    if isinstance(images, Dataset3d):
        image_list = images.to_dataset2d()
    elif isinstance(images, np.ndarray) and images.ndim == 3:
        image_list = [Dataset2d.from_array(ensure_valid_array(im, ndim=2)) for im in images]
    elif isinstance(images, list):
        if all(isinstance(im, Dataset2d) for im in images):
            image_list = images
        elif all(isinstance(im, np.ndarray) and im.ndim == 2 for im in images):
            image_list = [Dataset2d.from_array(ensure_valid_array(im, ndim=2)) for im in images]
        else:
            raise TypeError(
                "If passing a list, all elements must be either 2D numpy arrays or Dataset2d instances."
            )
    else:
        raise TypeError(
            "images must be a Dataset3d, a 3D ndarray, or a list of 2D arrays or Dataset2d instances."
        )

    return image_list


def validate_pad_value(pad_value: Union[float, str, List[float]], images: List[Dataset2d]):
    """ """

    if isinstance(pad_value, str):
        if pad_value == "median":
            pad_value = [np.median(im.array) for im in images]
        elif pad_value == "mean":
            pad_value = [np.mean(im.array) for im in images]
        elif pad_value == "min":
            pad_value = [np.min(im.array) for im in images]
        elif pad_value == "max":
            pad_value = [np.max(im.array) for im in images]
    elif isinstance(pad_value, numbers.Number):
        if float(pad_value) < 0.0:
            raise ValueError(f"pad_value of {pad_value} is < 0.0")
        if float(pad_value) > 1.0:
            raise ValueError(f"pad_value of {pad_value} is > 1.0")
        pad_value = [np.quantile(im.array, pad_value) for im in images]
    elif isinstance(pad_value, list) and all(isinstance(v, (int, float)) for v in pad_value):
        if len(pad_value) != len(images):
            raise ValueError("pad_value list length must match number of images.")
        pad_value = pad_value
    else:
        raise TypeError(
            f"pad_value must be a 0.0 < float < 1.0, or one of ['median', 'mean', 'min', 'max'], got {type(pad_value)}"
        )
    return pad_value
