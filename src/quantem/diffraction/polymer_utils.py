"""Small numerical helpers specific to polymer diffraction inference."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import uniform_filter


def parse_reciprocal_units(unit_string: str) -> tuple[str, float]:
    """Return a canonical reciprocal unit and its multiplier to inverse angstroms."""

    normalized = (
        str(unit_string)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("angstrom", "å")
        .replace("ang", "å")
    )
    nanometer = {"1/nm", "/nm", "nm^-1", "nm-1", "nm⁻¹", "inv_nm", "per_nm"}
    angstrom = {
        "1/a",
        "/a",
        "a^-1",
        "a-1",
        "a⁻¹",
        "1/å",
        "/å",
        "å^-1",
        "å-1",
        "å⁻¹",
        "inv_a",
        "inv_å",
        "per_a",
        "per_å",
    }
    if normalized in nanometer:
        return "1/nm", 0.1
    if normalized in angstrom:
        return "1/A", 1.0
    warnings.warn(
        f"Unrecognized reciprocal unit {unit_string!r}; assuming 1/Å.",
        UserWarning,
        stacklevel=2,
    )
    return "unknown", 1.0


def sample_average_from_image(
    image: np.ndarray,
    coordinates: np.ndarray,
    radius_dim1: int = 2,
    radius_dim2: int = 2,
) -> np.ndarray:
    """Sample local means from a polar image, wrapping only its angular axis."""

    image = np.asarray(image)
    coordinates = np.asarray(coordinates)
    if image.ndim != 2 or coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("image must be 2D and coordinates must have shape (n, 2)")
    height, width = image.shape
    padded = np.pad(image, ((radius_dim1, radius_dim1), (radius_dim2, radius_dim2)), mode="wrap")
    if radius_dim1:
        padded[:radius_dim1] = 0
        padded[-radius_dim1:] = 0
    means = uniform_filter(
        padded,
        size=(2 * radius_dim1 + 1, 2 * radius_dim2 + 1),
        mode="constant",
        cval=0,
    )
    rows = np.clip(coordinates[:, 0].astype(int) + radius_dim1, 0, height + 2 * radius_dim1 - 1)
    cols = coordinates[:, 1].astype(int) % width + radius_dim2
    return means[rows, cols]


__all__ = ["parse_reciprocal_units", "sample_average_from_image"]
