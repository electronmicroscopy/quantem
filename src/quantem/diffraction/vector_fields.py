"""Small accessors for the ragged peak :class:`~quantem.core.datastructures.Vector`.

Peak data is stored as a Vector of per-scan-position tables, one row per peak and
one column per named field. Pulling a single field out is a two-step dance that
reads badly inline, so it lives here -- shared by the analysis and plotting
modules without either importing the other.
"""

from __future__ import annotations

__all__ = ["vector_field_cell", "vector_field_flat"]


def vector_field_flat(vector, field):
    """Return one current-Vector field as a one-dimensional NumPy array."""
    return vector.select_fields(field).flatten()[:, 0]


def vector_field_cell(vector, field, row, col):
    """Return one current-Vector field from a scan cell as a 1D array."""
    return vector.select_fields(field)[row, col].array[:, 0]
