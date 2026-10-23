"""Scan-region masks for a 4D-STEM analysis.

A region is a circle, ellipse, square or rectangle, rotated ``angle`` degrees
anticlockwise, positioned in scan coordinates (x is the scan column, horizontal;
y is the scan row, vertical). A mask may also be edited pixel by pixel after it is
built from a shape; the state file keeps both the mask and the shape it started from.

The interactive editor (draw a shape, then paint or erase pixels) lives in
``quantem.widget``; this module holds the geometry and file format it uses, so the
outline drawn there and the mask selected here come from the same code.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import numpy as np

GEOMETRIES = ("circle", "ellipse", "square", "rectangle")
# 1: circle only. 2: adds geometry and size_x/size_y. 3: adds angle.
# 4: adds painted, and the shape fields become optional (a hand-painted mask has none).
SCHEMA_VERSION = 4


@dataclasses.dataclass(frozen=True)
class ScanMaskShape:
    """One scan-region shape. Sizes are radii for round shapes, half-sizes otherwise.

    ``size_y`` is tied to ``size_x`` for a circle and a square. ``angle`` wraps at
    180 degrees, and a circle ignores it.
    """

    x: int
    y: int
    geometry: str = "circle"
    size_x: int = 1
    size_y: int | None = None
    angle: float = 0.0

    def __post_init__(self):
        geometry = str(self.geometry).lower()
        if geometry not in GEOMETRIES:
            raise ValueError(
                f"Unknown mask geometry {self.geometry!r}; choose one of {', '.join(GEOMETRIES)}."
            )
        size_x = int(self.size_x)
        size_y = size_x if geometry in {"circle", "square"} or self.size_y is None else self.size_y
        angle = 0.0 if geometry == "circle" else float(self.angle) % 180.0
        for name, value in (
            ("x", int(self.x)),
            ("y", int(self.y)),
            ("geometry", geometry),
            ("size_x", size_x),
            ("size_y", int(size_y)),
            ("angle", angle),
        ):
            object.__setattr__(self, name, value)
        if self.size_x < 1 or self.size_y < 1:
            raise ValueError("Mask half-sizes must be at least one scan pixel.")

    def validate(self, scan_shape):
        rows, columns = scan_shape
        if not 0 <= self.y < rows or not 0 <= self.x < columns:
            raise ValueError(
                f"Mask center (x={self.x}, y={self.y}) is outside scan shape {tuple(scan_shape)}."
            )
        maximum = int(np.ceil(np.hypot(rows - 1, columns - 1))) + 1
        if self.size_x > maximum or self.size_y > maximum:
            raise ValueError(
                f"Mask half-size ({self.size_x}, {self.size_y}) exceeds the supported maximum "
                f"{maximum} for scan shape {tuple(scan_shape)}."
            )

    def mask(self, scan_shape):
        """Boolean ``scan_shape`` mask, True inside the shape.

        The shape is rotated by rotating the sampling grid the other way, so the
        axis-aligned inside-tests below are unchanged.
        """
        self.validate(scan_shape)
        yy, xx = np.ogrid[: scan_shape[0], : scan_shape[1]]
        dx, dy = self._to_shape_frame(xx - self.x, yy - self.y)
        if self.geometry == "circle":
            # Integer test, so boundary points such as (3, 4) at radius 5 stay inside.
            return dx**2 + dy**2 <= self.size_x**2
        if self.geometry == "ellipse":
            return (dx / self.size_x) ** 2 + (dy / self.size_y) ** 2 <= 1
        return (np.abs(dx) <= self.size_x) & (np.abs(dy) <= self.size_y)

    def outline(self, num_points=256):
        """Closed boundary as ``(N, 2)`` ``(x, y)`` vertices in scan pixels.

        Rectangles are drawn a half pixel outside their inclusive edges, so the
        outline encloses the pixel centers the mask selects.
        """
        if self.geometry in {"circle", "ellipse"}:
            t = np.linspace(0.0, 2.0 * np.pi, num_points)
            u, v = self.size_x * np.cos(t), self.size_y * np.sin(t)
        else:
            hx, hy = self.size_x + 0.5, self.size_y + 0.5
            u = np.array([-hx, hx, hx, -hx, -hx])
            v = np.array([-hy, -hy, hy, hy, -hy])
        radians = np.deg2rad(self.angle)
        cos, sin = np.cos(radians), np.sin(radians)
        return np.column_stack([self.x + u * cos - v * sin, self.y + u * sin + v * cos])

    def _to_shape_frame(self, dx, dy):
        if not self.angle:
            return dx, dy
        radians = np.deg2rad(self.angle)
        cos, sin = np.cos(radians), np.sin(radians)
        return dx * cos + dy * sin, -dx * sin + dy * cos


def save_scan_mask(path, mask, *, shape=None, sampling=None, units=None):
    """Write ``mask`` (and the ``shape`` it started from, if any) to an ``.npz`` file.

    ``painted`` records whether the mask was edited after it was built from
    ``shape``. The file is written to a temporary name and renamed into place, so
    an interrupted save never leaves a truncated file. Returns the path.
    """
    path = Path(path)
    mask = np.asarray(mask, dtype=bool)
    fields = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "mask": mask,
        "scan_shape": np.asarray(mask.shape, dtype=np.int64),
        "painted": np.asarray(shape is None or not np.array_equal(mask, shape.mask(mask.shape))),
    }
    if shape is not None:
        fields.update(
            geometry=np.asarray(shape.geometry),
            center_row=np.asarray(shape.y, dtype=np.int64),
            center_column=np.asarray(shape.x, dtype=np.int64),
            size_x=np.asarray(shape.size_x, dtype=np.int64),
            size_y=np.asarray(shape.size_y, dtype=np.int64),
            angle=np.asarray(shape.angle, dtype=np.float64),
        )
    if sampling is not None:
        fields["sampling"] = np.asarray(sampling, dtype=float)
    if units is not None:
        fields["units"] = np.asarray([str(u) for u in units], dtype="U32")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".npz", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        np.savez_compressed(f, **fields)
    temporary.replace(path)
    return path


def load_scan_mask(path, scan_shape):
    """Read a file written by :func:`save_scan_mask` (any schema version).

    Returns ``(mask, shape)``; ``shape`` is None for a mask saved without one.
    Raises ValueError if the file is unreadable or was saved for another scan shape.
    """
    scan_shape = tuple(int(v) for v in scan_shape)
    try:
        with np.load(path, allow_pickle=False) as state:
            version = int(state["schema_version"])
            if version not in range(1, SCHEMA_VERSION + 1):
                raise ValueError(
                    f"Unsupported scan-mask schema {version}; expected 1 to {SCHEMA_VERSION}."
                )
            saved_shape = tuple(int(v) for v in state["scan_shape"])
            mask = np.asarray(state["mask"], dtype=bool)
            if saved_shape != scan_shape or mask.shape != scan_shape:
                raise ValueError(
                    f"Saved scan-mask shape {mask.shape} does not match current "
                    f"scan shape {scan_shape}."
                )
            if version >= 4 and "geometry" not in state:
                return mask, None
            if version == 1:
                geometry, size_x, size_y = "circle", int(state["radius"]), int(state["radius"])
            else:
                geometry = str(state["geometry"].item())
                size_x, size_y = int(state["size_x"]), int(state["size_y"])
            shape = ScanMaskShape(
                x=int(state["center_column"]),
                y=int(state["center_row"]),
                geometry=geometry,
                size_x=size_x,
                size_y=size_y,
                angle=float(state["angle"]) if version >= 3 else 0.0,
            )
            return mask, shape
    except (OSError, KeyError) as exc:
        raise ValueError(f"Could not load scan-mask state from {path}: {exc}") from exc


__all__ = ["GEOMETRIES", "SCHEMA_VERSION", "ScanMaskShape", "load_scan_mask", "save_scan_mask"]
