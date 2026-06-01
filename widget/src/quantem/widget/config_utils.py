"""
Shared QuantEM reconstruction ``config.json`` helpers.

Both Show3D and Show3DSlices accept a ``config=`` convenience argument so a
multislice ptychography z-stack can be calibrated (pixel size) and aligned
(in-plane rotation + post-crop) straight from the reconstruction metadata,
with no manual array math in the notebook. These helpers hold that logic in
one place so the two widgets stay consistent. The module has no widget
dependency: only numpy / json / math / pathlib.
"""
import json
import math
import pathlib
from collections.abc import Mapping, Sequence

import numpy as np


def _load_quantem_config(config: Mapping | str | pathlib.Path | None) -> Mapping | None:
    """Accept a parsed QuantEM config dict or a path to its JSON file."""
    if config is None:
        return None
    if isinstance(config, (str, pathlib.Path)):
        return json.loads(pathlib.Path(config).read_text())
    if isinstance(config, Mapping):
        return config
    raise TypeError(
        "config must be a parsed mapping, a config.json path, or None; "
        f"got {type(config).__name__}"
    )


def _config_get(config: Mapping, *keys: str):
    current = config
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _config_float(config: Mapping, *keys: str) -> float | None:
    value = _config_get(config, *keys)
    if value is None or value == "":
        return None
    return float(value)


def _centered_crop_for_shape(
    source_shape: Sequence[int],
    target_shape: Sequence[int],
) -> tuple[int, int, int, int]:
    """Centered row/column crop needed to display target_shape from source_shape."""
    if len(source_shape) < 2 or len(target_shape) < 2:
        raise ValueError(
            "cropped_shape inference needs source and target shapes with row/col axes"
        )
    crop_y = max(0, int(source_shape[-2]) - int(target_shape[-2]))
    crop_x = max(0, int(source_shape[-1]) - int(target_shape[-1]))
    return (
        crop_y // 2,
        crop_y - crop_y // 2,
        crop_x // 2,
        crop_x - crop_x // 2,
    )


def _post_crop_from_quantem_config(
    source_shape: Sequence[int],
    config: Mapping,
) -> int | tuple[int, int, int, int]:
    """Infer the post-rotation crop from QuantEM reconstruction metadata."""
    cropped_shape = _config_get(config, "object", "cropped_shape")
    if cropped_shape:
        return _centered_crop_for_shape(source_shape, cropped_shape)
    post_crop_px = (
        _config_get(config, "reconstruction", "obj_padding_px")
        or _config_get(config, "input", "padding")
        or 0
    )
    return int(post_crop_px)


def _pixel_size_from_quantem_config(config: Mapping) -> list[float] | None:
    """Return [pz, py, px] sampling in Å from QuantEM reconstruction metadata."""
    z_sampling = _config_float(config, "reconstruction", "slice_thickness_A")
    xy_sampling = _config_float(config, "reconstruction", "obj_sampling_A_per_px")
    if z_sampling is None or xy_sampling is None:
        return None
    return [z_sampling, xy_sampling, xy_sampling]


def _is_default_pixel_size(pixel_size: float | Sequence[float] | None) -> bool:
    return pixel_size is None or (
        np.isscalar(pixel_size) and float(pixel_size) == 0.0
    )


def _normalize_rotation_deg(rotation_deg: float) -> float:
    """Normalize an in-plane rotation angle for constructor transforms."""
    if isinstance(rotation_deg, bool):
        raise ValueError("rotation_deg must be a finite number, not bool")
    try:
        angle = float(rotation_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError("rotation_deg must be a finite number") from exc
    if not math.isfinite(angle):
        raise ValueError(f"rotation_deg must be finite, got {rotation_deg!r}")
    return angle


def _rotate_stack_inplane(data: np.ndarray, rotation_deg: float) -> np.ndarray:
    """Rotate a (N, H, W) stack in the row/col plane without SciPy.

    Matches scipy.ndimage.rotate(..., axes=(-2, -1), reshape=False, order=1,
    mode="nearest", prefilter=False) closely enough for notebook alignment while
    keeping the widgets free of a SciPy dependency. The output shape is
    unchanged and values stay float32.
    """
    angle = _normalize_rotation_deg(rotation_deg)
    if math.isclose(angle % 360.0, 0.0, abs_tol=1e-12) or math.isclose(angle % 360.0, 360.0, abs_tol=1e-12):
        return data

    src = np.ascontiguousarray(data, dtype=np.float32)
    n, h, w = src.shape
    radians = math.radians(angle)
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)

    yy, xx = np.indices((h, w), dtype=np.float32)
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    x = xx - cx
    y = yy - cy
    src_x = cos_a * x - sin_a * y + cx
    src_y = sin_a * x + cos_a * y + cy
    np.clip(src_x, 0.0, float(w - 1), out=src_x)
    np.clip(src_y, 0.0, float(h - 1), out=src_y)

    x0 = np.floor(src_x).astype(np.intp, copy=False)
    y0 = np.floor(src_y).astype(np.intp, copy=False)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = src_x - x0
    wy = src_y - y0

    w00 = ((1.0 - wx) * (1.0 - wy)).astype(np.float32, copy=False).ravel()
    w01 = (wx * (1.0 - wy)).astype(np.float32, copy=False).ravel()
    w10 = ((1.0 - wx) * wy).astype(np.float32, copy=False).ravel()
    w11 = (wx * wy).astype(np.float32, copy=False).ravel()
    idx00 = (y0 * w + x0).ravel()
    idx01 = (y0 * w + x1).ravel()
    idx10 = (y1 * w + x0).ravel()
    idx11 = (y1 * w + x1).ravel()

    out = np.empty_like(src, dtype=np.float32)
    src_flat = src.reshape(n, h * w)
    out_flat = out.reshape(n, h * w)
    for iz in range(n):
        frame = src_flat[iz]
        dst = out_flat[iz]
        np.multiply(frame[idx00], w00, out=dst)
        dst += frame[idx01] * w01
        dst += frame[idx10] * w10
        dst += frame[idx11] * w11
    return out
