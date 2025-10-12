"""optimized center of mass calculations for dpc diffraction patterns"""

from typing import Optional, Tuple
import torch

_coordinate_cache: dict[tuple[int, int, torch.device, torch.dtype, float, float], tuple[torch.Tensor, torch.Tensor]] = {}

def clear_coordinate_cache() -> None:
    _coordinate_cache.clear()

def _get_dtype(tensor: torch.Tensor) -> torch.dtype:
    return tensor.dtype if tensor.dtype in (torch.float32, torch.float64) else torch.float32

def _prepare_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """tensor has appropriate dtype for computation"""
    dtype = _get_dtype(tensor)
    return tensor.to(dtype) if tensor.dtype != dtype else tensor

def _create_coordinates(
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype,
    center: Optional[Tuple[float, float]] = None,
    use_cache: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    if center is None:
        x0, y0 = (W - 1) * 0.5, (H - 1) * 0.5
    else:
        x0, y0 = float(center[0]), float(center[1])

    # only cache CPU tensors when enabled
    if use_cache and device.type == "cpu":
        cache_key = (H, W, device, dtype, x0, y0)
        if cache_key in _coordinate_cache:
            return _coordinate_cache[cache_key]

        x_coords = torch.arange(W, device=device, dtype=dtype) - x0
        y_coords = torch.arange(H, device=device, dtype=dtype) - y0

        if len(_coordinate_cache) >= 8:
            _coordinate_cache.pop(next(iter(_coordinate_cache)))
        _coordinate_cache[cache_key] = (x_coords, y_coords)
        return x_coords, y_coords

    x_coords = torch.arange(W, device=device, dtype=dtype) - x0
    y_coords = torch.arange(H, device=device, dtype=dtype) - y0
    return x_coords, y_coords

def _compute_center_of_mass(
    intensity: torch.Tensor,
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
    eps: float = 1e-12
) -> torch.Tensor:
    # center of mass logic, now shared
    total_intensity = intensity.sum(dim=(-2, -1))
    weighted_x = (intensity * x_coords).sum(dim=(-2, -1))
    weighted_y = (intensity * y_coords.unsqueeze(-1)).sum(dim=(-2, -1))

    zero_intensity_mask = torch.abs(total_intensity) < eps
    com_x = torch.where(zero_intensity_mask, torch.zeros_like(weighted_x), weighted_x / total_intensity)
    com_y = torch.where(zero_intensity_mask, torch.zeros_like(weighted_y), weighted_y / total_intensity)

    return torch.stack((com_x, com_y), dim=-1)


def prepare_center_of_mass_coordinates(
    height: int,
    width: int,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
    center: Optional[Tuple[float, float]] = None,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    resolved_device = torch.device(device) if device is not None else torch.device("cpu")
    if resolved_device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device type for CoM coordinates: {resolved_device.type}")
    return _create_coordinates(
        height,
        width,
        resolved_device,
        dtype,
        center=center,
        use_cache=use_cache,
    )

def _validate_inputs(
    diffraction_patterns: torch.Tensor,
    x_coords: Optional[torch.Tensor] = None,
    y_coords: Optional[torch.Tensor] = None
) -> None:
    if diffraction_patterns.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions (height, width)")

    if x_coords is not None and y_coords is not None:
        H, W = diffraction_patterns.shape[-2:]
        if x_coords.ndim != 1 or x_coords.shape[0] != W:
            raise ValueError("x_coords must be 1D with length match pattern width")
        if y_coords.ndim != 1 or y_coords.shape[0] != H:
            raise ValueError("y_coords must be 1D with length match pattern height")

def center_of_mass_with_coordinates(
    diffraction_patterns: torch.Tensor,
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    # compute the center of mass using already computed coord vectors
    _validate_inputs(diffraction_patterns, x_coords, y_coords)

    device = diffraction_patterns.device
    dtype = _get_dtype(diffraction_patterns)

    diffraction_patterns = _prepare_tensor(diffraction_patterns)
    if x_coords.device != device or x_coords.dtype != dtype:
        x_coords = x_coords.to(device=device, dtype=dtype)
    if y_coords.device != device or y_coords.dtype != dtype:
        y_coords = y_coords.to(device=device, dtype=dtype)

    return _compute_center_of_mass(diffraction_patterns, x_coords, y_coords, eps)

def _center_of_mass_dispatch(
    diffraction_patterns: torch.Tensor,
    pixel_size: Optional[float | Tuple[float, float]],
    mask: Optional[torch.Tensor],
    subtract_background: bool,
    center: Optional[Tuple[float, float]],
    coords: Optional[Tuple[torch.Tensor, torch.Tensor]],
    eps: float,
    vectorized_fn,
    full_fn,
) -> torch.Tensor:
    if diffraction_patterns.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions, height and width")

    is_single_pattern = diffraction_patterns.ndim == 2
    device = diffraction_patterns.device
    dtype = _get_dtype(diffraction_patterns)

    if coords is not None and any(
        option is not None
        for option in (mask, pixel_size, center)
    ):
        raise ValueError("coords can only be supplied when mask, pixel_size, and center are not provided.")

    if coords is not None and subtract_background:
        raise ValueError("coords cannot be supplied when subtract_background=True.")

    if coords is not None:
        return center_of_mass_with_coordinates(diffraction_patterns, coords[0], coords[1], eps)

    if mask is None and not subtract_background and pixel_size is None and center is None:
        if is_single_pattern:
            batched = diffraction_patterns.unsqueeze(0)
            return vectorized_fn(batched, eps).squeeze(0)
        return vectorized_fn(diffraction_patterns, eps)

    return full_fn(
        diffraction_patterns,
        pixel_size,
        mask,
        subtract_background,
        center,
        eps,
    )

@torch.no_grad()
def center_of_mass_optimized(
    diffraction_patterns: torch.Tensor,
    pixel_size: Optional[float | Tuple[float, float]] = None,
    mask: Optional[torch.Tensor] = None,
    subtract_background: bool = False,
    center: Optional[Tuple[float, float]] = None,
    coords: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    return _center_of_mass_dispatch(
        diffraction_patterns,
        pixel_size,
        mask,
        subtract_background,
        center,
        coords,
        eps,
        _vectorized_batch_center_of_mass,
        _full_featured_center_of_mass,
    )

def _vectorized_batch_center_of_mass(
    diffraction_patterns: torch.Tensor,
    eps: float = 1e-12
) -> torch.Tensor:
    H, W = diffraction_patterns.shape[-2:]
    device = diffraction_patterns.device
    dtype = _get_dtype(diffraction_patterns)

    diffraction_patterns = _prepare_tensor(diffraction_patterns)
    x_coords, y_coords = _create_coordinates(H, W, device, dtype)

    return _compute_center_of_mass(diffraction_patterns, x_coords, y_coords, eps)

def _full_featured_center_of_mass(
    diffraction_patterns: torch.Tensor,
    pixel_size: Optional[float | Tuple[float, float]] = None,
    mask: Optional[torch.Tensor] = None,
    subtract_background: bool = False,
    center: Optional[Tuple[float, float]] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    _validate_inputs(diffraction_patterns)

    H, W = diffraction_patterns.shape[-2:]
    device = diffraction_patterns.device
    dtype = _get_dtype(diffraction_patterns)

    diffraction_patterns = _prepare_tensor(diffraction_patterns)
    x_coords, y_coords = _create_coordinates(H, W, device, dtype, center, use_cache=False)

    # Apply pixel scaling
    if pixel_size is not None:
        if isinstance(pixel_size, (tuple, list)):
            x_coords *= pixel_size[0]
            y_coords *= pixel_size[1]
        else:
            x_coords *= pixel_size
            y_coords *= pixel_size

    # Apply mask and background subtraction
    intensity = diffraction_patterns
    if mask is not None:
        intensity = intensity * mask.to(dtype).clamp(0, 1)

    if subtract_background:
        median_bg = torch.quantile(intensity.flatten(-2), 0.5, dim=-1, keepdim=True).unsqueeze(-1)
        intensity = torch.clamp(intensity - median_bg, min=0.0)

    # Clean NaN and inf values and compute center of mass
    intensity = torch.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    return _compute_center_of_mass(intensity, x_coords, y_coords, eps)
