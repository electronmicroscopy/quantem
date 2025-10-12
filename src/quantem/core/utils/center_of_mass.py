from __future__ import annotations
from typing import Optional, Tuple
import torch
_CACHE_LIMIT = 8
_coord_cache: dict[tuple[int, int, torch.device, torch.dtype, float, float], tuple[torch.Tensor, torch.Tensor]] = {}

def clear_com_cache() -> None:
    _coord_cache.clear()

def _get_dtype(tensor: torch.Tensor) -> torch.dtype:
    return tensor.dtype if tensor.dtype in (torch.float32, torch.float64) else torch.float32

def _prepare_tensor(tensor: torch.Tensor) -> torch.Tensor:
    dtype = _get_dtype(tensor)
    return tensor.to(dtype) if tensor.dtype != dtype else tensor

def _create_coords(height: int, width: int, device: torch.device, dtype: torch.dtype, center: Optional[Tuple[float, float]]=None, use_cache: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
    if center is None:
        x0, y0 = ((width - 1) * 0.5, (height - 1) * 0.5)
    else:
        x0, y0 = (float(center[0]), float(center[1]))
    # Cache coordinate arrays on CPU
    if use_cache and device.type == 'cpu':
        cache_key = (height, width, device, dtype, x0, y0)
        if cache_key in _coord_cache:
            return _coord_cache[cache_key]
        x = torch.arange(width, device=device, dtype=dtype) - x0
        y = torch.arange(height, device=device, dtype=dtype) - y0
        if len(_coord_cache) >= _CACHE_LIMIT:
            _coord_cache.pop(next(iter(_coord_cache)))
        _coord_cache[cache_key] = (x, y)
        return (x, y)
    x = torch.arange(width, device=device, dtype=dtype) - x0
    y = torch.arange(height, device=device, dtype=dtype) - y0
    return (x, y)

def prepare_com_coords(height: int, width: int, device: Optional[torch.device | str]=None, dtype: torch.dtype=torch.float32, center: Optional[Tuple[float, float]]=None, use_cache: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
    resolved_device = torch.device(device) if device is not None else torch.device('cpu')
    if resolved_device.type not in {'cpu', 'cuda', 'mps'}:
        raise ValueError(f'Unsupported device type for CoM coordinates: {resolved_device.type}')
    return _create_coords(height, width, resolved_device, dtype, center=center, use_cache=use_cache)

def _validate_inputs(tensor: torch.Tensor, x_coords: Optional[torch.Tensor]=None, y_coords: Optional[torch.Tensor]=None) -> None:
    if tensor.ndim < 2:
        raise ValueError('Input must have at least 2 dimensions (height, width)')
    if x_coords is not None and y_coords is not None:
        height, width = tensor.shape[-2:]
        if x_coords.ndim != 1 or x_coords.shape[0] != width:
            raise ValueError('x_coords must be 1D with length matching pattern width')
        if y_coords.ndim != 1 or y_coords.shape[0] != height:
            raise ValueError('y_coords must be 1D with length matching pattern height')

def _compute_com(intensity: torch.Tensor, x_coords: torch.Tensor, y_coords: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    # Calculate total intensity, weighted coordinates
    total = intensity.sum(dim=(-2, -1))
    weighted_x = (intensity * x_coords).sum(dim=(-2, -1))
    weighted_y = (intensity * y_coords.unsqueeze(-1)).sum(dim=(-2, -1))

    zero_mask = torch.abs(total) < eps
    com_x = torch.where(zero_mask, torch.zeros_like(weighted_x), weighted_x / total)
    com_y = torch.where(zero_mask, torch.zeros_like(weighted_y), weighted_y / total)
    return torch.stack((com_x, com_y), dim=-1)

def com_with_coords(diffraction_patterns: torch.Tensor, x_coords: torch.Tensor, y_coords: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    _validate_inputs(diffraction_patterns, x_coords, y_coords)
    device = diffraction_patterns.device
    dtype = _get_dtype(diffraction_patterns)
    patterns = _prepare_tensor(diffraction_patterns)
    if x_coords.device != device or x_coords.dtype != dtype:
        x_coords = x_coords.to(device=device, dtype=dtype)
    if y_coords.device != device or y_coords.dtype != dtype:
        y_coords = y_coords.to(device=device, dtype=dtype)
    return _compute_com(patterns, x_coords, y_coords, eps)

def com(diffraction_patterns: torch.Tensor, eps: float=1e-12, use_cache: bool=True) -> torch.Tensor:
    _validate_inputs(diffraction_patterns)
    height, width = diffraction_patterns.shape[-2:]
    device = diffraction_patterns.device
    dtype = _get_dtype(diffraction_patterns)
    patterns = _prepare_tensor(diffraction_patterns)
    x_coords, y_coords = _create_coords(height, width, device, dtype, use_cache=use_cache)
    return _compute_com(patterns, x_coords, y_coords, eps)
__all__ = ['com_with_coords', 'clear_com_cache', 'prepare_com_coords', 'com']