"""Optimized center of mass calculations for diffraction patterns."""

from typing import Optional, Tuple

import numpy as np
import torch

try:
    import scipy.ndimage as ndi
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

_coordinate_cache = {}


def _get_dtype(tensor: torch.Tensor) -> torch.dtype:
    """Get appropriate dtype for calculations."""
    return (
        tensor.dtype
        if tensor.dtype in (torch.float32, torch.float64)
        else torch.float32
    )


def _create_coordinates(
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype,
    center: Optional[Tuple[float, float]] = None,
    use_cache: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create coordinate tensors with optional caching."""
    # Set coordinate origin
    if center is None:
        x0, y0 = (W - 1) * 0.5, (H - 1) * 0.5
    else:
        x0, y0 = float(center[0]), float(center[1])

    # Skip caching for MPS or when disabled
    if not use_cache or device.type == 'mps':
        x_coords = torch.arange(W, device=device, dtype=dtype) - x0
        y_coords = torch.arange(H, device=device, dtype=dtype) - y0
        return x_coords, y_coords

    # Use caching for other devices
    cache_key = (H, W, device, dtype, x0, y0)
    if cache_key not in _coordinate_cache:
        x_coords = torch.arange(W, device=device, dtype=dtype) - x0
        y_coords = torch.arange(H, device=device, dtype=dtype) - y0
        _coordinate_cache[cache_key] = (x_coords, y_coords)
    else:
        x_coords, y_coords = _coordinate_cache[cache_key]

    return x_coords, y_coords

@torch.no_grad()
def center_of_mass_optimized(
    diffraction_patterns: torch.Tensor,
    pixel_size: Optional[float | Tuple[float, float]] = None,
    mask: Optional[torch.Tensor] = None,
    subtract_background: bool = False,
    center: Optional[Tuple[float, float]] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    if diffraction_patterns.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions (height, width)")

    is_single_pattern = diffraction_patterns.ndim == 2
    device = diffraction_patterns.device
    dtype = _get_dtype(diffraction_patterns)

    # Use scipy for simple single-pattern CPU calculations
    if (is_single_pattern and SCIPY_AVAILABLE and device.type == 'cpu'
        and mask is None and not subtract_background and pixel_size is None and center is None):

        H, W = diffraction_patterns.shape
        com = ndi.center_of_mass(diffraction_patterns.detach().numpy())
        x_center = com[1] - (W - 1) * 0.5
        y_center = com[0] - (H - 1) * 0.5

        result_np = np.array([x_center, y_center], dtype=np.float32 if dtype == torch.float32 else np.float64)
        return torch.from_numpy(result_np).to(device=device)

    # Use optimized batch processing for simple cases
    elif (not is_single_pattern and mask is None and not subtract_background
          and pixel_size is None and center is None):
        return _ultra_fast_batch_center_of_mass(diffraction_patterns, eps)

    # Handle complex cases with full feature set
    else:
        return _full_featured_center_of_mass(diffraction_patterns, pixel_size, mask, subtract_background, center, eps)


def _ultra_fast_batch_center_of_mass(
    diffraction_patterns: torch.Tensor,
    eps: float = 1e-12
) -> torch.Tensor:
    H, W = diffraction_patterns.shape[-2:]
    device = diffraction_patterns.device
    dtype = _get_dtype(diffraction_patterns)

    if diffraction_patterns.dtype != dtype:
        diffraction_patterns = diffraction_patterns.to(dtype)

    x_coords, y_coords = _create_coordinates(H, W, device, dtype)

    # Vectorized center of mass calculation
    total_intensity = diffraction_patterns.sum(dim=(-2, -1))
    weighted_x = (diffraction_patterns * x_coords).sum(dim=(-2, -1))
    weighted_y = (diffraction_patterns * y_coords.unsqueeze(-1)).sum(dim=(-2, -1))

    # Handle zero intensity edge cases
    zero_intensity_mask = torch.abs(total_intensity) < eps
    com_x = torch.where(
        zero_intensity_mask,
        torch.zeros_like(weighted_x),
        weighted_x / total_intensity
    )
    com_y = torch.where(
        zero_intensity_mask,
        torch.zeros_like(weighted_y),
        weighted_y / total_intensity
    )

    return torch.stack((com_x, com_y), dim=-1)

def _compile_if_supported(func):
    """Apply torch.compile with MPS safety check."""
    try:
        compiled = torch.compile(func)

        def safe_compiled_func(*args, **kwargs):
            # Skip compilation for MPS due to symbolic expression issues
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.device.type == 'mps':
                    return func(*args, **kwargs)
            return compiled(*args, **kwargs)

        return safe_compiled_func
    except (AttributeError, ImportError):
        return func


_ultra_fast_batch_center_of_mass = _compile_if_supported(_ultra_fast_batch_center_of_mass)


def _full_featured_center_of_mass(
    diffraction_patterns: torch.Tensor,
    pixel_size: Optional[float | Tuple[float, float]] = None,
    mask: Optional[torch.Tensor] = None,
    subtract_background: bool = False,
    center: Optional[Tuple[float, float]] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    if diffraction_patterns.ndim < 2:
        raise ValueError("Input must have (at least) 2 dimensions (height, width)")

    H, W = diffraction_patterns.shape[-2:]
    device = diffraction_patterns.device
    dtype = _get_dtype(diffraction_patterns)

    if diffraction_patterns.dtype != dtype:
        diffraction_patterns = diffraction_patterns.to(dtype)

    # Create coordinates
    x_coords, y_coords = _create_coordinates(H, W, device, dtype, center, use_cache=False)

    # Apply pixel scaling
    if pixel_size is not None:
        if isinstance(pixel_size, (tuple, list)):
            x_coords *= pixel_size[0]
            y_coords *= pixel_size[1]
        else:
            x_coords *= pixel_size
            y_coords *= pixel_size

    # Apply mask if provided
    intensity = diffraction_patterns
    if mask is not None:
        intensity = intensity * mask.to(dtype).clamp(0, 1)

    # Subtract background
    if subtract_background:
        median_bg = torch.quantile(
            intensity.flatten(-2), 0.5, dim=-1, keepdim=True
        ).unsqueeze(-1)
        intensity = torch.clamp(intensity - median_bg, min=0.0)

    # Clean NaN/inf values
    intensity = torch.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)

    # Calculate center of mass
    total_intensity = intensity.sum(dim=(-2, -1))
    weighted_x = (intensity * x_coords).sum(dim=(-2, -1))
    weighted_y = (intensity * y_coords.unsqueeze(-1)).sum(dim=(-2, -1))

    # Handle zero intensity edge cases
    zero_intensity_mask = torch.abs(total_intensity) < eps
    com_x = torch.where(
        zero_intensity_mask,
        torch.zeros_like(weighted_x),
        weighted_x / total_intensity
    )
    com_y = torch.where(
        zero_intensity_mask,
        torch.zeros_like(weighted_y),
        weighted_y / total_intensity
    )

    return torch.stack((com_x, com_y), dim=-1)

_full_featured_center_of_mass = _compile_if_supported(_full_featured_center_of_mass)

def warmup_compiled_functions(
    batch_size: int = 100,
    pattern_size: Tuple[int, int] = (256, 256)
) -> None:
    """Pre-compile torch.compile functions to eliminate JIT overhead."""
    test_data = torch.randn(batch_size, *pattern_size, dtype=torch.float32)

    # Warmup batch function
    for _ in range(5):
        _ultra_fast_batch_center_of_mass(test_data)

    # Warmup full-featured function
    for _ in range(5):
        _full_featured_center_of_mass(test_data, None, None, False, None, 1e-12)
