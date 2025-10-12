"""Optimized center of mass calculations for DPC diffraction patterns"""

from __future__ import annotations

from typing import Optional, Tuple
import torch

from quantem.core.utils.center_of_mass import (
    com_with_coords,
    clear_com_cache,
    prepare_com_coords,
    com,
)


def _get_dtype(tensor: torch.Tensor) -> torch.dtype:
    return tensor.dtype if tensor.dtype in (torch.float32, torch.float64) else torch.float32


def _prepare_tensor(tensor: torch.Tensor) -> torch.Tensor:
    dtype = _get_dtype(tensor)
    return tensor.to(dtype) if tensor.dtype != dtype else tensor


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

    if coords is not None:
        if any(option is not None for option in (mask, pixel_size, center)):
            raise ValueError("coords can only be supplied when mask, pixel_size, and center are not provided.")
        if subtract_background:
            raise ValueError("coords cannot be supplied when subtract_background=True.")
        return com_with_coords(diffraction_patterns, coords[0], coords[1], eps)

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
    """Support masks, background subtraction, and scaling"""
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
    eps: float = 1e-12,
) -> torch.Tensor:
    return com(diffraction_patterns, eps=eps)


def _full_featured_center_of_mass(
    diffraction_patterns: torch.Tensor,
    pixel_size: Optional[float | Tuple[float, float]] = None,
    mask: Optional[torch.Tensor] = None,
    subtract_background: bool = False,
    center: Optional[Tuple[float, float]] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    if diffraction_patterns.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions (height, width)")

    height, width = diffraction_patterns.shape[-2:]
    device = diffraction_patterns.device
    dtype = _get_dtype(diffraction_patterns)

    x_coords, y_coords = prepare_com_coords(
        height,
        width,
        device=device,
        dtype=dtype,
        center=center,
        use_cache=False,
    )

    intensity = _prepare_tensor(diffraction_patterns)
    if mask is not None:
        intensity = intensity * mask.to(dtype=dtype).clamp(0, 1)

    if subtract_background:
        median_bg = torch.quantile(intensity.flatten(-2), 0.5, dim=-1, keepdim=True).unsqueeze(-1)
        intensity = torch.clamp(intensity - median_bg, min=0.0)

    if pixel_size is not None:
        if isinstance(pixel_size, (tuple, list)):
            x_coords = x_coords * pixel_size[0]
            y_coords = y_coords * pixel_size[1]
        else:
            x_coords = x_coords * pixel_size
            y_coords = y_coords * pixel_size

    # Clean up NaNs/Infs
    intensity = torch.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    return com_with_coords(intensity, x_coords, y_coords, eps)


__all__ = [
    "center_of_mass_optimized",
    "com_with_coords",
    "clear_com_cache",
    "prepare_com_coords",
]
