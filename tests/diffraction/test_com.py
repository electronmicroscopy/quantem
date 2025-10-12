"""Test DPC center-of-mass utils"""

import numpy as np
import pytest
import torch

from quantem.core.utils.center_of_mass import com
from quantem.diffraction.dpc import (
    center_of_mass_optimized,
    com_with_coords,
    prepare_com_coords,
)

"""Test single Pattern - single bright pixel in middle of 2d array"""
def test_single_pattern_center_at_origin():
    pattern = torch.zeros(5, 5, dtype=torch.float32)
    pattern[2, 2] = 1.0

    com = center_of_mass_optimized(pattern)

    assert com.shape == (2,)
    torch.testing.assert_close(com, torch.zeros(2), atol=1e-6, rtol=0.0)

"""Compare Torch Result with NumPy"""
def test_single_pattern_off_center_matches_numpy():
    pattern = torch.zeros(7, 7, dtype=torch.float32)
    pattern[1, 4] = 3.5
    pattern[5, 2] = 1.0

    torch_result = center_of_mass_optimized(pattern)

    np_pattern = pattern.numpy()
    coords = np.argwhere(np_pattern > 0)
    weights = np_pattern[np_pattern > 0]
    weighted_mean = (coords * weights[:, None]).sum(axis=0) / weights.sum()
    expected_y = weighted_mean[0] - (pattern.shape[-2] - 1) * 0.5
    expected_x = weighted_mean[1] - (pattern.shape[-1] - 1) * 0.5

    torch.testing.assert_close(
        torch_result,
        torch.tensor([expected_x, expected_y], dtype=torch_result.dtype),
        atol=1e-6,
        rtol=0.0,
    )


def test_batch_processing_shape_and_values():
    """Test batch processing with a small batch of patterns."""
    batch = torch.zeros(4, 9, 9, dtype=torch.float32)
    centres = [(4, 6), (1, 3), (7, 1), (0, 8)]
    for idx, (row, col) in enumerate(centres):
        batch[idx, row, col] = 5.0

    com = center_of_mass_optimized(batch)

    assert com.shape == (4, 2)
    expected = []
    for row, col in centres:
        expected.append([
            col - (batch.shape[-1] - 1) * 0.5,
            row - (batch.shape[-2] - 1) * 0.5,
        ])
    torch.testing.assert_close(
        com,
        torch.tensor(expected, dtype=torch.float32, device=com.device),
        atol=1e-6,
        rtol=0.0,
    )


def test_mask_and_background_subtraction():
    """Test masking and background subtraction"""
    pattern = torch.full((6, 6), 2.0, dtype=torch.float32)
    pattern[2, 1] += 8.0  # signal left of center
    mask = torch.ones_like(pattern)
    mask[:, :3] = 0  # drop left half including bright pixel

    com_masked = center_of_mass_optimized(pattern, mask=mask)
    com_unmasked = center_of_mass_optimized(pattern)

    assert com_unmasked[0] < 0  # x offset negative, left

    expected_x = 4 - (pattern.shape[-1] - 1) * 0.5  # right-half center
    torch.testing.assert_close(
        com_masked,
        torch.tensor([expected_x, 0.0], dtype=torch.float32),
        atol=1e-6,
        rtol=0.0,
    )

    noisy = pattern + torch.randn_like(pattern) * 0.01
    com_bg = center_of_mass_optimized(noisy, subtract_background=True)
    assert torch.isfinite(com_bg).all()

def test_precomputed_coordinates_match_default():
    batch = torch.randn(3, 11, 13, dtype=torch.float32)
    coords = prepare_com_coords(
        height=batch.shape[-2],
        width=batch.shape[-1],
        device=batch.device,
        dtype=torch.float32,
    )
    from_precomputed = com_with_coords(batch, *coords)
    default = center_of_mass_optimized(batch)
    torch.testing.assert_close(from_precomputed, default, atol=1e-6, rtol=0.0)


def test_com_with_coords_validates_inputs():
    pattern = torch.ones(5, 5)
    x_coords = torch.arange(5)
    y_coords = torch.arange(4)  

    with pytest.raises(ValueError):
        com_with_coords(pattern, x_coords, y_coords)

    coords = prepare_com_coords(5, 5)
    with pytest.raises(ValueError):
        center_of_mass_optimized(pattern, mask=torch.ones_like(pattern), coords=coords)

def test_vectorized_center_matches_optimized():
    batch = torch.rand(8, 9, 9)
    expected = center_of_mass_optimized(batch)
    result = com(batch)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=0.0)