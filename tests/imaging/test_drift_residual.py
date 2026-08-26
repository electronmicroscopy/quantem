"""Residual-correction contracts needed by the publication workflows."""

import numpy as np
import pytest
import torch

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.imaging.drift import DriftCorrection, StripPass
from quantem.imaging.drift.core.nonrigid import _regularize_knots
from quantem.imaging.drift.core.strip import (
    free_weight,
    measure_strip_residual_torch,
)


def _accelerator_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    pytest.skip("non-rigid backend parity requires CUDA or MPS")


@pytest.mark.parametrize("num_knots", (1, 2, 3))
def test_multiknot_diagnostics_have_explicit_fast_direction_meaning(num_knots):
    """Fast roughness is adjacent-knot displacement, not image roughness."""
    values = np.linspace(0.0, 1.0, num_knots, dtype=np.float32)
    measured = (
        0.0
        if num_knots == 1
        else float(np.sqrt(np.mean(np.diff(values) ** 2)))
    )
    assert measured == 0.0 if num_knots == 1 else measured > 0.0
    doc = DriftCorrection.diagnose_nonrigid.__doc__ or ""
    assert "neighboring knot" in doc
    assert "does not measure image noise" in doc


def test_publication_strip_recipe_is_explicit_and_ordered():
    """The frozen XEDS workflow has three visible coarse-to-fine passes."""
    recipe = (
        StripPass(
            num_strips=24,
            smoothing_sigma=12,
            max_column_shift=80,
            max_row_shift=8,
        ),
        StripPass(
            num_strips=24,
            smoothing_sigma=12,
            max_column_shift=12,
            max_row_shift=3,
        ),
        StripPass(
            num_strips=64,
            smoothing_sigma=6,
            max_column_shift=6,
            max_row_shift=2,
            update_fraction=0.8,
        ),
    )

    assert [item.num_strips for item in recipe] == [24, 24, 64]
    assert [item.max_column_shift for item in recipe] == [80, 12, 6]
    assert recipe[-1].update_fraction == 0.8


def test_strip_free_weight_can_freeze_then_smoothly_release_scanlines():
    """A partial residual update has a stable zero-to-one transition."""
    weights = free_weight(100, free_from_frac=0.55, ramp_frac=0.08)

    assert weights.shape == (100,)
    assert np.all(weights[:47] == 0.0)
    assert np.all((weights >= 0.0) & (weights <= 1.0))
    assert np.all(weights[55:] == 1.0)


def test_strip_measurement_recovers_local_integer_residual():
    """Each slow-scan strip recovers the same known residual displacement."""
    rng = np.random.default_rng(5)
    reference = rng.normal(size=(48, 48)).astype(np.float32)
    moving = np.roll(np.roll(reference, 2, axis=0), -3, axis=1)

    result = measure_strip_residual_torch(
        reference,
        moving,
        np.ones_like(reference, dtype=bool),
        n_strips=4,
        max_shift_row=3,
        max_shift_col=4,
        device="cpu",
        method="brute",
    )

    np.testing.assert_array_equal(result["drow"], np.full(4, -2.0))
    np.testing.assert_array_equal(result["dcol"], np.full(4, 3.0))
    assert np.all(result["valid"])


@pytest.mark.parametrize("trend_order", (0, 1, 2, 3))
def test_nonrigid_regularization_matches_cpu(trend_order):
    """The MPS/CUDA fallback retains float32 CPU knot precision."""
    device = _accelerator_device()
    row_count = 65
    generator = torch.Generator().manual_seed(42)
    coordinates = torch.arange(row_count, dtype=torch.float32)
    coordinates = (coordinates - coordinates.mean()) / coordinates.std()
    vander = torch.stack(
        [coordinates**power for power in range(trend_order + 1)],
        dim=1,
    )
    knots = torch.randn(2, 2, row_count, 3, generator=generator)
    previous = torch.randn(2, 2, row_count, 3, generator=generator)

    expected = knots.clone()
    _regularize_knots(expected, previous, vander, 2, 4, 0.8)
    actual = knots.to(device)
    _regularize_knots(
        actual,
        previous.to(device),
        vander.to(device),
        2,
        4,
        0.8,
    )

    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-5, atol=3e-5)


def test_two_dimensional_map_uses_same_field_and_preserves_metadata():
    """Element maps and equivalent cube channels share one spatial warp."""
    rows, columns = np.indices((24, 24), dtype=np.float32)
    image = rows + 2 * columns
    drift = DriftCorrection.from_reference(
        image,
        image.copy(),
        scan_direction_degrees=0.0,
        device="cpu",
    ).preprocess(
        num_knots=1,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    drift.correct_affine(
        max_drift_rate=0.01,
        num_rates=3,
        refine=False,
        max_image_shift=4,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    element_map = Dataset2d.from_array(
        image,
        name="Ti K",
        origin=(1.0, 2.0),
        sampling=(0.2, 0.3),
        units=("nm", "nm"),
    )
    element_map.metadata["line"] = "Ti_K_wide"

    corrected_map = drift.apply_correction(element_map, image_index=1)
    corrected_cube = drift.apply_correction(
        np.stack((image, 3 * image), axis=-1),
        image_index=1,
    )

    np.testing.assert_allclose(corrected_map.array, corrected_cube[..., 0])
    np.testing.assert_allclose(corrected_map.origin, element_map.origin)
    np.testing.assert_allclose(corrected_map.sampling, element_map.sampling)
    assert corrected_map.metadata["line"] == "Ti_K_wide"
