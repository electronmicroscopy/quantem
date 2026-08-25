"""Calibration and provenance invariants for computational downsampling."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.imaging.drift import DriftCorrection
from quantem.imaging.drift.core.affine import _affine_rate_from_knots
from quantem.imaging.drift.preparation import (
    average_downsample_2d,
    resolve_downsample,
)


def _calibrated_pair(size: int = 16):
    rows, columns = np.indices((size, size), dtype=np.float32)
    image = rows * 10 + columns
    metadata = {"source": "synthetic calibrated scan"}
    datasets = []
    for array in (image, np.rot90(image, k=-1).copy()):
        dataset = Dataset2d.from_array(
            array,
            origin=(1.0, 2.0),
            sampling=(0.2, 0.3),
            units=("nm", "nm"),
        )
        dataset.metadata.update(metadata)
        datasets.append(dataset)
    return datasets


def test_average_downsample_is_exact_block_mean():
    """Integer count images become float32 block averages, never decimation."""
    image = np.arange(64, dtype=np.uint16).reshape(8, 8)

    result = average_downsample_2d(image, 2)

    expected = image.reshape(4, 2, 4, 2).mean(axis=(1, 3)).astype(np.float32)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)


def test_preprocess_downsample_preserves_pixel_center_calibration():
    """Sampling grows and origin moves to the center of each averaged block."""
    datasets = _calibrated_pair()
    drift = DriftCorrection.from_data(
        datasets,
        scan_direction_degrees=(0.0, 90.0),
        device="cpu",
    ).preprocess(
        downsample=2,
        show_merged=False,
        show_images=False,
        show_knots=False,
    )

    assert drift.images[0].shape == (8, 8)
    np.testing.assert_allclose(drift.images[0].sampling, (0.4, 0.6))
    np.testing.assert_allclose(drift.images[0].origin, (1.1, 2.15))
    assert drift.images[0].units == ["nm", "nm"]
    assert drift.images[0].metadata["source"] == "synthetic calibrated scan"
    assert drift.downsample_metadata["original_images"][0]["shape"] == [16, 16]


def test_corrected_output_records_downsampling_provenance():
    """A corrected image tells readers which computational grid was fitted."""
    drift = DriftCorrection.from_data(
        _calibrated_pair(),
        scan_direction_degrees=(0.0, 90.0),
        device="cpu",
    ).preprocess(
        downsample=2,
        show_merged=False,
        show_images=False,
        show_knots=False,
    )

    corrected = drift.generate_corrected(
        upsample_factor=1,
        show_merged=False,
    )

    assert corrected.metadata["downsample"] == 2
    assert corrected.metadata["downsample_method"] == "average"
    assert corrected.metadata["downsample_metadata"]["factor"] == 2
    np.testing.assert_allclose(corrected.sampling, (0.4, 0.6))


def test_downsample_requires_exact_divisor_and_new_correction():
    """Grid changes fail with a corrective message instead of silent cropping."""
    with pytest.raises(ValueError, match="divisible"):
        resolve_downsample(4, (18, 18))

    drift = DriftCorrection.from_data(
        _calibrated_pair(),
        scan_direction_degrees=(0.0, 90.0),
        device="cpu",
    ).preprocess(
        show_merged=False,
        show_images=False,
        show_knots=False,
    )
    with pytest.raises(RuntimeError, match="Create a new DriftCorrection"):
        drift.preprocess(
            downsample=2,
            show_merged=False,
            show_images=False,
            show_knots=False,
        )


def test_automatic_factor_is_largest_exact_divisor_up_to_eight():
    """Automatic resolution selection is deterministic and shape-safe."""
    assert resolve_downsample("auto", (2048, 2048)) == 8
    assert resolve_downsample("auto", (1026, 1026)) == 2
    assert resolve_downsample("auto", (1025, 1025)) == 1


def test_affine_pyramid_refines_rate_on_native_grid():
    """A pooled broad search retains the native image and affine rate."""
    size = 64
    rng = np.random.default_rng(23)
    reference = gaussian_filter(rng.normal(size=(size, size)), 1.5).astype(np.float32)
    target = np.empty_like(reference)
    columns = np.arange(size, dtype=np.float32)
    for row in range(size):
        shift = 0.04 * (row - (size - 1) / 2)
        target[row] = np.interp(
            columns + shift,
            columns,
            reference[row],
            left=float(np.median(reference[row])),
            right=float(np.median(reference[row])),
        )

    native = DriftCorrection.from_reference(
        reference,
        target,
        scan_direction_degrees=0.0,
        device="cpu",
    ).preprocess(
        show_merged=False,
        show_images=False,
        show_knots=False,
    )
    pyramid = DriftCorrection.from_reference(
        reference,
        target,
        scan_direction_degrees=0.0,
        device="cpu",
    ).preprocess(
        show_merged=False,
        show_images=False,
        show_knots=False,
    )

    native.align_affine(
        step=0.02,
        num_tests=7,
        max_image_shift=8,
        downsample=1,
        show_merged=False,
        show_images=False,
        show_knots=False,
    )
    pyramid.align_affine(
        step=0.02,
        num_tests=7,
        max_image_shift=8,
        downsample=2,
        show_merged=False,
        show_images=False,
        show_knots=False,
    )

    assert pyramid.images[0].shape == (size, size)
    assert pyramid.affine_search_info["downsample_factor"] == 2
    np.testing.assert_allclose(
        _affine_rate_from_knots(pyramid, 1),
        _affine_rate_from_knots(native, 1),
        atol=0.01,
    )
