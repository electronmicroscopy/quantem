"""Scientist-facing API contracts for drift correction."""

import inspect

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.imaging import DriftCorrection


def _orthogonal_pair(size: int = 64) -> tuple[Dataset2d, Dataset2d]:
    """Return one calibrated synthetic field acquired at 0 and 90 degrees."""
    rng = np.random.default_rng(42)
    image = gaussian_filter(
        rng.normal(size=(size, size)).astype(np.float32),
        sigma=1.2,
    )
    rows, columns = np.indices(image.shape, dtype=np.float32)
    image += 2.0 * np.exp(
        -((rows - 19.0) ** 2 + (columns - 43.0) ** 2) / (2.0 * 4.0**2)
    )
    scans = []
    for array, angle in ((image, 0.0), (np.rot90(image, k=-1).copy(), 90.0)):
        dataset = Dataset2d.from_array(
            array,
            origin=(1.0, 2.0),
            sampling=(0.08, 0.08),
            units=("nm", "nm"),
        )
        dataset.metadata["scan_rotation_deg"] = angle
        scans.append(dataset)
    return scans[0], scans[1]


def test_primary_api_uses_scientist_facing_names():
    """The final PR exposes one concise chain without legacy solve names."""
    preprocess = inspect.signature(DriftCorrection.preprocess).parameters
    affine = inspect.signature(DriftCorrection.correct_affine).parameters
    nonrigid = inspect.signature(DriftCorrection.correct_nonrigid).parameters

    assert {"padding_fraction", "smoothing_sigma", "num_knots"} <= set(preprocess)
    assert {"max_drift_rate", "num_rates", "region"} <= set(affine)
    assert {"num_knots", "num_refine_cycles", "knot_smoothing_sigma"} <= set(
        nonrigid
    )
    for old_name in (
        "pad_fraction",
        "kde_sigma",
        "number_knots",
        "step",
        "num_tests",
        "num_iterations",
    ):
        assert old_name not in preprocess
        assert old_name not in affine
        assert old_name not in nonrigid
    for old_method in (
        "from_data",
        "align_affine",
        "align_nonrigid",
        "generate_corrected",
    ):
        assert not hasattr(DriftCorrection, old_method)


def test_from_images_requires_angles_for_bare_arrays():
    """Bare arrays never receive a silent scan-angle assumption."""
    image = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(TypeError, match="scan_direction_degrees is required"):
        DriftCorrection.from_images(image, image.copy(), device="cpu")


def test_metadata_driven_affine_workflow_returns_calibrated_dataset():
    """The normal image workflow is short, finite, and calibration preserving."""
    scan_0, scan_90 = _orthogonal_pair()
    drift = DriftCorrection.from_images(scan_0, scan_90, device="cpu")
    drift.preprocess(
        padding_fraction=0.25,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    drift.correct_affine(
        max_drift_rate=0.02,
        num_rates=5,
        refine=False,
        max_image_shift=8,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    corrected = drift.corrected(
        upsample_factor=1,
        output_original_shape=True,
        verbose=False,
    )

    assert isinstance(corrected, Dataset2d)
    assert corrected.shape == scan_0.shape
    assert np.isfinite(corrected.array).all()
    np.testing.assert_allclose(corrected.origin, scan_0.origin)
    np.testing.assert_allclose(corrected.sampling, scan_0.sampling)
    assert corrected.units == scan_0.units
    assert len(drift.drift_rate) == 2

    automatic = drift.corrected(verbose=False)
    canvas = drift.corrected(
        output_original_shape=False,
        verbose=False,
    )
    assert automatic.shape == tuple(drift.shape[-2:])
    assert canvas.shape == tuple(drift.shape[-2:])


def test_nonrigid_diagnostic_defines_fast_roughness():
    """The difficult multi-knot diagnostic states exactly what roughness means."""
    doc = DriftCorrection.diagnose_nonrigid.__doc__ or ""
    assert "root-mean-square difference between neighboring knot" in doc
    assert "does not measure image noise" in doc
