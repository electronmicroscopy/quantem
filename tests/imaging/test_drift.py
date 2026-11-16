"""Scientist-facing API contracts for drift correction."""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter, shift

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
    translation = inspect.signature(DriftCorrection.align_translation).parameters
    affine = inspect.signature(DriftCorrection.correct_affine).parameters
    nonrigid = inspect.signature(DriftCorrection.correct_nonrigid).parameters

    assert {"padding_fraction", "smoothing_sigma", "num_knots"} <= set(preprocess)
    assert {"max_image_shift", "fixed_scans"} <= set(translation)
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


def test_static_show_returns_one_closed_matplotlib_figure():
    """A bare static show call does not also queue an inline duplicate."""
    scan_0, scan_90 = _orthogonal_pair()
    drift = DriftCorrection.from_images(scan_0, scan_90, device="cpu")
    drift.preprocess(
        padding_fraction=0.25,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )

    figure = drift.show(mode="static", cmap="gray")

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 6
    assert figure.number not in plt.get_fignums()


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
        output_frame="input",
        verbose=False,
    )

    assert isinstance(corrected, Dataset2d)
    assert corrected.shape == scan_0.shape
    assert np.isfinite(corrected.array).all()
    np.testing.assert_allclose(corrected.origin, scan_0.origin)
    np.testing.assert_allclose(corrected.sampling, scan_0.sampling)
    assert corrected.units == scan_0.units
    assert len(drift.drift_rate) == 2

    automatic = drift.corrected(upsample_factor=1, verbose=False)
    canvas = drift.corrected(
        upsample_factor=1,
        output_frame="canvas",
        verbose=False,
    )
    assert automatic.shape == tuple(drift.shape[-2:])
    assert canvas.shape == tuple(drift.shape[-2:])


def test_manual_translation_alignment_reduces_global_offset():
    """Manual translation alignment registers scans without fitting drift."""
    rng = np.random.default_rng(7)
    reference = gaussian_filter(
        rng.normal(size=(64, 64)).astype(np.float32),
        sigma=1.5,
    )
    moving = np.rot90(
        shift(
            reference,
            shift=(3.0, -4.0),
            order=1,
            mode="constant",
            cval=float(np.median(reference)),
        ),
        k=1,
    ).astype(np.float32)
    drift = DriftCorrection.from_images(
        reference,
        moving,
        scan_direction_degrees=(0.0, 90.0),
        device="cpu",
    )

    returned = drift.align_translation(
        max_image_shift=8,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    initial = drift.corrected(
        stage="initial",
        merge=False,
        output_frame="input",
        verbose=False,
    )
    aligned = drift.corrected(
        merge=False,
        output_frame="input",
        verbose=False,
    )
    interior = np.s_[8:-8, 8:-8]
    initial_error = np.mean(
        np.abs(initial[0].array[interior] - initial[1].array[interior])
    )
    aligned_error = np.mean(
        np.abs(aligned[0].array[interior] - aligned[1].array[interior])
    )

    assert returned is drift
    assert aligned_error < initial_error * 0.6


def test_nonrigid_diagnostic_defines_fast_roughness():
    """The difficult multi-knot diagnostic states exactly what roughness means."""
    doc = DriftCorrection.diagnose_nonrigid.__doc__ or ""
    assert "root-mean-square difference between neighboring knot" in doc
    assert "does not measure image noise" in doc


def test_paired_position_plot_uses_native_sites_and_measured_scan_directions():
    first, second = _orthogonal_pair()
    for scan in (first, second):
        scan.metadata["upsample"] = 2
    drift = DriftCorrection.from_images(first, second, device="cpu")
    drift.preprocess(show_combined=False, show_knots=False, verbose=False)
    fig, (before, after) = drift.plot_probe_positions(image_index=(0, 1), stride=4, zoom=4)
    try:
        for i, arrow in enumerate(before.texts):
            sites = drift.probe_positions(i, corrected=False)
            row, col = sites[0, -1] - sites[0, 0]
            expected = np.array([col, -row])
            np.testing.assert_allclose(np.subtract(arrow.xy, arrow.xyann), 0.13 * expected / np.linalg.norm(expected), atol=1e-6)
            np.testing.assert_allclose(before.lines[i].get_xdata(), sites[::4, ::4, 1].ravel() / 0.8, atol=1e-5)
        assert len(before.texts) == 2 and not after.texts
        assert np.isclose(np.diff(before.get_xlim()).item(), first.shape[1] / 4)
        assert before.get_ylim() == after.get_ylim() and before.yaxis_inverted()
    finally:
        plt.close(fig)


def test_probe_positions_returns_the_same_full_arrays_with_or_without_show(monkeypatch):
    first, second = _orthogonal_pair()
    drift = DriftCorrection.from_images(first, second, device="cpu")
    drift.preprocess(show_combined=False, show_knots=False, verbose=False)
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    expected = np.stack([drift.probe_positions(0), drift.probe_positions(1)])
    np.testing.assert_array_equal(drift.probe_positions(), expected)
    assert not figures
    np.testing.assert_array_equal(drift.probe_positions(show=True), expected)
    assert len(figures) == 1
    plt.close(figures[0])


@pytest.mark.parametrize("downsample", [1, 2])
def test_probe_positions_native_angstrom_raster(downsample):
    """Recover each measured site, including edges, in the first scan's axes."""
    scans = _orthogonal_pair()
    for scan, angle in zip(scans, [25, 115], strict=True):
        scan.metadata.update(upsample=2, scan_rotation_deg=angle)
    drift = DriftCorrection.from_images(*scans, device="cpu")
    drift.preprocess(downsample=downsample, verbose=False, show_knots=False)
    positions = drift.probe_positions(corrected=False)
    assert positions.shape == (2, 32, 32, 2)
    row, col = np.indices((32, 32))
    np.testing.assert_allclose(positions[0], np.stack((row, col), axis=-1) * 1.6, atol=2e-5)
    np.testing.assert_allclose(positions[1, 1] - positions[1, 0], np.tile([0, -1.6], (32, 1)), atol=2e-5)
