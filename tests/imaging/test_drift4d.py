"""4D-STEM drift propagation keeps detector coordinates scientifically intact."""

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.imaging.drift import CorrectionResult, DriftCorrection


def _orthogonal_4dstem_pair(
    scan_size: int = 24,
    detector_shape: tuple[int, int] = (4, 5),
):
    """Build two orthogonal scans with fixed per-detector-pixel signatures."""
    rng = np.random.default_rng(14)
    image_0 = gaussian_filter(
        rng.normal(size=(scan_size, scan_size)).astype(np.float32),
        1.2,
    )
    image_0 += np.linspace(0, 2, scan_size, dtype=np.float32)[:, None]
    image_1 = np.rot90(image_0, k=-1).copy()
    detector_offset = np.arange(
        np.prod(detector_shape),
        dtype=np.float32,
    ).reshape(detector_shape)
    cube_0 = image_0[..., None, None] + detector_offset
    cube_1 = image_1[..., None, None] + detector_offset
    return cube_0, cube_1, detector_offset


def _fit_small_pair(cube_0, cube_1):
    drift = DriftCorrection.from_4dstem(
        cube_0,
        cube_1,
        scan_direction_degrees=(0.0, 90.0),
        scan_sampling=0.2,
        scan_units="nm",
        device="cpu",
    ).preprocess(
        padding_fraction=0.25,
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
        max_image_shift=8,
        chunk_size=1,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    return drift


def _add_known_raw_drift(drift, image_index, row, column):
    """Move canvas knots by one requested raw-frame trajectory."""
    scan_rows, scan_cols = drift.imgs[image_index].shape
    aspect = (scan_rows - 1) / (scan_cols - 1)
    slow = drift.scan_slow[image_index]
    fast = drift.scan_fast[image_index]
    row = torch.as_tensor(
        row,
        dtype=drift.knots[image_index].dtype,
        device=drift.knots[image_index].device,
    )
    column = torch.as_tensor(
        column,
        dtype=drift.knots[image_index].dtype,
        device=drift.knots[image_index].device,
    )
    drift.knots[image_index][0, :, 0] += slow[0] * row + fast[0] * aspect * column
    drift.knots[image_index][1, :, 0] += slow[1] * row + fast[1] * column
    drift._images_warped_stale = True


def test_virtual_detector_matches_numpy_and_torch_integer_inputs():
    """Virtual integration has the same exact integer sum on both backends."""
    data = np.arange(3 * 4 * 2 * 3, dtype=np.uint16).reshape(3, 4, 2, 3)
    mask = np.array([[True, False, True], [False, True, False]])
    expected = data[..., mask].sum(axis=-1, dtype=np.uint64).astype(np.float32)

    numpy_image = DriftCorrection.integrate_virtual_detector(
        data,
        mask,
        reduce="sum",
    )
    torch_image = DriftCorrection.integrate_virtual_detector(
        torch.from_numpy(data),
        mask,
        reduce="sum",
    )

    np.testing.assert_array_equal(numpy_image, expected)
    np.testing.assert_array_equal(torch_image, expected)


def test_corrected_4dstem_transforms_scan_axes_not_detector_axes():
    """Every detector pixel receives one shared scan transform."""
    cube_0, cube_1, detector_offset = _orthogonal_4dstem_pair()
    drift = _fit_small_pair(cube_0, cube_1)
    line_drift = np.linspace(-1.5, 1.5, cube_0.shape[0])
    _add_known_raw_drift(drift, 0, line_drift, 0.5 * line_drift)

    result = drift.corrected_4dstem(chunk_size=5, verbose=False)

    assert isinstance(result, CorrectionResult)
    assert result.corrected_4dstem_0.shape == cube_0.shape
    assert result.corrected_4dstem_1.shape == cube_1.shape
    assert result.corrected_4dstem.shape == cube_0.shape
    for corrected in (result.corrected_4dstem_0, result.corrected_4dstem_1):
        detector_difference = corrected - corrected[..., :1, :1]
        np.testing.assert_allclose(
            detector_difference,
            np.broadcast_to(
                detector_offset - detector_offset[0, 0],
                corrected.shape,
            ),
            atol=2e-5,
        )


def test_regional_patterns_average_native_detector_samples():
    """Region membership changes, but diffraction pixels are not interpolated."""
    cube_0, cube_1, _ = _orthogonal_4dstem_pair(scan_size=16)
    drift = DriftCorrection.from_4dstem(
        cube_0,
        cube_1,
        scan_direction_degrees=(0.0, 90.0),
        device="cpu",
    ).preprocess(
        num_knots=1,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    _add_known_raw_drift(
        drift,
        0,
        np.full(16, 2.0),
        np.zeros(16),
    )
    regions = {"feature": (8.0, 8.0)}

    comparison = drift.regional_diffraction_patterns(
        regions,
        radius_px=2.0,
        stages=("initial", "corrected"),
    )

    assert comparison["patterns"].shape == (2, 1, 2, 4, 5)
    for stage_index, corrected in enumerate((False, True)):
        for scan_index, cube in enumerate((cube_0, cube_1)):
            positions = drift.probe_positions(
                scan_index,
                corrected=corrected,
                strip_padding=True,
                plot=False,
            )
            mask = (
                (positions[..., 0] - 8.0) ** 2
                + (positions[..., 1] - 8.0) ** 2
                <= 2.0**2
            )
            np.testing.assert_allclose(
                comparison["patterns"][stage_index, 0, scan_index],
                cube[mask].mean(axis=0, dtype=np.float32),
            )
            assert comparison["sample_counts"][stage_index, 0, scan_index] == mask.sum()
    assert not np.array_equal(
        comparison["patterns"][0, 0, 0],
        comparison["patterns"][1, 0, 0],
    )


def test_canvas_combination_uses_union_coverage():
    """The combined canvas retains pixels covered by either corrected scan."""
    cube_0, cube_1, _ = _orthogonal_4dstem_pair(scan_size=16)
    drift = _fit_small_pair(cube_0, cube_1)
    image_0 = drift.integrate_virtual_detector(cube_0, np.ones((4, 5), dtype=bool))
    image_1 = drift.integrate_virtual_detector(cube_1, np.ones((4, 5), dtype=bool))

    result = drift.corrected_virtual_images(
        image_0,
        image_1,
        output_frame="canvas",
    )

    expected_union = np.maximum(
        result["coverage_image_0"],
        result["coverage_image_1"],
    )
    np.testing.assert_allclose(result["coverage_image"], expected_union)
    either_scan = expected_union >= 1e-3
    assert np.count_nonzero(result["corrected_image"][either_scan]) > 0


def test_saved_correction_accepts_explicit_4dstem_datasets():
    """Serialized corrections can analyze explicitly reattached raw cubes."""
    cube_0, cube_1, _ = _orthogonal_4dstem_pair(scan_size=16)
    drift = DriftCorrection.from_4dstem(
        cube_0,
        cube_1,
        scan_direction_degrees=(0.0, 90.0),
        device="cpu",
    ).preprocess(
        num_knots=1,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    drift._datasets = None

    result = drift.regional_diffraction_patterns(
        {"feature": (8.0, 8.0)},
        radius_px=2.0,
        datasets=(cube_0, cube_1),
        stages=("initial",),
    )

    assert result["patterns"].shape == (1, 1, 2, 4, 5)


def test_numpy_cube_can_return_torch_output_on_requested_device():
    """An explicit output device is honored without changing detector layout."""
    cube_0, cube_1, _ = _orthogonal_4dstem_pair(scan_size=16)
    drift = _fit_small_pair(cube_0, cube_1)

    result = drift.corrected_4dstem(
        merge=False,
        output_device="cpu",
        output_dtype=np.float32,
        verbose=False,
    )

    assert isinstance(result.corrected_4dstem_0, torch.Tensor)
    assert isinstance(result.corrected_4dstem_1, torch.Tensor)
    assert result.corrected_4dstem_0.device.type == "cpu"
    assert result.corrected_4dstem_0.shape == cube_0.shape


def test_dataset4dstem_metadata_supplies_rotation_and_scan_calibration():
    """QuantEM datasets retain scan metadata while exposing resident data."""
    cube_0, cube_1, _ = _orthogonal_4dstem_pair(scan_size=16)
    datasets = []
    for cube, angle in ((cube_0, 0.0), (cube_1, 90.0)):
        dataset = Dataset4dstem.from_array(
            cube,
            sampling=(0.2, 0.3, 0.01, 0.01),
            units=("nm", "nm", "1/nm", "1/nm"),
        )
        dataset.metadata["scan_rotation_deg"] = angle
        datasets.append(dataset)

    drift = DriftCorrection.from_4dstem(*datasets, device="cpu")

    np.testing.assert_allclose(drift.scan_direction_degrees, (0.0, 90.0))
    np.testing.assert_allclose(drift.imgs[0].sampling, (0.2, 0.3))
    assert drift.imgs[0].units == ["nm", "nm"]
    assert drift._datasets[0] is datasets[0].array


def test_drift_field_reports_raw_components_for_rotated_scan():
    """A 90-degree scan reports raw row/column drift, not canvas axes."""
    cube_0, cube_1, _ = _orthogonal_4dstem_pair(scan_size=16)
    drift = DriftCorrection.from_4dstem(
        cube_0,
        cube_1,
        scan_direction_degrees=(0.0, 90.0),
        device="cpu",
    ).preprocess(
        num_knots=1,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    expected_row = np.linspace(-2.0, 2.0, 16)
    expected_column = np.linspace(1.0, -1.0, 16)
    _add_known_raw_drift(drift, 1, expected_row, expected_column)

    field = drift.drift_field(1).cpu().numpy()

    np.testing.assert_allclose(field[0], expected_row, atol=2e-6)
    np.testing.assert_allclose(field[1], expected_column, atol=2e-6)


def test_virtual_detector_integration_commutes_with_scan_warp():
    """Integrating detector pixels before or after correction is equivalent."""
    cube_0, cube_1, _ = _orthogonal_4dstem_pair(scan_size=16)
    drift = _fit_small_pair(cube_0, cube_1)
    line_drift = np.linspace(-1.0, 1.0, 16)
    _add_known_raw_drift(drift, 0, line_drift, -0.25 * line_drift)
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:4] = True
    virtual_0 = drift.integrate_virtual_detector(cube_0, mask, reduce="sum")
    virtual_1 = drift.integrate_virtual_detector(cube_1, mask, reduce="sum")

    corrected_cube = drift.corrected_4dstem(merge=False, verbose=False)
    corrected_virtual = drift.corrected_virtual_images(virtual_0, virtual_1)

    np.testing.assert_allclose(
        drift.integrate_virtual_detector(
            corrected_cube.corrected_4dstem_0,
            mask,
            reduce="sum",
        ),
        corrected_virtual["corrected_image_0"],
        atol=2e-4,
    )


def test_preallocated_output_streams_without_full_device_allocation(monkeypatch):
    """A supplied output receives detector chunks without a full device cube."""
    cube_0, cube_1, _ = _orthogonal_4dstem_pair(scan_size=16)
    drift = _fit_small_pair(cube_0, cube_1)
    output_0 = np.empty_like(cube_0, dtype=np.float32)
    output_1 = np.empty_like(cube_1, dtype=np.float32)

    def reject_full_allocation(*shape, **kwargs):
        requested = (
            tuple(shape[0])
            if len(shape) == 1 and not isinstance(shape[0], int)
            else tuple(shape)
        )
        if requested in {cube_0.shape, (16, 16, 20)}:
            raise AssertionError("attempted full corrected-cube allocation")
        return original_empty(*shape, **kwargs)

    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", reject_full_allocation)
    corrected = drift.corrected_4dstem(
        merge=False,
        output_0=output_0,
        output_1=output_1,
        chunk_size=3,
        verbose=False,
    )

    assert corrected.corrected_4dstem_0 is output_0
    assert np.isfinite(output_1).all()
    assert corrected.corrected_4dstem_1.shape == output_1.shape
    assert np.isfinite(output_0).all()


def test_integer_merge_is_float32_and_backend_consistent():
    """Integer acquisitions retain half-counts in one float32 merge policy."""
    cube_0, cube_1, _ = _orthogonal_4dstem_pair(scan_size=16)
    minimum = min(float(cube_0.min()), float(cube_1.min()))
    cubes_np = [
        np.round((cube - minimum + 1.0) * 100).astype(np.uint16)
        for cube in (cube_0, cube_1)
    ]
    drift_np = _fit_small_pair(*cubes_np)
    result_np = drift_np.corrected_4dstem(
        output_dtype="same",
        verbose=False,
    )
    cubes_torch = [torch.from_numpy(cube) for cube in cubes_np]
    drift_torch = _fit_small_pair(*cubes_torch)
    result_torch = drift_torch.corrected_4dstem(
        output_dtype="same",
        verbose=False,
    )

    assert result_np.corrected_4dstem.dtype == np.float32
    assert result_torch.corrected_4dstem.dtype == torch.float32
    np.testing.assert_allclose(
        result_np.corrected_4dstem,
        result_torch.corrected_4dstem.cpu().numpy(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matches_cpu_for_uint32_native_detector_frames():
    """CUDA and CPU preserve one native 192-square detector field equally."""
    scan_size = 8
    detector_shape = (192, 192)
    row = np.arange(scan_size, dtype=np.uint32)[:, None, None, None]
    column = np.arange(scan_size, dtype=np.uint32)[None, :, None, None]
    detector = np.arange(
        np.prod(detector_shape),
        dtype=np.uint32,
    ).reshape(1, 1, *detector_shape)
    cube_0 = row * 100_000 + column * 10_000 + detector
    cube_1 = np.rot90(cube_0, k=-1, axes=(0, 1)).copy()
    cpu = DriftCorrection.from_4dstem(
        cube_0,
        cube_1,
        scan_direction_degrees=(0.0, 90.0),
        device="cpu",
    ).preprocess(
        num_knots=1,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    cuda = DriftCorrection.from_4dstem(
        torch.from_numpy(cube_0).cuda(),
        torch.from_numpy(cube_1).cuda(),
        scan_direction_degrees=(0.0, 90.0),
        device="cuda",
    ).preprocess(
        num_knots=1,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    drift_row = np.linspace(-0.75, 0.75, scan_size)
    drift_column = np.linspace(0.5, -0.5, scan_size)
    _add_known_raw_drift(cpu, 0, drift_row, drift_column)
    _add_known_raw_drift(cuda, 0, drift_row, drift_column)

    cpu_result = cpu.corrected_4dstem(
        merge=False,
        output_dtype=np.float32,
        chunk_size=4096,
        verbose=False,
    )
    cuda_result = cuda.corrected_4dstem(
        merge=False,
        output_dtype=torch.float32,
        chunk_size=4096,
        verbose=False,
    )

    np.testing.assert_allclose(
        cpu_result.corrected_4dstem_0,
        cuda_result.corrected_4dstem_0.cpu().numpy(),
        rtol=3e-7,
        atol=0.125,
    )
