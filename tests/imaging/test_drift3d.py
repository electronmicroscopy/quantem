"""Reference-based spectrum-image drift correction workflows."""

import json
import math

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.imaging.drift import (
    DriftCorrection,
    pair_spectrum_image_references,
    read_emd_eds,
)


def _column_drift(array: np.ndarray, rate: float) -> np.ndarray:
    """Apply a known scan-row-dependent column displacement."""
    scan_rows, scan_cols = array.shape[:2]
    columns = np.arange(scan_cols)
    drifted = np.empty_like(array)
    for scan_row in range(scan_rows):
        sample = np.clip(columns + rate * scan_row, 0, scan_cols - 1.001)
        lower = np.floor(sample).astype(int)
        fraction = sample - lower
        if array.ndim == 3:
            fraction = fraction[:, None]
        drifted[scan_row] = (
            array[scan_row, lower] * (1.0 - fraction)
            + array[scan_row, lower + 1] * fraction
        )
    return drifted


def test_reference_correction_preserves_spectra_and_calibration():
    """One HAADF-derived field corrects every spectrum channel identically."""
    scan_size = 80
    row, column = np.mgrid[:scan_size, :scan_size]
    rng = np.random.default_rng(4)
    reference = gaussian_filter(
        rng.normal(size=(scan_size, scan_size)).astype(np.float32),
        1.2,
    )
    for center_row, center_column in ((18, 20), (29, 61), (63, 24), (58, 65)):
        reference += 4.0 * np.exp(
            -(
                (row - center_row) ** 2
                + (column - center_column) ** 2
            )
            / (2.0 * 4.0**2)
        )
    clean_spectrum = np.stack(
        (
            reference,
            2.0 * reference + 3.0,
            np.zeros_like(reference),
            np.full_like(reference, 11.0),
        ),
        axis=-1,
    ).astype(np.float32)
    drift_rate = 0.16
    alignment_image = _column_drift(reference, drift_rate).astype(np.float32)
    spectrum_image = Dataset3d.from_array(
        _column_drift(clean_spectrum, drift_rate).astype(np.float32),
        name="SrTiO3 spectrum image",
        origin=[1.2, 2.4, 0.35],
        sampling=[0.08, 0.08, 0.01],
        units=["nm", "nm", "keV"],
        signal_units="counts",
    )
    spectrum_image.metadata.update(
        {"scan_rotation_deg": 0.0, "detector": "Super-X"}
    )

    drift = DriftCorrection.from_reference(
        reference,
        spectrum_image,
        alignment_image=alignment_image,
        scan_direction_degrees=0.0,
        device="cpu",
    ).preprocess(
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    initial_reference_knots = drift.knots[0].clone()
    drift.correct_affine(
        max_drift_rate=0.2,
        num_rates=11,
        refine=True,
        max_image_shift=16,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    corrected = drift.corrected(verbose=False)

    interior = np.s_[12:-12, 12:-12]
    raw_ncc = np.corrcoef(
        clean_spectrum[interior].ravel(),
        spectrum_image.array[interior].ravel(),
    )[0, 1]
    corrected_ncc = np.corrcoef(
        clean_spectrum[interior].ravel(),
        corrected.array[interior].ravel(),
    )[0, 1]
    assert corrected_ncc > raw_ncc + 0.02
    assert corrected_ncc > 0.99
    np.testing.assert_allclose(
        drift.knots[0].cpu(),
        initial_reference_knots.cpu(),
        atol=0.0,
    )
    np.testing.assert_allclose(
        corrected.array[..., 1],
        2.0 * corrected.array[..., 0] + 3.0,
        atol=2e-6,
    )
    np.testing.assert_array_equal(corrected.array[..., 2], 0.0)
    np.testing.assert_allclose(corrected.array[..., 3], 11.0, atol=2e-6)
    np.testing.assert_array_equal(corrected.origin, spectrum_image.origin)
    np.testing.assert_array_equal(corrected.sampling, spectrum_image.sampling)
    assert corrected.units == spectrum_image.units
    assert corrected.signal_units == spectrum_image.signal_units
    assert corrected.metadata == spectrum_image.metadata


def _write_metadata_emd(
    path,
    *,
    rotation_degrees: float,
    stage_position: tuple[float, float],
    spectrum_image: bool,
    timestamp: int,
    scan_shape: tuple[int, int] = (256, 256),
    pixel_size_m: float = 6.74e-9 / 256,
):
    """Write the minimal Velox metadata used by the pairing workflow."""
    metadata = {
        "Scan": {
            "ScanRotation": math.radians(rotation_degrees),
            "ScanSize": {"width": scan_shape[1], "height": scan_shape[0]},
        },
        "Optics": {"NominalMagnification": 15_000_000},
        "Stage": {
            "Position": {"x": stage_position[0], "y": stage_position[1]}
        },
        "BinaryResult": {"PixelSize": {"width": pixel_size_m}},
        "Acquisition": {
            "AcquisitionStartDatetime": {"DateTime": str(timestamp)}
        },
    }
    encoded = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    with h5py.File(path, "w") as handle:
        image = handle.create_group("Data/Image/0")
        image.create_dataset("Metadata", data=encoded)
        if spectrum_image:
            handle.create_group("Data/SpectrumImage")


def test_spectrum_image_pairing_uses_metadata_not_names_or_order(tmp_path):
    """A spectrum image is paired by rotation and stage metadata."""
    stage = (1.2e-6, -3.4e-6)
    _write_metadata_emd(
        tmp_path / "zzz_last_name.emd",
        rotation_degrees=0.0,
        stage_position=stage,
        spectrum_image=False,
        timestamp=30,
    )
    _write_metadata_emd(
        tmp_path / "aaa_first_name.emd",
        rotation_degrees=-90.0,
        stage_position=stage,
        spectrum_image=False,
        timestamp=10,
    )
    _write_metadata_emd(
        tmp_path / "middle_name.emd",
        rotation_degrees=0.0,
        stage_position=stage,
        spectrum_image=True,
        timestamp=20,
    )
    _write_metadata_emd(
        tmp_path / "nearby_but_wrong_area.emd",
        rotation_degrees=90.0,
        stage_position=(stage[0] + 100e-9, stage[1]),
        spectrum_image=False,
        timestamp=40,
    )
    _write_metadata_emd(
        tmp_path / "same_area_wrong_grid.emd",
        rotation_degrees=90.0,
        stage_position=stage,
        spectrum_image=False,
        timestamp=50,
        scan_shape=(128, 128),
        pixel_size_m=6.74e-9 / 128,
    )

    match = pair_spectrum_image_references(tmp_path)[0]

    assert match["status"] == "ready"
    assert match["spectrum_image"].name == "middle_name.emd"
    assert match["reference_zero"].name == "zzz_last_name.emd"
    assert match["reference_orthogonal"].name == "aaa_first_name.emd"


def test_spectrum_image_pairing_rejects_shared_reference_assignment(tmp_path):
    """One reference pair cannot be silently reused for multiple acquisitions."""
    stage = (1.2e-6, -3.4e-6)
    _write_metadata_emd(
        tmp_path / "reference_zero.emd",
        rotation_degrees=0.0,
        stage_position=stage,
        spectrum_image=False,
        timestamp=10,
    )
    _write_metadata_emd(
        tmp_path / "reference_orthogonal.emd",
        rotation_degrees=90.0,
        stage_position=stage,
        spectrum_image=False,
        timestamp=20,
    )
    for index in range(2):
        _write_metadata_emd(
            tmp_path / f"spectrum_{index}.emd",
            rotation_degrees=0.0,
            stage_position=stage,
            spectrum_image=True,
            timestamp=30 + index,
        )

    matches = pair_spectrum_image_references(tmp_path)

    assert len(matches) == 2
    assert {match["status"] for match in matches} == {"ambiguous"}
    assert all("matches 2 spectrum images" in match["reason"] for match in matches)
    assert all(match["reference_zero"] is None for match in matches)
    assert all(match["reference_orthogonal"] is None for match in matches)


def test_read_emd_eds_preserves_native_energy_axis(tmp_path, monkeypatch):
    """The EMD loader returns scan axes first and native energy calibration."""
    path = tmp_path / "spectrum.emd"
    _write_metadata_emd(
        path,
        rotation_degrees=0.0,
        stage_position=(0.0, 0.0),
        spectrum_image=True,
        timestamp=1,
    )
    native = np.arange(4 * 5 * 6, dtype=np.uint32).reshape(4, 5, 6)
    streams = [
        {
            "data": np.ones((5, 6), dtype=np.float32),
            "metadata": {"General": {"title": "HAADF"}},
            "axes": [
                {"index_in_array": 0, "scale": 0.2, "offset": 1.0, "units": "nm"},
                {"index_in_array": 1, "scale": 0.2, "offset": 2.0, "units": "nm"},
            ],
        },
        {
            "data": native,
            "metadata": {"General": {"title": "EDS"}},
            "axes": [
                {
                    "index_in_array": 0,
                    "name": "Energy",
                    "scale": 0.01,
                    "offset": 0.35,
                    "units": "keV",
                },
                {"index_in_array": 1, "scale": 0.2, "offset": 1.0, "units": "nm"},
                {"index_in_array": 2, "scale": 0.2, "offset": 2.0, "units": "nm"},
            ],
        },
    ]
    monkeypatch.setattr("rsciio.emd.file_reader", lambda *args, **kwargs: streams)

    acquisition = read_emd_eds(path, load_spectrum=True, verbose=False)
    spectrum = acquisition["spectrum"]

    assert isinstance(spectrum, Dataset3d)
    assert spectrum.shape == (5, 6, 4)
    np.testing.assert_array_equal(spectrum.array, np.moveaxis(native, 0, 2))
    np.testing.assert_array_equal(spectrum.origin, [1.0, 2.0, 0.35])
    np.testing.assert_array_equal(spectrum.sampling, [0.2, 0.2, 0.01])
    assert spectrum.units == ["nm", "nm", "keV"]


def test_read_emd_eds_extracts_requested_windows_without_dense_cube(
    tmp_path, monkeypatch
):
    """Requested EDS windows are counted directly from each sparse stream."""
    path = tmp_path / "spectrum_windows.emd"
    _write_metadata_emd(
        path,
        rotation_degrees=0.0,
        stage_position=(0.0, 0.0),
        spectrum_image=True,
        timestamp=1,
        scan_shape=(2, 3),
    )
    detector_metadata = {
        "BinaryResult": {"Detector": "SuperX-1"},
        "Detectors": {
            "Detector-0": {
                "DetectorName": "SuperX-1",
                "Dispersion": "100",
                "OffsetEnergy": "0",
            }
        },
    }
    acquisition_settings = {
        "bincount": "8",
        "StreamEncoding": "uint16",
        "RasterScanDefinition": {"Width": "3", "Height": "2"},
    }
    # Six pixels. Values other than 65535 are one X-ray count in that
    # energy-channel index; 65535 advances to the next scan pixel.
    stream = np.array(
        [1, 2, 65535, 2, 65535, 4, 65535, 1, 3, 65535, 2, 2, 65535, 65535],
        dtype=np.uint16,
    )
    with h5py.File(path, "a") as handle:
        group = handle.create_group("Data/SpectrumStream/stream-0")
        group.create_dataset(
            "AcquisitionSettings",
            data=np.array([json.dumps(acquisition_settings).encode()]),
        )
        encoded = np.frombuffer(json.dumps(detector_metadata).encode(), dtype=np.uint8)
        group.create_dataset("Metadata", data=encoded)
        group.create_dataset("Data", data=stream[:, None])

    streams = [
        {
            "data": np.ones((2, 3), dtype=np.float32),
            "metadata": {"General": {"title": "HAADF"}},
            "axes": [
                {"index_in_array": 0, "scale": 1.0, "offset": 0.0},
                {"index_in_array": 1, "scale": 1.0, "offset": 0.0},
            ],
        }
    ]
    monkeypatch.setattr("rsciio.emd.file_reader", lambda *args, **kwargs: streams)

    acquisition = read_emd_eds(
        path,
        energy_windows={"low": (0.1, 0.2), "high": (0.3, 0.4)},
        verbose=False,
    )

    np.testing.assert_array_equal(
        acquisition["window_maps"]["low"],
        [[2, 1, 0], [1, 2, 0]],
    )
    np.testing.assert_array_equal(
        acquisition["window_maps"]["high"],
        [[0, 0, 1], [1, 0, 0]],
    )
    np.testing.assert_allclose(
        acquisition["energy_axis_keV"],
        np.arange(8, dtype=np.float32) * np.float32(0.1),
    )
    assert acquisition["spectrum"] is None
