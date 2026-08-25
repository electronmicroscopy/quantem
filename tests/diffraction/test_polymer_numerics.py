import numpy as np

from quantem.diffraction.peak_detection import detect_blobs
from quantem.diffraction.polar_transform import polar_transform
from quantem.diffraction.polymer_utils import parse_reciprocal_units


def test_reciprocal_unit_conversion_is_explicit():
    assert parse_reciprocal_units("nm^-1") == ("1/nm", 0.1)
    assert parse_reciprocal_units("Å⁻¹") == ("1/A", 1.0)


def test_peak_coordinates_remain_row_column_order():
    yy, xx = np.mgrid[:17, :19]
    image = np.exp(-((yy - 6.25) ** 2 + (xx - 11.4) ** 2) / 2.0)
    peaks, _, success = detect_blobs(image, sigma=0.5, threshold=0.2)
    assert success.tolist() == [True]
    np.testing.assert_allclose(peaks[0], [6.25, 11.4], atol=0.15)


def test_polar_transform_orientation_and_shape():
    data = np.zeros((1, 1, 15, 15), dtype=np.float32)
    data[0, 0, 7, 11] = 1.0
    polar = polar_transform(
        data, origin_array=np.array([7.0, 7.0]), num_annular_bins=8,
        radial_min=0, radial_max=7, radial_step=1, device="cpu", show_progress=False,
    )
    assert polar.array.shape == (1, 1, 8, 7)
    assert np.unravel_index(np.argmax(polar.array[0, 0]), (8, 7)) == (0, 4)
