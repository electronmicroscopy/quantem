"""Baseline coverage for the model-inference support functions.

Written before extracting them. Neither had any tests: ``resize_images`` prepares
every diffraction pattern for the network, and ``_postprocess_single`` turns the
network's two output channels into peak coordinates and intensities -- so between
them they bracket the whole ML detection path.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.diffraction import BraggPeaksPolymer
from quantem.diffraction.peak_detection import peaks_from_model_output


@pytest.fixture
def bp():
    dataset = Dataset4dstem.from_array(
        array=np.zeros((1, 1, 8, 8), dtype=np.float32),
        name="ml_test",
        origin=(0, 0, 0, 0),
        sampling=(1.0, 1.0, 1.0, 1.0),
        units=["pixels"] * 4,
        signal_units="counts",
    )
    analysis = BraggPeaksPolymer.from_data(
        dataset,
        device="cpu",
        compute_parameters=lambda x, **kwargs: (0.0, 1.0),
        normalize_data=lambda x, lo, hi: x,
    )
    analysis._final_shape = (16, 16)
    return analysis


def _blob(shape=(32, 32), centres=((10.0, 12.0), (22.0, 20.0)), sigma=1.2):
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    image = np.zeros(shape, dtype=float)
    for cy, cx in centres:
        image += np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2))
    return image


def test_resize_images_intensity_convention(bp):
    """Pins the existing intensity convention, which is NOT count-conserving.

    Bilinear resizing to half the linear size already divides the sum by 4, and the
    code then multiplies by ``scale_factor = (16*16)/(32*32) = 1/4`` too, so the total
    is 16x smaller (conserving counts would need x4). Harmless today: resizing precedes
    a per-image min-max normalization, which cancels any uniform scale. It is latent
    fragility only, since ``scale_factor`` depends on the detector size and would matter
    under a normalization with a fixed absolute divisor.
    """

    class Wrapped:  # resize_images also accepts anything with an ``array`` attribute
        array = np.stack([_blob((32, 32))])

    cases = {
        "ndarray": np.stack([_blob((32, 32)), _blob((32, 32), centres=((5.0, 5.0),))]),
        "wrapped": Wrapped(),
    }
    scale_factor = (16 * 16) / (32 * 32)
    for name, images in cases.items():
        resized = bp.resize_images(images, device="cpu", initial_chunk_size=2)
        originals = getattr(images, "array", images)
        assert resized.shape == (len(originals), 16, 16), name
        for original, small in zip(originals, resized):
            assert small.sum() == pytest.approx(original.sum() * scale_factor / 4, rel=0.05), name


def test_peaks_from_model_output_finds_blobs_and_samples_intensities(bp):
    # show=True used to raise NameError (matplotlib not imported) once a peak was found.
    for show in (False, True):
        coords, intensities = peaks_from_model_output(
            _blob(), np.full((32, 32), 3.0), sigma=1.0, threshold=0.25, show=show
        )
        plt.close("all")

        # Both injected centres should be recovered to within a pixel.
        found = sorted(tuple(np.round(c, 0)) for c in coords)
        assert found == [(10.0, 12.0), (22.0, 20.0)], f"show={show}"
        # Intensities are sampled from the (uniform) intensity channel.
        np.testing.assert_allclose(intensities, 3.0, rtol=1e-6, err_msg=f"show={show}")

    # The method wrapper forwards, and nothing above threshold gives empty output.
    coords, intensities = bp._postprocess_single(np.zeros((32, 32)), np.zeros((32, 32)))
    assert len(coords) == 0 and len(intensities) == 0
