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

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.diffraction import BraggPeaksPolymer


@pytest.fixture
def bp():
    dataset = Dataset4dstem.from_array(
        array=np.zeros((1, 1, 8, 8), dtype=np.float32), name="ml_test",
        origin=(0, 0, 0, 0), sampling=(1.0, 1.0, 1.0, 1.0),
        units=["pixels"] * 4, signal_units="counts",
    )
    analysis = BraggPeaksPolymer.from_data(
        dataset, device="cpu",
        compute_parameters=lambda x, **kwargs: (0.0, 1.0),
        normalize_data=lambda x, lo, hi: x,
    )
    analysis._final_shape = (16, 16)
    return analysis


def _blob(shape=(32, 32), centres=((10.0, 12.0), (22.0, 20.0)), sigma=1.2):
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    image = np.zeros(shape, dtype=float)
    for cy, cx in centres:
        image += np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2))
    return image


def test_resize_images_shape_and_intensity_convention(bp):
    """Pins the existing intensity convention, which is NOT count-conserving.

    Bilinear interpolation to half the linear size already divides the sum by 4
    (each output pixel is the mean of its neighbourhood). The code then multiplies
    by ``scale_factor = (16*16)/(32*32) = 1/4`` as well, so the total comes out
    16x smaller. Conserving counts would need the reciprocal, x4.

    This has no effect on results, and not merely because training and inference
    share the pipeline: the factor is *annihilated* downstream. Resizing happens
    before normalization everywhere, and every normalization strategy here begins
    with a per-image min-max -- ``(x - min) / (max - min)`` is invariant under any
    positive uniform scaling, since min and max scale with it. Both reported
    intensity fields are post-normalization too (one from the model's intensity
    channel, one sampled from the already-normalized input), so no pre-normalization
    absolute intensity reaches any output.

    Worth keeping in mind only as latent fragility: ``scale_factor`` depends on the
    input detector size, so it is not constant across datasets. Harmless under any
    scale-equivariant normalization (per-image min-max and median/IQR both are), but
    a normalization with a fixed absolute divisor would make it matter, and matter
    differently per dataset.
    """
    images = np.stack([_blob((32, 32)), _blob((32, 32), centres=((5.0, 5.0),))])
    resized = bp.resize_images(images, device="cpu", initial_chunk_size=2)

    assert resized.shape == (2, 16, 16)
    scale_factor = (16 * 16) / (32 * 32)
    for original, small in zip(images, resized):
        assert small.sum() == pytest.approx(original.sum() * scale_factor / 4, rel=0.05)


def test_resize_images_accepts_a_dataset_with_an_array_attribute(bp):
    class Wrapped:
        array = np.stack([_blob((32, 32))])

    resized = bp.resize_images(Wrapped(), device="cpu")
    assert resized.shape == (1, 16, 16)


def test_postprocess_single_finds_blobs_and_samples_intensities(bp):
    position_map = _blob()
    intensity_map = np.full((32, 32), 3.0)

    coords, intensities = bp._postprocess_single(position_map, intensity_map,
                                                 sigma=1.0, threshold=0.25)

    assert len(coords) == 2
    assert coords.shape[1] == 2
    # Both injected centres should be recovered to within a pixel.
    found = sorted(tuple(np.round(c, 0)) for c in coords)
    assert found == [(10.0, 12.0), (22.0, 20.0)]
    # Intensities are sampled from the (uniform) intensity channel.
    np.testing.assert_allclose(intensities, 3.0, rtol=1e-6)


def test_postprocess_single_returns_empty_when_nothing_is_above_threshold(bp):
    coords, intensities = bp._postprocess_single(np.zeros((32, 32)), np.zeros((32, 32)))

    assert len(coords) == 0 and len(intensities) == 0
