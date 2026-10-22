"""Baseline coverage for the flowline pipeline.

Written before the orientation extraction, so it pins current behaviour.
``make_flowline_map`` and ``make_flowline_rainbow_image`` each appear in 51
notebooks and had no tests; the other two and the intensity helpers had none either.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.diffraction import BraggPeaksPolymer


@pytest.fixture(scope="module")
def bp():
    dataset = Dataset4dstem.from_array(
        array=np.zeros((1, 1, 4, 4), dtype=np.float32), name="flowline_test",
        origin=(0, 0, 0, 0), sampling=(1.0, 1.0, 1.0, 1.0), units=["pixels"] * 4,
        signal_units="counts",
    )
    analysis = BraggPeaksPolymer.from_data(
        dataset, device="cpu",
        compute_parameters=lambda x, **kwargs: (0.0, 1.0),
        normalize_data=lambda x, lo, hi: x,
    )
    yield analysis
    plt.close("all")


@pytest.fixture(scope="module")
def orient_hist():
    """One radial bin, a 12x12 scan, 18 theta bins, with a coherent stripe.

    A uniform histogram seeds no flowlines, so give the tracer an aligned band.
    """
    hist = np.zeros((1, 12, 12, 18), dtype=float)
    hist[0, 3:9, :, 4] = 1.0
    hist[0, 3:9, :, 5] = 0.6
    return hist


@pytest.fixture(scope="module")
def flowlines(bp, orient_hist):
    return bp.make_flowline_map(orient_hist, thresh_seed=0.5, thresh_grow=0.2)


def test_make_flowline_map_traces_the_aligned_band(flowlines, orient_hist):
    assert flowlines.shape == orient_hist.shape
    assert np.all(np.isfinite(flowlines))
    # Signal concentrated in the band. Not exactly zero outside it: the tracer steps
    # along directions and set_intensity spreads weight over neighbouring bins.
    assert flowlines.sum() > 0
    assert flowlines[0, 3:9].sum() / flowlines.sum() > 0.8


def test_flowline_images_are_rgb_in_unit_range(bp, flowlines):
    images = {
        "rainbow": bp.make_flowline_rainbow_image(
            flowlines, int_range=[0, 1], plot_images=False, sum_radial_bins=True),
        "combined": bp.make_flowline_combined_image(
            flowlines, int_range=[0, 1], plot_images=False, sum_radial_bins=True),
        "legend": bp.make_flowline_rainbow_legend(
            im_size=np.array([32, 32]), return_image=True, plot_legend=False),
    }
    for name, image in images.items():
        assert image.ndim in (3, 4) and image.shape[-1] == 3, f"{name}: {image.shape}"
        assert image.min() >= 0.0 and image.max() <= 1.0, name
    assert images["legend"].shape == (32, 32, 3)


def test_intensity_helpers_round_trip(bp):
    """``set_intensity`` deposits interpolated weight that ``get_intensity`` reads back."""
    written = bp.set_intensity(np.zeros((6, 6, 8)), np.array([[2.0, 3.0, 4.0, 1.0]]))

    assert written.sum() == pytest.approx(1.0)
    assert bp.get_intensity(written, np.array([2.0]), np.array([3.0]),
                            np.array([4.0]))[0] == pytest.approx(1.0)
