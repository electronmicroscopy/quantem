"""Baseline coverage for the flowline pipeline.

Written before the orientation extraction, so it pins current behaviour rather
than post-hoc behaviour. ``make_flowline_map`` and ``make_flowline_rainbow_image``
each appear in 51 notebooks and had no tests at all; the other two and the two
intensity helpers had none either.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.diffraction import BraggPeaksPolymer


@pytest.fixture
def bp():
    array = np.zeros((1, 1, 4, 4), dtype=np.float32)
    dataset = Dataset4dstem.from_array(
        array=array, name="flowline_test", origin=(0, 0, 0, 0),
        sampling=(1.0, 1.0, 1.0, 1.0), units=["pixels"] * 4, signal_units="counts",
    )
    analysis = BraggPeaksPolymer.from_data(
        dataset, device="cpu",
        compute_parameters=lambda x, **kwargs: (0.0, 1.0),
        normalize_data=lambda x, lo, hi: x,
    )
    yield analysis
    plt.close("all")


@pytest.fixture
def orient_hist():
    """One radial bin, a 12x12 scan, 18 theta bins, with a coherent stripe.

    A uniform histogram seeds no flowlines, so give it a band of aligned
    intensity for the tracer to follow.
    """
    hist = np.zeros((1, 12, 12, 18), dtype=float)
    hist[0, 3:9, :, 4] = 1.0
    hist[0, 3:9, :, 5] = 0.6
    return hist


def test_make_flowline_map_traces_the_aligned_band(bp, orient_hist):
    flowlines = bp.make_flowline_map(orient_hist, thresh_seed=0.5, thresh_grow=0.2)

    assert flowlines.shape == orient_hist.shape
    assert np.all(np.isfinite(flowlines))
    # The stripe must produce signal, concentrated where the histogram had some.
    # Not exactly zero outside the band: the tracer steps along directions and
    # set_intensity spreads weight over neighbouring bins, so a little bleeds out.
    assert flowlines.sum() > 0
    assert flowlines[0, 3:9].sum() / flowlines.sum() > 0.8


def test_rainbow_image_shape_and_range(bp, orient_hist):
    flowlines = bp.make_flowline_map(orient_hist, thresh_seed=0.5, thresh_grow=0.2)
    image = bp.make_flowline_rainbow_image(
        flowlines, int_range=[0, 1], plot_images=False, sum_radial_bins=True)

    assert image.ndim in (3, 4) and image.shape[-1] == 3
    assert image.min() >= 0.0 and image.max() <= 1.0


def test_combined_image_shape(bp, orient_hist):
    flowlines = bp.make_flowline_map(orient_hist, thresh_seed=0.5, thresh_grow=0.2)
    image = bp.make_flowline_combined_image(
        flowlines, int_range=[0, 1], plot_images=False, sum_radial_bins=True)

    assert image.shape[-1] == 3
    assert image.min() >= 0.0 and image.max() <= 1.0


def test_rainbow_legend_returns_an_image(bp):
    legend = bp.make_flowline_rainbow_legend(
        im_size=np.array([32, 32]), return_image=True, plot_legend=False)

    assert legend.shape == (32, 32, 3)
    assert legend.min() >= 0.0 and legend.max() <= 1.0


def test_intensity_helpers_round_trip(bp):
    """``set_intensity`` deposits interpolated weight that ``get_intensity`` reads back."""
    orient = np.zeros((6, 6, 8), dtype=float)
    written = bp.set_intensity(orient, np.array([[2.0, 3.0, 4.0, 1.0]]))

    assert written.sum() == pytest.approx(1.0)
    assert bp.get_intensity(written, np.array([2.0]), np.array([3.0]),
                            np.array([4.0]))[0] == pytest.approx(1.0)
