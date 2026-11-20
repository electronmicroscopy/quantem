"""Tests for radial peak windows, orientation histograms and flowlines."""

import numpy as np

from quantem.core.datastructures import Vector
from quantem.diffraction.polymer_orientation import (
    estimate_peak_windows,
    make_flowline_map,
    make_orientation_histogram,
)


def _peaks(cells, shape):
    nested = [
        [cells.get((r, c), np.zeros((0, 3))) for c in range(shape[1])] for r in range(shape[0])
    ]
    return Vector.from_data(nested, fields=["q", "theta", "intensity"], name="p")


def test_estimate_peak_windows_orders_families_by_q():
    rng = np.random.default_rng(0)
    q = np.concatenate([rng.normal(0.42, 0.004, 2000), rng.normal(0.25, 0.006, 300)])
    peaks = _peaks({(0, 0): np.column_stack([q, np.zeros_like(q), np.ones_like(q)])}, (1, 1))
    centers, windows, info = estimate_peak_windows(peaks, q_range=(0.1, 0.5), verbose=False)
    assert np.allclose(centers, [0.25, 0.42], atol=0.005), centers
    assert np.all((windows[:, 0] < centers) & (centers < windows[:, 1]))
    # prominences follow the centers: the strong family is the second
    assert info["prominences"][1] > info["prominences"][0], info["prominences"]
    # a minimum width below one bin used to raise in find_peaks
    centers, _, _ = estimate_peak_windows(
        peaks, num_bins=20, q_range=(0.1, 0.5), min_width=1e-4, verbose=False
    )
    assert np.isclose(centers, 0.42, atol=0.02).any(), centers


def test_orientation_histogram_places_and_folds_peaks():
    theta = np.deg2rad(30.0)
    cells = {
        (1, 2): np.array([[0.25, theta, 1.0], [0.25, theta + np.pi, 1.0], [0.42, theta, 5.0]]),
    }
    hist = make_orientation_histogram(
        _peaks(cells, (3, 4)), [0.2, 0.3], sigma_x=0, sigma_y=0, sigma_theta=0, normalize=None
    )
    # the Friedel pair folds onto one bin; the peak outside the window is left out
    assert np.isclose(hist.sum(), 2.0), hist.sum()
    weight = hist[0].sum(axis=-1)
    rows, cols = np.nonzero(weight)
    assert np.isclose((rows * weight[rows, cols]).sum() / 2.0, 5.5)
    assert np.isclose((cols * weight[rows, cols]).sum() / 2.0, 9.5)
    assert np.isclose(hist[0, ..., 30].sum(), 2.0)


def _field(theta, value, num_theta=90):
    bins = np.arange(num_theta) * np.pi / num_theta
    d = np.angle(np.exp(2j * (bins - theta[..., None]))) / 2
    return (value[..., None] * np.exp(-0.5 * (d / np.deg2rad(4)) ** 2))[None]


def test_flowlines_near_zero_and_along_curves():
    rows, cols = np.mgrid[:24, :24].astype(float)
    # scattering at 0.5 degrees: lines run along the rows (the old code drew none)
    flow = make_flowline_map(
        _field(np.full_like(rows, np.deg2rad(0.5)), np.ones_like(rows)), progress_bar=False
    )
    image = flow[0].sum(-1)
    assert image.sum() > 0
    along, across = image[4:-4].sum(axis=1), image.sum(axis=0)
    assert along.std() / along.mean() < 0.1 < across.std() / across.mean()

    # a single seed on a quarter circle: both halves of the line follow the arc
    # (the old forward half kept its first direction and left the arc)
    phi = np.arctan2(rows - 1, cols - 1)
    seed = np.abs(phi - np.deg2rad(54.5)) < np.deg2rad(6)
    value = np.exp(-0.5 * ((np.hypot(rows - 1, cols - 1) - 16) / 1.5) ** 2) * np.where(
        seed, 1.0, 0.5
    )
    flow = make_flowline_map(
        _field(phi, value), thresh_seed=0.6, thresh_grow=0.1, progress_bar=False
    )
    image = flow[0].sum(-1)
    drawn = np.rad2deg(phi[image > 0.05 * image.max()])
    assert drawn.min() < 25 and drawn.max() > 85, (drawn.min(), drawn.max())
