"""Tests for BraggPeaksPolymer: model detection, amorphous-ring calibration, polar peaks."""

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Vector
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffraction.bragg_vectors import PEAK_FIELDS
from quantem.diffraction.calibration import _ellipse_matrix, ellipse_from_axes, q_to_r_matrix
from quantem.diffraction.polymer import BraggPeaksPolymer


class _BatchEcho(torch.nn.Module):
    """Returns its input, plus a little of the batch mean so results depend on the batch."""

    def forward(self, x):
        out = x + 0.2 * x.mean(dim=0, keepdim=True)
        return torch.cat([out, out], dim=1)


def _gaussians(shape, centers, sigma, amplitudes):
    rows, cols = np.mgrid[: shape[0], : shape[1]].astype(float)
    image = np.zeros(shape)
    for (r, c), a in zip(centers, amplitudes):
        image += a * np.exp(-((rows - r) ** 2 + (cols - c) ** 2) / (2 * sigma**2))
    return image


def test_find_peaks_model_maps_to_detector_pixels_and_replays_batches():
    rng = np.random.default_rng(1)
    scan, detector = (2, 3), (384, 512)  # non-square, both axes resized
    spots = rng.uniform([40, 40], [detector[0] - 40, detector[1] - 40], size=(*scan, 3, 2))
    data = np.zeros((*scan, *detector), np.float32)
    for r, c in np.ndindex(scan):
        data[r, c] = 0.02 + _gaussians(detector, spots[r, c], 3.0, [1, 1, 1])
    bp = BraggPeaksPolymer.from_dataset(Dataset4dstem.from_array(data), device="cpu")
    bp.model = _BatchEcho()
    mask = np.array([[True, True, False], [True, True, True]])
    peaks = bp.find_peaks_model(
        scan_mask=mask, batch_size=2, percentiles=(0, 100), progressbar=False
    )

    for r, c in zip(*np.nonzero(mask)):
        found = peaks[r, c].numpy().astype(float)[:, :2]
        assert len(found) == 3, (r, c, found)
        error = np.min(np.linalg.norm(found[:, None] - spots[r, c][None], axis=-1), axis=0)
        assert error.max() < 0.1, f"position ({r}, {c}): errors {error}"
    assert len(peaks[0, 2].numpy()) == 0  # outside the mask

    # (1, 1) runs in its scan batch, so it matches exactly; (0, 2) is outside the mask
    sub = bp.find_peaks_model(
        positions=[(1, 1), (0, 2)], batch_size=2, percentiles=(0, 100), progressbar=False
    )
    assert np.array_equal(sub[0].numpy(), peaks[1, 1].numpy())
    assert len(sub[1].numpy()) == 3


def test_calibrate_amorphous_ring_and_polar_peaks():
    rng = np.random.default_rng(2)
    scan, detector, pixel_size, rotation = (4, 5), (128, 128), 0.01, 30.0
    correction = _ellipse_matrix(ellipse_from_axes(1.03, 0.97, 25.0))
    rows, cols = np.mgrid[: detector[0], : detector[1]].astype(float)
    data = np.zeros((*scan, *detector), np.float32)
    cells, expected = [], []
    for r, c in np.ndindex(scan):
        origin = np.array([64.3 + 0.7 * r, 63.6 - 0.5 * c])
        d = np.stack([rows - origin[0], cols - origin[1]], axis=-1) @ correction.T
        rho, phi = np.hypot(d[..., 0], d[..., 1]), np.arctan2(d[..., 0], d[..., 1])
        ring = np.exp(-((rho - 38) ** 2) / 18) * (1 + 0.5 * np.cos(2 * (phi - 0.2)))
        data[r, c] = 0.5 + 10 * ring + 50 * np.exp(-(rho**2) / 8)
        u = rng.uniform(20, 50, 3)[:, None] * np.stack(
            [np.sin(a := rng.uniform(0, 2 * np.pi, 3)), np.cos(a)], 1
        )
        spots = origin + u @ np.linalg.inv(correction).T
        cells.append(np.vstack([[*origin, 50.0], np.column_stack([spots, np.ones(3)])]))
        g = u @ q_to_r_matrix(rotation, False).T
        expected.append(
            np.column_stack([np.hypot(*g.T) * pixel_size, np.mod(np.arctan2(*g.T), 2 * np.pi)])
        )
    ds = Dataset4dstem.from_array(
        data, sampling=(1, 1, pixel_size, pixel_size), units=("A", "A", "A^-1", "A^-1")
    )
    ds.metadata["q_to_r_rotation_ccw_deg"] = rotation
    bp = BraggPeaksPolymer.from_dataset(ds, device="cpu")
    bp.peaks = Vector.from_data(
        [cells[r * scan[1] : (r + 1) * scan[1]] for r in range(scan[0])],
        fields=PEAK_FIELDS,
        name="p",
    )

    bp.calibrate_amorphous_ring(fit_radii=(25, 52))
    polar = bp.polar_peaks().numpy().astype(float)
    counts = np.asarray(bp.peaks.row_counts())
    found = np.delete(polar, np.cumsum(np.r_[0, counts[:-1]]), axis=0)  # drop the central beams
    expected = np.concatenate(expected)
    assert np.allclose(found[:, 0], expected[:, 0], rtol=3e-3), np.abs(
        found[:, 0] / expected[:, 0] - 1
    ).max()
    d_theta = np.angle(np.exp(1j * (found[:, 1] - expected[:, 1])))
    assert np.abs(np.rad2deg(d_theta)).max() < 0.3


def test_pixel_size_from_detector_units():
    wavelength = electron_wavelength_angstrom(300e3)
    for units, sampling, energy, expected in [
        ("A^-1", 0.02, None, 0.02),
        ("1/nm", 0.1, None, 0.01),
        ("mrad", 0.5, 300e3, 0.5e-3 / wavelength),
    ]:
        ds = Dataset4dstem.from_array(
            np.ones((1, 1, 4, 4)),
            sampling=(1, 1, sampling, sampling),
            units=("A", "A", units, units),
        )
        bp = BraggPeaksPolymer.from_dataset(ds, device="cpu")
        assert np.isclose(bp._pixel_size(energy=energy), expected), units
    with pytest.raises(ValueError, match="energy"):
        bp._pixel_size()


def test_detect_ice_measures_widths_on_the_patterns():
    from quantem.diffraction.polymer_ice import IceFlaggerParams

    detector, radius = (96, 96), 30.0
    pixel_size = 1.61 / radius  # the ice ring at q_target
    rows, cols = np.mgrid[: detector[0], : detector[1]].astype(float)
    rho, phi = np.hypot(rows - 48, cols - 48), np.arctan2(rows - 48, cols - 48)
    arms = np.deg2rad(10 + 60 * np.arange(6))
    spots = 48 + radius * np.stack([np.sin(arms), np.cos(arms)], axis=1)
    sharp = _gaussians(detector, spots, 1.2, np.ones(6))
    d = np.angle(np.exp(1j * (phi[..., None] - arms)))
    broad = np.exp(-((rho - radius) ** 2) / 3) * np.exp(-(d**2) / (2 * np.deg2rad(15) ** 2)).sum(
        -1
    )
    data = np.stack([sharp, broad])[None].astype(np.float32) + 0.05
    ds = Dataset4dstem.from_array(
        data, sampling=(1, 1, pixel_size, pixel_size), units=("A", "A", "A^-1", "A^-1")
    )
    bp = BraggPeaksPolymer.from_dataset(ds, device="cpu")
    cell = np.vstack([[48.0, 48.0, 50.0], np.column_stack([spots, np.ones(6)])])
    bp.peaks = Vector.from_data([[cell, cell]], fields=PEAK_FIELDS, name="p")

    params = IceFlaggerParams(intensity_cutoff=0.5, max_width_theta_deg=10.0)
    result = bp.detect_ice(params)
    # same positions on both patterns: only the sharp spots pass the width gate
    assert result.flagged_count_map.tolist() == [[6, 0]]
    assert [
        len(c)
        for c in (result.filter(bp.peaks)[0, 0].numpy(), result.filter(bp.peaks)[0, 1].numpy())
    ] == [1, 7]
