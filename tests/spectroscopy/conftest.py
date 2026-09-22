"""Shared fixtures for the spectroscopy test suite.

Covers both:
- EELS: the ``Dataset3deels`` ZLP-offset robustness fix (``measure_zlp_offset`` /
  ``calculate_thickness_log_ratio``) -- synthetic scans with a known, exactly-linear
  ZLP-center plane so recovered surface-fit coefficients can be checked against ground truth.
- XEDS: synthetic spectra built from the real X-ray line database (``xeds_factory``,
  ``line_spectrum_factory``) for line-lookup/peak-fitting/background tests.
"""

import numpy as np
import pytest

from quantem.spectroscopy import Dataset3dxeds
from quantem.spectroscopy.dataset3deels import Dataset3deels

# ---------------------------------------------------------------------------
# EELS: tilted zero-loss-peak plane
# ---------------------------------------------------------------------------

SCAN_ROW = 12
SCAN_COL = 12
N_ENERGY = 400
ENERGY_LO = -2.0
ENERGY_HI = 2.0

# Ground-truth plane: mu(row, col) = A_TRUE * row + B_TRUE * col + C_TRUE
A_TRUE = 0.01
B_TRUE = -0.005
C_TRUE = 0.02
SIGMA_TRUE = 0.1
AMP_TRUE = 1000.0
NOISE_STD = 0.5


def energy_axis() -> np.ndarray:
    return np.linspace(ENERGY_LO, ENERGY_HI, N_ENERGY)


def make_tilted_zlp_scan(
    a: float = A_TRUE,
    b: float = B_TRUE,
    c: float = C_TRUE,
    sigma: float = SIGMA_TRUE,
    amp: float = AMP_TRUE,
    noise: float = NOISE_STD,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (scan_row, scan_col, n_energy) array of Gaussian ZLPs whose
    centers lie exactly on the plane ``mu(row, col) = a*row + b*col + c``.

    Returns ``(array, plane)``, where ``plane`` is the (scan_row, scan_col)
    ground-truth ZLP center (eV) at every pixel.
    """
    rng = np.random.default_rng(seed)
    energy = energy_axis()
    array = np.empty((SCAN_ROW, SCAN_COL, N_ENERGY))
    plane = np.empty((SCAN_ROW, SCAN_COL))
    for i in range(SCAN_ROW):
        for j in range(SCAN_COL):
            mu = a * i + b * j + c
            plane[i, j] = mu
            array[i, j, :] = amp * np.exp(-0.5 * ((energy - mu) / sigma) ** 2)
            array[i, j, :] += rng.normal(0, noise, N_ENERGY)
    return array, plane


def dataset_from_array(array: np.ndarray) -> Dataset3deels:
    energy = energy_axis()
    return Dataset3deels.from_array(
        array=array,
        sampling=[1, 1, (energy[-1] - energy[0]) / (N_ENERGY - 1)],
        origin=[0, 0, energy[0]],
        units=["px", "px", "eV"],
    )


@pytest.fixture(scope="module")
def tilted_zlp_scan() -> tuple[np.ndarray, np.ndarray]:
    """Clean synthetic (array, ground_truth_plane) with a known tilted ZLP plane."""
    return make_tilted_zlp_scan()


@pytest.fixture
def tilted_zlp_dataset(tilted_zlp_scan) -> tuple[Dataset3deels, np.ndarray]:
    """``Dataset3deels`` built from ``tilted_zlp_scan``, plus its ground-truth plane.

    Function-scoped (fresh dataset per test) even though the underlying
    array is module-scoped and reused read-only.
    """
    array, plane = tilted_zlp_scan
    return dataset_from_array(array.copy()), plane


# ---------------------------------------------------------------------------
# XEDS: synthetic spectra from the real X-ray line database
# ---------------------------------------------------------------------------


@pytest.fixture
def xeds_factory():
    def _make(
        array,
        *,
        energy_min=0.15,
        energy_max=9.5,
        name="synthetic xeds",
    ):
        array = np.asarray(array, dtype=np.float32)
        n_channels = int(array.shape[-1])
        step = (float(energy_max) - float(energy_min)) / max(n_channels - 1, 1)
        return Dataset3dxeds.from_array(
            array=array,
            name=name,
            origin=[0.0, 0.0, float(energy_min)],
            sampling=[1.0, 1.0, step],
            units=["px", "px", "keV"],
            signal_units="counts",
        )

    return _make


@pytest.fixture
def line_spectrum_factory(xeds_factory):
    def _build(
        selectors,
        amplitudes,
        *,
        n_channels=512,
        energy_min=0.15,
        energy_max=9.5,
        peak_width=0.08,
        background=None,
    ):
        dataset = xeds_factory(
            np.zeros((1, 1, n_channels), dtype=np.float32),
            energy_min=energy_min,
            energy_max=energy_max,
        )
        energy_axis = np.asarray(dataset.energy_axis, dtype=np.float32)
        spectrum = np.zeros_like(energy_axis, dtype=np.float32)

        if background is None:
            pass
        elif np.isscalar(background):
            spectrum += float(background)
        else:
            background_arr = np.asarray(background, dtype=np.float32)
            if background_arr.shape != energy_axis.shape:
                raise ValueError("background must be scalar or match the energy axis")
            spectrum += background_arr

        sigma = float(peak_width) / 2.355
        for selector, amplitude in zip(selectors, amplitudes):
            line_energies, line_weights, _labels = dataset.x_ray_lookup(selector)
            keep = (line_energies >= float(energy_min)) & (line_energies <= float(energy_max))
            line_energies = line_energies[keep]
            line_weights = line_weights[keep]
            if not len(line_energies):
                raise ValueError(
                    f"No lines from {selector!r} are inside the requested energy range"
                )

            if np.all(line_weights <= 0):
                line_weights = np.full(
                    line_weights.shape, 1.0 / line_weights.size, dtype=np.float32
                )
            else:
                line_weights = line_weights / np.sum(line_weights)

            for energy, weight in zip(line_energies, line_weights):
                peak = np.exp(-0.5 * ((energy_axis - float(energy)) / sigma) ** 2).astype(
                    np.float32
                )
                peak /= max(float(np.sum(peak)), 1e-8)
                spectrum += float(amplitude) * float(weight) * peak

        return spectrum

    return _build
