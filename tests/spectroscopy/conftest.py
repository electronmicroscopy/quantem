import numpy as np
import pytest

from quantem.spectroscopy import Dataset3dxeds


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
