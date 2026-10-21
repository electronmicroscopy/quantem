"""Tests for the joint reconstruction of a tilt series."""

import numpy as np
import pytest

from quantem.core.utils.utils import to_numpy
from quantem.diffractive_imaging import DirectPtychography, DirectPtychographyTiltSeries

from .conftest import (
    ML_ABERRATIONS,
    ML_SAMPLING,
    layer_purity,
    ml_kwargs,
    thin_layer_4dstem,
    twisted_square_cell,
)

SPACING = 6.0
DEPTHS = (-SPACING / 2, SPACING / 2)
TILTS = [(0, 0), (10, 0), (-10, 0), (0, 10), (0, -10)]


def _layers():
    _, layers, _ = twisted_square_cell()
    return [0.3 * layer for layer in layers]


def _shifted(image, shift):
    q = np.fft.fftfreq(image.shape[0], ML_SAMPLING)
    ramp = np.exp(-2j * np.pi * (q[:, None] * shift[0] + q[None, :] * shift[1]))
    return np.fft.ifft2(np.fft.fft2(image) * ramp).real


def _acquisition(tilt_mrad, dose, seed, drift=(0.0, 0.0)):
    """The two layers seen at a specimen tilt: each displaced by its depth times the tilt."""
    theta = np.asarray(tilt_mrad, dtype=float) * 1e-3
    logs = [
        1j * _shifted(layer, z * theta + np.asarray(drift)) for layer, z in zip(_layers(), DEPTHS)
    ]
    return thin_layer_4dstem(logs, list(DEPTHS), ML_ABERRATIONS, dose=dose, seed=seed)


def _recon(dataset):
    return DirectPtychography.from_dataset4d(dataset, **ml_kwargs())


@pytest.fixture(scope="module")
def tilted():
    """``(series, one)``: five tilts sharing 100 electrons per A^2, and one acquisition with all
    of it, where two slices 6 A apart are noise-limited."""
    total = 100.0
    series = DirectPtychographyTiltSeries(
        [_recon(_acquisition(t, total / len(TILTS), seed=i + 1)) for i, t in enumerate(TILTS)],
        TILTS,
    )
    return series, _recon(_acquisition((0, 0), total, seed=99))


KWARGS = dict(obj_type="complex", num_slices=2, slice_mode="joint", slice_thicknesses=SPACING)


class TestTiltSeries:
    @pytest.mark.parametrize(
        "kernel, obj_type", [("obf", "complex"), ("wiener", "pure_phase"), ("obf", "pure_phase")]
    )
    def test_one_untilted_acquisition_is_its_own_reconstruction(self, kernel, obj_type):
        recon = _recon(_acquisition((0, 0), 300.0, seed=0))
        kwargs = dict(
            deconvolution_kernel=kernel, obj_type=obj_type, num_slices=2, slice_mode="joint"
        )
        expected = recon.reconstruct(slice_thicknesses=SPACING, verbose=False, **kwargs).obj
        series = DirectPtychographyTiltSeries([recon], [(0, 0)])
        series.reconstruct(slice_thicknesses=SPACING, **kwargs)

        assert np.array_equal(series.obj, expected)
        assert series.residual_loss() == pytest.approx(recon.residual_loss())

    def test_the_residual_prefers_the_true_tilts(self, tilted):
        series, _ = tilted
        flipped = DirectPtychographyTiltSeries(series.reconstructions, -series.tilt_angles)
        residuals, purities = {}, {}
        for name, s in (("true", series), ("flipped", flipped)):
            s.reconstruct(deconvolution_kernel="obf", **KWARGS)
            residuals[name] = s.residual_loss()
            purities[name], _ = layer_purity(to_numpy(s.corrected_bf), _layers())

        assert residuals["true"] < residuals["flipped"]
        # flipping the tilts moves each layer the wrong way, which mixes the slices
        assert purities["true"] > 0.99
        assert purities["flipped"] < 0.8

    @pytest.mark.parametrize("kernel, gain", [("obf", 0.04), ("wiener", 0.01)])
    def test_tilts_sharpen_close_slices_at_equal_dose(self, tilted, kernel, gain):
        series, one = tilted
        series.reconstruct(deconvolution_kernel=kernel, **KWARGS)
        purity, corr = layer_purity(to_numpy(series.corrected_bf), _layers())
        one.reconstruct(deconvolution_kernel=kernel, verbose=False, **KWARGS)
        _, corr_one = layer_purity(to_numpy(one.corrected_bf), _layers())

        # OBF 0.79 to 0.87, Wiener 0.96 to 0.98
        assert corr > corr_one + gain
        assert purity > 0.99

    def test_register_recovers_a_drift(self):
        drift = (0.6, -0.35)
        series = DirectPtychographyTiltSeries(
            [
                _recon(_acquisition((0, 0), 3000.0, seed=7)),
                _recon(_acquisition((10, 0), 3000.0, seed=8, drift=drift)),
            ],
            [(0, 0), (10, 0)],
        )
        # the lattice repeats every 3.2 A, so the search stays within half of that
        series.register(max_shift=1.5)

        assert series.shifts[0] == pytest.approx((0.0, 0.0))
        assert series.shifts[1] == pytest.approx(drift, abs=0.03)

    def test_upsampling(self, tilted):
        series, _ = tilted
        series.reconstruct(deconvolution_kernel="obf", upsampling_factor=2, **KWARGS)
        assert series.corrected_bf.shape == (2, 128, 128)
        assert np.isfinite(series.obj).all()

    def test_from_tilt_series_builds_each_acquisition(self):
        datasets = [_acquisition((0, 0), None, 0), _acquisition((10, 0), None, 0)]
        shared = ml_kwargs()
        shared.pop("aberration_coefs")
        series = DirectPtychography.from_tilt_series(
            datasets,
            [(0, 0), (10, 0)],
            per_dataset_kwargs=[
                {"aberration_coefs": ML_ABERRATIONS},
                {"aberration_coefs": dict(ML_ABERRATIONS, C10=60.0)},
            ],
            **shared,
        )

        assert isinstance(series, DirectPtychographyTiltSeries)
        assert len(series) == 2
        assert series[1].hyperparameter_state.current_aberrations()["C10"] == pytest.approx(60.0)

    @pytest.mark.parametrize(
        "make, match",
        [
            (lambda r: DirectPtychographyTiltSeries([r], [(0, 0), (1, 0)]), "one row per"),
            (lambda r: DirectPtychographyTiltSeries([r, "not one"], [(0, 0), (1, 0)]), "must be"),
            (lambda r: DirectPtychographyTiltSeries([], []), "at least one"),
        ],
    )
    def test_rejects_bad_series(self, tilted, make, match):
        series, _ = tilted
        with pytest.raises((ValueError, TypeError), match=match):
            make(series[0])

    def test_rejects_another_scan_grid(self, tilted):
        series, _ = tilted
        from quantem.core.datastructures import Dataset4d

        full = _acquisition((0, 0), None, 0)
        cropped = Dataset4d.from_array(
            np.asarray(full.array)[:32].copy(), sampling=full.sampling, units=full.units
        )
        with pytest.raises(ValueError, match="scan grid"):
            DirectPtychographyTiltSeries([series[0], _recon(cropped)], [(0, 0), (1, 0)])

    def test_survives_a_round_trip(self, tmp_path):
        from quantem.core.io.serialize import load

        series = DirectPtychography.from_tilt_series(
            [_acquisition((0, 0), 3000.0, seed=7), _acquisition((10, 0), 3000.0, seed=8)],
            [(0, 0), (10, 0)],
            per_dataset_kwargs=[{}, {"focal_spread": 20.0}],
            **ml_kwargs(),
        )
        series.shifts = np.array([[0.0, 0.0], [0.25, -0.5]])
        kwargs = dict(deconvolution_kernel="wiener", **KWARGS)
        series.reconstruct(**kwargs)
        before = series.obj.copy()

        path = str(tmp_path / "tilt_series.zip")
        series.save(path, mode="o")
        restored = load(path)

        assert isinstance(restored, DirectPtychographyTiltSeries)
        assert len(restored) == 2
        assert all(isinstance(r, DirectPtychography) for r in restored)
        assert np.array_equal(restored.tilt_angles, series.tilt_angles)
        assert np.array_equal(restored.shifts, series.shifts)
        assert restored[1].focal_spread == 20.0
        assert np.array_equal(restored.obj, before)
        assert restored.residual_loss() == pytest.approx(series.residual_loss())
        assert restored.noise_variance == pytest.approx(series.noise_variance)
        assert np.allclose(restored.slice_depths, series.slice_depths)

        # and it reconstructs again from what it loaded
        restored.reconstruct(**kwargs)
        assert np.allclose(restored.obj, before, atol=1e-6)

    def test_survives_a_round_trip_before_reconstructing(self, tmp_path):
        from quantem.core.io.serialize import load

        series = DirectPtychographyTiltSeries(
            [_recon(_acquisition((0, 0), None, 0)), _recon(_acquisition((10, 0), None, 0))],
            [(0, 0), (10, 0)],
        )
        path = str(tmp_path / "fresh.zip")
        series.save(path, mode="o")
        restored = load(path)

        assert restored.obj is None
        assert restored.residual_loss() is None
        assert restored.slice_depths is None
        restored.reconstruct(deconvolution_kernel="obf", **KWARGS)
        expected = series.reconstruct(deconvolution_kernel="obf", **KWARGS).obj
        assert np.allclose(restored.obj, expected, atol=1e-6)

    @pytest.mark.parametrize("kernel", ["ssb", "wdd", "prlx", "icom"])
    def test_only_the_normal_equation_kernels_combine(self, tilted, kernel):
        series, _ = tilted
        with pytest.raises(ValueError, match="normal"):
            series.reconstruct(deconvolution_kernel=kernel, num_slices=1)


def test_a_tilt_series_cannot_be_sectioned(tilted):
    series, _ = tilted
    with pytest.raises(ValueError, match="solved jointly"):
        series.reconstruct(num_slices=2, slice_thicknesses=SPACING, slice_mode="sectioned")
