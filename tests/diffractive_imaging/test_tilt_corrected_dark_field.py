"""Tests for tilt-corrected dark field and its fusion with a bright-field reconstruction.

Synthetic data from the thin-layer simulator in ``conftest.py``, whose 64-pixel detector
extends to twice the aperture radius, so the dark-field annulus holds real scattering.
"""

import math

import numpy as np
import pytest

from quantem.core.utils.utils import electron_wavelength_angstrom, to_numpy
from quantem.diffractive_imaging import (
    DirectPtychography,
    OptimizationParameter,
    TiltCorrectedDarkField,
    bright_field_transfer,
    fuse_full_field,
)

from .conftest import (
    ML_K_PROBE,
    ML_N_DET,
    ML_SEMIANGLE,
    PROBE_ENERGY,
    band_limit_ml,
    correlation,
    thin_layer_4dstem,
    twisted_square_cell,
)

DEFOCUS = 150.0


def _kwargs(c10=DEFOCUS):
    return dict(
        energy=PROBE_ENERGY,
        semiangle_cutoff=ML_SEMIANGLE,
        rotation_angle=0.0,
        aberration_coefs={"C10": c10},
        force_fitted_origin=(ML_N_DET // 2, ML_N_DET // 2),
        verbose=False,
    )


@pytest.fixture(scope="module")
def scene():
    """``(dataset, phase)`` for one atomic layer imaged with a defocused probe."""
    _, layers, _ = twisted_square_cell()
    phase = 0.5 * layers[0]
    return thin_layer_4dstem([1j * phase], [0.0], {"C10": DEFOCUS}), phase


@pytest.fixture(scope="module")
def tcdf(scene):
    dataset, _ = scene
    return TiltCorrectedDarkField.from_dataset4d(dataset, **_kwargs())


class TestTiltCorrectedDarkField:
    def test_two_thirds_rule_is_ma_et_al_s_shift(self, tcdf):
        """For pure defocus each segment shifts by (2/3) C10 alpha along its direction."""
        shifts = to_numpy(tcdf._segment_shifts({"C10": DEFOCUS}, 0.0, "two_thirds"))
        wavelength = electron_wavelength_angstrom(PROBE_ENERGY)
        # grad chi = 2 pi lambda C10 k, evaluated at k = (2/3) k_alpha
        expected = 2 * math.pi * wavelength * DEFOCUS * (2 / 3) * ML_K_PROBE

        assert np.allclose(np.linalg.norm(shifts, axis=-1), expected, rtol=1e-4)

    def test_correcting_the_shifts_recovers_the_object(self, scene, tcdf):
        _, phase = scene
        truth = band_limit_ml(phase**2)
        tcdf.reconstruct(verbose=False)
        corrected = correlation(to_numpy(tcdf.corrected_bf), truth)
        tcdf.reconstruct(override_aberration_coefs={"C10": 0.0}, verbose=False)
        uncorrected = correlation(to_numpy(tcdf.corrected_bf), truth)

        assert corrected > 0.5
        assert corrected > uncorrected + 0.5

    def test_two_thirds_beats_the_aperture_edge(self, scene, tcdf):
        """Shifting as at the aperture edge over-shifts, as Ma et al. found."""
        _, phase = scene
        truth = band_limit_ml(phase**2)
        scores = {}
        for shift in ("two_thirds", 1.0):
            tcdf.reconstruct(shift=shift, verbose=False)
            scores[shift] = correlation(to_numpy(tcdf.corrected_bf), truth)

        assert scores["two_thirds"] > scores[1.0]

    def test_the_residual_finds_the_defocus(self, tcdf):
        losses = {}
        for c10 in (DEFOCUS - 50, DEFOCUS, DEFOCUS + 50):
            tcdf.reconstruct(override_aberration_coefs={"C10": c10}, verbose=False)
            losses[c10] = tcdf.residual_loss()

        assert min(losses, key=losses.get) == DEFOCUS

    def test_a_grid_search_fits_the_defocus(self, scene):
        dataset, _ = scene
        recon = TiltCorrectedDarkField.from_dataset4d(dataset, **_kwargs(c10=0.0))
        recon.grid_search_hyperparameters(
            aberration_coefs={"C10": OptimizationParameter(low=100.0, high=200.0, n_points=5)},
            verbose=False,
        )

        assert recon.hyperparameter_state.optimized_aberrations["C10"] == pytest.approx(DEFOCUS)

    def test_every_pixel_can_be_its_own_segment(self, scene):
        dataset, _ = scene
        recon = TiltCorrectedDarkField.from_dataset4d(dataset, n_angular_bins=None, **_kwargs())

        assert recon.num_segments == int(recon._segment_masks.flatten(1).sum())
        assert np.isfinite(to_numpy(recon.reconstruct(verbose=False).corrected_bf)).all()

    def test_upsampling_interpolates_the_image(self, tcdf):
        """Zero-padding the spectrum passes through the scan-grid samples and adds no grid."""
        coarse = to_numpy(tcdf.reconstruct(verbose=False).corrected_bf)
        fine = to_numpy(tcdf.reconstruct(upsampling_factor=2, verbose=False).corrected_bf)

        assert fine.shape == tuple(2 * n for n in coarse.shape)
        assert np.allclose(fine[::2, ::2], coarse, atol=0.02 * np.abs(coarse).max())
        power = np.abs(np.fft.fft2(fine)) ** 2
        n = fine.shape[0]
        assert power[n // 2 - 1 : n // 2 + 2].sum() < 1e-6 * power.sum()

    def test_an_empty_annulus_raises(self, scene):
        dataset, _ = scene
        with pytest.raises(ValueError, match="annulus is empty"):
            TiltCorrectedDarkField.from_dataset4d(dataset, inner_radius=10.0, **_kwargs())

    def test_from_direct_ptychography_copies_the_hyperparameters(self, scene):
        dataset, _ = scene
        direct = DirectPtychography.from_dataset4d(dataset, edge_blend_pixels=0, **_kwargs(0.0))
        direct.hyperparameter_state.optimized_aberrations = {"C10": DEFOCUS}
        direct.hyperparameter_state.optimized_rotation_angle = 2.0
        recon = TiltCorrectedDarkField.from_direct_ptychography(
            direct, dataset, force_fitted_origin=(ML_N_DET // 2, ML_N_DET // 2)
        )

        state = recon.hyperparameter_state
        assert state.current_aberrations()["C10"] == pytest.approx(DEFOCUS)
        assert state.current_rotation_angle() == pytest.approx(2.0)


class TestFullFieldFusion:
    def test_transfer_is_one_at_its_peak_and_zero_past_the_band(self, scene):
        dataset, _ = scene
        direct = DirectPtychography.from_dataset4d(dataset, edge_blend_pixels=0, **_kwargs())
        transfer = bright_field_transfer(direct)

        qx = np.fft.fftfreq(transfer.shape[0], 0.25)
        q = np.hypot(qx[:, None], qx[None, :])
        assert transfer.max() == pytest.approx(1.0)
        # the mask reaches half a pixel past alpha, so the band ends a pixel past 2 alpha
        pixel = 1 / (ML_N_DET * 0.25)
        assert np.abs(transfer[q > 2 * ML_K_PROBE + 1.5 * pixel]).max() < 1e-6

    def test_the_weight_selects_each_channel(self):
        rng = np.random.default_rng(0)
        direct, dark = rng.standard_normal((2, 16, 16))

        def standardized(image):
            return (image - image.mean()) / image.std()

        assert np.allclose(fuse_full_field(direct, dark, np.ones((16, 16))), standardized(direct))
        assert np.allclose(fuse_full_field(direct, dark, np.zeros((16, 16))), standardized(dark))

    def test_shapes_must_match(self):
        with pytest.raises(ValueError, match="share a shape"):
            fuse_full_field(np.zeros((4, 4)), np.zeros((4, 5)), np.zeros((4, 4)))
