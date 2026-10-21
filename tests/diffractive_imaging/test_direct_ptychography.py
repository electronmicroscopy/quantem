"""Tests for the Fourier-space direct-ptychography reconstruction.

Covers all five deconvolution kernels, the aberration/rotation conventions they share, the
hyperparameter-fitting routines, and save/load. Synthetic data comes from
``conftest.py``: a white-noise pure-phase object imaged with a defocused soft aperture.
"""

import warnings

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset2d, Dataset3d
from quantem.core.io.serialize import load
from quantem.core.utils.utils import electron_wavelength_angstrom, to_numpy
from quantem.diffractive_imaging import DirectPtychography, OptimizationParameter
from quantem.diffractive_imaging.complex_probe import spatial_frequencies

from .conftest import (
    CTF_SAMPLING,
    CTF_SCAN_SAMPLING,
    DECONVOLUTION_KERNELS,
    ML_ABERRATIONS,
    ML_K_PROBE,
    ML_N_DET,
    ML_RECIPROCAL_SAMPLING,
    ML_SAMPLING,
    ML_SEMIANGLE,
    PROBE_ENERGY,
    Q_PROBE,
    SCAN_SAMPLING,
    SEMIANGLE_CUTOFF,
    N,
    analytic_probe_array,
    analytic_psi,
    band_limit_ml,
    band_limited_phase,
    correlation,
    ctf_band_scores,
    ctf_dataset3d,
    ctf_dataset4d,
    ctf_jittered_positions,
    ctf_kwargs,
    ctf_radial_correlation,
    ctf_raster_positions,
    ctf_simulate,
    direct_ptycho_kwargs,
    integer_shift_defocus,
    layer_purity,
    library_probe_fourier,
    make_dataset4d,
    measured_ctf,
    ml_kwargs,
    scan_positions_px,
    thin_layer_4dstem,
    twisted_honeycomb_cell,
    twisted_square_cell,
    white_noise_object_2D,
)

TRUE_C10 = integer_shift_defocus(1)


def _build(dataset4d, defocus=TRUE_C10, **overrides):
    kwargs = dict(direct_ptycho_kwargs(defocus), edge_blend_pixels=0)
    kwargs.update(overrides)
    return DirectPtychography.from_dataset4d(dataset4d, **kwargs)


@pytest.fixture(scope="module")
def recon(dataset4d):
    """A reconstruction seeded with the true defocus. Reconstruct before asserting."""
    return _build(dataset4d)


class TestConstruction:
    def test_geometry_matches_the_dataset(self, recon):
        assert recon.bf_mask.shape == recon.gpts
        assert recon.vbf_stack.shape == (recon.num_bf, N, N)
        assert recon.num_bf == int(recon.bf_mask.sum())
        assert recon.scan_gpts == (N, N)
        assert recon.scan_sampling[0] == pytest.approx(SCAN_SAMPLING)

    def test_bf_mask_is_the_model_aperture(self, recon):
        """By default the mask is every pixel the soft aperture reaches, half a pixel past alpha."""
        expected_area = np.pi * (Q_PROBE / recon.reciprocal_sampling[0] + 0.5) ** 2
        assert recon.num_bf == pytest.approx(expected_area, rel=0.15)

    def test_an_intensity_threshold_masks_the_mean_pattern(self, dataset4d):
        recon = _build(dataset4d, intensity_threshold=0.5)
        expected_area = np.pi * (Q_PROBE / recon.reciprocal_sampling[0]) ** 2
        assert recon.num_bf == pytest.approx(expected_area, rel=0.35)
        assert recon.num_bf < _build(dataset4d).num_bf

    def test_preprocess_zeroes_the_dc_bin(self, recon):
        assert torch.allclose(
            recon._vbf_fourier[..., 0, 0], torch.zeros_like(recon._vbf_fourier[..., 0, 0])
        )

    def test_from_virtual_bfs_reproduces_from_dataset4d(self, dataset4d, recon):
        """Re-wrapping the stored stack must give a bit-identical reconstruction."""
        vbf_dataset = Dataset3d.from_array(
            to_numpy(recon.vbf_stack),
            name="vBF stack",
            units=("index", "A", "A"),
            sampling=(1, SCAN_SAMPLING, SCAN_SAMPLING),
        )
        bf_mask_dataset = Dataset2d.from_array(
            to_numpy(recon.bf_mask),
            name="BF mask",
            units=("A^-1", "A^-1"),
            sampling=tuple(recon.reciprocal_sampling),
        )
        rebuilt = DirectPtychography.from_virtual_bfs(
            vbf_dataset,
            bf_mask_dataset,
            energy=PROBE_ENERGY,
            rotation_angle=0.0,
            semiangle_cutoff=SEMIANGLE_CUTOFF,
            aberration_coefs={"C10": TRUE_C10},
            crop_bf_mask=False,  # the stored mask is already cropped
            verbose=False,
        )

        recon.reconstruct(deconvolution_kernel="ssb", verbose=False)
        rebuilt.reconstruct(deconvolution_kernel="ssb", verbose=False)
        assert np.allclose(rebuilt.obj, recon.obj, rtol=1e-5, atol=1e-8)

    def test_cropping_the_bf_mask_preserves_the_reconstruction(self, dataset4d):
        """Cropping shrinks the detector grid but must not permute the vBF stack order."""
        cropped = _build(dataset4d, crop_bf_mask=True)
        uncropped = _build(dataset4d, crop_bf_mask=False)

        assert cropped.gpts[0] < uncropped.gpts[0]
        assert cropped.num_bf == uncropped.num_bf

        cropped.reconstruct(deconvolution_kernel="ssb", verbose=False)
        uncropped.reconstruct(deconvolution_kernel="ssb", verbose=False)
        assert correlation(cropped.obj, uncropped.obj) > 0.99

    def test_direct_instantiation_is_blocked(self):
        with pytest.raises(RuntimeError, match="from_virtual_bfs"):
            DirectPtychography(
                vbf_dataset=None,
                bf_mask_dataset=None,
                energy=PROBE_ENERGY,
                rotation_angle=0.0,
                aberration_coefs={},
                semiangle_cutoff=SEMIANGLE_CUTOFF,
                soft_edges=True,
                crop_bf_mask=False,
                bf_mask_padding_px=1,
                rng=None,
                device="cpu",
                verbose=False,
            )


class TestDeconvolutionKernels:
    @pytest.mark.parametrize("kernel", DECONVOLUTION_KERNELS)
    def test_recovers_the_band_limited_object(self, recon, kernel):
        """Every kernel must recover the object over the band the aperture transfers."""
        recon.reconstruct(deconvolution_kernel=kernel, verbose=False)

        assert recon.obj.shape == (N, N)
        assert np.isfinite(recon.obj).all()
        assert abs(correlation(recon.obj, band_limited_phase())) > 0.7

    @pytest.mark.parametrize("kernel", DECONVOLUTION_KERNELS)
    def test_aliases_resolve(self, recon, kernel):
        aliases = {
            "ssb": "single-sideband",
            "obf": "optimum-bright-field",
            "prlx": "tilt-corrected-bright-field",
            "icom": "center-of-mass",
        }
        by_short = recon.reconstruct(deconvolution_kernel=kernel, verbose=False).obj.copy()
        by_alias = recon.reconstruct(deconvolution_kernel=aliases[kernel], verbose=False).obj

        assert np.array_equal(by_short, by_alias)

    @pytest.mark.parametrize("kernel", ["mf", "matched-filter"])
    def test_the_matched_filter_has_been_removed(self, recon, kernel):
        with pytest.raises(ValueError, match="matched-filter kernel has been removed"):
            recon.reconstruct(deconvolution_kernel=kernel, verbose=False)

    @pytest.mark.parametrize("upsampling_factor", [1, 2, 3])
    def test_upsampling_preserves_the_field_of_view(self, recon, upsampling_factor):
        recon.reconstruct(
            deconvolution_kernel="ssb", upsampling_factor=upsampling_factor, verbose=False
        )

        assert recon.obj.shape == (N * upsampling_factor, N * upsampling_factor)
        assert recon.obj.shape[0] * recon._obj_sampling[0] == pytest.approx(N * SCAN_SAMPLING)

    def test_no_per_pixel_stack_is_kept(self, recon):
        """The kernels are summed over bright-field pixels in Fourier space, never stored."""
        recon.reconstruct(deconvolution_kernel="ssb", verbose=False)

        assert recon.corrected_bf.shape == (N, N)
        assert not hasattr(recon, "corrected_stack")
        assert not hasattr(recon, "variance_loss")

    def test_unknown_kernel_raises(self, recon):
        with pytest.raises(ValueError, match="Unknown deconvolution kernel"):
            recon.reconstruct(deconvolution_kernel="maximum-entropy", verbose=False)


class TestLateralShifts:
    """`_return_lateral_shifts` underpins both the parallax kernel and the montage class."""

    def test_pure_defocus_matches_the_analytic_shift(self, recon):
        """For pure C10 the lateral shift is exactly `wavelength * C10 * k` Angstrom."""
        shifts = recon._return_lateral_shifts(0.0, {"C10": TRUE_C10}, recon.bf_mask)

        kxa, kya = spatial_frequencies(recon.gpts, recon.sampling, device=recon.device)
        expected = (
            torch.stack((kxa[recon.bf_mask], kya[recon.bf_mask]), -1)
            * electron_wavelength_angstrom(PROBE_ENERGY)
            * TRUE_C10
        )

        assert torch.allclose(shifts, expected, rtol=1e-5, atol=1e-6)

    def test_no_aberrations_means_no_shift(self, recon):
        shifts = recon._return_lateral_shifts(0.0, {}, recon.bf_mask)

        assert torch.count_nonzero(shifts) == 0

    def test_rotation_rotates_the_shifts(self, recon):
        """`_passively_rotate_grid` sends (kx, ky) -> (-ky, kx) at 90 degrees."""
        coefs = {"C10": TRUE_C10}
        unrotated = recon._return_lateral_shifts(0.0, coefs, recon.bf_mask)
        rotated = recon._return_lateral_shifts(90.0, coefs, recon.bf_mask)

        expected = torch.stack((-unrotated[:, 1], unrotated[:, 0]), dim=-1)
        assert torch.allclose(rotated, expected, atol=1e-5)

    def test_astigmatism_breaks_the_radial_symmetry(self, recon):
        radial = recon._return_lateral_shifts(0.0, {"C10": TRUE_C10}, recon.bf_mask)
        astigmatic = recon._return_lateral_shifts(
            0.0, {"C10": TRUE_C10, "C12": 0.3 * TRUE_C10}, recon.bf_mask
        )

        assert not torch.allclose(radial, astigmatic)


class TestBrightFieldSubsets:
    def test_checkerboard_halves_are_additive(self, recon):
        """Reconstructions are linear in the BF sum once the per-subset weight is undone."""
        recon.reconstruct(deconvolution_kernel="prlx", parallax_flip_phase=False, verbose=False)
        full = recon.corrected_bf.clone()

        halves = []
        for mask in recon._make_checkerboard_bf_masks(recon.gpts, recon.bf_mask):
            recon.reconstruct(
                bf_mask=mask,
                deconvolution_kernel="prlx",
                parallax_flip_phase=False,
                verbose=False,
            )
            halves.append(recon.corrected_bf.clone() * int(mask.sum()))

        # each half is normalized by its own BF weight; undo that before comparing
        combined = to_numpy(halves[0] + halves[1]) / recon.num_bf
        assert correlation(combined, to_numpy(full)) > 0.99

    def test_halfsets_helper_returns_two_images(self, recon):
        first, second = recon._reconstruct_with_halfsets(deconvolution_kernel="ssb")

        assert first.shape == second.shape == (N, N)
        assert correlation(to_numpy(first), to_numpy(second)) > 0.5

    def test_subset_uses_fewer_bright_field_pixels(self, recon):
        mask = recon._make_checkerboard_bf_masks(recon.gpts, recon.bf_mask)[0]
        recon.reconstruct(deconvolution_kernel="ssb", verbose=False)
        full = recon.corrected_bf.clone()
        recon.reconstruct(bf_mask=mask, deconvolution_kernel="ssb", verbose=False)

        assert recon._return_bf_context(mask).num_bf == int(mask.sum()) < recon.num_bf
        assert not torch.equal(recon.corrected_bf, full)
        assert correlation(to_numpy(recon.corrected_bf), to_numpy(full)) > 0.9


class TestFilters:
    def test_lowpass_suppresses_high_frequencies(self, recon):
        recon.reconstruct(deconvolution_kernel="ssb", verbose=False)
        unfiltered = np.abs(np.fft.fft2(recon.obj))
        recon.reconstruct(deconvolution_kernel="ssb", q_lowpass=0.05, verbose=False)
        filtered = np.abs(np.fft.fft2(recon.obj))

        qx = qy = np.fft.fftfreq(N, SCAN_SAMPLING)
        high = np.hypot(qx[:, None], qy[None, :]) > 0.15

        assert filtered[high].sum() < 0.05 * unfiltered[high].sum()

    def test_highpass_suppresses_low_frequencies(self, recon):
        recon.reconstruct(deconvolution_kernel="ssb", verbose=False)
        unfiltered = np.abs(np.fft.fft2(recon.obj))
        recon.reconstruct(deconvolution_kernel="ssb", q_highpass=0.15, verbose=False)
        filtered = np.abs(np.fft.fft2(recon.obj))

        qx = qy = np.fft.fftfreq(N, SCAN_SAMPLING)
        low = (np.hypot(qx[:, None], qy[None, :]) < 0.05) & (
            np.hypot(qx[:, None], qy[None, :]) > 0
        )

        assert filtered[low].sum() < 0.2 * unfiltered[low].sum()

    def test_parallax_phase_flip_vanishes_at_zero_defocus(self, recon):
        """`sign(sin(chi))` is identically zero when chi is, so the image is too.

        Not a defect: an unaberrated probe transfers no phase contrast in this formulation.
        Worth pinning because the all-zero output is otherwise startling.
        """
        recon.reconstruct(
            deconvolution_kernel="prlx",
            override_aberration_coefs={"C10": 0.0},
            parallax_flip_phase=True,
            verbose=False,
        )
        assert np.ptp(recon.obj) == 0.0

        recon.reconstruct(
            deconvolution_kernel="prlx",
            override_aberration_coefs={"C10": 0.0},
            parallax_flip_phase=False,
            verbose=False,
        )
        assert np.ptp(recon.obj) > 0.0


class TestResidualLoss:
    """The least-squares residual of each kernel's forward model, as a hyperparameter objective."""

    def test_is_none_before_reconstructing(self, dataset4d):
        assert _build(dataset4d).residual_loss() is None

    @pytest.mark.parametrize("kernel", DECONVOLUTION_KERNELS)
    def test_is_a_fraction(self, recon, kernel):
        recon.reconstruct(deconvolution_kernel=kernel, verbose=False)

        assert 0.0 < recon.residual_loss() < 1.0

    @pytest.mark.parametrize("kernel", ("ssb", "obf", "prlx"))
    def test_is_minimized_at_the_true_defocus(self, recon, kernel):
        losses = {}
        for scale in (0.5, 0.8, 1.0, 1.2, 1.5):
            recon.reconstruct(
                deconvolution_kernel=kernel,
                override_aberration_coefs={"C10": scale * TRUE_C10},
                verbose=False,
            )
            losses[scale] = recon.residual_loss()

        assert min(losses, key=losses.get) == pytest.approx(1.0)

    def test_the_weak_object_kernels_share_one_model(self, recon):
        """ssb, obf and mf differ in how they normalize, not in what they model."""
        values = []
        for kernel in ("ssb", "obf"):
            recon.reconstruct(deconvolution_kernel=kernel, verbose=False)
            values.append(recon.residual_loss())

        assert values[1] == pytest.approx(values[0], rel=1e-6)

    def test_parallax_residual_is_the_spread_of_the_corrected_images(self, recon):
        """By Parseval, the parallax residual is the spread of the aligned images about their mean."""
        recon.reconstruct(
            deconvolution_kernel="prlx", parallax_flip_phase=False, band_limit=False, verbose=False
        )
        stack = recon._return_parallax_stack(band_limit=False).double()
        deviation = torch.fft.fft2(stack - stack.mean(0)).abs().square().sum(0)
        # a shift has unit modulus, so the aligned stack carries the data energy unchanged
        energy = torch.fft.fft2(stack).abs().square().sum(0)

        qx = torch.fft.fftfreq(N, SCAN_SAMPLING, dtype=torch.float64)
        q = torch.hypot(qx[:, None], qx[None, :])
        mask = (q > 0) & (q <= recon._return_band_limit())

        expected = float((deviation * mask).sum() / (energy * mask).sum())
        assert recon.residual_loss() == pytest.approx(expected, rel=1e-4)

    def test_variance_loss_has_been_removed(self, recon):
        recon.reconstruct(deconvolution_kernel="prlx", verbose=False)
        with pytest.raises(ValueError, match="loss='residual'"):
            recon._return_loss_value("variance")


class TestHyperparameterFitting:
    """All four routines must recover a seeded defocus from a blind start."""

    def test_grid_search(self, dataset4d):
        recon = _build(dataset4d, aberration_coefs={})
        recon.grid_search_hyperparameters(
            aberration_coefs={
                "C10": OptimizationParameter(low=0.4 * TRUE_C10, high=1.6 * TRUE_C10, n_points=7)
            },
            deconvolution_kernel="prlx",
            verbose=False,
        )
        fitted = recon.hyperparameter_state.optimized_aberrations["C10"]

        step = (1.6 - 0.4) * TRUE_C10 / 6
        assert abs(fitted - TRUE_C10) <= step

    def test_cross_correlation(self, dataset4d):
        recon = _build(dataset4d, aberration_coefs={})
        recon.fit_hyperparameters_cross_correlation(bin_factors=(2, 1), verbose=False)
        state = recon.hyperparameter_state

        assert state.optimized_aberrations["C10"] == pytest.approx(TRUE_C10, rel=0.3)
        assert state.optimized_rotation_angle == pytest.approx(0.0, abs=5.0)

    def test_cross_correlation_from_the_stored_state(self, dataset4d):
        """The images are pre-shifted by the stored starting point, so the fit must count those
        shifts too, not only what the alignment adds to them."""
        recon = _build(dataset4d, defocus=0.7 * TRUE_C10)
        recon.fit_hyperparameters_cross_correlation(bin_factors=(2, 1), verbose=False)

        assert recon.hyperparameter_state.optimized_aberrations["C10"] == pytest.approx(
            TRUE_C10, rel=0.1
        )

    def test_cross_correlation_keeps_the_twin_of_its_start(self, dataset4d):
        """Turning the detector by 180 degrees and negating C10 shifts every image alike, so
        the start decides which of the two the fit returns."""
        recon = _build(dataset4d, defocus=-0.7 * TRUE_C10, rotation_angle=180.0)
        recon.fit_hyperparameters_cross_correlation(bin_factors=(2, 1), verbose=False)
        state = recon.hyperparameter_state

        assert state.optimized_aberrations["C10"] == pytest.approx(-TRUE_C10, rel=0.1)
        assert abs(state.optimized_rotation_angle) == pytest.approx(180.0, abs=5.0)

    def test_cross_correlation_starts_from_a_stored_fit(self, dataset4d):
        """A rotation stored as optimized, by an earlier fit or by hand, is the start, so it
        decides the twin; ``use_initial_state`` starts from the constructor's values instead."""
        recon = _build(dataset4d, defocus=0.7 * TRUE_C10)
        state = recon.hyperparameter_state

        state.optimized_aberrations = {"C10": -0.7 * TRUE_C10}
        state.optimized_rotation_angle = 180.0
        recon.fit_hyperparameters_cross_correlation(bin_factors=(2, 1), verbose=False)
        assert state.optimized_aberrations["C10"] == pytest.approx(-TRUE_C10, rel=0.1)
        assert abs(state.optimized_rotation_angle) == pytest.approx(180.0, abs=5.0)

        state.optimized_aberrations = {"C10": -0.7 * TRUE_C10}
        state.optimized_rotation_angle = 180.0
        recon.fit_hyperparameters_cross_correlation(
            bin_factors=(2, 1), use_initial_state=True, verbose=False
        )
        assert state.optimized_aberrations["C10"] == pytest.approx(TRUE_C10, rel=0.1)
        assert state.optimized_rotation_angle == pytest.approx(0.0, abs=5.0)

    def test_the_shift_fit_returns_the_twin_nearest_its_reference(self, recon):
        from quantem.diffractive_imaging.direct_ptycho_utils import fit_aberrations_from_shifts

        aberrations = {"C10": -300.0, "C12": 40.0, "phi12": 0.3}
        shifts = recon._return_lateral_shifts(-168.0, aberrations, recon.bf_mask)
        args = (shifts, recon.bf_mask, recon.wavelength, recon.gpts, recon.sampling)

        near = fit_aberrations_from_shifts(*args, reference_rotation=-160.0)
        assert near["rotation_angle"] == pytest.approx(-168.0, abs=1e-2)
        assert near["C10"] == pytest.approx(-300.0, rel=1e-3)
        assert near["C12"] == pytest.approx(40.0, rel=1e-3)
        assert np.cos(2 * (near["phi12"] - 0.3)) == pytest.approx(1.0, abs=1e-5)

        # without a reference, the twin within 90 degrees of zero
        twin = fit_aberrations_from_shifts(*args)
        assert twin["rotation_angle"] == pytest.approx(12.0, abs=1e-2)
        assert twin["C10"] == pytest.approx(300.0, rel=1e-3)
        assert np.cos(2 * (twin["phi12"] - 0.3 - np.pi / 2)) == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.slow
    def test_optuna(self, dataset4d):
        recon = _build(dataset4d, aberration_coefs={})
        recon.optimize_hyperparameters(
            aberration_coefs={
                "C10": OptimizationParameter(low=0.4 * TRUE_C10, high=1.6 * TRUE_C10)
            },
            n_trials=25,
            deconvolution_kernel="prlx",
            verbose=False,
        )
        fitted = recon.hyperparameter_state.optimized_aberrations["C10"]

        assert fitted == pytest.approx(TRUE_C10, rel=0.15)
        assert recon.hyperparameter_state.study is not None

    def test_fitting_leaves_the_initial_state_intact(self, dataset4d):
        """`use_initial_state=True` must ignore whatever a fit wrote back."""
        recon = _build(dataset4d)
        recon.hyperparameter_state.optimized_aberrations = {"C10": 5.0 * TRUE_C10}

        recon.reconstruct(deconvolution_kernel="ssb", use_initial_state=True, verbose=False)
        from_initial = recon.obj.copy()
        recon.reconstruct(
            deconvolution_kernel="ssb",
            override_aberration_coefs={"C10": TRUE_C10},
            verbose=False,
        )

        assert np.allclose(from_initial, recon.obj)


class TestLossFunctions:
    """The searches accept an objective by name or as a callable, and minimize it."""

    def test_rms_gradient_is_none_before_reconstruct(self, dataset4d):
        assert _build(dataset4d).rms_gradient_loss() is None

    def test_rms_gradient_peaks_at_the_true_defocus(self, dataset4d):
        """It is a sharpness metric, so the correct deconvolution must maximize it."""
        recon = _build(dataset4d)

        losses = {}
        for scale in (0.5, 1.0, 1.5):
            recon.reconstruct(
                deconvolution_kernel="prlx",
                override_aberration_coefs={"C10": scale * TRUE_C10},
                verbose=False,
            )
            losses[scale] = recon.rms_gradient_loss()

        # negated, so the true defocus is the *smallest*
        assert losses[1.0] < losses[0.5]
        assert losses[1.0] < losses[1.5]

    def test_rms_gradient_follows_the_object_sampling(self, dataset4d):
        """Per Angstrom, not per pixel, so upsampling does not rescale it.

        ``band_limit=False`` keeps the tiled replicas, so the two canvases hold the same
        content and only the sampling differs, which is the property under test. With the
        band limit on, upsampling a parallax reconstruction removes those replicas and the
        image amplitude genuinely changes, so this would be measuring that instead.
        """
        recon = _build(dataset4d)
        kwargs = dict(deconvolution_kernel="prlx", band_limit=False, verbose=False)
        recon.reconstruct(upsampling_factor=1, **kwargs)
        coarse = recon.rms_gradient_loss()
        recon.reconstruct(upsampling_factor=2, **kwargs)
        fine = recon.rms_gradient_loss()

        # upsampling tiles the spectrum rather than adding detail, so the physical gradient
        # is the same quantity; a per-pixel metric would differ by the sampling ratio
        assert fine == pytest.approx(coarse, rel=0.35)

    def test_the_two_classes_agree(self, dataset4d):
        """`prlx` is the same operator in both, so the objective must match."""
        from quantem.diffractive_imaging import DirectPtychographyMontage

        kwargs = dict(direct_ptycho_kwargs(TRUE_C10), edge_blend_pixels=0)
        fourier = DirectPtychography.from_dataset4d(dataset4d, **kwargs)
        montage = DirectPtychographyMontage.from_dataset4d(dataset4d, boundary="wrap", **kwargs)
        for recon in (fourier, montage):
            recon.reconstruct(deconvolution_kernel="prlx", verbose=False)

        assert montage.rms_gradient_loss() == pytest.approx(fourier.rms_gradient_loss(), rel=1e-5)

    def test_grid_search_with_the_rms_gradient(self, dataset4d):
        recon = _build(dataset4d, aberration_coefs={})
        recon.grid_search_hyperparameters(
            aberration_coefs={
                "C10": OptimizationParameter(low=0.4 * TRUE_C10, high=1.6 * TRUE_C10, n_points=7)
            },
            loss="rms_gradient",
            deconvolution_kernel="prlx",
            verbose=False,
        )
        fitted = recon.hyperparameter_state.optimized_aberrations["C10"]

        step = (1.6 - 0.4) * TRUE_C10 / 6
        assert abs(fitted - TRUE_C10) <= step

    def test_grid_search_accepts_a_callable(self, dataset4d):
        """Anything that scores a reconstruction can drive a search."""
        recon = _build(dataset4d, aberration_coefs={})
        recon.grid_search_hyperparameters(
            aberration_coefs={
                "C10": OptimizationParameter(low=0.4 * TRUE_C10, high=1.6 * TRUE_C10, n_points=7)
            },
            loss=lambda r: -float(np.std(r.obj)),
            deconvolution_kernel="prlx",
            verbose=False,
        )
        fitted = recon.hyperparameter_state.optimized_aberrations["C10"]

        step = (1.6 - 0.4) * TRUE_C10 / 6
        assert abs(fitted - TRUE_C10) <= step

    def test_recorded_losses_are_the_requested_objective(self, dataset4d):
        """`_grid_search_results` must hold the loss actually optimized, not the default."""
        recon = _build(dataset4d, aberration_coefs={})
        recon.grid_search_hyperparameters(
            aberration_coefs={
                "C10": OptimizationParameter(low=0.4 * TRUE_C10, high=1.6 * TRUE_C10, n_points=3)
            },
            loss="rms_gradient",
            deconvolution_kernel="prlx",
            verbose=False,
        )

        # the RMS gradient loss is negative; the residual loss is positive
        assert all(loss < 0 for _, loss in recon._grid_search_results)

    @pytest.mark.parametrize("bad", ["sharpness", 3, None])
    def test_unknown_losses_are_rejected(self, dataset4d, bad):
        recon = _build(dataset4d)
        recon.reconstruct(deconvolution_kernel="prlx", verbose=False)
        with pytest.raises(ValueError, match="must be a callable or one of"):
            recon._return_loss_value(bad)


class TestHyperparameterState:
    def test_optimized_overrides_initial(self, recon):
        state = recon.hyperparameter_state
        state.clear_optimized()

        assert state.current_rotation_angle() == 0.0
        state.optimized_rotation_angle = 12.0
        assert state.current_rotation_angle() == 12.0
        assert state.current_rotation_angle(override_fixed=3.0) == 3.0

        state.clear_optimized()
        assert state.current_aberrations()["C10"] == pytest.approx(TRUE_C10)

    def test_defocus_alias_is_negated(self):
        from quantem.diffractive_imaging.direct_ptychography_base import HyperparameterState

        state = HyperparameterState(initial_aberrations={"defocus": 100.0})

        assert state.initial_aberrations == {"C10": -100.0}

    def test_rejects_unknown_aberrations(self):
        from quantem.diffractive_imaging.direct_ptychography_base import HyperparameterState

        with pytest.raises(ValueError):
            HyperparameterState(initial_aberrations={"C99": 1.0})


class TestSerialization:
    def test_round_trip_preserves_the_reconstruction(self, dataset4d, tmp_path):
        recon = _build(dataset4d)
        recon.reconstruct(deconvolution_kernel="ssb", verbose=False)
        before = recon.obj.copy()

        path = str(tmp_path / "direct.zip")
        recon.save(path, mode="o")
        restored = load(path)

        assert isinstance(restored, DirectPtychography)
        assert np.array_equal(restored.obj, before)
        assert restored.num_bf == recon.num_bf
        assert restored.gpts == recon.gpts

        restored.reconstruct(deconvolution_kernel="ssb", verbose=False)
        assert np.allclose(restored.obj, before)

    def test_the_hyperparameter_state_saves_natively(self, dataset4d, tmp_path, capsys):
        from quantem.diffractive_imaging.direct_ptychography_base import HyperparameterState

        recon = _build(dataset4d)
        recon.hyperparameter_state.optimized_aberrations = {"C10": 42.0, "C12": 3.0}
        recon.hyperparameter_state.optimized_rotation_angle = 1.5
        recon.hyperparameter_state.optimized_keys = {"C10", "C12", "rotation_angle"}
        recon.hyperparameter_state.optimized_slice_thicknesses = [6.0]

        path = str(tmp_path / "direct.zip")
        recon.save(path, mode="o")
        # nothing fell back to pickling, which the serializer announces on stdout
        assert "falling back" not in capsys.readouterr().out

        state = load(path).hyperparameter_state
        assert isinstance(state, HyperparameterState)
        assert state.current_aberrations() == recon.hyperparameter_state.current_aberrations()
        assert state.current_rotation_angle() == 1.5
        assert state.optimized_keys == {"C10", "C12", "rotation_angle"}
        assert state.current_slice_thicknesses() == [6.0]


class TestRotationSensitivity:
    def test_wrong_rotation_degrades_the_reconstruction(self, dataset4d):
        """The data is simulated unrotated, so 0 degrees must beat a large rotation."""
        recon = _build(dataset4d)
        reference = band_limited_phase()

        recon.reconstruct(deconvolution_kernel="prlx", override_rotation_angle=0.0, verbose=False)
        aligned = abs(correlation(recon.obj, reference))
        recon.reconstruct(deconvolution_kernel="prlx", override_rotation_angle=60.0, verbose=False)
        misaligned = abs(correlation(recon.obj, reference))

        assert aligned > misaligned

    def test_detector_rotation_is_estimated_when_not_supplied(self, dataset4d):
        recon = DirectPtychography.from_dataset4d(
            dataset4d,
            energy=PROBE_ENERGY,
            semiangle_cutoff=SEMIANGLE_CUTOFF,
            rotation_angle=None,
            aberration_coefs={"C10": TRUE_C10},
            edge_blend_pixels=0,
            verbose=False,
        )

        # simulated without rotation; the curl-minimizing estimate should land near 0 or 180
        estimated = abs(recon.rotation_angle) % 180
        assert min(estimated, 180 - estimated) < 20


class TestReconstructAllPermutations:
    def test_returns_one_image_per_kernel(self, recon):
        images = recon._reconstruct_all_permutations(verbose=False)

        assert len(images) == len(DECONVOLUTION_KERNELS) + 2  # with "wiener" and "wdd"
        assert all(image.shape == (N, N) for image in images)
        assert all(np.isfinite(image).all() for image in images)

    @pytest.mark.parametrize(
        "kwargs, count",
        [
            (dict(obj_type="complex"), 5),
            (dict(num_slices=2, slice_mode="joint", slice_thicknesses=5.0), 2),
            (dict(prior="ard"), 1),
        ],
    )
    def test_leaves_out_the_kernels_that_cannot_take_the_settings(self, recon, kwargs, count):
        assert len(recon._reconstruct_all_permutations(verbose=False, **kwargs)) == count


class TestNormalizationOrder:
    def test_linear_background_normalization_runs(self, dataset4d):
        recon = _build(dataset4d, normalization_order=1)
        recon.reconstruct(deconvolution_kernel="ssb", verbose=False)

        assert np.isfinite(recon.obj).all()
        assert abs(correlation(recon.obj, band_limited_phase())) > 0.5

    def test_rejects_unknown_order(self, dataset4d):
        with pytest.raises(ValueError, match="normalization_order"):
            _build(dataset4d, normalization_order=2)

    def test_edge_blending_tapers_the_stack(self, dataset4d):
        """A nonzero blend pulls the scan-edge vBF values toward unity."""
        blended = _build(dataset4d, edge_blend_pixels=4)
        sharp = _build(dataset4d, edge_blend_pixels=0)

        edge_blended = blended.vbf_stack[:, 0, :]
        edge_sharp = sharp.vbf_stack[:, 0, :]

        assert (edge_blended - 1).abs().mean() < (edge_sharp - 1).abs().mean()


class TestDatasetVariants:
    def test_reconstruction_tracks_the_simulated_defocus(self):
        """Data simulated at a different defocus must prefer that defocus."""
        other_C10 = 2 * TRUE_C10
        dataset = make_dataset4d(defocus=other_C10)
        recon = DirectPtychography.from_dataset4d(
            dataset, edge_blend_pixels=0, **direct_ptycho_kwargs(other_C10)
        )

        losses = {}
        for scale in (0.5, 1.0, 1.5):
            recon.reconstruct(
                deconvolution_kernel="prlx",
                override_aberration_coefs={"C10": scale * other_C10},
                parallax_flip_phase=False,
                verbose=False,
            )
            losses[scale] = recon.residual_loss()

        assert min(losses, key=losses.get) == pytest.approx(1.0)


class TestFromDataset3d:
    """Ungridded scans, by resampling the bright-field stack onto a grid first.

    Every kernel here is a scan-space Fourier multiplier and so needs a regular grid. Once
    regridded they all run unchanged, which is what makes this exact for positions that were
    already on a lattice.
    """

    @staticmethod
    def _dataset3d(dataset4d):
        return Dataset3d.from_array(
            np.asarray(dataset4d.array).reshape(-1, N, N),
            name="ungridded patterns",
            sampling=(1.0, dataset4d.sampling[-2], dataset4d.sampling[-1]),
            units=("index", "A^-1", "A^-1"),
        )

    @staticmethod
    def _build(dataset3d, positions, **overrides):
        kwargs = dict(
            energy=PROBE_ENERGY,
            semiangle_cutoff=SEMIANGLE_CUTOFF,
            rotation_angle=0.0,
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            aberration_coefs={"C10": TRUE_C10},
            force_fitted_origin=(N // 2, N // 2),
            verbose=False,
        )
        kwargs.update(overrides)
        return DirectPtychography.from_dataset3d(dataset3d, positions, **kwargs)

    @pytest.mark.parametrize("kernel", DECONVOLUTION_KERNELS)
    def test_lattice_positions_reproduce_from_dataset4d(self, dataset4d, kernel):
        """On a lattice the splat is an identity map, so this must be exact."""
        gridded = _build(dataset4d)
        ungridded = self._build(self._dataset3d(dataset4d), scan_positions_px() * SCAN_SAMPLING)

        assert ungridded.scan_gpts == gridded.scan_gpts
        assert np.allclose(
            ungridded.reconstruct(deconvolution_kernel=kernel, verbose=False).obj,
            gridded.reconstruct(deconvolution_kernel=kernel, verbose=False).obj,
            atol=1e-6,
        )

    def test_position_axis_order_is_row_col(self, dataset4d):
        """Swapping the columns must transpose the regridded stack, not scramble it.

        Checked with no aberrations, so the parallax shifts vanish. With a shift present the
        relation does not hold: the shifts come from the detector k-grid, which transposing
        the *positions* leaves alone.
        """
        dataset3d = self._dataset3d(dataset4d)
        positions = scan_positions_px() * SCAN_SAMPLING
        flat = dict(aberration_coefs={}, scan_gpts=(N, N))

        row_col = self._build(dataset3d, positions, **flat)
        col_row = self._build(dataset3d, positions[:, ::-1].copy(), **flat)

        assert np.allclose(
            to_numpy(col_row.vbf_stack),
            to_numpy(row_col.vbf_stack).transpose(0, 2, 1),
            atol=1e-5,
        )

    def test_jittered_positions_recover_the_object(self, dataset4d):
        rng = np.random.default_rng(0)
        positions = scan_positions_px() * SCAN_SAMPLING
        positions = positions + rng.uniform(-0.3, 0.3, positions.shape) * SCAN_SAMPLING

        reconstruction = self._build(self._dataset3d(dataset4d), positions)
        obj = reconstruction.reconstruct(deconvolution_kernel="prlx", verbose=False).obj

        assert correlation(obj[:N, :N], band_limited_phase()) > 0.5

    def test_warns_when_positions_are_clustered(self, dataset4d):
        """Enough positions to cover the grid, yet most of it empty: genuinely uneven."""
        positions = scan_positions_px() * SCAN_SAMPLING
        corner = positions[(positions[:, 0] < 10) & (positions[:, 1] < 10)]
        # more positions than grid pixels, but all piled onto 100 distinct spots
        repeated = np.repeat(corner, 10, axis=0)

        with pytest.warns(UserWarning, match="clustered rather than merely sparse"):
            DirectPtychography.from_dataset3d(
                Dataset3d.from_array(
                    np.asarray(dataset4d.array).reshape(-1, N, N)[: len(repeated)],
                    name="clustered",
                    sampling=(1.0, dataset4d.sampling[-2], dataset4d.sampling[-1]),
                    units=("index", "A^-1", "A^-1"),
                ),
                repeated,
                energy=PROBE_ENERGY,
                semiangle_cutoff=SEMIANGLE_CUTOFF,
                rotation_angle=0.0,
                scan_sampling=(SCAN_SAMPLING / 2, SCAN_SAMPLING / 2),
                aberration_coefs={"C10": TRUE_C10},
                force_fitted_origin=(N // 2, N // 2),
                verbose=False,
            )

    def test_a_finer_grid_does_not_warn_about_its_gaps(self, dataset4d):
        """Fewer positions than pixels is a comb, not missing data."""
        positions = scan_positions_px() * SCAN_SAMPLING

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            self._build(
                self._dataset3d(dataset4d),
                positions,
                scan_sampling=(SCAN_SAMPLING / 2, SCAN_SAMPLING / 2),
            )

    def test_no_warning_on_a_fully_covered_grid(self, dataset4d):
        positions = scan_positions_px() * SCAN_SAMPLING

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            self._build(self._dataset3d(dataset4d), positions)

    def test_scan_gpts_pads_without_rescaling_the_positions(self, dataset4d):
        """`scan_sampling` stays authoritative: a bigger grid pads, it does not rescale."""
        positions = scan_positions_px() * SCAN_SAMPLING
        plain = self._build(self._dataset3d(dataset4d), positions)
        padded = self._build(self._dataset3d(dataset4d), positions, scan_gpts=(N + 8, N + 8))

        assert padded.scan_gpts == (N + 8, N + 8)
        assert padded.scan_sampling[0] == pytest.approx(plain.scan_sampling[0])

    def test_scan_gpts_too_small_raises(self, dataset4d):
        positions = scan_positions_px() * SCAN_SAMPLING
        with pytest.raises(ValueError, match="positions would be dropped"):
            self._build(self._dataset3d(dataset4d), positions, scan_gpts=(N // 2, N // 2))

    def test_auto_scan_sampling_warns_and_infers(self, dataset4d):
        positions = scan_positions_px() * SCAN_SAMPLING

        with pytest.warns(UserWarning, match="Inferred scan_sampling"):
            reconstruction = self._build(
                self._dataset3d(dataset4d), positions, scan_sampling="auto"
            )

        assert reconstruction.scan_sampling[0] == pytest.approx(SCAN_SAMPLING)

    def test_accepts_a_dataset2d_of_positions(self, dataset4d):
        positions = Dataset2d.from_array(
            scan_positions_px() * SCAN_SAMPLING,
            name="positions",
            sampling=(1.0, 1.0),
            units=("A", "A"),
        )
        reconstruction = self._build(self._dataset3d(dataset4d), positions)

        assert reconstruction.scan_gpts == (N, N)

    def test_rejects_positions_in_the_wrong_units(self, dataset4d):
        positions = Dataset2d.from_array(
            scan_positions_px() * SCAN_SAMPLING,
            name="positions",
            sampling=(1.0, 1.0),
            units=("nm", "nm"),
        )
        with pytest.raises(ValueError, match="must be given in 'A'"):
            self._build(self._dataset3d(dataset4d), positions)

    def test_rejects_mismatched_position_count(self, dataset4d):
        with pytest.raises(ValueError, match="rows but `dataset` has"):
            self._build(self._dataset3d(dataset4d), scan_positions_px()[:10] * SCAN_SAMPLING)

    def test_rejects_linear_normalization(self, dataset4d):
        with pytest.raises(ValueError, match="needs a scan grid"):
            self._build(
                self._dataset3d(dataset4d),
                scan_positions_px() * SCAN_SAMPLING,
                normalization_order=1,
            )

    def test_survives_a_serialization_round_trip(self, dataset4d, tmp_path):
        reconstruction = self._build(
            self._dataset3d(dataset4d), scan_positions_px() * SCAN_SAMPLING
        )
        reconstruction.reconstruct(deconvolution_kernel="prlx", verbose=False)
        path = tmp_path / "ungridded.zip"
        reconstruction.save(path, mode="o")

        assert np.allclose(load(path).obj, reconstruction.obj)

    @staticmethod
    def _disk_masked(dataset4d, radius=14.0):
        """A non-rectangular scan subset -- the case that exposes hole handling."""
        rows_cols = scan_positions_px()
        center = (N - 1) / 2
        keep = ((rows_cols[:, 0] - center) ** 2 + (rows_cols[:, 1] - center) ** 2) < radius**2
        dataset3d = Dataset3d.from_array(
            np.asarray(dataset4d.array).reshape(-1, N, N)[keep],
            name="masked patterns",
            sampling=(1.0, dataset4d.sampling[-2], dataset4d.sampling[-1]),
            units=("index", "A^-1", "A^-1"),
        )
        return dataset3d, rows_cols[keep] * SCAN_SAMPLING, keep

    def test_mean_hole_fill_beats_zero_on_a_masked_scan(self, dataset4d):
        """Regression guard: zero-filled holes wreck the reconstruction, mean-filled do not.

        `_preprocess` zeroes the DC bin, subtracting the mean over the whole grid including
        holes, so zero-filled holes sit at `-mean` -- a hard-edged step the deconvolution
        smears everywhere. Measured 0.25 vs 0.69 correlation with ground truth.
        """
        dataset3d, positions, keep = self._disk_masked(dataset4d)
        rows_cols = scan_positions_px()[keep].astype(int)
        low, high = rows_cols.min(0), rows_cols.max(0) + 1
        truth = band_limited_phase()[low[0] : high[0], low[1] : high[1]]

        def score(hole_fill):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                recon = self._build(dataset3d, positions, hole_fill=hole_fill)
            obj = recon.reconstruct(deconvolution_kernel="prlx", verbose=False).obj
            return correlation(obj[: truth.shape[0], : truth.shape[1]], truth)

        assert score("mean") > 0.6
        assert score("mean") > score("zero") + 0.2

    def test_mean_fill_matches_the_montage_on_a_masked_scan(self, dataset4d):
        """With holes filled, the two formulations agree on data neither was built for."""
        from quantem.diffractive_imaging import DirectPtychographyMontage

        dataset3d, positions, keep = self._disk_masked(dataset4d)
        rows_cols = scan_positions_px()[keep].astype(int)
        low, high = rows_cols.min(0), rows_cols.max(0) + 1
        truth = band_limited_phase()[low[0] : high[0], low[1] : high[1]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fourier = self._build(dataset3d, positions)
        montage = DirectPtychographyMontage.from_dataset3d(
            dataset3d,
            positions,
            energy=PROBE_ENERGY,
            semiangle_cutoff=SEMIANGLE_CUTOFF,
            rotation_angle=0.0,
            scan_sampling=(SCAN_SAMPLING, SCAN_SAMPLING),
            aberration_coefs={"C10": TRUE_C10},
            force_fitted_origin=(N // 2, N // 2),
            verbose=False,
        )

        fourier_obj = fourier.reconstruct(deconvolution_kernel="prlx", verbose=False).obj
        montage.reconstruct(deconvolution_kernel="prlx", weight_normalize=False, verbose=False)
        origin = to_numpy(montage._canvas_origin_px)
        row0, col0 = int(round(-origin[0])), int(round(-origin[1]))
        montage_obj = montage.obj[row0 : row0 + truth.shape[0], col0 : col0 + truth.shape[1]]

        fourier_corr = correlation(fourier_obj[: truth.shape[0], : truth.shape[1]], truth)
        montage_corr = correlation(montage_obj, truth)

        assert fourier_corr > 0.6
        assert abs(fourier_corr - montage_corr) < 0.1

    def test_rejects_an_unknown_hole_fill(self, dataset4d):
        dataset3d, positions, _ = self._disk_masked(dataset4d)
        with pytest.raises(ValueError, match="`hole_fill` must be"):
            self._build(dataset3d, positions, hole_fill="interpolate")

    def test_upsampling_an_irregular_regrid_warns(self, dataset4d):
        """Regridding discards the sub-pixel positions upsampling needs to unfold."""
        rng = np.random.default_rng(3)
        positions = scan_positions_px() * SCAN_SAMPLING
        positions = positions + rng.uniform(-0.4, 0.4, positions.shape) * SCAN_SAMPLING
        recon = self._build(self._dataset3d(dataset4d), positions)

        assert recon._regrid_info["lattice_rms_px"] > 0.1
        with pytest.warns(UserWarning, match="cannot unfold"):
            recon.reconstruct(deconvolution_kernel="prlx", upsampling_factor=2, verbose=False)

    def test_no_unfolding_warning_on_a_lattice_or_without_upsampling(self, dataset4d):
        dataset3d = self._dataset3d(dataset4d)
        on_lattice = self._build(dataset3d, scan_positions_px() * SCAN_SAMPLING)

        assert on_lattice._regrid_info["lattice_rms_px"] == pytest.approx(0.0, abs=1e-9)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            on_lattice.reconstruct(deconvolution_kernel="prlx", upsampling_factor=2, verbose=False)

        rng = np.random.default_rng(3)
        positions = scan_positions_px() * SCAN_SAMPLING
        positions = positions + rng.uniform(-0.4, 0.4, positions.shape) * SCAN_SAMPLING
        irregular = self._build(dataset3d, positions)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            irregular.reconstruct(deconvolution_kernel="prlx", verbose=False)

    def test_gridded_reconstructions_never_warn_about_unfolding(self, dataset4d):
        """`from_dataset4d` has no regrid info, so the check must be a no-op there."""
        gridded = _build(dataset4d)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            gridded.reconstruct(deconvolution_kernel="prlx", upsampling_factor=4, verbose=False)


class TestUnfolding:
    """Does `upsampling_factor` extend the CTF, or merely replicate it?

    Uses the white-noise object, whose Fourier amplitude is constant, so
    `|FFT(reconstruction)|` is the contrast transfer function and can be compared against an
    analytical one. The scan pitch is half the aperture's transfer limit, so there is a
    genuine factor of two above the scan Nyquist to recover -- without that, an upsampling
    test measures nothing.
    """

    @staticmethod
    def _scores(obj, analytic, upsampling_factor):
        return ctf_band_scores(measured_ctf(obj), analytic, CTF_SCAN_SAMPLING / upsampling_factor)

    @staticmethod
    def _ungridded(scene, positions, **overrides):
        _, _, probe, _, _ = scene
        kwargs = dict(
            ctf_kwargs(),
            scan_sampling=(CTF_SCAN_SAMPLING, CTF_SCAN_SAMPLING),
        )
        kwargs.update(overrides)
        return DirectPtychography.from_dataset3d(
            ctf_dataset3d(ctf_simulate(scene[0], probe, positions)),
            positions * CTF_SAMPLING,
            **kwargs,
        )

    def test_gridded_upsampling_unfolds_the_ctf(self, ctf_scene):
        """The baseline the ungridded path is judged against."""
        obj, _, probe, ctf_full, ctf_sub = ctf_scene
        recon = DirectPtychography.from_dataset4d(
            ctf_dataset4d(ctf_simulate(obj, probe, ctf_raster_positions())),
            edge_blend_pixels=0,
            **ctf_kwargs(),
        )

        in_band, _ = self._scores(
            recon.reconstruct(deconvolution_kernel="prlx", verbose=False).obj, ctf_sub, 1
        )
        up = recon.reconstruct(deconvolution_kernel="prlx", upsampling_factor=2, verbose=False).obj
        up_in, up_ext = self._scores(up, ctf_full, 2)

        assert in_band > 0.95
        assert up_in > 0.95
        assert up_ext > 0.95  # genuinely extended, not replicated

    def test_lattice_positions_unfold_through_the_ungridded_path(self, ctf_scene):
        _, _, _, ctf_full, _ = ctf_scene
        recon = self._ungridded(ctf_scene, ctf_raster_positions())

        obj = recon.reconstruct(
            deconvolution_kernel="prlx", upsampling_factor=2, verbose=False
        ).obj

        assert self._scores(obj, ctf_full, 2)[1] > 0.95

    def test_upsampling_factor_degrades_on_an_irregular_scan(self, ctf_scene):
        """Binning first discards the sub-pixel positions upsampling needs.

        Compared against the same call on a lattice rather than an absolute threshold: what
        matters is that scattering the positions costs the extension band, not where the
        number happens to land.
        """
        _, _, _, ctf_full, _ = ctf_scene

        on_lattice = (
            self._ungridded(ctf_scene, ctf_raster_positions())
            .reconstruct(deconvolution_kernel="prlx", upsampling_factor=2, verbose=False)
            .obj
        )
        scattered_recon = self._ungridded(ctf_scene, ctf_jittered_positions(1.0))
        with pytest.warns(UserWarning, match="cannot unfold"):
            scattered = scattered_recon.reconstruct(
                deconvolution_kernel="prlx", upsampling_factor=2, verbose=False
            ).obj

        assert (
            self._scores(scattered, ctf_full, 2)[1]
            < self._scores(on_lattice, ctf_full, 2)[1] - 0.1
        )

    def test_a_finer_scan_sampling_recovers_it(self, ctf_scene):
        """Splatting straight onto a finer grid keeps the probes on it."""
        _, _, _, ctf_full, _ = ctf_scene
        positions = ctf_jittered_positions(1.0)

        binned = self._ungridded(ctf_scene, positions)
        with pytest.warns(UserWarning, match="cannot unfold"):
            coarse = binned.reconstruct(
                deconvolution_kernel="prlx", upsampling_factor=2, verbose=False
            ).obj

        refined = self._ungridded(ctf_scene, positions, scan_sampling=(CTF_SCAN_SAMPLING / 2,) * 2)
        fine = refined.reconstruct(deconvolution_kernel="prlx", verbose=False).obj

        fine_corr = ctf_radial_correlation(measured_ctf(fine), ctf_full, refined.scan_sampling[0])
        coarse_corr = ctf_radial_correlation(
            measured_ctf(coarse), ctf_full, binned.scan_sampling[0] / 2
        )
        assert fine_corr > 0.99
        assert fine_corr > coarse_corr

    def test_a_finer_scan_sampling_refines_the_grid(self, ctf_scene):
        plain = self._ungridded(ctf_scene, ctf_raster_positions())
        finer = self._ungridded(
            ctf_scene, ctf_raster_positions(), scan_sampling=(CTF_SCAN_SAMPLING / 2,) * 2
        )

        assert finer.scan_sampling[0] == pytest.approx(plain.scan_sampling[0] / 2)
        assert finer.scan_gpts[0] >= 2 * plain.scan_gpts[0] - 2

    def test_nearest_beats_bilinear_on_a_finer_grid(self, ctf_scene):
        """Bilinear smears each measurement over four pixels and blurs out the detail."""
        _, _, _, ctf_full, _ = ctf_scene
        positions = ctf_jittered_positions(1.0)
        fine = dict(scan_sampling=(CTF_SCAN_SAMPLING / 2,) * 2)

        scores = {}
        for scheme in ("nearest", "bilinear"):
            recon = self._ungridded(ctf_scene, positions, interpolation=scheme, **fine)
            obj = recon.reconstruct(deconvolution_kernel="prlx", verbose=False).obj
            scores[scheme] = ctf_radial_correlation(
                measured_ctf(obj), ctf_full, recon.scan_sampling[0]
            )

        assert scores["nearest"] > scores["bilinear"]

    def test_zero_fill_inverts_a_finer_grid(self, ctf_scene):
        """`hole_fill="mean"` centers the comb's gaps; leaving them at zero inverts it."""
        _, _, _, ctf_full, _ = ctf_scene
        positions = ctf_jittered_positions(1.0)
        fine = dict(scan_sampling=(CTF_SCAN_SAMPLING / 2,) * 2)

        scores = {}
        for fill in ("mean", "zero"):
            recon = self._ungridded(ctf_scene, positions, hole_fill=fill, **fine)
            obj = recon.reconstruct(deconvolution_kernel="prlx", verbose=False).obj
            scores[fill] = ctf_radial_correlation(
                measured_ctf(obj), ctf_full, recon.scan_sampling[0]
            )

        assert scores["mean"] > 0.99
        assert scores["zero"] < scores["mean"] - 0.1

    def test_montage_unfolds_without_any_regridding(self, ctf_scene):
        """The reference: the montage never bins, so irregularity costs it nothing."""
        from quantem.diffractive_imaging import DirectPtychographyMontage

        obj, _, probe, ctf_full, _ = ctf_scene
        positions = ctf_jittered_positions(1.0)
        montage = DirectPtychographyMontage.from_dataset3d(
            ctf_dataset3d(ctf_simulate(obj, probe, positions)),
            positions * CTF_SAMPLING,
            scan_sampling=(CTF_SCAN_SAMPLING, CTF_SCAN_SAMPLING),
            boundary="wrap",
            **ctf_kwargs(),
        )

        result = montage.reconstruct(
            deconvolution_kernel="prlx", upsampling_factor=2, verbose=False
        ).obj

        assert self._scores(result, ctf_full, 2)[1] > 0.95


class TestOptimizationParameterIsOneClass:
    """`OptimizationParameter` used to be defined twice, identically.

    One copy lived in ``direct_ptychography`` and the other in
    ``optimize_hyperparameters``. Both searches test candidate specifications with
    ``isinstance``, so a value built from one module was silently ignored by the other
    rather than raising. These pin the single definition in place.
    """

    def test_every_import_path_is_the_same_class(self):
        from quantem.diffractive_imaging import OptimizationParameter as package
        from quantem.diffractive_imaging.direct_ptychography import (
            OptimizationParameter as direct,
        )
        from quantem.diffractive_imaging.direct_ptychography_base import (
            OptimizationParameter as base,
        )
        from quantem.diffractive_imaging.optimize_hyperparameters import (
            OptimizationParameter as iterative,
        )
        from quantem.diffractive_imaging.ptycho_utils import (
            OptimizationParameter as canonical,
        )

        assert package is direct is base is iterative is canonical
        assert canonical.__module__ == "quantem.diffractive_imaging.ptycho_utils"

    def test_a_spec_built_anywhere_is_accepted_everywhere(self):
        """The failure the duplicate caused: isinstance across the two searches."""
        from quantem.diffractive_imaging import OptimizationParameter
        from quantem.diffractive_imaging.optimize_hyperparameters import (
            OptimizationParameter as iterative,
        )

        assert isinstance(OptimizationParameter(0.0, 1.0), iterative)


# ---------------------------------------------------------------------------
# complex-valued objects
# ---------------------------------------------------------------------------

WEAK = 0.02  # white-noise strength: first order in the object, second-order terms ~1e-4


def _white_noise_scene(channel, aberrations=ML_ABERRATIONS):
    """``(reconstruction, truth)`` for a weak pure-phase or pure-amplitude white-noise object."""
    truth = WEAK * white_noise_object_2D(n=64, phi0=1.0)
    log_t = 1j * truth if channel == "phase" else truth + 0j
    dataset = thin_layer_4dstem([log_t], [0.0], aberrations)
    return DirectPtychography.from_dataset4d(dataset, **ml_kwargs(aberrations)), truth


@pytest.fixture(scope="module")
def phase_scene():
    return _white_noise_scene("phase")


@pytest.fixture(scope="module")
def amplitude_scene():
    return _white_noise_scene("amplitude")


def _analytic_complex_transfer(recon, aberrations):
    """Per-frequency ``(phase, amplitude)`` transfer of the two-column solve, in numpy.

    Written independently of the library: the overlap functions from the closed-form probe,
    the Gram matrix and its pseudo-inverse, and the gains. The data are normalized per
    bright-field pixel by the mean intensity ``|psi(k)|^2``, so the Gram matrix carries the
    weight ``1 / |psi(k)|^2`` and the estimate is the truth; each channel's transfer is then
    just its gain.
    """
    wavelength = electron_wavelength_angstrom(PROBE_ENERGY)
    kxa, kya = spatial_frequencies(recon.gpts, recon.sampling)
    mask = to_numpy(recon.bf_mask)
    kx = to_numpy(kxa)[mask].astype(np.float64)
    ky = to_numpy(kya)[mask].astype(np.float64)
    psi_k = analytic_psi(kx, ky, aberrations, wavelength)

    q = np.fft.fftfreq(recon.corrected_bf.shape[0], ML_SAMPLING)
    qx, qy = np.meshgrid(q, q, indexing="ij")
    qx, qy = qx[..., None], qy[..., None]
    t_minus = analytic_psi(qx - kx, qy - ky, aberrations, wavelength) * psi_k.conj()
    t_plus = analytic_psi(qx + kx, qy + ky, aberrations, wavelength).conj() * psi_k
    columns = np.stack([t_minus - t_plus, t_minus + t_plus], -1)

    weights = np.abs(psi_k) ** 2
    gram = np.einsum("abki,abkj->abij", columns.conj(), columns / weights[:, None])

    # the library's conditioning: scale to unit diagonal, add a 1e-5 ridge, invert
    diagonal = np.real(np.diagonal(gram, axis1=-2, axis2=-1))
    scale = np.where(diagonal > 0, 1 / np.sqrt(np.where(diagonal > 0, diagonal, 1)), 0)
    scaled = gram * scale[..., :, None] * scale[..., None, :]
    inverse = np.linalg.inv(scaled + 1e-5 * np.eye(2))
    d = np.real(np.diagonal(inverse, axis1=-2, axis2=-1)) * scale**2
    s = np.where(d > 0, 1 / np.where(d > 0, d, 1), 0)
    gain = np.sqrt(s / weights.sum()) * np.array([1.0, 0.5])
    return gain[..., 0], gain[..., 1]


def _band(shape, sampling=ML_SAMPLING, lo=0.0, hi=2 * ML_K_PROBE):
    qx = np.fft.fftfreq(shape[0], sampling)
    qy = np.fft.fftfreq(shape[1], sampling)
    q = np.hypot(qx[:, None], qy[None, :])
    return (q > lo) & (q < hi)


class TestComplexObject:
    """``obj_type="complex"`` recovers the log-amplitude alongside the phase."""

    @pytest.mark.parametrize("channel", ["phase", "amplitude"])
    def test_transfer_matches_the_analytic_solve(self, phase_scene, amplitude_scene, channel):
        recon, truth = phase_scene if channel == "phase" else amplitude_scene
        recon.reconstruct(deconvolution_kernel="obf", obj_type="complex", verbose=False)
        image = recon.corrected_bf if channel == "phase" else recon.corrected_amplitude
        measured = np.abs(np.fft.fft2(to_numpy(image))) / np.abs(np.fft.fft2(truth))

        phase_t, amplitude_t = _analytic_complex_transfer(recon, ML_ABERRATIONS)
        expected = phase_t if channel == "phase" else amplitude_t
        sel = _band(truth.shape) & (expected > 0.05 * expected.max())

        assert np.corrcoef(measured[sel], expected[sel])[0, 1] > 0.999
        assert np.median(measured[sel] / expected[sel]) == pytest.approx(1.0, abs=0.02)

    @pytest.mark.parametrize("kernel", ["ssb", "obf"])
    def test_the_channels_do_not_leak(self, phase_scene, amplitude_scene, kernel):
        """With an aberrated probe the overlap functions are parallel on the triple overlap,
        so an independent kernel per channel would leak. The joint solve does not."""
        for (recon, _), own_is_phase in ((phase_scene, True), (amplitude_scene, False)):
            recon.reconstruct(deconvolution_kernel=kernel, obj_type="complex", verbose=False)
            phase = to_numpy(recon.corrected_bf)
            amplitude = to_numpy(recon.corrected_amplitude)
            own, other = (phase, amplitude) if own_is_phase else (amplitude, phase)
            assert other.std() < 1e-3 * own.std()

    def test_ssb_amplitude_transfer_is_aberration_independent(self, amplitude_scene):
        """Strategy D reads only the double overlaps, where ``Gamma+ = +-Gamma-``."""
        aberrated, truth = amplitude_scene
        in_focus, _ = _white_noise_scene("amplitude", {"C10": 0.0})
        transfers = []
        for recon in (aberrated, in_focus):
            recon.reconstruct(deconvolution_kernel="ssb", obj_type="complex", verbose=False)
            spectrum = np.abs(np.fft.fft2(to_numpy(recon.corrected_amplitude)))
            transfers.append(spectrum / np.abs(np.fft.fft2(truth)))
        sel = _band(truth.shape) & (transfers[1] > 0.05 * transfers[1].max())

        assert np.median(transfers[0][sel] / transfers[1][sel]) == pytest.approx(1.0, abs=0.02)

    def test_icom_amplitude_is_the_bright_field_sum(self, amplitude_scene):
        recon, _ = amplitude_scene
        recon.reconstruct(
            deconvolution_kernel="icom", obj_type="complex", band_limit=False, verbose=False
        )
        from quantem.diffractive_imaging.complex_probe import polar_coordinates

        stack = recon.vbf_stack.double()
        bf_sum = (stack - stack.mean((-2, -1), keepdim=True)).sum(0)
        k, phi = polar_coordinates(
            *spatial_frequencies(recon.gpts, recon.sampling, rotation_angle=0.0)
        )
        weights = recon._return_probe_on_grid(k, phi, ML_ABERRATIONS)[recon.bf_mask]
        expected = to_numpy(bf_sum / 2 / weights.abs().square().sum().double())

        amplitude = to_numpy(recon.corrected_amplitude)
        assert np.abs(amplitude - expected).max() < 1e-3 * np.abs(expected).max()

    def test_obj_is_the_complex_transmission(self, amplitude_scene):
        recon, _ = amplitude_scene
        recon.reconstruct(deconvolution_kernel="obf", obj_type="complex", verbose=False)
        obj = recon.obj

        assert np.iscomplexobj(obj)
        assert np.allclose(np.abs(obj), np.exp(to_numpy(recon.corrected_amplitude)), rtol=1e-6)
        assert np.allclose(np.angle(obj), to_numpy(recon.corrected_bf), atol=1e-6)
        assert recon.obj_type == "complex"

    def test_a_defocused_phase_layer_appears_as_amplitude(self):
        """Free-space propagation turns phase into amplitude, which the complex solve reports
        at the wrong plane and not at the right one."""
        truth = WEAK * white_noise_object_2D(n=64, phi0=1.0)
        depth = 40.0
        dataset = thin_layer_4dstem([1j * truth], [depth], ML_ABERRATIONS)
        recon = DirectPtychography.from_dataset4d(dataset, **ml_kwargs(ML_ABERRATIONS))

        amplitudes = {}
        for label, c10 in (
            ("at the layer", ML_ABERRATIONS["C10"] + depth),
            ("off it", ML_ABERRATIONS["C10"]),
        ):
            recon.reconstruct(
                deconvolution_kernel="obf",
                obj_type="complex",
                override_aberration_coefs={"C10": c10},
                verbose=False,
            )
            amplitudes[label] = (
                to_numpy(recon.corrected_amplitude).std() / to_numpy(recon.corrected_bf).std()
            )

        assert amplitudes["off it"] > 10 * amplitudes["at the layer"]

    def test_parallax_rejects_complex(self, recon):
        with pytest.raises(ValueError, match="cannot separate amplitude from phase"):
            recon.reconstruct(deconvolution_kernel="prlx", obj_type="complex", verbose=False)

    def test_unknown_obj_type_raises(self, recon):
        with pytest.raises(ValueError, match="obj_type"):
            recon.reconstruct(deconvolution_kernel="obf", obj_type="potential", verbose=False)

    def test_pure_phase_has_no_amplitude(self, recon):
        recon.reconstruct(deconvolution_kernel="obf", verbose=False)

        assert recon.corrected_amplitude is None
        assert not np.iscomplexobj(recon.obj)
        assert recon.obj_type == "pure_phase"

    def test_visualize_shows_both_channels(self, amplitude_scene):
        import matplotlib

        matplotlib.use("Agg")
        recon, _ = amplitude_scene
        recon.reconstruct(deconvolution_kernel="obf", obj_type="complex", verbose=False)
        fig, axs = recon.visualize(return_fig=True)

        assert np.size(axs) == 4

    def test_round_trip_preserves_both_channels(self, amplitude_scene, tmp_path):
        recon, _ = amplitude_scene
        recon.reconstruct(deconvolution_kernel="obf", obj_type="complex", verbose=False)
        before = recon.obj.copy()

        path = str(tmp_path / "complex.zip")
        recon.save(path, mode="o")
        restored = load(path)

        assert np.array_equal(restored.obj, before)
        assert restored.obj_type == "complex"

    @pytest.mark.parametrize("kernel", ["ssb", "obf"])
    def test_an_empirical_probe_reproduces_the_analytic_one(self, dataset4d, kernel):
        from quantem.diffractive_imaging.complex_probe import FourierProbe

        analytic = _build(dataset4d)
        psi = analytic_probe_array(analytic, {"C10": TRUE_C10})
        probe = FourierProbe.from_array(
            psi, analytic.reciprocal_sampling, analytic.wavelength, normalize=False
        )
        empirical = _build(dataset4d, fourier_probe=probe)

        kwargs = dict(deconvolution_kernel=kernel, obj_type="complex", verbose=False)
        expected = analytic.reconstruct(**kwargs).obj.copy()
        assert np.allclose(empirical.reconstruct(**kwargs).obj, expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# multislice
# ---------------------------------------------------------------------------


def _twisted_scene(kind):
    """``(reconstruction, true layers, spacing)`` for two twisted layers about the C10 plane."""
    if kind == "square":
        _, layers, _ = twisted_square_cell()
        layers, spacing = [0.3 * layer for layer in layers], 6.0
    else:
        _, layers, _ = twisted_honeycomb_cell()
        layers, spacing = [0.1 * layer for layer in layers], 3.35
    depths = [-spacing / 2, spacing / 2]
    dataset = thin_layer_4dstem([1j * layer for layer in layers], depths, ML_ABERRATIONS)
    recon = DirectPtychography.from_dataset4d(dataset, **ml_kwargs(ML_ABERRATIONS))
    return recon, layers, spacing


@pytest.fixture(scope="module")
def square_scene():
    return _twisted_scene("square")


@pytest.fixture(scope="module")
def honeycomb_scene():
    return _twisted_scene("honeycomb")


class TestSectioning:
    """``slice_mode="sectioned"``: one single-plane reconstruction per slice depth."""

    SECTIONS = dict(num_slices=3, slice_thicknesses=50.0, slice_mode="sectioned", verbose=False)

    @pytest.mark.parametrize("kernel", ["ssb", "obf", "wiener", "wdd", "prlx"])
    def test_sections_are_single_plane_reconstructions(self, recon, kernel):
        recon.reconstruct(deconvolution_kernel=kernel, **self.SECTIONS)
        stack = to_numpy(recon.corrected_bf).copy()
        assert recon.slice_mode == "sectioned"
        assert np.allclose(recon.slice_depths, [-50.0, 0.0, 50.0])

        for section, depth in zip(stack, recon.slice_depths):
            recon.reconstruct(
                deconvolution_kernel=kernel,
                override_aberration_coefs={"C10": TRUE_C10 + depth},
                verbose=False,
            )
            assert np.array_equal(section, to_numpy(recon.corrected_bf))

    def test_sections_start_from_the_initial_state_when_asked(self, dataset4d):
        recon = _build(dataset4d)
        recon.hyperparameter_state.optimized_aberrations = {"C10": 2.0 * TRUE_C10}
        recon.reconstruct(deconvolution_kernel="obf", use_initial_state=True, **self.SECTIONS)
        from_initial = to_numpy(recon.corrected_bf).copy()
        recon.hyperparameter_state.clear_optimized()
        recon.reconstruct(deconvolution_kernel="obf", **self.SECTIONS)

        assert np.array_equal(from_initial, to_numpy(recon.corrected_bf))

    def test_a_complex_object_has_amplitude_sections(self, recon):
        recon.reconstruct(deconvolution_kernel="obf", obj_type="complex", **self.SECTIONS)
        assert recon.corrected_amplitude.shape == recon.corrected_bf.shape == (3, N, N)
        assert recon.obj.shape == (3, N, N)

    def test_each_section_has_its_own_residual(self):
        """A layer at the reference plane is explained best by the section through it."""
        _, layers, _ = twisted_square_cell()
        dataset = thin_layer_4dstem([1j * 0.3 * layers[0]], [0.0], ML_ABERRATIONS)
        recon = DirectPtychography.from_dataset4d(dataset, **ml_kwargs())
        recon.reconstruct(
            deconvolution_kernel="obf",
            num_slices=3,
            slice_thicknesses=15.0,
            slice_mode="sectioned",
            verbose=False,
        )

        assert recon.residual_loss() is None
        assert len(recon.section_residuals) == 3
        assert int(np.argmin(recon.section_residuals)) == 1

    def test_icom_cannot_be_sectioned(self, recon):
        with pytest.raises(ValueError, match="iCoM does not depend on the aberrations"):
            recon.reconstruct(deconvolution_kernel="icom", **self.SECTIONS)

    def test_kernels_without_a_joint_solve_point_to_sectioning(self, recon):
        with pytest.raises(ValueError, match="slice_mode='sectioned'"):
            recon.reconstruct(
                deconvolution_kernel="ssb",
                num_slices=2,
                slice_thicknesses=5.0,
                slice_mode="joint",
                verbose=False,
            )

    def test_sectioning_is_the_default(self, recon):
        recon.reconstruct(
            deconvolution_kernel="ssb", num_slices=2, slice_thicknesses=5.0, verbose=False
        )
        assert recon.slice_mode == "sectioned"

    @pytest.mark.parametrize("alias", ["sectioned", "depth-section", "sweep", "Section"])
    def test_aliases(self, recon, alias):
        recon.reconstruct(deconvolution_kernel="ssb", **dict(self.SECTIONS, slice_mode=alias))
        assert recon.slice_mode == "sectioned"

    def test_rejects_an_unknown_mode(self, recon):
        with pytest.raises(ValueError, match="`slice_mode`"):
            recon.reconstruct(deconvolution_kernel="obf", **dict(self.SECTIONS, slice_mode="x"))

    def test_a_joint_reconstruction_resets_the_mode(self, recon):
        recon.reconstruct(deconvolution_kernel="obf", **self.SECTIONS)
        recon.reconstruct(deconvolution_kernel="obf", verbose=False)
        assert recon.slice_mode == "joint"
        assert recon.section_residuals is None
        assert recon.residual_loss() is not None

    def test_permutations_leave_out_only_icom(self, recon):
        images = recon._reconstruct_all_permutations(**self.SECTIONS)
        assert len(images) == 5
        assert all(image.shape == (3, N, N) for image in images)

    def test_fits_refuse_the_residual_of_a_sectioned_stack(self, recon):
        sections = dict(num_slices=3, slice_thicknesses=50.0, slice_mode="sectioned")
        with pytest.raises(ValueError, match="no residual to minimize"):
            recon.grid_search_hyperparameters(
                aberration_coefs={"C10": OptimizationParameter(TRUE_C10, TRUE_C10, n_points=1)},
                verbose=False,
                **sections,
            )
        with pytest.raises(ValueError, match="no residual to minimize"):
            recon.fit_hyperparameters_gradient(("C10",), verbose=False, **sections)

    def test_sections_carry_gradients(self, recon):
        c10 = torch.tensor(float(TRUE_C10), dtype=torch.float64, requires_grad=True)
        recon.reconstruct(
            deconvolution_kernel="obf",
            override_aberration_coefs={"C10": c10},
            differentiable=True,
            **self.SECTIONS,
        )
        recon.rms_gradient_loss().backward()
        assert c10.grad is not None and torch.isfinite(c10.grad)

    def test_survives_a_round_trip(self, dataset4d, tmp_path):
        recon = _build(dataset4d)
        recon.reconstruct(deconvolution_kernel="obf", **self.SECTIONS)
        path = str(tmp_path / "sections.zip")
        recon.save(path, mode="o")
        restored = load(path)

        assert restored.slice_mode == "sectioned"
        assert np.allclose(restored.section_residuals, recon.section_residuals)
        assert np.array_equal(restored.obj, recon.obj)


class TestMultislice:
    """Several thin slices solved jointly at each spatial frequency."""

    @pytest.mark.parametrize("scene", ["square_scene", "honeycomb_scene"])
    def test_twisted_layers_separate(self, scene, request):
        """Layers far closer than ``lambda / alpha**2`` (51 A here) come apart, which a C10
        sweep cannot do at any defocus."""
        recon, layers, spacing = request.getfixturevalue(scene)
        recon.reconstruct(
            deconvolution_kernel="obf",
            num_slices=2,
            slice_mode="joint",
            slice_thicknesses=spacing,
            verbose=False,
        )
        purity, corr = layer_purity(to_numpy(recon.corrected_bf), layers)
        assert purity > 0.95
        assert corr > 0.9

        c10 = ML_ABERRATIONS["C10"]
        sweep = [
            to_numpy(
                recon.reconstruct(
                    deconvolution_kernel="obf",
                    override_aberration_coefs={"C10": c10 + z},
                    verbose=False,
                ).corrected_bf
            )
            for z in (-spacing / 2, spacing / 2)
        ]
        assert layer_purity(sweep, layers)[0] < 0.6

    def test_each_slice_holds_one_rotation(self, square_scene):
        """Each lattice's (11) reflection sits on an integer pixel, (5, 5) and (1, 7)."""
        recon, _, spacing = square_scene
        recon.reconstruct(
            deconvolution_kernel="obf",
            num_slices=2,
            slice_mode="joint",
            slice_thicknesses=spacing,
            verbose=False,
        )
        spectra = np.abs(np.fft.fft2(to_numpy(recon.corrected_bf)))

        assert spectra[0, 5, 5] > 10 * spectra[0, 1, 7]
        assert spectra[1, 1, 7] > 10 * spectra[1, 5, 5]

    def test_amplitude_columns_help_a_strong_layer(self, square_scene):
        """Strong layers carry a second-order pseudo-amplitude that phase columns mis-fit."""
        recon, layers, spacing = square_scene
        corr = {}
        for obj_type in ("pure_phase", "complex"):
            recon.reconstruct(
                deconvolution_kernel="obf",
                obj_type=obj_type,
                num_slices=2,
                slice_mode="joint",
                slice_thicknesses=spacing,
                verbose=False,
            )
            corr[obj_type] = layer_purity(to_numpy(recon.corrected_bf), layers)[1]

        assert corr["complex"] > corr["pure_phase"] + 0.01

    def test_a_reference_plane_one_spacing_off_swaps_the_layers(self, square_scene):
        recon, layers, spacing = square_scene
        recon.reconstruct(
            deconvolution_kernel="obf",
            num_slices=2,
            slice_mode="joint",
            slice_thicknesses=spacing,
            override_aberration_coefs={"C10": ML_ABERRATIONS["C10"] + spacing},
            verbose=False,
        )

        assert layer_purity(to_numpy(recon.corrected_bf), layers)[0] < 0.5

    def test_shapes_and_depths(self, square_scene):
        recon, _, spacing = square_scene
        recon.reconstruct(
            deconvolution_kernel="obf",
            obj_type="complex",
            num_slices=3,
            slice_mode="joint",
            slice_thicknesses=[6.0, 12.0],
            verbose=False,
        )

        assert recon.corrected_bf.shape == recon.corrected_amplitude.shape == (3, 64, 64)
        assert recon.obj.shape == (3, 64, 64)
        # gaps of 6 and 12 A put the slices at 0, 6 and 18 A, centred on their mean
        assert np.allclose(recon.slice_depths, [-8.0, -2.0, 10.0])
        assert recon.num_slices == 3

    def test_one_slice_is_the_single_plane(self, recon):
        recon.reconstruct(deconvolution_kernel="obf", verbose=False)
        single = recon.obj.copy()
        recon.reconstruct(deconvolution_kernel="obf", num_slices=1, verbose=False)

        assert np.array_equal(recon.obj, single)

    @pytest.mark.parametrize("kernel", ["ssb", "prlx", "icom"])
    def test_kernels_without_a_joint_solve_raise(self, recon, kernel):
        with pytest.raises(ValueError, match="only 'obf'"):
            recon.reconstruct(
                deconvolution_kernel=kernel,
                num_slices=2,
                slice_mode="joint",
                slice_thicknesses=5.0,
                verbose=False,
            )

    @pytest.mark.parametrize(
        ("thicknesses", "match"),
        [(None, "cannot be None"), ([1.0, 2.0, 3.0], "num_slices - 1"), (-1.0, "positive")],
    )
    def test_slice_thicknesses_are_validated(self, recon, thicknesses, match):
        with pytest.raises(ValueError, match=match):
            recon.reconstruct(
                deconvolution_kernel="obf",
                num_slices=2,
                slice_mode="joint",
                slice_thicknesses=thicknesses,
                verbose=False,
            )

    def test_an_empirical_probe_reproduces_the_analytic_one(self, dataset4d):
        from quantem.diffractive_imaging.complex_probe import FourierProbe

        analytic = _build(dataset4d)
        psi = analytic_probe_array(analytic, {"C10": TRUE_C10})
        probe = FourierProbe.from_array(
            psi, analytic.reciprocal_sampling, analytic.wavelength, normalize=False
        )
        empirical = _build(dataset4d, fourier_probe=probe)

        kwargs = dict(
            deconvolution_kernel="obf",
            obj_type="complex",
            num_slices=2,
            slice_mode="joint",
            slice_thicknesses=20.0,
            verbose=False,
        )
        expected = analytic.reconstruct(**kwargs).obj.copy()
        # two descriptions of one probe differ by float32 rounding, which the near-parallel
        # slice columns at low q amplify up to the eigenvalue floor
        error = np.abs(empirical.reconstruct(**kwargs).obj - expected).max()
        assert error < 1e-4 * np.abs(expected).max()

    def test_the_residual_ignores_the_probe_normalization(self, dataset4d):
        """A measured probe normalized to unit power gives the analytic probe's residual."""
        from quantem.diffractive_imaging.complex_probe import FourierProbe

        analytic = _build(dataset4d)
        psi = analytic_probe_array(analytic, {"C10": TRUE_C10})
        psi = psi / psi.abs().square().sum().sqrt()
        probe = FourierProbe.from_array(
            psi, analytic.reciprocal_sampling, analytic.wavelength, normalize=False
        )
        empirical = _build(dataset4d, fourier_probe=probe)

        kwargs = dict(
            deconvolution_kernel="obf",
            num_slices=2,
            slice_mode="joint",
            slice_thicknesses=20.0,
            verbose=False,
        )
        expected = analytic.reconstruct(**kwargs).residual_loss()
        assert empirical.reconstruct(**kwargs).residual_loss() == pytest.approx(expected, abs=1e-4)

    def test_visualize_shows_one_row_per_slice(self, square_scene):
        import matplotlib

        matplotlib.use("Agg")
        recon, _, spacing = square_scene
        recon.reconstruct(
            deconvolution_kernel="obf",
            num_slices=2,
            slice_mode="joint",
            slice_thicknesses=spacing,
            verbose=False,
        )
        fig, axs = recon.visualize(return_fig=True)

        assert np.shape(axs) == (2, 2)

    @pytest.mark.parametrize("kernel", ["obf"])
    def test_a_large_ridge_is_the_sweep(self, square_scene, kernel):
        """With the ridge far above every eigenvalue each slice is its own plane alone."""
        recon, _, spacing = square_scene
        recon.reconstruct(
            deconvolution_kernel=kernel,
            num_slices=2,
            slice_mode="joint",
            slice_thicknesses=spacing,
            ridge=1e8,
            verbose=False,
        )
        slices = to_numpy(recon.corrected_bf)
        for image, z in zip(slices, (-spacing / 2, spacing / 2)):
            override = {"C10": ML_ABERRATIONS["C10"] + z}
            recon.reconstruct(
                deconvolution_kernel=kernel, override_aberration_coefs=override, verbose=False
            )
            plane = to_numpy(recon.corrected_bf)
            # the two differ only in how the soft aperture edge is weighted
            assert correlation(image, plane) > 0.999
            assert np.std(image) == pytest.approx(np.std(plane), rel=0.05)

    def test_the_ridge_trades_separation_for_noise(self, square_scene):
        recon, layers, spacing = square_scene
        purities = []
        for ridge in (1e-5, 3e-2, 1e2):
            recon.reconstruct(
                deconvolution_kernel="obf",
                num_slices=2,
                slice_mode="joint",
                slice_thicknesses=spacing,
                ridge=ridge,
                verbose=False,
            )
            purities.append(layer_purity(to_numpy(recon.corrected_bf), layers)[0])

        assert purities[0] > purities[1] > purities[2]
        assert purities[0] > 0.95
        assert purities[2] < 0.6

    def test_one_slice_finds_the_mean_plane(self, square_scene):
        """Fit C10 with one slice, then add slices about it.

        One slice pins the plane: its residual changes by about a fifth over +-3 A. Two slices
        span nearly the same columns at any offset, so theirs changes by well under a percent
        and is a poor guide to the reference plane.
        """
        recon, _, spacing = square_scene
        offsets = np.arange(-3.0, 3.01, 1.0)
        losses = {1: [], 2: []}
        for offset in offsets:
            override = {"C10": ML_ABERRATIONS["C10"] + offset}
            recon.reconstruct(
                deconvolution_kernel="obf", override_aberration_coefs=override, verbose=False
            )
            losses[1].append(recon.residual_loss())
            recon.reconstruct(
                deconvolution_kernel="obf",
                num_slices=2,
                slice_mode="joint",
                slice_thicknesses=spacing,
                override_aberration_coefs=override,
                verbose=False,
            )
            losses[2].append(recon.residual_loss())

        # the minimum is flat to 1e-4 between 0 and +0.5 A, so on a 1 A grid it sits at 0
        assert abs(offsets[int(np.argmin(losses[1]))]) <= 0.5
        spread = {n: (max(v) - min(v)) / min(v) for n, v in losses.items()}
        assert spread[1] > 10 * spread[2]


# ---------------------------------------------------------------------------
# empirical probes: the overlap convention and mixed states
# ---------------------------------------------------------------------------


def _empirical(array):
    from quantem.diffractive_imaging.complex_probe import FourierProbe

    return FourierProbe.from_array(
        torch.as_tensor(np.asarray(array), dtype=torch.complex64),
        (ML_RECIPROCAL_SAMPLING, ML_RECIPROCAL_SAMPLING),
        electron_wavelength_angstrom(PROBE_ENERGY),
    )


def _full_detector(dataset, aberrations=ML_ABERRATIONS):
    """A reconstruction on the whole detector window, so a probe array matches its grid."""
    return DirectPtychography.from_dataset4d(dataset, crop_bf_mask=False, **ml_kwargs(aberrations))


#: two equal modes 60 A apart in focus, a crude focal spread
FOCAL_SPREAD_MODES = np.stack(
    [
        np.sqrt(0.5) * library_probe_fourier(dict(ML_ABERRATIONS, C10=20.0)),
        np.sqrt(0.5) * library_probe_fourier(dict(ML_ABERRATIONS, C10=80.0)),
    ]
)


@pytest.fixture(scope="module")
def weak_scene():
    return thin_layer_4dstem([1j * 0.02 * white_noise_object_2D(n=64, phi0=1.0)], [0.0])


@pytest.fixture(scope="module")
def noisy_square_scenes():
    """``({dose: reconstruction}, layers, spacing)``: the square twisted layers with shot noise,
    at 300 electrons per A^2, where least squares is noise-limited, and at ten times that."""
    _, layers, _ = twisted_square_cell()
    layers, spacing = [0.3 * layer for layer in layers], 6.0
    recons = {}
    for dose in (300.0, 3000.0):
        dataset = thin_layer_4dstem(
            [1j * layer for layer in layers],
            [-spacing / 2, spacing / 2],
            ML_ABERRATIONS,
            dose=dose,
        )
        recons[dose] = DirectPtychography.from_dataset4d(dataset, **ml_kwargs(ML_ABERRATIONS))
    return recons, layers, spacing


class TestArdPrior:
    """``prior="ard"``: a variance per coefficient, learned from the data."""

    def test_it_lifts_noise_limited_slices(self, noisy_square_scenes):
        recons, layers, spacing = noisy_square_scenes
        recon = recons[300.0]
        kwargs = dict(
            deconvolution_kernel="obf",
            num_slices=2,
            slice_mode="joint",
            slice_thicknesses=spacing,
            verbose=False,
        )
        purity_ls, corr_ls = layer_purity(
            to_numpy(recon.reconstruct(**kwargs).corrected_bf), layers
        )
        recon.reconstruct(prior="ard", **kwargs)
        purity, corr = layer_purity(to_numpy(recon.corrected_bf), layers)

        # 0.886 and 0.962 at a purity of 0.999 for both
        assert corr > corr_ls + 0.05
        assert purity > 0.99

    def test_the_noise_variance_scales_with_the_dose(self, noisy_square_scenes):
        recons, _, spacing = noisy_square_scenes
        estimates = {}
        for dose, recon in recons.items():
            recon.reconstruct(
                deconvolution_kernel="obf",
                num_slices=2,
                slice_mode="joint",
                slice_thicknesses=spacing,
                prior="ard",
                verbose=False,
            )
            estimates[dose] = recon.noise_variance

        # Poisson noise on the mean-normalized data goes as one over the dose
        assert estimates[300.0] / estimates[3000.0] == pytest.approx(10.0, rel=0.05)

    def test_a_vanishing_noise_variance_is_least_squares(self, square_scene):
        recon, _, spacing = square_scene
        kwargs = dict(
            deconvolution_kernel="obf",
            num_slices=2,
            slice_mode="joint",
            slice_thicknesses=spacing,
            verbose=False,
        )
        expected = to_numpy(recon.reconstruct(**kwargs).corrected_bf)
        estimated = recon.reconstruct(prior="ard", **kwargs).noise_variance
        recon.reconstruct(prior="ard", noise_variance=1e-6 * estimated, **kwargs)

        error = np.abs(to_numpy(recon.corrected_bf) - expected).max() / np.abs(expected).max()
        assert error < 1e-2

    def test_the_residual_ignores_the_prior(self, noisy_square_scenes):
        recons, _, spacing = noisy_square_scenes
        recon = recons[300.0]
        kwargs = dict(
            deconvolution_kernel="obf",
            num_slices=2,
            slice_mode="joint",
            slice_thicknesses=spacing,
            verbose=False,
        )
        expected = recon.reconstruct(**kwargs).residual_loss()
        assert recon.noise_variance is None
        assert recon.reconstruct(prior="ard", **kwargs).residual_loss() == pytest.approx(expected)

    @pytest.mark.parametrize("kernel", ["obf"])
    @pytest.mark.parametrize("obj_type", ["pure_phase", "complex"])
    def test_one_plane(self, noisy_square_scenes, kernel, obj_type):
        recons, _, _ = noisy_square_scenes
        recon = recons[300.0]
        kwargs = dict(deconvolution_kernel=kernel, obj_type=obj_type, verbose=False)
        expected = to_numpy(recon.reconstruct(**kwargs).corrected_bf)
        image = to_numpy(recon.reconstruct(prior="ard", **kwargs).corrected_bf)

        assert image.shape == expected.shape
        assert np.isfinite(image).all()
        assert correlation(image, expected) > 0.9
        if obj_type == "complex":
            assert np.isfinite(to_numpy(recon.corrected_amplitude)).all()

    @pytest.mark.parametrize("kernel", ["ssb", "prlx", "icom"])
    def test_only_the_normal_equation_kernels_take_it(self, recon, kernel):
        with pytest.raises(ValueError, match="normal equations"):
            recon.reconstruct(deconvolution_kernel=kernel, prior="ard", verbose=False)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(prior="lasso"), "prior"),
            (dict(noise_variance=1.0), "only used"),
            (dict(prior="ard", noise_variance=-1.0), "positive"),
        ],
    )
    def test_rejects_bad_arguments(self, recon, kwargs, match):
        with pytest.raises(ValueError, match=match):
            recon.reconstruct(deconvolution_kernel="obf", verbose=False, **kwargs)


class TestOverlapConvention:
    """``t_minus = psi(k - q) psi*(k)``, which matters only for a non-centrosymmetric probe."""

    def test_a_shifted_probe_only_shifts_the_object(self, weak_scene):
        """Shifting the probe relabels the scan, so the residual cannot change.

        With ``psi(q - k)`` the two halves of the overlap function pick up different phases
        and a two-pixel shift ruins the fit.
        """
        recon = _full_detector(weak_scene)
        psi = library_probe_fourier(ML_ABERRATIONS)
        recon.fourier_probe = _empirical(psi)
        recon.reconstruct(deconvolution_kernel="obf", verbose=False)
        expected, residual = to_numpy(recon.corrected_bf), recon.residual_loss()

        k = np.fft.fftfreq(ML_N_DET, ML_SAMPLING)
        kx, ky = np.meshgrid(k, k, indexing="ij")
        ramp = np.exp(2j * np.pi * (kx * 2 - ky * 3) * ML_SAMPLING)
        recon.fourier_probe = _empirical(psi * ramp)
        recon.reconstruct(deconvolution_kernel="obf", verbose=False)

        assert recon.residual_loss() == pytest.approx(residual, abs=1e-6)
        shifted = np.roll(expected, (-2, 3), axis=(0, 1))
        assert correlation(to_numpy(recon.corrected_bf), shifted) > 0.9999

    @pytest.mark.parametrize("aberration", ["C21", "C23"])
    def test_odd_aberrations_are_fitted_with_their_sign(self, aberration):
        angle = aberration.replace("C", "phi")
        truth = {"C10": 50.0, aberration: 300.0, angle: 0.4}
        dataset = thin_layer_4dstem(
            [1j * 0.02 * white_noise_object_2D(n=64, phi0=1.0)],
            [0.0],
            probe_modes=library_probe_fourier(truth)[None],
        )
        recon = DirectPtychography.from_dataset4d(dataset, **ml_kwargs(truth))
        losses = {}
        for name, override in (
            ("true", truth),
            ("inverted", dict(truth, **{angle: 0.4 + np.pi})),
            ("absent", {"C10": 50.0, aberration: 0.0}),
        ):
            recon.reconstruct(
                deconvolution_kernel="obf", override_aberration_coefs=override, verbose=False
            )
            losses[name] = recon.residual_loss()

        # the inverted sign is worse than leaving the aberration out altogether
        assert losses["true"] < 1e-5
        assert losses["absent"] > 100 * losses["true"]
        assert losses["inverted"] > 2 * losses["absent"]


class TestRefinedProbe:
    """A probe refined by zero-padding is the same probe, sampled more finely off the grid."""

    def test_a_refined_probe_matches_the_native_one(self, weak_scene):
        recon = _full_detector(weak_scene)
        probe = _empirical(library_probe_fourier(ML_ABERRATIONS))
        recon.fourier_probe = probe
        recon.reconstruct(deconvolution_kernel="obf", verbose=False)
        expected = recon.obj.copy()
        step = ML_RECIPROCAL_SAMPLING / 2
        recon.fourier_probe = probe.resampled_to((step, step))
        recon.reconstruct(deconvolution_kernel="obf", verbose=False)

        assert np.allclose(recon.obj, expected, atol=1e-5 * np.abs(expected).max())

    @pytest.mark.parametrize("num_modes", [1, 2])
    def test_a_double_precision_probe_is_stored_in_single_precision(self, weak_scene, num_modes):
        """The reconstructions accumulate in complex64, and MPS has no complex128."""
        from quantem.diffractive_imaging.complex_probe import FourierProbe

        array = library_probe_fourier(ML_ABERRATIONS)
        if num_modes > 1:
            array = np.stack([array] * num_modes) / np.sqrt(num_modes)
        recon = _full_detector(weak_scene)
        recon.fourier_probe = _empirical(array)
        recon.reconstruct(deconvolution_kernel="obf", verbose=False)
        expected = recon.obj.copy()

        probe = FourierProbe.from_array(
            np.asarray(array, dtype=np.complex128),
            (ML_RECIPROCAL_SAMPLING, ML_RECIPROCAL_SAMPLING),
            electron_wavelength_angstrom(PROBE_ENERGY),
        )
        assert probe.array.dtype == torch.complex64
        assert probe.modes is None or probe.modes.dtype == torch.complex64
        recon.fourier_probe = probe
        recon.reconstruct(deconvolution_kernel="obf", verbose=False)

        assert np.allclose(recon.obj, expected, atol=1e-5 * np.abs(expected).max())

    def test_refinement_makes_rotated_sampling_accurate(self, weak_scene):
        """A rotation puts ``k - q`` between grid points, where bilinear sampling is only as
        good as the grid is fine."""
        recon = _full_detector(weak_scene)
        kwargs = dict(deconvolution_kernel="obf", override_rotation_angle=7.0, verbose=False)
        recon.reconstruct(**kwargs)
        analytic = to_numpy(recon.corrected_bf)

        from quantem.diffractive_imaging.complex_probe import FourierProbe

        # unnormalized, so the object's scale matches the analytic probe's
        probe = FourierProbe.from_array(
            torch.as_tensor(library_probe_fourier(ML_ABERRATIONS), dtype=torch.complex64),
            (ML_RECIPROCAL_SAMPLING, ML_RECIPROCAL_SAMPLING),
            electron_wavelength_angstrom(PROBE_ENERGY),
            normalize=False,
            interpolation="bilinear",
        )
        errors = {}
        for factor in (1, 4):
            step = ML_RECIPROCAL_SAMPLING / factor
            recon.fourier_probe = probe.resampled_to((step, step))
            recon.reconstruct(**kwargs)
            errors[factor] = np.abs(to_numpy(recon.corrected_bf) - analytic).max()

        scale = np.abs(analytic).max()
        assert errors[4] < 0.05 * scale
        assert errors[4] < errors[1] / 3

    def test_upsampling_stops_at_the_band(self):
        """Past the band a refined probe's overlap is round-off; upsampling must not divide by it.

        Without :data:`OVERLAP_FLOOR` this correlation is 0.63.
        """
        from quantem.diffractive_imaging.complex_probe import FourierProbe

        noisy = thin_layer_4dstem(
            [1j * 0.02 * white_noise_object_2D(n=64, phi0=1.0)], [0.0], dose=1e4
        )
        recon = _full_detector(noisy)
        kwargs = dict(deconvolution_kernel="obf", upsampling_factor=2, verbose=False)
        recon.reconstruct(**kwargs)
        analytic = to_numpy(recon.corrected_bf)
        probe = FourierProbe.from_array(
            torch.as_tensor(library_probe_fourier(ML_ABERRATIONS), dtype=torch.complex64),
            (ML_RECIPROCAL_SAMPLING, ML_RECIPROCAL_SAMPLING),
            electron_wavelength_angstrom(PROBE_ENERGY),
            normalize=False,
        )
        step = ML_RECIPROCAL_SAMPLING / 2
        recon.fourier_probe = probe.resampled_to((step, step))
        recon.reconstruct(**kwargs)

        assert correlation(to_numpy(recon.corrected_bf), analytic) > 0.98

    def test_a_probe_on_another_grid_is_refused(self, weak_scene):
        recon = _full_detector(weak_scene)
        wrong = _empirical(np.ones((ML_N_DET + 3, ML_N_DET + 3), dtype=complex))
        with pytest.raises(ValueError, match="refined by a whole number"):
            recon.fourier_probe = wrong


class TestMixedStateProbe:
    """Incoherent probe modes: the overlap function is summed over them."""

    def test_from_array_takes_modes(self):
        modes = FOCAL_SPREAD_MODES
        probe = _empirical(modes)

        assert probe.num_modes == 2
        assert not probe.is_analytic
        assert float(probe.modes.abs().square().sum()) == pytest.approx(1.0, rel=1e-5)
        intensity = np.abs(modes) ** 2
        expected = np.sqrt(intensity.sum(0) / intensity.sum())
        assert np.allclose(to_numpy(probe.array.abs()), expected, atol=1e-6)
        with pytest.raises(ValueError, match="no single psi"):
            probe.at(torch.zeros(1), torch.zeros(1))

    def test_resampling_refines_every_mode(self):
        probe = _empirical(FOCAL_SPREAD_MODES)
        step = ML_RECIPROCAL_SAMPLING / 2
        refined = probe.resampled_to((step, step))

        assert refined.num_modes == 2
        for mode, fine in zip(probe.coherent_modes(), refined.coherent_modes()):
            expected = mode.resampled_to((step, step)).array
            assert torch.allclose(fine.array, expected)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"obj_type": "complex"},
            {"num_slices": 2, "slice_mode": "joint", "slice_thicknesses": 10.0},
        ],
    )
    def test_one_mode_is_the_coherent_probe(self, weak_scene, kwargs):
        recon = _full_detector(weak_scene)
        psi = library_probe_fourier(ML_ABERRATIONS)
        results = []
        for array in (psi, psi[None]):
            recon.fourier_probe = _empirical(array)
            recon.reconstruct(deconvolution_kernel="obf", verbose=False, **kwargs)
            results.append((recon.obj.copy(), recon.residual_loss()))

        assert np.allclose(results[0][0], results[1][0])
        assert results[0][1] == pytest.approx(results[1][1])

    def test_the_mode_sum_explains_a_focal_spread(self):
        """No single focus describes two incoherent ones; their mode sum does exactly."""
        dataset = thin_layer_4dstem(
            [1j * 0.02 * white_noise_object_2D(n=64, phi0=1.0)],
            [0.0],
            probe_modes=FOCAL_SPREAD_MODES,
        )
        recon = _full_detector(dataset)
        coherent = []
        for c10 in (20.0, 50.0, 80.0):
            recon.reconstruct(
                deconvolution_kernel="obf", override_aberration_coefs={"C10": c10}, verbose=False
            )
            coherent.append(recon.residual_loss())
        recon.fourier_probe = _empirical(FOCAL_SPREAD_MODES)
        recon.reconstruct(deconvolution_kernel="obf", verbose=False)

        assert recon.residual_loss() < 1e-5
        assert min(coherent) > 0.05

    def test_layers_separate_under_a_mixed_state(self):
        """A coherent model of a mixed-state probe cannot place the layers; the mode sum can."""
        _, layers, _ = twisted_square_cell()
        layers, spacing = [0.3 * layer for layer in layers], 6.0
        dataset = thin_layer_4dstem(
            [1j * layer for layer in layers],
            [-spacing / 2, spacing / 2],
            probe_modes=FOCAL_SPREAD_MODES,
        )
        recon = _full_detector(dataset)
        kwargs = dict(
            deconvolution_kernel="obf",
            num_slices=2,
            slice_mode="joint",
            slice_thicknesses=spacing,
            verbose=False,
        )
        recon.reconstruct(**kwargs)
        coherent, _ = layer_purity(to_numpy(recon.corrected_bf), layers)
        recon.fourier_probe = _empirical(FOCAL_SPREAD_MODES)
        recon.reconstruct(**kwargs)
        purity, corr = layer_purity(to_numpy(recon.corrected_bf), layers)

        assert purity > 0.99
        assert corr > 0.9
        assert coherent < 0.6


# ---------------------------------------------------------------------------
# differentiability and gradient fitting
# ---------------------------------------------------------------------------


def _residual_at(
    recon, kernel, c10, rotation=0.0, thickness=None, obj_type="pure_phase", grad=False
):
    kwargs = dict(
        deconvolution_kernel=kernel,
        obj_type=obj_type,
        override_aberration_coefs={"C10": c10},
        override_rotation_angle=rotation,
        differentiable=grad,
        verbose=False,
    )
    if thickness is not None:
        kwargs.update(num_slices=2, slice_mode="joint", slice_thicknesses=thickness)
    recon.reconstruct(**kwargs)
    return recon.residual_loss()


class TestGradients:
    """Autograd through the reconstruction, checked against finite differences."""

    C10 = ML_ABERRATIONS["C10"] + 7.0

    @pytest.mark.parametrize(
        ("kernel", "obj_type", "parameter"),
        [
            ("obf", "pure_phase", "C10"),
            ("prlx", "pure_phase", "C10"),
            ("obf", "pure_phase", "rotation"),
            ("icom", "pure_phase", "rotation"),
            ("obf", "complex", "C10"),
            ("obf", "complex", "thickness"),
        ],
    )
    def test_residual_gradient_matches_finite_differences(
        self, square_scene, kernel, obj_type, parameter
    ):
        recon, _, spacing = square_scene
        thickness = spacing + 1.0 if parameter == "thickness" else None
        values = {"C10": self.C10, "rotation": 1.0, "thickness": thickness}
        leaf = torch.tensor(values[parameter], requires_grad=True)
        args = dict(values, **{parameter: leaf})

        _residual_at(
            recon, kernel, args["C10"], args["rotation"], args["thickness"], obj_type, grad=True
        ).backward()

        step = {"C10": 0.5, "rotation": 0.05, "thickness": 0.25}[parameter]
        shifted = []
        for sign in (1, -1):
            point = dict(values, **{parameter: values[parameter] + sign * step})
            shifted.append(
                _residual_at(
                    recon, kernel, point["C10"], point["rotation"], point["thickness"], obj_type
                )
            )
        finite_difference = (shifted[0] - shifted[1]) / (2 * step)

        assert leaf.grad.item() == pytest.approx(finite_difference, rel=0.05)

    def test_reconstruct_keeps_no_graph_by_default(self, square_scene):
        recon, _, _ = square_scene
        c10 = torch.tensor(self.C10, requires_grad=True)
        recon.reconstruct(
            deconvolution_kernel="obf", override_aberration_coefs={"C10": c10}, verbose=False
        )

        assert not recon.corrected_bf.requires_grad
        assert isinstance(recon.residual_loss(), float)

    def test_differentiable_reconstruction_backpropagates_through_the_image(self, square_scene):
        recon, _, _ = square_scene
        c10 = torch.tensor(self.C10, requires_grad=True)
        recon.reconstruct(
            deconvolution_kernel="obf",
            obj_type="complex",
            override_aberration_coefs={"C10": c10},
            differentiable=True,
            verbose=False,
        )
        recon.rms_gradient_loss().backward()

        assert recon.corrected_amplitude.requires_grad
        assert torch.isfinite(c10.grad)

    def test_gradient_fit_recovers_the_defocus(self):
        """OBF only: SSB shares its residual, so its fit follows the same path."""
        _, layers, _ = twisted_square_cell()
        dataset = thin_layer_4dstem([0.3j * layer for layer in layers], [-3.0, 3.0])
        kwargs = ml_kwargs()
        kwargs["aberration_coefs"] = dict(ML_ABERRATIONS, C10=ML_ABERRATIONS["C10"] + 30.0)
        recon = DirectPtychography.from_dataset4d(dataset, **kwargs)

        # converged after 4 iterations: the same +0.42 A as after the default 20
        recon.fit_hyperparameters_gradient(("C10",), deconvolution_kernel="obf", num_iterations=5)
        state = recon.hyperparameter_state

        assert state.optimized_aberrations["C10"] == pytest.approx(ML_ABERRATIONS["C10"], abs=0.5)
        assert state.optimized_aberrations["C12"] == pytest.approx(ML_ABERRATIONS["C12"])
        assert "C10" in state.optimized_keys

    def test_gradient_fit_recovers_the_rotation(self, square_scene):
        recon, _, _ = square_scene
        recon.hyperparameter_state.optimized_rotation_angle = 3.0
        recon.fit_hyperparameters_gradient(("rotation_angle",), deconvolution_kernel="obf")

        assert recon.hyperparameter_state.optimized_rotation_angle == pytest.approx(0.0, abs=0.05)
        recon.hyperparameter_state.clear_optimized()

    def test_gradient_fit_rejects_an_empty_request(self, square_scene):
        recon, _, _ = square_scene
        with pytest.raises(ValueError, match="Nothing to fit"):
            recon.fit_hyperparameters_gradient((), deconvolution_kernel="obf")


class TestSliceThicknessSearch:
    def test_grid_search_stores_the_thickness(self, square_scene):
        recon, _, _ = square_scene
        recon.grid_search_hyperparameters(
            slice_thicknesses=OptimizationParameter(low=4.0, high=8.0, n_points=3),
            deconvolution_kernel="obf",
            num_slices=2,
            slice_mode="joint",
            verbose=False,
        )
        stored = recon.hyperparameter_state.optimized_slice_thicknesses

        assert stored is not None and len(stored) == 1
        assert stored[0] in (4.0, 6.0, 8.0)
        # a later reconstruction with several slices picks it up
        recon.reconstruct(
            deconvolution_kernel="obf", num_slices=2, slice_mode="joint", verbose=False
        )
        assert np.ptp(recon.slice_depths) == pytest.approx(stored[0])
        recon.hyperparameter_state.clear_optimized()

    def test_optuna_searches_the_thickness(self, square_scene):
        recon, _, _ = square_scene
        recon.optimize_hyperparameters(
            slice_thicknesses=OptimizationParameter(low=4.0, high=8.0),
            n_trials=3,
            deconvolution_kernel="obf",
            num_slices=2,
            slice_mode="joint",
            verbose=False,
        )

        assert "slice_thicknesses" in recon.hyperparameter_state.study.best_params
        recon.hyperparameter_state.clear_optimized()


# ---------------------------------------------------------------------------
# partial coherence, the Wiener kernel and Wigner distribution deconvolution
# ---------------------------------------------------------------------------


def _with_rms(image, rms):
    return rms * image / image.std()


def _slope(image, truth):
    """The least-squares scale of ``image`` on ``truth`` over the band: one when calibrated."""
    image, truth = band_limit_ml(np.asarray(image)), band_limit_ml(truth)
    return float((image * truth).sum() / (truth * truth).sum())


def _with_source(dataset, source_size, dose, seed=0):
    """``dataset`` averaged over a Gaussian spread of probe positions, then with shot noise."""
    from quantem.core.datastructures import Dataset4d

    array = np.asarray(dataset.array)
    q = np.fft.fftfreq(array.shape[0], dataset.sampling[0])
    envelope = np.exp(-2 * np.pi**2 * source_size**2 * (q[:, None] ** 2 + q[None, :] ** 2))
    blurred = np.fft.ifft2(
        np.fft.fft2(array, axes=(0, 1)) * envelope[..., None, None], axes=(0, 1)
    )
    counts = np.random.default_rng(seed).poisson(blurred.real.clip(0) * dose * ML_SAMPLING**2)
    return Dataset4d.from_array(
        counts.astype(np.float32), sampling=dataset.sampling, units=dataset.units
    )


def _focal_modes(spread, count=9):
    """Gauss-Hermite modes of a Gaussian focal spread about ``ML_ABERRATIONS``."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(count)
    weights = weights / weights.sum()
    c10 = ML_ABERRATIONS["C10"]
    return np.stack(
        [
            np.sqrt(w) * library_probe_fourier(dict(ML_ABERRATIONS, C10=c10 + spread * x))
            for x, w in zip(nodes, weights)
        ]
    )


@pytest.fixture(scope="module")
def calibration_scene():
    """``(reconstruction, phase, log_amplitude)``: independent white noise in both channels,
    0.05 rad and 0.015 rms, at 1e5 electrons per A^2."""
    phase = _with_rms(white_noise_object_2D(n=64, phi0=1.0), 0.05)
    amplitude = _with_rms(white_noise_object_2D(n=64, phi0=1.0, seed=1), 0.015)
    dataset = thin_layer_4dstem([amplitude + 1j * phase], [0.0], ML_ABERRATIONS, dose=1e5)
    return DirectPtychography.from_dataset4d(dataset, **ml_kwargs()), phase, amplitude


@pytest.fixture(scope="module")
def source_scene():
    """``(reconstruction, layer, source_size)``: a square lattice imaged with a 0.5 A source."""
    _, layers, _ = twisted_square_cell()
    layer, source_size = 0.3 * layers[0], 0.5
    clean = thin_layer_4dstem([1j * layer], [0.0], ML_ABERRATIONS)
    dataset = _with_source(clean, source_size, dose=1e4)
    recon = DirectPtychography.from_dataset4d(dataset, intensity_threshold=0.05, **ml_kwargs())
    return recon, layer, source_size


class TestPartialCoherence:
    """``focal_spread`` and ``source_size``: a closed-form envelope on the overlap function."""

    def test_the_focal_envelope_is_the_average_over_defocus(self):
        from quantem.diffractive_imaging.complex_probe import FourierProbe, overlap_terms

        wavelength = electron_wavelength_angstrom(PROBE_ENERGY)
        angular = ML_RECIPROCAL_SAMPLING * wavelength * 1e3
        spread = 20.0
        k = torch.fft.fftfreq(ML_N_DET, ML_SAMPLING, dtype=torch.float64)
        kx, ky = (g.reshape(-1, 1, 1) for g in torch.meshgrid(k, k, indexing="ij"))
        q = torch.fft.fftfreq(16, 4 * ML_SAMPLING, dtype=torch.float64)
        qx, qy = torch.meshgrid(q, q, indexing="ij")

        def terms(probe):
            t_minus, t_plus, _, _ = overlap_terms(probe, probe.at(kx, ky), kx, ky, qx, qy)
            return t_minus, t_plus

        def probe_at(c10):
            aberrations = dict(ML_ABERRATIONS, C10=c10)
            return FourierProbe.from_aberrations(
                wavelength, ML_SEMIANGLE, (angular,) * 2, aberrations
            )

        nodes, weights = np.polynomial.hermite_e.hermegauss(61)
        weights = weights / weights.sum()
        average = [0, 0]
        for x, w in zip(nodes, weights):
            for i, t in enumerate(terms(probe_at(ML_ABERRATIONS["C10"] + spread * x))):
                average[i] = average[i] + w * t
        enveloped = terms(probe_at(ML_ABERRATIONS["C10"]).with_partial_coherence(spread, 0.0))

        for expected, actual in zip(average, enveloped):
            scale = float(expected.abs().max())
            assert float((actual - expected).abs().max()) < 1e-6 * scale

    @pytest.mark.parametrize("obj_type", ["pure_phase", "complex"])
    def test_a_finite_source_leaves_obf_and_the_residual_unchanged(self, weak_scene, obj_type):
        recon = DirectPtychography.from_dataset4d(weak_scene, **ml_kwargs())
        kwargs = dict(deconvolution_kernel="obf", obj_type=obj_type, verbose=False)
        expected = to_numpy(recon.reconstruct(**kwargs).corrected_bf)
        residual = recon.residual_loss()
        # small enough that no frequency inside the band falls below the round-off floor
        recon.source_size = 0.2
        image = to_numpy(recon.reconstruct(**kwargs).corrected_bf)

        # the source multiplies every column at q by the same factor, which OBF's gain and the
        # least-squares projection both divide out
        assert np.abs(image - expected).max() < 1e-4 * np.abs(expected).max()
        # the data are noiseless and the residual is 1e-7 to 1e-5 of the energy, so only its
        # absolute round-off is meaningful. That round-off differs between platforms (up to 2e-7
        # on Linux CI), while mishandling the source would move the residual by far more, since
        # its envelope falls well below one inside the band
        assert recon.residual_loss() == pytest.approx(residual, abs=1e-6)

    def test_ssb_keeps_its_coherent_weights(self, weak_scene):
        recon = DirectPtychography.from_dataset4d(weak_scene, **ml_kwargs())
        expected = to_numpy(recon.reconstruct(deconvolution_kernel="ssb", verbose=False).obj)
        residual = recon.residual_loss()
        recon.focal_spread = 40.0
        recon.reconstruct(deconvolution_kernel="ssb", verbose=False)

        assert np.array_equal(recon.obj, expected)
        assert recon.residual_loss() != pytest.approx(residual)

    def test_the_residual_finds_the_focal_spread(self):
        dataset = thin_layer_4dstem(
            [1j * 0.02 * white_noise_object_2D(n=64, phi0=1.0)],
            [0.0],
            probe_modes=_focal_modes(40.0),
        )
        recon = DirectPtychography.from_dataset4d(dataset, **ml_kwargs())
        residuals = {}
        for spread in (0.0, 40.0, 80.0):
            recon.focal_spread = spread
            residuals[spread] = recon.reconstruct(
                deconvolution_kernel="obf", verbose=False
            ).residual_loss()

        # 0.13, 5e-7 and 0.16: the envelope is the nine modes' average to quadrature accuracy
        assert residuals[40.0] < 1e-3 * min(residuals[0.0], residuals[80.0])

    def test_survives_a_round_trip(self, weak_scene, tmp_path):
        recon = DirectPtychography.from_dataset4d(
            weak_scene, focal_spread=30.0, source_size=0.2, **ml_kwargs()
        )
        path = str(tmp_path / "coherence.zip")
        recon.save(path, mode="o")
        restored = load(path)

        assert restored.focal_spread == 30.0
        assert restored.source_size == 0.2

    @pytest.mark.parametrize("name", ["focal_spread", "source_size"])
    def test_rejects_a_negative_spread(self, recon, name):
        with pytest.raises(ValueError, match="non-negative"):
            setattr(recon, name, -1.0)


class TestWienerKernel:
    """``"wiener"``: the posterior mean under the data's own radial spectrum."""

    @pytest.mark.parametrize("obj_type", ["pure_phase", "complex"])
    @pytest.mark.parametrize("upsampling_factor", [1, 2])
    def test_its_values_are_radians_and_nepers(
        self, calibration_scene, obj_type, upsampling_factor
    ):
        recon, phase, amplitude = calibration_scene
        recon.reconstruct(
            deconvolution_kernel="wiener",
            obj_type=obj_type,
            upsampling_factor=upsampling_factor,
            verbose=False,
        )
        step = upsampling_factor

        # 0.989 and 0.995; a Wiener filter shrinks a little where the signal is near the noise
        assert _slope(to_numpy(recon.corrected_bf)[::step, ::step], phase) == pytest.approx(
            1.0, abs=0.03
        )
        if obj_type == "complex":
            amplitude_image = to_numpy(recon.corrected_amplitude)[::step, ::step]
            assert _slope(amplitude_image, amplitude) == pytest.approx(1.0, abs=0.05)

    def test_it_divides_out_a_finite_source(self, source_scene):
        recon, layer, source_size = source_scene
        scores = {}
        for kernel, size in (("obf", source_size), ("wiener", 0.0), ("wiener", source_size)):
            recon.source_size = size
            recon.reconstruct(deconvolution_kernel=kernel, verbose=False)
            scores[(kernel, size)] = correlation(
                band_limit_ml(to_numpy(recon.corrected_bf)), band_limit_ml(layer)
            )
        recon.source_size = 0.0

        # 0.921 for OBF, 0.898 for the coherent model, 0.986 with the source
        assert scores[("wiener", source_size)] > scores[("obf", source_size)] + 0.04
        assert scores[("wiener", source_size)] > scores[("wiener", 0.0)] + 0.04

    def test_it_solves_several_slices(self, noisy_square_scenes):
        recons, layers, spacing = noisy_square_scenes
        recon = recons[300.0]
        kwargs = dict(num_slices=2, slice_mode="joint", slice_thicknesses=spacing, verbose=False)
        recon.reconstruct(deconvolution_kernel="obf", **kwargs)
        _, corr_ls = layer_purity(to_numpy(recon.corrected_bf), layers)
        recon.reconstruct(deconvolution_kernel="wiener", **kwargs)
        purity, corr = layer_purity(to_numpy(recon.corrected_bf), layers)

        assert recon.corrected_bf.shape[0] == 2
        assert corr > corr_ls + 0.03
        assert purity > 0.95

    def test_the_noise_variance(self, calibration_scene):
        recon, _, _ = calibration_scene
        recon.reconstruct(deconvolution_kernel="wiener", verbose=False)
        estimated = recon.noise_variance
        recon.reconstruct(
            deconvolution_kernel="wiener", noise_variance=2 * estimated, verbose=False
        )

        assert estimated > 0
        assert recon.noise_variance == 2 * estimated

    def test_shares_the_residual_of_obf(self, calibration_scene):
        recon, _, _ = calibration_scene
        expected = recon.reconstruct(deconvolution_kernel="obf", verbose=False).residual_loss()
        recon.reconstruct(deconvolution_kernel="wiener", verbose=False)
        assert recon.residual_loss() == pytest.approx(expected)

    def test_refuses_the_ard_prior(self, recon):
        with pytest.raises(ValueError, match="its own prior"):
            recon.reconstruct(deconvolution_kernel="wiener", prior="ard", verbose=False)


@pytest.fixture(scope="module")
def strong_scene():
    """``(reconstruction, phase)``: a square lattice with 2 rad peaks, noiseless, the mask
    reaching the aperture's whole soft edge."""
    _, layers, _ = twisted_square_cell()
    phase = 2.0 * layers[0] / layers[0].max()
    dataset = thin_layer_4dstem([1j * phase], [0.0], ML_ABERRATIONS)
    recon = DirectPtychography.from_dataset4d(dataset, intensity_threshold=0.005, **ml_kwargs())
    return recon, phase


def _complex_correlation(image, truth):
    keep = _band(truth.shape, hi=2 * ML_K_PROBE + 1e-9)
    a, b = np.fft.fft2(image) * keep, np.fft.fft2(truth) * keep
    a[0, 0] = b[0, 0] = 0
    return float(np.abs(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real))


class TestWignerDistributionDeconvolution:
    """``"wdd"``: the intensity as a convolution over the detector, with no weak-object expansion."""

    def test_it_recovers_a_strong_object(self, strong_scene):
        recon, phase = strong_scene
        recon.reconstruct(
            deconvolution_kernel="wdd", obj_type="complex", ridge=1e-6, verbose=False
        )
        wdd = _complex_correlation(recon.obj, np.exp(1j * phase))
        wdd_phase = correlation(band_limit_ml(to_numpy(recon.corrected_bf)), band_limit_ml(phase))
        wdd_residual = recon.residual_loss()
        recon.reconstruct(deconvolution_kernel="obf", obj_type="complex", verbose=False)
        obf_phase = correlation(band_limit_ml(to_numpy(recon.corrected_bf)), band_limit_ml(phase))

        # 0.9995 against the transmission; phases 0.9998 against 0.956 for OBF
        assert wdd > 0.995
        assert wdd_phase > obf_phase + 0.03
        # WDD reports the residual of the weak-object model OBF solves, not of its own estimate
        assert wdd_residual == pytest.approx(recon.residual_loss())

    def test_its_weak_object_transfer_tends_to_one_with_the_ridge(self):
        phase = _with_rms(white_noise_object_2D(n=64, phi0=1.0), 0.02)
        dataset = thin_layer_4dstem([1j * phase], [0.0], ML_ABERRATIONS)
        recon = DirectPtychography.from_dataset4d(
            dataset, intensity_threshold=0.005, **ml_kwargs()
        )
        slopes = {}
        for ridge in (1e-10, 1e-3):
            recon.reconstruct(deconvolution_kernel="wdd", ridge=ridge, verbose=False)
            slopes[ridge] = _slope(to_numpy(recon.corrected_bf), phase)

        # 0.994, and well below one at the default: the regularized filter's weight at zero offset
        assert slopes[1e-10] == pytest.approx(1.0, abs=0.02)
        assert slopes[1e-3] < 0.8

    def test_it_warns_when_the_mask_clips_the_aperture(self, weak_scene, strong_scene):
        clipped = DirectPtychography.from_dataset4d(
            weak_scene, intensity_threshold=0.5, **ml_kwargs()
        )
        with pytest.warns(UserWarning, match="probe intensity outside"):
            clipped.reconstruct(deconvolution_kernel="wdd", verbose=False)
        recon, _ = strong_scene
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            recon.reconstruct(deconvolution_kernel="wdd", verbose=False)

    def test_one_mode_is_the_coherent_probe(self, weak_scene):
        recon = DirectPtychography.from_dataset4d(
            weak_scene, crop_bf_mask=False, intensity_threshold=0.005, **ml_kwargs()
        )
        psi = library_probe_fourier(ML_ABERRATIONS)
        images = []
        for array in (psi, psi[None]):
            recon.fourier_probe = _empirical(array)
            recon.reconstruct(deconvolution_kernel="wdd", obj_type="complex", verbose=False)
            images.append(recon.obj.copy())

        assert np.allclose(images[0], images[1], atol=1e-5)

    @pytest.mark.slow
    def test_upsampling(self, strong_scene):
        recon, _ = strong_scene
        coarse = to_numpy(
            recon.reconstruct(deconvolution_kernel="wdd", verbose=False).corrected_bf
        )
        recon.reconstruct(deconvolution_kernel="wdd", upsampling_factor=2, verbose=False)
        fine = to_numpy(recon.corrected_bf)

        assert fine.shape == (128, 128)
        assert correlation(fine[::2, ::2], coarse) > 0.99

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (
                dict(num_slices=2, slice_mode="joint", slice_thicknesses=5.0),
                "single multiplicative object",
            ),
            (dict(prior="ard"), "normal equations"),
        ],
    )
    def test_rejects(self, recon, kwargs, match):
        with pytest.raises(ValueError, match=match):
            recon.reconstruct(deconvolution_kernel="wdd", verbose=False, **kwargs)


@pytest.fixture(scope="module")
def half_scan():
    """``(reconstruction, phase, log_amplitude)``: every other position of a scan whose Nyquist
    is the band edge, so half the band lies above the scan's Nyquist."""
    from quantem.core.datastructures import Dataset4d

    phase = _with_rms(white_noise_object_2D(n=64, phi0=1.0), 0.05)
    amplitude = _with_rms(white_noise_object_2D(n=64, phi0=1.0, seed=1), 0.015)
    full = thin_layer_4dstem([amplitude + 1j * phase], [0.0], ML_ABERRATIONS, dose=1e5)
    array = np.asarray(full.array)[::2, ::2].copy()
    dataset = Dataset4d.from_array(
        array, sampling=(2 * ML_SAMPLING,) * 2 + tuple(full.sampling[2:]), units=full.units
    )
    return DirectPtychography.from_dataset4d(dataset, **ml_kwargs()), phase, amplitude


def _extension_correlation(image, truth):
    """Correlation with ``truth`` over the band above the half scan's Nyquist, 1-2 / A."""
    q = np.fft.fftfreq(image.shape[0], ML_SAMPLING)
    radius = np.hypot(q[:, None], q[None, :])
    keep = (radius > ML_K_PROBE) & (radius <= 2 * ML_K_PROBE)
    a, b = np.fft.fft2(image) * keep, np.fft.fft2(truth) * keep
    return float(np.vdot(a, b).real / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real))


class TestUpsampling:
    """``upsampling_factor`` recovers the band above the scan's Nyquist, for every kernel."""

    def test_alias_tiles_round_trip(self):
        from quantem.diffractive_imaging.direct_ptycho_utils import alias_tiles, from_alias_tiles

        array = torch.randn(3, 8, 6, 2)
        tiled = alias_tiles(array, 2)

        assert tiled.shape == (3, 4, 3, 8)
        # column tile * K + channel: tile (1, 0) of channel 1 at coarse (0, 0) is fine (4, 0)
        assert tiled[0, 0, 0, 2 * 2 + 1] == array[0, 4, 0, 1]
        assert torch.equal(from_alias_tiles(tiled, 2, 2), array)

    # correlation over the extension band from the half scan at U = 2; a full scan gives
    # 0.83, 0.93, 1.00, 0.92, 0.71 and 0.32 for the phase, and 0.82, 0.89, 0.99, 0.92 and 0.24
    # for the log-amplitude
    @pytest.mark.parametrize(
        "kernel, obj_type, phase_min, amplitude_min",
        [
            ("ssb", "pure_phase", 0.78, None),
            ("obf", "pure_phase", 0.86, None),
            ("wiener", "pure_phase", 0.97, None),
            ("wdd", "pure_phase", 0.85, None),
            ("prlx", "pure_phase", 0.64, None),
            ("icom", "pure_phase", 0.14, None),
            ("ssb", "complex", 0.78, 0.71),
            ("obf", "complex", 0.86, 0.85),
            ("wiener", "complex", 0.97, 0.95),
            ("wdd", "complex", 0.85, 0.62),
            ("icom", "complex", 0.14, 0.04),
        ],
    )
    def test_recovers_the_band_above_the_scan_nyquist(
        self, half_scan, kernel, obj_type, phase_min, amplitude_min
    ):
        recon, phase, amplitude = half_scan
        recon.reconstruct(
            deconvolution_kernel=kernel, obj_type=obj_type, upsampling_factor=2, verbose=False
        )

        assert recon.corrected_bf.shape == (64, 64)
        assert np.isfinite(recon.obj).all()
        assert _extension_correlation(to_numpy(recon.corrected_bf), phase) > phase_min
        if amplitude_min is not None:
            image = to_numpy(recon.corrected_amplitude)
            assert _extension_correlation(image, amplitude) > amplitude_min

    @pytest.mark.parametrize("kernel", ["wiener", "wdd"])
    def test_the_object_estimates_keep_their_scale(self, half_scan, kernel):
        recon, phase, _ = half_scan
        coarse = to_numpy(recon.reconstruct(deconvolution_kernel=kernel, verbose=False).obj)
        fine = to_numpy(
            recon.reconstruct(deconvolution_kernel=kernel, upsampling_factor=2, verbose=False).obj
        )
        below_nyquist = np.fft.ifft2(np.fft.fft2(fine) * _band(fine.shape, hi=ML_K_PROBE)).real

        # same values on the coarse grid's pixels, so no 1 / U**2 to undo
        assert below_nyquist[::2, ::2].std() == pytest.approx(coarse.std(), rel=0.1)
