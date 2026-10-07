"""Tests for the Fourier-space direct-ptychography reconstruction.

Covers all five deconvolution kernels, the aberration/rotation conventions they share, the
four hyperparameter-fitting routines, and save/load. Synthetic data comes from
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
    PROBE_ENERGY,
    Q_PROBE,
    SCAN_SAMPLING,
    SEMIANGLE_CUTOFF,
    N,
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
    make_dataset4d,
    measured_ctf,
    scan_positions_px,
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

    def test_bf_mask_is_about_the_aperture_size(self, recon):
        """The mask is thresholded from the mean pattern, so it should track the BF disk."""
        expected_area = np.pi * (Q_PROBE / recon.reciprocal_sampling[0]) ** 2
        assert recon.num_bf == pytest.approx(expected_area, rel=0.35)

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
            "mf": "matched-filter",
            "prlx": "tilt-corrected-bright-field",
            "icom": "center-of-mass",
        }
        by_short = recon.reconstruct(deconvolution_kernel=kernel, verbose=False).obj.copy()
        by_alias = recon.reconstruct(deconvolution_kernel=aliases[kernel], verbose=False).obj

        assert np.array_equal(by_short, by_alias)

    @pytest.mark.parametrize("upsampling_factor", [1, 2, 3])
    def test_upsampling_preserves_the_field_of_view(self, recon, upsampling_factor):
        recon.reconstruct(
            deconvolution_kernel="ssb", upsampling_factor=upsampling_factor, verbose=False
        )

        assert recon.obj.shape == (N * upsampling_factor, N * upsampling_factor)
        assert recon.obj.shape[0] * recon._obj_sampling[0] == pytest.approx(N * SCAN_SAMPLING)

    def test_corrected_stack_sums_to_corrected_bf(self, recon):
        recon.reconstruct(deconvolution_kernel="ssb", verbose=False)

        assert recon.corrected_stack.shape == (recon.num_bf, N, N)
        assert torch.allclose(recon.corrected_stack.sum(0), recon.corrected_bf)

    def test_unknown_kernel_raises(self, recon):
        with pytest.raises(ValueError, match="Unknown deconvolution kernel"):
            recon.reconstruct(deconvolution_kernel="wiener", verbose=False)


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
            halves.append(recon.corrected_bf.clone() * recon.corrected_stack.shape[0])

        # each half is normalized by its own BF weight; undo that before comparing
        combined = to_numpy(halves[0] + halves[1]) / recon.num_bf
        assert correlation(combined, to_numpy(full)) > 0.99

    def test_halfsets_helper_returns_two_images(self, recon):
        first, second = recon._reconstruct_with_halfsets(deconvolution_kernel="ssb")

        assert first.shape == second.shape == (N, N)
        assert correlation(to_numpy(first), to_numpy(second)) > 0.5

    def test_subset_uses_fewer_bright_field_pixels(self, recon):
        mask = recon._make_checkerboard_bf_masks(recon.gpts, recon.bf_mask)[0]
        recon.reconstruct(bf_mask=mask, deconvolution_kernel="ssb", verbose=False)

        assert recon.corrected_stack.shape[0] == int(mask.sum())
        assert int(mask.sum()) < recon.num_bf


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


class TestVarianceLoss:
    def test_is_positive_after_reconstructing(self, recon):
        recon.reconstruct(deconvolution_kernel="prlx", verbose=False)

        assert float(recon.variance_loss()) > 0

    def test_is_minimized_at_the_true_defocus(self, recon):
        losses = {}
        for scale in (0.5, 0.8, 1.0, 1.2, 1.5):
            recon.reconstruct(
                deconvolution_kernel="prlx",
                override_aberration_coefs={"C10": scale * TRUE_C10},
                parallax_flip_phase=False,
                verbose=False,
            )
            losses[scale] = float(recon.variance_loss())

        assert min(losses, key=losses.get) == pytest.approx(1.0)


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

    def test_least_squares(self, dataset4d):
        recon = _build(dataset4d, aberration_coefs={})
        recon.fit_hyperparameters_least_squares(
            cartesian_basis="defocus", fit_method="global", verbose=False
        )
        fitted = recon.hyperparameter_state.optimized_aberrations["C10"]

        assert fitted == pytest.approx(TRUE_C10, rel=0.15)

    def test_cross_correlation(self, dataset4d):
        recon = _build(dataset4d, aberration_coefs={})
        recon.fit_hyperparameters_cross_correlation(bin_factors=(2, 1), verbose=False)
        state = recon.hyperparameter_state

        assert state.optimized_aberrations["C10"] == pytest.approx(TRUE_C10, rel=0.3)
        assert state.optimized_rotation_angle == pytest.approx(0.0, abs=5.0)

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
        """Per Angstrom, not per pixel, so upsampling does not rescale it."""
        recon = _build(dataset4d)
        recon.reconstruct(deconvolution_kernel="prlx", upsampling_factor=1, verbose=False)
        coarse = recon.rms_gradient_loss()
        recon.reconstruct(deconvolution_kernel="prlx", upsampling_factor=2, verbose=False)
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

        # the RMS gradient loss is negative; the variance loss is positive
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

        assert len(images) == len(DECONVOLUTION_KERNELS)
        assert all(image.shape == (N, N) for image in images)
        assert all(np.isfinite(image).all() for image in images)


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
            losses[scale] = float(recon.variance_loss())

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
