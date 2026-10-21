import numpy as np
import pytest
import torch

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem
from quantem.diffraction.polar import PairDistributionFunction
from quantem.diffraction.polar_transform import (
    find_origin_angular_descent,
    find_origin_angular_grid,
    polar_transform,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def synthetic_diffraction_pattern():
    """Create a synthetic diffraction pattern with concentric rings."""
    ny, nx = 256, 256
    y, x = np.ogrid[:ny, :nx]
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0

    # Create rings with Gaussian profiles at specific radii
    pattern = np.zeros((ny, nx), dtype=np.float32)
    ring_radii = [10, 20, 30, 40]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    for radius in ring_radii:
        pattern += 100 * np.exp(-((r - radius) ** 2) / (2 * 2**2))
    # central beam
    pattern += 1000 * np.exp(-(r**2) / (2 * 3**2))
    # noise
    rng = np.random.default_rng(42)
    pattern += rng.poisson(5, size=(ny, nx))

    return pattern.astype(np.float32)


@pytest.fixture
def synthetic_4dstem_dataset(synthetic_diffraction_pattern):
    """Create a synthetic 4D-STEM dataset with 3x3 scan."""
    scan_y, scan_x = 3, 3
    ny, nx = synthetic_diffraction_pattern.shape

    array_4d = np.zeros((scan_y, scan_x, ny, nx), dtype=np.float32)
    for iy in range(scan_y):
        for ix in range(scan_x):
            # Add slight variations
            rng = np.random.default_rng(42 + iy * scan_x + ix)
            variation = 1.0 + 0.1 * rng.standard_normal()
            array_4d[iy, ix] = synthetic_diffraction_pattern * variation

    return Dataset4dstem.from_array(
        array=array_4d,
        name="test_4dstem",
        origin=(0, 0, 0, 0),
        sampling=(1.0, 1.0, 0.015, 0.015),
        units=["nm", "nm", "1/Angstrom", "1/Angstrom"],
        signal_units="counts",
    )


@pytest.fixture
def synthetic_dataset2d(synthetic_diffraction_pattern):
    """Create a synthetic 2D diffraction dataset."""
    return Dataset2d.from_array(
        array=synthetic_diffraction_pattern,
        name="test_2d_diffraction",
        origin=(0, 0),
        sampling=(0.015, 0.015),
        units=["1/Angstrom", "1/Angstrom"],
        signal_units="counts",
    )


# ============================================================================
# Test PairDistributionFunction Construction
# ============================================================================


class TestPairDistributionFunctionConstruction:
    """Test PairDistributionFunction initialization from various input types."""

    def test_from_data_with_dataset4dstem(self, synthetic_4dstem_dataset):
        """Test construction from a Dataset4dstem object."""
        pdf = PairDistributionFunction.from_data(
            synthetic_4dstem_dataset,
            find_origin=False,
        )
        assert isinstance(pdf.polar, Polar4dstem)
        assert pdf.input_data is synthetic_4dstem_dataset
        assert pdf.polar.shape[0] == 3  # scan_y
        assert pdf.polar.shape[1] == 3  # scan_x
        assert pdf.polar.shape[2] == 180  # num_annular_bins

    def test_direct_init_without_token_raises(self, synthetic_dataset2d):
        """Test that direct __init__ without token raises RuntimeError."""
        pdf_valid = PairDistributionFunction.from_data(synthetic_dataset2d, find_origin=False)
        with pytest.raises(RuntimeError, match="Use PairDistributionFunction.from_data"):
            PairDistributionFunction(polar=pdf_valid.polar, device="cpu")

    def test_from_data_origin_method(self, synthetic_4dstem_dataset):
        """from_data routes origin_method to the two finders and rejects bad values."""
        for method in ("grid", "descent"):
            pdf = PairDistributionFunction.from_data(
                synthetic_4dstem_dataset, find_origin=True, origin_method=method
            )
            assert pdf.polar.shape[:2] == (3, 3)
        with pytest.raises(ValueError, match="origin_method"):
            PairDistributionFunction.from_data(
                synthetic_4dstem_dataset, find_origin=True, origin_method="bogus"
            )


@pytest.mark.parametrize("finder", [find_origin_angular_grid, find_origin_angular_descent])
class TestOriginFinding:
    """Shared coverage for both origin finders (``find_origin_angular_grid`` and ``find_origin_angular_descent``). Each test runs against both."""

    @staticmethod
    def _ring_pattern(ny, nx, cy, cx, radii=(12, 24, 36, 48), beam_sigma=3.0):
        """Concentric circular rings + a central beam, centered at (cy, cx)."""
        y, x = np.ogrid[:ny, :nx]
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        p = np.zeros((ny, nx), dtype=np.float32)
        for radius in radii:
            p += 100.0 * np.exp(-((r - radius) ** 2) / (2 * 2.0**2))
        p += 1000.0 * np.exp(-(r**2) / (2 * beam_sigma**2))
        return p.astype(np.float32)

    @staticmethod
    def _wrap(arr):
        return Dataset4dstem.from_array(
            array=arr,
            name="origin_test",
            origin=(0, 0, 0, 0),
            sampling=(1.0, 1.0, 1.0, 1.0),
            units=["nm", "nm", "1/Angstrom", "1/Angstrom"],
            signal_units="counts",
        )

    def test_recovers_center(self, finder, synthetic_4dstem_dataset):
        """Recovers the known center of the shared 3x3 fixture."""
        origins = finder(synthetic_4dstem_dataset, radial_min=4, radial_max=50, device="cpu")
        assert origins.shape == (3, 3, 2)
        assert np.all(np.abs(origins - 127.5) < 0.5)

    def test_subpixel(self, finder):
        """Recovers distinct fractional centers to sub-pixel precision (also
        exercises per-position batching)."""
        ny = nx = 128
        true_centers = [(63.3, 64.7), (64.6, 63.4), (62.8, 65.2), (65.1, 62.9)]
        arr = np.stack([self._ring_pattern(ny, nx, cy, cx) for cy, cx in true_centers]).reshape(
            2, 2, ny, nx
        )
        origins = finder(self._wrap(arr), radial_min=4, radial_max=54, device="cpu")
        assert origins.shape == (2, 2, 2)
        for k, (cy, cx) in enumerate(true_centers):
            row, col = origins[k // 2, k % 2]
            err = float(np.hypot(row - cy, col - cx))
            assert err < 0.1, f"sub-pixel origin off by {err:.3f} px at position {k}"

    def test_small_detector(self, finder):
        """Works on a small (64 px) detector -- the regime where the grid finder's
        fixed search margin used to collapse the search annulus."""
        ny = nx = 64
        cy, cx = 30.4, 30.6
        arr = self._ring_pattern(ny, nx, cy, cx, radii=(8, 16, 24))[None, None]
        origins = finder(self._wrap(arr), radial_min=3, radial_max=28, device="cpu")
        assert origins.shape == (1, 1, 2)
        assert float(np.hypot(origins[0, 0, 0] - cy, origins[0, 0, 1] - cx)) < 0.3

    def test_accepts_tensor_backed_input(self, finder):
        """Runs on a tensor-backed dataset (whose .array is None)."""
        arr = self._ring_pattern(128, 128, 64.0, 64.0)[None, None]
        ds_t = Dataset4dstem.from_tensor(torch.from_numpy(arr))
        assert ds_t.array is None  # tensor-backed
        origins = finder(ds_t, radial_min=4, radial_max=54, device="cpu")
        assert origins.shape == (1, 1, 2)
        assert float(np.hypot(origins[0, 0, 0] - 64.0, origins[0, 0, 1] - 64.0)) < 0.3

    def test_ellipse_params(self, finder):
        """ellipse_params is accepted and the center is recovered on an elliptical
        ring when the correct correction is supplied."""
        ny = nx = 128
        cy, cx, theta_deg = 64.3, 63.7, 25.0
        theta = np.radians(theta_deg)
        a, b, ring_r = 40 * np.sqrt(1.6), 40 / np.sqrt(1.6), 40.0
        y, x = np.ogrid[:ny, :nx]
        u_col, u_row = x - cx, y - cy
        u_a = u_col * np.cos(theta) + u_row * np.sin(theta)
        u_b = -u_col * np.sin(theta) + u_row * np.cos(theta)
        r_ell = np.sqrt((u_a / a) ** 2 + (u_b / b) ** 2)
        p = 300.0 * np.exp(-(((r_ell - 1) * ring_r) ** 2) / (2 * 4.0**2))
        p += 1000.0 * np.exp(-(u_col**2 + u_row**2) / (2 * 3.0**2))
        ds = self._wrap(p.astype(np.float32)[None, None])
        origins = finder(
            ds, ellipse_params=(a, b, theta_deg), radial_min=8, radial_max=58, device="cpu"
        )
        err = float(np.hypot(origins[0, 0, 0] - cy, origins[0, 0, 1] - cx))
        assert err < 0.3, f"elliptical origin off by {err:.3f} px"


# ============================================================================
# Test Polar Transform
# ============================================================================


class TestPolarTransform:
    """Test polar coordinate transformation."""

    def test_polar_transform_basic(self, synthetic_4dstem_dataset):
        """Test basic polar transformation."""
        polar = polar_transform(synthetic_4dstem_dataset)
        assert isinstance(polar, Polar4dstem)
        assert polar.shape[0] == 3  # scan_y
        assert polar.shape[1] == 3  # scan_x
        assert polar.shape[2] == 180  # num_annular_bins
        assert polar.shape[3] > 0  # radial bins

    def test_polar_transform_single_origin(self, synthetic_4dstem_dataset):
        """Test polar transformation with single origin broadcast to all positions."""
        origin = np.array([128.0, 128.0])
        polar = polar_transform(
            synthetic_4dstem_dataset,
            origin_array=origin,
        )
        assert isinstance(polar, Polar4dstem)

    def test_polar_transform_radial_range(self, synthetic_4dstem_dataset):
        """Test polar transformation with custom radial range."""
        polar = polar_transform(
            synthetic_4dstem_dataset,
            radial_min=5.0,
            radial_max=50.0,
            radial_step=2.0,
        )
        assert isinstance(polar, Polar4dstem)
        # Check that radial dimension matches expected size
        expected_n_r = int(np.ceil((50.0 - 5.0) / 2.0))
        assert polar.shape[3] == expected_n_r

    def test_polar_transform_scan_pos(self, synthetic_4dstem_dataset):
        """Test polar transformation for a single scan position."""
        polar_2d = polar_transform(
            synthetic_4dstem_dataset,
            scan_pos=(0, 0),
        )
        # should return 2D tensor (phi, r)
        assert polar_2d.ndim == 2
        assert polar_2d.shape[0] == 180  # num_annular_bins


# ============================================================================
# Test Radial Mean Calculation
# ============================================================================


class TestRadialMeanCalculation:
    """Test radial mean intensity calculation."""

    def test_calculate_radial_mean_with_mask(self, synthetic_4dstem_dataset):
        """Test radial mean calculation with real-space mask."""
        pdf = PairDistributionFunction.from_data(
            synthetic_4dstem_dataset,
            find_origin=False,
        )
        mask = np.zeros((3, 3), dtype=bool)
        mask[0:2, 0:2] = True
        radial_mean = pdf.calculate_radial_mean(
            mask_realspace=mask,
            returnval=True,
        )
        assert radial_mean is not None

    def test_pipeline_torch_native(self, synthetic_4dstem_dataset):
        """The pipeline keeps the polar data on-device (tensor-backed) and
        gives identical results for numpy-backed vs tensor-backed input."""
        ds_np = synthetic_4dstem_dataset
        ds_t = Dataset4dstem.from_tensor(torch.from_numpy(ds_np.array.copy()))
        pdf_np = PairDistributionFunction.from_data(ds_np, find_origin=False)
        pdf_t = PairDistributionFunction.from_data(ds_t, find_origin=False)
        # Polar data is tensor-backed regardless of input backing (no round-trip).
        assert pdf_np.polar.array is None
        assert pdf_t.polar.array is None
        assert isinstance(pdf_t.polar.tensor, torch.Tensor)
        # numpy-backed and tensor-backed inputs give identical I(k).
        ik_np = pdf_np.calculate_radial_mean(returnval=True).cpu().numpy()
        ik_t = pdf_t.calculate_radial_mean(returnval=True).cpu().numpy()
        assert np.allclose(ik_np, ik_t, rtol=1e-5, atol=1e-6)


# ============================================================================
# Test Background Fitting
# ============================================================================


class TestBackgroundFitting:
    """Test background fitting."""

    def test_fit_bg_basic(self, synthetic_dataset2d):
        """Test basic background fitting."""
        pdf = PairDistributionFunction.from_data(
            synthetic_dataset2d,
            find_origin=False,
        )
        Ik = pdf.calculate_radial_mean(returnval=True)
        k = np.asarray(pdf.qq)
        kmin, kmax = float(k.min()), float(k.max())
        bg, f = pdf.fit_bg(Ik, kmin=kmin * 0.1, kmax=kmax * 0.9)
        assert bg.shape == Ik.shape
        assert f.shape == Ik.shape
        # Check that background is positive
        assert (bg >= 0).all()

    def test_fit_bg_recovers_known_background(self, synthetic_4dstem_dataset):
        """The torch fit recovers a known smooth background under a structure
        modulation, on a realistic high-dynamic-range I(k) / calibrated axis."""
        pdf = PairDistributionFunction.from_data(synthetic_4dstem_dataset, find_origin=False)
        nb = len(np.asarray(pdf.qq))
        pdf.polar.sampling[3] = 2.3 / (nb - 1)  # realistic 1/A q-axis
        k = np.asarray(pdf.qq)
        rng = np.random.RandomState(0)
        bg_true = (
            0.5 + 900.0 * np.exp(-(k**2) / (2 * 0.09**2)) + 6.0 * np.exp(-(k**4) / (2 * 0.8**4))
        )
        # add 35% amplitude ripple to the true bg to simulate scattering
        modulation = 1.0 + 0.35 * np.sin(2 * np.pi * k / 0.42 + 0.5) * np.exp(-k / 1.2)
        Ik = np.clip(bg_true * modulation + np.abs(0.5 * rng.randn(k.size)), 1e-6, None).astype(
            np.float32
        )
        # try fitting to this simulated Ik
        bg, _ = pdf.fit_bg(Ik.copy())
        bg = bg.cpu().numpy()
        assert (bg >= 0).all()
        # The fit tracks the modulated data, so pointwise error reflects the
        # ~35% structure modulation
        # the recovered shape of the smooth background should still correlate strongly with the truth.
        corr = float(np.corrcoef(bg, bg_true)[0, 1])
        assert corr > 0.99, f"bg shape far from truth: corr {corr:.4g}"

    def test_fit_bg_batched_matches_per_curve(self, synthetic_4dstem_dataset):
        """fit_bg_batched reproduces the per-curve torch fit for a stack of
        realistic radial means (the line-scan fits one background per bin)."""
        pdf = PairDistributionFunction.from_data(synthetic_4dstem_dataset, find_origin=False)
        nb = len(np.asarray(pdf.qq))
        pdf.polar.sampling[3] = 2.3 / (nb - 1)
        k = np.asarray(pdf.qq)
        # A stack of 8 distinct I(k) (central beam + envelope + a gentle, varying structure modulation).
        curves = []
        for seed in range(8):
            rng = np.random.RandomState(seed)
            bg_true = (
                0.5
                + (400.0 + 300.0 * rng.rand()) * np.exp(-(k**2) / (2 * 0.09**2))
                + 6.0 * np.exp(-(k**4) / (2 * 0.8**4))
            )
            mod = 1.0 + 0.2 * np.sin(
                2 * np.pi * k / (0.4 + 0.2 * rng.rand()) + rng.rand()
            ) * np.exp(-k / 1.2)
            curves.append((bg_true * mod).astype(np.float32))
        stack = np.stack(curves)

        bg_batched, _ = pdf.fit_bg_batched(stack, kmin=0.1)
        bg_batched = bg_batched.cpu().numpy()
        for i in range(stack.shape[0]):
            bg_single, _ = pdf.fit_bg(stack[i].copy(), kmin=0.1)
            bg_single = bg_single.cpu().numpy()
            rel = np.abs(bg_batched[i] - bg_single) / np.clip(np.abs(bg_single), 1e-6, None)
            assert rel.max() < 1e-3, f"batched vs per-curve bg mismatch (row {i}): {rel.max():.4g}"


# ============================================================================
# Test PDF Calculation
# ============================================================================


class TestPDFCalculation:
    """Test the PDF calculation pipeline."""

    def test_calculate_Gr_with_bandpass(self, synthetic_dataset2d):
        """Test PDF calculation with bandpass filtering."""
        pdf = PairDistributionFunction.from_data(
            synthetic_dataset2d,
            find_origin=False,
        )
        pdf.calculate_Gr(
            k_min_fit=0.1,
            k_max_fit=2.0,
            k_lowpass=0.02,
            k_highpass=0.001,
        )
        assert pdf.reduced_pdf is not None

    def test_calculate_Gr_with_mask(self, synthetic_4dstem_dataset):
        """Test PDF calculation with real-space mask."""
        pdf = PairDistributionFunction.from_data(
            synthetic_4dstem_dataset,
            find_origin=False,
        )
        mask = np.zeros((3, 3), dtype=bool)
        mask[0:2, 0:2] = True
        pdf.calculate_Gr(
            k_min_fit=0.1,
            k_max_fit=2.0,
            mask_realspace=mask,
        )
        assert pdf.reduced_pdf is not None

    def test_calculate_gr_requires_Gr(self, synthetic_dataset2d):
        """Test that calculate_gr raises if calculate_Gr has not been run."""
        pdf = PairDistributionFunction.from_data(
            synthetic_dataset2d,
            find_origin=False,
        )
        with pytest.raises(RuntimeError, match="Reduced PDF not computed"):
            pdf.calculate_gr(density=0.05)

    def test_calculate_gr_requires_density(self, synthetic_dataset2d):
        """The density is an input: calculate_gr raises without one and never estimates it."""
        pdf = PairDistributionFunction.from_data(
            synthetic_dataset2d,
            find_origin=False,
        )
        pdf.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0)
        with pytest.raises(ValueError, match="needs the number density"):
            pdf.calculate_gr()
        r, gr = pdf.calculate_gr(density=0.05, returnval=True)
        assert isinstance(gr, np.ndarray)
        assert len(gr) == len(r)
        # a density cached by a damped calculate_Gr is reused
        pdf.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0, damp_origin_oscillations=True, density=0.05)
        assert pdf.rho0 == 0.05
        assert pdf.calculate_gr(returnval=True) is not None

    def test_damping_requires_density_and_Gr(self, synthetic_dataset2d):
        pdf = PairDistributionFunction.from_data(
            synthetic_dataset2d,
            find_origin=False,
        )
        with pytest.raises(RuntimeError, match="depends on Sk, reduced_pdf, and r"):
            pdf.damp_origin_oscillations(density=0.05)
        with pytest.raises(ValueError, match="requires `density`"):
            pdf.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0, damp_origin_oscillations=True)


# ============================================================================
# Integration Workflows
# ============================================================================


class TestIntegrationWorkflows:
    """Test complete end-to-end workflows."""

    def test_complete_pdf_workflow_2d(self, synthetic_dataset2d):
        """Test: 2D diffraction → polar transform → G(r) → g(r)."""
        pdf = PairDistributionFunction.from_data(
            synthetic_dataset2d,
            find_origin=False,
        )
        Gr_results = pdf.calculate_Gr(
            k_min_fit=0.1,
            k_max_fit=2.0,
            r_max=10.0,
            r_step=0.05,
            returnval=True,
        )
        assert Gr_results is not None
        r, Gr = Gr_results
        assert not np.isnan(r).any()
        assert not np.isnan(Gr).any()
        assert not np.isinf(Gr).any()
        assert len(r) > 0
        assert len(Gr) == len(r)
        gr_results = pdf.calculate_gr(
            density=0.05,
            returnval=True,
        )
        assert gr_results is not None
        r_gr, gr = gr_results
        assert not np.isnan(gr).any()
        assert not np.isinf(gr).any()
        assert len(gr) == len(r_gr)

    def test_complete_pdf_workflow_4dstem(self, synthetic_4dstem_dataset):
        """Test: 4D-STEM → origin finding → polar transform → G(r)."""
        pdf = PairDistributionFunction.from_data(
            synthetic_4dstem_dataset,
            find_origin=True,
        )
        mask = np.zeros((3, 3), dtype=bool)
        mask[0:2, 0:2] = True
        pdf.calculate_Gr(
            k_min_fit=0.1,
            k_max_fit=2.0,
            mask_realspace=mask,
        )
        assert pdf.reduced_pdf is not None
        assert not np.isnan(pdf.reduced_pdf).any()
        assert not np.isinf(pdf.reduced_pdf).any()

    def test_polar_transform_input_types(self, synthetic_diffraction_pattern):
        """Test from_data works with Dataset2d and Dataset4dstem."""
        # Test with Dataset2d
        ds2 = Dataset2d.from_array(
            array=synthetic_diffraction_pattern,
            name="test",
        )
        pdf_ds2 = PairDistributionFunction.from_data(
            ds2,
            find_origin=False,
        )
        assert pdf_ds2.polar.shape[2] == 180

        # Test with Dataset4dstem
        array_4d = synthetic_diffraction_pattern[None, None, :, :]  # (1, 1, ny, nx)
        ds4 = Dataset4dstem.from_array(array_4d, name="test")
        pdf_ds4 = PairDistributionFunction.from_data(
            ds4,
            find_origin=False,
        )
        assert pdf_ds4.polar.shape[2] == 180
        assert pdf_ds2.polar.shape == pdf_ds4.polar.shape

    def test_low_r_correction_workflow(self, synthetic_dataset2d):
        """Test: G(r) calculation → low-r correction with a known density → g(r)."""
        pdf = PairDistributionFunction.from_data(
            synthetic_dataset2d,
            find_origin=False,
        )
        pdf.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0)
        Fk_damped, G_cor = pdf.damp_origin_oscillations(
            density=0.05,
            max_iter=5,
            tolerance=1e-2,
        )
        assert torch.isfinite(G_cor).all()
        assert len(G_cor) == len(pdf.r)
        results = pdf.calculate_gr(
            density=0.05,
            returnval=True,
        )
        assert results is not None
        r, gr = results
        assert not np.isnan(gr).any()


# ============================================================================
# Test state handling across repeated calls
# ============================================================================


class TestStateHandling:
    """Repeated calls must give the same result as a fresh object."""

    def test_refit_background_when_fit_range_changes(self, synthetic_dataset2d):
        pdf = PairDistributionFunction.from_data(synthetic_dataset2d, find_origin=False)
        pdf.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0)
        pdf.calculate_Gr(k_min_fit=0.3, k_max_fit=1.5)

        fresh = PairDistributionFunction.from_data(synthetic_dataset2d, find_origin=False)
        fresh.calculate_Gr(k_min_fit=0.3, k_max_fit=1.5)

        np.testing.assert_allclose(pdf.bg.cpu().numpy(), fresh.bg.cpu().numpy())
        np.testing.assert_allclose(pdf.reduced_pdf, fresh.reduced_pdf)

    def test_unmasked_call_after_masked_recomputes_radial_mean(self, synthetic_4dstem_dataset):
        pdf = PairDistributionFunction.from_data(synthetic_4dstem_dataset, find_origin=False)
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        pdf.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0, mask_realspace=mask)
        pdf.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0)

        fresh = PairDistributionFunction.from_data(synthetic_4dstem_dataset, find_origin=False)
        fresh.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0)

        np.testing.assert_allclose(pdf.Ik.cpu().numpy(), fresh.Ik.cpu().numpy())
        np.testing.assert_allclose(pdf.reduced_pdf, fresh.reduced_pdf)

    def test_direct_fit_bg_does_not_poison_calculate_Gr(self, synthetic_dataset2d):
        pdf = PairDistributionFunction.from_data(synthetic_dataset2d, find_origin=False)
        Ik = pdf.calculate_radial_mean(returnval=True)
        pdf.fit_bg(Ik, kmin=0.5, kmax=1.0)
        pdf.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0)

        fresh = PairDistributionFunction.from_data(synthetic_dataset2d, find_origin=False)
        fresh.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0)

        np.testing.assert_allclose(pdf.bg.cpu().numpy(), fresh.bg.cpu().numpy())
        np.testing.assert_allclose(pdf.reduced_pdf, fresh.reduced_pdf)

    def test_rerun_clears_stale_pdf(self, synthetic_dataset2d):
        pdf = PairDistributionFunction.from_data(synthetic_dataset2d, find_origin=False)
        assert pdf.Fk_masked is None
        pdf.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0, r_max=10.0, r_step=0.05)
        pdf.calculate_gr(density=0.05)
        assert pdf.Fk_masked is not None

        pdf.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0, r_max=5.0, r_step=0.02)
        assert pdf.pdf is None
        r, gr = pdf.calculate_gr(density=0.05, returnval=True)
        assert len(gr) == len(r) == len(pdf.r)

    def test_calculate_gr_new_density_after_damping_matches_fresh_run(self, synthetic_dataset2d):
        pdf = PairDistributionFunction.from_data(synthetic_dataset2d, find_origin=False)
        pdf.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0, damp_origin_oscillations=True, density=0.05)
        pdf.calculate_gr(density=0.1)

        fresh = PairDistributionFunction.from_data(synthetic_dataset2d, find_origin=False)
        fresh.calculate_Gr(
            k_min_fit=0.1, k_max_fit=2.0, damp_origin_oscillations=True, density=0.1
        )
        fresh.calculate_gr(density=0.1)

        assert pdf.rho0 == 0.1
        np.testing.assert_allclose(
            pdf.reduced_pdf_damped.cpu().numpy(), fresh.reduced_pdf_damped.cpu().numpy()
        )
        np.testing.assert_allclose(pdf.pdf, fresh.pdf)

    def test_max_iter_reaches_the_low_r_correction(self, synthetic_dataset2d):
        pdf = PairDistributionFunction.from_data(synthetic_dataset2d, find_origin=False)
        kw = dict(k_min_fit=0.1, k_max_fit=2.0, damp_origin_oscillations=True, density=0.05)
        # no iterations: the corrected G(r) is the uncorrected one
        pdf.calculate_Gr(**kw, max_iter=0)
        G0 = pdf.reduced_pdf
        np.testing.assert_allclose(
            pdf.reduced_pdf_damped.cpu().numpy(), G0, rtol=0, atol=1e-4 * np.abs(G0).max()
        )
        pdf.calculate_Gr(**kw, max_iter=1)
        G1 = pdf.reduced_pdf_damped.cpu().numpy()
        pdf.calculate_Gr(**kw, max_iter=6)
        assert not np.allclose(G1, pdf.reduced_pdf_damped.cpu().numpy())


# ============================================================================
# Known-answer tests: analytic hard-sphere S(k) with known density
# ============================================================================


def _percus_yevick_S(k, rho0=0.05, eta=0.45, u=0.15):
    """Analytic hard-sphere S(k) (Percus-Yevick, Gaussian damped): physical, known rho0."""
    q = 2 * np.pi * k
    sig = (6 * eta / (np.pi * rho0)) ** (1 / 3)
    a = (1 + 2 * eta) ** 2 / (1 - eta) ** 4
    b = -6 * eta * (1 + eta / 2) ** 2 / (1 - eta) ** 4
    g = eta * (1 + 2 * eta) ** 2 / (2 * (1 - eta) ** 4)
    r = np.linspace(0, sig, 4000)
    x = r / sig
    c = -(a + b * x + g * x**3)
    qr = np.outer(q, r)
    j0 = np.ones_like(qr)
    nz = qr != 0
    j0[nz] = np.sin(qr[nz]) / qr[nz]
    chat = 4 * np.pi * np.trapezoid(r**2 * c * j0, r, axis=1)
    S = 1 / (1 - rho0 * chat)
    return 1 + (S - 1) * np.exp(-(q**2) * u**2 / 2)


class TestKnownAnswer:
    """Drive the transform with a known S(k) (background fit bypassed: bg = 0, f = 1)."""

    RHO0 = 0.05

    @classmethod
    def _pdf_with_injected_S(cls):
        img = np.random.default_rng(0).random((401, 401)).astype(np.float32)
        ds = Dataset2d.from_array(
            array=img,
            name="hard_spheres",
            origin=(0, 0),
            sampling=(0.01, 0.01),
            units=["1/Angstrom", "1/Angstrom"],
            signal_units="counts",
        )
        pdf = PairDistributionFunction.from_data(ds, find_origin=False)
        k = np.asarray(pdf.qq)
        S = _percus_yevick_S(k, rho0=cls.RHO0)
        pdf.Ik = torch.tensor(S - 1, dtype=torch.float32)
        pdf.bg = torch.zeros(len(k))
        pdf.f = torch.ones(len(k))
        pdf.fit_bg = lambda Ik, kmin=None, kmax=None: (pdf.bg, pdf.f)
        return pdf, k, S

    def test_G_matches_windowed_transform_of_true_S(self):
        """G(r) = (2/pi) sum F(q) sin(qr) dq with F = 2 pi k [S-1]: catches any 2pi, dk or
        r-scaling error in the transform (py4DSTEM's G is 4 pi^2 too small)."""
        pdf, k, S = self._pdf_with_injected_S()
        pdf.calculate_Gr(k_min_fit=0.01, k_max_fit=float(k[-1]), r_max=10.0, r_step=0.02)
        r = pdf.r
        wk = pdf._k_window(
            torch.tensor(k, dtype=torch.float32), pdf.kmin_window, pdf.kmax_window
        ).numpy()
        Fk = k * (S - 1)
        F = 2 * np.pi * Fk * wk
        dk = k[1] - k[0]
        G_ref = (2 / np.pi) * dk * 2 * np.pi * (np.sin(2 * np.pi * np.outer(r, k)) @ F)
        G_ref[0] = 0.0
        np.testing.assert_allclose(pdf.reduced_pdf, G_ref, rtol=0, atol=2e-3 * np.abs(G_ref).max())
        # first-neighbour peak of the hard-sphere shell (diameter 2.58 A)
        assert abs(r[np.argmax(pdf.reduced_pdf)] - 2.58) < 0.35

    def test_low_r_correction_with_known_density(self):
        """With the true density and r_cut below the contact distance, the corrected G(r)
        is -4 pi rho0 r there, g(r) is ~0, and the first peak is not moved."""
        pdf, k, S = self._pdf_with_injected_S()
        pdf.calculate_Gr(k_min_fit=0.01, k_max_fit=float(k[-1]), r_max=10.0, r_step=0.02)
        r = pdf.r
        peak_before = r[np.argmax(pdf.reduced_pdf)]
        pdf.calculate_Gr(
            k_min_fit=0.01,
            k_max_fit=float(k[-1]),
            r_max=10.0,
            r_step=0.02,
            damp_origin_oscillations=True,
            density=self.RHO0,
            r_cut=2.0,  # below the hard-sphere contact at 2.58 A
        )
        G_cor = pdf.reduced_pdf_damped.cpu().numpy()
        m = (r > 0.3) & (r < 1.5)
        np.testing.assert_allclose(G_cor[m] / r[m], -4 * np.pi * self.RHO0, rtol=0.05)
        r_g, g = pdf.calculate_gr(returnval=True)
        assert np.max(np.abs(g[m])) < 0.1
        assert abs(r[np.argmax(G_cor)] - peak_before) < 0.1
