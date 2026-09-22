"""Tests for the ``Dataset3deels`` ZLP-offset robustness fix.

Covers ``measure_zlp_offset`` and ``calculate_thickness_log_ratio``: both had
an unguarded per-pixel ``curve_fit`` (crashing on a single bad pixel out of
thousands -- the InSitu (16) reproducer), ``np.argmax`` hijacked by NaN, a
missing ``return`` in the ``fit_to_polynomial`` branch, and no way to size
``fit_window`` for monochromated (sub-eV ZLP) data. See PROMPT_zlp_fix.md /
ADDENDUM_zlp_fix_stage2.md at the repo root for the full design rationale.
"""

import inspect
import warnings

import numpy as np
import pytest

from quantem.spectroscopy.dataset3deels import (
    Dataset3deels,
    _estimate_zlp_fwhm,
    _resolve_auto_fit_window,
)

from .conftest import (
    SCAN_COL,
    SCAN_ROW,
    dataset_from_array,
    energy_axis,
    make_tilted_zlp_scan,
)


class _no_warnings:
    """Assert no warnings fire in the block (the straightforward replacement
    for the now-removed ``pytest.warns(None)``)."""

    def __enter__(self):
        self._cm = warnings.catch_warnings(record=True)
        self._records = self._cm.__enter__()
        warnings.simplefilter("always")
        return self._records

    def __exit__(self, exc_type, exc, tb):
        self._cm.__exit__(exc_type, exc, tb)
        if exc_type is None:
            assert not self._records, (
                f"unexpected warnings: {[str(r.message) for r in self._records]}"
            )
        return False


class TestMeasureZlpOffset:
    def test_known_tilted_plane_recovery(self, tilted_zlp_dataset):
        ds, true_plane = tilted_zlp_dataset
        with _no_warnings():
            plane = ds.measure_zlp_offset(fit_window=0.5, fit_to_plane=True, fit_zlp=True)
        np.testing.assert_allclose(plane, true_plane, atol=1e-2)

    def test_injected_nan_run_is_excluded_and_interpolated(self, tilted_zlp_scan):
        array, true_plane = tilted_zlp_scan
        array = array.copy()
        array[3, 4, :] = np.nan
        ds = dataset_from_array(array)

        with pytest.warns(UserWarning, match=r"1 failed"):
            plane = ds.measure_zlp_offset(fit_window=0.5, fit_to_plane=True, fit_zlp=True)

        assert np.isfinite(plane).all()
        # the excluded pixel's value comes from the plane fit over the other
        # 143 pixels, which (noise aside) should still land close to truth.
        assert plane[3, 4] == pytest.approx(true_plane[3, 4], abs=0.05)

    def test_injected_all_negative_pixel_is_excluded(self, tilted_zlp_scan):
        array, _ = tilted_zlp_scan
        array = array.copy()
        array[5, 5, :] = -np.abs(array[5, 5, :]) - 1.0
        ds = dataset_from_array(array)

        with pytest.warns(UserWarning, match=r"ValueError"):
            ds.measure_zlp_offset(fit_window=0.5, fit_to_plane=True, fit_zlp=True)

    def test_curve_fit_runtime_error_is_excluded_and_interpolated(
        self, tilted_zlp_dataset, monkeypatch
    ):
        ds, true_plane = tilted_zlp_dataset
        import quantem.spectroscopy.dataset3deels as mod

        real_curve_fit = mod.curve_fit
        call_count = {"n": 0}

        def flaky_curve_fit(f, xw, yw, p0=None, bounds=None, **kw):
            call_count["n"] += 1
            if call_count["n"] == 7:
                raise RuntimeError("Optimal parameters not found: forced test failure")
            return real_curve_fit(f, xw, yw, p0=p0, bounds=bounds, **kw)

        monkeypatch.setattr(mod, "curve_fit", flaky_curve_fit)

        with pytest.warns(UserWarning, match=r"RuntimeError"):
            plane = ds.measure_zlp_offset(fit_window=0.5, fit_to_plane=True, fit_zlp=True)

        assert np.isfinite(plane).all()
        np.testing.assert_allclose(plane, true_plane, atol=0.05)

    def test_boundary_pinned_fit_is_rejected(self, tilted_zlp_dataset, monkeypatch):
        """A fit that "succeeds" with mu pinned at the window edge must be
        treated as a failure, not a silently-accepted value."""
        ds, true_plane = tilted_zlp_dataset
        import quantem.spectroscopy.dataset3deels as mod

        real_curve_fit = mod.curve_fit
        call_count = {"n": 0}

        def pinning_curve_fit(f, xw, yw, p0=None, bounds=None, **kw):
            call_count["n"] += 1
            if call_count["n"] == 10:
                A0, _mu0, sigma0 = p0
                lo = bounds[0][1]
                # scipy would return something extremely close to, but not
                # exactly at, the bound -- emulate that.
                pinned_mu = lo + 1e-9
                return np.array([A0, pinned_mu, sigma0]), None
            return real_curve_fit(f, xw, yw, p0=p0, bounds=bounds, **kw)

        monkeypatch.setattr(mod, "curve_fit", pinning_curve_fit)

        with pytest.warns(UserWarning, match=r"1 boundary-pinned"):
            plane = ds.measure_zlp_offset(fit_window=0.5, fit_to_plane=True, fit_zlp=True)

        assert np.isfinite(plane).all()
        np.testing.assert_allclose(plane, true_plane, atol=0.05)

    def test_masked_region_is_excluded_and_interpolated(self, tilted_zlp_dataset):
        ds, true_plane = tilted_zlp_dataset
        mask = np.zeros((SCAN_ROW, SCAN_COL), dtype=bool)
        mask[7, 7] = True

        with pytest.warns(UserWarning, match=r"1 masked"):
            plane = ds.measure_zlp_offset(
                fit_window=0.5, fit_to_plane=True, fit_zlp=True, mask=mask
            )

        assert np.isfinite(plane).all()
        assert plane[7, 7] == pytest.approx(true_plane[7, 7], abs=0.05)

    def test_mask_shape_mismatch_raises(self, tilted_zlp_dataset):
        ds, _ = tilted_zlp_dataset
        bad_mask = np.zeros((SCAN_ROW + 1, SCAN_COL), dtype=bool)
        with pytest.raises(ValueError, match=r"mask shape"):
            ds.measure_zlp_offset(fit_window=0.5, fit_to_plane=True, mask=bad_mask)

    def test_too_few_good_pixels_raises(self, tilted_zlp_dataset):
        ds, _ = tilted_zlp_dataset
        mask = np.ones((SCAN_ROW, SCAN_COL), dtype=bool)
        mask[0, 0] = False  # only 1 pixel survives
        with pytest.raises(ValueError, match=r"usable ZLP fit"):
            ds.measure_zlp_offset(fit_window=0.5, fit_to_plane=True, mask=mask)

    def test_fit_to_polynomial_returns_array_not_none(self, tilted_zlp_dataset):
        ds, _ = tilted_zlp_dataset
        result = ds.measure_zlp_offset(fit_window=0.5, fit_to_polynomial=True, fit_zlp=True)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (SCAN_ROW, SCAN_COL)
        assert np.isfinite(result).all()

    def test_fit_window_default_is_unchanged(self):
        sig = inspect.signature(Dataset3deels.measure_zlp_offset)
        assert sig.parameters["fit_window"].default == 0.8

    def test_estimate_zlp_fwhm_accuracy_on_flat_scan(self):
        # Un-tilted, single Gaussian (no plane-spread broadening to account
        # for) on a fine energy grid, so the half-max-crossing walk isn't
        # dominated by channel quantization -- isolates the FWHM estimator's
        # own accuracy.
        narrow_sigma = 0.05
        fine_energy = np.linspace(-2, 2, 4000)
        mean_spectrum = 1000.0 * np.exp(-0.5 * (fine_energy / narrow_sigma) ** 2)

        true_fwhm = narrow_sigma * 2 * np.sqrt(2 * np.log(2))
        estimated_fwhm = _estimate_zlp_fwhm(fine_energy, mean_spectrum)
        assert estimated_fwhm == pytest.approx(true_fwhm, rel=0.02)

    def test_fit_window_auto_resolves_within_fwhm_band_and_recovers_plane(self):
        # Deliberately narrow, tilted ZLP (monochromated-data regime). The
        # mean-spectrum FWHM estimate is necessarily broadened here by the
        # spread of per-pixel centers across the tilted plane on top of the
        # intrinsic sigma=0.05 width -- that's expected (see the flat-scan
        # test above for the intrinsic-width accuracy check) -- so this test
        # checks the "auto" mechanism self-consistently against whatever
        # FWHM it estimates, plus that plane recovery still works.
        narrow_sigma = 0.05
        array, true_plane = make_tilted_zlp_scan(sigma=narrow_sigma, noise=0.2, seed=1)
        energy = energy_axis()
        mean_spectrum = array.mean(axis=(0, 1))

        estimated_fwhm = _estimate_zlp_fwhm(energy, mean_spectrum)

        multiplier = 3.5
        resolved = _resolve_auto_fit_window("auto", energy, mean_spectrum, multiplier=multiplier)
        assert 2.5 * estimated_fwhm <= resolved <= 5.0 * estimated_fwhm

        ds = dataset_from_array(array)
        with _no_warnings():
            plane = ds.measure_zlp_offset(
                fit_window="auto",
                fit_to_plane=True,
                fit_zlp=True,
                fit_window_fwhm_multiplier=multiplier,
            )
        np.testing.assert_allclose(plane, true_plane, atol=0.02)

    def test_fit_window_invalid_string_raises(self, tilted_zlp_dataset):
        ds, _ = tilted_zlp_dataset
        with pytest.raises(ValueError, match=r"fit_window must be"):
            ds.measure_zlp_offset(fit_window="not-auto", fit_to_plane=True)


class TestApplyZlpCorrectionMaskPassthrough:
    def test_mask_and_fit_window_auto_forwarded(self, tilted_zlp_dataset):
        ds, _ = tilted_zlp_dataset
        mask = np.zeros((SCAN_ROW, SCAN_COL), dtype=bool)
        mask[2, 2] = True

        with pytest.warns(UserWarning, match=r"1 masked"):
            corrected, shifts = ds.apply_zlp_correction(
                measure_offset=True,
                fit_to_plane=True,
                fit_window="auto",
                mask=mask,
                return_3d_dataset=True,
                return_shifts=True,
            )

        assert np.isfinite(shifts).all()
        assert corrected.array.shape[:2] == (SCAN_ROW, SCAN_COL)


class TestCalculateThicknessLogRatio:
    def test_returns_finite_map_on_clean_data(self, tilted_zlp_dataset):
        ds, _ = tilted_zlp_dataset
        with _no_warnings():
            t_map = ds.calculate_thickness_log_ratio(zlp_window=1.0, plot=False)
        assert np.isfinite(t_map).all()
        assert t_map.shape == (SCAN_ROW, SCAN_COL)

    def test_injected_nan_pixel_is_handled(self, tilted_zlp_scan):
        array, _ = tilted_zlp_scan
        array = array.copy()
        array[3, 4, :] = np.nan
        ds = dataset_from_array(array)

        with pytest.warns(UserWarning, match=r"1 failed"):
            t_map = ds.calculate_thickness_log_ratio(zlp_window=1.0, plot=False)
        assert np.isfinite(t_map).all()

    def test_masked_region_is_handled(self, tilted_zlp_dataset):
        ds, _ = tilted_zlp_dataset
        mask = np.zeros((SCAN_ROW, SCAN_COL), dtype=bool)
        mask[6, 6] = True
        with pytest.warns(UserWarning, match=r"1 masked"):
            t_map = ds.calculate_thickness_log_ratio(zlp_window=1.0, mask=mask, plot=False)
        assert np.isfinite(t_map).all()

    def test_too_few_good_pixels_raises(self, tilted_zlp_dataset):
        ds, _ = tilted_zlp_dataset
        mask = np.ones((SCAN_ROW, SCAN_COL), dtype=bool)
        mask[0, 0] = False
        with pytest.raises(ValueError, match=r"usable ZLP fit"):
            ds.calculate_thickness_log_ratio(zlp_window=1.0, mask=mask, plot=False)

    def test_zlp_window_auto(self, tilted_zlp_dataset):
        ds, _ = tilted_zlp_dataset
        with _no_warnings():
            t_map = ds.calculate_thickness_log_ratio(zlp_window="auto", plot=False)
        assert np.isfinite(t_map).all()
