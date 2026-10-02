"""Tests for the reflected/mirrored-tail ZLP model and the Cherenkov
radiation feasibility check -- promoted from the sandbox's
``pipeline/stem_eels_pipeline.py`` / ``pipeline/build_dataset_notebooks.py``
into the library proper (``quantem.spectroscopy.spectroscopy_visualzitions``
and ``quantem.spectroscopy.utils`` respectively).

``build_reflected_zlp_model`` mirrors a spectrum's negative-energy-loss side
(pure instrument response, since energy loss cannot physically be negative)
onto the positive side to model the ZLP tail without fitting a functional
form -- see its docstring for the literature basis (Rafferty & Brown-style
ZLP deconvolution; Stoeger-Pollach's low-loss EELS review).

``cherenkov_feasibility_check`` is a standalone relativistic-kinematics
calculation (beta = v/c for an electron accelerated through a given voltage,
compared against a literature refractive-index range) used to flag whether
Cherenkov emission is an energetically-allowed confound for a near-ZLP
low-loss feature.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from quantem.spectroscopy.spectroscopy_visualzitions import (
    build_reflected_zlp_model,
    plot_reflected_zlp_steps,
)
from quantem.spectroscopy.utils import cherenkov_feasibility_check


def _symmetric_zlp_with_bump(bump_center=2.0, bump_amp=3.0, bump_fwhm=0.4):
    """A synthetic spectrum: a symmetric Gaussian ZLP (so the negative side
    is a perfect model of the positive-side tail) plus a one-sided Gaussian
    bump on the positive side only -- the "real signal" the reflected-tail
    method should recover."""
    E = np.linspace(-5.0, 10.0, 600)
    sigma = bump_fwhm / 2.3548
    zlp = 100.0 * np.exp(-0.5 * (E / 0.3) ** 2)
    bump = bump_amp * np.exp(-0.5 * ((E - bump_center) / sigma) ** 2) * (E > 0)
    spectrum = zlp + bump + 0.01
    return E, spectrum


class TestBuildReflectedZlpModel:
    def test_recovers_injected_bump_amplitude(self):
        E, spectrum = _symmetric_zlp_with_bump(bump_center=2.0, bump_amp=3.0)
        result = build_reflected_zlp_model(E, spectrum, zlp_center=0.0)

        near_bump = (result["E_shifted"] > 1.5) & (result["E_shifted"] < 2.5)
        recovered_peak = np.nanmax(result["reflected_subtracted"][near_bump])
        # the symmetric ZLP's mirrored tail should cancel almost exactly,
        # leaving close to the injected bump amplitude (3.0) as the residual
        assert recovered_peak == pytest.approx(3.0, abs=0.05)

    def test_negative_coverage_matches_axis_extent(self):
        E, spectrum = _symmetric_zlp_with_bump()
        result = build_reflected_zlp_model(E, spectrum, zlp_center=0.0)
        # axis runs to -5 eV, so the model should be trustworthy out to +5 eV
        assert result["negative_coverage_eV"] == pytest.approx(5.0, abs=1e-6)

    def test_model_is_nan_beyond_negative_coverage(self):
        E, spectrum = _symmetric_zlp_with_bump()
        result = build_reflected_zlp_model(E, spectrum, zlp_center=0.0)
        beyond = result["E_shifted"] > result["negative_coverage_eV"]
        assert beyond.any()
        assert np.all(np.isnan(result["reflected_zlp_model"][beyond]))

    def test_model_is_nan_on_negative_side(self):
        E, spectrum = _symmetric_zlp_with_bump()
        result = build_reflected_zlp_model(E, spectrum, zlp_center=0.0)
        neg_side = result["E_shifted"] <= 0
        assert np.all(np.isnan(result["reflected_zlp_model"][neg_side]))

    def test_zlp_center_shift_is_applied(self):
        E, spectrum = _symmetric_zlp_with_bump()
        shift = 1.2
        result = build_reflected_zlp_model(E + shift, spectrum, zlp_center=shift)
        np.testing.assert_allclose(result["E_shifted"], np.sort(E), atol=1e-9)

    def test_shape_mismatch_raises(self):
        E = np.linspace(-5, 5, 100)
        spectrum = np.ones(50)
        with pytest.raises(ValueError, match="same shape"):
            build_reflected_zlp_model(E, spectrum)

    def test_no_negative_channels_raises(self):
        E = np.linspace(0.1, 10, 100)  # entirely positive
        spectrum = np.ones_like(E)
        with pytest.raises(ValueError, match="nothing to mirror"):
            build_reflected_zlp_model(E, spectrum, zlp_center=0.0)

    def test_unsorted_energy_axis_is_handled(self):
        E, spectrum = _symmetric_zlp_with_bump()
        rng = np.random.default_rng(0)
        order = rng.permutation(len(E))
        result_shuffled = build_reflected_zlp_model(E[order], spectrum[order], zlp_center=0.0)
        result_sorted = build_reflected_zlp_model(E, spectrum, zlp_center=0.0)
        np.testing.assert_allclose(
            result_shuffled["reflected_subtracted"],
            result_sorted["reflected_subtracted"],
            equal_nan=True,
        )


class TestPlotReflectedZlpSteps:
    def test_returns_figure_and_result_without_showing(self):
        E, spectrum = _symmetric_zlp_with_bump()
        fig, result = plot_reflected_zlp_steps(E, spectrum, zlp_center=0.0, show=False)
        assert len(fig.axes) == 4
        assert "reflected_subtracted" in result
        matplotlib.pyplot.close(fig)

    def test_near_gap_window_clipped_to_coverage(self):
        # axis only extends to -1 eV on the negative side -> coverage is 1 eV,
        # so a near_gap_window reaching to 3 eV must be clipped, not silently
        # plotted past what the model actually covers
        E = np.linspace(-1.0, 8.0, 400)
        sigma = 0.4 / 2.3548
        zlp = 100.0 * np.exp(-0.5 * (E / 0.3) ** 2)
        bump = 3.0 * np.exp(-0.5 * ((E - 2.0) / sigma) ** 2) * (E > 0)
        spectrum = zlp + bump + 0.01
        fig, result = plot_reflected_zlp_steps(
            E, spectrum, zlp_center=0.0, near_gap_window=(0.5, 3.0), show=False
        )
        assert result["negative_coverage_eV"] == pytest.approx(1.0, abs=1e-6)
        matplotlib.pyplot.close(fig)


class TestCherenkovFeasibilityCheck:
    def test_300kV_water_ice_is_allowed(self):
        # matches the real DM4-derived value measured on this project's own
        # 300 kV monochromated STEM-EELS data: beta=0.77653, n_threshold=1.28779
        out = cherenkov_feasibility_check(300_000)
        assert out["beta"] == pytest.approx(0.77653, abs=1e-5)
        assert out["n_threshold"] == pytest.approx(1.28779, abs=1e-5)
        assert out["allowed_lo"] is True
        assert out["allowed_hi"] is True
        assert "IS energetically allowed" in out["verdict"]

    def test_low_voltage_is_not_allowed(self):
        out = cherenkov_feasibility_check(60_000)
        assert out["allowed_lo"] is False
        assert out["allowed_hi"] is False
        assert "NOT energetically allowed" in out["verdict"]

    def test_borderline_case_reports_borderline_verdict(self):
        # pick a voltage whose n_threshold sits strictly between n_lo and n_hi
        out = cherenkov_feasibility_check(300_000, n_lo=1.0, n_hi=1.335)
        assert out["allowed_lo"] is False
        assert out["allowed_hi"] is True
        assert "Borderline" in out["verdict"]

    def test_custom_refractive_index_range_is_respected(self):
        out_default = cherenkov_feasibility_check(300_000)
        out_custom = cherenkov_feasibility_check(300_000, n_lo=1.0, n_hi=1.1)
        assert out_custom["n_lo"] == 1.0
        assert out_custom["n_hi"] == 1.1
        assert out_custom["allowed_lo"] is False
        assert out_custom["allowed_hi"] is False
        assert out_default["n_threshold"] == out_custom["n_threshold"]  # voltage-only quantity
