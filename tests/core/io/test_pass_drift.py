"""Tests for the multi-pass drift measurement/correction primitives in
``quantem.core.io.file_readers``: ``estimate_pass_shifts`` (FFT subpixel
cross-correlation against a fixed reference), ``apply_pass_shifts``
(per-pass rigid-shift correction), ``suggest_drift_frames_to_drop``
(flagging unsettled leading/trailing passes before combining),
``crop_alignment_border`` (cropping the zero-filled border those shifts
leave, with origin update), and ``suggest_pass_range_for_analysis``
(combining a drift suggestion with a dose series into one recommended
pass range).

``crop_alignment_border`` is split into a pure margin-computation helper
(``_compute_crop_margins`` -- plain arrays/numbers in, a ``(top, bottom,
left, right)`` tuple out) and a thin ``StemEelsRaw``-aware wrapper that
calls it and then performs the actual array crop via ``Dataset.crop()``.
Both the pure helper and the full wrapper (built from a real
``Dataset3deels``/``StemEelsRaw``, fed the kind of shifts
``apply_pass_shifts`` produces) are tested below.

``suggest_pass_range_for_analysis`` already takes only a
``DriftFrameSuggestion`` dataclass (plain list/tuple/float fields) and a
list of plain dicts -- no array-pulling wrapper was needed, so it's tested
directly.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from scipy.ndimage import shift as ndi_shift

from quantem.core.io.file_readers import (
    DriftFrameSuggestion,
    StemEelsRaw,
    _compute_crop_margins,
    apply_pass_shifts,
    crop_alignment_border,
    estimate_pass_shifts,
    suggest_drift_frames_to_drop,
    suggest_pass_range_for_analysis,
)
from quantem.spectroscopy.dataset3deels import Dataset3deels


def _blob_image(size=64, cy=32.0, cx=32.0, sigma=4.0):
    y, x = np.mgrid[0:size, 0:size].astype(float)
    return 100.0 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2)) + 5.0


def _shifted_stack(true_shifts):
    """Build a stack of frames, each the base blob shifted by -dy,-dx (so
    that the ground-truth correction needed to re-align each frame back to
    the reference is +dy,+dx -- matching what estimate_pass_shifts should
    report, since it measures the shift of each frame relative to the mean)."""
    base = _blob_image()
    frames = []
    for dy, dx in true_shifts:
        frames.append(ndi_shift(base, shift=[-dy, -dx], order=1, mode="constant", cval=0.0))
    return np.stack(frames, axis=0), base


class TestEstimatePassShifts:
    def test_recovers_known_injected_shifts(self):
        true_shifts = np.array([[0.0, 0.0], [1.5, -2.0], [-0.8, 3.1], [2.2, 0.4]])
        stack, _ = _shifted_stack(true_shifts)
        measured = estimate_pass_shifts(stack, reference="mean")
        assert measured.shape == true_shifts.shape
        # reference="mean" measures each frame's shift relative to the
        # average of all frames, not relative to an absolute zero -- so
        # compare mean-subtracted true shifts, which cancels the reference
        # offset and isolates the pairwise drift pattern this is meant to
        # recover.
        true_relative = true_shifts - true_shifts.mean(axis=0)
        np.testing.assert_allclose(measured, true_relative, atol=0.15)

    def test_zero_shift_for_identical_frames(self):
        base = _blob_image()
        stack = np.stack([base, base, base], axis=0)
        measured = estimate_pass_shifts(stack, reference="mean")
        np.testing.assert_allclose(measured, np.zeros((3, 2)), atol=1e-6)

    def test_passes_subset_selects_frames(self):
        true_shifts = np.array([[0.0, 0.0], [1.5, -2.0], [-0.8, 3.1], [2.2, 0.4]])
        stack, _ = _shifted_stack(true_shifts)
        measured_subset = estimate_pass_shifts(stack, passes=[1, 2], reference="mean")
        assert measured_subset.shape == (2, 2)


class TestApplyPassShifts:
    def test_shift_then_inverse_shift_recovers_original(self):
        base = _blob_image()
        dy, dx = 2.0, -1.5
        shifted = ndi_shift(base, shift=[dy, dx], order=1, mode="constant", cval=0.0)
        stack = shifted[None, ...]  # (1, ny, nx)
        corrected = apply_pass_shifts(stack, shifts=np.array([[-dy, -dx]]), order=1)
        # interior (away from the zero-filled border) should match the original blob closely
        interior = (slice(10, -10), slice(10, -10))
        np.testing.assert_allclose(corrected[0][interior], base[interior], atol=2.0)

    def test_output_shape_matches_input(self):
        stack = np.random.default_rng(0).random((3, 20, 20, 20))  # (pass, energy, ny, nx)
        shifts = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]])
        out = apply_pass_shifts(stack, shifts)
        assert out.shape == stack.shape

    def test_energy_axis_untouched_for_4d_stack(self):
        # a stack with a middle (energy) axis must only shift the trailing
        # two (spatial) axes, not the energy axis
        n_energy = 5
        base = _blob_image(size=32)
        stack = np.stack([base] * n_energy, axis=0)[None, ...]  # (1 pass, n_energy, ny, nx)
        shifts = np.array([[1.0, 0.0]])
        out = apply_pass_shifts(stack, shifts, order=1)
        assert out.shape == (1, n_energy, 32, 32)
        # every energy slice shifted identically (same spatial shift applied to all)
        for e in range(1, n_energy):
            np.testing.assert_allclose(out[0, 0], out[0, e])

    def test_shift_mismatch_length_raises(self):
        stack = np.zeros((3, 10, 10))
        with pytest.raises(ValueError, match="shifts must have shape"):
            apply_pass_shifts(stack, shifts=np.zeros((2, 2)))

    def test_unlisted_passes_copied_through_unshifted(self):
        base = _blob_image(size=32)
        other = _blob_image(size=32, cy=10.0, cx=10.0)
        stack = np.stack([base, other], axis=0)
        # only shift pass 0; pass 1 isn't in `passes` so should pass through unchanged
        out = apply_pass_shifts(stack, shifts=np.array([[0.0, 0.0]]), passes=[0], order=1)
        np.testing.assert_allclose(out[1], other)


class TestSuggestDriftFramesToDrop:
    def test_flags_settling_leading_run(self):
        # large shifts at the start that settle to a small plateau -- classic
        # beam-settling/thermal-drift signature at the start of an acquisition
        shifts = np.array(
            [
                [5.0, 4.0],
                [3.0, 2.5],
                [1.5, 1.2],
                [0.1, 0.05],
                [0.08, 0.1],
                [0.1, 0.07],
                [0.05, 0.1],
            ]
        )
        suggestion, _ = suggest_drift_frames_to_drop(shifts, show=False)
        assert 0 in suggestion.drop
        assert suggestion.leading_run is not None
        # the settled plateau passes should be kept
        assert len(suggestion.keep) >= 2

    def test_flat_stack_suggests_no_drops(self):
        rng = np.random.default_rng(1)
        shifts = rng.normal(0, 0.05, size=(10, 2))  # tiny noise, no real drift
        suggestion, _ = suggest_drift_frames_to_drop(shifts, show=False)
        assert suggestion.drop == []

    def test_min_keep_is_respected(self):
        # even a pathologically drifting series must keep at least min_keep passes
        shifts = np.array([[i * 2.0, i * 2.0] for i in range(6)])  # monotonic runaway drift
        suggestion, _ = suggest_drift_frames_to_drop(shifts, min_keep=3, show=False)
        assert len(suggestion.keep) >= 3

    def test_passes_used_labels_are_respected(self):
        shifts = np.array([[5.0, 4.0], [0.1, 0.1], [0.08, 0.1], [0.1, 0.07]])
        passes_used = [10, 11, 12, 13]  # non-default, non-0-indexed-from-0 labeling
        suggestion, _ = suggest_drift_frames_to_drop(shifts, passes_used=passes_used, show=False)
        # drop/keep still reported as 0-indexed positions into `shifts`, not into passes_used
        assert all(0 <= i < len(shifts) for i in suggestion.drop + suggestion.keep)


class TestComputeCropMargins:
    """Pure-function tests for the margin arithmetic behind
    ``crop_alignment_border`` -- the part actually at risk of a sign or
    off-by-one bug (every downstream pixel coordinate depends on it)."""

    def test_known_mixed_sign_shifts(self):
        # dy in [-0.8, 2.2] -> top=ceil(2.2)=3, bottom=ceil(0.8)=1
        # dx in [-2.0, 3.1] -> left=ceil(3.1)=4, right=ceil(2.0)=2
        shifts = np.array([[0.0, 0.0], [2.2, -2.0], [-0.8, 3.1]])
        top, bottom, left, right = _compute_crop_margins(shifts, ny=50, nx=50)
        assert (top, bottom, left, right) == (3, 1, 4, 2)

    def test_zero_shifts_give_zero_margins(self):
        shifts = np.zeros((4, 2))
        top, bottom, left, right = _compute_crop_margins(shifts, ny=50, nx=50)
        assert (top, bottom, left, right) == (0, 0, 0, 0)

    def test_extra_margin_is_added_on_every_side(self):
        shifts = np.zeros((2, 2))
        top, bottom, left, right = _compute_crop_margins(shifts, ny=50, nx=50, extra_margin_px=3)
        assert (top, bottom, left, right) == (3, 3, 3, 3)

    def test_one_sided_shift_only_margins_that_side(self):
        # all shifts positive in dy -> only a top margin, bottom stays 0
        shifts = np.array([[1.0, 0.0], [2.0, 0.0]])
        top, bottom, left, right = _compute_crop_margins(shifts, ny=50, nx=50)
        assert top == 2
        assert bottom == 0

    def test_shift_larger_than_field_raises(self):
        shifts = np.array([[60.0, 60.0]])
        with pytest.raises(ValueError, match="would leave no pixels"):
            _compute_crop_margins(shifts, ny=50, nx=50)


def _make_stem_eels_raw(array_ll, array_hl, adf, pass_shifts_px):
    """Minimal but real StemEelsRaw: Dataset3deels.from_array() for
    eels_ll/eels_hl (so crop_alignment_border exercises the actual
    Dataset.crop() origin-update logic, not a mock), a plain ndarray adf,
    and the kind of pass_shifts_px apply_pass_shifts/estimate_pass_shifts
    produce."""
    ll = Dataset3deels.from_array(
        array=array_ll, origin=[0, 0, 0], sampling=[1, 1, 1], units=["px", "px", "eV"]
    )
    hl = Dataset3deels.from_array(
        array=array_hl, origin=[0, 0, 0], sampling=[1, 1, 1], units=["px", "px", "eV"]
    )
    return StemEelsRaw(
        folder=Path("/tmp/synthetic"),
        dm4_path=Path("/tmp/synthetic/STEM SI.dm4"),
        is_multipass=True,
        n_passes=len(pass_shifts_px),
        eels_ll=ll,
        eels_hl=hl,
        adf=adf,
        energy_axis_ll=None,
        energy_axis_hl=None,
        pixel_size_nm=None,
        passes_used=list(range(1, len(pass_shifts_px) + 1)),
        combine_method="sum",
        pass_shifts_px=np.asarray(pass_shifts_px, dtype=float),
    )


class TestCropAlignmentBorder:
    def test_crops_and_updates_origin_for_known_shifts(self):
        ny, nx, n_energy = 30, 30, 5
        array = np.random.default_rng(0).random((ny, nx, n_energy))
        adf = np.random.default_rng(1).random((ny, nx))
        # dy in [0.0, 2.0] -> top=2, bottom=0; dx in [-1.0, 3.0] -> left=3, right=1
        shifts = np.array([[2.0, -1.0], [0.0, 3.0]])
        raw = _make_stem_eels_raw(array, array, adf, shifts)

        out = crop_alignment_border(raw)

        assert out.eels_ll.shape == (ny - 2 - 0, nx - 3 - 1, n_energy)
        assert out.eels_hl.shape == (ny - 2 - 0, nx - 3 - 1, n_energy)
        assert out.adf.shape == (ny - 2 - 0, nx - 3 - 1)
        # origin updated by crop_start * sampling (sampling=1 here, so
        # origin shifts by exactly the top/left margin)
        np.testing.assert_allclose(out.eels_ll.origin[:2], [2, 3])
        np.testing.assert_allclose(out.eels_hl.origin[:2], [2, 3])
        # cropped content matches the corresponding slice of the original
        np.testing.assert_allclose(out.eels_ll.array, array[2:30, 3:29, :])
        np.testing.assert_allclose(out.adf, adf[2:30, 3:29])
        # input untouched (modify_in_place=False default)
        assert raw.eels_ll.shape == (ny, nx, n_energy)

    def test_zero_shifts_crop_nothing(self):
        ny, nx, n_energy = 20, 20, 4
        array = np.random.default_rng(2).random((ny, nx, n_energy))
        adf = np.random.default_rng(3).random((ny, nx))
        shifts = np.zeros((3, 2))
        raw = _make_stem_eels_raw(array, array, adf, shifts)

        out = crop_alignment_border(raw)

        assert out.eels_ll.shape == (ny, nx, n_energy)
        np.testing.assert_allclose(out.eels_ll.origin, [0, 0, 0])
        np.testing.assert_allclose(out.eels_ll.array, array)

    def test_explicit_shifts_used_overrides_attribute(self):
        ny, nx, n_energy = 20, 20, 4
        array = np.random.default_rng(4).random((ny, nx, n_energy))
        adf = np.random.default_rng(5).random((ny, nx))
        # object carries zero shifts, but an explicit shifts_used is passed in
        raw = _make_stem_eels_raw(array, array, adf, np.zeros((2, 2)))
        explicit_shifts = np.array([[1.0, 1.0], [2.0, 0.0]])

        out = crop_alignment_border(raw, shifts_used=explicit_shifts)

        assert out.eels_ll.shape == (ny - 2, nx - 1, n_energy)

    def test_no_shifts_at_all_raises(self):
        ny, nx, n_energy = 20, 20, 4
        array = np.zeros((ny, nx, n_energy))
        adf = np.zeros((ny, nx))
        raw = _make_stem_eels_raw(array, array, adf, np.zeros((0, 2)))
        raw.pass_shifts_px = None
        with pytest.raises(ValueError, match="no pass_shifts_px"):
            crop_alignment_border(raw)

    def test_shift_larger_than_field_raises(self):
        ny, nx, n_energy = 20, 20, 4
        array = np.zeros((ny, nx, n_energy))
        adf = np.zeros((ny, nx))
        shifts = np.array([[25.0, 25.0]])  # larger than the 20x20 field
        raw = _make_stem_eels_raw(array, array, adf, shifts)
        with pytest.raises(ValueError, match="would leave no pixels"):
            crop_alignment_border(raw)

    def test_modify_in_place_mutates_and_returns_same_object(self):
        ny, nx, n_energy = 20, 20, 4
        array = np.random.default_rng(6).random((ny, nx, n_energy))
        adf = np.random.default_rng(7).random((ny, nx))
        shifts = np.array([[1.0, 1.0], [0.0, 0.0]])
        raw = _make_stem_eels_raw(array, array, adf, shifts)

        out = crop_alignment_border(raw, modify_in_place=True)

        assert out is raw
        assert raw.eels_ll.shape == (ny - 1, nx - 1, n_energy)


class TestSuggestPassRangeForAnalysis:
    """suggest_pass_range_for_analysis already takes only a
    DriftFrameSuggestion dataclass and a list of plain dicts -- both are
    trivial to construct directly, so no array-pulling wrapper/pure-split
    was needed here; this tests the function as-is."""

    def _drift_suggestion(self, keep):
        return DriftFrameSuggestion(
            drop=[],
            keep=list(keep),
            leading_run=None,
            trailing_run=None,
            mid_outliers=[],
            mid_outliers_dropped=[],
            plateau_magnitude_px=0.1,
            threshold_px=0.5,
            reason="synthetic fixture",
        )

    def test_recommends_the_known_settled_range(self):
        # 8 dose blocks: a flat "settled" plateau for the first 4, then a
        # clear, large, sustained jump for the last 4 -- the dose-damage
        # signature this function is meant to catch.
        values = [1.00, 1.02, 1.01, 1.00, 1.50, 1.60, 1.55, 1.65]
        dose_block_results = [{"ratio": v} for v in values]
        # 16 passes spread evenly over the 8 dose blocks (2 passes/block)
        drift_suggestion = self._drift_suggestion(keep=range(16))

        result = suggest_pass_range_for_analysis(drift_suggestion, dose_block_results)

        assert result["damage_detected"] is True
        assert result["direction"] == "increasing"
        assert result["cutoff_block"] == 4
        # cutoff at dose block 4 of 8 -> first half (8 of 16) of the kept passes
        assert result["recommended_passes"] == list(range(8))

    def test_no_trend_keeps_every_drift_surviving_pass(self):
        rng = np.random.default_rng(0)
        values = 1.0 + rng.normal(0, 0.01, size=8)  # flat, noisy, no real trend
        dose_block_results = [{"ratio": v} for v in values]
        drift_suggestion = self._drift_suggestion(keep=range(10))

        result = suggest_pass_range_for_analysis(drift_suggestion, dose_block_results)

        assert result["damage_detected"] is False
        assert result["recommended_passes"] == list(range(10))

    def test_too_few_blocks_keeps_every_pass(self):
        dose_block_results = [{"ratio": 1.0}, {"ratio": 1.1}]  # only 2 blocks
        drift_suggestion = self._drift_suggestion(keep=range(5))

        result = suggest_pass_range_for_analysis(drift_suggestion, dose_block_results)

        assert result["damage_detected"] is False
        assert result["recommended_passes"] == list(range(5))
        assert "too few" in result["reason"]

    def test_significant_but_tiny_range_is_not_flagged(self):
        # a real, significant monotonic trend (Spearman will find it) but
        # with a peak-to-peak range well under the 5% relative-range gate
        values = [1.000, 1.002, 1.004, 1.006, 1.008, 1.010, 1.012, 1.014]
        dose_block_results = [{"ratio": v} for v in values]
        drift_suggestion = self._drift_suggestion(keep=range(8))

        result = suggest_pass_range_for_analysis(drift_suggestion, dose_block_results)

        assert result["damage_detected"] is False
        assert result["recommended_passes"] == list(range(8))
        assert "too small" in result["reason"]

    def test_custom_metric_key_is_used(self):
        values = [1.0, 1.0, 1.0, 1.0, 0.5, 0.4, 0.45, 0.4]  # decreasing -> knock-on signature
        dose_block_results = [{"t_mean": v, "ratio": 999.0} for v in values]
        drift_suggestion = self._drift_suggestion(keep=range(8))

        result = suggest_pass_range_for_analysis(
            drift_suggestion, dose_block_results, metric_key="t_mean"
        )

        assert result["damage_detected"] is True
        assert result["direction"] == "decreasing"
