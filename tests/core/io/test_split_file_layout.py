"""Tests for the split-file dual-EELS layout in ``quantem.core.io.file_readers``
(DigitalMicrograph saving ``EELS LL SI.dm4`` / ``EELS HL SI.dm4`` / ``ADF Image.dm4`` as
separate files instead of one ``STEM SI.dm4``) and for ``crop_unacquired_rows`` (dropping the
all-zero rows an acquisition stopped part way through leaves behind).

File discovery is tested on empty placeholder files -- it only looks at names. Reading real
DM4 content needs instrument data and is not covered here.
"""

from pathlib import Path

import numpy as np
import pytest

from quantem.core.io.file_readers import (
    StemEelsRaw,
    _load_alignment_stack,
    crop_unacquired_rows,
    find_stem_si_files,
)
from quantem.spectroscopy.dataset3deels import Dataset3deels


def _touch(folder: Path, *names: str) -> None:
    for name in names:
        (folder / name).write_bytes(b"")


class TestFindSplitFiles:
    def test_split_layout_is_detected(self, tmp_path):
        _touch(
            tmp_path,
            "EELS LL SI.dm4",
            "EELS HL SI.dm4",
            "ADF Image.dm4",
            "ADF Image (SI Survey).dm4",
        )
        files = find_stem_si_files(tmp_path)
        assert files["dm4_ll"].name == "EELS LL SI.dm4"
        assert files["dm4_hl"].name == "EELS HL SI.dm4"
        assert files["dm4_adf"].name == "ADF Image.dm4"
        assert files["dm4"] == files["dm4_hl"]
        assert files["eels_ll_raw"] is None and files["eels_hl_raw"] is None

    def test_extracted_spectrum_files_are_not_mistaken_for_the_cube(self, tmp_path):
        # "(12) Spectrum of EELS LL SI.dm4" also ends in "EELS LL SI.dm4" and sorts first.
        _touch(tmp_path, "(12) Spectrum of EELS LL SI.dm4", "EELS LL SI.dm4", "EELS HL SI.dm4")
        files = find_stem_si_files(tmp_path)
        assert files["dm4_ll"].name == "EELS LL SI.dm4"
        assert files["dm4_adf"] is None

    def test_one_channel_alone_is_not_the_split_layout(self, tmp_path):
        _touch(tmp_path, "EELS HL SI.dm4")
        assert "dm4_ll" not in find_stem_si_files(tmp_path)

    def test_single_file_layout_is_unchanged(self, tmp_path):
        _touch(
            tmp_path,
            "STEM SI.dm4",
            "STEM SI_EELS LL SI.raw",
            "STEM SI_EELS HL SI.raw",
            "STEM SI_ADF Image.raw",
        )
        files = find_stem_si_files(tmp_path)
        assert set(files) == {"dm4", "adf_raw", "eels_hl_raw", "eels_ll_raw"}
        assert files["dm4"].name == "STEM SI.dm4"
        assert files["eels_ll_raw"].name == "STEM SI_EELS LL SI.raw"

    def test_split_layout_has_no_drift_series(self, tmp_path):
        _touch(tmp_path, "EELS LL SI.dm4", "EELS HL SI.dm4")
        with pytest.raises(ValueError, match="single-pass"):
            _load_alignment_stack(tmp_path)


def _raw(total_per_pixel: np.ndarray, n_energy: int = 8) -> StemEelsRaw:
    """A StemEelsRaw whose LL/HL spectra sum to ``total_per_pixel`` at each scan position."""
    array = np.repeat(total_per_pixel[:, :, None] / n_energy, n_energy, axis=2).astype(float)

    def cube():
        return Dataset3deels.from_array(
            array=array.copy(), origin=[0, 0, 0], sampling=[1, 1, 1], units=["px", "px", "eV"]
        )

    return StemEelsRaw(
        folder=Path("/tmp/synthetic"),
        dm4_path=Path("/tmp/synthetic/EELS HL SI.dm4"),
        is_multipass=False,
        n_passes=1,
        eels_ll=cube(),
        eels_hl=cube(),
        adf=total_per_pixel.copy(),
        energy_axis_ll=None,
        energy_axis_hl=None,
        pixel_size_nm=2.0,
        passes_used=None,
        combine_method=None,
    )


class TestCropUnacquiredRows:
    def test_trailing_empty_rows_and_the_partial_row_are_dropped(self):
        total = np.full((6, 5), 100.0)
        total[3, 2:] = 0.0  # the scan stopped in the middle of row 3
        total[4:] = 0.0
        cropped = crop_unacquired_rows(_raw(total))
        assert cropped.eels_ll.shape[:2] == (3, 5)
        assert cropped.eels_hl.shape[:2] == (3, 5)
        assert cropped.adf.shape == (3, 5)
        assert (np.asarray(cropped.eels_ll.array).sum(axis=-1) > 0).all()
        assert cropped.pixel_size_nm == 2.0

    def test_complete_scan_is_returned_untouched(self):
        raw = _raw(np.full((4, 5), 100.0))
        assert crop_unacquired_rows(raw) is raw

    def test_scan_with_no_complete_row_is_returned_untouched(self):
        total = np.full((4, 5), 100.0)
        total[:, 0] = 0.0
        raw = _raw(total)
        assert crop_unacquired_rows(raw) is raw
