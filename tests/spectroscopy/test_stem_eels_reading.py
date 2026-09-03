"""Tests for the STEM-EELS multipass/single-pass reading capability in
``quantem.core.io.file_readers`` (``read_stem_eels_folder`` and its helpers).

Logic that doesn't need real DM4 files (pass selection/combination, dataset
wrapping, and file discovery) is tested with synthetic data and runs
everywhere. The end-to-end read against real acquisition folders under
``Data/`` is skipped when that folder isn't present (it's local instrument
data, not checked into the repo).
"""

from pathlib import Path

import numpy as np
import pytest

from quantem.core.io.file_readers import (
    StemEelsRaw,
    array_to_spectroscopy3d,
    combine_passes,
    find_stem_si_files,
    read_stem_eels_folder,
    select_passes,
)
from quantem.core.io.serialize import AutoSerialize, load
from quantem.spectroscopy.dataset3deels import Dataset3deels

DATA_DIR = Path(__file__).resolve().parents[2] / "Data"
SAMPLE_FOLDERS = [
    DATA_DIR / "jaden_omiec_",
    DATA_DIR / "jaden_omiec_1",
    DATA_DIR / "jaden_omiec_2",
    DATA_DIR / "jaden_omiec_3",
    DATA_DIR / "jaden_omiec_4",
]
requires_sample_data = pytest.mark.skipif(
    not all(f.exists() for f in SAMPLE_FOLDERS),
    reason="local instrument data under Data/ is not checked into the repo",
)


# --------------------------------------------------------------------------- #
# Pure-logic tests (no file I/O)
# --------------------------------------------------------------------------- #


def test_select_passes_manual_spec():
    assert select_passes(10, mode="manual", passes="1-3,5,8-10") == [0, 1, 2, 4, 7, 8, 9]
    assert select_passes(5, mode="all") == [0, 1, 2, 3, 4]
    assert select_passes(5, mode="manual", passes=[2, 4]) == [1, 3]


def test_select_passes_out_of_range_raises():
    with pytest.raises(ValueError):
        select_passes(5, mode="manual", passes="1-6")


def test_select_passes_manual_without_passes_raises():
    with pytest.raises(ValueError):
        select_passes(5, mode="manual")


def test_combine_passes_sum_and_mean():
    stack = np.ones((4, 3, 2))  # 4 passes of a (3, 2) frame
    stack[1] *= 2  # pass index 1 (pass 2) is worth double

    summed = combine_passes(stack, indices=[0, 1], method="sum")
    np.testing.assert_allclose(summed, np.full((3, 2), 3.0))

    meaned = combine_passes(stack, indices=[0, 1], method="mean")
    np.testing.assert_allclose(meaned, np.full((3, 2), 1.5))


def test_combine_passes_invalid_method_raises():
    with pytest.raises(ValueError):
        combine_passes(np.ones((2, 2, 2)), indices=[0], method="bogus")


def test_array_to_spectroscopy3d_roundtrip():
    n_energy = 50
    energy_axis = np.linspace(-5.0, 10.0, n_energy)
    data = np.random.default_rng(0).normal(size=(6, 7, n_energy))

    ds = array_to_spectroscopy3d(data, energy_axis, pixel_size_nm=2.5, name="test EELS")

    assert isinstance(ds, Dataset3deels)
    assert ds.array.shape == data.shape
    np.testing.assert_allclose(ds.energy_axis, energy_axis, rtol=1e-6)
    assert ds.sampling[0] == pytest.approx(2.5)
    assert ds.units[2] == "eV"


def test_array_to_spectroscopy3d_shape_mismatch_raises():
    with pytest.raises(ValueError):
        array_to_spectroscopy3d(np.zeros((4, 4, 10)), energy_axis=np.linspace(0, 1, 5))


def test_find_stem_si_files_ignores_picker_and_postacq(tmp_path):
    """Regression test: a '(1) Picker of STEM SI.dm4' companion file sorts
    before 'STEM SI.dm4' alphabetically and also matches the '*SI.dm4' glob
    -- find_stem_si_files must not pick it over the real acquisition file."""
    (tmp_path / "(1) Picker of STEM SI.dm4").write_bytes(b"")
    (tmp_path / "STEM SI.dm4").write_bytes(b"")
    (tmp_path / "STEM SI_ADF Image.raw").write_bytes(b"")
    (tmp_path / "STEM SI_EELS HL SI.raw").write_bytes(b"")
    (tmp_path / "STEM SI_EELS LL SI.raw").write_bytes(b"")

    files = find_stem_si_files(tmp_path)

    assert files["dm4"].name == "STEM SI.dm4"
    assert files["adf_raw"].name == "STEM SI_ADF Image.raw"
    assert files["eels_hl_raw"].name == "STEM SI_EELS HL SI.raw"
    assert files["eels_ll_raw"].name == "STEM SI_EELS LL SI.raw"


def test_find_stem_si_files_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_stem_si_files(tmp_path)


# --------------------------------------------------------------------------- #
# End-to-end against real single-pass acquisitions
# --------------------------------------------------------------------------- #


@requires_sample_data
@pytest.mark.parametrize("folder", SAMPLE_FOLDERS, ids=lambda f: f.name)
def test_read_stem_eels_folder_single_pass(folder):
    """Regression test: dataset_index for the HL object was previously taken
    from ncempy's raw ImageList index, which doesn't match rsciio's
    data_list position and raised IndexError on real acquisitions."""
    raw = read_stem_eels_folder(folder)

    assert raw.is_multipass is False
    assert raw.n_passes == 1
    assert isinstance(raw.eels_ll, Dataset3deels)
    assert isinstance(raw.eels_hl, Dataset3deels)
    assert raw.eels_ll.array.ndim == 3
    assert raw.eels_hl.array.ndim == 3
    assert raw.eels_ll.array.shape[:2] == raw.eels_hl.array.shape[:2]


def test_stem_eels_raw_is_autoserialize():
    assert issubclass(StemEelsRaw, AutoSerialize)


@requires_sample_data
def test_stem_eels_raw_save_load_roundtrip(tmp_path):
    """StemEelsRaw bundles two live Dataset3deels objects plus provenance
    (is_multipass, passes_used, ...) -- it should checkpoint and reload as
    one unit rather than requiring eels_ll/eels_hl to be saved separately."""
    raw = read_stem_eels_folder(SAMPLE_FOLDERS[0])

    path = tmp_path / "stem_eels_raw.zip"
    raw.save(path)
    reloaded = load(path)

    assert isinstance(reloaded, StemEelsRaw)
    assert reloaded.is_multipass == raw.is_multipass
    assert reloaded.n_passes == raw.n_passes
    np.testing.assert_allclose(reloaded.eels_ll.array, raw.eels_ll.array)
    np.testing.assert_allclose(reloaded.eels_hl.array, raw.eels_hl.array)
