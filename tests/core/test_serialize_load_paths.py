"""load() finds a saved object one folder down and never writes while loading."""

import os
import shutil
from zipfile import ZipFile

import numpy as np
import pytest

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.io.serialize import load


@pytest.fixture
def saved(tmp_path):
    obj = Dataset2d.from_array(np.arange(12.0).reshape(3, 4), name="df")
    path = tmp_path / "saved" / "IM_DF"
    obj.save(str(path), mode="o")
    return path


def _check(obj):
    assert isinstance(obj, Dataset2d)
    assert np.array_equal(np.asarray(obj.array), np.arange(12.0).reshape(3, 4))


def test_load_direct_and_one_level_down(saved):
    _check(load(saved))
    _check(load(saved.parent))  # e.g. a folder dragged into a folder


def test_load_zip_with_wrapper_folder(saved, tmp_path):
    # an operating-system zip wraps the contents in a folder
    zpath = tmp_path / "wrapped.zip"
    with ZipFile(zpath, "w") as zf:
        for root, _, files in os.walk(saved):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, saved.parent))
    _check(load(zpath))


def test_load_ignores_stray_root_metadata(saved):
    # an earlier load wrote an empty zarr.json into the parent folder
    (saved.parent / "zarr.json").write_text(
        '{"attributes": {}, "zarr_format": 3, "node_type": "group"}'
    )
    _check(load(saved.parent))


def test_load_nothing_there_raises_and_writes_nothing(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No saved quantem object"):
        load(empty)
    assert os.listdir(empty) == []
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing")


def test_load_two_objects_is_ambiguous(saved):
    shutil.copytree(saved, saved.parent / "IM_DF_copy")
    with pytest.raises(FileNotFoundError, match="2 saved objects"):
        load(saved.parent)
