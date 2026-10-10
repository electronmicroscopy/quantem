import numpy as np
import pytest

from quantem.core.io import file_readers
from quantem.core.io.file_readers import _resolve_rsciio_plugin, read_4dstem


class TestResolveRsciioPlugin:
    def test_plugin_name(self):
        assert _resolve_rsciio_plugin("x.dat", "digitalmicrograph") == "rsciio.digitalmicrograph"
        assert _resolve_rsciio_plugin("x.dat", "DigitalMicrograph") == "rsciio.digitalmicrograph"

    def test_unique_extension(self):
        assert _resolve_rsciio_plugin("scan.dm4") == "rsciio.digitalmicrograph"
        assert _resolve_rsciio_plugin("scan.DM3") == "rsciio.digitalmicrograph"
        assert _resolve_rsciio_plugin("scan.mib") == "rsciio.quantumdetector"
        assert _resolve_rsciio_plugin("x.dat", file_type="mib") == "rsciio.quantumdetector"

    def test_extension_equal_to_plugin_name_wins(self):
        # "mrc" is listed by both the MRC and MRCZ plugins
        assert _resolve_rsciio_plugin("stack.mrc") == "rsciio.mrc"

    def test_ambiguous_extension_raises(self):
        with pytest.raises(ValueError, match="file_type") as err:
            _resolve_rsciio_plugin("data.h5")
        assert "arina" in str(err.value)

    def test_unknown_or_missing_type_raises(self):
        with pytest.raises(ValueError, match="No RosettaSciIO reader"):
            _resolve_rsciio_plugin("data.notaformat")
        with pytest.raises(ValueError, match="Cannot infer"):
            _resolve_rsciio_plugin("data_without_extension")


def _fake_reader(monkeypatch, entries):
    def file_reader(path, **kwargs):
        return entries

    monkeypatch.setattr(
        file_readers, "_rsciio_reader", lambda path, file_type=None: ("rsciio.fake", file_reader)
    )


def _axis(scale, offset, units, name):
    return {"scale": scale, "offset": offset, "units": units, "name": name}


@pytest.mark.parametrize("scan_axis", [0, 1])
def test_read_4dstem_reshapes_3d_stack(monkeypatch, scan_axis):
    n_frames, ny, nx, scan_length = 12, 5, 7, 4
    frames = np.arange(n_frames * ny * nx, dtype=np.float32).reshape(n_frames, ny, nx)
    det_y = _axis(0.1, -0.25, "1/nm", "ky")
    det_x = _axis(0.2, -0.7, "1/nm", "kx")
    scan = _axis(1.0, 0.0, "1", "frame")
    if scan_axis == 0:
        data, axes = frames, [scan, det_y, det_x]
    else:
        data, axes = np.moveaxis(frames, 0, 1), [det_y, scan, det_x]
    _fake_reader(monkeypatch, [{"data": data, "axes": axes}])

    ds = read_4dstem("stack.fake", scan_length=scan_length, scan_axis=scan_axis)

    assert ds.shape == (n_frames // scan_length, scan_length, ny, nx)
    assert np.array_equal(ds.array[1, 2], frames[1 * scan_length + 2])
    assert np.allclose(ds.sampling, [1.0, 1.0, 0.1, 0.2])
    assert np.allclose(ds.origin, [0.0, 0.0, -0.25, -0.7])
    assert list(ds.units[2:]) == ["1/nm", "1/nm"]


def test_read_4dstem_transpose_scan_axes(monkeypatch):
    frames = np.random.default_rng(0).random((6, 3, 3))
    axes = [_axis(1.0, 0.0, "1", "f"), _axis(1.0, 0.0, "1", "y"), _axis(1.0, 0.0, "1", "x")]
    _fake_reader(monkeypatch, [{"data": frames, "axes": axes}])
    ds = read_4dstem("stack.fake", scan_length=3, transpose_scan_axes=True)
    assert ds.shape == (3, 2, 3, 3)
    assert np.array_equal(ds.array[2, 1], frames[1 * 3 + 2])


def test_read_4dstem_3d_needs_scan_length(monkeypatch):
    axes = [_axis(1.0, 0.0, "1", n) for n in "fyx"]
    _fake_reader(monkeypatch, [{"data": np.zeros((4, 3, 3)), "axes": axes}])
    with pytest.raises(ValueError, match="scan_length"):
        read_4dstem("stack.fake")
    with pytest.raises(ValueError, match="divisible"):
        read_4dstem("stack.fake", scan_length=3)


def test_read_4dstem_rejects_bad_scan_axis(monkeypatch):
    axes = [_axis(1.0, 0.0, "1", n) for n in "fyx"]
    _fake_reader(monkeypatch, [{"data": np.zeros((4, 3, 3)), "axes": axes}])
    for scan_axis in (2, -1):
        with pytest.raises(ValueError, match="scan_axis must be 0 or 1"):
            read_4dstem("stack.fake", scan_length=2, scan_axis=scan_axis)
