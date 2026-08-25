from __future__ import annotations

import hashlib
import io

import pytest

from quantem.diffraction.polymer_models import PolymerModelError, resolve_polymer_model


def _registry(payload: bytes, *, architecture=None):
    checksum = hashlib.sha256(payload).hexdigest()
    spec = {
        "schema_version": 1,
        "model_id": "test-model",
        "architecture": architecture
        or {"input_channels": 1, "output_channels": 2, "kernel_size": 3},
        "weights": {"filename": "weights.pth", "sha256": checksum},
    }
    return {"test-model": {"v1": {"specification": spec, "weights_url": "mock://weights"}}}


def test_download_then_offline_cache_hit(tmp_path):
    payload = b"immutable model weights"
    calls = []

    def download(url):
        calls.append(url)
        return io.BytesIO(payload)

    first = resolve_polymer_model(
        "test-model", version="v1", cache_dir=tmp_path, registry=_registry(payload), downloader=download
    )
    assert first.weights_path.read_bytes() == payload
    assert first.weights_path.parent.name == first.checksum
    second = resolve_polymer_model(
        "test-model", version="v1", cache_dir=tmp_path, registry=_registry(payload),
        downloader=lambda _: pytest.fail("offline cache hit attempted a download"),
    )
    assert second == first
    assert calls == ["mock://weights"]


def test_local_override_uses_verified_weights(tmp_path):
    payload = b"private weights"
    model_dir = tmp_path / "test-model" / "v1"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.pth").write_bytes(payload)
    result = resolve_polymer_model(
        "test-model", version="v1", local_model_dir=tmp_path, registry=_registry(payload)
    )
    assert result.weights_path == (model_dir / "weights.pth").resolve()


def test_checksum_failure_leaves_no_partial_file(tmp_path):
    with pytest.raises(PolymerModelError, match="Checksum failure after downloading"):
        resolve_polymer_model(
            "test-model", version="v1", cache_dir=tmp_path, registry=_registry(b"expected"),
            downloader=lambda _: io.BytesIO(b"corrupt"),
        )
    assert not list(tmp_path.rglob("weights.pth"))
    assert not list(tmp_path.rglob(".download-*"))


def test_interrupted_download_leaves_no_partial_file(tmp_path):
    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            raise OSError("connection interrupted")

    with pytest.raises(PolymerModelError, match="Could not download"):
        resolve_polymer_model(
            "test-model", version="v1", cache_dir=tmp_path, registry=_registry(b"expected"),
            downloader=lambda _: Interrupted(b"partial"),
        )
    assert not list(tmp_path.rglob(".download-*"))


def test_incompatible_specification_is_rejected(tmp_path):
    with pytest.raises(PolymerModelError, match="incompatible"):
        resolve_polymer_model(
            "test-model", version="v1", cache_dir=tmp_path,
            registry=_registry(b"weights", architecture={"input_channels": 3, "output_channels": 2, "kernel_size": 3}),
        )


def test_latest_is_explicit_and_exclusive(tmp_path):
    payload = b"weights"
    versions = _registry(payload)["test-model"]
    versions["v2"] = versions["v1"]
    result = resolve_polymer_model(
        "test-model", latest=True, cache_dir=tmp_path, registry={"test-model": versions},
        downloader=lambda _: io.BytesIO(payload),
    )
    assert result.version == "v2"
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_polymer_model("test-model", version="v1", latest=True, registry={"test-model": versions})
