"""Verify integrity failures never become usable cached model files."""

import hashlib
import io
from urllib.error import URLError

import pytest

from aesthetic_detector import download


@pytest.fixture
def release(monkeypatch, tmp_path):
    payload = b"verified fixture weights"
    spec = {
        "filename": "fixture.onnx",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": "https://example.invalid/releases/fixture.onnx",
    }
    monkeypatch.setattr(
        download, "manifest", lambda: {"version": "test-version", "models": {"binary": spec}}
    )
    folder = tmp_path / "cache" / "test-version"
    return payload, spec, folder


def test_verified_cache_does_not_attempt_download(release):
    payload, spec, folder = release
    folder.mkdir(parents=True)
    destination = folder / spec["filename"]
    destination.write_bytes(payload)

    assert download.get_model("binary", folder.parent) == destination
    assert destination.read_bytes() == payload


def test_corrupted_cache_is_replaced_only_after_verification(release, monkeypatch):
    payload, spec, folder = release
    folder.mkdir(parents=True)
    destination = folder / spec["filename"]
    corrupted = b"x" * len(payload)
    destination.write_bytes(corrupted)
    requests = []

    def response(request, *, timeout):
        requests.append(request.full_url)
        assert destination.read_bytes() == corrupted
        assert timeout > 0
        return io.BytesIO(payload)

    monkeypatch.setattr(download, "urlopen", response)
    assert download.get_model("binary", folder.parent) == destination
    assert requests == [spec["url"]]
    assert destination.read_bytes() == payload
    assert not list(folder.glob("*.part"))


@pytest.mark.parametrize("damage", ["truncated", "wrong_hash", "oversized"])
def test_bad_download_never_populates_cache_and_removes_temp(release, monkeypatch, damage):
    payload, spec, folder = release
    bad_payload = {
        "truncated": payload[:-1],
        "wrong_hash": b"x" * len(payload),
        "oversized": payload + b"extra",
    }[damage]
    monkeypatch.setattr(download, "urlopen", lambda *args, **kwargs: io.BytesIO(bad_payload))

    with pytest.raises(ValueError, match="verification|declared size"):
        download.get_model("binary", folder.parent)

    assert not (folder / spec["filename"]).exists()
    assert not list(folder.glob("*.part"))


def test_interrupted_download_cleans_temp_and_preserves_existing_file(release, monkeypatch):
    payload, spec, folder = release
    folder.mkdir(parents=True)
    destination = folder / spec["filename"]
    destination.write_bytes(b"old damaged data")

    class InterruptedResponse(io.BytesIO):
        def read(self, size=-1):
            if self.tell() > 0:
                raise URLError("connection interrupted")
            return super().read(3)

    monkeypatch.setattr(download, "urlopen", lambda *args, **kwargs: InterruptedResponse(payload))
    with pytest.raises(RuntimeError, match="local model_path"):
        download.get_model("binary", folder.parent)

    assert destination.read_bytes() == b"old damaged data"
    assert not list(folder.glob("*.part"))


def test_unknown_model_is_rejected_without_creating_cache(release):
    _, _, folder = release

    with pytest.raises(ValueError, match="Unknown model"):
        download.get_model("unlisted", folder.parent)

    assert not folder.exists()
