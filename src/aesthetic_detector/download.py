"""Versioned model downloads with integrity checks and atomic cache writes."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from importlib.resources import files
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def manifest() -> dict:
    return json.loads(files("aesthetic_detector").joinpath("model_manifest.json").read_text())


def default_cache_dir() -> Path:
    override = os.environ.get("AESTHETIC_DETECTOR_CACHE")
    if override:
        return Path(override).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "aesthetic-detector"


def _matches(path: Path, spec: dict) -> bool:
    if not path.is_file() or path.stat().st_size != spec["bytes"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == spec["sha256"]


def get_model(name: str, cache_dir: str | Path | None = None) -> Path:
    """Return a verified cached model, downloading it if absent or damaged."""
    catalog = manifest()
    if name not in catalog["models"]:
        raise ValueError(f"Unknown model {name!r}; choose binary or regional.")
    spec = catalog["models"][name]
    folder = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    folder = folder / catalog["version"]
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / spec["filename"]
    if _matches(destination, spec):
        return destination

    if not spec["url"].startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS model URL for {spec['filename']}.")
    logger.info("Downloading %s (%.1f MiB)", spec["filename"], spec["bytes"] / 1024**2)
    temporary: Path | None = None
    try:
        request = Request(spec["url"], headers={"User-Agent": "aesthetic-detector/1.0.0"})
        # URL comes from the packaged manifest and is HTTPS-only; payload is size/SHA-256 checked.
        with urlopen(request, timeout=60) as response:  # nosec B310
            with tempfile.NamedTemporaryFile(dir=folder, suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                count = 0
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > spec["bytes"]:
                        raise ValueError("Downloaded model exceeds its declared size.")
                    output.write(chunk)
        if not _matches(temporary, spec):
            raise ValueError("Model download failed SHA-256 or file-size verification.")
        os.replace(temporary, destination)
    except (URLError, TimeoutError) as error:
        raise RuntimeError(
            f"Could not download {spec['filename']}. Check your connection or supply "
            "a local model_path / --model-path."
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    logger.info("Model cached at %s", destination)
    return destination
