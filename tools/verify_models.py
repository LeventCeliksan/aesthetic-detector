"""Run real-model smoke tests; these checks do not measure classification quality."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from PIL import Image
from PIL import __version__ as pillow_version

from aesthetic_detector import AestheticDetector, RegionalScorer


def samples() -> list[Image.Image]:
    rng = np.random.default_rng(42)
    noise = Image.fromarray(rng.integers(0, 256, (320, 240, 3), dtype=np.uint8))
    gradient = np.linspace(0, 255, 400, dtype=np.uint8)
    gradient_image = Image.fromarray(np.tile(gradient[None, :, None], (300, 1, 3)))
    return [Image.new("RGB", (224, 224), color) for color in ("black", "white", "gray")] + [
        noise,
        gradient_image,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("validation/model_smoke.json"))
    args = parser.parse_args()
    images = samples()
    report = {
        "purpose": "Software execution and export-equivalence checks, not an accuracy benchmark.",
        "environment": {
            "system": platform.system(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "onnxruntime": onnxruntime.__version__,
            "onnx": onnx.__version__,
            "numpy": np.__version__,
            "pillow": pillow_version,
        },
        "input_description": "Five deterministic synthetic images; no human photos or labels.",
        "models": [],
    }
    for model_type, filename in (
        (AestheticDetector, "binary_model_fp16.onnx"),
        (RegionalScorer, "regional_model.onnx"),
    ):
        path = args.models / filename
        onnx.checker.check_model(str(path))
        model = model_type(model_path=path)
        results = [model.predict(image).to_dict() for image in images]
        assert results == [model.predict(image).to_dict() for image in images]
        del model
        gc.collect()
        compared = False
        if args.reference_dir:
            original = model_type(model_path=args.reference_dir / filename)
            assert results == [original.predict(image).to_dict() for image in images]
            del original
            gc.collect()
            compared = True
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        report["models"].append(
            {
                "filename": filename,
                "sha256": checksum,
                "sample_count": len(images),
                "onnx_checker": "passed",
                "inference": "passed",
                "repeated_results_identical": True,
                "outputs_identical_to_original_export": compared,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2) + "\n"
    args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
