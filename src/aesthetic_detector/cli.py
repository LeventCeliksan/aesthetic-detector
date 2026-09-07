"""Command-line interface with machine-readable JSON on standard output."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .download import get_model, manifest
from .models import AestheticDetector, RegionalScorer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fine-tuned Aesthetic Detector ONNX models."
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("predict", "Classify a close, frontal portrait."),
        ("score-region", "Score an already-cropped face region."),
    ):
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument("image", help="Path to a local image.")
        sub.add_argument("--model-path", help="Use a local ONNX file without downloading.")
        sub.add_argument("--cache-dir", help="Directory for downloaded model files.")
        if command == "predict":
            sub.add_argument("--threshold", type=float, default=0.52)
    download = commands.add_parser("download", help="Download and verify model weights.")
    download.add_argument("--model", choices=["binary", "regional", "all"], default="binary")
    download.add_argument("--cache-dir")
    commands.add_parser("info", help="Show model versions, checksums, and download URLs.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        if args.command in {"predict", "score-region"} and not Path(args.image).is_file():
            raise FileNotFoundError(f"Image file does not exist: {args.image}")
        if args.command == "info":
            result = manifest()
        elif args.command == "download":
            names = ["binary", "regional"] if args.model == "all" else [args.model]
            result = {name: str(get_model(name, args.cache_dir)) for name in names}
        elif args.command == "predict":
            model = AestheticDetector(args.model_path, args.cache_dir, args.threshold)
            result = model.predict(args.image).to_dict()
        else:
            region_model = RegionalScorer(args.model_path, args.cache_dir)
            result = region_model.predict(args.image).to_dict()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
