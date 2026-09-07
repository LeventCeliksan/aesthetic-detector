"""Validate release exports and put ONNX nodes into dependency order.

Usage: python tools/prepare_model_assets.py SOURCE_DIRECTORY --output .release
Requires the development dependencies. Original files are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx


def order_nodes(graph: onnx.GraphProto) -> None:
    available = {value.name for value in graph.input}
    available.update(value.name for value in graph.initializer)
    pending = list(graph.node)
    ordered = []
    while pending:
        remaining = []
        for node in pending:
            if all(not name or name in available for name in node.input):
                ordered.append(node)
                available.update(node.output)
            else:
                remaining.append(node)
        if len(remaining) == len(pending):
            raise ValueError("Graph contains unresolved dependencies or a cycle.")
        pending = remaining
    del graph.node[:]
    graph.node.extend(ordered)


def tensor_digest(model: onnx.ModelProto) -> str:
    digest = hashlib.sha256()
    for tensor in model.graph.initializer:
        digest.update(tensor.SerializeToString())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(".release"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for filename in ("binary_model_fp16.onnx", "regional_model.onnx"):
        source = args.source / filename
        destination = args.output / filename
        if source.resolve() == destination.resolve():
            raise ValueError("Output must differ from the original model path.")
        model = onnx.load(source)
        before = tensor_digest(model)
        order_nodes(model.graph)
        onnx.checker.check_model(model)
        onnx.save(model, destination)
        saved = onnx.load(destination)
        assert tensor_digest(saved) == before, "Model tensors changed unexpectedly."
        with source.open("rb") as stream:
            original_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        with destination.open("rb") as stream:
            release_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        reports.append(
            {
                "filename": filename,
                "original_sha256": original_sha256,
                "release_sha256": release_sha256,
                "bytes": destination.stat().st_size,
                "tensors_unchanged": True,
                "onnx_checker": "passed",
            }
        )
    report = json.dumps(reports, indent=2) + "\n"
    (args.output / "export_validation.json").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
