"""Small, valid ONNX graphs keep inference tests independent of release weights."""

from __future__ import annotations

from itertools import count

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from PIL import Image


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch):
    def unexpected_download(*args, **kwargs):
        raise AssertionError("Tests must not make network requests.")

    monkeypatch.setattr("aesthetic_detector.download.urlopen", unexpected_download)


@pytest.fixture
def portrait():
    return Image.new("RGB", (300, 400), (80, 120, 160))


@pytest.fixture
def make_model(tmp_path):
    sequence = count()

    def create(
        values,
        *,
        kind="binary",
        input_name=None,
        input_type=TensorProto.FLOAT,
        input_shape=(1, 3, 224, 224),
        output_name=None,
    ):
        input_name = input_name or ("pixel_values" if kind == "binary" else "input")
        output_name = output_name or ("logits" if kind == "binary" else "score")
        values = np.asarray(values, dtype=np.float32)
        node = helper.make_node(
            "Constant", [], [output_name], value=numpy_helper.from_array(values)
        )
        graph = helper.make_graph(
            [node],
            "inference_contract_fixture",
            [helper.make_tensor_value_info(input_name, input_type, list(input_shape))],
            [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, list(values.shape))],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 10
        onnx.checker.check_model(model)
        path = tmp_path / f"fixture_{next(sequence)}.onnx"
        onnx.save(model, path)
        return path

    return create
