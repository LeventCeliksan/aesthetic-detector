"""Exercise public inference against real, deliberately controlled ONNX outputs."""

import math

import pytest
from onnx import TensorProto

from aesthetic_detector import AestheticDetector, RegionalScorer


@pytest.mark.parametrize(
    ("logits", "expected_label", "expected_intervention"),
    [
        ([[-2.0, 2.0]], "cosmetic_intervention", 1 / (1 + math.exp(-4))),
        ([[2.0, -2.0]], "natural", 1 / (1 + math.exp(4))),
        ([[10000.0, 10002.0]], "cosmetic_intervention", 1 / (1 + math.exp(-2))),
    ],
)
def test_binary_mapping_and_stable_softmax(
    make_model, portrait, logits, expected_label, expected_intervention
):
    result = AestheticDetector(model_path=make_model(logits)).predict(portrait)

    assert result.label == expected_label
    assert result.intervention_score == pytest.approx(expected_intervention)
    assert result.natural_score + result.intervention_score == pytest.approx(1)
    assert result.model_version == "custom"


def test_decision_threshold_is_inclusive_and_does_not_change_scores(make_model, portrait):
    path = make_model([[0.0, 0.0]])
    default = AestheticDetector(model_path=path).predict(portrait)
    boundary = AestheticDetector(model_path=path, threshold=0.5).predict(portrait)

    assert default.threshold == 0.52
    assert default.label == "natural"
    assert boundary.label == "cosmetic_intervention"
    assert boundary.intervention_score == default.intervention_score == 0.5


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_threshold_is_rejected_before_model_download(threshold):
    with pytest.raises(ValueError, match="threshold"):
        AestheticDetector(threshold=threshold)


@pytest.mark.parametrize(
    "contract",
    [
        {"input_name": "unexpected"},
        {"input_type": TensorProto.FLOAT16},
        {"input_shape": (1, 224, 224, 3)},
        {"output_name": "unexpected"},
    ],
)
def test_incompatible_onnx_contract_is_rejected(make_model, contract):
    path = make_model([[0.0, 1.0]], **contract)
    with pytest.raises(ValueError, match="documented|missing output"):
        AestheticDetector(model_path=path)


@pytest.mark.parametrize(
    ("model_class", "kind", "values"),
    [(AestheticDetector, "binary", [0.0, 1.0]), (RegionalScorer, "regional", [[0.8]])],
)
def test_wrong_output_shape_is_rejected(make_model, portrait, model_class, kind, values):
    model = model_class(model_path=make_model(values, kind=kind))
    with pytest.raises(ValueError, match="shape"):
        model.predict(portrait)


@pytest.mark.parametrize(
    ("model_class", "kind", "values"),
    [
        (AestheticDetector, "binary", [[0.0, float("nan")]]),
        (RegionalScorer, "regional", [float("inf")]),
    ],
)
def test_nonfinite_output_never_produces_prediction(
    make_model, portrait, model_class, kind, values
):
    model = model_class(model_path=make_model(values, kind=kind))
    with pytest.raises(ValueError, match="non-finite"):
        model.predict(portrait)


def test_regional_output_is_already_sigmoid_and_is_not_transformed(make_model, portrait):
    model = RegionalScorer(model_path=make_model([0.8], kind="regional"))
    result = model.predict(portrait)

    assert result.score == pytest.approx(0.8)
    assert result.model_version == "custom"


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_out_of_range_regional_score_is_rejected(make_model, portrait, value):
    model = RegionalScorer(model_path=make_model([value], kind="regional"))
    with pytest.raises(ValueError, match="outside"):
        model.predict(portrait)
