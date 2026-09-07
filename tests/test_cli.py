"""Check machine-readable command output and useful failure reporting."""

import json

import pytest

from aesthetic_detector.cli import main


@pytest.mark.parametrize(
    ("command", "kind", "values", "score_key", "score"),
    [
        ("predict", "binary", [[0, 0]], "intervention_score", 0.5),
        ("score-region", "regional", [0.8], "score", 0.8),
    ],
)
def test_prediction_commands_emit_json(
    tmp_path, portrait, make_model, capsys, command, kind, values, score_key, score
):
    path = tmp_path / "portrait.png"
    portrait.save(path)
    model = make_model(values, kind=kind)

    assert main([command, str(path), "--model-path", str(model)]) == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result[score_key] == pytest.approx(score)
    assert result["model_version"] == "custom"
    assert "Traceback" not in captured.err


def test_invalid_image_returns_nonzero_with_error_on_stderr(make_model, tmp_path, capsys):
    model = make_model([[0, 0]])

    result = main(["predict", str(tmp_path / "missing.png"), "--model-path", str(model)])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err


def test_invalid_model_returns_nonzero_without_traceback(tmp_path, portrait, capsys):
    image = tmp_path / "portrait.png"
    portrait.save(image)
    invalid_model = tmp_path / "invalid.onnx"
    invalid_model.write_bytes(b"This is not an ONNX graph.")

    result = main(["predict", str(image), "--model-path", str(invalid_model)])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err
