"""CPU inference interfaces for the released binary and regional models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .download import get_model, manifest
from .preprocessing import ImageSource, prepare_portrait, prepare_region


@dataclass(frozen=True)
class Prediction:
    label: str
    intervention_score: float
    natural_score: float
    threshold: float
    model_version: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RegionalPrediction:
    score: float
    model_version: str

    def to_dict(self) -> dict:
        return asdict(self)


class _OnnxModel:
    def __init__(self, name: str, model_path: str | Path | None, cache_dir: str | Path | None):
        import onnxruntime as ort

        catalog = manifest()
        self.spec = catalog["models"][name]
        self.model_version = "custom" if model_path is not None else catalog["version"]
        self.model_path = (
            Path(model_path).expanduser() if model_path else get_model(name, cache_dir)
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        try:
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise ValueError(
                "Could not load the ONNX model. Verify the file and ONNX Runtime compatibility."
            ) from error
        inputs = self._session.get_inputs()
        if (
            len(inputs) != 1
            or inputs[0].name != self.spec["input_name"]
            or inputs[0].type != "tensor(float)"
            or inputs[0].shape != [1, 3, 224, 224]
        ):
            raise ValueError("Model must expose the documented float32 [1, 3, 224, 224] input.")
        if self.spec["output_name"] not in {output.name for output in self._session.get_outputs()}:
            raise ValueError(f"Model is missing output {self.spec['output_name']!r}.")

    def _run(self, tensor: np.ndarray) -> np.ndarray:
        try:
            values = self._session.run(
                [self.spec["output_name"]], {self.spec["input_name"]: tensor}
            )[0]
        except Exception as error:
            raise RuntimeError(
                "ONNX inference failed; verify model input/output compatibility."
            ) from error
        if not np.isfinite(values).all():
            raise ValueError("Model returned non-finite values; no prediction was produced.")
        return np.asarray(values, dtype=np.float64)


class AestheticDetector(_OnnxModel):
    """Classify a close, frontal portrait with the fine-tuned binary model.

    Scores are model outputs, not calibrated probabilities of a person's history.
    This interface does not detect faces or validate portrait composition.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        threshold: float = 0.52,
    ):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a finite number between 0 and 1.")
        self.threshold = float(threshold)
        super().__init__("binary", model_path, cache_dir)

    def predict(self, image: ImageSource) -> Prediction:
        logits = self._run(prepare_portrait(image))
        if logits.shape != (1, 2):
            raise ValueError(f"Expected binary logits shape (1, 2), received {logits.shape}.")
        exp_logits = np.exp(logits[0] - logits[0].max())
        scores = exp_logits / exp_logits.sum()
        intervention = float(scores[1])
        return Prediction(
            label="cosmetic_intervention" if intervention >= self.threshold else "natural",
            intervention_score=intervention,
            natural_score=float(scores[0]),
            threshold=self.threshold,
            model_version=self.model_version,
        )


class RegionalScorer(_OnnxModel):
    """Score a caller-supplied crop of a nose, lips, chin, cheekbone, cheek, or eye."""

    def __init__(self, model_path: str | Path | None = None, cache_dir: str | Path | None = None):
        super().__init__("regional", model_path, cache_dir)

    def predict(self, image: ImageSource) -> RegionalPrediction:
        values = self._run(prepare_region(image))
        if values.shape != (1,):
            raise ValueError(f"Expected regional score shape (1,), received {values.shape}.")
        score = float(values[0])
        if not 0 <= score <= 1:
            raise ValueError("Regional model returned a score outside [0, 1].")
        return RegionalPrediction(score=score, model_version=self.model_version)
