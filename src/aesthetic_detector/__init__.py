"""Public API for image classification and regional ONNX inference."""

from .models import AestheticDetector, Prediction, RegionalPrediction, RegionalScorer

__all__ = ["AestheticDetector", "Prediction", "RegionalPrediction", "RegionalScorer"]
__version__ = "1.0.0"
