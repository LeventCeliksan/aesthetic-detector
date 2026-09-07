"""Run with: python examples/predict.py path/to/portrait.jpg"""

import json
import sys

from aesthetic_detector import AestheticDetector

if len(sys.argv) != 2:
    raise SystemExit("Usage: python examples/predict.py path/to/portrait.jpg")

detector = AestheticDetector()
print(json.dumps(detector.predict(sys.argv[1]).to_dict(), indent=2))
