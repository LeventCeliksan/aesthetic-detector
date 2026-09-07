# Aesthetic Detector v1.0.0

Initial public release of the trained ONNX models and a standalone Python inference library. Developers can download the weights and run local predictions without repeating training or running the original Android application and backend.

## Included

- Binary portrait classification through `AestheticDetector` and `aesthetic-detector predict`.
- Optional scoring of caller-supplied facial-region crops through `RegionalScorer` and `aesthetic-detector score-region`.
- CPU inference with ONNX Runtime and Python 3.11 or later.
- Versioned model downloads, SHA-256 verification, local caching, and support for local ONNX files.
- A Python wheel, source package, English usage documentation, and a model card.

The binary weights are approximately 164.2 MiB and the regional weights approximately 16.5 MiB. The CLI can download both in advance:

```bash
aesthetic-detector download --model all
```

## Getting started

```bash
python -m pip install "https://github.com/LeventCeliksan/aesthetic-detector/releases/download/v1.0.0/aesthetic_detector-1.0.0-py3-none-any.whl"
aesthetic-detector predict portrait.jpg
```

See the [README](https://github.com/LeventCeliksan/aesthetic-detector/tree/v1.0.0#readme) for Python examples, local-weight loading, and threshold configuration.

## Evaluation and scope

The author reports fine-tuning on 1,000 photographs and a 97% evaluation result. The metric definition, held-out image count, train/test split, and evaluation artifacts are unavailable, and this release has not independently reproduced that result. Python preprocessing is not byte-for-byte equivalent to the original Android pipeline, so no equivalent quality benchmark is claimed for the package.

Provide close, frontal, single-person portraits or prepare regional crops yourself. There is no built-in face detection, automatic cropping, or content detection. Scores are uncalibrated model outputs and do not establish whether a person has undergone a procedure. The release is intended for research and experimentation.

The local validation environment is macOS on Apple Silicon with Python 3.14. Runtime compatibility on other platforms depends on the available ONNX Runtime wheels and the repository's current test results.

Training scripts and source photographs are not included. The package code and the author's weight contributions use MIT licensing; exact upstream checkpoint and training-data provenance remain incomplete. See the [model card](https://github.com/LeventCeliksan/aesthetic-detector/blob/v1.0.0/MODEL_CARD.md) for details.

Feedback and integration examples are welcome in [Discussions](https://github.com/LeventCeliksan/aesthetic-detector/discussions); report reproducible package bugs in [Issues](https://github.com/LeventCeliksan/aesthetic-detector/issues).
