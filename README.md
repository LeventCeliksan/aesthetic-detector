# Aesthetic Detector

A fine-tuned image classifier and Python inference library for experimenting with visual cues associated with cosmetic intervention in portrait photographs. Download the trained ONNX weights and run predictions locally without retraining or an API key.

The package includes a binary portrait classifier and an optional regional scorer for manually cropped facial regions. It is an independent inference package; the original Android application and backend are not required.

[Download v1.0.0](https://github.com/LeventCeliksan/aesthetic-detector/releases/tag/v1.0.0) · [Model card](MODEL_CARD.md) · [Discussions](https://github.com/LeventCeliksan/aesthetic-detector/discussions) · [Issues](https://github.com/LeventCeliksan/aesthetic-detector/issues)

## Install

Use Python 3.11 or later in a virtual environment. Installation from GitHub also requires Git:

```bash
python -m pip install "git+https://github.com/LeventCeliksan/aesthetic-detector.git@v1.0.0"
```

Alternatively, install the release wheel without Git:

```bash
python -m pip install "https://github.com/LeventCeliksan/aesthetic-detector/releases/download/v1.0.0/aesthetic_detector-1.0.0-py3-none-any.whl"
```

Inference uses ONNX Runtime on the CPU. No GPU, training framework, or account is required. Runtime wheel availability depends on your Python version and operating system. Local validation uses macOS on Apple Silicon with Python 3.14; this does not establish compatibility with every platform.

## Classify a portrait

Choose a close, frontal photograph containing one person. The library does not locate faces, crop them automatically, or check image content. Input quality and framing affect predictions.

```python
from aesthetic_detector import AestheticDetector

detector = AestheticDetector()
result = detector.predict("portrait.jpg")
print(result.to_dict())
```

`predict` accepts an image path or a Pillow image. The first call that needs the model downloads approximately 164.2 MiB of weights from the release, verifies their SHA-256 checksum, and caches them for reuse. The input image is processed locally.

The equivalent command prints JSON to standard output:

```bash
aesthetic-detector predict portrait.jpg
```

The result contains:

| Field | Meaning |
| --- | --- |
| `label` | `natural` or `cosmetic_intervention`, selected using the threshold. |
| `intervention_score` | Softmax score for the cosmetic-intervention class. |
| `natural_score` | Softmax score for the natural class. |
| `threshold` | Decision threshold, defaulting to `0.52`. |
| `model_version` | Version identifier associated with the model. |

These class names describe the training task. The scores are uncalibrated model outputs, not percentages of medical certainty. A prediction does not establish whether someone has undergone a procedure or reveal their medical history.

You can set a different threshold and cache directory:

```python
detector = AestheticDetector(threshold=0.60, cache_dir="./model-cache")
print(detector.predict("portrait.jpg").to_dict())
```

```bash
aesthetic-detector predict portrait.jpg --threshold 0.60 --cache-dir ./model-cache
```

Changing the threshold changes the decision rule; it does not improve the underlying model or establish a new benchmark result.

## Score a cropped region

The regional model produces one score for a supplied crop. Prepare the crop yourself; this API does not identify a nose, lips, or other region within a full portrait.

```python
from aesthetic_detector import RegionalScorer

scorer = RegionalScorer()
result = scorer.predict("nose-crop.jpg")
print(result.to_dict())
```

```bash
aesthetic-detector score-region nose-crop.jpg
```

The result contains `score` and `model_version`. Regional weights are approximately 16.5 MiB and download only when needed. The scalar output is a task-specific model score; no published calibration or procedure-specific interpretation is provided.

## Download in advance or use local weights

```bash
aesthetic-detector download --model all --cache-dir ./model-cache
aesthetic-detector info
```

Use `--model binary` or `--model regional` to download just one model. After downloading, point predictions at the same cache directory for offline use:

```bash
aesthetic-detector predict portrait.jpg --cache-dir ./model-cache
```

You can also supply an ONNX model directly:

```python
detector = AestheticDetector(model_path="./weights/classifier.onnx")
scorer = RegionalScorer(model_path="./weights/regional.onnx")
```

```bash
aesthetic-detector predict portrait.jpg --model-path ./weights/classifier.onnx
aesthetic-detector score-region nose-crop.jpg --model-path ./weights/regional.onnx
```

Local files must match the input/output contract expected by the respective API. SHA-256 verification applies to the release weights managed by the downloader; supplying another model does not establish its provenance or accuracy.

## Training and reported result

The author reports fine-tuning with **1,000 photographs** and a **97% evaluation result**. The metric definition, evaluation image count, and train/test split are not available in this release. Training scripts, source photographs, and evaluation artifacts are not included, so this result has not been independently reproduced.

The Python preprocessing uses a bicubic resize and center crop for the binary classifier. It is not byte-for-byte equivalent to the original Android preprocessing, so the reported result should not be treated as a measured result for this package. See the [model card](MODEL_CARD.md) for architecture, provenance, and evaluation details.

## Intended use

Use this release for research, experimentation, and building local inference integrations. It has not been clinically validated. Lighting, pose, makeup, filters, compression, framing, and differences from the training images can affect the scores. Performance across demographic groups and image sources has not been established.

## License and attribution

The package code and the author's contributions to the weights are offered under the [MIT license](LICENSE). The exact upstream checkpoints and training-data provenance are incomplete; this release does not claim that all underlying checkpoint and data terms have been audited. See the [model card](MODEL_CARD.md#provenance-and-license) for the known upstream references and scope of the license.

## Contribute and share feedback

[Open an issue](https://github.com/LeventCeliksan/aesthetic-detector/issues) for reproducible bugs or [start a discussion](https://github.com/LeventCeliksan/aesthetic-detector/discussions) for integration questions and suggestions. Pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

If this project helps your work, you can [star the repository](https://github.com/LeventCeliksan/aesthetic-detector) or share how you used it in Discussions.
