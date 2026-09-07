# Model card

## Overview

| Item | Description |
| --- | --- |
| Release | `v1.0.0` |
| Author | Levent Celiksan |
| Task | Experimental classification of visual cues associated with cosmetic intervention in portrait photographs. |
| Binary labels | `natural`, `cosmetic_intervention` |
| Additional model | A scalar scorer for manually cropped facial regions. |
| Format | ONNX |
| Runtime | ONNX Runtime, CPU execution. |
| Package | `aesthetic-detector`; Python import `aesthetic_detector`. |

This release packages previously exported models for reuse without retraining. It does not include training scripts, the training images, the original Android application, backend services, or a content-filter model.

## Models and architecture

The binary ONNX graph has a **CLIP ViT-B/16-style** visual backbone and a custom **768 → 256 → 2** classification head. The regional graph has an **EfficientNet-B0-style** architecture. These descriptions come from inspection of the exported graph structure; the exact upstream checkpoint identifiers and training implementation are unavailable.

| Model | Approximate download | Input | Output |
| --- | --- | --- | --- |
| Binary classifier | 164.2 MiB | A close, frontal, single-person portrait. | Two class scores and a threshold-based label. |
| Regional scorer | 16.5 MiB | A facial-region crop prepared by the caller. | One task-specific scalar score. |

Release weights are downloaded lazily from GitHub Releases and verified against a SHA-256 manifest before use. The Python package also accepts compatible local ONNX files.

## Training data and development

The author reports using **1,000 photographs for fine-tuning**. The available release materials do not establish the class balance, data sources, annotation process, participant characteristics, duplicate handling, or training/validation/test allocation. The exact starting checkpoints, hyperparameters, training duration, and random seeds are not available.

The photographs and training scripts are not distributed. The inference package should not be described as a reproducible training pipeline or as a published dataset.

## Evaluation

The author reports a **97% evaluation result** from the original work. The specific metric, evaluation set size, split protocol, model variant, and supporting predictions or logs are unavailable. In particular, the result cannot currently be identified as accuracy, F1, precision, or recall.

This release has not independently reproduced the reported result. Software installation checks and inference smoke tests verify that the package can execute; they do not measure classification quality.

The Python binary pipeline uses bicubic image resizing and a center crop with CLIP normalization. The original Android pipeline is not reproduced byte-for-byte. Consequently, the original reported result does not transfer automatically to this implementation, even though the exported weights are reused.

To establish a reproducible benchmark, an evaluation should identify the exact model checksum, preprocessing version, threshold, metric definition, held-out image count, class distribution, and method for preventing overlap between training and test data. It should also include per-class results and a confusion matrix. None of these missing measurements are implied by the reported 97% figure.

## Inputs, preprocessing, and outputs

`AestheticDetector.predict` accepts an image path or a Pillow image. The caller is responsible for providing suitable portrait framing. The package does not detect faces, validate the number of people, assess image content, or create region crops automatically.

For the binary classifier, the default intervention threshold is **0.52**. The output includes both softmax scores, the selected label, the threshold, and a model version identifier. Softmax scores are not calibrated estimates of real-world procedure prevalence or medical certainty. The label `natural` does not verify an absence of procedures; `cosmetic_intervention` does not verify their presence.

`RegionalScorer.predict` takes a caller-supplied crop and returns `score` and `model_version`. The release does not establish calibration, validated cutoffs, or a procedure-specific meaning for this scalar. Applying it to a full portrait does not perform automatic analysis of separate facial regions.

## Intended use and limitations

The models are intended for research, experimentation, and integration of local image inference. They are not clinically validated and should not be used to establish a person's medical or cosmetic-procedure history.

Known limits of the release include:

- No published held-out evaluation or independently reproduced quality measurement.
- No established performance breakdown across demographic groups or image sources.
- Potential sensitivity to lighting, pose, makeup, filters, compression, facial framing, and other differences from the training images.
- No built-in rejection of unsuitable images, multiple faces, or non-portrait content.
- No automatic face detection, alignment, or region cropping.
- No calibrated confidence estimates or validated procedure-specific diagnoses.

Treat outputs as fallible model predictions. Evaluate them on representative, appropriately labeled images before making claims about performance in another application.

## Software validation

The local validation environment is **macOS on Apple Silicon with Python 3.14**. This statement identifies the environment used for package checks; it is not a claim that all operating systems or Python versions have been tested. The package requires Python 3.11 or later, with compatibility also depending on available ONNX Runtime wheels.

Check the repository's test results for the current software-validation status. Model-quality evaluation remains separate from those software checks.

## Provenance and license

The package code and the author's contributions to the released weights are offered under the [MIT license](LICENSE). This grant covers the rights held by the author; it does not establish additional rights over third-party checkpoints or training data.

Relevant upstream references are:

- [OpenAI CLIP](https://github.com/openai/CLIP), whose repository uses the [MIT license](https://github.com/openai/CLIP/blob/main/LICENSE).
- [Torchvision](https://github.com/pytorch/vision), whose source uses the [BSD 3-Clause license](https://github.com/pytorch/vision/blob/main/LICENSE).

The architecture inspection does not establish which upstream checkpoint or implementation produced either export. Torchvision also documents that [pretrained weights can have terms derived from their training datasets](https://docs.pytorch.org/vision/stable/models.html#general-information-on-pre-trained-weights). Exact checkpoint and training-data provenance remain incomplete, and no complete audit of those underlying terms is claimed.

## Feedback

Use [Issues](https://github.com/LeventCeliksan/aesthetic-detector/issues) for reproducible package defects and [Discussions](https://github.com/LeventCeliksan/aesthetic-detector/discussions) for usage questions, evaluation proposals, and integration feedback. Share test images only when you have permission to publish them.
