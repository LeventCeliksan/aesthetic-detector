# Contributing

Contributions that make the models easier to integrate, evaluate, and document are welcome.

## Questions and bug reports

Use [Discussions](https://github.com/LeventCeliksan/aesthetic-detector/discussions) for integration questions and proposals. Use [Issues](https://github.com/LeventCeliksan/aesthetic-detector/issues) for reproducible defects.

A useful bug report includes the package version, Python version, operating system and architecture, relevant command or a short code example, expected behavior, and the complete error message. Include the model version or checksum when the problem concerns inference. Remove credentials and personal information from logs and paths before posting them.

Do not attach identifiable portraits without permission to publish them. When possible, reproduce software failures with a synthetic image or describe the input dimensions, format, and color mode.

## Local development

Use Python 3.11 or later and an isolated virtual environment:

```bash
git clone https://github.com/LeventCeliksan/aesthetic-detector.git
cd aesthetic-detector
python -m venv .venv
```

Activate the environment with the command appropriate to your shell, then install the package in editable mode:

```bash
python -m pip install -e .
aesthetic-detector info
```

For an inference check, download the appropriate release weights and use a local image you are permitted to process:

```bash
aesthetic-detector download --model binary
aesthetic-detector predict portrait.jpg
```

Model files are release assets and should not be committed to Git. Keep local test photographs, model caches, virtual environments, and generated build artifacts out of commits.

## Pull requests

Keep a pull request focused on one problem and explain the resulting behavior. Include the validation you performed and relevant environment details. Add or update tests when behavior changes, and update the usage documentation when public APIs or command-line options change.

Use English for code comments, documentation, and issue templates. Prefer clear names and short, purposeful explanations. Preserve applicable copyright and license notices.

Changes to preprocessing, label order, thresholds, or score interpretation can change predictions. Explain those changes explicitly and do not imply that existing evaluation claims remain valid after them. Changes to release downloads should preserve checksum verification and safe handling of partial files.

## Evaluation contributions

Reproducible evaluation is especially useful. State the model checksum, package version, preprocessing, threshold, metric definition, held-out image count, class balance, and data-split method. Describe how training/test overlap and duplicates were handled. Include per-class metrics and a confusion matrix when available.

Keep predictions and measurements distinct from the author's reported result. Do not present a small demonstration, successful execution, or a few handpicked photographs as a measured accuracy benchmark.

Before contributing data or model files, describe their source and the permissions that allow redistribution. The current release does not include the original training dataset.
