# Third-party notices

The original Python library and the author's contributions to the released model
weights are provided under the MIT license in `LICENSE`. That grant covers the
rights held by the author; it does not replace terms attached to upstream models
or training data.

The binary ONNX graph contains a CLIP ViT-B/16-style vision encoder and a custom
classifier. The regional graph is consistent with an EfficientNet-B0-style
network. Exact base checkpoint identifiers and their accompanying training-data
terms were not retained with these exports. Architecture identification alone
does not establish checkpoint provenance. See `MODEL_CARD.md` for the available
training and evaluation information.

## CLIP attribution

Upstream reference: https://github.com/openai/CLIP

The upstream CLIP project's MIT notice is reproduced below for attribution. This
does not assert that a particular undocumented checkpoint was used.

MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Runtime dependencies

NumPy, Pillow, and ONNX Runtime are installed separately and retain their own
licenses. Their licenses are not replaced by this project's license.
