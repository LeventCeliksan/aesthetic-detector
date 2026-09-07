"""Image preparation matching the exported models' normalization contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ImageSource = str | Path | Image.Image
IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _rgb_image(source: ImageSource) -> Image.Image:
    try:
        if isinstance(source, Image.Image):
            return ImageOps.exif_transpose(source).convert("RGB")
        with Image.open(source) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Image could not be decoded. Supply a valid local image file.") from error


def _tensor(image: Image.Image, mean: tuple, std: tuple) -> np.ndarray:
    pixels = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    normalized = (pixels - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)


def prepare_portrait(source: ImageSource) -> np.ndarray:
    """Apply EXIF orientation, CLIP bicubic resize, center crop, and normalization."""
    image = _rgb_image(source)
    scale = IMAGE_SIZE / min(image.size)
    size = tuple(max(IMAGE_SIZE, int(dimension * scale)) for dimension in image.size)
    image = image.resize(size, Image.Resampling.BICUBIC)
    left = (image.width - IMAGE_SIZE) // 2
    top = (image.height - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    return _tensor(image, CLIP_MEAN, CLIP_STD)


def prepare_region(source: ImageSource) -> np.ndarray:
    """Normalize an already-cropped face region using ImageNet statistics."""
    image = _rgb_image(source).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return _tensor(image, IMAGENET_MEAN, IMAGENET_STD)
