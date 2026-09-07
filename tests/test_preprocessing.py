"""Protect orientation, framing, and RGB normalization expected by the weights."""

import numpy as np
import pytest
from PIL import Image

from aesthetic_detector.preprocessing import prepare_portrait, prepare_region


@pytest.mark.parametrize(
    ("prepare", "mean", "std"),
    [
        (
            prepare_portrait,
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711],
        ),
        (prepare_region, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ],
)
def test_model_specific_rgb_normalization(prepare, mean, std):
    image = Image.new("RGB", (224, 224), (255, 0, 128))
    tensor = prepare(image)
    expected = (np.array([1, 0, 128 / 255]) - mean) / std

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    np.testing.assert_allclose(tensor[0, :, 100, 100], expected, rtol=1e-6, atol=1e-6)


def test_portrait_center_crop_and_region_stretch_keep_different_framing():
    image = Image.new("RGB", (672, 224), "green")
    image.paste((255, 0, 0), (0, 0, 224, 224))
    image.paste((0, 0, 255), (448, 0, 672, 224))
    portrait = prepare_portrait(image)
    region = prepare_region(image)
    center = prepare_portrait(Image.new("RGB", (224, 224), "green"))

    np.testing.assert_array_equal(portrait, center)
    assert region[0, 0, 100, 10] > region[0, 2, 100, 10]
    assert region[0, 2, 100, 210] > region[0, 0, 100, 210]


@pytest.mark.parametrize("prepare", [prepare_portrait, prepare_region])
@pytest.mark.parametrize(
    ("orientation", "transpose"),
    [(5, Image.Transpose.TRANSPOSE), (7, Image.Transpose.TRANSVERSE)],
)
def test_exif_mirrored_rotations_match_physically_oriented_pixels(
    tmp_path, prepare, orientation, transpose
):
    image = Image.new("RGB", (240, 320), (20, 30, 40))
    image.paste((220, 15, 30), (0, 0, 80, 100))
    image.paste((10, 200, 60), (80, 100, 240, 220))
    image.paste((20, 40, 230), (0, 220, 170, 320))
    exif = image.getexif()
    exif[274] = orientation
    path = tmp_path / "oriented.png"
    image.save(path, exif=exif)
    expected = image.transpose(transpose)
    expected.getexif().clear()

    np.testing.assert_array_equal(prepare(path), prepare(expected))


def test_image_input_is_not_modified_or_closed():
    image = Image.new("RGBA", (240, 320), (20, 30, 40, 255))
    before = image.tobytes()

    prepare_portrait(image)
    prepare_region(image)

    assert image.size == (240, 320)
    assert image.mode == "RGBA"
    assert image.tobytes() == before


def test_unreadable_image_has_actionable_error(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"This is not image data.")

    with pytest.raises(ValueError, match="valid local image"):
        prepare_portrait(path)
