import io

import numpy as np
from PIL import Image

from plant_health.features.augmentation import build_augmentation
from plant_health.features.preprocessing import preprocess


def _jpeg_bytes(size=(500, 500), mode="RGB") -> bytes:
    rng = np.random.default_rng(0)
    channels = {"RGB": 3, "L": None, "RGBA": 4}[mode]
    shape = (size[1], size[0]) if channels is None else (size[1], size[0], channels)
    img = Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8), mode)
    buffer = io.BytesIO()
    fmt = "PNG" if mode == "RGBA" else "JPEG"
    img.save(buffer, format=fmt)
    return buffer.getvalue()


def test_output_shape_dtype_and_range():
    out = preprocess(_jpeg_bytes())
    assert out.shape == (160, 160, 3)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 255.0


def test_non_square_image_gives_square_output():
    assert preprocess(_jpeg_bytes(size=(300, 200))).shape == (160, 160, 3)


def test_grayscale_and_rgba_become_rgb():
    assert preprocess(_jpeg_bytes(mode="L")).shape == (160, 160, 3)
    assert preprocess(_jpeg_bytes(mode="RGBA")).shape == (160, 160, 3)


def test_same_bytes_give_same_array():
    data = _jpeg_bytes()
    assert np.array_equal(preprocess(data), preprocess(data))


def test_custom_image_size():
    assert preprocess(_jpeg_bytes(), image_size=224).shape == (224, 224, 3)


def test_augmentation_only_active_in_training():
    aug = build_augmentation(seed=1)
    x = np.random.default_rng(0).uniform(0, 255, (2, 160, 160, 3)).astype("float32")
    assert np.allclose(aug(x, training=False), x)
    assert not np.allclose(aug(x, training=True), x)