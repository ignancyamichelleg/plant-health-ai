"""Image preprocessing shared by training and the serving API.

Never re-implement this elsewhere: training and serving must import this function.
Output values stay in 0-255; scaling is done by the first layer of the model.
"""
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

DEFAULT_IMAGE_SIZE = 160


def preprocess(image_bytes: bytes, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Decode image bytes into a float32 array of shape (size, size, 3), values 0-255.

    - applies the EXIF rotation (phone photos are often stored sideways)
    - converts any mode (grayscale, RGBA, ...) to RGB
    - center-crops to a square and resizes, so non-square photos are not distorted
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img = ImageOps.fit(img, (image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32)


def preprocess_file(path: str | Path, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    return preprocess(Path(path).read_bytes(), image_size)