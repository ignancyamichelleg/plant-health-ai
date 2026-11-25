"""Image quality metrics, shared by the dataset manifest, the API and monitoring.

Both metrics are computed on a fixed-size grayscale copy so that photos of
different resolutions are comparable.
"""
import numpy as np
from PIL import Image

QUALITY_SIZE = (256, 256)


def _gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L").resize(QUALITY_SIZE), dtype=np.float32)


def brightness(img: Image.Image) -> float:
    """Mean grayscale intensity (0 = black, 255 = white)."""
    return float(_gray(img).mean())


def blur_score(img: Image.Image) -> float:
    """Variance of the Laplacian: low values mean a blurry image."""
    g = _gray(img)
    lap = (
        -4 * g[1:-1, 1:-1]
        + g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
    )
    return float(lap.var())


def image_quality(img: Image.Image) -> dict:
    return {"brightness": brightness(img), "blur": blur_score(img)}