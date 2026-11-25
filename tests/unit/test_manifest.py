import numpy as np
from PIL import Image, ImageFilter

from plant_health.data.manifest import build_manifest
from plant_health.monitoring.quality import image_quality


def _noise_image(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))


def _save(img: Image.Image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def test_manifest_columns_and_counts(tmp_path):
    _save(_noise_image(0), tmp_path / "train" / "healthy" / "a.jpg")
    _save(_noise_image(1), tmp_path / "validation" / "bean_rust" / "b.jpg")
    df = build_manifest(tmp_path)
    assert len(df) == 2
    assert set(df["split"]) == {"train", "validation"}
    assert df["readable"].all()
    assert {"sha256", "width", "height", "channels", "brightness", "blur"} <= set(df.columns)


def test_corrupt_file_is_marked_unreadable(tmp_path):
    bad = tmp_path / "train" / "healthy" / "bad.jpg"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"this is not an image")
    df = build_manifest(tmp_path)
    assert not df.loc[0, "readable"]


def test_dark_image_has_lower_brightness():
    dark = Image.new("RGB", (64, 64), (20, 20, 20))
    light = Image.new("RGB", (64, 64), (200, 200, 200))
    assert image_quality(dark)["brightness"] < image_quality(light)["brightness"]


def test_blur_lowers_blur_score():
    sharp = _noise_image()
    blurred = sharp.filter(ImageFilter.GaussianBlur(3))
    assert image_quality(blurred)["blur"] < image_quality(sharp)["blur"]