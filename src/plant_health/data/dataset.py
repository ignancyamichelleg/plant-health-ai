"""Load images of one split into memory using the shared preprocessing."""
import numpy as np
import pandas as pd

from plant_health.features.preprocessing import preprocess_file


def load_split(
    manifest: pd.DataFrame, split: str, classes: list[str], image_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (images uint8 [N, size, size, 3], integer labels [N]) for a split.

    Label index = position in the `classes` list from the config.
    preprocess() output is whole numbers in 0-255, so uint8 storage is lossless.
    """
    rows = manifest[(manifest["split"] == split) & manifest["readable"]]
    index = {name: i for i, name in enumerate(classes)}
    images = np.stack([preprocess_file(p, image_size) for p in rows["path"]])
    labels = rows["label"].map(index).to_numpy(dtype=np.int64)
    return images.astype(np.uint8), labels