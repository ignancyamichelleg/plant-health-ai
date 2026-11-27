"""Reference statistics that monitoring compares live traffic against."""
import numpy as np
import pandas as pd


def describe(values) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def build_reference_stats(manifest: pd.DataFrame) -> dict:
    """Brightness, blur and class mix of the training images."""
    train = manifest[(manifest["split"] == "train") & manifest["readable"]]
    return {
        "source_split": "train",
        "n_images": len(train),
        "brightness": describe(train["brightness"]),
        "blur": describe(train["blur"]),
        "class_mix": {
            label: float(share)
            for label, share in train["label"].value_counts(normalize=True).items()
        },
    }