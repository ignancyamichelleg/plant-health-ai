"""Download the Beans dataset and save it as image files.

Output layout: data/raw/<split>/<label>/<split>_<index>.jpg
"""
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "AI-Lab-Makerere/beans"


def download(out_dir: str = "data/raw") -> None:
    ds = load_dataset(DATASET_ID)
    label_names = ds["train"].features["labels"].names  # angular_leaf_spot, bean_rust, healthy

    for split in ("train", "validation", "test"):
        for i, row in enumerate(ds[split]):
            label = label_names[row["labels"]]
            target = Path(out_dir) / split / label
            target.mkdir(parents=True, exist_ok=True)
            row["image"].convert("RGB").save(target / f"{split}_{i:04d}.jpg", quality=95)
        print(f"{split}: {len(ds[split])} images saved")


if __name__ == "__main__":
    download()