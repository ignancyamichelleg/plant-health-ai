"""Build data/manifest.csv: one row per image with file facts and quality stats."""
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

from plant_health.config import load_config
from plant_health.monitoring.quality import image_quality

MANIFEST_PATH = Path("data/manifest.csv")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dataset_hash(df: pd.DataFrame) -> str:
    """Short fingerprint of the dataset content (changes if any image changes)."""
    joined = "".join(sorted(df["sha256"]))
    return hashlib.sha256(joined.encode()).hexdigest()[:12]
def build_manifest(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    rows = []
    for path in sorted(raw_dir.rglob("*.jpg")):
        split, label = path.relative_to(raw_dir).parts[:2]
        row = {
            "path": path.as_posix(),
            "split": split,
            "label": label,
            "file_size": path.stat().st_size,
            "sha256": sha256_of(path),
            "readable": True,
        }
        try:
            with Image.open(path) as img:
                img.load()  # forces a full decode, so corrupt files fail here
                row.update(
                    width=img.width,
                    height=img.height,
                    channels=len(img.getbands()),
                    **image_quality(img),
                )
        except (OSError, ValueError):
            row["readable"] = False  # keep the row so validation can report it
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    df = build_manifest(cfg["data"]["raw_dir"])
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"Wrote {len(df)} rows to {MANIFEST_PATH}")
    print(df.groupby(["split", "label"]).size().unstack())
    print(df[["brightness", "blur"]].describe().round(1))


if __name__ == "__main__":
    main()