"""Save a trained model together with the metadata needed to serve and trace it."""
import json
from pathlib import Path

from tensorflow import keras


def save_model(model: keras.Model, cfg: dict, info: dict) -> Path:
    """Write models/<name>/model.keras and metadata.json; return the folder."""
    out_dir = Path(cfg["paths"]["models_dir"]) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.keras")
    metadata = {
        "version": cfg["name"],
        "classes": cfg["classes"],
        "image_size": cfg["data"]["image_size"],
        "input": "float32 (H, W, 3), values 0-255; scaling is inside the model",
        **info,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return out_dir