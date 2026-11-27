import json

import numpy as np
from PIL import Image

from plant_health.config import ROOT, load_config
from plant_health.models.builders import build_model
from plant_health.models.metrics import compute_metrics
from plant_health.models.train import run

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]


def _tiny_cfg(tmp_path, experiment="baseline_cnn"):
    cfg = load_config(str(ROOT / "configs" / "experiments" / f"{experiment}.yaml"))
    cfg["data"].update(raw_dir=str(tmp_path / "raw"), image_size=32, batch_size=4)
    cfg["training"].update(epochs=1)
    cfg["paths"]["models_dir"] = str(tmp_path / "models")
    cfg["mlflow"].update(
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        artifact_root=str(tmp_path / "mlruns"),
    )
    return cfg


def _write_images(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 4), ("validation", 2)):
        for label in CLASSES:
            folder = tmp_path / "raw" / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                arr = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
                Image.fromarray(arr).save(folder / f"{split}_{i}.jpg")


def test_build_model_outputs_class_probabilities(tmp_path):
    model = build_model(_tiny_cfg(tmp_path))
    out = model.predict(np.zeros((2, 32, 32, 3), dtype="float32"), verbose=0)
    assert out.shape == (2, 3)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_augmentation_flag_adds_augmentation_layer(tmp_path):
    plain = build_model(_tiny_cfg(tmp_path))
    aug = build_model(_tiny_cfg(tmp_path, "cnn_aug"))
    assert "augmentation" not in [layer.name for layer in plain.layers]
    assert "augmentation" in [layer.name for layer in aug.layers]


def test_compute_metrics_perfect_and_wrong():
    y = np.array([0, 1, 2, 2])
    perfect = compute_metrics(y, y, CLASSES, prefix="p")
    assert perfect["p_accuracy"] == 1.0
    assert perfect["p_macro_f1"] == 1.0
    wrong = compute_metrics(y, np.array([1, 2, 0, 0]), CLASSES, prefix="p")
    assert wrong["p_accuracy"] == 0.0


def test_training_smoke_run_saves_model_and_metadata(tmp_path):
    _write_images(tmp_path)
    result = run(_tiny_cfg(tmp_path))
    assert 0.0 <= result["final_val_macro_f1"] <= 1.0
    model_dir = tmp_path / "models" / "baseline_cnn"
    assert (model_dir / "model.keras").exists()
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["classes"] == CLASSES
    assert metadata["run_id"] == result["run_id"]