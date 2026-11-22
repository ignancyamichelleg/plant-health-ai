import json

import numpy as np
from PIL import Image

from plant_health.config import ROOT, load_config
from plant_health.pipeline import run_pipeline

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]


def _cfg(tmp_path):
    cfg = load_config(str(ROOT / "configs" / "experiments" / "baseline_cnn.yaml"))
    cfg["data"].update(raw_dir=str(tmp_path / "raw"), image_size=32, batch_size=4)
    cfg["training"].update(epochs=1)
    cfg["paths"].update(
        models_dir=str(tmp_path / "models"), reports_dir=str(tmp_path / "reports")
    )
    cfg["mlflow"].update(
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        artifact_root=str(tmp_path / "mlruns"),
    )
    return cfg


def _write_images(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 4), ("validation", 2), ("test", 2)):
        for label in CLASSES:
            folder = tmp_path / "raw" / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                arr = rng.integers(0, 256, (72, 72, 3), dtype=np.uint8)
                Image.fromarray(arr).save(folder / f"{split}_{i}.jpg")


def test_pipeline_end_to_end_writes_all_artifacts(tmp_path):
    _write_images(tmp_path)
    assert run_pipeline(_cfg(tmp_path)) == 0

    reports = tmp_path / "reports"
    assert json.loads((reports / "validation_report.json").read_text())["passed"]
    stats = json.loads((reports / "reference_stats.json").read_text())
    assert "brightness" in stats and "confidence_validation" in stats
    assert (reports / "pipeline_run.json").exists()
    assert (tmp_path / "models" / "baseline_cnn" / "model.keras").exists()


def test_pipeline_stops_before_training_when_validation_fails(tmp_path):
    _write_images(tmp_path)
    (tmp_path / "raw" / "train" / "healthy" / "bad.jpg").write_bytes(b"not an image")

    assert run_pipeline(_cfg(tmp_path)) == 1

    report = json.loads((tmp_path / "reports" / "validation_report.json").read_text())
    assert not report["passed"]
    assert not (tmp_path / "models").exists()  # training never started