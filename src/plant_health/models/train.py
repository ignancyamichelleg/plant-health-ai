"""Train one experiment from a YAML config and log it to MLflow.

Usage:
    python -m plant_health.models.train --config configs/experiments/baseline_cnn.yaml

Only the train and validation splits are used here; the test set is never touched.
"""
import argparse
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from tensorflow import keras

from plant_health.config import load_config
from plant_health.data.dataset import load_split
from plant_health.data.manifest import build_manifest, dataset_hash
from plant_health.models.builders import build_model
from plant_health.models.metrics import compute_metrics
from plant_health.models.registry import save_model
from plant_health.monitoring.reference import describe


def _flatten(d: dict, parent: str = "") -> dict:
    items: dict = {}
    for key, value in d.items():
        name = f"{parent}{key}"
        if isinstance(value, dict):
            items.update(_flatten(value, f"{name}."))
        else:
            items[name] = str(value) if isinstance(value, list) else value
    return items


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _setup_mlflow(cfg: dict) -> None:
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    name = cfg["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        root = Path(cfg["mlflow"]["artifact_root"]).resolve()
        mlflow.create_experiment(name, artifact_location=root.as_uri())
    mlflow.set_experiment(name)


def run(cfg: dict, manifest: pd.DataFrame | None = None) -> dict:
    """Train, evaluate on the validation split, log to MLflow, save the model."""
    keras.utils.set_random_seed(cfg["seed"])  # Python, NumPy and TensorFlow seeds
    if manifest is None:
        manifest = build_manifest(cfg["data"]["raw_dir"])

    classes, size = cfg["classes"], cfg["data"]["image_size"]
    x_train, y_train = load_split(manifest, "train", classes, size)
    x_val, y_val = load_split(manifest, "validation", classes, size)

    model = build_model(cfg)
    train_cfg = cfg["training"]
    _setup_mlflow(cfg)

    with mlflow.start_run(run_name=cfg["name"]) as active:
        git_sha, data_hash = _git_sha(), dataset_hash(manifest)
        mlflow.log_params(_flatten(cfg))
        mlflow.set_tags({"git_sha": git_sha, "data_hash": data_hash})

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=train_cfg["epochs"],
            batch_size=cfg["data"]["batch_size"],
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=train_cfg["early_stopping_patience"],
                    restore_best_weights=True,
                )
            ],
            verbose=2,
        )
        for epoch in range(len(history.history["loss"])):
            mlflow.log_metrics(
                {k: float(v[epoch]) for k, v in history.history.items()}, step=epoch
            )

        probs = model.predict(x_val, verbose=0)
        y_pred = probs.argmax(axis=1)
        metrics = compute_metrics(y_val, y_pred, classes, prefix="final_val")
        mlflow.log_metrics(metrics)

        model_dir = save_model(
            model,
            cfg,
            {
                "run_id": active.info.run_id,
                "git_sha": git_sha,
                "data_hash": data_hash,
                "val_macro_f1": metrics["final_val_macro_f1"],
                "val_accuracy": metrics["final_val_accuracy"],
            },
        )
        mlflow.log_artifacts(str(model_dir), artifact_path="model")

        return {
        "run_id": active.info.run_id,
        "model_dir": str(model_dir),
        "val_confidence": describe(probs.max(axis=1)),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="experiment YAML file")
    args = parser.parse_args()
    result = run(load_config(args.config))
    print(f"\nRun {result['run_id']} saved to {result['model_dir']}")
    for key, value in result.items():
        if key.startswith("final_val"):
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()