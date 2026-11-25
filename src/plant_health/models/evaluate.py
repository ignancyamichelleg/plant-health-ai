"""Evaluate a saved model on one split, and select the final model.

    # every candidate, on validation
    python -m plant_health.models.evaluate --model models/baseline_cnn
    python -m plant_health.models.evaluate --model models/cnn_aug
    python -m plant_health.models.evaluate --model models/mobilenet_frozen

    # pick the winner from the validation reports -> reports/model_selection.md
    python -m plant_health.models.evaluate --select

    # once, at the very end, for the selected model only
    python -m plant_health.models.evaluate --model models/<winner> --split test --final

Outputs per run in reports/: eval_<model>_<split>.json (metrics, per-class,
confusion matrix, slices by brightness/blur) and misclassified_<model>_<split>.csv.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from plant_health.config import load_config
from plant_health.models.metrics import compute_metrics

TEST_LOCK = "test_evaluation_done.json"
REFERENCE_FILE = "reference_stats.json"


def _bucket(values: pd.Series) -> pd.Series:
    """Split a quality metric into low / mid / high thirds of this split."""
    try:
        return pd.qcut(values, 3, labels=["low", "mid", "high"])
    except ValueError:  # too few distinct values to cut into thirds
        return pd.Series(["all"] * len(values), index=values.index)


def slice_metrics(frame: pd.DataFrame, column: str, classes: list[str]) -> list[dict]:
    """Accuracy and macro-F1 within each bucket of `column` (brightness or blur)."""
    frame = frame.assign(bucket=_bucket(frame[column]))
    rows = []
    for bucket, part in frame.groupby("bucket", observed=True):
        m = compute_metrics(part["y_true"].to_numpy(), part["y_pred"].to_numpy(),
                            classes, prefix="s")
        rows.append({
            "bucket": str(bucket),
            "n": len(part),
            f"{column}_min": float(part[column].min()),
            f"{column}_max": float(part[column].max()),
            "accuracy": m["s_accuracy"],
            "macro_f1": m["s_macro_f1"],
            "mean_confidence": float(part["confidence"].mean()),
        })
    return rows


def evaluate_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: list[str],
    rows: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Metrics for one split. `rows` holds path, brightness, blur per image (same order).

    Returns (report dict, table of misclassified images).
    """
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    labels = list(range(len(classes)))

    overall = compute_metrics(y_true, y_pred, classes, prefix="eval")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        name: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for name, p, r, f, s in zip(classes, precision, recall, f1, support, strict=True)
    }

    frame = rows[["path", "brightness", "blur"]].reset_index(drop=True).assign(
        y_true=y_true, y_pred=y_pred, confidence=confidence
    )
    report = {
        "n": len(frame),
        "accuracy": overall["eval_accuracy"],
        "macro_f1": overall["eval_macro_f1"],
        "min_class_recall": float(recall.min()),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": classes,
            "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
        "confidence": {
            "mean": float(confidence.mean()),
            "std": float(confidence.std()),
            "p05": float(np.percentile(confidence, 5)),
            "p50": float(np.percentile(confidence, 50)),
            "p95": float(np.percentile(confidence, 95)),
            "mean_correct": float(confidence[y_true == y_pred].mean())
            if (y_true == y_pred).any() else None,
            "mean_wrong": float(confidence[y_true != y_pred].mean())
            if (y_true != y_pred).any() else None,
        },
        "slices": {
            "brightness": slice_metrics(frame, "brightness", classes),
            "blur": slice_metrics(frame, "blur", classes),
        },
    }

    wrong = frame[frame["y_true"] != frame["y_pred"]].copy()
    wrong["true_class"] = [classes[i] for i in wrong["y_true"]]
    wrong["predicted_class"] = [classes[i] for i in wrong["y_pred"]]
    wrong = wrong[["path", "true_class", "predicted_class", "confidence",
                   "brightness", "blur"]].sort_values("confidence", ascending=False)
    return report, wrong


def select_model(reports: list[dict]) -> list[dict]:
    """Rank validation reports: macro-F1 first, then the weakest class recall
    (so a model that quietly misses one disease does not win on average)."""
    return sorted(reports, key=lambda r: (r["macro_f1"], r["min_class_recall"]), reverse=True)


def selection_markdown(ranked: list[dict]) -> str:
    lines = [
        "# Model selection",
        "",
        "Ranked on the validation split by macro-F1, ties broken by the lowest",
        "per-class recall. The test split is not used for selection.",
        "",
        "| Rank | Model | Macro-F1 | Accuracy | Min class recall | "
        "Recall ALS | Recall rust | Recall healthy |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(ranked, 1):
        pc = r["per_class"]
        lines.append(
            f"| {i} | {r['model']} | {r['macro_f1']:.3f} | {r['accuracy']:.3f} | "
            f"{r['min_class_recall']:.3f} | "
            f"{pc.get('angular_leaf_spot', {}).get('recall', float('nan')):.3f} | "
            f"{pc.get('bean_rust', {}).get('recall', float('nan')):.3f} | "
            f"{pc.get('healthy', {}).get('recall', float('nan')):.3f} |"
        )
    if ranked:
        lines += ["", f"**Selected: `{ranked[0]['model']}`**", ""]
    return "\n".join(lines)


def update_reference(reference_path: Path, winner: dict) -> bool:
    """Point the monitoring reference at the selected model.

    The training pipeline rewrites reference_stats.json after every run, so its
    confidence block belongs to whichever experiment ran last. Monitoring must
    compare live confidence with the model that is actually served.
    Returns False if there is nothing to update.
    """
    conf = winner.get("confidence", {})
    if not reference_path.is_file() or "p50" not in conf:
        return False
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference["confidence_validation"] = {
        k: conf[k] for k in ("mean", "std", "p05", "p50", "p95")
    }
    reference["selected_model"] = winner["model"]
    reference_path.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    return True


def run_evaluation(model_dir: Path, split: str, cfg: dict, reports_dir: Path) -> dict:
    """Load a saved model, predict on a split and write the reports."""
    from tensorflow import keras  # imported here so tests of the logic need no TF

    from plant_health.data.dataset import load_split
    from plant_health.data.manifest import build_manifest

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    classes, size = metadata["classes"], int(metadata["image_size"])

    manifest = build_manifest(cfg["data"]["raw_dir"])
    rows = manifest[(manifest["split"] == split) & manifest["readable"]]
    x, y = load_split(manifest, split, classes, size)
    model = keras.models.load_model(model_dir / "model.keras", compile=False)
    probs = model.predict(x, verbose=0)

    report, wrong = evaluate_predictions(y, probs, classes, rows)
    report.update(model=model_dir.name, split=split, version=metadata.get("version"),
                  run_id=metadata.get("run_id"), data_hash=metadata.get("data_hash"))

    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_dir.name}_{split}"
    (reports_dir / f"eval_{stem}.json").write_text(json.dumps(report, indent=2),
                                                     encoding="utf-8")
    wrong.to_csv(reports_dir / f"misclassified_{stem}.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="model folder, e.g. models/baseline_cnn")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--final", action="store_true",
                        help="required for the one-time test-set evaluation")
    parser.add_argument("--select", action="store_true",
                        help="rank all validation reports and write model_selection.md")
    args = parser.parse_args()

    cfg = load_config()
    reports_dir = Path(cfg["paths"]["reports_dir"])

    if args.select:
        reports = [json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted(reports_dir.glob("eval_*_validation.json"))]
        if not reports:
            parser.error("no reports/eval_*_validation.json found; evaluate models first")
        ranked = select_model(reports)
        md = selection_markdown(ranked)
        (reports_dir / "model_selection.md").write_text(md, encoding="utf-8")
        print(md)
        if update_reference(reports_dir / REFERENCE_FILE, ranked[0]):
            print(f"{REFERENCE_FILE}: confidence now from {ranked[0]['model']}")
        else:
            print(f"{REFERENCE_FILE} not updated: re-run `make evaluate` for "
                  f"{ranked[0]['model']} (older reports lack the confidence summary)")
        return

    if not args.model:
        parser.error("--model is required unless --select is given")

    if args.split == "test":
        lock = reports_dir / TEST_LOCK
        if not args.final:
            parser.error("the test split is used once; pass --final for the selected model")
        if lock.exists():
            parser.error(f"test set already used ({lock}); delete it only if you must redo")

    report = run_evaluation(Path(args.model), args.split, cfg, reports_dir)

    if args.split == "test":
        (reports_dir / TEST_LOCK).write_text(
            json.dumps({"model": report["model"], "macro_f1": report["macro_f1"]}, indent=2),
            encoding="utf-8",
        )
    print(f"{report['model']} on {args.split}: macro-F1 {report['macro_f1']:.3f}, "
          f"accuracy {report['accuracy']:.3f}, min class recall "
          f"{report['min_class_recall']:.3f}")


if __name__ == "__main__":
    main()
