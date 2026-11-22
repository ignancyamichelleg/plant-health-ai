"""Tests for evaluation metrics, slice analysis and model selection (no TensorFlow)."""
import numpy as np
import pandas as pd

from plant_health.models.evaluate import evaluate_predictions, select_model, selection_markdown

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]


def _rows(n):
    return pd.DataFrame({
        "path": [f"img_{i}.jpg" for i in range(n)],
        "brightness": np.linspace(40, 200, n),
        "blur": np.linspace(100, 900, n),
    })


def _one_hot(pred, conf=0.8):
    probs = np.full((len(pred), 3), (1 - conf) / 2)
    probs[np.arange(len(pred)), pred] = conf
    return probs


def test_perfect_predictions():
    y = np.array([0, 1, 2] * 4)
    report, wrong = evaluate_predictions(y, _one_hot(y), CLASSES, _rows(len(y)))
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["min_class_recall"] == 1.0
    assert wrong.empty
    assert np.trace(report["confusion_matrix"]["rows_true_cols_pred"]) == 12


def test_errors_are_counted_and_listed():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 2, 1, 1, 2, 2])  # one angular_leaf_spot missed
    report, wrong = evaluate_predictions(y, _one_hot(pred), CLASSES, _rows(6))
    assert report["per_class"]["angular_leaf_spot"]["recall"] == 0.5
    assert report["min_class_recall"] == 0.5
    assert report["confusion_matrix"]["rows_true_cols_pred"][0] == [1, 0, 1]
    assert list(wrong["true_class"]) == ["angular_leaf_spot"]
    assert list(wrong["predicted_class"]) == ["healthy"]


def test_slices_cover_all_images():
    y = np.array([0, 1, 2] * 5)
    report, _ = evaluate_predictions(y, _one_hot(y), CLASSES, _rows(len(y)))
    for column in ("brightness", "blur"):
        buckets = report["slices"][column]
        assert [b["bucket"] for b in buckets] == ["low", "mid", "high"]
        assert sum(b["n"] for b in buckets) == len(y)


def test_selection_uses_min_recall_as_tiebreak():
    a = {"model": "a", "macro_f1": 0.90, "min_class_recall": 0.70, "accuracy": 0.9,
         "per_class": {}}
    b = {"model": "b", "macro_f1": 0.90, "min_class_recall": 0.85, "accuracy": 0.9,
         "per_class": {}}
    c = {"model": "c", "macro_f1": 0.80, "min_class_recall": 0.80, "accuracy": 0.8,
         "per_class": {}}
    ranked = select_model([a, c, b])
    assert [r["model"] for r in ranked] == ["b", "a", "c"]
    assert "Selected: `b`" in selection_markdown(ranked)
