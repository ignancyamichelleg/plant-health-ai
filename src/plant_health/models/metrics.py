"""Classification metrics used to compare experiments."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], prefix: str
) -> dict[str, float]:
    labels = list(range(len(classes)))
    metrics = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
    }
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for name, value in zip(classes, recalls, strict=True):
        metrics[f"{prefix}_recall_{name}"] = float(value)
    return metrics