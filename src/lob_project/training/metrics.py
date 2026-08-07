from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        include = (confidence > lower) & (confidence <= upper)
        if include.any():
            ece += include.mean() * abs(correct[include].mean() - confidence[include].mean())
    return float(ece)


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=[0, 1, 2], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "macro_f1": float(f1.mean()),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "ece_10": expected_calibration_error(y_true, probabilities, n_bins=10),
        "precision_by_class": precision.tolist(),
        "recall_by_class": recall.tolist(),
        "f1_by_class": f1.tolist(),
        "support_by_class": support.tolist(),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1, 2],
            target_names=["down", "stationary", "up"],
            zero_division=0,
            output_dict=True,
        ),
    }
