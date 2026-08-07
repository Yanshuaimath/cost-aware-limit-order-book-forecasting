from __future__ import annotations

import numpy as np


def class_balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    counts = np.bincount(values, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"all three classes must be present; counts={counts.astype(int).tolist()}")
    class_weights = len(values) / (3.0 * counts)
    return class_weights[values].astype(np.float32)


def make_xgboost(
    *,
    seed: int = 42,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    n_jobs: int = -1,
):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depends on optional runtime installation
        raise RuntimeError("XGBoost is not installed. Run: uv add xgboost") from exc

    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=1.0,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=n_jobs,
    )
