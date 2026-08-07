from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_baseline(max_iter: int = 1000, seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=None,
                ),
            ),
        ]
    )
