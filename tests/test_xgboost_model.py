from __future__ import annotations

import numpy as np
import pytest

from lob_project.models.xgboost_model import (
    class_balanced_sample_weights,
    make_xgboost,
)


def test_class_balanced_weights_equalize_total_class_weight() -> None:
    labels = np.array([0, 0, 0, 1, 2, 2])
    weights = class_balanced_sample_weights(labels)
    totals = [weights[labels == label].sum() for label in range(3)]
    assert np.allclose(totals, totals[0])


def test_xgboost_probability_shape() -> None:
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(42)
    x = rng.normal(size=(60, 8))
    y = np.repeat(np.arange(3), 20)
    model = make_xgboost(n_estimators=5, max_depth=2, n_jobs=1)
    model.fit(x, y, sample_weight=class_balanced_sample_weights(y))
    probabilities = model.predict_proba(x[:4])
    assert probabilities.shape == (4, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
