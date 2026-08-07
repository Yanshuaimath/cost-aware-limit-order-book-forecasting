from __future__ import annotations

import numpy as np

from lob_project.features.microstructure import microstructure_features


def test_microstructure_features_are_finite_and_interpretable() -> None:
    features = np.zeros((5, 40), dtype=np.float32)
    for level in range(10):
        features[:, 4 * level] = 100.01 + 0.01 * level
        features[:, 4 * level + 1] = 2.0
        features[:, 4 * level + 2] = 99.99 - 0.01 * level
        features[:, 4 * level + 3] = 3.0
    output, names = microstructure_features(features)
    assert output.shape == (5, len(names))
    assert "imbalance_10" in names
    assert "microprice" in names
    assert np.isfinite(output).all()
    imbalance = output[:, names.index("imbalance_1")]
    assert np.allclose(imbalance, 0.2)
