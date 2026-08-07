from pathlib import Path

import numpy as np

from lob_project.data.fi2010 import WindowedLOBDataset, load_fi2010_matrix
from lob_project.data.synthetic import generate_synthetic_fi2010


def test_loader_accepts_both_orientations(tmp_path: Path) -> None:
    matrix = generate_synthetic_fi2010(n_observations=1200, seed=1)
    path_a = tmp_path / "a.txt"
    path_b = tmp_path / "b.txt"
    np.savetxt(path_a, matrix)
    np.savetxt(path_b, matrix.T)
    a = load_fi2010_matrix(path_a)
    b = load_fi2010_matrix(path_b)
    assert a.features.shape == (1200, 40)
    np.testing.assert_allclose(a.features, b.features)
    assert set(np.unique(a.labels(50))).issubset({0, 1, 2})


def test_lazy_dataset_window_shape() -> None:
    features = np.arange(300 * 40, dtype=np.float32).reshape(300, 40)
    labels = np.arange(300) % 3
    dataset = WindowedLOBDataset(features, labels, np.array([99, 120]), 100)
    x, y = dataset[1]
    assert x.shape == (100, 40)
    assert y.item() == labels[120]
    np.testing.assert_array_equal(x[0].numpy(), features[21])
