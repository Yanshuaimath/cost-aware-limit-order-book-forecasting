from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lob_project.constants import FI2010_HORIZONS, N_LOB_FEATURES


@dataclass(frozen=True)
class FI2010Data:
    features: np.ndarray
    labels_by_horizon: np.ndarray

    def labels(self, horizon: int) -> np.ndarray:
        try:
            column = FI2010_HORIZONS.index(horizon)
        except ValueError as exc:
            raise ValueError(f"Unsupported horizon {horizon}; choose {FI2010_HORIZONS}") from exc
        labels = self.labels_by_horizon[:, column].astype(np.int64, copy=False)
        unique = set(np.unique(labels).tolist())
        if unique.issubset({1, 2, 3}):
            labels = labels - 1
        if not set(np.unique(labels).tolist()).issubset({0, 1, 2}):
            raise ValueError("Labels must be encoded as 1/2/3 or 0/1/2")
        return labels


def _orient_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, received shape={matrix.shape}")
    if matrix.shape[0] >= 45 and matrix.shape[0] <= 300 and matrix.shape[1] > matrix.shape[0]:
        return matrix
    if matrix.shape[1] >= 45 and matrix.shape[1] <= 300 and matrix.shape[0] > matrix.shape[1]:
        return matrix.T
    if matrix.shape[0] == 149:
        return matrix
    if matrix.shape[1] == 149:
        return matrix.T
    raise ValueError(
        "Could not infer FI-2010 orientation. Expected approximately 149 x N or N x 149."
    )


def load_fi2010_matrix(path: str | Path) -> FI2010Data:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    matrix = np.loadtxt(path, dtype=np.float32)
    matrix = _orient_matrix(matrix)
    if matrix.shape[0] < N_LOB_FEATURES + len(FI2010_HORIZONS):
        raise ValueError(f"Matrix has too few rows: {matrix.shape}")
    features = np.ascontiguousarray(matrix[:N_LOB_FEATURES].T, dtype=np.float32)
    labels = np.ascontiguousarray(matrix[-len(FI2010_HORIZONS):].T, dtype=np.int64)
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinite values")
    return FI2010Data(features=features, labels_by_horizon=labels)


class WindowedLOBDataset(Dataset):
    """Lazy rolling windows indexed by target observation.

    `target_indices` contain the absolute index of each label. The corresponding
    input is `[target-sequence_length+1, ..., target]`. No 3D sequence tensor is
    materialized, which keeps memory use practical for FI-2010-sized datasets.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        target_indices: np.ndarray,
        sequence_length: int,
    ) -> None:
        if features.ndim != 2 or features.shape[1] != N_LOB_FEATURES:
            raise ValueError(f"Expected features [N, 40], received {features.shape}")
        if len(features) != len(labels):
            raise ValueError("Features and labels must have equal length")
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        indices = np.asarray(target_indices, dtype=np.int64)
        if len(indices) and indices.min() < sequence_length - 1:
            raise ValueError("A target index does not have enough history")
        if len(indices) and indices.max() >= len(features):
            raise ValueError("A target index exceeds the feature array")
        self.features = features
        self.labels = labels
        self.target_indices = indices
        self.sequence_length = int(sequence_length)

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        target = int(self.target_indices[item])
        start = target - self.sequence_length + 1
        x = np.array(self.features[start : target + 1], dtype=np.float32, copy=True)
        y = int(self.labels[target])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
