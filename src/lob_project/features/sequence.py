from __future__ import annotations

import numpy as np


def summarize_windows(
    features: np.ndarray,
    target_indices: np.ndarray,
    sequence_length: int,
    max_samples: int | None = None,
) -> np.ndarray:
    """Create interpretable sequence summaries for classical models.

    Per LOB feature, the output contains last value, mean, standard deviation,
    and change from first to last observation: 40 x 4 = 160 features.
    """
    indices = np.asarray(target_indices, dtype=np.int64)
    if max_samples is not None and len(indices) > max_samples:
        positions = np.linspace(0, len(indices) - 1, max_samples, dtype=np.int64)
        indices = indices[positions]
    result = np.empty((len(indices), features.shape[1] * 4), dtype=np.float32)
    for row, target in enumerate(indices):
        window = np.asarray(
            features[target - sequence_length + 1 : target + 1], dtype=np.float32
        )
        result[row] = np.concatenate(
            [window[-1], window.mean(axis=0), window.std(axis=0), window[-1] - window[0]]
        )
    return result


def subsample_indices(indices: np.ndarray, max_samples: int | None) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if max_samples is None or len(indices) <= max_samples:
        return indices
    positions = np.linspace(0, len(indices) - 1, max_samples, dtype=np.int64)
    return indices[positions]
