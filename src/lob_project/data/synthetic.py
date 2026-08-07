from __future__ import annotations

from pathlib import Path

import numpy as np

from lob_project.constants import FI2010_HORIZONS


def generate_synthetic_fi2010(
    n_observations: int = 8000,
    seed: int = 42,
    tick_size: float = 0.01,
) -> np.ndarray:
    """Generate an FI-2010-shaped matrix for software tests only.

    The generator creates a persistent latent pressure process so models have a
    learnable signal. It is deliberately unrealistic and must not be used for
    financial conclusions.
    """
    if n_observations < 1000:
        raise ValueError("Use at least 1000 observations for useful split sizes")
    rng = np.random.default_rng(seed)
    pressure = np.zeros(n_observations, dtype=np.float64)
    innovations = rng.normal(0.0, 1.0, n_observations)
    for t in range(1, n_observations):
        pressure[t] = 0.96 * pressure[t - 1] + 0.28 * innovations[t]

    returns = 0.00018 * np.tanh(pressure) + rng.normal(0.0, 0.00008, n_observations)
    mid = 100.0 * np.exp(np.cumsum(returns))
    spread_ticks = 1 + (np.abs(pressure) < 0.25).astype(int)
    spread = spread_ticks * tick_size

    rows = []
    for level in range(10):
        distance = spread / 2 + level * tick_size
        ask = mid + distance
        bid = mid - distance
        base_volume = 120.0 + 12.0 * level
        ask_volume = np.maximum(
            1.0, base_volume * np.exp(-0.20 * pressure + rng.normal(0, 0.22, n_observations))
        )
        bid_volume = np.maximum(
            1.0, base_volume * np.exp(0.20 * pressure + rng.normal(0, 0.22, n_observations))
        )
        rows.extend([ask, ask_volume, bid, bid_volume])

    matrix = np.zeros((149, n_observations), dtype=np.float32)
    matrix[:40] = np.asarray(rows, dtype=np.float32)

    for j, horizon in enumerate(FI2010_HORIZONS):
        future = np.full(n_observations, np.nan, dtype=np.float64)
        csum = np.concatenate([[0.0], np.cumsum(mid)])
        valid = np.arange(0, n_observations - horizon)
        future[valid] = (csum[valid + horizon + 1] - csum[valid + 1]) / horizon
        move = (future - mid) / mid
        threshold = 0.00008 + horizon * 0.0000012
        labels = np.full(n_observations, 2, dtype=np.int64)
        labels[move < -threshold] = 1
        labels[move > threshold] = 3
        labels[-horizon:] = 2
        matrix[-5 + j] = labels
    return matrix


def write_synthetic_fi2010(
    path: str | Path,
    n_observations: int = 8000,
    seed: int = 42,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = generate_synthetic_fi2010(n_observations=n_observations, seed=seed)
    np.savetxt(path, matrix, fmt="%.7g")
    return path
