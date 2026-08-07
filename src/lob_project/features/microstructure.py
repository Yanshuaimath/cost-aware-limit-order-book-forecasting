from __future__ import annotations

import numpy as np

from lob_project.constants import N_LOB_FEATURES


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(values)
    if window <= 1:
        return result
    csum = np.r_[0.0, np.cumsum(values)]
    csum_sq = np.r_[0.0, np.cumsum(values * values)]
    for end in range(1, len(values) + 1):
        start = max(0, end - window)
        count = end - start
        mean = (csum[end] - csum[start]) / count
        variance = (csum_sq[end] - csum_sq[start]) / count - mean * mean
        result[end - 1] = np.sqrt(max(variance, 0.0))
    return result


def microstructure_features(features: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Create interpretable per-snapshot features from canonical ten-level LOB data."""

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != N_LOB_FEATURES:
        raise ValueError(f"Expected features with shape [N, 40], received {matrix.shape}")
    book = matrix.reshape(len(matrix), 10, 4)
    ask_price = book[:, :, 0]
    ask_size = book[:, :, 1]
    bid_price = book[:, :, 2]
    bid_size = book[:, :, 3]

    best_ask = ask_price[:, 0]
    best_bid = bid_price[:, 0]
    mid = (best_ask + best_bid) / 2.0
    spread = best_ask - best_bid
    relative_spread = spread / mid
    top_depth = ask_size[:, 0] + bid_size[:, 0]
    microprice = np.divide(
        best_ask * bid_size[:, 0] + best_bid * ask_size[:, 0],
        top_depth,
        out=mid.copy(),
        where=top_depth > 0.0,
    )

    columns: list[np.ndarray] = [
        spread,
        relative_spread,
        mid,
        microprice,
        (microprice - mid) / mid,
    ]
    names = [
        "spread",
        "relative_spread",
        "midprice",
        "microprice",
        "microprice_deviation",
    ]

    for depth in (1, 5, 10):
        total_ask = ask_size[:, :depth].sum(axis=1)
        total_bid = bid_size[:, :depth].sum(axis=1)
        total = total_ask + total_bid
        imbalance = np.divide(
            total_bid - total_ask,
            total,
            out=np.zeros_like(total),
            where=total > 0.0,
        )
        columns.extend([total_ask, total_bid, imbalance])
        names.extend(
            [
                f"ask_depth_{depth}",
                f"bid_depth_{depth}",
                f"imbalance_{depth}",
            ]
        )

    log_mid = np.log(mid)
    log_return = np.r_[0.0, np.diff(log_mid)]
    spread_change = np.r_[0.0, np.diff(relative_spread)]
    imbalance_1 = columns[names.index("imbalance_1")]
    imbalance_change = np.r_[0.0, np.diff(imbalance_1)]
    columns.extend(
        [
            log_return,
            spread_change,
            imbalance_change,
            _rolling_std(log_return, 4),
            _rolling_std(log_return, 20),
        ]
    )
    names.extend(
        [
            "log_mid_return_1",
            "relative_spread_change_1",
            "imbalance_1_change_1",
            "realized_volatility_4",
            "realized_volatility_20",
        ]
    )

    output = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("engineered microstructure features contain non-finite values")
    return output, tuple(names)
