from __future__ import annotations

import numpy as np


def diagnostic_round_trip_returns(
    probabilities: np.ndarray,
    target_indices: np.ndarray,
    best_ask: np.ndarray,
    best_bid: np.ndarray,
    horizon: int,
    confidence: float = 0.50,
    fee_bps: float = 0.0,
) -> dict:
    """Evaluate independent fixed-horizon trades at executable top-of-book prices.

    This diagnostic permits overlapping trades and assumes immediate fills. It
    is useful for comparing signal quality but is not a production backtester.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    indices = np.asarray(target_indices, dtype=np.int64)
    ask = np.asarray(best_ask, dtype=np.float64)
    bid = np.asarray(best_bid, dtype=np.float64)
    if len(probabilities) != len(indices):
        raise ValueError("Probabilities and target indices must have equal length")
    valid = indices + horizon < len(ask)
    probabilities = probabilities[valid]
    indices = indices[valid]
    predictions = probabilities.argmax(axis=1)
    conf = probabilities.max(axis=1)
    active = (conf >= confidence) & (predictions != 1)

    pnl = np.zeros(len(indices), dtype=np.float64)
    long_mask = active & (predictions == 2)
    short_mask = active & (predictions == 0)
    fee_rate = 2.0 * fee_bps / 10_000.0
    pnl[long_mask] = (
        bid[indices[long_mask] + horizon] - ask[indices[long_mask]]
    ) / ask[indices[long_mask]] - fee_rate
    pnl[short_mask] = (
        bid[indices[short_mask]] - ask[indices[short_mask] + horizon]
    ) / bid[indices[short_mask]] - fee_rate

    trades = pnl[active]
    cumulative = np.cumsum(trades) if len(trades) else np.array([], dtype=np.float64)
    running_peak = np.maximum.accumulate(np.r_[0.0, cumulative])
    drawdown = np.r_[0.0, cumulative] - running_peak
    return {
        "n_signals": int(len(indices)),
        "n_trades": int(active.sum()),
        "trade_rate": float(active.mean()) if len(active) else 0.0,
        "mean_return_per_trade": float(trades.mean()) if len(trades) else 0.0,
        "hit_rate": float((trades > 0).mean()) if len(trades) else 0.0,
        "sum_independent_returns": float(trades.sum()),
        "max_drawdown_independent_returns": float(drawdown.min()),
        "returns": trades,
    }
