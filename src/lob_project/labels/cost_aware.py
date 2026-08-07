from __future__ import annotations

import numpy as np


def cost_aware_direction_labels(
    best_ask: np.ndarray,
    best_bid: np.ndarray,
    horizon: int,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> np.ndarray:
    """Create down/stationary/up labels using executable round-trip prices.

    Up is feasible when selling at the future bid beats buying at the current
    ask after fees and slippage. Down is feasible under the symmetric short
    trade. Returned labels use 0=down, 1=stationary, 2=up.
    """
    ask = np.asarray(best_ask, dtype=np.float64)
    bid = np.asarray(best_bid, dtype=np.float64)
    if ask.shape != bid.shape or ask.ndim != 1:
        raise ValueError("best_ask and best_bid must be equal-length 1D arrays")
    if np.any(ask < bid):
        raise ValueError("Best ask must not be below best bid")
    if horizon <= 0 or horizon >= len(ask):
        raise ValueError("horizon must be positive and shorter than the series")

    labels = np.full(len(ask), 1, dtype=np.int64)
    cost_rate = (fee_bps + slippage_bps) * 2.0 / 10_000.0
    current_ask = ask[:-horizon]
    current_bid = bid[:-horizon]
    future_ask = ask[horizon:]
    future_bid = bid[horizon:]

    long_return = (future_bid - current_ask) / current_ask - cost_rate
    short_return = (current_bid - future_ask) / current_bid - cost_rate
    labels[:-horizon][long_return > 0] = 2
    labels[:-horizon][short_return > 0] = 0
    both = (long_return > 0) & (short_return > 0)
    labels[:-horizon][both] = np.where(long_return[both] >= short_return[both], 2, 0)
    return labels
