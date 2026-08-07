from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExecutableLabelResult:
    labels: np.ndarray
    long_returns: np.ndarray
    short_returns: np.ndarray
    valid_mask: np.ndarray


def _validate_prices(best_ask: np.ndarray, best_bid: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    ask = np.asarray(best_ask, dtype=np.float64)
    bid = np.asarray(best_bid, dtype=np.float64)
    if ask.ndim != 1 or bid.ndim != 1 or ask.shape != bid.shape:
        raise ValueError("best_ask and best_bid must be equal-length one-dimensional arrays")
    if horizon <= 0 or horizon >= len(ask):
        raise ValueError("horizon must be positive and shorter than the series")
    if not np.isfinite(ask).all() or not np.isfinite(bid).all():
        raise ValueError("prices contain non-finite values")
    if np.any(ask <= 0.0) or np.any(bid <= 0.0):
        raise ValueError("prices must be positive")
    if np.any(ask < bid):
        raise ValueError("crossed book detected")
    return ask, bid


def executable_action_labels(
    best_ask: np.ndarray,
    best_bid: np.ndarray,
    *,
    horizon: int,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    minimum_edge_bps: float = 0.0,
    group_ids: np.ndarray | None = None,
) -> ExecutableLabelResult:
    """Create short/flat/long labels from exact executable round-trip returns.

    Labels are encoded as ``0=short``, ``1=flat``, ``2=long``. The final
    ``horizon`` rows and observations whose exit crosses a group/day boundary
    are assigned ``-1`` and excluded from training.

    Fees and slippage are specified per transaction side. Fee treatment is
    multiplicative rather than the small-cost additive approximation.
    """

    ask, bid = _validate_prices(best_ask, best_bid, horizon)
    if fee_bps < 0.0 or slippage_bps < 0.0 or minimum_edge_bps < 0.0:
        raise ValueError("costs and minimum edge must be non-negative")

    n = len(ask)
    labels = np.full(n, -1, dtype=np.int64)
    long_returns = np.full(n, np.nan, dtype=np.float64)
    short_returns = np.full(n, np.nan, dtype=np.float64)
    valid_mask = np.zeros(n, dtype=bool)

    current = np.arange(n - horizon, dtype=np.int64)
    future = current + horizon
    valid = np.ones(len(current), dtype=bool)
    if group_ids is not None:
        groups = np.asarray(group_ids)
        if groups.ndim != 1 or len(groups) != n:
            raise ValueError("group_ids must be one-dimensional and match prices")
        valid &= groups[current] == groups[future]

    fee_rate = fee_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0
    minimum_edge = minimum_edge_bps / 10_000.0

    long_entry = ask[current] * (1.0 + slippage_rate)
    long_exit = bid[future] * (1.0 - slippage_rate)
    long_net = long_exit * (1.0 - fee_rate) / (long_entry * (1.0 + fee_rate)) - 1.0

    short_entry = bid[current] * (1.0 - slippage_rate)
    short_exit = ask[future] * (1.0 + slippage_rate)
    short_net = (
        short_entry * (1.0 - fee_rate) - short_exit * (1.0 + fee_rate)
    ) / short_entry

    valid_indices = current[valid]
    long_returns[valid_indices] = long_net[valid]
    short_returns[valid_indices] = short_net[valid]
    valid_mask[valid_indices] = True

    valid_long = long_net[valid]
    valid_short = short_net[valid]
    chosen = np.full(len(valid_indices), 1, dtype=np.int64)
    long_ok = valid_long > minimum_edge
    short_ok = valid_short > minimum_edge
    chosen[long_ok & (~short_ok | (valid_long >= valid_short))] = 2
    chosen[short_ok & (~long_ok | (valid_short > valid_long))] = 0
    labels[valid_indices] = chosen

    return ExecutableLabelResult(
        labels=labels,
        long_returns=long_returns,
        short_returns=short_returns,
        valid_mask=valid_mask,
    )


def point_midprice_direction_labels(
    best_ask: np.ndarray,
    best_bid: np.ndarray,
    *,
    horizon: int,
    threshold_bps: float,
    group_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Create exact-horizon down/stationary/up mid-price labels.

    Unlike the synthetic engineering target, this compares the mid-price at
    exactly ``t+horizon`` with the current mid-price. Invalid terminal or
    cross-day observations receive ``-1``.
    """

    ask, bid = _validate_prices(best_ask, best_bid, horizon)
    if threshold_bps < 0.0:
        raise ValueError("threshold_bps must be non-negative")
    n = len(ask)
    labels = np.full(n, -1, dtype=np.int64)
    current = np.arange(n - horizon, dtype=np.int64)
    future = current + horizon
    valid = np.ones(len(current), dtype=bool)
    if group_ids is not None:
        groups = np.asarray(group_ids)
        if groups.ndim != 1 or len(groups) != n:
            raise ValueError("group_ids must be one-dimensional and match prices")
        valid &= groups[current] == groups[future]

    mid = (ask + bid) / 2.0
    returns = mid[future] / mid[current] - 1.0
    threshold = threshold_bps / 10_000.0
    selected = np.full(int(valid.sum()), 1, dtype=np.int64)
    selected[returns[valid] > threshold] = 2
    selected[returns[valid] < -threshold] = 0
    labels[current[valid]] = selected
    return labels
