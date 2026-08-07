from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """Outputs from a single-position, fixed-horizon portfolio backtest."""

    metrics: dict[str, Any]
    trades: tuple[dict[str, Any], ...]
    equity_curve: tuple[dict[str, Any], ...]


def _validate_inputs(
    probabilities: np.ndarray,
    target_indices: np.ndarray,
    best_ask: np.ndarray,
    best_bid: np.ndarray,
    *,
    horizon: int,
    confidence: float,
    initial_equity: float,
    notional_per_trade: float,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    indices = np.asarray(target_indices, dtype=np.int64)
    ask = np.asarray(best_ask, dtype=np.float64)
    bid = np.asarray(best_bid, dtype=np.float64)

    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("probabilities must have shape (n_samples, 3)")
    if len(probabilities) != len(indices):
        raise ValueError("probabilities and target_indices must have equal length")
    if ask.ndim != 1 or bid.ndim != 1 or len(ask) != len(bid):
        raise ValueError("best_ask and best_bid must be equal-length one-dimensional arrays")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if initial_equity <= 0.0:
        raise ValueError("initial_equity must be positive")
    if notional_per_trade <= 0.0:
        raise ValueError("notional_per_trade must be positive")
    if fee_bps < 0.0 or slippage_bps < 0.0:
        raise ValueError("fee_bps and slippage_bps must be non-negative")
    if len(indices) and (indices.min() < 0 or indices.max() >= len(ask)):
        raise ValueError("target_indices contain values outside the price arrays")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("target_indices must be unique")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities contain non-finite values")
    if np.any(probabilities < 0.0):
        raise ValueError("probabilities must be non-negative")
    if len(probabilities) and not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("each probability row must sum to one")
    if not np.all(np.isfinite(ask)) or not np.all(np.isfinite(bid)):
        raise ValueError("price arrays contain non-finite values")
    if np.any(ask <= 0.0) or np.any(bid <= 0.0):
        raise ValueError("prices must be positive")
    if np.any(ask < bid):
        raise ValueError("crossed book detected: best ask is below best bid")

    order = np.argsort(indices, kind="stable")
    return probabilities[order], indices[order], ask, bid


def single_position_backtest(
    probabilities: np.ndarray,
    target_indices: np.ndarray,
    best_ask: np.ndarray,
    best_bid: np.ndarray,
    *,
    horizon: int,
    confidence: float = 0.50,
    initial_equity: float = 100_000.0,
    notional_per_trade: float = 10_000.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> PortfolioBacktestResult:
    """Backtest non-overlapping, fixed-horizon trades at executable prices.

    Class mapping is 0=short/down, 1=flat/stationary, 2=long/up.

    Assumptions:
    - At most one position is open.
    - A trade enters immediately at the best quote, worsened by slippage.
    - The trade exits exactly ``horizon`` events later at the opposite best quote.
    - Fees are charged on entry and exit traded notional.
    - New signals are ignored while a position is open.
    - Position notional is capped by current equity, so the simulation does not
      introduce leverage when equity falls below ``notional_per_trade``.

    The simulator does not model queue position, partial fills, market impact,
    funding, borrowing constraints, or exchange-specific liquidation rules.
    """
    probabilities, indices, ask, bid = _validate_inputs(
        probabilities,
        target_indices,
        best_ask,
        best_bid,
        horizon=horizon,
        confidence=confidence,
        initial_equity=initial_equity,
        notional_per_trade=notional_per_trade,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )

    fee_rate = fee_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0
    equity = float(initial_equity)
    next_available_index = -1

    counters = {
        "skipped_overlap": 0,
        "skipped_low_confidence": 0,
        "skipped_flat": 0,
        "skipped_missing_exit": 0,
        "skipped_insufficient_equity": 0,
    }
    trades: list[dict[str, Any]] = []
    first_curve_index = int(indices[0]) if len(indices) else 0
    equity_curve: list[dict[str, Any]] = [
        {
            "event_index": first_curve_index,
            "equity": equity,
            "drawdown": 0.0,
            "trade_id": -1,
        }
    ]

    for probability, signal_index in zip(probabilities, indices, strict=True):
        signal_index = int(signal_index)
        exit_index = signal_index + horizon

        if signal_index < next_available_index:
            counters["skipped_overlap"] += 1
            continue
        if exit_index >= len(ask):
            counters["skipped_missing_exit"] += 1
            continue

        predicted_class = int(np.argmax(probability))
        prediction_confidence = float(np.max(probability))
        if prediction_confidence < confidence:
            counters["skipped_low_confidence"] += 1
            continue
        if predicted_class == 1:
            counters["skipped_flat"] += 1
            continue

        trade_notional = min(float(notional_per_trade), equity)
        if trade_notional <= 0.0:
            counters["skipped_insufficient_equity"] += 1
            continue

        if predicted_class == 2:
            side = "long"
            entry_quote = float(ask[signal_index])
            exit_quote = float(bid[exit_index])
            entry_price = entry_quote * (1.0 + slippage_rate)
            exit_price = exit_quote * (1.0 - slippage_rate)
            quantity = trade_notional / entry_price
            gross_pnl = quantity * (exit_price - entry_price)
        else:
            side = "short"
            entry_quote = float(bid[signal_index])
            exit_quote = float(ask[exit_index])
            entry_price = entry_quote * (1.0 - slippage_rate)
            exit_price = exit_quote * (1.0 + slippage_rate)
            quantity = trade_notional / entry_price
            gross_pnl = quantity * (entry_price - exit_price)

        entry_fee = quantity * entry_price * fee_rate
        exit_fee = quantity * exit_price * fee_rate
        total_fees = entry_fee + exit_fee
        net_pnl = gross_pnl - total_fees
        net_return = net_pnl / trade_notional
        equity_before = equity
        equity += net_pnl
        next_available_index = exit_index

        trade_id = len(trades)
        trades.append(
            {
                "trade_id": trade_id,
                "side": side,
                "predicted_class": predicted_class,
                "confidence": prediction_confidence,
                "signal_index": signal_index,
                "exit_index": exit_index,
                "holding_events": horizon,
                "entry_quote": entry_quote,
                "entry_price": entry_price,
                "exit_quote": exit_quote,
                "exit_price": exit_price,
                "quantity": quantity,
                "notional": trade_notional,
                "gross_pnl": gross_pnl,
                "entry_fee": entry_fee,
                "exit_fee": exit_fee,
                "total_fees": total_fees,
                "net_pnl": net_pnl,
                "net_return": net_return,
                "equity_before": equity_before,
                "equity_after": equity,
            }
        )
        equity_curve.append(
            {
                "event_index": exit_index,
                "equity": equity,
                "drawdown": 0.0,
                "trade_id": trade_id,
            }
        )

    equity_values = np.asarray([point["equity"] for point in equity_curve], dtype=np.float64)
    running_peaks = np.maximum.accumulate(equity_values)
    drawdowns = equity_values / running_peaks - 1.0
    for point, drawdown in zip(equity_curve, drawdowns, strict=True):
        point["drawdown"] = float(drawdown)

    returns = np.asarray([trade["net_return"] for trade in trades], dtype=np.float64)
    pnls = np.asarray([trade["net_pnl"] for trade in trades], dtype=np.float64)
    notionals = np.asarray([trade["notional"] for trade in trades], dtype=np.float64)
    positive_pnl = pnls[pnls > 0.0].sum() if len(pnls) else 0.0
    negative_pnl = pnls[pnls < 0.0].sum() if len(pnls) else 0.0
    return_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    trade_sharpe = float(returns.mean() / return_std) if return_std > 0.0 else None
    profit_factor = (
        float(positive_pnl / abs(negative_pnl)) if negative_pnl < 0.0 else None
    )

    if trades:
        covered_events = trades[-1]["exit_index"] - trades[0]["signal_index"] + 1
        exposure = min(1.0, len(trades) * horizon / max(covered_events, 1))
    else:
        exposure = 0.0

    metrics: dict[str, Any] = {
        "n_signals": int(len(indices)),
        "n_trades": int(len(trades)),
        "long_trades": int(sum(trade["side"] == "long" for trade in trades)),
        "short_trades": int(sum(trade["side"] == "short" for trade in trades)),
        **counters,
        "initial_equity": float(initial_equity),
        "final_equity": float(equity),
        "net_pnl": float(equity - initial_equity),
        "total_return": float(equity / initial_equity - 1.0),
        "mean_pnl_per_trade": float(pnls.mean()) if len(pnls) else 0.0,
        "mean_return_per_trade": float(returns.mean()) if len(returns) else 0.0,
        "median_return_per_trade": float(np.median(returns)) if len(returns) else 0.0,
        "trade_return_std": return_std,
        "trade_return_sharpe_unannualized": trade_sharpe,
        "win_rate": float((pnls > 0.0).mean()) if len(pnls) else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown": float(-drawdowns.min()) if len(drawdowns) else 0.0,
        "turnover": float(notionals.sum() / initial_equity) if len(notionals) else 0.0,
        "event_exposure": float(exposure),
        "confidence": float(confidence),
        "horizon": int(horizon),
        "fee_bps_per_side": float(fee_bps),
        "slippage_bps_per_side": float(slippage_bps),
        "target_notional_per_trade": float(notional_per_trade),
        "assumptions": [
            "single position maximum",
            "fixed-horizon exit",
            "immediate full fill at top of book",
            "no market impact or queue-position model",
            "equity and drawdown are updated at trade exits only",
            "trade-level Sharpe is not annualized",
        ],
    }
    return PortfolioBacktestResult(
        metrics=metrics,
        trades=tuple(trades),
        equity_curve=tuple(equity_curve),
    )
