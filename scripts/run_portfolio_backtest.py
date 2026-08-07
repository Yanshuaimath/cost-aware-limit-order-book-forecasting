#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from lob_project.backtest.portfolio import single_position_backtest
from lob_project.data.processed import load_processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single-position executable-price portfolio backtest"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to RUN_DIR/portfolio_backtest",
    )
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--initial-equity", type=float, default=100_000.0)
    parser.add_argument("--notional-per-trade", type=float, default=10_000.0)
    parser.add_argument("--fee-bps", type=float, default=0.0, help="Per side")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="Per side")
    return parser.parse_args()


def write_csv(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data = load_processed(args.processed_dir)
    if not data.metadata.get("executable_prices", False):
        raise SystemExit(
            "Portfolio backtest refused: metadata does not assert raw executable "
            "bid/ask prices. Pre-normalized FI-2010 is classification-only."
        )

    source_path = args.processed_dir / "source_features.npy"
    prediction_path = args.run_dir / "test_predictions.npz"
    if not source_path.exists():
        raise SystemExit(f"Missing executable source prices: {source_path}")
    if not prediction_path.exists():
        raise SystemExit(
            f"Missing model predictions: {prediction_path}. Run evaluate_model.py first."
        )

    source = np.load(source_path, mmap_mode="r")
    best_ask = source[:, int(data.metadata["price_columns"]["best_ask"])]
    best_bid = source[:, int(data.metadata["price_columns"]["best_bid"])]
    predictions = np.load(prediction_path)

    result = single_position_backtest(
        probabilities=predictions["probabilities"],
        target_indices=predictions["target_indices"],
        best_ask=best_ask,
        best_bid=best_bid,
        horizon=int(data.metadata["horizon"]),
        confidence=args.confidence,
        initial_equity=args.initial_equity,
        notional_per_trade=args.notional_per_trade,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    output_dir = args.output_dir or args.run_dir / "portfolio_backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
    write_csv(output_dir / "trades.csv", result.trades)
    write_csv(output_dir / "equity_curve.csv", result.equity_curve)
    np.savez(
        output_dir / "backtest_arrays.npz",
        net_returns=np.asarray([trade["net_return"] for trade in result.trades]),
        net_pnl=np.asarray([trade["net_pnl"] for trade in result.trades]),
        equity=np.asarray([point["equity"] for point in result.equity_curve]),
        event_indices=np.asarray(
            [point["event_index"] for point in result.equity_curve], dtype=np.int64
        ),
    )

    print(json.dumps(result.metrics, indent=2))
    print(f"Wrote portfolio backtest outputs to {output_dir}")


if __name__ == "__main__":
    main()
