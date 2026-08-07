#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lob_project.backtest.execution import diagnostic_round_trip_returns
from lob_project.data.processed import load_processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run executable-price diagnostic backtest")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_processed(args.processed_dir)
    if not data.metadata.get("executable_prices", False):
        raise SystemExit(
            "Backtest refused: metadata does not assert raw executable bid/ask prices. "
            "Pre-normalized FI-2010 is classification-only."
        )
    source = np.load(args.processed_dir / "source_features.npy", mmap_mode="r")
    best_ask = source[:, int(data.metadata["price_columns"]["best_ask"])]
    best_bid = source[:, int(data.metadata["price_columns"]["best_bid"])]
    predictions = np.load(args.run_dir / "test_predictions.npz")
    result = diagnostic_round_trip_returns(
        predictions["probabilities"],
        predictions["target_indices"],
        best_ask,
        best_bid,
        horizon=int(data.metadata["horizon"]),
        confidence=args.confidence,
        fee_bps=args.fee_bps,
    )
    returns = result.pop("returns")
    np.save(args.run_dir / "diagnostic_trade_returns.npy", returns)
    result.update(
        {
            "confidence": args.confidence,
            "fee_bps": args.fee_bps,
            "warning": "Independent overlapping fixed-horizon trades; immediate fills assumed.",
        }
    )
    (args.run_dir / "backtest_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
