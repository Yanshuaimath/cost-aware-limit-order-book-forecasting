#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate one model across saved folds")
    parser.add_argument("--model", choices=["baseline", "xgboost", "deeplob", "transformer"], required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-n-jobs", type=int, default=-1)
    parser.add_argument("--run-backtest", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--initial-equity", type=float, default=100_000.0)
    parser.add_argument("--notional-per-trade", type=float, default=10_000.0)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=0.5)
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    fold_files = sorted(args.folds_dir.glob("fold_*.npz"))
    if not fold_files:
        raise SystemExit(f"No fold_*.npz files found in {args.folds_dir}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold_file in fold_files:
        fold_name = fold_file.stem
        run_dir = args.output_root / fold_name
        train_command = [
            sys.executable,
            str(script_dir / "train_model.py"),
            "--model",
            args.model,
            "--processed-dir",
            str(args.processed_dir),
            "--split-file",
            str(fold_file),
            "--output-dir",
            str(run_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
        ]
        if args.model == "xgboost":
            train_command.extend(
                [
                    "--xgb-estimators",
                    str(args.xgb_estimators),
                    "--xgb-max-depth",
                    str(args.xgb_max_depth),
                    "--xgb-learning-rate",
                    str(args.xgb_learning_rate),
                    "--xgb-n-jobs",
                    str(args.xgb_n_jobs),
                ]
            )
        _run(train_command)
        _run(
            [
                sys.executable,
                str(script_dir / "evaluate_model.py"),
                "--run-dir",
                str(run_dir),
                "--processed-dir",
                str(args.processed_dir),
                "--split-file",
                str(fold_file),
            ]
        )
        metrics = json.loads((run_dir / "test_metrics.json").read_text())
        row = {"fold": fold_name, **{key: value for key, value in metrics.items() if isinstance(value, (int, float))}}

        if args.run_backtest:
            backtest_dir = run_dir / "portfolio_backtest"
            _run(
                [
                    sys.executable,
                    str(script_dir / "run_portfolio_backtest.py"),
                    "--run-dir",
                    str(run_dir),
                    "--processed-dir",
                    str(args.processed_dir),
                    "--output-dir",
                    str(backtest_dir),
                    "--confidence",
                    str(args.confidence),
                    "--initial-equity",
                    str(args.initial_equity),
                    "--notional-per-trade",
                    str(args.notional_per_trade),
                    "--fee-bps",
                    str(args.fee_bps),
                    "--slippage-bps",
                    str(args.slippage_bps),
                ]
            )
            backtest_metrics = json.loads((backtest_dir / "metrics.json").read_text())
            for key in ("n_trades", "net_pnl", "total_return", "max_drawdown", "win_rate", "profit_factor"):
                row[f"backtest_{key}"] = backtest_metrics.get(key)
        rows.append(row)

    (args.output_root / "walk_forward_metrics.json").write_text(json.dumps(rows, indent=2))
    fieldnames = sorted({key for row in rows for key in row})
    with (args.output_root / "walk_forward_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote aggregated metrics to {args.output_root}")


if __name__ == "__main__":
    main()
