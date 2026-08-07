#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lob_project.data.processed import load_processed
from lob_project.data.walk_forward import (
    expanding_walk_forward_folds,
    valid_targets_within_days,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create expanding day-based walk-forward folds")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-train-days", type=int, default=5)
    parser.add_argument("--validation-days", type=int, default=1)
    parser.add_argument("--test-days", type=int, default=1)
    parser.add_argument("--step-days", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_processed(args.processed_dir)
    day_path = args.processed_dir / "day_ids.npy"
    if not day_path.exists():
        raise SystemExit("day_ids.npy is required; use prepare_btc_lob.py first")
    day_ids = np.load(day_path, mmap_mode="r")
    valid_targets = valid_targets_within_days(
        day_ids,
        sequence_length=int(data.metadata["sequence_length"]),
        horizon=int(data.metadata["horizon"]),
        label_valid_mask=np.asarray(data.labels) >= 0,
    )
    folds = expanding_walk_forward_folds(
        day_ids,
        valid_targets=valid_targets,
        minimum_train_days=args.minimum_train_days,
        validation_days=args.validation_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for fold in folds:
        path = args.output_dir / f"fold_{fold.fold:02d}.npz"
        np.savez(path, train=fold.train, validation=fold.validation, test=fold.test)
        summary.append(
            {
                "fold": fold.fold,
                "split_file": str(path),
                "train_days": list(fold.train_days),
                "validation_days": list(fold.validation_days),
                "test_days": list(fold.test_days),
                "sizes": {
                    "train": int(len(fold.train)),
                    "validation": int(len(fold.validation)),
                    "test": int(len(fold.test)),
                },
            }
        )
    (args.output_dir / "folds.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
