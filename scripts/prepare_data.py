#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lob_project.data.fi2010 import load_fi2010_matrix
from lob_project.data.split import assert_no_window_overlap, purged_chronological_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FI-2010-format data")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=[10, 20, 30, 50, 100], default=50)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--normalize", choices=["none", "train"], default="none")
    parser.add_argument(
        "--executable-prices",
        action="store_true",
        help="Assert that columns contain raw executable prices; false for pre-normalized FI-2010.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_fi2010_matrix(args.input)
    labels = data.labels(args.horizon)
    splits = purged_chronological_split(
        len(data.features), args.sequence_length, args.horizon
    )
    assert_no_window_overlap(splits.train, splits.validation, args.sequence_length)
    assert_no_window_overlap(splits.validation, splits.test, args.sequence_length)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    source_features = np.asarray(data.features, dtype=np.float32)
    features = source_features.copy()
    scaler = None
    if args.normalize == "train":
        train_raw_stop = splits.train_boundary
        mean = source_features[:train_raw_stop].mean(axis=0, dtype=np.float64)
        std = source_features[:train_raw_stop].std(axis=0, dtype=np.float64)
        std[std < 1e-8] = 1.0
        features = ((source_features - mean) / std).astype(np.float32)
        scaler = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
        np.savez(output / "scaler.npz", **scaler)

    np.save(output / "features.npy", features)
    np.save(output / "labels.npy", labels)
    np.save(output / "source_features.npy", source_features)
    np.savez(
        output / "splits.npz",
        train=splits.train,
        validation=splits.validation,
        test=splits.test,
    )
    metadata = {
        "source": str(args.input),
        "n_observations": int(len(features)),
        "n_features": int(features.shape[1]),
        "horizon": args.horizon,
        "sequence_length": args.sequence_length,
        "normalization": args.normalize,
        "executable_prices": bool(args.executable_prices),
        "train_boundary": splits.train_boundary,
        "validation_boundary": splits.validation_boundary,
        "purge_size": splits.purge_size,
        "split_sizes": {
            "train": int(len(splits.train)),
            "validation": int(len(splits.validation)),
            "test": int(len(splits.test)),
        },
        "label_mapping": {"0": "down", "1": "stationary", "2": "up"},
        "price_columns": {"best_ask": 0, "best_bid": 2},
        "warning": (
            "Executable-price analysis enabled." if args.executable_prices else
            "Source prices are not asserted executable; do not run transaction-cost backtests."
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
