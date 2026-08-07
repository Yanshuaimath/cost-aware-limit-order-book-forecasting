#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lob_project.data.bitcoin import load_raw_lob_table
from lob_project.data.walk_forward import fixed_day_split, valid_targets_within_days
from lob_project.features.microstructure import microstructure_features
from lob_project.labels.executable import (
    executable_action_labels,
    point_midprice_direction_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare raw ten-level BTC limit-order-book data")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--column-map", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--drop-duplicate-timestamps", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=4, help="Snapshot horizon")
    parser.add_argument("--label-type", choices=["midprice", "cost-aware"], default="cost-aware")
    parser.add_argument("--threshold-bps", type=float, default=1.0)
    parser.add_argument("--fee-bps", type=float, default=1.0, help="Per side")
    parser.add_argument("--slippage-bps", type=float, default=0.5, help="Per side")
    parser.add_argument("--minimum-edge-bps", type=float, default=0.0)
    parser.add_argument("--train-days", type=int, default=8)
    parser.add_argument("--validation-days", type=int, default=2)
    parser.add_argument("--normalize", choices=["none", "train"], default="train")
    return parser.parse_args()


def _normalise_from_training_rows(
    values: np.ndarray,
    training_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[training_rows].mean(axis=0, dtype=np.float64)
    std = values[training_rows].std(axis=0, dtype=np.float64)
    std[std < 1e-8] = 1.0
    normalised = ((values - mean) / std).astype(np.float32)
    return normalised, mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    args = parse_args()
    input_name = args.input.name.lower()
    is_csv = args.input.suffix.lower() in {".csv", ".txt", ".gz"} or input_name.endswith(".csv.gz")
    if (
        is_csv
        and args.max_rows is None
        and args.input.exists()
        and args.input.stat().st_size >= 750_000_000
    ):
        raise RuntimeError(
            "This CSV is too large for the in-memory preparer. "
            "Use scripts/prepare_btc_lob_low_memory.py, which reads chunks and writes "
            "memory-mapped NumPy arrays."
        )
    raw = load_raw_lob_table(
        args.input,
        column_map=args.column_map,
        max_rows=args.max_rows,
        drop_duplicate_timestamps=args.drop_duplicate_timestamps,
    )
    source_features = raw.features
    best_ask = source_features[:, 0]
    best_bid = source_features[:, 2]

    label_returns = None
    if args.label_type == "cost-aware":
        result = executable_action_labels(
            best_ask,
            best_bid,
            horizon=args.horizon,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            minimum_edge_bps=args.minimum_edge_bps,
            group_ids=raw.day_ids,
        )
        labels = result.labels
        label_valid_mask = result.valid_mask
        label_returns = result
    else:
        labels = point_midprice_direction_labels(
            best_ask,
            best_bid,
            horizon=args.horizon,
            threshold_bps=args.threshold_bps,
            group_ids=raw.day_ids,
        )
        label_valid_mask = labels >= 0

    valid_targets = valid_targets_within_days(
        raw.day_ids,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        label_valid_mask=label_valid_mask,
    )
    split = fixed_day_split(
        raw.day_ids,
        valid_targets=valid_targets,
        n_train_days=args.train_days,
        n_validation_days=args.validation_days,
    )
    if min(len(split.train), len(split.validation), len(split.test)) == 0:
        raise ValueError("one data split is empty")

    train_day_mask = np.isin(raw.day_ids, split.train_days)
    micro_raw, micro_names = microstructure_features(source_features)
    features = source_features.copy()
    tabular_features = micro_raw.copy()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    if args.normalize == "train":
        features, feature_mean, feature_std = _normalise_from_training_rows(
            source_features, train_day_mask
        )
        tabular_features, tabular_mean, tabular_std = _normalise_from_training_rows(
            micro_raw, train_day_mask
        )
        np.savez(output / "scaler.npz", mean=feature_mean, std=feature_std)
        np.savez(
            output / "tabular_scaler.npz", mean=tabular_mean, std=tabular_std
        )

    np.save(output / "features.npy", features.astype(np.float32))
    np.save(output / "source_features.npy", source_features.astype(np.float32))
    np.save(output / "labels.npy", labels.astype(np.int64))
    np.save(output / "timestamps.npy", raw.timestamps)
    np.save(output / "day_ids.npy", raw.day_ids)
    np.save(output / "tabular_features.npy", tabular_features.astype(np.float32))
    (output / "tabular_feature_names.json").write_text(json.dumps(micro_names, indent=2))
    np.savez(
        output / "splits.npz",
        train=split.train,
        validation=split.validation,
        test=split.test,
    )
    if label_returns is not None:
        np.savez(
            output / "label_returns.npz",
            long_returns=label_returns.long_returns,
            short_returns=label_returns.short_returns,
            valid_mask=label_returns.valid_mask,
        )

    class_counts = {
        name: np.bincount(labels[indices], minlength=3).astype(int).tolist()
        for name, indices in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
    }
    metadata = {
        "source": str(args.input),
        "dataset_type": "raw_lob_snapshots",
        "n_observations": int(len(source_features)),
        "n_features": int(source_features.shape[1]),
        "n_days": int(len(raw.day_labels)),
        "day_labels": list(raw.day_labels),
        "horizon": int(args.horizon),
        "sequence_length": int(args.sequence_length),
        "label_type": args.label_type,
        "threshold_bps": float(args.threshold_bps),
        "fee_bps_per_side": float(args.fee_bps),
        "slippage_bps_per_side": float(args.slippage_bps),
        "minimum_edge_bps": float(args.minimum_edge_bps),
        "normalization": args.normalize,
        "executable_prices": True,
        "price_columns": {"best_ask": 0, "best_bid": 2},
        "train_days": list(split.train_days),
        "validation_days": list(split.validation_days),
        "test_days": list(split.test_days),
        "split_sizes": {
            "train": int(len(split.train)),
            "validation": int(len(split.validation)),
            "test": int(len(split.test)),
        },
        "class_counts_down_flat_up": class_counts,
        "label_mapping": {"0": "short/down", "1": "flat/stationary", "2": "long/up"},
        "resolved_columns": raw.resolved_columns,
        "warning": (
            "Historical top-of-book execution research only; immediate fills, queue position, "
            "market impact, funding, and exchange-specific rules remain modeling assumptions."
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
