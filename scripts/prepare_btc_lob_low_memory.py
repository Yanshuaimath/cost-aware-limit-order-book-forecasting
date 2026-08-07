#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from lob_project.data.bitcoin import load_raw_lob_table
from lob_project.data.walk_forward import fixed_day_split, valid_targets_within_days
from lob_project.labels.executable import (
    executable_action_labels,
    point_midprice_direction_labels,
)


TABULAR_NAMES = (
    "spread",
    "relative_spread",
    "midprice",
    "microprice",
    "microprice_deviation",
    "ask_depth_1",
    "bid_depth_1",
    "imbalance_1",
    "ask_depth_5",
    "bid_depth_5",
    "imbalance_5",
    "ask_depth_10",
    "bid_depth_10",
    "imbalance_10",
    "log_mid_return_1",
    "relative_spread_change_1",
    "imbalance_1_change_1",
    "realized_volatility_4",
    "realized_volatility_20",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Memory-safe preparation of a large raw BTC LOB CSV"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--column-map", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument(
        "--label-type", choices=["midprice", "cost-aware"], default="cost-aware"
    )
    parser.add_argument("--threshold-bps", type=float, default=1.0)
    parser.add_argument("--fee-bps", type=float, default=1.0, help="Per side")
    parser.add_argument("--slippage-bps", type=float, default=0.5, help="Per side")
    parser.add_argument("--minimum-edge-bps", type=float, default=0.0)
    parser.add_argument("--train-days", type=int, default=8)
    parser.add_argument("--validation-days", type=int, default=2)
    parser.add_argument("--normalize", choices=["none", "train"], default="train")
    return parser.parse_args()


def _count_csv_rows(path: Path) -> int:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    count = 0
    with opener(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            count += block.count(b"\n")
    # One header line. Support a final line without a newline.
    with opener(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size == 0:
            return 0
        handle.seek(-1, os.SEEK_END)
        has_final_newline = handle.read(1) == b"\n"
    data_rows = count - 1 + (0 if has_final_newline else 1)
    return max(data_rows, 0)


def _parse_timestamp_series(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
        finite = numeric[np.isfinite(numeric)]
        if finite.size == 0:
            raise ValueError("timestamp column contains no finite values")
        magnitude = float(np.median(np.abs(finite)))
        if magnitude >= 1e17:
            unit = "ns"
        elif magnitude >= 1e14:
            unit = "us"
        elif magnitude >= 1e11:
            unit = "ms"
        else:
            unit = "s"
        parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
    if parsed.isna().any():
        raise ValueError(f"timestamp parsing failed for {int(parsed.isna().sum())} rows")
    return parsed.to_numpy(dtype="datetime64[ns]")


def _validate_feature_chunk(values: np.ndarray) -> None:
    if values.ndim != 2 or values.shape[1] != 40:
        raise ValueError(f"Expected [N, 40] features, received {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("LOB features contain missing or non-finite values")
    ask_prices = values[:, 0::4]
    ask_sizes = values[:, 1::4]
    bid_prices = values[:, 2::4]
    bid_sizes = values[:, 3::4]
    if np.any(ask_prices <= 0.0) or np.any(bid_prices <= 0.0):
        raise ValueError("LOB prices must be positive")
    if np.any(ask_sizes < 0.0) or np.any(bid_sizes < 0.0):
        raise ValueError("LOB quantities must be non-negative")
    if np.any(ask_prices[:, 0] < bid_prices[:, 0]):
        raise ValueError("crossed book detected: best ask is below best bid")
    if np.any(np.diff(ask_prices, axis=1) < -1e-6):
        raise ValueError("ask prices must be non-decreasing with depth")
    if np.any(np.diff(bid_prices, axis=1) > 1e-6):
        raise ValueError("bid prices must be non-increasing with depth")


def _write_raw_arrays(
    *,
    path: Path,
    output: Path,
    resolved: dict[str, str],
    n_rows: int,
    chunksize: int,
) -> None:
    timestamp_column = resolved["timestamp"]
    feature_columns: list[str] = []
    for level in range(1, 11):
        for side, field in (
            ("ask", "price"),
            ("ask", "size"),
            ("bid", "price"),
            ("bid", "size"),
        ):
            feature_columns.append(resolved[f"{side}_{field}_{level}"])

    selected_columns = list(dict.fromkeys([timestamp_column, *feature_columns]))
    source = np.lib.format.open_memmap(
        output / "source_features.npy", mode="w+", dtype=np.float32, shape=(n_rows, 40)
    )
    timestamps = np.lib.format.open_memmap(
        output / "timestamps.npy",
        mode="w+",
        dtype="datetime64[ns]",
        shape=(n_rows,),
    )

    row = 0
    previous_timestamp: np.datetime64 | None = None
    reader = pd.read_csv(
        path,
        usecols=selected_columns,
        chunksize=chunksize,
        nrows=n_rows,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        chunk_timestamps = _parse_timestamp_series(chunk[timestamp_column])
        chunk_features = chunk[feature_columns].to_numpy(dtype=np.float32, copy=True)
        _validate_feature_chunk(chunk_features)

        if len(chunk_timestamps) > 1 and np.any(chunk_timestamps[1:] <= chunk_timestamps[:-1]):
            raise ValueError(
                "timestamps must be strictly increasing; duplicate or unsorted timestamps found"
            )
        if previous_timestamp is not None and chunk_timestamps[0] <= previous_timestamp:
            raise ValueError(
                "timestamps must be strictly increasing across CSV chunks"
            )
        previous_timestamp = chunk_timestamps[-1]

        stop = row + len(chunk)
        source[row:stop] = chunk_features
        timestamps[row:stop] = chunk_timestamps
        row = stop
        print(f"loaded chunk {chunk_number}: {row:,}/{n_rows:,} rows", flush=True)

    if row != n_rows:
        raise RuntimeError(f"Expected {n_rows:,} rows but loaded {row:,}")
    source.flush()
    timestamps.flush()


def _day_ids_from_timestamps(
    timestamps: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    days = np.asarray(timestamps).astype("datetime64[D]")
    if len(days) == 0:
        raise ValueError("no timestamps")
    changes = np.r_[True, days[1:] != days[:-1]]
    day_ids = np.cumsum(changes, dtype=np.int32) - 1
    labels = tuple(np.datetime_as_string(days[changes], unit="D").tolist())
    return day_ids.astype(np.int32, copy=False), labels


def _chunked_mean_std(
    values: np.ndarray,
    training_mask: np.ndarray,
    *,
    chunksize: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(values.shape[1], dtype=np.float64)
    total_sq = np.zeros(values.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(values), chunksize):
        stop = min(start + chunksize, len(values))
        mask = training_mask[start:stop]
        if not np.any(mask):
            continue
        chunk = np.asarray(values[start:stop][mask], dtype=np.float64)
        total += chunk.sum(axis=0, dtype=np.float64)
        total_sq += np.square(chunk).sum(axis=0, dtype=np.float64)
        count += len(chunk)
    if count == 0:
        raise ValueError("no training rows available for normalization")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _normalise_to_file(
    values: np.ndarray,
    destination: Path,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    chunksize: int,
) -> None:
    output = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float32,
        shape=values.shape,
    )
    for start in range(0, len(values), chunksize):
        stop = min(start + chunksize, len(values))
        chunk = np.asarray(values[start:stop], dtype=np.float32)
        output[start:stop] = (chunk - mean) / std
    output.flush()


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    ends = np.arange(1, n + 1, dtype=np.int64)
    starts = np.maximum(0, ends - window)
    csum = np.r_[0.0, np.cumsum(x, dtype=np.float64)]
    csum_sq = np.r_[0.0, np.cumsum(x * x, dtype=np.float64)]
    counts = ends - starts
    mean = (csum[ends] - csum[starts]) / counts
    variance = (csum_sq[ends] - csum_sq[starts]) / counts - mean * mean
    return np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)


def _write_tabular_raw(
    source: np.ndarray,
    day_ids: np.ndarray,
    destination: Path,
    *,
    chunksize: int,
) -> np.ndarray:
    output = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float32,
        shape=(len(source), len(TABULAR_NAMES)),
    )
    for start in range(0, len(source), chunksize):
        stop = min(start + chunksize, len(source))
        book = np.asarray(source[start:stop], dtype=np.float32).reshape(-1, 10, 4)
        ask_price = book[:, :, 0]
        ask_size = book[:, :, 1]
        bid_price = book[:, :, 2]
        bid_size = book[:, :, 3]
        best_ask = ask_price[:, 0]
        best_bid = bid_price[:, 0]
        mid = (best_ask + best_bid) * 0.5
        spread = best_ask - best_bid
        relative_spread = spread / mid
        top_depth = ask_size[:, 0] + bid_size[:, 0]
        microprice = np.divide(
            best_ask * bid_size[:, 0] + best_bid * ask_size[:, 0],
            top_depth,
            out=mid.copy(),
            where=top_depth > 0.0,
        )
        output[start:stop, 0] = spread
        output[start:stop, 1] = relative_spread
        output[start:stop, 2] = mid
        output[start:stop, 3] = microprice
        output[start:stop, 4] = (microprice - mid) / mid
        column = 5
        for depth in (1, 5, 10):
            total_ask = ask_size[:, :depth].sum(axis=1)
            total_bid = bid_size[:, :depth].sum(axis=1)
            total = total_ask + total_bid
            imbalance = np.divide(
                total_bid - total_ask,
                total,
                out=np.zeros_like(total),
                where=total > 0.0,
            )
            output[start:stop, column] = total_ask
            output[start:stop, column + 1] = total_bid
            output[start:stop, column + 2] = imbalance
            column += 3

    starts = np.r_[0, np.flatnonzero(day_ids[1:] != day_ids[:-1]) + 1]
    stops = np.r_[starts[1:], len(day_ids)]
    for start, stop in zip(starts, stops, strict=True):
        mid = np.asarray(output[start:stop, 2], dtype=np.float64)
        relative_spread = np.asarray(output[start:stop, 1], dtype=np.float64)
        imbalance = np.asarray(output[start:stop, 7], dtype=np.float64)
        log_mid = np.log(mid)
        log_return = np.r_[0.0, np.diff(log_mid)]
        spread_change = np.r_[0.0, np.diff(relative_spread)]
        imbalance_change = np.r_[0.0, np.diff(imbalance)]
        output[start:stop, 14] = log_return.astype(np.float32)
        output[start:stop, 15] = spread_change.astype(np.float32)
        output[start:stop, 16] = imbalance_change.astype(np.float32)
        output[start:stop, 17] = _rolling_std(log_return, 4)
        output[start:stop, 18] = _rolling_std(log_return, 20)
    output.flush()
    return output


def main() -> None:
    args = parse_args()
    path = args.input.expanduser().resolve()
    if path.suffix.lower() not in {".csv", ".txt", ".gz"} and not path.name.lower().endswith(".csv.gz"):
        raise ValueError("The low-memory preparer currently supports CSV/CSV.GZ only")
    if args.chunksize <= 0:
        raise ValueError("chunksize must be positive")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    # Resolve the schema from one row using the already-tested canonical loader.
    sample = load_raw_lob_table(path, column_map=args.column_map, max_rows=1)
    resolved = sample.resolved_columns

    available_rows = _count_csv_rows(path)
    n_rows = min(available_rows, args.max_rows) if args.max_rows is not None else available_rows
    if n_rows <= args.sequence_length + args.horizon:
        raise ValueError(f"Not enough rows: {n_rows:,}")
    print(f"preparing {n_rows:,} rows in chunks of {args.chunksize:,}", flush=True)

    _write_raw_arrays(
        path=path,
        output=output,
        resolved=resolved,
        n_rows=n_rows,
        chunksize=args.chunksize,
    )

    source = np.load(output / "source_features.npy", mmap_mode="r")
    timestamps = np.load(output / "timestamps.npy", mmap_mode="r")
    day_ids, day_labels = _day_ids_from_timestamps(timestamps)
    np.save(output / "day_ids.npy", day_ids)

    best_ask = source[:, 0]
    best_bid = source[:, 2]
    label_returns = None
    if args.label_type == "cost-aware":
        label_returns = executable_action_labels(
            best_ask,
            best_bid,
            horizon=args.horizon,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            minimum_edge_bps=args.minimum_edge_bps,
            group_ids=day_ids,
        )
        labels = label_returns.labels
        label_valid_mask = label_returns.valid_mask
    else:
        labels = point_midprice_direction_labels(
            best_ask,
            best_bid,
            horizon=args.horizon,
            threshold_bps=args.threshold_bps,
            group_ids=day_ids,
        )
        label_valid_mask = labels >= 0

    valid_targets = valid_targets_within_days(
        day_ids,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        label_valid_mask=label_valid_mask,
    )
    split = fixed_day_split(
        day_ids,
        valid_targets=valid_targets,
        n_train_days=args.train_days,
        n_validation_days=args.validation_days,
    )
    if min(len(split.train), len(split.validation), len(split.test)) == 0:
        raise ValueError("one data split is empty")

    np.save(output / "labels.npy", labels.astype(np.int64, copy=False))
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

    training_mask = np.isin(day_ids, split.train_days)
    raw_tabular_path = output / "_tabular_raw.npy"
    raw_tabular = _write_tabular_raw(
        source,
        day_ids,
        raw_tabular_path,
        chunksize=args.chunksize,
    )
    (output / "tabular_feature_names.json").write_text(
        json.dumps(TABULAR_NAMES, indent=2)
    )

    if args.normalize == "train":
        feature_mean, feature_std = _chunked_mean_std(
            source, training_mask, chunksize=args.chunksize
        )
        tabular_mean, tabular_std = _chunked_mean_std(
            raw_tabular, training_mask, chunksize=args.chunksize
        )
        _normalise_to_file(
            source,
            output / "features.npy",
            feature_mean,
            feature_std,
            chunksize=args.chunksize,
        )
        _normalise_to_file(
            raw_tabular,
            output / "tabular_features.npy",
            tabular_mean,
            tabular_std,
            chunksize=args.chunksize,
        )
        np.savez(output / "scaler.npz", mean=feature_mean, std=feature_std)
        np.savez(
            output / "tabular_scaler.npz", mean=tabular_mean, std=tabular_std
        )
        del raw_tabular
        raw_tabular_path.unlink(missing_ok=True)
    else:
        # Hard links avoid duplicate 600 MB files on Linux/WSL.
        os.link(output / "source_features.npy", output / "features.npy")
        del raw_tabular
        raw_tabular_path.rename(output / "tabular_features.npy")

    class_counts = {
        name: np.bincount(labels[indices], minlength=3).astype(int).tolist()
        for name, indices in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
    }
    metadata = {
        "source": str(path),
        "dataset_type": "raw_lob_snapshots",
        "preparation_mode": "chunked_memory_mapped",
        "n_observations": int(n_rows),
        "n_features": 40,
        "n_days": int(len(day_labels)),
        "day_labels": list(day_labels),
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
        "resolved_columns": resolved,
        "warning": (
            "Historical top-of-book execution research only; immediate fills, queue position, "
            "market impact, funding, and exchange-specific rules remain modeling assumptions."
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
