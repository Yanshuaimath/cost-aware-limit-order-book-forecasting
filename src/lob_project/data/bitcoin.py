from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from lob_project.constants import N_LOB_FEATURES


@dataclass(frozen=True)
class RawLOBData:
    """Canonical ten-level limit-order-book snapshots.

    Features use the project-wide ordering
    ``[ask_price, ask_size, bid_price, bid_size]`` for levels 1 through 10.
    Timestamps are stored in UTC as ``datetime64[ns]``.
    """

    timestamps: np.ndarray
    day_ids: np.ndarray
    day_labels: tuple[str, ...]
    features: np.ndarray
    resolved_columns: dict[str, str]


def _normalise_name(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower())
    return value.strip("_")


def _timestamp_candidates() -> tuple[str, ...]:
    return (
        "timestamp",
        "time",
        "datetime",
        "date_time",
        "event_time",
        "transact_time",
        "local_timestamp",
        "exchange_timestamp",
        "ts",
    )


def _field_candidates(side: str, field: str, level: int) -> tuple[str, ...]:
    side_short = "a" if side == "ask" else "b"
    field_aliases = (
        ("price", "p")
        if field == "price"
        else ("size", "volume", "qty", "quantity", "amount", "v")
    )
    plural = f"{side}s"
    zero_level = level - 1
    values: list[str] = []
    for alias in field_aliases:
        values.extend(
            [
                f"{side}_{alias}_{level}",
                f"{side}{level}_{alias}",
                f"{side}_{level}_{alias}",
                f"{side}{alias}{level}",
                f"{side_short}{alias[0]}{level}",
                f"{plural}_{zero_level}_{alias}",
                f"{plural}_{level}_{alias}",
                f"{plural}{zero_level}{alias}",
            ]
        )
    return tuple(dict.fromkeys(_normalise_name(value) for value in values))


def _load_column_map(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("column map must be a JSON object")
    return {str(key): str(value) for key, value in raw.items()}


def _resolve_column(
    columns_by_normalised_name: Mapping[str, str],
    *,
    canonical_name: str,
    candidates: tuple[str, ...],
    explicit: Mapping[str, str],
) -> str:
    if canonical_name in explicit:
        requested = explicit[canonical_name]
        normalised = _normalise_name(requested)
        if normalised not in columns_by_normalised_name:
            raise ValueError(
                f"Column map requested {requested!r} for {canonical_name}, but it was not found"
            )
        return columns_by_normalised_name[normalised]
    for candidate in candidates:
        if candidate in columns_by_normalised_name:
            return columns_by_normalised_name[candidate]
    raise ValueError(
        f"Could not resolve {canonical_name}. Supply --column-map with an explicit mapping. "
        f"Tried aliases: {candidates[:8]}..."
    )


def _read_table(path: Path, max_rows: int | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".gz"} or path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path, nrows=max_rows)
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        return frame.iloc[:max_rows] if max_rows is not None else frame
    if suffix in {".feather", ".ftr"}:
        frame = pd.read_feather(path)
        return frame.iloc[:max_rows] if max_rows is not None else frame
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _parse_timestamps(values: pd.Series) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="raise").astype("float64")
        finite = numeric[np.isfinite(numeric)]
        if finite.empty:
            raise ValueError("timestamp column contains no finite values")
        magnitude = float(np.nanmedian(np.abs(finite)))
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
        bad = int(parsed.isna().sum())
        raise ValueError(f"timestamp parsing failed for {bad} rows")
    return pd.DatetimeIndex(parsed)


def load_raw_lob_table(
    path: str | Path,
    *,
    column_map: str | Path | None = None,
    max_rows: int | None = None,
    drop_duplicate_timestamps: bool = False,
) -> RawLOBData:
    """Load a CSV/Parquet/Feather LOB table into the canonical 40-column format.

    The loader auto-detects common names such as ``ask_price_1``, ``ask1_price``,
    ``asks[0].price``, ``bid_size_1``, and related variants. For other schemas,
    pass a JSON column map whose keys are canonical names, for example::

        {
          "timestamp": "local_timestamp",
          "ask_price_1": "asks[0].price",
          "ask_size_1": "asks[0].amount",
          "bid_price_1": "bids[0].price",
          "bid_size_1": "bids[0].amount"
        }
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = _read_table(path, max_rows=max_rows)
    if frame.empty:
        raise ValueError("input table is empty")

    explicit = _load_column_map(column_map)
    columns_by_normalised_name: dict[str, str] = {}
    for original in frame.columns:
        normalised = _normalise_name(original)
        if normalised in columns_by_normalised_name:
            raise ValueError(
                f"Columns {columns_by_normalised_name[normalised]!r} and {original!r} "
                "normalise to the same name"
            )
        columns_by_normalised_name[normalised] = str(original)

    timestamp_column = _resolve_column(
        columns_by_normalised_name,
        canonical_name="timestamp",
        candidates=_timestamp_candidates(),
        explicit=explicit,
    )

    resolved: dict[str, str] = {"timestamp": timestamp_column}
    feature_columns: list[str] = []
    for level in range(1, 11):
        for side, field in (
            ("ask", "price"),
            ("ask", "size"),
            ("bid", "price"),
            ("bid", "size"),
        ):
            canonical = f"{side}_{field}_{level}"
            column = _resolve_column(
                columns_by_normalised_name,
                canonical_name=canonical,
                candidates=_field_candidates(side, field, level),
                explicit=explicit,
            )
            resolved[canonical] = column
            feature_columns.append(column)

    timestamps = _parse_timestamps(frame[timestamp_column])
    features = frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float64
    )
    if features.shape[1] != N_LOB_FEATURES:
        raise AssertionError(f"Expected 40 features, received {features.shape}")

    order = np.argsort(timestamps.asi8, kind="stable")
    timestamps = timestamps[order]
    features = features[order]

    if drop_duplicate_timestamps:
        keep = ~timestamps.duplicated(keep="last")
        timestamps = timestamps[keep]
        features = features[keep]
    elif timestamps.duplicated().any():
        raise ValueError(
            "duplicate timestamps detected; use --drop-duplicate-timestamps only after auditing them"
        )

    if not np.isfinite(features).all():
        raise ValueError("LOB features contain missing or non-finite values")
    if np.any(features[:, 0::4] <= 0.0) or np.any(features[:, 2::4] <= 0.0):
        raise ValueError("LOB prices must be positive")
    if np.any(features[:, 1::4] < 0.0) or np.any(features[:, 3::4] < 0.0):
        raise ValueError("LOB quantities must be non-negative")

    ask_prices = features[:, 0::4]
    bid_prices = features[:, 2::4]
    if np.any(ask_prices[:, 0] < bid_prices[:, 0]):
        raise ValueError("crossed book detected: best ask is below best bid")
    if np.any(np.diff(ask_prices, axis=1) < -1e-10):
        raise ValueError("ask prices must be non-decreasing with depth")
    if np.any(np.diff(bid_prices, axis=1) > 1e-10):
        raise ValueError("bid prices must be non-increasing with depth")

    utc_dates = timestamps.strftime("%Y-%m-%d")
    day_codes, unique_days = pd.factorize(utc_dates, sort=False)
    return RawLOBData(
        timestamps=timestamps.to_numpy(dtype="datetime64[ns]"),
        day_ids=day_codes.astype(np.int32, copy=False),
        day_labels=tuple(str(day) for day in unique_days.tolist()),
        features=np.ascontiguousarray(features, dtype=np.float32),
        resolved_columns=resolved,
    )
