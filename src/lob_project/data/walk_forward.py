from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DaySplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_days: tuple[int, ...]
    validation_days: tuple[int, ...]
    test_days: tuple[int, ...]


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_days: tuple[int, ...]
    validation_days: tuple[int, ...]
    test_days: tuple[int, ...]


def _validate_day_ids(day_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(day_ids)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("day_ids must be a non-empty one-dimensional array")
    if np.any(values[1:] < values[:-1]):
        raise ValueError("day_ids must be chronologically non-decreasing")
    return values


def valid_targets_within_days(
    day_ids: np.ndarray,
    *,
    sequence_length: int,
    horizon: int,
    label_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return targets whose input and exact-horizon exit stay in one day."""

    days = _validate_day_ids(day_ids)
    if sequence_length < 2 or horizon <= 0:
        raise ValueError("sequence_length must be >=2 and horizon must be positive")
    mask = None
    if label_valid_mask is not None:
        mask = np.asarray(label_valid_mask, dtype=bool)
        if mask.ndim != 1 or len(mask) != len(days):
            raise ValueError("label_valid_mask must match day_ids")

    targets: list[np.ndarray] = []
    starts = np.r_[0, np.flatnonzero(days[1:] != days[:-1]) + 1]
    stops = np.r_[starts[1:], len(days)]
    for start, stop in zip(starts, stops, strict=True):
        first_target = int(start + sequence_length - 1)
        stop_exclusive = int(stop - horizon)
        if first_target >= stop_exclusive:
            continue
        candidates = np.arange(first_target, stop_exclusive, dtype=np.int64)
        if mask is not None:
            candidates = candidates[mask[candidates]]
        if len(candidates):
            targets.append(candidates)
    return np.concatenate(targets) if targets else np.empty(0, dtype=np.int64)


def fixed_day_split(
    day_ids: np.ndarray,
    *,
    valid_targets: np.ndarray,
    n_train_days: int,
    n_validation_days: int,
) -> DaySplit:
    days = _validate_day_ids(day_ids)
    unique_days = np.unique(days)
    if n_train_days <= 0 or n_validation_days <= 0:
        raise ValueError("training and validation day counts must be positive")
    if n_train_days + n_validation_days >= len(unique_days):
        raise ValueError("at least one complete test day is required")

    train_days = tuple(int(value) for value in unique_days[:n_train_days])
    validation_days = tuple(
        int(value) for value in unique_days[n_train_days : n_train_days + n_validation_days]
    )
    test_days = tuple(int(value) for value in unique_days[n_train_days + n_validation_days :])
    targets = np.asarray(valid_targets, dtype=np.int64)
    return DaySplit(
        train=targets[np.isin(days[targets], train_days)],
        validation=targets[np.isin(days[targets], validation_days)],
        test=targets[np.isin(days[targets], test_days)],
        train_days=train_days,
        validation_days=validation_days,
        test_days=test_days,
    )


def expanding_walk_forward_folds(
    day_ids: np.ndarray,
    *,
    valid_targets: np.ndarray,
    minimum_train_days: int,
    validation_days: int = 1,
    test_days: int = 1,
    step_days: int = 1,
) -> tuple[WalkForwardFold, ...]:
    days = _validate_day_ids(day_ids)
    unique_days = np.unique(days)
    if min(minimum_train_days, validation_days, test_days, step_days) <= 0:
        raise ValueError("all day-count parameters must be positive")

    targets = np.asarray(valid_targets, dtype=np.int64)
    folds: list[WalkForwardFold] = []
    train_stop = minimum_train_days
    fold_number = 0
    while train_stop + validation_days + test_days <= len(unique_days):
        train_values = tuple(int(value) for value in unique_days[:train_stop])
        validation_values = tuple(
            int(value) for value in unique_days[train_stop : train_stop + validation_days]
        )
        test_values = tuple(
            int(value)
            for value in unique_days[
                train_stop + validation_days : train_stop + validation_days + test_days
            ]
        )
        folds.append(
            WalkForwardFold(
                fold=fold_number,
                train=targets[np.isin(days[targets], train_values)],
                validation=targets[np.isin(days[targets], validation_values)],
                test=targets[np.isin(days[targets], test_values)],
                train_days=train_values,
                validation_days=validation_values,
                test_days=test_values,
            )
        )
        fold_number += 1
        train_stop += step_days
    if not folds:
        raise ValueError("not enough days for one walk-forward fold")
    return tuple(folds)
