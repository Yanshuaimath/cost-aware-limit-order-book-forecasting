from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PurgedSplits:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_boundary: int
    validation_boundary: int
    purge_size: int


def purged_chronological_split(
    n_observations: int,
    sequence_length: int,
    forecast_horizon: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> PurgedSplits:
    """Create conservative chronological target-index splits.

    No raw observation is shared by input windows across adjacent splits, and
    each training/validation target's future horizon remains inside its own
    time partition. This is intentionally stricter than many benchmark setups.
    """
    if n_observations <= sequence_length + 2 * forecast_horizon:
        raise ValueError("Not enough observations for the requested window and horizon")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Fractions must lie between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than one")

    train_boundary = int(n_observations * train_fraction)
    validation_boundary = int(n_observations * (train_fraction + validation_fraction))
    purge_size = sequence_length - 1

    train_start = sequence_length - 1
    train_stop = train_boundary - forecast_horizon

    validation_start = train_boundary + purge_size
    validation_stop = validation_boundary - forecast_horizon

    test_start = validation_boundary + purge_size
    test_stop = n_observations - forecast_horizon

    train = np.arange(train_start, max(train_start, train_stop), dtype=np.int64)
    validation = np.arange(
        validation_start, max(validation_start, validation_stop), dtype=np.int64
    )
    test = np.arange(test_start, max(test_start, test_stop), dtype=np.int64)

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            "One split is empty. Increase data size or reduce sequence length/horizon."
        )
    return PurgedSplits(
        train=train,
        validation=validation,
        test=test,
        train_boundary=train_boundary,
        validation_boundary=validation_boundary,
        purge_size=purge_size,
    )


def assert_no_window_overlap(
    left_indices: np.ndarray, right_indices: np.ndarray, sequence_length: int
) -> None:
    if not len(left_indices) or not len(right_indices):
        return
    left_last_raw = int(left_indices.max())
    right_first_raw = int(right_indices.min()) - sequence_length + 1
    if left_last_raw >= right_first_raw:
        raise AssertionError(
            f"Window overlap detected: left ends {left_last_raw}, right starts {right_first_raw}"
        )
