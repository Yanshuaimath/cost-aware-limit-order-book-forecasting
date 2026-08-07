from __future__ import annotations

import numpy as np

from lob_project.data.walk_forward import (
    expanding_walk_forward_folds,
    fixed_day_split,
    valid_targets_within_days,
)


def test_valid_targets_do_not_cross_day_boundaries() -> None:
    day_ids = np.repeat(np.arange(3), 10)
    targets = valid_targets_within_days(day_ids, sequence_length=4, horizon=2)
    assert targets.tolist() == [3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 23, 24, 25, 26, 27]


def test_fixed_and_expanding_day_splits() -> None:
    day_ids = np.repeat(np.arange(6), 10)
    targets = valid_targets_within_days(day_ids, sequence_length=4, horizon=2)
    split = fixed_day_split(
        day_ids,
        valid_targets=targets,
        n_train_days=3,
        n_validation_days=1,
    )
    assert set(day_ids[split.train]) == {0, 1, 2}
    assert set(day_ids[split.validation]) == {3}
    assert set(day_ids[split.test]) == {4, 5}

    folds = expanding_walk_forward_folds(
        day_ids,
        valid_targets=targets,
        minimum_train_days=3,
        validation_days=1,
        test_days=1,
    )
    assert len(folds) == 2
    assert folds[0].train_days == (0, 1, 2)
    assert folds[0].validation_days == (3,)
    assert folds[0].test_days == (4,)
    assert folds[1].test_days == (5,)
