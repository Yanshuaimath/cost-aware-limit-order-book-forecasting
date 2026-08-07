from __future__ import annotations

import numpy as np

from lob_project.labels.executable import (
    executable_action_labels,
    point_midprice_direction_labels,
)


def test_executable_labels_use_realized_action_return() -> None:
    ask = np.array([100.00, 100.01, 100.03, 100.05])
    bid = np.array([99.99, 100.00, 100.02, 100.04])
    result = executable_action_labels(ask, bid, horizon=2)
    assert result.labels.tolist() == [2, 2, -1, -1]
    assert result.long_returns[0] > 0.0
    assert result.short_returns[0] < 0.0


def test_executable_labels_invalidate_cross_day_exit() -> None:
    ask = np.array([100.00, 100.01, 100.02, 100.03])
    bid = ask - 0.01
    groups = np.array([0, 0, 1, 1])
    result = executable_action_labels(ask, bid, horizon=1, group_ids=groups)
    assert result.labels[0] in {0, 1, 2}
    assert result.labels[1] == -1
    assert result.labels[2] in {0, 1, 2}
    assert result.labels[3] == -1


def test_point_midprice_labels_exact_horizon() -> None:
    ask = np.array([100.01, 100.01, 100.04, 99.98])
    bid = np.array([99.99, 99.99, 100.02, 99.96])
    labels = point_midprice_direction_labels(
        ask, bid, horizon=2, threshold_bps=1.0
    )
    assert labels.tolist() == [2, 0, -1, -1]
