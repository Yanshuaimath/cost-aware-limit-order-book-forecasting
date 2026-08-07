from __future__ import annotations

import numpy as np
import pandas as pd

from lob_project.data.bitcoin import load_raw_lob_table


def _frame(rows: int = 8) -> pd.DataFrame:
    data: dict[str, object] = {
        "timestamp": pd.date_range("2024-01-01", periods=rows, freq="250ms", tz="UTC")
    }
    mid = 100.0 + np.arange(rows) * 0.01
    for level in range(1, 11):
        offset = 0.01 * level
        data[f"ask_price_{level}"] = mid + offset
        data[f"ask_size_{level}"] = np.full(rows, 2.0 + level)
        data[f"bid_price_{level}"] = mid - offset
        data[f"bid_size_{level}"] = np.full(rows, 3.0 + level)
    return pd.DataFrame(data)


def test_load_raw_lob_table_canonical_order(tmp_path) -> None:
    path = tmp_path / "lob.csv"
    _frame().to_csv(path, index=False)
    result = load_raw_lob_table(path)
    assert result.features.shape == (8, 40)
    assert result.features[0, 0] == np.float32(100.01)
    assert result.features[0, 2] == np.float32(99.99)
    assert result.day_labels == ("2024-01-01",)
    assert np.all(result.day_ids == 0)


def test_duplicate_timestamps_require_explicit_policy(tmp_path) -> None:
    frame = _frame()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    path = tmp_path / "lob.csv"
    frame.to_csv(path, index=False)
    try:
        load_raw_lob_table(path)
    except ValueError as exc:
        assert "duplicate timestamps" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate timestamps should be rejected")
