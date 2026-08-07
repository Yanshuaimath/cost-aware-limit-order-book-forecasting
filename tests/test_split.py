from lob_project.data.split import assert_no_window_overlap, purged_chronological_split


def test_purged_splits_do_not_share_windows() -> None:
    splits = purged_chronological_split(8000, sequence_length=100, forecast_horizon=50)
    assert len(splits.train) > len(splits.validation) > 0
    assert len(splits.test) > 0
    assert_no_window_overlap(splits.train, splits.validation, 100)
    assert_no_window_overlap(splits.validation, splits.test, 100)
    assert splits.train.max() + 50 < splits.train_boundary
    assert splits.validation.max() + 50 < splits.validation_boundary
