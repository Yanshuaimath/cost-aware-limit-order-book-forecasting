import numpy as np

from lob_project.labels.cost_aware import cost_aware_direction_labels


def test_cost_aware_labels_use_executable_prices() -> None:
    ask = np.array([100.01, 100.02, 100.03, 100.50, 100.51])
    bid = np.array([99.99, 100.00, 100.01, 100.48, 100.49])
    labels = cost_aware_direction_labels(ask, bid, horizon=2)
    assert labels[0] == 1
    assert labels[1] == 2
    assert labels[-1] == 1


def test_fees_can_remove_a_trade() -> None:
    ask = np.array([100.00, 100.00, 100.01])
    bid = np.array([99.99, 99.99, 100.005])
    no_fee = cost_aware_direction_labels(ask, bid, horizon=2, fee_bps=0.0)
    with_fee = cost_aware_direction_labels(ask, bid, horizon=2, fee_bps=1.0)
    assert no_fee[0] == 2
    assert with_fee[0] == 1
