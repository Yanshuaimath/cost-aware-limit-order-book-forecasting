import numpy as np

from lob_project.backtest.execution import diagnostic_round_trip_returns


def test_long_trade_uses_ask_then_future_bid() -> None:
    probabilities = np.array([[0.01, 0.01, 0.98]])
    indices = np.array([0])
    ask = np.array([100.0, 101.0])
    bid = np.array([99.9, 100.8])
    result = diagnostic_round_trip_returns(
        probabilities, indices, ask, bid, horizon=1, confidence=0.5
    )
    assert result["n_trades"] == 1
    np.testing.assert_allclose(result["returns"], np.array([0.008]))
