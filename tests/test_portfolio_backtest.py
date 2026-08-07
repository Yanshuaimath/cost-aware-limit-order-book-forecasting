import numpy as np

from lob_project.backtest.portfolio import single_position_backtest


def test_long_trade_updates_equity_from_executable_prices() -> None:
    result = single_position_backtest(
        probabilities=np.array([[0.01, 0.01, 0.98]]),
        target_indices=np.array([0]),
        best_ask=np.array([100.0, 101.0]),
        best_bid=np.array([99.9, 100.8]),
        horizon=1,
        confidence=0.5,
        initial_equity=1_000.0,
        notional_per_trade=1_000.0,
    )

    assert result.metrics["n_trades"] == 1
    np.testing.assert_allclose(result.trades[0]["net_return"], 0.008)
    np.testing.assert_allclose(result.trades[0]["net_pnl"], 8.0)
    np.testing.assert_allclose(result.metrics["final_equity"], 1_008.0)


def test_short_trade_uses_bid_then_future_ask() -> None:
    result = single_position_backtest(
        probabilities=np.array([[0.98, 0.01, 0.01]]),
        target_indices=np.array([0]),
        best_ask=np.array([100.1, 99.2]),
        best_bid=np.array([100.0, 99.0]),
        horizon=1,
        initial_equity=100.0,
        notional_per_trade=100.0,
    )

    np.testing.assert_allclose(result.trades[0]["net_return"], 0.008)
    np.testing.assert_allclose(result.metrics["final_equity"], 100.8)


def test_overlapping_signals_are_skipped() -> None:
    probabilities = np.tile(np.array([[0.01, 0.01, 0.98]]), (4, 1))
    result = single_position_backtest(
        probabilities=probabilities,
        target_indices=np.array([0, 1, 2, 3]),
        best_ask=np.array([100.0, 100.1, 100.2, 100.3, 100.4]),
        best_bid=np.array([99.9, 100.0, 100.1, 100.2, 100.3]),
        horizon=2,
        initial_equity=1_000.0,
        notional_per_trade=100.0,
    )

    assert result.metrics["n_trades"] == 2
    assert result.metrics["skipped_overlap"] == 2
    assert [trade["signal_index"] for trade in result.trades] == [0, 2]


def test_fees_and_slippage_are_applied_on_both_sides() -> None:
    result = single_position_backtest(
        probabilities=np.array([[0.01, 0.01, 0.98]]),
        target_indices=np.array([0]),
        best_ask=np.array([100.0, 101.0]),
        best_bid=np.array([99.9, 101.0]),
        horizon=1,
        initial_equity=10_000.0,
        notional_per_trade=10_000.0,
        fee_bps=5.0,
        slippage_bps=10.0,
    )

    trade = result.trades[0]
    expected_entry = 100.0 * 1.001
    expected_exit = 101.0 * 0.999
    expected_quantity = 10_000.0 / expected_entry
    expected_fees = expected_quantity * (expected_entry + expected_exit) * 0.0005
    expected_pnl = expected_quantity * (expected_exit - expected_entry) - expected_fees

    np.testing.assert_allclose(trade["entry_price"], expected_entry)
    np.testing.assert_allclose(trade["exit_price"], expected_exit)
    np.testing.assert_allclose(trade["total_fees"], expected_fees)
    np.testing.assert_allclose(trade["net_pnl"], expected_pnl)
