# Single-position portfolio backtest

The original `scripts/run_backtest.py` is intentionally retained as an
independent-signal diagnostic. It permits overlapping trades and should not be
interpreted as a capital-constrained portfolio.

The new `scripts/run_portfolio_backtest.py` uses a chronological single-position
state machine.

## Trading rule

For class probabilities `(short, flat, long)`, take a position only when the
largest probability is at least the confidence threshold. A long enters at the
best ask and exits at the future best bid. A short enters at the best bid and
exits at the future best ask. Signals arriving before the current trade's exit
index are ignored.

Slippage worsens both execution prices. Fees are charged on both entry and exit
traded notional. The spread is already represented by using ask/bid quotes, so it
is not subtracted again.

For a long trade with quantity `q`:

```text
entry_price = ask[t] * (1 + slippage_rate)
exit_price  = bid[t+h] * (1 - slippage_rate)
gross_pnl   = q * (exit_price - entry_price)
fees        = fee_rate * q * (entry_price + exit_price)
net_pnl     = gross_pnl - fees
```

For a short trade:

```text
entry_price = bid[t] * (1 - slippage_rate)
exit_price  = ask[t+h] * (1 + slippage_rate)
gross_pnl   = q * (entry_price - exit_price)
fees        = fee_rate * q * (entry_price + exit_price)
net_pnl     = gross_pnl - fees
```

Equity and drawdown update only when the trade exits:

```text
equity_after = equity_before + net_pnl
```

## Run

```bash
uv run python scripts/run_portfolio_backtest.py \
  --run-dir outputs/baseline_h50 \
  --processed-dir data/processed/synthetic_h50 \
  --confidence 0.50 \
  --initial-equity 100000 \
  --notional-per-trade 10000 \
  --fee-bps 1.0 \
  --slippage-bps 0.5
```

Outputs are written under `RUN_DIR/portfolio_backtest/`:

- `metrics.json`
- `trades.csv`
- `equity_curve.csv`
- `backtest_arrays.npz`

## Limitations

This is a defensible research backtest, not an exchange simulator. It still
assumes immediate full fills at displayed top-of-book prices and does not model
queue position, partial fills, market impact, funding, borrowing, or liquidation.
The included synthetic data remains an engineering test only.
