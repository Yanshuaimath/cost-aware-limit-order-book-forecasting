# Final BTC h=40 results

This directory contains the lightweight artifacts from the final out-of-sample experiment. Large model checkpoints, processed arrays, predictions, and raw market data are intentionally excluded from Git.

## Headline result

| Model | Macro-F1 | Balanced accuracy | MCC | Net P&L | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | **0.405** | **0.512** | **0.192** | -$12,978.51 | 5,642 |
| DeepLOB | 0.319 | 0.334 | 0.028 | $0.00 | 0 |
| Transformer | 0.320 | 0.335 | 0.035 | +$7.44 | 6 |

The test set was about 90.6% flat. DeepLOB and the Transformer mostly predicted that majority class, so their ~90.6% raw accuracy is not evidence of useful directional forecasting. XGBoost improved minority-class recall and balanced metrics, but its false-positive rate generated enough losing trades that costs dominated the signal.

The Transformer made only six trades, so its small positive P&L is not statistically meaningful.

## Contents

- `model_comparison.csv/json` — compact final table.
- `metrics/*_test_metrics.json` — full classification metrics and confusion matrices.
- `backtests/*_portfolio_metrics.json` — economic metrics and execution assumptions.
- `training/*_run_config.json` — exact training CLI configuration captured by the run.
- `training/*_training_summary.json` — neural best epoch / validation loss.
- `figures/classification_metrics.png` — comparison of imbalance-aware classification metrics.
- `figures/equity_curves.png` — test-period equity curves at the fixed 0.50 confidence threshold.
