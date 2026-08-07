# Real-data pipeline

This extension adds the files needed to begin the real BTC limit-order-book experiment.
It does not download a dataset automatically because public copies use different file
names and column schemas. The loader accepts CSV, Parquet, and Feather files and can
auto-detect common ten-level LOB column names.

## 1. Synchronize dependencies

```bash
uv sync
uv run pytest -q
```

## 2. Inspect the source columns

```bash
uv run python - <<'PY'
import pandas as pd
path = "/path/to/btc_lob.csv"
print(pd.read_csv(path, nrows=3).columns.tolist())
PY
```

When automatic column detection fails, create `configs/btc_column_map.json`:

```json
{
  "timestamp": "local_timestamp",
  "ask_price_1": "asks[0].price",
  "ask_size_1": "asks[0].amount",
  "bid_price_1": "bids[0].price",
  "bid_size_1": "bids[0].amount"
}
```

Continue the same pattern through level 10.

## 3. Prepare a cost-aware dataset

For 250 ms snapshots, horizon 40 is approximately ten seconds.

```bash
uv run python scripts/prepare_btc_lob.py \
  --input /path/to/btc_lob.csv \
  --output-dir data/processed/btc_h40_cost \
  --column-map configs/btc_column_map.json \
  --sequence-length 100 \
  --horizon 40 \
  --label-type cost-aware \
  --fee-bps 1.0 \
  --slippage-bps 0.5 \
  --minimum-edge-bps 0.0 \
  --train-days 8 \
  --validation-days 2
```

For the academic mid-price comparison:

```bash
uv run python scripts/prepare_btc_lob.py \
  --input /path/to/btc_lob.csv \
  --output-dir data/processed/btc_h40_mid \
  --sequence-length 100 \
  --horizon 40 \
  --label-type midprice \
  --threshold-bps 1.0 \
  --train-days 8 \
  --validation-days 2
```

The preparation script:

- validates ten levels of bid/ask prices and quantities;
- sorts snapshots by UTC timestamp;
- rejects duplicate timestamps unless explicitly allowed;
- prevents input windows and exits from crossing UTC-day boundaries;
- creates exact-horizon mid-price or executable-return labels;
- fits normalization using training days only;
- saves engineered microstructure features for XGBoost.

## 4. Train models on the fixed split

```bash
uv run python scripts/train_model.py \
  --model xgboost \
  --processed-dir data/processed/btc_h40_cost \
  --output-dir outputs/btc_h40_cost/xgboost

uv run python scripts/evaluate_model.py \
  --run-dir outputs/btc_h40_cost/xgboost \
  --processed-dir data/processed/btc_h40_cost

uv run python scripts/run_portfolio_backtest.py \
  --run-dir outputs/btc_h40_cost/xgboost \
  --processed-dir data/processed/btc_h40_cost \
  --confidence 0.50 \
  --initial-equity 100000 \
  --notional-per-trade 10000 \
  --fee-bps 1.0 \
  --slippage-bps 0.5
```

DeepLOB and Transformer use the same processed directory and commands, replacing
`--model xgboost` with `deeplob` or `transformer`.

## 5. Create walk-forward folds

```bash
uv run python scripts/create_walk_forward_folds.py \
  --processed-dir data/processed/btc_h40_cost \
  --output-dir data/processed/btc_h40_cost/folds \
  --minimum-train-days 5 \
  --validation-days 1 \
  --test-days 1
```

Run XGBoost over all folds:

```bash
uv run python scripts/run_walk_forward.py \
  --model xgboost \
  --processed-dir data/processed/btc_h40_cost \
  --folds-dir data/processed/btc_h40_cost/folds \
  --output-root outputs/btc_h40_cost/xgboost_walk_forward \
  --run-backtest \
  --xgb-estimators 300 \
  --confidence 0.50 \
  --fee-bps 1.0 \
  --slippage-bps 0.5
```

The confidence threshold is fixed at 0.50 in the published final experiment.
Validation-selected thresholds and full walk-forward aggregation are left as optional extensions.

## Current limitations

- Immediate full fills at displayed quotes.
- No queue-position or partial-fill model.
- No market impact.
- No perpetual-futures funding calculation.
- Fixed confidence threshold rather than validation optimization.
- Top-of-book execution rather than consuming multiple depth levels.

