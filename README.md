# Cost-Aware Limit Order Book Forecasting

Short-horizon BTCUSDT limit-order-book forecasting with **XGBoost, DeepLOB, and a compact Transformer**, evaluated with executable bid/ask prices and transaction costs.

The project asks a simple question:

> Do better classification metrics on limit-order-book data translate into better executable trading performance after spread, fees, and slippage?

The answer across the original and weighted-loss experiments was **no**. In the original run, DeepLOB and the Transformer achieved high raw accuracy largely by predicting the dominant no-trade class. Class-balanced weighted loss improved their directional discrimination, but all three models lost money after execution costs in the follow-up run.

![Original-run classification metrics (unweighted neural loss)](results/figures/classification_metrics.png)

## Results

Both runs used roughly 3.7M BTCUSDT perpetual LOB snapshots sampled at about 250 ms, ten book levels, 100-snapshot input windows (~25 s), and a 40-snapshot forecast horizon (~10 s). The out-of-sample test set contained 605,175 labeled events and was strongly imbalanced: 4.42% short/down, 90.57% flat, and 5.01% long/up.

### Original run: unweighted neural loss

The original [Kaggle notebook](notebooks/08_kaggle_deeplob_transformer_unweighted_loss.ipynb) used unweighted cross-entropy for DeepLOB and the Transformer. XGBoost used class-balanced sample weights.

| Model | Accuracy | Macro-F1 | Balanced accuracy | MCC | Net P&L | Return | Trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.708 | **0.405** | **0.512** | **0.192** | -$12,978.51 | -12.98% | 5,642 |
| DeepLOB | 0.906 | 0.319 | 0.334 | 0.028 | $0.00 | 0.00% | 0 |
| Transformer | 0.906 | 0.320 | 0.335 | 0.035 | +$7.44 | +0.007% | 6 |

The Transformer result is **not evidence of profitability**: only six trades passed the fixed 0.50 confidence threshold. DeepLOB placed no trades. XGBoost produced many more directional signals, but its minority-class precision was only about 12–13%, and transaction costs overwhelmed the predictive edge.

### Follow-up: class-balanced weighted neural loss

The [weighted-loss Kaggle notebook](notebooks/09_kaggle_deeplob_transformer_weighted_loss.ipynb) enables `--balance-weights` for DeepLOB and the Transformer, applying inverse-frequency class weights to cross-entropy. It retains the same input windows, forecast horizon, neural training sample limits, five training epochs, and portfolio assumptions. XGBoost remains the class-balanced comparison model.

The notebook's saved evaluation and backtest outputs report:

| Model | Accuracy | Macro-F1 | Balanced accuracy | MCC | Net P&L | Return | Trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.708 | **0.405** | **0.512** | **0.192** | -$12,978.51 | -12.98% | 5,642 |
| DeepLOB | 0.626 | 0.372 | 0.511 | 0.169 | -$11,338.99 | -11.34% | 4,682 |
| Transformer | 0.646 | 0.369 | 0.476 | 0.150 | -$9,669.95 | -9.67% | 3,819 |

Weighting reduced majority-class collapse: DeepLOB's balanced accuracy rose from **0.334 to 0.511**, and the Transformer's from **0.335 to 0.476**. Both generated thousands of trades at the fixed 0.50 confidence threshold, but their signals still lost money after costs. Better imbalance-aware metrics did not translate into profitable execution.

The metrics, run configurations, and figures under [`results/`](results/) describe the **original run**. The weighted-loss results above come from the saved outputs in notebook `09`.

## Why accuracy is misleading here

The test set is about 90.6% flat. A model that almost always predicts flat can therefore report ~90% accuracy while having little directional value.

The original DeepLOB and Transformer runs illustrate this failure mode:

- DeepLOB balanced accuracy: **0.334**
- Transformer balanced accuracy: **0.335**
- Three-class chance-level balanced accuracy: approximately **0.333**

XGBoost reduced raw accuracy but improved balanced accuracy to **0.512** and MCC to **0.192**, showing substantially better directional discrimination than the original neural runs. The weighted-loss follow-up also improved neural balanced accuracy while lowering raw accuracy. In both experiments, classification improvement alone was not enough to overcome trading costs.

## Experiment design

### Input

Each snapshot contains ten LOB levels. Internally the 40 raw features are represented as:

```text
[ask_price_1, ask_size_1, bid_price_1, bid_size_1, ... level 10]
```

A model receives 100 consecutive snapshots:

```text
X_t in R^(100 x 40)
```

### Cost-aware target

Instead of labeling only future mid-price direction, the primary target uses executable entry and exit prices.

For a long action:

```text
buy now at current ask -> sell at future bid
```

For a short action:

```text
sell now at current bid -> buy back at future ask
```

The target assigns `short`, `flat`, or `long` after accounting for a 1.0 bp fee per side and 0.5 bp slippage per side. Spread is naturally represented by using ask prices for buys and bid prices for sells.

### Split

The real-data experiment is chronological:

- first 8 days: training
- next 2 days: validation
- final 2 days: test

No random train/test shuffle is used. Sequence/forecast boundaries are handled by the data pipeline to avoid leakage across windows.

### Models

- **XGBoost** — engineered microstructure features plus window summaries; trained on the full training split with class-balanced sample weights.
- **DeepLOB** — convolutional feature extraction, inception block, and LSTM; trained on 1,000,000 representative windows for five epochs.
- **Transformer** — compact 2-layer, 4-head encoder with `d_model=64`; trained on 500,000 representative windows for five epochs.

The neural subsampling was a compute choice; adjacent 100-snapshot windows overlap heavily. All models were evaluated on the full test period. The original neural runs used unweighted cross-entropy; the follow-up enabled class-balanced weighted cross-entropy with `--balance-weights`.

## Backtest assumptions

The portfolio backtest is deliberately simple and transparent:

- one open position maximum;
- fixed 40-snapshot holding horizon;
- fixed $10,000 notional per trade;
- $100,000 initial equity;
- immediate full fill at top of book;
- 1.0 bp fee per side;
- 0.5 bp slippage per side;
- fixed 0.50 model-confidence threshold;
- no queue-position model, partial fills, market impact, or funding;
- equity/drawdown updated at trade exits.

This is an **evaluation backtest**, not a production execution simulator.

## Repository structure

```text
configs/                 Data/model configuration examples
data/sample/             Small synthetic fixture for pipeline checks
docs/                    Real-data and backtest documentation
notebooks/               EDA, modeling, backtesting, and Kaggle run notebooks
results/                 Lightweight original-run metrics and figures
scripts/                 Reproducible CLI entry points
src/lob_project/         Reusable Python package
tests/                    Unit tests
```

Raw BTC data, processed arrays, predictions, and trained checkpoints are intentionally not committed.

## Installation

The project uses Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

Notebook dependencies are optional:

```bash
uv sync --extra notebooks
```

Run the tests:

```bash
uv run pytest -q
```

The published repository contains a small synthetic fixture, so core pipeline tests do not require the multi-gigabyte BTC dataset.

## Quick synthetic smoke test

Synthetic data is included only to verify code paths. It is **not** financial evidence.

```bash
uv run python scripts/prepare_data.py \
  --input data/sample/synthetic_fi2010.txt \
  --output-dir data/processed/synthetic_h50 \
  --horizon 50 \
  --sequence-length 100 \
  --normalize train \
  --executable-prices

uv run python scripts/train_model.py \
  --model baseline \
  --processed-dir data/processed/synthetic_h50 \
  --output-dir outputs/baseline_h50

uv run python scripts/evaluate_model.py \
  --run-dir outputs/baseline_h50 \
  --processed-dir data/processed/synthetic_h50
```

## Reproducing the BTC experiment

### 1. Obtain the data

The final experiment used the public Kaggle dataset **Bitcoin Limit Order Book (LOB) Data**:

https://www.kaggle.com/datasets/siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data

The repository does not redistribute the source dataset. Place the downloaded CSV/CSV.GZ under `data/raw/`.

### 2. Prepare the h=40 cost-aware dataset

For the full multi-million-row CSV, use the low-memory preparer rather than the original in-memory path:

```bash
uv run python scripts/prepare_btc_lob_low_memory.py \
  --input data/raw/YOUR_BTC_FILE.csv \
  --column-map configs/btc_kaggle_numeric_columns.json \
  --output-dir data/processed/btc_h40_cost \
  --chunksize 100000 \
  --sequence-length 100 \
  --horizon 40 \
  --label-type cost-aware \
  --fee-bps 1.0 \
  --slippage-bps 0.5 \
  --minimum-edge-bps 0.0 \
  --train-days 8 \
  --validation-days 2
```

### 3. Train the models

These commands reproduce the original run with unweighted neural loss:

XGBoost:

```bash
uv run python scripts/train_model.py \
  --model xgboost \
  --processed-dir data/processed/btc_h40_cost \
  --output-dir outputs/btc_h40_cost/xgboost
```

DeepLOB:

```bash
uv run python scripts/train_model.py \
  --model deeplob \
  --processed-dir data/processed/btc_h40_cost \
  --output-dir outputs/btc_h40_cost/deeplob \
  --max-train-samples 1000000 \
  --epochs 5 \
  --batch-size 512
```

Transformer:

```bash
uv run python scripts/train_model.py \
  --model transformer \
  --processed-dir data/processed/btc_h40_cost \
  --output-dir outputs/btc_h40_cost/transformer \
  --max-train-samples 500000 \
  --epochs 5 \
  --batch-size 512
```

GPU training is recommended for the neural models. The final runs used CUDA on Kaggle.

For the weighted-loss follow-up, add `--balance-weights` to both neural training commands. Use separate output directories, such as `outputs/btc_h40_cost_weighted/deeplob` and `outputs/btc_h40_cost_weighted/transformer`, to preserve the original artifacts. The XGBoost command is unchanged.

### 4. Evaluate and backtest

For each model:

```bash
uv run python scripts/evaluate_model.py \
  --run-dir outputs/btc_h40_cost/MODEL \
  --processed-dir data/processed/btc_h40_cost

uv run python scripts/run_portfolio_backtest.py \
  --run-dir outputs/btc_h40_cost/MODEL \
  --processed-dir data/processed/btc_h40_cost \
  --confidence 0.50 \
  --initial-equity 100000 \
  --notional-per-trade 10000 \
  --fee-bps 1.0 \
  --slippage-bps 0.5
```

Replace `MODEL` with `xgboost`, `deeplob`, or `transformer`.

For weighted models, point `--run-dir` in both commands to the corresponding weighted output directory.

The complete Kaggle workflows, including saved outputs, are available in:

- [`08_kaggle_deeplob_transformer_unweighted_loss.ipynb`](notebooks/08_kaggle_deeplob_transformer_unweighted_loss.ipynb): original run with unweighted neural loss.
- [`09_kaggle_deeplob_transformer_weighted_loss.ipynb`](notebooks/09_kaggle_deeplob_transformer_weighted_loss.ipynb): follow-up with class-balanced weighted loss for DeepLOB and the Transformer.

When switching between notebooks in the same Kaggle session, choose a fresh `OUTPUT_ROOT` for the weighted run: the notebooks reuse existing checkpoints when `FORCE_RETRAIN = False`.

## Limitations

The results should be interpreted as research experiments, not as evidence of a deployable trading strategy.

- Only one short BTCUSDT market period is evaluated.
- Both reported tables use one chronological train/validation/test split rather than a full walk-forward aggregate.
- Neural models were trained on representative subsets due to compute constraints.
- Class-balanced weighted loss improved neural directional discrimination in the follow-up, but neither neural model was profitable after costs. The comparison does not establish robustness across seeds or market periods.
- Backtesting assumes immediate top-of-book fills and ignores queue position, partial fills, market impact, and perpetual funding.
- The confidence threshold was fixed at 0.50 for comparability rather than optimized for P&L.

These limitations are intentional: the project focuses on the gap between **forecast metrics** and **executable economic performance**.

## References

- Zhang, Zohren & Roberts, *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books*, IEEE Transactions on Signal Processing / arXiv: https://arxiv.org/abs/1808.03668
- Public BTCUSDT LOB dataset: https://www.kaggle.com/datasets/siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data

## License

Code in this repository is released under the MIT License. The external datasets retain their own licenses and are not redistributed here.
