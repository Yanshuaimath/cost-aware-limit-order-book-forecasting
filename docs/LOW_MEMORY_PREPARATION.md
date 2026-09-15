# Low-memory BTC LOB preparation

The Kaggle BTCUSDT LOB CSV has roughly 3.7 million rows. The original
`prepare_btc_lob.py` loads the complete table into pandas and is intended only
for samples and smaller files. On WSL, the full file can exceed the available
RAM and cause the VS Code server or the WSL VM to restart.

Use the chunked preparer for the complete CSV:

```bash
rm -rf data/processed/btc_h40_cost

uv run python scripts/prepare_btc_lob_low_memory.py \
  --input "$DATA_FILE" \
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

The script:

- reads only the timestamp and 40 required LOB columns;
- processes the CSV in bounded pandas chunks;
- writes `source_features.npy`, `features.npy`, timestamps, and tabular features
  as disk-backed NumPy arrays;
- computes normalization statistics from training days in streaming passes;
- resets return/change/volatility features at UTC day boundaries;
- creates the same labels, splits, metadata, and downstream file layout expected
  by the training and backtesting scripts.

Before the full run, validate the schema with a small sample:

```bash
rm -rf data/processed/btc_smoke

/usr/bin/time -v uv run python scripts/prepare_btc_lob_low_memory.py \
  --input "$DATA_FILE" \
  --column-map configs/btc_kaggle_numeric_columns.json \
  --output-dir data/processed/btc_smoke \
  --max-rows 200000 \
  --chunksize 50000 \
  --sequence-length 100 \
  --horizon 40 \
  --label-type cost-aware \
  --fee-bps 1.0 \
  --slippage-bps 0.5 \
  --train-days 1 \
  --validation-days 1
```

The smoke command requires at least three UTC days in the first 200,000 rows.
If it does not span three days, increase `--max-rows` or run the full command. The completed processed dataset requires approximately 1.5–2.0 GB of disk
space, depending on row count and metadata outputs.
