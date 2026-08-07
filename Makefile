UV ?= uv
RUN := $(UV) run
BTC_DIR := data/processed/btc_h40_cost
OUT := outputs/btc_h40_cost

.PHONY: setup setup-notebooks lock test lint clean demo synthetic-prepare btc-prepare btc-xgboost btc-deeplob btc-transformer

setup:
	$(UV) sync

setup-notebooks:
	$(UV) sync --extra notebooks

lock:
	$(UV) lock

test:
	$(RUN) pytest -q

lint:
	$(RUN) ruff check .

clean:
	rm -rf data/processed outputs .pytest_cache .ruff_cache

demo:
	$(RUN) python scripts/generate_synthetic.py --output data/sample/synthetic_fi2010.txt

synthetic-prepare:
	$(RUN) python scripts/prepare_data.py --input data/sample/synthetic_fi2010.txt --output-dir data/processed/synthetic_h50 --horizon 50 --sequence-length 100 --normalize train --executable-prices

btc-prepare:
	@test -n "$(BTC_INPUT)" || (echo "Set BTC_INPUT=/path/to/raw.csv" && exit 1)
	$(RUN) python scripts/prepare_btc_lob_low_memory.py --input "$(BTC_INPUT)" --column-map configs/btc_kaggle_numeric_columns.json --output-dir $(BTC_DIR) --chunksize 100000 --sequence-length 100 --horizon 40 --label-type cost-aware --fee-bps 1.0 --slippage-bps 0.5 --minimum-edge-bps 0.0 --train-days 8 --validation-days 2

btc-xgboost:
	$(RUN) python scripts/train_model.py --model xgboost --processed-dir $(BTC_DIR) --output-dir $(OUT)/xgboost
	$(RUN) python scripts/evaluate_model.py --run-dir $(OUT)/xgboost --processed-dir $(BTC_DIR)
	$(RUN) python scripts/run_portfolio_backtest.py --run-dir $(OUT)/xgboost --processed-dir $(BTC_DIR) --confidence 0.50 --initial-equity 100000 --notional-per-trade 10000 --fee-bps 1.0 --slippage-bps 0.5

btc-deeplob:
	$(RUN) python scripts/train_model.py --model deeplob --processed-dir $(BTC_DIR) --output-dir $(OUT)/deeplob --max-train-samples 1000000 --epochs 5 --batch-size 512
	$(RUN) python scripts/evaluate_model.py --run-dir $(OUT)/deeplob --processed-dir $(BTC_DIR)
	$(RUN) python scripts/run_portfolio_backtest.py --run-dir $(OUT)/deeplob --processed-dir $(BTC_DIR) --confidence 0.50 --initial-equity 100000 --notional-per-trade 10000 --fee-bps 1.0 --slippage-bps 0.5

btc-transformer:
	$(RUN) python scripts/train_model.py --model transformer --processed-dir $(BTC_DIR) --output-dir $(OUT)/transformer --max-train-samples 500000 --epochs 5 --batch-size 512
	$(RUN) python scripts/evaluate_model.py --run-dir $(OUT)/transformer --processed-dir $(BTC_DIR)
	$(RUN) python scripts/run_portfolio_backtest.py --run-dir $(OUT)/transformer --processed-dir $(BTC_DIR) --confidence 0.50 --initial-equity 100000 --notional-per-trade 10000 --fee-bps 1.0 --slippage-bps 0.5
