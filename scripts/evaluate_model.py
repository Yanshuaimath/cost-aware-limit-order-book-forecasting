#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from lob_project.data.fi2010 import WindowedLOBDataset
from lob_project.data.processed import load_processed
from lob_project.features.sequence import summarize_windows
from lob_project.models.factory import make_torch_model
from lob_project.training.engine import predict_probabilities
from lob_project.training.metrics import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained LOB model")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((args.run_dir / "run_config.json").read_text())
    model_name = config["model"]
    data = load_processed(args.processed_dir, split_path=args.split_file)
    sequence_length = int(data.metadata["sequence_length"])
    started = time.perf_counter()

    if model_name in {"baseline", "xgboost"}:
        model = joblib.load(args.run_dir / "model.joblib")
        classical_source = data.features
        if config.get("classical_feature_source") == "microstructure_features":
            classical_source = np.load(
                args.processed_dir / "tabular_features.npy", mmap_mode="r"
            )
        x_test = summarize_windows(classical_source, data.test_indices, sequence_length)
        probabilities = model.predict_proba(x_test)
        y_true = np.asarray(data.labels[data.test_indices], dtype=np.int64)
        classes = np.asarray(model.classes_, dtype=np.int64)
        if not np.array_equal(classes, np.array([0, 1, 2])):
            full = np.zeros((len(probabilities), 3), dtype=np.float64)
            full[:, classes] = probabilities
            probabilities = full
    else:
        checkpoint = torch.load(args.run_dir / "model.pt", map_location="cpu", weights_only=False)
        model = make_torch_model(
            checkpoint["model_name"],
            sequence_length=checkpoint["sequence_length"],
            **checkpoint.get("model_kwargs", {}),
        )
        model.load_state_dict(checkpoint["model_state"])
        dataset = WindowedLOBDataset(
            data.features, data.labels, data.test_indices, sequence_length
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        probabilities, y_true = predict_probabilities(model, loader, device)

    elapsed = time.perf_counter() - started
    metrics = classification_metrics(y_true, probabilities)
    metrics["evaluation_seconds"] = elapsed
    metrics["samples_per_second"] = len(y_true) / max(elapsed, 1e-9)
    np.savez(
        args.run_dir / "test_predictions.npz",
        probabilities=probabilities.astype(np.float32),
        y_true=y_true,
        target_indices=data.test_indices,
    )
    (args.run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}, indent=2))


if __name__ == "__main__":
    main()
