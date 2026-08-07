#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from lob_project.data.fi2010 import WindowedLOBDataset
from lob_project.data.processed import load_processed
from lob_project.features.sequence import subsample_indices, summarize_windows
from lob_project.models.baseline import make_baseline
from lob_project.models.factory import make_torch_model
from lob_project.models.xgboost_model import (
    class_balanced_sample_weights,
    make_xgboost,
)
from lob_project.training.engine import train_torch_model
from lob_project.training.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LOB forecasting model")
    parser.add_argument(
        "--model",
        choices=["baseline", "xgboost", "deeplob", "transformer"],
        required=True,
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    data = load_processed(args.processed_dir, split_path=args.split_file)
    sequence_length = int(data.metadata["sequence_length"])
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["processed_dir"] = str(args.processed_dir)
    config["output_dir"] = str(args.output_dir)
    config["split_file"] = str(args.split_file) if args.split_file is not None else None
    config["sequence_length"] = sequence_length
    (output / "run_config.json").write_text(json.dumps(config, indent=2))

    train_indices = subsample_indices(data.train_indices, args.max_train_samples)
    if args.model in {"baseline", "xgboost"}:
        classical_source = data.features
        classical_feature_source = "lob_features"
        tabular_path = args.processed_dir / "tabular_features.npy"
        if args.model == "xgboost" and tabular_path.exists():
            classical_source = np.load(tabular_path, mmap_mode="r")
            classical_feature_source = "microstructure_features"
        x_train = summarize_windows(classical_source, train_indices, sequence_length)
        y_train = np.asarray(data.labels[train_indices], dtype=np.int64)
        if np.any(y_train < 0):
            raise ValueError("training indices contain invalid labels")
        if args.model == "baseline":
            model = make_baseline(seed=args.seed)
            model.fit(x_train, y_train)
        else:
            model = make_xgboost(
                seed=args.seed,
                n_estimators=args.xgb_estimators,
                max_depth=args.xgb_max_depth,
                learning_rate=args.xgb_learning_rate,
                subsample=args.xgb_subsample,
                colsample_bytree=args.xgb_colsample_bytree,
                n_jobs=args.xgb_n_jobs,
            )
            model.fit(
                x_train,
                y_train,
                sample_weight=class_balanced_sample_weights(y_train),
            )
        config["classical_feature_source"] = classical_feature_source
        (output / "run_config.json").write_text(json.dumps(config, indent=2))
        joblib.dump(model, output / "model.joblib")
        print(f"Saved {args.model} model to {output / 'model.joblib'}")
        return

    train_dataset = WindowedLOBDataset(
        data.features, data.labels, train_indices, sequence_length
    )
    validation_dataset = WindowedLOBDataset(
        data.features, data.labels, data.validation_indices, sequence_length
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = make_torch_model(
        args.model,
        sequence_length=sequence_length,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = train_torch_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
    )
    torch.save(
        {
            "model_name": args.model,
            "model_state": result.model_state,
            "sequence_length": sequence_length,
            "model_kwargs": {
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
            },
        },
        output / "model.pt",
    )
    (output / "history.json").write_text(json.dumps(result.history, indent=2))
    summary = {
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "device": str(device),
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
