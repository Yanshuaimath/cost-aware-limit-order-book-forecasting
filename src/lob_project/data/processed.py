from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ProcessedData:
    root: Path
    features: np.ndarray
    labels: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    metadata: dict


def load_processed(
    root: str | Path,
    mmap_mode: str | None = "r",
    split_path: str | Path | None = None,
) -> ProcessedData:
    root = Path(root)
    required = ["features.npy", "labels.npy", "splits.npz", "metadata.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed files in {root}: {missing}")
    features = np.load(root / "features.npy", mmap_mode=mmap_mode)
    labels = np.load(root / "labels.npy", mmap_mode=mmap_mode)
    resolved_split_path = Path(split_path) if split_path is not None else root / "splits.npz"
    if not resolved_split_path.exists():
        raise FileNotFoundError(resolved_split_path)
    split_file = np.load(resolved_split_path)
    metadata = json.loads((root / "metadata.json").read_text())
    return ProcessedData(
        root=root,
        features=features,
        labels=labels,
        train_indices=split_file["train"],
        validation_indices=split_file["validation"],
        test_indices=split_file["test"],
        metadata=metadata,
    )
