#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import shlex
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a controlled experiment command grid")
    parser.add_argument("--grid", type=Path, default=Path("configs/experiment_grid.yaml"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = yaml.safe_load(args.grid.read_text())
    for model, horizon, seed in itertools.product(
        grid["models"], grid["horizons"], grid["seeds"]
    ):
        processed = args.processed_root / f"fi2010_h{horizon}"
        output = args.output_root / f"{model}_h{horizon}_s{seed}"
        command = [
            "python", "scripts/train_model.py", "--model", model,
            "--processed-dir", str(processed), "--output-dir", str(output),
            "--seed", str(seed),
        ]
        print(" ".join(shlex.quote(part) for part in command))


if __name__ == "__main__":
    main()
