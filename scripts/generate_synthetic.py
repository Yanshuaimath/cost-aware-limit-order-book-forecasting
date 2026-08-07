#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from lob_project.data.synthetic import write_synthetic_fi2010


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic FI-2010-shaped test data")
    parser.add_argument("--output", type=Path, default=Path("data/sample/synthetic_fi2010.txt"))
    parser.add_argument("--n-observations", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = write_synthetic_fi2010(args.output, args.n_observations, args.seed)
    print(f"Wrote synthetic software-test data to {path}")


if __name__ == "__main__":
    main()
