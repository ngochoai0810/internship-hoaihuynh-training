"""Create a deterministic training subset for the Week 9 retrain simulation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_DIR / "data" / "raw" / "train.csv"
DEFAULT_DESTINATION = PROJECT_DIR / "data" / "retrain" / "train_subset.csv"


def create_retrain_subset(
    source: Path,
    destination: Path,
    *,
    fraction: float = 0.6,
    random_state: int = 2026,
) -> pd.DataFrame:
    """Sample and persist a reproducible subset without changing source data."""

    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")

    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("source and destination must be different files")
    if not source_path.exists():
        raise FileNotFoundError(f"Training data not found: {source_path}")

    data = pd.read_csv(source_path)
    subset = (
        data.sample(frac=fraction, random_state=random_state)
        .sort_index()
        .reset_index(drop=True)
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(destination_path, index=False)
    return subset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--random-state", type=int, default=2026)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    subset = create_retrain_subset(
        args.source,
        args.destination,
        fraction=args.fraction,
        random_state=args.random_state,
    )
    print(f"Created {len(subset)} rows at {args.destination}")
    return Path(args.destination)


if __name__ == "__main__":
    main()
