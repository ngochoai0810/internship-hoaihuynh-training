from pathlib import Path

import pandas as pd
from scripts.create_retrain_subset import create_retrain_subset


def test_subset_is_reproducible_and_preserves_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.csv"
    first_destination = tmp_path / "first.csv"
    second_destination = tmp_path / "second.csv"
    original = pd.DataFrame(
        {
            "Id": range(1, 9),
            "LotFrontage": range(60, 68),
            "SalePrice": range(100_000, 180_000, 10_000),
        }
    )
    original.to_csv(source, index=False)

    first = create_retrain_subset(
        source,
        first_destination,
        fraction=0.5,
        random_state=2026,
    )
    second = create_retrain_subset(
        source,
        second_destination,
        fraction=0.5,
        random_state=2026,
    )

    assert len(first) == 4
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(pd.read_csv(source), original)
    pd.testing.assert_frame_equal(pd.read_csv(first_destination), first)
