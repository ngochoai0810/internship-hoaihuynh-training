"""Download and extract the Kaggle House Prices competition dataset."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from zipfile import ZipFile

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DESTINATION = PROJECT_DIR / "data" / "raw"
COMPETITION = "house-prices-advanced-regression-techniques"
ARCHIVE_NAME = f"{COMPETITION}.zip"


class KaggleApiLike(Protocol):
    def authenticate(self) -> None: ...

    def competition_download_files(
        self,
        competition: str,
        path: str,
        quiet: bool,
    ) -> None: ...


@dataclass(frozen=True)
class DownloadResult:
    train_csv: Path
    test_csv: Path


def _default_api() -> KaggleApiLike:
    from kaggle.api.kaggle_api_extended import KaggleApi

    return cast(KaggleApiLike, KaggleApi())


def _extract_required_csvs(archive_path: Path, destination: Path) -> DownloadResult:
    required = {"train.csv", "test.csv"}
    with ZipFile(archive_path) as archive:
        names = {Path(name).name: name for name in archive.namelist()}
        missing = required - set(names)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise FileNotFoundError(f"Kaggle archive is missing: {missing_list}")

        for filename in sorted(required):
            (destination / filename).write_bytes(archive.read(names[filename]))

    return DownloadResult(
        train_csv=destination / "train.csv",
        test_csv=destination / "test.csv",
    )


def download_house_prices(
    destination: Path = DEFAULT_DESTINATION,
    *,
    api: KaggleApiLike | None = None,
) -> DownloadResult:
    """Authenticate, download, and extract the competition train/test CSVs."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    kaggle_api = api or _default_api()
    kaggle_api.authenticate()
    kaggle_api.competition_download_files(
        COMPETITION,
        path=str(destination),
        quiet=True,
    )
    return _extract_required_csvs(destination / ARCHIVE_NAME, destination)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> DownloadResult:
    args = parse_args(argv)
    result = download_house_prices(args.destination)
    print(f"Downloaded train data to {result.train_csv}")
    print(f"Downloaded test data to {result.test_csv}")
    return result


if __name__ == "__main__":
    main()
