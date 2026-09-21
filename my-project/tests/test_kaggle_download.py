from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest
from scripts.download_house_prices import download_house_prices


class FakeKaggleApi:
    def __init__(self, archive_path: Path) -> None:
        self.archive_path = archive_path
        self.authenticated = False
        self.download_args: tuple[str, str, bool] | None = None

    def authenticate(self) -> None:
        self.authenticated = True

    def competition_download_files(
        self,
        competition: str,
        path: str,
        quiet: bool,
    ) -> None:
        self.download_args = (competition, path, quiet)
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "house-prices-advanced-regression-techniques.zip").write_bytes(
            self.archive_path.read_bytes()
        )


def _create_competition_zip(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("train.csv", "Id,SalePrice\n1,100000\n")
        archive.writestr("test.csv", "Id\n1\n")


def test_download_house_prices_authenticates_downloads_and_extracts_csvs(
    tmp_path: Path,
) -> None:
    source_zip = tmp_path / "source.zip"
    _create_competition_zip(source_zip)
    fake_api = FakeKaggleApi(source_zip)

    result = download_house_prices(tmp_path / "raw", api=fake_api)

    assert fake_api.authenticated
    assert fake_api.download_args == (
        "house-prices-advanced-regression-techniques",
        str(tmp_path / "raw"),
        True,
    )
    assert result.train_csv == tmp_path / "raw" / "train.csv"
    assert result.test_csv == tmp_path / "raw" / "test.csv"
    assert result.train_csv.read_text() == "Id,SalePrice\n1,100000\n"
    assert result.test_csv.read_text() == "Id\n1\n"


def test_download_house_prices_rejects_archives_missing_train_csv(
    tmp_path: Path,
) -> None:
    source_zip = tmp_path / "source.zip"
    with ZipFile(source_zip, "w") as archive:
        archive.writestr("test.csv", "Id\n1\n")
    fake_api = FakeKaggleApi(source_zip)

    with pytest.raises(FileNotFoundError, match="train.csv"):
        download_house_prices(tmp_path / "raw", api=fake_api)
