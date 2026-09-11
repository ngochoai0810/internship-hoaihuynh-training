import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ml import finalize


def _raw_training_frame(n_rows: int = 20) -> pd.DataFrame:
    """Create compact raw data covering the 15 selected features and the target."""
    quality = ["TA", "Gd", "Ex", "Fa"]
    garage_types = ["Attchd", "Detchd", "BuiltIn", "CarPort"]
    neighborhoods = ["CollgCr", "OldTown", "Edwards", "Somerst"]
    return pd.DataFrame(
        {
            "Id": range(1, n_rows + 1),
            "OverallQual": [4 + (index % 7) for index in range(n_rows)],
            "GrLivArea": [900.0 + 80 * index for index in range(n_rows)],
            "GarageCars": [
                np.nan if index == 3 else float(1 + index % 3)
                for index in range(n_rows)
            ],
            "GarageArea": [200.0 + 30 * index for index in range(n_rows)],
            "TotalBsmtSF": [600.0 + 40 * index for index in range(n_rows)],
            "1stFlrSF": [650.0 + 45 * index for index in range(n_rows)],
            "FullBath": [1 + index % 3 for index in range(n_rows)],
            "TotRmsAbvGrd": [4 + index % 6 for index in range(n_rows)],
            "YearBuilt": [1950 + 3 * index for index in range(n_rows)],
            "YearRemodAdd": [1980 + 2 * index for index in range(n_rows)],
            "Neighborhood": [neighborhoods[index % 4] for index in range(n_rows)],
            "GarageType": [garage_types[index % 4] for index in range(n_rows)],
            "ExterQual": [quality[index % 4] for index in range(n_rows)],
            "KitchenQual": [quality[(index + 1) % 4] for index in range(n_rows)],
            "BsmtQual": [
                None if index == 5 else quality[(index + 2) % 4]
                for index in range(n_rows)
            ],
            "SalePrice": [120_000.0 + 9_000 * index for index in range(n_rows)],
        }
    )


def _round3_metrics(winner: str = "ridge") -> pd.DataFrame:
    """Build a Round 3 metrics frame whose selection rule picks `winner`."""
    return pd.DataFrame(
        [
            {
                "model_name": "ridge",
                "rmse_log_mean": 0.15848,
                "passes_threshold": False,
            },
            {
                "model_name": "gradient_boosting",
                "rmse_log_mean": 0.14599,
                "passes_threshold": winner == "gradient_boosting",
            },
        ]
    )


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    """Write the raw data and the two upstream selection artifacts."""
    data_path = tmp_path / "train.csv"
    _raw_training_frame().to_csv(data_path, index=False)

    metrics_path = tmp_path / "round3_metrics.csv"
    _round3_metrics().to_csv(metrics_path, index=False)

    search_path = tmp_path / "ridge_alpha_search.csv"
    pd.DataFrame(
        [
            {"alpha": 1.0, "selected": True},
            {"alpha": 10.0, "selected": False},
        ]
    ).to_csv(search_path, index=False)

    return {
        "data_path": data_path,
        "round3_metrics_path": metrics_path,
        "ridge_search_path": search_path,
        "artifacts_dir": tmp_path / "final",
        "model_path": tmp_path / "model.pkl",
    }


def test_load_selected_model_name_returns_ridge(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    _round3_metrics().to_csv(metrics_path, index=False)

    assert finalize.load_selected_model_name(metrics_path) == "ridge"


def test_load_selected_model_name_rejects_unsupported_winner(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    _round3_metrics(winner="gradient_boosting").to_csv(metrics_path, index=False)

    with pytest.raises(NotImplementedError, match="gradient_boosting"):
        finalize.load_selected_model_name(metrics_path)


def test_load_selected_ridge_alpha_reads_selected_row(tmp_path: Path) -> None:
    search_path = tmp_path / "search.csv"
    pd.DataFrame(
        [
            {"alpha": 0.1, "selected": False},
            {"alpha": 3.0, "selected": True},
        ]
    ).to_csv(search_path, index=False)

    assert finalize.load_selected_ridge_alpha(search_path) == 3.0


@pytest.mark.parametrize("selected_flags", [[False, False], [True, True]])
def test_load_selected_ridge_alpha_requires_exactly_one_row(
    tmp_path: Path, selected_flags: list[bool]
) -> None:
    search_path = tmp_path / "search.csv"
    pd.DataFrame(
        [
            {"alpha": 1.0, "selected": selected_flags[0]},
            {"alpha": 10.0, "selected": selected_flags[1]},
        ]
    ).to_csv(search_path, index=False)

    with pytest.raises(ValueError, match="exactly one selected Ridge alpha"):
        finalize.load_selected_ridge_alpha(search_path)


def test_run_finalize_writes_one_artifact_and_report(inputs: dict[str, Path]) -> None:
    report = finalize.run_finalize(test_size=0.25, **inputs)

    assert report["model_name"] == "ridge"
    assert report["ridge_alpha"] == 1.0
    assert report["holdout_rows"] == 5
    assert report["train_eval_rows"] == 15
    assert inputs["model_path"].exists()

    report_path = inputs["artifacts_dir"] / "final_holdout_report.json"
    saved = json.loads(report_path.read_text())
    assert saved["model_sha256"] == report["model_sha256"]
    assert set(saved["metrics"]) == {
        "rmse_log",
        "mae_log",
        "r2_log",
        "rmse_price",
        "mae_price",
        "r2_price",
    }
    assert (inputs["artifacts_dir"] / "README.md").exists()


def test_run_finalize_writes_demo_samples_matching_schema(
    inputs: dict[str, Path],
) -> None:
    finalize.run_finalize(test_size=0.25, **inputs)

    samples = json.loads((inputs["artifacts_dir"] / "demo_samples.json").read_text())
    assert len(samples) == finalize.DEMO_SAMPLE_COUNT
    for sample in samples:
        assert sample["actual_price"] > 0
        assert set(sample["payload"]) == set(finalize.PAYLOAD_FIELD_BY_COLUMN.values())
        assert isinstance(sample["payload"]["overall_qual"], int)


def test_run_finalize_refuses_to_score_holdout_twice(inputs: dict[str, Path]) -> None:
    finalize.run_finalize(test_size=0.25, **inputs)

    with pytest.raises(RuntimeError, match="single-use report number"):
        finalize.run_finalize(test_size=0.25, **inputs)

    second = finalize.run_finalize(test_size=0.25, allow_overwrite=True, **inputs)
    assert second["holdout_rows"] == 5


def test_run_finalize_smoke_tests_the_reloaded_artifact(
    inputs: dict[str, Path],
) -> None:
    report = finalize.run_finalize(test_size=0.25, **inputs)

    prices = report["smoke_test_prices"]
    assert len(prices) == finalize.DEMO_SAMPLE_COUNT
    assert all(0.0 < price < finalize.MAX_PLAUSIBLE_PRICE for price in prices)


def test_run_finalize_keeps_holdout_rows_out_of_training(
    inputs: dict[str, Path],
) -> None:
    finalize.run_finalize(test_size=0.25, **inputs)

    samples = json.loads((inputs["artifacts_dir"] / "demo_samples.json").read_text())
    holdout_indices = {sample["index"] for sample in samples}

    from ml import baseline_round2

    x_raw, y_log = baseline_round2.load_raw_training_data(inputs["data_path"])
    x_train_eval, x_holdout, _y_train, _y_holdout = baseline_round2.split_raw_holdout(
        x_raw=x_raw,
        y_log=y_log,
        test_size=0.25,
        random_state=baseline_round2.DEFAULT_RANDOM_STATE,
    )
    assert holdout_indices <= set(x_holdout.index)
    assert not holdout_indices & set(x_train_eval.index)
