from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from ml import baseline_round2, ridge_tuning
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline


def _cv_results(
    mean_test_scores: list[float],
    baseline_std: float,
) -> dict[str, Any]:
    return {
        "params": [
            {"model__alpha": 0.1},
            {"model__alpha": 1.0},
            {"model__alpha": 10.0},
        ],
        "mean_test_score": np.asarray(mean_test_scores, dtype=float),
        "std_test_score": np.asarray([0.01, baseline_std, 0.01], dtype=float),
    }


def _raw_training_data(n_rows: int = 40) -> pd.DataFrame:
    row_numbers = np.arange(n_rows, dtype=float)
    data: dict[str, object] = {
        column: row_numbers + feature_number
        for feature_number, column in enumerate(baseline_round2.NUMERIC_FEATURES)
    }
    data.update(
        {
            "GarageType": np.resize(["Attchd", "Detchd", "BuiltIn"], n_rows),
            "Neighborhood": np.resize(["NAmes", "CollgCr", "OldTown"], n_rows),
            "ExterQual": np.resize(["Fa", "TA", "Gd", "Ex"], n_rows),
            "KitchenQual": np.resize(["TA", "Gd", "Ex"], n_rows),
            "BsmtQual": np.resize(["Fa", "TA", "Gd"], n_rows),
        }
    )
    frame = pd.DataFrame(data, columns=baseline_round2.FEATURE_COLUMNS)
    frame.insert(0, "Id", range(1, n_rows + 1))
    frame[baseline_round2.TARGET_COLUMN] = 100_000.0 + row_numbers**2 * 750.0
    return frame


class RecordingKFold(KFold):
    def __init__(self) -> None:
        super().__init__(n_splits=5, shuffle=True, random_state=42)
        self.seen_indices: set[int] = set()
        self.split_calls = 0

    def split(
        self,
        x: pd.DataFrame,
        y: pd.Series | None = None,
        groups: object = None,
    ) -> Any:
        self.split_calls += 1
        self.seen_indices = set(x.index)
        return super().split(x, y, groups)


def test_refit_policy_selects_raw_best_only_above_baseline_std() -> None:
    results = _cv_results(
        mean_test_scores=[-0.10, -0.25, -0.30],
        baseline_std=0.125,
    )

    selected_index = ridge_tuning.select_refit_index(results)

    assert selected_index == 0


@pytest.mark.parametrize(
    "candidate_score",
    [-0.125, -0.20],
)
def test_refit_policy_keeps_alpha_one_at_or_below_threshold(
    candidate_score: float,
) -> None:
    results = _cv_results(
        mean_test_scores=[candidate_score, -0.25, -0.30],
        baseline_std=0.125,
    )

    selected_index = ridge_tuning.select_refit_index(results)

    assert selected_index == 1


def test_refit_policy_keeps_alpha_one_when_it_is_raw_best() -> None:
    results = _cv_results(
        mean_test_scores=[-0.30, -0.20, -0.25],
        baseline_std=0.05,
    )

    selected_index = ridge_tuning.select_refit_index(results)

    assert selected_index == 1


def test_build_search_uses_shared_cv_train_scores_and_callable_refit() -> None:
    cv = baseline_round2.make_cv()

    search = ridge_tuning.build_ridge_alpha_search(cv)

    assert isinstance(search, GridSearchCV)
    assert search.cv is cv
    assert search.scoring == "neg_root_mean_squared_error"
    assert search.return_train_score is True
    assert search.refit is ridge_tuning.select_refit_index
    assert search.param_grid == {
        "model__alpha": [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    }


def test_run_returns_fitted_policy_selection_and_writes_nine_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_data = _raw_training_data()
    data_path = tmp_path / "train.csv"
    artifacts_dir = tmp_path / "ridge_tuning"
    raw_data.to_csv(data_path, index=False)
    shared_cv = RecordingKFold()
    monkeypatch.setattr(baseline_round2, "make_cv", lambda: shared_cv)
    expected_x_train_eval, expected_x_holdout, _, _ = baseline_round2.split_raw_holdout(
        x_raw=raw_data[baseline_round2.FEATURE_COLUMNS],
        y_log=np.log1p(raw_data[baseline_round2.TARGET_COLUMN]),
        test_size=0.2,
        random_state=42,
    )

    def fail_on_step_11_inference(*args: object, **kwargs: object) -> None:
        raise AssertionError("Step 11 must not predict or score the final holdout")

    monkeypatch.setattr(GridSearchCV, "predict", fail_on_step_11_inference)
    monkeypatch.setattr(GridSearchCV, "score", fail_on_step_11_inference)

    result = ridge_tuning.run_ridge_tuning(
        data_path=data_path,
        artifacts_dir=artifacts_dir,
        test_size=0.2,
        random_state=42,
    )

    assert isinstance(result, ridge_tuning.RidgeTuningResult)
    assert result.search.cv is shared_cv
    assert shared_cv.split_calls == 1
    assert shared_cv.seen_indices == set(expected_x_train_eval.index)
    assert set(expected_x_holdout.index).isdisjoint(shared_cv.seen_indices)

    assert len(result.candidates) == 9
    assert list(result.candidates.columns) == [
        "alpha",
        "mean_train_rmse_log",
        "std_train_rmse_log",
        "mean_cv_rmse_log",
        "std_cv_rmse_log",
        "generalization_gap",
        "rank",
        "improvement_vs_alpha_1",
        "selection_threshold",
        "is_raw_best",
        "passes_threshold",
        "selected",
    ]
    selected_row = result.candidates.loc[result.candidates["selected"]].iloc[0]
    assert int(result.candidates["selected"].sum()) == 1
    assert float(selected_row["alpha"]) == result.selected_alpha
    assert float(selected_row["mean_cv_rmse_log"]) == result.selected_cv_rmse_mean
    assert float(selected_row["std_cv_rmse_log"]) == result.selected_cv_rmse_std
    np.testing.assert_allclose(
        result.candidates["generalization_gap"],
        result.candidates["mean_cv_rmse_log"]
        - result.candidates["mean_train_rmse_log"],
    )

    assert "mean_train_score" in result.search.cv_results_
    for fold in range(5):
        assert f"split{fold}_train_score" in result.search.cv_results_
        assert f"split{fold}_test_score" in result.search.cv_results_
    assert not hasattr(result.search, "best_score_")
    assert result.search.best_index_ == int(selected_row.name)
    assert float(result.search.best_params_["model__alpha"]) == result.selected_alpha
    best_pipeline = cast(Pipeline, result.search.best_estimator_)
    best_ridge = cast(Ridge, best_pipeline.named_steps["model"])
    assert float(best_ridge.alpha) == result.selected_alpha
    predictions = best_pipeline.predict(
        raw_data.loc[list(shared_cv.seen_indices)[:2], baseline_round2.FEATURE_COLUMNS]
    )
    assert np.isfinite(predictions).all()

    saved_candidates = pd.read_csv(artifacts_dir / "ridge_alpha_search.csv")
    note = (artifacts_dir / "README.md").read_text(encoding="utf-8")
    assert len(saved_candidates) == 9
    assert int(saved_candidates["selected"].sum()) == 1
    pd.testing.assert_frame_equal(
        saved_candidates,
        result.candidates,
        check_dtype=False,
    )
    assert "best-of-9" in note
    assert "callable refit" in note
    assert "same five CV folds" in note
    assert "final holdout" in note
    assert not any(
        "holdout" in column.lower()
        and ("rmse" in column.lower() or "score" in column.lower())
        for column in saved_candidates.columns
    )
    assert not any(
        path.suffix in {".pkl", ".joblib"} for path in artifacts_dir.iterdir()
    )
