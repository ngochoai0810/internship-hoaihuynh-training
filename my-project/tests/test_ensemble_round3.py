from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from ml import baseline_round2, ensemble_round3
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline


def _two_row_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{column: [1.0, 2.0] for column in baseline_round2.NUMERIC_FEATURES},
            "GarageType": ["Attchd", "Detchd"],
            "Neighborhood": ["NAmes", "CollgCr"],
            "ExterQual": ["TA", "Gd"],
            "KitchenQual": ["TA", "Gd"],
            "BsmtQual": ["TA", "Gd"],
        },
        columns=baseline_round2.FEATURE_COLUMNS,
    )


def _training_feature_frame(n_rows: int = 30) -> pd.DataFrame:
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
    return pd.DataFrame(data, columns=baseline_round2.FEATURE_COLUMNS)


class CountingKFold(KFold):
    def __init__(self) -> None:
        super().__init__(n_splits=5, shuffle=True, random_state=42)
        self.split_calls = 0

    def split(
        self,
        x: object,
        y: object = None,
        groups: object = None,
    ) -> Any:
        self.split_calls += 1
        return super().split(x, y, groups)


def test_preflight_rejects_non_ignoring_one_hot_branch() -> None:
    preprocessor = baseline_round2.build_preprocessor()
    preprocessor_runtime = cast(Any, preprocessor)
    garage_pipeline = preprocessor_runtime.transformers[1][1]
    garage_pipeline.named_steps["onehot"].set_params(handle_unknown="error")

    with pytest.raises(RuntimeError, match="GarageType.*handle_unknown='ignore'"):
        ensemble_round3.validate_one_hot_unknown_handling(preprocessor)


def test_preflight_accepts_round2_one_hot_branches() -> None:
    preprocessor = baseline_round2.build_preprocessor()

    ensemble_round3.validate_one_hot_unknown_handling(preprocessor)


def test_round2_preprocessor_ignores_unseen_one_hot_categories() -> None:
    train = _two_row_feature_frame()
    unseen = train.iloc[[0]].copy()
    unseen.loc[:, "GarageType"] = "RareGarage"
    unseen.loc[:, "Neighborhood"] = "RareNeighborhood"
    preprocessor: ColumnTransformer = baseline_round2.build_preprocessor()

    preprocessor.fit(train)
    transformed = preprocessor.transform(unseen)

    garage_values = transformed[0, preprocessor.output_indices_["cat_none"]]
    neighborhood_values = transformed[0, preprocessor.output_indices_["cat_missing"]]
    assert np.isfinite(transformed).all()
    assert float(garage_values.sum()) == 0.0
    assert float(neighborhood_values.sum()) == 0.0


def test_build_random_forest_pipeline_uses_round3_configuration() -> None:
    pipeline = ensemble_round3.build_random_forest_pipeline()
    model = cast(Any, pipeline.named_steps["model"])

    assert isinstance(pipeline, Pipeline)
    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 500
    assert model.random_state == 42
    assert model.n_jobs == -1


def test_gradient_boosting_search_reuses_cv_and_has_twelve_candidates() -> None:
    cv = baseline_round2.make_cv()

    search = ensemble_round3.build_gradient_boosting_search(cv)

    assert isinstance(search, GridSearchCV)
    assert search.cv is cv
    assert search.scoring == "neg_root_mean_squared_error"
    assert isinstance(search.estimator, Pipeline)
    assert isinstance(search.estimator.named_steps["model"], GradientBoostingRegressor)
    assert search.param_grid == {
        "model__n_estimators": [100, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3],
    }
    candidate_count = int(
        np.prod([len(values) for values in search.param_grid.values()])
    )
    assert candidate_count == 12


@pytest.mark.parametrize(
    ("improvement", "threshold", "expected"),
    [(0.03, 0.02, True), (0.02, 0.02, False), (0.01, 0.02, False)],
)
def test_selection_threshold_requires_strict_improvement(
    improvement: float, threshold: float, expected: bool
) -> None:
    assert (
        ensemble_round3.passes_selection_threshold(improvement, threshold) is expected
    )


@pytest.mark.parametrize(
    ("random_forest_passes", "gradient_boosting_passes", "expected"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_stacking_requires_both_tree_models_to_pass(
    random_forest_passes: bool,
    gradient_boosting_passes: bool,
    expected: bool,
) -> None:
    assert (
        ensemble_round3.should_evaluate_stacking(
            random_forest_passes=random_forest_passes,
            gradient_boosting_passes=gradient_boosting_passes,
        )
        is expected
    )


def test_build_stacking_regressor_uses_linear_and_tree_candidates() -> None:
    best_gb_params = {
        "model__n_estimators": 300,
        "model__learning_rate": 0.1,
        "model__max_depth": 2,
    }

    stack = ensemble_round3.build_stacking_regressor(
        best_gradient_boosting_params=best_gb_params,
        ridge_alpha=2.5,
    )

    assert isinstance(stack, StackingRegressor)
    assert [name for name, _estimator in stack.estimators] == [
        "ridge",
        "random_forest",
        "gradient_boosting",
    ]
    assert isinstance(stack.final_estimator, Ridge)
    assert cast(Any, stack.final_estimator).alpha == 2.5
    assert isinstance(stack.cv, KFold)
    assert cast(Any, stack.cv).n_splits == 5
    assert cast(Any, stack.cv).shuffle is True
    assert cast(Any, stack.cv).random_state == 42
    gradient_boosting = dict(stack.estimators)["gradient_boosting"]
    assert gradient_boosting.get_params()["model__n_estimators"] == 300
    assert gradient_boosting.get_params()["model__learning_rate"] == 0.1
    assert gradient_boosting.get_params()["model__max_depth"] == 2


def test_random_forest_importance_is_aggregated_to_fifteen_raw_features() -> None:
    x_train = _training_feature_frame()
    y_train = pd.Series(
        np.log1p(100_000.0 + np.arange(len(x_train)) * 5_000.0),
        name=baseline_round2.TARGET_LOG_COLUMN,
    )
    pipeline = ensemble_round3.build_random_forest_pipeline()
    pipeline.set_params(model__n_estimators=10)
    pipeline.fit(x_train, y_train)

    importance = ensemble_round3.aggregate_random_forest_importance(pipeline)

    assert list(importance.columns) == ["feature", "importance", "rank"]
    assert len(importance) == 15
    assert set(importance["feature"]) == set(baseline_round2.FEATURE_COLUMNS)
    assert (importance["importance"] >= 0.0).all()
    assert float(importance["importance"].sum()) == pytest.approx(1.0)
    assert list(importance["rank"]) == list(range(1, 16))


def test_choose_selected_model_falls_back_to_ridge_or_uses_best_passing_model() -> None:
    no_pass = pd.DataFrame(
        [
            {"model_name": "ridge", "rmse_log_mean": 0.15, "passes_threshold": False},
            {
                "model_name": "random_forest",
                "rmse_log_mean": 0.16,
                "passes_threshold": False,
            },
            {
                "model_name": "gradient_boosting",
                "rmse_log_mean": 0.14,
                "passes_threshold": False,
            },
        ]
    )
    two_pass = no_pass.copy()
    two_pass.loc[two_pass["model_name"] != "ridge", "passes_threshold"] = True

    assert ensemble_round3.choose_selected_model(no_pass) == "ridge"
    assert ensemble_round3.choose_selected_model(two_pass) == "gradient_boosting"


def test_run_fails_preflight_before_loading_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    invalid = baseline_round2.build_preprocessor()
    invalid_runtime = cast(Any, invalid)
    neighborhood_pipeline = invalid_runtime.transformers[2][1]
    neighborhood_pipeline.named_steps["onehot"].set_params(handle_unknown="error")
    monkeypatch.setattr(baseline_round2, "build_preprocessor", lambda: invalid)

    with pytest.raises(RuntimeError, match="Neighborhood.*handle_unknown='ignore'"):
        ensemble_round3.run_ensemble_round3(
            data_path=tmp_path / "does-not-exist.csv",
            artifacts_dir=tmp_path / "artifacts",
        )


def test_run_writes_cv_only_artifacts_and_reuses_one_splitter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_data = _training_feature_frame(40)
    raw_data.insert(0, "Id", range(1, len(raw_data) + 1))
    raw_data[baseline_round2.TARGET_COLUMN] = (
        100_000.0 + np.arange(len(raw_data), dtype=float) ** 2 * 750.0
    )
    data_path = tmp_path / "train.csv"
    artifacts_dir = tmp_path / "ensemble_round3"
    raw_data.to_csv(data_path, index=False)
    shared_cv = CountingKFold()
    monkeypatch.setattr(baseline_round2, "make_cv", lambda: shared_cv)
    monkeypatch.setattr(ensemble_round3, "RANDOM_FOREST_N_ESTIMATORS", 10)
    monkeypatch.setattr(
        ensemble_round3,
        "GRADIENT_BOOSTING_PARAM_GRID",
        {
            "model__n_estimators": [10],
            "model__learning_rate": [0.1],
            "model__max_depth": [2],
        },
    )

    metrics = ensemble_round3.run_ensemble_round3(
        data_path=data_path,
        artifacts_dir=artifacts_dir,
        test_size=0.2,
        random_state=42,
        ridge_alpha=1.0,
    )

    assert shared_cv.split_calls == 3
    assert set(metrics["model_name"]) == {
        "ridge",
        "random_forest",
        "gradient_boosting",
    }
    assert int(metrics["selected"].sum()) == 1
    assert metrics.loc[metrics["model_name"] == "ridge", "selected"].item()
    assert {
        "improvement_vs_ridge",
        "selection_threshold",
        "passes_threshold",
        "selected",
    }.issubset(metrics.columns)

    metrics_saved = pd.read_csv(artifacts_dir / "metrics.csv")
    folds_saved = pd.read_csv(artifacts_dir / "fold_metrics.csv")
    search_saved = pd.read_csv(artifacts_dir / "gradient_boosting_search.csv")
    importance_saved = pd.read_csv(
        artifacts_dir / "random_forest_feature_importance.csv"
    )
    note = (artifacts_dir / "README.md").read_text(encoding="utf-8")

    assert len(metrics_saved) == 3
    assert len(folds_saved) == 15
    assert set(folds_saved["fold"]) == {1, 2, 3, 4, 5}
    assert len(search_saved) == 1
    assert len(importance_saved) == 15
    assert "best-of-12" in note
    assert "final holdout" in note
    assert not any(
        "holdout" in column.lower()
        and ("rmse" in column.lower() or "score" in column.lower())
        for column in metrics_saved.columns
    )
    assert not any(
        path.suffix in {".pkl", ".joblib"} for path in artifacts_dir.iterdir()
    )


def test_split_train_eval_only_drops_final_holdout_rows() -> None:
    x_raw = pd.DataFrame({"value": range(10)})
    y_log = pd.Series(np.arange(10, dtype=float), name="SalePriceLog")

    x_train_eval, y_train_eval = ensemble_round3.split_train_eval_only(
        x_raw=x_raw,
        y_log=y_log,
        test_size=0.2,
        random_state=42,
    )

    assert list(x_train_eval.index) == [5, 0, 7, 2, 9, 4, 3, 6]
    assert list(y_train_eval.index) == [5, 0, 7, 2, 9, 4, 3, 6]
    assert {1, 8}.isdisjoint(x_train_eval.index)


def test_optional_stacking_skips_evaluation_unless_both_trees_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_evaluation(*_args: object, **_kwargs: object) -> list[float]:
        raise AssertionError("stacking evaluation must be skipped")

    monkeypatch.setattr(
        ensemble_round3, "_cross_validate_rmse_log", unexpected_evaluation
    )

    scores = ensemble_round3.evaluate_stacking_if_eligible(
        random_forest_passes=True,
        gradient_boosting_passes=False,
        best_gradient_boosting_params={
            "model__n_estimators": 10,
            "model__learning_rate": 0.1,
            "model__max_depth": 2,
        },
        ridge_alpha=1.0,
        x_train_eval=_training_feature_frame(10),
        y_train_eval=pd.Series(np.arange(10, dtype=float)),
        cv=baseline_round2.make_cv(),
    )

    assert scores is None


def test_optional_stacking_uses_shared_outer_cv_when_both_trees_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_train_eval = _training_feature_frame(10)
    y_train_eval = pd.Series(np.arange(10, dtype=float))
    shared_cv = baseline_round2.make_cv()
    observed: dict[str, object] = {}

    def record_evaluation(
        estimator: object,
        x_data: pd.DataFrame,
        y_data: pd.Series,
        cv: KFold,
    ) -> list[float]:
        observed.update({"estimator": estimator, "x": x_data, "y": y_data, "cv": cv})
        return [0.08, 0.09, 0.07, 0.08, 0.1]

    monkeypatch.setattr(ensemble_round3, "_cross_validate_rmse_log", record_evaluation)

    scores = ensemble_round3.evaluate_stacking_if_eligible(
        random_forest_passes=True,
        gradient_boosting_passes=True,
        best_gradient_boosting_params={
            "model__n_estimators": 10,
            "model__learning_rate": 0.1,
            "model__max_depth": 2,
        },
        ridge_alpha=1.0,
        x_train_eval=x_train_eval,
        y_train_eval=y_train_eval,
        cv=shared_cv,
    )

    assert scores == [0.08, 0.09, 0.07, 0.08, 0.1]
    assert isinstance(observed["estimator"], StackingRegressor)
    assert observed["x"] is x_train_eval
    assert observed["y"] is y_train_eval
    assert observed["cv"] is shared_cv


def test_run_adds_passing_stack_and_never_sends_holdout_downstream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_data = _training_feature_frame(40)
    raw_data[baseline_round2.TARGET_COLUMN] = (
        100_000.0 + np.arange(len(raw_data), dtype=float) ** 2 * 750.0
    )
    data_path = tmp_path / "train.csv"
    artifacts_dir = tmp_path / "ensemble_round3"
    raw_data.to_csv(data_path, index=False)
    expected_train_indices = {
        39,
        6,
        25,
        9,
        13,
        31,
        34,
        8,
        17,
        24,
        0,
        33,
        5,
        11,
        1,
        29,
        21,
        2,
        30,
        36,
        3,
        35,
        23,
        32,
        10,
        22,
        18,
        20,
        7,
        14,
        28,
        38,
    }
    holdout_indices = {19, 16, 15, 26, 4, 12, 37, 27}
    shared_cv = baseline_round2.make_cv()
    cv_inputs: list[tuple[set[int], KFold]] = []

    class RecordingForest:
        def __init__(self) -> None:
            self.fit_indices: set[int] = set()

        def fit(self, x_data: pd.DataFrame, _y_data: pd.Series) -> "RecordingForest":
            self.fit_indices = set(x_data.index)
            return self

    class RecordingSearch:
        def __init__(self, cv: KFold) -> None:
            self.cv = cv
            self.fit_indices: set[int] = set()
            self.best_params_ = {
                "model__n_estimators": 300,
                "model__learning_rate": 0.1,
                "model__max_depth": 2,
            }
            self.best_index_ = 0
            self.n_splits_ = 5
            self.cv_results_ = {
                "params": [self.best_params_],
                "mean_test_score": np.array([-0.09]),
                "std_test_score": np.array([0.0]),
                "rank_test_score": np.array([1]),
                **{f"split{fold}_test_score": np.array([-0.09]) for fold in range(5)},
            }

        def fit(self, x_data: pd.DataFrame, _y_data: pd.Series) -> "RecordingSearch":
            self.fit_indices = set(x_data.index)
            return self

    recording_forest = RecordingForest()
    recording_search = RecordingSearch(shared_cv)

    def record_cv(
        estimator: object,
        x_data: pd.DataFrame,
        _y_data: pd.Series,
        cv: KFold,
    ) -> list[float]:
        cv_inputs.append((set(x_data.index), cv))
        if isinstance(estimator, StackingRegressor):
            return [0.08, 0.08, 0.08, 0.08, 0.08]
        if isinstance(estimator, RecordingForest):
            return [0.10, 0.10, 0.10, 0.10, 0.10]
        return [0.19, 0.20, 0.21, 0.20, 0.20]

    monkeypatch.setattr(baseline_round2, "make_cv", lambda: shared_cv)
    monkeypatch.setattr(
        ensemble_round3,
        "build_random_forest_pipeline",
        lambda: recording_forest,
    )
    monkeypatch.setattr(
        ensemble_round3,
        "build_gradient_boosting_search",
        lambda cv: recording_search,
    )
    monkeypatch.setattr(ensemble_round3, "_cross_validate_rmse_log", record_cv)
    monkeypatch.setattr(
        ensemble_round3,
        "aggregate_random_forest_importance",
        lambda _pipeline: pd.DataFrame(
            {
                "feature": baseline_round2.FEATURE_COLUMNS,
                "importance": [1.0 / 15.0] * 15,
                "rank": range(1, 16),
            }
        ),
    )

    metrics = ensemble_round3.run_ensemble_round3(
        data_path=data_path,
        artifacts_dir=artifacts_dir,
    )

    assert len(cv_inputs) == 3
    assert all(indices == expected_train_indices for indices, _cv in cv_inputs)
    assert all(cv is shared_cv for _indices, cv in cv_inputs)
    assert recording_search.cv is shared_cv
    assert recording_search.fit_indices == expected_train_indices
    assert recording_forest.fit_indices == expected_train_indices
    assert all(holdout_indices.isdisjoint(indices) for indices, _cv in cv_inputs)
    assert set(metrics["model_name"]) == {
        "ridge",
        "random_forest",
        "gradient_boosting",
        "stacking",
    }
    assert metrics.loc[metrics["selected"], "model_name"].item() == "stacking"
    fold_metrics = pd.read_csv(artifacts_dir / "fold_metrics.csv")
    assert len(fold_metrics) == 20
    assert len(fold_metrics.loc[fold_metrics["model_name"] == "stacking"]) == 5
