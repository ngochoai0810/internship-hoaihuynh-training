"""Cross-validated ensemble experiments for House Prices Round 3."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from ml import baseline_round2
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DEFAULT_RANDOM_STATE = baseline_round2.DEFAULT_RANDOM_STATE
DEFAULT_DATA_PATH = baseline_round2.DEFAULT_DATA_PATH
DEFAULT_ARTIFACTS_DIR = baseline_round2.PROJECT_DIR / "artifacts" / "ensemble_round3"
ROUND_NAME = "ensemble_round3"
RANDOM_FOREST_N_ESTIMATORS = 500
GRADIENT_BOOSTING_PARAM_GRID: dict[str, list[float | int]] = {
    "model__n_estimators": [100, 300],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [2, 3],
}
RMSE_SCORING = "neg_root_mean_squared_error"
METHODOLOGY_NOTE = (
    "The best-of-12 Gradient Boosting CV score is selected and compared on the "
    "same folds, so it may be mildly optimistic. The final holdout stays untouched "
    "until step 13 and provides the unbiased final estimate."
)


ONE_HOT_BRANCHES: tuple[tuple[str, str], ...] = (
    ("cat_none", "GarageType"),
    ("cat_missing", "Neighborhood"),
)


def validate_one_hot_unknown_handling(preprocessor: ColumnTransformer) -> None:
    """Fail unless both Round 2 one-hot branches ignore unseen categories."""
    transformers = {
        name: transformer
        for name, transformer, _columns in cast(Any, preprocessor).transformers
    }
    for branch_name, feature_name in ONE_HOT_BRANCHES:
        transformer = transformers.get(branch_name)
        if not isinstance(transformer, Pipeline):
            raise RuntimeError(
                f"{feature_name} branch must be a preprocessing Pipeline"
            )
        encoder = transformer.named_steps.get("onehot")
        if (
            not isinstance(encoder, OneHotEncoder)
            or encoder.get_params()["handle_unknown"] != "ignore"
        ):
            raise RuntimeError(
                f"{feature_name} one-hot branch must use handle_unknown='ignore'"
            )


def build_random_forest_pipeline() -> Pipeline:
    """Build the fixed Random Forest candidate with Round 2 preprocessing."""
    return Pipeline(
        steps=[
            ("preprocessor", baseline_round2.build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=RANDOM_FOREST_N_ESTIMATORS,
                    random_state=DEFAULT_RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _build_gradient_boosting_pipeline() -> Pipeline:
    """Build an untuned Gradient Boosting candidate."""
    return Pipeline(
        steps=[
            ("preprocessor", baseline_round2.build_preprocessor()),
            ("model", GradientBoostingRegressor(random_state=DEFAULT_RANDOM_STATE)),
        ]
    )


def build_gradient_boosting_search(cv: KFold) -> GridSearchCV:
    """Build the 12-candidate Gradient Boosting search using the shared CV."""
    return GridSearchCV(
        estimator=_build_gradient_boosting_pipeline(),
        param_grid=GRADIENT_BOOSTING_PARAM_GRID,
        scoring=RMSE_SCORING,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )


def passes_selection_threshold(improvement: float, threshold: float) -> bool:
    """Return whether an RMSE improvement strictly exceeds the Ridge CV std."""
    return improvement > threshold


def should_evaluate_stacking(
    random_forest_passes: bool,
    gradient_boosting_passes: bool,
) -> bool:
    """Return whether both tree candidates justify evaluating a stack."""
    return random_forest_passes and gradient_boosting_passes


def split_train_eval_only(
    x_raw: pd.DataFrame,
    y_log: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return only train/eval data so Round 3 cannot consume final holdout rows."""
    x_train_eval, _x_final_holdout, y_train_eval, _y_final_holdout = (
        baseline_round2.split_raw_holdout(
            x_raw=x_raw,
            y_log=y_log,
            test_size=test_size,
            random_state=random_state,
        )
    )
    return x_train_eval, y_train_eval


def build_stacking_regressor(
    best_gradient_boosting_params: dict[str, Any],
    ridge_alpha: float,
) -> StackingRegressor:
    """Build a leak-free linear/tree stack with internal out-of-fold CV."""
    gradient_boosting = _build_gradient_boosting_pipeline()
    gradient_boosting.set_params(**best_gradient_boosting_params)
    return StackingRegressor(
        estimators=[
            (
                "ridge",
                baseline_round2.build_baseline_pipeline(
                    "ridge", ridge_alpha=ridge_alpha
                ),
            ),
            ("random_forest", build_random_forest_pipeline()),
            ("gradient_boosting", gradient_boosting),
        ],
        final_estimator=Ridge(alpha=ridge_alpha),
        cv=KFold(
            n_splits=baseline_round2.CV_N_SPLITS,
            shuffle=True,
            random_state=DEFAULT_RANDOM_STATE,
        ),
        n_jobs=-1,
    )


def aggregate_random_forest_importance(fitted_pipeline: Pipeline) -> pd.DataFrame:
    """Aggregate encoded Random Forest importance back to 15 raw features."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError("Random Forest pipeline must contain a ColumnTransformer")
    if not isinstance(model, RandomForestRegressor):
        raise TypeError("Random Forest pipeline must contain RandomForestRegressor")

    encoded_importance = model.feature_importances_
    raw_importance = {feature: 0.0 for feature in baseline_round2.FEATURE_COLUMNS}

    for transformer_name, raw_features in (
        ("num", baseline_round2.NUMERIC_FEATURES),
        ("ord", baseline_round2.ORDINAL_FEATURES),
    ):
        output_slice = preprocessor.output_indices_[transformer_name]
        values = encoded_importance[output_slice]
        if len(values) != len(raw_features):
            raise RuntimeError(
                f"Unexpected encoded width for {transformer_name}: {len(values)}"
            )
        for feature, importance in zip(raw_features, values, strict=True):
            raw_importance[feature] = float(importance)

    for transformer_name, raw_features in (
        ("cat_none", baseline_round2.CATEGORICAL_NONE_FEATURES),
        ("cat_missing", baseline_round2.CATEGORICAL_MISSING_FEATURES),
    ):
        if len(raw_features) != 1:
            raise RuntimeError(
                "Round 3 importance aggregation expects one feature in "
                f"{transformer_name}"
            )
        output_slice = preprocessor.output_indices_[transformer_name]
        raw_importance[raw_features[0]] = float(encoded_importance[output_slice].sum())

    importance = pd.DataFrame(
        {
            "feature": list(raw_importance),
            "importance": list(raw_importance.values()),
        }
    ).sort_values("importance", ascending=False, kind="stable")
    importance = importance.reset_index(drop=True)
    importance["rank"] = range(1, len(importance) + 1)
    return importance


def choose_selected_model(metrics: pd.DataFrame) -> str:
    """Choose the lowest-RMSE passing candidate or fall back to Ridge."""
    passing = metrics.loc[metrics["passes_threshold"].astype(bool)]
    if passing.empty:
        return "ridge"
    best_index = passing["rmse_log_mean"].astype(float).idxmin()
    return str(passing.loc[best_index, "model_name"])


def _cross_validate_rmse_log(
    estimator: BaseEstimator,
    x_train_eval: pd.DataFrame,
    y_train_eval: pd.Series,
    cv: KFold,
) -> list[float]:
    """Return positive log-RMSE scores from the shared outer CV."""
    negative_scores = cross_val_score(
        estimator,
        x_train_eval,
        y_train_eval,
        cv=cv,
        scoring=RMSE_SCORING,
        n_jobs=-1,
    )
    return [float(-score) for score in negative_scores]


def evaluate_stacking_if_eligible(
    random_forest_passes: bool,
    gradient_boosting_passes: bool,
    best_gradient_boosting_params: dict[str, Any],
    ridge_alpha: float,
    x_train_eval: pd.DataFrame,
    y_train_eval: pd.Series,
    cv: KFold,
) -> list[float] | None:
    """Evaluate stacking on shared outer folds only when both trees qualify."""
    if not should_evaluate_stacking(
        random_forest_passes=random_forest_passes,
        gradient_boosting_passes=gradient_boosting_passes,
    ):
        return None
    stacking = build_stacking_regressor(
        best_gradient_boosting_params=best_gradient_boosting_params,
        ridge_alpha=ridge_alpha,
    )
    return _cross_validate_rmse_log(
        stacking,
        x_train_eval,
        y_train_eval,
        cv,
    )


def _best_gradient_boosting_fold_scores(search: GridSearchCV) -> list[float]:
    """Extract the five positive RMSE scores for the selected grid candidate."""
    return [
        float(-search.cv_results_[f"split{fold}_test_score"][search.best_index_])
        for fold in range(search.n_splits_)
    ]


def _gradient_boosting_search_frame(search: GridSearchCV) -> pd.DataFrame:
    """Convert GridSearchCV results into a compact Round 3 artifact."""
    rows: list[dict[str, float | int]] = []
    for index, params in enumerate(search.cv_results_["params"]):
        rows.append(
            {
                "n_estimators": int(params["model__n_estimators"]),
                "learning_rate": float(params["model__learning_rate"]),
                "max_depth": int(params["model__max_depth"]),
                "rmse_log_mean": float(-search.cv_results_["mean_test_score"][index]),
                "rmse_log_std": float(search.cv_results_["std_test_score"][index]),
                "rank": int(search.cv_results_["rank_test_score"][index]),
            }
        )
    return pd.DataFrame(rows).sort_values("rank", kind="stable").reset_index(drop=True)


def _metrics_row(
    model_name: str,
    scores: Iterable[float],
    params: dict[str, Any],
    ridge_mean: float,
    ridge_std: float,
    train_eval_rows: int,
    test_size: float,
    random_state: int,
    timestamp: str,
) -> dict[str, Any]:
    """Build one model-comparison row with the Ridge selection rule."""
    score_array = np.asarray(list(scores), dtype=np.float64)
    rmse_mean = float(score_array.mean())
    rmse_std = float(score_array.std())
    improvement = ridge_mean - rmse_mean
    passes = model_name != "ridge" and passes_selection_threshold(
        improvement=improvement,
        threshold=ridge_std,
    )
    return {
        "round": ROUND_NAME,
        "model_name": model_name,
        "n_features": len(baseline_round2.FEATURE_COLUMNS),
        "target": baseline_round2.TARGET_LOG_COLUMN,
        "split_type": baseline_round2.SPLIT_TYPE,
        "test_size": test_size,
        "random_state": random_state,
        "train_eval_rows": train_eval_rows,
        "cv_n_splits": baseline_round2.CV_N_SPLITS,
        "cv_shuffle": True,
        "cv_random_state": DEFAULT_RANDOM_STATE,
        "params_json": json.dumps(params, sort_keys=True),
        "timestamp": timestamp,
        "rmse_log_mean": rmse_mean,
        "rmse_log_std": rmse_std,
        "improvement_vs_ridge": improvement,
        "selection_threshold": ridge_std,
        "passes_threshold": passes,
    }


def _fold_metric_rows(
    model_name: str,
    scores: Iterable[float],
    timestamp: str,
) -> list[dict[str, float | int | str]]:
    """Build one traceable positive log-RMSE row per outer fold."""
    return [
        {
            "round": ROUND_NAME,
            "model_name": model_name,
            "fold": fold,
            "target": baseline_round2.TARGET_LOG_COLUMN,
            "rmse_log": float(score),
            "timestamp": timestamp,
        }
        for fold, score in enumerate(scores, start=1)
    ]


def _write_artifacts(
    artifacts_dir: Path,
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    search_results: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    """Write CV-only Round 3 results and the methodology caveat."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(artifacts_dir / "metrics.csv", index=False)
    fold_metrics.to_csv(artifacts_dir / "fold_metrics.csv", index=False)
    search_results.to_csv(artifacts_dir / "gradient_boosting_search.csv", index=False)
    feature_importance.to_csv(
        artifacts_dir / "random_forest_feature_importance.csv", index=False
    )
    (artifacts_dir / "README.md").write_text(
        f"# Round 3 methodology note\n\n{METHODOLOGY_NOTE}\n",
        encoding="utf-8",
    )


def run_ensemble_round3(
    data_path: Path = DEFAULT_DATA_PATH,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    test_size: float = baseline_round2.DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    ridge_alpha: float = baseline_round2.DEFAULT_RIDGE_ALPHA,
) -> pd.DataFrame:
    """Run Round 3 CV experiments without fitting or scoring the final holdout."""
    preflight_preprocessor = baseline_round2.build_preprocessor()
    validate_one_hot_unknown_handling(preflight_preprocessor)

    x_raw, y_log = baseline_round2.load_raw_training_data(Path(data_path))
    x_train_eval, y_train_eval = split_train_eval_only(
        x_raw=x_raw,
        y_log=y_log,
        test_size=test_size,
        random_state=random_state,
    )

    cv = baseline_round2.make_cv()
    ridge_pipeline = baseline_round2.build_baseline_pipeline(
        "ridge", ridge_alpha=ridge_alpha
    )
    random_forest_pipeline = build_random_forest_pipeline()
    ridge_scores = _cross_validate_rmse_log(
        ridge_pipeline, x_train_eval, y_train_eval, cv
    )
    random_forest_scores = _cross_validate_rmse_log(
        random_forest_pipeline, x_train_eval, y_train_eval, cv
    )

    gradient_boosting_search = build_gradient_boosting_search(cv)
    gradient_boosting_search.fit(x_train_eval, y_train_eval)
    gradient_boosting_scores = _best_gradient_boosting_fold_scores(
        gradient_boosting_search
    )

    random_forest_pipeline.fit(x_train_eval, y_train_eval)
    feature_importance = aggregate_random_forest_importance(random_forest_pipeline)

    timestamp = datetime.now().isoformat(timespec="seconds")
    ridge_mean = float(np.mean(ridge_scores))
    ridge_std = float(np.std(ridge_scores))
    scored_models: list[tuple[str, list[float], dict[str, Any]]] = [
        ("ridge", ridge_scores, {"alpha": ridge_alpha}),
        (
            "random_forest",
            random_forest_scores,
            {
                "n_estimators": RANDOM_FOREST_N_ESTIMATORS,
                "random_state": DEFAULT_RANDOM_STATE,
            },
        ),
        (
            "gradient_boosting",
            gradient_boosting_scores,
            dict(gradient_boosting_search.best_params_),
        ),
    ]

    initial_metrics = pd.DataFrame(
        [
            _metrics_row(
                model_name=model_name,
                scores=scores,
                params=params,
                ridge_mean=ridge_mean,
                ridge_std=ridge_std,
                train_eval_rows=len(x_train_eval),
                test_size=test_size,
                random_state=random_state,
                timestamp=timestamp,
            )
            for model_name, scores, params in scored_models
        ]
    )
    random_forest_passes = bool(
        initial_metrics.loc[
            initial_metrics["model_name"] == "random_forest", "passes_threshold"
        ].item()
    )
    gradient_boosting_passes = bool(
        initial_metrics.loc[
            initial_metrics["model_name"] == "gradient_boosting",
            "passes_threshold",
        ].item()
    )
    stacking_scores = evaluate_stacking_if_eligible(
        random_forest_passes=random_forest_passes,
        gradient_boosting_passes=gradient_boosting_passes,
        best_gradient_boosting_params=dict(gradient_boosting_search.best_params_),
        ridge_alpha=ridge_alpha,
        x_train_eval=x_train_eval,
        y_train_eval=y_train_eval,
        cv=cv,
    )
    if stacking_scores is not None:
        scored_models.append(
            (
                "stacking",
                stacking_scores,
                {
                    "base_estimators": [
                        "ridge",
                        "random_forest",
                        "gradient_boosting",
                    ],
                    "final_estimator": "ridge",
                },
            )
        )

    metrics = pd.DataFrame(
        [
            _metrics_row(
                model_name=model_name,
                scores=scores,
                params=params,
                ridge_mean=ridge_mean,
                ridge_std=ridge_std,
                train_eval_rows=len(x_train_eval),
                test_size=test_size,
                random_state=random_state,
                timestamp=timestamp,
            )
            for model_name, scores, params in scored_models
        ]
    )
    selected_model = choose_selected_model(metrics)
    metrics["selected"] = metrics["model_name"] == selected_model
    fold_metrics = pd.DataFrame(
        [
            row
            for model_name, scores, _params in scored_models
            for row in _fold_metric_rows(model_name, scores, timestamp)
        ]
    )
    search_results = _gradient_boosting_search_frame(gradient_boosting_search)
    _write_artifacts(
        artifacts_dir=Path(artifacts_dir),
        metrics=metrics,
        fold_metrics=fold_metrics,
        search_results=search_results,
        feature_importance=feature_importance,
    )
    return metrics


if __name__ == "__main__":
    print(run_ensemble_round3().to_string(index=False))
