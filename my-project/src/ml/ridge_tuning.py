"""Disciplined Ridge alpha tuning on the Round 2 train/eval partition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ml import baseline_round2
from sklearn.model_selection import GridSearchCV, KFold

BASELINE_ALPHA = 1.0
DEFAULT_DATA_PATH = baseline_round2.DEFAULT_DATA_PATH
DEFAULT_ARTIFACTS_DIR = baseline_round2.PROJECT_DIR / "artifacts" / "ridge_tuning"
RIDGE_ALPHA_GRID: list[float] = [
    0.001,
    0.01,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
]
RMSE_SCORING = "neg_root_mean_squared_error"
METHODOLOGY_NOTE = (
    "The best-of-9 Ridge candidates share the same five CV folds, so the raw "
    "minimum may be mildly optimistic. A callable refit keeps alpha=1.0 unless "
    "the raw best improves by more than the alpha=1.0 CV standard deviation. "
    "The final holdout stays untouched until step 13."
)


@dataclass(frozen=True)
class RidgeTuningResult:
    """In-memory search and traceable selected Ridge metrics."""

    search: GridSearchCV
    candidates: pd.DataFrame
    selected_alpha: float
    selected_cv_rmse_mean: float
    selected_cv_rmse_std: float


def select_refit_index(cv_results: dict[str, Any]) -> int:
    """Select raw best only when it beats alpha=1 by more than baseline std."""
    params = cv_results["params"]
    baseline_indices = [
        index
        for index, candidate in enumerate(params)
        if float(candidate["model__alpha"]) == BASELINE_ALPHA
    ]
    if len(baseline_indices) != 1:
        raise ValueError("Ridge alpha grid must contain alpha=1.0 exactly once")

    baseline_index = baseline_indices[0]
    mean_test_scores = np.asarray(cv_results["mean_test_score"], dtype=np.float64)
    std_test_scores = np.asarray(cv_results["std_test_score"], dtype=np.float64)
    raw_best_index = int(np.argmax(mean_test_scores))
    baseline_mean = float(-mean_test_scores[baseline_index])
    raw_best_mean = float(-mean_test_scores[raw_best_index])
    baseline_std = float(std_test_scores[baseline_index])
    improvement = baseline_mean - raw_best_mean
    if improvement > baseline_std:
        return raw_best_index
    return baseline_index


def build_ridge_alpha_search(cv: KFold) -> GridSearchCV:
    """Build the nine-candidate Ridge search using shared CV and train scores."""
    return GridSearchCV(
        estimator=baseline_round2.build_baseline_pipeline(
            "ridge", ridge_alpha=BASELINE_ALPHA
        ),
        param_grid={"model__alpha": RIDGE_ALPHA_GRID},
        scoring=RMSE_SCORING,
        cv=cv,
        n_jobs=-1,
        refit=select_refit_index,
        return_train_score=True,
    )


def _ridge_alpha_search_frame(search: GridSearchCV) -> pd.DataFrame:
    """Convert Ridge GridSearchCV output to positive train/CV RMSE diagnostics."""
    params = search.cv_results_["params"]
    mean_test_scores = np.asarray(
        search.cv_results_["mean_test_score"], dtype=np.float64
    )
    std_test_scores = np.asarray(search.cv_results_["std_test_score"], dtype=np.float64)
    mean_train_scores = np.asarray(
        search.cv_results_["mean_train_score"], dtype=np.float64
    )
    std_train_scores = np.asarray(
        search.cv_results_["std_train_score"], dtype=np.float64
    )
    baseline_index = next(
        index
        for index, candidate in enumerate(params)
        if float(candidate["model__alpha"]) == BASELINE_ALPHA
    )
    raw_best_index = int(np.argmax(mean_test_scores))
    baseline_mean = float(-mean_test_scores[baseline_index])
    baseline_std = float(std_test_scores[baseline_index])

    rows: list[dict[str, float | int | bool]] = []
    for index, candidate in enumerate(params):
        mean_train_rmse = float(-mean_train_scores[index])
        mean_cv_rmse = float(-mean_test_scores[index])
        improvement = baseline_mean - mean_cv_rmse
        rows.append(
            {
                "alpha": float(candidate["model__alpha"]),
                "mean_train_rmse_log": mean_train_rmse,
                "std_train_rmse_log": float(std_train_scores[index]),
                "mean_cv_rmse_log": mean_cv_rmse,
                "std_cv_rmse_log": float(std_test_scores[index]),
                "generalization_gap": mean_cv_rmse - mean_train_rmse,
                "rank": int(search.cv_results_["rank_test_score"][index]),
                "improvement_vs_alpha_1": improvement,
                "selection_threshold": baseline_std,
                "is_raw_best": index == raw_best_index,
                "passes_threshold": improvement > baseline_std,
                "selected": index == search.best_index_,
            }
        )
    return pd.DataFrame(rows)


def _write_artifacts(artifacts_dir: Path, candidates: pd.DataFrame) -> None:
    """Write candidate diagnostics and the selection-bias caveat."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(artifacts_dir / "ridge_alpha_search.csv", index=False)
    (artifacts_dir / "README.md").write_text(
        f"# Ridge tuning methodology note\n\n{METHODOLOGY_NOTE}\n",
        encoding="utf-8",
    )


def run_ridge_tuning(
    data_path: Path = DEFAULT_DATA_PATH,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    test_size: float = baseline_round2.DEFAULT_TEST_SIZE,
    random_state: int = baseline_round2.DEFAULT_RANDOM_STATE,
) -> RidgeTuningResult:
    """Tune Ridge alpha on train/eval only and refit the policy-selected model."""
    x_raw, y_log = baseline_round2.load_raw_training_data(Path(data_path))
    x_train_eval, _x_holdout, y_train_eval, _y_holdout = (
        baseline_round2.split_raw_holdout(
            x_raw=x_raw,
            y_log=y_log,
            test_size=test_size,
            random_state=random_state,
        )
    )
    del _x_holdout, _y_holdout

    cv = baseline_round2.make_cv()
    search = build_ridge_alpha_search(cv)
    search.fit(x_train_eval, y_train_eval)
    candidates = _ridge_alpha_search_frame(search)
    selected_row = candidates.iloc[search.best_index_]
    _write_artifacts(Path(artifacts_dir), candidates)

    return RidgeTuningResult(
        search=search,
        candidates=candidates,
        selected_alpha=float(selected_row["alpha"]),
        selected_cv_rmse_mean=float(selected_row["mean_cv_rmse_log"]),
        selected_cv_rmse_std=float(selected_row["std_cv_rmse_log"]),
    )
