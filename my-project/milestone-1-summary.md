# Milestone 1 Summary

## Problem

Milestone 1 needed to create a solid foundation for backend and ML integration:
use a professional Git workflow, write Python with explicit type hints, organize
the project into clear layers, and turn exploratory data work into reusable
preprocessing code.

## Approach

- Git workflow: separated the work into `feature/python-basics`,
  `feature/project-init`, and `feature/eda-house-prices`, with Conventional
  Commits and PRs into `develop`.
- Python typing: practiced `List`, `Dict`, `Optional`, `Any`, and `TypedDict`,
  then used mypy for static analysis.
- Project structure: prepared `src/api`, `src/models`, `src/schemas`, and
  `src/ml` so Milestone 2 can add FastAPI cleanly.
- Data engineering: analyzed the Kaggle House Prices dataset, chose a missing
  value strategy, engineered reusable features, and packaged the logic in
  `preprocessing.py`.

## Results

- EDA notebooks cover data overview, missing values, correlation, outliers,
  encoding, scaling, and feature selection.
- `ml.preprocessing` includes missing value handling, feature engineering,
  sklearn `ColumnTransformer`, train/test preprocessing, and joblib persistence.
- `my-project/src/run_pipeline.py` can run preprocessing from raw local data.
- Unit tests cover core logic, immutability, error handling, unseen categories,
  and serialization consistency.

## Lessons Learned

- Git flow makes the work easier to review and easier for a mentor to follow.
- Type hints and mypy catch mistakes before runtime.
- Moving notebook prototypes into `.py` modules makes the work testable,
  reusable, and easier to integrate with an API.
- Train and test preprocessing must separate `fit_transform` from `transform` to
  avoid data leakage.

## Beyond The Original Target

The original mentor target only required EDA notebooks and extracting the logic
into `preprocessing.py`. Milestone 1 goes beyond that target with PR workflow,
Conventional Commits, project structure, sklearn Pipeline, mypy strict checks,
unit tests, edge-case tests, a reusable runner, and documentation that can be
reused for the final presentation.
