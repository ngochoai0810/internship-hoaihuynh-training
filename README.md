# Internship - Agility Vietnam

**Intern:** Hoai Huynh  
**Role:** Backend & ML Integration  
**Period:** June 2026 - September 2026  
**Mentor:** Thong Nguyen

## Milestone 1: Git Workflow, Python Typing & Data Engineering

Milestone 1 builds the foundation for the following backend and ML integration
work: enterprise-style Git workflow, Python typing and static analysis,
industrial project structure, and reusable data preprocessing for the Kaggle
House Prices dataset.

### 1. Git Enterprise Basics

- Work with `main`, `develop`, and `feature/*` branches.
- Use Conventional Commits such as `feat(eda): ...`, `fix(tests): ...`, and
  `docs: ...`.
- Practice pull request handling through a sequence of focused PRs:
  - PR 1: `feature/python-basics` -> `develop`
  - PR 2: `feature/project-init` -> `develop`, or the existing project skeleton
    PR evidence if it has already been merged
  - PR 3: `feature/eda-house-prices` -> `develop`
- Learning references: W3C Git Tutorial and Learn Git Branching.

### 2. Python Typing

The `feature/python-basics` branch records Python data structure and typing
practice: `List`, `Dict`, `Optional`, `Any`, `TypedDict`, dynamic typing versus
explicit type hints, and mypy checks.

Main evidence:

- `feature/python-basics` commit `9647976`: practiced `Optional`, `Union`, and
  `Any`.
- `feature/python-basics` commit `c1efbc5`: practiced dictionaries,
  `TypedDict`, and mypy checks.

### 3. Industrial Project Structure

The project is organized into clear layers so Milestone 2 can add FastAPI on top
of the ML preprocessing work:

```text
my-project/
  src/
    api/       # FastAPI router layer
    models/    # SQLAlchemy DB models
    schemas/   # Pydantic schemas
    ml/        # Training and preprocessing scripts
  tests/
```

Main evidence:

- `develop` contains `my-project/src/api`, `models`, `schemas`, and `ml`.
- `feature/project-init` commit `35c020f`: added the House Prices exploration
  script.
- `develop` commit `d834cf6`: project skeleton merged through PR `#1`.

### 4. Data Extraction & Preparation

The `feature/eda-house-prices` branch packages the data engineering work:

- EDA notebooks under `my-project/notebooks/house_prices/`.
- Missing value strategy:
  - `NA` means `"None"` for columns such as `PoolQC`, `Alley`, and `Fence`.
  - `NA` means `0` for selected garage and basement numeric columns.
  - Median fill for `LotFrontage` and `MasVnrArea`.
  - Median/mode fallback for remaining columns.
- Feature engineering:
  - `TotalSF`
  - `HouseAge`
  - `RemodAge`
- Sklearn preprocessing:
  - `StandardScaler` for numeric features.
  - `OneHotEncoder(handle_unknown="ignore")` for nominal categorical features.
  - `OrdinalEncoder` for ordinal features.
  - `ColumnTransformer` to combine the preprocessing steps.

## How To Run

### 1. Prepare the environment

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install development tools if they are not already available:

```powershell
.\.venv\Scripts\python.exe -m pip install black ruff mypy pytest
```

### 2. Prepare the data

Place the Kaggle House Prices data locally. Do not commit it to Git:

```text
my-project/data/raw/train.csv
my-project/data/raw/test.csv
```

`.gitignore` excludes `data/`, `*.csv`, and `*.zip`.

### 3. Run preprocessing

```powershell
.\.venv\Scripts\python.exe my-project/src/run_pipeline.py
```

The preprocessing artifact is written to:

```text
my-project/artifacts/preprocessor.pkl
```

### 4. Run quality gates

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m mypy my-project/src my-project/tests
.\.venv\Scripts\python.exe -m black --check .
.\.venv\Scripts\python.exe -m ruff check .
```

## Beyond The Original Target

The original mentor target was to produce EDA notebooks and extract the logic
into `preprocessing.py`. Milestone 1 goes beyond that target with concrete
evidence:

- Multiple feature branches and PRs following Git flow.
- Conventional Commits.
- Python typing and mypy strict checks.
- Industrial project structure for FastAPI and ML integration.
- Sklearn `Pipeline` and `ColumnTransformer`.
- Unit tests and edge-case tests.
- A reusable runner at `my-project/src/run_pipeline.py`.
- README and a one-page summary that can be reused for the final presentation.
