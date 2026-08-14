# Internship - Model Training

## Overview

This branch contains the machine learning training work for the Kaggle House
Prices workflow. It keeps the reusable preprocessing project and removes the
backend/authentication milestone folders so the branch can focus on notebook
training, model evaluation, and script conversion.

## Repository Structure

```text
internship-hoaihuynh-training/
|-- my-project/            # ML preprocessing, notebooks, tests, and training code
|-- .env.example           # Root example environment file
|-- .gitignore             # Ignored local data, caches, and generated artifacts
|-- pyproject.toml         # pytest, mypy, black, and ruff config
|-- requirements.txt       # ML, notebook, and test dependencies
`-- README.md              # Branch overview
```

## Week 7 Focus

- Reuse `my-project/src/ml/preprocessing.py` instead of copying preprocessing
  logic.
- Create training notebooks for train/test split, baseline linear regression,
  Ridge regression, and model evaluation.
- Convert notebook logic into a standalone `my-project/src/ml/train.py` script.
- Save the first fitted model artifact with `joblib`, for example under
  `my-project/models/model.pkl`.

## Useful Paths

- `my-project/notebooks/`: exploratory and training notebooks.
- `my-project/src/ml/preprocessing.py`: reusable preprocessing helpers.
- `my-project/src/run_pipeline.py`: preprocessing pipeline entrypoint.
- `my-project/tests/`: preprocessing tests and edge-case coverage.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Checks

```powershell
python -m pytest -q
python -m mypy my-project/src
python -m ruff check .
python -m black --check .
```

## Data And Artifacts

The Kaggle dataset is expected to be stored locally and is not committed to Git.
Generated model files and pipeline artifacts should stay out of Git.
