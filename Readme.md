# Week 9 - ML Model Integration & Retrain Simulation

This branch consolidates the house-price ML pipeline, FastAPI backend,
database-backed JWT authentication, prediction history, and Streamlit client
inside `my-project`.

## Structure

```text
my-project/
|-- alembic/                    # Database migrations
|-- data/                       # Local/ignored training data
|-- src/
|   |-- api/                    # FastAPI app and routes
|   |-- core/                   # Settings and JWT/password security
|   |-- dependencies/           # FastAPI auth dependencies
|   |-- ml/                     # Preprocessing, training, runtime loading
|   |-- models/                 # SQLAlchemy entities
|   |-- schemas/                # API contracts
|   |-- scripts/                # Deterministic retrain subset helper
|   `-- streamlit_app/          # Streamlit UI and HTTP client
`-- tests/                      # Unit and integration tests
```

Generated CSV files, SQLite databases, experiment artifacts, `.env`, and
`my-project/model.pkl` are intentionally not committed.

## Setup

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env
alembic -c my-project/alembic.ini upgrade head
```

Replace `SECRET_KEY` in `.env` before using the app outside local practice.

## Train the Initial Model

The API deliberately fails during startup when `model.pkl` is missing,
corrupt, or does not expose `predict()`.

```powershell
python my-project/src/ml/train.py
```

The command trains the existing Ridge pipeline, appends metrics under
`my-project/artifacts/`, and atomically replaces `my-project/model.pkl`.

## Run FastAPI and Streamlit

Use separate terminals from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --app-dir my-project/src
```

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run my-project/src/streamlit_app/app.py
```

Register and log in at `http://localhost:8501`. Streamlit retains the returned
JWT in session state and sends it as `Authorization: Bearer <token>` when it
calls `POST /api/v1/predict`.

The dashboard includes overview, holdout comparison, paced load testing, and
authenticated prediction history.

## API Contract

Protected prediction request:

```json
{
  "overall_qual": 7,
  "gr_liv_area": 1710.0,
  "garage_cars": 2.0,
  "garage_area": 548.0,
  "total_bsmt_sf": 856.0,
  "first_flr_sf": 856.0,
  "full_bath": 2,
  "tot_rms_abv_grd": 8,
  "year_built": 2003,
  "year_remod_add": 2003,
  "neighborhood": "CollgCr",
  "garage_type": "Attchd",
  "exter_qual": "Gd",
  "kitchen_qual": "Gd",
  "bsmt_qual": "Gd"
}
```

Routes:

- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`
- `POST /api/v1/predict`
- `GET /api/v1/predictions/history`
- `GET /health`

Successful predictions are inverse-transformed with `expm1` and saved to
`prediction_history` with the input JSON, authenticated user, price, and the
SHA-256 fingerprint of the loaded artifact.

## Retrain and Restart Simulation

Record the current fingerprint from `GET http://localhost:8000/health`, then:

```powershell
python my-project/src/scripts/create_retrain_subset.py
python my-project/src/ml/train.py --data-path my-project/data/retrain/train_subset.csv
```

The reportable artifact is produced instead by the packaging step, which refits
the selected model on train+eval, saves one pipeline file, and scores the final
holdout exactly once:

```powershell
python -m ml.finalize
```

Stop Uvicorn with `Ctrl+C` and start it again with the same command. A new
`GET /health` response must report a different `model_sha256`, and predictions
must use the new model without route or Streamlit changes. Running Uvicorn
without `--reload` makes the manual restart behavior explicit.

## Verification

```powershell
python -m pytest -q
python -m ruff check .
python -m black --check .
python -m mypy my-project/src
```
