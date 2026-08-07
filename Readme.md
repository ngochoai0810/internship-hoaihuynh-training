# Internship - Agility Vietnam

## Overview

- **Intern:** Hoai Huynh
- **Role:** Backend and ML Integration
- **Period:** June 2026 - September 2026
- **Mentor:** Thong Nguyen

This repository documents the Backend and ML Integration training work for the
Agility Vietnam internship. The training plan focuses on production-minded
backend and machine learning integration skills while avoiding heavy frontend
frameworks and complex DevOps.

The main technical stack is Python 3.11+, FastAPI, SQLAlchemy, Alembic,
Streamlit, PyJWT, Scikit-Learn, Joblib, SQLite, venv, and pip. The Kaggle House
Prices Competition dataset is used as the core data backbone for data
engineering and future ML integration work.

Current repository status:

- Milestone 1 is implemented through Python practice, typed exercises, project
  structure, EDA notebooks, reusable preprocessing code, and tests.
- Milestone 2 is implemented through FastAPI routes, Pydantic schemas,
  SQLAlchemy ORM models, Alembic migrations, JWT authentication, and a Streamlit
  auth UI.
- Milestone 3 

Repository structure:

```text
internship-hoaihuynh-training/
|-- prepare-python/        # Python basics and early practice logs
|-- python-type/           # List, dict, and type-hint exercises
|-- my-project/            # Milestone 1 ML preprocessing project
|-- fastapi-house-api/     # Milestone 2 FastAPI foundation work
|-- orm_migrations/        # Milestone 2 SQLAlchemy and Alembic work
|-- auth_flow/             # Milestone 2 JWT auth and Streamlit UI
|-- .env.example           # Root example settings for FastAPI House API
|-- pyproject.toml         # Root pytest, mypy, black, and ruff config
|-- requirements.txt       # Shared training dependencies
`-- README.md              # Repository overview
```

## Milestone 1: Git Workflow, Python Typing and Data Engineering

Training plan scope:

- Git workflow with `main`, `develop`, and `feature/*` branches.
- Conventional commit and pull request practice.
- Python type declarations for safer code and clearer architecture.
- Industrial project structure with separate `api`, `models`, `schemas`, and
  `ml` layers.
- Exploratory data analysis and feature processing for the Kaggle House Prices
  dataset.
- Extraction of notebook logic into reusable, typed Python modules.

Timeline mapping:

| Week | Training focus | Repository output |
| --- | --- | --- |
| Week 1 | Project setup and Git workflow | `prepare-python/`, project skeleton, work logs |
| Week 2-3 | EDA and feature processing | `python-type/`, `my-project/notebooks/`, `my-project/src/ml/preprocessing.py` |

Implemented folders:

- `prepare-python/`: early Python syntax, variable practice, exercise files, and
  work log.
- `python-type/`: list, dictionary, and typing exercises using concepts such as
  `List`, `Dict`, `Optional`, `Any`, and typed house records.
- `my-project/`: reusable ML preprocessing project for the Kaggle House Prices
  workflow.

Important outputs:

- EDA notebooks in `my-project/notebooks/house_prices/` for data overview,
  missing values, encoding, scaling, correlation analysis, transformations,
  outlier handling, and feature selection.
- `my-project/src/ml/preprocessing.py` for reusable preprocessing helpers.
- `my-project/src/run_pipeline.py` for running the preprocessing pipeline from
  raw local data.
- `my-project/tests/` for preprocessing behavior, edge cases, immutability,
  unseen categories, and serialization consistency.
- `my-project/milestone-1-summary.md` and `my-project/project-structure.md` for
  milestone documentation.

Milestone 1 setup:

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Milestone 1 testing and checks:

```bash
python -m pytest -q
python -m mypy my-project/src
python -m ruff check .
python -m black --check .
```

Notes:

- The root `pyproject.toml` points pytest at `my-project/tests` and sets
  `my-project/src` as the Python path.
- The Kaggle dataset is expected to be stored locally and is not committed to
  Git.
- Generated artifacts under `my-project/artifacts/` are ignored by Git.

## Milestone 2: Core FastAPI, Migrations and User Authentication

Training plan scope:

- FastAPI framework initialization, Uvicorn server running, and Swagger API docs.
- Pydantic schema segregation for request and response contracts.
- SQLAlchemy ORM modeling with shared inheritance for audit fields.
- Alembic database migrations for versioned schema changes.
- Password hashing with bcrypt and token-based authentication with JWT.
- Simple Streamlit UI integration with a decoupled FastAPI backend.

Timeline mapping:

| Week | Training focus | Repository output |
| --- | --- | --- |
| Week 4 | FastAPI foundations and Pydantic schemas | `fastapi-house-api/` |
| Week 5 | Relational DB design and Alembic migrations | `orm_migrations/` |
| Week 6 | User authentication and Streamlit UI integration | `auth_flow/` |

### Week 4: FastAPI Foundations and Pydantic Schemas

`fastapi-house-api/` is a FastAPI training API for house listings, user schemas,
safe response models, and a mock prediction endpoint.

Implemented outputs:

- FastAPI app in `fastapi-house-api/app/main.py`.
- Versioned routes under `/api/v1` plus unversioned practice aliases.
- Pydantic schemas for house, user, and prediction request/response models.
- Swagger documentation at `/docs`.
- CORS enabled for Streamlit at `http://localhost:8501`.

Key endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | API welcome message |
| GET | `/health` | Health check |
| GET | `/api/v1/houses` | List houses with `limit`, `min_price`, and `max_rooms` filters |
| GET | `/api/v1/houses/{house_id}` | Get one house by id |
| POST | `/api/v1/houses` | Create a house |
| PATCH | `/api/v1/houses/{house_id}` | Partially update a house |
| GET | `/api/v1/users` | List mock users |
| GET | `/api/v1/users/me` | Return the current mock user |
| GET | `/api/v1/users/{user_id}` | Get one mock user by id |
| POST | `/api/v1/users` | Create a mock user response |
| POST | `/api/v1/predict` | Return a mock house price prediction |

FastAPI House API setup and running:

```bash
copy .env.example .env
```

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = ".;fastapi-house-api"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

FastAPI House API tests:

```bash
cd fastapi-house-api
..\.venv\Scripts\python.exe -m pytest -q
```

### Week 5: Relational DB Design and Alembic Migrations

`orm_migrations/` contains the SQLAlchemy 2.0 and Alembic milestone work.

Implemented outputs:

- SQLite engine, `SessionLocal`, and `get_db` in
  `orm_migrations/src/database.py`.
- Shared `Base` and `AuditMixin` with inherited `id`, `created_at`, and
  `updated_at` fields.
- ORM models for `House`, `User`, and `PredictionHistory`.
- One-to-many `User` to `PredictionHistory` relationship with cascading delete.
- Alembic migrations for initial tables and unique user emails.
- Seed and query example scripts.
- Tests for table creation, relationships, cascade behavior, SQLite file
  creation, and unique constraints.

ORM setup and running:

```bash
python orm_migrations\scripts\init_db.py
python orm_migrations\scripts\seed.py
python orm_migrations\scripts\query_examples.py
```

Alembic configuration lives in `orm_migrations/alembic.ini`. Run Alembic commands
from inside `orm_migrations/` unless a command explicitly sets the config path.

ORM tests:

```bash
cd orm_migrations
..\.venv\Scripts\python.exe -m pytest -q
```

### Week 6: User Authentication and Streamlit UI Integration

`auth_flow/` contains the JWT authentication milestone and a minimal Streamlit
UI for testing backend communication outside Swagger.

Implemented outputs:

- `POST /register` for creating users with email and password.
- Password hashing with bcrypt through `pwdlib`.
- `POST /login` using OAuth2 form fields: `username` and `password`.
- JWT access token creation and validation with PyJWT.
- Protected `GET /me` route for the current user profile.
- Protected `PATCH /me` route for email and password updates.
- Public response schemas that exclude `password` and `hashed_password`.
- Streamlit UI for registration, login, token storage, `/me` calls, and logout.

Auth flow request pattern:

```text
Authorization: Bearer <token>
```

Auth flow environment setup:

```bash
copy auth_flow\.env.example auth_flow\.env
```

Set a real `SECRET_KEY` in `auth_flow/.env` before starting the auth API.

Auth Flow API running:

```bash
cd auth_flow
..\.venv\Scripts\python.exe -m uvicorn src.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Streamlit UI running:

```bash
cd auth_flow
..\.venv\Scripts\python.exe -m streamlit run streamlit_ui\app.py
```

Manual Streamlit test flow:

1. Register a new user.
2. Log in with the same email and password.
3. Click `Call /me`.
4. Confirm the response contains public user fields only.
5. Confirm missing or invalid tokens return `401 Unauthorized`.

Auth flow tests:

```bash
cd auth_flow
..\.venv\Scripts\python.exe -m pytest -q
```
