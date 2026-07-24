# House Price API

A small FastAPI training API for house listings, user schemas, and safe response models.

## Quick Start

```bash
pip install -r requirements.txt
copy ..\.env.example .env
uvicorn app.main:app --reload
```

Interactive API docs:

```txt
http://127.0.0.1:8000/docs
```

## Tests

From this directory:

```bash
..\.venv\Scripts\python.exe -m pytest -q
```

## API Overview

Primary routes use the `/api/v1` prefix. Day 20 compatibility aliases are also exposed
without the version prefix for router and CRUD skeleton practice.

| Method | Primary Endpoint | Day 20 Alias | Purpose |
|--------|------------------|--------------|---------|
| GET | `/` | - | Basic API welcome message |
| GET | `/health` | - | Fast process health check |
| GET | `/api/v1/houses` | `/houses` | List mock houses with `limit`, `min_price`, and `max_rooms` filters |
| GET | `/api/v1/houses/{house_id}` | `/houses/{house_id}` | Get one mock house by id |
| POST | `/api/v1/houses` | `/houses` | Create a mock house response |
| PATCH | `/api/v1/houses/{house_id}` | `/houses/{house_id}` | Partially update a mock house |
| GET | `/api/v1/users` | `/users` | List mock users |
| GET | `/api/v1/users/me` | `/users/me` | Return the current mock user |
| GET | `/api/v1/users/{user_id}` | `/users/{user_id}` | Get one mock user by id |
| POST | `/api/v1/users` | `/users` | Create a mock user response |
| POST | `/api/v1/predict` | `/predict` | Return a mock house price prediction |

Routes are grouped by domain tags in `/docs`.

## CORS

The API allows browser requests from Streamlit at `http://localhost:8501` with
credentials, all methods, and all headers. This supports the future Streamlit frontend
without allowing every origin.

## Schema Pattern

The project separates schemas by use case:

- `Base` schemas hold shared fields.
- `Create` schemas describe request bodies for create endpoints.
- `Update` schemas describe partial PATCH payloads with optional fields.
- `Response` schemas describe the JSON returned to clients.

For PATCH, the router uses `model_dump(exclude_unset=True)` so fields omitted by the
client are not overwritten with `None`.

## Security Note

Do not use one "fat" user model for both input and output. `UserCreate` accepts
`password`, while `UserResponse` intentionally excludes both `password` and
`hashed_password`.

FastAPI's `response_model` validates and filters returned data before sending JSON to the
client. This keeps internal fields out of API responses even when a route builds a mock
object that contains `hashed_password`.

## Current Limitations

- Data is stored in mock in-memory objects.
- Authentication is mocked.
- There is no real database, password hashing library, or ML integration yet.
