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

- `GET /` returns a basic health message.
- `GET /api/v1/houses/` lists mock houses and supports `limit`, `min_price`, and `max_rooms`.
- `GET /api/v1/houses/{house_id}` returns one mock house by path parameter.
- `POST /api/v1/houses/` creates a mock house response.
- `PATCH /api/v1/houses/{house_id}` partially updates a mock house.
- `GET /api/v1/users/me` returns the current mock user.
- `POST /api/v1/users/` creates a mock user response.

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
