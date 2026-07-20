# House Price API

Week 4 FastAPI foundation for the internship training project.

## Setup

```bash
pip install -r requirements.txt
copy ..\.env.example .env
```

## Run the API

```bash
uvicorn app.main:app --reload
```

Open the interactive API docs at:

```txt
http://127.0.0.1:8000/docs
```

## Run Tests

```bash
pytest
```

## Week 4 Scope

- Initialize a FastAPI application.
- Define Pydantic schemas for houses and users.
- Separate request and response schemas.
- Add mock routers for houses and users.
- Add TestClient coverage for house endpoints.

## FastAPI Notes

- FastAPI builds `/docs` from route decorators, type hints, and response models.
- `house_id: int` tells FastAPI to validate the path parameter as an integer.
- Calling `/api/v1/houses/abc` returns `422` because `abc` cannot be parsed as an integer.
- Pydantic handles validation and serialization for request and response data.
- Starlette provides the ASGI foundation that lets FastAPI run efficiently with async I/O.

Database models, authentication, and ML integration are intentionally left for later weeks.
