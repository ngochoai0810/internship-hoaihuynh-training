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

Database models, authentication, and ML integration are intentionally left for later weeks.
