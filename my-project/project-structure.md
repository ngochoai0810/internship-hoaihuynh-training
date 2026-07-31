# Project Structure Evidence

Milestone 1 established the base structure for backend and ML integration. The
current `develop` branch already contains the skeleton needed for the next
milestone:

```text
my-project/
  src/
    api/       # FastAPI routers
    models/    # SQLAlchemy models
    schemas/   # Pydantic request/response schemas
    ml/        # ML exploration, preprocessing, and training code
  tests/       # Automated tests
```

## Why This Structure Matters

- `api` keeps HTTP routing separate from business logic.
- `models` keeps database persistence definitions isolated.
- `schemas` keeps validation and serialization contracts explicit.
- `ml` keeps data engineering and model code separate from the API layer.
- `tests` gives each layer a clear place for automated verification.

This layout lets Milestone 2 add FastAPI without mixing API code with the
preprocessing pipeline from Milestone 1.
