# SQLAlchemy ORM Setup

## Goal

Set up the first ORM layer with SQLAlchemy 2.0 style:

- Create an `engine` for local SQLite.
- Create `SessionLocal` to make database sessions.
- Create `Base` with `DeclarativeBase`.
- Create a `get_db()` dependency with `yield`.
- Create the `House` model with `Mapped[]` and `mapped_column()`.
- Add tests for table creation, columns, insert, and query.
- Add a script that creates the local SQLite file `app.db`.

## Key Ideas

### Engine, Connection, And Session

`Engine` manages database access. `Connection` is a lower-level database
connection. `Session` is the main ORM object. It groups changes before
`commit()`.

### Typed ORM Mapping

SQLAlchemy 2.0 uses `Mapped[T]` and `mapped_column()`. This makes the model
clear for SQLAlchemy and useful for type checkers.

### Local SQLite

The project uses the local SQLite file `app.db` for practice. This file is a
local artifact and should not be committed to Git.

Run this command from the repo root to create the database file inside
`orm_migrations/`:

```powershell
.\.venv\Scripts\python.exe .\orm_migrations\scripts\init_db.py
```

### `create_all()` And Alembic

`Base.metadata.create_all()` is useful for a small local setup and tests. The
main schema workflow later should use Alembic migrations.

### FastAPI Dependency

`get_db()` opens a `Session`, yields it to the route, and closes it in
`finally`. This pattern will be used when FastAPI routes read and write the
real database.

## `House` Model

The `House` model matches the main API schema in
`fastapi-house-api/app/schemas/house.py` on branch `feature/fastapi-core-crud`.
The house routes import and use that schema.

The model has these columns:

- `id`
- `area`
- `rooms`
- `location`
- `price`
- `created_at`

The `house_schema.py` file on the old branch is only a learning/demo schema.
It is not the source of truth for this ORM model.
