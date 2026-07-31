# Relational DB Design And Alembic Migrations

**Branch:** `feature/orm-migrations`

## Main Goal

Move the API from mock data to a real relational database. Use SQLAlchemy 2.0
ORM and Alembic migrations.

## Plan

| Topic | Main output |
| --- | --- |
| Basic SQLAlchemy ORM | `engine`, `SessionLocal`, `Base`, `House` model |
| OOP models and one-to-many links | `AuditMixin`, `User`, `PredictionHistory`, `relationship()` |
| First Alembic setup and migration | `alembic/`, migration for `user`, `house`, `prediction_history` |
| Real queries and unique constraints | `seed.py`, migration with `unique=True` |
| Connect API routes to the database | `House` routes with `Depends(get_db)`, schema notes |

## Key Topics

- SQLAlchemy 2.0 core ideas: `Engine`, `Connection`, `Session`.
- A `Session` is a unit of work for database changes.
- Use `DeclarativeBase` instead of old `declarative_base()`.
- Use typed ORM mapping with `Mapped[T]` and `mapped_column()`.
- Use mixins for shared audit fields.
- Use `ForeignKey`, `relationship()`, and `back_populates`.
- Use Alembic commands: `init`, `--autogenerate`, `upgrade`, `downgrade`.
- Use query 2.0 style: `select()` and `session.execute(...).scalars().all()`.
- Use FastAPI dependency pattern with `get_db()` and `yield`.

## Project Layout

```text
internship-hoaihuynh-training/
|-- orm_migrations/
|   |-- alembic/
|   |-- alembic.ini
|   |-- app.db
|   |-- docs/
|   |   |-- README.md
|   |   `-- orm_setup.md
|   |-- scripts/
|   |   `-- init_db.py
|   |-- src/
|   |   `-- models/
|   |       |-- base.py
|   |       `-- house.py
|   `-- tests/
|       `-- test_orm_setup.py
`-- Readme.md
```

`orm_migrations/docs/` has learning notes only. Real code is in
`orm_migrations/src/models/`.

The `House` model matches the API schema used by the house routes on branch
`feature/fastapi-core-crud`: `area`, `rooms`, `location`, and `price`.
