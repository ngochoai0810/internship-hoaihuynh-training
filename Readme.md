# Internship - Agility Vietnam

**Intern:** Hoai Huynh
**Role:** Backend & ML Integration
**Period:** June 2026 - September 2026
**Mentor:** Thong Nguyen

## Overview

This repository documents my internship at Agility Vietnam, including training
progress and project source code.

## Week 5 - SQLite, SQLAlchemy, And Alembic

The House API now uses real SQLite data through SQLAlchemy sessions instead of
mock dictionaries. The FastAPI routes receive a database session with
`Depends(get_db)`, and the schema is managed by Alembic migrations in
`orm_migrations/`.

### Run The Database Migration

Run Alembic from the `orm_migrations/` directory:

```bash
cd orm_migrations
python -m alembic upgrade head
```

This creates or upgrades `app.db` to the latest schema.

### Run The API

Install API dependencies, then start FastAPI from `fastapi-house-api/`:

```bash
pip install -r fastapi-house-api/requirements.txt
cd fastapi-house-api
uvicorn app.main:app --reload
```

Open `/docs`, create a house, list houses, restart the server, and list again to
confirm the row persists in SQLite.

## Database Schema

### `houses`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | Integer | Primary key, indexed |
| `area` | Float | House area, required |
| `rooms` | Integer | Number of rooms, required |
| `location` | String | Required, indexed |
| `price` | Float | Required, defaults to `0.0` in the API create route |
| `created_at` | DateTime | Server default timestamp |
| `updated_at` | DateTime | Server default timestamp, updates on ORM changes |

### `users`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | Integer | Primary key, indexed |
| `email` | String | Required, unique, indexed |
| `hashed_password` | String | Required |
| `created_at` | DateTime | Server default timestamp |
| `updated_at` | DateTime | Server default timestamp, updates on ORM changes |

### `prediction_history`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | Integer | Primary key, indexed |
| `user_id` | Integer | Foreign key to `users.id`, indexed |
| `input_area` | Float | Prediction input |
| `input_rooms` | Integer | Prediction input |
| `input_location` | String | Prediction input |
| `predicted_price` | Float | Prediction output |
| `created_at` | DateTime | Server default timestamp |
| `updated_at` | DateTime | Server default timestamp, updates on ORM changes |

### `alembic_version`

| Column | Type | Notes |
| --- | --- | --- |
| `version_num` | String | Current Alembic revision applied to the database |

## Entity Relationships

| Relationship | Meaning |
| --- | --- |
| `User 1--n PredictionHistory` | One user can own many prediction records. Deleting a user cascades to their prediction history in the ORM relationship. |
| `House` standalone | House rows support the CRUD/listing API and are not linked to users or prediction history yet. |
