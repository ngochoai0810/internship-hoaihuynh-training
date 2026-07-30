# SQLAlchemy 2.0 Queries, Relationships, And Seed Data

## Goal

Practice common ORM tasks with SQLAlchemy 2.0 style:

- Query ORM models with `select()`.
- Load related data with `relationship()`.
- Protect `User.email` with a database unique constraint.
- Add the unique email change with Alembic.
- Seed sample data safely many times.

## Key Ideas

### SQLAlchemy 2.0 Query Style

Use `select()` for SQLAlchemy 2.0 queries. Then run the statement with
`session.execute()`.

```python
stmt = select(User).where(User.email == "alice@example.com")
user = session.execute(stmt).scalar_one()
```

Use `.scalars()` when the query returns ORM objects:

```python
stmt = select(PredictionHistory).where(PredictionHistory.user_id == user.id)
predictions = session.execute(stmt).scalars().all()
```

### Relationship Loading

`relationship()` creates a Python link between models. The real database link is
still the foreign key column.

```python
class User(AuditMixin, Base):
    predictions: Mapped[list["PredictionHistory"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class PredictionHistory(AuditMixin, Base):
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    user: Mapped["User"] = relationship(back_populates="predictions")
```

By default, SQLAlchemy uses lazy loading. This means `user.predictions` sends a
query only when the attribute is first accessed.

## Query Examples

Load predictions through the relationship:

```python
stmt = select(User).where(User.email == email)
user = session.execute(stmt).scalar_one()
predictions = user.predictions
```

Load predictions directly with `select()` and `join()`:

```python
stmt = (
    select(PredictionHistory)
    .join(User, PredictionHistory.user_id == User.id)
    .where(User.email == email)
)
predictions = session.execute(stmt).scalars().all()
```

Both styles should return the same prediction rows for the same user.

## Relationship Notes

- `User.predictions` points to many `PredictionHistory` rows.
- `PredictionHistory.user` points back to one `User`.
- `back_populates` must match on both sides.
- `ForeignKey("users.id")` tells SQLAlchemy how the tables are connected.
- `cascade="all, delete-orphan"` deletes a user's predictions when the user is
  deleted.

## Unique Constraint

`unique=True` makes the database reject duplicate emails.

```python
class User(AuditMixin, Base):
    email: Mapped[str] = mapped_column(unique=True, index=True)
```

Application checks are useful for friendly error messages, but they do not
replace the database constraint. The database raises `IntegrityError` when an
email is duplicated.

```python
with pytest.raises(IntegrityError):
    session.commit()
```

Alembic stores this schema change in a migration:

```python
def upgrade() -> None:
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
```

## Seed And Verification

The seed script uses a check-before-insert pattern. This makes it safe to run
many times on the same database.

```python
stmt = select(User).where(User.email == email)
user = session.execute(stmt).scalar_one_or_none()
if user is None:
    session.add(User(email=email, hashed_password=hashed_password))
```

Run these commands from `orm_migrations/`:

```bash
python -m alembic upgrade head
python -m scripts.seed
python -m scripts.seed
python -m scripts.query_examples
pytest
```

Expected results:

- The second seed run skips existing users and predictions.
- The query example prints the same rows for both query styles.
- The unique constraint test passes.
