"""Demonstrate SQLAlchemy 2.0 select() and relationship-based loading."""

from pathlib import Path
import os
import sys

from sqlalchemy import select
from sqlalchemy.orm import Session

ORM_MIGRATIONS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ORM_MIGRATIONS_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(ORM_MIGRATIONS_ROOT)

from orm_migrations.src.database import SessionLocal  # noqa: E402
from orm_migrations.src.models import PredictionHistory, User  # noqa: E402


def query_via_relationship(session: Session, email: str) -> list[PredictionHistory]:
    """Load a user with select(), then lazy-load user.predictions."""

    stmt = select(User).where(User.email == email)
    user = session.execute(stmt).scalar_one()
    return user.predictions


def query_via_select(session: Session, email: str) -> list[PredictionHistory]:
    """Load prediction rows directly with select() and an explicit join."""

    stmt = (
        select(PredictionHistory)
        .join(User, PredictionHistory.user_id == User.id)
        .where(User.email == email)
    )
    return list(session.execute(stmt).scalars().all())


def verify_bidirectional_relationship(session: Session, email: str) -> None:
    """Assert that User.predictions and PredictionHistory.user agree."""

    user = session.execute(select(User).where(User.email == email)).scalar_one()
    for prediction in user.predictions:
        assert prediction.user.id == user.id
        assert prediction.user.email == user.email

    print(f"[OK] Bidirectional relationship works for {email}.")


def main() -> None:
    """Print both query styles and verify they return the same rows."""

    email = "alice@example.com"
    session = SessionLocal()
    try:
        print("\n--- Via relationship: user.predictions ---")
        relationship_rows = query_via_relationship(session, email)
        for prediction in relationship_rows:
            print(f"  {prediction}")

        print("\n--- Via select(PredictionHistory).join(User) ---")
        select_rows = query_via_select(session, email)
        for prediction in select_rows:
            print(f"  {prediction}")

        assert {row.id for row in relationship_rows} == {row.id for row in select_rows}
        print("\n[OK] Both query styles returned the same prediction IDs.")
        verify_bidirectional_relationship(session, email)
    finally:
        session.close()


if __name__ == "__main__":
    main()
