import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from orm_migrations.src.models import Base, User


def test_duplicate_user_email_raises_integrity_error() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    testing_session_local = sessionmaker(bind=engine)
    db = testing_session_local()

    try:
        db.add(User(email="duplicate@example.com", hashed_password="first"))
        db.commit()

        db.add(User(email="duplicate@example.com", hashed_password="second"))
        with pytest.raises(IntegrityError):
            db.commit()
    finally:
        db.rollback()
        db.close()
