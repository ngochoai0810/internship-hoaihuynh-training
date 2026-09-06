"""SQLAlchemy engine, session factory, and request dependency helpers."""

from collections.abc import Generator

from fastapi import Request
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def create_database(
    database_url: str,
) -> tuple[Engine, sessionmaker[Session]]:
    """Create an engine and its configured session factory."""

    connect_args = (
        {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    )
    engine = create_engine(database_url, connect_args=connect_args)
    return engine, sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db(request: Request) -> Generator[Session, None, None]:
    """Yield a request-scoped database session."""

    with request.app.state.session_factory() as session:
        try:
            yield session
        finally:
            session.close()
