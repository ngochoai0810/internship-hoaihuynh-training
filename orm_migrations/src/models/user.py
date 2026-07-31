"""User ORM model."""

from typing import TYPE_CHECKING

from sqlalchemy.orm import Mapped, mapped_column, relationship

from orm_migrations.src.models.base import AuditMixin, Base

if TYPE_CHECKING:
    from orm_migrations.src.models.prediction_history import PredictionHistory


class User(AuditMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(unique=True, index=True)
    hashed_password: Mapped[str]

    predictions: Mapped[list["PredictionHistory"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"User(id={self.id!r}, email={self.email!r})"
