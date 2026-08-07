"""Prediction history ORM model."""

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from orm_migrations.src.models.base import AuditMixin, Base

if TYPE_CHECKING:
    from orm_migrations.src.models.user import User


class PredictionHistory(AuditMixin, Base):
    __tablename__ = "prediction_history"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    input_area: Mapped[float]
    input_rooms: Mapped[int]
    input_location: Mapped[str]
    predicted_price: Mapped[float]

    user: Mapped["User"] = relationship(back_populates="predictions")

    def __repr__(self) -> str:
        return (
            f"PredictionHistory(id={self.id!r}, user_id={self.user_id!r}, "
            f"predicted_price={self.predicted_price!r})"
        )
