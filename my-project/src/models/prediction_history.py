"""Persisted model prediction transaction."""

from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import AuditMixin, Base

if TYPE_CHECKING:
    from .user import User


class PredictionHistory(AuditMixin, Base):
    """Input, output, and artifact identity for one prediction."""

    __tablename__ = "prediction_history"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    predicted_price: Mapped[float] = mapped_column(Float)
    model_sha256: Mapped[str] = mapped_column(String(64))
    user: Mapped["User"] = relationship(back_populates="predictions")
