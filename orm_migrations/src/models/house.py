"""
First ORM model: House.

The columns match the API schema in use: area, rooms, location, and price.
"""

from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from orm_migrations.src.models.base import Base


class House(Base):
    __tablename__ = "houses"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    area: Mapped[float] = mapped_column(comment="House area")
    rooms: Mapped[int] = mapped_column(comment="Number of rooms")
    location: Mapped[str] = mapped_column(index=True)
    price: Mapped[float] = mapped_column(default=0.0)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    def __repr__(self) -> str:
        return (
            f"House(id={self.id!r}, location={self.location!r}, "
            f"area={self.area!r}, price={self.price!r})"
        )
