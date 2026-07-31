"""House ORM model."""

from sqlalchemy.orm import Mapped, mapped_column

from orm_migrations.src.models.base import AuditMixin, Base


class House(AuditMixin, Base):
    __tablename__ = "houses"

    area: Mapped[float] = mapped_column(comment="House area")
    rooms: Mapped[int] = mapped_column(comment="Number of rooms")
    location: Mapped[str] = mapped_column(index=True)
    price: Mapped[float] = mapped_column(default=0.0)

    def __repr__(self) -> str:
        return (
            f"House(id={self.id!r}, location={self.location!r}, "
            f"area={self.area!r}, price={self.price!r})"
        )
