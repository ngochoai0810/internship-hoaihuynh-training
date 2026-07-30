"""add unique email constraint

Revision ID: 9c1a7f4a2d8b
Revises: 54b4c66d53f3
Create Date: 2026-07-30 10:30:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "9c1a7f4a2d8b"
down_revision: Union[str, Sequence[str], None] = "54b4c66d53f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Make users.email unique at the database layer."""

    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)


def downgrade() -> None:
    """Return users.email to a normal non-unique index."""

    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=False)
