"""add current age to players

Revision ID: fa78f3a94f8d
Revises: bc4d7bf5b515
Create Date: 2026-03-14 20:15:41.973418

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fa78f3a94f8d'
down_revision: Union[str, Sequence[str], None] = 'bc4d7bf5b515'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("players", sa.Column("current_age_years", sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("players", "current_age_years")
