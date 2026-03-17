"""add physical_score to predictions_log

Revision ID: d4f1e8c2a017
Revises: c8e2f1a3b906
Create Date: 2026-03-16 00:00:00.000000

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "d4f1e8c2a017"
down_revision = "c8e2f1a3b906"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "predictions_log",
        sa.Column("physical_score", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("predictions_log", "physical_score")
