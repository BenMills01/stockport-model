"""Per-90 normalisation utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import select

from db.schema import MatchPerformance
from db.session import session_scope


PER90_VOLUME_METRICS = [
    "goals_scored",
    "assists",
    "shots_total",
    "shots_on_target",
    "passes_total",
    "passes_key",
    "tackles_total",
    "tackles_blocks",
    "tackles_interceptions",
    "duels_total",
    "duels_won",
    "dribbles_attempts",
    "dribbles_success",
    "dribbles_past",
    "fouls_committed",
    "fouls_drawn",
    "yellow_cards",
    "saves",
    "pen_won",
]


def compute_per90(player_id: int, season: str | None = None) -> pd.DataFrame:
    """Load match performances and compute per-90 metrics."""

    with session_scope() as session:
        statement = select(MatchPerformance).where(MatchPerformance.player_id == player_id)
        if season is not None:
            statement = statement.where(MatchPerformance.season == season)
        statement = statement.order_by(MatchPerformance.date.asc())
        rows = list(session.scalars(statement))

    frame = pd.DataFrame([_row_to_dict(row) for row in rows])
    return _compute_per90_frame(frame)


def _compute_per90_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    result = frame.copy()
    result = result[result["minutes"].fillna(0) >= 1].copy()
    result["low_minutes"] = result["minutes"].fillna(0) < 20

    for metric in PER90_VOLUME_METRICS:
        if metric not in result.columns:
            result[metric] = pd.NA
        result[f"{metric}_per90"] = (pd.to_numeric(result[metric], errors="coerce") / result["minutes"]) * 90.0

    return result.reset_index(drop=True)


def _row_to_dict(row: MatchPerformance) -> dict[str, Any]:
    return {
        column.name: getattr(row, column.name)
        for column in MatchPerformance.__table__.columns
    }
