"""Board-facing recommendation pack output."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import desc, select

from db.schema import Brief, Override, Player, PredictionLog, ScoutNote
from db.session import session_scope
from outputs.common import render_template


def generate_recommendation_pack(
    brief_id: int,
    recommended_player_id: int,
    alternatives: list[int],
) -> str:
    """Render the board-ready recommendation pack to HTML."""

    context = _build_recommendation_context(brief_id, recommended_player_id, alternatives)
    return render_template("recommendation.html", **context)


def _build_recommendation_context(
    brief_id: int,
    recommended_player_id: int,
    alternatives: list[int],
) -> dict[str, Any]:
    with session_scope() as session:
        brief = session.get(Brief, brief_id)
        player_ids = [recommended_player_id, *alternatives]
        players = {
            player.player_id: player
            for player in session.scalars(select(Player).where(Player.player_id.in_(player_ids)))
        }
        notes = list(
            session.scalars(select(ScoutNote).where(ScoutNote.player_id == recommended_player_id))
        )
        overrides = list(
            session.scalars(
                select(Override)
                .where(
                    Override.brief_id == brief_id,
                    Override.player_id == recommended_player_id,
                )
                .order_by(desc(Override.override_date))
            )
        )

    primary_prediction = _latest_prediction(brief_id, recommended_player_id)
    alternative_predictions = [
        {
            "player": players.get(player_id),
            "prediction": _latest_prediction(brief_id, player_id),
        }
        for player_id in alternatives
    ]
    return {
        "brief": brief,
        "player": players.get(recommended_player_id),
        "prediction": primary_prediction,
        "alternatives": alternative_predictions,
        "scout_notes": notes,
        "overrides": overrides,
        "generated_at": datetime.utcnow(),
    }


def _latest_prediction(brief_id: int, player_id: int) -> PredictionLog | None:
    with session_scope() as session:
        return session.scalar(
            select(PredictionLog)
            .where(
                PredictionLog.brief_id == brief_id,
                PredictionLog.player_id == player_id,
            )
            .order_by(desc(PredictionLog.prediction_date))
        )
