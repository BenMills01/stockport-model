"""Single-player shortlist card."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import desc, select

from db.schema import Brief, Pipeline, Player, PredictionLog, ScoutNote
from db.session import session_scope
from outputs.common import render_template


def generate_shortlist_card(brief_id: int, player_id: int) -> str:
    """Render a deep shortlist card to HTML."""

    context = _build_shortlist_context(brief_id, player_id)
    return render_template("shortlist_card.html", **context)


def _build_shortlist_context(brief_id: int, player_id: int) -> dict[str, Any]:
    with session_scope() as session:
        brief = session.get(Brief, brief_id)
        player = session.get(Player, player_id)
        prediction = session.scalar(
            select(PredictionLog)
            .where(
                PredictionLog.player_id == player_id,
                PredictionLog.brief_id == brief_id,
            )
            .order_by(desc(PredictionLog.prediction_date))
        )
        notes = list(
            session.scalars(
                select(ScoutNote).where(ScoutNote.player_id == player_id).order_by(desc(ScoutNote.date))
            )
        )
    return {
        "brief": brief,
        "player": player,
        "prediction": prediction,
        "generated_at": datetime.utcnow(),
        "scout_notes": notes,
        "projection_band": {
            "p10": prediction.championship_projection_10th if prediction else None,
            "p50": prediction.championship_projection_50th if prediction else None,
            "p90": prediction.championship_projection_90th if prediction else None,
        },
    }
