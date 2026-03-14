"""Side-by-side candidate comparison output."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import desc, select

from db.schema import Brief, Player, PredictionLog
from db.session import session_scope
from outputs.common import render_template


def generate_comparison(brief_id: int, player_ids: list[int]) -> str:
    """Render a multi-player comparison sheet to HTML."""

    context = _build_comparison_context(brief_id, player_ids)
    return render_template("comparison.html", **context)


def _build_comparison_context(brief_id: int, player_ids: list[int]) -> dict[str, Any]:
    with session_scope() as session:
        brief = session.get(Brief, brief_id)
        players = {
            player.player_id: player
            for player in session.scalars(select(Player).where(Player.player_id.in_(player_ids)))
        }
    cards = []
    for player_id in player_ids:
        prediction = _latest_prediction(brief_id, player_id)
        player = players.get(player_id)
        cards.append(
            {
                "player_id": player_id,
                "player_name": player.player_name if player else f"Player {player_id}",
                "team": player.current_team if player else None,
                "role_fit": prediction.role_fit_score if prediction else None,
                "current_performance": prediction.l1_performance_score if prediction else None,
                "projection": prediction.championship_projection_50th if prediction else None,
                "availability": prediction.availability_risk_prob if prediction else None,
                "financial": prediction.var_score if prediction else None,
                "composite": prediction.composite_score if prediction else None,
                "projection_band": {
                    "p10": prediction.championship_projection_10th if prediction else None,
                    "p50": prediction.championship_projection_50th if prediction else None,
                    "p90": prediction.championship_projection_90th if prediction else None,
                },
            }
        )
    return {"brief": brief, "generated_at": datetime.utcnow(), "players": cards}


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
