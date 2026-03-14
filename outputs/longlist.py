"""Ranked longlist output."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, select

from db.schema import Brief, Pipeline, Player, PredictionLog
from db.session import session_scope
from outputs.common import render_template
from scoring.action_tiers import board_score_equation, classify_composite_action, composite_to_board_score, load_board_action_tiers, summarise_action_tiers
from scoring.composite import projection_score_from_logged_p50


def generate_longlist_report(brief_id: int) -> str:
    """Render the ranked longlist report to HTML."""

    context = _build_longlist_context(brief_id)
    return render_template("longlist.html", **context)


def _build_longlist_context(brief_id: int) -> dict[str, Any]:
    with session_scope() as session:
        brief = session.get(Brief, brief_id)
        pipeline_rows = list(
            session.scalars(
                select(Pipeline).where(
                    Pipeline.brief_id == brief_id,
                    Pipeline.stage == "longlist",
                )
            )
        )
        players = {
            player.player_id: player
            for player in session.scalars(
                select(Player).where(Player.player_id.in_([row.player_id for row in pipeline_rows]))
            )
        }

    player_cards = []
    for row in pipeline_rows:
        latest_prediction = _latest_prediction(row.player_id, brief_id)
        if latest_prediction is None:
            continue
        player = players.get(row.player_id)
        player_cards.append(
            {
                "player_id": row.player_id,
                "player_name": player.player_name if player else f"Player {row.player_id}",
                "team": player.current_team if player else None,
                "composite_score": latest_prediction.composite_score,
                "board_score": composite_to_board_score(latest_prediction.composite_score),
                "action_tier": classify_composite_action(latest_prediction.composite_score),
                "confidence_tier": "Unknown",
                "archetype_primary": row.archetype_primary,
                "archetype_secondary": row.archetype_secondary,
                "projection_band": {
                    "p10": latest_prediction.championship_projection_10th,
                    "p50": latest_prediction.championship_projection_50th,
                    "p90": latest_prediction.championship_projection_90th,
                },
                "per_layer_scores": {
                    "role_fit": latest_prediction.role_fit_score,
                    "current_performance": latest_prediction.l1_performance_score,
                    "financial": latest_prediction.var_score,
                    "availability": latest_prediction.availability_risk_prob,
                },
                "model_warnings": list(getattr(latest_prediction, "model_warnings", None) or []),
                "top_strengths": _top_strengths(latest_prediction),
                "top_concerns": _top_concerns(latest_prediction),
                "pathway_flag": bool(brief.pathway_player_id if brief else False),
            }
        )
    player_cards.sort(key=lambda item: float(item["composite_score"] or 0.0), reverse=True)
    return {
        "brief": brief,
        "generated_at": datetime.now(UTC),
        "players": player_cards,
        "board_score_equation": board_score_equation(),
        "action_tiers": load_board_action_tiers(),
        "action_tier_summary": summarise_action_tiers([card.get("composite_score") for card in player_cards]),
    }


def _latest_prediction(player_id: int, brief_id: int) -> PredictionLog | None:
    with session_scope() as session:
        return session.scalar(
            select(PredictionLog)
            .where(
                PredictionLog.player_id == player_id,
                PredictionLog.brief_id == brief_id,
            )
            .order_by(desc(PredictionLog.prediction_date))
        )


def _top_strengths(prediction: PredictionLog) -> list[str]:
    strengths = []
    if prediction.role_fit_score and prediction.role_fit_score >= 70:
        strengths.append("Strong role fit")
    if prediction.l1_performance_score and prediction.l1_performance_score >= 70:
        strengths.append("Strong current performance")
    if prediction.var_score and prediction.var_score >= 1.0:
        strengths.append("Good value-adjusted return")
    return strengths[:3] or ["No standout strengths logged yet"]


def _top_concerns(prediction: PredictionLog) -> list[str]:
    concerns = []
    if prediction.availability_risk_prob and prediction.availability_risk_prob > 0.4:
        concerns.append("Elevated availability risk")
    if projection_score_from_logged_p50(prediction.championship_projection_50th) < 40:
        concerns.append("Projection below Championship threshold")
    if any(bool(value) for value in (getattr(prediction, "component_fallbacks", None) or {}).values()):
        concerns.append("Model fallback used in at least one layer")
    return concerns[:2] or ["No major concerns logged yet"]
