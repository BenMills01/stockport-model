"""Explainable role-fit scoring."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from sqlalchemy import desc, select

from db.schema import PlayerRole, RoleTemplate
from db.session import session_scope
from features.confidence import compute_confidence, shrink_low_sample_value
from features.league_adjust import compute_league_percentile


def score_role_fit(player_id: int, template_id: int, season: str) -> dict[str, Any]:
    """Score a player against a role template using role-relative percentiles."""

    with session_scope() as session:
        template = session.get(RoleTemplate, template_id)
        if template is None or not template.is_active:
            raise ValueError(f"Active role template {template_id} not found")

    metrics = template.metrics_json or {}
    league_adjust = compute_league_percentile(player_id=player_id, season=season, role=template.role_name)
    percentiles = league_adjust.get("percentiles", {})
    confidence = compute_confidence(player_id, season)

    contributions = {}
    raw_score = 0.0
    for metric_name, weight in metrics.items():
        lookup_metric = metric_name.removesuffix("_per90")
        player_percentile = float(percentiles.get(lookup_metric, 0.0) or 0.0)
        contribution = player_percentile * float(weight)
        raw_score += contribution
        contributions[metric_name] = {
            "player_percentile": player_percentile,
            "weight": float(weight),
            "contribution": contribution,
        }

    shrinkage_factor = float(confidence.get("shrinkage_factor", 1.0) or 1.0)
    score = shrink_low_sample_value(
        player_value=raw_score,
        league_role_average=50.0,
        shrinkage_factor=shrinkage_factor,
    )
    return {
        "score": float(max(0.0, min(100.0, score))),
        "raw_score": float(max(0.0, min(100.0, raw_score))),
        "shrinkage_factor": shrinkage_factor,
        "decomposition": contributions,
        "template_version": template.version,
        "confidence_tier": confidence["confidence_tier"],
    }


@lru_cache(maxsize=128)
def get_active_template_for_role(role_name: str) -> RoleTemplate | None:
    """Return the latest active role template for a role.

    The result is expunged from the session before the session closes so that
    session.commit() cannot expire its column attributes.  All columns are
    accessed inside the session to guarantee they are loaded before expunge.
    """

    with session_scope() as session:
        result = session.scalar(
            select(RoleTemplate)
            .where(
                RoleTemplate.role_name == role_name,
                RoleTemplate.is_active.is_(True),
            )
            .order_by(desc(RoleTemplate.created_date))
        )
        if result is None:
            return None
        # Touch every column used downstream to ensure they are in instance __dict__.
        _ = (
            result.template_id,
            result.role_name,
            result.version,
            result.is_active,
            result.metrics_json,
            result.created_date,
        )
        session.expunge(result)
        return result
