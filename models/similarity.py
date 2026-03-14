"""Similarity and alternatives engine."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import or_, select

from db.schema import Player, PlayerRole
from db.session import session_scope
from features.league_adjust import compute_league_percentile


def find_similar(
    player_id: int,
    role: str,
    n: int = 10,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Find similar players by cosine similarity on role-specific feature vectors."""

    filters = filters or {}
    target = compute_league_percentile(player_id=player_id, season=str(filters.get("season") or ""), role=role)
    target_vector = _vector_from_percentiles(target.get("percentiles", {}))
    if target_vector.size == 0:
        return []

    with session_scope() as session:
        role_rows = list(
            session.scalars(
                select(PlayerRole).where(
                    or_(PlayerRole.primary_role == role, PlayerRole.secondary_role == role)
                )
            )
        )
        players = {
            player.player_id: player
            for player in session.scalars(select(Player).where(Player.player_id.in_([row.player_id for row in role_rows])))
        }

    results = []
    for role_row in role_rows:
        if role_row.player_id == player_id:
            continue
        if not _passes_filters(players.get(role_row.player_id), filters):
            continue
        season = str(filters.get("season") or role_row.season)
        candidate = compute_league_percentile(player_id=role_row.player_id, season=season, role=role)
        candidate_vector = _vector_from_percentiles(candidate.get("percentiles", {}))
        if candidate_vector.size == 0 or candidate_vector.shape != target_vector.shape:
            continue
        similarity = float(cosine_similarity([target_vector], [candidate_vector])[0][0])
        delta = {
            metric: float(candidate.get("percentiles", {}).get(metric, 0.0) - target.get("percentiles", {}).get(metric, 0.0))
            for metric in target.get("percentiles", {})
        }
        results.append(
            {
                "player_id": role_row.player_id,
                "name": players.get(role_row.player_id).player_name if players.get(role_row.player_id) else None,
                "similarity_score": similarity,
                "per_metric_comparison": delta,
                "hard_gate_flags": [],
            }
        )

    return sorted(results, key=lambda item: item["similarity_score"], reverse=True)[:n]


def _vector_from_percentiles(percentiles: dict[str, Any]) -> np.ndarray:
    if not percentiles:
        return np.array([])
    ordered = [float(percentiles[key] or 0.0) for key in sorted(percentiles)]
    return np.array(ordered, dtype=float)


def _passes_filters(player: Player | None, filters: dict[str, Any]) -> bool:
    if player is None:
        return False
    league = filters.get("league")
    if league is not None and player.current_league_id != league:
        return False
    return True
