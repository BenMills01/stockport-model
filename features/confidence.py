"""Sample-size confidence tiers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from sqlalchemy import func, select

from db.schema import MatchPerformance
from db.session import session_scope

LOW_MINUTES_SAMPLE_THRESHOLD = 500.0
FULL_MINUTES_CONFIDENCE_THRESHOLD = 900.0
MINIMUM_MINUTES_EVIDENCE_MULTIPLIER = 0.35

# Bayesian shrinkage prior: equivalent to assuming each player has this many
# "phantom" appearances at the league-role average before real data is observed.
# Higher values = more aggressive pull toward the mean for low-sample players.
SHRINKAGE_PRIOR_APPEARANCES = 10


@lru_cache(maxsize=16384)
def compute_confidence(player_id: int, season: str) -> dict[str, Any]:
    """Compute appearance-based confidence and shrinkage."""

    with session_scope() as session:
        appearances, total_minutes = session.execute(
            select(
                func.count(),
                func.coalesce(func.sum(MatchPerformance.minutes), 0.0),
            )
            .select_from(MatchPerformance)
            .where(
                MatchPerformance.player_id == player_id,
                MatchPerformance.season == season,
                MatchPerformance.minutes > 0,
            )
        ).one()
    return _compute_confidence_from_sample(
        appearances=int(appearances or 0),
        total_minutes=float(total_minutes or 0.0),
    )


def shrink_low_sample_value(
    *,
    player_value: float,
    league_role_average: float,
    shrinkage_factor: float,
) -> float:
    """Apply the low-sample shrinkage rule from the spec."""

    return (player_value * shrinkage_factor) + (league_role_average * (1.0 - shrinkage_factor))


def _compute_confidence_from_appearances(appearances: int) -> dict[str, Any]:
    return _compute_confidence_from_sample(appearances=appearances, total_minutes=None)


def _compute_confidence_from_sample(
    *,
    appearances: int,
    total_minutes: float | None,
) -> dict[str, Any]:
    minutes = None if total_minutes is None else max(0.0, float(total_minutes))
    return {
        "appearances": appearances,
        "total_minutes": minutes,
        "confidence_tier": _tier_for_appearances(appearances),
        "shrinkage_factor": appearances / (appearances + SHRINKAGE_PRIOR_APPEARANCES) if appearances >= 0 else 0.0,
        "minutes_evidence_multiplier": minutes_evidence_multiplier(minutes),
        "below_minutes_threshold": bool(minutes is not None and minutes < LOW_MINUTES_SAMPLE_THRESHOLD),
    }


def _tier_for_appearances(appearances: int) -> str:
    if appearances < 10:
        return "Low"
    if appearances < 20:
        return "Medium"
    if appearances < 30:
        return "High"
    return "VeryHigh"


def minutes_evidence_multiplier(total_minutes: float | None) -> float:
    """Convert minutes played into an evidence multiplier for ranking."""

    if total_minutes is None:
        return 1.0
    bounded_minutes = max(0.0, float(total_minutes))
    if bounded_minutes <= 0.0:
        return MINIMUM_MINUTES_EVIDENCE_MULTIPLIER
    scaled = min(1.0, bounded_minutes / FULL_MINUTES_CONFIDENCE_THRESHOLD)
    return max(MINIMUM_MINUTES_EVIDENCE_MULTIPLIER, scaled ** 0.5)
