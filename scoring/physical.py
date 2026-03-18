"""Physical score (0–100) derived from SkillCorner metrics.

The score is a weighted composite of percentile-ranked physical and
game-intelligence metrics within a peer group (the candidate pool for the
current brief/role).  When no SkillCorner data exists for a player the
function returns ``None`` so callers can fall back gracefully.

Usage::

    physical_score = score_physical(player_id, peer_player_ids)
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from features.skillcorner import compute_skillcorner_features


# ---------------------------------------------------------------------------
# Metric weights: which features matter most and how much
# ---------------------------------------------------------------------------

# Physical output (sprint / high-speed running intensity)
_PHYSICAL_WEIGHTS: dict[str, float] = {
    "sc_physical_hsr_dist_per_match": 0.20,
    "sc_physical_sprint_dist_per_match": 0.20,
    "sc_physical_dist_per_match": 0.10,
    "sc_physical_count_sprint_per_match": 0.10,
    "sc_physical_count_hsr_per_match": 0.10,
    "sc_physical_count_high_accel_per_match": 0.08,
    "sc_physical_count_high_decel_per_match": 0.07,
    "sc_physical_top_speed_per_match": 0.15,
}

# Game-intelligence: pressure resilience and passing quality
# Off-ball run metrics (dangerous runs in behind, run threat, exploitation rate)
# are excluded as they are position-specific to forwards/runners and unfairly
# penalise defenders and midfielders for not making those movements.
_GI_WEIGHTS: dict[str, float] = {
    "sc_pressure_ball_retention_ratio_under_high_pressure": 0.25,
    "sc_pressure_pass_completion_ratio_under_high_pressure": 0.25,
    "sc_passes_count_completed_pass_to_run_in_behind_per_match": 0.25,
    "sc_passes_opportunity_take_rate": 0.25,
}

# Blend between the two sub-scores
_PHYSICAL_SUB_WEIGHT = 0.80
_GI_SUB_WEIGHT = 0.20

# Minimum number of peers with data before percentile ranking is meaningful
_MIN_PEERS_WITH_DATA = 3


def score_physical(
    player_id: int,
    peer_player_ids: list[int],
    *,
    physical_weights: Mapping[str, float] | None = None,
    gi_weights: Mapping[str, float] | None = None,
    physical_sub_weight: float | None = None,
    gi_sub_weight: float | None = None,
) -> float | None:
    """Return a 0–100 physical score for *player_id* relative to *peer_player_ids*.

    Returns ``None`` if the player has no SkillCorner data.

    The score is the weighted average of percentile ranks (0–100) across
    physical output metrics and game-intelligence metrics. Each sub-score
    is computed independently and blended with the configured sub-weights.

    ``physical_weights`` and ``gi_weights`` let callers override the default
    profile templates for dashboard-specific physical emphases.

    If fewer than ``_MIN_PEERS_WITH_DATA`` peers have data for a metric that
    metric is skipped for that sub-score.
    """
    # Collect features for the full peer pool (including target player).
    all_ids = list(dict.fromkeys([player_id, *peer_player_ids]))  # deduplicated, order-stable
    feature_rows: list[dict[str, Any]] = [
        {"player_id": pid, **compute_skillcorner_features(pid)} for pid in all_ids
    ]
    pool = pd.DataFrame(feature_rows).set_index("player_id")

    # Player must have SC data.
    if not pool.loc[player_id].get("sc_has_physical", False):
        return None

    active_physical_weights = dict(physical_weights or _PHYSICAL_WEIGHTS)
    active_gi_weights = dict(gi_weights or _GI_WEIGHTS)
    active_physical_sub_weight = float(
        _PHYSICAL_SUB_WEIGHT if physical_sub_weight is None else physical_sub_weight
    )
    active_gi_sub_weight = float(_GI_SUB_WEIGHT if gi_sub_weight is None else gi_sub_weight)

    physical_sub = _weighted_percentile_score(player_id, pool, active_physical_weights)
    gi_sub = _weighted_percentile_score(player_id, pool, active_gi_weights)

    if physical_sub is None and gi_sub is None:
        return None

    # Blend — if one sub-score is missing, use only the available one.
    if physical_sub is None:
        return round(gi_sub, 2)  # type: ignore[arg-type]
    if gi_sub is None:
        return round(physical_sub, 2)

    total_sub_weight = active_physical_sub_weight + active_gi_sub_weight
    if total_sub_weight <= 0.0:
        return round((physical_sub + gi_sub) / 2.0, 2)

    blended = (
        active_physical_sub_weight * physical_sub + active_gi_sub_weight * gi_sub
    ) / total_sub_weight
    return round(blended, 2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _weighted_percentile_score(
    player_id: int,
    pool: pd.DataFrame,
    weights: dict[str, float],
) -> float | None:
    """Compute the weighted-average percentile score (0–100) for *player_id*."""
    total_weight = 0.0
    weighted_sum = 0.0

    for metric, weight in weights.items():
        if metric not in pool.columns:
            continue
        col = pd.to_numeric(pool[metric], errors="coerce")
        valid = col.dropna()
        if len(valid) < _MIN_PEERS_WITH_DATA:
            continue
        player_val = col.get(player_id)
        if player_val is None or np.isnan(float(player_val)):
            continue

        # Percentile rank: fraction of peers strictly below, averaged with fraction
        # at or below (handles ties gracefully).
        below = float((valid < player_val).sum())
        at_or_below = float((valid <= player_val).sum())
        pct = 100.0 * (below + at_or_below) / 2.0 / len(valid)

        weighted_sum += weight * pct
        total_weight += weight

    if total_weight == 0.0:
        return None

    # Re-normalise in case some metrics were skipped.
    return weighted_sum / total_weight
