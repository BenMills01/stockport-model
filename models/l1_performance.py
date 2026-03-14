"""League One current-performance scoring."""

from __future__ import annotations

from typing import Any

import pandas as pd

from db.read_cache import load_player_match_frame
from features.confidence import compute_confidence, shrink_low_sample_value
from features.league_adjust import compute_league_percentile
from features.opposition import compute_opposition_splits
from features.per90 import _compute_per90_frame
from features.rolling import compute_rolling
from models.role_fit import get_active_template_for_role


def score_l1_performance(player_id: int, season: str, role: str) -> dict[str, Any]:
    """Score current League One performance for a player-role."""

    template = get_active_template_for_role(role)
    if template is None:
        raise ValueError(f"No active template found for role '{role}'")

    league_adjust = compute_league_percentile(player_id=player_id, season=season, role=role)
    percentiles = league_adjust.get("percentiles", {})
    opposition = compute_opposition_splits(player_id=player_id, season=season)

    match_frame = load_player_match_frame(player_id, season).copy()
    per90 = _compute_per90_frame(match_frame)
    rolling = compute_rolling(per90)

    metrics_with_weights = template.metrics_json  # {metric_name: weight}
    metric_names = list(metrics_with_weights.keys())
    metric_percentiles = [
        float(percentiles.get(metric.removesuffix("_per90"), 0.0) or 0.0)
        for metric in metric_names
    ]
    role_metric_score = float(sum(metric_percentiles) / len(metric_percentiles)) if metric_percentiles else 0.0

    # Compute consistency, trend, and vs-top-tier scores as a weighted average
    # across ALL role template metrics rather than proxying from the first metric.
    total_weight = sum(float(w) for w in metrics_with_weights.values()) or 1.0
    consistency_sum = 0.0
    slope_numerator = 0.0
    slope_denominator = 0.0
    tier1_sum = 0.0
    for metric_name, raw_weight in metrics_with_weights.items():
        metric_key = metric_name.removesuffix("_per90")
        w = float(raw_weight) / total_weight
        roll_info = rolling.get(metric_key, {})
        opp_info = opposition.get(metric_key) or {}
        consistency_sum += _consistency_to_score(roll_info.get("roll_10_cv")) * w
        slope = roll_info.get("trend_slope_10")
        if slope is not None:
            slope_numerator += slope * w
            slope_denominator += w
        tier1_sum += _tier1_percentile_score(
            tier1_value=opp_info.get("tier1"),
            baseline_value=opp_info.get("tier3"),
        ) * w

    consistency_score = consistency_sum
    weighted_slope: float | None = (slope_numerator / slope_denominator) if slope_denominator > 0.0 else None
    trend_score = _trend_to_score(weighted_slope)
    vs_tier1_percentile = tier1_sum

    score = (
        (0.50 * role_metric_score)
        + (0.20 * consistency_score)
        + (0.15 * trend_score)
        + (0.15 * vs_tier1_percentile)
    )
    confidence = compute_confidence(player_id, season)
    shrinkage_factor = float(confidence.get("shrinkage_factor", 1.0) or 1.0)
    shrunk_score = shrink_low_sample_value(
        player_value=score,
        league_role_average=50.0,
        shrinkage_factor=shrinkage_factor,
    )

    return {
        "score": float(max(0.0, min(100.0, shrunk_score))),
        "raw_score": float(max(0.0, min(100.0, score))),
        "shrinkage_factor": shrinkage_factor,
        "percentile_in_role": role_metric_score,
        "form_trend": _trend_label(weighted_slope),
        "consistency": consistency_score,
        "vs_tier1_percentile": vs_tier1_percentile,
    }


def _consistency_to_score(cv: float | None) -> float:
    if cv is None:
        return 50.0
    return float(max(0.0, min(100.0, 100.0 / (1.0 + cv))))


def _trend_to_score(trend_slope: float | None) -> float:
    if trend_slope is None:
        return 50.0
    return float(max(0.0, min(100.0, 50.0 + (trend_slope * 50.0))))


def _trend_label(trend_slope: float | None) -> str:
    if trend_slope is None or abs(trend_slope) < 0.02:
        return "stable"
    return "improving" if trend_slope > 0 else "declining"


def _tier1_percentile_score(*, tier1_value: float | None, baseline_value: float | None) -> float:
    if tier1_value is None:
        return 50.0
    if baseline_value is None or baseline_value == 0:
        return 100.0 if tier1_value > 0 else 50.0
    ratio = tier1_value / baseline_value
    return float(max(0.0, min(100.0, ratio * 50.0)))
