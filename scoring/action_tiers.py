"""Composite score action-tier helpers."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any

from config import get_settings


@lru_cache(maxsize=1)
def load_action_tiers() -> list[dict[str, Any]]:
    """Load and normalise the composite action-tier config."""

    payload = get_settings().load_json("composite_action_tiers.json")
    tiers = []
    for row in payload:
        tiers.append(
            {
                "label": str(row["label"]),
                "min_score": float(row["min_score"]),
                "action": str(row["action"]),
                "summary": str(row["summary"]),
                "style": str(row.get("style") or "neutral"),
            }
        )
    tiers.sort(key=lambda item: item["min_score"], reverse=True)
    return tiers


def composite_to_board_score(composite_score: float | None) -> float:
    """Translate the compressed raw composite into a more intuitive board score."""

    raw_score = max(0.0, float(composite_score or 0.0))
    if raw_score == 0.0:
        return 0.0
    scale = float(_score_calibration()["board_score"]["saturation_scale"])
    board_score = 100.0 * (1.0 - math.exp(-(raw_score / scale)))
    return float(max(0.0, min(100.0, board_score)))


def board_score_equation() -> str:
    """Return a human-readable version of the display-score equation."""

    scale = float(_score_calibration()["board_score"]["saturation_scale"])
    return f"Board Score = 100 x (1 - exp(-Composite / {scale:.0f})), capped between 0 and 100."


@lru_cache(maxsize=1)
def load_board_action_tiers() -> list[dict[str, Any]]:
    """Return the action tiers translated into board-score thresholds."""

    return [
        {
            **tier,
            "board_min_score": composite_to_board_score(tier["min_score"]),
        }
        for tier in load_action_tiers()
    ]


def classify_composite_action(composite_score: float | None) -> dict[str, Any]:
    """Return the configured action tier for a composite score."""

    score = float(composite_score or 0.0)
    for tier in load_board_action_tiers():
        if score >= tier["min_score"]:
            return {**tier, "score": score}
    fallback = load_board_action_tiers()[-1]
    return {**fallback, "score": score}


def summarise_action_tiers(scores: list[float | None]) -> list[dict[str, Any]]:
    """Count how many scores fall into each configured tier."""

    counts = {tier["label"]: 0 for tier in load_board_action_tiers()}
    for score in scores:
        tier = classify_composite_action(score)
        counts[tier["label"]] += 1
    return [{**tier, "count": counts[tier["label"]]} for tier in load_board_action_tiers()]


@lru_cache(maxsize=1)
def _score_calibration() -> dict[str, Any]:
    return dict(get_settings().load_json("score_calibration.json"))
