"""Composite scoring engine."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any

from sqlalchemy import select

from config import get_settings
from db.schema import PathwayPlayer, PlayerRole, PredictionLog
from db.session import session_scope
from features.confidence import compute_confidence
from models.availability_risk import predict_availability_risk
from models.championship_projection import project_to_championship
from models.financial_value import estimate_value
from models.l1_performance import score_l1_performance
from models.role_fit import get_active_template_for_role, score_role_fit

_MODEL_VERSION = "0.1.0"


def compute_composite(player_id: int, brief: dict[str, Any], season: str) -> dict[str, Any]:
    """Compute the composite recruitment score for a player against a brief."""

    role_name = brief["role_name"]
    template = get_active_template_for_role(role_name)
    if template is None:
        raise ValueError(f"No active template found for role '{role_name}'")

    weights = _blended_weights(
        primary=str(brief["archetype_primary"]),
        secondary=brief.get("archetype_secondary"),
    )
    effective_weights = effective_layer_weights(weights)
    model_warnings: list[str] = []
    component_fallbacks: dict[str, bool] = {}
    role_fit = _call_component_with_fallback(
        "role_fit",
        lambda: score_role_fit(player_id, template.template_id, season),
        _fallback_role_fit,
        model_warnings,
        component_fallbacks,
    )
    tactical_fit = role_fit["score"]
    current_performance = _call_component_with_fallback(
        "current_performance",
        lambda: score_l1_performance(player_id, season, role_name),
        _fallback_current_performance,
        model_warnings,
        component_fallbacks,
    )
    projection = _call_component_with_fallback(
        "projection",
        lambda: project_to_championship(player_id, season, brief=brief),
        _fallback_projection,
        model_warnings,
        component_fallbacks,
        note_keys=("confidence_note",),
    )
    projection_score = _projection_score_from_bundle(projection)
    availability = _call_component_with_fallback(
        "availability",
        lambda: predict_availability_risk(player_id),
        _fallback_availability,
        model_warnings,
        component_fallbacks,
        note_keys=("caveat",),
    )
    availability_multiplier = availability["probability_available_75pct"]
    confidence = _call_component_with_fallback(
        "confidence",
        lambda: compute_confidence(player_id, season),
        _fallback_confidence,
        model_warnings,
        component_fallbacks,
        note_keys=("caveat",),
    )
    minutes_evidence_multiplier = float(confidence.get("minutes_evidence_multiplier", 1.0) or 1.0)
    financial = _call_component_with_fallback(
        "financial",
        lambda: estimate_value(
            player_id,
            {
                **brief,
                "quality_score": (
                    (role_fit["score"] * 0.5)
                    + (current_performance["score"] * 0.3)
                    + (projection_score * 0.2)
                ),
            },
        ),
        _fallback_financial,
        model_warnings,
        component_fallbacks,
        note_keys=("caveat",),
    )
    financial_score = _financial_score(financial["var_score"])

    current_score_for_sum = current_performance["score"]
    if (
        str(brief["archetype_primary"]).lower() == "emerging_asset"
        and projection_score >= 70.0
    ):
        current_score_for_sum = max(30.0, current_score_for_sum)

    raw_composite = (
        (role_fit["score"] * effective_weights["role_fit"])
        + (current_score_for_sum * effective_weights["current_performance"])
        + (projection_score * effective_weights["upward_projection"])
        + (financial_score * effective_weights["financial_value"])
    )
    tactical_fit_cap_applied = tactical_fit < 40.0
    if tactical_fit_cap_applied:
        raw_composite = min(raw_composite, 60.0)
    raw_composite *= minutes_evidence_multiplier
    composite_score = raw_composite * availability_multiplier

    pathway_comparison = _pathway_comparison(player_id=player_id, brief=brief, season=season, role_fit_score=role_fit["score"], projection_score=projection_score)
    total_minutes = confidence.get("total_minutes")
    minutes_sample_limited = bool(
        confidence.get(
            "below_minutes_threshold",
            total_minutes is not None and float(total_minutes) < 500.0,
        )
    )
    result = {
        "composite_score": composite_score,
        "per_layer_scores": {
            "tactical_fit": tactical_fit,
            "role_fit": role_fit["score"],
            "current_performance": current_performance["score"],
            "projection": projection_score,
            "financial": financial_score,
            "availability": availability_multiplier,
        },
        "per_layer_weights_used": effective_weights,
        "archetype_primary": brief["archetype_primary"],
        "archetype_secondary": brief.get("archetype_secondary"),
        "tactical_fit_cap_applied": tactical_fit_cap_applied,
        "confidence_tier": confidence["confidence_tier"],
        "sample_minutes": total_minutes,
        "minutes_evidence_multiplier": minutes_evidence_multiplier,
        "minutes_sample_limited": minutes_sample_limited,
        "projection_band": _flatten_projection_band(projection),
        "internal_pathway_comparison": pathway_comparison,
        "model_warnings": model_warnings,
        "component_fallbacks": component_fallbacks,
    }
    _log_prediction(player_id=player_id, brief=brief, role_fit=role_fit, current_performance=current_performance, projection=projection, availability=availability, financial=financial, composite=result)
    return result


def _blended_weights(primary: str, secondary: str | None) -> dict[str, float]:
    config = get_settings().load_json("archetype_weights.json")
    primary_weights = config[primary]["weights_pct"]
    if not secondary:
        return {key: value / 100.0 for key, value in primary_weights.items() if key != "availability"}
    secondary_weights = config[secondary]["weights_pct"]
    blended = {}
    for key in primary_weights:
        if key == "availability":
            continue
        blended[key] = ((0.70 * primary_weights[key]) + (0.30 * secondary_weights[key])) / 100.0
    return blended


def effective_layer_weights(weights: dict[str, float]) -> dict[str, float]:
    """Remove duplicate role-fit weighting while preserving the overall score scale."""

    adjusted = dict(weights)
    tactical_weight = float(adjusted.get("tactical_fit", 0.0) or 0.0)
    adjusted["tactical_fit"] = 0.0

    redistribution_keys = [
        "role_fit",
        "current_performance",
        "upward_projection",
        "financial_value",
    ]
    remaining_total = sum(float(adjusted.get(key, 0.0) or 0.0) for key in redistribution_keys)
    if tactical_weight <= 0.0 or remaining_total <= 0.0:
        return adjusted

    scale = (remaining_total + tactical_weight) / remaining_total
    for key in redistribution_keys:
        adjusted[key] = float(adjusted.get(key, 0.0) or 0.0) * scale
    return adjusted


def _call_component_with_fallback(
    component_name: str,
    loader: Any,
    fallback_factory: Any,
    warnings: list[str],
    component_fallbacks: dict[str, bool],
    *,
    note_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    try:
        payload = loader()
        component_fallbacks[component_name] = False
    except Exception as exc:
        warnings.append(
            f"{component_name.replace('_', ' ').title()} fell back to a neutral/default output after "
            f"{exc.__class__.__name__}: {exc}"
        )
        component_fallbacks[component_name] = True
        return fallback_factory(str(exc))

    for key in note_keys:
        note = payload.get(key)
        if note:
            warnings.append(f"{component_name.replace('_', ' ').title()}: {note}")
    return payload


def _fallback_role_fit(_error: str) -> dict[str, Any]:
    return {
        "score": 50.0,
        "raw_score": 50.0,
        "confidence_tier": "Unknown",
        "component_fallback": True,
    }


def _fallback_current_performance(_error: str) -> dict[str, Any]:
    return {
        "score": 50.0,
        "raw_score": 50.0,
        "component_fallback": True,
    }


def _fallback_projection(error: str) -> dict[str, Any]:
    return {
        "projected_performance": {},
        "projected_minutes_share": 0.5,
        "projected_adaptation_months": 5.0,
        "sample_size": 0,
        "confidence_note": f"Projection fallback used because live scoring failed ({error}).",
        "component_fallback": True,
    }


def _fallback_availability(error: str) -> dict[str, Any]:
    return {
        "probability_available_75pct": 0.75,
        "risk_tier": "Unknown",
        "contributing_factors": [],
        "caveat": f"Availability fallback used because live scoring failed ({error}).",
        "component_fallback": True,
    }


def _fallback_confidence(error: str) -> dict[str, Any]:
    return {
        "confidence_tier": "Unknown",
        "shrinkage_factor": 1.0,
        "minutes_evidence_multiplier": 1.0,
        "total_minutes": None,
        "below_minutes_threshold": False,
        "caveat": f"Confidence fallback used because live scoring failed ({error}).",
        "component_fallback": True,
    }


def _fallback_financial(error: str) -> dict[str, Any]:
    return {
        "fair_value_band": {"low": None, "mid": None, "high": None},
        "wage_fit": "unknown",
        "resale_band_2yr": {"low": None, "mid": None, "high": None},
        "replacement_cost": None,
        "var_score": 0.0,
        "comparable_transactions": [],
        "caveat": f"Financial fallback used because live scoring failed ({error}).",
        "component_fallback": True,
    }


def _projection_score_from_bundle(bundle: dict[str, Any]) -> float:
    performances = bundle.get("projected_performance", {})
    medians = [values["p50"] for values in performances.values()]
    if not medians:
        return 50.0
    avg = sum(medians) / len(medians)
    return _calibrated_projection_score(avg)


def projection_score_from_logged_p50(value: float | None) -> float:
    """Translate a stored flattened projection p50 into the 0-100 score scale."""

    if value is None:
        return 50.0
    return _calibrated_projection_score(float(value))


def _financial_score(var_score: float) -> float:
    return max(0.0, min(100.0, 50.0 + (50.0 * float(var_score))))


def _calibrated_projection_score(value: float) -> float:
    calibration = _score_calibration()["projection_score"]
    anchors = calibration["anchors"]
    score_points = calibration["score_at_anchors"]
    p10 = float(anchors["p10"])
    p50 = float(anchors["p50"])
    p90 = float(anchors["p90"])
    score10 = float(score_points["p10"])
    score50 = float(score_points["p50"])
    score90 = float(score_points["p90"])
    raw = max(0.0, float(value))

    if raw <= 0.0:
        return 0.0
    if raw <= p10:
        return max(0.0, min(score10, (raw / p10) * score10))
    if raw <= p50:
        proportion = (raw - p10) / max(1e-9, p50 - p10)
        return score10 + (proportion * (score50 - score10))
    if raw <= p90:
        proportion = (raw - p50) / max(1e-9, p90 - p50)
        return score50 + (proportion * (score90 - score50))

    tail_scale = float(calibration.get("tail_scale") or max(1.0, p90 - p50))
    tail_gain = 20.0 * (1.0 - math.exp(-(raw - p90) / tail_scale))
    return max(0.0, min(100.0, score90 + tail_gain))


@lru_cache(maxsize=1)
def _score_calibration() -> dict[str, Any]:
    return dict(get_settings().load_json("score_calibration.json"))


def _flatten_projection_band(bundle: dict[str, Any]) -> dict[str, float | None]:
    performances = bundle.get("projected_performance", {})
    if not performances:
        return {"p10": None, "p50": None, "p90": None}
    return {
        band: float(sum(values[band] for values in performances.values()) / len(performances))
        for band in ("p10", "p50", "p90")
    }


def _pathway_comparison(
    *,
    player_id: int,
    brief: dict[str, Any],
    season: str,
    role_fit_score: float,
    projection_score: float,
) -> dict[str, Any] | None:
    pathway_player_id = brief.get("pathway_player_id")
    if not pathway_player_id:
        return None
    role_name = brief["role_name"]
    template = get_active_template_for_role(role_name)
    if template is None:
        return None
    try:
        pathway_role_fit = score_role_fit(pathway_player_id, template.template_id, season)
        pathway_projection = project_to_championship(pathway_player_id, season, brief=brief)
    except Exception:
        return {"pathway_player_id": pathway_player_id, "pathway_gap_pct": None}

    pathway_projection_score = _projection_score_from_bundle(pathway_projection)
    pathway_composite = (pathway_role_fit["score"] + pathway_projection_score) / 2.0
    target_composite = (role_fit_score + projection_score) / 2.0
    if pathway_composite == 0:
        gap_pct = 100.0
    else:
        gap_pct = ((target_composite - pathway_composite) / pathway_composite) * 100.0
    return {"pathway_player_id": pathway_player_id, "pathway_gap_pct": gap_pct}


def _log_prediction(
    *,
    player_id: int,
    brief: dict[str, Any],
    role_fit: dict[str, Any],
    current_performance: dict[str, Any],
    projection: dict[str, Any],
    availability: dict[str, Any],
    financial: dict[str, Any],
    composite: dict[str, Any],
) -> None:
    projection_band = _flatten_projection_band(projection)
    with session_scope() as session:
        session.add(
            PredictionLog(
                player_id=player_id,
                brief_id=brief["brief_id"],
                model_version=_MODEL_VERSION,
                role_fit_score=role_fit["score"],
                l1_performance_score=current_performance["score"],
                championship_projection_50th=projection_band["p50"],
                championship_projection_10th=projection_band["p10"],
                championship_projection_90th=projection_band["p90"],
                projected_minutes_share=projection["projected_minutes_share"],
                projected_adaptation_months=projection["projected_adaptation_months"],
                availability_risk_prob=1.0 - availability["probability_available_75pct"],
                financial_value_band_low=financial["fair_value_band"]["low"],
                financial_value_band_high=financial["fair_value_band"]["high"],
                var_score=financial["var_score"],
                composite_score=composite["composite_score"],
                archetype_weights_used=composite["per_layer_weights_used"],
                model_warnings=composite.get("model_warnings"),
                component_fallbacks=composite.get("component_fallbacks"),
            )
        )
