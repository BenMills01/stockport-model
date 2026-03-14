"""Hard gate filtering engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

import pandas as pd
from sqlalchemy import and_, desc, or_, select

import logging

from config import get_settings
from db.read_cache import load_latest_market_value_row, load_player_lineup_frame, load_player_match_frame, load_player_role_row, load_player_row
from db.schema import MarketValue, PathwayPlayer, Player, PlayerRole
from db.session import session_scope
from features.confidence import compute_confidence
from features.gbe import estimate_gbe_score

_LOGGER = logging.getLogger(__name__)


ABSOLUTE_GATES = {
    "registration_eligibility",
    "affordability",
    "role_relevance",
    "role_profile_fit",
    "pathway_blocking_hard",
    "age_band_fit",
}

AGE_CURVE_BY_ROLE_FAMILY = {
    "centre_back": "centre_back",
    "full_back_wing_back": "full_back_wing_back",
    "midfield": "central_midfielder",
    "transition_midfield": "central_midfielder",
    "wide_creator_runner": "wide_attacker_winger",
    "striker": "striker",
}


@dataclass(frozen=True)
class GateResult:
    gate_name: str
    gate_class: str
    result: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "gate_name": self.gate_name,
            "class": self.gate_class,
            "result": self.result,
            "detail": self.detail,
        }


def apply_gates(player_id: int, brief: dict[str, Any]) -> dict[str, Any]:
    """Apply all gates in spec order to a single player."""

    context = _load_player_gate_context(player_id=player_id, brief=brief)
    return _evaluate_gates_with_context(context=context, brief=brief)


def filter_universe(brief: dict[str, Any]) -> pd.DataFrame:
    """Apply gates across the scoped player universe and return passers."""

    league_scope = brief.get("league_scope") or [league["league_id"] for league in get_settings().load_json("leagues.json")]
    role_name = brief.get("role_name")
    season = str(brief.get("season") or "")
    with session_scope() as session:
        query = select(Player).where(Player.current_league_id.in_(league_scope))
        if role_name and season:
            query = (
                query.join(
                    PlayerRole,
                    and_(
                        PlayerRole.player_id == Player.player_id,
                        PlayerRole.season == season,
                    ),
                )
                .where(
                    or_(
                        PlayerRole.primary_role == role_name,
                        PlayerRole.secondary_role == role_name,
                    )
                )
            )
        players = list(session.scalars(query).unique())

    rows = []
    for player in players:
        result = apply_gates(player.player_id, brief)
        if result["passed"]:
            rows.append(
                {
                    "player_id": player.player_id,
                    "player_name": player.player_name,
                    "current_team": player.current_team,
                    "current_league_id": player.current_league_id,
                    "passed": True,
                    "caution_flags": [gate["gate_name"] for gate in result["gate_results"] if gate["result"] == "caution"],
                    "exception_flags": [gate["gate_name"] for gate in result["gate_results"] if gate["result"] == "exception"],
                    "gate_results": result["gate_results"],
                }
            )
    return pd.DataFrame(rows)


_THRESHOLD_SPEC: dict[str, tuple[type, float, float]] = {
    # key: (expected_type, min_valid, max_valid)
    "age_band_leeway_years":              (float, 0.0,  10.0),
    "min_role_profile_label_rows":        (int,   1.0,  50.0),
    "min_role_profile_grid_rows":         (int,   1.0,  50.0),
    "pathway_hard_readiness_months":      (int,   1.0,  36.0),
    "pathway_hard_max_contract_years":    (int,   1.0,  10.0),
    "wage_discipline_multiplier":         (float, 1.0,   3.0),
    "availability_risk_caution_threshold":(float, 0.0,   1.0),
    "championship_projection_floor":      (float, 0.0, 100.0),
    "pathway_soft_readiness_min_months":  (int,   1.0,  36.0),
    "pathway_soft_readiness_max_months":  (int,   1.0,  60.0),
    "pathway_soft_max_contract_years":    (int,   1.0,  10.0),
}


def _get_gate_thresholds() -> dict[str, Any]:
    """Load gate numeric thresholds from config, with safe defaults."""
    defaults: dict[str, Any] = {
        "age_band_leeway_years": 2.0,
        "min_role_profile_label_rows": 5,
        "min_role_profile_grid_rows": 5,
        "pathway_hard_readiness_months": 12,
        "pathway_hard_max_contract_years": 2,
        "wage_discipline_multiplier": 1.2,
        "availability_risk_caution_threshold": 0.40,
        "championship_projection_floor": 40.0,
        "pathway_soft_readiness_min_months": 12,
        "pathway_soft_readiness_max_months": 24,
        "pathway_soft_max_contract_years": 2,
    }
    try:
        loaded = dict(get_settings().load_json("gate_thresholds.json"))
    except Exception:
        _LOGGER.warning("gate_thresholds.json missing or unreadable; using built-in defaults.")
        return defaults

    merged = {**defaults, **loaded}
    for key, (expected_type, lo, hi) in _THRESHOLD_SPEC.items():
        raw = merged.get(key)
        try:
            value = expected_type(raw)
            if not (lo <= value <= hi):
                raise ValueError(f"{value} outside [{lo}, {hi}]")
            merged[key] = value
        except (TypeError, ValueError) as exc:
            _LOGGER.warning(
                "gate_thresholds.json: invalid value for '%s' (%r): %s — using default %r.",
                key, raw, exc, defaults[key],
            )
            merged[key] = defaults[key]

    return merged


def _evaluate_gates_with_context(
    *,
    context: dict[str, Any],
    brief: dict[str, Any],
) -> dict[str, Any]:
    gate_results: list[GateResult] = []
    thresholds = _get_gate_thresholds()

    gbe = context.get("gbe_result", {})
    registration_gate = _registration_gate(gbe)
    gate_results.append(registration_gate)

    affordability_gate = _affordability_gate(context, brief)
    gate_results.append(affordability_gate)

    role_relevance_gate = _role_relevance_gate(context, brief)
    gate_results.append(role_relevance_gate)

    gate_results.append(_role_profile_gate(context, brief, thresholds))

    hard_pathway_gate = _pathway_hard_gate(context, brief, thresholds)
    gate_results.append(hard_pathway_gate)

    gate_results.append(_wage_discipline_gate(context, brief, thresholds))
    gate_results.append(_brief_age_gate(context, brief, thresholds))
    gate_results.append(_age_profile_gate(context, brief))
    gate_results.append(_availability_risk_gate(context, brief, thresholds))
    gate_results.append(_data_confidence_gate(context))
    gate_results.append(_championship_projection_gate(context, brief, thresholds))
    gate_results.append(_pathway_soft_gate(context, brief, thresholds))

    passed = not any(
        gate.gate_name in ABSOLUTE_GATES and gate.result == "fail"
        for gate in gate_results
    )
    return {
        "passed": passed,
        "gate_results": [gate.as_dict() for gate in gate_results],
    }


def _registration_gate(gbe_result: dict[str, Any]) -> GateResult:
    status = gbe_result.get("status")
    if status in (None, "green", "amber"):
        detail = gbe_result.get("notes", "Player is registrable or likely registrable from current estimate.")
        return GateResult("registration_eligibility", "absolute", "pass", detail)
    return GateResult(
        "registration_eligibility",
        "absolute",
        "fail",
        "GBE estimate is red or registration could not be established from available data.",
    )


def _affordability_gate(context: dict[str, Any], brief: dict[str, Any]) -> GateResult:
    market_value = context.get("market_value_eur")
    wage_estimate = context.get("wage_estimate")
    contract_years = int(brief.get("budget_max_contract_years") or 0)
    max_fee_raw = brief.get("budget_max_fee")
    max_wage_raw = brief.get("budget_max_wage")
    max_fee = Decimal(str(max_fee_raw)) if max_fee_raw not in (None, "") else None
    max_wage = Decimal(str(max_wage_raw)) if max_wage_raw not in (None, "") else None

    if max_fee is None and max_wage is None:
        return GateResult(
            "affordability",
            "absolute",
            "pass",
            "No fee or wage cap set on the brief; affordability gate skipped.",
        )

    if max_fee is not None and max_wage is not None:
        if market_value is None or wage_estimate is None:
            return GateResult(
                "affordability",
                "absolute",
                "fail",
                "Missing market value or wage estimate; affordability cannot be established.",
            )
        total_cost = market_value + (wage_estimate * contract_years)
        budget_total = max_fee + (max_wage * contract_years)
        if total_cost <= budget_total:
            return GateResult("affordability", "absolute", "pass", f"Estimated total cost {total_cost} within budget {budget_total}.")
        return GateResult("affordability", "absolute", "fail", f"Estimated total cost {total_cost} exceeds budget {budget_total}.")

    if max_fee is not None:
        if market_value is None:
            return GateResult(
                "affordability",
                "absolute",
                "fail",
                "Missing market value; fee affordability cannot be established.",
            )
        if market_value <= max_fee:
            return GateResult("affordability", "absolute", "pass", f"Estimated fee {market_value} within max fee {max_fee}.")
        return GateResult(
            "affordability",
            "absolute",
            "fail",
            f"Estimated fee {market_value} exceeds max fee {max_fee}.",
        )

    if wage_estimate is None or max_wage is None:
        return GateResult(
            "affordability",
            "absolute",
            "fail",
            "Missing wage estimate; wage affordability cannot be established.",
        )
    if wage_estimate <= max_wage:
        return GateResult("affordability", "absolute", "pass", f"Estimated wage {wage_estimate} within max wage {max_wage}.")
    return GateResult("affordability", "absolute", "fail", f"Estimated wage {wage_estimate} exceeds max wage {max_wage}.")


def _role_relevance_gate(context: dict[str, Any], brief: dict[str, Any]) -> GateResult:
    role_name = brief.get("role_name")
    player_roles = {context.get("primary_role"), context.get("secondary_role")}
    if role_name in player_roles:
        return GateResult("role_relevance", "absolute", "pass", f"Player role matches brief role '{role_name}'.")
    return GateResult("role_relevance", "absolute", "fail", f"Player roles {sorted(role for role in player_roles if role)} do not match brief role '{role_name}'.")


def _role_profile_gate(context: dict[str, Any], brief: dict[str, Any], thresholds: dict[str, Any]) -> GateResult:
    role_name = str(brief.get("role_name") or "")
    rules = get_settings().load_json("role_profile_rules.json").get(role_name)
    if not rules:
        return GateResult("role_profile_fit", "absolute", "pass", "No additional physical or positional restrictions configured for this role.")

    profile = context.get("role_profile") or {}
    failures: list[str] = []
    cautions: list[str] = []
    min_label_rows = int(thresholds["min_role_profile_label_rows"])
    min_grid_rows = int(thresholds["min_role_profile_grid_rows"])

    height_cm = profile.get("height_cm")
    minimum_height_cm = rules.get("minimum_height_cm")
    if minimum_height_cm is not None:
        if height_cm is None:
            cautions.append("Height unavailable for role-profile check.")
        elif float(height_cm) < float(minimum_height_cm):
            failures.append(f"Height {height_cm}cm is below the {minimum_height_cm}cm role-profile floor.")

    starter_label_rows = int(profile.get("starter_label_rows") or 0)
    if starter_label_rows >= min_label_rows:
        _append_role_profile_threshold_failure(
            failures=failures,
            value=profile.get("forward_label_share"),
            threshold=rules.get("minimum_forward_label_share"),
            comparator="min",
            label="starter appearances logged as forwards",
        )
        _append_role_profile_threshold_failure(
            failures=failures,
            value=profile.get("midfielder_label_share"),
            threshold=rules.get("maximum_midfielder_label_share"),
            comparator="max",
            label="starter appearances logged as midfielders",
        )
    else:
        cautions.append("Not enough labelled starter rows to fully verify positional profile.")

    attack_zone_rows = int(profile.get("attack_zone_rows") or 0)
    if attack_zone_rows >= min_grid_rows:
        _append_role_profile_threshold_failure(
            failures=failures,
            value=profile.get("central_attack_share"),
            threshold=rules.get("minimum_central_attack_share"),
            comparator="min",
            label="central attacking usage share",
        )
        _append_role_profile_threshold_failure(
            failures=failures,
            value=profile.get("wide_attack_share"),
            threshold=rules.get("maximum_wide_attack_share"),
            comparator="max",
            label="wide attacking usage share",
        )
    else:
        cautions.append("Not enough lineup grid rows to fully verify central versus wide usage.")

    if failures:
        return GateResult("role_profile_fit", "absolute", "fail", " ; ".join(failures))
    if cautions:
        return GateResult("role_profile_fit", "caution", "caution", " ; ".join(cautions))
    return GateResult("role_profile_fit", "absolute", "pass", "Physical and positional profile fits the role restrictions.")


def _pathway_hard_gate(context: dict[str, Any], brief: dict[str, Any], thresholds: dict[str, Any]) -> GateResult:
    pathway = context.get("pathway_player")
    if pathway is None:
        return GateResult("pathway_blocking_hard", "absolute", "pass", "No hard pathway conflict.")

    readiness = pathway.get("readiness_estimate_months")
    pathway_age = pathway.get("age")
    player_age = context.get("age")
    contract_years = int(brief.get("budget_max_contract_years") or 0)
    readiness_months = int(thresholds["pathway_hard_readiness_months"])
    max_contract_years = int(thresholds["pathway_hard_max_contract_years"])
    if (
        readiness is not None
        and readiness < readiness_months
        and pathway_age is not None
        and player_age is not None
        and player_age <= pathway_age
        and contract_years > max_contract_years
    ):
        return GateResult(
            "pathway_blocking_hard",
            "absolute",
            "fail",
            f"Pathway player is within {readiness_months} months of readiness and would be blocked by a long external deal.",
        )
    return GateResult("pathway_blocking_hard", "absolute", "pass", "No hard pathway conflict.")


def _wage_discipline_gate(context: dict[str, Any], brief: dict[str, Any], thresholds: dict[str, Any]) -> GateResult:
    wage_estimate = context.get("wage_estimate")
    if wage_estimate is None:
        return GateResult("wage_discipline", "caution", "caution", "No external wage estimate available.")

    wage_band_raw = brief.get("club_wage_band")
    if wage_band_raw in (None, ""):
        wage_band_raw = brief.get("budget_max_wage")
    if wage_band_raw in (None, ""):
        return GateResult("wage_discipline", "caution", "pass", "No wage band set on the brief.")

    multiplier = Decimal(str(thresholds["wage_discipline_multiplier"]))
    pct = int(multiplier * 100)
    wage_band = Decimal(str(wage_band_raw))
    if wage_estimate > (multiplier * wage_band):
        return GateResult("wage_discipline", "caution", "caution", f"Estimated wage exceeds {pct}% of the role band.")
    return GateResult("wage_discipline", "caution", "pass", "Estimated wage is within the role band.")


def _brief_age_gate(context: dict[str, Any], brief: dict[str, Any], thresholds: dict[str, Any]) -> GateResult:
    age = context.get("age")
    if age is None:
        return GateResult("age_band_fit", "absolute", "fail", "Player age unavailable; brief age band cannot be verified.")

    age_min = brief.get("age_min")
    age_max = brief.get("age_max")
    leeway = float(thresholds["age_band_leeway_years"])

    if age_min is not None and age < (age_min - leeway):
        return GateResult(
            "age_band_fit",
            "absolute",
            "fail",
            f"Player age {age:.1f} is more than {leeway:.0f} years below brief minimum {age_min}.",
        )
    if age_max is not None and age > (age_max + leeway):
        return GateResult(
            "age_band_fit",
            "absolute",
            "fail",
            f"Player age {age:.1f} is more than {leeway:.0f} years above brief maximum {age_max}.",
        )

    if age_min is not None and age < age_min:
        return GateResult(
            "age_band_fit",
            "caution",
            "caution",
            f"Player age {age:.1f} is below the brief band but within the {leeway:.0f}-year tolerance.",
        )
    if age_max is not None and age > age_max:
        return GateResult(
            "age_band_fit",
            "caution",
            "caution",
            f"Player age {age:.1f} is above the brief band but within the {leeway:.0f}-year tolerance.",
        )
    return GateResult("age_band_fit", "absolute", "pass", "Player age is within the brief band.")


def _age_profile_gate(context: dict[str, Any], brief: dict[str, Any]) -> GateResult:
    age = context.get("age")
    if age is None:
        return GateResult("age_profile_fit", "caution", "caution", "Player age unavailable for positional age-curve assessment.")

    acceptable_older_age = context.get("acceptable_older_signing_age")
    if acceptable_older_age is not None and age > acceptable_older_age:
        return GateResult(
            "age_profile_fit",
            "caution",
            "caution",
            f"Player age {age:.1f} exceeds the position-specific older-signing threshold of {acceptable_older_age}.",
        )
    return GateResult("age_profile_fit", "caution", "pass", "Age fits the brief and positional curve.")


def _availability_risk_gate(context: dict[str, Any], brief: dict[str, Any], thresholds: dict[str, Any]) -> GateResult:
    risk = context.get("availability_risk_prob", brief.get("availability_risk_prob"))
    if risk is None:
        return GateResult("availability_risk", "caution", "caution", "Availability risk model output not yet supplied.")
    threshold = float(thresholds["availability_risk_caution_threshold"])
    if float(risk) > threshold:
        return GateResult("availability_risk", "caution", "caution", f"Availability risk probability {float(risk):.2f} exceeds {threshold:.2f}.")
    return GateResult("availability_risk", "caution", "pass", "Availability risk is within tolerance.")


def _data_confidence_gate(context: dict[str, Any]) -> GateResult:
    confidence = context.get("confidence", {})
    tier = confidence.get("confidence_tier")
    if tier == "Low":
        return GateResult("data_confidence", "caution", "caution", "Fewer than 10 tracked appearances.")
    return GateResult("data_confidence", "caution", "pass", f"Confidence tier is {tier or 'unknown'}." )


def _championship_projection_gate(context: dict[str, Any], brief: dict[str, Any], thresholds: dict[str, Any]) -> GateResult:
    archetypes = {
        str(brief.get("archetype_primary") or "").lower(),
        str(brief.get("archetype_secondary") or "").lower(),
    }
    if not archetypes.intersection({"championship_transition", "emerging_asset"}):
        return GateResult("championship_projection", "exception", "pass", "Projection exception does not apply to this archetype.")

    projection = context.get("championship_projection_50th", brief.get("championship_projection_50th"))
    if projection is None:
        return GateResult("championship_projection", "exception", "exception", "Championship projection output not yet supplied.")
    floor = float(thresholds["championship_projection_floor"])
    if float(projection) < floor:
        return GateResult("championship_projection", "exception", "exception", f"Base-case projection {float(projection):.1f} is below the {floor:.0f}th percentile threshold.")
    return GateResult("championship_projection", "exception", "pass", "Championship projection clears the threshold.")


def _pathway_soft_gate(context: dict[str, Any], brief: dict[str, Any], thresholds: dict[str, Any]) -> GateResult:
    pathway = context.get("pathway_player")
    if pathway is None:
        return GateResult("pathway_blocking_soft", "exception", "pass", "No soft pathway conflict.")
    readiness = pathway.get("readiness_estimate_months")
    contract_years = int(brief.get("budget_max_contract_years") or 0)
    min_months = int(thresholds["pathway_soft_readiness_min_months"])
    max_months = int(thresholds["pathway_soft_readiness_max_months"])
    max_contract = int(thresholds["pathway_soft_max_contract_years"])
    if readiness is not None and min_months <= readiness <= max_months and contract_years > max_contract:
        return GateResult(
            "pathway_blocking_soft",
            "exception",
            "exception",
            f"Pathway player is {min_months}-{max_months} months from readiness and external deal would run beyond {max_contract} years.",
        )
    return GateResult("pathway_blocking_soft", "exception", "pass", "No soft pathway conflict.")


def _load_player_gate_context(player_id: int, brief: dict[str, Any]) -> dict[str, Any]:
    season = str(brief.get("season") or _infer_latest_role_season(player_id))
    with session_scope() as session:
        pathway_row = None
        if brief.get("pathway_player_id"):
            pathway_row = session.get(PathwayPlayer, brief["pathway_player_id"])

    player_record = load_player_row(player_id)
    role_record = load_player_role_row(player_id, season)
    market_record = load_latest_market_value_row(player_id)
    pathway_record = _row_to_dict(pathway_row, PathwayPlayer) if pathway_row is not None else None

    context = {
        "player_id": player_id,
        "player": player_record,
        "primary_role": role_record.get("primary_role"),
        "secondary_role": role_record.get("secondary_role"),
        "market_value_eur": market_record.get("market_value_eur"),
        "wage_estimate": market_record.get("wage_estimate"),
        "confidence": compute_confidence(player_id, season),
        "gbe_result": estimate_gbe_score(player_id),
        "age": _player_age(player_record.get("birth_date")),
        "pathway_player": _pathway_record_with_age(pathway_record),
        "availability_risk_prob": brief.get("availability_risk_prob"),
        "championship_projection_50th": brief.get("championship_projection_50th"),
        "acceptable_older_signing_age": _acceptable_older_signing_age(brief.get("role_name")),
        "role_profile": _build_role_profile_context(player_id=player_id, season=season, player_record=player_record),
    }
    return context


def _infer_latest_role_season(player_id: int) -> str:
    with session_scope() as session:
        role_row = session.scalar(
            select(PlayerRole)
            .where(PlayerRole.player_id == player_id)
            .order_by(desc(PlayerRole.season))
        )
    return str(role_row.season) if role_row is not None else ""


def _acceptable_older_signing_age(role_name: str | None) -> int | None:
    if not role_name:
        return None
    templates = get_settings().load_json("role_templates.json")
    role_family = None
    for template in templates:
        if template["role_name"] == role_name:
            role_family = template.get("role_family")
            break
    curve_key = AGE_CURVE_BY_ROLE_FAMILY.get(str(role_family))
    if curve_key is None:
        return None
    age_curves = get_settings().load_json("age_curves.json")
    return int(age_curves[curve_key]["acceptable_older_signing_age"])


def _player_age(birth_date: date | None) -> float | None:
    if birth_date is None:
        return None
    return round((date.today() - birth_date).days / 365.25, 2)


def _pathway_record_with_age(pathway_record: dict[str, Any] | None) -> dict[str, Any] | None:
    if pathway_record is None:
        return None
    record = dict(pathway_record)
    record["age"] = _player_age(record.get("birth_date"))
    return record


def _row_to_dict(row: Any, model: Any) -> dict[str, Any]:
    if row is None:
        return {}
    return {
        column.name: getattr(row, column.name)
        for column in model.__table__.columns
    }


def _build_role_profile_context(
    *,
    player_id: int,
    season: str,
    player_record: dict[str, Any],
) -> dict[str, Any]:
    lineups = load_player_lineup_frame(player_id, season)
    matches = load_player_match_frame(player_id, season)
    starters = lineups[lineups.get("is_starter", False).fillna(False)].copy() if not lineups.empty else lineups

    labelled = starters[starters["position_label"].notna()] if not starters.empty and "position_label" in starters else starters.iloc[0:0]
    attack_usage = _summarise_attack_usage(starters)

    total_minutes = float(pd.to_numeric(matches["minutes"], errors="coerce").fillna(0.0).sum()) if not matches.empty and "minutes" in matches else 0.0
    total_duels = float(pd.to_numeric(matches["duels_total"], errors="coerce").fillna(0.0).sum()) if not matches.empty and "duels_total" in matches else 0.0

    return {
        "height_cm": player_record.get("height_cm"),
        "starter_label_rows": int(len(labelled.index)),
        "forward_label_share": _share_of_rows(labelled, "position_label", "F"),
        "midfielder_label_share": _share_of_rows(labelled, "position_label", "M"),
        "attack_zone_rows": attack_usage["attack_zone_rows"],
        "central_attack_share": attack_usage["central_attack_share"],
        "wide_attack_share": attack_usage["wide_attack_share"],
        "total_minutes": total_minutes,
        "duels_total_per90": (total_duels * 90.0 / total_minutes) if total_minutes > 0 else None,
    }


def _share_of_rows(frame: pd.DataFrame, column: str, value: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    return float((frame[column] == value).mean())


def _summarise_attack_usage(starters: pd.DataFrame) -> dict[str, Any]:
    if starters.empty or "grid_position" not in starters or "position_label" not in starters:
        return {"attack_zone_rows": 0, "central_attack_share": None, "wide_attack_share": None}

    frame = starters.copy()
    grid_parts = frame["grid_position"].astype(str).str.extract(r"^(?P<row>\d+):(?P<col>\d+)$")
    frame["grid_row"] = pd.to_numeric(grid_parts["row"], errors="coerce")
    frame["grid_col"] = pd.to_numeric(grid_parts["col"], errors="coerce")
    frame = frame.dropna(subset=["grid_row", "grid_col"])
    if frame.empty:
        return {"attack_zone_rows": 0, "central_attack_share": None, "wide_attack_share": None}

    frame["grid_row"] = frame["grid_row"].astype(int)
    frame["grid_col"] = frame["grid_col"].astype(int)
    frame = frame[frame["position_label"].isin({"F", "M"}) & (frame["grid_row"] >= 3)]
    if frame.empty:
        return {"attack_zone_rows": 0, "central_attack_share": None, "wide_attack_share": None}

    frame["row_width"] = frame.groupby(["fixture_id", "team", "grid_row"])["grid_col"].transform("max")
    central_mask = (
        (frame["row_width"] == 1)
        | ((frame["row_width"] == 2) & (frame["position_label"] == "F"))
        | ((frame["row_width"] >= 3) & (frame["grid_col"] > 1) & (frame["grid_col"] < frame["row_width"]))
    )
    wide_mask = (frame["row_width"] >= 3) & frame["grid_col"].isin({1}) | (
        (frame["row_width"] >= 3) & (frame["grid_col"] == frame["row_width"])
    )
    considered = frame[central_mask | wide_mask].copy()
    if considered.empty:
        return {"attack_zone_rows": 0, "central_attack_share": None, "wide_attack_share": None}

    central_share = float(central_mask.loc[considered.index].mean())
    wide_share = float(wide_mask.loc[considered.index].mean())
    return {
        "attack_zone_rows": int(len(considered.index)),
        "central_attack_share": central_share,
        "wide_attack_share": wide_share,
    }


def _append_role_profile_threshold_failure(
    *,
    failures: list[str],
    value: float | None,
    threshold: float | None,
    comparator: str,
    label: str,
) -> None:
    if threshold is None or value is None:
        return
    numeric_value = float(value)
    numeric_threshold = float(threshold)
    if comparator == "min" and numeric_value < numeric_threshold:
        failures.append(f"{label.capitalize()} {numeric_value:.2f} is below the {numeric_threshold:.2f} minimum.")
    if comparator == "max" and numeric_value > numeric_threshold:
        failures.append(f"{label.capitalize()} {numeric_value:.2f} is above the {numeric_threshold:.2f} maximum.")
