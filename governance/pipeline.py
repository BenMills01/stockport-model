"""Pipeline and workflow management."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import and_, func, select

from db.schema import Brief, Override, Pipeline as PipelineRow, Player, PlayerRole, PredictionLog
from db.schema import ScoutNote, WyscoutZoneStat
from db.session import session_scope
from features.confidence import compute_confidence
from gates.filtering import filter_universe
from scoring.composite import compute_composite
from scoring.composite import projection_score_from_logged_p50


REQUIRED_BRIEF_FIELDS = [
    "role_name",
    "archetype_primary",
    "intent",
    "budget_max_contract_years",
    "age_min",
    "age_max",
    "league_scope",
    "timeline",
    "pathway_check_done",
    "created_by",
    "approved_by",
]

TIMELINE_LABEL_TO_DATE = {
    "summer": (7, 1),
    "winter": (1, 1),
    "january": (1, 1),
}


def create_brief(params: dict[str, Any]) -> int:
    """Validate and create a live brief."""

    missing = [field for field in REQUIRED_BRIEF_FIELDS if field not in params or params[field] in (None, "", [])]
    if missing:
        raise ValueError(f"Missing required brief fields: {', '.join(missing)}")
    if not params.get("pathway_check_done"):
        raise ValueError("Pathway check required before brief goes live")

    with session_scope() as session:
        brief = Brief(
            role_name=params["role_name"],
            archetype_primary=params["archetype_primary"],
            archetype_secondary=params.get("archetype_secondary"),
            intent=params["intent"],
            budget_max_fee=params.get("budget_max_fee"),
            budget_max_wage=params.get("budget_max_wage"),
            budget_max_contract_years=params["budget_max_contract_years"],
            age_min=params["age_min"],
            age_max=params["age_max"],
            league_scope=params["league_scope"],
            timeline=_normalise_timeline(params["timeline"]),
            pathway_check_done=params["pathway_check_done"],
            pathway_player_id=params.get("pathway_player_id"),
            status="live",
            created_by=params["created_by"],
            approved_by=params["approved_by"],
        )
        session.add(brief)
        session.flush()
        return int(brief.brief_id)


def generate_longlist(brief_id: int) -> pd.DataFrame:
    """Generate and persist a scored longlist for a brief."""

    brief = _load_brief_dict(brief_id)
    passed = filter_universe(brief)
    if passed.empty:
        empty = pd.DataFrame(columns=["player_id"])
        empty.attrs["skipped_players"] = []
        _upsert_pipeline_rows(empty, brief)
        return passed

    scored_rows = []
    skipped_players = []
    season = str(brief["season"])
    for _, row in passed.iterrows():
        try:
            score = compute_composite(int(row["player_id"]), brief, season)
        except Exception as exc:
            skipped_players.append(
                {
                    "player_id": int(row["player_id"]),
                    "player_name": row.get("player_name"),
                    "error": str(exc),
                }
            )
            continue
        scored_rows.append({**row.to_dict(), **score, "brief_id": brief_id})

    if not scored_rows:
        empty = pd.DataFrame(scored_rows)
        empty.attrs["skipped_players"] = skipped_players
        _upsert_pipeline_rows(pd.DataFrame(columns=["player_id"]), brief)
        return empty

    longlist = pd.DataFrame(scored_rows).sort_values("composite_score", ascending=False).head(60).reset_index(drop=True)
    longlist.attrs["skipped_players"] = skipped_players
    _upsert_pipeline_rows(longlist, brief)
    return longlist


def promote_to_shortlist(brief_id: int, player_id: int) -> dict[str, Any]:
    """Check Stage 3→5 criteria and promote a player when satisfied."""

    brief = _load_brief_dict(brief_id)
    with session_scope() as session:
        latest_prediction = session.scalar(
            select(PredictionLog)
            .where(
                PredictionLog.player_id == player_id,
                PredictionLog.brief_id == brief_id,
            )
            .order_by(PredictionLog.prediction_date.desc())
        )
    unmet = []
    if latest_prediction is None:
        unmet.append("No logged prediction found")
    else:
        if (latest_prediction.role_fit_score or 0) < 60:
            unmet.append("Role fit below 60th percentile threshold")
        confidence_tier = compute_confidence(player_id, str(brief["season"]))["confidence_tier"]
        if confidence_tier == "Low":
            unmet.append("Confidence below Medium")
        if (
            brief["archetype_primary"] in {"championship_transition", "emerging_asset"}
            and projection_score_from_logged_p50(latest_prediction.championship_projection_50th) < 40
        ):
            unmet.append("Championship projection below threshold")
        if latest_prediction.financial_value_band_low is None or latest_prediction.financial_value_band_high is None:
            unmet.append("Financial assessment incomplete")
        if brief.get("pathway_player_id") and latest_prediction.composite_score is None:
            unmet.append("Pathway comparison incomplete")

    if unmet:
        return {"ok": False, "unmet_conditions": unmet}

    with session_scope() as session:
        pipeline_row = session.scalar(
            select(PipelineRow).where(
                PipelineRow.brief_id == brief_id,
                PipelineRow.player_id == player_id,
            )
        )
        if pipeline_row is None:
            raise ValueError("Player is not currently in the pipeline for this brief")
        pipeline_row.stage = "shortlist"
        pipeline_row.stage_changed_date = datetime.now(UTC)
    return {"ok": True, "unmet_conditions": []}


def check_scouting_requirements(brief_id: int, player_id: int) -> dict[str, Any]:
    """Check whether a player is ready for cross-functional review."""

    brief = _load_brief_dict(brief_id)
    with session_scope() as session:
        notes = list(
            session.scalars(
                select(ScoutNote).where(ScoutNote.player_id == player_id)
            )
        )
        wyscout_count = session.scalar(
            select(func.count())
            .select_from(WyscoutZoneStat)
            .where(WyscoutZoneStat.player_id == player_id)
        ) or 0

    missing = []
    if len(notes) < 3:
        missing.append("Need at least 3 scout notes")
    if not any("top-6" in (note.notes_text or "").lower() for note in notes):
        missing.append("Need at least one top-6 opposition review")
    if not any("fits our system" in (note.notes_text or "").lower() or note.system_fit_rating >= 4 for note in notes):
        missing.append("Need coach system-fit sign-off")
    if brief["archetype_primary"] in {"championship_transition", "emerging_asset"} and wyscout_count < 6:
        missing.append("Need six Wyscout filtered imports")
    return {"ready": not missing, "missing": missing}


def log_override(
    player_id: int,
    brief_id: int,
    original_output: dict[str, Any],
    decision: str,
    reason_category: str,
    reason_text: str,
    overridden_by: str,
) -> None:
    """Persist an override event with validated taxonomy."""

    valid_categories = {"tactical", "financial", "medical", "intelligence", "alternatives", "timing"}
    if reason_category not in valid_categories:
        raise ValueError(f"Invalid reason_category '{reason_category}'")
    with session_scope() as session:
        session.add(
            Override(
                player_id=player_id,
                brief_id=brief_id,
                overridden_by=overridden_by,
                original_model_output=original_output,
                decision_made=decision,
                reason_category=reason_category,
                reason_text=reason_text,
            )
        )


def _load_brief_dict(brief_id: int) -> dict[str, Any]:
    with session_scope() as session:
        brief = session.get(Brief, brief_id)
    if brief is None:
        raise ValueError(f"Brief {brief_id} not found")
    brief_dict = {
        column.name: getattr(brief, column.name)
        for column in Brief.__table__.columns
    }
    brief_dict["season"] = _resolve_brief_season(brief_dict)
    return brief_dict


def _normalise_timeline(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError("Brief timeline must be a date, datetime, ISO date string, or window label.")

    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return date.fromisoformat(cleaned)
    except ValueError:
        pass

    normalised = cleaned.lower().replace("-", "_").replace(" ", "_")
    label, _, year_text = normalised.rpartition("_")
    if not label or not year_text.isdigit():
        raise ValueError(f"Could not parse brief timeline '{value}'")
    month_day = TIMELINE_LABEL_TO_DATE.get(label)
    if month_day is None:
        raise ValueError(f"Unsupported brief timeline label '{value}'")
    year = int(year_text)
    month, day = month_day
    return date(year, month, day)


def _resolve_brief_season(brief: dict[str, Any]) -> str:
    explicit = brief.get("season")
    if explicit not in (None, ""):
        return str(explicit)

    available = _available_player_role_seasons()
    if not available:
        return str(datetime.today().year - 1)

    timeline = brief.get("timeline")
    if isinstance(timeline, datetime):
        timeline = timeline.date()
    if isinstance(timeline, date):
        target_season = timeline.year - 1
        eligible = [season for season in available if season <= target_season]
        if eligible:
            return str(max(eligible))
    return str(max(available))


def _available_player_role_seasons() -> list[int]:
    with session_scope() as session:
        seasons = list(session.scalars(select(PlayerRole.season).distinct()))
    numeric = sorted({int(season) for season in seasons if str(season).isdigit()})
    return numeric


def _upsert_pipeline_rows(frame: pd.DataFrame, brief: dict[str, Any]) -> None:
    selected_ids = {
        int(player_id)
        for player_id in frame.get("player_id", pd.Series(dtype=int)).dropna().tolist()
    }
    with session_scope() as session:
        existing_rows = list(
            session.scalars(
                select(PipelineRow).where(PipelineRow.brief_id == brief["brief_id"])
            )
        )
        existing_by_player = {int(row.player_id): row for row in existing_rows}

        for player_id, existing in existing_by_player.items():
            if existing.stage == "longlist" and player_id not in selected_ids:
                existing.stage = "filtered_out"
                existing.stage_changed_by = "model"
                existing.stage_changed_date = datetime.now(UTC)

        for _, row in frame.iterrows():
            player_id = int(row["player_id"])
            existing = existing_by_player.get(player_id)
            if existing is None:
                session.add(
                    PipelineRow(
                        brief_id=brief["brief_id"],
                        player_id=player_id,
                        stage="longlist",
                        archetype_primary=brief["archetype_primary"],
                        archetype_secondary=brief.get("archetype_secondary"),
                        intent=brief["intent"],
                        stage_changed_by="model",
                    )
                )
            else:
                existing.stage = "longlist"
                existing.stage_changed_by = "model"
                existing.stage_changed_date = datetime.now(UTC)
