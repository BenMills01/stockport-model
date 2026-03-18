"""Database queries backing the local data viewer."""

from __future__ import annotations

from collections import defaultdict
import csv
from datetime import date as _date
from datetime import datetime
import json
from math import ceil
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy import text

from config import get_settings
from db import session_scope
from db.read_cache import load_player_match_frame
from db.schema import MatchPerformance, WyscoutSeasonStat
from features.confidence import compute_confidence, minutes_evidence_multiplier, shrink_low_sample_value
from features.opposition import compute_opposition_splits
from features.per90 import _compute_per90_frame
from features.rolling import compute_rolling
from governance.pipeline import create_brief as pipeline_create_brief
from governance.pipeline import generate_longlist, _load_brief_dict
from ingestion.common import normalise_text
from ingestion.matching import build_source_lookup_key, save_source_player_mapping
from ingestion.wyscout_import import LEAGUE_FOLDER_ALIASES, import_wyscout_root
from models.championship_projection import project_to_championship
from outputs.longlist import generate_longlist_report
from scoring.action_tiers import board_score_equation, classify_composite_action, composite_to_board_score, load_board_action_tiers, summarise_action_tiers
from scoring.composite import effective_layer_weights, projection_score_from_logged_p50
from scoring.physical import score_physical


_WYSCOUT_IMPORT_LOCK = Lock()

_BRIEF_SCORE_COLUMN_HELP = {
    "board_score": "A rescaled display score for board readability. It keeps the same ranking order as the raw composite but spreads scores into a more intuitive 0-100 range.",
    "composite_score": "Final ranking score for the brief. It blends role fit, current level, projection, and financial value, then suppresses low-minute samples and applies availability as a multiplier.",
    "role_fit_score": "Role-template fit on a 0-100 scale using league-adjusted percentiles for the target role metrics, then shrunk back toward neutral when the sample is light.",
    "current_score": "Present-level performance score on a 0-100 scale, combining role metric percentile, consistency, form trend, and output versus stronger opposition, then shrunk toward neutral for low-confidence samples.",
    "projection_score": "Projection score on a 0-100 scale. The smaller raw Championship median underneath is converted into the score the model actually uses.",
    "availability_risk": "Injury and availability risk. Lower is better. The composite uses the inverse of this risk as a multiplier.",
    "financial_score": "Value-for-money score on a 0-100 scale derived from a bounded return score that blends deal economics with player quality. Higher is better.",
}

_AGE_CURVE_BY_ROLE_FAMILY = {
    "centre_back": "centre_back",
    "full_back_wing_back": "full_back_wing_back",
    "midfield": "central_midfielder",
    "transition_midfield": "central_midfielder",
    "central_midfielder": "central_midfielder",
    "wide_creator_runner": "wide_attacker_winger",
    "wide_attacker_winger": "wide_attacker_winger",
    "striker": "striker",
    "goalkeeper": "goalkeeper",
}


def get_dashboard_context() -> dict[str, Any]:
    """Build the homepage context for the data browser."""

    league_catalog = _league_catalog()
    review_path = _latest_wyscout_review_path()
    wyscout_review_count = len(_load_wyscout_review_rows(review_path))

    with session_scope() as session:
        totals = session.execute(
            text(
                """
                select
                    (select count(*) from players) as players,
                    (select count(*) from fixtures) as fixtures,
                    (select count(*) from match_performances) as match_performances,
                    (select count(*) from fixture_team_stats) as fixture_team_stats,
                    (select count(*) from match_events) as match_events,
                    (select count(*) from lineups) as lineups,
                    (select count(*) from standings_snapshots) as standings_snapshots,
                    (select count(*) from wyscout_season_stats) as wyscout_season_stats
                """
            )
        ).mappings().one()

        fixture_rows = session.execute(
            text(
                """
                select
                    league_id,
                    season,
                    count(*) as fixture_count,
                    min(date)::date as first_fixture_date,
                    max(date)::date as last_fixture_date
                from fixtures
                group by league_id, season
                order by league_id, season desc
                """
            )
        ).mappings().all()
        performance_rows = _count_rows_by_league_season(session, "match_performances")
        team_stat_fixture_counts = _count_distinct_fixtures_by_league_season(session, "fixture_team_stats")
        event_fixture_counts = _count_distinct_fixtures_by_league_season(session, "match_events")
        lineup_fixture_counts = _count_distinct_fixtures_by_league_season(session, "lineups")

        recent_fixtures = session.execute(
            text(
                """
                with perf as (
                    select fixture_id, count(*) as player_rows
                    from match_performances
                    group by fixture_id
                ),
                stats as (
                    select fixture_id, count(*) as team_stat_rows
                    from fixture_team_stats
                    group by fixture_id
                ),
                events as (
                    select fixture_id, count(*) as event_rows
                    from match_events
                    group by fixture_id
                ),
                lineups as (
                    select fixture_id, count(*) as lineup_rows
                    from lineups
                    group by fixture_id
                )
                select
                    f.fixture_id,
                    f.league_id,
                    f.season,
                    f.date,
                    f.home_team,
                    f.away_team,
                    f.home_score,
                    f.away_score,
                    f.status,
                    coalesce(perf.player_rows, 0) as player_rows,
                    coalesce(stats.team_stat_rows, 0) as team_stat_rows,
                    coalesce(events.event_rows, 0) as event_rows,
                    coalesce(lineups.lineup_rows, 0) as lineup_rows
                from fixtures f
                left join perf on perf.fixture_id = f.fixture_id
                left join stats on stats.fixture_id = f.fixture_id
                left join events on events.fixture_id = f.fixture_id
                left join lineups on lineups.fixture_id = f.fixture_id
                order by f.date desc
                limit 16
                """
            )
        ).mappings().all()

    coverage_rows = []
    for row in fixture_rows:
        league_id = int(row["league_id"])
        season = str(row["season"])
        meta = league_catalog.get(league_id, {})
        key = (league_id, season)
        coverage_rows.append(
            {
                "league_id": league_id,
                "season": season,
                "league_name": meta.get("name", f"League {league_id}"),
                "country": meta.get("country"),
                "tier": meta.get("tier"),
                "polling_priority": meta.get("polling_priority"),
                "fixture_count": int(row["fixture_count"]),
                "performance_rows": performance_rows.get(key, 0),
                "team_stat_fixtures": team_stat_fixture_counts.get(key, 0),
                "event_fixtures": event_fixture_counts.get(key, 0),
                "lineup_fixtures": lineup_fixture_counts.get(key, 0),
                "first_fixture_date": row["first_fixture_date"],
                "last_fixture_date": row["last_fixture_date"],
            }
        )

    recent_fixture_cards = []
    for row in recent_fixtures:
        league_id = int(row["league_id"])
        meta = league_catalog.get(league_id, {})
        recent_fixture_cards.append(
            {
                "fixture_id": int(row["fixture_id"]),
                "league_id": league_id,
                "league_name": meta.get("name", f"League {league_id}"),
                "season": str(row["season"]),
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": row["home_score"],
                "away_score": row["away_score"],
                "status": row["status"],
                "player_rows": int(row["player_rows"]),
                "team_stat_rows": int(row["team_stat_rows"]),
                "event_rows": int(row["event_rows"]),
                "lineup_rows": int(row["lineup_rows"]),
            }
        )

    coverage_rows.sort(
        key=lambda row: (
            row.get("polling_priority", 99),
            row.get("country") or "",
            row.get("tier") or 99,
            row["league_name"],
            -int(row["season"]),
        )
    )

    recent_briefs = get_recent_briefs()
    pipeline_summary = _pipeline_stage_summary()
    attention_items = _build_attention_items(
        wyscout_unmatched=wyscout_review_count,
        recent_briefs=recent_briefs,
    )

    return {
        "title": "Stockport Data Viewer",
        "generated_at": datetime.now(),
        "totals": dict(totals),
        "coverage_rows": coverage_rows,
        "recent_fixtures": recent_fixture_cards,
        "recent_briefs": recent_briefs,
        "brief_builder": get_brief_builder_context(),
        "wyscout_review": {
            "unmatched_rows": wyscout_review_count,
            "review_path": str(review_path) if review_path else None,
            "source_root": str(_wyscout_source_root()),
        },
        "pipeline_summary": pipeline_summary,
        "attention_items": attention_items,
    }


def get_on_pitch_profiles_context(
    *,
    role_name: str | None = None,
    season: str | None = None,
) -> dict[str, Any]:
    """Build a brief-free on-pitch dashboard for a selected role profile."""

    config = _on_pitch_dashboard_config()
    profile_map = _on_pitch_profile_map()
    physical_profile_map = _on_pitch_physical_profile_map()
    role_options = _on_pitch_role_options()
    season_options = _on_pitch_season_options()
    selected_role = role_name or str(config.get("default_role_name") or (role_options[0][0] if role_options else "controller"))
    if role_options and selected_role not in {role for role, _family in role_options}:
        selected_role = role_options[0][0]
    selected_season = str(season or (season_options[0] if season_options else datetime.now().year - 1))
    if season_options and selected_season not in season_options:
        selected_season = season_options[0]

    overall_weights = _dashboard_weight_rows("score_weights_pct")
    present_weights = _dashboard_weight_rows("present_weights_pct")
    upside_weights = _dashboard_weight_rows("upside_weights_pct")
    profile = profile_map.get(selected_role)
    role_family = (str(profile.get("role_family") or "") or None) if profile else None
    selected_physical_profile = (
        physical_profile_map.get(str(profile.get("physical_profile") or "").strip())
        if profile
        else None
    )

    if profile is None:
        return {
            "title": "On-Pitch Profiles",
            "generated_at": datetime.now(),
            "role_options": role_options,
            "season_options": season_options,
            "selected_role": selected_role,
            "selected_role_family": role_family,
            "selected_season": selected_season,
            "selected_profile_label": selected_role.upper(),
            "selected_profile_points": [],
            "selected_physical_label": None,
            "selected_physical_points": [],
            "selected_physical_blend": [],
            "overall_weights": overall_weights,
            "present_weights": present_weights,
            "upside_weights": upside_weights,
            "score_guides": [],
            "rows": [],
            "top_players": [],
            "present_top_players": [],
            "upside_top_players": [],
            "technical_top_players": [],
            "physical_top_players": [],
            "combined_league_top_fives": [],
            "on_pitch_league_top_fives": [],
            "present_league_top_fives": [],
            "upside_league_top_fives": [],
            "candidate_count": 0,
            "scored_count": 0,
            "skipped_count": 0,
            "minimum_minutes": int(config.get("minimum_minutes") or 0),
            "candidate_limit": int(config.get("candidate_limit") or 0),
        }

    candidate_bundle = _load_on_pitch_profile_candidates(
        profile=profile,
        season=selected_season,
        minimum_minutes=int(config.get("minimum_minutes") or 0),
        candidate_limit=int(config.get("candidate_limit") or 250),
    )
    candidates = list(candidate_bundle.get("candidates") or [])
    league_catalog = _league_catalog()
    metric_frame = _load_on_pitch_metric_frame(
        tuple(sorted(int(candidate["player_id"]) for candidate in candidates)),
        selected_season,
    )

    rows: list[dict[str, Any]] = []
    skipped_count = 0
    peer_player_ids = [c["player_id"] for c in candidates]
    for candidate in candidates:
        try:
            role_fit = _score_on_pitch_profile_fit(
                player_id=int(candidate["player_id"]),
                season=selected_season,
                metrics=dict(profile.get("metrics") or {}),
                role_family=str(profile.get("role_family") or ""),
                metric_frame=metric_frame,
            )
            current = _score_on_pitch_profile_current(
                player_id=int(candidate["player_id"]),
                season=selected_season,
                metrics=dict(profile.get("metrics") or {}),
                role_family=str(profile.get("role_family") or ""),
                metric_frame=metric_frame,
            )
            projection = project_to_championship(candidate["player_id"], selected_season, brief=None)
            projection_score = _projection_score_from_projection_bundle(projection)
        except Exception:
            skipped_count += 1
            continue

        soft_multiplier = _soft_on_pitch_minutes_multiplier(candidate.get("total_minutes"))
        row = {
            **candidate,
            "projection_score": projection_score,
            "role_fit_score": float(role_fit.get("score") or 0.0),
            "primary_score": float(role_fit.get("primary_score") or 0.0),
            "secondary_score": float(role_fit.get("secondary_score") or 0.0),
            "current_score": float(current.get("score") or 0.0),
            "soft_minutes_multiplier": soft_multiplier,
        }
        age_score, age_penalty_multiplier = _compute_upside_age_adjustment(
            role_name=selected_role,
            age_years=row.get("age_years"),
        )
        row["age_upside_score"] = age_score
        row["unknown_age_penalty_multiplier"] = age_penalty_multiplier

        # Technical score = weighted blend of role fit, current performance, projection
        raw_technical_score = _compute_dashboard_weighted_score(
            row,
            weight_rows=overall_weights,
            fields={"Role Fit": "role_fit_score", "Current": "current_score", "Projection": "projection_score"},
            soft_minutes_multiplier=soft_multiplier,
        )
        raw_present_score = _compute_dashboard_weighted_score(
            row,
            weight_rows=present_weights,
            fields={"Role Fit": "role_fit_score", "Current": "current_score"},
            soft_minutes_multiplier=soft_multiplier,
        )
        raw_upside_score = _compute_dashboard_weighted_score(
            row,
            weight_rows=upside_weights,
            fields={"Role Fit": "role_fit_score", "Projection": "projection_score", "Age": "age_upside_score"},
            soft_minutes_multiplier=soft_multiplier * age_penalty_multiplier,
        )
        row["technical_score_raw"] = raw_technical_score
        row["present_on_pitch_score_raw"] = raw_present_score
        row["upside_on_pitch_score_raw"] = raw_upside_score

        league_id = candidate.get("current_league_id")
        strength_factor = float(league_catalog.get(league_id, {}).get("strength_factor") or 1.0)
        row["league_strength_factor"] = strength_factor
        row["technical_score"] = _apply_league_strength_factor(raw_technical_score, strength_factor)
        row["present_on_pitch_score"] = _apply_league_strength_factor(raw_present_score, strength_factor)
        row["upside_on_pitch_score"] = _apply_league_strength_factor(raw_upside_score, strength_factor)

        # Physical score = peer-percentile ranked SkillCorner metrics (None if no SC data)
        try:
            physical_weights = selected_physical_profile.get("physical_weights") if selected_physical_profile else None
            gi_weights = selected_physical_profile.get("gi_weights") if selected_physical_profile else None
            row["physical_score"] = score_physical(
                candidate["player_id"],
                peer_player_ids,
                physical_weights=dict(physical_weights) if isinstance(physical_weights, dict) and physical_weights else None,
                gi_weights=dict(gi_weights) if isinstance(gi_weights, dict) and gi_weights else None,
                physical_sub_weight=(
                    float(selected_physical_profile["physical_sub_weight"])
                    if selected_physical_profile and "physical_sub_weight" in selected_physical_profile
                    else None
                ),
                gi_sub_weight=(
                    float(selected_physical_profile["gi_sub_weight"])
                    if selected_physical_profile and "gi_sub_weight" in selected_physical_profile
                    else None
                ),
            )
        except Exception:
            row["physical_score"] = None

        # On-pitch score blends raw football and physical scores, then applies league strength
        # so the cross-league board stays comparable while physical remains readable on its own.
        technical = raw_technical_score
        physical = row["physical_score"]
        if technical is not None and physical is not None:
            raw_on_pitch = round(0.60 * float(technical) + 0.40 * float(physical), 2)
        else:
            raw_on_pitch = technical
        row["on_pitch_score_raw"] = raw_on_pitch
        row["on_pitch_score"] = _apply_league_strength_factor(raw_on_pitch, strength_factor)

        rows.append(row)

    top_n = int(config.get("top_n") or 10)
    shortlist_n = int(config.get("shortlist_n") or 5)
    rows_by_on_pitch = sorted(
        rows,
        key=lambda row: (float(row.get("on_pitch_score") or 0.0), float(row.get("projection_score") or 0.0)),
        reverse=True,
    )
    present_rows = sorted(
        rows,
        key=lambda row: (float(row.get("present_on_pitch_score") or 0.0), float(row.get("current_score") or 0.0)),
        reverse=True,
    )
    upside_rows = sorted(
        rows,
        key=lambda row: (float(row.get("upside_on_pitch_score") or 0.0), float(row.get("projection_score") or 0.0)),
        reverse=True,
    )

    league_top_n = int(config.get("league_top_n") or 5)
    on_pitch_league_top_fives = _build_dashboard_league_top_fives(
        rows_by_on_pitch,
        top_n=league_top_n,
        score_key="on_pitch_score",
        tie_break_keys=("technical_score", "projection_score"),
    )
    technical_rows = sorted(
        rows,
        key=lambda row: (float(row.get("technical_score") or 0.0), float(row.get("current_score") or 0.0)),
        reverse=True,
    )
    technical_league_top_fives = _build_dashboard_league_top_fives(
        technical_rows,
        top_n=league_top_n,
        score_key="technical_score",
        tie_break_keys=("current_score", "projection_score"),
    )
    physical_rows = sorted(
        [row for row in rows if row.get("physical_score") is not None],
        key=lambda row: float(row.get("physical_score") or 0.0),
        reverse=True,
    )
    physical_league_top_fives = _build_dashboard_league_top_fives(
        physical_rows,
        top_n=league_top_n,
        score_key="physical_score",
        tie_break_keys=("on_pitch_score",),
    )

    return {
        "title": "On-Pitch Profiles",
        "generated_at": datetime.now(),
        "role_options": role_options,
        "season_options": season_options,
        "selected_role": selected_role,
        "selected_profile_label": str(profile.get("label") or selected_role.upper()),
        "selected_profile_points": list(profile.get("profile_points") or []),
        "selected_role_family": role_family,
        "selected_season": selected_season,
        "candidate_match_mode": candidate_bundle.get("match_mode", "exact_role"),
        "candidate_match_note": candidate_bundle.get("match_note"),
        "candidate_match_roles": list(candidate_bundle.get("match_roles") or [selected_role]),
        "selected_physical_label": (
            str(selected_physical_profile.get("label") or "")
            if selected_physical_profile
            else None
        ),
        "selected_physical_points": list(
            selected_physical_profile.get("profile_points") or []
        )
        if selected_physical_profile
        else [],
        "selected_physical_blend": _physical_blend_rows(selected_physical_profile),
        "overall_weights": overall_weights,
        "present_weights": present_weights,
        "upside_weights": upside_weights,
        "score_guides": _build_on_pitch_score_guides(rows),
        "rows": rows_by_on_pitch,
        "top_players": rows_by_on_pitch[:top_n],
        "present_top_players": present_rows[:shortlist_n],
        "upside_top_players": upside_rows[:shortlist_n],
        "technical_top_players": technical_rows[:shortlist_n],
        "physical_top_players": physical_rows[:shortlist_n],
        "combined_league_top_fives": _combine_dashboard_league_top_fives(
            on_pitch_league_top_fives,
            technical_league_top_fives,
            physical_league_top_fives,
        ),
        "on_pitch_league_top_fives": on_pitch_league_top_fives,
        "technical_league_top_fives": technical_league_top_fives,
        "physical_league_top_fives": physical_league_top_fives,
        "candidate_count": len(candidates),
        "scored_count": len(rows),
        "skipped_count": skipped_count,
        "minimum_minutes": int(config.get("minimum_minutes") or 0),
        "candidate_limit": int(config.get("candidate_limit") or 0),
    }


def _on_pitch_dashboard_config() -> dict[str, Any]:
    return dict(get_settings().load_json("on_pitch_dashboard.json"))


def _on_pitch_profile_payloads() -> list[dict[str, Any]]:
    return list(get_settings().load_json("on_pitch_profiles.json"))


def _on_pitch_profile_map() -> dict[str, dict[str, Any]]:
    return {
        str(profile.get("role_name")): dict(profile)
        for profile in _on_pitch_profile_payloads()
        if str(profile.get("role_name") or "").strip()
    }


def _on_pitch_physical_profile_map() -> dict[str, dict[str, Any]]:
    payload = get_settings().load_json("on_pitch_physical_profiles.json")
    return {
        str(profile_name): dict(profile)
        for profile_name, profile in dict(payload).items()
        if str(profile_name or "").strip()
    }


def _on_pitch_role_options() -> list[tuple[str, str | None]]:
    options: list[tuple[str, str | None]] = []
    for profile in _on_pitch_profile_payloads():
        role_name = str(profile.get("role_name") or "").strip()
        if not role_name:
            continue
        options.append((role_name, profile.get("role_family")))
    return options


def _on_pitch_season_options() -> list[str]:
    with session_scope() as session:
        rows = session.execute(
            text(
                """
                select distinct season
                from player_roles
                order by season desc
                """
            )
        ).scalars().all()
    return [str(value) for value in rows]


def _dashboard_weight_rows(config_key: str) -> list[dict[str, Any]]:
    config = _on_pitch_dashboard_config()
    labels = {
        "role_fit": "Role Fit",
        "current_performance": "Current",
        "upward_projection": "Projection",
        "age_upside": "Age",
    }
    weight_map = config.get(config_key) or {}
    return [
        {"label": labels[key], "percent": float(weight_map[key])}
        for key in labels
        if key in weight_map
    ]


def _physical_blend_rows(profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not profile:
        return []
    physical_pct = 100.0 * float(profile.get("physical_sub_weight") or 0.0)
    gi_pct = 100.0 * float(profile.get("gi_sub_weight") or 0.0)
    total = physical_pct + gi_pct
    if total <= 0.0:
        return []
    return [
        {"label": "Physical Output", "percent": round(physical_pct / total, 1)},
        {"label": "Pressure / GI", "percent": round(gi_pct / total, 1)},
    ]


def _load_on_pitch_profile_candidates(
    *,
    profile: dict[str, Any],
    season: str,
    minimum_minutes: int,
    candidate_limit: int,
) -> dict[str, Any]:
    league_catalog = _league_catalog()
    role_name = str(profile.get("role_name") or "")
    minimum_height_cm = profile.get("minimum_height_cm")
    candidate_roles = {
        str(value).strip()
        for value in (profile.get("candidate_roles") or [])
        if str(value or "").strip()
    }
    allowed_sc_positions: set[str] | None = (
        {str(value).upper() for value in (profile.get("allowed_sc_positions") or []) if str(value or "").strip()}
        if profile.get("allowed_sc_positions")
        else None
    )
    recruitment_leagues = {
        int(league_id)
        for league_id, meta in league_catalog.items()
        if meta.get("recruitment_board")
    }

    with session_scope() as session:
        rows = session.execute(
            text(
                """
                with season_minutes as (
                    select
                        mp.player_id,
                        coalesce(sum(mp.minutes), 0.0) as total_minutes
                    from match_performances mp
                    where mp.season = :season
                    group by mp.player_id
                ),
                sc_modal_position as (
                    select
                        spm.player_id,
                        sp.position as modal_sc_position,
                        row_number() over (
                            partition by spm.player_id
                            order by count(*) desc, sp.position
                        ) as rn
                    from skillcorner_physical sp
                    join skillcorner_player_map spm on spm.sc_player_id = sp.sc_player_id
                    where sp.position is not null and sp.position <> 'SUB'
                    group by spm.player_id, sp.position
                )
                select
                    pr.player_id,
                    pr.primary_role,
                    pr.secondary_role,
                    pl.player_name,
                    pl.current_team,
                    pl.current_league_id,
                    pl.birth_date,
                    pl.current_age_years,
                    pl.height_cm,
                    coalesce(sm.total_minutes, 0.0) as total_minutes,
                    scp.modal_sc_position
                from player_roles pr
                join players pl on pl.player_id = pr.player_id
                left join season_minutes sm on sm.player_id = pr.player_id
                left join sc_modal_position scp on scp.player_id = pr.player_id and scp.rn = 1
                where pr.season = :season
                order by coalesce(sm.total_minutes, 0.0) desc, pl.player_name
                """
            ),
            {"season": season},
        ).mappings().all()

    candidates = _filter_on_pitch_profile_candidates(
        rows,
        role_names=candidate_roles or {role_name},
        tracked_leagues=recruitment_leagues,
        minimum_minutes=minimum_minutes,
        minimum_height_cm=minimum_height_cm,
        allowed_sc_positions=allowed_sc_positions,
        league_catalog=league_catalog,
        candidate_limit=candidate_limit,
    )
    match_roles = sorted(candidate_roles or {role_name})
    note = None
    if match_roles:
        note = (
            f"This Stockport profile is using the underlying candidate pool "
            f"({', '.join(match_roles)}) and then scoring those players against the {role_name.upper()} template."
        )
    return {
        "candidates": candidates,
        "match_mode": "configured_pool",
        "match_roles": match_roles or [role_name],
        "match_note": note,
    }


def _filter_on_pitch_profile_candidates(
    rows: list[dict[str, Any]],
    *,
    role_names: set[str],
    tracked_leagues: set[int],
    minimum_minutes: int,
    minimum_height_cm: float | None,
    allowed_sc_positions: set[str] | None,
    league_catalog: dict[int, dict[str, Any]],
    candidate_limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        player_roles = {
            str(row.get("primary_role") or "").strip(),
            str(row.get("secondary_role") or "").strip(),
        }
        if not player_roles.intersection(role_names):
            continue
        modal_sc_pos = row.get("modal_sc_position")
        if allowed_sc_positions is not None and modal_sc_pos is not None:
            if str(modal_sc_pos).upper() not in allowed_sc_positions:
                continue
        league_id = _parse_optional_int(row.get("current_league_id"))
        if league_id is None or league_id not in tracked_leagues:
            continue
        total_minutes = float(row.get("total_minutes") or 0.0)
        if total_minutes < float(minimum_minutes):
            continue
        height_cm = _parse_optional_float(row.get("height_cm"))
        if minimum_height_cm is not None and height_cm is not None and height_cm < float(minimum_height_cm):
            continue

        candidates.append(
            {
                "player_id": int(row["player_id"]),
                "player_name": row["player_name"],
                "current_team": row["current_team"],
                "current_league_id": league_id,
                "league_name": league_catalog.get(league_id, {}).get("name", f"League {league_id}"),
                "primary_role": row.get("primary_role"),
                "secondary_role": row.get("secondary_role"),
                "birth_date": row.get("birth_date"),
                "current_age_years": _parse_optional_float(row.get("current_age_years")),
                "age_years": _player_age_years(
                    row.get("birth_date"),
                    row.get("current_age_years"),
                ),
                "age_label": _age_label(
                    row.get("birth_date"),
                    row.get("current_age_years"),
                ),
                "height_cm": height_cm,
                "total_minutes": total_minutes,
                "modal_sc_position": modal_sc_pos,
            }
        )
        if len(candidates) >= candidate_limit:
            break
    return candidates


# Maps our internal Wyscout metric names to possible normalized column aliases in metrics_json.
# Column names are normalised by _normalise_column_name in wyscout_import.py:
#   strip → lower → non-alnum → '_' → collapse multiple '_'
# CRITICAL: Wyscout "won, %" columns normalise WITHOUT a _pct suffix:
#   "Aerial duels won, %" → "aerial_duels_won"   ← a percentage, handled in RATIO aliases below
#   "Accurate crosses, %" → "accurate_crosses"   ← a percentage, NOT a count
# All of these are volume metrics → stored as {name}_per90 in the frame.
_WYSCOUT_METRIC_ALIASES: dict[str, list[str]] = {
    # Expected goals / assists — prefer the pre-computed per-90 column from the export
    "xg": ["xg_per_90", "xg"],
    "xa": ["xa_per_90", "xa"],
    # Non-penalty goals per 90 (Wyscout does not export npxG, only np goals)
    "non_penalty_goals": ["non_penalty_goals_per_90", "non_penalty_goals"],
    # Passing / creation — all already per-90 in the Wyscout Search Results export
    "progressive_passes": ["progressive_passes_per_90"],
    "deep_completions": ["deep_completions_per_90"],
    "shot_assists": ["shot_assists_per_90"],
    "passes_to_final_third": ["passes_to_final_third_per_90"],
    # Carrying / running
    "progressive_runs": ["progressive_runs_per_90"],
    "touches_in_box": ["touches_in_box_per_90"],
    # Crossing — volume only; the "Accurate crosses, %" column is a ratio (handled below)
    "crosses": ["crosses_per_90"],
    # Aerial — volume of aerial challenges (win % is handled as a ratio metric below)
    "aerial_duels": ["aerial_duels_per_90"],
    # Defensive
    "padj_interceptions": ["padj_interceptions"],  # PAdj season total → divided by minutes
    "padj_tackles": ["padj_sliding_tackles"],       # PAdj season total → divided by minutes
    "ball_recoveries": ["ball_recoveries_per_90", "ball_recoveries"],
    # GK
    "successful_exits": ["exits_per_90"],
}

# Ratio metrics are not per-90 normalized — stored as their own name (no _per90 suffix).
# IMPORTANT: these aliases are the exact normalized DB keys (result of _normalise_column_name).
#   "Aerial duels won, %"   → key "aerial_duels_won"   (NOT "aerial_duels_won_pct")
#   "Defensive duels won, %" → key "defensive_duels_won"
#   "Accurate long passes, %" → key "accurate_long_passes"
#   "Save rate, %"           → key "save_rate"
_WYSCOUT_RATIO_ALIASES: dict[str, list[str]] = {
    "aerial_duels_won_pct": ["aerial_duels_won"],
    "defensive_duels_won_pct": ["defensive_duels_won"],
    "offensive_duels_won_pct": ["offensive_duels_won"],
    "long_pass_accuracy": ["accurate_long_passes"],
    "save_pct": ["save_rate"],
}

# All ratio metric names — looked up in the frame without _per90 suffix (like pass_accuracy).
_WYSCOUT_RATIO_METRICS: frozenset[str] = frozenset(_WYSCOUT_RATIO_ALIASES.keys())


def _extract_wyscout_per90(
    metrics: dict[str, Any], minutes_played: int | None, aliases: list[str]
) -> float | None:
    """Return a per-90 value for the first matching alias in a Wyscout metrics_json dict."""
    for alias in aliases:
        raw = metrics.get(alias)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        # Already a per-90 value if the column name says so
        if "per_90" in alias or alias.endswith("_90"):
            return value
        # Convert total → per-90 using stored minutes
        if minutes_played and minutes_played >= 45:
            return value / (minutes_played / 90.0)
    return None


def _extract_wyscout_ratio(metrics: dict[str, Any], aliases: list[str]) -> float | None:
    """Return a ratio/percentage value (no per-90 conversion) from a Wyscout metrics_json dict."""
    for alias in aliases:
        raw = metrics.get(alias)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _load_wyscout_metric_frame(player_ids: tuple[int, ...], season: str) -> pd.DataFrame:
    """Load Wyscout advanced metrics for a set of players, returned as per-90 and ratio columns."""
    if not player_ids:
        return pd.DataFrame()
    with session_scope() as session:
        rows = list(
            session.scalars(
                select(WyscoutSeasonStat).where(
                    WyscoutSeasonStat.season == season,
                    WyscoutSeasonStat.player_id.in_(player_ids),
                )
            )
        )
    if not rows:
        return pd.DataFrame()

    # Where a player has multiple rows (played in multiple leagues), keep the one with most minutes
    best: dict[int, WyscoutSeasonStat] = {}
    for row in rows:
        existing = best.get(row.player_id)
        if existing is None or (row.minutes_played or 0) > (existing.minutes_played or 0):
            best[row.player_id] = row

    records = []
    for row in best.values():
        metrics = row.metrics_json or {}
        record: dict[str, Any] = {"player_id": row.player_id}
        for metric_name, aliases in _WYSCOUT_METRIC_ALIASES.items():
            record[f"{metric_name}_per90"] = _extract_wyscout_per90(metrics, row.minutes_played, aliases)
        for metric_name, aliases in _WYSCOUT_RATIO_ALIASES.items():
            record[metric_name] = _extract_wyscout_ratio(metrics, aliases)
        records.append(record)

    return pd.DataFrame(records)


def _load_on_pitch_metric_frame(player_ids_key: tuple[int, ...], season: str) -> pd.DataFrame:
    if not player_ids_key:
        return pd.DataFrame()
    with session_scope() as session:
        rows = list(
            session.scalars(
                select(MatchPerformance).where(
                    MatchPerformance.season == season,
                    MatchPerformance.player_id.in_(player_ids_key),
                )
            )
        )
    frame = pd.DataFrame([_match_performance_row_to_dict(row) for row in rows])
    if frame.empty:
        return pd.DataFrame()
    per90 = _compute_per90_frame(frame)
    if per90.empty:
        return pd.DataFrame()

    per90_columns = [column for column in per90.columns if column.endswith("_per90")]
    aggregations: dict[str, Any] = {column: "mean" for column in per90_columns}
    if "pass_accuracy" in per90.columns:
        aggregations["pass_accuracy"] = "mean"
    aggregations["minutes"] = "sum"
    aggregations["league_id"] = _mode_value
    base = per90.groupby("player_id", as_index=False).agg(aggregations)

    # Blend in Wyscout advanced metrics — left join so players without data get NaN
    wyscout = _load_wyscout_metric_frame(player_ids_key, season)
    if not wyscout.empty:
        base = base.merge(wyscout, on="player_id", how="left", suffixes=("", "_wyscout"))
    return base


_PRIMARY_METRIC_CONFIG: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parents[1] / "config" / "on_pitch_primary_metrics.json").read_text()
)
_PRIMARY_WEIGHT: float = float(_PRIMARY_METRIC_CONFIG.get("primary_weight", 0.40))
_PRIMARY_FAMILIES: dict[str, dict[str, Any]] = _PRIMARY_METRIC_CONFIG.get("families", {})

_RATIO_METRIC_NAMES: frozenset[str] = frozenset({"pass_accuracy"}) | _WYSCOUT_RATIO_METRICS


def _score_metric_set(
    metric_dict: dict[str, float],
    target_row: Any,
    peers: pd.DataFrame,
) -> tuple[float, dict[str, dict[str, float]], dict[str, float]]:
    """Score a weighted metric dict against same-league peers.

    Returns (renormalised_raw_score_0_to_100, contributions, percentile_scores).
    Missing metrics (no column or no player value) are skipped and weights renormalised.
    """
    contributions: dict[str, dict[str, float]] = {}
    percentile_scores: dict[str, float] = {}
    raw_score = 0.0
    total_weight = 0.0

    for metric_name, weight in metric_dict.items():
        metric_column = metric_name if metric_name in _RATIO_METRIC_NAMES else f"{metric_name}_per90"
        if metric_column not in peers.columns:
            continue
        player_value = _parse_optional_float(target_row.get(metric_column))
        if player_value is None:
            continue
        peer_values = pd.to_numeric(peers[metric_column], errors="coerce").dropna()
        percentile = _percentile_from_series(player_value, peer_values)
        percentile_scores[metric_name] = percentile
        contribution = float(weight) * percentile
        raw_score += contribution
        total_weight += float(weight)
        contributions[metric_name] = {
            "player_percentile": percentile,
            "weight": float(weight),
            "contribution": contribution,
        }

    if 0.0 < total_weight < 1.0:
        raw_score = raw_score / total_weight

    return raw_score, contributions, percentile_scores


def _score_on_pitch_profile_fit(
    *,
    player_id: int,
    season: str,
    metrics: dict[str, float],
    role_family: str,
    metric_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Two-layer profile fit score.

    Layer 1 (primary, 40%): universal quality for this role family — is this a good player at all?
    Layer 2 (secondary, 60%): profile-specific metrics — what type of player are they?
    """
    if metric_frame.empty:
        raise ValueError("No on-pitch metric frame available")

    target = metric_frame[metric_frame["player_id"] == player_id]
    if target.empty:
        raise ValueError(f"No metric row found for player {player_id}")

    target_row = target.iloc[0]
    player_league_id = int(target_row["league_id"])
    peers = metric_frame[metric_frame["league_id"] == player_league_id].copy()
    if peers.empty:
        peers = metric_frame.copy()

    # --- Layer 1: primary (universal family quality) ---
    primary_metrics = (_PRIMARY_FAMILIES.get(role_family) or {}).get("metrics") or {}
    primary_raw, primary_contributions, primary_percentiles = _score_metric_set(
        primary_metrics, target_row, peers
    )

    # --- Layer 2: secondary (profile-specific fit) ---
    secondary_raw, secondary_contributions, secondary_percentiles = _score_metric_set(
        metrics, target_row, peers
    )

    secondary_weight = 1.0 - _PRIMARY_WEIGHT
    raw_score = _PRIMARY_WEIGHT * primary_raw + secondary_weight * secondary_raw

    confidence = compute_confidence(player_id, season)
    shrinkage_factor = float(confidence.get("shrinkage_factor", 1.0) or 1.0)
    score = shrink_low_sample_value(
        player_value=raw_score,
        league_role_average=50.0,
        shrinkage_factor=shrinkage_factor,
    )
    return {
        "score": float(max(0.0, min(100.0, score))),
        "raw_score": float(max(0.0, min(100.0, raw_score))),
        "primary_score": float(max(0.0, min(100.0, primary_raw))),
        "secondary_score": float(max(0.0, min(100.0, secondary_raw))),
        "shrinkage_factor": shrinkage_factor,
        "primary_decomposition": primary_contributions,
        "decomposition": secondary_contributions,
        "percentiles": {**primary_percentiles, **secondary_percentiles},
        "confidence_tier": confidence["confidence_tier"],
    }


def _score_on_pitch_profile_current(
    *,
    player_id: int,
    season: str,
    metrics: dict[str, float],
    role_family: str,
    metric_frame: pd.DataFrame,
) -> dict[str, Any]:
    role_fit = _score_on_pitch_profile_fit(
        player_id=player_id,
        season=season,
        metrics=metrics,
        role_family=role_family,
        metric_frame=metric_frame,
    )
    opposition = compute_opposition_splits(player_id=player_id, season=season)
    match_frame = load_player_match_frame(player_id, season).copy()
    per90 = _compute_per90_frame(match_frame)
    rolling = compute_rolling(per90)

    total_weight = sum(float(weight) for weight in metrics.values()) or 1.0
    role_metric_score = float(role_fit.get("raw_score") or 0.0)
    consistency_sum = 0.0
    slope_numerator = 0.0
    slope_denominator = 0.0
    tier1_sum = 0.0

    for metric_name, raw_weight in metrics.items():
        weight = float(raw_weight) / total_weight
        roll_info = rolling.get(metric_name, {})
        opp_info = opposition.get(metric_name) or {}
        consistency_sum += _consistency_to_score(roll_info.get("roll_10_cv")) * weight
        slope = roll_info.get("trend_slope_10")
        if slope is not None:
            slope_numerator += float(slope) * weight
            slope_denominator += weight
        tier1_sum += _tier1_percentile_score(
            tier1_value=opp_info.get("tier1"),
            baseline_value=opp_info.get("tier3"),
        ) * weight

    consistency_score = consistency_sum
    weighted_slope: float | None = (slope_numerator / slope_denominator) if slope_denominator > 0.0 else None
    trend_score = _trend_to_score(weighted_slope)
    vs_tier1_percentile = tier1_sum
    raw_score = (
        (0.50 * role_metric_score)
        + (0.20 * consistency_score)
        + (0.15 * trend_score)
        + (0.15 * vs_tier1_percentile)
    )

    confidence = compute_confidence(player_id, season)
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
        "percentile_in_role": role_metric_score,
        "form_trend": _trend_label(weighted_slope),
        "consistency": consistency_score,
        "vs_tier1_percentile": vs_tier1_percentile,
    }


def _projection_score_from_projection_bundle(bundle: dict[str, Any]) -> float:
    performances = bundle.get("projected_performance") or {}
    if not performances:
        return 50.0
    medians = [
        float(values.get("p50"))
        for values in performances.values()
        if values.get("p50") is not None
    ]
    if not medians:
        return 50.0
    return projection_score_from_logged_p50(sum(medians) / len(medians))


def _role_family_for_role(role_name: str) -> str | None:
    profile = _on_pitch_profile_map().get(role_name)
    if profile is not None:
        return str(profile.get("role_family") or "") or None
    templates = get_settings().load_json("role_templates.json")
    for template in templates:
        if str(template.get("role_name")) == role_name:
            return str(template.get("role_family") or "") or None
    return None


def _role_names_for_family(role_name: str) -> list[str]:
    family = _role_family_for_role(role_name)
    if not family:
        return [role_name]
    templates = get_settings().load_json("role_templates.json")
    role_names = [
        str(template.get("role_name"))
        for template in templates
        if str(template.get("role_family") or "") == family and str(template.get("role_name") or "").strip()
    ]
    if role_name not in role_names:
        role_names.append(role_name)
    return sorted(dict.fromkeys(role_names))


def _percentile_from_series(player_value: float | None, peer_values: pd.Series) -> float:
    if player_value is None or peer_values.empty:
        return 0.0
    return float((peer_values <= player_value).mean() * 100.0)


def _mode_value(series: pd.Series) -> Any:
    modes = series.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else None


def _match_performance_row_to_dict(row: MatchPerformance) -> dict[str, Any]:
    return {
        column.name: getattr(row, column.name)
        for column in MatchPerformance.__table__.columns
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


def _player_age_years(birth_date: Any, current_age_years: Any = None) -> float | None:
    fallback_age = _parse_optional_float(current_age_years)
    if birth_date is None:
        return fallback_age
    try:
        if not isinstance(birth_date, _date):
            birth_date = datetime.strptime(str(birth_date)[:10], "%Y-%m-%d").date()
        return round((_date.today() - birth_date).days / 365.25, 1)
    except Exception:
        return fallback_age


def _age_label(birth_date: Any, current_age_years: Any = None) -> str:
    age = _player_age_years(birth_date, current_age_years)
    if age is None:
        return "Unknown"
    return f"{age:.1f}"


def _compute_upside_age_adjustment(*, role_name: str, age_years: float | None) -> tuple[float, float]:
    config = _on_pitch_dashboard_config()
    if age_years is None:
        return (
            float(config.get("unknown_age_upside_score") or 25.0),
            float(config.get("unknown_age_penalty_multiplier") or 0.85),
        )

    curves = get_settings().load_json("age_curves.json")
    profile = _on_pitch_profile_map().get(role_name)
    curve_key = str(profile.get("age_curve_key") or "") if profile else ""
    if not curve_key:
        role_family = _role_family_for_role(role_name)
        curve_key = str(_AGE_CURVE_BY_ROLE_FAMILY.get(str(role_family)) or "")
    if not curve_key or curve_key not in curves:
        return 50.0, 1.0

    curve = curves[curve_key]
    resale_low, resale_high = [float(value) for value in curve["resale_window"]]
    peak_low, peak_high = [float(value) for value in curve["output_peak"]]
    decline_low, decline_high = [float(value) for value in curve["decline_onset"]]
    minimum_score = float(config.get("minimum_known_age_upside_score") or 5.0)
    age = float(age_years)

    if age <= resale_low:
        score = 100.0
    elif age <= resale_high:
        proportion = (age - resale_low) / max(1e-9, resale_high - resale_low)
        score = 100.0 - (proportion * 5.0)
    elif age <= peak_high:
        proportion = (age - resale_high) / max(1e-9, peak_high - resale_high)
        score = 95.0 - (proportion * 25.0)
    elif age <= decline_high:
        proportion = (age - peak_high) / max(1e-9, decline_high - peak_high)
        score = 70.0 - (proportion * 45.0)
    else:
        years_beyond_decline = age - decline_high
        score = max(minimum_score, 25.0 - (years_beyond_decline * 8.0))

    score = max(minimum_score, min(100.0, score))
    return score, 1.0


def _compute_dashboard_weighted_score(
    row: dict[str, Any],
    *,
    weight_rows: list[dict[str, Any]],
    fields: dict[str, str],
    soft_minutes_multiplier: float,
) -> float:
    score = 0.0
    for label, field_name in fields.items():
        weight = float(next((item["percent"] for item in weight_rows if item["label"] == label), 0.0)) / 100.0
        score += float(row.get(field_name) or 0.0) * weight
    return score * soft_minutes_multiplier


def _build_dashboard_league_top_fives(
    rows: list[dict[str, Any]],
    *,
    top_n: int,
    score_key: str,
    tie_break_keys: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["current_league_id"])].append(row)

    cards: list[dict[str, Any]] = []
    for league_id, league_rows in grouped.items():
        ranked_rows = sorted(
            league_rows,
            key=lambda row: tuple(
                float(row.get(key) or 0.0)
                for key in (score_key, *tie_break_keys)
            ),
            reverse=True,
        )[:top_n]
        if not ranked_rows:
            continue
        cards.append(
            {
                "league_id": league_id,
                "league_name": ranked_rows[0]["league_name"],
                "players": ranked_rows,
                "top_score": float(ranked_rows[0].get(score_key) or 0.0),
                "score_key": score_key,
            }
        )
    return sorted(cards, key=lambda card: (card["top_score"], card["league_name"]), reverse=True)


def _combine_dashboard_league_top_fives(
    on_pitch_cards: list[dict[str, Any]],
    technical_cards: list[dict[str, Any]],
    physical_cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    league_catalog = _league_catalog()
    combined: dict[int, dict[str, Any]] = {}

    def _ensure_card(league_id: int, league_name: str) -> dict[str, Any]:
        if league_id not in combined:
            meta = league_catalog.get(league_id, {})
            combined[league_id] = {
                "league_id": league_id,
                "league_name": league_name,
                "country": meta.get("country"),
                "tier": meta.get("tier"),
                "polling_priority": meta.get("polling_priority", 99),
                "on_pitch": {"players": [], "top_score": 0.0},
                "technical": {"players": [], "top_score": 0.0},
                "physical": {"players": [], "top_score": 0.0},
            }
        return combined[league_id]

    for key, cards in (
        ("on_pitch", on_pitch_cards),
        ("technical", technical_cards),
        ("physical", physical_cards),
    ):
        for card in cards:
            league_id = int(card["league_id"])
            entry = _ensure_card(league_id, str(card["league_name"]))
            entry[key] = {
                "players": list(card.get("players") or []),
                "top_score": float(card.get("top_score") or 0.0),
            }

    return sorted(
        combined.values(),
        key=lambda row: (
            int(row.get("polling_priority") or 99),
            str(row.get("country") or ""),
            int(row.get("tier") or 99),
            str(row["league_name"]),
        ),
    )


def _build_on_pitch_score_guides(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    guides: list[dict[str, Any]] = []
    for label, key in (
        ("On-Pitch", "on_pitch_score"),
        ("Technical", "technical_score"),
        ("Physical", "physical_score"),
    ):
        values = sorted(float(row.get(key) or 0.0) for row in rows)
        if not values:
            continue
        median = _score_percentile(values, 0.50)
        p75 = _score_percentile(values, 0.75)
        p90 = _score_percentile(values, 0.90)
        guides.append(
            {
                "label": label,
                "score_key": key,
                "median": median,
                "p75": p75,
                "p90": p90,
                "bands": [
                    {"label": "Excellent", "threshold": p90, "description": "Top 10% of this board"},
                    {"label": "Strong", "threshold": p75, "description": "Top 25% of this board"},
                    {"label": "Good", "threshold": median, "description": "At or above the board median"},
                ],
            }
        )
    return guides


def _score_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * percentile)))
    return round(float(values[index]), 1)


def get_brief_builder_context(
    *,
    form_values: dict[str, Any] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Build form options and defaults for the browser brief creator."""

    settings = get_settings()
    templates = settings.load_json("role_templates.json")
    archetypes = settings.load_json("archetype_weights.json")
    league_catalog = _league_catalog()
    defaults = {
        "role_name": "controller",
        "archetype_primary": "promotion_accelerator",
        "archetype_secondary": "",
        "intent": "first_team",
        "budget_max_fee": "",
        "budget_max_wage": "",
        "budget_max_contract_years": "3",
        "age_min": "20",
        "age_max": "29",
        "timeline": f"summer_{datetime.now().year}",
        "league_scope": [40, 41, 42, 62, 79, 89, 144, 145, 179],
        "created_by": "Ben Mills",
        "approved_by": "Ben Mills",
        "pathway_check_done": True,
        "pathway_player_id": "",
    }
    values = {**defaults, **(form_values or {})}

    role_options = sorted(
        {
            (
                str(template["role_name"]),
                str(template.get("role_family") or "").replace("_", " ").title(),
            )
            for template in templates
        },
        key=lambda item: item[0],
    )
    league_options = sorted(
        (
            {
                "league_id": league_id,
                "label": f"{meta['name']} ({meta.get('country')}, Tier {meta.get('tier')})",
            }
            for league_id, meta in league_catalog.items()
        ),
        key=lambda row: row["label"],
    )
    return {
        "form_values": values,
        "errors": errors or [],
        "role_options": role_options,
        "archetype_options": list(archetypes.keys()),
        "intent_options": [
            ("first_team", "First-team upgrade"),
            ("development", "Development / pathway"),
            ("succession", "Succession planning"),
        ],
        "timeline_options": _timeline_options(),
        "league_options": league_options,
    }


def _pipeline_stage_summary() -> list[dict[str, Any]]:
    """Return counts of pipeline rows per stage, ordered by funnel depth."""

    stage_order = ["longlist", "shortlist", "recommendation", "engaged", "signed"]
    with session_scope() as session:
        rows = session.execute(
            text("select stage, count(*) as player_count from pipeline group by stage")
        ).mappings().all()
    counts = {str(row["stage"]): int(row["player_count"]) for row in rows}
    return [
        {"stage": stage, "player_count": counts.get(stage, 0)}
        for stage in stage_order
    ]


def _build_attention_items(
    *,
    wyscout_unmatched: int,
    recent_briefs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a list of actionable attention items for the dashboard header."""

    items: list[dict[str, Any]] = []
    if wyscout_unmatched > 0:
        items.append({
            "type": "warn",
            "text": f"{wyscout_unmatched:,} unmatched Wyscout row(s) need resolution",
            "href": "/wyscout-review",
            "cta": "Resolve now",
        })
    unrun = [b for b in recent_briefs if int(b.get("longlist_count") or 0) == 0]
    if unrun:
        labels = ", ".join(f"#{b['brief_id']}" for b in unrun[:3])
        items.append({
            "type": "info",
            "text": f"Brief(s) {labels} have not been run yet",
            "href": f"/brief/{unrun[0]['brief_id']}",
            "cta": "Run longlist",
        })
    return items


def get_recent_briefs(limit: int = 10) -> list[dict[str, Any]]:
    """Return the most recent browser-visible briefs with run-state summary."""

    with session_scope() as session:
        rows = session.execute(
            text(
                """
                with prediction_counts as (
                    select brief_id, count(distinct player_id) as prediction_count, max(prediction_date) as last_prediction_at
                    from predictions_log
                    group by brief_id
                ),
                pipeline_counts as (
                    select brief_id, count(*) as longlist_count
                    from pipeline
                    where stage = 'longlist'
                    group by brief_id
                )
                select
                    b.brief_id,
                    b.role_name,
                    b.archetype_primary,
                    b.archetype_secondary,
                    b.intent,
                    b.timeline,
                    b.status,
                    b.created_date,
                    coalesce(pc.prediction_count, 0) as prediction_count,
                    coalesce(pl.longlist_count, 0) as longlist_count,
                    pc.last_prediction_at
                from briefs b
                left join prediction_counts pc on pc.brief_id = b.brief_id
                left join pipeline_counts pl on pl.brief_id = b.brief_id
                order by b.created_date desc, b.brief_id desc
                limit :limit
                """
            ),
            {"limit": int(limit)},
        ).mappings().all()

    briefs = []
    for row in rows:
        brief_dict = _load_brief_dict(int(row["brief_id"]))
        briefs.append(
            {
                **dict(row),
                "season": brief_dict.get("season"),
                "report_exists": _brief_report_path(int(row["brief_id"])).exists(),
            }
        )
    return briefs


def create_brief_from_form(form: dict[str, list[str]]) -> dict[str, Any]:
    """Parse browser form input and create a brief."""

    params = {
        "role_name": _required_form_text(form, "role_name"),
        "archetype_primary": _required_form_text(form, "archetype_primary"),
        "archetype_secondary": _optional_form_text(form, "archetype_secondary"),
        "intent": _required_form_text(form, "intent"),
        "budget_max_fee": _optional_form_number(form, "budget_max_fee"),
        "budget_max_wage": _optional_form_number(form, "budget_max_wage"),
        "budget_max_contract_years": int(_required_form_text(form, "budget_max_contract_years")),
        "age_min": int(_required_form_text(form, "age_min")),
        "age_max": int(_required_form_text(form, "age_max")),
        "league_scope": [int(value) for value in form.get("league_scope", []) if str(value).strip()],
        "timeline": _required_form_text(form, "timeline"),
        "pathway_check_done": bool(form.get("pathway_check_done")),
        "pathway_player_id": _optional_form_int(form, "pathway_player_id"),
        "created_by": _required_form_text(form, "created_by"),
        "approved_by": _required_form_text(form, "approved_by"),
    }
    if not params["league_scope"]:
        raise ValueError("Select at least one league for the brief.")
    action = _optional_form_text(form, "action") or "create_run"
    brief_id = pipeline_create_brief(params)
    return {"brief_id": brief_id, "action": action, "params": params}


def run_brief_longlist(brief_id: int) -> dict[str, Any]:
    """Run the longlist pipeline and persist a browser-openable report."""

    started = perf_counter()
    frame = generate_longlist(brief_id)
    duration_seconds = perf_counter() - started
    html = generate_longlist_report(brief_id)
    report_path = _brief_report_path(brief_id)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    return {
        "brief_id": brief_id,
        "row_count": int(len(frame.index)),
        "skipped_players": frame.attrs.get("skipped_players", []),
        "duration_seconds": round(duration_seconds, 2),
        "report_path": str(report_path),
    }


def get_brief_context(brief_id: int, *, message: str | None = None) -> dict[str, Any] | None:
    """Build a detail page for a single brief and its latest longlist results."""

    try:
        brief = _load_brief_dict(brief_id)
    except ValueError:
        return None

    league_catalog = _league_catalog()
    with session_scope() as session:
        summary = session.execute(
            text(
                """
                select
                    (select count(distinct player_id) from predictions_log where brief_id = :brief_id) as prediction_count,
                    (select count(*) from pipeline where brief_id = :brief_id and stage = 'longlist') as longlist_count,
                    (select max(prediction_date) from predictions_log where brief_id = :brief_id) as last_prediction_at
                """
            ),
            {"brief_id": brief_id},
        ).mappings().one()
        predictions = session.execute(
            text(
                """
                with current_longlist as (
                    select player_id
                    from pipeline
                    where brief_id = :brief_id
                      and stage = 'longlist'
                ),
                season_minutes as (
                    select
                        mp.player_id,
                        coalesce(sum(mp.minutes), 0.0) as total_minutes
                    from match_performances mp
                    where mp.season = :season
                    group by mp.player_id
                ),
                latest_predictions as (
                    select distinct on (p.player_id)
                        p.player_id,
                        p.composite_score,
                        p.role_fit_score,
                        p.l1_performance_score,
                        p.championship_projection_50th,
                        p.availability_risk_prob,
                        p.var_score,
                        p.model_warnings,
                        p.component_fallbacks
                    from predictions_log p
                    join current_longlist cl on cl.player_id = p.player_id
                    where p.brief_id = :brief_id
                    order by p.player_id, p.prediction_date desc
                )
                select
                    p.player_id,
                    pl.player_name,
                    pl.current_team,
                    pl.current_league_id,
                    coalesce(sm.total_minutes, 0.0) as total_minutes,
                    p.composite_score,
                    p.role_fit_score,
                    p.l1_performance_score,
                    p.championship_projection_50th,
                    p.availability_risk_prob,
                    p.var_score,
                    p.model_warnings,
                    p.component_fallbacks
                from latest_predictions p
                join players pl on pl.player_id = p.player_id
                left join season_minutes sm on sm.player_id = p.player_id
                order by p.composite_score desc nulls last, pl.player_name
                limit 60
                """
            ),
            {"brief_id": brief_id, "season": str(brief["season"])},
        ).mappings().all()
        all_predictions = session.execute(
            text(
                """
                with season_minutes as (
                    select
                        mp.player_id,
                        coalesce(sum(mp.minutes), 0.0) as total_minutes
                    from match_performances mp
                    where mp.season = :season
                    group by mp.player_id
                ),
                latest_predictions as (
                    select distinct on (p.player_id)
                        p.player_id,
                        p.composite_score,
                        p.role_fit_score,
                        p.l1_performance_score,
                        p.championship_projection_50th,
                        p.availability_risk_prob,
                        p.var_score,
                        p.model_warnings,
                        p.component_fallbacks
                    from predictions_log p
                    where p.brief_id = :brief_id
                    order by p.player_id, p.prediction_date desc
                )
                select
                    p.player_id,
                    pl.player_name,
                    pl.current_team,
                    pl.current_league_id,
                    coalesce(sm.total_minutes, 0.0) as total_minutes,
                    p.composite_score,
                    p.role_fit_score,
                    p.l1_performance_score,
                    p.championship_projection_50th,
                    p.availability_risk_prob,
                    p.var_score,
                    p.model_warnings,
                    p.component_fallbacks
                from latest_predictions p
                join players pl on pl.player_id = p.player_id
                left join season_minutes sm on sm.player_id = p.player_id
                order by p.composite_score desc nulls last, pl.player_name
                """
            ),
            {"brief_id": brief_id, "season": str(brief["season"])},
        ).mappings().all()

    report_path = _brief_report_path(brief_id)
    on_pitch_weights = _build_on_pitch_weight_rows(brief)
    present_on_pitch_weights = _build_present_on_pitch_weight_rows(brief)
    upside_on_pitch_weights = _build_upside_on_pitch_weight_rows(brief)
    prediction_rows = []
    for row in predictions:
        prediction_rows.append(
            _decorate_prediction_row(
                dict(row),
                league_catalog,
                on_pitch_weights=on_pitch_weights,
                present_on_pitch_weights=present_on_pitch_weights,
                upside_on_pitch_weights=upside_on_pitch_weights,
            )
        )

    all_prediction_rows = [
        _decorate_prediction_row(
            dict(row),
            league_catalog,
            on_pitch_weights=on_pitch_weights,
            present_on_pitch_weights=present_on_pitch_weights,
            upside_on_pitch_weights=upside_on_pitch_weights,
        )
        for row in all_predictions
    ]
    on_pitch_top_players = sorted(
        all_prediction_rows,
        key=lambda row: (float(row.get("on_pitch_score") or 0.0), float(row.get("projection_score") or 0.0)),
        reverse=True,
    )[:10]
    present_on_pitch_top_players = sorted(
        all_prediction_rows,
        key=lambda row: (float(row.get("present_on_pitch_score") or 0.0), float(row.get("current_score") or 0.0)),
        reverse=True,
    )[:5]
    upside_on_pitch_top_players = sorted(
        all_prediction_rows,
        key=lambda row: (float(row.get("upside_on_pitch_score") or 0.0), float(row.get("projection_score") or 0.0)),
        reverse=True,
    )[:5]
    on_pitch_league_top_fives = _build_on_pitch_league_top_fives(all_prediction_rows)

    scoped_leagues = [
        {
            "league_id": int(league_id),
            "league_name": league_catalog.get(int(league_id), {}).get("name", f"League {league_id}"),
        }
        for league_id in (brief.get("league_scope") or [])
    ]
    return {
        "title": f"Brief {brief_id}",
        "generated_at": datetime.now(),
        "message": message,
        "brief": brief,
        "summary": dict(summary),
        "predictions": prediction_rows,
        "score_column_help": dict(_BRIEF_SCORE_COLUMN_HELP),
        "composite_weights": _build_composite_weight_rows(brief),
        "on_pitch_weights": on_pitch_weights,
        "present_on_pitch_weights": present_on_pitch_weights,
        "upside_on_pitch_weights": upside_on_pitch_weights,
        "on_pitch_top_players": on_pitch_top_players,
        "present_on_pitch_top_players": present_on_pitch_top_players,
        "upside_on_pitch_top_players": upside_on_pitch_top_players,
        "on_pitch_league_top_fives": on_pitch_league_top_fives,
        "availability_mode_note": "Availability is not added into the weighted sum. The final composite is multiplied by the player’s availability probability.",
        "board_score_equation": board_score_equation(),
        "action_tiers": load_board_action_tiers(),
        "action_tier_summary": summarise_action_tiers([row.get("composite_score") for row in prediction_rows]),
        "warning_player_count": sum(
            1
            for row in prediction_rows
            if any(bool(value) for value in row.get("component_fallbacks", {}).values())
        ),
        "scoped_leagues": scoped_leagues,
        "report_exists": report_path.exists(),
        "report_url": f"/brief/{brief_id}/report",
        "report_path": str(report_path),
    }


def render_brief_report(brief_id: int) -> str:
    """Render the current longlist report HTML for a brief."""

    return generate_longlist_report(brief_id)


def get_league_context(league_id: int, season: str | None = None) -> dict[str, Any] | None:
    """Build a league-season detail view."""

    league_catalog = _league_catalog()
    meta = league_catalog.get(league_id)
    if meta is None:
        return None

    with session_scope() as session:
        available_seasons = [
            str(value)
            for value in session.execute(
                text(
                    """
                    select distinct season
                    from fixtures
                    where league_id = :league_id
                    order by season desc
                    """
                ),
                {"league_id": league_id},
            ).scalars()
        ]
        if not available_seasons:
            return None

        selected_season = str(season or available_seasons[0])
        if selected_season not in available_seasons:
            return None

        summary = session.execute(
            text(
                """
                with season_fixtures as (
                    select fixture_id, date
                    from fixtures
                    where league_id = :league_id and season = :season
                )
                select
                    (select count(*) from season_fixtures) as fixture_count,
                    (select min(date)::date from season_fixtures) as first_fixture_date,
                    (select max(date)::date from season_fixtures) as last_fixture_date,
                    (select count(*) from match_performances mp join season_fixtures sf on sf.fixture_id = mp.fixture_id) as performance_rows,
                    (select count(distinct fixture_id) from fixture_team_stats where fixture_id in (select fixture_id from season_fixtures)) as team_stat_fixtures,
                    (select count(distinct fixture_id) from match_events where fixture_id in (select fixture_id from season_fixtures)) as event_fixtures,
                    (select count(distinct fixture_id) from lineups where fixture_id in (select fixture_id from season_fixtures)) as lineup_fixtures
                """
            ),
            {"league_id": league_id, "season": selected_season},
        ).mappings().one()

        fixtures = session.execute(
            text(
                """
                with perf as (
                    select fixture_id, count(*) as player_rows
                    from match_performances
                    group by fixture_id
                ),
                stats as (
                    select fixture_id, count(*) as team_stat_rows
                    from fixture_team_stats
                    group by fixture_id
                ),
                events as (
                    select fixture_id, count(*) as event_rows
                    from match_events
                    group by fixture_id
                ),
                lineups as (
                    select fixture_id, count(*) as lineup_rows
                    from lineups
                    group by fixture_id
                )
                select
                    f.fixture_id,
                    f.date,
                    f.home_team,
                    f.away_team,
                    f.home_score,
                    f.away_score,
                    f.status,
                    coalesce(perf.player_rows, 0) as player_rows,
                    coalesce(stats.team_stat_rows, 0) as team_stat_rows,
                    coalesce(events.event_rows, 0) as event_rows,
                    coalesce(lineups.lineup_rows, 0) as lineup_rows
                from fixtures f
                left join perf on perf.fixture_id = f.fixture_id
                left join stats on stats.fixture_id = f.fixture_id
                left join events on events.fixture_id = f.fixture_id
                left join lineups on lineups.fixture_id = f.fixture_id
                where f.league_id = :league_id and f.season = :season
                order by f.date desc, f.fixture_id desc
                limit 200
                """
            ),
            {"league_id": league_id, "season": selected_season},
        ).mappings().all()

    fixture_rows = [dict(row) for row in fixtures]
    return {
        "title": f"{meta['name']} {selected_season}",
        "generated_at": datetime.now(),
        "league": meta,
        "league_id": league_id,
        "selected_season": selected_season,
        "available_seasons": available_seasons,
        "summary": dict(summary),
        "fixtures": fixture_rows,
    }


def get_fixture_context(fixture_id: int) -> dict[str, Any] | None:
    """Build a single-fixture detail view."""

    league_catalog = _league_catalog()

    with session_scope() as session:
        fixture = session.execute(
            text(
                """
                select *
                from fixtures
                where fixture_id = :fixture_id
                """
            ),
            {"fixture_id": fixture_id},
        ).mappings().first()
        if fixture is None:
            return None

        team_stats = session.execute(
            text(
                """
                select *
                from fixture_team_stats
                where fixture_id = :fixture_id
                order by team_name
                """
            ),
            {"fixture_id": fixture_id},
        ).mappings().all()

        performances = session.execute(
            text(
                """
                select
                    mp.*,
                    p.player_name,
                    p.current_team,
                    p.photo_url
                from match_performances mp
                left join players p on p.player_id = mp.player_id
                where mp.fixture_id = :fixture_id
                order by mp.team, coalesce(mp.rating, -1) desc, coalesce(mp.minutes, 0) desc
                """
            ),
            {"fixture_id": fixture_id},
        ).mappings().all()

        events = session.execute(
            text(
                """
                select *
                from match_events
                where fixture_id = :fixture_id
                order by coalesce(time_elapsed, 0), coalesce(time_extra, 0), id
                """
            ),
            {"fixture_id": fixture_id},
        ).mappings().all()

        lineups = session.execute(
            text(
                """
                select *
                from lineups
                where fixture_id = :fixture_id
                order by team, is_starter desc, coalesce(shirt_number, 999), player_id
                """
            ),
            {"fixture_id": fixture_id},
        ).mappings().all()

    performance_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in performances:
        performance_groups[str(row["team"])].append(dict(row))

    lineup_groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in lineups:
        team_name = str(row["team"])
        group = lineup_groups.setdefault(team_name, {"starters": [], "substitutes": []})
        bucket = "starters" if row["is_starter"] else "substitutes"
        group[bucket].append(dict(row))

    league_id = int(fixture["league_id"])
    return {
        "title": f"{fixture['home_team']} vs {fixture['away_team']}",
        "generated_at": datetime.now(),
        "fixture": dict(fixture),
        "league": league_catalog.get(league_id, {"name": f"League {league_id}"}),
        "team_stats": [dict(row) for row in team_stats],
        "performance_groups": dict(performance_groups),
        "events": [dict(row) for row in events],
        "lineup_groups": lineup_groups,
    }


def get_player_context(player_id: int) -> dict[str, Any] | None:
    """Build a player detail page."""

    league_catalog = _league_catalog()

    with session_scope() as session:
        player = session.execute(
            text(
                """
                select *
                from players
                where player_id = :player_id
                """
            ),
            {"player_id": player_id},
        ).mappings().first()
        if player is None:
            return None

        season_summary = session.execute(
            text(
                """
                select
                    mp.season,
                    mp.league_id,
                    max(mp.team) as team,
                    count(*) as appearances,
                    sum(coalesce(mp.minutes, 0)) as minutes,
                    sum(coalesce(mp.goals_scored, 0)) as goals,
                    sum(coalesce(mp.assists, 0)) as assists,
                    round(avg(mp.rating)::numeric, 2) as average_rating
                from match_performances mp
                where mp.player_id = :player_id
                group by mp.season, mp.league_id
                order by mp.season desc, minutes desc
                """
            ),
            {"player_id": player_id},
        ).mappings().all()

        recent_matches = session.execute(
            text(
                """
                select
                    mp.fixture_id,
                    mp.season,
                    mp.date,
                    mp.league_id,
                    mp.home_team,
                    mp.away_team,
                    mp.team,
                    mp.minutes,
                    mp.position,
                    mp.rating,
                    mp.goals_scored,
                    mp.assists
                from match_performances mp
                where mp.player_id = :player_id
                order by mp.date desc, mp.fixture_id desc
                limit 25
                """
            ),
            {"player_id": player_id},
        ).mappings().all()

        role_rows = session.execute(
            text(
                """
                select season, primary_role, secondary_role, cluster_confidence
                from player_roles
                where player_id = :player_id
                order by season desc
                limit 3
                """
            ),
            {"player_id": player_id},
        ).mappings().all()

        injury_summary = session.execute(
            text(
                """
                select count(*) as injury_count,
                       max(date) as last_injury_date
                from injuries
                where player_id = :player_id
                """
            ),
            {"player_id": player_id},
        ).mappings().first()

        scout_note_rows = session.execute(
            text(
                """
                select
                    id,
                    scout_name,
                    date,
                    technical_rating,
                    tactical_rating,
                    physical_rating,
                    mental_rating,
                    system_fit_rating,
                    notes_text,
                    video_urls,
                    fixture_id,
                    round(
                        (technical_rating + tactical_rating + physical_rating
                         + mental_rating + system_fit_rating) / 5.0,
                        1
                    ) as avg_rating
                from scout_notes
                where player_id = :player_id
                order by date desc
                limit 10
                """
            ),
            {"player_id": player_id},
        ).mappings().all()

        per90_stats = session.execute(
            text(
                """
                select
                    season,
                    sum(coalesce(minutes, 0)) as total_minutes,
                    case when sum(coalesce(minutes, 0)) > 0
                         then round((sum(coalesce(goals_scored, 0)) * 90.0 / sum(coalesce(minutes, 0)))::numeric, 2)
                         else null end as goals_per90,
                    case when sum(coalesce(minutes, 0)) > 0
                         then round((sum(coalesce(assists, 0)) * 90.0 / sum(coalesce(minutes, 0)))::numeric, 2)
                         else null end as assists_per90,
                    case when sum(coalesce(minutes, 0)) > 0
                         then round((sum(coalesce(shots_total, 0)) * 90.0 / sum(coalesce(minutes, 0)))::numeric, 2)
                         else null end as shots_per90,
                    case when sum(coalesce(minutes, 0)) > 0
                         then round((sum(coalesce(passes_total, 0)) * 90.0 / sum(coalesce(minutes, 0)))::numeric, 2)
                         else null end as passes_per90,
                    case when sum(coalesce(minutes, 0)) > 0
                         then round((sum(coalesce(tackles, 0)) * 90.0 / sum(coalesce(minutes, 0)))::numeric, 2)
                         else null end as tackles_per90,
                    case when sum(coalesce(minutes, 0)) > 0
                         then round((sum(coalesce(duels_total, 0)) * 90.0 / sum(coalesce(minutes, 0)))::numeric, 2)
                         else null end as duels_per90,
                    case when sum(coalesce(minutes, 0)) > 0
                         then round((sum(coalesce(dribbles_attempts, 0)) * 90.0 / sum(coalesce(minutes, 0)))::numeric, 2)
                         else null end as dribbles_per90
                from match_performances
                where player_id = :player_id
                group by season
                order by season desc
                limit 3
                """
            ),
            {"player_id": player_id},
        ).mappings().all()

        market_value_history = session.execute(
            text(
                """
                select date, value_eur
                from market_value_history
                where player_id = :player_id
                order by date asc
                """
            ),
            {"player_id": player_id},
        ).mappings().all()

    season_rows = []
    for row in season_summary:
        season_rows.append(
            {
                **dict(row),
                "league_name": league_catalog.get(int(row["league_id"]), {}).get("name", f"League {row['league_id']}"),
            }
        )

    recent_rows = []
    for row in recent_matches:
        recent_rows.append(
            {
                **dict(row),
                "league_name": league_catalog.get(int(row["league_id"]), {}).get("name", f"League {row['league_id']}"),
            }
        )

    return {
        "title": player["player_name"],
        "generated_at": datetime.now(),
        "player": dict(player),
        "season_rows": season_rows,
        "recent_rows": recent_rows,
        "role_history": [dict(r) for r in role_rows],
        "injury_summary": dict(injury_summary) if injury_summary else {},
        "per90_by_season": [dict(r) for r in per90_stats],
        "market_value_history": [dict(r) for r in market_value_history],
        "scout_notes": [dict(r) for r in scout_note_rows],
    }


def get_wyscout_review_context(
    *,
    league_id: int | None = None,
    season: str | None = None,
    page: int = 1,
    message: str | None = None,
    page_size: int = 24,
) -> dict[str, Any]:
    """Build the unresolved Wyscout review page."""

    review_path = _latest_wyscout_review_path()
    all_rows = _load_wyscout_review_rows(review_path)
    league_catalog = _league_catalog()

    filtered_rows = [
        row
        for row in all_rows
        if (league_id is None or row["league_id"] == league_id)
        and (season is None or row["season"] == str(season))
    ]

    total_pages = max(1, ceil(len(filtered_rows) / page_size)) if filtered_rows else 1
    selected_page = min(max(page, 1), total_pages)
    start = (selected_page - 1) * page_size
    end = start + page_size

    available_leagues = sorted(
        (
            {
                "league_id": league_key,
                "league_name": league_catalog.get(league_key, {}).get("name", f"League {league_key}"),
                "count": sum(1 for row in all_rows if row["league_id"] == league_key),
            }
            for league_key in sorted({row["league_id"] for row in all_rows})
        ),
        key=lambda row: row["league_name"],
    )
    available_seasons = sorted({row["season"] for row in all_rows}, reverse=True)
    league_breakdown = sorted(available_leagues, key=lambda row: (-row["count"], row["league_name"]))

    rows_with_candidates = sum(1 for row in all_rows if row["candidate_options"])
    rows_with_team_matches = sum(1 for row in all_rows if row["suggested_historical_team"])
    rows_without_candidates = sum(1 for row in all_rows if not row["candidate_options"])

    return {
        "title": "Wyscout Review",
        "generated_at": datetime.now(),
        "message": message,
        "review_path": str(review_path) if review_path else None,
        "source_root": str(_wyscout_source_root()),
        "selected_league_id": league_id,
        "selected_season": str(season) if season else None,
        "available_leagues": available_leagues,
        "available_seasons": available_seasons,
        "summary": {
            "unmatched_rows": len(all_rows),
            "filtered_rows": len(filtered_rows),
            "rows_with_candidates": rows_with_candidates,
            "rows_with_team_matches": rows_with_team_matches,
            "rows_without_candidates": rows_without_candidates,
        },
        "league_breakdown": league_breakdown,
        "rows": filtered_rows[start:end],
        "page": selected_page,
        "total_pages": total_pages,
        "page_size": page_size,
    }


def apply_wyscout_review_mapping(
    *,
    source_player_name: str,
    source_team_name: str | None,
    player_id: int,
    league_id: int | None,
    match_score: float | None = None,
    source_player_external_id: str | None = None,
    rerun: bool = True,
) -> dict[str, Any]:
    """Persist a manual Wyscout mapping and optionally rerun the relevant import."""

    save_source_player_mapping(
        "wyscout",
        player_id=player_id,
        source_player_name=source_player_name,
        source_team_name=source_team_name,
        source_player_external_id=source_player_external_id,
        league_id=league_id,
        match_score=match_score,
        matched_by="manual_review",
    )

    rerun_summary = None
    if rerun:
        rerun_summary = rerun_wyscout_review_import(league_id=league_id)
    return {"player_id": player_id, "rerun_summary": rerun_summary}


def apply_wyscout_review_mappings(
    selections: list[dict[str, Any]],
    *,
    rerun: bool = True,
) -> dict[str, Any]:
    """Persist several manual Wyscout mappings and rerun once for the affected leagues."""

    if not selections:
        raise ValueError("Select at least one Wyscout match before applying the batch")

    unique_selections: dict[str, dict[str, Any]] = {}
    affected_league_ids: set[int] = set()
    for selection in selections:
        player_id = int(selection["player_id"])
        source_player_name = str(selection["source_player_name"])
        source_team_name = selection.get("source_team_name")
        source_player_external_id = selection.get("source_player_external_id")
        league_id = selection.get("league_id")
        league_id = int(league_id) if league_id is not None else None
        lookup_key = build_source_lookup_key(
            source_player_name,
            source_team_name=source_team_name,
            source_player_external_id=source_player_external_id,
        )
        unique_selections[lookup_key] = {
            "player_id": player_id,
            "source_player_name": source_player_name,
            "source_team_name": source_team_name,
            "source_player_external_id": source_player_external_id,
            "league_id": league_id,
            "match_score": selection.get("match_score"),
        }
        if league_id is not None:
            affected_league_ids.add(league_id)

    for selection in unique_selections.values():
        save_source_player_mapping(
            "wyscout",
            player_id=selection["player_id"],
            source_player_name=selection["source_player_name"],
            source_team_name=selection["source_team_name"],
            source_player_external_id=selection.get("source_player_external_id"),
            league_id=selection["league_id"],
            match_score=selection.get("match_score"),
            matched_by="manual_review_batch",
        )

    rerun_summary = None
    if rerun:
        rerun_summary = rerun_wyscout_review_import(
            league_ids=sorted(affected_league_ids) if affected_league_ids else None,
        )

    return {
        "saved_count": len(unique_selections),
        "affected_league_ids": sorted(affected_league_ids),
        "rerun_summary": rerun_summary,
    }


def rerun_wyscout_review_import(
    *,
    league_id: int | None = None,
    league_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Rerun season-average Wyscout import from the configured source root."""

    if league_id is not None and league_ids is not None:
        raise ValueError("Pass either league_id or league_ids, not both")

    settings = get_settings()
    source_root = _wyscout_source_root()
    target_leagues = league_ids if league_ids is not None else ([league_id] if league_id is not None else None)
    folder_names = _folder_names_for_leagues(source_root, target_leagues)
    if target_leagues and not folder_names:
        raise ValueError(f"No Wyscout source folder found for leagues {target_leagues}")

    acquired = _WYSCOUT_IMPORT_LOCK.acquire(blocking=False)
    if not acquired:
        raise RuntimeError("A Wyscout import is already running in the viewer")

    try:
        return import_wyscout_root(
            source_root,
            folder_names=folder_names,
            unmatched_output_dir=settings.data_dir / "wyscout" / "unmatched",
        )
    finally:
        _WYSCOUT_IMPORT_LOCK.release()


def _league_catalog() -> dict[int, dict[str, Any]]:
    settings = get_settings()
    return {int(row["league_id"]): row for row in settings.load_json("leagues.json")}


def _count_rows_by_league_season(session: Any, table_name: str) -> dict[tuple[int, str], int]:
    rows = session.execute(
        text(
            f"""
            select f.league_id, f.season, count(*) as row_count
            from {table_name} t
            join fixtures f on f.fixture_id = t.fixture_id
            group by f.league_id, f.season
            """
        )
    ).mappings().all()
    return {(int(row["league_id"]), str(row["season"])): int(row["row_count"]) for row in rows}


def _count_distinct_fixtures_by_league_season(session: Any, table_name: str) -> dict[tuple[int, str], int]:
    rows = session.execute(
        text(
            f"""
            select f.league_id, f.season, count(distinct t.fixture_id) as fixture_count
            from {table_name} t
            join fixtures f on f.fixture_id = t.fixture_id
            group by f.league_id, f.season
            """
        )
    ).mappings().all()
    return {(int(row["league_id"]), str(row["season"])): int(row["fixture_count"]) for row in rows}


def _latest_wyscout_review_path() -> Path | None:
    settings = get_settings()
    unmatched_dir = settings.data_dir / "wyscout" / "unmatched"
    review_paths = sorted(unmatched_dir.glob("review_suggestions_*.csv"))
    if not review_paths:
        return None
    return review_paths[-1]


def _load_wyscout_review_rows(review_path: Path | None) -> list[dict[str, Any]]:
    if review_path is None or not review_path.exists():
        return []

    resolved_lookup_keys = _load_resolved_wyscout_lookup_keys()
    rows: list[dict[str, Any]] = []
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            player_name = (raw_row.get("player_name") or "").strip()
            team_name = (raw_row.get("team_name") or "").strip()
            try:
                lookup_key = build_source_lookup_key(player_name, source_team_name=team_name or None)
            except ValueError:
                lookup_key = None
            if lookup_key is not None and lookup_key in resolved_lookup_keys:
                continue

            candidate_map: dict[int, dict[str, Any]] = {}
            for label, id_key, name_key, score_key in (
                ("Historical roster", "suggested_player_id", "suggested_player_name", "suggested_player_score"),
                ("Global suggestion 1", "global_candidate_1_id", "global_candidate_1_name", "global_candidate_1_score"),
                ("Global suggestion 2", "global_candidate_2_id", "global_candidate_2_name", "global_candidate_2_score"),
                ("Global suggestion 3", "global_candidate_3_id", "global_candidate_3_name", "global_candidate_3_score"),
            ):
                player_id = _parse_optional_int(raw_row.get(id_key))
                if player_id is None:
                    continue
                score = _parse_optional_float(raw_row.get(score_key))
                candidate = {
                    "player_id": player_id,
                    "player_name": (raw_row.get(name_key) or "").strip() or f"Player {player_id}",
                    "score": score,
                    "label": label,
                }
                existing = candidate_map.get(player_id)
                if existing is None or (score or 0.0) > (existing.get("score") or 0.0):
                    candidate_map[player_id] = candidate

            candidate_options = sorted(
                candidate_map.values(),
                key=lambda candidate: (candidate.get("score") or 0.0, candidate["player_name"]),
                reverse=True,
            )
            selection_group = lookup_key or f"{raw_row.get('league_id')}:{raw_row.get('season')}:{player_name}:{team_name}"
            for candidate in candidate_options:
                candidate["selection_token"] = _build_selection_token(
                    player_id=candidate["player_id"],
                    source_player_name=player_name,
                    source_team_name=team_name or None,
                    league_id=int(raw_row["league_id"]),
                    match_score=candidate.get("score"),
                )

            rows.append(
                {
                    "source_file": (raw_row.get("source_file") or "").strip(),
                    "league_id": int(raw_row["league_id"]),
                    "season": str(raw_row["season"]),
                    "player_name": player_name,
                    "team_name": team_name,
                    "current_team": (raw_row.get("current_team") or "").strip(),
                    "suggested_historical_team": (raw_row.get("suggested_historical_team") or "").strip(),
                    "suggested_team_score": _parse_optional_float(raw_row.get("suggested_team_score")),
                    "candidate_options": candidate_options,
                    "selection_group": selection_group,
                }
            )

    rows.sort(
        key=lambda row: (
            row["league_id"],
            row["season"],
            -(row["candidate_options"][0]["score"] if row["candidate_options"] else 0.0),
            row["player_name"],
        )
    )
    return rows


def _load_resolved_wyscout_lookup_keys() -> set[str]:
    with session_scope() as session:
        rows = session.execute(
            text(
                """
                select source_lookup_key
                from source_player_mappings
                where source = 'wyscout'
                """
            )
        ).scalars()
        return {str(row) for row in rows}


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    return float(text_value)


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    return int(float(text_value))


def _required_form_text(form: dict[str, list[str]], key: str) -> str:
    values = form.get(key) or []
    if not values or not str(values[0]).strip():
        raise ValueError(f"Missing required field: {key}")
    return str(values[0]).strip()


def _optional_form_text(form: dict[str, list[str]], key: str) -> str | None:
    values = form.get(key) or []
    if not values:
        return None
    text_value = str(values[0]).strip()
    return text_value or None


def _optional_form_int(form: dict[str, list[str]], key: str) -> int | None:
    value = _optional_form_text(form, key)
    if value is None:
        return None
    return int(value)


def _optional_form_number(form: dict[str, list[str]], key: str) -> int | float | None:
    value = _optional_form_text(form, key)
    if value is None:
        return None
    numeric = float(value.replace(",", ""))
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _financial_score_from_logged_var(value: Any) -> float:
    if value in (None, ""):
        return 50.0
    return float(max(0.0, min(100.0, 50.0 + (50.0 * float(value)))))


def _risk_probability_to_percent(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value) * 100.0


def _build_composite_weight_rows(brief: dict[str, Any]) -> list[dict[str, Any]]:
    effective_weights = _effective_weight_map(brief)

    labels = {
        "role_fit": "Role Fit",
        "current_performance": "Current",
        "upward_projection": "Projection",
        "financial_value": "Financial",
    }
    rows = []
    for key, label in labels.items():
        rows.append({"label": label, "percent": round(float(effective_weights.get(key, 0.0)) * 100.0, 1)})

    return rows


def _build_on_pitch_weight_rows(brief: dict[str, Any]) -> list[dict[str, Any]]:
    return _build_subscore_weight_rows(
        brief,
        score_keys=("role_fit", "current_performance", "upward_projection"),
        labels={
            "role_fit": "Role Fit",
            "current_performance": "Current",
            "upward_projection": "Projection",
        },
        fallback_rows=[
            {"label": "Role Fit", "percent": 33.3},
            {"label": "Current", "percent": 33.3},
            {"label": "Projection", "percent": 33.4},
        ],
    )


def _build_present_on_pitch_weight_rows(brief: dict[str, Any]) -> list[dict[str, Any]]:
    return _build_subscore_weight_rows(
        brief,
        score_keys=("role_fit", "current_performance"),
        labels={
            "role_fit": "Role Fit",
            "current_performance": "Current",
        },
        fallback_rows=[
            {"label": "Role Fit", "percent": 50.0},
            {"label": "Current", "percent": 50.0},
        ],
    )


def _build_upside_on_pitch_weight_rows(brief: dict[str, Any]) -> list[dict[str, Any]]:
    return _build_subscore_weight_rows(
        brief,
        score_keys=("role_fit", "upward_projection"),
        labels={
            "role_fit": "Role Fit",
            "upward_projection": "Projection",
        },
        fallback_rows=[
            {"label": "Role Fit", "percent": 50.0},
            {"label": "Projection", "percent": 50.0},
        ],
    )


def _build_subscore_weight_rows(
    brief: dict[str, Any],
    *,
    score_keys: tuple[str, ...],
    labels: dict[str, str],
    fallback_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    effective_weights = _effective_weight_map(brief)
    total = sum(float(effective_weights.get(key, 0.0) or 0.0) for key in score_keys)
    if total <= 0:
        return fallback_rows

    return [
        {
            "label": labels[key],
            "percent": round((float(effective_weights.get(key, 0.0) or 0.0) / total) * 100.0, 1),
        }
        for key in score_keys
    ]


def _effective_weight_map(brief: dict[str, Any]) -> dict[str, float]:
    archetypes = get_settings().load_json("archetype_weights.json")
    primary_name = str(brief.get("archetype_primary") or "")
    secondary_name = brief.get("archetype_secondary")
    primary = (archetypes.get(primary_name) or {}).get("weights_pct") or {}
    secondary = (archetypes.get(str(secondary_name)) or {}).get("weights_pct") or {} if secondary_name else {}

    base_weights = {}
    for key in ("tactical_fit", "role_fit", "current_performance", "upward_projection", "financial_value"):
        if secondary_name and key in secondary:
            base_weights[key] = ((0.70 * float(primary.get(key, 0.0))) + (0.30 * float(secondary.get(key, 0.0)))) / 100.0
        else:
            base_weights[key] = float(primary.get(key, 0.0)) / 100.0
    return effective_layer_weights(base_weights)


def _decorate_prediction_row(
    row_dict: dict[str, Any],
    league_catalog: dict[int, dict[str, Any]],
    *,
    on_pitch_weights: list[dict[str, Any]],
    present_on_pitch_weights: list[dict[str, Any]],
    upside_on_pitch_weights: list[dict[str, Any]],
) -> dict[str, Any]:
    league_meta = league_catalog.get(int(row_dict["current_league_id"]), {})
    row_dict["league_name"] = league_meta.get("name", f"League {row_dict['current_league_id']}")
    strength_factor = float(league_meta.get("strength_factor") or 1.0)
    row_dict["league_strength_factor"] = strength_factor
    row_dict["board_score"] = composite_to_board_score(row_dict.get("composite_score"))
    row_dict["projection_score"] = projection_score_from_logged_p50(row_dict.get("championship_projection_50th"))
    row_dict["current_score"] = float(row_dict.get("l1_performance_score") or 0.0)
    row_dict["availability_risk_pct"] = _risk_probability_to_percent(row_dict.get("availability_risk_prob"))
    row_dict["financial_score"] = _financial_score_from_logged_var(row_dict.get("var_score"))
    row_dict["action_tier"] = classify_composite_action(row_dict.get("composite_score"))
    row_dict["model_warnings"] = list(row_dict.get("model_warnings") or [])
    row_dict["component_fallbacks"] = dict(row_dict.get("component_fallbacks") or {})
    row_dict["soft_minutes_multiplier"] = _soft_on_pitch_minutes_multiplier(row_dict.get("total_minutes"))
    raw_on_pitch_score = _compute_on_pitch_score(
        role_fit_score=row_dict.get("role_fit_score"),
        current_score=row_dict.get("current_score"),
        projection_score=row_dict.get("projection_score"),
        on_pitch_weights=on_pitch_weights,
        soft_minutes_multiplier=row_dict["soft_minutes_multiplier"],
    )
    raw_present_score = _compute_dual_component_score(
        left_score=row_dict.get("role_fit_score"),
        right_score=row_dict.get("current_score"),
        weight_rows=present_on_pitch_weights,
        left_label="Role Fit",
        right_label="Current",
        soft_minutes_multiplier=row_dict["soft_minutes_multiplier"],
    )
    raw_upside_score = _compute_dual_component_score(
        left_score=row_dict.get("role_fit_score"),
        right_score=row_dict.get("projection_score"),
        weight_rows=upside_on_pitch_weights,
        left_label="Role Fit",
        right_label="Projection",
        soft_minutes_multiplier=row_dict["soft_minutes_multiplier"],
    )
    row_dict["on_pitch_score_raw"] = raw_on_pitch_score
    row_dict["present_on_pitch_score_raw"] = raw_present_score
    row_dict["upside_on_pitch_score_raw"] = raw_upside_score
    row_dict["on_pitch_score"] = _apply_league_strength_factor(raw_on_pitch_score, strength_factor)
    row_dict["present_on_pitch_score"] = _apply_league_strength_factor(raw_present_score, strength_factor)
    row_dict["upside_on_pitch_score"] = _apply_league_strength_factor(raw_upside_score, strength_factor)
    return row_dict


def _compute_on_pitch_score(
    *,
    role_fit_score: Any,
    current_score: Any,
    projection_score: Any,
    on_pitch_weights: list[dict[str, Any]],
    soft_minutes_multiplier: float,
) -> float:
    weight_lookup = {
        "Role Fit": float(next((row["percent"] for row in on_pitch_weights if row["label"] == "Role Fit"), 33.3)) / 100.0,
        "Current": float(next((row["percent"] for row in on_pitch_weights if row["label"] == "Current"), 33.3)) / 100.0,
        "Projection": float(next((row["percent"] for row in on_pitch_weights if row["label"] == "Projection"), 33.4)) / 100.0,
    }
    base_score = (
        (float(role_fit_score or 0.0) * weight_lookup["Role Fit"])
        + (float(current_score or 0.0) * weight_lookup["Current"])
        + (float(projection_score or 0.0) * weight_lookup["Projection"])
    )
    return base_score * soft_minutes_multiplier


def _compute_dual_component_score(
    *,
    left_score: Any,
    right_score: Any,
    weight_rows: list[dict[str, Any]],
    left_label: str,
    right_label: str,
    soft_minutes_multiplier: float,
) -> float:
    weight_lookup = {
        left_label: float(next((row["percent"] for row in weight_rows if row["label"] == left_label), 50.0)) / 100.0,
        right_label: float(next((row["percent"] for row in weight_rows if row["label"] == right_label), 50.0)) / 100.0,
    }
    base_score = (
        (float(left_score or 0.0) * weight_lookup[left_label])
        + (float(right_score or 0.0) * weight_lookup[right_label])
    )
    return base_score * soft_minutes_multiplier


def _apply_league_strength_factor(score: Any, strength_factor: Any) -> float | None:
    if score is None:
        return None
    try:
        score_value = float(score)
        factor_value = float(strength_factor or 1.0)
    except (TypeError, ValueError):
        return None
    return round(score_value * factor_value, 2)


def _soft_on_pitch_minutes_multiplier(total_minutes: Any) -> float:
    base_multiplier = float(minutes_evidence_multiplier(_parse_optional_float(total_minutes)) or 1.0)
    return 0.55 + (0.45 * base_multiplier)


def _build_on_pitch_league_top_fives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        league_id = int(row["current_league_id"])
        grouped.setdefault(league_id, []).append(row)

    league_cards: list[dict[str, Any]] = []
    for league_id, league_rows in grouped.items():
        ranked_rows = sorted(
            league_rows,
            key=lambda row: (
                float(row.get("on_pitch_score") or 0.0),
                float(row.get("projection_score") or 0.0),
                float(row.get("role_fit_score") or 0.0),
            ),
            reverse=True,
        )[:5]
        if not ranked_rows:
            continue
        league_cards.append(
            {
                "league_id": league_id,
                "league_name": ranked_rows[0].get("league_name", f"League {league_id}"),
                "players": ranked_rows,
                "top_on_pitch_score": float(ranked_rows[0].get("on_pitch_score") or 0.0),
            }
        )

    return sorted(
        league_cards,
        key=lambda card: (float(card.get("top_on_pitch_score") or 0.0), card.get("league_name", "")),
        reverse=True,
    )


def _timeline_options() -> list[str]:
    year = datetime.now().year
    return [
        f"summer_{year}",
        f"winter_{year + 1}",
        f"summer_{year + 1}",
    ]


def _brief_report_path(brief_id: int) -> Path:
    return get_settings().project_root / "artifacts" / f"longlist_brief_{brief_id}.html"


def _build_selection_token(
    *,
    player_id: int,
    source_player_name: str,
    source_team_name: str | None,
    league_id: int | None,
    match_score: float | None,
) -> str:
    payload = {
        "player_id": int(player_id),
        "source_player_name": source_player_name,
        "source_team_name": source_team_name,
        "league_id": int(league_id) if league_id is not None else None,
        "match_score": match_score,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _wyscout_source_root() -> Path:
    return get_settings().wyscout_source_dir


def _folder_names_for_leagues(source_root: Path, league_ids: list[int] | None) -> list[str] | None:
    if league_ids is None:
        return None
    if not source_root.exists():
        return None

    league_id_set = {int(league_id) for league_id in league_ids}
    folder_names = [
        path.name
        for path in source_root.iterdir()
        if path.is_dir() and LEAGUE_FOLDER_ALIASES.get(_normalise_folder_name(path.name)) in league_id_set
    ]
    return folder_names or None


def _normalise_folder_name(value: str) -> str:
    return normalise_text(value)
