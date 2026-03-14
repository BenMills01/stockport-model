"""Approximate Governing Body Endorsement estimation."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd

from config import get_settings
from db.read_cache import load_player_match_frame, load_player_row


UK_IRISH_NATIONALITIES = {
    "england",
    "english",
    "scotland",
    "scottish",
    "wales",
    "welsh",
    "northern ireland",
    "northern irish",
    "republic of ireland",
    "irish",
    "ireland",
}

CONTINENTAL_COMPETITION_NAMES = (
    "champions league",
    "europa league",
    "conference league",
    "copa libertadores",
    "copa sudamericana",
)


@lru_cache(maxsize=4096)
def estimate_gbe_score(player_id: int) -> dict[str, Any]:
    """Estimate a player's GBE status from available public-style data.

    This is intentionally approximate and flags missing evidence rather than
    pretending to replace a legal or regulatory determination.
    """

    player_frame = load_player_row(player_id)
    match_frame = load_player_match_frame(player_id).copy()
    return _estimate_gbe_from_frames(
        player_frame=player_frame,
        match_frame=match_frame,
        leagues=get_settings().load_json("leagues.json"),
        today=date.today(),
    )


def _estimate_gbe_from_frames(
    *,
    player_frame: dict[str, Any],
    match_frame: pd.DataFrame,
    leagues: list[dict[str, Any]],
    today: date,
) -> dict[str, Any]:
    nationality = str(player_frame.get("nationality") or "").strip().lower()
    if nationality in UK_IRISH_NATIONALITIES:
        return {
            "points_estimate": 15,
            "status": "green",
            "gaps": [],
            "notes": "UK/Irish player; GBE not required under the current workflow.",
        }

    league_lookup = {
        int(league["league_id"]): {
            "name": league["name"],
            "tier": int(league["tier"]),
        }
        for league in leagues
    }
    gaps: list[str] = []
    points = 0

    if match_frame.empty:
        gaps.append("No tracked match data available to estimate league minutes share.")
        return {
            "points_estimate": 0,
            "status": "red",
            "gaps": gaps,
            "notes": "Estimate unavailable without tracked appearances; manual review required.",
        }

    match_frame = match_frame.copy()
    match_frame["date"] = pd.to_datetime(match_frame["date"])
    valid_seasons = match_frame["season"].dropna()
    if valid_seasons.empty:
        gaps.append("No valid season data in match history; GBE estimate unavailable.")
        return {
            "points_estimate": 0,
            "status": "red",
            "gaps": gaps,
            "notes": "Estimate unavailable without season data; manual review required.",
        }
    latest_season = str(valid_seasons.astype(str).iloc[-1])
    season_matches = match_frame[match_frame["season"].astype(str) == latest_season].copy()
    league_id_mode = season_matches["league_id"].dropna().mode()
    if league_id_mode.empty:
        gaps.append("No valid league_id in season matches; GBE estimate unavailable.")
        return {
            "points_estimate": 0,
            "status": "red",
            "gaps": gaps,
            "notes": "Estimate unavailable without valid league data; manual review required.",
        }
    dominant_league_id = int(league_id_mode.iloc[0])
    league_meta = league_lookup.get(dominant_league_id, {"name": "Unknown", "tier": 5})
    tier = int(league_meta["tier"])

    starts = int((~season_matches["is_substitute"].fillna(False)).sum())
    appearances = int((season_matches["minutes"].fillna(0) > 0).sum())
    start_share = starts / appearances if appearances else 0.0
    points += _domestic_points(tier=tier, start_share=start_share)

    competition_names = " ".join(
        value.lower()
        for value in [
            str(player_frame.get("current_team") or ""),
            str(league_meta["name"]),
        ]
        if value
    )
    continental_matches = 0
    if any(name in competition_names for name in CONTINENTAL_COMPETITION_NAMES):
        continental_matches = appearances
    elif "competition_name" in match_frame.columns:
        continental_matches = int(
            match_frame["competition_name"]
            .fillna("")
            .astype(str)
            .str.lower()
            .apply(lambda text: any(name in text for name in CONTINENTAL_COMPETITION_NAMES))
            .sum()
        )
    else:
        gaps.append("No structured continental competition tracking available.")
    points += _continental_points(continental_matches)

    international_matches = 0
    if "is_international" in match_frame.columns:
        international_matches = int(match_frame["is_international"].fillna(False).sum())
    else:
        national_team_markers = Counter(
            str(team).strip().lower()
            for team in match_frame["team"].dropna().tolist()
            if str(team).strip()
        )
        if nationality and nationality in national_team_markers:
            international_matches = national_team_markers[nationality]
        else:
            gaps.append("No reliable international cap tracking available.")
    points += _international_points(international_matches)

    status = "green" if points >= 15 else "amber" if points >= 10 else "red"
    notes = (
        "Approximate estimate using tracked league level, appearance share, and any available "
        "continental/international indicators. Missing data is surfaced in gaps."
    )
    return {
        "points_estimate": points,
        "status": status,
        "gaps": gaps,
        "notes": notes,
    }


def _domestic_points(*, tier: int, start_share: float) -> int:
    if tier <= 1:
        return 10 if start_share >= 0.7 else 8
    if tier == 2:
        return 8 if start_share >= 0.7 else 6
    if tier == 3:
        return 4 if start_share >= 0.7 else 2
    if tier == 4:
        return 2 if start_share >= 0.5 else 1
    return 0


def _continental_points(continental_matches: int) -> int:
    if continental_matches >= 10:
        return 5
    if continental_matches > 0:
        return 3
    return 0


def _international_points(international_matches: int) -> int:
    if international_matches >= 10:
        return 5
    if international_matches >= 3:
        return 3
    if international_matches > 0:
        return 1
    return 0
