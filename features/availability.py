"""Availability and injury-history features."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd
from sqlalchemy import or_, select

from db.read_cache import load_player_injury_frame, load_player_match_frame, load_player_sidelined_frame
from db.schema import Fixture, Injury, MatchEvent, MatchPerformance, Sidelined
from db.session import session_scope


MUSCLE_TERMS = ("muscle", "hamstring", "thigh", "calf", "groin")


@lru_cache(maxsize=4096)
def compute_availability_features(player_id: int) -> dict[str, Any]:
    """Compute injury and appearance continuity features."""

    match_frame = load_player_match_frame(player_id).copy()
    sidelined_frame = load_player_sidelined_frame(player_id).copy()
    injury_frame = load_player_injury_frame(player_id).copy()

    fixture_frame = _load_relevant_fixtures(match_frame)
    event_frame = _load_relevant_events(match_frame, player_id)
    return _compute_availability_features_from_frames(
        match_frame=match_frame,
        fixture_frame=fixture_frame,
        sidelined_frame=sidelined_frame,
        injury_frame=injury_frame,
        event_frame=event_frame,
        today=date.today(),
    )


def _compute_availability_features_from_frames(
    *,
    match_frame: pd.DataFrame,
    fixture_frame: pd.DataFrame,
    sidelined_frame: pd.DataFrame,
    injury_frame: pd.DataFrame,
    event_frame: pd.DataFrame,
    today: date,
) -> dict[str, Any]:
    if match_frame.empty:
        return {
            "availability_rate_season": None,
            "availability_rate_3yr": None,
            "injury_frequency_3yr": 0,
            "avg_injury_duration": None,
            "max_injury_duration": None,
            "muscle_injury_count": 0,
            "recurrence_rate": None,
            "days_since_last_injury": None,
            "minutes_continuity": 0,
            "subbed_off_rate": None,
        }

    match_frame = match_frame.copy()
    match_frame["date"] = pd.to_datetime(match_frame["date"])
    current_season = str(match_frame["season"].iloc[-1])
    seasons = list(dict.fromkeys(match_frame["season"].astype(str).tolist()))
    recent_seasons = seasons[-3:]

    availability_rate_season = _season_availability_rate(
        match_frame,
        fixture_frame,
        season=current_season,
    )
    three_year_rates = [
        rate
        for season in recent_seasons
        if (rate := _season_availability_rate(match_frame, fixture_frame, season=season)) is not None
    ]
    availability_rate_3yr = (
        float(sum(three_year_rates) / len(three_year_rates)) if three_year_rates else None
    )

    sidelined_frame = sidelined_frame.copy()
    for column, default in {
        "start_date": pd.NaT,
        "end_date": pd.NaT,
        "type": None,
    }.items():
        if column not in sidelined_frame.columns:
            sidelined_frame[column] = default
    if not sidelined_frame.empty:
        sidelined_frame["start_date"] = pd.to_datetime(sidelined_frame["start_date"])
        sidelined_frame["end_date"] = pd.to_datetime(sidelined_frame["end_date"])
    injury_frame = injury_frame.copy()
    if "date" not in injury_frame.columns:
        injury_frame["date"] = pd.NaT
    if not injury_frame.empty:
        injury_frame["date"] = pd.to_datetime(injury_frame["date"])

    cutoff = pd.Timestamp(today - timedelta(days=365 * 3))
    recent_sidelined = sidelined_frame[sidelined_frame["start_date"].fillna(cutoff) >= cutoff].copy()
    recent_injuries = injury_frame[injury_frame["date"].fillna(cutoff) >= cutoff].copy()

    injury_frequency_3yr = int(len(recent_sidelined.index))
    durations = _injury_durations(recent_sidelined, today)
    avg_injury_duration = float(sum(durations) / len(durations)) if durations else None
    max_injury_duration = int(max(durations)) if durations else None
    muscle_injury_count = int(
        recent_sidelined["type"]
        .fillna("")
        .astype(str)
        .str.lower()
        .apply(lambda text: any(term in text for term in MUSCLE_TERMS))
        .sum()
    ) if not recent_sidelined.empty else 0

    recurrence_rate = _recurrence_rate(recent_sidelined)
    days_since_last_injury = _days_since_last_injury(recent_sidelined, recent_injuries, today)
    minutes_continuity = _minutes_continuity(match_frame)
    subbed_off_rate = _subbed_off_rate(match_frame, event_frame)

    return {
        "availability_rate_season": availability_rate_season,
        "availability_rate_3yr": availability_rate_3yr,
        "injury_frequency_3yr": injury_frequency_3yr,
        "avg_injury_duration": avg_injury_duration,
        "max_injury_duration": max_injury_duration,
        "muscle_injury_count": muscle_injury_count,
        "recurrence_rate": recurrence_rate,
        "days_since_last_injury": days_since_last_injury,
        "minutes_continuity": minutes_continuity,
        "subbed_off_rate": subbed_off_rate,
    }


def _load_relevant_fixtures(match_frame: pd.DataFrame) -> pd.DataFrame:
    if match_frame.empty:
        return pd.DataFrame()

    latest_season = str(match_frame["season"].astype(str).iloc[-1])
    current_team = _current_team_for_season(match_frame, latest_season)
    if current_team is None:
        return pd.DataFrame()

    seasons = set(match_frame["season"].astype(str))
    with session_scope() as session:
        fixtures = list(
            session.scalars(
                select(Fixture).where(
                    Fixture.season.in_(seasons),
                    or_(Fixture.home_team == current_team, Fixture.away_team == current_team),
                )
            )
        )
    return pd.DataFrame([_row_to_dict(row, Fixture) for row in fixtures])


def _load_relevant_events(match_frame: pd.DataFrame, player_id: int) -> pd.DataFrame:
    if match_frame.empty:
        return pd.DataFrame()

    fixture_ids = [int(fixture_id) for fixture_id in match_frame["fixture_id"].dropna().unique().tolist()]
    if not fixture_ids:
        return pd.DataFrame()
    with session_scope() as session:
        events = list(
            session.scalars(
                select(MatchEvent).where(
                    MatchEvent.fixture_id.in_(fixture_ids),
                    MatchEvent.player_id == player_id,
                )
            )
        )
    return pd.DataFrame([_row_to_dict(row, MatchEvent) for row in events])


def _season_availability_rate(
    match_frame: pd.DataFrame,
    fixture_frame: pd.DataFrame,
    *,
    season: str,
) -> float | None:
    season_matches = match_frame[match_frame["season"].astype(str) == str(season)].copy()
    if season_matches.empty:
        return None

    starts = season_matches[~season_matches["is_substitute"].fillna(False)]
    total_matchdays = _team_matchdays_for_season(season_matches, fixture_frame, str(season))
    if total_matchdays == 0:
        return None
    return float(len(starts.index) / total_matchdays)


def _team_matchdays_for_season(
    season_matches: pd.DataFrame,
    fixture_frame: pd.DataFrame,
    season: str,
) -> int:
    current_team = _current_team_for_season(season_matches, season)
    if current_team is None:
        return int(season_matches["fixture_id"].nunique())
    if fixture_frame.empty:
        return int(season_matches["fixture_id"].nunique())
    relevant = fixture_frame[
        (fixture_frame["season"].astype(str) == season)
        & (
            (fixture_frame["home_team"] == current_team)
            | (fixture_frame["away_team"] == current_team)
        )
    ]
    return int(relevant["fixture_id"].nunique()) if not relevant.empty else int(season_matches["fixture_id"].nunique())


def _current_team_for_season(match_frame: pd.DataFrame, season: str) -> str | None:
    season_matches = match_frame[match_frame["season"].astype(str) == season]
    if season_matches.empty:
        return None
    modes = season_matches["team"].dropna().mode()
    return str(modes.iloc[0]) if not modes.empty else None


def _injury_durations(sidelined_frame: pd.DataFrame, today: date) -> list[int]:
    if sidelined_frame.empty:
        return []
    end_dates = sidelined_frame["end_date"].fillna(pd.Timestamp(today))
    start_dates = sidelined_frame["start_date"].fillna(end_dates)
    durations = (end_dates - start_dates).dt.days.clip(lower=0)
    return [int(value) for value in durations.tolist()]


def _recurrence_rate(sidelined_frame: pd.DataFrame) -> float | None:
    if sidelined_frame.empty:
        return None
    labels = [
        str(value).strip().lower()
        for value in sidelined_frame["type"].fillna("").tolist()
        if str(value).strip()
    ]
    if not labels:
        return None
    counts = Counter(labels)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return float(repeated / len(labels))


def _days_since_last_injury(
    sidelined_frame: pd.DataFrame,
    injury_frame: pd.DataFrame,
    today: date,
) -> int | None:
    candidates: list[pd.Timestamp] = []
    if not sidelined_frame.empty and not sidelined_frame["end_date"].dropna().empty:
        candidates.append(sidelined_frame["end_date"].dropna().max())
    if not injury_frame.empty and not injury_frame["date"].dropna().empty:
        candidates.append(injury_frame["date"].dropna().max())
    if not candidates:
        return None
    most_recent = max(candidates)
    return int((pd.Timestamp(today) - most_recent).days)


def _minutes_continuity(match_frame: pd.DataFrame) -> int:
    continuity = 0
    for minutes in reversed(match_frame.sort_values("date")["minutes"].fillna(0).tolist()):
        if float(minutes) >= 60:
            continuity += 1
        else:
            break
    return continuity


def _subbed_off_rate(match_frame: pd.DataFrame, event_frame: pd.DataFrame) -> float | None:
    starts = match_frame[~match_frame["is_substitute"].fillna(False)].copy()
    if starts.empty:
        return None
    if event_frame.empty:
        return 0.0

    event_frame = event_frame.copy()
    event_frame["event_type"] = event_frame["event_type"].fillna("").astype(str).str.lower()
    event_frame["event_detail"] = event_frame["event_detail"].fillna("").astype(str).str.lower()
    event_frame["time_elapsed"] = pd.to_numeric(event_frame["time_elapsed"], errors="coerce")
    subbed_off_fixtures = set(
        event_frame[
            (
                event_frame["event_type"].str.contains("subst")
                | event_frame["event_detail"].str.contains("subst")
                | event_frame["event_detail"].str.contains("substitution")
            )
            & (event_frame["time_elapsed"].fillna(999) < 80)
        ]["fixture_id"].tolist()
    )
    return float(len(subbed_off_fixtures.intersection(set(starts["fixture_id"]))) / len(starts.index))


def _row_to_dict(row: Any, model: Any) -> dict[str, Any]:
    return {
        column.name: getattr(row, column.name)
        for column in model.__table__.columns
    }
