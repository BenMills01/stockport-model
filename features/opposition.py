"""Opposition-quality split features."""

from __future__ import annotations

from typing import Any

import pandas as pd

from db.read_cache import load_player_match_frame, load_standings_frame_for_leagues
from features.per90 import _compute_per90_frame


def compute_opposition_splits(player_id: int, season: str) -> dict[str, dict[str, float | None]]:
    """Compute per-tier opposition splits for a player's per-90 metrics."""

    match_frame = load_player_match_frame(player_id, season).copy()
    league_ids_key = tuple(sorted({int(value) for value in match_frame.get("league_id", pd.Series(dtype=int)).dropna().tolist()}))
    standings_frame = load_standings_frame_for_leagues(league_ids_key).copy()
    return _compute_opposition_splits_from_frames(match_frame, standings_frame)


def _compute_opposition_splits_from_frames(
    match_frame: pd.DataFrame,
    standings_frame: pd.DataFrame,
) -> dict[str, dict[str, float | None]]:
    if match_frame.empty:
        return {}

    per90 = _compute_per90_frame(match_frame)
    if per90.empty:
        return {}

    standings = standings_frame.copy()
    if not standings.empty:
        standings["date"] = pd.to_datetime(standings["date"], utc=True).dt.tz_convert(None)

    per90["match_date"] = pd.to_datetime(per90["date"], utc=True).dt.tz_convert(None)
    per90["opponent"] = per90.apply(
        lambda row: row["away_team"] if row["is_home"] else row["home_team"],
        axis=1,
    )
    per90["tier"] = per90.apply(lambda row: _lookup_tier(row, standings), axis=1)

    metric_columns = [column for column in per90.columns if column.endswith("_per90")]
    output: dict[str, dict[str, float | None]] = {}
    for column in metric_columns:
        output[column.removesuffix("_per90")] = {}
        for tier in ("tier1", "tier2", "tier3", "tier4"):
            tier_values = pd.to_numeric(
                per90.loc[per90["tier"] == tier, column],
                errors="coerce",
            ).dropna()
            output[column.removesuffix("_per90")][tier] = (
                float(tier_values.mean()) if not tier_values.empty else None
            )
    return output


def _lookup_tier(row: pd.Series, standings_frame: pd.DataFrame) -> str | None:
    if standings_frame.empty:
        return None

    candidates = standings_frame[
        (standings_frame["league_id"] == row["league_id"])
        & (standings_frame["team_name"] == row["opponent"])
        & (standings_frame["date"] <= row["match_date"])
    ].sort_values("date", ascending=False)
    if candidates.empty:
        return None

    position = int(candidates.iloc[0]["position"])

    # Derive league size from the most recent snapshot for this league so that
    # tier boundaries scale correctly for 16-, 20-, and 24-team competitions.
    league_snapshot = standings_frame[
        (standings_frame["league_id"] == row["league_id"])
        & (standings_frame["date"] <= row["match_date"])
    ]
    if league_snapshot.empty:
        league_size = 20  # safe default
    else:
        latest_date = league_snapshot["date"].max()
        league_size = int(
            league_snapshot[league_snapshot["date"] == latest_date]["team_name"].nunique()
        )
        if league_size < 4:
            league_size = 20  # guard against sparse snapshot data

    quarter = max(1, league_size // 4)
    if position <= quarter:
        return "tier1"
    if position <= quarter * 2:
        return "tier2"
    if position <= quarter * 3:
        return "tier3"
    return "tier4"
