"""League intensity normalisation and role-relative percentiles."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from sqlalchemy import and_, or_, select

from config import get_settings
from db.schema import MatchPerformance, PlayerRole
from db.session import session_scope
from features.per90 import _compute_per90_frame


def compute_league_percentile(player_id: int, season: str, role: str) -> dict[str, Any]:
    """Compute role-relative percentiles and Championship-referenced absolutes."""

    return _compute_league_percentile_cached(player_id=player_id, season=season, role=role)


def _compute_league_percentile_from_frames(
    *,
    player_id: int,
    season: str,
    role: str,
    match_frame: pd.DataFrame,
    role_frame: pd.DataFrame,
) -> dict[str, Any]:
    if match_frame.empty:
        return {
            "player_id": player_id,
            "season": season,
            "role": role,
            "league_id": None,
            "reference_league": get_settings().reference_league_name,
            "percentiles": {},
            "league_adjusted_absolute": {},
            "league_role_average": {},
            "reference_role_average": {},
        }

    season_matches = match_frame[match_frame["season"] == season].copy()
    per90 = _compute_per90_frame(season_matches)
    if per90.empty:
        return {
            "player_id": player_id,
            "season": season,
            "role": role,
            "league_id": None,
            "reference_league": get_settings().reference_league_name,
            "percentiles": {},
            "league_adjusted_absolute": {},
            "league_role_average": {},
            "reference_role_average": {},
        }

    season_averages = _season_player_averages(per90)
    target_row = season_averages[season_averages["player_id"] == player_id]
    if target_row.empty:
        return {
            "player_id": player_id,
            "season": season,
            "role": role,
            "league_id": None,
            "reference_league": get_settings().reference_league_name,
            "percentiles": {},
            "league_adjusted_absolute": {},
            "league_role_average": {},
            "reference_role_average": {},
        }

    target_row = target_row.iloc[0]
    player_league_id = int(target_row["league_id"])

    peer_ids = set(role_frame["player_id"]) if not role_frame.empty else {player_id}
    league_peer_frame = season_averages[
        (season_averages["league_id"] == player_league_id)
        & (season_averages["player_id"].isin(peer_ids))
    ].copy()
    if league_peer_frame.empty:
        league_peer_frame = season_averages[season_averages["league_id"] == player_league_id].copy()

    reference_league_ids = _reference_league_ids()
    reference_peer_frame = season_averages[
        (season_averages["league_id"].isin(reference_league_ids))
        & (season_averages["player_id"].isin(peer_ids))
    ].copy()
    if reference_peer_frame.empty:
        reference_peer_frame = league_peer_frame.copy()

    metric_columns = [column for column in season_averages.columns if column.endswith("_per90")]
    percentiles: dict[str, float | None] = {}
    league_adjusted_absolute: dict[str, float | None] = {}
    league_role_average: dict[str, float | None] = {}
    reference_role_average: dict[str, float | None] = {}

    for metric in metric_columns:
        player_value = _coerce_float(target_row.get(metric))
        peer_values = pd.to_numeric(league_peer_frame[metric], errors="coerce").dropna()
        reference_values = pd.to_numeric(reference_peer_frame[metric], errors="coerce").dropna()
        league_avg = float(peer_values.mean()) if not peer_values.empty else None
        reference_avg = float(reference_values.mean()) if not reference_values.empty else None

        percentiles[metric.removesuffix("_per90")] = _percentile(player_value, peer_values)
        league_role_average[metric.removesuffix("_per90")] = league_avg
        reference_role_average[metric.removesuffix("_per90")] = reference_avg
        league_adjusted_absolute[metric.removesuffix("_per90")] = _league_adjusted_absolute(
            player_value=player_value,
            league_average=league_avg,
            reference_average=reference_avg,
        )

    return {
        "player_id": player_id,
        "season": season,
        "role": role,
        "league_id": player_league_id,
        "reference_league": get_settings().reference_league_name,
        "percentiles": percentiles,
        "league_adjusted_absolute": league_adjusted_absolute,
        "league_role_average": league_role_average,
        "reference_role_average": reference_role_average,
    }


@lru_cache(maxsize=16384)
def _compute_league_percentile_cached(player_id: int, season: str, role: str) -> dict[str, Any]:
    season_averages = _season_player_averages_for_season(season)
    if season_averages.empty:
        return {
            "player_id": player_id,
            "season": season,
            "role": role,
            "league_id": None,
            "reference_league": get_settings().reference_league_name,
            "percentiles": {},
            "league_adjusted_absolute": {},
            "league_role_average": {},
            "reference_role_average": {},
        }

    target_row = season_averages[season_averages["player_id"] == player_id]
    if target_row.empty:
        return {
            "player_id": player_id,
            "season": season,
            "role": role,
            "league_id": None,
            "reference_league": get_settings().reference_league_name,
            "percentiles": {},
            "league_adjusted_absolute": {},
            "league_role_average": {},
            "reference_role_average": {},
        }

    target_row = target_row.iloc[0]
    player_league_id = int(target_row["league_id"])
    peer_ids = set(_role_peer_ids_for_season_role(season, role)) or {player_id}
    league_peer_frame = season_averages[
        (season_averages["league_id"] == player_league_id)
        & (season_averages["player_id"].isin(peer_ids))
    ].copy()
    if league_peer_frame.empty:
        league_peer_frame = season_averages[season_averages["league_id"] == player_league_id].copy()

    reference_league_ids = _reference_league_ids()
    reference_peer_frame = season_averages[
        (season_averages["league_id"].isin(reference_league_ids))
        & (season_averages["player_id"].isin(peer_ids))
    ].copy()
    if reference_peer_frame.empty:
        reference_peer_frame = league_peer_frame.copy()

    metric_columns = [column for column in season_averages.columns if column.endswith("_per90")]
    percentiles: dict[str, float | None] = {}
    league_adjusted_absolute: dict[str, float | None] = {}
    league_role_average: dict[str, float | None] = {}
    reference_role_average: dict[str, float | None] = {}

    for metric in metric_columns:
        player_value = _coerce_float(target_row.get(metric))
        peer_values = pd.to_numeric(league_peer_frame[metric], errors="coerce").dropna()
        reference_values = pd.to_numeric(reference_peer_frame[metric], errors="coerce").dropna()
        league_avg = float(peer_values.mean()) if not peer_values.empty else None
        reference_avg = float(reference_values.mean()) if not reference_values.empty else None

        key = metric.removesuffix("_per90")
        percentiles[key] = _percentile(player_value, peer_values)
        league_role_average[key] = league_avg
        reference_role_average[key] = reference_avg
        league_adjusted_absolute[key] = _league_adjusted_absolute(
            player_value=player_value,
            league_average=league_avg,
            reference_average=reference_avg,
        )

    return {
        "player_id": player_id,
        "season": season,
        "role": role,
        "league_id": player_league_id,
        "reference_league": get_settings().reference_league_name,
        "percentiles": percentiles,
        "league_adjusted_absolute": league_adjusted_absolute,
        "league_role_average": league_role_average,
        "reference_role_average": reference_role_average,
    }


@lru_cache(maxsize=16)
def _season_player_averages_for_season(season: str) -> pd.DataFrame:
    with session_scope() as session:
        match_rows = list(
            session.scalars(
                select(MatchPerformance).where(MatchPerformance.season == season)
            )
        )
    match_frame = pd.DataFrame([_match_row_to_dict(row) for row in match_rows])
    if match_frame.empty:
        return pd.DataFrame()
    per90 = _compute_per90_frame(match_frame)
    if per90.empty:
        return pd.DataFrame()
    return _season_player_averages(per90)


@lru_cache(maxsize=128)
def _role_peer_ids_for_season_role(season: str, role: str) -> tuple[int, ...]:
    with session_scope() as session:
        role_rows = list(
            session.scalars(
                select(PlayerRole).where(
                    and_(
                        PlayerRole.season == season,
                        or_(
                            PlayerRole.primary_role == role,
                            PlayerRole.secondary_role == role,
                        ),
                    )
                )
            )
        )
    return tuple(sorted({int(row.player_id) for row in role_rows}))


def _season_player_averages(per90_frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in per90_frame.columns if column.endswith("_per90")]
    aggregations: dict[str, Any] = {metric: "mean" for metric in metric_columns}
    aggregations["league_id"] = _mode_value
    grouped = per90_frame.groupby("player_id", as_index=False).agg(aggregations)
    return grouped


@lru_cache(maxsize=1)
def _reference_league_ids() -> set[int]:
    settings = get_settings()
    leagues = settings.load_json("leagues.json")
    return {
        int(league["league_id"])
        for league in leagues
        if league["name"] == settings.reference_league_name
    }


def _percentile(player_value: float | None, peer_values: pd.Series) -> float | None:
    if player_value is None or peer_values.empty:
        return None
    return float((peer_values <= player_value).mean() * 100.0)


def _league_adjusted_absolute(
    *,
    player_value: float | None,
    league_average: float | None,
    reference_average: float | None,
) -> float | None:
    if (
        player_value is None
        or league_average is None
        or reference_average is None
        or league_average == 0
    ):
        return None
    return float((player_value / league_average) * reference_average)


def _mode_value(series: pd.Series) -> Any:
    modes = series.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return series.dropna().iloc[0] if not series.dropna().empty else None


def _coerce_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _match_row_to_dict(row: MatchPerformance) -> dict[str, Any]:
    return {
        column.name: getattr(row, column.name)
        for column in MatchPerformance.__table__.columns
    }


def _role_row_to_dict(row: PlayerRole) -> dict[str, Any]:
    return {
        column.name: getattr(row, column.name)
        for column in PlayerRole.__table__.columns
    }
