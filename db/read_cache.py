"""Cached read helpers for analytics-heavy workflows."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from sqlalchemy import desc, select

from db.schema import Fixture, Injury, Lineup, MarketValue, MatchEvent, MatchPerformance, Player, PlayerRole
from db.schema import Sidelined, StandingsSnapshot, Transfer
from db.session import session_scope


@lru_cache(maxsize=4096)
def load_player_row(player_id: int) -> dict[str, Any]:
    with session_scope() as session:
        player = session.get(Player, player_id)
    return _row_to_dict(player, Player) if player is not None else {}


@lru_cache(maxsize=4096)
def load_latest_market_value_row(player_id: int) -> dict[str, Any]:
    with session_scope() as session:
        market_value = session.scalar(
            select(MarketValue)
            .where(MarketValue.player_id == player_id)
            .order_by(desc(MarketValue.date))
        )
    return _row_to_dict(market_value, MarketValue) if market_value is not None else {}


@lru_cache(maxsize=8192)
def load_player_role_row(player_id: int, season: str | None = None) -> dict[str, Any]:
    with session_scope() as session:
        query = select(PlayerRole).where(PlayerRole.player_id == player_id)
        if season is not None:
            query = query.where(PlayerRole.season == season)
        role_row = session.scalar(query.order_by(desc(PlayerRole.season)))
    return _row_to_dict(role_row, PlayerRole) if role_row is not None else {}


@lru_cache(maxsize=4096)
def load_player_match_frame(player_id: int, season: str | None = None) -> pd.DataFrame:
    with session_scope() as session:
        query = select(MatchPerformance).where(MatchPerformance.player_id == player_id)
        if season is not None:
            query = query.where(MatchPerformance.season == season)
        rows = list(session.scalars(query.order_by(MatchPerformance.date.asc())))
    return pd.DataFrame([_row_to_dict(row, MatchPerformance) for row in rows])


@lru_cache(maxsize=4096)
def load_player_lineup_frame(player_id: int, season: str | None = None) -> pd.DataFrame:
    with session_scope() as session:
        query = (
            select(Lineup, Fixture.season, Fixture.date)
            .join(Fixture, Fixture.fixture_id == Lineup.fixture_id)
            .where(Lineup.player_id == player_id)
        )
        if season is not None:
            query = query.where(Fixture.season == season)
        rows = list(session.execute(query.order_by(Fixture.date.asc())))

    return pd.DataFrame(
        [
            {
                **_row_to_dict(lineup, Lineup),
                "season": fixture_season,
                "date": fixture_date,
            }
            for lineup, fixture_season, fixture_date in rows
        ]
    )


@lru_cache(maxsize=4096)
def load_player_transfer_frame(player_id: int) -> pd.DataFrame:
    with session_scope() as session:
        rows = list(
            session.scalars(
                select(Transfer)
                .where(Transfer.player_id == player_id)
                .order_by(Transfer.date.asc())
            )
        )
    return pd.DataFrame([_row_to_dict(row, Transfer) for row in rows])


@lru_cache(maxsize=4096)
def load_player_sidelined_frame(player_id: int) -> pd.DataFrame:
    with session_scope() as session:
        rows = list(
            session.scalars(
                select(Sidelined)
                .where(Sidelined.player_id == player_id)
                .order_by(Sidelined.start_date.asc())
            )
        )
    return pd.DataFrame([_row_to_dict(row, Sidelined) for row in rows])


@lru_cache(maxsize=4096)
def load_player_injury_frame(player_id: int) -> pd.DataFrame:
    with session_scope() as session:
        rows = list(
            session.scalars(
                select(Injury)
                .where(Injury.player_id == player_id)
                .order_by(Injury.date.asc())
            )
        )
    return pd.DataFrame([_row_to_dict(row, Injury) for row in rows])


@lru_cache(maxsize=8192)
def load_player_event_frame(player_id: int, fixture_ids_key: tuple[int, ...]) -> pd.DataFrame:
    if not fixture_ids_key:
        return pd.DataFrame()
    with session_scope() as session:
        rows = list(
            session.scalars(
                select(MatchEvent).where(
                    MatchEvent.fixture_id.in_(fixture_ids_key),
                    MatchEvent.player_id == player_id,
                )
            )
        )
    return pd.DataFrame([_row_to_dict(row, MatchEvent) for row in rows])


@lru_cache(maxsize=512)
def load_standings_frame_for_leagues(league_ids_key: tuple[int, ...]) -> pd.DataFrame:
    if not league_ids_key:
        return pd.DataFrame()
    with session_scope() as session:
        rows = list(
            session.scalars(
                select(StandingsSnapshot).where(StandingsSnapshot.league_id.in_(league_ids_key))
            )
        )
    return pd.DataFrame([_row_to_dict(row, StandingsSnapshot) for row in rows])


def clear_read_caches() -> None:
    """Clear cached DB reads after bulk data updates."""

    load_player_row.cache_clear()
    load_latest_market_value_row.cache_clear()
    load_player_role_row.cache_clear()
    load_player_match_frame.cache_clear()
    load_player_lineup_frame.cache_clear()
    load_player_transfer_frame.cache_clear()
    load_player_sidelined_frame.cache_clear()
    load_player_injury_frame.cache_clear()
    load_player_event_frame.cache_clear()
    load_standings_frame_for_leagues.cache_clear()


def _row_to_dict(row: Any, model: Any) -> dict[str, Any]:
    if row is None:
        return {}
    return {
        column.name: getattr(row, column.name)
        for column in model.__table__.columns
    }
