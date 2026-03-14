"""Backfill player bio fields from API-Football player profile pages."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from sqlalchemy import func, select, text

from config import get_settings
from db import session_scope
from db.schema import Fixture
from ingestion.api_football import fetch_api_usage, fetch_player_profiles


LOGGER = logging.getLogger(__name__)


def backfill_player_profiles(
    *,
    league_ids: list[int] | None = None,
    seasons: list[str] | None = None,
    current_season_only: bool = True,
    limit_pairs: int | None = None,
) -> dict[str, Any]:
    """Backfill Player bio fields from API-Football across league-season pairs."""

    pairs = discover_league_season_pairs(
        league_ids=league_ids,
        seasons=seasons,
        current_season_only=current_season_only,
    )
    if limit_pairs is not None:
        pairs = pairs[: max(limit_pairs, 0)]

    before = player_bio_coverage()
    summary: dict[str, Any] = {
        "pairs_requested": len(pairs),
        "league_ids": sorted({league_id for league_id, _season in pairs}) or [],
        "seasons": sorted({season for _league_id, season in pairs}) or [],
        "pairs": [],
        "coverage_before": before,
    }

    try:
        summary["api_usage_before"] = fetch_api_usage()
    except Exception as exc:  # pragma: no cover - live API only
        LOGGER.warning("Could not fetch API usage before player-profile backfill: %s", exc)
        summary["api_usage_before_error"] = str(exc)

    total_rows = 0
    for league_id, season in pairs:
        LOGGER.info("Backfilling API-Football player profiles: league=%s season=%s", league_id, season)
        row_count = fetch_player_profiles(league_id, season)
        total_rows += row_count
        summary["pairs"].append(
            {
                "league_id": league_id,
                "season": season,
                "player_rows_upserted": row_count,
            }
        )

    summary["player_rows_upserted"] = total_rows
    summary["coverage_after"] = player_bio_coverage()
    try:
        summary["api_usage_after"] = fetch_api_usage()
    except Exception as exc:  # pragma: no cover - live API only
        LOGGER.warning("Could not fetch API usage after player-profile backfill: %s", exc)
        summary["api_usage_after_error"] = str(exc)
    return summary


def discover_league_season_pairs(
    *,
    league_ids: list[int] | None = None,
    seasons: list[str] | None = None,
    current_season_only: bool = True,
) -> list[tuple[int, str]]:
    """Return distinct tracked league-season pairs present in the fixtures table."""

    tracked_league_ids = {int(item["league_id"]) for item in get_settings().load_json("leagues.json")}
    selected_league_ids = tracked_league_ids if not league_ids else {int(value) for value in league_ids} & tracked_league_ids
    if not selected_league_ids:
        return []

    selected_seasons = {str(value) for value in seasons} if seasons else None

    with session_scope() as session:
        query = (
            select(Fixture.league_id, Fixture.season)
            .where(Fixture.league_id.in_(sorted(selected_league_ids)))
            .distinct()
        )
        if selected_seasons is not None:
            query = query.where(Fixture.season.in_(sorted(selected_seasons)))
        rows = list(session.execute(query.order_by(Fixture.league_id.asc(), Fixture.season.desc())))

    pairs = [(int(league_id), str(season)) for league_id, season in rows]
    if not current_season_only:
        return pairs

    latest_by_league: dict[int, str] = {}
    for league_id, season in pairs:
        latest_by_league.setdefault(league_id, season)
    return [(league_id, latest_by_league[league_id]) for league_id in sorted(latest_by_league)]


def player_bio_coverage() -> dict[str, Any]:
    """Summarise how much player bio coverage is available in the DB."""

    with session_scope() as session:
        row = session.execute(
            text(
                """
                select
                    count(*) as total_players,
                    count(*) filter (where birth_date is not null) as players_with_birth_date,
                    count(*) filter (where current_age_years is not null) as players_with_current_age,
                    count(*) filter (where birth_date is not null or current_age_years is not null) as players_with_any_age,
                    round(100.0 * count(*) filter (where birth_date is not null) / nullif(count(*), 0), 1) as pct_with_birth_date,
                    round(100.0 * count(*) filter (where current_age_years is not null) / nullif(count(*), 0), 1) as pct_with_current_age,
                    round(100.0 * count(*) filter (where birth_date is not null or current_age_years is not null) / nullif(count(*), 0), 1) as pct_with_any_age
                from players
                """
            )
        ).mappings().one()

        current_season = session.scalar(select(func.max(Fixture.season)))
        if current_season is None:
            return {**dict(row), "current_season": None, "current_season_candidates": None}

        candidate_row = session.execute(
            text(
                """
                with current_season_players as (
                    select distinct mp.player_id
                    from match_performances mp
                    where mp.season = :season
                )
                select
                    count(*) as players_in_current_season,
                    count(*) filter (where pl.birth_date is not null) as current_season_with_birth_date,
                    count(*) filter (where pl.current_age_years is not null) as current_season_with_current_age,
                    count(*) filter (where pl.birth_date is not null or pl.current_age_years is not null) as current_season_with_any_age
                from current_season_players csp
                join players pl on pl.player_id = csp.player_id
                """
            ),
            {"season": str(current_season)},
        ).mappings().one()

    return {
        **dict(row),
        "current_season": str(current_season),
        **dict(candidate_row),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill player DOB/current age from API-Football profiles.")
    parser.add_argument("--league-id", dest="league_ids", action="append", type=int, help="League ID to backfill. Repeat for multiple leagues.")
    parser.add_argument("--season", dest="seasons", action="append", help="Season label to backfill, e.g. 2025. Repeat for multiple seasons.")
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="Backfill every distinct tracked league-season in the fixtures table instead of only the latest season per league.",
    )
    parser.add_argument("--limit-pairs", type=int, default=None, help="Optional cap on the number of league-season pairs processed.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = backfill_player_profiles(
        league_ids=args.league_ids,
        seasons=args.seasons,
        current_season_only=not args.all_seasons,
        limit_pairs=args.limit_pairs,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
