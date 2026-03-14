"""Seed Stockport tables from the football_model raw player-stats CSV."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, time
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config import get_settings
from db.schema import Fixture, MatchPerformance, Player
from ingestion.common import upsert_rows


LOGGER = logging.getLogger(__name__)
DEFAULT_LEGACY_CSV_PATH = Path.home() / "football_model" / "data" / "raw" / "raw_player_stats.csv"
MATCH_PERFORMANCE_UPSERT_BATCH_SIZE = 1000
LEGACY_REQUIRED_COLUMNS = {
    "league_id",
    "fixture_id",
    "season",
    "date",
    "home_team",
    "away_team",
    "referee",
    "player_id",
    "player_name",
    "team",
    "is_home",
    "position",
    "minutes",
    "rating",
    "fouls_committed",
    "fouls_drawn",
    "shots_total",
    "shots_on",
    "tackles",
    "saves",
    "offsides",
    "yellow_cards",
    "red_cards",
    "passes",
    "key_passes",
    "duels_total",
    "duels_won",
    "dribbles",
}


def import_legacy_raw_player_stats(
    csv_path: str | Path | None = None,
    *,
    chunk_size: int = 10000,
    batch_size: int = 5000,
    league_ids: list[int] | None = None,
    seasons: list[int] | None = None,
    tracked_only: bool = False,
) -> dict[str, Any]:
    """Import football_model raw player stats into Stockport Postgres tables."""

    path = Path(csv_path) if csv_path is not None else DEFAULT_LEGACY_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"Legacy raw-player-stats CSV not found: {path}")

    tracked_league_ids = _tracked_league_ids() if tracked_only else None
    requested_league_ids = {int(league_id) for league_id in league_ids} if league_ids else None
    requested_seasons = {int(season) for season in seasons} if seasons else None

    player_latest: dict[int, dict[str, Any]] = {}
    fixtures_by_id: dict[int, dict[str, Any]] = {}
    summary = {
        "csv_path": str(path),
        "chunk_size": chunk_size,
        "batch_size": batch_size,
        "tracked_only": tracked_only,
        "league_ids": sorted(requested_league_ids) if requested_league_ids else None,
        "seasons": sorted(requested_seasons) if requested_seasons else None,
        "chunks_processed": 0,
        "rows_read": 0,
        "rows_after_filters": 0,
        "rows_skipped_missing_keys": 0,
        "match_performance_rows_upserted": 0,
        "fixture_rows_upserted": 0,
        "player_rows_upserted": 0,
    }

    for chunk_index, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size), start=1):
        summary["chunks_processed"] = chunk_index
        summary["rows_read"] += len(chunk.index)
        _validate_legacy_columns(chunk.columns)

        filtered = _filter_chunk(
            chunk,
            tracked_league_ids=tracked_league_ids,
            requested_league_ids=requested_league_ids,
            requested_seasons=requested_seasons,
        )
        if filtered.empty:
            continue

        filtered = filtered.drop_duplicates(subset=["fixture_id", "player_id"], keep="last")
        filtered = filtered.where(pd.notnull(filtered), None)
        summary["rows_after_filters"] += len(filtered.index)

        match_rows: list[dict[str, Any]] = []
        for record in filtered.to_dict(orient="records"):
            parsed = _parse_common_fields(record)
            if parsed is None:
                summary["rows_skipped_missing_keys"] += 1
                continue

            fixtures_by_id[parsed["fixture_id"]] = _build_fixture_row(parsed)
            _update_player_latest(player_latest, parsed)
            match_rows.append(_build_match_performance_row(parsed))

        if match_rows:
            summary["match_performance_rows_upserted"] += _upsert_batched(
                MatchPerformance,
                match_rows,
                ["fixture_id", "player_id"],
                batch_size=MATCH_PERFORMANCE_UPSERT_BATCH_SIZE,
            )

        LOGGER.info(
            "Legacy import chunk %s complete: read=%s filtered=%s match_rows=%s",
            chunk_index,
            len(chunk.index),
            len(filtered.index),
            len(match_rows),
        )

    summary["fixture_rows_upserted"] = _upsert_batched(
        Fixture,
        list(fixtures_by_id.values()),
        ["fixture_id"],
        batch_size=batch_size,
    )
    summary["player_rows_upserted"] = _upsert_batched(
        Player,
        [_strip_player_sort_key(row) for row in player_latest.values()],
        ["player_id"],
        batch_size=batch_size,
    )
    return summary


def _validate_legacy_columns(columns: Any) -> None:
    missing = sorted(LEGACY_REQUIRED_COLUMNS - set(columns))
    if missing:
        raise ValueError(f"Legacy raw-player-stats CSV is missing required columns: {missing}")


def _filter_chunk(
    chunk: pd.DataFrame,
    *,
    tracked_league_ids: set[int] | None,
    requested_league_ids: set[int] | None,
    requested_seasons: set[int] | None,
) -> pd.DataFrame:
    filtered = chunk

    if tracked_league_ids is not None:
        filtered = filtered[filtered["league_id"].astype("Int64").isin(sorted(tracked_league_ids))]
    if requested_league_ids is not None:
        filtered = filtered[filtered["league_id"].astype("Int64").isin(sorted(requested_league_ids))]
    if requested_seasons is not None:
        filtered = filtered[filtered["season"].astype("Int64").isin(sorted(requested_seasons))]

    return filtered.copy()


def _parse_common_fields(record: dict[str, Any]) -> dict[str, Any] | None:
    fixture_id = _coerce_int(record.get("fixture_id"))
    player_id = _coerce_int(record.get("player_id"))
    league_id = _coerce_int(record.get("league_id"))
    match_datetime = _parse_match_datetime(record.get("date"))
    season = _coerce_int(record.get("season"))
    home_team = _clean_text(record.get("home_team"))
    away_team = _clean_text(record.get("away_team"))
    player_name = _clean_text(record.get("player_name"))
    team = _clean_text(record.get("team"))

    required = (fixture_id, player_id, league_id, match_datetime, season)
    if any(value is None for value in required):
        return None
    if fixture_id <= 0 or player_id <= 0 or league_id <= 0:
        return None
    if not all((home_team, away_team, player_name, team)):
        return None

    is_home = _coerce_bool(record.get("is_home"))
    if is_home is None:
        is_home = team == home_team

    return {
        "fixture_id": fixture_id,
        "player_id": player_id,
        "league_id": league_id,
        "season": str(season),
        "date": match_datetime,
        "home_team": home_team,
        "away_team": away_team,
        "referee": _clean_text(record.get("referee")),
        "player_name": player_name,
        "team": team,
        "is_home": is_home,
        "position": _clean_text(record.get("position")),
        "minutes": _coerce_int(record.get("minutes")),
        "rating": _coerce_float(record.get("rating")),
        "fouls_committed": _coerce_int(record.get("fouls_committed")),
        "fouls_drawn": _coerce_int(record.get("fouls_drawn")),
        "shots_total": _coerce_int(record.get("shots_total")),
        "shots_on_target": _coerce_int(record.get("shots_on")),
        "tackles_total": _coerce_int(record.get("tackles")),
        "saves": _coerce_int(record.get("saves")),
        "offsides": _coerce_int(record.get("offsides")),
        "yellow_cards": _coerce_int(record.get("yellow_cards")),
        "red_cards": _coerce_int(record.get("red_cards")),
        "passes_total": _coerce_int(record.get("passes")),
        "passes_key": _coerce_int(record.get("key_passes")),
        "duels_total": _coerce_int(record.get("duels_total")),
        "duels_won": _coerce_int(record.get("duels_won")),
        "dribbles_success": _coerce_int(record.get("dribbles")),
    }


def _build_fixture_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "fixture_id": record["fixture_id"],
        "league_id": record["league_id"],
        "season": record["season"],
        "date": record["date"],
        "home_team": record["home_team"],
        "away_team": record["away_team"],
        "home_score": None,
        "away_score": None,
        "referee": record["referee"],
        "status": None,
    }


def _update_player_latest(
    player_latest: dict[int, dict[str, Any]],
    record: dict[str, Any],
) -> None:
    current = player_latest.get(record["player_id"])
    candidate = {
        "player_id": record["player_id"],
        "player_name": record["player_name"],
        "current_team": record["team"],
        "current_league_id": record["league_id"],
        "_latest_date": record["date"],
    }

    if current is None or record["date"] >= current["_latest_date"]:
        player_latest[record["player_id"]] = candidate


def _build_match_performance_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "fixture_id": record["fixture_id"],
        "player_id": record["player_id"],
        "league_id": record["league_id"],
        "season": record["season"],
        "date": record["date"],
        "home_team": record["home_team"],
        "away_team": record["away_team"],
        "team": record["team"],
        "is_home": record["is_home"],
        "referee": record["referee"],
        "minutes": record["minutes"],
        "position": record["position"],
        "rating": record["rating"],
        "is_substitute": False,
        "is_captain": False,
        "goals_scored": None,
        "goals_conceded": None,
        "assists": None,
        "saves": record["saves"],
        "shots_total": record["shots_total"],
        "shots_on_target": record["shots_on_target"],
        "passes_total": record["passes_total"],
        "passes_key": record["passes_key"],
        "pass_accuracy": None,
        "tackles_total": record["tackles_total"],
        "tackles_blocks": None,
        "tackles_interceptions": None,
        "duels_total": record["duels_total"],
        "duels_won": record["duels_won"],
        "dribbles_attempts": None,
        "dribbles_success": record["dribbles_success"],
        "dribbles_past": None,
        "fouls_committed": record["fouls_committed"],
        "fouls_drawn": record["fouls_drawn"],
        "yellow_cards": record["yellow_cards"],
        "red_cards": record["red_cards"],
        "pen_won": None,
        "pen_committed": None,
        "pen_scored": None,
        "pen_missed": None,
        "pen_saved": None,
        "offsides": record["offsides"],
    }


def _strip_player_sort_key(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key != "_latest_date"
    }


def _upsert_batched(
    model: type[Any],
    rows: list[dict[str, Any]],
    conflict_columns: list[str],
    *,
    batch_size: int,
) -> int:
    total = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        total += upsert_rows(model, batch, conflict_columns)
    return total


def _tracked_league_ids() -> set[int]:
    settings = get_settings()
    return {int(row["league_id"]) for row in settings.load_json("leagues.json")}


def _parse_match_datetime(value: Any) -> datetime | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    try:
        if "T" in text:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        parsed_date = date.fromisoformat(text[:10])
        return datetime.combine(parsed_date, time.min, tzinfo=UTC)
    except ValueError:
        return None


def _coerce_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        result = float(value)
        return None if pd.isna(result) else result
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if _is_missing(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes"}:
        return True
    if text in {"0", "false", "f", "no"}:
        return False
    return None


def _clean_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if text.lower() == "nan":
        return None
    return text or None


def _is_missing(value: Any) -> bool:
    if value in (None, ""):
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import football_model raw player stats into Stockport tables.")
    parser.add_argument("--csv-path", default=str(DEFAULT_LEGACY_CSV_PATH))
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--league-id", dest="league_ids", action="append", type=int, default=None)
    parser.add_argument("--season", dest="seasons", action="append", type=int, default=None)
    parser.add_argument("--tracked-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    summary = import_legacy_raw_player_stats(
        csv_path=args.csv_path,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        league_ids=args.league_ids,
        seasons=args.seasons,
        tracked_only=args.tracked_only,
    )
    LOGGER.info("Legacy raw-stats import complete: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
