"""API-Football ingestion helpers for raw match and player data."""

from __future__ import annotations

from datetime import date, datetime
import logging
from math import ceil
import time
from typing import Any, TypeVar

import requests
from requests import Response

from config import get_settings
from db.schema import Fixture, FixtureTeamStat, Injury, Lineup, MatchEvent, MatchPerformance
from db.schema import Player, Sidelined, StandingsSnapshot, Transfer
from ingestion.common import upsert_rows


LOGGER = logging.getLogger(__name__)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
COMPLETED_STATUSES = {"FT", "AET", "PEN"}
PLAYER_PAGE_SIZE = 20
BATCH_FIXTURE_SIZE = 20
UPSERT_BATCH_SIZE = 1000
_LAST_REQUEST_AT = 0.0
T = TypeVar("T")


class ApiFootballError(RuntimeError):
    """Raised when the API-Football client cannot complete a request."""


def api_get(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    """Execute a rate-limited GET request against API-Football."""

    settings = get_settings()
    if not settings.api_football_api_key:
        raise ApiFootballError("API_FOOTBALL_API_KEY is not configured")

    headers = {
        "x-apisports-key": settings.api_football_api_key,
    }
    url = f"{settings.api_football_base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    last_error: Exception | None = None
    for attempt in range(1, 4):
        _respect_rate_limit(settings.api_rate_limit_seconds)
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            _raise_for_status(response)
            payload = response.json()
            if payload.get("errors"):
                raise ApiFootballError(f"API-Football returned errors: {payload['errors']}")
            return payload
        except (requests.RequestException, ValueError, ApiFootballError) as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(float(2 ** (attempt - 1)))

    raise ApiFootballError(f"API-Football request failed for {endpoint}: {last_error}")


def fetch_fixtures(
    from_date: date | str,
    to_date: date | str,
    league_ids: list[int] | None = None,
    fixtures: list[dict[str, Any]] | None = None,
) -> int:
    """Fetch completed fixtures across tracked leagues and upsert them."""

    rows = list(fixtures) if fixtures is not None else collect_completed_fixtures(
        from_date,
        to_date,
        league_ids=league_ids,
    )
    return upsert_rows(Fixture, rows, ["fixture_id"])


def fetch_match_performances(
    from_date: date | str,
    to_date: date | str,
    league_ids: list[int] | None = None,
    fixtures: list[dict[str, Any]] | None = None,
    fixture_details: dict[int, dict[str, Any]] | None = None,
) -> int:
    """Fetch per-player match statistics for completed fixtures."""

    fixtures = list(fixtures) if fixtures is not None else collect_completed_fixtures(
        from_date,
        to_date,
        league_ids=league_ids,
    )
    fixture_lookup = {int(fixture["fixture_id"]): fixture for fixture in fixtures}
    detail_lookup = fixture_details if fixture_details is not None else collect_fixture_details(fixtures)
    rows: list[dict[str, Any]] = []
    fallback_fixture_ids: list[int] = []

    for fixture_id in sorted(fixture_lookup):
        item = detail_lookup.get(fixture_id)
        players_block = (item or {}).get("players") or []
        if players_block:
            rows.extend(_build_match_performance_rows(fixture_lookup[fixture_id], players_block))
        else:
            fallback_fixture_ids.append(fixture_id)

    for fixture_id in _stable_unique(fallback_fixture_ids):
        payload = api_get("/fixtures/players", {"fixture": fixture_id})
        rows.extend(
            _build_match_performance_rows(
                fixture_lookup[fixture_id],
                payload.get("response", []),
            )
        )
    return _upsert_in_batches(
        MatchPerformance,
        rows,
        ["fixture_id", "player_id"],
        batch_size=UPSERT_BATCH_SIZE,
    )


def fetch_fixture_team_stats(
    from_date: date | str,
    to_date: date | str,
    league_ids: list[int] | None = None,
    fixtures: list[dict[str, Any]] | None = None,
    fixture_details: dict[int, dict[str, Any]] | None = None,
) -> int:
    """Fetch team-level match context for completed fixtures."""

    fixtures = list(fixtures) if fixtures is not None else collect_completed_fixtures(
        from_date,
        to_date,
        league_ids=league_ids,
    )
    detail_lookup = fixture_details if fixture_details is not None else collect_fixture_details(fixtures)
    rows: list[dict[str, Any]] = []
    fallback_fixture_ids: list[int] = []

    for fixture in fixtures:
        fixture_id = int(fixture["fixture_id"])
        item = detail_lookup.get(fixture_id)
        if item is not None and "statistics" in item:
            rows.extend(_build_fixture_team_stat_rows(fixture_id, item.get("statistics") or []))
        else:
            fallback_fixture_ids.append(fixture_id)

    for fixture_id in _stable_unique(fallback_fixture_ids):
        payload = api_get("/fixtures/statistics", {"fixture": fixture_id})
        rows.extend(_build_fixture_team_stat_rows(fixture_id, payload.get("response", [])))
    return _upsert_in_batches(
        FixtureTeamStat,
        rows,
        ["fixture_id", "team_name"],
        batch_size=UPSERT_BATCH_SIZE,
    )


def fetch_match_events(
    from_date: date | str,
    to_date: date | str,
    league_ids: list[int] | None = None,
    fixtures: list[dict[str, Any]] | None = None,
    fixture_details: dict[int, dict[str, Any]] | None = None,
) -> int:
    """Fetch all recorded events for completed fixtures."""

    fixtures = list(fixtures) if fixtures is not None else collect_completed_fixtures(
        from_date,
        to_date,
        league_ids=league_ids,
    )
    detail_lookup = fixture_details if fixture_details is not None else collect_fixture_details(fixtures)
    rows: list[dict[str, Any]] = []
    fallback_fixture_ids: list[int] = []

    for fixture in fixtures:
        fixture_id = int(fixture["fixture_id"])
        item = detail_lookup.get(fixture_id)
        if item is not None and "events" in item:
            rows.extend(_build_match_event_rows(fixture_id, item.get("events") or []))
        else:
            fallback_fixture_ids.append(fixture_id)

    for fixture_id in _stable_unique(fallback_fixture_ids):
        payload = api_get("/fixtures/events", {"fixture": fixture_id})
        rows.extend(_build_match_event_rows(fixture_id, payload.get("response", [])))
    return _upsert_in_batches(
        MatchEvent,
        rows,
        [
            "fixture_id",
            "time_elapsed",
            "time_extra",
            "event_type",
            "event_detail",
            "player_id",
            "assist_player_id",
            "team",
        ],
        batch_size=UPSERT_BATCH_SIZE,
    )


def fetch_lineups(
    from_date: date | str,
    to_date: date | str,
    league_ids: list[int] | None = None,
    fixtures: list[dict[str, Any]] | None = None,
    fixture_details: dict[int, dict[str, Any]] | None = None,
) -> int:
    """Fetch starting XI and substitute lineups for completed fixtures."""

    fixtures = list(fixtures) if fixtures is not None else collect_completed_fixtures(
        from_date,
        to_date,
        league_ids=league_ids,
    )
    detail_lookup = fixture_details if fixture_details is not None else collect_fixture_details(fixtures)
    rows: list[dict[str, Any]] = []
    fallback_fixture_ids: list[int] = []

    for fixture in fixtures:
        fixture_id = int(fixture["fixture_id"])
        item = detail_lookup.get(fixture_id)
        if item is not None and "lineups" in item:
            rows.extend(_build_lineup_rows(fixture_id, item.get("lineups") or []))
        else:
            fallback_fixture_ids.append(fixture_id)

    for fixture_id in _stable_unique(fallback_fixture_ids):
        payload = api_get("/fixtures/lineups", {"fixture": fixture_id})
        rows.extend(_build_lineup_rows(fixture_id, payload.get("response", [])))
    return _upsert_in_batches(
        Lineup,
        rows,
        ["fixture_id", "player_id"],
        batch_size=UPSERT_BATCH_SIZE,
    )


def fetch_standings(league_id: int, season: int | str) -> int:
    """Fetch a point-in-time league standings snapshot."""

    payload = api_get("/standings", {"league": league_id, "season": season})
    rows = _build_standings_rows(league_id, payload.get("response", []))
    return upsert_rows(StandingsSnapshot, rows, ["league_id", "date", "team_name"])


def fetch_player_profiles(league_id: int, season: int | str) -> int:
    """Fetch player bio data for a league-season."""

    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = api_get(
            "/players",
            {"league": league_id, "season": season, "page": page},
        )
        response_rows = payload.get("response", [])
        rows.extend(_build_player_rows(response_rows))
        paging = payload.get("paging", {})
        current = int(paging.get("current", page))
        total = int(paging.get("total", current))
        if not response_rows or current >= total:
            break
        page += 1
    count = upsert_rows(Player, rows, ["player_id"])
    if count:
        from db.read_cache import clear_read_caches

        clear_read_caches()
    return count


def fetch_transfers(player_id: int) -> int:
    """Fetch transfer history for a player."""

    payload = api_get("/transfers", {"player": player_id})
    rows = _build_transfer_rows(payload.get("response", []))
    return upsert_rows(Transfer, rows, ["player_id", "date", "type", "team_in", "team_out"])


def fetch_sidelined(player_id: int) -> int:
    """Fetch absence history for a player."""

    payload = api_get("/sidelined", {"player": player_id})
    rows = _build_sidelined_rows(player_id, payload.get("response", []))
    return upsert_rows(Sidelined, rows, ["player_id", "type", "start_date", "end_date"])


def fetch_injuries(player_id_or_league: int | dict[str, int]) -> int:
    """Fetch injury updates for a player or a full league."""

    params = _coerce_injury_params(player_id_or_league)
    payload = api_get("/injuries", params)
    rows = _build_injury_rows(payload.get("response", []))
    return upsert_rows(Injury, rows, ["player_id", "fixture_id", "type", "reason", "date"])


def collect_completed_fixtures(
    from_date: date | str,
    to_date: date | str,
    league_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Return completed fixture rows for the requested leagues and date range."""

    tracked_leagues = _tracked_leagues(league_ids)
    rows_by_fixture_id: dict[int, dict[str, Any]] = {}
    seasons = _season_candidates_for_range(from_date, to_date)

    for league in tracked_leagues:
        for season in seasons:
            payload = api_get(
                "/fixtures",
                {
                    "league": league["league_id"],
                    "season": season,
                    "from": _format_date(from_date),
                    "to": _format_date(to_date),
                },
            )
            for item in payload.get("response", []):
                row = _build_fixture_row(item)
                if row and _fixture_is_completed(item):
                    rows_by_fixture_id[row["fixture_id"]] = row

    return list(rows_by_fixture_id.values())


def collect_fixture_details(fixtures: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Return batched fixture-detail payloads keyed by fixture ID."""

    fixture_lookup = {int(fixture["fixture_id"]): fixture for fixture in fixtures}
    details_by_fixture_id: dict[int, dict[str, Any]] = {}

    for batch in _chunked(sorted(fixture_lookup), BATCH_FIXTURE_SIZE):
        payload = api_get("/fixtures", {"ids": "-".join(str(fixture_id) for fixture_id in batch)})
        for item in payload.get("response", []):
            fixture = item.get("fixture") or {}
            fixture_id = _coerce_int(fixture.get("id"))
            if fixture_id is None or fixture_id not in fixture_lookup:
                continue
            details_by_fixture_id[fixture_id] = item

    return details_by_fixture_id


def fetch_api_usage() -> dict[str, Any]:
    """Return the current API-Football usage summary."""

    payload = api_get("/status", {})
    response = payload.get("response") or {}
    requests_block = response.get("requests") or {}
    subscription = response.get("subscription") or {}

    current = _coerce_int(requests_block.get("current"))
    limit_day = _coerce_int(requests_block.get("limit_day"))
    remaining = None if current is None or limit_day is None else max(limit_day - current, 0)

    return {
        "requests_current": current,
        "requests_limit_day": limit_day,
        "requests_remaining": remaining,
        "plan": subscription.get("plan"),
        "subscription_active": subscription.get("active"),
        "subscription_end": subscription.get("end"),
    }


def load_player_stats_coverage(
    season: int | str,
    league_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Return tracked leagues that advertise player-stat coverage for a season."""

    usable_leagues: list[dict[str, Any]] = []
    for league in _tracked_leagues(league_ids):
        payload = api_get("/leagues", {"id": league["league_id"], "season": int(season)})
        response = payload.get("response", [])
        if not response:
            continue
        season_block = _find_exact_season_block(response[0], int(season))
        fixtures = ((season_block or {}).get("coverage") or {}).get("fixtures") or {}
        if fixtures.get("statistics_players"):
            usable_leagues.append(league)
    return usable_leagues


def estimate_ingest_request_plan(
    from_date: date | str,
    to_date: date | str,
    *,
    league_ids: list[int] | None = None,
    fixture_count: int = 0,
) -> dict[str, int]:
    """Estimate the request footprint for a daily ingest window."""

    league_count = len(_tracked_leagues(league_ids))
    season_count = len(_season_candidates_for_range(from_date, to_date))
    fixture_discovery_calls = league_count * season_count
    fixture_detail_batches = 0 if fixture_count == 0 else ceil(fixture_count / BATCH_FIXTURE_SIZE)
    estimated_detail_calls = fixture_detail_batches
    return {
        "league_count": league_count,
        "season_count": season_count,
        "fixture_count": fixture_count,
        "fixture_discovery_calls": fixture_discovery_calls,
        "detail_batch_size": BATCH_FIXTURE_SIZE,
        "fixture_detail_batches": fixture_detail_batches,
        "estimated_detail_calls": estimated_detail_calls,
        "estimated_total_calls": fixture_discovery_calls + estimated_detail_calls,
    }


def _tracked_leagues(league_ids: list[int] | None = None) -> list[dict[str, Any]]:
    """Return tracked leagues, optionally filtered to a requested subset."""

    tracked_leagues = get_settings().load_json("leagues.json")
    if league_ids is None:
        return list(tracked_leagues)

    requested_ids = {int(league_id) for league_id in league_ids}
    filtered = [
        league
        for league in tracked_leagues
        if int(league["league_id"]) in requested_ids
    ]
    found_ids = {int(league["league_id"]) for league in filtered}
    missing_ids = sorted(requested_ids - found_ids)
    if missing_ids:
        raise ValueError(f"Requested league IDs are not tracked: {missing_ids}")
    return filtered


def _find_exact_season_block(response_item: dict[str, Any], season: int) -> dict[str, Any] | None:
    for season_block in response_item.get("seasons", []):
        if _coerce_int(season_block.get("year")) == season:
            return season_block
    return None


def _season_candidates_for_range(from_date: date | str, to_date: date | str) -> list[int]:
    """Infer API-Football season start years for a date window.

    The tracked leagues are all July-to-June competitions, so API-Football expects
    the season's starting calendar year.
    """

    start = _coerce_date(from_date)
    end = _coerce_date(to_date)
    if end < start:
        raise ValueError("to_date must be on or after from_date")
    return sorted({_season_for_date(start), _season_for_date(end)})


def _season_for_date(value: date) -> int:
    """Map a fixture date to the API-Football season start year."""

    return value.year if value.month >= 7 else value.year - 1


def _coerce_date(value: date | str) -> date:
    """Parse a date-like input into a date object."""

    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _chunked(values: list[T], size: int) -> list[list[T]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def _stable_unique(values: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _upsert_in_batches(
    model: type[Any],
    rows: list[dict[str, Any]],
    conflict_columns: list[str],
    *,
    batch_size: int,
) -> int:
    total = 0
    for batch in _chunked(rows, batch_size):
        total += upsert_rows(model, batch, conflict_columns)
    return total


def _build_fixture_row(item: dict[str, Any]) -> dict[str, Any] | None:
    fixture = item.get("fixture") or {}
    league = item.get("league") or {}
    teams = item.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    goals = item.get("goals") or {}
    fixture_id = fixture.get("id")
    if fixture_id is None:
        return None

    return {
        "fixture_id": int(fixture_id),
        "league_id": _coerce_int(league.get("id")),
        "season": str(league.get("season")),
        "date": _parse_datetime(fixture.get("date")),
        "home_team": home.get("name"),
        "away_team": away.get("name"),
        "home_score": _coerce_int(goals.get("home")),
        "away_score": _coerce_int(goals.get("away")),
        "referee": fixture.get("referee"),
        "status": ((fixture.get("status") or {}).get("short")),
    }


def _build_match_performance_rows(
    fixture: dict[str, Any],
    response_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for team_block in response_rows:
        team = (team_block.get("team") or {}).get("name")
        for player_block in team_block.get("players", []):
            player = player_block.get("player") or {}
            statistics = (player_block.get("statistics") or [{}])[0]
            games = statistics.get("games") or {}
            goals = statistics.get("goals") or {}
            shots = statistics.get("shots") or {}
            passes = statistics.get("passes") or {}
            tackles = statistics.get("tackles") or {}
            duels = statistics.get("duels") or {}
            dribbles = statistics.get("dribbles") or {}
            fouls = statistics.get("fouls") or {}
            cards = statistics.get("cards") or {}
            penalty = statistics.get("penalty") or {}

            player_id = player.get("id")
            if player_id is None:
                continue

            rows.append(
                {
                    "fixture_id": fixture["fixture_id"],
                    "player_id": int(player_id),
                    "league_id": fixture["league_id"],
                    "season": fixture["season"],
                    "date": fixture["date"],
                    "home_team": fixture["home_team"],
                    "away_team": fixture["away_team"],
                    "team": team,
                    "is_home": team == fixture["home_team"],
                    "referee": fixture["referee"],
                    "minutes": _coerce_int(games.get("minutes")),
                    "position": games.get("position"),
                    "rating": _coerce_float(games.get("rating")),
                    "is_substitute": bool(games.get("substitute")),
                    "is_captain": bool(games.get("captain")),
                    "goals_scored": _coerce_int(goals.get("total")),
                    "goals_conceded": _coerce_int(goals.get("conceded")),
                    "assists": _coerce_int(goals.get("assists")),
                    "saves": _coerce_int(goals.get("saves")),
                    "shots_total": _coerce_int(shots.get("total")),
                    "shots_on_target": _coerce_int(shots.get("on")),
                    "passes_total": _coerce_int(passes.get("total")),
                    "passes_key": _coerce_int(passes.get("key")),
                    "pass_accuracy": _coerce_float(passes.get("accuracy")),
                    "tackles_total": _coerce_int(tackles.get("total")),
                    "tackles_blocks": _coerce_int(tackles.get("blocks")),
                    "tackles_interceptions": _coerce_int(tackles.get("interceptions")),
                    "duels_total": _coerce_int(duels.get("total")),
                    "duels_won": _coerce_int(duels.get("won")),
                    "dribbles_attempts": _coerce_int(dribbles.get("attempts")),
                    "dribbles_success": _coerce_int(dribbles.get("success")),
                    "dribbles_past": _coerce_int(dribbles.get("past")),
                    "fouls_committed": _coerce_int(fouls.get("committed")),
                    "fouls_drawn": _coerce_int(fouls.get("drawn")),
                    "yellow_cards": _coerce_int(cards.get("yellow")),
                    "red_cards": _coerce_int(cards.get("red")),
                    "pen_won": _coerce_int(penalty.get("won")),
                    "pen_committed": _coerce_int(penalty.get("commited")),
                    "pen_scored": _coerce_int(penalty.get("scored")),
                    "pen_missed": _coerce_int(penalty.get("missed")),
                    "pen_saved": _coerce_int(penalty.get("saved")),
                    "offsides": _coerce_int(statistics.get("offsides")),
                }
            )
    return rows


def _build_fixture_team_stat_rows(
    fixture_id: int,
    response_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for team_block in response_rows:
        team_name = ((team_block.get("team") or {}).get("name"))
        stat_map = _normalise_statistics_map(team_block.get("statistics", []))
        rows.append(
            {
                "fixture_id": fixture_id,
                "team_name": team_name,
                "possession": _coerce_float(stat_map.get("ball possession")),
                "total_shots": _coerce_int(stat_map.get("total shots")),
                "shots_on_target": _coerce_int(stat_map.get("shots on goal")),
                "corners": _coerce_int(stat_map.get("corner kicks")),
                "fouls": _coerce_int(stat_map.get("fouls")),
                "expected_goals": _coerce_float(
                    stat_map.get("expected goals") or stat_map.get("xg")
                ),
                "passes_total": _coerce_int(stat_map.get("total passes")),
                "passes_accuracy": _coerce_float(stat_map.get("passes %")),
            }
        )
    return rows


def _build_match_event_rows(
    fixture_id: int,
    response_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in response_rows:
        rows.append(
            {
                "fixture_id": fixture_id,
                "time_elapsed": _coerce_int((item.get("time") or {}).get("elapsed")),
                "time_extra": _coerce_int((item.get("time") or {}).get("extra")),
                "event_type": item.get("type"),
                "event_detail": item.get("detail"),
                "player_id": _coerce_int((item.get("player") or {}).get("id")),
                "player_name": (item.get("player") or {}).get("name"),
                "assist_player_id": _coerce_int((item.get("assist") or {}).get("id")),
                "assist_player_name": (item.get("assist") or {}).get("name"),
                "team": (item.get("team") or {}).get("name"),
                "comments": item.get("comments"),
            }
        )
    return rows


def _build_lineup_rows(
    fixture_id: int,
    response_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for team_block in response_rows:
        team_name = ((team_block.get("team") or {}).get("name"))
        formation = team_block.get("formation")
        coach = team_block.get("coach") or {}

        for is_starter, squad_key in ((True, "startXI"), (False, "substitutes")):
            for slot in team_block.get(squad_key, []):
                player = slot.get("player") or {}
                player_id = player.get("id")
                if player_id is None:
                    continue
                rows.append(
                    {
                        "fixture_id": fixture_id,
                        "player_id": int(player_id),
                        "team": team_name,
                        "is_starter": is_starter,
                        "position_label": player.get("pos"),
                        "grid_position": player.get("grid"),
                        "shirt_number": _coerce_int(player.get("number")),
                        "formation": formation,
                        "coach_name": coach.get("name"),
                        "coach_id": _coerce_int(coach.get("id")),
                    }
                )
    return rows


def _build_standings_rows(
    league_id: int,
    response_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    snapshot_date = date.today()
    rows: list[dict[str, Any]] = []
    for league_block in response_rows:
        league = league_block.get("league") or {}
        for group in league.get("standings", []):
            for item in group:
                team = item.get("team") or {}
                all_stats = item.get("all") or {}
                goals_diff = item.get("goalsDiff")
                rows.append(
                    {
                        "league_id": league_id,
                        "date": snapshot_date,
                        "team_name": team.get("name"),
                        "position": _coerce_int(item.get("rank")),
                        "points": _coerce_int(item.get("points")),
                        "goal_diff": _coerce_int(goals_diff),
                        "form": item.get("form"),
                        "played": _coerce_int(all_stats.get("played")),
                        "won": _coerce_int(all_stats.get("win")),
                        "drawn": _coerce_int(all_stats.get("draw")),
                        "lost": _coerce_int(all_stats.get("lose")),
                    }
                )
    return rows


def _build_player_rows(response_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in response_rows:
        player = item.get("player") or {}
        statistics = item.get("statistics") or []
        stat_block = statistics[0] if statistics else {}
        team = stat_block.get("team") or {}
        league = stat_block.get("league") or {}
        player_id = player.get("id")
        if player_id is None:
            continue
        rows.append(
            {
                "player_id": int(player_id),
                "player_name": player.get("name"),
                "nationality": player.get("nationality"),
                "birth_date": _parse_date((player.get("birth") or {}).get("date")),
                "current_age_years": _coerce_float(player.get("age")),
                "height_cm": _extract_height_cm(player.get("height")),
                "weight_kg": _extract_weight_kg(player.get("weight")),
                "photo_url": player.get("photo"),
                "current_team": team.get("name"),
                "current_league_id": _coerce_int(league.get("id")),
            }
        )
    return rows


def _build_transfer_rows(response_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in response_rows:
        player = item.get("player") or {}
        player_id = player.get("id")
        for transfer in item.get("transfers", []):
            teams = transfer.get("teams") or {}
            team_in = teams.get("in") or {}
            team_out = teams.get("out") or {}
            rows.append(
                {
                    "player_id": _coerce_int(player_id),
                    "date": _parse_date(transfer.get("date")),
                    "type": transfer.get("type"),
                    "team_in": team_in.get("name"),
                    "team_in_id": _coerce_int(team_in.get("id")),
                    "team_out": team_out.get("name"),
                    "team_out_id": _coerce_int(team_out.get("id")),
                }
            )
    return rows


def _build_sidelined_rows(
    player_id: int,
    response_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in response_rows:
        rows.append(
            {
                "player_id": player_id,
                "type": item.get("type"),
                "start_date": _parse_date(item.get("start")),
                "end_date": _parse_date(item.get("end")),
            }
        )
    return rows


def _build_injury_rows(response_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in response_rows:
        player = item.get("player") or {}
        fixture = item.get("fixture") or {}
        rows.append(
            {
                "player_id": _coerce_int(player.get("id")),
                "fixture_id": _coerce_int(fixture.get("id")),
                "type": item.get("type"),
                "reason": item.get("reason"),
                "date": _parse_date((fixture.get("date") or item.get("date"))),
            }
        )
    return rows

def _coerce_injury_params(player_id_or_league: int | dict[str, int]) -> dict[str, int]:
    if isinstance(player_id_or_league, dict):
        if "player" in player_id_or_league or "league" in player_id_or_league:
            return player_id_or_league
        raise ValueError("Injury params dict must include 'player' or 'league'")

    return {"player": player_id_or_league}


def _fixture_is_completed(item: dict[str, Any]) -> bool:
    status = ((item.get("fixture") or {}).get("status") or {}).get("short")
    return status in COMPLETED_STATUSES


def _normalise_statistics_map(statistics: list[dict[str, Any]]) -> dict[str, Any]:
    return {str(item.get("type", "")).strip().lower(): item.get("value") for item in statistics}


def _respect_rate_limit(rate_limit_seconds: int) -> None:
    global _LAST_REQUEST_AT
    now = time.monotonic()
    elapsed = now - _LAST_REQUEST_AT
    if _LAST_REQUEST_AT and elapsed < rate_limit_seconds:
        time.sleep(rate_limit_seconds - elapsed)
    _LAST_REQUEST_AT = time.monotonic()


def _raise_for_status(response: Response) -> None:
    if response.status_code in RETRYABLE_STATUS_CODES:
        raise requests.HTTPError(f"retryable response {response.status_code}", response=response)
    response.raise_for_status()


def _format_date(value: date | str) -> str:
    if isinstance(value, str):
        return value
    return value.isoformat()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_date(value: str | None) -> date | None:
    parsed = _parse_datetime(value) if value and "T" in value else None
    if parsed is not None:
        return parsed.date()
    if not value:
        return None
    return date.fromisoformat(value)


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    cleaned = str(value).strip().replace("%", "")
    if not cleaned:
        return None
    return int(float(cleaned))


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).strip().replace("%", "")
    if not cleaned:
        return None
    return float(cleaned)


def _extract_height_cm(value: Any) -> int | None:
    if value in (None, ""):
        return None
    cleaned = str(value).replace("cm", "").strip()
    return _coerce_int(cleaned)


def _extract_weight_kg(value: Any) -> int | None:
    if value in (None, ""):
        return None
    cleaned = str(value).replace("kg", "").strip()
    return _coerce_int(cleaned)
