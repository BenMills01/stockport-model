"""SkillCorner API ingestion — physical, off-ball runs, pressure, and passing data."""

from __future__ import annotations

from datetime import date, datetime, timezone
import difflib
import json
import logging
import time
import threading
from pathlib import Path
from typing import Any

import requests
from requests import Response
from requests.auth import HTTPBasicAuth

from config import get_settings
from db.schema import (
    Fixture,
    Lineup,
    Player,
    SkillCornerMatchMap,
    SkillCornerOffBallRuns,
    SkillCornerPasses,
    SkillCornerPhysical,
    SkillCornerPlayerMap,
    SkillCornerPressure,
)
from db.session import session_scope
from ingestion.common import normalise_text, upsert_rows


LOGGER = logging.getLogger(__name__)

_RATE_LOCK = threading.Lock()
_LAST_REQUEST_AT: float = 0.0
_SC_RATE_LIMIT_SECONDS: int = 1  # SkillCorner is generous; 1 s is conservative

_NAME_SIM_THRESHOLD_SHIRT = 0.80
_NAME_SIM_THRESHOLD_BIRTHDAY = 0.70
_NAME_SIM_THRESHOLD_NAME_ONLY = 0.85
_MATCH_DATE_SIM_THRESHOLD = 0.65  # lowered: both SC and API-Football use short names

# Known abbreviation mismatches between SC short_name and API-Football team names
_TEAM_ALIASES: dict[str, str] = {
    "queens park": "qpr",
    "queens park rangers": "qpr",
    "wolverhampton": "wolves",
    "wolverhampton wanderers": "wolves",
    "brighton and hove albion": "brighton",
    "brighton & hove albion": "brighton",
    "tottenham hotspur": "tottenham",
    "west ham united": "west ham",
    "manchester united": "man united",
    "manchester city": "man city",
    "newcastle united": "newcastle",
    "nottingham forest": "nott'm forest",
    "sheffield united": "sheffield utd",
}

UPSERT_BATCH_SIZE = 500


class SkillCornerError(RuntimeError):
    """Raised when the SkillCorner client cannot complete a request."""


# ---------------------------------------------------------------------------
# Core HTTP client
# ---------------------------------------------------------------------------


def sc_get(endpoint: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Execute a rate-limited, auto-paginated GET against the SkillCorner API.

    SkillCorner uses a ``count / results / next / previous`` cursor pattern for
    list endpoints.  Single-object endpoints (e.g. ``/match/{id}/``) return a
    plain dict, which is wrapped in a list for a uniform return type.

    Returns a flat list of result dicts.
    """

    settings = get_settings()
    if not settings.skillcorner_username or not settings.skillcorner_password:
        raise SkillCornerError(
            "SKILLCORNER_USERNAME and SKILLCORNER_PASSWORD must be set"
        )

    auth = HTTPBasicAuth(settings.skillcorner_username, settings.skillcorner_password)
    base_url = settings.skillcorner_base_url.rstrip("/")
    url: str | None = f"{base_url}/{endpoint.lstrip('/')}"
    all_results: list[dict[str, Any]] = []

    while url:
        payload = _sc_request(url, params=params, auth=auth)
        params = None  # only pass params on the first request; subsequent pages use the full next URL

        if isinstance(payload, list):
            all_results.extend(payload)
            break
        if isinstance(payload, dict) and "results" in payload:
            all_results.extend(payload["results"])
            url = payload.get("next")  # None terminates pagination
        else:
            # Single-object response (e.g. /match/{id}/)
            all_results.append(payload)
            break

    return all_results


def _sc_request(
    url: str,
    *,
    params: dict[str, Any] | None,
    auth: HTTPBasicAuth,
) -> Any:
    """Execute one HTTP GET with rate limiting, 429 / 5xx retry logic."""

    last_error: Exception | None = None

    for attempt in range(1, 4):
        _respect_rate_limit()
        try:
            response = requests.get(url, params=params, auth=auth, timeout=120)
            _raise_for_status(response)
            return response.json()
        except requests.HTTPError as exc:
            last_error = exc
            resp: Response | None = exc.response
            status = resp.status_code if resp is not None else 0
            if status == 429:
                retry_after = _parse_retry_after(resp)
                LOGGER.warning(
                    "SkillCorner 429 rate limit — sleeping %ss (attempt %d/3)",
                    retry_after,
                    attempt,
                )
                time.sleep(retry_after)
            elif 500 <= status < 600:
                backoff = float(2 ** (attempt - 1))
                LOGGER.warning(
                    "SkillCorner %s server error — backoff %.0fs (attempt %d/3)",
                    status,
                    backoff,
                    attempt,
                )
                time.sleep(backoff)
            else:
                raise
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(float(2 ** (attempt - 1)))

    raise SkillCornerError(f"SkillCorner request failed for {url}: {last_error}")


def _respect_rate_limit() -> None:
    global _LAST_REQUEST_AT
    with _RATE_LOCK:
        now = time.monotonic()
        elapsed = now - _LAST_REQUEST_AT
        if _LAST_REQUEST_AT and elapsed < _SC_RATE_LIMIT_SECONDS:
            time.sleep(_SC_RATE_LIMIT_SECONDS - elapsed)
        _LAST_REQUEST_AT = time.monotonic()


def _raise_for_status(response: Response) -> None:
    if response.status_code in {429, 500, 502, 503, 504}:
        raise requests.HTTPError(
            f"retryable response {response.status_code}", response=response
        )
    response.raise_for_status()


def _parse_retry_after(response: Response | None) -> float:
    if response is None:
        return 60.0
    header = response.headers.get("Retry-After", "")
    try:
        return max(float(header), 1.0)
    except (TypeError, ValueError):
        return 60.0


# ---------------------------------------------------------------------------
# Match ID reconciliation
# ---------------------------------------------------------------------------


def reconcile_matches(competition_edition_ids: list[int]) -> int:
    """Fetch SC matches for the given competition editions and reconcile against Fixture.

    Returns the number of rows upserted into SkillCornerMatchMap.
    """

    ce_param = ",".join(str(i) for i in competition_edition_ids)
    sc_matches = sc_get("/matches/", {"competition_edition": ce_param, "user": "true"})

    # Load all fixtures once so we can fuzzy-match without repeated DB hits.
    with session_scope() as session:
        db_fixtures: list[Fixture] = session.query(Fixture).all()
        # Detach — we only need plain data
        fixture_rows = [
            {
                "fixture_id": f.fixture_id,
                "match_date": f.date.date() if isinstance(f.date, datetime) else f.date,
                "home_team": f.home_team,
                "away_team": f.away_team,
            }
            for f in db_fixtures
        ]

    rows: list[dict[str, Any]] = []
    unmatched: list[int] = []

    for sc_match in sc_matches:
        sc_match_id = sc_match.get("id")
        if sc_match_id is None:
            continue

        sc_date_raw = sc_match.get("date_time") or sc_match.get("match_date")
        sc_date = _parse_date(sc_date_raw)

        home_raw = _extract_team_name(sc_match, "home")
        away_raw = _extract_team_name(sc_match, "away")

        matched_fixture_id, confidence = _match_fixture(
            sc_date, home_raw, away_raw, fixture_rows
        )

        row: dict[str, Any] = {
            "sc_match_id": sc_match_id,
            "fixture_id": matched_fixture_id,
            "sc_competition_id": _nested_id(sc_match, "competition_edition", "competition", "id"),
            "sc_season_id": _nested_id(sc_match, "competition_edition", "season", "id"),
            "sc_competition_edition_id": _nested_id(sc_match, "competition_edition", "id"),
            "match_date": sc_date,
            "home_team_sc": home_raw,
            "away_team_sc": away_raw,
            "match_confidence": confidence,
            "matched_at": datetime.now(tz=timezone.utc) if matched_fixture_id else None,
        }
        rows.append(row)

        if matched_fixture_id is None:
            unmatched.append(sc_match_id)

    for sc_mid in unmatched:
        LOGGER.warning(
            "SkillCorner match %s could not be reconciled to a Fixture row", sc_mid
        )

    return upsert_rows(SkillCornerMatchMap, rows, ["sc_match_id"])


def _match_fixture(
    sc_date: date | None,
    home_raw: str | None,
    away_raw: str | None,
    fixture_rows: list[dict[str, Any]],
) -> tuple[int | None, float | None]:
    """Return (fixture_id, confidence) for the best matching Fixture, or (None, None).

    Uses fuzzy similarity with a reduced threshold (0.65) since both SC and
    API-Football use abbreviated team names, and adds a partial-word overlap
    fallback for edge cases where abbreviations diverge (e.g. "QPR" vs "Queens Park").
    """

    if sc_date is None:
        return None, None

    home_norm = normalise_text(home_raw)
    away_norm = normalise_text(away_raw)
    best_id: int | None = None
    best_score: float = 0.0

    for fixture in fixture_rows:
        fx_date = fixture["match_date"]
        if fx_date != sc_date:
            continue  # date must match exactly (ignoring time)

        fx_home = normalise_text(fixture["home_team"])
        fx_away = normalise_text(fixture["away_team"])
        home_sim = _team_name_sim(home_norm, fx_home)
        away_sim = _team_name_sim(away_norm, fx_away)
        score = (home_sim + away_sim) / 2.0

        if score >= 0.65 and score > best_score:
            best_score = score
            best_id = fixture["fixture_id"]

    if best_id is not None:
        return best_id, round(best_score, 4)
    return None, None


def _team_name_sim(a: str, b: str) -> float:
    """Similarity between two normalised team name strings.

    Applies known alias normalisation first (e.g. 'queens park' → 'qpr'),
    then combines sequence ratio with token-overlap so abbreviated names
    still score reasonably when one is a prefix of the other.
    """
    a = _TEAM_ALIASES.get(a, a)
    b = _TEAM_ALIASES.get(b, b)
    seq = _name_sim(a, b)
    # Token overlap: fraction of tokens from the shorter that appear in the longer
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if a_tokens and b_tokens:
        shorter = a_tokens if len(a_tokens) <= len(b_tokens) else b_tokens
        longer = a_tokens if len(a_tokens) > len(b_tokens) else b_tokens
        overlap = len(shorter & longer) / len(shorter)
    else:
        overlap = 0.0
    return max(seq, overlap * 0.9)  # overlap capped at 0.9 to prefer exact matches


# ---------------------------------------------------------------------------
# Player ID reconciliation
# ---------------------------------------------------------------------------


def reconcile_players(sc_match_ids: list[int]) -> int:
    """For each SC match, reconcile SC player IDs against the Player / Lineup tables.

    Returns the number of rows upserted into SkillCornerPlayerMap.
    """

    # Load existing match map so we know which fixture_id goes with each sc_match_id
    with session_scope() as session:
        match_map: dict[int, int | None] = {
            row.sc_match_id: row.fixture_id
            for row in session.query(SkillCornerMatchMap).filter(
                SkillCornerMatchMap.sc_match_id.in_(sc_match_ids)
            )
        }

    rows: list[dict[str, Any]] = []

    for sc_match_id in sc_match_ids:
        fixture_id = match_map.get(sc_match_id)
        match_detail = sc_get(f"/match/{sc_match_id}/")
        if not match_detail:
            continue
        detail = match_detail[0]
        players_raw = detail.get("players") or []

        # Load lineup + player bio data for this fixture (if reconciled)
        lineup_rows: list[dict[str, Any]] = []
        player_bio: dict[int, dict[str, Any]] = {}

        if fixture_id is not None:
            with session_scope() as session:
                lineups = (
                    session.query(Lineup)
                    .filter(Lineup.fixture_id == fixture_id)
                    .all()
                )
                lineup_rows = [
                    {
                        "player_id": ln.player_id,
                        "shirt_number": ln.shirt_number,
                    }
                    for ln in lineups
                ]
                pids = [ln["player_id"] for ln in lineup_rows]
                if pids:
                    players_db = (
                        session.query(Player)
                        .filter(Player.player_id.in_(pids))
                        .all()
                    )
                    for p in players_db:
                        player_bio[p.player_id] = {
                            "player_name": p.player_name,
                            "birth_date": p.birth_date,
                        }

        for sc_player in players_raw:
            sc_pid = sc_player.get("id")
            if sc_pid is None:
                continue

            sc_name = sc_player.get("name") or ""
            sc_first = sc_player.get("first_name") or ""
            sc_last = sc_player.get("last_name") or ""
            sc_short = sc_player.get("short_name") or sc_name
            sc_birthday = _parse_date(sc_player.get("birthdate"))
            sc_shirt = _coerce_int(sc_player.get("shirt_number"))

            matched_pid, method, confidence = _match_player(
                sc_name=sc_short or f"{sc_first} {sc_last}".strip(),
                sc_birthday=sc_birthday,
                sc_shirt=sc_shirt,
                lineup_rows=lineup_rows,
                player_bio=player_bio,
            )

            rows.append(
                {
                    "sc_player_id": sc_pid,
                    "player_id": matched_pid,
                    "sc_first_name": sc_first or None,
                    "sc_last_name": sc_last or None,
                    "sc_short_name": sc_short or None,
                    "sc_birthday": sc_birthday,
                    "match_method": method,
                    "match_confidence": confidence,
                    "matched_at": datetime.now(tz=timezone.utc) if matched_pid else None,
                }
            )

    return upsert_rows(SkillCornerPlayerMap, rows, ["sc_player_id"])


def _match_player(
    sc_name: str,
    sc_birthday: date | None,
    sc_shirt: int | None,
    lineup_rows: list[dict[str, Any]],
    player_bio: dict[int, dict[str, Any]],
) -> tuple[int | None, str, float | None]:
    """Return (player_id, method, confidence) using three-tier matching."""

    sc_name_norm = normalise_text(sc_name)

    # Tier 1 — shirt number + name similarity
    if sc_shirt is not None:
        for lineup in lineup_rows:
            if lineup.get("shirt_number") == sc_shirt:
                pid = lineup["player_id"]
                bio = player_bio.get(pid, {})
                sim = _name_sim(sc_name_norm, normalise_text(bio.get("player_name")))
                if sim >= _NAME_SIM_THRESHOLD_SHIRT:
                    return pid, "shirt_number_name", 0.95

    # Tier 2 — birthday + name similarity
    if sc_birthday is not None:
        for pid, bio in player_bio.items():
            if bio.get("birth_date") == sc_birthday:
                sim = _name_sim(sc_name_norm, normalise_text(bio.get("player_name")))
                if sim >= _NAME_SIM_THRESHOLD_BIRTHDAY:
                    return pid, "birthdate_name", 0.90

    # Tier 3 — name only
    best_pid: int | None = None
    best_sim: float = 0.0
    for pid, bio in player_bio.items():
        sim = _name_sim(sc_name_norm, normalise_text(bio.get("player_name")))
        if sim > best_sim:
            best_sim = sim
            best_pid = pid

    if best_pid is not None and best_sim >= _NAME_SIM_THRESHOLD_NAME_ONLY:
        return best_pid, "name_only", round(best_sim, 4)

    return None, "unmatched", None


# ---------------------------------------------------------------------------
# Data ingestion helpers
# ---------------------------------------------------------------------------


def ingest_physical(
    competition_edition_ids: list[int],
    **filter_kwargs: Any,
) -> int:
    """Fetch /physical/ data and upsert into SkillCornerPhysical."""

    ce_param = ",".join(str(i) for i in competition_edition_ids)
    params: dict[str, Any] = {
        "competition_edition": ce_param,
        "group_by": "match,player",
        "response_format": "json",
        "average_per": "match,p90",
    }
    params.update(filter_kwargs)

    results = sc_get("/physical/", params)
    rows = [_build_physical_row(r) for r in results]
    rows = [r for r in rows if r is not None]

    count = _upsert_in_batches(SkillCornerPhysical, rows, ["sc_match_id", "sc_player_id"])
    _reconcile_fixture_player(SkillCornerPhysical, "sc_match_id", "sc_player_id")
    LOGGER.info("SkillCornerPhysical: upserted %d rows", count)
    return count


def ingest_off_ball_runs(competition_edition_ids: list[int]) -> int:
    """Fetch /in_possession/off_ball_runs/ and upsert into SkillCornerOffBallRuns."""

    total = 0
    for ce_id in competition_edition_ids:
        results = sc_get(
            "/in_possession/off_ball_runs/",
            {"competition_edition": ce_id, "group_by": "match,player"},
        )
        rows = [_build_off_ball_runs_row(r) for r in results]
        rows = [r for r in rows if r is not None]
        total += _upsert_in_batches(SkillCornerOffBallRuns, rows, ["sc_match_id", "sc_player_id"])
        LOGGER.info("SkillCornerOffBallRuns: edition %s — %d rows", ce_id, len(rows))

    _reconcile_fixture_player(SkillCornerOffBallRuns, "sc_match_id", "sc_player_id")
    LOGGER.info("SkillCornerOffBallRuns: upserted %d rows total", total)
    return total


def ingest_pressure(competition_edition_ids: list[int]) -> int:
    """Fetch /in_possession/on_ball_pressures/ and upsert into SkillCornerPressure."""

    total = 0
    for ce_id in competition_edition_ids:
        results = sc_get(
            "/in_possession/on_ball_pressures/",
            {"competition_edition": ce_id, "group_by": "match,player"},
        )
        rows = [_build_pressure_row(r) for r in results]
        rows = [r for r in rows if r is not None]
        total += _upsert_in_batches(SkillCornerPressure, rows, ["sc_match_id", "sc_player_id"])
        LOGGER.info("SkillCornerPressure: edition %s — %d rows", ce_id, len(rows))

    _reconcile_fixture_player(SkillCornerPressure, "sc_match_id", "sc_player_id")
    LOGGER.info("SkillCornerPressure: upserted %d rows total", total)
    return total


def ingest_passes(competition_edition_ids: list[int]) -> int:
    """Fetch /in_possession/passes/ and upsert into SkillCornerPasses."""

    total = 0
    for ce_id in competition_edition_ids:
        results = sc_get(
            "/in_possession/passes/",
            {"competition_edition": ce_id, "group_by": "match,player"},
        )
        rows = [_build_passes_row(r) for r in results]
        rows = [r for r in rows if r is not None]
        total += _upsert_in_batches(SkillCornerPasses, rows, ["sc_match_id", "sc_player_id"])
        LOGGER.info("SkillCornerPasses: edition %s — %d rows", ce_id, len(rows))

    _reconcile_fixture_player(SkillCornerPasses, "sc_match_id", "sc_player_id")
    LOGGER.info("SkillCornerPasses: upserted %d rows total", total)
    return total


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def discover_competition_edition_ids(*, save_to_config: bool = True) -> list[int]:
    """Return all SkillCorner competition edition IDs the account has access to.

    Calls ``/competition_editions/?user=true`` and optionally writes the
    discovered IDs back into ``config/skillcorner_leagues.json`` so subsequent
    calls can skip discovery.  Returns the flat list of edition IDs.
    """

    editions = sc_get("/competition_editions/", {"user": "true"})
    all_ids = [int(e["id"]) for e in editions if e.get("id") is not None]

    if save_to_config and all_ids:
        settings = get_settings()
        config_path: Path = settings.config_dir / "skillcorner_leagues.json"
        try:
            existing: list[dict[str, Any]] = json.loads(
                config_path.read_text(encoding="utf-8")
            )
        except (FileNotFoundError, json.JSONDecodeError):
            existing = []

        # Build a name→edition lookup from what the API returned
        name_to_edition_ids: dict[str, list[int]] = {}
        for edition in editions:
            eid = edition.get("id")
            if eid is None:
                continue
            comp = edition.get("competition") or {}
            name = comp.get("name", "")
            name_to_edition_ids.setdefault(name, [])
            name_to_edition_ids[name].append(int(eid))

        updated = False
        for entry in existing:
            league_name = entry.get("name", "")
            if league_name in name_to_edition_ids:
                entry["sc_competition_edition_ids"] = sorted(
                    name_to_edition_ids[league_name]
                )
                updated = True

        if updated:
            config_path.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            LOGGER.info(
                "skillcorner_leagues.json updated with %d competition edition IDs",
                len(all_ids),
            )

    return all_ids


def load_competition_edition_ids() -> list[int]:
    """Return competition edition IDs from config, running discovery if all are empty."""

    settings = get_settings()
    config_path: Path = settings.config_dir / "skillcorner_leagues.json"
    try:
        leagues: list[dict[str, Any]] = json.loads(
            config_path.read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError):
        leagues = []

    ids: list[int] = []
    for entry in leagues:
        ids.extend(entry.get("sc_competition_edition_ids") or [])

    if not ids:
        LOGGER.info(
            "No SkillCorner competition edition IDs in config — running discovery"
        )
        ids = discover_competition_edition_ids(save_to_config=True)

    return sorted(set(ids))


def run_skillcorner_ingest(competition_edition_ids: list[int]) -> dict[str, Any]:
    """Full SkillCorner ingest pipeline: reconcile then fetch all data tables.

    Returns a summary dict with row counts per table and reconciliation breakdown.
    """

    LOGGER.info(
        "SkillCorner ingest started for competition_edition_ids=%s",
        competition_edition_ids,
    )

    # 1. Match reconciliation
    matches_upserted = reconcile_matches(competition_edition_ids)
    LOGGER.info("reconcile_matches: %d rows upserted", matches_upserted)

    # 2. Collect matched sc_match_ids
    with session_scope() as session:
        matched_rows = (
            session.query(SkillCornerMatchMap.sc_match_id)
            .filter(SkillCornerMatchMap.fixture_id.isnot(None))
            .all()
        )
    sc_match_ids = [r.sc_match_id for r in matched_rows]
    LOGGER.info("Matched sc_match_ids available for player reconciliation: %d", len(sc_match_ids))

    # 3. Player reconciliation
    players_upserted = reconcile_players(sc_match_ids)
    LOGGER.info("reconcile_players: %d rows upserted", players_upserted)

    # 4. Data ingestion
    physical_count = ingest_physical(competition_edition_ids)
    off_ball_count = ingest_off_ball_runs(competition_edition_ids)
    pressure_count = ingest_pressure(competition_edition_ids)
    passes_count = ingest_passes(competition_edition_ids)

    # 5. Summary
    with session_scope() as session:
        method_counts: dict[str, int] = {}
        for method_val, in (
            session.query(SkillCornerPlayerMap.match_method)
            .filter(SkillCornerPlayerMap.sc_player_id.in_(
                session.query(SkillCornerPlayerMap.sc_player_id)
            ))
            .distinct()
        ):
            pass  # just a placeholder; real count below
        method_rows = (
            session.query(SkillCornerPlayerMap.match_method)
            .all()
        )
        for (m,) in method_rows:
            method_counts[m or "unknown"] = method_counts.get(m or "unknown", 0) + 1

    summary: dict[str, Any] = {
        "matches_reconciled": matches_upserted,
        "players_reconciled_by_method": method_counts,
        "rows_upserted": {
            "skillcorner_physical": physical_count,
            "skillcorner_off_ball_runs": off_ball_count,
            "skillcorner_pressure": pressure_count,
            "skillcorner_passes": passes_count,
        },
    }

    LOGGER.info(
        "SkillCorner ingest complete: matches=%d, players=%d, "
        "physical=%d, off_ball_runs=%d, pressure=%d, passes=%d",
        matches_upserted,
        sum(method_counts.values()),
        physical_count,
        off_ball_count,
        pressure_count,
        passes_count,
    )

    return summary


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _build_physical_row(r: dict[str, Any]) -> dict[str, Any] | None:
    sc_match_id = _coerce_int(r.get("match_id"))
    sc_player_id = _coerce_int(r.get("player_id"))
    if sc_match_id is None or sc_player_id is None:
        return None

    # Physical endpoint uses different field names to the DB column names.
    # API field → DB column mapping:
    #   total_distance_full_all       → dist_per_match
    #   hsr_distance_full_all         → hsr_dist_per_match
    #   sprint_distance_full_all      → sprint_dist_per_match
    #   hsr_count_full_all            → count_hsr_per_match
    #   sprint_count_full_all         → count_sprint_per_match
    #   highaccel_count_full_all      → count_high_accel_per_match
    #   highdecel_count_full_all      → count_high_decel_per_match
    #   psv99                         → top_speed_per_match
    #   minutes_full_all              → minutes_played_per_match
    #   physical_check_passed         → quality_check
    #   *_p90 suffix columns follow the same pattern
    return {
        "sc_match_id": sc_match_id,
        "sc_player_id": sc_player_id,
        "player_name": r.get("player_name"),
        "player_birthdate": _parse_date(r.get("player_birthdate")),
        "match_name": r.get("match_name"),
        "match_date": _parse_date(r.get("match_date")),
        "team_id": _coerce_int(r.get("team_id")),
        "team_name": r.get("team_name"),
        "competition_id": _coerce_int(r.get("competition_id")),
        "competition_name": r.get("competition_name"),
        "season_id": _coerce_int(r.get("season_id")),
        "season_name": r.get("season_name"),
        "competition_edition_id": _coerce_int(r.get("competition_edition_id")),
        "position": r.get("position") or r.get("position_group"),
        "group": r.get("group") or r.get("position_group"),
        "quality_check": r.get("physical_check_passed"),
        "count_match": None,
        "count_match_failed": None,
        "minutes_played_per_match": _coerce_float(r.get("minutes_full_all")),
        "adjusted_min_tip_per_match": None,
        "adjusted_min_otip_per_match": None,
        # per match (actual match totals — group_by=match,player gives one row per match)
        "dist_per_match": _coerce_float(r.get("total_distance_full_all")),
        "hsr_dist_per_match": _coerce_float(r.get("hsr_distance_full_all")),
        "sprint_dist_per_match": _coerce_float(r.get("sprint_distance_full_all")),
        "count_hsr_per_match": _coerce_float(r.get("hsr_count_full_all")),
        "count_sprint_per_match": _coerce_float(r.get("sprint_count_full_all")),
        "count_high_accel_per_match": _coerce_float(r.get("highaccel_count_full_all")),
        "count_high_decel_per_match": _coerce_float(r.get("highdecel_count_full_all")),
        "top_speed_per_match": _coerce_float(r.get("psv99")),
        "dist_tip_per_match": None,
        "dist_otip_per_match": None,
        "hsr_dist_tip_per_match": None,
        "hsr_dist_otip_per_match": None,
        "sprint_dist_tip_per_match": None,
        "sprint_dist_otip_per_match": None,
        # p90
        "dist_p90": _coerce_float(r.get("total_distance_full_all_p90")),
        "hsr_dist_p90": _coerce_float(r.get("hsr_distance_full_all_p90")),
        "sprint_dist_p90": _coerce_float(r.get("sprint_distance_full_all_p90")),
        "count_hsr_p90": _coerce_float(r.get("hsr_count_full_all_p90")),
        "count_sprint_p90": _coerce_float(r.get("sprint_count_full_all_p90")),
        "count_high_accel_p90": _coerce_float(r.get("highaccel_count_full_all_p90")),
        "count_high_decel_p90": _coerce_float(r.get("highdecel_count_full_all_p90")),
        # p60bip / p30tip / p30otip — not returned by the API at match-level grouping
        "dist_p60bip": None,
        "hsr_dist_p60bip": None,
        "sprint_dist_p60bip": None,
        "count_hsr_p60bip": None,
        "count_sprint_p60bip": None,
        "count_high_accel_p60bip": None,
        "count_high_decel_p60bip": None,
        "dist_p30tip": None,
        "hsr_dist_p30tip": None,
        "sprint_dist_p30tip": None,
        "dist_tip_p30tip": None,
        "hsr_dist_tip_p30tip": None,
        "dist_p30otip": None,
        "hsr_dist_p30otip": None,
        "sprint_dist_p30otip": None,
        "dist_otip_p30otip": None,
        "hsr_dist_otip_p30otip": None,
    }


def _build_off_ball_runs_row(r: dict[str, Any]) -> dict[str, Any] | None:
    sc_match_id = _coerce_int(r.get("match_id"))
    sc_player_id = _coerce_int(r.get("player_id"))
    if sc_match_id is None or sc_player_id is None:
        return None

    return {
        "sc_match_id": sc_match_id,
        "sc_player_id": sc_player_id,
        **_common_identity_fields(r),
        "third": r.get("third"),
        "channel": r.get("channel"),
        "minutes_played_per_match": _coerce_float(r.get("minutes_played_per_match")),
        "adjusted_min_tip_per_match": _coerce_float(r.get("adjusted_min_tip_per_match")),
        # API uses "runs" shorthand; DB columns use "run_in_behind" prefix
        "count_run_in_behind_in_sample": _coerce_float(r.get("count_runs_in_sample")),
        "count_dangerous_run_in_behind_per_match": _coerce_float(r.get("count_dangerous_runs_per_match")),
        "run_in_behind_threat_per_match": _coerce_float(r.get("runs_threat_per_match")),
        "count_run_in_behind_leading_to_goal_per_match": _coerce_float(r.get("count_runs_leading_to_goal_per_match")),
        "count_run_in_behind_targeted_per_match": _coerce_float(r.get("count_runs_targeted_per_match")),
        "count_run_in_behind_received_per_match": _coerce_float(r.get("count_runs_received_per_match")),
        "count_run_in_behind_leading_to_shot_per_match": _coerce_float(r.get("count_runs_leading_to_shot_per_match")),
        "run_in_behind_targeted_threat_per_match": _coerce_float(r.get("runs_targeted_threat_per_match")),
        "run_in_behind_received_threat_per_match": _coerce_float(r.get("runs_received_threat_per_match")),
        "count_dangerous_run_in_behind_targeted_per_match": _coerce_float(r.get("count_dangerous_runs_targeted_per_match")),
        "count_dangerous_run_in_behind_received_per_match": _coerce_float(r.get("count_dangerous_runs_received_per_match")),
    }


def _build_pressure_row(r: dict[str, Any]) -> dict[str, Any] | None:
    sc_match_id = _coerce_int(r.get("match_id"))
    sc_player_id = _coerce_int(r.get("player_id"))
    if sc_match_id is None or sc_player_id is None:
        return None

    return {
        "sc_match_id": sc_match_id,
        "sc_player_id": sc_player_id,
        **_common_identity_fields(r),
        "third": r.get("third"),
        "channel": r.get("channel"),
        "minutes_played_per_match": _coerce_float(r.get("minutes_played_per_match")),
        "adjusted_min_tip_per_match": _coerce_float(r.get("adjusted_min_tip_per_match")),
        # API omits "_high_" from field names; DB columns include it
        "count_high_pressure_received_in_sample": _coerce_float(r.get("count_pressures_received_in_sample")),
        "count_high_pressure_received_per_match": _coerce_float(r.get("count_pressures_received_per_match")),
        "count_forced_losses_under_high_pressure_per_match": _coerce_float(r.get("count_forced_losses_under_pressure_per_match")),
        "count_ball_retention_under_high_pressure_per_match": _coerce_float(r.get("count_ball_retentions_under_pressure_per_match")),
        "ball_retention_ratio_under_high_pressure": _coerce_float(r.get("ball_retention_ratio_under_pressure")),
        "ball_retention_added_under_high_pressure_per_match": None,  # not in API
        "pass_completion_ratio_under_high_pressure": _coerce_float(r.get("pass_completion_ratio_under_pressure")),
        "count_pass_attempts_under_high_pressure_per_match": _coerce_float(r.get("count_pass_attempts_under_pressure_per_match")),
        "count_completed_passes_under_high_pressure_per_match": _coerce_float(r.get("count_completed_passes_under_pressure_per_match")),
        "count_dangerous_pass_attemps_under_high_pressure_per_match": _coerce_float(r.get("count_dangerous_pass_attempts_under_pressure_per_match")),
        "count_completed_dangerous_passes_under_high_pressure_per_match": _coerce_float(r.get("count_completed_dangerous_passes_under_pressure_per_match")),
        "dangerous_pass_completion_ratio_under_high_pressure": _coerce_float(r.get("dangerous_pass_completion_ratio_under_pressure")),
        "count_difficult_pass_attempts_under_high_pressure_per_match": _coerce_float(r.get("count_difficult_pass_attempts_under_pressure_per_match")),
        "count_completed_difficult_passes_under_high_pressure_per_match": _coerce_float(r.get("count_completed_difficult_passes_under_pressure_per_match")),
        "difficult_pass_completion_ratio_under_high_pressure": _coerce_float(r.get("difficult_pass_completion_ratio_under_pressure")),
    }


def _build_passes_row(r: dict[str, Any]) -> dict[str, Any] | None:
    sc_match_id = _coerce_int(r.get("match_id"))
    sc_player_id = _coerce_int(r.get("player_id"))
    if sc_match_id is None or sc_player_id is None:
        return None

    return {
        "sc_match_id": sc_match_id,
        "sc_player_id": sc_player_id,
        **_common_identity_fields(r),
        "third": r.get("third"),
        "channel": r.get("channel"),
        "minutes_played_per_match": _coerce_float(r.get("minutes_played_per_match")),
        "adjusted_min_tip_per_match": _coerce_float(r.get("adjusted_min_tip_per_match")),
        # API uses "runs" shorthand; DB columns use "run_in_behind" prefix
        "count_opportunities_to_pass_to_run_in_behind_in_sample": _coerce_float(r.get("count_opportunities_to_pass_to_runs_in_sample")),
        "count_opportunities_to_pass_to_run_in_behind_per_match": _coerce_float(r.get("count_opportunities_to_pass_to_runs_per_match")),
        "count_pass_attempts_to_run_in_behind_per_match": _coerce_float(r.get("count_pass_attempts_to_runs_per_match")),
        "pass_opportunities_to_run_in_behind_threat_per_match": _coerce_float(r.get("pass_opportunities_to_runs_threat_per_match")),
        "run_in_behind_to_which_pass_attempted_threat_per_match": _coerce_float(r.get("runs_to_which_pass_attempted_threat_per_match")),
        "pass_completion_ratio_to_run_in_behind": _coerce_float(r.get("pass_completion_ratio_to_runs")),
        "count_run_in_behind_by_teammate_per_match": _coerce_float(r.get("count_runs_by_teammate_per_match")),
        "run_in_behind_to_which_pass_completed_threat_per_match": _coerce_float(r.get("runs_to_which_pass_completed_threat_per_match")),
        "count_completed_pass_to_run_in_behind_per_match": _coerce_float(r.get("count_completed_pass_to_runs_per_match")),
        "count_completed_pass_to_run_in_behind_leading_to_shot_per_match": _coerce_float(r.get("count_completed_pass_to_runs_leading_to_shot_per_match")),
        "count_completed_pass_to_run_in_behind_leading_to_goal_per_match": _coerce_float(r.get("count_completed_pass_to_runs_leading_to_goal_per_match")),
        "count_pass_opportunities_to_dangerous_run_in_behind_per_match": _coerce_float(r.get("count_pass_opportunities_to_dangerous_runs_per_match")),
        "count_pass_attempts_to_dangerous_run_in_behind_per_match": _coerce_float(r.get("count_pass_attempts_to_dangerous_runs_per_match")),
        "count_completed_pass_to_dangerous_run_in_behind_per_match": _coerce_float(r.get("count_completed_pass_to_dangerous_runs_per_match")),
    }


def _common_identity_fields(r: dict[str, Any]) -> dict[str, Any]:
    """Extract the identity / context fields shared across all GI endpoints."""

    return {
        "player_name": r.get("player_name"),
        "player_birthdate": _parse_date(r.get("player_birthdate")),
        "match_name": r.get("match_name"),
        "match_date": _parse_date(r.get("match_date")),
        "team_id": _coerce_int(r.get("team_id")),
        "team_name": r.get("team_name"),
        "competition_id": _coerce_int(r.get("competition_id")),
        "competition_name": r.get("competition_name"),
        "season_id": _coerce_int(r.get("season_id")),
        "season_name": r.get("season_name"),
        "competition_edition_id": _coerce_int(r.get("competition_edition_id")),
        "position": r.get("position"),
        "group": r.get("group"),
        "quality_check": r.get("quality_check"),
        "count_match": None,
        "count_match_failed": None,
    }


# ---------------------------------------------------------------------------
# Post-upsert reconciliation (populate fixture_id / player_id from map tables)
# ---------------------------------------------------------------------------


def _reconcile_fixture_player(
    model: type[Any],
    sc_match_col: str,
    sc_player_col: str,
) -> None:
    """Back-fill fixture_id and player_id on a data table using the map tables.

    Rows that already have both IDs populated are skipped.
    """

    table = model.__table__
    sc_match_attr = getattr(model, sc_match_col)
    sc_player_attr = getattr(model, sc_player_col)

    with session_scope() as session:
        rows_missing = (
            session.query(model)
            .filter(
                (model.fixture_id.is_(None)) | (model.player_id.is_(None))
            )
            .all()
        )

        # Load relevant map entries in bulk
        sc_match_ids = {getattr(r, sc_match_col) for r in rows_missing}
        sc_player_ids = {getattr(r, sc_player_col) for r in rows_missing}

        match_map: dict[int, int | None] = {}
        if sc_match_ids:
            for mm in session.query(SkillCornerMatchMap).filter(
                SkillCornerMatchMap.sc_match_id.in_(sc_match_ids)
            ):
                match_map[mm.sc_match_id] = mm.fixture_id

        player_map: dict[int, int | None] = {}
        if sc_player_ids:
            for pm in session.query(SkillCornerPlayerMap).filter(
                SkillCornerPlayerMap.sc_player_id.in_(sc_player_ids)
            ):
                player_map[pm.sc_player_id] = pm.player_id

        for row in rows_missing:
            new_fixture = match_map.get(getattr(row, sc_match_col))
            new_player = player_map.get(getattr(row, sc_player_col))
            if new_fixture is not None:
                row.fixture_id = new_fixture
            if new_player is not None:
                row.player_id = new_player


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _upsert_in_batches(
    model: type[Any],
    rows: list[dict[str, Any]],
    conflict_columns: list[str],
) -> int:
    total = 0
    for i in range(0, len(rows), UPSERT_BATCH_SIZE):
        batch = rows[i : i + UPSERT_BATCH_SIZE]
        total += upsert_rows(model, batch, conflict_columns)
    return total


def _name_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _extract_team_name(sc_match: dict[str, Any], side: str) -> str | None:
    """Pull team name from a SkillCorner match dict for 'home' or 'away'."""

    team_str = sc_match.get(f"{side}_team")
    if isinstance(team_str, str):
        return team_str
    if isinstance(team_str, dict):
        # API returns {"id": ..., "short_name": "..."} — prefer short_name
        return team_str.get("name") or team_str.get("short_name")
    return None


def _nested_id(obj: dict[str, Any], *keys: str) -> int | None:
    """Drill into nested dicts and return the final value as int."""

    current: Any = obj
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _coerce_int(current)


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    raw = str(value).strip()
    # ISO datetime with time component
    if "T" in raw or " " in raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
        except ValueError:
            pass
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    cleaned = str(value).strip()
    if not cleaned:
        return None
    try:
        return int(float(cleaned))
    except (ValueError, TypeError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None
