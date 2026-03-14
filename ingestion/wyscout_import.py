"""Wyscout imports for filtered files and season-average workbook batches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from difflib import SequenceMatcher
from itertools import permutations
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
from sqlalchemy import select

from config import get_settings
from db.schema import Fixture, Lineup, MatchPerformance, Player, WyscoutSeasonStat, WyscoutZoneStat
from db.session import session_scope
from ingestion.common import normalise_text, upsert_rows
from ingestion.matching import resolve_source_player_id, save_source_player_mapping


PLAYER_NAME_ALIASES = {"player", "player_name", "name", "playername"}
TEAM_NAME_ALIASES = {"team", "team_name", "club", "squad"}
TEAM_TIMEFRAME_ALIASES = {
    "team_within_selected_timeframe",
    "team_within_selected_period",
    "team_in_selected_timeframe",
}
PLAYER_EXTERNAL_ID_ALIASES = {"wyscout_id", "wyid", "player_id", "playerid"}
POSITION_ALIASES = {"position", "primary_position"}
MATCHES_PLAYED_ALIASES = {"matches_played", "games_played"}
MINUTES_PLAYED_ALIASES = {"minutes_played", "minutes"}
WORKBOOK_EXTENSIONS = {".xlsx", ".xlsm", ".csv"}
SEASON_PATTERN = re.compile(r"(?P<start>\d{2})\s*[:/-]\s*(?P<end>\d{2})")
UPSERT_BATCH_SIZE = 250

LEAGUE_FOLDER_ALIASES = {
    normalise_text("EFL Championship"): 40,
    normalise_text("Championship"): 40,
    normalise_text("League One"): 41,
    normalise_text("League 1"): 41,
    normalise_text("League Two"): 42,
    normalise_text("League 2"): 42,
    normalise_text("France Ligue 1"): 61,
    normalise_text("France Ligue 2"): 62,
    normalise_text("Germany Bundesliga"): 78,
    normalise_text("Ger 2. Bundesliga"): 79,
    normalise_text("Dutch Eredivise"): 88,
    normalise_text("Dutch Eredivisie"): 88,
    normalise_text("Dutch Eerste Divise"): 89,
    normalise_text("Dutch Eerste Divisie"): 89,
    normalise_text("Belgium Jupiler Pro"): 144,
    normalise_text("Bel Challenger"): 145,
    normalise_text("Scotland Prem"): 179,
    normalise_text("Scottish Premiership"): 179,
}

TEAM_ALIAS_GROUPS = (
    ("rwd molenbeek", "rwdm", "rwdm"),
    ("sk beveren", "waasland beveren", "waasland beveren", "beveren"),
    ("beerschot va", "beerschot wilrijk", "beerschot"),
    ("lommel sk", "lommel united", "lommel"),
    ("genk ii", "krc genk ii"),
    ("anderlecht ii", "rsc anderlecht ii"),
    ("ajax ii", "jong ajax"),
    ("az ii", "jong az"),
    ("psv ii", "jong psv", "jong psv u21"),
    ("utrecht ii", "jong utrecht"),
    ("top oss", "fc oss"),
    ("almere city", "almere city fc"),
    ("clermont", "clermont foot"),
    ("red star", "red star fc 93"),
    ("paris", "paris fc"),
    ("psg", "paris saint germain"),
    ("le havre", "le havre", "le havre ac"),
    ("saint etienne", "saint etienne", "saint etienne"),
    ("angers sco", "angers"),
    ("marseille", "olympique marseille"),
    ("stade brestois 29", "brest"),
    ("eupen", "as eupen"),
    ("standard liege", "standard liege", "standard liege"),
    ("standard liege ii", "standard liege ii"),
    ("sint truiden", "st truiden"),
    ("mechelen", "kv mechelen"),
    ("club brugge", "club brugge kv"),
    ("union saint gilloise", "union st gilloise"),
    ("zulte waregem", "zulte waregem", "zulte waregem"),
    ("seraing", "seraing united", "rfc seraing"),
    ("la louviere", "raal la louviere"),
)

TEAM_ALIAS_MAP = {
    normalise_text(alias): canonical
    for group in TEAM_ALIAS_GROUPS
    for canonical, alias in ((normalise_text(group[0]), alias) for alias in group)
}


def import_wyscout_export(
    csv_path: str,
    player_id: int | None,
    season: str,
    zone: str,
    *,
    player_name: str | None = None,
    team_name: str | None = None,
    league_id: int | None = None,
    source_player_external_id: str | None = None,
    persist_mapping: bool = True,
    threshold: float = 0.72,
) -> int:
    """Import a single-player Wyscout export into JSONB storage.

    The historical `zone` parameter name is retained for backward compatibility,
    but it stores the analyst-applied Wyscout filter label rather than a
    hard-coded pitch-zone taxonomy.
    """

    if not str(zone).strip():
        raise ValueError("Wyscout filter label cannot be empty")

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError("Wyscout export is empty")
    if len(frame.index) != 1:
        raise ValueError("Expected a single-player Wyscout export with exactly one row")

    source_row = frame.iloc[0].to_dict()
    resolved_player_name = player_name or _extract_source_value(source_row, PLAYER_NAME_ALIASES)
    resolved_team_name = team_name or _extract_source_value(source_row, TEAM_NAME_ALIASES)
    resolved_external_id = source_player_external_id or _extract_source_value(
        source_row,
        PLAYER_EXTERNAL_ID_ALIASES,
    )

    if player_id is None:
        player_id = resolve_source_player_id(
            "wyscout",
            resolved_player_name,
            source_team_name=resolved_team_name,
            source_player_external_id=resolved_external_id,
            league_id=league_id,
            threshold=threshold,
            persist_mapping=persist_mapping,
        )
        if player_id is None:
            raise ValueError(
                "Could not resolve Wyscout player to an internal player_id. "
                "Provide player_id explicitly or include a matchable player/team identity."
            )
    elif persist_mapping and (resolved_player_name or resolved_external_id):
        save_source_player_mapping(
            "wyscout",
            player_id=player_id,
            source_player_name=resolved_player_name,
            source_team_name=resolved_team_name,
            source_player_external_id=resolved_external_id,
            league_id=league_id,
            match_score=1.0,
            matched_by="manual",
        )

    metrics_json = _serialise_row(source_row)
    rows = [
        {
            "player_id": player_id,
            "season": str(season),
            "zone": zone,
            "metrics_json": metrics_json,
            "export_date": datetime.now(timezone.utc),
        }
    ]
    return upsert_rows(WyscoutZoneStat, rows, ["player_id", "season", "zone"])


def import_wyscout_league_folder(
    folder_path: str | Path,
    *,
    persist_mapping: bool = True,
    threshold: float = 0.72,
    unmatched_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Import season-average Wyscout workbooks for a single league folder."""

    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Wyscout league folder not found: {folder}")

    league_id = _infer_league_id_from_folder(folder)
    workbook_paths = sorted(path for path in folder.iterdir() if path.suffix.lower() in WORKBOOK_EXTENSIONS)
    if not workbook_paths:
        raise ValueError(f"No Wyscout workbooks found in {folder}")

    season_map = _assign_file_seasons(workbook_paths, league_id)
    all_rows: dict[tuple[int, str, int], dict[str, Any]] = {}
    unmatched_records: list[dict[str, Any]] = []
    duplicate_rows = 0
    file_summaries: list[dict[str, Any]] = []
    resolution_cache: dict[tuple[str, str, int, float], int | None] = {}
    historical_team_cache: dict[tuple[str, int], tuple[tuple[str, str], ...]] = {}
    historical_roster_cache: dict[tuple[str, int, str], tuple[tuple[int, str], ...]] = {}

    for workbook_path in workbook_paths:
        frame = _read_wyscout_frame(workbook_path)
        prepared = _prepare_season_rows(
            frame,
            workbook_path=workbook_path,
            league_id=league_id,
            season=season_map[workbook_path],
            persist_mapping=persist_mapping,
            threshold=threshold,
            unmatched_records=unmatched_records,
            resolution_cache=resolution_cache,
            historical_team_cache=historical_team_cache,
            historical_roster_cache=historical_roster_cache,
        )
        file_summaries.append(prepared["summary"])
        for key, row in prepared["rows"].items():
            if key in all_rows:
                duplicate_rows += 1
                all_rows[key] = _select_preferred_row(all_rows[key], row)
            else:
                all_rows[key] = row

    imported_rows = _upsert_season_rows(list(all_rows.values()))
    unmatched_path = _write_unmatched_records(
        folder.name,
        unmatched_records,
        output_dir=Path(unmatched_output_dir) if unmatched_output_dir is not None else None,
    )

    return {
        "folder": str(folder),
        "league_id": league_id,
        "season_files": {path.name: season for path, season in season_map.items()},
        "source_files": len(workbook_paths),
        "distinct_rows": len(all_rows),
        "duplicate_rows_merged": duplicate_rows,
        "imported_rows": imported_rows,
        "unmatched_rows": len(unmatched_records),
        "unmatched_output_path": str(unmatched_path) if unmatched_path else None,
        "files": file_summaries,
    }


def import_wyscout_root(
    root_path: str | Path,
    *,
    persist_mapping: bool = True,
    threshold: float = 0.72,
    folder_names: list[str] | None = None,
    unmatched_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Import all recognised Wyscout league folders under a root directory."""

    root = Path(root_path)
    if not root.is_dir():
        raise ValueError(f"Wyscout root folder not found: {root}")

    requested = {normalise_text(name) for name in folder_names or []}
    folders = [
        folder
        for folder in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name.lower())
        if normalise_text(folder.name) in LEAGUE_FOLDER_ALIASES
        and (not requested or normalise_text(folder.name) in requested)
    ]
    if not folders:
        raise ValueError(f"No recognised Wyscout league folders found in {root}")

    summaries = [
        import_wyscout_league_folder(
            folder,
            persist_mapping=persist_mapping,
            threshold=threshold,
            unmatched_output_dir=unmatched_output_dir,
        )
        for folder in folders
    ]
    return {
        "root": str(root),
        "folders_processed": len(summaries),
        "imported_rows": sum(summary["imported_rows"] for summary in summaries),
        "distinct_rows": sum(summary["distinct_rows"] for summary in summaries),
        "duplicate_rows_merged": sum(summary["duplicate_rows_merged"] for summary in summaries),
        "unmatched_rows": sum(summary["unmatched_rows"] for summary in summaries),
        "folders": summaries,
    }


def _prepare_season_rows(
    frame: pd.DataFrame,
    *,
    workbook_path: Path,
    league_id: int,
    season: str,
    persist_mapping: bool,
    threshold: float,
    unmatched_records: list[dict[str, Any]],
    resolution_cache: dict[tuple[str, str, int, float], int | None],
    historical_team_cache: dict[tuple[str, int], tuple[tuple[str, str], ...]],
    historical_roster_cache: dict[tuple[str, int, str], tuple[tuple[int, str], ...]],
) -> dict[str, Any]:
    rows: dict[tuple[int, str, int], dict[str, Any]] = {}
    duplicate_rows = 0
    source_rows = 0

    for record in frame.to_dict(orient="records"):
        source_rows += 1
        row = _serialise_row(record)
        player_name = _extract_source_value(row, PLAYER_NAME_ALIASES)
        timeframe_team = _extract_source_value(row, TEAM_TIMEFRAME_ALIASES) or _extract_source_value(row, TEAM_NAME_ALIASES)
        current_team = _extract_source_value(row, TEAM_NAME_ALIASES)
        cache_key = (
            normalise_text(player_name),
            normalise_text(timeframe_team),
            normalise_text(current_team),
            league_id,
            round(threshold, 4),
        )
        if cache_key in resolution_cache:
            player_id = resolution_cache[cache_key]
        else:
            player_id = _resolve_wyscout_player_id(
                player_name,
                timeframe_team,
                current_team_name=current_team,
                league_id=league_id,
                season=season,
                threshold=threshold,
                persist_mapping=persist_mapping,
                historical_team_cache=historical_team_cache,
                historical_roster_cache=historical_roster_cache,
            )
            resolution_cache[cache_key] = player_id
        if player_id is None:
            unmatched_records.append(
                {
                    "source_file": workbook_path.name,
                    "league_id": league_id,
                    "season": season,
                    "player_name": player_name,
                    "team_name": timeframe_team,
                    "current_team": current_team,
                }
            )
            continue

        season_row = {
            "player_id": player_id,
            "season": str(season),
            "league_id": league_id,
            "source_player_name": player_name,
            "source_team_name": timeframe_team,
            "current_team_name": current_team,
            "position": _extract_source_value(row, POSITION_ALIASES),
            "matches_played": _extract_int_value(row, MATCHES_PLAYED_ALIASES),
            "minutes_played": _extract_int_value(row, MINUTES_PLAYED_ALIASES),
            "metrics_json": row,
            "source_file": workbook_path.name,
            "import_date": datetime.now(timezone.utc),
        }
        key = (player_id, str(season), league_id)
        if key in rows:
            duplicate_rows += 1
            rows[key] = _select_preferred_row(rows[key], season_row)
        else:
            rows[key] = season_row

    return {
        "rows": rows,
        "summary": {
            "source_file": workbook_path.name,
            "season": str(season),
            "league_id": league_id,
            "source_rows": source_rows,
            "resolved_rows": len(rows),
            "duplicate_rows_merged": duplicate_rows,
            "unmatched_rows": source_rows - len(rows) - duplicate_rows,
        },
    }


def _resolve_wyscout_player_id(
    player_name: str | None,
    team_name: str | None,
    *,
    current_team_name: str | None,
    league_id: int,
    season: str,
    threshold: float,
    persist_mapping: bool,
    historical_team_cache: dict[tuple[str, int], tuple[tuple[str, str], ...]],
    historical_roster_cache: dict[tuple[str, int, str], tuple[tuple[int, str], ...]],
) -> int | None:
    player_id = resolve_source_player_id(
        "wyscout",
        player_name,
        source_team_name=team_name,
        league_id=league_id,
        threshold=threshold,
        persist_mapping=persist_mapping,
    )
    if player_id is not None:
        return player_id

    if current_team_name and normalise_text(current_team_name) != normalise_text(team_name):
        player_id = resolve_source_player_id(
            "wyscout",
            player_name,
            source_team_name=current_team_name,
            league_id=None,
            threshold=threshold,
            persist_mapping=persist_mapping,
        )
        if player_id is not None:
            return player_id

    historical_match = _resolve_via_historical_roster(
        player_name,
        team_name,
        season=season,
        league_id=league_id,
        historical_team_cache=historical_team_cache,
        historical_roster_cache=historical_roster_cache,
    )
    if historical_match is None:
        return None

    if persist_mapping:
        save_source_player_mapping(
            "wyscout",
            player_id=historical_match["player_id"],
            source_player_name=player_name,
            source_team_name=team_name,
            league_id=league_id,
            match_score=historical_match["score"],
            matched_by="season_team_roster",
        )
    return historical_match["player_id"]


def _resolve_via_historical_roster(
    player_name: str | None,
    team_name: str | None,
    *,
    season: str,
    league_id: int,
    historical_team_cache: dict[tuple[str, int], tuple[tuple[str, str], ...]],
    historical_roster_cache: dict[tuple[str, int, str], tuple[tuple[int, str], ...]],
) -> dict[str, Any] | None:
    if not player_name or not team_name:
        return None

    historical_team = _match_historical_team_name(
        season,
        league_id,
        team_name,
        historical_team_cache=historical_team_cache,
    )
    if historical_team is None:
        return None

    roster = _load_historical_roster(
        season,
        league_id,
        historical_team,
        historical_roster_cache=historical_roster_cache,
    )
    if not roster:
        return None

    ranked = sorted(
        (
            {
                "player_id": candidate_player_id,
                "player_name": candidate_name,
                "score": _score_roster_candidate(player_name, candidate_name),
            }
            for candidate_player_id, candidate_name in roster
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    best = ranked[0]
    second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
    if best["score"] < 0.88:
        return None
    if best["score"] < 0.94 and (best["score"] - second_score) < 0.05:
        return None
    return best


def _match_historical_team_name(
    season: str,
    league_id: int,
    source_team_name: str,
    *,
    historical_team_cache: dict[tuple[str, int], tuple[tuple[str, str], ...]],
) -> str | None:
    candidates = historical_team_cache.get((str(season), league_id))
    if candidates is None:
        candidates = _load_historical_team_names(str(season), league_id)
        historical_team_cache[(str(season), league_id)] = candidates
    if not candidates:
        return None

    ranked = sorted(
        (
            (team_name, _score_team_name(source_team_name, team_name))
            for team_name, _normalized in candidates
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    best_team, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score < 0.85:
        return None
    if best_score < 0.98 and (best_score - second_score) < 0.05:
        return None
    return best_team


def _load_historical_team_names(season: str, league_id: int) -> tuple[tuple[str, str], ...]:
    with session_scope() as session:
        rows = session.execute(
            select(MatchPerformance.team).where(
                MatchPerformance.season == str(season),
                MatchPerformance.league_id == league_id,
            ).distinct()
        ).all()
    teams = sorted((row.team, normalise_text(row.team)) for row in rows if row.team)
    return tuple(teams)


def _load_historical_roster(
    season: str,
    league_id: int,
    team_name: str,
    *,
    historical_roster_cache: dict[tuple[str, int, str], tuple[tuple[int, str], ...]],
) -> tuple[tuple[int, str], ...]:
    cache_key = (str(season), league_id, team_name)
    cached = historical_roster_cache.get(cache_key)
    if cached is not None:
        return cached

    with session_scope() as session:
        performance_rows = session.execute(
            select(Player.player_id, Player.player_name)
            .join(MatchPerformance, MatchPerformance.player_id == Player.player_id)
            .where(
                MatchPerformance.season == str(season),
                MatchPerformance.league_id == league_id,
                MatchPerformance.team == team_name,
            )
            .distinct()
        ).all()
        lineup_rows = session.execute(
            select(Player.player_id, Player.player_name)
            .join(Lineup, Lineup.player_id == Player.player_id)
            .join(Fixture, Fixture.fixture_id == Lineup.fixture_id)
            .where(
                Fixture.season == str(season),
                Fixture.league_id == league_id,
                Lineup.team == team_name,
            )
            .distinct()
        ).all()

    merged = {(row.player_id, row.player_name) for row in performance_rows}
    merged.update((row.player_id, row.player_name) for row in lineup_rows)
    roster = tuple(sorted(merged, key=lambda item: (normalise_text(item[1]), item[0])))
    historical_roster_cache[cache_key] = roster
    return roster


def _score_team_name(source_team_name: str, candidate_team_name: str) -> float:
    source = _canonical_team_name(source_team_name)
    candidate = _canonical_team_name(candidate_team_name)
    if not source or not candidate:
        return 0.0
    if source == candidate:
        return 1.0
    if source in candidate or candidate in source:
        return 0.92
    return SequenceMatcher(None, source, candidate).ratio()


def _score_roster_candidate(source_player_name: str, candidate_player_name: str) -> float:
    source = normalise_text(source_player_name)
    candidate = normalise_text(candidate_player_name)
    if not source or not candidate:
        return 0.0

    score = SequenceMatcher(None, source, candidate).ratio()
    source_tokens = source.split()
    candidate_tokens = candidate.split()
    source_meaningful = {token for token in source_tokens if len(token) > 1}
    candidate_meaningful = {token for token in candidate_tokens if len(token) > 1}
    shared_meaningful = source_meaningful & candidate_meaningful
    if shared_meaningful:
        if source_tokens and candidate_tokens and source_tokens[0][:1] == candidate_tokens[0][:1]:
            score = max(score, 0.9)
        else:
            score = max(score, 0.83)
    if sorted(source_tokens) == sorted(candidate_tokens):
        score = max(score, 0.95)
    if source_tokens and candidate_tokens:
        source_initial = source_tokens[0][:1]
        candidate_initial = candidate_tokens[0][:1]
        if source_initial == candidate_initial and source_tokens[-1] == candidate_tokens[-1]:
            score = max(score, 0.9)
        elif source_tokens[-1] == candidate_tokens[-1]:
            score = max(score, 0.82)
        elif source_tokens[-1] in candidate_tokens or candidate_tokens[-1] in source_tokens:
            if source_initial == candidate_initial:
                score = max(score, 0.89)
            else:
                score = max(score, 0.84)
        if len(source_tokens) >= 2 and len(candidate_tokens) >= 2:
            if source_tokens[0] == candidate_tokens[-1] and source_tokens[-1] == candidate_tokens[0]:
                score = max(score, 0.88)
    return score


def _canonical_team_name(team_name: str | None) -> str:
    normalized = normalise_text(team_name)
    return TEAM_ALIAS_MAP.get(normalized, normalized)


def _select_preferred_row(existing: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if _row_rank(candidate) > _row_rank(existing):
        return candidate
    return existing


def _row_rank(row: dict[str, Any]) -> tuple[int, int, int]:
    metrics = row.get("metrics_json", {})
    non_null_values = sum(1 for value in metrics.values() if value is not None)
    return (
        non_null_values,
        int(row.get("minutes_played") or 0),
        int(row.get("matches_played") or 0),
    )


def _upsert_season_rows(rows: list[dict[str, Any]]) -> int:
    total = 0
    for start in range(0, len(rows), UPSERT_BATCH_SIZE):
        batch = rows[start : start + UPSERT_BATCH_SIZE]
        total += upsert_rows(WyscoutSeasonStat, batch, ["player_id", "season", "league_id"])
    return total


def _assign_file_seasons(workbook_paths: list[Path], league_id: int) -> dict[Path, str]:
    assigned: dict[Path, str] = {}
    ambiguous_paths: list[Path] = []

    for workbook_path in workbook_paths:
        season = _infer_season_from_filename(workbook_path.name)
        if season is None:
            ambiguous_paths.append(workbook_path)
        else:
            assigned[workbook_path] = season

    if not ambiguous_paths:
        return assigned

    season_team_map = _load_league_team_names_by_season(league_id)
    remaining_seasons = [season for season in sorted(season_team_map) if season not in set(assigned.values())]
    if len(ambiguous_paths) > len(remaining_seasons):
        raise ValueError(
            f"Could not infer seasons for {len(ambiguous_paths)} Wyscout files in league {league_id}; "
            f"only {len(remaining_seasons)} candidate seasons remain"
        )

    overlap_scores: dict[Path, dict[str, int]] = {}
    for workbook_path in ambiguous_paths:
        frame = _read_wyscout_frame(workbook_path)
        team_names = _extract_team_names(frame)
        overlap_scores[workbook_path] = {
            season: len(team_names & season_team_map[season])
            for season in remaining_seasons
        }

    best_score = -1
    best_mapping: dict[Path, str] | None = None
    for season_assignment in permutations(remaining_seasons, len(ambiguous_paths)):
        score = sum(
            overlap_scores[workbook_path][season]
            for workbook_path, season in zip(ambiguous_paths, season_assignment, strict=True)
        )
        if score > best_score:
            best_score = score
            best_mapping = {
                workbook_path: season
                for workbook_path, season in zip(ambiguous_paths, season_assignment, strict=True)
            }

    if best_mapping is None or best_score <= 0:
        raise ValueError(f"Could not infer seasons for Wyscout files in {workbook_paths[0].parent}")

    assigned.update(best_mapping)
    return assigned


def _infer_league_id_from_folder(folder: Path) -> int:
    folder_key = normalise_text(folder.name)
    league_id = LEAGUE_FOLDER_ALIASES.get(folder_key)
    if league_id is None:
        raise ValueError(f"Unrecognised Wyscout league folder name: {folder.name}")
    return league_id


def _infer_season_from_filename(filename: str) -> str | None:
    match = SEASON_PATTERN.search(filename)
    if match is None:
        return None

    start_year = int(match.group("start"))
    return str(2000 + start_year)


def _load_league_team_names_by_season(league_id: int) -> dict[str, set[str]]:
    with session_scope() as session:
        rows = session.execute(
            select(Fixture.season, Fixture.home_team, Fixture.away_team).where(Fixture.league_id == league_id)
        ).all()

    season_team_map: dict[str, set[str]] = {}
    for season, home_team, away_team in rows:
        season_team_map.setdefault(str(season), set()).add(normalise_text(home_team))
        season_team_map.setdefault(str(season), set()).add(normalise_text(away_team))
    return season_team_map


def _extract_team_names(frame: pd.DataFrame) -> set[str]:
    row_count = len(frame.index)
    if row_count == 0:
        return set()

    records = frame.to_dict(orient="records")
    team_names = set()
    for record in records:
        team_name = _extract_source_value(record, TEAM_TIMEFRAME_ALIASES) or _extract_source_value(record, TEAM_NAME_ALIASES)
        if team_name:
            team_names.add(normalise_text(team_name))
    return team_names


def _read_wyscout_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported Wyscout file type: {path.suffix}")

    frame = frame.dropna(how="all")
    return frame.reset_index(drop=True)


def _write_unmatched_records(
    folder_name: str,
    rows: list[dict[str, Any]],
    *,
    output_dir: Path | None,
) -> Path | None:
    if not rows:
        return None

    settings = get_settings()
    target_dir = output_dir or (settings.data_dir / "wyscout" / "unmatched")
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = target_dir / f"{normalise_text(folder_name).replace(' ', '_')}_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def _extract_source_value(row: dict[str, Any], aliases: set[str]) -> str | None:
    for key, value in row.items():
        if _normalise_column_name(key) not in aliases:
            continue
        if pd.isna(value):
            continue
        resolved = value.item() if hasattr(value, "item") else value
        text = str(resolved).strip()
        if text:
            return text
    return None


def _extract_int_value(row: dict[str, Any], aliases: set[str]) -> int | None:
    raw_value = _extract_source_value(row, aliases)
    if raw_value in (None, ""):
        return None
    try:
        return int(float(raw_value))
    except ValueError:
        return None


def _normalise_column_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = "".join(character if character.isalnum() else "_" for character in text)
    return "_".join(segment for segment in text.split("_") if segment)


def _serialise_row(row: dict[str, Any]) -> dict[str, Any]:
    serialised: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            serialised[key] = None
        elif hasattr(value, "item"):
            serialised[key] = value.item()
        elif isinstance(value, Path):
            serialised[key] = str(value)
        else:
            serialised[key] = value
    return serialised


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Wyscout workbook exports.")
    parser.add_argument("path", nargs="?", default=None, help="League folder or root folder containing league folders")
    parser.add_argument("--folder", dest="folders", action="append", default=None, help="Specific league folder name to import")
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--no-persist-mapping", action="store_true")
    parser.add_argument("--unmatched-output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    path = Path(args.path) if args.path else settings.wyscout_import_dir

    if path.is_dir() and normalise_text(path.name) in LEAGUE_FOLDER_ALIASES:
        result = import_wyscout_league_folder(
            path,
            persist_mapping=not args.no_persist_mapping,
            threshold=args.threshold,
            unmatched_output_dir=args.unmatched_output_dir,
        )
    else:
        result = import_wyscout_root(
            path,
            persist_mapping=not args.no_persist_mapping,
            threshold=args.threshold,
            folder_names=args.folders,
            unmatched_output_dir=args.unmatched_output_dir,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
