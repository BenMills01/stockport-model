"""FBref ingestion for expected metrics and progressive actions."""

from __future__ import annotations

from typing import Any

from bs4 import BeautifulSoup, Comment, Tag
import pandas as pd
import requests

from config import get_settings
from db.schema import ExpectedMetric
from ingestion.common import upsert_rows
from ingestion.matching import match_player_id


FBREF_TABLE_IDS = {
    "standard": "stats_standard",
    "shooting": "stats_shooting",
    "passing": "stats_passing",
    "possession": "stats_possession",
}


def scrape_fbref_player_stats(league_url: str, season: str) -> pd.DataFrame:
    """Scrape FBref player tables and return a merged metrics dataframe."""

    html = _fetch_fbref_html(league_url)
    tables = {
        name: _read_fbref_table(html, table_id)
        for name, table_id in FBREF_TABLE_IDS.items()
    }

    base = _extract_columns(
        tables["standard"],
        {"player_name": ["player"], "team_name": ["squad", "team"]},
    )
    shooting = _extract_columns(
        tables["shooting"],
        {
            "player_name": ["player"],
            "team_name": ["squad", "team"],
            "xg": ["performance_xg", "expected_xg", "xg"],
            "npxg": ["performance_npxg", "expected_npxg", "npxg"],
            "xg_per_shot": ["standard_xg_sh", "xg_sh", "performance_xg_sh"],
        },
    )
    passing = _extract_columns(
        tables["passing"],
        {
            "player_name": ["player"],
            "team_name": ["squad", "team"],
            "xa": ["expected_xag", "xag", "xa", "expected_xa"],
            "progressive_passes": ["total_prgp", "prgp", "progression_prgp"],
        },
    )
    possession = _extract_columns(
        tables["possession"],
        {
            "player_name": ["player"],
            "team_name": ["squad", "team"],
            "progressive_carries": ["carries_prgc", "prgc", "carry_prgc"],
            "progressive_receptions": ["receiving_prgr", "prgr", "receive_prgr"],
        },
    )

    merged = base.merge(shooting, on=["player_name", "team_name"], how="left")
    merged = merged.merge(passing, on=["player_name", "team_name"], how="left")
    merged = merged.merge(possession, on=["player_name", "team_name"], how="left")

    merged["season"] = season
    metric_columns = [
        "xg",
        "npxg",
        "xa",
        "xg_per_shot",
        "progressive_passes",
        "progressive_carries",
        "progressive_receptions",
    ]
    for column in metric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    return merged[
        ["player_name", "team_name", "season", *metric_columns]
    ].sort_values(["team_name", "player_name"]).reset_index(drop=True)


def ingest_fbref_player_stats(league_url: str, season: str, league_id: int) -> int:
    """Scrape a league page and upsert expected metrics into the database."""

    data = scrape_fbref_player_stats(league_url, season)
    rows: list[dict[str, Any]] = []
    for record in data.to_dict(orient="records"):
        player_id = match_player_id(
            record["player_name"],
            team_name=record.get("team_name"),
            league_id=league_id,
        )
        if player_id is None:
            continue

        rows.append(
            {
                "player_id": player_id,
                "season": str(record["season"]),
                "source": "fbref",
                "league_id": league_id,
                "xg": _none_if_nan(record.get("xg")),
                "npxg": _none_if_nan(record.get("npxg")),
                "xa": _none_if_nan(record.get("xa")),
                "xg_per_shot": _none_if_nan(record.get("xg_per_shot")),
                "goals_minus_xg": None,
                "assists_minus_xa": None,
                "progressive_passes": _none_if_nan(record.get("progressive_passes")),
                "progressive_carries": _none_if_nan(record.get("progressive_carries")),
                "progressive_receptions": _none_if_nan(record.get("progressive_receptions")),
            }
        )

    return upsert_rows(ExpectedMetric, rows, ["player_id", "season", "source"])


def _fetch_fbref_html(league_url: str) -> str:
    settings = get_settings()
    response = requests.get(
        league_url,
        headers={"User-Agent": settings.http_user_agent},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def _read_fbref_table(html: str, table_id: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id=table_id)
    if table is None:
        for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
            if table_id not in comment:
                continue
            comment_soup = BeautifulSoup(str(comment), "html.parser")
            table = comment_soup.find("table", id=table_id)
            if table is not None:
                break
    if table is None:
        raise ValueError(f"Could not find FBref table '{table_id}'")

    return _clean_fbref_frame(_table_to_frame(table))


def _clean_fbref_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = _flatten_columns(frame.columns)

    player_column = _resolve_column(frame, ["player"])
    if player_column:
        frame = frame[frame[player_column].astype(str).str.strip().ne("Player")]
    squad_column = _resolve_column(frame, ["squad", "team"])
    if squad_column:
        frame = frame[frame[squad_column].astype(str).str.strip().ne("Squad")]
    return frame.reset_index(drop=True)


def _flatten_columns(columns: Any) -> list[str]:
    flattened: list[str] = []
    for column in columns:
        if isinstance(column, tuple):
            parts = [str(part).strip() for part in column if str(part).strip() and "Unnamed" not in str(part)]
            name = "_".join(parts)
        else:
            name = str(column).strip()
        name = name.lower().replace("%", "pct").replace("/", "_")
        name = "".join(character if character.isalnum() or character == "_" else "_" for character in name)
        name = "_".join(segment for segment in name.split("_") if segment)
        flattened.append(name)
    return flattened


def _table_to_frame(table: Tag) -> pd.DataFrame:
    rows = []
    for row in table.select("tbody tr"):
        parsed = _parse_row_from_data_stat(row)
        if parsed:
            rows.append(parsed)
    if rows:
        return pd.DataFrame(rows)

    headers = _combine_header_rows(table)
    body_rows = [_expand_row_cells(row.find_all(["th", "td"])) for row in table.select("tbody tr")]
    return pd.DataFrame(body_rows, columns=headers)


def _parse_row_from_data_stat(row: Tag) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for cell in row.find_all(["th", "td"], recursive=False):
        key = cell.get("data-stat")
        if key:
            parsed[key] = cell.get_text(" ", strip=True)
    return parsed


def _combine_header_rows(table: Tag) -> list[str]:
    header_rows = [_expand_row_cells(row.find_all(["th", "td"])) for row in table.select("thead tr")]
    if not header_rows:
        return []

    width = max(len(row) for row in header_rows)
    padded = [row + [""] * (width - len(row)) for row in header_rows]
    headers: list[str] = []
    for index in range(width):
        parts: list[str] = []
        for row in padded:
            text = row[index].strip()
            if text and (not parts or parts[-1] != text):
                parts.append(text)
        headers.append("_".join(parts) if parts else f"column_{index}")
    return headers


def _expand_row_cells(cells: list[Tag]) -> list[str]:
    expanded: list[str] = []
    for cell in cells:
        colspan = int(cell.get("colspan", 1))
        text = cell.get_text(" ", strip=True)
        expanded.extend([text] * colspan)
    return expanded


def _extract_columns(frame: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    selected: dict[str, str] = {}
    for target, alias_list in aliases.items():
        source = _resolve_column(frame, alias_list)
        if source is None:
            if target in {"player_name", "team_name"}:
                raise KeyError(f"Required FBref column missing for {target}")
            continue
        selected[target] = source

    result = frame[[*selected.values()]].rename(columns={value: key for key, value in selected.items()})
    for target in aliases:
        if target not in result:
            result[target] = None
    return result[[*aliases.keys()]]


def _resolve_column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    normalised_columns = {column.lower(): column for column in frame.columns}
    for alias in aliases:
        if alias in normalised_columns:
            return normalised_columns[alias]
    return None


def _none_if_nan(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)
