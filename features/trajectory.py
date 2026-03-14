"""Career trajectory and age-curve features."""

from __future__ import annotations

from collections import Counter
from datetime import date
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from config import get_settings
from db.read_cache import load_player_match_frame, load_player_row, load_player_transfer_frame
from features.per90 import _compute_per90_frame


AGE_CURVE_GROUPS = {
    "g": "goalkeeper",
    "cb": "centre_back",
    "c": "centre_back",
    "d": "centre_back",
    "rb": "full_back_wing_back",
    "lb": "full_back_wing_back",
    "wb": "full_back_wing_back",
    "dm": "central_midfielder",
    "cm": "central_midfielder",
    "am": "central_midfielder",
    "m": "central_midfielder",
    "rw": "wide_attacker_winger",
    "lw": "wide_attacker_winger",
    "w": "wide_attacker_winger",
    "f": "striker",
    "st": "striker",
}

TRAJECTORY_METRICS = [
    "goals_scored_per90",
    "assists_per90",
    "shots_total_per90",
    "passes_key_per90",
    "tackles_interceptions_per90",
    "duels_won_per90",
]


@lru_cache(maxsize=4096)
def compute_trajectory_features(player_id: int) -> dict[str, Any]:
    """Compute historical trajectory features for a player."""

    player_dict = load_player_row(player_id)
    transfer_frame = load_player_transfer_frame(player_id).copy()
    match_frame = load_player_match_frame(player_id).copy()
    return _compute_trajectory_features_from_frames(
        player_frame=player_dict,
        transfer_frame=transfer_frame,
        match_frame=match_frame,
        leagues=get_settings().load_json("leagues.json"),
        age_curves=get_settings().load_json("age_curves.json"),
        today=date.today(),
    )


def _compute_trajectory_features_from_frames(
    *,
    player_frame: dict[str, Any],
    transfer_frame: pd.DataFrame,
    match_frame: pd.DataFrame,
    leagues: list[dict[str, Any]],
    age_curves: dict[str, Any],
    today: date,
) -> dict[str, Any]:
    league_lookup = {
        int(league["league_id"]): {"tier": int(league["tier"]), "country": league["country"]}
        for league in leagues
    }

    if match_frame.empty:
        return {
            "league_level_history": [],
            "loan_count": 0,
            "clubs_count": 0,
            "countries_played_in": 0,
            "age_at_first_senior_appearance": None,
            "seasons_at_current_club": 0,
            "output_trajectory_2yr": None,
            "age_curve_position": None,
        }

    match_frame = match_frame.copy()
    match_frame["date"] = pd.to_datetime(match_frame["date"])
    per90 = _compute_per90_frame(match_frame)
    season_summary = per90.groupby("season", as_index=False).agg(
        league_id=("league_id", _mode_value),
        team=("team", _mode_value),
        **{metric: (metric, "mean") for metric in TRAJECTORY_METRICS if metric in per90.columns},
    )
    league_level_history = [
        {
            "season": str(row["season"]),
            "tier": league_lookup.get(int(row["league_id"]), {}).get("tier"),
        }
        for _, row in season_summary.sort_values("season").iterrows()
    ]

    clubs = set()
    if not transfer_frame.empty:
        clubs.update(value for value in transfer_frame["team_in"].dropna().tolist() if value)
        clubs.update(value for value in transfer_frame["team_out"].dropna().tolist() if value)
    if player_frame.get("current_team"):
        clubs.add(player_frame["current_team"])

    countries = set()
    for league_id in season_summary["league_id"].dropna().tolist():
        country = league_lookup.get(int(league_id), {}).get("country")
        if country:
            countries.add(country)

    loan_count = 0
    if not transfer_frame.empty and "type" in transfer_frame:
        loan_count = int(
            transfer_frame["type"].fillna("").astype(str).str.lower().str.contains("loan").sum()
        )

    age_at_first_senior_appearance = None
    birth_date = player_frame.get("birth_date")
    if birth_date is not None:
        first_match_date = match_frame["date"].min().date()
        age_at_first_senior_appearance = round((first_match_date - birth_date).days / 365.25, 2)

    seasons_at_current_club = _seasons_at_current_club(
        season_summary.sort_values("season"),
        current_team=player_frame.get("current_team"),
    )
    output_trajectory_2yr = _output_trajectory_two_years(season_summary.sort_values("season"))
    age_curve_position = _age_curve_position(
        player_frame=player_frame,
        match_frame=match_frame,
        age_curves=age_curves,
        today=today,
    )

    return {
        "league_level_history": league_level_history,
        "loan_count": loan_count,
        "clubs_count": len(clubs),
        "countries_played_in": len(countries),
        "age_at_first_senior_appearance": age_at_first_senior_appearance,
        "seasons_at_current_club": seasons_at_current_club,
        "output_trajectory_2yr": output_trajectory_2yr,
        "age_curve_position": age_curve_position,
    }


def _seasons_at_current_club(season_summary: pd.DataFrame, current_team: str | None) -> int:
    if season_summary.empty or not current_team:
        return 0
    count = 0
    for _, row in season_summary.sort_values("season", ascending=False).iterrows():
        if row["team"] == current_team:
            count += 1
        else:
            break
    return count


def _output_trajectory_two_years(season_summary: pd.DataFrame) -> float | None:
    if season_summary.empty:
        return None
    tail = season_summary.tail(2).copy()
    metric_columns = [column for column in tail.columns if column in TRAJECTORY_METRICS]
    metric_slopes = []
    for metric in metric_columns:
        series = pd.to_numeric(tail[metric], errors="coerce").dropna()
        if len(series.index) < 2:
            continue
        x_values = np.arange(len(series), dtype=float)
        metric_slopes.append(float(np.polyfit(x_values, series.to_numpy(dtype=float), 1)[0]))
    if not metric_slopes:
        return None
    return float(sum(metric_slopes) / len(metric_slopes))


def _age_curve_position(
    *,
    player_frame: dict[str, Any],
    match_frame: pd.DataFrame,
    age_curves: dict[str, Any],
    today: date,
) -> str | None:
    birth_date = player_frame.get("birth_date")
    if birth_date is None:
        return None
    age = round((today - birth_date).days / 365.25, 2)
    position_group = _infer_age_curve_group(match_frame)
    if position_group is None:
        return None
    curve = age_curves[position_group]
    output_peak_start = float(curve["output_peak"][0])
    decline_start = float(curve["decline_onset"][0])
    if age < output_peak_start:
        return "pre_peak"
    if age >= decline_start:
        return "post_peak"
    return "peak"


def _infer_age_curve_group(match_frame: pd.DataFrame) -> str | None:
    labels = [
        str(value).strip().lower()
        for value in match_frame["position"].dropna().tolist()
        if str(value).strip()
    ]
    if not labels:
        return None
    label = Counter(labels).most_common(1)[0][0]
    for token, curve_group in AGE_CURVE_GROUPS.items():
        if label.startswith(token) or token in label:
            return curve_group
    return None


def _mode_value(series: pd.Series) -> Any:
    modes = series.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return series.dropna().iloc[0] if not series.dropna().empty else None
