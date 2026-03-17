"""SkillCorner-derived physical and game-intelligence features."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from db.schema import (
    SkillCornerOffBallRuns,
    SkillCornerPasses,
    SkillCornerPhysical,
    SkillCornerPressure,
)
from db.session import session_scope


# ---------------------------------------------------------------------------
# Physical metric columns we aggregate (per-match normalisation)
# ---------------------------------------------------------------------------
_PHYSICAL_METRICS = [
    "dist_per_match",
    "hsr_dist_per_match",
    "sprint_dist_per_match",
    "count_hsr_per_match",
    "count_sprint_per_match",
    "count_high_accel_per_match",
    "count_high_decel_per_match",
    "top_speed_per_match",
    "dist_tip_per_match",
    "dist_otip_per_match",
    "hsr_dist_p90",
    "sprint_dist_p90",
]

_OFF_BALL_METRICS = [
    "count_run_in_behind_in_sample",
    "count_dangerous_run_in_behind_per_match",
    "run_in_behind_threat_per_match",
    "count_run_in_behind_targeted_per_match",
    "count_run_in_behind_received_per_match",
    "run_in_behind_targeted_threat_per_match",
    "run_in_behind_received_threat_per_match",
    "count_dangerous_run_in_behind_targeted_per_match",
    "count_dangerous_run_in_behind_received_per_match",
]

_PRESSURE_METRICS = [
    "count_high_pressure_received_per_match",
    "ball_retention_ratio_under_high_pressure",
    "ball_retention_added_under_high_pressure_per_match",
    "pass_completion_ratio_under_high_pressure",
    "dangerous_pass_completion_ratio_under_high_pressure",
    "difficult_pass_completion_ratio_under_high_pressure",
]

_PASSES_METRICS = [
    "count_opportunities_to_pass_to_run_in_behind_per_match",
    "count_pass_attempts_to_run_in_behind_per_match",
    "pass_completion_ratio_to_run_in_behind",
    "count_completed_pass_to_run_in_behind_per_match",
    "count_completed_pass_to_run_in_behind_leading_to_shot_per_match",
    "count_completed_pass_to_run_in_behind_leading_to_goal_per_match",
    "count_pass_opportunities_to_dangerous_run_in_behind_per_match",
    "count_pass_attempts_to_dangerous_run_in_behind_per_match",
    "count_completed_pass_to_dangerous_run_in_behind_per_match",
]

# How many recent matches to use for the rolling window
_RECENT_N = 5


@lru_cache(maxsize=4096)
def compute_skillcorner_features(player_id: int) -> dict[str, Any]:
    """Return a flat dict of SkillCorner-derived features for a player.

    Returns an empty dict (with all keys set to None) if the player has no
    SkillCorner data.  Features are grouped into four blocks:

    - ``sc_physical_*``   — physical output (distance, HSR, sprint, accel/decel, speed)
    - ``sc_off_ball_*``   — in-possession off-ball running threat and counts
    - ``sc_pressure_*``   — on-ball retention and passing quality under pressure
    - ``sc_passes_*``     — passing to runs in behind and opportunity exploitation
    """

    physical_frame = _load_physical(player_id)
    off_ball_frame = _load_off_ball_runs(player_id)
    pressure_frame = _load_pressure(player_id)
    passes_frame = _load_passes(player_id)

    features: dict[str, Any] = {}
    features.update(_aggregate_physical(physical_frame))
    features.update(_aggregate_off_ball(off_ball_frame))
    features.update(_aggregate_pressure(pressure_frame))
    features.update(_aggregate_passes(passes_frame))

    # Overall data availability flag
    features["sc_has_physical"] = not physical_frame.empty
    features["sc_has_gi"] = not (
        off_ball_frame.empty and pressure_frame.empty and passes_frame.empty
    )
    features["sc_match_count"] = int(
        physical_frame["sc_match_id"].nunique()
        if not physical_frame.empty
        else 0
    )

    return features


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_physical(player_id: int) -> pd.DataFrame:
    with session_scope() as session:
        rows = list(
            session.scalars(
                select(SkillCornerPhysical)
                .where(SkillCornerPhysical.player_id == player_id)
                .order_by(SkillCornerPhysical.match_date.asc())
            )
        )
    return _to_frame(rows, SkillCornerPhysical)


def _load_off_ball_runs(player_id: int) -> pd.DataFrame:
    with session_scope() as session:
        rows = list(
            session.scalars(
                select(SkillCornerOffBallRuns)
                .where(SkillCornerOffBallRuns.player_id == player_id)
                .order_by(SkillCornerOffBallRuns.match_date.asc())
            )
        )
    return _to_frame(rows, SkillCornerOffBallRuns)


def _load_pressure(player_id: int) -> pd.DataFrame:
    with session_scope() as session:
        rows = list(
            session.scalars(
                select(SkillCornerPressure)
                .where(SkillCornerPressure.player_id == player_id)
                .order_by(SkillCornerPressure.match_date.asc())
            )
        )
    return _to_frame(rows, SkillCornerPressure)


def _load_passes(player_id: int) -> pd.DataFrame:
    with session_scope() as session:
        rows = list(
            session.scalars(
                select(SkillCornerPasses)
                .where(SkillCornerPasses.player_id == player_id)
                .order_by(SkillCornerPasses.match_date.asc())
            )
        )
    return _to_frame(rows, SkillCornerPasses)


# ---------------------------------------------------------------------------
# Aggregation blocks
# ---------------------------------------------------------------------------


def _aggregate_physical(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}

    if frame.empty:
        for metric in _PHYSICAL_METRICS:
            key = f"sc_physical_{metric}"
            out[key] = None
            out[f"{key}_recent{_RECENT_N}"] = None
        out["sc_physical_top_speed_max"] = None
        out["sc_physical_sample_matches"] = 0
        return out

    # Only use rows where quality_check is not False
    frame = frame[frame["quality_check"].fillna(True) != False].copy()  # noqa: E712

    out["sc_physical_sample_matches"] = int(frame["sc_match_id"].nunique())

    for metric in _PHYSICAL_METRICS:
        if metric not in frame.columns:
            out[f"sc_physical_{metric}"] = None
            out[f"sc_physical_{metric}_recent{_RECENT_N}"] = None
            continue
        series = pd.to_numeric(frame[metric], errors="coerce").dropna()
        out[f"sc_physical_{metric}"] = _safe_mean(series)
        out[f"sc_physical_{metric}_recent{_RECENT_N}"] = _safe_mean(series.tail(_RECENT_N))

    # Peak top speed (max, not mean — a counting stat)
    if "top_speed_per_match" in frame.columns:
        series = pd.to_numeric(frame["top_speed_per_match"], errors="coerce").dropna()
        out["sc_physical_top_speed_max"] = float(series.max()) if not series.empty else None
    else:
        out["sc_physical_top_speed_max"] = None

    # Physical intensity index: HSR + sprint dist combined, season average
    hsr = pd.to_numeric(frame.get("hsr_dist_per_match"), errors="coerce").fillna(0)
    sprint = pd.to_numeric(frame.get("sprint_dist_per_match"), errors="coerce").fillna(0)
    intensity = hsr + sprint
    out["sc_physical_intensity_index"] = _safe_mean(intensity.replace(0, float("nan")).dropna())

    # Trend slope on HSR (last 10 matches)
    hsr_series = pd.to_numeric(frame.get("hsr_dist_per_match"), errors="coerce").dropna()
    out["sc_physical_hsr_trend_10"] = _trend_slope(hsr_series.tail(10))

    return out


def _aggregate_off_ball(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}

    if frame.empty:
        for metric in _OFF_BALL_METRICS:
            out[f"sc_off_ball_{metric}"] = None
            out[f"sc_off_ball_{metric}_recent{_RECENT_N}"] = None
        out["sc_off_ball_sample_matches"] = 0
        out["sc_off_ball_run_exploitation_rate"] = None
        return out

    frame = frame[frame["quality_check"].fillna(True) != False].copy()  # noqa: E712
    out["sc_off_ball_sample_matches"] = int(frame["sc_match_id"].nunique())

    for metric in _OFF_BALL_METRICS:
        if metric not in frame.columns:
            out[f"sc_off_ball_{metric}"] = None
            out[f"sc_off_ball_{metric}_recent{_RECENT_N}"] = None
            continue
        series = pd.to_numeric(frame[metric], errors="coerce").dropna()
        out[f"sc_off_ball_{metric}"] = _safe_mean(series)
        out[f"sc_off_ball_{metric}_recent{_RECENT_N}"] = _safe_mean(series.tail(_RECENT_N))

    # Run exploitation rate: received / targeted (how often the pass finds the runner)
    targeted = pd.to_numeric(frame.get("count_run_in_behind_targeted_per_match"), errors="coerce")
    received = pd.to_numeric(frame.get("count_run_in_behind_received_per_match"), errors="coerce")
    combined = pd.concat([targeted.rename("t"), received.rename("r")], axis=1).dropna()
    if not combined.empty and combined["t"].sum() > 0:
        out["sc_off_ball_run_exploitation_rate"] = float(
            combined["r"].sum() / combined["t"].sum()
        )
    else:
        out["sc_off_ball_run_exploitation_rate"] = None

    return out


def _aggregate_pressure(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}

    if frame.empty:
        for metric in _PRESSURE_METRICS:
            out[f"sc_pressure_{metric}"] = None
            out[f"sc_pressure_{metric}_recent{_RECENT_N}"] = None
        out["sc_pressure_sample_matches"] = 0
        return out

    frame = frame[frame["quality_check"].fillna(True) != False].copy()  # noqa: E712
    out["sc_pressure_sample_matches"] = int(frame["sc_match_id"].nunique())

    for metric in _PRESSURE_METRICS:
        if metric not in frame.columns:
            out[f"sc_pressure_{metric}"] = None
            out[f"sc_pressure_{metric}_recent{_RECENT_N}"] = None
            continue
        series = pd.to_numeric(frame[metric], errors="coerce").dropna()
        out[f"sc_pressure_{metric}"] = _safe_mean(series)
        out[f"sc_pressure_{metric}_recent{_RECENT_N}"] = _safe_mean(series.tail(_RECENT_N))

    return out


def _aggregate_passes(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}

    if frame.empty:
        for metric in _PASSES_METRICS:
            out[f"sc_passes_{metric}"] = None
            out[f"sc_passes_{metric}_recent{_RECENT_N}"] = None
        out["sc_passes_sample_matches"] = 0
        out["sc_passes_opportunity_take_rate"] = None
        return out

    frame = frame[frame["quality_check"].fillna(True) != False].copy()  # noqa: E712
    out["sc_passes_sample_matches"] = int(frame["sc_match_id"].nunique())

    for metric in _PASSES_METRICS:
        if metric not in frame.columns:
            out[f"sc_passes_{metric}"] = None
            out[f"sc_passes_{metric}_recent{_RECENT_N}"] = None
            continue
        series = pd.to_numeric(frame[metric], errors="coerce").dropna()
        out[f"sc_passes_{metric}"] = _safe_mean(series)
        out[f"sc_passes_{metric}_recent{_RECENT_N}"] = _safe_mean(series.tail(_RECENT_N))

    # Opportunity take rate: attempts / opportunities
    opps = pd.to_numeric(
        frame.get("count_opportunities_to_pass_to_run_in_behind_per_match"), errors="coerce"
    )
    attempts = pd.to_numeric(
        frame.get("count_pass_attempts_to_run_in_behind_per_match"), errors="coerce"
    )
    combined = pd.concat([opps.rename("o"), attempts.rename("a")], axis=1).dropna()
    if not combined.empty and combined["o"].sum() > 0:
        out["sc_passes_opportunity_take_rate"] = float(
            combined["a"].sum() / combined["o"].sum()
        )
    else:
        out["sc_passes_opportunity_take_rate"] = None

    return out


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _to_frame(rows: list[Any], model: Any) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    # Use mapper attribute names (not DB column names) — they differ when a column
    # name clashes with a Python builtin (e.g. DB col "group" → attr "group_").
    from sqlalchemy import inspect as sa_inspect
    attr_names = [prop.key for prop in sa_inspect(model).mapper.column_attrs]
    return pd.DataFrame(
        [{attr: getattr(row, attr) for attr in attr_names} for row in rows]
    )


def _safe_mean(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else None


def _trend_slope(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 2:
        return None
    x = np.arange(len(clean), dtype=float)
    slope = float(np.polyfit(x, clean.to_numpy(dtype=float), 1)[0])
    return slope
