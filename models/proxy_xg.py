"""Proxy xG fallback model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import select

from config import get_settings
from db.schema import ExpectedMetric, MatchEvent, MatchPerformance, Player
from db.session import session_scope
from ingestion.common import upsert_rows


PROXY_XG_MODEL_PATH = Path("data/proxy_xg_model.joblib")


@dataclass(frozen=True)
class ProxyShotFeatures:
    is_header: int
    angle_to_goal: float
    distance_to_goal: float
    is_penalty: int
    is_direct_free_kick: int
    game_state: str


def train_proxy_xg(statsbomb_data_path: str) -> Pipeline:
    """Train a logistic-regression-based proxy xG model from shot-event data."""

    training_data = _load_statsbomb_shot_data(statsbomb_data_path)
    feature_columns = [
        "is_header",
        "angle_to_goal",
        "distance_to_goal",
        "is_penalty",
        "is_direct_free_kick",
        "game_state",
    ]
    target = training_data["goal"].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                    ]
                ),
                ["is_header", "angle_to_goal", "distance_to_goal", "is_penalty", "is_direct_free_kick"],
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["game_state"],
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(training_data[feature_columns], target)

    output_path = get_settings().project_root / PROXY_XG_MODEL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return model


def predict_xg(player_id: int, season: str) -> float:
    """Estimate xG for a player-season using fallback shot features."""

    model = _load_proxy_model()
    shot_frame = _build_player_proxy_shot_frame(player_id, season)
    if shot_frame.empty:
        raise ValueError("No shot-like data available for proxy xG prediction")

    probabilities = model.predict_proba(
        shot_frame[
            [
                "is_header",
                "angle_to_goal",
                "distance_to_goal",
                "is_penalty",
                "is_direct_free_kick",
                "game_state",
            ]
        ]
    )[:, 1]
    total_xg = float(probabilities.sum())

    with session_scope() as session:
        latest_match = session.scalar(
            select(MatchPerformance)
            .where(
                MatchPerformance.player_id == player_id,
                MatchPerformance.season == season,
            )
            .order_by(MatchPerformance.date.desc())
        )
    if latest_match is not None:
        upsert_rows(
            ExpectedMetric,
            [
                {
                    "player_id": player_id,
                    "season": season,
                    "source": "proxy",
                    "league_id": latest_match.league_id,
                    "xg": total_xg,
                    "npxg": total_xg,
                    "xa": None,
                    "xg_per_shot": total_xg / len(shot_frame.index) if len(shot_frame.index) else None,
                    "goals_minus_xg": None,
                    "assists_minus_xa": None,
                    "progressive_passes": None,
                    "progressive_carries": None,
                    "progressive_receptions": None,
                }
            ],
            ["player_id", "season", "source"],
        )
    return total_xg


def _load_statsbomb_shot_data(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {
        "is_header",
        "angle_to_goal",
        "distance_to_goal",
        "is_penalty",
        "is_direct_free_kick",
        "game_state",
        "goal",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required StatsBomb-derived columns: {sorted(missing)}")
    return frame[list(required_columns)].copy()


def _load_proxy_model() -> Pipeline:
    output_path = get_settings().project_root / PROXY_XG_MODEL_PATH
    if not output_path.exists():
        raise FileNotFoundError(
            f"Proxy xG model not found at {output_path}. Train it with train_proxy_xg() first."
        )
    return joblib.load(output_path)


def _build_player_proxy_shot_frame(player_id: int, season: str) -> pd.DataFrame:
    with session_scope() as session:
        events = list(
            session.scalars(
                select(MatchEvent).where(
                    MatchEvent.player_id == player_id,
                    MatchEvent.event_type.ilike("%goal%"),
                )
            )
        )
        matches = list(
            session.scalars(
                select(MatchPerformance).where(
                    MatchPerformance.player_id == player_id,
                    MatchPerformance.season == season,
                )
            )
        )
    match_frame = pd.DataFrame([_row_to_dict(row, MatchPerformance) for row in matches])
    event_frame = pd.DataFrame([_row_to_dict(row, MatchEvent) for row in events])
    return _build_proxy_shot_frame_from_frames(match_frame, event_frame)


def _build_proxy_shot_frame_from_frames(
    match_frame: pd.DataFrame,
    event_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not event_frame.empty:
        for _, event in event_frame.iterrows():
            detail = str(event.get("event_detail") or "").lower()
            comments = str(event.get("comments") or "").lower()
            rows.append(
                {
                    "is_header": int("header" in detail or "header" in comments),
                    "angle_to_goal": 0.35 if "header" in detail else 0.55,
                    "distance_to_goal": 11.0 if "pen" in detail else 16.0,
                    "is_penalty": int("pen" in detail),
                    "is_direct_free_kick": int("free kick" in detail or "free-kick" in comments),
                    "game_state": "drawing",
                }
            )

    if not match_frame.empty:
        for _, match in match_frame.iterrows():
            shots_total = int(match.get("shots_total") or 0)
            on_target = int(match.get("shots_on_target") or 0)
            non_event_shots = max(shots_total - len(rows), 0)
            for index in range(non_event_shots):
                rows.append(
                    {
                        "is_header": 0,
                        "angle_to_goal": 0.45 if index < on_target else 0.30,
                        "distance_to_goal": 18.0 if index >= on_target else 15.0,
                        "is_penalty": 0,
                        "is_direct_free_kick": 0,
                        "game_state": "drawing",
                    }
                )
    return pd.DataFrame(rows)


def _row_to_dict(row: Any, model: Any) -> dict[str, Any]:
    return {
        column.name: getattr(row, column.name)
        for column in model.__table__.columns
    }
