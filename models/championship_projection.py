"""Championship transition projection model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from datetime import date as _date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import get_settings
from db.read_cache import load_player_match_frame, load_player_role_row, load_player_row, load_standings_frame_for_leagues
from features.per90 import _compute_per90_frame
from models.role_fit import get_active_template_for_role

_FALLBACK_AGE = 24.0
_FALLBACK_LEAGUE_POSITION = 12.0


PROJECTION_MODEL_PATH = Path("data/championship_projection_model.joblib")


@dataclass
class ProjectionBundle:
    metric_models: dict[str, Pipeline]
    starter_model: Pipeline
    residual_quantiles: dict[str, tuple[float, float]]
    sample_size_by_pair: dict[str, int]
    feature_columns: list[str]
    role_name: str


def train_projection_model(training_data: pd.DataFrame) -> dict[str, Any]:
    """Train per-metric Championship projection models.

    Uses a practical sklearn fallback when `statsmodels` mixed-effects tooling
    is unavailable in the environment.
    """

    if training_data.empty:
        raise ValueError("Projection training data is empty")

    base_features = [
        "origin_league_id",
        "destination_league_id",
        "league_pair",
        "age_at_transfer",
        "primary_role",
        "origin_team_league_position",
        "destination_team_league_position",
    ]
    origin_metric_columns = [
        column
        for column in training_data.columns
        if column.startswith("origin_") and column not in base_features
    ]
    feature_columns = base_features + origin_metric_columns
    target_columns = [column for column in training_data.columns if column.startswith("target_")]
    if not target_columns:
        raise ValueError("Projection training data must include target_* columns")

    preprocessor = _projection_preprocessor(feature_columns)
    metric_models: dict[str, Pipeline] = {}
    residual_quantiles: dict[str, tuple[float, float]] = {}

    for target in target_columns:
        estimator = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]
        )
        estimator.fit(training_data[feature_columns], training_data[target])
        metric_models[target.removeprefix("target_")] = estimator
        fitted = estimator.predict(training_data[feature_columns])
        residuals = training_data[target].to_numpy(dtype=float) - fitted
        residual_quantiles[target.removeprefix("target_")] = (
            float(np.quantile(residuals, 0.10)),
            float(np.quantile(residuals, 0.90)),
        )

    starter_target = training_data.get("target_starter")
    if starter_target is None:
        raise ValueError("Projection training data must include target_starter")
    starter_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler(with_mean=False)),
            ("model", LogisticRegression(max_iter=5000, solver="saga")),
        ]
    )
    starter_model.fit(training_data[feature_columns], starter_target.astype(int))

    sample_size_by_pair = (
        training_data.groupby("league_pair").size().astype(int).to_dict()
    )
    bundle = ProjectionBundle(
        metric_models=metric_models,
        starter_model=starter_model,
        residual_quantiles=residual_quantiles,
        sample_size_by_pair=sample_size_by_pair,
        feature_columns=feature_columns,
        role_name=str(training_data["primary_role"].mode().iloc[0]),
    )
    output_path = _resolve_model_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    return {
        "metrics": list(metric_models),
        "sample_size_by_pair": sample_size_by_pair,
        "model_path": str(output_path),
    }


def project_to_championship(player_id: int, season: str, brief: dict[str, Any] | None = None) -> dict[str, Any]:
    """Project a player's Championship performance, minutes share, and adaptation time."""

    try:
        bundle = _load_projection_bundle()
    except FileNotFoundError:
        return _heuristic_projection(player_id=player_id, season=season, brief=brief)

    feature_frame = _build_projection_feature_frame(
        player_id=player_id,
        season=season,
        role=bundle.role_name,
        brief=brief,
    )
    if feature_frame.empty:
        raise ValueError("Could not build projection feature frame for player")
    feature_frame = _align_projection_feature_frame(feature_frame, bundle.feature_columns)

    projected_performance: dict[str, dict[str, float]] = {}
    for metric_name, model in bundle.metric_models.items():
        median = float(model.predict(feature_frame[bundle.feature_columns])[0])
        q10_resid, q90_resid = bundle.residual_quantiles[metric_name]
        projected_performance[metric_name] = {
            "p10": max(0.0, median + q10_resid),
            "p50": max(0.0, median),
            "p90": max(0.0, median + q90_resid),
        }

    starter_probability = float(
        bundle.starter_model.predict_proba(feature_frame[bundle.feature_columns])[0][1]
    )
    adaptation_months = _estimate_adaptation_months(
        age=float(feature_frame.iloc[0]["age_at_transfer"]),
        role_name=str(feature_frame.iloc[0]["primary_role"]),
    )
    league_pair = str(feature_frame.iloc[0]["league_pair"])
    sample_size = int(bundle.sample_size_by_pair.get(league_pair, 0))

    return {
        "projected_performance": projected_performance,
        "projected_minutes_share": starter_probability,
        "projected_adaptation_months": adaptation_months,
        "sample_size": sample_size,
        "confidence_note": (
            "Low sample — wide uncertainty bands, proceed with caution"
            if sample_size < 20
            else None
        ),
    }


def _projection_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_columns = [
        column
        for column in feature_columns
        if (
            column.startswith("origin_")
            and column != "origin_league_id"
        )
        or column.endswith("_position")
        or column == "age_at_transfer"
    ]
    numeric_columns = list(dict.fromkeys(numeric_columns))
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def _align_projection_feature_frame(
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    aligned = feature_frame.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned


def _load_projection_bundle() -> ProjectionBundle:
    model_path = _resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Projection model not found at {model_path}. Train it first."
        )
    return joblib.load(model_path)


def _heuristic_projection(
    player_id: int,
    season: str,
    brief: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fallback projection when no trained model artifact exists."""

    role_name = _infer_role_name(player_id, season)
    feature_frame = _build_projection_feature_frame(
        player_id=player_id,
        season=season,
        role=role_name,
        brief=brief,
    )
    if feature_frame.empty:
        player_record = load_player_row(player_id)
        return {
            "projected_performance": {},
            "projected_minutes_share": _heuristic_starter_probability(player_id, season),
            "projected_adaptation_months": _estimate_adaptation_months(
                age=_player_age_years(
                    player_record.get("birth_date"),
                    player_record.get("current_age_years"),
                ),
                role_name=role_name,
            ),
            "sample_size": 0,
            "confidence_note": (
                "Heuristic Championship translation used because no trained projection artifact is present "
                "and player-level projection features were incomplete."
            ),
        }

    row = feature_frame.iloc[0]
    multiplier = _championship_translation_multiplier(int(row["origin_league_id"]))
    projected_performance: dict[str, dict[str, float]] = {}
    for column, value in row.items():
        if not str(column).startswith("origin_") or not str(column).endswith("_per90"):
            continue
        target_name = str(column).removeprefix("origin_")
        median = max(0.0, float(value or 0.0) * multiplier)
        spread = max(0.05, median * 0.25)
        projected_performance[target_name] = {
            "p10": max(0.0, median - spread),
            "p50": median,
            "p90": median + spread,
        }

    starter_probability = _heuristic_starter_probability(player_id, season)
    adaptation_months = _estimate_adaptation_months(
        age=float(row.get("age_at_transfer") or 24.0),
        role_name=str(row.get("primary_role") or role_name),
    )
    return {
        "projected_performance": projected_performance,
        "projected_minutes_share": starter_probability,
        "projected_adaptation_months": adaptation_months,
        "sample_size": 0,
        "confidence_note": "Heuristic Championship translation used because no trained projection artifact is present.",
    }


def _build_projection_feature_frame(
    player_id: int,
    season: str,
    role: str,
    brief: dict[str, Any] | None = None,
) -> pd.DataFrame:
    match_frame = load_player_match_frame(player_id, season).copy()
    role_row = load_player_role_row(player_id, season)
    player_record = load_player_row(player_id)
    if match_frame.empty:
        return pd.DataFrame()
    per90 = _compute_per90_frame(match_frame)
    summary = per90.mean(numeric_only=True).to_dict()
    origin_league_id = int(match_frame["league_id"].mode().iloc[0])
    origin_position = _lookup_team_league_position(match_frame)
    primary_role = role_row.get("primary_role") or role
    destination = _resolve_projection_destination(brief)
    feature_row = {
        "origin_league_id": origin_league_id,
        "destination_league_id": destination["destination_league_id"],
        "league_pair": f"{origin_league_id}->{destination['destination_league_id']}",
        "age_at_transfer": _player_age_years(
            player_record.get("birth_date"),
            player_record.get("current_age_years"),
        ),
        "primary_role": primary_role,
        "origin_team_league_position": origin_position,
        "destination_team_league_position": destination["destination_team_league_position"],
    }
    for key, value in summary.items():
        if key.endswith("_per90"):
            feature_row[f"origin_{key}"] = float(value)
    return pd.DataFrame([feature_row])


def _lookup_team_league_position(match_frame: pd.DataFrame) -> float:
    """Return the player's team most recent league table position, or fallback."""

    if match_frame.empty or "team" not in match_frame.columns:
        return _FALLBACK_LEAGUE_POSITION
    team_series = match_frame["team"].dropna()
    if team_series.empty:
        return _FALLBACK_LEAGUE_POSITION
    team_name = str(team_series.mode().iloc[0])
    league_ids = tuple(sorted({int(v) for v in match_frame["league_id"].dropna().tolist()}))
    if not league_ids:
        return _FALLBACK_LEAGUE_POSITION
    standings = load_standings_frame_for_leagues(league_ids)
    if standings.empty or "team_name" not in standings.columns:
        return _FALLBACK_LEAGUE_POSITION
    team_rows = standings[standings["team_name"] == team_name]
    if team_rows.empty:
        return _FALLBACK_LEAGUE_POSITION
    latest = team_rows.sort_values("date", ascending=False).iloc[0]
    return float(latest["position"])


def _player_age_years(birth_date: Any, current_age_years: Any = None) -> float:
    """Return player age in years, falling back to _FALLBACK_AGE if unavailable."""

    if current_age_years not in (None, ""):
        try:
            fallback_age = float(current_age_years)
        except (TypeError, ValueError):
            fallback_age = _FALLBACK_AGE
    else:
        fallback_age = _FALLBACK_AGE

    if birth_date is None:
        return fallback_age
    try:
        if not isinstance(birth_date, _date):
            from datetime import datetime

            birth_date = datetime.strptime(str(birth_date)[:10], "%Y-%m-%d").date()
        return round((_date.today() - birth_date).days / 365.25, 1)
    except (ValueError, TypeError):
        return fallback_age


def _estimate_adaptation_months(*, age: float, role_name: str) -> float:
    config = _projection_heuristics()["adaptation_months"]
    base = float(config["base"])
    if age < 22:
        base += float(config["under_22_adjustment"])
    elif age > 28:
        base += float(config["over_28_adjustment"])
    if "forward" in role_name or "striker" in role_name:
        base += float(config["forward_adjustment"])
    return max(float(config["minimum"]), round(base, 1))


def _heuristic_starter_probability(player_id: int, season: str) -> float:
    match_frame = load_player_match_frame(player_id, season).copy()
    if match_frame.empty:
        return 0.5
    average_minutes = float(pd.to_numeric(match_frame["minutes"], errors="coerce").fillna(0).mean())
    starts = int((~match_frame["is_substitute"].fillna(False)).sum())
    start_rate = starts / len(match_frame.index)
    probability = (0.45 * start_rate) + (0.55 * min(average_minutes / 90.0, 1.0))
    return float(np.clip(probability, 0.20, 0.92))


def _championship_translation_multiplier(origin_league_id: int) -> float:
    config = _projection_heuristics()["translation_multipliers"]
    leagues = {
        int(league["league_id"]): league
        for league in get_settings().load_json("leagues.json")
    }
    league = leagues.get(origin_league_id, {})
    tier = int(league.get("tier") or 3)
    country = str(league.get("country") or "")

    if origin_league_id == 40:
        return float(config["same_league"])
    if tier == 1:
        if country == "Scotland":
            return float(config["tier_1_scotland"])
        return float(config["tier_1"])
    if tier == 2:
        return float(config["tier_2"])
    if tier == 3:
        return float(config["tier_3"])
    return float(config["default"])


def _infer_role_name(player_id: int, season: str) -> str:
    role_row = load_player_role_row(player_id, season)
    if role_row.get("primary_role"):
        return str(role_row["primary_role"])
    return "complete_forward"


def _resolve_projection_destination(brief: dict[str, Any] | None = None) -> dict[str, float | int]:
    defaults = _club_profile_defaults()
    brief = brief or {}
    destination_league_id = int(
        brief.get("destination_league_id")
        or defaults.get("destination_league_id")
        or 40
    )
    destination_team_position = float(
        brief.get("destination_team_league_position")
        or defaults.get("destination_team_league_position")
        or 16.0
    )
    return {
        "destination_league_id": destination_league_id,
        "destination_team_league_position": destination_team_position,
    }


@lru_cache(maxsize=1)
def _club_profile_defaults() -> dict[str, Any]:
    return dict(get_settings().load_json("club_profile.json"))


@lru_cache(maxsize=1)
def _projection_heuristics() -> dict[str, Any]:
    return dict(get_settings().load_json("projection_heuristics.json"))


def _resolve_model_path() -> Path:
    from config import get_settings

    return get_settings().project_root / PROJECTION_MODEL_PATH
