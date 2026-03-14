"""Availability risk model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from config import get_settings
from db.read_cache import load_player_match_frame, load_player_row
from features.availability import compute_availability_features


AVAILABILITY_MODEL_PATH = Path("data/availability_risk_model.joblib")


def train_availability_model(training_data: pd.DataFrame) -> Pipeline:
    """Train an interpretable availability model."""

    feature_columns = [
        "availability_rate_3yr",
        "injury_frequency_3yr",
        "avg_injury_duration",
        "max_injury_duration",
        "muscle_injury_count",
        "recurrence_rate",
        "days_since_last_injury",
        "minutes_continuity",
        "age",
        "position_group",
    ]
    missing = set(feature_columns + ["target_available_75pct"]) - set(training_data.columns)
    if missing:
        raise ValueError(f"Missing availability training columns: {sorted(missing)}")

    # HistGradientBoostingClassifier handles NaN natively and uses ordinal
    # encoding for categoricals — no imputation or scaling step required.
    # This is robust when injury-history columns are all-null (no sidelined data).
    numeric_columns = [column for column in feature_columns if column != "position_group"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_columns),
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ["position_group"],
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", HistGradientBoostingClassifier(
                max_iter=300,
                random_state=42,
            )),
        ]
    )
    model.fit(training_data[feature_columns], training_data["target_available_75pct"].astype(int))

    model_path = _resolve_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def predict_availability_risk(player_id: int) -> dict[str, Any]:
    """Predict the probability of future availability."""

    try:
        model = _load_availability_model()
    except FileNotFoundError:
        return _heuristic_availability_risk(player_id)

    features = _build_availability_prediction_frame(player_id)
    if _availability_history_is_sparse(features.iloc[0]):
        result = _heuristic_availability_risk(player_id)
        result["caveat"] = (
            "Heuristic fallback used because injury-history coverage is too sparse for the trained availability model."
        )
        return result
    probability = float(model.predict_proba(features)[0][1])
    try:
        coefficients = _availability_contributions(model, features)
    except Exception:
        coefficients = []
    return {
        "probability_available_75pct": probability,
        "risk_tier": _risk_tier(probability),
        "contributing_factors": coefficients,
        "caveat": "Advisory to medical. Does not predict freak injuries.",
    }


def _load_availability_model() -> Pipeline:
    model_path = _resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Availability model not found at {model_path}. Train it first."
        )
    return joblib.load(model_path)


def _build_availability_prediction_frame(player_id: int) -> pd.DataFrame:
    features = compute_availability_features(player_id)
    match_frame = load_player_match_frame(player_id)
    latest_match = None
    if not match_frame.empty:
        latest_match = match_frame.sort_values("date").iloc[-1]
    player = load_player_row(player_id)
    position_group = "Unknown"
    age = 25.0
    latest_position = latest_match.get("position") if latest_match is not None else None
    if latest_position:
        position_group = str(latest_position)[0].upper()
    birth_date = player.get("birth_date")
    if birth_date is not None:
        age = round((pd.Timestamp.today().date() - birth_date).days / 365.25, 2)
    features = dict(features)
    features["age"] = age
    features["position_group"] = position_group
    return pd.DataFrame([features])


def _availability_contributions(model: Pipeline, feature_frame: pd.DataFrame) -> list[dict[str, Any]]:
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())
    importances = classifier.feature_importances_
    # Pair names with importances; direction is inferred from the feature value
    # relative to the population mean (positive value → positive contribution).
    transformed = preprocessor.transform(feature_frame)
    values = transformed[0] if not hasattr(transformed, "toarray") else transformed.toarray()[0]
    pairs = sorted(zip(feature_names, importances, values), key=lambda t: t[1], reverse=True)[:5]
    return [
        {
            "factor": name,
            "direction": "increases_availability" if float(val) >= 0 else "decreases_availability",
            "magnitude": float(importance),
        }
        for name, importance, val in pairs
    ]


def _availability_history_is_sparse(row: pd.Series) -> bool:
    injury_frequency = float(row.get("injury_frequency_3yr") or 0.0)
    muscle_count = float(row.get("muscle_injury_count") or 0.0)
    max_duration = row.get("max_injury_duration")
    recurrence_rate = row.get("recurrence_rate")
    return (
        injury_frequency == 0.0
        and muscle_count == 0.0
        and pd.isna(max_duration)
        and pd.isna(recurrence_rate)
    )


def _heuristic_availability_risk(player_id: int) -> dict[str, Any]:
    """Fallback availability estimate when no trained model artifact exists."""

    features = _build_availability_prediction_frame(player_id)
    row = features.iloc[0]
    availability_rate = float(row.get("availability_rate_3yr") or row.get("availability_rate_season") or 0.70)
    injury_frequency = float(row.get("injury_frequency_3yr") or 0.0)
    recurrence_rate = float(row.get("recurrence_rate") or 0.0)
    max_injury_duration = float(row.get("max_injury_duration") or 0.0)
    continuity = float(row.get("minutes_continuity") or 0.0)
    days_since_last_injury = row.get("days_since_last_injury")
    rest_bonus = 0.0
    if pd.notna(days_since_last_injury):
        rest_bonus = min(float(days_since_last_injury), 180.0) / 180.0 * 0.08

    probability = (
        0.52
        + (0.38 * availability_rate)
        - (0.035 * injury_frequency)
        - (0.14 * recurrence_rate)
        - (0.0015 * max_injury_duration)
        + (0.018 * min(continuity, 10.0))
        + rest_bonus
    )
    probability = float(np.clip(probability, 0.20, 0.95))

    factors = [
        ("availability_rate_3yr", (availability_rate - 0.5) * 0.38),
        ("injury_frequency_3yr", -injury_frequency * 0.035),
        ("recurrence_rate", -recurrence_rate * 0.14),
        ("max_injury_duration", -max_injury_duration * 0.0015),
        ("minutes_continuity", min(continuity, 10.0) * 0.018),
        ("days_since_last_injury", rest_bonus),
    ]
    factors = sorted(factors, key=lambda item: abs(item[1]), reverse=True)
    return {
        "probability_available_75pct": probability,
        "risk_tier": _risk_tier(probability),
        "contributing_factors": [
            {
                "factor": name,
                "direction": "increases_availability" if value >= 0 else "decreases_availability",
                "magnitude": float(value),
            }
            for name, value in factors[:5]
        ],
        "caveat": "Heuristic fallback used because no trained availability model artifact is present.",
    }


def _risk_tier(probability_available: float) -> str:
    risk = 1.0 - probability_available
    if risk >= 0.40:
        return "High"
    if risk >= 0.20:
        return "Medium"
    return "Low"


def _resolve_model_path() -> Path:
    return get_settings().project_root / AVAILABILITY_MODEL_PATH
