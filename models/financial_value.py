"""Financial value and VAR estimation."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import and_, desc, func, select

from db.read_cache import load_latest_market_value_row, load_player_match_frame, load_player_role_row, load_player_row
from db.schema import MarketValue, MatchPerformance, Player, PlayerRole, Transfer
from db.session import session_scope
from features.trajectory import compute_trajectory_features
from features.per90 import _compute_per90_frame


VALUE_MODEL_PATH = Path("data/financial_value_model.joblib")


def train_value_model(training_data: pd.DataFrame) -> Pipeline:
    """Train quantile value models for transfer valuation."""

    feature_columns = [
        "age",
        "position_group",
        "role",
        "league_level",
        "contract_remaining_years",
        "market_value_pretransfer",
        "per90_output",
    ]
    missing = set(feature_columns + ["fee_paid"]) - set(training_data.columns)
    if missing:
        raise ValueError(f"Missing value training columns: {sorted(missing)}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median"))]),
                ["age", "league_level", "contract_remaining_years", "market_value_pretransfer", "per90_output"],
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["position_group", "role"],
            ),
        ]
    )

    bundle = {}
    for quantile in (0.1, 0.5, 0.9):
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", QuantileRegressor(quantile=quantile, alpha=0.01, solver="highs")),
            ]
        )
        model.fit(training_data[feature_columns], training_data["fee_paid"])
        bundle[str(quantile)] = model

    model_path = _resolve_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle["0.5"]


def estimate_value(player_id: int, brief: dict[str, Any]) -> dict[str, Any]:
    """Estimate fair-value bands, wage fit, resale band, and VAR score."""

    try:
        bundle = _load_value_model_bundle()
    except FileNotFoundError:
        return _heuristic_estimate_value(player_id, brief)

    feature_frame = _build_value_prediction_frame(player_id)
    low = Decimal(str(round(float(bundle["0.1"].predict(feature_frame)[0]), 2)))
    mid = Decimal(str(round(float(bundle["0.5"].predict(feature_frame)[0]), 2)))
    high = Decimal(str(round(float(bundle["0.9"].predict(feature_frame)[0]), 2)))

    market_value = Decimal(str(float(feature_frame.iloc[0]["market_value_pretransfer"] or 0.0)))
    proposed_wage = Decimal(str(brief.get("proposed_wage") or brief.get("budget_max_wage") or 0))
    role_band = Decimal(str(brief.get("club_wage_band") or brief.get("budget_max_wage") or 0))
    wage_fit = _wage_fit(proposed_wage, role_band)
    resale_mid = Decimal(str(round(_resale_projection(feature_frame.iloc[0], float(mid)), 2)))
    resale_band = {
        "low": round(resale_mid * Decimal("0.85"), 2),
        "mid": round(resale_mid, 2),
        "high": round(resale_mid * Decimal("1.15"), 2),
    }
    replacement_cost = round(max(high, market_value) * Decimal("1.1"), 2)
    total_cost = float(mid + proposed_wage * Decimal(str(brief.get("budget_max_contract_years") or 1)))
    quality_score = float(brief.get("quality_score") or 50.0)
    var_score = _value_adjusted_return_score(
        total_cost=total_cost,
        fair_value_mid=float(mid),
        resale_mid=float(resale_mid),
        quality_score=quality_score,
    )

    comparables = _find_comparable_transactions(feature_frame.iloc[0], n=5, player_id=player_id)
    return {
        "fair_value_band": {"low": low, "mid": mid, "high": high},
        "wage_fit": wage_fit,
        "resale_band_2yr": resale_band,
        "replacement_cost": replacement_cost,
        "var_score": var_score,
        "comparable_transactions": comparables,
    }


def _heuristic_estimate_value(player_id: int, brief: dict[str, Any]) -> dict[str, Any]:
    """Fallback valuation when no trained model artifact exists."""

    feature_frame = _build_value_prediction_frame(player_id)
    row = feature_frame.iloc[0]
    market_value = Decimal(str(float(row.get("market_value_pretransfer") or 0.0)))
    if market_value > 0:
        mid = market_value
    else:
        mid = Decimal(str(_implied_market_value(row)))
    low = max(Decimal("0"), mid * Decimal("0.80"))
    high = max(low, mid * Decimal("1.20"))

    proposed_wage = Decimal(str(brief.get("proposed_wage") or brief.get("budget_max_wage") or 0))
    role_band = Decimal(str(brief.get("club_wage_band") or brief.get("budget_max_wage") or 0))
    wage_fit = _wage_fit(proposed_wage, role_band)
    resale_mid = Decimal(str(round(_resale_projection(row, float(mid)), 2)))
    resale_band = {
        "low": round(resale_mid * Decimal("0.85"), 2),
        "mid": round(resale_mid, 2),
        "high": round(resale_mid * Decimal("1.15"), 2),
    }
    replacement_cost = round(max(high, market_value) * Decimal("1.1"), 2)
    total_cost = float(mid + proposed_wage * Decimal(str(brief.get("budget_max_contract_years") or 1)))
    quality_score = float(brief.get("quality_score") or 50.0)
    var_score = _value_adjusted_return_score(
        total_cost=total_cost,
        fair_value_mid=float(mid),
        resale_mid=float(resale_mid),
        quality_score=quality_score,
    )

    return {
        "fair_value_band": {"low": round(low, 2), "mid": round(mid, 2), "high": round(high, 2)},
        "wage_fit": wage_fit,
        "resale_band_2yr": resale_band,
        "replacement_cost": replacement_cost,
        "var_score": var_score,
        "comparable_transactions": [],
        "caveat": "Heuristic valuation used because no trained financial value model artifact is present.",
    }


def _load_value_model_bundle() -> dict[str, Pipeline]:
    model_path = _resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Financial value model not found at {model_path}. Train it first."
        )
    return joblib.load(model_path)


def _build_value_prediction_frame(player_id: int) -> pd.DataFrame:
    player = load_player_row(player_id)
    market_value = load_latest_market_value_row(player_id)
    role_row = load_player_role_row(player_id)
    match_frame = load_player_match_frame(player_id).copy()
    per90 = _compute_per90_frame(match_frame)
    per90_output = float(per90[[column for column in per90.columns if column.endswith("_per90")]].mean(numeric_only=True).mean()) if not per90.empty else 0.0
    trajectory = compute_trajectory_features(player_id)
    latest_position = None
    if not match_frame.empty and "position" in match_frame.columns:
        latest_position = match_frame.sort_values("date").iloc[-1].get("position")
    origin_league_level = _resolve_origin_league_level(match_frame, player)
    return pd.DataFrame(
        [
            {
                "age": _player_age_value(player),
                "position_group": (str(latest_position)[0].upper() if latest_position else "U"),
                "role": role_row.get("primary_role") or "unknown",
                "league_level": origin_league_level,
                "contract_remaining_years": _contract_remaining_years(market_value.get("contract_expiry")),
                "market_value_pretransfer": float(market_value["market_value_eur"]) if market_value.get("market_value_eur") is not None else 0.0,
                "per90_output": per90_output,
                "trajectory": float(trajectory.get("output_trajectory_2yr") or 0.0),
            }
        ]
    )


def _resale_projection(row: pd.Series, fair_value_mid: float) -> float:
    trajectory = float(row.get("trajectory", 0.0))
    age = float(row.get("age", 24.0))
    age_modifier = 1.08 if 22 <= age <= 24 else 1.02 if age <= 27 else 0.9
    trajectory_modifier = 1.0 + max(min(trajectory, 0.2), -0.2)
    return fair_value_mid * age_modifier * trajectory_modifier


def _wage_fit(proposed_wage: Decimal, role_band: Decimal) -> str:
    if role_band == 0:
        return "caution"
    if proposed_wage <= role_band:
        return "ok"
    if proposed_wage <= (Decimal("1.2") * role_band):
        return "caution"
    return "exceeds"


def _find_comparable_transactions(player_row: pd.Series, n: int, player_id: int | None = None) -> list[dict[str, Any]]:
    transfers = _recent_transfer_candidates(limit=max(n * 12, 60))
    comparable_meta = _load_comparable_player_metadata([int(transfer.player_id) for transfer in transfers])
    target_role = str(player_row.get("role") or "").strip().lower()
    target_league_level = float(player_row.get("league_level") or 3.0)
    ranked: list[tuple[float, Transfer, dict[str, Any]]] = []

    for transfer in transfers:
        if player_id is not None and int(transfer.player_id) == int(player_id):
            continue
        meta = comparable_meta.get(int(transfer.player_id), {})
        candidate_role = str(meta.get("primary_role") or "").strip().lower()
        candidate_league_level = meta.get("league_level")
        score = 0.0
        if target_role and candidate_role == target_role:
            score += 100.0
        if candidate_league_level is not None:
            score += max(0.0, 15.0 - (10.0 * abs(float(candidate_league_level) - target_league_level)))
        if transfer.date is not None:
            years_old = max(0.0, (date.today() - transfer.date).days / 365.25)
            score += max(0.0, 10.0 - years_old)
        ranked.append((score, transfer, meta))

    ranked.sort(
        key=lambda item: (
            item[0],
            item[1].date or date.min,
            float(item[1].fee_paid or 0.0),
        ),
        reverse=True,
    )

    comparables = []
    for _, transfer, meta in ranked[:n]:
        comparables.append(
            {
                "player_id": transfer.player_id,
                "date": transfer.date.isoformat() if transfer.date else None,
                "type": transfer.type,
                "team_in": transfer.team_in,
                "team_out": transfer.team_out,
                "primary_role": meta.get("primary_role"),
                "league_level": meta.get("league_level"),
            }
        )
    return comparables


def _resolve_origin_league_level(match_frame: pd.DataFrame, player: dict[str, Any]) -> float:
    candidate_league_ids: list[int] = []
    if not match_frame.empty and "league_id" in match_frame.columns:
        league_series = pd.to_numeric(match_frame["league_id"], errors="coerce").dropna()
        if not league_series.empty:
            candidate_league_ids.append(int(league_series.mode().iloc[0]))
    current_league_id = player.get("current_league_id")
    if current_league_id not in (None, ""):
        candidate_league_ids.append(int(current_league_id))

    lookup = _league_tier_lookup()
    for league_id in candidate_league_ids:
        if league_id in lookup:
            return float(lookup[league_id])
    return 3.0


def _value_adjusted_return_score(
    *,
    total_cost: float,
    fair_value_mid: float,
    resale_mid: float,
    quality_score: float,
) -> float:
    """Return a bounded, dimensionless value-adjusted return score."""

    if total_cost <= 0:
        return 0.0

    asset_support = (0.60 * max(fair_value_mid, 0.0)) + (0.40 * max(resale_mid, 0.0))
    economic_margin = (asset_support / total_cost) - 1.0
    quality_adjustment = ((quality_score - 50.0) / 50.0) * 0.30
    blended = (0.70 * economic_margin) + quality_adjustment
    return float(max(-1.0, min(1.0, blended)))


def _recent_transfer_candidates(limit: int, lookback_years: int = 5) -> list[Transfer]:
    cutoff = date.today() - timedelta(days=365 * lookback_years)
    with session_scope() as session:
        return list(
            session.scalars(
                select(Transfer)
                .where(
                    Transfer.date.is_not(None),
                    Transfer.date >= cutoff,
                    Transfer.fee_paid.is_not(None),
                )
                .order_by(desc(Transfer.date))
                .limit(limit)
            )
        )


def _load_comparable_player_metadata(player_ids: list[int]) -> dict[int, dict[str, Any]]:
    candidate_ids = sorted({int(player_id) for player_id in player_ids})
    if not candidate_ids:
        return {}

    with session_scope() as session:
        latest_role = (
            select(
                PlayerRole.player_id.label("player_id"),
                func.max(PlayerRole.season).label("latest_season"),
            )
            .where(PlayerRole.player_id.in_(candidate_ids))
            .group_by(PlayerRole.player_id)
            .subquery()
        )
        role_rows = session.execute(
            select(PlayerRole.player_id, PlayerRole.primary_role)
            .join(
                latest_role,
                and_(
                    PlayerRole.player_id == latest_role.c.player_id,
                    PlayerRole.season == latest_role.c.latest_season,
                ),
            )
        ).all()
        player_rows = session.execute(
            select(Player.player_id, Player.current_league_id)
            .where(Player.player_id.in_(candidate_ids))
        ).all()

    role_lookup = {int(player_id): primary_role for player_id, primary_role in role_rows}
    league_lookup = {
        int(player_id): _league_tier_lookup().get(int(current_league_id))
        for player_id, current_league_id in player_rows
        if current_league_id not in (None, "")
    }
    return {
        player_id: {
            "primary_role": role_lookup.get(player_id),
            "league_level": league_lookup.get(player_id),
        }
        for player_id in candidate_ids
    }


@lru_cache(maxsize=1)
def _league_tier_lookup() -> dict[int, int]:
    from config import get_settings

    return {
        int(row["league_id"]): int(row.get("tier") or 3)
        for row in get_settings().load_json("leagues.json")
    }


def _implied_market_value(row: pd.Series) -> float:
    age = float(row.get("age") or 24.0)
    position_group = str(row.get("position_group") or "U")
    league_level = float(row.get("league_level") or 3.0)
    per90_output = float(row.get("per90_output") or 0.0)
    trajectory = float(row.get("trajectory") or 0.0)

    position_multiplier = {
        "F": 1.20,
        "M": 1.00,
        "D": 0.92,
        "G": 0.80,
    }.get(position_group, 0.90)
    age_multiplier = 1.15 if age <= 23 else 1.05 if age <= 26 else 0.90 if age <= 29 else 0.75
    league_multiplier = max(0.55, 1.30 - (0.15 * league_level))
    trajectory_multiplier = 1.0 + max(min(trajectory, 0.20), -0.15)

    base_value = 120000.0 + (per90_output * 175000.0)
    return round(max(150000.0, base_value * position_multiplier * age_multiplier * league_multiplier * trajectory_multiplier), 2)


def _age(birth_date: date) -> float:
    return round((date.today() - birth_date).days / 365.25, 2)


def _player_age_value(player: dict[str, Any]) -> float:
    birth_date = player.get("birth_date")
    if birth_date is not None:
        return _age(birth_date)
    current_age_years = player.get("current_age_years")
    if current_age_years not in (None, ""):
        try:
            return float(current_age_years)
        except (TypeError, ValueError):
            pass
    return 24.0


def _contract_remaining_years(contract_expiry: date | None) -> float:
    if contract_expiry is None:
        return 0.0
    return max(0.0, round((contract_expiry - date.today()).days / 365.25, 2))


def _row_to_dict(row: Any, model: Any) -> dict[str, Any]:
    return {
        column.name: getattr(row, column.name)
        for column in model.__table__.columns
    }


def _resolve_model_path() -> Path:
    from config import get_settings

    return get_settings().project_root / VALUE_MODEL_PATH
