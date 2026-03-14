"""Validation and audit helpers."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, roc_auc_score
from sqlalchemy import and_, desc, select

from db.schema import Brief, Outcome, Override, Pipeline, PredictionLog
from db.session import session_scope


def temporal_backtest(model_name: str, window_dates: list[str | date]) -> dict[str, Any]:
    """Evaluate historical predictions by window date using logged predictions."""

    score_column = _score_column_for_model(model_name)
    with session_scope() as session:
        predictions = list(session.scalars(select(PredictionLog)))
        outcomes = list(session.scalars(select(Outcome)))
    prediction_frame = pd.DataFrame([_row_to_dict(row, PredictionLog) for row in predictions])
    outcome_frame = pd.DataFrame([_row_to_dict(row, Outcome) for row in outcomes])
    return _temporal_backtest_from_frames(
        prediction_frame=prediction_frame,
        outcome_frame=outcome_frame,
        score_column=score_column,
        window_dates=window_dates,
    )


def compute_outcome_metrics(window: str | None = None) -> dict[str, Any]:
    """Summarise hit/miss outcomes for signed players."""

    with session_scope() as session:
        outcomes = list(session.scalars(select(Outcome)))
    frame = pd.DataFrame([_row_to_dict(row, Outcome) for row in outcomes])
    return _compute_outcome_metrics_from_frame(frame, window=window)


def post_window_audit(window: str) -> dict[str, Any]:
    """Produce a structured audit summary for a recruitment window."""

    with session_scope() as session:
        briefs = list(session.scalars(select(Brief)))
        pipeline_rows = list(session.scalars(select(Pipeline)))
        overrides = list(session.scalars(select(Override)))
        outcomes = list(session.scalars(select(Outcome)))
    return _post_window_audit_from_frames(
        briefs=pd.DataFrame([_row_to_dict(row, Brief) for row in briefs]),
        pipeline=pd.DataFrame([_row_to_dict(row, Pipeline) for row in pipeline_rows]),
        overrides=pd.DataFrame([_row_to_dict(row, Override) for row in overrides]),
        outcomes=pd.DataFrame([_row_to_dict(row, Outcome) for row in outcomes]),
        window=window,
    )


def calibration_check() -> dict[str, Any]:
    """Compare predicted and realised success rates by decile."""

    with session_scope() as session:
        predictions = list(session.scalars(select(PredictionLog)))
        outcomes = list(session.scalars(select(Outcome)))
    prediction_frame = pd.DataFrame([_row_to_dict(row, PredictionLog) for row in predictions])
    outcome_frame = pd.DataFrame([_row_to_dict(row, Outcome) for row in outcomes])
    return _calibration_check_from_frames(prediction_frame=prediction_frame, outcome_frame=outcome_frame)


def _temporal_backtest_from_frames(
    *,
    prediction_frame: pd.DataFrame,
    outcome_frame: pd.DataFrame,
    score_column: str,
    window_dates: list[str | date],
) -> dict[str, Any]:
    if prediction_frame.empty or outcome_frame.empty:
        return {"windows": []}

    merged = prediction_frame.merge(outcome_frame, on=["player_id", "brief_id"], how="inner", suffixes=("_pred", "_out"))
    merged["prediction_date"] = pd.to_datetime(merged["prediction_date"])
    merged["success"] = _success_series(merged)

    windows = []
    for window_date in window_dates:
        window_ts = pd.Timestamp(window_date)
        subset = merged[merged["prediction_date"] <= window_ts].copy()
        if subset.empty or subset["success"].nunique() < 2:
            windows.append(
                {
                    "window_date": window_ts.date().isoformat(),
                    "auc": None,
                    "precision@20": None,
                    "coverage_of_prediction_intervals": None,
                    "calibration_curve": [],
                }
            )
            continue
        scores = pd.to_numeric(subset[score_column], errors="coerce").fillna(0.0)
        target = subset["success"].astype(int)
        ranked = subset.assign(score=scores).sort_values("score", ascending=False)
        precision_at_20 = float(ranked.head(20)["success"].mean())
        windows.append(
            {
                "window_date": window_ts.date().isoformat(),
                "auc": float(roc_auc_score(target, scores)),
                "precision@20": precision_at_20,
                "coverage_of_prediction_intervals": None,
                "calibration_curve": _decile_calibration(scores, target),
            }
        )
    return {"windows": windows}


def _compute_outcome_metrics_from_frame(frame: pd.DataFrame, window: str | None = None) -> dict[str, Any]:
    if frame.empty:
        return {
            "total_signed": 0,
            "total_assessed": 0,
            "hit_rates": {},
            "model_added_wins": 0,
            "missed_opportunities_identified": 0,
        }

    if window:
        frame = frame[frame["signed_date"].astype(str).str.contains(window, na=False)].copy()
    assessed = frame[
        frame[["performance_hit", "financial_hit", "availability_hit"]]
        .notna()
        .any(axis=1)
    ].copy()
    hit_rates = {
        "performance": float(assessed["performance_hit"].fillna(False).mean()) if not assessed.empty else None,
        "financial": float(assessed["financial_hit"].fillna(False).mean()) if not assessed.empty else None,
        "availability": float(assessed["availability_hit"].fillna(False).mean()) if not assessed.empty else None,
    }
    return {
        "total_signed": int(len(frame.index)),
        "total_assessed": int(len(assessed.index)),
        "hit_rates": hit_rates,
        "model_added_wins": int(
            (
                assessed["performance_hit"].fillna(False)
                & assessed["financial_hit"].fillna(False)
            ).sum()
        ),
        "missed_opportunities_identified": int(assessed["failure_type"].fillna("").str.contains("missed", case=False).sum()),
    }


def _post_window_audit_from_frames(
    *,
    briefs: pd.DataFrame,
    pipeline: pd.DataFrame,
    overrides: pd.DataFrame,
    outcomes: pd.DataFrame,
    window: str,
) -> dict[str, Any]:
    brief_subset = briefs[briefs["created_date"].astype(str).str.contains(window, na=False)] if not briefs.empty else briefs
    pipeline_subset = pipeline[pipeline["added_date"].astype(str).str.contains(window, na=False)] if not pipeline.empty else pipeline
    override_subset = overrides[overrides["override_date"].astype(str).str.contains(window, na=False)] if not overrides.empty else overrides
    outcome_subset = outcomes[outcomes["signed_date"].astype(str).str.contains(window, na=False)] if not outcomes.empty else outcomes
    stage_counts = pipeline_subset["stage"].value_counts().to_dict() if not pipeline_subset.empty else {}
    return {
        "window": window,
        "briefs_created": int(len(brief_subset.index)) if not brief_subset.empty else 0,
        "players_progressed_by_stage": {str(key): int(value) for key, value in stage_counts.items()},
        "signed": int(len(outcome_subset.index)) if not outcome_subset.empty else 0,
        "overrides_logged": int(len(override_subset.index)) if not override_subset.empty else 0,
        "override_outcomes": override_subset["outcome"].fillna("pending").value_counts().to_dict() if not override_subset.empty else {},
        "data_quality_incidents": [],
        "coverage_gaps_discovered": [],
    }


def _calibration_check_from_frames(
    *,
    prediction_frame: pd.DataFrame,
    outcome_frame: pd.DataFrame,
) -> dict[str, Any]:
    if prediction_frame.empty or outcome_frame.empty:
        return {"calibration_curve": [], "drift_alert": False}
    merged = prediction_frame.merge(outcome_frame, on=["player_id", "brief_id"], how="inner")
    if merged.empty:
        return {"calibration_curve": [], "drift_alert": False}
    predicted = pd.to_numeric(merged["availability_risk_prob"], errors="coerce").fillna(0.0)
    actual = _success_series(merged).astype(int)
    curve = _decile_calibration(1.0 - predicted, actual)
    drift_alert = any(abs(bin_row["predicted_rate"] - bin_row["actual_rate"]) > 0.05 for bin_row in curve)
    return {"calibration_curve": curve, "drift_alert": drift_alert}


def _score_column_for_model(model_name: str) -> str:
    mapping = {
        "availability_risk": "availability_risk_prob",
        "financial_value": "var_score",
        "role_fit": "role_fit_score",
        "l1_performance": "l1_performance_score",
        "championship_projection": "championship_projection_50th",
        "composite": "composite_score",
    }
    if model_name not in mapping:
        raise ValueError(f"Unknown model_name '{model_name}'")
    return mapping[model_name]


def _success_series(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["performance_hit"].fillna(False)
        & frame["financial_hit"].fillna(False)
        & frame["availability_hit"].fillna(False)
    )


def _decile_calibration(predicted: pd.Series, actual: pd.Series) -> list[dict[str, Any]]:
    if predicted.empty:
        return []
    bins = pd.qcut(predicted.rank(method="first"), q=min(10, len(predicted)), duplicates="drop")
    grouped = pd.DataFrame({"predicted": predicted, "actual": actual, "bin": bins}).groupby("bin", observed=False)
    curve = []
    for idx, (_, group) in enumerate(grouped, start=1):
        curve.append(
            {
                "bin": idx,
                "predicted_rate": float(group["predicted"].mean()),
                "actual_rate": float(group["actual"].mean()),
                "count": int(len(group.index)),
            }
        )
    return curve


def _row_to_dict(row: Any, model: Any) -> dict[str, Any]:
    return {
        column.name: getattr(row, column.name)
        for column in model.__table__.columns
    }
