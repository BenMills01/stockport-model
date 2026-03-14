"""Rolling aggregate feature generation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_rolling(player_per90_df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Compute rolling windows, variability, and trend slopes for per-90 metrics."""

    if player_per90_df.empty:
        return {}

    starts = player_per90_df[player_per90_df["minutes"].fillna(0) >= 45].copy()
    if starts.empty:
        return {}

    metric_columns = [column for column in starts.columns if column.endswith("_per90")]
    output: dict[str, dict[str, float | None]] = {}
    for column in metric_columns:
        series = pd.to_numeric(starts[column], errors="coerce").dropna()
        output[column.removesuffix("_per90")] = {
            "roll_3": _window_mean(series, 3),
            "roll_5": _window_mean(series, 5),
            "roll_10": _window_mean(series, 10),
            "season_avg": float(series.mean()) if not series.empty else None,
            "roll_10_std": _window_std(series, 10),
            "roll_10_cv": _window_cv(series, 10),
            "trend_slope_10": _trend_slope(series.tail(10)),
            "trend_slope_season": _trend_slope(series),
        }
    return output


def _window_mean(series: pd.Series, window: int) -> float | None:
    sample = series.tail(window)
    return float(sample.mean()) if not sample.empty else None


def _window_std(series: pd.Series, window: int) -> float | None:
    sample = series.tail(window)
    if sample.empty:
        return None
    return float(sample.std(ddof=0))


def _window_cv(series: pd.Series, window: int) -> float | None:
    sample = series.tail(window)
    if sample.empty:
        return None
    mean = float(sample.mean())
    if math.isclose(mean, 0.0):
        return None
    std = float(sample.std(ddof=0))
    return std / mean


def _trend_slope(series: pd.Series) -> float | None:
    if len(series.index) < 2:
        return None
    values = series.to_numpy(dtype=float)
    x_values = np.arange(len(values), dtype=float)
    slope = np.polyfit(x_values, values, 1)[0]
    return float(slope)
