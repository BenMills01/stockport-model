"""Role classification via k-means clustering."""

from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select

from config import get_settings
from db.schema import Fixture, Lineup, MatchPerformance, Player, PlayerRole
from db.session import session_scope
from features.per90 import _compute_per90_frame
from ingestion.common import upsert_rows


POSITION_GROUP_METRICS = {
    "G": ["saves_per90", "passes_total_per90", "pass_accuracy"],
    "D": [
        "passes_total_per90",
        "pass_accuracy",
        "passes_key_per90",
        "tackles_total_per90",
        "tackles_interceptions_per90",
        "duels_total_per90",
        "duels_won_per90",
        "dribbles_past_per90",
        "fouls_committed_per90",
        "yellow_cards_per90",
    ],
    "M": [
        "passes_total_per90",
        "pass_accuracy",
        "passes_key_per90",
        "tackles_total_per90",
        "tackles_interceptions_per90",
        "duels_won_per90",
        "shots_total_per90",
        "assists_per90",
        "dribbles_success_per90",
        "fouls_drawn_per90",
    ],
    "F": [
        "goals_scored_per90",
        "shots_total_per90",
        "shots_on_target_per90",
        "assists_per90",
        "passes_key_per90",
        "duels_won_per90",
        "duels_total_per90",
        "dribbles_success_per90",
        "fouls_drawn_per90",
        "passes_total_per90",
    ],
}

TEMPLATE_POSITION_GROUPS = {
    "G": set(),
    "D": {"centre_back", "full_back_wing_back"},
    "M": {"midfield", "transition_midfield"},
    "F": {"wide_creator_runner", "striker"},
}


def classify_roles(season: str, position_group: str) -> pd.DataFrame:
    """Cluster players in a position group and assign primary/secondary roles."""

    position_group = position_group.upper()
    with session_scope() as session:
        match_rows = list(
            session.scalars(
                select(MatchPerformance).where(MatchPerformance.season == season)
            )
        )
        lineup_rows = list(
            session.execute(
                select(Lineup, Fixture.season).join(Fixture, Fixture.fixture_id == Lineup.fixture_id).where(Fixture.season == season)
            )
        )
        player_rows = list(session.scalars(select(Player)))

    match_frame = pd.DataFrame([_row_to_dict(row, MatchPerformance) for row in match_rows])
    lineup_frame = pd.DataFrame(
        [
            {
                **_row_to_dict(lineup, Lineup),
                "season": fixture_season,
            }
            for lineup, fixture_season in lineup_rows
        ]
    )
    player_frame = pd.DataFrame([_row_to_dict(row, Player) for row in player_rows])
    result = _classify_roles_from_frames(
        season=season,
        position_group=position_group,
        match_frame=match_frame,
        lineup_frame=lineup_frame,
        player_frame=player_frame,
        templates=get_settings().load_json("role_templates.json"),
    )

    rows = result[["player_id", "primary_role", "secondary_role", "cluster_confidence"]].to_dict("records")
    for row in rows:
        row["season"] = season
    if rows:
        upsert_rows(PlayerRole, rows, ["player_id", "season"])
    return result


def _classify_roles_from_frames(
    *,
    season: str,
    position_group: str,
    match_frame: pd.DataFrame,
    lineup_frame: pd.DataFrame | None = None,
    player_frame: pd.DataFrame | None = None,
    templates: list[dict[str, Any]],
) -> pd.DataFrame:
    if match_frame.empty:
        return pd.DataFrame(columns=["player_id", "cluster", "primary_role", "secondary_role", "cluster_confidence"])

    season_matches = match_frame[match_frame["season"].astype(str) == str(season)].copy()
    if season_matches.empty:
        return pd.DataFrame(columns=["player_id", "cluster", "primary_role", "secondary_role", "cluster_confidence"])

    per90 = _compute_per90_frame(season_matches)
    if per90.empty:
        return pd.DataFrame(columns=["player_id", "cluster", "primary_role", "secondary_role", "cluster_confidence"])

    player_features = _build_position_group_feature_matrix(
        per90,
        position_group,
        lineup_frame=lineup_frame,
        player_frame=player_frame,
        season=season,
    )
    if player_features.empty:
        return pd.DataFrame(columns=["player_id", "cluster", "primary_role", "secondary_role", "cluster_confidence"])

    feature_columns = [column for column in player_features.columns if column not in {"player_id"}]
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(player_features[feature_columns].fillna(player_features[feature_columns].mean()))

    if len(player_features.index) == 1:
        return _single_player_role_assignment(player_features, position_group, templates)

    cluster_count = _choose_cluster_count(feature_matrix)
    if cluster_count <= 1:
        return _single_player_role_assignment(player_features, position_group, templates)

    kmeans = KMeans(n_clusters=cluster_count, n_init=20, random_state=42)
    clusters = kmeans.fit_predict(feature_matrix)
    distances = kmeans.transform(feature_matrix)

    cluster_role_map = _map_clusters_to_roles(
        centroids=pd.DataFrame(kmeans.cluster_centers_, columns=feature_columns),
        position_group=position_group,
        templates=templates,
    )

    output_rows = []
    for index, player_row in player_features.reset_index(drop=True).iterrows():
        best_idx, second_idx = _two_smallest_indices(distances[index].tolist())
        primary_role = cluster_role_map.get(best_idx, _fallback_role_for_group(position_group))
        secondary_role = None
        if second_idx is not None:
            secondary_distance = float(distances[index][second_idx])
            best_distance = float(distances[index][best_idx])
            if best_distance > 0 and (secondary_distance / best_distance) <= 1.15:
                secondary_role = cluster_role_map.get(second_idx, _fallback_role_for_group(position_group))
        confidence = _cluster_confidence(distances[index].tolist())
        output_rows.append(
            {
                "player_id": int(player_row["player_id"]),
                "cluster": int(clusters[index]),
                "primary_role": primary_role,
                "secondary_role": secondary_role if secondary_role != primary_role else None,
                "cluster_confidence": confidence,
            }
        )

    return pd.DataFrame(output_rows).sort_values("player_id").reset_index(drop=True)


def _build_position_group_feature_matrix(
    per90_frame: pd.DataFrame,
    position_group: str,
    *,
    lineup_frame: pd.DataFrame | None = None,
    player_frame: pd.DataFrame | None = None,
    season: str | None = None,
) -> pd.DataFrame:
    position_group = position_group.upper()
    candidate_metrics = POSITION_GROUP_METRICS[position_group]
    frame = per90_frame.copy()
    frame["position_group"] = frame["position"].map(_position_to_group)
    frame = frame[frame["position_group"] == position_group]
    if frame.empty:
        return pd.DataFrame()

    aggregations: dict[str, Any] = {}
    for metric in candidate_metrics:
        if metric in frame.columns:
            aggregations[metric] = "mean"
    if "pass_accuracy" in frame.columns and "pass_accuracy" not in aggregations:
        aggregations["pass_accuracy"] = "mean"

    grouped = frame.groupby(["player_id", "league_id"], as_index=False).agg(aggregations)
    feature_columns = [column for column in grouped.columns if column not in {"player_id", "league_id"}]
    if not feature_columns:
        return pd.DataFrame()

    for metric in feature_columns:
        grouped[metric] = grouped.groupby("league_id")[metric].rank(pct=True) * 100.0
    collapsed = grouped.groupby("player_id", as_index=False)[feature_columns].mean()
    if position_group == "D":
        collapsed = _merge_defender_context_features(
            collapsed,
            lineup_frame=lineup_frame,
            player_frame=player_frame,
            season=season,
        )
    final_columns = [column for column in collapsed.columns if column != "player_id"]
    return collapsed[["player_id", *final_columns]]


def _merge_defender_context_features(
    player_features: pd.DataFrame,
    *,
    lineup_frame: pd.DataFrame | None,
    player_frame: pd.DataFrame | None,
    season: str | None,
) -> pd.DataFrame:
    merged = player_features.copy()

    if lineup_frame is not None and not lineup_frame.empty:
        context = _build_defender_lineup_context(lineup_frame, season)
        if not context.empty:
            merged = merged.merge(context, on="player_id", how="left")
    if player_frame is not None and not player_frame.empty and "height_cm" in player_frame.columns:
        merged = merged.merge(player_frame[["player_id", "height_cm"]], on="player_id", how="left")

    for column in ("central_def_share", "wide_def_share"):
        if column not in merged.columns:
            merged[column] = 0.0
        else:
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0) * 100.0

    if "height_cm" not in merged.columns:
        merged["height_cm"] = pd.NA
    height_numeric = pd.to_numeric(merged["height_cm"], errors="coerce")
    median_height = height_numeric.median() if height_numeric.notna().any() else 182.0
    merged["height_cm"] = height_numeric.fillna(median_height if pd.notna(median_height) else 182.0)
    return merged


def _build_defender_lineup_context(lineup_frame: pd.DataFrame, season: str | None) -> pd.DataFrame:
    frame = lineup_frame.copy()
    if season is not None and "season" in frame.columns:
        frame = frame[frame["season"].astype(str) == str(season)]
    if frame.empty:
        return pd.DataFrame(columns=["player_id", "central_def_share", "wide_def_share"])

    if "is_starter" in frame.columns:
        frame = frame[frame["is_starter"].fillna(False)]
    if frame.empty or "position_label" not in frame.columns or "grid_position" not in frame.columns:
        return pd.DataFrame(columns=["player_id", "central_def_share", "wide_def_share"])

    frame = frame[frame["position_label"].isin({"D", "M"})].copy()
    parts = frame["grid_position"].astype(str).str.extract(r"^(?P<row>\d+):(?P<col>\d+)$")
    frame["grid_row"] = pd.to_numeric(parts["row"], errors="coerce")
    frame["grid_col"] = pd.to_numeric(parts["col"], errors="coerce")
    frame = frame.dropna(subset=["grid_row", "grid_col"])
    if frame.empty:
        return pd.DataFrame(columns=["player_id", "central_def_share", "wide_def_share"])

    frame["grid_row"] = frame["grid_row"].astype(int)
    frame["grid_col"] = frame["grid_col"].astype(int)
    frame = frame[frame["grid_row"] <= 3].copy()
    if frame.empty:
        return pd.DataFrame(columns=["player_id", "central_def_share", "wide_def_share"])

    frame["row_width"] = frame.groupby(["fixture_id", "team", "grid_row"])["grid_col"].transform("max")
    frame["central_def"] = (
        (frame["grid_row"] <= 2)
        & (frame["row_width"] >= 3)
        & (frame["grid_col"] > 1)
        & (frame["grid_col"] < frame["row_width"])
    )
    frame["wide_def"] = (
        (
            (frame["grid_row"] <= 2)
            & (frame["row_width"] >= 3)
            & ((frame["grid_col"] == 1) | (frame["grid_col"] == frame["row_width"]))
        )
        | (
            (frame["grid_row"] == 3)
            & ((frame["grid_col"] == 1) | (frame["grid_col"] == frame["row_width"]))
        )
    )

    summary = frame.groupby("player_id", as_index=False).agg(
        central_def_share=("central_def", "mean"),
        wide_def_share=("wide_def", "mean"),
    )
    return summary


def _choose_cluster_count(feature_matrix: np.ndarray) -> int:
    sample_count = feature_matrix.shape[0]
    if sample_count <= 2:
        return 1

    min_k = 4
    max_k = 6
    feasible = [k for k in range(min_k, max_k + 1) if 1 < k < sample_count]
    if not feasible:
        return min(sample_count - 1, 2)

    best_k = feasible[0]
    best_score = -1.0
    for k in feasible:
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = model.fit_predict(feature_matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(feature_matrix, labels)
        if score > best_score:
            best_k = k
            best_score = score
    return best_k


def _map_clusters_to_roles(
    *,
    centroids: pd.DataFrame,
    position_group: str,
    templates: list[dict[str, Any]],
) -> dict[int, str]:
    candidates = [
        template
        for template in templates
        if template.get("role_family") in TEMPLATE_POSITION_GROUPS[position_group]
    ]
    if not candidates:
        return {index: _fallback_role_for_group(position_group) for index in centroids.index}

    mapping: dict[int, str] = {}
    for cluster_index, row in centroids.iterrows():
        best_role = None
        best_score = -1.0
        for template in candidates:
            score = 0.0
            for metric, weight in template.get("metrics", {}).items():
                metric_name = metric if metric.endswith("_per90") or metric == "pass_accuracy" else f"{metric}_per90"
                if metric_name in row.index and pd.notna(row[metric_name]):
                    score += float(row[metric_name]) * float(weight)
            if score > best_score:
                best_score = score
                best_role = template["role_name"]
        mapping[int(cluster_index)] = best_role or _fallback_role_for_group(position_group)
    return mapping


def _single_player_role_assignment(
    player_features: pd.DataFrame,
    position_group: str,
    templates: list[dict[str, Any]],
) -> pd.DataFrame:
    fallback_role = _fallback_role_for_group(position_group)
    primary_role = _map_clusters_to_roles(
        centroids=pd.DataFrame([player_features.drop(columns=["player_id"]).iloc[0]]),
        position_group=position_group,
        templates=templates,
    ).get(0, fallback_role)
    return pd.DataFrame(
        [
            {
                "player_id": int(player_features.iloc[0]["player_id"]),
                "cluster": 0,
                "primary_role": primary_role,
                "secondary_role": None,
                "cluster_confidence": 1.0,
            }
        ]
    )


def _two_smallest_indices(values: list[float]) -> tuple[int, int | None]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    best_idx = ordered[0][0]
    second_idx = ordered[1][0] if len(ordered) > 1 else None
    return best_idx, second_idx


def _cluster_confidence(distances: list[float]) -> float:
    positive = [distance for distance in distances if isfinite(distance)]
    if not positive:
        return 0.0
    total = sum(positive)
    if total == 0:
        return 1.0
    best = min(positive)
    return float(max(0.0, min(1.0, 1.0 - (best / total))))


def _position_to_group(position: Any) -> str | None:
    if position is None:
        return None
    label = str(position).strip().upper()
    if not label:
        return None
    if label.startswith("G"):
        return "G"
    if label.startswith(("D", "C", "RWB", "LWB", "RB", "LB")):
        return "D"
    if label.startswith(("M", "DM", "CM", "AM")):
        return "M"
    if label.startswith(("F", "S", "W", "RW", "LW", "ST", "CF")):
        return "F"
    return None


def _fallback_role_for_group(position_group: str) -> str:
    return {
        "G": "goalkeeper",
        "D": "covering_cb",
        "M": "controller",
        "F": "complete_forward",
    }[position_group]


def _row_to_dict(row: Any, model: Any) -> dict[str, Any]:
    return {
        column.name: getattr(row, column.name)
        for column in model.__table__.columns
    }
