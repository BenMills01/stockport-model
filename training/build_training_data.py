"""Training dataset construction for the three unfitted ML models.

Each builder queries the DB and returns a plain DataFrame ready to pass to the
corresponding train_*() function in models/.

Usage
-----
    from training.build_training_data import (
        build_championship_projection_training_data,
        build_availability_training_data,
        build_financial_value_training_data,
    )
    df = build_championship_projection_training_data()
    # inspect df.shape, df.head(), df.describe() before fitting

Notes
-----
The builders work with whatever rows are currently in the DB.  When the DB is
freshly seeded, these return empty DataFrames and the train_*() functions will
raise ValueError — that's by design (fit once you have data).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import text

from db.session import session_scope


# Minimum minutes a player must have played in a season to be used as a
# training example.  Below this the per-90 stats are too noisy to be useful.
_MIN_ORIGIN_MINUTES = 540   # 6 full matches
_MIN_DEST_MINUTES = 450     # 5 full matches

# Championship league_id in the API-Football / DB convention.
_CHAMPIONSHIP_LEAGUE_ID = 40


# ---------------------------------------------------------------------------
# 1. Championship projection
# ---------------------------------------------------------------------------

def build_championship_projection_training_data() -> pd.DataFrame:
    """Build one training row per player × inter-league season transition.

    Approach
    --------
    Finds every player who appeared in two consecutive seasons in different
    leagues (both with sufficient minutes).  For each such transition, computes:

    * ``origin_*`` columns  — per-90 stats from the pre-move season
    * ``target_*`` columns  — per-90 stats from the post-move season
    * ``target_starter``    — bool: started ≥75 % of destination-side matches

    The model is useful in any direction (lower-league → Championship or vice
    versa), so we keep all transitions, not just ones ending in league 40.
    Callers who want only specific pairs can filter on ``destination_league_id``.

    Returns
    -------
    pd.DataFrame with columns matching what ``train_projection_model()`` expects:
    origin_league_id, destination_league_id, league_pair, age_at_transfer,
    primary_role, origin_team_league_position, destination_team_league_position,
    origin_{metric}_per90 …, target_{metric}_per90 …, target_starter.
    """
    with session_scope() as session:
        # One row per (player, season, league) with aggregated match stats.
        season_agg_rows = session.execute(text("""
            SELECT
                player_id,
                season,
                league_id,
                SUM(COALESCE(minutes, 0))           AS total_minutes,
                COUNT(*)                             AS appearances,
                SUM(COALESCE(goals_scored, 0))       AS goals,
                SUM(COALESCE(assists, 0))            AS assists,
                SUM(COALESCE(shots_total, 0))        AS shots_total,
                SUM(COALESCE(shots_on_target, 0))    AS shots_on_target,
                SUM(COALESCE(passes_total, 0))       AS passes_total,
                SUM(COALESCE(passes_key, 0))         AS passes_key,
                AVG(COALESCE(pass_accuracy, 0))      AS pass_accuracy_avg,
                SUM(COALESCE(tackles_total, 0))      AS tackles_total,
                SUM(COALESCE(tackles_interceptions,0)) AS interceptions,
                SUM(COALESCE(dribbles_attempts, 0))  AS dribbles_attempts,
                SUM(COALESCE(dribbles_success, 0))   AS dribbles_success,
                SUM(COALESCE(duels_total, 0))        AS duels_total,
                SUM(COALESCE(duels_won, 0))          AS duels_won,
                SUM(COALESCE(saves, 0))              AS saves,
                SUM(CASE WHEN NOT is_substitute THEN 1 ELSE 0 END) AS starts,
                MAX(team)                            AS primary_team
            FROM match_performances
            GROUP BY player_id, season, league_id
            HAVING SUM(COALESCE(minutes, 0)) >= :min_minutes
            ORDER BY player_id, season, league_id
        """), {"min_minutes": _MIN_ORIGIN_MINUTES}).mappings().all()

        # Player birth dates for age calculation.
        player_birth_dates: dict[int, date | None] = {
            row["player_id"]: row["birth_date"]
            for row in session.execute(
                text("SELECT player_id, birth_date FROM players")
            ).mappings().all()
        }

        # Primary role per (player, season).
        role_rows: dict[tuple[int, str], str] = {
            (row["player_id"], row["season"]): row["primary_role"]
            for row in session.execute(
                text("SELECT player_id, season, primary_role FROM player_roles")
            ).mappings().all()
        }

        # Most recent league position per (league, team) — used for both sides
        # of the transition.  We take the latest snapshot date as the position.
        position_rows = session.execute(text("""
            SELECT DISTINCT ON (league_id, team_name)
                league_id,
                team_name,
                position
            FROM standings_snapshots
            ORDER BY league_id, team_name, date DESC
        """)).mappings().all()
        team_position: dict[tuple[int, str], int] = {
            (int(r["league_id"]), r["team_name"]): int(r["position"])
            for r in position_rows
        }

    # Build a dict keyed by (player_id, season, league_id) → agg row.
    agg: dict[tuple[int, str, int], dict[str, Any]] = {
        (int(r["player_id"]), str(r["season"]), int(r["league_id"])): dict(r)
        for r in season_agg_rows
    }

    # Group by player_id to find consecutive transitions.
    player_seasons: dict[int, list[tuple[str, int]]] = {}
    for player_id, season, league_id in agg:
        player_seasons.setdefault(player_id, []).append((season, league_id))

    metric_names = [
        "goals", "assists", "shots_total", "shots_on_target", "passes_total",
        "passes_key", "pass_accuracy_avg", "tackles_total", "interceptions",
        "dribbles_attempts", "dribbles_success", "duels_total", "duels_won",
        "saves",
    ]

    rows: list[dict[str, Any]] = []
    for player_id, season_league_pairs in player_seasons.items():
        # Sort by season string (works for "2021", "2022", "2022/23", etc.).
        sorted_pairs = sorted(season_league_pairs, key=lambda sl: sl[0])

        for idx in range(len(sorted_pairs) - 1):
            origin_season, origin_league_id = sorted_pairs[idx]
            dest_season, dest_league_id = sorted_pairs[idx + 1]

            # Only create a training row if the player changed leagues.
            if origin_league_id == dest_league_id:
                continue

            origin_key = (player_id, origin_season, origin_league_id)
            dest_key = (player_id, dest_season, dest_league_id)

            if origin_key not in agg or dest_key not in agg:
                continue

            origin = agg[origin_key]
            dest = agg[dest_key]

            if int(dest["total_minutes"]) < _MIN_DEST_MINUTES:
                continue

            origin_minutes = max(int(origin["total_minutes"]), 1)
            dest_minutes = max(int(dest["total_minutes"]), 1)

            # Age at the approximate transfer date — start of destination season.
            birth_date = player_birth_dates.get(player_id)
            age_at_transfer = _age_years(birth_date) if birth_date else 24.0

            # League position for origin team, destination team.
            origin_pos = team_position.get(
                (origin_league_id, str(origin.get("primary_team") or "")), 12
            )
            dest_pos = team_position.get(
                (dest_league_id, str(dest.get("primary_team") or "")), 12
            )

            row: dict[str, Any] = {
                "player_id": player_id,
                "origin_league_id": origin_league_id,
                "destination_league_id": dest_league_id,
                "league_pair": f"{origin_league_id}->{dest_league_id}",
                "age_at_transfer": age_at_transfer,
                "primary_role": role_rows.get(
                    (player_id, origin_season),
                    role_rows.get((player_id, dest_season), "unknown"),
                ),
                "origin_team_league_position": float(origin_pos),
                "destination_team_league_position": float(dest_pos),
            }

            # Per-90 metrics for origin (features) and destination (targets).
            for metric in metric_names:
                origin_val = float(origin.get(metric) or 0.0)
                dest_val = float(dest.get(metric) or 0.0)
                row[f"origin_{metric}_per90"] = origin_val / origin_minutes * 90.0
                row[f"target_{metric}_per90"] = dest_val / dest_minutes * 90.0

            # target_starter: started ≥75 % of destination appearances.
            dest_starts = int(dest.get("starts") or 0)
            dest_appearances = int(dest.get("appearances") or 1)
            row["target_starter"] = int((dest_starts / dest_appearances) >= 0.75)

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    # Drop the internal player_id column before returning — the model doesn't need it.
    return frame.drop(columns=["player_id"])


# ---------------------------------------------------------------------------
# 2. Availability risk
# ---------------------------------------------------------------------------

def build_availability_training_data() -> pd.DataFrame:
    """Build one training row per player × completed-season window.

    Features
    --------
    Computed from injury/sidelined history and match appearances in the 3 years
    *before* the reference date (end of a completed season).

    Target
    ------
    ``target_available_75pct``: 1 if the player started ≥75 % of their team's
    fixtures in the *following* season, 0 otherwise.

    The predictor is forward-looking: features from season N predict
    availability in season N+1.  This mirrors how the model is used in
    production (evaluate a player *before* signing them).

    Returns
    -------
    pd.DataFrame with columns matching what ``train_availability_model()`` expects.
    """
    with session_scope() as session:
        # All (player, season) pairs with enough appearances.
        player_season_rows = session.execute(text("""
            SELECT
                player_id,
                season,
                league_id,
                SUM(COALESCE(minutes, 0))                            AS total_minutes,
                SUM(CASE WHEN NOT is_substitute THEN 1 ELSE 0 END)  AS starts,
                COUNT(*)                                             AS appearances,
                MAX(team)                                            AS primary_team,
                MAX(position)                                        AS primary_position
            FROM match_performances
            GROUP BY player_id, season, league_id
            HAVING COUNT(*) >= 5
            ORDER BY player_id, season
        """)).mappings().all()

        # Team total fixtures per (team, season, league) — needed for the target.
        team_fixture_counts = session.execute(text("""
            SELECT
                league_id,
                season,
                home_team AS team_name,
                COUNT(*) AS fixture_count
            FROM fixtures
            GROUP BY league_id, season, home_team
            UNION ALL
            SELECT
                league_id,
                season,
                away_team,
                COUNT(*)
            FROM fixtures
            GROUP BY league_id, season, away_team
        """)).mappings().all()

        # Sidelined records: (player_id, start_date, end_date, type).
        sidelined_rows = session.execute(text("""
            SELECT player_id, start_date, end_date, type
            FROM sidelined
            ORDER BY player_id, start_date
        """)).mappings().all()

        # Injury records.
        injury_rows = session.execute(text("""
            SELECT player_id, date, type
            FROM injuries
            ORDER BY player_id, date
        """)).mappings().all()

        # Player birth dates and positions.
        player_meta: dict[int, dict[str, Any]] = {
            row["player_id"]: dict(row)
            for row in session.execute(
                text("SELECT player_id, birth_date FROM players")
            ).mappings().all()
        }

    # Index sidelined and injury records by player.
    sidelined_by_player: dict[int, list[dict[str, Any]]] = {}
    for row in sidelined_rows:
        sidelined_by_player.setdefault(int(row["player_id"]), []).append(dict(row))

    injury_by_player: dict[int, list[dict[str, Any]]] = {}
    for row in injury_rows:
        injury_by_player.setdefault(int(row["player_id"]), []).append(dict(row))

    # Index team fixture counts: (team_name, season) → count.
    team_fixtures: dict[tuple[str, str], int] = {}
    for row in team_fixture_counts:
        key = (str(row["team_name"]), str(row["season"]))
        team_fixtures[key] = team_fixtures.get(key, 0) + int(row["fixture_count"])

    # Group by player.
    player_seasons: dict[int, list[dict[str, Any]]] = {}
    for row in player_season_rows:
        player_seasons.setdefault(int(row["player_id"]), []).append(dict(row))

    training_rows: list[dict[str, Any]] = []
    for player_id, seasons in player_seasons.items():
        sorted_seasons = sorted(seasons, key=lambda r: r["season"])

        for idx in range(len(sorted_seasons) - 1):
            feature_season = sorted_seasons[idx]
            target_season = sorted_seasons[idx + 1]

            feature_season_str = str(feature_season["season"])
            target_season_str = str(target_season["season"])

            # Reference date = approximate end of feature season.
            # We use the feature season year as a rough date (Dec 31).
            try:
                ref_year = int(str(feature_season_str)[:4])
                reference_date = date(ref_year, 12, 31)
            except (ValueError, TypeError):
                continue

            # Compute feature-side availability metrics using the 3yr lookback
            # window ending at the reference date.
            features = _compute_availability_features_for_window(
                player_id=player_id,
                sidelined_records=sidelined_by_player.get(player_id, []),
                injury_records=injury_by_player.get(player_id, []),
                match_seasons=sorted_seasons[: idx + 1],
                reference_date=reference_date,
            )

            # Compute target: was the player available ≥75 % in the following season?
            target_team = str(target_season.get("primary_team") or "")
            target_starts = int(target_season.get("starts") or 0)
            total_team_fixtures = team_fixtures.get((target_team, target_season_str), 0)
            if total_team_fixtures == 0:
                # Fall back to how many fixtures the player appeared in.
                total_team_fixtures = max(int(target_season.get("appearances") or 1), 1)
            target_available_75pct = int(target_starts / total_team_fixtures >= 0.75)

            # Age at reference date.
            birth_date = player_meta.get(player_id, {}).get("birth_date")
            age = _age_years(birth_date, reference_date) if birth_date else 25.0

            # Position group from the feature season.
            raw_position = str(feature_season.get("primary_position") or "U")
            position_group = raw_position[0].upper() if raw_position else "U"

            training_rows.append(
                {
                    **features,
                    "age": age,
                    "position_group": position_group,
                    "target_available_75pct": target_available_75pct,
                }
            )

    if not training_rows:
        return pd.DataFrame()

    return pd.DataFrame(training_rows)


def _compute_availability_features_for_window(
    *,
    player_id: int,
    sidelined_records: list[dict[str, Any]],
    injury_records: list[dict[str, Any]],
    match_seasons: list[dict[str, Any]],
    reference_date: date,
) -> dict[str, Any]:
    """Compute availability features for a player up to reference_date."""

    lookback_start = reference_date - timedelta(days=365 * 3)

    # Sidelined episodes within lookback window.
    recent_sidelined = [
        r for r in sidelined_records
        if r.get("start_date") and r["start_date"] >= lookback_start
    ]

    recent_injuries = [
        r for r in injury_records
        if r.get("date") and r["date"] >= lookback_start
    ]

    # Injury frequency = count of sidelined episodes.
    injury_frequency_3yr = len(recent_sidelined)

    # Duration stats.
    durations = [
        max(0, ((r["end_date"] or reference_date) - r["start_date"]).days)
        for r in recent_sidelined
        if r.get("start_date")
    ]
    avg_injury_duration = float(sum(durations) / len(durations)) if durations else None
    max_injury_duration = int(max(durations)) if durations else None

    # Muscle injury count.
    muscle_terms = ("muscle", "hamstring", "thigh", "calf", "groin")
    muscle_injury_count = sum(
        1 for r in recent_sidelined
        if any(term in str(r.get("type") or "").lower() for term in muscle_terms)
    )

    # Recurrence rate: fraction of injury types that recurred.
    from collections import Counter
    type_labels = [
        str(r.get("type") or "").strip().lower()
        for r in recent_sidelined
        if str(r.get("type") or "").strip()
    ]
    recurrence_rate: float | None = None
    if type_labels:
        counts = Counter(type_labels)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        recurrence_rate = float(repeated / len(type_labels))

    # Days since last injury.
    days_since_last_injury: int | None = None
    all_end_dates = (
        [r["end_date"] for r in recent_sidelined if r.get("end_date")]
        + [r["date"] for r in recent_injuries if r.get("date")]
    )
    if all_end_dates:
        last_event = max(all_end_dates)
        days_since_last_injury = (reference_date - last_event).days

    # Availability rate over 3yr lookback — use match seasons in window.
    lookback_seasons = [s for s in match_seasons if str(s["season"])[:4] >= str(lookback_start.year)]
    avail_rates = []
    for s in lookback_seasons:
        starts = int(s.get("starts") or 0)
        appearances = int(s.get("appearances") or 0)
        if appearances > 0:
            avail_rates.append(starts / appearances)
    availability_rate_3yr = float(sum(avail_rates) / len(avail_rates)) if avail_rates else None

    # Minutes continuity = consecutive full-match seasons.
    minutes_continuity = 0
    for s in reversed(lookback_seasons):
        minutes = int(s.get("total_minutes") or 0)
        appearances = int(s.get("appearances") or 1)
        avg_minutes = minutes / appearances
        if avg_minutes >= 60:
            minutes_continuity += 1
        else:
            break

    return {
        "availability_rate_3yr": availability_rate_3yr,
        "injury_frequency_3yr": injury_frequency_3yr,
        "avg_injury_duration": avg_injury_duration,
        "max_injury_duration": max_injury_duration,
        "muscle_injury_count": muscle_injury_count,
        "recurrence_rate": recurrence_rate,
        "days_since_last_injury": days_since_last_injury,
        "minutes_continuity": float(minutes_continuity),
    }


# ---------------------------------------------------------------------------
# 3. Financial value
# ---------------------------------------------------------------------------

def build_financial_value_training_data() -> pd.DataFrame:
    """Build one training row per signed player with a known fee.

    Source
    ------
    ``outcomes`` table — rows where ``fee_paid`` is non-null represent actual
    completed signings with a verified transfer fee.

    Features are read from the player's profile, last market value snapshot
    before the signing date, and match performance in the season immediately
    before signing.

    Returns
    -------
    pd.DataFrame with columns matching what ``train_value_model()`` expects:
    age, position_group, role, league_level, contract_remaining_years,
    market_value_pretransfer, per90_output, fee_paid.
    """
    with session_scope() as session:
        outcome_rows = session.execute(text("""
            SELECT
                o.player_id,
                o.brief_id,
                o.signed_date,
                o.fee_paid,
                o.wage_annual,
                o.contract_years
            FROM outcomes o
            WHERE o.fee_paid IS NOT NULL
              AND o.signed_date IS NOT NULL
        """)).mappings().all()

        if not outcome_rows:
            return pd.DataFrame()

        player_ids = list({int(r["player_id"]) for r in outcome_rows})

        # Market values — we want the snapshot closest to (but before) signing.
        market_value_rows = session.execute(text("""
            SELECT player_id, date, market_value_eur, contract_expiry, wage_estimate
            FROM market_values
            WHERE player_id = ANY(:ids)
            ORDER BY player_id, date DESC
        """), {"ids": player_ids}).mappings().all()

        # Player meta.
        player_meta: dict[int, dict[str, Any]] = {
            row["player_id"]: dict(row)
            for row in session.execute(text("""
                SELECT player_id, birth_date, current_league_id
                FROM players
                WHERE player_id = ANY(:ids)
            """), {"ids": player_ids}).mappings().all()
        }

        # Player roles — latest before signing date.
        player_role_rows = session.execute(text("""
            SELECT player_id, season, primary_role
            FROM player_roles
            WHERE player_id = ANY(:ids)
            ORDER BY player_id, season DESC
        """), {"ids": player_ids}).mappings().all()

        # Per-season match aggregates for feature players.
        match_agg_rows = session.execute(text("""
            SELECT
                player_id,
                season,
                league_id,
                SUM(COALESCE(minutes, 0))           AS total_minutes,
                SUM(COALESCE(goals_scored, 0))       AS goals,
                SUM(COALESCE(assists, 0))            AS assists,
                SUM(COALESCE(shots_total, 0))        AS shots_total,
                SUM(COALESCE(passes_total, 0))       AS passes_total,
                SUM(COALESCE(tackles_total, 0))      AS tackles_total,
                MAX(position)                        AS primary_position
            FROM match_performances
            WHERE player_id = ANY(:ids)
            GROUP BY player_id, season, league_id
            ORDER BY player_id, season DESC
        """), {"ids": player_ids}).mappings().all()

        # League tier lookup.
        league_tier_rows = session.execute(text("""
            SELECT DISTINCT league_id, position
            FROM standings_snapshots
        """)).mappings().all()

    # Index market values by player.
    mv_by_player: dict[int, list[dict[str, Any]]] = {}
    for row in market_value_rows:
        mv_by_player.setdefault(int(row["player_id"]), []).append(dict(row))

    # Index roles by player (latest season first).
    role_by_player: dict[int, str] = {}
    for row in player_role_rows:
        pid = int(row["player_id"])
        if pid not in role_by_player:
            role_by_player[pid] = str(row["primary_role"])

    # Index match aggregates: player_id → list sorted by season desc.
    match_by_player: dict[int, list[dict[str, Any]]] = {}
    for row in match_agg_rows:
        match_by_player.setdefault(int(row["player_id"]), []).append(dict(row))

    # League tier: league_id → tier (approximate from available standings data).
    # We don't have a direct tier column in standings, so we use the player's
    # current_league_id from their profile and the leagues.json via settings.
    from config import get_settings
    settings = get_settings()
    league_config = {
        int(league["league_id"]): int(league.get("tier") or 3)
        for league in settings.load_json("leagues.json")
    }

    training_rows: list[dict[str, Any]] = []
    for outcome in outcome_rows:
        player_id = int(outcome["player_id"])
        signed_date: date = outcome["signed_date"]
        fee_paid = float(outcome["fee_paid"])

        if fee_paid <= 0:
            continue

        meta = player_meta.get(player_id, {})

        # Market value snapshot before signing.
        mv_snapshot = _latest_mv_before(mv_by_player.get(player_id, []), signed_date)
        market_value_pretransfer = float(mv_snapshot.get("market_value_eur") or 0.0)
        contract_expiry: date | None = mv_snapshot.get("contract_expiry")
        contract_remaining_years = (
            max(0.0, (contract_expiry - signed_date).days / 365.25)
            if contract_expiry else 0.0
        )

        # Age at signing.
        birth_date = meta.get("birth_date")
        age = _age_years(birth_date, signed_date) if birth_date else 24.0

        # Position from most recent match data.
        latest_match_season = (match_by_player.get(player_id) or [{}])[0]
        raw_position = str(latest_match_season.get("primary_position") or "U")
        position_group = raw_position[0].upper() if raw_position else "U"

        # Role.
        role = role_by_player.get(player_id, "unknown")

        # League level (tier) of origin club.
        origin_league_id = int(latest_match_season.get("league_id") or 0)
        league_level = float(league_config.get(origin_league_id, 3))

        # Per-90 output: mean across key output metrics in last season before signing.
        per90_output = _compute_per90_output(latest_match_season)

        training_rows.append(
            {
                "age": age,
                "position_group": position_group,
                "role": role,
                "league_level": league_level,
                "contract_remaining_years": contract_remaining_years,
                "market_value_pretransfer": market_value_pretransfer,
                "per90_output": per90_output,
                "fee_paid": fee_paid,
            }
        )

    if not training_rows:
        return pd.DataFrame()

    return pd.DataFrame(training_rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _age_years(birth_date: date, reference: date | None = None) -> float:
    ref = reference or date.today()
    return round((ref - birth_date).days / 365.25, 1)


def _latest_mv_before(mv_records: list[dict[str, Any]], cutoff: date) -> dict[str, Any]:
    """Return the market value snapshot closest to but not after cutoff."""
    candidates = [r for r in mv_records if r.get("date") and r["date"] <= cutoff]
    if not candidates:
        return {}
    return max(candidates, key=lambda r: r["date"])


def _compute_per90_output(match_season: dict[str, Any]) -> float:
    """Compute a single per-90 output index from a match-season aggregate row."""
    total_minutes = max(float(match_season.get("total_minutes") or 0.0), 1.0)
    goals_per90 = float(match_season.get("goals") or 0.0) / total_minutes * 90.0
    assists_per90 = float(match_season.get("assists") or 0.0) / total_minutes * 90.0
    shots_per90 = float(match_season.get("shots_total") or 0.0) / total_minutes * 90.0
    passes_per90 = float(match_season.get("passes_total") or 0.0) / total_minutes * 90.0
    tackles_per90 = float(match_season.get("tackles_total") or 0.0) / total_minutes * 90.0
    # Weighted sum of key output metrics — equal weights here, can be refined.
    return round(
        (goals_per90 * 0.30)
        + (assists_per90 * 0.20)
        + (shots_per90 * 0.15)
        + (passes_per90 * 0.05)
        + (tackles_per90 * 0.05),
        4,
    )
