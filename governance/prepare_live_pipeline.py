"""Bootstrap live analytical tables and readiness from current database data."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import desc, func, select

from config import get_settings
from db.init_db import create_all_tables
from db.schema import MarketValue, MatchPerformance, PlayerRole, WyscoutSeasonStat
from db.session import session_scope
from features.role_classification import classify_roles
from ingestion.common import parse_money_to_eur, upsert_rows


POSITION_GROUPS = ("G", "D", "M", "F")
MARKET_VALUE_KEYS = ("Market value", "market_value", "Market Value")
CONTRACT_KEYS = ("Contract expires", "Contract expiry", "contract_expiry", "contract_expires")


def prepare_live_pipeline(
    *,
    seasons: list[str] | None = None,
    reference_date: date | None = None,
) -> dict[str, Any]:
    """Create missing tables and bootstrap the live analytical layer."""

    create_all_tables()
    role_summary = backfill_player_roles(seasons=seasons)
    market_summary = backfill_market_values_from_wyscout(reference_date=reference_date)
    return {
        "prepared_at": datetime.now(UTC).isoformat(),
        "role_summary": role_summary,
        "market_value_summary": market_summary,
        "summary": summarise_prepared_state(),
    }


def available_seasons() -> list[str]:
    """Return seasons present in match performance data."""

    with session_scope() as session:
        seasons = session.scalars(
            select(MatchPerformance.season).distinct().order_by(desc(MatchPerformance.season))
        )
        return [str(season) for season in seasons]


def backfill_player_roles(*, seasons: list[str] | None = None) -> dict[str, Any]:
    """Populate player_roles for the requested seasons and position groups."""

    target_seasons = [str(season) for season in (seasons or available_seasons())]
    results: list[dict[str, Any]] = []
    total_rows = 0

    for season in target_seasons:
        for position_group in POSITION_GROUPS:
            classified = classify_roles(season=season, position_group=position_group)
            row_count = int(len(classified.index))
            total_rows += row_count
            results.append(
                {
                    "season": season,
                    "position_group": position_group,
                    "rows": row_count,
                }
            )

    return {
        "seasons": target_seasons,
        "runs": results,
        "rows_written": total_rows,
    }


def backfill_market_values_from_wyscout(*, reference_date: date | None = None) -> dict[str, Any]:
    """Populate market_values from the latest Wyscout season-average snapshots."""

    settings = get_settings()
    target_date = reference_date or date.today()

    with session_scope() as session:
        season_rows = list(
            session.scalars(
                select(WyscoutSeasonStat).order_by(
                    desc(WyscoutSeasonStat.season),
                    desc(WyscoutSeasonStat.import_date),
                )
            )
        )

    latest_by_player: dict[int, WyscoutSeasonStat] = {}
    for row in season_rows:
        if row.player_id in latest_by_player:
            continue
        latest_by_player[row.player_id] = row

    rows: list[dict[str, Any]] = []
    with_market_value = 0
    with_contract_expiry = 0
    for record in latest_by_player.values():
        metrics = record.metrics_json or {}
        market_value = _extract_market_value_eur(metrics, settings)
        contract_expiry = _extract_contract_expiry(metrics)
        if market_value is None and contract_expiry is None:
            continue
        if market_value is not None:
            with_market_value += 1
        if contract_expiry is not None:
            with_contract_expiry += 1

        rows.append(
            {
                "player_id": record.player_id,
                "date": target_date,
                "market_value_eur": market_value,
                "contract_expiry": contract_expiry,
                "wage_estimate": _estimate_wage_from_market_value(market_value, record.league_id),
            }
        )

    inserted = upsert_rows(MarketValue, rows, ["player_id", "date"])
    return {
        "reference_date": target_date.isoformat(),
        "rows_written": inserted,
        "rows_with_market_value": with_market_value,
        "rows_with_contract_expiry": with_contract_expiry,
    }


def summarise_prepared_state() -> dict[str, int]:
    """Return key counts for the live analytical layer."""

    with session_scope() as session:
        player_roles = session.scalar(select(func.count()).select_from(PlayerRole))
        market_values = session.scalar(select(func.count()).select_from(MarketValue))
        players = session.scalar(
            select(func.count(func.distinct(MatchPerformance.player_id))).select_from(MatchPerformance)
        )
    return {
        "player_roles": int(player_roles or 0),
        "market_values": int(market_values or 0),
        "players_with_match_data": int(players or 0),
    }


def _extract_market_value_eur(metrics: dict[str, Any], settings: Any) -> Decimal | None:
    for key in MARKET_VALUE_KEYS:
        if key not in metrics:
            continue
        raw_value = metrics.get(key)
        if raw_value in (None, "", "-", "?", "Not available"):
            return None
        if isinstance(raw_value, (int, float, Decimal)):
            return Decimal(str(raw_value)).quantize(Decimal("0.01"))
        parsed = parse_money_to_eur(
            raw_value,
            gbp_to_eur_rate=settings.gbp_to_eur_rate,
            usd_to_eur_rate=settings.usd_to_eur_rate,
            chf_to_eur_rate=settings.chf_to_eur_rate,
        )
        return parsed
    return None


def _extract_contract_expiry(metrics: dict[str, Any]) -> date | None:
    for key in CONTRACT_KEYS:
        raw_value = metrics.get(key)
        if raw_value in (None, "", "-", "?", "Not available"):
            continue
        if isinstance(raw_value, datetime):
            return raw_value.date()
        if isinstance(raw_value, date):
            return raw_value
        if isinstance(raw_value, (int, float)):
            year = int(raw_value)
            if 2000 <= year <= 2100:
                return date(year, 6, 30)
            continue

        text_value = str(raw_value).strip()
        if not text_value:
            continue
        if text_value.isdigit() and len(text_value) == 4:
            year = int(text_value)
            if 2000 <= year <= 2100:
                return date(year, 6, 30)
        try:
            parsed = datetime.fromisoformat(text_value.replace("Z", "+00:00"))
            return parsed.date()
        except ValueError:
            pass
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%b %d, %Y", "%d %b %Y"):
            try:
                return datetime.strptime(text_value, fmt).date()
            except ValueError:
                continue
    return None


def _estimate_wage_from_market_value(market_value_eur: Decimal | None, league_id: int) -> Decimal | None:
    if market_value_eur is None:
        return None

    leagues = {
        int(league["league_id"]): league
        for league in get_settings().load_json("leagues.json")
    }
    tier = int(leagues.get(int(league_id), {}).get("tier") or 3)
    ratio = {
        1: Decimal("0.20"),
        2: Decimal("0.16"),
        3: Decimal("0.12"),
        4: Decimal("0.09"),
    }.get(tier, Decimal("0.12"))
    floor = {
        1: Decimal("180000"),
        2: Decimal("90000"),
        3: Decimal("45000"),
        4: Decimal("30000"),
    }.get(tier, Decimal("45000"))
    estimate = max(floor, market_value_eur * ratio)
    cap = market_value_eur * Decimal("0.40")
    return min(estimate, cap).quantize(Decimal("0.01"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the live Stockport pipeline from current DB data.")
    parser.add_argument("--season", action="append", dest="seasons", default=None, help="Limit preparation to one or more seasons.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = prepare_live_pipeline(seasons=args.seasons)
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
