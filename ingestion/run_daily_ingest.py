"""Daily orchestrator for the Component 2 ingestion pipeline."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Any

from config import get_settings
from ingestion.api_football import collect_completed_fixtures, collect_fixture_details
from ingestion.api_football import estimate_ingest_request_plan, fetch_api_usage
from ingestion.api_football import fetch_fixture_team_stats, fetch_fixtures
from ingestion.api_football import fetch_lineups, fetch_match_events, fetch_match_performances
from ingestion.fbref import ingest_fbref_player_stats
from ingestion.skillcorner import load_competition_edition_ids, run_skillcorner_ingest
from ingestion.transfermarkt import (
    ingest_market_values,
    ingest_player_profiles,
    ingest_transfer_fees,
    ingest_value_history,
)


LOGGER = logging.getLogger(__name__)

# Run TM + SkillCorner enrichment on Mondays (weekday 0).
_TM_ENRICHMENT_WEEKDAY = 0
_SC_ENRICHMENT_WEEKDAY = 0


def run_tm_enrichment(
    season: str | None = None,
) -> dict[str, Any]:
    """Run the weekly Transfermarkt enrichment pipeline.

    Steps
    -----
    1. Scrape squad market values for all configured TM leagues (and save player
       TM profile URL mappings into SourcePlayerMapping along the way).
    2. Enrich Player rows with preferred foot, secondary nationality, and agent.
    3. Backfill ``fee_paid`` on Transfer rows from scraped transfer history.
    4. Upsert market value time-series into MarketValueHistory.

    Args:
        season: Four-digit year string (e.g. ``"2023"``). Defaults to the current
                calendar year, which matches how TM labels the ongoing season.

    Returns:
        Dict with step-level counts and any error messages.
    """

    if season is None:
        season = str(date.today().year)

    settings = get_settings()
    tm_leagues = settings.load_json("transfermarkt_leagues.json")

    results: dict[str, Any] = {
        "season": season,
        "steps": {},
        "errors": [],
    }

    # Step 1: squad market values (also populates TM source mappings).
    mv_total = 0
    for league_conf in tm_leagues:
        slug: str = league_conf["tm_slug"]
        league_id: int = league_conf["league_id"]
        try:
            count = ingest_market_values(slug, season, league_id=league_id)
            mv_total += count
            LOGGER.info("TM market values: %d rows for %s", count, league_conf["name"])
        except Exception as exc:
            LOGGER.exception("TM market value ingest failed for %s", league_conf["name"])
            results["errors"].append({"step": "market_values", "league": league_conf["name"], "error": str(exc)})
    results["steps"]["market_values"] = mv_total

    # Step 2: player profile enrichment (foot, secondary nationality, agent).
    try:
        profile_result = ingest_player_profiles()
        results["steps"]["player_profiles"] = profile_result
    except Exception as exc:
        LOGGER.exception("TM player profile enrichment failed")
        results["errors"].append({"step": "player_profiles", "error": str(exc)})
        results["steps"]["player_profiles"] = {}

    # Step 3: transfer fee backfill.
    try:
        fee_result = ingest_transfer_fees()
        results["steps"]["transfer_fees"] = fee_result
    except Exception as exc:
        LOGGER.exception("TM transfer fee backfill failed")
        results["errors"].append({"step": "transfer_fees", "error": str(exc)})
        results["steps"]["transfer_fees"] = {}

    # Step 4: market value history time series.
    try:
        history_result = ingest_value_history()
        results["steps"]["value_history"] = history_result
    except Exception as exc:
        LOGGER.exception("TM value history ingest failed")
        results["errors"].append({"step": "value_history", "error": str(exc)})
        results["steps"]["value_history"] = {}

    # Step 5: FBref expected metrics (xG, xA, progressive actions).
    fbref_leagues = settings.load_json("fbref_leagues.json")
    fbref_total = 0
    for league_conf in fbref_leagues:
        fbref_url: str = league_conf["fbref_url"]
        league_id: int = league_conf["league_id"]
        try:
            count = ingest_fbref_player_stats(fbref_url, season, league_id)
            fbref_total += count
            LOGGER.info("FBref expected metrics: %d rows for %s", count, league_conf["name"])
        except Exception as exc:
            LOGGER.exception("FBref ingest failed for %s", league_conf["name"])
            results["errors"].append({"step": "fbref", "league": league_conf["name"], "error": str(exc)})
    results["steps"]["fbref_expected_metrics"] = fbref_total

    return results


def run_skillcorner_enrichment() -> dict[str, Any]:
    """Run the weekly SkillCorner enrichment pipeline.

    Discovers or loads competition edition IDs from config, then runs the full
    SkillCorner ingest (match reconciliation → player reconciliation → physical,
    off-ball runs, pressure, passes).

    Returns the summary dict from ``run_skillcorner_ingest``.
    """

    try:
        edition_ids = load_competition_edition_ids()
    except Exception as exc:
        LOGGER.exception("SkillCorner competition edition discovery failed")
        return {"error": str(exc)}

    if not edition_ids:
        LOGGER.warning("No SkillCorner competition edition IDs available — skipping ingest")
        return {"skipped": True, "reason": "no_edition_ids"}

    return run_skillcorner_ingest(edition_ids)


def run_daily_ingest(
    *,
    from_date: date | None = None,
    to_date: date | None = None,
    state_path: str | Path | None = None,
    lookback_days: int = 2,
    league_ids: list[int] | None = None,
    request_buffer: int = 100,
    persist_state: bool = True,
    run_tm_enrichment_flag: bool = False,
    run_skillcorner_flag: bool = False,
) -> dict[str, Any]:
    """Run the daily ingestion routine and persist the last successful run."""

    settings = get_settings()
    state_file = Path(state_path) if state_path is not None else settings.daily_ingest_state_path
    end_date = to_date or date.today()
    previous_run = None if from_date is not None else load_last_run(state_file)
    start_date = from_date or previous_run or (end_date - timedelta(days=lookback_days))

    results: dict[str, Any] = {
        "from_date": start_date.isoformat(),
        "to_date": end_date.isoformat(),
        "league_ids": list(league_ids) if league_ids is not None else None,
        "steps": {},
        "errors": [],
        "api_budget": {},
    }

    fixture_rows: list[dict[str, Any]] = []
    fixture_details: dict[int, dict[str, Any]] | None = None
    steps = (
        ("fixtures", fetch_fixtures),
        ("match_performances", fetch_match_performances),
        ("fixture_team_stats", fetch_fixture_team_stats),
        ("match_events", fetch_match_events),
        ("lineups", fetch_lineups),
    )

    try:
        fixture_rows = collect_completed_fixtures(start_date, end_date, league_ids=league_ids)
        results["api_budget"] = estimate_ingest_request_plan(
            start_date,
            end_date,
            league_ids=league_ids,
            fixture_count=len(fixture_rows),
        )
    except Exception as exc:  # pragma: no cover - covered via patching around the public function
        LOGGER.exception("Daily ingestion step failed: fixtures")
        results["errors"].append({"step": "fixtures", "error": str(exc)})
        results["steps"]["fixtures"] = 0

    budget_blocked = False
    if "fixtures" not in results["steps"]:
        try:
            usage = fetch_api_usage()
            estimated_detail_calls = int(results["api_budget"].get("estimated_detail_calls", 0))
            remaining = usage.get("requests_remaining")
            results["api_budget"].update(usage)
            results["api_budget"]["request_buffer"] = request_buffer
            results["api_budget"]["estimated_remaining_after_detail"] = (
                None if remaining is None else int(remaining) - estimated_detail_calls
            )
            if remaining is not None and (int(remaining) - estimated_detail_calls) < request_buffer:
                budget_blocked = True
                results["errors"].append(
                    {
                        "step": "api_budget",
                        "error": (
                            "Stopping before detail endpoints to protect API quota "
                            f"(remaining={remaining}, estimated_detail_calls={estimated_detail_calls}, "
                            f"buffer={request_buffer})"
                        ),
                    }
                )
        except Exception as exc:  # pragma: no cover - exercised via mocks in tests
            LOGGER.exception("Daily ingestion step failed: api_budget")
            results["errors"].append({"step": "api_budget", "error": str(exc)})
            budget_blocked = True

    if fixture_rows and not budget_blocked:
        try:
            fixture_details = collect_fixture_details(fixture_rows)
        except Exception:  # pragma: no cover - exercised via live runtime rather than mocks
            LOGGER.warning(
                "Fixture detail prefetch failed; falling back to per-step detail fetches",
                exc_info=True,
            )

    for step_name, function in steps:
        if step_name == "fixtures" and "fixtures" in results["steps"]:
            continue
        if step_name != "fixtures" and budget_blocked:
            results["steps"][step_name] = 0
            continue
        try:
            kwargs: dict[str, Any] = {
                "league_ids": league_ids,
                "fixtures": fixture_rows,
            }
            if step_name != "fixtures":
                kwargs["fixture_details"] = fixture_details
            count = function(start_date, end_date, **kwargs)
            results["steps"][step_name] = count
        except Exception as exc:  # pragma: no cover - exercised via mocks in tests
            LOGGER.exception("Daily ingestion step failed: %s", step_name)
            results["errors"].append({"step": step_name, "error": str(exc)})
            results["steps"][step_name] = 0

    results["fixture_count"] = results["steps"].get("fixtures", 0)
    results["player_count"] = results["steps"].get("match_performances", 0)

    # Run Transfermarkt enrichment weekly (Mondays) or when explicitly requested.
    should_run_tm = run_tm_enrichment_flag or (end_date.weekday() == _TM_ENRICHMENT_WEEKDAY)
    if should_run_tm:
        try:
            tm_results = run_tm_enrichment(season=str(end_date.year))
            results["tm_enrichment"] = tm_results
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Weekly TM enrichment run failed")
            results["errors"].append({"step": "tm_enrichment", "error": str(exc)})

    # Run SkillCorner enrichment weekly (Mondays) or when explicitly requested.
    should_run_sc = run_skillcorner_flag or (end_date.weekday() == _SC_ENRICHMENT_WEEKDAY)
    if should_run_sc:
        try:
            sc_results = run_skillcorner_enrichment()
            results["skillcorner_enrichment"] = sc_results
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Weekly SkillCorner enrichment run failed")
            results["errors"].append({"step": "skillcorner_enrichment", "error": str(exc)})

    if persist_state and not results["errors"]:
        save_last_run(state_file, end_date)

    return results


def load_last_run(state_path: str | Path) -> date | None:
    """Load the last successful ingestion date from disk."""

    path = Path(state_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    last_run = payload.get("last_successful_run")
    return date.fromisoformat(last_run) if last_run else None


def save_last_run(state_path: str | Path, run_date: date) -> None:
    """Persist the last successful ingestion date."""

    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_successful_run": run_date.isoformat(),
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily Stockport ingestion routine.")
    parser.add_argument("--from-date", type=date.fromisoformat, default=None)
    parser.add_argument("--to-date", type=date.fromisoformat, default=None)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--lookback-days", type=int, default=2)
    parser.add_argument("--league-id", dest="league_ids", action="append", type=int, default=None)
    parser.add_argument("--request-buffer", type=int, default=100)
    parser.add_argument("--skip-state", action="store_true")
    parser.add_argument(
        "--tm-enrichment",
        action="store_true",
        help="Force Transfermarkt enrichment regardless of day of week.",
    )
    parser.add_argument(
        "--skillcorner",
        action="store_true",
        help="Force SkillCorner enrichment regardless of day of week.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    results = run_daily_ingest(
        from_date=args.from_date,
        to_date=args.to_date,
        state_path=args.state_path,
        lookback_days=args.lookback_days,
        league_ids=args.league_ids,
        request_buffer=args.request_buffer,
        persist_state=not args.skip_state,
        run_tm_enrichment_flag=args.tm_enrichment,
        run_skillcorner_flag=args.skillcorner,
    )
    LOGGER.info("Daily ingest complete: %s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
