from __future__ import annotations

from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ingestion.run_daily_ingest import load_last_run, run_daily_ingest, save_last_run


class DailyIngestTests(unittest.TestCase):
    def test_save_and_load_last_run_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            save_last_run(path, date(2026, 3, 12))
            self.assertEqual(load_last_run(path), date(2026, 3, 12))

    @patch("ingestion.run_daily_ingest.fetch_lineups", return_value=44)
    @patch("ingestion.run_daily_ingest.fetch_match_events", return_value=81)
    @patch("ingestion.run_daily_ingest.fetch_fixture_team_stats", return_value=12)
    @patch("ingestion.run_daily_ingest.fetch_match_performances", return_value=275)
    @patch("ingestion.run_daily_ingest.fetch_fixtures", return_value=6)
    @patch("ingestion.run_daily_ingest.fetch_api_usage", return_value={"requests_remaining": 1000})
    @patch("ingestion.run_daily_ingest.collect_fixture_details")
    @patch("ingestion.run_daily_ingest.collect_completed_fixtures")
    def test_run_daily_ingest_executes_steps_and_persists_state(
        self,
        mock_collect_fixtures: object,
        mock_collect_fixture_details: object,
        _mock_api_usage: object,
        mock_fixtures: object,
        mock_performances: object,
        mock_team_stats: object,
        mock_events: object,
        mock_lineups: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            fixture_rows = [{"fixture_id": 101}, {"fixture_id": 102}]
            fixture_details = {101: {"fixture": {"id": 101}}, 102: {"fixture": {"id": 102}}}
            mock_collect_fixtures.return_value = fixture_rows
            mock_collect_fixture_details.return_value = fixture_details
            results = run_daily_ingest(
                from_date=date(2026, 3, 10),
                to_date=date(2026, 3, 12),
                state_path=state_path,
            )

            self.assertEqual(results["fixture_count"], 6)
            self.assertEqual(results["player_count"], 275)
            self.assertEqual(results["errors"], [])
            self.assertEqual(load_last_run(state_path), date(2026, 3, 12))
            mock_fixtures.assert_called_once_with(
                date(2026, 3, 10),
                date(2026, 3, 12),
                league_ids=None,
                fixtures=fixture_rows,
            )
            mock_performances.assert_called_once_with(
                date(2026, 3, 10),
                date(2026, 3, 12),
                league_ids=None,
                fixtures=fixture_rows,
                fixture_details=fixture_details,
            )
            mock_team_stats.assert_called_once_with(
                date(2026, 3, 10),
                date(2026, 3, 12),
                league_ids=None,
                fixtures=fixture_rows,
                fixture_details=fixture_details,
            )
            mock_events.assert_called_once_with(
                date(2026, 3, 10),
                date(2026, 3, 12),
                league_ids=None,
                fixtures=fixture_rows,
                fixture_details=fixture_details,
            )
            mock_lineups.assert_called_once_with(
                date(2026, 3, 10),
                date(2026, 3, 12),
                league_ids=None,
                fixtures=fixture_rows,
                fixture_details=fixture_details,
            )

    @patch("ingestion.run_daily_ingest.fetch_lineups", return_value=44)
    @patch("ingestion.run_daily_ingest.fetch_match_events", side_effect=RuntimeError("boom"))
    @patch("ingestion.run_daily_ingest.fetch_fixture_team_stats", return_value=12)
    @patch("ingestion.run_daily_ingest.fetch_match_performances", return_value=275)
    @patch("ingestion.run_daily_ingest.fetch_fixtures", return_value=6)
    @patch("ingestion.run_daily_ingest.fetch_api_usage", return_value={"requests_remaining": 1000})
    @patch("ingestion.run_daily_ingest.collect_fixture_details")
    @patch("ingestion.run_daily_ingest.collect_completed_fixtures")
    @patch("ingestion.run_daily_ingest.LOGGER.exception")
    def test_run_daily_ingest_records_errors_without_advancing_state(
        self,
        _mock_logger_exception: object,
        mock_collect_fixtures: object,
        mock_collect_fixture_details: object,
        _mock_api_usage: object,
        mock_fixtures: object,
        _mock_performances: object,
        _mock_team_stats: object,
        _mock_events: object,
        _mock_lineups: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            fixture_rows = [{"fixture_id": 101}, {"fixture_id": 102}]
            mock_collect_fixtures.return_value = fixture_rows
            mock_collect_fixture_details.return_value = {101: {"fixture": {"id": 101}}}
            results = run_daily_ingest(
                from_date=date(2026, 3, 10),
                to_date=date(2026, 3, 12),
                state_path=state_path,
            )

            self.assertEqual(len(results["errors"]), 1)
            self.assertIsNone(load_last_run(state_path))
            mock_fixtures.assert_called_once_with(
                date(2026, 3, 10),
                date(2026, 3, 12),
                league_ids=None,
                fixtures=fixture_rows,
            )

    @patch("ingestion.run_daily_ingest.fetch_lineups", return_value=4)
    @patch("ingestion.run_daily_ingest.fetch_match_events", return_value=3)
    @patch("ingestion.run_daily_ingest.fetch_fixture_team_stats", return_value=2)
    @patch("ingestion.run_daily_ingest.fetch_match_performances", return_value=11)
    @patch("ingestion.run_daily_ingest.fetch_fixtures", return_value=1)
    @patch("ingestion.run_daily_ingest.fetch_api_usage", return_value={"requests_remaining": 1000})
    @patch("ingestion.run_daily_ingest.collect_fixture_details")
    @patch("ingestion.run_daily_ingest.collect_completed_fixtures")
    def test_run_daily_ingest_passes_league_filters_to_steps(
        self,
        mock_collect_fixtures: object,
        mock_collect_fixture_details: object,
        _mock_api_usage: object,
        mock_fixtures: object,
        mock_performances: object,
        mock_team_stats: object,
        mock_events: object,
        mock_lineups: object,
    ) -> None:
        fixture_rows = [{"fixture_id": 101}]
        fixture_details = {101: {"fixture": {"id": 101}}}
        mock_collect_fixtures.return_value = fixture_rows
        mock_collect_fixture_details.return_value = fixture_details
        results = run_daily_ingest(
            from_date=date(2026, 3, 10),
            to_date=date(2026, 3, 12),
            league_ids=[41],
            persist_state=False,
        )

        self.assertEqual(results["league_ids"], [41])
        mock_fixtures.assert_called_once_with(
            date(2026, 3, 10),
            date(2026, 3, 12),
            league_ids=[41],
            fixtures=fixture_rows,
        )
        mock_performances.assert_called_once_with(
            date(2026, 3, 10),
            date(2026, 3, 12),
            league_ids=[41],
            fixtures=fixture_rows,
            fixture_details=fixture_details,
        )
        mock_team_stats.assert_called_once_with(
            date(2026, 3, 10),
            date(2026, 3, 12),
            league_ids=[41],
            fixtures=fixture_rows,
            fixture_details=fixture_details,
        )
        mock_events.assert_called_once_with(
            date(2026, 3, 10),
            date(2026, 3, 12),
            league_ids=[41],
            fixtures=fixture_rows,
            fixture_details=fixture_details,
        )
        mock_lineups.assert_called_once_with(
            date(2026, 3, 10),
            date(2026, 3, 12),
            league_ids=[41],
            fixtures=fixture_rows,
            fixture_details=fixture_details,
        )

    @patch("ingestion.run_daily_ingest.fetch_lineups", return_value=4)
    @patch("ingestion.run_daily_ingest.fetch_match_events", return_value=3)
    @patch("ingestion.run_daily_ingest.fetch_fixture_team_stats", return_value=2)
    @patch("ingestion.run_daily_ingest.fetch_match_performances", return_value=11)
    @patch("ingestion.run_daily_ingest.fetch_fixtures", return_value=1)
    @patch("ingestion.run_daily_ingest.fetch_api_usage", return_value={"requests_remaining": 100})
    @patch("ingestion.run_daily_ingest.collect_fixture_details")
    @patch("ingestion.run_daily_ingest.collect_completed_fixtures")
    def test_run_daily_ingest_stops_when_budget_buffer_would_be_breached(
        self,
        mock_collect_fixtures: object,
        _mock_collect_fixture_details: object,
        _mock_api_usage: object,
        mock_fixtures: object,
        mock_performances: object,
        mock_team_stats: object,
        mock_events: object,
        mock_lineups: object,
    ) -> None:
        fixture_rows = [{"fixture_id": idx} for idx in range(101, 111)]
        mock_collect_fixtures.return_value = fixture_rows

        results = run_daily_ingest(
            from_date=date(2026, 3, 10),
            to_date=date(2026, 3, 12),
            persist_state=False,
            request_buffer=100,
        )

        self.assertEqual(results["steps"]["fixtures"], 1)
        self.assertEqual(results["steps"]["match_performances"], 0)
        self.assertEqual(results["errors"][0]["step"], "api_budget")
        mock_fixtures.assert_called_once()
        mock_performances.assert_not_called()
        mock_team_stats.assert_not_called()
        mock_events.assert_not_called()
        mock_lineups.assert_not_called()


if __name__ == "__main__":
    unittest.main()
