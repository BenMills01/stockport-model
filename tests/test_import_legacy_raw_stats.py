from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from db.schema import Fixture, MatchPerformance, Player
from ingestion.import_legacy_raw_stats import import_legacy_raw_player_stats


class LegacyRawStatsImportTests(unittest.TestCase):
    def test_import_legacy_raw_player_stats_maps_csv_into_stockport_tables(self) -> None:
        csv_text = """league_id,league_name,country,fixture_id,season,date,home_team,away_team,referee,player_id,player_name,team,is_home,position,minutes,rating,fouls_committed,fouls_drawn,shots_total,shots_on,tackles,saves,offsides,yellow_cards,red_cards,passes,key_passes,duels_total,duels_won,dribbles
41,League One,England,1001,2025,2026-03-10,Stockport County,Blackpool,J. Brooks,9,Isaac Olaofe,Stockport County,True,F,90,7.5,1,3,4,2,1,0,2,0,0,18,2,9,5,2
41,League One,England,1001,2025,2026-03-10,Stockport County,Blackpool,J. Brooks,18,Callum Camps,Stockport County,True,M,88,7.1,0,1,1,1,2,0,0,1,0,42,3,10,6,1
39,Premier League,England,2001,2025,2026-03-12,Arsenal,Chelsea,M. Oliver,20,Cole Palmer,Chelsea,False,M,90,8.2,1,2,5,2,0,0,0,0,0,44,4,7,4,3
41,League One,England,1002,2025,2026-03-11,Reading,Barnsley,S. Hooper,9,Isaac Olaofe,Reading,False,F,85,7.0,1,1,3,1,0,0,1,0,0,14,1,6,3,1
"""

        calls: list[tuple[type[object], list[dict[str, object]], list[str]]] = []

        def fake_upsert_rows(
            model: type[object],
            rows: list[dict[str, object]],
            conflict_columns: list[str],
        ) -> int:
            calls.append((model, rows, conflict_columns))
            return len(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "raw_player_stats.csv"
            csv_path.write_text(csv_text, encoding="utf-8")

            with patch("ingestion.import_legacy_raw_stats.upsert_rows", side_effect=fake_upsert_rows):
                summary = import_legacy_raw_player_stats(
                    csv_path=csv_path,
                    chunk_size=2,
                    batch_size=2,
                )

        self.assertEqual(summary["rows_read"], 4)
        self.assertEqual(summary["rows_after_filters"], 4)
        self.assertEqual(summary["match_performance_rows_upserted"], 4)
        self.assertEqual(summary["fixture_rows_upserted"], 3)
        self.assertEqual(summary["player_rows_upserted"], 3)

        match_calls = [rows for model, rows, keys in calls if model is MatchPerformance]
        fixture_calls = [rows for model, rows, keys in calls if model is Fixture]
        player_calls = [rows for model, rows, keys in calls if model is Player]

        self.assertEqual(len(match_calls), 2)
        self.assertEqual(sum(len(rows) for rows in match_calls), 4)
        self.assertEqual(sum(len(rows) for rows in fixture_calls), 3)
        self.assertEqual(sum(len(rows) for rows in player_calls), 3)

        first_match_row = match_calls[0][0]
        self.assertEqual(first_match_row["shots_on_target"], 2)
        self.assertEqual(first_match_row["passes_total"], 18)
        self.assertEqual(first_match_row["dribbles_success"], 2)
        self.assertFalse(first_match_row["is_substitute"])

        latest_player_row = next(
            row
            for rows in player_calls
            for row in rows
            if row["player_id"] == 9
        )
        self.assertEqual(latest_player_row["current_team"], "Reading")
        self.assertEqual(latest_player_row["current_league_id"], 41)

        fixture_row = next(
            row
            for rows in fixture_calls
            for row in rows
            if row["fixture_id"] == 1001
        )
        self.assertEqual(
            fixture_row["date"],
            datetime(2026, 3, 10, 0, 0, tzinfo=UTC),
        )

    def test_import_legacy_raw_player_stats_can_restrict_to_tracked_leagues(self) -> None:
        csv_text = """league_id,league_name,country,fixture_id,season,date,home_team,away_team,referee,player_id,player_name,team,is_home,position,minutes,rating,fouls_committed,fouls_drawn,shots_total,shots_on,tackles,saves,offsides,yellow_cards,red_cards,passes,key_passes,duels_total,duels_won,dribbles
41,League One,England,1001,2025,2026-03-10,Stockport County,Blackpool,J. Brooks,9,Isaac Olaofe,Stockport County,True,F,90,7.5,1,3,4,2,1,0,2,0,0,18,2,9,5,2
2,UEFA Champions League,Europe,3001,2025,2026-03-10,Barcelona,PSG,F. Letexier,7,Ousmane Dembele,PSG,False,F,90,8.0,0,1,5,3,0,0,1,0,0,30,4,8,6,4
"""
        match_calls: list[list[dict[str, object]]] = []

        def fake_upsert_rows(
            model: type[object],
            rows: list[dict[str, object]],
            conflict_columns: list[str],
        ) -> int:
            if model is MatchPerformance:
                match_calls.append(rows)
            return len(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "raw_player_stats.csv"
            csv_path.write_text(csv_text, encoding="utf-8")

            with patch(
                "ingestion.import_legacy_raw_stats.get_settings",
                return_value=type(
                    "FakeSettings",
                    (),
                    {"load_json": staticmethod(lambda _filename: [{"league_id": 41}])},
                )(),
            ), patch("ingestion.import_legacy_raw_stats.upsert_rows", side_effect=fake_upsert_rows):
                summary = import_legacy_raw_player_stats(
                    csv_path=csv_path,
                    tracked_only=True,
                )

        self.assertEqual(summary["rows_after_filters"], 1)
        self.assertEqual(summary["match_performance_rows_upserted"], 1)
        self.assertEqual(len(match_calls), 1)
        self.assertEqual(match_calls[0][0]["league_id"], 41)


if __name__ == "__main__":
    unittest.main()
