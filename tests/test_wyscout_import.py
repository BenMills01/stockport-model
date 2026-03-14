from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from ingestion.wyscout_import import (
    _canonical_team_name,
    _resolve_via_historical_roster,
    _score_roster_candidate,
    _score_team_name,
    _assign_file_seasons,
    _infer_league_id_from_folder,
    _infer_season_from_filename,
    import_wyscout_export,
    import_wyscout_league_folder,
)


class WyscoutImportTests(unittest.TestCase):
    def test_import_wyscout_export_validates_and_upserts_single_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "zone.csv"
            csv_path.write_text("player,team,metric_a,metric_b\nLouie Barry,Stockport County,10,hello\n", encoding="utf-8")

            with patch("ingestion.wyscout_import.save_source_player_mapping", return_value=1) as mocked_save, patch(
                "ingestion.wyscout_import.upsert_rows",
                return_value=1,
            ) as mocked_upsert:
                inserted = import_wyscout_export(str(csv_path), player_id=9, season="2025", zone="central")

        self.assertEqual(inserted, 1)
        mocked_save.assert_called_once()
        self.assertEqual(mocked_save.call_args.kwargs["matched_by"], "manual")
        args, _kwargs = mocked_upsert.call_args
        rows = args[1]
        self.assertEqual(rows[0]["player_id"], 9)
        self.assertEqual(rows[0]["season"], "2025")
        self.assertEqual(rows[0]["zone"], "central")
        self.assertEqual(rows[0]["metrics_json"]["metric_a"], 10)
        self.assertEqual(rows[0]["metrics_json"]["metric_b"], "hello")

    def test_import_wyscout_export_resolves_player_id_from_saved_or_fuzzy_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "zone.csv"
            csv_path.write_text("player,team,metric_a\nLewis Fiorini,Stockport County,10\n", encoding="utf-8")

            with patch(
                "ingestion.wyscout_import.resolve_source_player_id",
                return_value=18,
            ) as mocked_resolve, patch("ingestion.wyscout_import.upsert_rows", return_value=1) as mocked_upsert:
                inserted = import_wyscout_export(
                    str(csv_path),
                    player_id=None,
                    season="2025",
                    zone="left",
                    league_id=41,
                )

        self.assertEqual(inserted, 1)
        mocked_resolve.assert_called_once()
        self.assertEqual(mocked_resolve.call_args.args[:2], ("wyscout", "Lewis Fiorini"))
        self.assertEqual(mocked_resolve.call_args.kwargs["source_team_name"], "Stockport County")
        self.assertEqual(mocked_resolve.call_args.kwargs["league_id"], 41)
        args, _kwargs = mocked_upsert.call_args
        rows = args[1]
        self.assertEqual(rows[0]["player_id"], 18)

    def test_import_wyscout_export_rejects_empty_filter_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "zone.csv"
            csv_path.write_text("metric_a\n10\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                import_wyscout_export(str(csv_path), player_id=9, season="2025", zone=" ")

    def test_import_wyscout_export_requires_resolvable_identity_when_player_id_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "zone.csv"
            csv_path.write_text("metric_a\n10\n", encoding="utf-8")

            with patch("ingestion.wyscout_import.resolve_source_player_id", return_value=None):
                with self.assertRaises(ValueError):
                    import_wyscout_export(str(csv_path), player_id=None, season="2025", zone="central")

    def test_infer_league_id_from_folder_uses_aliases(self) -> None:
        folder = Path("/tmp/League One")

        self.assertEqual(_infer_league_id_from_folder(folder), 41)

    def test_infer_season_from_filename_parses_start_year(self) -> None:
        self.assertEqual(_infer_season_from_filename("Champ 25:26 pt 1.xlsx"), "2025")
        self.assertEqual(_infer_season_from_filename("2. Bundesliga 24-25.xlsx"), "2024")

    def test_assign_file_seasons_uses_team_overlap_when_filename_is_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "France Ligue 1"
            root.mkdir()
            file_a = root / "Search results (13).xlsx"
            file_b = root / "Search results (14).xlsx"
            pd.DataFrame({"Player": ["A"], "Team within selected timeframe": ["Auxerre"]}).to_excel(file_a, index=False)
            pd.DataFrame({"Player": ["B"], "Team within selected timeframe": ["Le Havre"]}).to_excel(file_b, index=False)

            with patch(
                "ingestion.wyscout_import._load_league_team_names_by_season",
                return_value={"2024": {"auxerre"}, "2025": {"le havre"}},
            ):
                assignments = _assign_file_seasons([file_a, file_b], 61)

        self.assertEqual(assignments[file_a], "2024")
        self.assertEqual(assignments[file_b], "2025")

    def test_import_wyscout_league_folder_imports_multi_file_workbooks_and_merges_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir) / "League One"
            folder.mkdir()
            file_one = folder / "Lge 1 25:26 pt1.xlsx"
            file_two = folder / "Lge 1 25:26 pt2.xlsx"
            pd.DataFrame(
                [
                    {
                        "Player": "C. Rushworth",
                        "Team": "Brighton",
                        "Team within selected timeframe": "Coventry City",
                        "Position": "GK",
                        "Matches played": 10,
                        "Minutes played": 900,
                        "xG": 0.0,
                    }
                ]
            ).to_excel(file_one, index=False)
            pd.DataFrame(
                [
                    {
                        "Player": "C. Rushworth",
                        "Team": "Brighton",
                        "Team within selected timeframe": "Coventry City",
                        "Position": "GK",
                        "Matches played": 12,
                        "Minutes played": 1080,
                        "xG": 0.0,
                    },
                    {
                        "Player": "L. Fiorini",
                        "Team": "Stockport County",
                        "Team within selected timeframe": "Stockport County",
                        "Position": "AMF",
                        "Matches played": 8,
                        "Minutes played": 620,
                        "xG": 1.2,
                    },
                ]
            ).to_excel(file_two, index=False)

            with patch(
                "ingestion.wyscout_import.resolve_source_player_id",
                side_effect=[101, 102],
            ), patch("ingestion.wyscout_import.upsert_rows", side_effect=lambda _model, rows, _conflict: len(rows)):
                summary = import_wyscout_league_folder(folder)

        self.assertEqual(summary["league_id"], 41)
        self.assertEqual(summary["distinct_rows"], 2)
        self.assertEqual(summary["duplicate_rows_merged"], 1)
        self.assertEqual(summary["imported_rows"], 2)

    def test_score_team_name_handles_shortened_api_labels(self) -> None:
        self.assertGreaterEqual(_score_team_name("West Bromwich Albion", "West Brom"), 0.9)
        self.assertGreaterEqual(_score_team_name("Olympique Marseille", "Marseille"), 0.9)
        self.assertEqual(_canonical_team_name("SK Beveren"), _canonical_team_name("Waasland-beveren"))
        self.assertEqual(_canonical_team_name("RFC Seraing"), _canonical_team_name("Seraing United"))

    def test_score_roster_candidate_handles_initials_and_full_names(self) -> None:
        self.assertGreaterEqual(_score_roster_candidate("P. Højbjerg", "Pierre-Emile Højbjerg"), 0.9)
        self.assertGreaterEqual(_score_roster_candidate("K. Grant", "Karlan Ahearne-Grant"), 0.9)
        self.assertGreaterEqual(_score_roster_candidate("J. Obambi Bapela", "Joaquin Obambi"), 0.89)

    def test_resolve_via_historical_roster_accepts_clear_match(self) -> None:
        with patch(
            "ingestion.wyscout_import._match_historical_team_name",
            return_value="West Brom",
        ), patch(
            "ingestion.wyscout_import._load_historical_roster",
            return_value=((19055, "Karlan Ahearne-Grant"), (17450, "Jorge Grant")),
        ):
            match = _resolve_via_historical_roster(
                "K. Grant",
                "West Bromwich Albion",
                season="2022",
                league_id=40,
                historical_team_cache={},
                historical_roster_cache={},
            )

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match["player_id"], 19055)
        self.assertGreaterEqual(match["score"], 0.9)


if __name__ == "__main__":
    unittest.main()
