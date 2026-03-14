from __future__ import annotations

from datetime import UTC, datetime
import unittest

import pandas as pd

from features.opposition import _compute_opposition_splits_from_frames
from features.per90 import _compute_per90_frame
from features.rolling import compute_rolling


class FeatureEngineeringTests(unittest.TestCase):
    def test_compute_per90_frame_adds_low_minutes_flag_and_filters_zeroes(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "fixture_id": 1,
                    "player_id": 9,
                    "league_id": 41,
                    "season": "2025",
                    "date": datetime(2026, 1, 1),
                    "home_team": "Stockport County",
                    "away_team": "Blackpool",
                    "team": "Stockport County",
                    "is_home": True,
                    "minutes": 90,
                    "goals_scored": 1,
                    "shots_total": 3,
                    "passes_total": 30,
                },
                {
                    "fixture_id": 2,
                    "player_id": 9,
                    "league_id": 41,
                    "season": "2025",
                    "date": datetime(2026, 1, 8),
                    "home_team": "Bolton",
                    "away_team": "Stockport County",
                    "team": "Stockport County",
                    "is_home": False,
                    "minutes": 10,
                    "goals_scored": 1,
                    "shots_total": 1,
                    "passes_total": 4,
                },
                {
                    "fixture_id": 3,
                    "player_id": 9,
                    "league_id": 41,
                    "season": "2025",
                    "date": datetime(2026, 1, 15),
                    "home_team": "Stockport County",
                    "away_team": "Wigan",
                    "team": "Stockport County",
                    "is_home": True,
                    "minutes": 0,
                    "goals_scored": 0,
                    "shots_total": 0,
                    "passes_total": 0,
                },
            ]
        )

        result = _compute_per90_frame(frame)

        self.assertEqual(len(result.index), 2)
        self.assertFalse(bool(result.iloc[0]["low_minutes"]))
        self.assertTrue(bool(result.iloc[1]["low_minutes"]))
        self.assertAlmostEqual(float(result.iloc[0]["goals_scored_per90"]), 1.0)
        self.assertAlmostEqual(float(result.iloc[1]["goals_scored_per90"]), 9.0)

    def test_compute_rolling_summarises_recent_starts(self) -> None:
        per90 = pd.DataFrame(
            {
                "minutes": [90, 90, 90, 90],
                "goals_scored_per90": [0.5, 1.0, 1.5, 2.0],
                "shots_total_per90": [2.0, 2.5, 3.0, 3.5],
            }
        )

        result = compute_rolling(per90)

        self.assertAlmostEqual(result["goals_scored"]["roll_3"], (1.0 + 1.5 + 2.0) / 3)
        self.assertAlmostEqual(result["goals_scored"]["season_avg"], 1.25)
        self.assertGreater(result["goals_scored"]["trend_slope_10"], 0.0)
        self.assertAlmostEqual(result["shots_total"]["roll_5"], 2.75)

    def test_compute_opposition_splits_groups_by_standings_tier(self) -> None:
        match_frame = pd.DataFrame(
            [
                {
                    "fixture_id": 1,
                    "player_id": 9,
                    "league_id": 41,
                    "season": "2025",
                    "date": datetime(2026, 2, 1),
                    "home_team": "Stockport County",
                    "away_team": "Blackpool",
                    "team": "Stockport County",
                    "is_home": True,
                    "minutes": 90,
                    "goals_scored": 1,
                    "shots_total": 3,
                },
                {
                    "fixture_id": 2,
                    "player_id": 9,
                    "league_id": 41,
                    "season": "2025",
                    "date": datetime(2026, 2, 8),
                    "home_team": "Bolton",
                    "away_team": "Stockport County",
                    "team": "Stockport County",
                    "is_home": False,
                    "minutes": 90,
                    "goals_scored": 2,
                    "shots_total": 4,
                },
            ]
        )
        standings_frame = pd.DataFrame(
            [
                {"league_id": 41, "date": datetime(2026, 1, 31), "team_name": "Blackpool", "position": 3},
                {"league_id": 41, "date": datetime(2026, 2, 7), "team_name": "Bolton", "position": 14},
            ]
        )

        result = _compute_opposition_splits_from_frames(match_frame, standings_frame)

        self.assertAlmostEqual(result["goals_scored"]["tier1"], 1.0)
        self.assertAlmostEqual(result["goals_scored"]["tier3"], 2.0)
        self.assertAlmostEqual(result["shots_total"]["tier1"], 3.0)
        self.assertIsNone(result["goals_scored"]["tier2"])

    def test_compute_opposition_splits_handles_timezone_mismatches(self) -> None:
        match_frame = pd.DataFrame(
            [
                {
                    "fixture_id": 1,
                    "player_id": 9,
                    "league_id": 41,
                    "season": "2025",
                    "date": datetime(2026, 2, 1, 15, 0, tzinfo=UTC),
                    "home_team": "Stockport County",
                    "away_team": "Blackpool",
                    "team": "Stockport County",
                    "is_home": True,
                    "minutes": 90,
                    "goals_scored": 1,
                    "shots_total": 3,
                }
            ]
        )
        standings_frame = pd.DataFrame(
            [
                {"league_id": 41, "date": datetime(2026, 1, 31, 0, 0), "team_name": "Blackpool", "position": 3},
            ]
        )

        result = _compute_opposition_splits_from_frames(match_frame, standings_frame)

        self.assertAlmostEqual(result["goals_scored"]["tier1"], 1.0)


if __name__ == "__main__":
    unittest.main()
