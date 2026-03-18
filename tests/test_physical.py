from __future__ import annotations

import unittest
from unittest.mock import patch

from scoring.physical import score_physical


class PhysicalScoreTests(unittest.TestCase):
    def test_score_physical_supports_profile_specific_weighting(self) -> None:
        feature_map = {
            1: {
                "sc_has_physical": True,
                "sc_physical_top_speed_per_match": 36.0,
                "sc_physical_sprint_dist_per_match": 520.0,
                "sc_physical_dist_per_match": 9300.0,
                "sc_physical_count_hsr_per_match": 48.0,
                "sc_pressure_ball_retention_ratio_under_high_pressure": 0.58,
            },
            2: {
                "sc_has_physical": True,
                "sc_physical_top_speed_per_match": 33.0,
                "sc_physical_sprint_dist_per_match": 360.0,
                "sc_physical_dist_per_match": 10200.0,
                "sc_physical_count_hsr_per_match": 56.0,
                "sc_pressure_ball_retention_ratio_under_high_pressure": 0.60,
            },
            3: {
                "sc_has_physical": True,
                "sc_physical_top_speed_per_match": 31.0,
                "sc_physical_sprint_dist_per_match": 240.0,
                "sc_physical_dist_per_match": 12100.0,
                "sc_physical_count_hsr_per_match": 69.0,
                "sc_pressure_ball_retention_ratio_under_high_pressure": 0.62,
            },
        }

        with patch(
            "scoring.physical.compute_skillcorner_features",
            side_effect=lambda player_id: feature_map[player_id],
        ):
            winger_score = score_physical(
                1,
                [2, 3],
                physical_weights={
                    "sc_physical_top_speed_per_match": 0.5,
                    "sc_physical_sprint_dist_per_match": 0.5,
                },
                gi_weights={"sc_pressure_ball_retention_ratio_under_high_pressure": 1.0},
                physical_sub_weight=1.0,
                gi_sub_weight=0.0,
            )
            engine_score = score_physical(
                1,
                [2, 3],
                physical_weights={
                    "sc_physical_dist_per_match": 0.5,
                    "sc_physical_count_hsr_per_match": 0.5,
                },
                gi_weights={"sc_pressure_ball_retention_ratio_under_high_pressure": 1.0},
                physical_sub_weight=1.0,
                gi_sub_weight=0.0,
            )

        self.assertIsNotNone(winger_score)
        self.assertIsNotNone(engine_score)
        self.assertGreater(float(winger_score), float(engine_score))


if __name__ == "__main__":
    unittest.main()
