from __future__ import annotations

from datetime import date, datetime
import unittest

import pandas as pd

from config import get_settings
from features.availability import _compute_availability_features_from_frames
from features.confidence import _compute_confidence_from_appearances, minutes_evidence_multiplier, shrink_low_sample_value
from features.gbe import _estimate_gbe_from_frames
from features.league_adjust import _compute_league_percentile_from_frames
from features.per90 import _compute_per90_frame
from features.role_classification import _build_position_group_feature_matrix, _classify_roles_from_frames
from features.trajectory import _compute_trajectory_features_from_frames


class ExpandedFeatureStackTests(unittest.TestCase):
    def test_compute_league_percentile_from_frames_returns_role_relative_output(self) -> None:
        match_frame = pd.DataFrame(
            [
                _match_row(1, 101, 41, "2025", 90, goals=1, assists=0, team="Stockport", position="F"),
                _match_row(2, 102, 41, "2025", 90, goals=0, assists=0, team="Blackpool", position="F"),
                _match_row(3, 103, 41, "2025", 90, goals=0, assists=0, team="Bolton", position="F"),
                _match_row(4, 104, 40, "2025", 90, goals=0.8, assists=0, team="Hull", position="F"),
            ]
        )
        role_frame = pd.DataFrame(
            [
                {"player_id": 1, "season": "2025", "primary_role": "complete_forward", "secondary_role": None, "cluster_confidence": 0.9},
                {"player_id": 2, "season": "2025", "primary_role": "complete_forward", "secondary_role": None, "cluster_confidence": 0.9},
                {"player_id": 3, "season": "2025", "primary_role": "complete_forward", "secondary_role": None, "cluster_confidence": 0.9},
                {"player_id": 4, "season": "2025", "primary_role": "complete_forward", "secondary_role": None, "cluster_confidence": 0.9},
            ]
        )

        result = _compute_league_percentile_from_frames(
            player_id=1,
            season="2025",
            role="complete_forward",
            match_frame=match_frame,
            role_frame=role_frame,
        )

        self.assertEqual(result["league_id"], 41)
        self.assertEqual(result["percentiles"]["goals_scored"], 100.0)
        self.assertAlmostEqual(result["league_adjusted_absolute"]["goals_scored"], 2.4)

    def test_compute_availability_features_from_frames_summarises_injury_history(self) -> None:
        match_frame = pd.DataFrame(
            [
                _match_row(9, 201, 41, "2024", 90, team="Stockport", position="M", is_substitute=False, played_on=datetime(2025, 4, 1)),
                _match_row(9, 202, 41, "2024", 75, team="Stockport", position="M", is_substitute=False, played_on=datetime(2025, 4, 8)),
                _match_row(9, 301, 41, "2025", 90, team="Stockport", position="M", is_substitute=False, played_on=datetime(2026, 1, 1)),
                _match_row(9, 302, 41, "2025", 70, team="Stockport", position="M", is_substitute=False, played_on=datetime(2026, 1, 8)),
                _match_row(9, 303, 41, "2025", 90, team="Stockport", position="M", is_substitute=False, played_on=datetime(2026, 1, 15)),
            ]
        )
        fixture_frame = pd.DataFrame(
            [
                {"fixture_id": 201, "season": "2024", "home_team": "Stockport", "away_team": "Blackpool"},
                {"fixture_id": 202, "season": "2024", "home_team": "Bolton", "away_team": "Stockport"},
                {"fixture_id": 301, "season": "2025", "home_team": "Stockport", "away_team": "Wigan"},
                {"fixture_id": 302, "season": "2025", "home_team": "Port Vale", "away_team": "Stockport"},
                {"fixture_id": 303, "season": "2025", "home_team": "Stockport", "away_team": "Rotherham"},
                {"fixture_id": 304, "season": "2025", "home_team": "Huddersfield", "away_team": "Stockport"},
            ]
        )
        sidelined_frame = pd.DataFrame(
            [
                {"player_id": 9, "type": "Hamstring", "start_date": datetime(2025, 6, 1), "end_date": datetime(2025, 6, 21)},
                {"player_id": 9, "type": "Hamstring", "start_date": datetime(2025, 9, 1), "end_date": datetime(2025, 9, 10)},
            ]
        )
        injury_frame = pd.DataFrame(
            [
                {"player_id": 9, "fixture_id": 302, "type": "Hamstring", "reason": "Tightness", "date": datetime(2025, 9, 1)},
            ]
        )
        event_frame = pd.DataFrame(
            [
                {"fixture_id": 302, "event_type": "subst", "event_detail": "Substitution 1", "time_elapsed": 65},
            ]
        )

        result = _compute_availability_features_from_frames(
            match_frame=match_frame,
            fixture_frame=fixture_frame,
            sidelined_frame=sidelined_frame,
            injury_frame=injury_frame,
            event_frame=event_frame,
            today=date(2026, 3, 12),
        )

        self.assertAlmostEqual(result["availability_rate_season"], 0.75)
        self.assertGreater(result["availability_rate_3yr"], 0.8)
        self.assertEqual(result["injury_frequency_3yr"], 2)
        self.assertEqual(result["muscle_injury_count"], 2)
        self.assertEqual(result["minutes_continuity"], 5)
        self.assertAlmostEqual(result["subbed_off_rate"], 0.2)

    def test_compute_availability_features_handles_empty_medical_frames(self) -> None:
        match_frame = pd.DataFrame(
            [
                _match_row(9, 301, 41, "2025", 90, team="Stockport", position="M", is_substitute=False, played_on=datetime(2026, 1, 1)),
                _match_row(9, 302, 41, "2025", 70, team="Stockport", position="M", is_substitute=False, played_on=datetime(2026, 1, 8)),
            ]
        )
        fixture_frame = pd.DataFrame(
            [
                {"fixture_id": 301, "season": "2025", "home_team": "Stockport", "away_team": "Wigan"},
                {"fixture_id": 302, "season": "2025", "home_team": "Port Vale", "away_team": "Stockport"},
            ]
        )

        result = _compute_availability_features_from_frames(
            match_frame=match_frame,
            fixture_frame=fixture_frame,
            sidelined_frame=pd.DataFrame(),
            injury_frame=pd.DataFrame(),
            event_frame=pd.DataFrame(),
            today=date(2026, 3, 12),
        )

        self.assertEqual(result["injury_frequency_3yr"], 0)
        self.assertIsNone(result["days_since_last_injury"])
        self.assertEqual(result["subbed_off_rate"], 0.0)

    def test_compute_trajectory_features_from_frames_summarises_career_shape(self) -> None:
        player_frame = {
            "birth_date": date(2002, 6, 1),
            "current_team": "Stockport",
        }
        transfer_frame = pd.DataFrame(
            [
                {"player_id": 9, "date": date(2023, 7, 1), "type": "transfer", "team_in": "Stockport", "team_out": "Lincoln"},
                {"player_id": 9, "date": date(2022, 7, 1), "type": "loan", "team_in": "Lincoln", "team_out": "Man City U21"},
            ]
        )
        match_frame = pd.DataFrame(
            [
                _match_row(9, 401, 41, "2024", 90, goals=0, assists=1, team="Stockport", position="CM", played_on=datetime(2025, 4, 1)),
                _match_row(9, 402, 41, "2024", 90, goals=0, assists=0, team="Stockport", position="CM", played_on=datetime(2025, 4, 8)),
                _match_row(9, 501, 41, "2025", 90, goals=1, assists=1, team="Stockport", position="CM", played_on=datetime(2026, 1, 1)),
                _match_row(9, 502, 41, "2025", 90, goals=1, assists=0, team="Stockport", position="CM", played_on=datetime(2026, 1, 8)),
            ]
        )

        result = _compute_trajectory_features_from_frames(
            player_frame=player_frame,
            transfer_frame=transfer_frame,
            match_frame=match_frame,
            leagues=get_settings().load_json("leagues.json"),
            age_curves=get_settings().load_json("age_curves.json"),
            today=date(2026, 3, 12),
        )

        self.assertEqual(result["loan_count"], 1)
        self.assertGreaterEqual(result["clubs_count"], 3)
        self.assertEqual(result["countries_played_in"], 1)
        self.assertEqual(result["seasons_at_current_club"], 2)
        self.assertGreater(result["age_at_first_senior_appearance"], 20)
        self.assertIn(result["age_curve_position"], {"pre_peak", "peak", "post_peak"})

    def test_confidence_helpers_apply_expected_thresholds(self) -> None:
        result = _compute_confidence_from_appearances(18)
        self.assertEqual(result["confidence_tier"], "Medium")
        self.assertAlmostEqual(result["shrinkage_factor"], 18 / 28)
        self.assertIsNone(result["total_minutes"])
        self.assertEqual(result["minutes_evidence_multiplier"], 1.0)
        self.assertAlmostEqual(
            shrink_low_sample_value(player_value=2.0, league_role_average=1.0, shrinkage_factor=0.5),
            1.5,
        )
        self.assertAlmostEqual(minutes_evidence_multiplier(400), (400 / 900) ** 0.5)
        self.assertEqual(minutes_evidence_multiplier(0), 0.35)

    def test_estimate_gbe_from_frames_flags_missing_inputs(self) -> None:
        non_uk = _estimate_gbe_from_frames(
            player_frame={"nationality": "Spanish", "current_team": "Stockport"},
            match_frame=pd.DataFrame(
                [
                    _match_row(11, 601, 40, "2025", 90, team="Hull", position="F", is_substitute=False),
                    _match_row(11, 602, 40, "2025", 90, team="Hull", position="F", is_substitute=False),
                    _match_row(11, 603, 40, "2025", 90, team="Hull", position="F", is_substitute=False),
                ]
            ),
            leagues=get_settings().load_json("leagues.json"),
            today=date(2026, 3, 12),
        )
        uk = _estimate_gbe_from_frames(
            player_frame={"nationality": "English", "current_team": "Stockport"},
            match_frame=pd.DataFrame(),
            leagues=get_settings().load_json("leagues.json"),
            today=date(2026, 3, 12),
        )

        self.assertEqual(uk["status"], "green")
        self.assertGreaterEqual(uk["points_estimate"], 15)
        self.assertIn(non_uk["status"], {"red", "amber", "green"})
        self.assertTrue(non_uk["gaps"])

    def test_classify_roles_from_frames_assigns_roles_for_position_group(self) -> None:
        templates = get_settings().load_json("role_templates.json")
        match_frame = pd.DataFrame(
            [
                _match_row(1, 701, 41, "2025", 90, team="Stockport", position="CB", passes=60, tackles=1, interceptions=1, duels_won=6, dribbles_past=0),
                _match_row(1, 702, 41, "2025", 90, team="Stockport", position="CB", passes=58, tackles=1, interceptions=2, duels_won=5, dribbles_past=0),
                _match_row(2, 703, 41, "2025", 90, team="Blackpool", position="CB", passes=25, tackles=5, interceptions=4, duels_won=7, dribbles_past=1),
                _match_row(2, 704, 41, "2025", 90, team="Blackpool", position="CB", passes=28, tackles=4, interceptions=5, duels_won=8, dribbles_past=1),
                _match_row(3, 705, 41, "2025", 90, team="Bolton", position="RB", passes=45, passes_key=3, tackles=3, interceptions=2, duels_won=5, dribbles_past=1),
                _match_row(3, 706, 41, "2025", 90, team="Bolton", position="RB", passes=47, passes_key=4, tackles=2, interceptions=2, duels_won=4, dribbles_past=1),
                _match_row(4, 707, 41, "2025", 90, team="Wigan", position="LB", passes=52, passes_key=4, tackles=2, interceptions=2, duels_won=4, dribbles_past=1),
                _match_row(4, 708, 41, "2025", 90, team="Wigan", position="LB", passes=50, passes_key=5, tackles=2, interceptions=3, duels_won=4, dribbles_past=1),
                _match_row(5, 709, 41, "2025", 90, team="Reading", position="CB", passes=35, tackles=3, interceptions=4, duels_won=6, dribbles_past=1),
                _match_row(5, 710, 41, "2025", 90, team="Reading", position="CB", passes=33, tackles=3, interceptions=4, duels_won=7, dribbles_past=1),
            ]
        )

        result = _classify_roles_from_frames(
            season="2025",
            position_group="D",
            match_frame=match_frame,
            templates=templates,
        )

        self.assertEqual(len(result.index), 5)
        self.assertTrue(result["primary_role"].notna().all())
        self.assertTrue(
            set(result["primary_role"]).issubset(
                {"ball_playing_cb", "aggressive_cb", "covering_cb", "attacking_fb", "inverted_fb", "defensive_fb"}
            )
        )

    def test_build_position_group_feature_matrix_adds_defender_context(self) -> None:
        match_frame = pd.DataFrame(
            [
                _match_row(1, 801, 41, "2025", 90, team="Stockport", position="D", passes=70, tackles=1, interceptions=2, duels_won=6, dribbles_past=0),
                _match_row(1, 802, 41, "2025", 90, team="Stockport", position="D", passes=68, tackles=1, interceptions=2, duels_won=5, dribbles_past=0),
                _match_row(2, 803, 41, "2025", 90, team="Blackpool", position="D", passes=30, tackles=5, interceptions=5, duels_won=8, dribbles_past=1),
                _match_row(2, 804, 41, "2025", 90, team="Blackpool", position="D", passes=28, tackles=4, interceptions=4, duels_won=8, dribbles_past=1),
                _match_row(3, 805, 41, "2025", 90, team="Bolton", position="D", passes=50, passes_key=3, tackles=3, interceptions=2, duels_won=5, dribbles_past=1),
                _match_row(3, 806, 41, "2025", 90, team="Bolton", position="D", passes=49, passes_key=4, tackles=2, interceptions=2, duels_won=4, dribbles_past=1),
                _match_row(4, 807, 41, "2025", 90, team="Wigan", position="D", passes=46, passes_key=4, tackles=2, interceptions=2, duels_won=4, dribbles_past=1),
                _match_row(4, 808, 41, "2025", 90, team="Wigan", position="D", passes=45, passes_key=5, tackles=2, interceptions=3, duels_won=4, dribbles_past=1),
                _match_row(5, 809, 41, "2025", 90, team="Reading", position="D", passes=36, tackles=3, interceptions=4, duels_won=6, dribbles_past=1),
                _match_row(5, 810, 41, "2025", 90, team="Reading", position="D", passes=34, tackles=3, interceptions=4, duels_won=7, dribbles_past=1),
            ]
        )
        lineup_rows = []
        defensive_layouts = {
            801: ("Stockport", [(90, "2:1"), (1, "2:2"), (91, "2:3"), (92, "2:4")]),
            802: ("Stockport", [(90, "2:1"), (1, "2:2"), (91, "2:3"), (92, "2:4")]),
            803: ("Blackpool", [(93, "2:1"), (94, "2:2"), (2, "2:3"), (95, "2:4")]),
            804: ("Blackpool", [(93, "2:1"), (94, "2:2"), (2, "2:3"), (95, "2:4")]),
            805: ("Bolton", [(3, "2:1"), (96, "2:2"), (97, "2:3"), (98, "2:4")]),
            806: ("Bolton", [(3, "2:1"), (96, "2:2"), (97, "2:3"), (98, "2:4")]),
            807: ("Wigan", [(99, "2:1"), (100, "2:2"), (101, "2:3"), (4, "2:4")]),
            808: ("Wigan", [(99, "2:1"), (100, "2:2"), (101, "2:3"), (4, "2:4")]),
            809: ("Reading", [(102, "2:1"), (5, "2:2"), (103, "2:3"), (104, "2:4")]),
            810: ("Reading", [(102, "2:1"), (5, "2:2"), (103, "2:3"), (104, "2:4")]),
        }
        for fixture_id, (team_name, slots) in defensive_layouts.items():
            for player_id, grid_position in slots:
                lineup_rows.append(
                    {
                        "fixture_id": fixture_id,
                        "player_id": player_id,
                        "team": team_name,
                        "is_starter": True,
                        "position_label": "D",
                        "grid_position": grid_position,
                        "season": "2025",
                    }
                )
        lineup_frame = pd.DataFrame(lineup_rows)
        player_frame = pd.DataFrame(
            [
                {"player_id": 1, "height_cm": 190},
                {"player_id": 2, "height_cm": 188},
                {"player_id": 3, "height_cm": 176},
                {"player_id": 4, "height_cm": 175},
                {"player_id": 5, "height_cm": 186},
            ]
        )
        per90 = _compute_per90_frame(match_frame)

        result = _build_position_group_feature_matrix(
            per90,
            "D",
            lineup_frame=lineup_frame,
            player_frame=player_frame,
            season="2025",
        )

        player_one = result.set_index("player_id").loc[1]
        player_three = result.set_index("player_id").loc[3]
        self.assertGreater(player_one["central_def_share"], player_three["central_def_share"])
        self.assertGreater(player_three["wide_def_share"], player_one["wide_def_share"])
        self.assertGreater(player_one["height_cm"], player_three["height_cm"])


def _match_row(
    player_id: int,
    fixture_id: int,
    league_id: int,
    season: str,
    minutes: float,
    *,
    goals: float = 0,
    assists: float = 0,
    shots: float = 0,
    team: str,
    position: str,
    is_substitute: bool = False,
    played_on: datetime | None = None,
    passes: float = 20,
    passes_key: float = 0,
    tackles: float = 0,
    interceptions: float = 0,
    duels_won: float = 0,
    duels_total: float | None = None,
    dribbles_past: float = 0,
) -> dict[str, object]:
    when = played_on or datetime(2026, 1, 1)
    return {
        "fixture_id": fixture_id,
        "player_id": player_id,
        "league_id": league_id,
        "season": season,
        "date": when,
        "home_team": team,
        "away_team": "Opponent",
        "team": team,
        "is_home": True,
        "referee": None,
        "minutes": minutes,
        "position": position,
        "rating": None,
        "is_substitute": is_substitute,
        "is_captain": False,
        "goals_scored": goals,
        "goals_conceded": 0,
        "assists": assists,
        "saves": 0,
        "shots_total": shots if shots else goals + assists + 1,
        "shots_on_target": goals if goals else 0,
        "passes_total": passes,
        "passes_key": passes_key,
        "pass_accuracy": 80.0,
        "tackles_total": tackles,
        "tackles_blocks": 0,
        "tackles_interceptions": interceptions,
        "duels_total": duels_total if duels_total is not None else duels_won + 2,
        "duels_won": duels_won,
        "dribbles_attempts": 2,
        "dribbles_success": 1,
        "dribbles_past": dribbles_past,
        "fouls_committed": 1,
        "fouls_drawn": 1,
        "yellow_cards": 0,
        "red_cards": 0,
        "pen_won": 0,
        "pen_committed": 0,
        "pen_scored": 0,
        "pen_missed": 0,
        "pen_saved": 0,
        "offsides": 0,
    }


if __name__ == "__main__":
    unittest.main()
