from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from viewer.service import _apply_league_strength_factor, _combine_dashboard_league_top_fives, _compute_upside_age_adjustment, _decorate_prediction_row, _load_on_pitch_profile_candidates, _player_age_years
from viewer.service import _role_names_for_family
from viewer.service import _load_wyscout_review_rows
from viewer.service import create_brief_from_form, get_brief_builder_context, get_brief_context, run_brief_longlist


class ViewerServiceTests(unittest.TestCase):
    def test_apply_league_strength_factor_scales_scores(self) -> None:
        self.assertEqual(_apply_league_strength_factor(50.0, 1.0), 50.0)
        self.assertEqual(_apply_league_strength_factor(50.0, 0.913), 45.65)
        self.assertIsNone(_apply_league_strength_factor(None, 0.913))

    def test_role_names_for_family_returns_related_profiles(self) -> None:
        role_names = _role_names_for_family("false_9")

        self.assertIn("false_9", role_names)
        self.assertIn("poacher", role_names)
        self.assertIn("target_forward", role_names)

    def test_combine_dashboard_league_top_fives_aligns_sections_by_league(self) -> None:
        combined = _combine_dashboard_league_top_fives(
            [
                {
                    "league_id": 40,
                    "league_name": "Championship",
                    "players": [{"player_name": "A. Morris"}],
                    "top_score": 54.2,
                }
            ],
            [
                {
                    "league_id": 40,
                    "league_name": "Championship",
                    "players": [{"player_name": "G. McEachran"}],
                    "top_score": 50.3,
                }
            ],
            [
                {
                    "league_id": 40,
                    "league_name": "Championship",
                    "players": [{"player_name": "J. Varane"}],
                    "top_score": 63.6,
                }
            ],
        )

        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0]["league_name"], "Championship")
        self.assertEqual(combined[0]["on_pitch"]["players"][0]["player_name"], "A. Morris")
        self.assertEqual(combined[0]["technical"]["players"][0]["player_name"], "G. McEachran")
        self.assertEqual(combined[0]["physical"]["players"][0]["player_name"], "J. Varane")

    def test_player_age_years_uses_current_age_fallback_when_birth_date_missing(self) -> None:
        self.assertEqual(_player_age_years(None, 28), 28.0)
        self.assertEqual(_player_age_years(None, "29"), 29.0)
        self.assertIsNone(_player_age_years(None, None))

    def test_upside_age_adjustment_rewards_younger_known_ages_and_penalises_unknown(self) -> None:
        young_score, young_multiplier = _compute_upside_age_adjustment(role_name="controller", age_years=23.0)
        peak_score, peak_multiplier = _compute_upside_age_adjustment(role_name="controller", age_years=28.0)
        old_score, old_multiplier = _compute_upside_age_adjustment(role_name="controller", age_years=34.0)
        unknown_score, unknown_multiplier = _compute_upside_age_adjustment(role_name="controller", age_years=None)

        self.assertGreaterEqual(young_score, peak_score)
        self.assertGreater(peak_score, old_score)
        self.assertEqual(young_multiplier, 1.0)
        self.assertEqual(old_multiplier, 1.0)
        self.assertLess(unknown_score, peak_score)
        self.assertLess(unknown_multiplier, 1.0)

    def test_load_wyscout_review_rows_skips_already_resolved_mappings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            review_path = Path(tmpdir) / "review_suggestions_test.csv"
            review_path.write_text(
                "\n".join(
                    [
                        "source_file,league_id,season,player_name,team_name,current_team,suggested_historical_team,suggested_team_score,suggested_player_id,suggested_player_name,suggested_player_score,global_candidate_1_id,global_candidate_1_name,global_candidate_1_team,global_candidate_1_score,global_candidate_2_id,global_candidate_2_name,global_candidate_2_team,global_candidate_2_score,global_candidate_3_id,global_candidate_3_name,global_candidate_3_team,global_candidate_3_score",
                        "Champ 22:23.xlsx,40,2022,K. Paal,Queens Park Rangers,QPR,Sunderland,0.34,,, ,36921,Kenneth Paal,QPR,0.90,,,,,,,,",
                        "Champ 22:23.xlsx,40,2022,L. Fiorini,Stockport County,Stockport County,Stockport County,0.99,18,Lewis Fiorini,0.99,,,,,,,,,,,,",
                    ]
                ),
                encoding="utf-8",
            )

            with patch(
                "viewer.service._load_resolved_wyscout_lookup_keys",
                return_value={"name:k paal|team:queens park rangers"},
            ):
                rows = _load_wyscout_review_rows(review_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["player_name"], "L. Fiorini")

    def test_load_on_pitch_profile_candidates_uses_configured_candidate_pool(self) -> None:
        class FakeExecuteResult:
            def mappings(self):
                return self

            def all(self):
                return [
                    {
                        "player_id": 21,
                        "primary_role": "poacher",
                        "secondary_role": None,
                        "player_name": "Striker One",
                        "current_team": "Test FC",
                        "current_league_id": 40,
                        "birth_date": None,
                        "current_age_years": 23,
                        "height_cm": 188,
                        "total_minutes": 920,
                    }
                ]

        class FakeSession:
            def execute(self, _query, _params=None):
                return FakeExecuteResult()

        class FakeSessionScope:
            def __enter__(self):
                return FakeSession()

            def __exit__(self, exc_type, exc, tb):
                return False

        with patch(
            "viewer.service.session_scope",
            return_value=FakeSessionScope(),
        ), patch(
            "viewer.service._league_catalog",
            return_value={40: {"name": "Championship", "recruitment_board": True}},
        ):
            result = _load_on_pitch_profile_candidates(
                profile={
                    "role_name": "point",
                    "candidate_roles": ["poacher"],
                    "allowed_sc_positions": ["ST", "CF"],
                },
                season="2025",
                minimum_minutes=180,
                candidate_limit=250,
            )

        self.assertEqual(result["match_mode"], "configured_pool")
        self.assertTrue(result["match_note"])
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["player_name"], "Striker One")

    def test_decorate_prediction_row_applies_league_strength_to_on_pitch_sections(self) -> None:
        row = _decorate_prediction_row(
            {
                "player_id": 9,
                "current_league_id": 41,
                "composite_score": 42.0,
                "role_fit_score": 60.0,
                "l1_performance_score": 50.0,
                "championship_projection_50th": 5.0,
                "availability_risk_prob": 0.2,
                "var_score": 0.01,
                "model_warnings": [],
                "component_fallbacks": {},
                "total_minutes": 900.0,
            },
            {41: {"name": "League One", "strength_factor": 0.913}},
            on_pitch_weights=[
                {"label": "Role Fit", "percent": 40.0},
                {"label": "Current", "percent": 35.0},
                {"label": "Projection", "percent": 25.0},
            ],
            present_on_pitch_weights=[
                {"label": "Role Fit", "percent": 55.0},
                {"label": "Current", "percent": 45.0},
            ],
            upside_on_pitch_weights=[
                {"label": "Role Fit", "percent": 30.0},
                {"label": "Projection", "percent": 70.0},
            ],
        )

        self.assertEqual(row["league_strength_factor"], 0.913)
        self.assertAlmostEqual(row["on_pitch_score_raw"], 57.48547717842324)
        self.assertAlmostEqual(row["on_pitch_score"], 52.48)
        self.assertAlmostEqual(row["present_on_pitch_score_raw"], 55.5)
        self.assertAlmostEqual(row["present_on_pitch_score"], 50.67)
        self.assertAlmostEqual(row["upside_on_pitch_score_raw"], 62.75933609958506)
        self.assertAlmostEqual(row["upside_on_pitch_score"], 57.3)

    def test_get_brief_builder_context_exposes_defaults_and_options(self) -> None:
        context = get_brief_builder_context()

        self.assertIn("controller", [role_name for role_name, _ in context["role_options"]])
        self.assertIn("promotion_accelerator", context["archetype_options"])
        self.assertTrue(context["form_values"]["pathway_check_done"])

    def test_create_brief_from_form_parses_browser_payload(self) -> None:
        form = {
            "role_name": ["controller"],
            "archetype_primary": ["promotion_accelerator"],
            "archetype_secondary": [""],
            "intent": ["first_team"],
            "budget_max_fee": ["5000000"],
            "budget_max_wage": ["30000"],
            "budget_max_contract_years": ["3"],
            "age_min": ["20"],
            "age_max": ["29"],
            "league_scope": ["40", "41"],
            "timeline": ["summer_2026"],
            "pathway_check_done": ["1"],
            "created_by": ["Ben Mills"],
            "approved_by": ["Ben Mills"],
            "action": ["create_run"],
        }

        with patch("viewer.service.pipeline_create_brief", return_value=11) as mocked_create:
            result = create_brief_from_form(form)

        self.assertEqual(result["brief_id"], 11)
        self.assertEqual(result["action"], "create_run")
        self.assertEqual(result["params"]["league_scope"], [40, 41])
        mocked_create.assert_called_once()

    def test_create_brief_from_form_allows_blank_budget_fields(self) -> None:
        form = {
            "role_name": ["controller"],
            "archetype_primary": ["promotion_accelerator"],
            "archetype_secondary": [""],
            "intent": ["first_team"],
            "budget_max_contract_years": ["3"],
            "age_min": ["20"],
            "age_max": ["29"],
            "league_scope": ["40", "41"],
            "timeline": ["summer_2026"],
            "pathway_check_done": ["1"],
            "created_by": ["Ben Mills"],
            "approved_by": ["Ben Mills"],
            "action": ["create"],
        }

        with patch("viewer.service.pipeline_create_brief", return_value=12) as mocked_create:
            result = create_brief_from_form(form)

        self.assertEqual(result["brief_id"], 12)
        self.assertIsNone(result["params"]["budget_max_fee"])
        self.assertIsNone(result["params"]["budget_max_wage"])
        mocked_create.assert_called_once()

    def test_run_brief_longlist_writes_report(self) -> None:
        frame = pd.DataFrame([{"player_id": 9, "composite_score": 77.0}])
        frame.attrs["skipped_players"] = [{"player_id": 10, "error": "x"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "longlist_brief_1.html"
            with patch("viewer.service.generate_longlist", return_value=frame), patch(
                "viewer.service.generate_longlist_report",
                return_value="<html>report</html>",
            ), patch(
                "viewer.service._brief_report_path",
                return_value=report_path,
            ):
                result = run_brief_longlist(1)
                report_body = report_path.read_text(encoding="utf-8")

        self.assertEqual(result["row_count"], 1)
        self.assertEqual(len(result["skipped_players"]), 1)
        self.assertEqual(report_body, "<html>report</html>")

    def test_get_brief_context_limits_predictions_to_current_longlist(self) -> None:
        executed_queries: list[str] = []

        class FakeExecuteResult:
            def __init__(self, payload):
                self._payload = payload

            def mappings(self):
                return self

            def one(self):
                return self._payload

            def all(self):
                return self._payload

        class FakeSession:
            def execute(self, query, params=None):
                executed_queries.append(str(query))
                if len(executed_queries) == 1:
                    return FakeExecuteResult(
                        {
                            "prediction_count": 99,
                            "longlist_count": 12,
                            "last_prediction_at": None,
                        }
                    )
                return FakeExecuteResult(
                    [
                        {
                            "player_id": 1,
                            "player_name": "Player One",
                            "current_team": "Test FC",
                            "current_league_id": 40,
                            "composite_score": 75.0,
                            "role_fit_score": 70.0,
                            "l1_performance_score": 68.0,
                            "championship_projection_50th": 6.2,
                            "availability_risk_prob": 0.15,
                            "var_score": 0.04,
                            "model_warnings": ["Projection: Heuristic fallback used."],
                            "component_fallbacks": {"projection": True},
                        }
                    ]
                )

        class FakeSessionScope:
            def __enter__(self):
                return FakeSession()

            def __exit__(self, exc_type, exc, tb):
                return False

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "viewer.service._load_brief_dict",
            return_value={"brief_id": 7, "season": "2025", "league_scope": [40]},
        ), patch(
            "viewer.service.session_scope",
            return_value=FakeSessionScope(),
        ), patch(
            "viewer.service._league_catalog",
            return_value={40: {"name": "Championship"}},
        ), patch(
            "viewer.service._brief_report_path",
            return_value=Path(tmpdir) / "longlist_brief_7.html",
        ):
            context = get_brief_context(7)

        self.assertIsNotNone(context)
        self.assertEqual(len(context["predictions"]), 1)
        self.assertAlmostEqual(context["predictions"][0]["board_score"], 98.07, places=2)
        self.assertGreater(context["predictions"][0]["projection_score"], 75.0)
        self.assertEqual(context["predictions"][0]["financial_score"], 52.0)
        self.assertEqual(context["predictions"][0]["availability_risk_pct"], 15.0)
        self.assertEqual(context["predictions"][0]["action_tier"]["action"], "Priority shortlist")
        self.assertEqual(context["warning_player_count"], 1)
        self.assertTrue(context["predictions"][0]["model_warnings"])
        self.assertTrue(context["composite_weights"])
        self.assertEqual([row["label"] for row in context["on_pitch_weights"]], ["Role Fit", "Current", "Projection"])
        self.assertAlmostEqual(sum(row["percent"] for row in context["on_pitch_weights"]), 100.0, places=1)
        self.assertEqual([row["label"] for row in context["present_on_pitch_weights"]], ["Role Fit", "Current"])
        self.assertAlmostEqual(sum(row["percent"] for row in context["present_on_pitch_weights"]), 100.0, places=1)
        self.assertEqual([row["label"] for row in context["upside_on_pitch_weights"]], ["Role Fit", "Projection"])
        self.assertAlmostEqual(sum(row["percent"] for row in context["upside_on_pitch_weights"]), 100.0, places=1)
        self.assertEqual(len(context["on_pitch_top_players"]), 1)
        self.assertEqual(context["on_pitch_top_players"][0]["player_name"], "Player One")
        self.assertGreater(context["on_pitch_top_players"][0]["on_pitch_score"], 0.0)
        self.assertEqual(len(context["present_on_pitch_top_players"]), 1)
        self.assertEqual(context["present_on_pitch_top_players"][0]["player_name"], "Player One")
        self.assertEqual(len(context["upside_on_pitch_top_players"]), 1)
        self.assertEqual(context["upside_on_pitch_top_players"][0]["player_name"], "Player One")
        self.assertEqual(len(context["on_pitch_league_top_fives"]), 1)
        self.assertEqual(context["on_pitch_league_top_fives"][0]["league_name"], "Championship")
        self.assertEqual(len(context["on_pitch_league_top_fives"][0]["players"]), 1)
        self.assertEqual(context["on_pitch_league_top_fives"][0]["players"][0]["player_name"], "Player One")
        self.assertTrue(context["action_tier_summary"])
        self.assertTrue(context["board_score_equation"])
        self.assertIn("current_longlist", executed_queries[1].lower())
        self.assertIn("stage = 'longlist'", executed_queries[1].lower())


if __name__ == "__main__":
    unittest.main()
