from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from models.validation import _calibration_check_from_frames
from models.validation import _compute_outcome_metrics_from_frame, _post_window_audit_from_frames
from models.validation import _temporal_backtest_from_frames
from outputs.comparison import generate_comparison
from outputs.longlist import _top_concerns, generate_longlist_report
from outputs.recommendation import generate_recommendation_pack
from outputs.shortlist_card import generate_shortlist_card


class OutputAndValidationTests(unittest.TestCase):
    def test_generate_longlist_report_renders_html(self) -> None:
        context = {
            "brief": SimpleNamespace(role_name="controller", archetype_primary="promotion_accelerator", archetype_secondary=None, pathway_player_id=None),
            "generated_at": datetime(2026, 3, 12, 12, 0),
            "board_score_equation": "Board Score = 100 x (1 - exp(-Composite / 19)), capped between 0 and 100.",
            "action_tier_summary": [
                {"label": "Tier 1", "min_score": 40.0, "board_min_score": 83.7, "action": "Priority shortlist", "count": 1},
            ],
            "players": [
                {
                    "player_name": "Lewis Fiorini",
                    "team": "Stockport County",
                    "composite_score": 78.5,
                    "board_score": 100.0,
                    "action_tier": {"label": "Tier 1", "action": "Priority shortlist"},
                    "projection_band": {"p10": 35, "p50": 52, "p90": 68},
                    "model_warnings": [],
                    "top_strengths": ["Strong role fit"],
                    "top_concerns": ["Elevated availability risk"],
                }
            ],
        }
        with patch("outputs.longlist._build_longlist_context", return_value=context):
            html = generate_longlist_report(1)

        self.assertIn("Lewis Fiorini", html)
        self.assertIn("Longlist", html)
        self.assertIn("78.5", html)
        self.assertIn("Priority shortlist", html)

    def test_generate_shortlist_and_comparison_reports_render(self) -> None:
        shortlist_context = {
            "brief": SimpleNamespace(role_name="controller"),
            "player": SimpleNamespace(player_name="Lewis Fiorini"),
            "prediction": SimpleNamespace(role_fit_score=75, l1_performance_score=70, availability_risk_prob=0.2, var_score=1.1, composite_score=77),
            "generated_at": datetime(2026, 3, 12, 12, 0),
            "scout_notes": [SimpleNamespace(scout_name="Scout A", notes_text="Fits our system")],
            "projection_band": {"p10": 30, "p50": 48, "p90": 62},
        }
        comparison_context = {
            "brief": SimpleNamespace(role_name="controller"),
            "generated_at": datetime(2026, 3, 12, 12, 0),
            "players": [
                {
                    "player_name": "Lewis Fiorini",
                    "team": "Stockport County",
                    "role_fit": 75,
                    "current_performance": 70,
                    "projection_band": {"p10": 30, "p50": 48, "p90": 62},
                    "availability": 0.2,
                    "financial": 1.1,
                    "composite": 77,
                }
            ],
        }
        with patch("outputs.shortlist_card._build_shortlist_context", return_value=shortlist_context):
            shortlist_html = generate_shortlist_card(1, 9)
        with patch("outputs.comparison._build_comparison_context", return_value=comparison_context):
            comparison_html = generate_comparison(1, [9])

        self.assertIn("Shortlist Card", shortlist_html)
        self.assertIn("Lewis Fiorini", shortlist_html)
        self.assertIn("Comparison", comparison_html)
        self.assertIn("controller", comparison_html)

    def test_generate_recommendation_pack_renders(self) -> None:
        context = {
            "brief": SimpleNamespace(role_name="controller", archetype_primary="promotion_accelerator"),
            "player": SimpleNamespace(player_name="Lewis Fiorini"),
            "prediction": SimpleNamespace(
                composite_score=80,
                role_fit_score=78,
                championship_projection_10th=35,
                championship_projection_50th=50,
                championship_projection_90th=67,
            ),
            "alternatives": [
                {
                    "player": SimpleNamespace(player_name="Player B"),
                    "prediction": SimpleNamespace(composite_score=74, championship_projection_50th=45),
                }
            ],
            "scout_notes": [SimpleNamespace(scout_name="Scout A", notes_text="Good profile")],
            "overrides": [],
            "generated_at": datetime(2026, 3, 12, 12, 0),
        }
        with patch("outputs.recommendation._build_recommendation_context", return_value=context):
            html = generate_recommendation_pack(1, 9, [10])

        self.assertIn("Recommendation", html)
        self.assertIn("Player B", html)
        self.assertIn("Lewis Fiorini", html)

    def test_top_concerns_uses_projection_score_scale(self) -> None:
        prediction = SimpleNamespace(
            availability_risk_prob=0.1,
            championship_projection_50th=6.2,
        )
        concerns = _top_concerns(prediction)

        self.assertNotIn("Projection below Championship threshold", concerns)

        low_projection = SimpleNamespace(
            availability_risk_prob=0.1,
            championship_projection_50th=3.0,
        )
        low_concerns = _top_concerns(low_projection)

        self.assertIn("Projection below Championship threshold", low_concerns)

    def test_temporal_backtest_and_calibration_helpers(self) -> None:
        prediction_frame = pd.DataFrame(
            [
                {"player_id": 1, "brief_id": 1, "prediction_date": "2026-01-01", "composite_score": 0.9, "availability_risk_prob": 0.1},
                {"player_id": 2, "brief_id": 1, "prediction_date": "2026-01-02", "composite_score": 0.2, "availability_risk_prob": 0.8},
                {"player_id": 3, "brief_id": 1, "prediction_date": "2026-01-03", "composite_score": 0.8, "availability_risk_prob": 0.3},
                {"player_id": 4, "brief_id": 1, "prediction_date": "2026-01-04", "composite_score": 0.1, "availability_risk_prob": 0.7},
            ]
        )
        outcome_frame = pd.DataFrame(
            [
                {"player_id": 1, "brief_id": 1, "performance_hit": True, "financial_hit": True, "availability_hit": True, "signed_date": "2026-01-20", "failure_type": None},
                {"player_id": 2, "brief_id": 1, "performance_hit": False, "financial_hit": False, "availability_hit": False, "signed_date": "2026-01-20", "failure_type": "performance"},
                {"player_id": 3, "brief_id": 1, "performance_hit": True, "financial_hit": True, "availability_hit": True, "signed_date": "2026-01-20", "failure_type": None},
                {"player_id": 4, "brief_id": 1, "performance_hit": False, "financial_hit": False, "availability_hit": False, "signed_date": "2026-01-20", "failure_type": "availability"},
            ]
        )

        backtest = _temporal_backtest_from_frames(
            prediction_frame=prediction_frame,
            outcome_frame=outcome_frame,
            score_column="composite_score",
            window_dates=["2026-01-10"],
        )
        calibration = _calibration_check_from_frames(
            prediction_frame=prediction_frame,
            outcome_frame=outcome_frame,
        )

        self.assertEqual(len(backtest["windows"]), 1)
        self.assertIsNotNone(backtest["windows"][0]["auc"])
        self.assertIn("drift_alert", calibration)
        self.assertTrue(calibration["calibration_curve"])

    def test_outcome_metrics_and_post_window_audit_helpers(self) -> None:
        outcome_frame = pd.DataFrame(
            [
                {"player_id": 1, "brief_id": 1, "performance_hit": True, "financial_hit": True, "availability_hit": True, "signed_date": "2026-01-10", "failure_type": None},
                {"player_id": 2, "brief_id": 1, "performance_hit": False, "financial_hit": False, "availability_hit": True, "signed_date": "2026-01-11", "failure_type": "missed opportunity"},
            ]
        )
        briefs = pd.DataFrame([{"brief_id": 1, "created_date": "2026-01-01"}])
        pipeline = pd.DataFrame(
            [
                {"brief_id": 1, "player_id": 1, "stage": "longlist", "added_date": "2026-01-02"},
                {"brief_id": 1, "player_id": 1, "stage": "shortlist", "added_date": "2026-01-05"},
            ]
        )
        overrides = pd.DataFrame([{"brief_id": 1, "player_id": 1, "override_date": "2026-01-06", "outcome": "pending"}])

        metrics = _compute_outcome_metrics_from_frame(outcome_frame, window="2026-01")
        audit = _post_window_audit_from_frames(
            briefs=briefs,
            pipeline=pipeline,
            overrides=overrides,
            outcomes=outcome_frame,
            window="2026-01",
        )

        self.assertEqual(metrics["total_signed"], 2)
        self.assertEqual(audit["briefs_created"], 1)
        self.assertEqual(audit["overrides_logged"], 1)
        self.assertIn("longlist", audit["players_progressed_by_stage"])


if __name__ == "__main__":
    unittest.main()
