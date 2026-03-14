from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from db.schema import Pipeline as PipelineRow
from models.availability_risk import predict_availability_risk
from governance.pipeline import _normalise_timeline, _resolve_brief_season, _upsert_pipeline_rows, create_brief, generate_longlist, log_override, promote_to_shortlist
from models.availability_risk import _risk_tier, train_availability_model
from models.championship_projection import _align_projection_feature_frame, _build_projection_feature_frame, _estimate_adaptation_months, train_projection_model
from models.championship_projection import project_to_championship
from models.financial_value import _build_value_prediction_frame, _find_comparable_transactions, _value_adjusted_return_score, _wage_fit, train_value_model
from models.financial_value import estimate_value
from models.l1_performance import _consistency_to_score, _trend_label, _trend_to_score, score_l1_performance
from models.proxy_xg import _build_proxy_shot_frame_from_frames, train_proxy_xg
from models.role_fit import score_role_fit
from models.similarity import _vector_from_percentiles
from scoring.composite import _blended_weights, _financial_score, _projection_score_from_bundle, compute_composite, effective_layer_weights


class ModelStackTests(unittest.TestCase):
    def test_train_proxy_xg_saves_model_and_builds_proxy_shots(self) -> None:
        training_frame = pd.DataFrame(
            [
                {"is_header": 0, "angle_to_goal": 0.4, "distance_to_goal": 12.0, "is_penalty": 0, "is_direct_free_kick": 0, "game_state": "drawing", "goal": 1},
                {"is_header": 1, "angle_to_goal": 0.3, "distance_to_goal": 8.0, "is_penalty": 0, "is_direct_free_kick": 0, "game_state": "winning", "goal": 1},
                {"is_header": 0, "angle_to_goal": 0.1, "distance_to_goal": 25.0, "is_penalty": 0, "is_direct_free_kick": 1, "game_state": "losing", "goal": 0},
                {"is_header": 0, "angle_to_goal": 0.2, "distance_to_goal": 18.0, "is_penalty": 0, "is_direct_free_kick": 0, "game_state": "drawing", "goal": 0},
            ]
        )
        match_frame = pd.DataFrame([{"shots_total": 3, "shots_on_target": 1}])
        event_frame = pd.DataFrame([{"event_detail": "Normal Goal", "comments": ""}])

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "shots.csv"
            training_frame.to_csv(csv_path, index=False)
            with patch("models.proxy_xg.get_settings", return_value=SimpleNamespace(project_root=Path(tmpdir))):
                model = train_proxy_xg(str(csv_path))

            self.assertTrue((Path(tmpdir) / "data" / "proxy_xg_model.joblib").exists())
            self.assertIsNotNone(model)

        proxy_shots = _build_proxy_shot_frame_from_frames(match_frame, event_frame)
        self.assertGreaterEqual(len(proxy_shots.index), 3)

    def test_score_role_fit_combines_weights_and_percentiles(self) -> None:
        template = SimpleNamespace(
            is_active=True,
            role_name="controller",
            version="2026-Q1",
            metrics_json={"passes_total": 0.6, "passes_key": 0.4},
        )

        @contextmanager
        def fake_session_scope() -> object:
            yield SimpleNamespace(get=lambda _model, _id: template)

        with patch("models.role_fit.session_scope", fake_session_scope), patch(
            "models.role_fit.compute_league_percentile",
            return_value={"percentiles": {"passes_total": 80.0, "passes_key": 50.0}},
        ), patch(
            "models.role_fit.compute_confidence",
            return_value={"confidence_tier": "High"},
        ):
            result = score_role_fit(player_id=9, template_id=1, season="2025")

        self.assertAlmostEqual(result["score"], 68.0)
        self.assertAlmostEqual(result["raw_score"], 68.0)
        self.assertEqual(result["confidence_tier"], "High")

    def test_score_role_fit_shrinks_low_sample_scores_toward_neutral(self) -> None:
        template = SimpleNamespace(
            is_active=True,
            role_name="controller",
            version="2026-Q1",
            metrics_json={"passes_total": 0.6, "passes_key": 0.4},
        )

        @contextmanager
        def fake_session_scope() -> object:
            yield SimpleNamespace(get=lambda _model, _id: template)

        with patch("models.role_fit.session_scope", fake_session_scope), patch(
            "models.role_fit.compute_league_percentile",
            return_value={"percentiles": {"passes_total": 80.0, "passes_key": 50.0}},
        ), patch(
            "models.role_fit.compute_confidence",
            return_value={"confidence_tier": "Low", "shrinkage_factor": 0.4},
        ):
            result = score_role_fit(player_id=9, template_id=1, season="2025")

        self.assertAlmostEqual(result["raw_score"], 68.0)
        self.assertAlmostEqual(result["score"], 57.2)

    def test_score_l1_performance_shrinks_low_sample_scores_toward_neutral(self) -> None:
        template = SimpleNamespace(metrics_json={"passes_total": 0.5, "passes_key": 0.5})

        with patch("models.l1_performance.get_active_template_for_role", return_value=template), patch(
            "models.l1_performance.compute_league_percentile",
            return_value={"percentiles": {"passes_total": 80.0, "passes_key": 70.0}},
        ), patch(
            "models.l1_performance.compute_opposition_splits",
            return_value={},
        ), patch(
            "models.l1_performance.load_player_match_frame",
            return_value=pd.DataFrame(),
        ), patch(
            "models.l1_performance._compute_per90_frame",
            return_value=pd.DataFrame(),
        ), patch(
            "models.l1_performance.compute_rolling",
            return_value={"passes_total": {"roll_10_cv": None, "trend_slope_10": None}},
        ), patch(
            "models.l1_performance.compute_confidence",
            return_value={"shrinkage_factor": 0.4},
        ):
            result = score_l1_performance(player_id=9, season="2025", role="controller")

        self.assertAlmostEqual(result["raw_score"], 62.5)
        self.assertAlmostEqual(result["score"], 55.0)

    def test_projection_training_and_helpers_work_on_synthetic_data(self) -> None:
        training_data = pd.DataFrame(
            [
                {
                    "origin_league_id": 41,
                    "destination_league_id": 40,
                    "league_pair": "41->40",
                    "age_at_transfer": 23,
                    "primary_role": "complete_forward",
                    "origin_team_league_position": 3,
                    "destination_team_league_position": 15,
                    "origin_goals_scored_per90": 0.7,
                    "origin_shots_total_per90": 3.1,
                    "target_goals_scored_per90": 0.32,
                    "target_shots_total_per90": 2.4,
                    "target_starter": 1,
                },
                {
                    "origin_league_id": 41,
                    "destination_league_id": 40,
                    "league_pair": "41->40",
                    "age_at_transfer": 25,
                    "primary_role": "complete_forward",
                    "origin_team_league_position": 8,
                    "destination_team_league_position": 18,
                    "origin_goals_scored_per90": 0.5,
                    "origin_shots_total_per90": 2.7,
                    "target_goals_scored_per90": 0.25,
                    "target_shots_total_per90": 2.0,
                    "target_starter": 0,
                },
                {
                    "origin_league_id": 41,
                    "destination_league_id": 40,
                    "league_pair": "41->40",
                    "age_at_transfer": 21,
                    "primary_role": "complete_forward",
                    "origin_team_league_position": 1,
                    "destination_team_league_position": 12,
                    "origin_goals_scored_per90": 0.8,
                    "origin_shots_total_per90": 3.5,
                    "target_goals_scored_per90": 0.40,
                    "target_shots_total_per90": 2.8,
                    "target_starter": 1,
                },
                {
                    "origin_league_id": 88,
                    "destination_league_id": 40,
                    "league_pair": "88->40",
                    "age_at_transfer": 24,
                    "primary_role": "complete_forward",
                    "origin_team_league_position": 5,
                    "destination_team_league_position": 10,
                    "origin_goals_scored_per90": 0.6,
                    "origin_shots_total_per90": 3.0,
                    "target_goals_scored_per90": 0.34,
                    "target_shots_total_per90": 2.5,
                    "target_starter": 1,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "models.championship_projection._resolve_model_path",
            return_value=Path(tmpdir) / "projection.joblib",
        ):
            summary = train_projection_model(training_data)

        self.assertIn("goals_scored_per90", summary["metrics"])
        self.assertGreaterEqual(_estimate_adaptation_months(age=21, role_name="striker"), 2.0)

    def test_train_availability_model_and_risk_tier(self) -> None:
        training_data = pd.DataFrame(
            [
                {"availability_rate_3yr": 0.9, "injury_frequency_3yr": 1, "avg_injury_duration": 8, "max_injury_duration": 12, "muscle_injury_count": 0, "recurrence_rate": 0.0, "days_since_last_injury": 200, "minutes_continuity": 8, "age": 24, "position_group": "M", "target_available_75pct": 1},
                {"availability_rate_3yr": 0.6, "injury_frequency_3yr": 4, "avg_injury_duration": 20, "max_injury_duration": 60, "muscle_injury_count": 2, "recurrence_rate": 0.5, "days_since_last_injury": 20, "minutes_continuity": 1, "age": 29, "position_group": "D", "target_available_75pct": 0},
                {"availability_rate_3yr": 0.8, "injury_frequency_3yr": 2, "avg_injury_duration": 10, "max_injury_duration": 20, "muscle_injury_count": 1, "recurrence_rate": 0.2, "days_since_last_injury": 120, "minutes_continuity": 5, "age": 26, "position_group": "F", "target_available_75pct": 1},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "models.availability_risk._resolve_model_path",
            return_value=Path(tmpdir) / "availability.joblib",
        ):
            model = train_availability_model(training_data)

        self.assertIsNotNone(model)
        self.assertEqual(_risk_tier(0.85), "Low")
        self.assertEqual(_risk_tier(0.65), "Medium")
        self.assertEqual(_risk_tier(0.45), "High")

    def test_availability_prediction_uses_heuristic_fallback_without_model_artifact(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "availability_rate_3yr": 0.88,
                    "availability_rate_season": 0.92,
                    "injury_frequency_3yr": 1,
                    "recurrence_rate": 0.1,
                    "max_injury_duration": 12,
                    "minutes_continuity": 7,
                    "days_since_last_injury": 120,
                }
            ]
        )
        with patch("models.availability_risk._load_availability_model", side_effect=FileNotFoundError), patch(
            "models.availability_risk._build_availability_prediction_frame",
            return_value=frame,
        ):
            result = predict_availability_risk(9)

        self.assertGreater(result["probability_available_75pct"], 0.5)
        self.assertIn("Heuristic fallback", result["caveat"])

    def test_availability_prediction_uses_heuristic_when_injury_history_is_sparse(self) -> None:
        sparse_frame = pd.DataFrame(
            [
                {
                    "availability_rate_3yr": 0.88,
                    "availability_rate_season": 0.92,
                    "injury_frequency_3yr": 0,
                    "muscle_injury_count": 0,
                    "recurrence_rate": None,
                    "max_injury_duration": None,
                    "minutes_continuity": 7,
                    "days_since_last_injury": None,
                    "age": 24,
                    "position_group": "F",
                }
            ]
        )
        mock_model = SimpleNamespace(predict_proba=lambda _features: [[0.1, 0.9]])
        with patch("models.availability_risk._load_availability_model", return_value=mock_model), patch(
            "models.availability_risk._build_availability_prediction_frame",
            return_value=sparse_frame,
        ):
            result = predict_availability_risk(9)

        self.assertGreater(result["probability_available_75pct"], 0.5)
        self.assertIn("injury-history coverage is too sparse", result["caveat"])

    def test_train_value_model_and_value_helpers(self) -> None:
        training_data = pd.DataFrame(
            [
                {"age": 22, "position_group": "F", "role": "complete_forward", "league_level": 3, "contract_remaining_years": 2.5, "market_value_pretransfer": 300000, "per90_output": 2.0, "fee_paid": 450000},
                {"age": 25, "position_group": "M", "role": "controller", "league_level": 2, "contract_remaining_years": 1.5, "market_value_pretransfer": 500000, "per90_output": 1.5, "fee_paid": 600000},
                {"age": 28, "position_group": "D", "role": "covering_cb", "league_level": 3, "contract_remaining_years": 1.0, "market_value_pretransfer": 200000, "per90_output": 1.1, "fee_paid": 250000},
                {"age": 21, "position_group": "F", "role": "inside_forward", "league_level": 1, "contract_remaining_years": 3.0, "market_value_pretransfer": 1200000, "per90_output": 2.8, "fee_paid": 1600000},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "models.financial_value._resolve_model_path",
            return_value=Path(tmpdir) / "value.joblib",
        ):
            model = train_value_model(training_data)

        self.assertIsNotNone(model)
        self.assertEqual(_wage_fit(proposed_wage=10000, role_band=12000), "ok")
        self.assertEqual(_wage_fit(proposed_wage=13000, role_band=12000), "caution")
        self.assertEqual(_wage_fit(proposed_wage=16000, role_band=12000), "exceeds")

    def test_projection_and_value_use_heuristic_fallbacks_without_artifacts(self) -> None:
        projection_frame = pd.DataFrame(
            [
                {
                    "origin_league_id": 41,
                    "destination_league_id": 40,
                    "league_pair": "41->40",
                    "age_at_transfer": 23.0,
                    "primary_role": "complete_forward",
                    "origin_team_league_position": 6.0,
                    "destination_team_league_position": 16.0,
                    "origin_goals_scored_per90": 0.55,
                    "origin_shots_total_per90": 2.8,
                }
            ]
        )
        value_frame = pd.DataFrame(
            [
                {
                    "age": 23.0,
                    "position_group": "F",
                    "role": "complete_forward",
                    "league_level": 3.0,
                    "contract_remaining_years": 2.0,
                    "market_value_pretransfer": 450000.0,
                    "per90_output": 1.9,
                    "trajectory": 0.08,
                }
            ]
        )
        brief = {
            "budget_max_wage": 120000,
            "budget_max_contract_years": 3,
            "quality_score": 62,
        }

        with patch("models.championship_projection._load_projection_bundle", side_effect=FileNotFoundError), patch(
            "models.championship_projection._build_projection_feature_frame",
            return_value=projection_frame,
        ), patch(
            "models.championship_projection._heuristic_starter_probability",
            return_value=0.64,
        ):
            projection = project_to_championship(9, "2025")

        with patch("models.financial_value._load_value_model_bundle", side_effect=FileNotFoundError), patch(
            "models.financial_value._build_value_prediction_frame",
            return_value=value_frame,
        ):
            financial = estimate_value(9, brief)

        self.assertIn("goals_scored_per90", projection["projected_performance"])
        self.assertEqual(projection["projected_minutes_share"], 0.64)
        self.assertIn("Heuristic", projection["confidence_note"])
        self.assertEqual(financial["fair_value_band"]["mid"], 450000.0)
        self.assertIn("Heuristic valuation", financial["caveat"])

    def test_projection_heuristic_fallback_handles_missing_feature_frame(self) -> None:
        with patch("models.championship_projection._load_projection_bundle", side_effect=FileNotFoundError), patch(
            "models.championship_projection._build_projection_feature_frame",
            return_value=pd.DataFrame(),
        ), patch(
            "models.championship_projection._infer_role_name",
            return_value="controller",
        ), patch(
            "models.championship_projection.load_player_row",
            return_value={"birth_date": None, "current_age_years": 27.0},
        ):
            projection = project_to_championship(9, "2025")

        self.assertEqual(projection["projected_performance"], {})
        self.assertIn("incomplete", projection["confidence_note"])

    def test_projection_feature_frame_uses_brief_override_for_destination_context(self) -> None:
        match_frame = pd.DataFrame(
            [
                {"league_id": 41, "date": "2025-02-01", "goals_scored_per90": 0.5},
            ]
        )
        per90 = pd.DataFrame(
            [
                {"goals_scored_per90": 0.5, "shots_total_per90": 2.1},
            ]
        )
        with patch("models.championship_projection.load_player_match_frame", return_value=match_frame), patch(
            "models.championship_projection.load_player_role_row",
            return_value={"primary_role": "complete_forward"},
        ), patch(
            "models.championship_projection.load_player_row",
            return_value={"birth_date": None},
        ), patch(
            "models.championship_projection._compute_per90_frame",
            return_value=per90,
        ), patch(
            "models.championship_projection._lookup_team_league_position",
            return_value=7.0,
        ):
            frame = _build_projection_feature_frame(
                player_id=9,
                season="2025",
                role="complete_forward",
                brief={"destination_league_id": 94, "destination_team_league_position": 4.0},
            )

        self.assertEqual(float(frame.iloc[0]["destination_league_id"]), 94.0)
        self.assertEqual(float(frame.iloc[0]["destination_team_league_position"]), 4.0)
        self.assertEqual(frame.iloc[0]["league_pair"], "41->94")

    def test_projection_feature_frame_uses_current_age_when_birth_date_missing(self) -> None:
        match_frame = pd.DataFrame(
            [
                {"league_id": 41, "date": "2025-02-01", "goals_scored_per90": 0.5},
            ]
        )
        per90 = pd.DataFrame(
            [
                {"goals_scored_per90": 0.5, "shots_total_per90": 2.1},
            ]
        )
        with patch("models.championship_projection.load_player_match_frame", return_value=match_frame), patch(
            "models.championship_projection.load_player_role_row",
            return_value={"primary_role": "complete_forward"},
        ), patch(
            "models.championship_projection.load_player_row",
            return_value={"birth_date": None, "current_age_years": 29.0},
        ), patch(
            "models.championship_projection._compute_per90_frame",
            return_value=per90,
        ), patch(
            "models.championship_projection._lookup_team_league_position",
            return_value=7.0,
        ):
            frame = _build_projection_feature_frame(
                player_id=9,
                season="2025",
                role="complete_forward",
                brief={"destination_league_id": 40, "destination_team_league_position": 16.0},
            )

        self.assertEqual(float(frame.iloc[0]["age_at_transfer"]), 29.0)

    def test_projection_feature_frame_alignment_backfills_missing_model_columns(self) -> None:
        frame = pd.DataFrame([{"origin_league_id": 41, "league_pair": "41->40"}])
        aligned = _align_projection_feature_frame(
            frame,
            ["origin_league_id", "league_pair", "origin_goals_per90"],
        )

        self.assertIn("origin_goals_per90", aligned.columns)
        self.assertTrue(pd.isna(aligned.iloc[0]["origin_goals_per90"]))

    def test_value_prediction_frame_derives_league_level_from_current_league(self) -> None:
        with patch("models.financial_value.load_player_row", return_value={"birth_date": None, "current_age_years": 27.0, "current_league_id": 61}), patch(
            "models.financial_value.load_latest_market_value_row",
            return_value={},
        ), patch(
            "models.financial_value.load_player_role_row",
            return_value={"primary_role": "controller"},
        ), patch(
            "models.financial_value.load_player_match_frame",
            return_value=pd.DataFrame(),
        ), patch(
            "models.financial_value._compute_per90_frame",
            return_value=pd.DataFrame(),
        ), patch(
            "models.financial_value.compute_trajectory_features",
            return_value={},
        ):
            frame = _build_value_prediction_frame(9)

        self.assertEqual(float(frame.iloc[0]["league_level"]), 1.0)
        self.assertEqual(float(frame.iloc[0]["age"]), 27.0)

    def test_comparable_transactions_prefer_matching_role_and_recent_context(self) -> None:
        recent_matching = SimpleNamespace(
            player_id=11,
            date=pd.Timestamp("2026-01-10").date(),
            type="Permanent",
            team_in="Team A",
            team_out="Team B",
            fee_paid=500000,
        )
        recent_non_matching = SimpleNamespace(
            player_id=12,
            date=pd.Timestamp("2026-02-10").date(),
            type="Permanent",
            team_in="Team C",
            team_out="Team D",
            fee_paid=600000,
        )

        with patch("models.financial_value._recent_transfer_candidates", return_value=[recent_non_matching, recent_matching]), patch(
            "models.financial_value._load_comparable_player_metadata",
            return_value={
                11: {"primary_role": "controller", "league_level": 2.0},
                12: {"primary_role": "ball_winner", "league_level": 2.0},
            },
        ):
            comparables = _find_comparable_transactions(
                pd.Series({"role": "controller", "league_level": 2.0}),
                n=2,
                player_id=9,
            )

        self.assertEqual(comparables[0]["player_id"], 11)

    def test_scoring_helpers_and_pipeline_validation(self) -> None:
        weights = _blended_weights("promotion_accelerator", "championship_transition")
        self.assertAlmostEqual(sum(weights.values()), 0.88)
        effective_weights = effective_layer_weights(weights)
        self.assertEqual(effective_weights["tactical_fit"], 0.0)
        self.assertAlmostEqual(sum(effective_weights.values()), sum(weights.values()))
        self.assertAlmostEqual(
            _projection_score_from_bundle({"projected_performance": {"goals": {"p10": 1.0, "p50": 5.0, "p90": 9.0}}}),
            63.94,
            places=2,
        )
        self.assertEqual(_financial_score(1.0), 100.0)
        self.assertEqual(_trend_label(0.1), "improving")
        self.assertEqual(_trend_label(-0.1), "declining")
        self.assertEqual(_trend_label(0.0), "stable")
        self.assertGreater(_consistency_to_score(0.2), _consistency_to_score(1.0))
        self.assertGreater(_trend_to_score(0.1), 50.0)
        self.assertEqual(_vector_from_percentiles({"b": 20, "a": 10}).tolist(), [10.0, 20.0])

        with self.assertRaises(ValueError):
            create_brief({"role_name": "controller"})
        with self.assertRaises(ValueError):
            log_override(
                player_id=1,
                brief_id=1,
                original_output={},
                decision="sign",
                reason_category="bad-category",
                reason_text="x",
                overridden_by="tester",
            )

    def test_pipeline_timeline_normalisation_and_season_resolution(self) -> None:
        self.assertEqual(str(_normalise_timeline("summer_2026")), "2026-07-01")
        self.assertEqual(str(_normalise_timeline("2026-08-15")), "2026-08-15")
        with self.assertRaises(ValueError):
            _normalise_timeline("deadline_day")

        with patch("governance.pipeline._available_player_role_seasons", return_value=[2022, 2023, 2024, 2025]):
            self.assertEqual(_resolve_brief_season({"timeline": _normalise_timeline("summer_2026")}), "2025")
            self.assertEqual(_resolve_brief_season({"timeline": _normalise_timeline("2025-01-10")}), "2024")
            self.assertEqual(_resolve_brief_season({"timeline": None}), "2025")

    def test_generate_longlist_skips_failed_players_and_persists_successes(self) -> None:
        passed = pd.DataFrame(
            [
                {"player_id": 1, "player_name": "Player A", "current_team": "Team A", "current_league_id": 40},
                {"player_id": 2, "player_name": "Player B", "current_team": "Team B", "current_league_id": 40},
            ]
        )
        captured = {}

        def fake_upsert(frame, brief):
            captured["frame"] = frame.copy()
            captured["brief"] = brief

        with patch("governance.pipeline._load_brief_dict", return_value={"brief_id": 7, "season": "2025"}), patch(
            "governance.pipeline.filter_universe",
            return_value=passed,
        ), patch(
            "governance.pipeline.compute_composite",
            side_effect=[
                {"composite_score": 75.0, "confidence_tier": "Medium"},
                RuntimeError("projection failed"),
            ],
        ), patch(
            "governance.pipeline._upsert_pipeline_rows",
            side_effect=fake_upsert,
        ):
            frame = generate_longlist(7)

        self.assertEqual(len(frame.index), 1)
        self.assertEqual(frame.iloc[0]["player_id"], 1)
        self.assertEqual(frame.attrs["skipped_players"][0]["player_id"], 2)
        self.assertEqual(captured["brief"]["brief_id"], 7)

    def test_compute_composite_falls_back_when_component_fails(self) -> None:
        template = SimpleNamespace(
            template_id=1,
            role_name="complete_forward",
            is_active=True,
            version="2026-Q1",
            metrics_json={"goals_scored": 1.0},
        )
        brief = {
            "brief_id": 1,
            "role_name": "complete_forward",
            "archetype_primary": "championship_transition",
            "archetype_secondary": None,
        }

        with patch("scoring.composite.get_active_template_for_role", return_value=template), patch(
            "scoring.composite.score_role_fit",
            side_effect=RuntimeError("role fit failed"),
        ), patch(
            "scoring.composite.score_l1_performance",
            return_value={"score": 60.0},
        ), patch(
            "scoring.composite.project_to_championship",
            return_value={
                "projected_performance": {"goals": {"p10": 0.1, "p50": 5.0, "p90": 9.0}},
                "projected_minutes_share": 0.65,
                "projected_adaptation_months": 5.0,
                "sample_size": 42,
                "confidence_note": None,
            },
        ), patch(
            "scoring.composite.predict_availability_risk",
            return_value={"probability_available_75pct": 0.9, "risk_tier": "Low", "contributing_factors": []},
        ), patch(
            "scoring.composite.compute_confidence",
            return_value={"confidence_tier": "High", "minutes_evidence_multiplier": 1.0, "total_minutes": 900},
        ), patch(
            "scoring.composite.estimate_value",
            return_value={
                "fair_value_band": {"low": 1, "mid": 2, "high": 3},
                "wage_fit": "ok",
                "resale_band_2yr": {"low": 1, "mid": 2, "high": 3},
                "replacement_cost": 3,
                "var_score": 0.0,
                "comparable_transactions": [],
            },
        ), patch(
            "scoring.composite._pathway_comparison",
            return_value=None,
        ), patch(
            "scoring.composite._log_prediction",
        ):
            result = compute_composite(player_id=1, brief=brief, season="2025")

        self.assertGreater(result["composite_score"], 0.0)
        self.assertTrue(result["component_fallbacks"]["role_fit"])
        self.assertTrue(result["model_warnings"])

    def test_value_adjusted_return_score_is_bounded_and_interpretable(self) -> None:
        strong = _value_adjusted_return_score(
            total_cost=1_000_000,
            fair_value_mid=1_200_000,
            resale_mid=1_800_000,
            quality_score=70.0,
        )
        weak = _value_adjusted_return_score(
            total_cost=2_500_000,
            fair_value_mid=900_000,
            resale_mid=700_000,
            quality_score=48.0,
        )
        self.assertGreater(strong, 0.0)
        self.assertLess(weak, 0.0)
        self.assertLessEqual(strong, 1.0)
        self.assertGreaterEqual(weak, -1.0)

    def test_upsert_pipeline_rows_filters_out_removed_longlist_entries(self) -> None:
        existing_kept = PipelineRow(
            brief_id=7,
            player_id=1,
            stage="longlist",
            archetype_primary="promotion_accelerator",
            archetype_secondary=None,
            intent="first_team",
            stage_changed_by="model",
        )
        existing_removed = PipelineRow(
            brief_id=7,
            player_id=2,
            stage="longlist",
            archetype_primary="promotion_accelerator",
            archetype_secondary=None,
            intent="first_team",
            stage_changed_by="model",
        )
        added: list[PipelineRow] = []

        class FakeSession:
            def __init__(self) -> None:
                self._existing = [existing_kept, existing_removed]

            def scalars(self, _statement: object):
                return list(self._existing)

            def add(self, row: PipelineRow) -> None:
                added.append(row)

        @contextmanager
        def fake_session_scope():
            yield FakeSession()

        frame = pd.DataFrame([{"player_id": 1}, {"player_id": 3}])
        brief = {
            "brief_id": 7,
            "archetype_primary": "promotion_accelerator",
            "archetype_secondary": None,
            "intent": "first_team",
        }
        with patch("governance.pipeline.session_scope", fake_session_scope):
            _upsert_pipeline_rows(frame, brief)

        self.assertEqual(existing_kept.stage, "longlist")
        self.assertEqual(existing_removed.stage, "filtered_out")
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].player_id, 3)

    def test_compute_composite_applies_minutes_penalty_and_removes_duplicate_role_weight(self) -> None:
        brief = {
            "brief_id": 8,
            "role_name": "controller",
            "archetype_primary": "promotion_accelerator",
            "archetype_secondary": None,
            "budget_max_wage": 10000,
        }
        template = SimpleNamespace(template_id=5)
        projection = {
            "projected_performance": {"goals": {"p10": 2.0, "p50": 5.0, "p90": 8.0}},
            "projected_minutes_share": 0.64,
            "projected_adaptation_months": 2.0,
        }
        availability = {"probability_available_75pct": 0.9}
        financial = {"var_score": 0.0, "fair_value_band": {"low": 1.0, "high": 2.0}}

        with patch("scoring.composite.get_active_template_for_role", return_value=template), patch(
            "scoring.composite.score_role_fit",
            return_value={"score": 70.0, "confidence_tier": "Low"},
        ), patch(
            "scoring.composite.score_l1_performance",
            return_value={"score": 60.0},
        ), patch(
            "scoring.composite.project_to_championship",
            return_value=projection,
        ), patch(
            "scoring.composite.predict_availability_risk",
            return_value=availability,
        ), patch(
            "scoring.composite.estimate_value",
            return_value=financial,
        ), patch(
            "scoring.composite.compute_confidence",
            return_value={
                "confidence_tier": "Low",
                "total_minutes": 400.0,
                "minutes_evidence_multiplier": (400 / 900) ** 0.5,
            },
        ), patch("scoring.composite._log_prediction"):
            result = compute_composite(player_id=9, brief=brief, season="2025")

        expected_weights = effective_layer_weights(_blended_weights("promotion_accelerator", None))
        self.assertEqual(result["per_layer_weights_used"]["tactical_fit"], 0.0)
        self.assertAlmostEqual(result["per_layer_weights_used"]["role_fit"], expected_weights["role_fit"])
        self.assertAlmostEqual(result["minutes_evidence_multiplier"], (400 / 900) ** 0.5)
        self.assertTrue(result["minutes_sample_limited"])
        self.assertLess(result["composite_score"], 35.0)

    def test_promote_to_shortlist_uses_projection_score_not_raw_p50(self) -> None:
        latest_prediction = SimpleNamespace(
            role_fit_score=70.0,
            championship_projection_50th=6.2,
            financial_value_band_low=100000.0,
            financial_value_band_high=200000.0,
            composite_score=45.0,
        )
        pipeline_row = SimpleNamespace(stage="longlist", stage_changed_date=None)

        class FakeSession:
            def __init__(self) -> None:
                self._calls = 0

            def scalar(self, statement):
                self._calls += 1
                if self._calls == 1:
                    return latest_prediction
                return pipeline_row

        @contextmanager
        def fake_session_scope():
            yield FakeSession()

        brief = {
            "brief_id": 8,
            "season": "2025",
            "archetype_primary": "championship_transition",
            "pathway_player_id": None,
        }
        with patch("governance.pipeline._load_brief_dict", return_value=brief), patch(
            "governance.pipeline.session_scope", fake_session_scope
        ), patch(
            "governance.pipeline.compute_confidence", return_value={"confidence_tier": "High"}
        ):
            result = promote_to_shortlist(8, 9)

        self.assertTrue(result["ok"])
        self.assertEqual(result["unmet_conditions"], [])


if __name__ == "__main__":
    unittest.main()
