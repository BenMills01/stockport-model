"""Integration tests for the Stockport recruitment model pipeline.

These tests exercise the *interfaces between modules* rather than unit-testing
each function in isolation.  No real database connection is required — DB calls
are either patched to return synthetic DataFrames / dicts, or the pure
``_from_frames`` variants of feature functions are called directly.

Test categories
---------------
1. Feature pipeline — per-90 → rolling → confidence → availability → opposition
2. Gate pipeline — ``_evaluate_gates_with_context`` with constructed contexts
3. Scoring pipeline — ``compute_composite`` with all sub-models patched
4. Training data builders — graceful empty-DB behaviour
5. Output generation — action tiers and longlist report rendering
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch
import unittest

import pandas as pd

from features.availability import _compute_availability_features_from_frames
from features.opposition import _compute_opposition_splits_from_frames
from features.per90 import _compute_per90_frame
from features.rolling import compute_rolling
from gates.filtering import _evaluate_gates_with_context
from scoring.action_tiers import (
    classify_composite_action,
    composite_to_board_score,
    summarise_action_tiers,
)
from scoring.composite import (
    _financial_score,
    _projection_score_from_bundle,
    compute_composite,
    effective_layer_weights,
)


# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------

def _make_match_frame(
    n: int = 12,
    player_id: int = 1,
    league_id: int = 39,
    season: str = "2024",
    goals_per_match: float = 0.4,
    minutes_per_match: int = 85,
) -> pd.DataFrame:
    """Build a synthetic MatchPerformance-like DataFrame."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "fixture_id": 10000 + i,
                "player_id": player_id,
                "league_id": league_id,
                "season": season,
                "date": datetime(2024, 8 + i // 4, 3 + (i % 4) * 7),
                "home_team": "Team A",
                "away_team": "Team B",
                "team": "Team A",
                "is_home": i % 2 == 0,
                "minutes": minutes_per_match,
                "position": "M",
                "rating": 7.0 + (i % 3) * 0.2,
                "is_substitute": i >= n - 2,
                "is_captain": False,
                "goals_scored": 1 if i % max(1, round(1.0 / goals_per_match)) == 0 else 0,
                "goals_conceded": None,
                "assists": 1 if i % 4 == 0 else 0,
                "saves": None,
                "shots_total": 2,
                "shots_on_target": 1,
                "passes_total": 45,
                "passes_key": 3,
                "pass_accuracy": 82.0,
                "tackles_total": 4,
                "tackles_blocks": 1,
                "tackles_interceptions": 2,
                "duels_total": 8,
                "duels_won": 5,
                "dribbles_attempts": 2,
                "dribbles_success": 1,
                "dribbles_past": None,
                "fouls_committed": 1,
                "fouls_drawn": 1,
                "yellow_cards": 0,
                "red_cards": 0,
                "pen_won": None,
                "pen_committed": None,
                "pen_scored": None,
                "pen_missed": None,
                "pen_saved": None,
                "offsides": 0,
            }
        )
    return pd.DataFrame(rows)


def _make_gate_context(
    *,
    role: str = "complete_forward",
    age: float = 24.0,
    gbe_status: str = "green",
    market_value_eur: float | None = 400_000,
    wage_estimate: float | None = 40_000,
    availability_risk_prob: float = 0.18,
    projection_p50: float = 0.45,
    pathway_player: dict[str, Any] | None = None,
    height_cm: int = 182,
    confidence_tier: str = "High",
) -> dict[str, Any]:
    """Build a synthetic gate context dict matching what _evaluate_gates_with_context expects."""
    return {
        "player_id": 1,
        "player_name": "Test Player",
        "birth_date": date(2000, 6, 15),
        "age": age,
        "height_cm": height_cm,
        "primary_role": role,
        "secondary_role": None,
        "gbe_result": {"status": gbe_status, "points_estimate": 20, "notes": "ok"},
        "market_value_eur": Decimal(str(market_value_eur)) if market_value_eur is not None else None,
        "wage_estimate": Decimal(str(wage_estimate)) if wage_estimate is not None else None,
        "contract_expiry": date(2026, 6, 30),
        "availability_risk_prob": availability_risk_prob,
        "championship_projection": {
            "projected_minutes_share": 0.60,
            "projected_performance": {
                "goals": {"p10": 0.1, "p50": projection_p50, "p90": 0.8},
            },
        },
        "pathway_player": pathway_player,
        "role_profile": {
            "total_rows": 20,
            "starter_label_rows": 0,   # Below MIN_ROLE_PROFILE_LABEL_ROWS → caution, not fail
            "attack_zone_rows": 0,     # Below MIN_ROLE_PROFILE_GRID_ROWS → caution, not fail
            "height_cm": height_cm,
        },
        "confidence": {"confidence_tier": confidence_tier, "shrinkage_factor": 1.0},
        "acceptable_older_signing_age": None,
    }


def _make_brief(
    *,
    role_name: str = "complete_forward",
    archetype: str = "championship_transition",
    budget_max_fee: float | None = 600_000,
    budget_max_wage: float | None = 55_000,
    age_min: int = 20,
    age_max: int = 27,
    league_scope: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "brief_id": 99,
        "role_name": role_name,
        "archetype_primary": archetype,
        "archetype_secondary": None,
        "intent": "Signing for depth",
        "budget_max_fee": Decimal(str(budget_max_fee)) if budget_max_fee is not None else None,
        "budget_max_wage": Decimal(str(budget_max_wage)) if budget_max_wage is not None else None,
        "budget_max_contract_years": 2,
        "age_min": age_min,
        "age_max": age_max,
        "league_scope": league_scope or [39, 40, 41],
        "pathway_check_done": True,
        "pathway_player_id": None,
        "club_wage_band": None,
        "proposed_wage": None,
        "timeline": date(2025, 7, 1),
        "created_by": "test",
        "approved_by": "test",
    }


# ---------------------------------------------------------------------------
# 1. Feature pipeline integration
# ---------------------------------------------------------------------------

class FeaturePipelineIntegrationTests(unittest.TestCase):

    def test_per90_produces_numeric_columns_for_all_metrics(self) -> None:
        frame = _make_match_frame(n=10)
        per90 = _compute_per90_frame(frame)
        self.assertFalse(per90.empty)
        per90_cols = [c for c in per90.columns if c.endswith("_per90")]
        self.assertGreaterEqual(len(per90_cols), 5, "Expected multiple per-90 metric columns")
        for col in per90_cols:
            non_null = per90[col].dropna()
            if not non_null.empty:
                self.assertTrue(
                    all(v >= 0 for v in non_null),
                    f"Negative per-90 value in {col}",
                )

    def test_per90_to_rolling_chain(self) -> None:
        """Rolling stats can be computed directly on per-90 output DataFrame."""
        frame = _make_match_frame(n=15)
        per90 = _compute_per90_frame(frame)
        self.assertFalse(per90.empty)
        # compute_rolling takes the per-90 DataFrame directly (no DB call).
        rolling = compute_rolling(per90)
        self.assertIsInstance(rolling, dict)
        self.assertGreater(len(rolling), 0, "Expected rolling output for at least one metric")

    def test_availability_features_from_frames_empty_match(self) -> None:
        """Empty match frame returns a valid zero-filled feature dict."""
        features = _compute_availability_features_from_frames(
            match_frame=pd.DataFrame(),
            fixture_frame=pd.DataFrame(),
            sidelined_frame=pd.DataFrame(),
            injury_frame=pd.DataFrame(),
            event_frame=pd.DataFrame(),
            today=date.today(),
        )
        self.assertIn("availability_rate_season", features)
        self.assertIn("injury_frequency_3yr", features)
        self.assertEqual(features["injury_frequency_3yr"], 0)

    def test_availability_features_from_synthetic_frames(self) -> None:
        match_frame = _make_match_frame(n=20)
        sidelined_frame = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "start_date": pd.Timestamp("2024-01-10"),
                    "end_date": pd.Timestamp("2024-02-01"),
                    "type": "Hamstring",
                }
            ]
        )
        injury_frame = pd.DataFrame(
            [{"player_id": 1, "fixture_id": None, "type": "Hamstring", "reason": None, "date": pd.Timestamp("2024-01-10")}]
        )
        fixture_frame = pd.DataFrame(
            [
                {
                    "fixture_id": 10000 + i,
                    "league_id": 39,
                    "season": "2024",
                    "date": pd.Timestamp(f"2024-{8 + i // 4:02d}-{3 + (i % 4) * 7:02d}"),
                    "home_team": "Team A",
                    "away_team": "Team B",
                    "status": "FT",
                }
                for i in range(20)
            ]
        )
        features = _compute_availability_features_from_frames(
            match_frame=match_frame,
            fixture_frame=fixture_frame,
            sidelined_frame=sidelined_frame,
            injury_frame=injury_frame,
            event_frame=pd.DataFrame(),
            today=date(2024, 12, 31),
        )
        self.assertEqual(features["injury_frequency_3yr"], 1)
        self.assertEqual(features["muscle_injury_count"], 1)
        self.assertIsNotNone(features["max_injury_duration"])
        self.assertGreaterEqual(features["max_injury_duration"], 20)

    def test_opposition_splits_from_frames_returns_tier_keyed_dict(self) -> None:
        match_frame = _make_match_frame(n=20)
        standings_frame = pd.DataFrame(
            [
                {"league_id": 39, "team_name": "Team B", "position": 2, "date": pd.Timestamp("2024-08-01")},
                {"league_id": 39, "team_name": "Team B", "position": 3, "date": pd.Timestamp("2024-09-01")},
            ]
        )
        result = _compute_opposition_splits_from_frames(match_frame, standings_frame)
        self.assertIsInstance(result, dict)
        if result:
            first_metric = next(iter(result.values()))
            self.assertIn("tier1", first_metric)
            self.assertIn("tier2", first_metric)

    def test_opposition_splits_empty_standings(self) -> None:
        """Empty standings → splits dict with None values, no crash."""
        match_frame = _make_match_frame(n=8)
        result = _compute_opposition_splits_from_frames(match_frame, pd.DataFrame())
        self.assertIsInstance(result, dict)
        if result:
            for metric_splits in result.values():
                self.assertTrue(all(v is None for v in metric_splits.values()))

    def test_rolling_returns_expected_keys_per_metric(self) -> None:
        frame = _make_match_frame(n=15)
        per90 = _compute_per90_frame(frame)
        rolling = compute_rolling(per90)
        if rolling:
            first_val = next(iter(rolling.values()))
            for key in ("roll_3", "roll_5", "roll_10", "season_avg"):
                self.assertIn(key, first_val, f"Missing rolling key '{key}'")


# ---------------------------------------------------------------------------
# 2. Gate pipeline integration
# ---------------------------------------------------------------------------

class GatePipelineIntegrationTests(unittest.TestCase):

    def test_all_gates_pass_or_caution_with_good_player(self) -> None:
        """A well-qualified player clears all absolute gates (may still have cautions)."""
        context = _make_gate_context()
        brief = _make_brief()
        result = _evaluate_gates_with_context(context=context, brief=brief)
        self.assertTrue(result["passed"], f"Expected pass; gate_results={result['gate_results']}")
        absolute_failures = [
            g for g in result["gate_results"]
            if g["result"] == "fail" and g["gate_name"] in {
                "registration_eligibility", "affordability", "role_relevance",
                "role_profile_fit", "pathway_blocking_hard", "age_band_fit",
            }
        ]
        self.assertEqual(absolute_failures, [])

    def test_red_gbe_fails_registration_gate(self) -> None:
        context = _make_gate_context(gbe_status="red")
        brief = _make_brief()
        result = _evaluate_gates_with_context(context=context, brief=brief)
        self.assertFalse(result["passed"])
        reg_gate = next(g for g in result["gate_results"] if g["gate_name"] == "registration_eligibility")
        self.assertEqual(reg_gate["result"], "fail")

    def test_player_over_budget_fails_affordability_gate(self) -> None:
        context = _make_gate_context(market_value_eur=1_500_000, wage_estimate=90_000)
        brief = _make_brief(budget_max_fee=400_000, budget_max_wage=40_000)
        result = _evaluate_gates_with_context(context=context, brief=brief)
        self.assertFalse(result["passed"])
        aff_gate = next(g for g in result["gate_results"] if g["gate_name"] == "affordability")
        self.assertEqual(aff_gate["result"], "fail")

    def test_player_far_outside_age_band_fails(self) -> None:
        """Age 35 is > 27 + 2 leeway → hard fail."""
        context = _make_gate_context(age=35.0)
        brief = _make_brief(age_min=20, age_max=27)
        result = _evaluate_gates_with_context(context=context, brief=brief)
        self.assertFalse(result["passed"])
        age_gate = next(g for g in result["gate_results"] if g["gate_name"] == "age_band_fit")
        self.assertEqual(age_gate["result"], "fail")

    def test_player_within_age_leeway_gives_caution_not_fail(self) -> None:
        """Age 29 is above brief max 27 but within the ±2 yr tolerance — caution, pipeline still passes."""
        context = _make_gate_context(age=29.0)
        brief = _make_brief(age_min=20, age_max=27)
        result = _evaluate_gates_with_context(context=context, brief=brief)
        age_gate = next(g for g in result["gate_results"] if g["gate_name"] == "age_band_fit")
        # Within leeway → caution, not fail → gate class is "caution" (not absolute) so overall still passes.
        self.assertIn(age_gate["result"], ("caution", "pass"))
        self.assertTrue(result["passed"])

    def test_player_within_age_band_passes_cleanly(self) -> None:
        context = _make_gate_context(age=24.0)
        brief = _make_brief(age_min=20, age_max=27)
        result = _evaluate_gates_with_context(context=context, brief=brief)
        age_gate = next(g for g in result["gate_results"] if g["gate_name"] == "age_band_fit")
        self.assertEqual(age_gate["result"], "pass")

    def test_missing_market_value_fails_affordability_when_caps_set(self) -> None:
        context = _make_gate_context(market_value_eur=None, wage_estimate=None)
        brief = _make_brief(budget_max_fee=500_000, budget_max_wage=50_000)
        result = _evaluate_gates_with_context(context=context, brief=brief)
        aff_gate = next(g for g in result["gate_results"] if g["gate_name"] == "affordability")
        self.assertEqual(aff_gate["result"], "fail")

    def test_no_budget_caps_passes_affordability(self) -> None:
        context = _make_gate_context(market_value_eur=None, wage_estimate=None)
        brief = _make_brief(budget_max_fee=None, budget_max_wage=None)
        result = _evaluate_gates_with_context(context=context, brief=brief)
        aff_gate = next(g for g in result["gate_results"] if g["gate_name"] == "affordability")
        self.assertEqual(aff_gate["result"], "pass")

    def test_wrong_role_fails_role_relevance(self) -> None:
        context = _make_gate_context(role="goalkeeper")
        brief = _make_brief(role_name="complete_forward")
        result = _evaluate_gates_with_context(context=context, brief=brief)
        self.assertFalse(result["passed"])
        role_gate = next(g for g in result["gate_results"] if g["gate_name"] == "role_relevance")
        self.assertEqual(role_gate["result"], "fail")

    def test_gate_results_contain_expected_keys(self) -> None:
        context = _make_gate_context()
        brief = _make_brief()
        result = _evaluate_gates_with_context(context=context, brief=brief)
        for gate in result["gate_results"]:
            self.assertIn("gate_name", gate)
            self.assertIn("class", gate)
            self.assertIn("result", gate)
            self.assertIn("detail", gate)
            self.assertIn(gate["result"], {"pass", "fail", "caution", "exception"})

    def test_gate_results_list_is_non_empty(self) -> None:
        context = _make_gate_context()
        brief = _make_brief()
        result = _evaluate_gates_with_context(context=context, brief=brief)
        self.assertGreater(len(result["gate_results"]), 5)


# ---------------------------------------------------------------------------
# 3. Scoring pipeline integration
# ---------------------------------------------------------------------------

def _stub_role_fit(score: float = 72.0) -> dict[str, Any]:
    return {"score": score, "raw_score": score, "confidence_tier": "High", "template_id": 1}


def _stub_l1_performance(score: float = 65.0) -> dict[str, Any]:
    return {"score": score, "raw_score": score, "confidence_tier": "High"}


def _stub_projection(p50: float = 0.50) -> dict[str, Any]:
    return {
        "projected_performance": {"goals": {"p10": 0.1, "p50": p50, "p90": 0.9}},
        "projected_minutes_share": 0.65,
        "projected_adaptation_months": 5.0,
        "sample_size": 42,
        "confidence_note": None,
    }


def _stub_availability(prob: float = 0.80) -> dict[str, Any]:
    return {
        "probability_available_75pct": prob,
        "risk_tier": "Low",
        "contributing_factors": [],
        "caveat": None,
    }


def _stub_financial() -> dict[str, Any]:
    return {
        "fair_value_band": {"low": 300_000.0, "mid": 400_000.0, "high": 550_000.0},
        "wage_fit": "ok",
        "resale_band_2yr": {"low": 320_000.0, "mid": 380_000.0, "high": 440_000.0},
        "replacement_cost": 605_000.0,
        "var_score": 1.20,
        "comparable_transactions": [],
        "caveat": None,
    }


def _stub_confidence() -> dict[str, Any]:
    return {
        "confidence_tier": "High",
        "shrinkage_factor": 1.0,
        "minutes_evidence_multiplier": 1.0,
        "total_minutes": 950,
        "below_minutes_threshold": False,
    }


class ScoringPipelineIntegrationTests(unittest.TestCase):

    def _run_compute_composite(
        self,
        role_fit_score: float = 72.0,
        l1_score: float = 65.0,
        projection_p50: float = 0.50,
        availability_prob: float = 0.80,
    ) -> dict[str, Any]:
        template = SimpleNamespace(
            template_id=1,
            role_name="complete_forward",
            is_active=True,
            version="2026-Q1",
            metrics_json={"goals_scored": 0.6, "assists": 0.4},
        )
        brief = _make_brief()  # uses "championship_transition" archetype key

        with patch("scoring.composite.get_active_template_for_role", return_value=template), \
             patch("scoring.composite.score_role_fit", return_value=_stub_role_fit(role_fit_score)), \
             patch("scoring.composite.score_l1_performance", return_value=_stub_l1_performance(l1_score)), \
             patch("scoring.composite.project_to_championship", return_value=_stub_projection(projection_p50)), \
             patch("scoring.composite.predict_availability_risk", return_value=_stub_availability(availability_prob)), \
             patch("scoring.composite.estimate_value", return_value=_stub_financial()), \
             patch("scoring.composite.compute_confidence", return_value=_stub_confidence()), \
             patch("scoring.composite._log_prediction"), \
             patch("scoring.composite._pathway_comparison", return_value=None):
            return compute_composite(player_id=1, brief=brief, season="2024")

    def test_composite_score_is_float_and_non_negative(self) -> None:
        result = self._run_compute_composite()
        self.assertIsInstance(result["composite_score"], float)
        self.assertGreaterEqual(result["composite_score"], 0.0)

    def test_composite_score_bounded_by_availability_multiplier(self) -> None:
        """Score with low availability probability should be lower."""
        high_avail = self._run_compute_composite(availability_prob=0.95)
        low_avail = self._run_compute_composite(availability_prob=0.30)
        self.assertGreater(high_avail["composite_score"], low_avail["composite_score"])

    def test_composite_result_contains_all_required_keys(self) -> None:
        result = self._run_compute_composite()
        required = {
            "composite_score",
            "per_layer_scores",
            "per_layer_weights_used",
            "archetype_primary",
            "tactical_fit_cap_applied",
            "confidence_tier",
            "sample_minutes",
            "minutes_evidence_multiplier",
            "projection_band",
        }
        for key in required:
            self.assertIn(key, result, f"Missing key '{key}' from composite result")

    def test_per_layer_scores_all_present(self) -> None:
        result = self._run_compute_composite()
        layers = result["per_layer_scores"]
        for key in ("tactical_fit", "role_fit", "current_performance", "projection", "financial", "availability"):
            self.assertIn(key, layers)

    def test_tactical_fit_cap_applied_when_role_fit_below_40(self) -> None:
        result = self._run_compute_composite(role_fit_score=35.0)
        self.assertTrue(result["tactical_fit_cap_applied"])

    def test_tactical_fit_cap_not_applied_when_role_fit_above_40(self) -> None:
        result = self._run_compute_composite(role_fit_score=75.0)
        self.assertFalse(result["tactical_fit_cap_applied"])

    def test_projection_score_converts_correctly(self) -> None:
        """Low raw p50 values still convert to a low but non-zero score."""
        bundle = _stub_projection(p50=0.50)
        score = _projection_score_from_bundle(bundle)
        self.assertAlmostEqual(score, 5.56, places=1)

    def test_projection_score_clamped_at_100(self) -> None:
        """Extremely high p50 values approach the top of the display range."""
        bundle = _stub_projection(p50=15.0)
        score = _projection_score_from_bundle(bundle)
        self.assertGreater(score, 98.0)
        self.assertLessEqual(score, 100.0)

    def test_financial_score_converts_var_correctly(self) -> None:
        self.assertAlmostEqual(_financial_score(1.0), 100.0)
        self.assertAlmostEqual(_financial_score(0.0), 50.0)
        self.assertAlmostEqual(_financial_score(-1.0), 0.0)

    def test_effective_weights_redistribute_tactical_fit(self) -> None:
        raw = {
            "tactical_fit": 0.20,
            "role_fit": 0.20,
            "current_performance": 0.25,
            "upward_projection": 0.20,
            "financial_value": 0.15,
        }
        effective = effective_layer_weights(raw)
        self.assertEqual(effective["tactical_fit"], 0.0)
        remaining = sum(
            effective[k]
            for k in ("role_fit", "current_performance", "upward_projection", "financial_value")
        )
        self.assertAlmostEqual(remaining, 1.0, places=6)

    def test_no_active_template_raises_value_error(self) -> None:
        brief = _make_brief()
        with patch("scoring.composite.get_active_template_for_role", return_value=None):
            with self.assertRaises(ValueError):
                compute_composite(player_id=1, brief=brief, season="2024")

    def test_compute_composite_with_no_projection_performance(self) -> None:
        """When projection returns empty performance dict, score defaults to 50."""
        template = SimpleNamespace(
            template_id=1, role_name="complete_forward", is_active=True,
            version="2026-Q1", metrics_json={"goals_scored": 1.0},
        )
        brief = _make_brief()
        empty_projection = {
            "projected_performance": {},
            "projected_minutes_share": 0.5,
            "projected_adaptation_months": 5.0,
            "sample_size": 0,
            "confidence_note": "Heuristic used.",
        }
        with patch("scoring.composite.get_active_template_for_role", return_value=template), \
             patch("scoring.composite.score_role_fit", return_value=_stub_role_fit()), \
             patch("scoring.composite.score_l1_performance", return_value=_stub_l1_performance()), \
             patch("scoring.composite.project_to_championship", return_value=empty_projection), \
             patch("scoring.composite.predict_availability_risk", return_value=_stub_availability()), \
             patch("scoring.composite.estimate_value", return_value=_stub_financial()), \
             patch("scoring.composite.compute_confidence", return_value=_stub_confidence()), \
             patch("scoring.composite._log_prediction"), \
             patch("scoring.composite._pathway_comparison", return_value=None):
            result = compute_composite(player_id=1, brief=brief, season="2024")

        self.assertIsNotNone(result["composite_score"])
        self.assertGreater(result["composite_score"], 0.0)
        # Empty projection → projection_score = 50 → composite is a weighted blend.
        self.assertEqual(result["per_layer_scores"]["projection"], 50.0)


# ---------------------------------------------------------------------------
# 4. Training data builder — graceful empty-DB behaviour
# ---------------------------------------------------------------------------

class TrainingDataBuilderTests(unittest.TestCase):
    """Verify builders return empty DataFrames (not crash) when the DB is empty."""

    def _empty_session_scope(self):
        @contextmanager
        def _scope(*args, **kwargs):
            mock = MagicMock()
            mock.execute.return_value.mappings.return_value.all.return_value = []
            mock.execute.return_value.mappings.return_value.first.return_value = None
            yield mock
        return _scope

    def test_championship_builder_empty_db(self) -> None:
        from training.build_training_data import build_championship_projection_training_data

        with patch("training.build_training_data.session_scope", self._empty_session_scope()):
            df = build_championship_projection_training_data()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_availability_builder_empty_db(self) -> None:
        from training.build_training_data import build_availability_training_data

        with patch("training.build_training_data.session_scope", self._empty_session_scope()):
            df = build_availability_training_data()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_financial_builder_empty_db(self) -> None:
        from training.build_training_data import build_financial_value_training_data

        with patch("training.build_training_data.session_scope", self._empty_session_scope()):
            df = build_financial_value_training_data()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_championship_builder_returns_expected_columns_when_data_present(self) -> None:
        """Verify column schema matches what train_projection_model() expects."""
        from training.build_training_data import build_championship_projection_training_data

        season_agg_result = [
            {
                "player_id": 1, "season": "2022", "league_id": 41,
                "total_minutes": 2100, "appearances": 23, "goals": 8,
                "assists": 4, "shots_total": 40, "shots_on_target": 18,
                "passes_total": 920, "passes_key": 55, "pass_accuracy_avg": 78.0,
                "tackles_total": 30, "interceptions": 20, "dribbles_attempts": 30,
                "dribbles_success": 18, "duels_total": 150, "duels_won": 90,
                "saves": 0, "starts": 22, "primary_team": "Wrexham",
            },
            {
                "player_id": 1, "season": "2023", "league_id": 40,
                "total_minutes": 1980, "appearances": 22, "goals": 7,
                "assists": 5, "shots_total": 38, "shots_on_target": 16,
                "passes_total": 880, "passes_key": 50, "pass_accuracy_avg": 79.0,
                "tackles_total": 28, "interceptions": 19, "dribbles_attempts": 28,
                "dribbles_success": 16, "duels_total": 140, "duels_won": 85,
                "saves": 0, "starts": 18, "primary_team": "Stockport County",
            },
        ]
        player_birth_result = [{"player_id": 1, "birth_date": date(1999, 3, 12)}]
        role_result = [{"player_id": 1, "season": "2022", "primary_role": "complete_forward"}]
        position_result: list = []

        @contextmanager
        def _mock_session(*args, **kwargs):
            mock = MagicMock()

            def _execute_side_effect(query, *a, **kw):
                inner = MagicMock()
                query_str = str(query)
                if "SUM" in query_str and "match_performances" in query_str:
                    inner.mappings.return_value.all.return_value = season_agg_result
                elif "birth_date" in query_str:
                    inner.mappings.return_value.all.return_value = player_birth_result
                elif "player_roles" in query_str:
                    inner.mappings.return_value.all.return_value = role_result
                elif "standings_snapshots" in query_str:
                    inner.mappings.return_value.all.return_value = position_result
                else:
                    inner.mappings.return_value.all.return_value = []
                return inner

            mock.execute.side_effect = _execute_side_effect
            yield mock

        with patch("training.build_training_data.session_scope", _mock_session):
            df = build_championship_projection_training_data()

        self.assertFalse(df.empty, "Expected one training row from two-season data")
        self.assertIn("origin_league_id", df.columns)
        self.assertIn("destination_league_id", df.columns)
        self.assertIn("age_at_transfer", df.columns)
        self.assertIn("target_starter", df.columns)
        self.assertIn("league_pair", df.columns)
        self.assertEqual(df["league_pair"].iloc[0], "41->40")
        self.assertIn(int(df["target_starter"].iloc[0]), (0, 1))

    def test_train_all_skips_gracefully_when_data_insufficient(self) -> None:
        """train_all_models reports 'skipped' when builders return empty DataFrames."""
        from training.train_all import train_all_models

        with patch("training.train_all.build_championship_projection_training_data", return_value=pd.DataFrame()), \
             patch("training.train_all.build_availability_training_data", return_value=pd.DataFrame()), \
             patch("training.train_all.build_financial_value_training_data", return_value=pd.DataFrame()):
            results = train_all_models()

        for model_name in ("championship_projection", "availability_risk", "financial_value"):
            self.assertIn(model_name, results)
            self.assertEqual(results[model_name]["status"], "skipped")


# ---------------------------------------------------------------------------
# 5. Output generation integration
# ---------------------------------------------------------------------------

class OutputGenerationIntegrationTests(unittest.TestCase):

    def test_action_tier_label_for_various_scores(self) -> None:
        tier_checks = [
            (45.0, "Tier 1"),
            (36.0, "Tier 2"),
            (30.0, "Tier 3"),
            (24.0, "Tier 4"),
            (10.0, "Tier 5"),
        ]
        for score, expected_label in tier_checks:
            result = classify_composite_action(score)
            self.assertIsInstance(result, dict)
            self.assertIn("label", result)
            self.assertEqual(result["label"], expected_label, f"Score {score} should be '{expected_label}'")

    def test_board_score_monotonic_with_composite(self) -> None:
        """Higher composite score → higher board score."""
        scores = [10.0, 20.0, 30.0, 40.0, 50.0]
        board_scores = [composite_to_board_score(s) for s in scores]
        for i in range(len(board_scores) - 1):
            self.assertLessEqual(board_scores[i], board_scores[i + 1])

    def test_board_score_bounded_0_to_100(self) -> None:
        for raw in (0.0, 10.0, 30.0, 55.0, 100.0):
            bs = composite_to_board_score(raw)
            self.assertGreaterEqual(bs, 0.0)
            self.assertLessEqual(bs, 100.0)

    def test_summarise_action_tiers_returns_list_with_counts(self) -> None:
        scores = [45.0, 38.0, 32.0, 25.0, 10.0, None]
        summary = summarise_action_tiers(scores)
        self.assertIsInstance(summary, list)
        # Find Tier 1 entry.
        tier1 = next((t for t in summary if t["label"] == "Tier 1"), None)
        self.assertIsNotNone(tier1)
        self.assertEqual(tier1["count"], 1)

    def test_projection_score_from_bundle_empty_performances(self) -> None:
        bundle: dict[str, Any] = {
            "projected_performance": {},
            "projected_minutes_share": 0.5,
            "projected_adaptation_months": 5.0,
            "sample_size": 0,
        }
        score = _projection_score_from_bundle(bundle)
        self.assertEqual(score, 50.0)

    def test_longlist_report_renders_with_mocked_db(self) -> None:
        """generate_longlist_report renders valid HTML without a real DB."""
        from outputs.longlist import generate_longlist_report

        mock_brief = SimpleNamespace(
            brief_id=1,
            role_name="complete_forward",
            archetype_primary="championship_transition",
            archetype_secondary=None,
            pathway_player_id=None,
        )
        mock_pipeline_row = SimpleNamespace(
            player_id=42,
            archetype_primary="championship_transition",
            archetype_secondary=None,
            stage="longlist",
        )
        mock_player = SimpleNamespace(player_id=42, player_name="Test Player", current_team="Test FC")
        mock_prediction = SimpleNamespace(
            player_id=42,
            brief_id=1,
            composite_score=35.0,
            role_fit_score=72.0,
            l1_performance_score=65.0,
            championship_projection_50th=0.45,
            championship_projection_10th=0.20,
            championship_projection_90th=0.75,
            projected_minutes_share=0.65,
            availability_risk_prob=0.18,
            var_score=1.10,
        )

        @contextmanager
        def _mock_session(*args, **kwargs):
            mock = MagicMock()
            mock.get.return_value = mock_brief

            # longlist.py calls list(session.scalars(...)) twice:
            #   1st → pipeline_rows (Pipeline ORM objects)
            #   2nd → players (Player ORM objects, unique())
            pipeline_mock = MagicMock()
            pipeline_mock.__iter__ = MagicMock(return_value=iter([mock_pipeline_row]))
            player_mock = MagicMock()
            player_mock.__iter__ = MagicMock(return_value=iter([mock_player]))
            player_mock.unique.return_value.__iter__ = MagicMock(return_value=iter([mock_player]))

            mock.scalars.side_effect = [pipeline_mock, player_mock]
            yield mock

        with patch("outputs.longlist.session_scope", _mock_session), \
             patch("outputs.longlist._latest_prediction", return_value=mock_prediction):
            html = generate_longlist_report(brief_id=1)

        self.assertIsInstance(html, str)
        self.assertIn("Test Player", html)
        self.assertIn("Test FC", html)


if __name__ == "__main__":
    unittest.main()
