from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gates.filtering import _evaluate_gates_with_context, _get_gate_thresholds, filter_universe


class GateFilteringTests(unittest.TestCase):
    def test_evaluate_gates_with_context_fails_absolute_violations(self) -> None:
        context = {
            "primary_role": "controller",
            "secondary_role": None,
            "market_value_eur": Decimal("600000"),
            "wage_estimate": Decimal("60000"),
            "confidence": {"confidence_tier": "Medium"},
            "gbe_result": {"status": "red", "notes": "Failed"},
            "age": 23.0,
            "pathway_player": {
                "readiness_estimate_months": 8,
                "age": 23.5,
            },
            "availability_risk_prob": 0.2,
            "championship_projection_50th": 55.0,
            "acceptable_older_signing_age": 30,
        }
        brief = {
            "role_name": "controller",
            "archetype_primary": "promotion_accelerator",
            "budget_max_fee": Decimal("300000"),
            "budget_max_wage": Decimal("40000"),
            "budget_max_contract_years": 3,
            "age_min": 21,
            "age_max": 27,
            "pathway_player_id": 100,
        }

        result = _evaluate_gates_with_context(context=context, brief=brief)

        self.assertFalse(result["passed"])
        failures = {gate["gate_name"] for gate in result["gate_results"] if gate["result"] == "fail"}
        self.assertIn("registration_eligibility", failures)
        self.assertIn("affordability", failures)
        self.assertIn("pathway_blocking_hard", failures)

    def test_evaluate_gates_with_context_preserves_cautions_and_exceptions(self) -> None:
        context = {
            "primary_role": "ball_playing_cb",
            "secondary_role": "covering_cb",
            "role_profile": {},
            "market_value_eur": Decimal("250000"),
            "wage_estimate": Decimal("18000"),
            "confidence": {"confidence_tier": "Low"},
            "gbe_result": {"status": "green", "notes": "Likely qualifies"},
            "age": 30.5,
            "pathway_player": {
                "readiness_estimate_months": 18,
                "age": 20.0,
            },
            "availability_risk_prob": 0.55,
            "championship_projection_50th": 35.0,
            "acceptable_older_signing_age": 29,
        }
        brief = {
            "role_name": "ball_playing_cb",
            "archetype_primary": "championship_transition",
            "budget_max_fee": Decimal("400000"),
            "budget_max_wage": Decimal("25000"),
            "budget_max_contract_years": 3,
            "age_min": 22,
            "age_max": 31,
            "club_wage_band": Decimal("14000"),
            "pathway_player_id": 200,
        }

        result = _evaluate_gates_with_context(context=context, brief=brief)

        self.assertTrue(result["passed"])
        cautions = {gate["gate_name"] for gate in result["gate_results"] if gate["result"] == "caution"}
        exceptions = {gate["gate_name"] for gate in result["gate_results"] if gate["result"] == "exception"}
        self.assertIn("wage_discipline", cautions)
        self.assertIn("age_profile_fit", cautions)
        self.assertIn("availability_risk", cautions)
        self.assertIn("data_confidence", cautions)
        self.assertIn("championship_projection", exceptions)
        self.assertIn("pathway_blocking_soft", exceptions)

    def test_evaluate_gates_with_context_allows_age_within_two_year_leeway(self) -> None:
        context = {
            "primary_role": "controller",
            "secondary_role": None,
            "role_profile": {},
            "market_value_eur": Decimal("250000"),
            "wage_estimate": Decimal("6000"),
            "confidence": {"confidence_tier": "High"},
            "gbe_result": {"status": "green", "notes": "Likely qualifies"},
            "age": 25.5,
            "pathway_player": None,
            "availability_risk_prob": 0.15,
            "championship_projection_50th": 55.0,
            "acceptable_older_signing_age": 30,
        }
        brief = {
            "role_name": "controller",
            "archetype_primary": "promotion_accelerator",
            "budget_max_fee": Decimal("500000"),
            "budget_max_wage": Decimal("12000"),
            "budget_max_contract_years": 3,
            "age_min": 18,
            "age_max": 24,
        }

        result = _evaluate_gates_with_context(context=context, brief=brief)

        self.assertTrue(result["passed"])
        cautions = {gate["gate_name"] for gate in result["gate_results"] if gate["result"] == "caution"}
        self.assertIn("age_band_fit", cautions)

    def test_evaluate_gates_with_context_fails_when_age_is_outside_leeway(self) -> None:
        context = {
            "primary_role": "controller",
            "secondary_role": None,
            "role_profile": {},
            "market_value_eur": Decimal("250000"),
            "wage_estimate": Decimal("6000"),
            "confidence": {"confidence_tier": "High"},
            "gbe_result": {"status": "green", "notes": "Likely qualifies"},
            "age": 32.0,
            "pathway_player": None,
            "availability_risk_prob": 0.15,
            "championship_projection_50th": 55.0,
            "acceptable_older_signing_age": 30,
        }
        brief = {
            "role_name": "controller",
            "archetype_primary": "promotion_accelerator",
            "budget_max_fee": Decimal("500000"),
            "budget_max_wage": Decimal("12000"),
            "budget_max_contract_years": 3,
            "age_min": 18,
            "age_max": 24,
        }

        result = _evaluate_gates_with_context(context=context, brief=brief)

        self.assertFalse(result["passed"])
        failures = {gate["gate_name"] for gate in result["gate_results"] if gate["result"] == "fail"}
        self.assertIn("age_band_fit", failures)

    def test_evaluate_gates_with_context_skips_affordability_when_budget_is_blank(self) -> None:
        context = {
            "primary_role": "controller",
            "secondary_role": None,
            "role_profile": {},
            "market_value_eur": Decimal("250000"),
            "wage_estimate": Decimal("6000"),
            "confidence": {"confidence_tier": "High"},
            "gbe_result": {"status": "green", "notes": "Likely qualifies"},
            "age": 23.0,
            "pathway_player": None,
            "availability_risk_prob": 0.15,
            "championship_projection_50th": 55.0,
            "acceptable_older_signing_age": 30,
        }
        brief = {
            "role_name": "controller",
            "archetype_primary": "promotion_accelerator",
            "budget_max_fee": None,
            "budget_max_wage": None,
            "budget_max_contract_years": 3,
            "age_min": 18,
            "age_max": 24,
        }

        result = _evaluate_gates_with_context(context=context, brief=brief)

        self.assertTrue(result["passed"])
        affordability = next(gate for gate in result["gate_results"] if gate["gate_name"] == "affordability")
        self.assertEqual(affordability["result"], "pass")

    def test_evaluate_gates_with_context_fails_role_profile_for_wide_target_forward(self) -> None:
        context = {
            "primary_role": "target_forward",
            "secondary_role": None,
            "role_profile": {
                "height_cm": 181,
                "starter_label_rows": 20,
                "forward_label_share": 0.15,
                "midfielder_label_share": 0.85,
                "attack_zone_rows": 20,
                "central_attack_share": 0.20,
                "wide_attack_share": 0.80,
            },
            "market_value_eur": Decimal("250000"),
            "wage_estimate": Decimal("6000"),
            "confidence": {"confidence_tier": "High"},
            "gbe_result": {"status": "green", "notes": "Likely qualifies"},
            "age": 22.0,
            "pathway_player": None,
            "availability_risk_prob": 0.15,
            "championship_projection_50th": 55.0,
            "acceptable_older_signing_age": 30,
        }
        brief = {
            "role_name": "target_forward",
            "archetype_primary": "promotion_accelerator",
            "budget_max_fee": None,
            "budget_max_wage": None,
            "budget_max_contract_years": 3,
            "age_min": 18,
            "age_max": 24,
        }

        result = _evaluate_gates_with_context(context=context, brief=brief)

        self.assertFalse(result["passed"])
        failures = {gate["gate_name"] for gate in result["gate_results"] if gate["result"] == "fail"}
        self.assertIn("role_profile_fit", failures)

    def test_filter_universe_prefilters_by_role_and_season(self) -> None:
        captured = {}

        class FakeScalarResult:
            def __init__(self, items: list[object]) -> None:
                self._items = items

            def unique(self) -> "FakeScalarResult":
                return self

            def __iter__(self):
                return iter(self._items)

        class FakeSession:
            def scalars(self, query):
                captured["query"] = str(query)
                return FakeScalarResult(
                    [
                        SimpleNamespace(player_id=1, player_name="A", current_team="Team A", current_league_id=40),
                        SimpleNamespace(player_id=2, player_name="B", current_team="Team B", current_league_id=41),
                    ]
                )

        class FakeSessionScope:
            def __enter__(self):
                return FakeSession()

            def __exit__(self, exc_type, exc, tb):
                return False

        with patch("gates.filtering.session_scope", return_value=FakeSessionScope()), patch(
            "gates.filtering.apply_gates",
            side_effect=[{"passed": True, "gate_results": []}, {"passed": False, "gate_results": []}],
        ):
            frame = filter_universe(
                {
                    "role_name": "controller",
                    "season": "2025",
                    "league_scope": [40, 41],
                }
            )

        self.assertEqual(len(frame.index), 1)
        self.assertIn("JOIN player_roles", captured["query"])
        self.assertIn("player_roles.season", captured["query"])
        self.assertIn("player_roles.primary_role", captured["query"])
        self.assertIn("player_roles.secondary_role", captured["query"])


class GateThresholdLoadingTests(unittest.TestCase):
    def test_returns_defaults_when_file_missing(self) -> None:
        with patch("gates.filtering.get_settings", side_effect=FileNotFoundError("no settings")):
            thresholds = _get_gate_thresholds()
        self.assertAlmostEqual(thresholds["age_band_leeway_years"], 2.0)
        self.assertAlmostEqual(thresholds["availability_risk_caution_threshold"], 0.40)

    def test_valid_overrides_are_applied(self) -> None:
        with patch(
            "gates.filtering.get_settings",
            return_value=SimpleNamespace(
                load_json=lambda _f: {"age_band_leeway_years": 3.0, "championship_projection_floor": 50.0}
            ),
        ):
            thresholds = _get_gate_thresholds()
        self.assertAlmostEqual(thresholds["age_band_leeway_years"], 3.0)
        self.assertAlmostEqual(thresholds["championship_projection_floor"], 50.0)
        # Unoverridden keys still carry defaults
        self.assertEqual(thresholds["min_role_profile_label_rows"], 5)

    def test_out_of_range_value_falls_back_to_default(self) -> None:
        # availability_risk_caution_threshold must be in [0, 1]; 5.0 is invalid
        with patch(
            "gates.filtering.get_settings",
            return_value=SimpleNamespace(
                load_json=lambda _f: {"availability_risk_caution_threshold": 5.0}
            ),
        ):
            thresholds = _get_gate_thresholds()
        self.assertAlmostEqual(thresholds["availability_risk_caution_threshold"], 0.40)

    def test_wrong_type_value_falls_back_to_default(self) -> None:
        with patch(
            "gates.filtering.get_settings",
            return_value=SimpleNamespace(
                load_json=lambda _f: {"age_band_leeway_years": "forty"}
            ),
        ):
            thresholds = _get_gate_thresholds()
        self.assertAlmostEqual(thresholds["age_band_leeway_years"], 2.0)

    def test_thresholds_used_by_availability_gate(self) -> None:
        """A config-overridden threshold is actually applied in gate evaluation."""
        context = {
            "primary_role": "controller",
            "secondary_role": None,
            "role_profile": {},
            "market_value_eur": None,
            "wage_estimate": None,
            "confidence": {"confidence_tier": "High"},
            "gbe_result": {"status": "green"},
            "age": 24.0,
            "pathway_player": None,
            # Risk of 0.50 — above the default 0.40 threshold but below a raised 0.60 threshold
            "availability_risk_prob": 0.50,
            "championship_projection_50th": 60.0,
            "acceptable_older_signing_age": 30,
        }
        brief = {
            "role_name": "controller",
            "archetype_primary": "promotion_accelerator",
            "budget_max_fee": None,
            "budget_max_wage": None,
            "budget_max_contract_years": 1,
        }

        # With default threshold (0.40) risk=0.50 → caution
        with patch(
            "gates.filtering.get_settings",
            return_value=SimpleNamespace(load_json=lambda _f: {}),
        ):
            result_default = _evaluate_gates_with_context(context=context, brief=brief)

        avail_default = next(g for g in result_default["gate_results"] if g["gate_name"] == "availability_risk")
        self.assertEqual(avail_default["result"], "caution")

        # With raised threshold (0.60) risk=0.50 → pass
        with patch(
            "gates.filtering.get_settings",
            return_value=SimpleNamespace(
                load_json=lambda _f: {"availability_risk_caution_threshold": 0.60}
            ),
        ):
            result_raised = _evaluate_gates_with_context(context=context, brief=brief)

        avail_raised = next(g for g in result_raised["gate_results"] if g["gate_name"] == "availability_risk")
        self.assertEqual(avail_raised["result"], "pass")


if __name__ == "__main__":
    unittest.main()
