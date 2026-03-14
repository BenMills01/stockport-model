from __future__ import annotations

import unittest

from scoring.action_tiers import classify_composite_action, composite_to_board_score, load_action_tiers, load_board_action_tiers, summarise_action_tiers


class ActionTierTests(unittest.TestCase):
    def test_action_tiers_load_in_descending_score_order(self) -> None:
        tiers = load_action_tiers()

        self.assertGreaterEqual(tiers[0]["min_score"], tiers[-1]["min_score"])
        self.assertEqual(tiers[0]["label"], "Tier 1")
        self.assertEqual(tiers[-1]["label"], "Tier 5")

    def test_classify_composite_action_uses_expected_thresholds(self) -> None:
        self.assertEqual(classify_composite_action(41.0)["action"], "Priority shortlist")
        self.assertEqual(classify_composite_action(35.0)["action"], "Scout next")
        self.assertEqual(classify_composite_action(29.0)["action"], "Active longlist")
        self.assertEqual(classify_composite_action(24.0)["action"], "Monitor")
        self.assertEqual(classify_composite_action(18.0)["action"], "Park")

    def test_board_score_translation_is_monotonic_and_higher_for_display(self) -> None:
        self.assertLess(composite_to_board_score(20.0), composite_to_board_score(30.0))
        self.assertAlmostEqual(composite_to_board_score(40.0), 87.82, places=2)
        self.assertAlmostEqual(composite_to_board_score(49.2), 92.49, places=2)

    def test_board_action_tiers_include_translated_thresholds(self) -> None:
        tiers = load_board_action_tiers()

        self.assertIn("board_min_score", tiers[0])
        self.assertGreater(tiers[0]["board_min_score"], tiers[-1]["board_min_score"])

    def test_summarise_action_tiers_counts_each_bucket(self) -> None:
        summary = summarise_action_tiers([42.0, 37.0, 31.0, 24.0, 19.0, 18.0])
        counts = {row["label"]: row["count"] for row in summary}

        self.assertEqual(counts["Tier 1"], 1)
        self.assertEqual(counts["Tier 2"], 1)
        self.assertEqual(counts["Tier 3"], 1)
        self.assertEqual(counts["Tier 4"], 1)
        self.assertEqual(counts["Tier 5"], 2)


if __name__ == "__main__":
    unittest.main()
