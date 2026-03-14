from __future__ import annotations

from decimal import Decimal
import unittest

from ingestion.common import _deduplicate_rows, normalise_text, parse_money_to_eur


class CommonIngestionTests(unittest.TestCase):
    def test_normalise_text_strips_accents_and_punctuation(self) -> None:
        self.assertEqual(normalise_text("Málaga C.F."), "malaga c f")
        self.assertEqual(normalise_text("C. O&apos;Riordan"), "c o riordan")

    def test_parse_money_to_eur_supports_common_units(self) -> None:
        self.assertEqual(
            parse_money_to_eur(
                "£1.2m",
                gbp_to_eur_rate=1.17,
                usd_to_eur_rate=0.92,
                chf_to_eur_rate=1.04,
            ),
            Decimal("1404000.00"),
        )
        self.assertEqual(
            parse_money_to_eur(
                "€750k",
                gbp_to_eur_rate=1.17,
                usd_to_eur_rate=0.92,
                chf_to_eur_rate=1.04,
            ),
            Decimal("750000.00"),
        )

    def test_deduplicate_rows_keeps_last_row_for_duplicate_conflict_key(self) -> None:
        rows = [
            {"fixture_id": 10, "player_id": 5, "minutes": 12},
            {"fixture_id": 11, "player_id": 6, "minutes": 90},
            {"fixture_id": 10, "player_id": 5, "minutes": 18},
        ]

        deduplicated = _deduplicate_rows(rows, ["fixture_id", "player_id"])

        self.assertEqual(len(deduplicated), 2)
        self.assertIn({"fixture_id": 11, "player_id": 6, "minutes": 90}, deduplicated)
        self.assertIn({"fixture_id": 10, "player_id": 5, "minutes": 18}, deduplicated)


if __name__ == "__main__":
    unittest.main()
