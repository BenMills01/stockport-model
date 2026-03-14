from __future__ import annotations

from datetime import date
from decimal import Decimal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from governance.prepare_live_pipeline import (
    _estimate_wage_from_market_value,
    _extract_contract_expiry,
    _extract_market_value_eur,
)


class PrepareLivePipelineTests(unittest.TestCase):
    def test_extract_market_value_eur_handles_numeric_and_money_strings(self) -> None:
        settings = SimpleNamespace(gbp_to_eur_rate=1.17, usd_to_eur_rate=0.92, chf_to_eur_rate=1.04)

        numeric = _extract_market_value_eur({"Market value": 250000}, settings)
        sterling = _extract_market_value_eur({"Market value": "£500k"}, settings)

        self.assertEqual(numeric, Decimal("250000.00"))
        self.assertEqual(sterling, Decimal("585000.00"))

    def test_extract_contract_expiry_parses_year_and_iso_dates(self) -> None:
        self.assertEqual(_extract_contract_expiry({"Contract expires": 2028}), date(2028, 6, 30))
        self.assertEqual(_extract_contract_expiry({"Contract expiry": "2027-06-30"}), date(2027, 6, 30))

    @patch("governance.prepare_live_pipeline.get_settings")
    def test_estimate_wage_from_market_value_uses_league_tier(self, mocked_settings: object) -> None:
        mocked_settings.return_value = SimpleNamespace(
            load_json=lambda _name: [
                {"league_id": 40, "tier": 2},
                {"league_id": 42, "tier": 4},
            ]
        )

        champ = _estimate_wage_from_market_value(Decimal("500000"), 40)
        l2 = _estimate_wage_from_market_value(Decimal("250000"), 42)

        self.assertEqual(champ, Decimal("90000.00"))
        self.assertEqual(l2, Decimal("30000.00"))


if __name__ == "__main__":
    unittest.main()
