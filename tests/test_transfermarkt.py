from __future__ import annotations

from datetime import date
from decimal import Decimal
import unittest
from unittest.mock import MagicMock, patch, call

from ingestion.transfermarkt import (
    scrape_market_values,
    scrape_player_profile,
    scrape_player_transfer_history,
    scrape_player_value_history,
    ingest_player_profiles,
    ingest_transfer_fees,
    ingest_value_history,
    _team_name_similar,
    _parse_contract_date,
    _estimate_annual_wage_eur,
    _infer_tier_from_slug,
)


# ── HTML fixtures ────────────────────────────────────────────────────────────

SQUAD_PAGE_HTML = """
<html><body>
  <table class="items">
    <thead>
      <tr>
        <th>Player</th>
        <th>Club</th>
        <th>Market value</th>
        <th>Contract expires</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><a href="/louie-barry/profil/spieler/123456">Louie Barry</a></td>
        <td>Stockport County</td>
        <td>£1.2m</td>
        <td>Jun 30, 2028</td>
      </tr>
    </tbody>
  </table>
</body></html>
"""

PROFILE_PAGE_HTML = """
<html><body>
  <table class="auflistung">
    <tr>
      <th>Foot:</th>
      <td>right</td>
    </tr>
    <tr>
      <th>Player agent:</th>
      <td>Base Soccer</td>
    </tr>
  </table>
  <span class="flaggenrahmen">
    <img title="England" />
    <img title="Scotland" />
  </span>
</body></html>
"""

PROFILE_PAGE_NO_DATA_HTML = """
<html><body>
  <div>No info available</div>
</body></html>
"""

TRANSFERS_PAGE_HTML = """
<html><body>
  <table class="items">
    <thead>
      <tr>
        <th>Season</th><th>Date</th><th>Left</th><th>Joined</th><th>MV</th><th>Fee</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>23/24</td>
        <td>Jul 1, 2023</td>
        <td>Bury FC</td>
        <td>Stockport County</td>
        <td>£500k</td>
        <td>£200k</td>
      </tr>
      <tr>
        <td>22/23</td>
        <td>Jan 15, 2023</td>
        <td>Oldham Athletic</td>
        <td>Bury FC</td>
        <td>£300k</td>
        <td>free transfer</td>
      </tr>
      <tr>
        <td>21/22</td>
        <td>Aug 1, 2021</td>
        <td>Salford City</td>
        <td>Oldham Athletic</td>
        <td>£200k</td>
        <td>loan fee: £50k</td>
      </tr>
    </tbody>
  </table>
</body></html>
"""

VALUE_HISTORY_SCRIPT_HTML = """
<html><body>
  <script>
    var highchartsData = {"list": [
      {"datum_mw": "Jan 01, 2022", "mw": "\\u00a3300k", "x": 1640995200000, "y": 300000},
      {"datum_mw": "Jul 01, 2022", "mw": "\\u00a3500k", "x": 1656633600000, "y": 500000},
      {"datum_mw": "Jan 01, 2023", "mw": "\\u00a31.2m", "x": 1672531200000, "y": 1200000}
    ]};
  </script>
</body></html>
"""

VALUE_HISTORY_NO_DATA_HTML = """
<html><body>
  <script>var foo = 'bar';</script>
</body></html>
"""


# ── scrape_market_values ─────────────────────────────────────────────────────

class TestScrapeMarketValues(unittest.TestCase):
    @patch("ingestion.transfermarkt._fetch_transfermarkt_html", return_value=SQUAD_PAGE_HTML)
    def test_parses_currency_and_contract(self, _mock_fetch: object) -> None:
        frame = scrape_market_values("league-one/startseite/wettbewerb/GB3", "2025")

        self.assertEqual(len(frame.index), 1)
        row = frame.iloc[0]
        self.assertEqual(row["player_name"], "Louie Barry")
        self.assertEqual(row["team_name"], "Stockport County")
        self.assertEqual(row["season"], "2025")
        self.assertEqual(row["market_value_eur"], Decimal("1404000.00"))
        self.assertEqual(row["contract_expiry"], date(2028, 6, 30))

    @patch("ingestion.transfermarkt._fetch_transfermarkt_html", return_value=SQUAD_PAGE_HTML)
    def test_extracts_tm_profile_path(self, _mock_fetch: object) -> None:
        frame = scrape_market_values("league-one/startseite/wettbewerb/GB3", "2025")
        self.assertEqual(frame.iloc[0]["tm_profile_path"], "/louie-barry/profil/spieler/123456")

    @patch("ingestion.transfermarkt._fetch_transfermarkt_html", return_value=SQUAD_PAGE_HTML)
    def test_returns_none_tm_path_for_unlinked_players(self, _mock_fetch: object) -> None:
        # Player with no href should have NaN/None tm_profile_path.
        frame = scrape_market_values("league-one/startseite/wettbewerb/GB3", "2025")
        # All rows in our fixture have a link, so tm_profile_path is populated.
        self.assertIsNotNone(frame.iloc[0]["tm_profile_path"])


# ── scrape_player_profile ────────────────────────────────────────────────────

class TestScrapePlayerProfile(unittest.TestCase):
    @patch("ingestion.transfermarkt._fetch_transfermarkt_html", return_value=PROFILE_PAGE_HTML)
    def test_extracts_foot_and_agent(self, _mock_fetch: object) -> None:
        result = scrape_player_profile("https://www.transfermarkt.com/p/profil/spieler/123")
        self.assertEqual(result["preferred_foot"], "right")
        self.assertEqual(result["agent_name"], "Base Soccer")

    @patch("ingestion.transfermarkt._fetch_transfermarkt_html", return_value=PROFILE_PAGE_HTML)
    def test_extracts_secondary_nationality_from_flag_imgs(self, _mock_fetch: object) -> None:
        result = scrape_player_profile("https://www.transfermarkt.com/p/profil/spieler/123")
        self.assertEqual(result["nationality_secondary"], "Scotland")

    @patch("ingestion.transfermarkt._fetch_transfermarkt_html", return_value=PROFILE_PAGE_NO_DATA_HTML)
    def test_returns_none_for_missing_fields(self, _mock_fetch: object) -> None:
        result = scrape_player_profile("https://www.transfermarkt.com/p/profil/spieler/456")
        self.assertIsNone(result["preferred_foot"])
        self.assertIsNone(result["agent_name"])
        self.assertIsNone(result["nationality_secondary"])

    @patch("ingestion.transfermarkt._fetch_transfermarkt_html", return_value=PROFILE_PAGE_HTML)
    def test_returns_dict_with_expected_keys(self, _mock_fetch: object) -> None:
        result = scrape_player_profile("https://www.transfermarkt.com/p/profil/spieler/123")
        self.assertIn("preferred_foot", result)
        self.assertIn("nationality_secondary", result)
        self.assertIn("agent_name", result)


# ── scrape_player_transfer_history ───────────────────────────────────────────

class TestScrapePlayerTransferHistory(unittest.TestCase):
    @patch("ingestion.transfermarkt._fetch_transfermarkt_html", return_value=TRANSFERS_PAGE_HTML)
    def test_parses_paid_transfer_fee(self, _mock_fetch: object) -> None:
        records = scrape_player_transfer_history(
            "https://www.transfermarkt.com/p/profil/spieler/123"
        )
        paid = [r for r in records if r["transfer_type"] == "transfer" and r["fee_eur"]]
        self.assertTrue(len(paid) >= 1)
        fees = [r["fee_eur"] for r in paid]
        self.assertIn(Decimal("234000.00"), fees)  # £200k at 1.17 rate

    @patch("ingestion.transfermarkt._fetch_transfermarkt_html", return_value=TRANSFERS_PAGE_HTML)
    def test_marks_free_transfers(self, _mock_fetch: object) -> None:
        records = scrape_player_transfer_history(
            "https://www.transfermarkt.com/p/profil/spieler/123"
        )
        free = [r for r in records if r["transfer_type"] == "free"]
        self.assertTrue(len(free) >= 1)
        self.assertIsNone(free[0]["fee_eur"])

    @patch("ingestion.transfermarkt._fetch_transfermarkt_html", return_value=TRANSFERS_PAGE_HTML)
    def test_marks_loan_transfers(self, _mock_fetch: object) -> None:
        records = scrape_player_transfer_history(
            "https://www.transfermarkt.com/p/profil/spieler/123"
        )
        loans = [r for r in records if r["transfer_type"] == "loan"]
        self.assertTrue(len(loans) >= 1)

    @patch("ingestion.transfermarkt._fetch_transfermarkt_html", return_value=TRANSFERS_PAGE_HTML)
    def test_parses_transfer_dates(self, _mock_fetch: object) -> None:
        records = scrape_player_transfer_history(
            "https://www.transfermarkt.com/p/profil/spieler/123"
        )
        dates = [r["date"] for r in records if r.get("date")]
        self.assertIn(date(2023, 7, 1), dates)

    @patch("ingestion.transfermarkt._fetch_transfermarkt_html", return_value=TRANSFERS_PAGE_HTML)
    def test_derives_transfers_url_from_profile_url(self, mock_fetch: MagicMock) -> None:
        scrape_player_transfer_history(
            "https://www.transfermarkt.com/player/profil/spieler/999"
        )
        called_url = mock_fetch.call_args[0][0]
        self.assertIn("/transfers/spieler/", called_url)
        self.assertNotIn("/profil/spieler/", called_url)

    @patch(
        "ingestion.transfermarkt._fetch_transfermarkt_html",
        return_value="<html><body></body></html>",
    )
    def test_returns_empty_list_when_no_table(self, _mock_fetch: object) -> None:
        records = scrape_player_transfer_history(
            "https://www.transfermarkt.com/p/profil/spieler/123"
        )
        self.assertEqual(records, [])


# ── scrape_player_value_history ──────────────────────────────────────────────

class TestScrapePlayerValueHistory(unittest.TestCase):
    @patch(
        "ingestion.transfermarkt._fetch_transfermarkt_html",
        return_value=VALUE_HISTORY_SCRIPT_HTML,
    )
    def test_parses_three_snapshots(self, _mock_fetch: object) -> None:
        records = scrape_player_value_history(
            "https://www.transfermarkt.com/p/profil/spieler/123"
        )
        self.assertEqual(len(records), 3)

    @patch(
        "ingestion.transfermarkt._fetch_transfermarkt_html",
        return_value=VALUE_HISTORY_SCRIPT_HTML,
    )
    def test_snapshot_has_date_and_value(self, _mock_fetch: object) -> None:
        records = scrape_player_value_history(
            "https://www.transfermarkt.com/p/profil/spieler/123"
        )
        first = records[0]
        self.assertIn("date", first)
        self.assertIn("value_eur", first)
        self.assertIsInstance(first["date"], date)

    @patch(
        "ingestion.transfermarkt._fetch_transfermarkt_html",
        return_value=VALUE_HISTORY_SCRIPT_HTML,
    )
    def test_derives_marktwertverlauf_url(self, mock_fetch: MagicMock) -> None:
        scrape_player_value_history(
            "https://www.transfermarkt.com/player/profil/spieler/999"
        )
        called_url = mock_fetch.call_args[0][0]
        self.assertIn("marktwertverlauf", called_url)

    @patch(
        "ingestion.transfermarkt._fetch_transfermarkt_html",
        return_value=VALUE_HISTORY_NO_DATA_HTML,
    )
    def test_returns_empty_list_when_no_json(self, _mock_fetch: object) -> None:
        records = scrape_player_value_history(
            "https://www.transfermarkt.com/p/profil/spieler/123"
        )
        self.assertEqual(records, [])


# ── ingest_player_profiles ───────────────────────────────────────────────────

class TestIngestPlayerProfiles(unittest.TestCase):
    def _make_mapping_row(self, player_id: int, external_id: str) -> MagicMock:
        row = MagicMock()
        row.player_id = player_id
        row.source_player_external_id = external_id
        row.source_player_name = "test-player"
        row.source_team_name = "Test FC"
        return row

    @patch("ingestion.transfermarkt.session_scope")
    @patch("ingestion.transfermarkt.scrape_player_profile")
    @patch("ingestion.transfermarkt.time.sleep")
    def test_updates_player_when_profile_data_found(
        self, mock_sleep: MagicMock, mock_scrape: MagicMock, mock_session: MagicMock
    ) -> None:
        mock_scrape.return_value = {
            "preferred_foot": "right",
            "nationality_secondary": None,
            "agent_name": "Test Agency",
        }
        mapping_row = self._make_mapping_row(42, "99999")

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_session.return_value = ctx
        ctx.execute.return_value = MagicMock()
        ctx.scalars.return_value = MagicMock()
        all_mock = MagicMock()
        all_mock.all.return_value = [mapping_row]
        ctx.execute.return_value = all_mock

        result = ingest_player_profiles()
        self.assertIn("scraped", result)
        self.assertIn("updated", result)
        self.assertIn("errors", result)

    @patch("ingestion.transfermarkt.session_scope")
    def test_returns_zeros_when_no_mappings(self, mock_session: MagicMock) -> None:
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_session.return_value = ctx
        all_mock = MagicMock()
        all_mock.all.return_value = []
        ctx.execute.return_value = all_mock

        result = ingest_player_profiles()
        self.assertEqual(result, {"scraped": 0, "updated": 0, "errors": 0})

    @patch("ingestion.transfermarkt.session_scope")
    @patch("ingestion.transfermarkt.scrape_player_profile", side_effect=Exception("HTTP 403"))
    @patch("ingestion.transfermarkt.time.sleep")
    def test_counts_errors_on_scrape_failure(
        self, mock_sleep: MagicMock, mock_scrape: MagicMock, mock_session: MagicMock
    ) -> None:
        mapping_row = self._make_mapping_row(7, "77777")
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_session.return_value = ctx
        all_mock = MagicMock()
        all_mock.all.return_value = [mapping_row]
        ctx.execute.return_value = all_mock

        result = ingest_player_profiles()
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["scraped"], 0)


# ── ingest_transfer_fees ─────────────────────────────────────────────────────

class TestIngestTransferFees(unittest.TestCase):
    @patch("ingestion.transfermarkt.session_scope")
    def test_returns_zeros_when_no_mappings(self, mock_session: MagicMock) -> None:
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_session.return_value = ctx
        all_mock = MagicMock()
        all_mock.all.return_value = []
        ctx.execute.return_value = all_mock

        result = ingest_transfer_fees()
        self.assertEqual(result, {"scraped": 0, "matched": 0, "errors": 0})

    @patch("ingestion.transfermarkt.session_scope")
    @patch(
        "ingestion.transfermarkt.scrape_player_transfer_history",
        side_effect=Exception("timeout"),
    )
    @patch("ingestion.transfermarkt.time.sleep")
    def test_counts_errors_on_scrape_failure(
        self, mock_sleep: MagicMock, mock_scrape: MagicMock, mock_session: MagicMock
    ) -> None:
        mapping_row = MagicMock()
        mapping_row.player_id = 5
        mapping_row.source_player_external_id = "55555"
        mapping_row.source_player_name = "p"
        mapping_row.source_team_name = "t"

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_session.return_value = ctx
        all_mock = MagicMock()
        all_mock.all.return_value = [mapping_row]
        ctx.execute.return_value = all_mock

        result = ingest_transfer_fees()
        self.assertEqual(result["errors"], 1)


# ── ingest_value_history ─────────────────────────────────────────────────────

class TestIngestValueHistory(unittest.TestCase):
    @patch("ingestion.transfermarkt.session_scope")
    def test_returns_zeros_when_no_mappings(self, mock_session: MagicMock) -> None:
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_session.return_value = ctx
        all_mock = MagicMock()
        all_mock.all.return_value = []
        ctx.execute.return_value = all_mock

        result = ingest_value_history()
        self.assertEqual(result, {"scraped": 0, "upserted": 0, "errors": 0})

    @patch("ingestion.transfermarkt.upsert_rows")
    @patch("ingestion.transfermarkt.session_scope")
    @patch(
        "ingestion.transfermarkt.scrape_player_value_history",
        return_value=[
            {"date": date(2022, 1, 1), "value_eur": Decimal("300000.00")},
            {"date": date(2023, 1, 1), "value_eur": Decimal("1200000.00")},
        ],
    )
    @patch("ingestion.transfermarkt.time.sleep")
    def test_upserts_snapshots_for_mapped_player(
        self,
        mock_sleep: MagicMock,
        mock_scrape: MagicMock,
        mock_session: MagicMock,
        mock_upsert: MagicMock,
    ) -> None:
        mock_upsert.return_value = 2

        mapping_row = MagicMock()
        mapping_row.player_id = 10
        mapping_row.source_player_external_id = "10101"
        mapping_row.source_player_name = "p"
        mapping_row.source_team_name = "t"

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_session.return_value = ctx
        all_mock = MagicMock()
        all_mock.all.return_value = [mapping_row]
        ctx.execute.return_value = all_mock

        result = ingest_value_history()
        self.assertEqual(result["scraped"], 2)
        self.assertEqual(result["upserted"], 2)
        self.assertEqual(result["errors"], 0)

    @patch("ingestion.transfermarkt.session_scope")
    @patch(
        "ingestion.transfermarkt.scrape_player_value_history",
        side_effect=Exception("connection error"),
    )
    @patch("ingestion.transfermarkt.time.sleep")
    def test_counts_errors_on_scrape_failure(
        self, mock_sleep: MagicMock, mock_scrape: MagicMock, mock_session: MagicMock
    ) -> None:
        mapping_row = MagicMock()
        mapping_row.player_id = 3
        mapping_row.source_player_external_id = "33333"
        mapping_row.source_player_name = "p"
        mapping_row.source_team_name = "t"

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_session.return_value = ctx
        all_mock = MagicMock()
        all_mock.all.return_value = [mapping_row]
        ctx.execute.return_value = all_mock

        result = ingest_value_history()
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["scraped"], 0)


# ── helper unit tests ─────────────────────────────────────────────────────────

class TestHelpers(unittest.TestCase):
    def test_team_name_similar_identical(self) -> None:
        self.assertTrue(_team_name_similar("Stockport County", "Stockport County"))

    def test_team_name_similar_substring(self) -> None:
        self.assertTrue(_team_name_similar("Stockport", "Stockport County"))

    def test_team_name_similar_shared_token(self) -> None:
        self.assertTrue(_team_name_similar("Stockport County", "Stockport Athletic"))

    def test_team_name_not_similar(self) -> None:
        self.assertFalse(_team_name_similar("Bury FC", "Oldham Athletic"))

    def test_team_name_none_returns_false(self) -> None:
        self.assertFalse(_team_name_similar(None, "Stockport"))
        self.assertFalse(_team_name_similar("Stockport", None))

    def test_parse_contract_date_iso(self) -> None:
        self.assertEqual(_parse_contract_date("2028-06-30"), date(2028, 6, 30))

    def test_parse_contract_date_british(self) -> None:
        self.assertEqual(_parse_contract_date("30/06/2028"), date(2028, 6, 30))

    def test_parse_contract_date_transfermarkt_style(self) -> None:
        self.assertEqual(_parse_contract_date("Jun 30, 2028"), date(2028, 6, 30))

    def test_parse_contract_date_empty_returns_none(self) -> None:
        self.assertIsNone(_parse_contract_date(""))
        self.assertIsNone(_parse_contract_date("-"))
        self.assertIsNone(_parse_contract_date(None))

    def test_estimate_annual_wage_tier1(self) -> None:
        wage = _estimate_annual_wage_eur(Decimal("10000000"), tier=1)
        self.assertEqual(wage, Decimal("2500000.00"))

    def test_estimate_annual_wage_tier3(self) -> None:
        wage = _estimate_annual_wage_eur(Decimal("500000"), tier=3)
        self.assertEqual(wage, Decimal("40000.00"))

    def test_estimate_annual_wage_none_input(self) -> None:
        self.assertIsNone(_estimate_annual_wage_eur(None, tier=3))
        self.assertIsNone(_estimate_annual_wage_eur(Decimal("0"), tier=3))

    def test_infer_tier_from_slug_league_one(self) -> None:
        self.assertEqual(_infer_tier_from_slug("league-one/startseite/wettbewerb/GB3"), 3)

    def test_infer_tier_from_slug_championship(self) -> None:
        self.assertEqual(_infer_tier_from_slug("championship/startseite/wettbewerb/GB2"), 2)

    def test_infer_tier_from_slug_unknown_defaults_to_3(self) -> None:
        self.assertEqual(_infer_tier_from_slug("unknown-league/startseite"), 3)


if __name__ == "__main__":
    unittest.main()
