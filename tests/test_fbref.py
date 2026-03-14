from __future__ import annotations

import unittest
from unittest.mock import patch

from ingestion.fbref import scrape_fbref_player_stats


FBREF_HTML = """
<html><body>
<!--
<table id="stats_standard">
  <thead>
    <tr><th>Player</th><th>Squad</th></tr>
  </thead>
  <tbody>
    <tr><td>Lewis Fiorini</td><td>Stockport County</td></tr>
  </tbody>
</table>
-->
<!--
<table id="stats_shooting">
  <thead>
    <tr><th>Player</th><th>Squad</th><th>Performance</th><th>Performance</th><th>Standard</th></tr>
    <tr><th>Player</th><th>Squad</th><th>xG</th><th>npxG</th><th>xG/Sh</th></tr>
  </thead>
  <tbody>
    <tr><td>Lewis Fiorini</td><td>Stockport County</td><td>4.2</td><td>4.0</td><td>0.11</td></tr>
  </tbody>
</table>
-->
<!--
<table id="stats_passing">
  <thead>
    <tr><th>Player</th><th>Squad</th><th>Expected</th><th>Total</th></tr>
    <tr><th>Player</th><th>Squad</th><th>xAG</th><th>PrgP</th></tr>
  </thead>
  <tbody>
    <tr><td>Lewis Fiorini</td><td>Stockport County</td><td>3.4</td><td>121</td></tr>
  </tbody>
</table>
-->
<!--
<table id="stats_possession">
  <thead>
    <tr><th>Player</th><th>Squad</th><th>Carries</th><th>Receiving</th></tr>
    <tr><th>Player</th><th>Squad</th><th>PrgC</th><th>PrgR</th></tr>
  </thead>
  <tbody>
    <tr><td>Lewis Fiorini</td><td>Stockport County</td><td>88</td><td>73</td></tr>
  </tbody>
</table>
-->
</body></html>
"""


class FbrefTests(unittest.TestCase):
    @patch("ingestion.fbref._fetch_fbref_html", return_value=FBREF_HTML)
    def test_scrape_fbref_player_stats_merges_required_metrics(self, _mock_fetch: object) -> None:
        frame = scrape_fbref_player_stats("https://fbref.com/fake", "2025-2026")

        self.assertEqual(len(frame.index), 1)
        row = frame.iloc[0]
        self.assertEqual(row["player_name"], "Lewis Fiorini")
        self.assertEqual(row["team_name"], "Stockport County")
        self.assertEqual(row["season"], "2025-2026")
        self.assertAlmostEqual(float(row["xg"]), 4.2)
        self.assertAlmostEqual(float(row["npxg"]), 4.0)
        self.assertAlmostEqual(float(row["xa"]), 3.4)
        self.assertAlmostEqual(float(row["xg_per_shot"]), 0.11)
        self.assertAlmostEqual(float(row["progressive_passes"]), 121.0)
        self.assertAlmostEqual(float(row["progressive_carries"]), 88.0)
        self.assertAlmostEqual(float(row["progressive_receptions"]), 73.0)


if __name__ == "__main__":
    unittest.main()
