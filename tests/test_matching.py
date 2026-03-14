from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ingestion.matching import (
    PlayerMatch,
    _score_candidate,
    build_source_lookup_key,
    resolve_source_player_id,
    save_source_player_mapping,
)


class MatchingTests(unittest.TestCase):
    def test_build_source_lookup_key_prefers_external_id(self) -> None:
        key = build_source_lookup_key(
            "Lewis Fiorini",
            source_team_name="Stockport County",
            source_player_external_id=" ABC123 ",
        )

        self.assertEqual(key, "id:abc123")

    def test_build_source_lookup_key_normalises_name_and_team(self) -> None:
        key = build_source_lookup_key("Málaga C.F.", source_team_name="RKC Waalwijk")

        self.assertEqual(key, "name:malaga c f|team:rkc waalwijk")

    def test_resolve_source_player_id_uses_saved_mapping_before_fuzzy_match(self) -> None:
        saved_mapping = SimpleNamespace(player_id=44)

        with patch("ingestion.matching.get_source_player_mapping", return_value=saved_mapping), patch(
            "ingestion.matching.find_player_match"
        ) as mocked_match:
            player_id = resolve_source_player_id(
                "wyscout",
                "Lewis Fiorini",
                source_team_name="Stockport County",
            )

        self.assertEqual(player_id, 44)
        mocked_match.assert_not_called()

    def test_resolve_source_player_id_persists_new_fuzzy_match(self) -> None:
        with patch("ingestion.matching.get_source_player_mapping", return_value=None), patch(
            "ingestion.matching.find_player_match",
            return_value=PlayerMatch(player_id=18, score=0.91),
        ), patch("ingestion.matching.save_source_player_mapping", return_value=1) as mocked_save:
            player_id = resolve_source_player_id(
                "wyscout",
                "Lewis Fiorini",
                source_team_name="Stockport County",
                league_id=41,
            )

        self.assertEqual(player_id, 18)
        mocked_save.assert_called_once()
        self.assertEqual(mocked_save.call_args.kwargs["matched_by"], "fuzzy_match")
        self.assertAlmostEqual(mocked_save.call_args.kwargs["match_score"], 0.91)

    def test_resolve_source_player_id_falls_back_without_league_filter(self) -> None:
        with patch("ingestion.matching.get_source_player_mapping", return_value=None), patch(
            "ingestion.matching.find_player_match",
            side_effect=[None, PlayerMatch(player_id=25, score=0.83)],
        ) as mocked_match, patch("ingestion.matching.save_source_player_mapping", return_value=1) as mocked_save:
            player_id = resolve_source_player_id(
                "wyscout",
                "Carl Rushworth",
                source_team_name="Coventry City",
                league_id=40,
            )

        self.assertEqual(player_id, 25)
        self.assertEqual(mocked_match.call_count, 2)
        self.assertIsNone(mocked_match.call_args_list[1].kwargs["league_id"])
        self.assertEqual(mocked_save.call_args.kwargs["matched_by"], "fuzzy_match_no_league")

    def test_save_source_player_mapping_upserts_expected_row(self) -> None:
        with patch("ingestion.matching.upsert_rows", return_value=1) as mocked_upsert:
            inserted = save_source_player_mapping(
                "wyscout",
                player_id=9,
                source_player_name="Louie Barry",
                source_team_name="Stockport County",
                league_id=41,
                match_score=1.0,
                matched_by="manual",
            )

        self.assertEqual(inserted, 1)
        args, _kwargs = mocked_upsert.call_args
        rows = args[1]
        self.assertEqual(rows[0]["source"], "wyscout")
        self.assertEqual(rows[0]["source_lookup_key"], "name:louie barry|team:stockport county")
        self.assertEqual(rows[0]["player_id"], 9)
        self.assertEqual(rows[0]["matched_by"], "manual")

    def test_score_candidate_does_not_penalise_name_match_when_team_is_different(self) -> None:
        score = _score_candidate("c rushworth", "carl rushworth", "coventry city", "brighton")

        self.assertGreaterEqual(score, 0.88)


if __name__ == "__main__":
    unittest.main()
