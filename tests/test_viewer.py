from __future__ import annotations

from datetime import datetime
from io import BytesIO
import unittest
from base64 import b64encode
from unittest.mock import patch

from viewer.app import application


def _call_app(
    path: str,
    query_string: str = "",
    *,
    method: str = "GET",
    body: bytes = b"",
) -> tuple[str, list[tuple[str, str]], bytes]:
    captured: dict[str, object] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = headers

    body = b"".join(
        application(
            {
                "PATH_INFO": path,
                "QUERY_STRING": query_string,
                "REQUEST_METHOD": method,
                "CONTENT_LENGTH": str(len(body)),
                "wsgi.input": BytesIO(body),
            },
            start_response,
        )
    )
    return str(captured["status"]), list(captured["headers"]), body


class ViewerAppTests(unittest.TestCase):
    @patch("viewer.app.get_dashboard_context")
    def test_homepage_renders(self, mock_context: object) -> None:
        mock_context.return_value = {
            "title": "Stockport Data Viewer",
            "generated_at": datetime(2026, 3, 12, 19, 0),
            "totals": {
                "players": 1,
                "fixtures": 2,
                "match_performances": 3,
                "fixture_team_stats": 4,
                "match_events": 5,
                "lineups": 6,
                "standings_snapshots": 7,
                "wyscout_season_stats": 8,
            },
            "coverage_rows": [],
            "recent_fixtures": [],
            "recent_briefs": [],
            "brief_builder": {
                "errors": [],
                "form_values": {
                    "role_name": "controller",
                    "intent": "first_team",
                    "archetype_primary": "promotion_accelerator",
                    "archetype_secondary": "",
                    "budget_max_fee": "5000000",
                    "budget_max_wage": "30000",
                    "budget_max_contract_years": "3",
                    "timeline": "summer_2026",
                    "age_min": "20",
                    "age_max": "29",
                    "created_by": "Ben Mills",
                    "approved_by": "Ben Mills",
                    "pathway_player_id": "",
                    "pathway_check_done": True,
                    "league_scope": [40],
                },
                "role_options": [("controller", "Midfield")],
                "archetype_options": ["promotion_accelerator"],
                "intent_options": [("first_team", "First-team upgrade")],
                "timeline_options": ["summer_2026"],
                "league_options": [{"league_id": 40, "label": "Championship (England, Tier 2)"}],
            },
            "wyscout_review": {
                "unmatched_rows": 12,
                "review_path": "/tmp/review.csv",
                "source_root": "/Users/benmills/Downloads",
            },
        }

        status, _headers, body = _call_app("/")

        self.assertTrue(status.startswith("200"))
        self.assertIn(b"Stockport Data Viewer", body)

    @patch("viewer.app._viewer_read_only", return_value=True)
    @patch("viewer.app.get_dashboard_context")
    def test_homepage_shows_read_only_banner(
        self,
        mock_context: object,
        _mock_read_only: object,
    ) -> None:
        mock_context.return_value = {
            "title": "Stockport Data Viewer",
            "generated_at": datetime(2026, 3, 12, 19, 0),
            "totals": {
                "players": 1,
                "fixtures": 2,
                "match_performances": 3,
                "fixture_team_stats": 4,
                "match_events": 5,
                "lineups": 6,
                "standings_snapshots": 7,
                "wyscout_season_stats": 8,
            },
            "coverage_rows": [],
            "recent_fixtures": [],
            "recent_briefs": [],
            "brief_builder": {
                "errors": [],
                "form_values": {
                    "role_name": "controller",
                    "intent": "first_team",
                    "archetype_primary": "promotion_accelerator",
                    "archetype_secondary": "",
                    "budget_max_fee": "",
                    "budget_max_wage": "",
                    "budget_max_contract_years": "3",
                    "timeline": "summer_2026",
                    "age_min": "20",
                    "age_max": "29",
                    "created_by": "Ben Mills",
                    "approved_by": "Ben Mills",
                    "pathway_player_id": "",
                    "pathway_check_done": True,
                    "league_scope": [40],
                },
                "role_options": [("controller", "Midfield")],
                "archetype_options": ["promotion_accelerator"],
                "intent_options": [("first_team", "First-team upgrade")],
                "timeline_options": ["summer_2026"],
                "league_options": [{"league_id": 40, "label": "Championship (England, Tier 2)"}],
            },
            "wyscout_review": {
                "unmatched_rows": 12,
                "review_path": "/tmp/review.csv",
                "source_root": "/Users/benmills/Downloads",
            },
        }

        status, _headers, body = _call_app("/")

        self.assertTrue(status.startswith("200"))
        self.assertIn(b"Read-only mode", body)
        self.assertIn(b"browse-only", body)
        self.assertIn(b"Brief creation is disabled", body)

    @patch("viewer.app.get_on_pitch_profiles_context")
    def test_on_pitch_dashboard_renders(self, mock_context: object) -> None:
        mock_context.return_value = {
            "title": "On-Pitch Profiles",
            "generated_at": datetime(2026, 3, 14, 12, 0),
            "role_options": [("controller", "Midfield")],
            "season_options": ["2025"],
            "selected_role": "controller",
            "selected_season": "2025",
            "overall_weights": [{"label": "Role Fit", "percent": 40.0}],
            "present_weights": [{"label": "Role Fit", "percent": 55.0}],
            "upside_weights": [{"label": "Role Fit", "percent": 45.0}],
            "score_guides": [
                {
                    "label": "On-Pitch",
                    "bands": [
                        {"label": "Excellent", "threshold": 48.0, "description": "Top 10% of this board"},
                        {"label": "Strong", "threshold": 45.0, "description": "Top 25% of this board"},
                        {"label": "Good", "threshold": 42.0, "description": "At or above the board median"},
                    ],
                    "median": 42.0,
                    "p75": 45.0,
                    "p90": 48.0,
                }
            ],
            "top_players": [
                {
                    "player_id": 9,
                    "player_name": "Lewis Fiorini",
                    "current_team": "Stockport County",
                    "league_name": "League One",
                    "on_pitch_score": 72.5,
                    "present_on_pitch_score": 70.0,
                    "upside_on_pitch_score": 74.1,
                    "total_minutes": 1450,
                }
            ],
            "present_top_players": [],
            "upside_top_players": [],
            "combined_league_top_fives": [],
            "on_pitch_league_top_fives": [],
            "present_league_top_fives": [],
            "upside_league_top_fives": [],
            "candidate_count": 12,
            "scored_count": 10,
            "skipped_count": 2,
            "minimum_minutes": 180,
            "candidate_limit": 250,
        }

        status, _headers, body = _call_app("/on-pitch", "role_name=controller&season=2025")

        self.assertTrue(status.startswith("200"))
        self.assertIn(b"On-Pitch Rankings", body)
        self.assertIn(b"Lewis Fiorini", body)
        self.assertIn(b"What's Good?", body)

    @patch("viewer.app.get_league_context")
    def test_league_page_renders(self, mock_context: object) -> None:
        mock_context.return_value = {
            "title": "League One 2025",
            "generated_at": datetime(2026, 3, 12, 19, 0),
            "league": {"name": "League One", "country": "England", "tier": 3},
            "league_id": 41,
            "selected_season": "2025",
            "available_seasons": ["2025"],
            "summary": {
                "fixture_count": 1,
                "performance_rows": 20,
                "team_stat_fixtures": 1,
                "event_fixtures": 1,
                "lineup_fixtures": 1,
                "first_fixture_date": "2025-08-01",
                "last_fixture_date": "2026-03-10",
            },
            "fixtures": [],
        }

        status, _headers, body = _call_app("/league", "league_id=41&season=2025")

        self.assertTrue(status.startswith("200"))
        self.assertIn(b"League One", body)

    @patch("viewer.app.get_fixture_context", return_value=None)
    def test_missing_fixture_returns_404(self, _mock_context: object) -> None:
        status, _headers, body = _call_app("/fixture/999")

        self.assertTrue(status.startswith("404"))
        self.assertIn(b"Fixture not found", body)

    @patch("viewer.app.get_player_context")
    def test_player_page_renders(self, mock_context: object) -> None:
        mock_context.return_value = {
            "title": "Player Name",
            "generated_at": datetime(2026, 3, 12, 19, 0),
            "player": {
                "player_name": "Player Name",
                "current_team": "Stockport County",
                "current_league_id": 41,
                "nationality": "England",
                "nationality_secondary": None,
                "birth_date": None,
                "height_cm": None,
                "weight_kg": None,
                "preferred_foot": None,
                "photo_url": None,
                "agent_name": None,
            },
            "injury_summary": {"injury_count": 0, "last_injury_date": None},
            "role_history": [],
            "per90_by_season": [],
            "market_value_history": [],
            "season_rows": [],
            "recent_rows": [],
            "scout_notes": [],
        }

        status, _headers, body = _call_app("/player/123")

        self.assertTrue(status.startswith("200"))
        self.assertIn(b"Player Name", body)

    @patch("viewer.app.get_wyscout_review_context")
    def test_wyscout_review_page_renders(self, mock_context: object) -> None:
        mock_context.return_value = {
            "title": "Wyscout Review",
            "generated_at": datetime(2026, 3, 12, 23, 30),
            "message": None,
            "review_path": "/tmp/review.csv",
            "source_root": "/Users/benmills/Downloads",
            "selected_league_id": None,
            "selected_season": None,
            "available_leagues": [],
            "available_seasons": [],
            "summary": {
                "unmatched_rows": 12,
                "filtered_rows": 12,
                "rows_with_candidates": 10,
                "rows_with_team_matches": 8,
                "rows_without_candidates": 2,
            },
            "league_breakdown": [],
            "rows": [],
            "page": 1,
            "total_pages": 1,
            "page_size": 24,
        }

        status, _headers, body = _call_app("/wyscout-review")

        self.assertTrue(status.startswith("200"))
        self.assertIn(b"Wyscout Review", body)

    @patch("viewer.app.apply_wyscout_review_mapping")
    def test_wyscout_apply_redirects_after_saving_mapping(self, mock_apply: object) -> None:
        mock_apply.return_value = {
            "player_id": 101,
            "rerun_summary": {"folders_processed": 1, "unmatched_rows": 7},
        }

        status, headers, _body = _call_app(
            "/wyscout-review/apply",
            method="POST",
            body=(
                b"player_id=101&source_player_name=Lee+Hyun-Ju&source_team_name=Wehen+Wiesbaden"
                b"&league_id=79&season=2023"
            ),
        )

        self.assertTrue(status.startswith("303"))
        self.assertIn(("Location", "/wyscout-review?league_id=79&season=2023&message=Saved+mapping+to+player+101.+Reran+1+folder%28s%29+and+now+have+7+unmatched+row%28s%29."), headers)

    @patch("viewer.app.apply_wyscout_review_mappings")
    def test_wyscout_batch_apply_redirects_after_saving_mappings(self, mock_apply: object) -> None:
        mock_apply.return_value = {
            "saved_count": 2,
            "rerun_summary": {"folders_processed": 2, "unmatched_rows": 5},
        }

        status, headers, _body = _call_app(
            "/wyscout-review/apply-batch",
            method="POST",
            body=(
                b"selected_match=%7B%22player_id%22%3A101%2C%22source_player_name%22%3A%22Lee+Hyun-Ju%22%2C%22source_team_name%22%3A%22Wehen+Wiesbaden%22%2C%22league_id%22%3A79%7D"
                b"&selected_match=%7B%22player_id%22%3A102%2C%22source_player_name%22%3A%22Lewis+Fiorini%22%2C%22source_team_name%22%3A%22Stockport+County%22%2C%22league_id%22%3A41%7D"
                b"&season=2023"
            ),
        )

        self.assertTrue(status.startswith("303"))
        self.assertIn(
            (
                "Location",
                "/wyscout-review?season=2023&message=Saved+2+mapping%28s%29.+Reran+2+folder%28s%29+and+now+have+5+unmatched+row%28s%29.",
            ),
            headers,
        )

    @patch("viewer.app._viewer_read_only", return_value=True)
    def test_brief_create_is_forbidden_in_read_only_mode(self, _mock_read_only: object) -> None:
        status, _headers, body = _call_app(
            "/briefs/create",
            method="POST",
            body=b"role_name=controller",
        )

        self.assertTrue(status.startswith("403"))
        self.assertIn(b"read-only mode", body)

    @patch("viewer.app._viewer_read_only", return_value=True)
    def test_brief_run_is_forbidden_in_read_only_mode(self, _mock_read_only: object) -> None:
        status, _headers, body = _call_app("/brief/8/run", method="POST", body=b"")

        self.assertTrue(status.startswith("403"))
        self.assertIn(b"read-only mode", body)

    @patch("viewer.app._viewer_read_only", return_value=True)
    def test_wyscout_actions_are_forbidden_in_read_only_mode(self, _mock_read_only: object) -> None:
        status, _headers, body = _call_app("/wyscout-review/reimport", method="POST", body=b"")

        self.assertTrue(status.startswith("403"))
        self.assertIn(b"read-only mode", body)

    @patch("viewer.app._viewer_basic_auth_credentials", return_value=("stockport", "secret-pass"))
    def test_homepage_requires_basic_auth_when_enabled(self, _mock_auth: object) -> None:
        status, headers, body = _call_app("/")

        self.assertTrue(status.startswith("401"))
        self.assertIn(("WWW-Authenticate", 'Basic realm="Stockport Viewer"'), headers)
        self.assertIn(b"Authentication required", body)

    @patch("viewer.app.get_dashboard_context")
    @patch("viewer.app._viewer_basic_auth_credentials", return_value=("stockport", "secret-pass"))
    def test_homepage_allows_valid_basic_auth(
        self,
        _mock_auth: object,
        mock_context: object,
    ) -> None:
        mock_context.return_value = {
            "title": "Stockport Data Viewer",
            "generated_at": datetime(2026, 3, 12, 19, 0),
            "totals": {
                "players": 1,
                "fixtures": 2,
                "match_performances": 3,
                "fixture_team_stats": 4,
                "match_events": 5,
                "lineups": 6,
                "standings_snapshots": 7,
                "wyscout_season_stats": 8,
            },
            "coverage_rows": [],
            "recent_fixtures": [],
            "recent_briefs": [],
            "brief_builder": {
                "errors": [],
                "form_values": {
                    "role_name": "controller",
                    "intent": "first_team",
                    "archetype_primary": "promotion_accelerator",
                    "archetype_secondary": "",
                    "budget_max_fee": "",
                    "budget_max_wage": "",
                    "budget_max_contract_years": "3",
                    "timeline": "summer_2026",
                    "age_min": "20",
                    "age_max": "29",
                    "created_by": "Ben Mills",
                    "approved_by": "Ben Mills",
                    "pathway_player_id": "",
                    "pathway_check_done": True,
                    "league_scope": [40],
                },
                "role_options": [("controller", "Midfield")],
                "archetype_options": ["promotion_accelerator"],
                "intent_options": [("first_team", "First-team upgrade")],
                "timeline_options": ["summer_2026"],
                "league_options": [{"league_id": 40, "label": "Championship (England, Tier 2)"}],
            },
            "wyscout_review": {
                "unmatched_rows": 12,
                "review_path": "/tmp/review.csv",
                "source_root": "/Users/benmills/Downloads",
            },
        }
        credentials = b64encode(b"stockport:secret-pass").decode("ascii")

        captured: dict[str, object] = {}

        def start_response(status: str, headers: list[tuple[str, str]]) -> None:
            captured["status"] = status
            captured["headers"] = headers

        body = b"".join(
            application(
                {
                    "PATH_INFO": "/",
                    "QUERY_STRING": "",
                    "REQUEST_METHOD": "GET",
                    "CONTENT_LENGTH": "0",
                    "wsgi.input": BytesIO(b""),
                    "HTTP_AUTHORIZATION": f"Basic {credentials}",
                },
                start_response,
            )
        )

        self.assertTrue(str(captured["status"]).startswith("200"))
        self.assertIn(b"Stockport Data Viewer", body)


if __name__ == "__main__":
    unittest.main()
