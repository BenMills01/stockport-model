from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ingestion.api_football import _build_fixture_row, _build_player_rows
from ingestion.api_football import _build_fixture_team_stat_rows, _build_lineup_rows
from ingestion.api_football import _build_match_event_rows, _build_match_performance_rows
from ingestion.api_football import _coerce_injury_params, _season_candidates_for_range, _tracked_leagues
from ingestion.api_football import collect_completed_fixtures, estimate_ingest_request_plan, fetch_api_usage
from ingestion.api_football import fetch_fixture_team_stats, fetch_lineups, fetch_match_events
from ingestion.api_football import fetch_match_performances, load_player_stats_coverage


class ApiFootballParsingTests(unittest.TestCase):
    def test_build_fixture_row_filters_expected_fields(self) -> None:
        payload = {
            "fixture": {
                "id": 101,
                "date": "2026-03-10T19:45:00+00:00",
                "referee": "J. Brooks",
                "status": {"short": "FT"},
            },
            "league": {"id": 41, "season": 2025},
            "teams": {
                "home": {"name": "Stockport County"},
                "away": {"name": "Blackpool"},
            },
            "goals": {"home": 2, "away": 1},
        }

        row = _build_fixture_row(payload)

        self.assertEqual(row["fixture_id"], 101)
        self.assertEqual(row["league_id"], 41)
        self.assertEqual(row["season"], "2025")
        self.assertEqual(row["home_team"], "Stockport County")
        self.assertEqual(row["away_team"], "Blackpool")
        self.assertEqual(row["home_score"], 2)
        self.assertEqual(row["away_score"], 1)

    def test_build_match_performance_rows_maps_nested_statistics(self) -> None:
        fixture = {
            "fixture_id": 101,
            "league_id": 41,
            "season": "2025",
            "date": datetime(2026, 3, 10, 19, 45, tzinfo=timezone.utc),
            "home_team": "Stockport County",
            "away_team": "Blackpool",
            "referee": "J. Brooks",
        }
        payload = [
            {
                "team": {"name": "Stockport County"},
                "players": [
                    {
                        "player": {"id": 9, "name": "Isaac Olaofe"},
                        "statistics": [
                            {
                                "games": {
                                    "minutes": 90,
                                    "position": "F",
                                    "rating": "7.5",
                                    "substitute": False,
                                    "captain": False,
                                },
                                "offsides": 2,
                                "shots": {"total": 4, "on": 2},
                                "goals": {"total": 1, "conceded": 0, "assists": 0, "saves": None},
                                "passes": {"total": 18, "key": 2, "accuracy": "78%"},
                                "tackles": {"total": 1, "blocks": 0, "interceptions": 1},
                                "duels": {"total": 9, "won": 5},
                                "dribbles": {"attempts": 3, "success": 2, "past": 1},
                                "fouls": {"committed": 1, "drawn": 3},
                                "cards": {"yellow": 0, "red": 0},
                                "penalty": {
                                    "won": 0,
                                    "commited": 0,
                                    "scored": 0,
                                    "missed": 0,
                                    "saved": 0,
                                },
                            }
                        ],
                    }
                ],
            }
        ]

        rows = _build_match_performance_rows(fixture, payload)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row["is_home"])
        self.assertEqual(row["player_id"], 9)
        self.assertEqual(row["minutes"], 90)
        self.assertEqual(row["pass_accuracy"], 78.0)
        self.assertEqual(row["dribbles_success"], 2)
        self.assertEqual(row["offsides"], 2)

    def test_build_fixture_team_stat_rows_parses_percentage_fields(self) -> None:
        payload = [
            {
                "team": {"name": "Stockport County"},
                "statistics": [
                    {"type": "Ball Possession", "value": "61%"},
                    {"type": "Total Shots", "value": 12},
                    {"type": "Shots on Goal", "value": 5},
                    {"type": "Corner Kicks", "value": 7},
                    {"type": "Fouls", "value": 11},
                    {"type": "Expected Goals", "value": "1.48"},
                    {"type": "Total passes", "value": 428},
                    {"type": "Passes %", "value": "84%"},
                ],
            }
        ]

        rows = _build_fixture_team_stat_rows(101, payload)

        self.assertEqual(rows[0]["possession"], 61.0)
        self.assertEqual(rows[0]["expected_goals"], 1.48)
        self.assertEqual(rows[0]["passes_accuracy"], 84.0)

    def test_build_match_event_rows_extracts_player_context(self) -> None:
        payload = [
            {
                "time": {"elapsed": 54, "extra": None},
                "team": {"name": "Stockport County"},
                "player": {"id": 9, "name": "Isaac Olaofe"},
                "assist": {"id": 18, "name": "Callum Camps"},
                "type": "Goal",
                "detail": "Normal Goal",
                "comments": None,
            }
        ]

        rows = _build_match_event_rows(101, payload)

        self.assertEqual(rows[0]["fixture_id"], 101)
        self.assertEqual(rows[0]["player_id"], 9)
        self.assertEqual(rows[0]["assist_player_id"], 18)
        self.assertEqual(rows[0]["event_type"], "Goal")

    def test_build_lineup_rows_marks_starters_and_substitutes(self) -> None:
        payload = [
            {
                "team": {"name": "Stockport County"},
                "formation": "4-3-3",
                "coach": {"id": 1, "name": "Dave Challinor"},
                "startXI": [{"player": {"id": 9, "number": 9, "pos": "F", "grid": "1:1"}}],
                "substitutes": [{"player": {"id": 21, "number": 21, "pos": "M", "grid": None}}],
            }
        ]

        rows = _build_lineup_rows(101, payload)

        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["is_starter"])
        self.assertFalse(rows[1]["is_starter"])
        self.assertEqual(rows[0]["formation"], "4-3-3")

    def test_build_player_rows_keeps_current_age_when_birth_date_missing(self) -> None:
        payload = [
            {
                "player": {
                    "id": 9,
                    "name": "Lewis Fiorini",
                    "nationality": "Scotland",
                    "age": 27,
                    "birth": {"date": None},
                    "height": "182 cm",
                    "weight": "73 kg",
                    "photo": "https://example.com/photo.png",
                },
                "statistics": [
                    {
                        "team": {"name": "Stockport County"},
                        "league": {"id": 41},
                    }
                ],
            }
        ]

        rows = _build_player_rows(payload)

        self.assertEqual(rows[0]["player_id"], 9)
        self.assertIsNone(rows[0]["birth_date"])
        self.assertEqual(rows[0]["current_age_years"], 27.0)
        self.assertEqual(rows[0]["current_league_id"], 41)

    def test_coerce_injury_params_defaults_to_player_and_accepts_explicit_league(self) -> None:
        self.assertEqual(_coerce_injury_params(1234), {"player": 1234})
        self.assertEqual(_coerce_injury_params({"league": 41}), {"league": 41})

    def test_tracked_leagues_filters_requested_subset(self) -> None:
        with patch(
            "ingestion.api_football.get_settings",
            return_value=SimpleNamespace(
                load_json=lambda _filename: [
                    {"league_id": 40, "name": "Championship"},
                    {"league_id": 41, "name": "League One"},
                ]
            ),
        ):
            leagues = _tracked_leagues([41])

        self.assertEqual(leagues, [{"league_id": 41, "name": "League One"}])

    def test_tracked_leagues_rejects_unknown_ids(self) -> None:
        with patch(
            "ingestion.api_football.get_settings",
            return_value=SimpleNamespace(
                load_json=lambda _filename: [{"league_id": 40, "name": "Championship"}]
            ),
        ):
            with self.assertRaises(ValueError):
                _tracked_leagues([999])

    def test_collect_completed_fixtures_uses_requested_league_ids(self) -> None:
        api_calls: list[dict[str, int | str]] = []

        def fake_api_get(_endpoint: str, params: dict[str, int | str]) -> dict[str, object]:
            api_calls.append(params)
            return {
                "response": [
                    {
                        "fixture": {
                            "id": int(params["league"]) * 10,
                            "date": "2026-03-10T19:45:00+00:00",
                            "referee": "J. Brooks",
                            "status": {"short": "FT"},
                        },
                        "league": {"id": params["league"], "season": 2025},
                        "teams": {
                            "home": {"name": "Stockport County"},
                            "away": {"name": "Blackpool"},
                        },
                        "goals": {"home": 2, "away": 1},
                    }
                ]
            }

        with patch(
            "ingestion.api_football.get_settings",
            return_value=SimpleNamespace(
                load_json=lambda _filename: [
                    {"league_id": 40, "name": "Championship"},
                    {"league_id": 41, "name": "League One"},
                ]
            ),
        ), patch("ingestion.api_football.api_get", side_effect=fake_api_get):
            rows = collect_completed_fixtures(date(2026, 3, 10), date(2026, 3, 10), league_ids=[41])

        self.assertEqual([call["league"] for call in api_calls], [41])
        self.assertEqual([call["season"] for call in api_calls], [2025])
        self.assertEqual(rows[0]["league_id"], 41)

    def test_season_candidates_follow_july_to_june_schedule(self) -> None:
        self.assertEqual(_season_candidates_for_range("2026-03-10", "2026-03-12"), [2025])
        self.assertEqual(_season_candidates_for_range("2026-07-01", "2026-07-02"), [2026])
        self.assertEqual(_season_candidates_for_range("2026-06-30", "2026-07-02"), [2025, 2026])

    def test_fetch_api_usage_parses_status_payload(self) -> None:
        with patch(
            "ingestion.api_football.api_get",
            return_value={
                "response": {
                    "requests": {"current": 12, "limit_day": 7500},
                    "subscription": {"plan": "Pro", "active": True, "end": "2026-04-03T19:54:28+00:00"},
                }
            },
        ):
            usage = fetch_api_usage()

        self.assertEqual(usage["requests_current"], 12)
        self.assertEqual(usage["requests_limit_day"], 7500)
        self.assertEqual(usage["requests_remaining"], 7488)
        self.assertEqual(usage["plan"], "Pro")

    def test_estimate_ingest_request_plan_counts_discovery_and_detail_calls(self) -> None:
        with patch(
            "ingestion.api_football.get_settings",
            return_value=SimpleNamespace(
                load_json=lambda _filename: [
                    {"league_id": 40, "name": "Championship"},
                    {"league_id": 41, "name": "League One"},
                ]
            ),
        ):
            estimate = estimate_ingest_request_plan(
                "2026-03-10",
                "2026-03-12",
                league_ids=[41],
                fixture_count=7,
            )

        self.assertEqual(estimate["league_count"], 1)
        self.assertEqual(estimate["season_count"], 1)
        self.assertEqual(estimate["fixture_discovery_calls"], 1)
        self.assertEqual(estimate["detail_batch_size"], 20)
        self.assertEqual(estimate["fixture_detail_batches"], 1)
        self.assertEqual(estimate["estimated_detail_calls"], 1)
        self.assertEqual(estimate["estimated_total_calls"], 2)

    def test_load_player_stats_coverage_filters_supported_leagues(self) -> None:
        responses = {
            40: {
                "response": [
                    {
                        "seasons": [
                            {
                                "year": 2025,
                                "coverage": {"fixtures": {"statistics_players": True}},
                            }
                        ]
                    }
                ]
            },
            41: {
                "response": [
                    {
                        "seasons": [
                            {
                                "year": 2025,
                                "coverage": {"fixtures": {"statistics_players": False}},
                            }
                        ]
                    }
                ]
            },
        }

        def fake_api_get(_endpoint: str, params: dict[str, int]) -> dict[str, object]:
            return responses[int(params["id"])]

        with patch(
            "ingestion.api_football.get_settings",
            return_value=SimpleNamespace(
                load_json=lambda _filename: [
                    {"league_id": 40, "name": "Championship"},
                    {"league_id": 41, "name": "League One"},
                ]
            ),
        ), patch("ingestion.api_football.api_get", side_effect=fake_api_get):
            usable = load_player_stats_coverage(2025)

        self.assertEqual(usable, [{"league_id": 40, "name": "Championship"}])

    def test_fetch_match_performances_batches_fixture_ids_and_falls_back_when_needed(self) -> None:
        fixtures = [
            {
                "fixture_id": 101,
                "league_id": 41,
                "season": "2025",
                "date": datetime(2026, 3, 10, 19, 45, tzinfo=timezone.utc),
                "home_team": "Stockport County",
                "away_team": "Blackpool",
                "referee": "J. Brooks",
            },
            {
                "fixture_id": 102,
                "league_id": 41,
                "season": "2025",
                "date": datetime(2026, 3, 10, 19, 45, tzinfo=timezone.utc),
                "home_team": "Bolton",
                "away_team": "Wigan",
                "referee": "J. Brooks",
            },
            {
                "fixture_id": 103,
                "league_id": 41,
                "season": "2025",
                "date": datetime(2026, 3, 10, 19, 45, tzinfo=timezone.utc),
                "home_team": "Reading",
                "away_team": "Barnsley",
                "referee": "J. Brooks",
            },
        ]
        api_calls: list[tuple[str, dict[str, object]]] = []

        def fake_api_get(endpoint: str, params: dict[str, object]) -> dict[str, object]:
            api_calls.append((endpoint, params))
            if endpoint == "/fixtures":
                return {
                    "response": [
                        {
                            "fixture": {"id": 101},
                            "players": [
                                {
                                    "team": {"name": "Stockport County"},
                                    "players": [
                                        {
                                            "player": {"id": 9, "name": "Isaac Olaofe"},
                                            "statistics": [
                                                {
                                                    "games": {
                                                        "minutes": 90,
                                                        "position": "F",
                                                        "rating": "7.5",
                                                        "substitute": False,
                                                        "captain": False,
                                                    },
                                                    "offsides": 0,
                                                    "shots": {"total": 4, "on": 2},
                                                    "goals": {"total": 1, "conceded": 0, "assists": 0, "saves": None},
                                                    "passes": {"total": 18, "key": 2, "accuracy": "78%"},
                                                    "tackles": {"total": 1, "blocks": 0, "interceptions": 1},
                                                    "duels": {"total": 9, "won": 5},
                                                    "dribbles": {"attempts": 3, "success": 2, "past": 1},
                                                    "fouls": {"committed": 1, "drawn": 3},
                                                    "cards": {"yellow": 0, "red": 0},
                                                    "penalty": {"won": 0, "commited": 0, "scored": 0, "missed": 0, "saved": 0},
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                        {"fixture": {"id": 102}, "players": []},
                    ]
                }
            if endpoint == "/fixtures/players":
                fixture_id = int(params["fixture"])
                return {
                    "response": [
                        {
                            "team": {"name": fixtures[fixture_id - 101]["home_team"]},
                            "players": [
                                {
                                    "player": {"id": fixture_id, "name": f"Player {fixture_id}"},
                                    "statistics": [
                                        {
                                            "games": {
                                                "minutes": 90,
                                                "position": "F",
                                                "rating": "7.0",
                                                "substitute": False,
                                                "captain": False,
                                            },
                                            "offsides": 0,
                                            "shots": {"total": 1, "on": 1},
                                            "goals": {"total": 0, "conceded": 0, "assists": 0, "saves": None},
                                            "passes": {"total": 10, "key": 1, "accuracy": "80%"},
                                            "tackles": {"total": 0, "blocks": 0, "interceptions": 0},
                                            "duels": {"total": 1, "won": 1},
                                            "dribbles": {"attempts": 1, "success": 1, "past": 0},
                                            "fouls": {"committed": 0, "drawn": 0},
                                            "cards": {"yellow": 0, "red": 0},
                                            "penalty": {"won": 0, "commited": 0, "scored": 0, "missed": 0, "saved": 0},
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                }
            raise AssertionError(endpoint)

        upsert_batches: list[int] = []

        def fake_upsert_rows(
            _model: object,
            rows: list[dict[str, object]],
            _keys: list[str],
        ) -> int:
            upsert_batches.append(len(rows))
            return len(rows)

        with patch("ingestion.api_football.api_get", side_effect=fake_api_get), patch(
            "ingestion.api_football.upsert_rows",
            side_effect=fake_upsert_rows,
        ), patch("ingestion.api_football.UPSERT_BATCH_SIZE", 2):
            inserted = fetch_match_performances(
                date(2026, 3, 10),
                date(2026, 3, 10),
                fixtures=fixtures,
            )

        self.assertEqual(inserted, 3)
        self.assertEqual(api_calls[0][0], "/fixtures")
        self.assertEqual(api_calls[0][1]["ids"], "101-102-103")
        self.assertEqual([endpoint for endpoint, _params in api_calls[1:]], ["/fixtures/players", "/fixtures/players"])
        self.assertEqual(upsert_batches, [2, 1])

    def test_fetch_fixture_team_stats_uses_batched_fixture_details_with_fallback(self) -> None:
        fixtures = [{"fixture_id": 101}, {"fixture_id": 102}]
        api_calls: list[tuple[str, dict[str, object]]] = []

        def fake_api_get(endpoint: str, params: dict[str, object]) -> dict[str, object]:
            api_calls.append((endpoint, params))
            if endpoint == "/fixtures":
                return {
                    "response": [
                        {
                            "fixture": {"id": 101},
                            "statistics": [
                                {
                                    "team": {"name": "Stockport County"},
                                    "statistics": [
                                        {"type": "Ball Possession", "value": "61%"},
                                        {"type": "Total Shots", "value": 12},
                                    ],
                                }
                            ],
                        },
                        {"fixture": {"id": 102}},
                    ]
                }
            if endpoint == "/fixtures/statistics":
                return {
                    "response": [
                        {
                            "team": {"name": "Blackpool"},
                            "statistics": [
                                {"type": "Ball Possession", "value": "39%"},
                                {"type": "Total Shots", "value": 8},
                            ],
                        }
                    ]
                }
            raise AssertionError(endpoint)

        with patch("ingestion.api_football.api_get", side_effect=fake_api_get), patch(
            "ingestion.api_football.upsert_rows",
            side_effect=lambda _model, rows, _keys: len(rows),
        ):
            inserted = fetch_fixture_team_stats(
                date(2026, 3, 10),
                date(2026, 3, 10),
                fixtures=fixtures,
            )

        self.assertEqual(inserted, 2)
        self.assertEqual([endpoint for endpoint, _params in api_calls], ["/fixtures", "/fixtures/statistics"])

    def test_fetch_match_events_uses_batched_fixture_details_with_fallback(self) -> None:
        fixtures = [{"fixture_id": 101}, {"fixture_id": 102}]
        api_calls: list[tuple[str, dict[str, object]]] = []

        def fake_api_get(endpoint: str, params: dict[str, object]) -> dict[str, object]:
            api_calls.append((endpoint, params))
            if endpoint == "/fixtures":
                return {
                    "response": [
                        {
                            "fixture": {"id": 101},
                            "events": [
                                {
                                    "time": {"elapsed": 54, "extra": None},
                                    "team": {"name": "Stockport County"},
                                    "player": {"id": 9, "name": "Isaac Olaofe"},
                                    "assist": {"id": 18, "name": "Callum Camps"},
                                    "type": "Goal",
                                    "detail": "Normal Goal",
                                    "comments": None,
                                }
                            ],
                        },
                        {"fixture": {"id": 102}},
                    ]
                }
            if endpoint == "/fixtures/events":
                return {
                    "response": [
                        {
                            "time": {"elapsed": 12, "extra": None},
                            "team": {"name": "Blackpool"},
                            "player": {"id": 4, "name": "Defender"},
                            "assist": {"id": None, "name": None},
                            "type": "Card",
                            "detail": "Yellow Card",
                            "comments": None,
                        }
                    ]
                }
            raise AssertionError(endpoint)

        with patch("ingestion.api_football.api_get", side_effect=fake_api_get), patch(
            "ingestion.api_football.upsert_rows",
            side_effect=lambda _model, rows, _keys: len(rows),
        ):
            inserted = fetch_match_events(
                date(2026, 3, 10),
                date(2026, 3, 10),
                fixtures=fixtures,
            )

        self.assertEqual(inserted, 2)
        self.assertEqual([endpoint for endpoint, _params in api_calls], ["/fixtures", "/fixtures/events"])

    def test_fetch_lineups_uses_batched_fixture_details_with_fallback(self) -> None:
        fixtures = [{"fixture_id": 101}, {"fixture_id": 102}]
        api_calls: list[tuple[str, dict[str, object]]] = []

        def fake_api_get(endpoint: str, params: dict[str, object]) -> dict[str, object]:
            api_calls.append((endpoint, params))
            if endpoint == "/fixtures":
                return {
                    "response": [
                        {
                            "fixture": {"id": 101},
                            "lineups": [
                                {
                                    "team": {"name": "Stockport County"},
                                    "formation": "4-3-3",
                                    "coach": {"id": 1, "name": "Dave Challinor"},
                                    "startXI": [{"player": {"id": 9, "number": 9, "pos": "F", "grid": "1:1"}}],
                                    "substitutes": [],
                                }
                            ],
                        },
                        {"fixture": {"id": 102}},
                    ]
                }
            if endpoint == "/fixtures/lineups":
                return {
                    "response": [
                        {
                            "team": {"name": "Blackpool"},
                            "formation": "4-4-2",
                            "coach": {"id": 2, "name": "Coach"},
                            "startXI": [{"player": {"id": 10, "number": 10, "pos": "F", "grid": "1:1"}}],
                            "substitutes": [],
                        }
                    ]
                }
            raise AssertionError(endpoint)

        with patch("ingestion.api_football.api_get", side_effect=fake_api_get), patch(
            "ingestion.api_football.upsert_rows",
            side_effect=lambda _model, rows, _keys: len(rows),
        ):
            inserted = fetch_lineups(
                date(2026, 3, 10),
                date(2026, 3, 10),
                fixtures=fixtures,
            )

        self.assertEqual(inserted, 2)
        self.assertEqual([endpoint for endpoint, _params in api_calls], ["/fixtures", "/fixtures/lineups"])


if __name__ == "__main__":
    unittest.main()
