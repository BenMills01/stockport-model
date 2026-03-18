from __future__ import annotations

import unittest

from sqlalchemy import UniqueConstraint

from db.schema import Base


EXPECTED_TABLES = {
    "players",
    "match_performances",
    "fixtures",
    "fixture_team_stats",
    "match_events",
    "lineups",
    "standings_snapshots",
    "transfers",
    "sidelined",
    "injuries",
    "expected_metrics",
    "market_values",
    "market_value_history",
    "player_roles",
    "role_templates",
    "briefs",
    "pipeline",
    "scout_notes",
    "predictions_log",
    "overrides",
    "outcomes",
    "wyscout_zone_stats",
    "wyscout_season_stats",
    "source_player_mappings",
    "pathway_players",
    "skillcorner_player_map",
    "skillcorner_match_map",
    "skillcorner_physical",
    "skillcorner_off_ball_runs",
    "skillcorner_pressure",
    "skillcorner_passes",
}


class SchemaTests(unittest.TestCase):
    def test_all_required_tables_are_registered(self) -> None:
        self.assertEqual(EXPECTED_TABLES, set(Base.metadata.tables))

    def test_players_table_includes_current_age_fallback_column(self) -> None:
        players = Base.metadata.tables["players"]
        self.assertIn("current_age_years", players.columns)

    def test_pipeline_has_unique_player_brief_pair(self) -> None:
        table = Base.metadata.tables["pipeline"]
        unique_constraints = {
            tuple(column.name for column in constraint.columns)
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        }
        self.assertIn(("brief_id", "player_id"), unique_constraints)

    def test_jsonb_columns_cover_config_driven_entities(self) -> None:
        expected_jsonb_columns = {
            "role_templates": {"metrics_json"},
            "briefs": {"league_scope"},
            "scout_notes": {"video_urls"},
            "predictions_log": {"archetype_weights_used", "model_warnings", "component_fallbacks"},
            "overrides": {"original_model_output"},
            "wyscout_zone_stats": {"metrics_json"},
            "wyscout_season_stats": {"metrics_json"},
        }

        for table_name, expected_columns in expected_jsonb_columns.items():
            actual = {
                column.name
                for column in Base.metadata.tables[table_name].columns
                if column.type.__class__.__name__ == "JSONB"
            }
            self.assertTrue(expected_columns <= actual, table_name)


if __name__ == "__main__":
    unittest.main()
