from __future__ import annotations

from math import isclose
from pathlib import Path
import tempfile
import unittest

from config import get_settings
from config.settings import load_env_file, normalise_database_url


class ConfigAssetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = get_settings()

    def test_config_files_load(self) -> None:
        for filename in (
            "leagues.json",
            "role_templates.json",
            "archetype_weights.json",
            "composite_action_tiers.json",
            "club_profile.json",
            "hard_gates.json",
            "age_curves.json",
            "role_profile_rules.json",
            "score_calibration.json",
            "projection_heuristics.json",
            "on_pitch_dashboard.json",
        ):
            payload = self.settings.load_json(filename)
            self.assertTrue(payload, filename)

    def test_archetype_weights_sum_to_100(self) -> None:
        payload = self.settings.load_json("archetype_weights.json")

        for archetype, config in payload.items():
            total = sum(config["weights_pct"].values())
            self.assertEqual(total, 100, archetype)

    def test_role_templates_are_normalised(self) -> None:
        templates = self.settings.load_json("role_templates.json")

        for template in templates:
            weights = template["metrics"].values()
            self.assertGreaterEqual(len(template["metrics"]), 8)
            self.assertLessEqual(len(template["metrics"]), 12)
            self.assertTrue(
                isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9),
                template["role_name"],
            )

    def test_age_curve_payloads_have_required_keys(self) -> None:
        curves = self.settings.load_json("age_curves.json")
        required_keys = {
            "physical_peak",
            "output_peak",
            "decline_onset",
            "resale_window",
            "acceptable_older_signing_age",
        }

        for position_group, payload in curves.items():
            self.assertEqual(required_keys, set(payload), position_group)

    def test_role_profile_rules_have_known_threshold_keys(self) -> None:
        payload = self.settings.load_json("role_profile_rules.json")
        allowed_keys = {
            "minimum_height_cm",
            "minimum_forward_label_share",
            "maximum_midfielder_label_share",
            "minimum_central_attack_share",
            "maximum_wide_attack_share",
        }
        for role_name, rules in payload.items():
            self.assertTrue(set(rules).issubset(allowed_keys), role_name)

    def test_league_scope_includes_requested_expansion(self) -> None:
        leagues = self.settings.load_json("leagues.json")
        league_ids = {int(row["league_id"]) for row in leagues}

        self.assertEqual(len(league_ids), len(leagues))
        self.assertTrue({62, 79, 89, 144, 145, 179}.issubset(league_ids))

    def test_project_paths_resolve_inside_workspace(self) -> None:
        self.assertEqual(self.settings.project_root.name, "Stockport_Model")
        self.assertEqual(self.settings.data_dir, self.settings.project_root / "data")
        self.assertEqual(
            self.settings.templates_dir,
            self.settings.project_root / "templates",
        )

    def test_dotenv_loader_respects_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dotenv_path = Path(tmpdir) / ".env"
            dotenv_path.write_text(
                "\n".join(
                    (
                        "# comment",
                        "STOCKPORT_DATABASE_URL=postgresql+psycopg://dotenv-user@localhost/db",
                        'API_FOOTBALL_API_KEY="quoted-key"',
                        "INVALID_LINE",
                    )
                ),
                encoding="utf-8",
            )
            environ = {"STOCKPORT_DATABASE_URL": "postgresql+psycopg://shell-user@localhost/db"}

            load_env_file(dotenv_path, environ)

            self.assertEqual(
                environ["STOCKPORT_DATABASE_URL"],
                "postgresql+psycopg://shell-user@localhost/db",
            )
            self.assertEqual(environ["API_FOOTBALL_API_KEY"], "quoted-key")
            self.assertNotIn("INVALID_LINE", environ)

    def test_dotenv_loader_ignores_missing_file(self) -> None:
        environ: dict[str, str] = {}

        load_env_file(Path("/tmp/stockport-missing-dotenv"), environ)

        self.assertEqual(environ, {})

    def test_database_url_normalisation_supports_hosted_postgres_urls(self) -> None:
        self.assertEqual(
            normalise_database_url("postgres://user:pass@host:5432/dbname"),
            "postgresql+psycopg://user:pass@host:5432/dbname",
        )
        self.assertEqual(
            normalise_database_url("postgresql://user:pass@host:5432/dbname"),
            "postgresql+psycopg://user:pass@host:5432/dbname",
        )
        self.assertEqual(
            normalise_database_url("postgresql+psycopg://user:pass@host:5432/dbname"),
            "postgresql+psycopg://user:pass@host:5432/dbname",
        )


if __name__ == "__main__":
    unittest.main()
