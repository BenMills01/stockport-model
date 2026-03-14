from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from db.schema import RoleTemplate
from db.seed_reference_data import seed_reference_data, sync_role_templates, verify_role_templates


class FakeSession:
    def __init__(self, rows: list[RoleTemplate]) -> None:
        self.rows = rows
        self.added: list[RoleTemplate] = []

    def scalars(self, _statement: object) -> list[RoleTemplate]:
        return list(self.rows)

    def add(self, row: RoleTemplate) -> None:
        self.rows.append(row)
        self.added.append(row)


class SeedReferenceDataTests(unittest.TestCase):
    def test_sync_role_templates_inserts_and_deactivates_old_versions(self) -> None:
        session = FakeSession(
            [
                RoleTemplate(
                    role_name="controller",
                    version="2025-Q4",
                    created_by="old-seed",
                    metrics_json={"passes_total": 1.0},
                    is_active=True,
                )
            ]
        )

        summary = sync_role_templates(
            session,
            [
                {
                    "role_name": "controller",
                    "version": "2026-Q1",
                    "metrics": {"passes_total": 0.6, "passes_key": 0.4},
                },
                {
                    "role_name": "ball_winning_6",
                    "version": "2026-Q1",
                    "metrics": {"tackles_total": 0.5, "duels_won": 0.5},
                },
            ],
        )

        self.assertEqual(summary["loaded_templates"], 2)
        self.assertEqual(summary["inserted"], 2)
        self.assertEqual(summary["updated"], 0)
        self.assertEqual(summary["deactivated"], 1)
        self.assertFalse(session.rows[0].is_active)
        self.assertEqual(len(session.added), 2)

    def test_sync_role_templates_updates_existing_current_version(self) -> None:
        existing = RoleTemplate(
            role_name="controller",
            version="2026-Q1",
            created_by=None,
            metrics_json={"passes_total": 1.0},
            is_active=False,
        )
        session = FakeSession([existing])

        summary = sync_role_templates(
            session,
            [
                {
                    "role_name": "controller",
                    "version": "2026-Q1",
                    "metrics": {"passes_total": 0.7, "passes_key": 0.3},
                }
            ],
        )

        self.assertEqual(summary["inserted"], 0)
        self.assertEqual(summary["updated"], 1)
        self.assertEqual(existing.metrics_json, {"passes_total": 0.7, "passes_key": 0.3})
        self.assertTrue(existing.is_active)
        self.assertEqual(existing.created_by, "config_seed")

    def test_seed_reference_data_loads_config_and_returns_summary(self) -> None:
        fake_session = FakeSession([])

        @contextmanager
        def fake_session_scope(_database_url: str | None = None) -> object:
            yield fake_session

        with patch(
            "db.seed_reference_data.get_settings",
            return_value=SimpleNamespace(
                load_json=lambda filename: [
                    {
                        "role_name": "controller",
                        "version": "2026-Q1",
                        "metrics": {"passes_total": 1.0},
                    }
                ]
                if filename == "role_templates.json"
                else None
            ),
        ), patch("db.seed_reference_data.session_scope", fake_session_scope):
            summary = seed_reference_data()

        self.assertEqual(summary["role_templates"]["loaded_templates"], 1)
        self.assertEqual(summary["role_templates"]["inserted"], 1)


class VerifyRoleTemplatesTests(unittest.TestCase):
    def _make_settings(self, config_templates: list[dict]) -> object:
        return SimpleNamespace(
            load_json=lambda filename: config_templates
            if filename == "role_templates.json"
            else []
        )

    def _make_db_rows(self, specs: list[tuple[str, str, bool]]) -> list[RoleTemplate]:
        return [
            RoleTemplate(role_name=rn, version=v, is_active=active, metrics_json={}, created_by="test")
            for rn, v, active in specs
        ]

    @contextmanager
    def _fake_scope(self, rows: list[RoleTemplate]):
        @contextmanager
        def _inner(_db_url=None):
            yield SimpleNamespace(
                scalars=lambda _stmt, **kw: rows
            )
        return _inner()

    def test_all_templates_present_returns_ok(self) -> None:
        config = [
            {"role_name": "controller", "version": "2026-Q1"},
            {"role_name": "ball_playing_cb", "version": "2026-Q1"},
        ]
        db_rows = self._make_db_rows([
            ("controller", "2026-Q1", True),
            ("ball_playing_cb", "2026-Q1", True),
        ])

        @contextmanager
        def fake_scope(_db_url=None):
            yield SimpleNamespace(scalars=lambda _stmt, **kw: db_rows)

        with patch("db.seed_reference_data.get_settings", return_value=self._make_settings(config)), \
             patch("db.seed_reference_data.session_scope", fake_scope):
            result = verify_role_templates()

        self.assertTrue(result["ok"])
        self.assertEqual(result["missing"], [])
        self.assertEqual(result["extra_active"], [])
        self.assertEqual(result["expected_count"], 2)

    def test_missing_template_detected(self) -> None:
        config = [
            {"role_name": "controller", "version": "2026-Q1"},
            {"role_name": "ball_playing_cb", "version": "2026-Q1"},
        ]
        db_rows = self._make_db_rows([
            ("controller", "2026-Q1", True),
            # ball_playing_cb is absent
        ])

        @contextmanager
        def fake_scope(_db_url=None):
            yield SimpleNamespace(scalars=lambda _stmt, **kw: db_rows)

        with patch("db.seed_reference_data.get_settings", return_value=self._make_settings(config)), \
             patch("db.seed_reference_data.session_scope", fake_scope):
            result = verify_role_templates()

        self.assertFalse(result["ok"])
        self.assertEqual(len(result["missing"]), 1)
        self.assertEqual(result["missing"][0]["role_name"], "ball_playing_cb")
        self.assertEqual(result["extra_active"], [])

    def test_inactive_db_row_treated_as_missing(self) -> None:
        config = [{"role_name": "controller", "version": "2026-Q1"}]
        # Row exists in DB but is_active=False
        db_rows = self._make_db_rows([("controller", "2026-Q1", False)])

        @contextmanager
        def fake_scope(_db_url=None):
            # Only active rows are returned by the query filter
            yield SimpleNamespace(scalars=lambda _stmt, **kw: [])

        with patch("db.seed_reference_data.get_settings", return_value=self._make_settings(config)), \
             patch("db.seed_reference_data.session_scope", fake_scope):
            result = verify_role_templates()

        self.assertFalse(result["ok"])
        self.assertEqual(len(result["missing"]), 1)

    def test_extra_active_db_row_detected(self) -> None:
        config = [{"role_name": "controller", "version": "2026-Q1"}]
        db_rows = self._make_db_rows([
            ("controller", "2026-Q1", True),
            ("old_role", "2025-Q4", True),  # no longer in config
        ])

        @contextmanager
        def fake_scope(_db_url=None):
            yield SimpleNamespace(scalars=lambda _stmt, **kw: db_rows)

        with patch("db.seed_reference_data.get_settings", return_value=self._make_settings(config)), \
             patch("db.seed_reference_data.session_scope", fake_scope):
            result = verify_role_templates()

        self.assertFalse(result["ok"])
        self.assertEqual(result["missing"], [])
        self.assertEqual(len(result["extra_active"]), 1)
        self.assertEqual(result["extra_active"][0]["role_name"], "old_role")


if __name__ == "__main__":
    unittest.main()
