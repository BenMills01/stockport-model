"""Seed config-backed reference data into the live database."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import get_settings
from db.schema import RoleTemplate
from db.session import session_scope


def sync_role_templates(
    session: Session,
    template_payloads: list[dict[str, Any]],
    *,
    created_by: str = "config_seed",
) -> dict[str, int]:
    """Upsert role templates from config and retire superseded versions."""

    existing_rows = list(session.scalars(select(RoleTemplate)))
    existing = {(row.role_name, row.version): row for row in existing_rows}
    current_versions_by_role: dict[str, set[str]] = {}

    inserted = 0
    updated = 0
    deactivated = 0

    for payload in template_payloads:
        role_name = str(payload["role_name"])
        version = str(payload["version"])
        metrics = dict(payload["metrics"])
        current_versions_by_role.setdefault(role_name, set()).add(version)

        row = existing.get((role_name, version))
        if row is None:
            session.add(
                RoleTemplate(
                    role_name=role_name,
                    version=version,
                    created_by=created_by,
                    metrics_json=metrics,
                    is_active=True,
                )
            )
            inserted += 1
            continue

        changed = False
        if row.metrics_json != metrics:
            row.metrics_json = metrics
            changed = True
        if not row.is_active:
            row.is_active = True
            changed = True
        if not row.created_by:
            row.created_by = created_by
            changed = True
        if changed:
            updated += 1

    for row in existing_rows:
        live_versions = current_versions_by_role.get(row.role_name)
        if live_versions and row.version not in live_versions and row.is_active:
            row.is_active = False
            deactivated += 1

    return {
        "loaded_templates": len(template_payloads),
        "inserted": inserted,
        "updated": updated,
        "deactivated": deactivated,
    }


def verify_role_templates(
    database_url: str | None = None,
) -> dict[str, Any]:
    """Check that all config-defined role templates are active in the DB.

    Returns a summary dict with keys:
    - ``ok``: True when config and DB are fully in sync.
    - ``missing``: list of (role_name, version) tuples defined in config but
      absent or inactive in the DB.
    - ``extra_active``: list of (role_name, version) tuples active in the DB
      but not present in the current config file.
    """
    settings = get_settings()
    template_payloads = list(settings.load_json("role_templates.json"))
    expected = {(str(p["role_name"]), str(p["version"])) for p in template_payloads}

    with session_scope(database_url) as session:
        db_rows = list(session.scalars(select(RoleTemplate).where(RoleTemplate.is_active == True)))  # noqa: E712

    active_in_db = {(row.role_name, row.version) for row in db_rows}

    missing = sorted(expected - active_in_db)
    extra_active = sorted(active_in_db - expected)

    return {
        "ok": not missing and not extra_active,
        "expected_count": len(expected),
        "active_in_db_count": len(active_in_db),
        "missing": [{"role_name": r, "version": v} for r, v in missing],
        "extra_active": [{"role_name": r, "version": v} for r, v in extra_active],
    }


def seed_reference_data(
    database_url: str | None = None,
    *,
    created_by: str = "config_seed",
) -> dict[str, dict[str, int]]:
    """Sync config-backed reference data into the configured database."""

    settings = get_settings()
    role_templates = settings.load_json("role_templates.json")

    with session_scope(database_url) as session:
        role_template_summary = sync_role_templates(
            session,
            list(role_templates),
            created_by=created_by,
        )

    return {"role_templates": role_template_summary}


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Seed or verify reference data.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check that DB matches config without writing any changes.",
    )
    args = parser.parse_args()

    if args.verify:
        result = verify_role_templates()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)
    else:
        print(json.dumps(seed_reference_data(), indent=2))
