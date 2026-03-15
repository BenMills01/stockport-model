"""Build a static GitHub Pages snapshot from the live local database."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
PROFILES_DIR = DOCS_DIR / "profiles"
REPORTS_DIR = DOCS_DIR / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from outputs.common import render_template
from viewer.service import _on_pitch_role_options, _on_pitch_season_options, get_on_pitch_profiles_context


def build_pages_site() -> dict[str, Any]:
    """Generate a static site snapshot under docs/ for GitHub Pages."""

    selected_season = _on_pitch_season_options()[0]
    role_options = _on_pitch_role_options()
    generated_at = datetime.now()

    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    profile_rows: list[dict[str, Any]] = []
    for role_name, role_family in role_options:
        context = get_on_pitch_profiles_context(role_name=role_name, season=selected_season)
        filename = f"{role_name}.html"
        profile_rows.append(
            {
                "role_name": role_name,
                "role_family": role_family,
                "href": f"profiles/{filename}",
                "candidate_count": context["candidate_count"],
                "scored_count": context["scored_count"],
                "fallback_note": context.get("candidate_match_note"),
                "top_player": (context.get("top_players") or [None])[0],
            }
        )
        profile_html = render_template(
            "github_pages_profile.html",
            **{
                **context,
                "title": f"{role_name} · On-Pitch Snapshot",
                "site_generated_at": generated_at,
                "site_home_href": "../index.html",
                "site_reports_href": "../reports/index.html",
                "all_profile_links": [
                    {
                        "role_name": candidate_role,
                        "role_family": candidate_family,
                        "href": f"{candidate_role}.html",
                        "active": candidate_role == role_name,
                    }
                    for candidate_role, candidate_family in role_options
                ],
            },
        )
        (PROFILES_DIR / filename).write_text(profile_html, encoding="utf-8")

    report_rows = _copy_report_artifacts()
    reports_index_html = render_template(
        "github_pages_reports.html",
        title="Brief Reports",
        generated_at=generated_at,
        reports=report_rows,
        site_home_href="../index.html",
    )
    (REPORTS_DIR / "index.html").write_text(reports_index_html, encoding="utf-8")

    index_html = render_template(
        "github_pages_index.html",
        title="Stockport Recruitment Snapshot",
        generated_at=generated_at,
        selected_season=selected_season,
        profiles=profile_rows,
        reports=report_rows[:8],
        reports_href="reports/index.html",
    )
    (DOCS_DIR / "index.html").write_text(index_html, encoding="utf-8")
    (DOCS_DIR / "404.html").write_text(index_html, encoding="utf-8")
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")

    return {
        "generated_at": generated_at,
        "selected_season": selected_season,
        "profile_count": len(profile_rows),
        "report_count": len(report_rows),
    }


def _copy_report_artifacts() -> list[dict[str, Any]]:
    report_rows: list[dict[str, Any]] = []
    for artifact in sorted(ARTIFACTS_DIR.glob("longlist_brief_*.html")):
        target = REPORTS_DIR / artifact.name
        shutil.copy2(artifact, target)
        report_rows.append(
            {
                "brief_id": _extract_brief_id(artifact.name),
                "filename": artifact.name,
                "href": artifact.name,
                "label": artifact.stem.replace("_", " ").title(),
                "updated_at": datetime.fromtimestamp(artifact.stat().st_mtime),
            }
        )
    return report_rows


def _extract_brief_id(filename: str) -> int | None:
    stem = Path(filename).stem
    if not stem.startswith("longlist_brief_"):
        return None
    suffix = stem.removeprefix("longlist_brief_")
    return int(suffix) if suffix.isdigit() else None


def main() -> int:
    summary = build_pages_site()
    print(
        "Built GitHub Pages snapshot for "
        f"{summary['profile_count']} profiles and {summary['report_count']} reports "
        f"(season {summary['selected_season']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
