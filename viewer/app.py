"""Local WSGI app for browsing enriched Stockport recruitment data."""

from __future__ import annotations

import argparse
from base64 import b64decode
from datetime import datetime
from http import HTTPStatus
import json
import os
import secrets
from typing import Any
from urllib.parse import parse_qs, quote_plus
import webbrowser
from wsgiref.simple_server import make_server

from config import get_settings
from outputs.common import render_template
from viewer.service import (
    apply_wyscout_review_mappings,
    apply_wyscout_review_mapping,
    create_brief_from_form,
    get_dashboard_context,
    get_brief_context,
    get_fixture_context,
    get_league_context,
    get_on_pitch_profiles_context,
    get_player_context,
    get_wyscout_review_context,
    get_brief_builder_context,
    render_brief_report,
    rerun_wyscout_review_import,
    run_brief_longlist,
)

_FORCE_READ_ONLY = False


def application(environ: dict[str, Any], start_response: Any) -> list[bytes]:
    """Serve a tiny local browser UI for the database and review flows."""

    path = environ.get("PATH_INFO", "/")
    query = parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=False)
    method = str(environ.get("REQUEST_METHOD", "GET")).upper()
    form = _parse_form_data(environ) if method == "POST" else {}
    viewer_read_only = _viewer_read_only()

    try:
        auth_credentials = _viewer_basic_auth_credentials()
        if path != "/health" and auth_credentials is not None and not _request_has_valid_basic_auth(
            environ,
            *auth_credentials,
        ):
            return _basic_auth_required_response(start_response)
        if viewer_read_only and method == "POST":
            return _error_response(
                start_response,
                HTTPStatus.FORBIDDEN,
                "Viewer is running in read-only mode. Create, rerun, and mapping actions are disabled.",
            )
        if path == "/":
            message = _parse_optional_text(query, "message")
            return _html_response(
                start_response,
                _render_view("data_browser_home.html", **{**get_dashboard_context(), "message": message}),
            )
        if path == "/on-pitch":
            role_name = _parse_optional_text(query, "role_name")
            season = _parse_optional_text(query, "season")
            return _html_response(
                start_response,
                _render_view(
                    "data_browser_on_pitch.html",
                    **get_on_pitch_profiles_context(role_name=role_name, season=season),
                ),
            )
        if path == "/briefs/create":
            if method != "POST":
                return _error_response(start_response, HTTPStatus.METHOD_NOT_ALLOWED, "POST required")
            try:
                result = create_brief_from_form(form)
            except ValueError as exc:
                context = {
                    **get_dashboard_context(),
                    "message": None,
                    "brief_builder": get_brief_builder_context(
                        form_values=_normalise_brief_form_values(form),
                        errors=[str(exc)],
                    ),
                }
                return _html_response(start_response, _render_view("data_browser_home.html", **context), status=HTTPStatus.BAD_REQUEST)

            brief_id = int(result["brief_id"])
            if result["action"] == "create_run":
                summary = run_brief_longlist(brief_id)
                message = (
                    f"Created brief {brief_id} and ran the longlist in "
                    f"{summary['duration_seconds']:.2f}s with {summary['row_count']} row(s)."
                )
                return _redirect_response(start_response, f"/brief/{brief_id}?message={quote_plus(message)}")
            return _redirect_response(
                start_response,
                f"/brief/{brief_id}?message={quote_plus(f'Created brief {brief_id}.')}",
            )
        if path.startswith("/brief/") and path.endswith("/run"):
            if method != "POST":
                return _error_response(start_response, HTTPStatus.METHOD_NOT_ALLOWED, "POST required")
            brief_id = _parse_path_int(path, "/brief/", suffix="/run")
            summary = run_brief_longlist(brief_id)
            message = (
                f"Ran longlist in {summary['duration_seconds']:.2f}s with "
                f"{summary['row_count']} row(s)."
            )
            return _redirect_response(start_response, f"/brief/{brief_id}?message={quote_plus(message)}")
        if path.startswith("/brief/") and path.endswith("/report"):
            brief_id = _parse_path_int(path, "/brief/", suffix="/report")
            return _html_response(start_response, render_brief_report(brief_id))
        if path.startswith("/brief/"):
            brief_id = _parse_path_int(path, "/brief/")
            message = _parse_optional_text(query, "message")
            context = get_brief_context(brief_id, message=message)
            if context is None:
                return _error_response(start_response, HTTPStatus.NOT_FOUND, "Brief not found")
            return _html_response(start_response, _render_view("data_browser_brief.html", **context))
        if path == "/wyscout-review":
            league_id = _parse_optional_int(query, "league_id")
            season = _parse_optional_text(query, "season")
            page = _parse_optional_int(query, "page") or 1
            message = _parse_optional_text(query, "message")
            context = get_wyscout_review_context(
                league_id=league_id,
                season=season,
                page=page,
                message=message,
            )
            return _html_response(start_response, _render_view("data_browser_wyscout_review.html", **context))
        if path == "/wyscout-review/apply":
            if method != "POST":
                return _error_response(start_response, HTTPStatus.METHOD_NOT_ALLOWED, "POST required")
            league_id = _parse_optional_int(form, "league_id")
            season = _parse_optional_text(form, "season")
            page = _parse_optional_int(form, "page") or 1
            selected_player_id = _parse_required_int(form, "player_id")
            result = apply_wyscout_review_mapping(
                source_player_name=_parse_required_text(form, "source_player_name"),
                source_team_name=_parse_optional_text(form, "source_team_name"),
                player_id=selected_player_id,
                league_id=league_id,
                match_score=_parse_optional_float(form, "match_score"),
                source_player_external_id=_parse_optional_text(form, "source_player_external_id"),
                rerun=True,
            )
            summary = result.get("rerun_summary") or {}
            message = (
                f"Saved mapping to player {selected_player_id}. "
                f"Reran {summary.get('folders_processed', 0)} folder(s) and now have "
                f"{summary.get('unmatched_rows', 0)} unmatched row(s)."
            )
            return _redirect_response(
                start_response,
                _build_wyscout_review_url(
                    league_id=league_id,
                    season=season,
                    page=page,
                    message=message,
                ),
            )
        if path == "/wyscout-review/apply-batch":
            if method != "POST":
                return _error_response(start_response, HTTPStatus.METHOD_NOT_ALLOWED, "POST required")
            league_id = _parse_optional_int(form, "league_id")
            season = _parse_optional_text(form, "season")
            page = _parse_optional_int(form, "page") or 1
            raw_selections = form.get("selected_match") or []
            if not raw_selections:
                return _redirect_response(
                    start_response,
                    _build_wyscout_review_url(
                        league_id=league_id,
                        season=season,
                        page=page,
                        message="Select at least one suggested match before running the batch.",
                    ),
                )
            selections = [json.loads(value) for value in raw_selections]
            result = apply_wyscout_review_mappings(selections, rerun=True)
            summary = result.get("rerun_summary") or {}
            message = (
                f"Saved {result.get('saved_count', 0)} mapping(s). "
                f"Reran {summary.get('folders_processed', 0)} folder(s) and now have "
                f"{summary.get('unmatched_rows', 0)} unmatched row(s)."
            )
            return _redirect_response(
                start_response,
                _build_wyscout_review_url(
                    league_id=league_id,
                    season=season,
                    page=page,
                    message=message,
                ),
            )
        if path == "/wyscout-review/reimport":
            if method != "POST":
                return _error_response(start_response, HTTPStatus.METHOD_NOT_ALLOWED, "POST required")
            league_id = _parse_optional_int(form, "league_id")
            season = _parse_optional_text(form, "season")
            page = _parse_optional_int(form, "page") or 1
            summary = rerun_wyscout_review_import(league_id=league_id)
            scope = f"league {league_id}" if league_id is not None else "all tracked folders"
            message = (
                f"Reran {scope}: {summary.get('folders_processed', 0)} folder(s), "
                f"{summary.get('imported_rows', 0)} imported row(s), "
                f"{summary.get('unmatched_rows', 0)} unmatched row(s) remaining."
            )
            return _redirect_response(
                start_response,
                _build_wyscout_review_url(
                    league_id=league_id,
                    season=season,
                    page=page,
                    message=message,
                ),
            )
        if path == "/league":
            league_id = _parse_required_int(query, "league_id")
            season = _parse_optional_text(query, "season")
            context = get_league_context(league_id, season)
            if context is None:
                return _error_response(start_response, HTTPStatus.NOT_FOUND, "League view not found")
            return _html_response(start_response, _render_view("data_browser_league.html", **context))
        if path.startswith("/fixture/"):
            fixture_id = _parse_path_int(path, "/fixture/")
            context = get_fixture_context(fixture_id)
            if context is None:
                return _error_response(start_response, HTTPStatus.NOT_FOUND, "Fixture not found")
            return _html_response(start_response, _render_view("data_browser_fixture.html", **context))
        if path.startswith("/player/"):
            player_id = _parse_path_int(path, "/player/")
            context = get_player_context(player_id)
            if context is None:
                return _error_response(start_response, HTTPStatus.NOT_FOUND, "Player not found")
            return _html_response(start_response, _render_view("data_browser_player.html", **context))
        if path == "/health":
            return _plain_response(start_response, HTTPStatus.OK, "ok\n")
    except ValueError as exc:
        return _error_response(start_response, HTTPStatus.BAD_REQUEST, str(exc))
    except Exception as exc:  # pragma: no cover - handled in live runtime
        return _error_response(start_response, HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    return _error_response(start_response, HTTPStatus.NOT_FOUND, "Page not found")


def main(argv: list[str] | None = None) -> int:
    """Run the local browser app."""

    parser = argparse.ArgumentParser(description="Launch the Stockport data browser.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args(argv)

    global _FORCE_READ_ONLY
    _FORCE_READ_ONLY = bool(args.read_only)
    if _FORCE_READ_ONLY:
        os.environ["STOCKPORT_VIEWER_READ_ONLY"] = "true"

    url = f"http://{args.host}:{args.port}/"
    if args.open_browser:
        webbrowser.open(url)

    with make_server(args.host, args.port, application) as server:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] Stockport data viewer running at {url}")
        server.serve_forever()
    return 0


def _html_response(start_response: Any, body: str, status: HTTPStatus = HTTPStatus.OK) -> list[bytes]:
    payload = body.encode("utf-8")
    start_response(
        f"{status.value} {status.phrase}",
        [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(payload))),
        ],
    )
    return [payload]


def _plain_response(start_response: Any, status: HTTPStatus, body: str) -> list[bytes]:
    payload = body.encode("utf-8")
    start_response(
        f"{status.value} {status.phrase}",
        [
            ("Content-Type", "text/plain; charset=utf-8"),
            ("Content-Length", str(len(payload))),
        ],
    )
    return [payload]


def _redirect_response(start_response: Any, location: str) -> list[bytes]:
    start_response("303 See Other", [("Location", location), ("Content-Length", "0")])
    return [b""]


def _error_response(start_response: Any, status: HTTPStatus, message: str) -> list[bytes]:
    body = _render_view(
        "data_browser_error.html",
        title=f"{status.value} {status.phrase}",
        status=status,
        message=message,
        generated_at=datetime.now(),
    )
    return _html_response(start_response, body, status=status)


def _render_view(template_name: str, **context: Any) -> str:
    return render_template(
        template_name,
        **{
            **context,
            "viewer_read_only": _viewer_read_only(),
        },
    )


def _viewer_read_only() -> bool:
    return bool(_FORCE_READ_ONLY or get_settings().viewer_read_only)


def _viewer_basic_auth_credentials() -> tuple[str, str] | None:
    settings = get_settings()
    user = (settings.viewer_basic_auth_user or "").strip()
    password = settings.viewer_basic_auth_password or ""
    if not user or not password:
        return None
    return user, password


def _request_has_valid_basic_auth(environ: dict[str, Any], username: str, password: str) -> bool:
    auth_header = str(environ.get("HTTP_AUTHORIZATION") or "")
    if not auth_header.startswith("Basic "):
        return False
    encoded = auth_header[6:].strip()
    if not encoded:
        return False
    try:
        decoded = b64decode(encoded).decode("utf-8")
    except Exception:
        return False
    if ":" not in decoded:
        return False
    supplied_user, supplied_password = decoded.split(":", 1)
    return secrets.compare_digest(supplied_user, username) and secrets.compare_digest(
        supplied_password,
        password,
    )


def _basic_auth_required_response(start_response: Any) -> list[bytes]:
    payload = b"Authentication required\n"
    start_response(
        "401 Unauthorized",
        [
            ("Content-Type", "text/plain; charset=utf-8"),
            ("Content-Length", str(len(payload))),
            ("WWW-Authenticate", 'Basic realm="Stockport Viewer"'),
        ],
    )
    return [payload]


def _parse_required_int(query: dict[str, list[str]], key: str) -> int:
    value = _parse_optional_text(query, key)
    if value is None:
        raise ValueError(f"Missing required query parameter: {key}")
    return int(value)


def _parse_required_text(query: dict[str, list[str]], key: str) -> str:
    value = _parse_optional_text(query, key)
    if value is None:
        raise ValueError(f"Missing required query parameter: {key}")
    return value


def _parse_optional_int(query: dict[str, list[str]], key: str) -> int | None:
    value = _parse_optional_text(query, key)
    if value is None:
        return None
    return int(value)


def _parse_optional_float(query: dict[str, list[str]], key: str) -> float | None:
    value = _parse_optional_text(query, key)
    if value is None:
        return None
    return float(value)


def _parse_optional_text(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key) or []
    if not values:
        return None
    value = values[0].strip()
    return value or None


def _parse_form_data(environ: dict[str, Any]) -> dict[str, list[str]]:
    try:
        content_length = int(environ.get("CONTENT_LENGTH") or 0)
    except (TypeError, ValueError):
        content_length = 0
    raw_body = environ.get("wsgi.input").read(content_length) if content_length else b""
    return parse_qs(raw_body.decode("utf-8"), keep_blank_values=False)


def _build_wyscout_review_url(
    *,
    league_id: int | None,
    season: str | None,
    page: int | None,
    message: str | None = None,
) -> str:
    parts = []
    if league_id is not None:
        parts.append(f"league_id={league_id}")
    if season:
        parts.append(f"season={quote_plus(season)}")
    if page and page > 1:
        parts.append(f"page={page}")
    if message:
        parts.append(f"message={quote_plus(message)}")
    query = "&".join(parts)
    if not query:
        return "/wyscout-review"
    return f"/wyscout-review?{query}"


def _parse_path_int(path: str, prefix: str, suffix: str | None = None) -> int:
    if not path.startswith(prefix):
        raise ValueError("Invalid path")
    tail = path[len(prefix) :]
    if suffix and tail.endswith(suffix):
        tail = tail[: -len(suffix)]
    tail = tail.strip("/")
    if not tail:
        raise ValueError("Missing identifier")
    return int(tail)


def _normalise_brief_form_values(form: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "role_name": _parse_optional_text(form, "role_name") or "",
        "archetype_primary": _parse_optional_text(form, "archetype_primary") or "",
        "archetype_secondary": _parse_optional_text(form, "archetype_secondary") or "",
        "intent": _parse_optional_text(form, "intent") or "",
        "budget_max_fee": _parse_optional_text(form, "budget_max_fee") or "",
        "budget_max_wage": _parse_optional_text(form, "budget_max_wage") or "",
        "budget_max_contract_years": _parse_optional_text(form, "budget_max_contract_years") or "",
        "age_min": _parse_optional_text(form, "age_min") or "",
        "age_max": _parse_optional_text(form, "age_max") or "",
        "timeline": _parse_optional_text(form, "timeline") or "",
        "league_scope": [int(value) for value in form.get("league_scope", []) if str(value).strip()],
        "created_by": _parse_optional_text(form, "created_by") or "",
        "approved_by": _parse_optional_text(form, "approved_by") or "",
        "pathway_check_done": bool(form.get("pathway_check_done")),
        "pathway_player_id": _parse_optional_text(form, "pathway_player_id") or "",
    }


if __name__ == "__main__":  # pragma: no cover - exercised manually
    raise SystemExit(main())
