"""Shared output rendering helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from config import get_settings


@lru_cache(maxsize=1)
def get_template_environment() -> Environment:
    """Return a cached Jinja environment rooted at the templates directory."""

    settings = get_settings()
    return Environment(
        loader=FileSystemLoader(str(settings.templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_template(template_name: str, **context: Any) -> str:
    """Render a named Jinja template into HTML."""

    environment = get_template_environment()
    template = environment.get_template(template_name)
    return template.render(**context)
