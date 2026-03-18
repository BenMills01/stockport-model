#!/usr/bin/env python3
"""Build a branded HTML and PDF handout from the manager-facing markdown note."""

from __future__ import annotations

import argparse
import html
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "artifacts" / "on_pitch_methodology_manager_note.md"
DEFAULT_HTML = ROOT / "artifacts" / "on_pitch_methodology_manager_note.html"
DEFAULT_PDF = ROOT / "artifacts" / "on_pitch_methodology_manager_note.pdf"
TEMPLATE_NAME = "manager_note.html"
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chrome",
    "chromium",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--prepared-by", default="Ben Mills")
    parser.add_argument("--prepared-for", default="Stockport County")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    html_path = args.html.resolve()
    pdf_path = args.pdf.resolve()

    markdown_text = source.read_text(encoding="utf-8")
    body_html, toc, title = render_markdown(markdown_text)

    env = Environment(
        loader=FileSystemLoader(str(ROOT / "templates")),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(TEMPLATE_NAME)
    document_title = "On-Pitch Section Methodology"

    html_output = template.render(
        title=document_title,
        subtitle=(
            "A manager-facing explanation of how the football-only ranking layer currently works, "
            "with emphasis on candidate selection, score construction, and how to interpret the outputs."
        ),
        prepared_by=args.prepared_by,
        prepared_for=args.prepared_for,
        generated_at=datetime.now().strftime("%d %B %Y"),
        toc=toc,
        body_html=body_html,
        summary_cards=[
            {"label": "Primary Purpose", "value": "Football-first discovery and longlisting for a chosen role profile."},
            {"label": "Main Inputs", "value": "Role fit, current performance, Championship projection, and physical data where available."},
            {"label": "Main Output", "value": "On-Pitch, Technical, Physical, Present, and Upside rankings by profile and by league."},
            {"label": "Key Safeguard", "value": "Low-sample protection through shrinkage and a softer minutes-evidence multiplier."},
        ],
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_output, encoding="utf-8")

    chrome = find_chrome()
    if chrome is None:
        raise RuntimeError("Google Chrome was not found, so the PDF could not be generated.")

    render_pdf(chrome, html_path, pdf_path)
    return 0


def render_markdown(markdown_text: str) -> tuple[str, list[dict[str, str]], str]:
    toc: list[dict[str, str]] = []
    html_parts: list[str] = []
    paragraph_lines: list[str] = []
    list_stack: list[str] = []
    blockquote_lines: list[str] = []
    code_lines: list[str] = []
    in_code = False
    title = "On-Pitch Section Methodology"

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        text = " ".join(line.strip() for line in paragraph_lines)
        html_parts.append(f"<p>{inline_markup(text)}</p>")
        paragraph_lines = []

    def flush_lists() -> None:
        nonlocal list_stack
        while list_stack:
            html_parts.append(f"</{list_stack.pop()}>")

    def flush_blockquote() -> None:
        nonlocal blockquote_lines
        if not blockquote_lines:
            return
        quote = " ".join(line.strip() for line in blockquote_lines)
        html_parts.append(f"<blockquote>{inline_markup(quote)}</blockquote>")
        blockquote_lines = []

    def open_list(kind: str) -> None:
        if list_stack and list_stack[-1] == kind:
            return
        flush_paragraph()
        flush_blockquote()
        flush_lists()
        html_parts.append(f"<{kind}>")
        list_stack.append(kind)

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            flush_blockquote()
            flush_lists()
            if in_code:
                html_parts.append("<pre><code>{}</code></pre>".format(html.escape("\n".join(code_lines))))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            flush_paragraph()
            flush_blockquote()
            flush_lists()
            continue

        heading_match = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph()
            flush_blockquote()
            flush_lists()
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_id = slugify(heading_text)
            if level == 1:
                title = heading_text
            if level == 2:
                toc.append({"id": heading_id, "title": heading_text})
            html_parts.append(f"<h{level} id=\"{heading_id}\">{inline_markup(heading_text)}</h{level}>")
            continue

        if stripped.startswith("> "):
            flush_paragraph()
            flush_lists()
            blockquote_lines.append(stripped[2:])
            continue

        ordered_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if ordered_match:
            flush_paragraph()
            flush_blockquote()
            open_list("ol")
            html_parts.append(f"<li>{inline_markup(ordered_match.group(1))}</li>")
            continue

        unordered_match = re.match(r"^-\s+(.*)$", stripped)
        if unordered_match:
            flush_paragraph()
            flush_blockquote()
            open_list("ul")
            html_parts.append(f"<li>{inline_markup(unordered_match.group(1))}</li>")
            continue

        flush_blockquote()
        paragraph_lines.append(stripped)

    flush_paragraph()
    flush_blockquote()
    flush_lists()
    if in_code:
        html_parts.append("<pre><code>{}</code></pre>".format(html.escape("\n".join(code_lines))))

    return "\n".join(html_parts), toc, title


def inline_markup(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def find_chrome() -> str | None:
    for candidate in CHROME_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return str(path)
        resolved = shutil_which(candidate)
        if resolved:
            return resolved
    return None


def shutil_which(binary: str) -> str | None:
    result = subprocess.run(
        ["sh", "-c", f"command -v {binary!s}"],
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return value or None


def render_pdf(chrome: str, html_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
