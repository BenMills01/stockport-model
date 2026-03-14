"""Shared ingestion helpers."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import html
import re
from typing import Any
import unicodedata

from sqlalchemy.dialects.postgresql import insert

from db.session import session_scope


_MULTIPLIER_MAP = {
    "k": Decimal("1000"),
    "th": Decimal("1000"),
    "m": Decimal("1000000"),
    "bn": Decimal("1000000000"),
}


def upsert_rows(
    model: type[Any],
    rows: list[dict[str, Any]],
    conflict_columns: list[str],
) -> int:
    """Upsert rows into a PostgreSQL-backed table."""

    if not rows:
        return 0

    deduplicated_rows = _deduplicate_rows(rows, conflict_columns)
    table = model.__table__
    statement = insert(table).values(deduplicated_rows)
    row_keys = set().union(*(row.keys() for row in deduplicated_rows))
    update_columns = {
        column.name: getattr(statement.excluded, column.name)
        for column in table.columns
        if column.name in row_keys and column.name not in conflict_columns
    }

    with session_scope() as session:
        session.execute(
            statement.on_conflict_do_update(
                index_elements=conflict_columns,
                set_=update_columns,
            )
        )
    return len(deduplicated_rows)


def _deduplicate_rows(
    rows: list[dict[str, Any]],
    conflict_columns: list[str],
) -> list[dict[str, Any]]:
    """Collapse duplicate conflict keys within one insert payload."""

    deduplicated: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(column) for column in conflict_columns)
        deduplicated[key] = row
    return list(deduplicated.values())


def normalise_text(value: str | None) -> str:
    """Normalise text for fuzzy matching across sources."""

    if value is None:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_decimal_number(value: Any) -> Decimal | None:
    """Convert common numeric strings into Decimal values."""

    if value in (None, "", "-", "?", "nan"):
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value).replace(",", "").strip())
    except InvalidOperation:
        return None


def parse_money_to_eur(
    raw_value: Any,
    *,
    gbp_to_eur_rate: float,
    usd_to_eur_rate: float,
    chf_to_eur_rate: float,
) -> Decimal | None:
    """Parse a Transfermarkt-style money string into EUR."""

    if raw_value in (None, "", "-", "?", "Not available"):
        return None

    value = str(raw_value).strip().replace(",", ".")
    multiplier_match = re.search(r"(bn|m|th|k)\b", value.lower())
    amount_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", value)
    if not amount_match:
        return None

    amount = Decimal(amount_match.group(1))
    multiplier = _MULTIPLIER_MAP.get(multiplier_match.group(1).lower(), Decimal("1")) if multiplier_match else Decimal("1")
    base_value = amount * multiplier

    if "£" in value:
        return (base_value * Decimal(str(gbp_to_eur_rate))).quantize(Decimal("0.01"))
    if "$" in value:
        return (base_value * Decimal(str(usd_to_eur_rate))).quantize(Decimal("0.01"))
    if "CHF" in value.upper():
        return (base_value * Decimal(str(chf_to_eur_rate))).quantize(Decimal("0.01"))
    return base_value.quantize(Decimal("0.01"))
