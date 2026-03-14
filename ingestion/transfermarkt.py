"""Transfermarkt ingestion for market values, contract dates, and wage estimates."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from bs4 import BeautifulSoup, Tag
import pandas as pd
import requests
from sqlalchemy import select, update

from config import get_settings
from db.schema import MarketValue, MarketValueHistory, Player, Transfer
from db.session import session_scope
from ingestion.common import parse_money_to_eur, upsert_rows
from ingestion.matching import match_player_id, save_source_player_mapping


LOGGER = logging.getLogger(__name__)

# Annual wage as a fraction of market value, indexed by approximate league tier.
# Derived from public wage reporting for English football (2023-24 season).
# Tier 1 (Premier League): ~25 %  — far above our scope, listed for completeness.
# Tier 2 (Championship):   ~12 %
# Tier 3 (League One):     ~ 8 %
# Tier 4 (League Two):     ~ 6 %
# Unknown / default:        ~ 8 %
_ANNUAL_WAGE_FRACTION_BY_TIER = {
    1: 0.25,
    2: 0.12,
    3: 0.08,
    4: 0.06,
}
_DEFAULT_ANNUAL_WAGE_FRACTION = 0.08

# League tier lookup from the slug keyword — coarse but sufficient for estimates.
_SLUG_TIER_HINTS = {
    "GB1": 1, "premier-league": 1,
    "GB2": 2, "championship": 2,
    "GB3": 3, "league-one": 3,
    "GB4": 4, "league-two": 4,
}

# Polite delay between profile page requests to avoid hammering TM.
_PROFILE_REQUEST_DELAY_SECONDS = 2.0

# TM profile URL path pattern used to identify player links in squad pages.
_TM_PLAYER_HREF_PATTERN = re.compile(r"/[^/]+/profil/spieler/(\d+)$")


def scrape_market_values(league_slug: str, season: str) -> pd.DataFrame:
    """Scrape a Transfermarkt squad-value page into a normalised dataframe.

    `league_slug` may be a full URL or a path such as
    `league-one/startseite/wettbewerb/GB3`.

    Returns a DataFrame with columns:
    player_name, team_name, season, market_value_eur, contract_expiry,
    wage_eur (if the page carries wage data, otherwise None),
    tm_profile_path (relative TM URL for the player, or None).
    """

    html = _fetch_transfermarkt_html(_build_transfermarkt_url(league_slug, season))
    frame, player_links = _read_transfermarkt_table_with_links(html)
    columns = _flatten_columns(frame.columns)
    frame.columns = columns

    player_col = _resolve_column(columns, ["player", "name"])
    team_col = _resolve_column(columns, ["club", "team", "squad"])
    value_col = _resolve_column(columns, ["market_value", "marketvalue", "mv"])
    expiry_col = _resolve_column(columns, ["contract_expires", "contract_until", "contract"])
    wage_col = _resolve_column(columns, ["wage", "salary", "weekly_wage", "annual_wage"])
    if not player_col or not team_col or not value_col:
        raise KeyError("Transfermarkt table missing required player/team/value columns")

    settings = get_settings()

    def _parse_money(value: Any) -> Decimal | None:
        return parse_money_to_eur(
            value,
            gbp_to_eur_rate=settings.gbp_to_eur_rate,
            usd_to_eur_rate=settings.usd_to_eur_rate,
            chf_to_eur_rate=settings.chf_to_eur_rate,
        )

    result = pd.DataFrame(
        {
            "player_name": frame[player_col],
            "team_name": frame[team_col],
            "market_value_eur": frame[value_col].map(_parse_money),
            "contract_expiry": frame[expiry_col].map(_parse_contract_date) if expiry_col else None,
            "wage_eur": frame[wage_col].map(_parse_money) if wage_col else None,
        }
    )

    result = result.dropna(subset=["player_name"]).reset_index(drop=True)
    result["season"] = season
    result["tm_profile_path"] = result["player_name"].map(player_links)
    return result[[
        "player_name", "team_name", "season", "market_value_eur",
        "contract_expiry", "wage_eur", "tm_profile_path",
    ]]


def ingest_market_values(league_slug: str, season: str, league_id: int | None = None) -> int:
    """Scrape and upsert Transfermarkt market values with wage estimates.

    Also saves TM player profile path mappings in SourcePlayerMapping so that
    subsequent profile/history scrapers can look up the TM URL for each player.

    Wage strategy
    -------------
    1. Use the scraped ``wage_eur`` column if the page carries wage data (rare
       on squad-overview pages but present on some detailed views).
    2. Otherwise estimate from market value using a league-tier multiplier:
       ``annual_wage_eur ≈ market_value_eur × tier_fraction``.
       The result is stored as an annual figure in ``wage_estimate`` to match
       the MarketValue schema column (which the affordability gate interprets
       as annual wages when ``budget_max_wage`` is also annual).
    """

    data = scrape_market_values(league_slug, season)
    snapshot_date = date.today()
    tier = _infer_tier_from_slug(league_slug)

    rows: list[dict[str, Any]] = []
    for record in data.to_dict(orient="records"):
        player_id = match_player_id(
            record["player_name"],
            team_name=record.get("team_name"),
            league_id=league_id,
        )
        if player_id is None:
            continue

        market_value_eur: Decimal | None = record.get("market_value_eur")
        scraped_wage: Decimal | None = record.get("wage_eur")

        # Prefer scraped wage; fall back to market-value estimate.
        if scraped_wage is not None:
            wage_estimate = scraped_wage
        else:
            wage_estimate = _estimate_annual_wage_eur(market_value_eur, tier)

        rows.append(
            {
                "player_id": player_id,
                "date": snapshot_date,
                "market_value_eur": market_value_eur,
                "contract_expiry": record.get("contract_expiry"),
                "wage_estimate": wage_estimate,
            }
        )

        # Persist the TM profile URL as a source mapping for later profile scrapes.
        tm_path: str | None = record.get("tm_profile_path")
        if tm_path:
            tm_id_match = _TM_PLAYER_HREF_PATTERN.search(tm_path)
            tm_external_id = tm_id_match.group(1) if tm_id_match else None
            try:
                save_source_player_mapping(
                    "transfermarkt",
                    player_id=player_id,
                    source_player_name=record["player_name"],
                    source_team_name=record.get("team_name"),
                    source_player_external_id=tm_external_id,
                    league_id=league_id,
                    matched_by="squad_page_scrape",
                )
            except Exception:
                LOGGER.debug(
                    "TM mapping save skipped for player_id=%d (likely duplicate)", player_id
                )

    return upsert_rows(MarketValue, rows, ["player_id", "date"])


# ── Player profile page scrapers ─────────────────────────────────────────────


def scrape_player_profile(tm_player_url: str) -> dict[str, str | None]:
    """Scrape preferred foot, secondary nationality, and agent from a TM profile page.

    Returns a dict with keys: ``preferred_foot``, ``nationality_secondary``, ``agent_name``.
    Values are ``None`` when not found on the page.
    """

    html = _fetch_transfermarkt_html(tm_player_url)
    soup = BeautifulSoup(html, "html.parser")

    preferred_foot: str | None = None
    nationality_secondary: str | None = None
    agent_name: str | None = None

    # The info table on TM profile pages holds foot, citizenship, agent etc.
    # Try the structured table first (class "auflistung" or "info-table").
    info_tables = soup.find_all("table", class_=re.compile(r"auflistung|info.?table"))
    for table in info_tables:
        rows = table.find_all("tr")
        for row in rows:
            label_cell = row.find("th") or row.find("td", class_=re.compile(r"label"))
            all_tds = row.find_all("td")
            value_cell = row.find("td", class_=re.compile(r"value|hauptlink|zentriert")) or (
                all_tds[-1] if all_tds else None
            )
            if not label_cell or not value_cell:
                continue
            label_text = label_cell.get_text(" ", strip=True).lower()
            value_text = value_cell.get_text(" ", strip=True)

            if "foot" in label_text and not preferred_foot:
                foot = value_text.lower().strip()
                if foot in ("right", "left", "both"):
                    preferred_foot = foot
                elif foot:
                    preferred_foot = foot[:16]  # truncate to column width

            elif "agent" in label_text and not agent_name:
                agent_name = value_text or None

    # Secondary citizenship: TM shows multiple flag icons in the header area.
    # The first is primary nationality; additional ones are secondary.
    citizenship_spans = soup.select(
        "span[itemprop='nationality'], .data-header__items .data-header__content"
    )
    nationalities: list[str] = []
    for span in citizenship_spans:
        text = span.get_text(" ", strip=True)
        if text and text not in nationalities:
            nationalities.append(text)
    if len(nationalities) >= 2:
        nationality_secondary = nationalities[1]

    # Alternative: look for flag images in the detail section.
    if nationality_secondary is None:
        flag_imgs = soup.select("span.flaggenrahmen img, .data-header__items img.flagge")
        titles = [img.get("title", "").strip() for img in flag_imgs if img.get("title")]
        if len(titles) >= 2:
            nationality_secondary = titles[1]

    return {
        "preferred_foot": preferred_foot,
        "nationality_secondary": nationality_secondary,
        "agent_name": agent_name,
    }


def scrape_player_transfer_history(tm_player_url: str) -> list[dict[str, Any]]:
    """Scrape transfer history (with fees) from a TM player profile page.

    The transfers tab is at ``/transfers/spieler/{id}`` — we derive this URL
    automatically from the profile URL.

    Returns a list of dicts with keys:
    ``date``, ``team_out``, ``team_in``, ``fee_eur``, ``transfer_type``.
    ``fee_eur`` is ``None`` for free transfers or when unparseable.
    """

    # Derive transfers URL from profile URL.
    transfers_url = re.sub(r"/profil/spieler/", "/transfers/spieler/", tm_player_url)
    if "/profil/" not in tm_player_url:
        # Already a transfers URL or unknown format — use as-is.
        transfers_url = tm_player_url

    html = _fetch_transfermarkt_html(transfers_url)
    soup = BeautifulSoup(html, "html.parser")
    settings = get_settings()

    def _parse_money(value: Any) -> Decimal | None:
        return parse_money_to_eur(
            value,
            gbp_to_eur_rate=settings.gbp_to_eur_rate,
            usd_to_eur_rate=settings.usd_to_eur_rate,
            chf_to_eur_rate=settings.chf_to_eur_rate,
        )

    records: list[dict[str, Any]] = []

    # TM transfer tables have class "items" inside a transfer-history block.
    for table in soup.select("table.items"):
        for row in table.select("tbody tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            cell_texts = [c.get_text(" ", strip=True) for c in cells]

            # Heuristic: look for a date-like cell and a fee-like cell.
            transfer_date: date | None = None
            team_out: str | None = None
            team_in: str | None = None
            fee_raw: str | None = None
            transfer_type: str = "transfer"

            for text in cell_texts:
                if transfer_date is None:
                    parsed_date = _parse_transfer_date(text)
                    if parsed_date:
                        transfer_date = parsed_date

            # TM layout: Season | Date | Left (team_out) | Joined (team_in) | MV | Fee
            if len(cell_texts) >= 6:
                team_out = cell_texts[2] or None
                team_in = cell_texts[3] or None
                fee_raw = cell_texts[5]
            elif len(cell_texts) >= 5:
                team_out = cell_texts[2] or None
                team_in = cell_texts[3] or None
                fee_raw = cell_texts[4]

            if fee_raw:
                lower = fee_raw.lower()
                if "loan" in lower:
                    transfer_type = "loan"
                    fee_raw = re.sub(r"loan\s*(fee:?\s*)?", "", lower, flags=re.I).strip() or None
                elif "free" in lower:
                    transfer_type = "free"
                    fee_raw = None

            fee_eur = _parse_money(fee_raw) if fee_raw else None

            if transfer_date or team_in or team_out:
                records.append(
                    {
                        "date": transfer_date,
                        "team_out": team_out,
                        "team_in": team_in,
                        "fee_eur": fee_eur,
                        "transfer_type": transfer_type,
                    }
                )

    return records


def scrape_player_value_history(tm_player_url: str) -> list[dict[str, Any]]:
    """Scrape the market value time series from a TM player profile page.

    Tries to parse the Highcharts JSON embedded in the marktwertverlauf page.
    Returns a list of dicts with keys: ``date``, ``value_eur``.
    """

    history_url = re.sub(r"/profil/spieler/", "/marktwertverlauf/spieler/", tm_player_url)
    if "/profil/" not in tm_player_url:
        history_url = tm_player_url

    html = _fetch_transfermarkt_html(history_url)
    soup = BeautifulSoup(html, "html.parser")
    settings = get_settings()

    def _parse_money(value: Any) -> Decimal | None:
        return parse_money_to_eur(
            value,
            gbp_to_eur_rate=settings.gbp_to_eur_rate,
            usd_to_eur_rate=settings.usd_to_eur_rate,
            chf_to_eur_rate=settings.chf_to_eur_rate,
        )

    records: list[dict[str, Any]] = []

    # TM embeds chart data as a JSON blob in a script tag.
    # Pattern: var highchartsData = {...} or similar; the list key holds snapshots.
    for script_tag in soup.find_all("script"):
        script_text = script_tag.string or ""
        if '"list"' not in script_text and "marktwertverlauf" not in script_text.lower():
            continue

        # Extract the JSON object from the script.
        json_match = re.search(r'\{[^{}]*"list"\s*:\s*\[.*?\]\s*[^{}]*\}', script_text, re.S)
        if not json_match:
            # Try broader extraction.
            json_match = re.search(r'(\{.*"datum_mw".*\})', script_text, re.S)
        if not json_match:
            continue

        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            continue

        snapshot_list = data.get("list") or data.get("data", {}).get("list", [])
        for entry in snapshot_list:
            raw_date = entry.get("datum_mw") or entry.get("date") or entry.get("x")
            raw_value = entry.get("mw") or entry.get("y") or entry.get("value")

            snapshot_date: date | None = None
            if isinstance(raw_date, str):
                snapshot_date = _parse_contract_date(raw_date)
            elif isinstance(raw_date, (int, float)):
                # Highcharts uses millisecond Unix timestamps.
                try:
                    snapshot_date = datetime.utcfromtimestamp(raw_date / 1000).date()
                except (ValueError, OSError):
                    pass

            value_eur = _parse_money(raw_value) if raw_value else None

            if snapshot_date:
                records.append({"date": snapshot_date, "value_eur": value_eur})

        if records:
            break  # Found and parsed successfully.

    return records


# ── Batch ingest helpers ──────────────────────────────────────────────────────


def ingest_player_profiles(player_ids: list[int] | None = None) -> dict[str, int]:
    """Enrich Player rows with foot, secondary nationality, and agent from TM profile pages.

    Iterates all players that have a Transfermarkt source mapping with an external ID
    (i.e. a known TM player URL). Scrapes each profile page and updates the Player row.

    Args:
        player_ids: If given, restrict to this subset; otherwise process all known TM mappings.

    Returns:
        Dict with ``{"scraped": N, "updated": N, "errors": N}``.
    """

    tm_mappings = _load_tm_mappings(player_ids)
    if not tm_mappings:
        LOGGER.info("No Transfermarkt player mappings found — skipping profile enrichment")
        return {"scraped": 0, "updated": 0, "errors": 0}

    settings = get_settings()
    base_url = settings.transfermarkt_base_url.rstrip("/")
    scraped = updated = errors = 0

    for player_id, tm_external_id, tm_player_name, tm_team_name in tm_mappings:
        profile_url = f"{base_url}/{tm_player_name or 'player'}/profil/spieler/{tm_external_id}"
        try:
            profile = scrape_player_profile(profile_url)
            time.sleep(_PROFILE_REQUEST_DELAY_SECONDS)
        except Exception as exc:
            LOGGER.warning("TM profile scrape failed for player_id=%d: %s", player_id, exc)
            errors += 1
            continue

        scraped += 1
        updates: dict[str, Any] = {
            k: v for k, v in profile.items() if v is not None
        }
        if not updates:
            continue

        with session_scope() as session:
            session.execute(
                update(Player)
                .where(Player.player_id == player_id)
                .values(**updates)
            )
        updated += 1
        LOGGER.debug("Updated profile for player_id=%d: %s", player_id, list(updates.keys()))

    LOGGER.info(
        "TM profile enrichment complete: scraped=%d updated=%d errors=%d",
        scraped,
        updated,
        errors,
    )
    return {"scraped": scraped, "updated": updated, "errors": errors}


def ingest_transfer_fees(player_ids: list[int] | None = None) -> dict[str, int]:
    """Scrape transfer history from TM and backfill ``fee_paid`` on Transfer rows.

    For each player with a known TM mapping, fetches their transfer history and
    attempts to match each scraped transfer against an existing Transfer row by
    player_id, date (within ±30 days), and team names.

    Returns:
        Dict with ``{"scraped": N, "matched": N, "errors": N}``.
    """

    tm_mappings = _load_tm_mappings(player_ids)
    if not tm_mappings:
        return {"scraped": 0, "matched": 0, "errors": 0}

    settings = get_settings()
    base_url = settings.transfermarkt_base_url.rstrip("/")
    scraped = matched = errors = 0

    for player_id, tm_external_id, _tm_player_name, _tm_team_name in tm_mappings:
        profile_url = (
            f"{base_url}/{_tm_player_name or 'player'}/profil/spieler/{tm_external_id}"
        )
        try:
            transfers = scrape_player_transfer_history(profile_url)
            time.sleep(_PROFILE_REQUEST_DELAY_SECONDS)
        except Exception as exc:
            LOGGER.warning(
                "TM transfer history scrape failed for player_id=%d: %s", player_id, exc
            )
            errors += 1
            continue

        scraped += len(transfers)

        with session_scope() as session:
            db_transfers = session.scalars(
                select(Transfer).where(Transfer.player_id == player_id)
            ).all()

            for scraped_tx in transfers:
                fee = scraped_tx.get("fee_eur")
                if fee is None:
                    continue

                best_match: Transfer | None = None
                best_delta = 999
                for db_tx in db_transfers:
                    if db_tx.fee_paid is not None:
                        continue  # Already populated.
                    if not _team_name_similar(
                        scraped_tx.get("team_in"), db_tx.team_in
                    ) and not _team_name_similar(
                        scraped_tx.get("team_out"), db_tx.team_out
                    ):
                        continue
                    if scraped_tx.get("date") and db_tx.date:
                        delta = abs((scraped_tx["date"] - db_tx.date).days)
                        if delta <= 60 and delta < best_delta:
                            best_delta = delta
                            best_match = db_tx
                    elif best_match is None:
                        best_match = db_tx

                if best_match is not None:
                    best_match.fee_paid = fee
                    matched += 1

    LOGGER.info(
        "TM transfer fee backfill complete: scraped=%d matched=%d errors=%d",
        scraped,
        matched,
        errors,
    )
    return {"scraped": scraped, "matched": matched, "errors": errors}


def ingest_value_history(player_ids: list[int] | None = None) -> dict[str, int]:
    """Scrape market value history timeline from TM and store in MarketValueHistory.

    Returns:
        Dict with ``{"scraped": N, "upserted": N, "errors": N}``.
    """

    tm_mappings = _load_tm_mappings(player_ids)
    if not tm_mappings:
        return {"scraped": 0, "upserted": 0, "errors": 0}

    settings = get_settings()
    base_url = settings.transfermarkt_base_url.rstrip("/")
    total_scraped = total_upserted = errors = 0

    for player_id, tm_external_id, _tm_player_name, _tm_team_name in tm_mappings:
        profile_url = (
            f"{base_url}/{_tm_player_name or 'player'}/profil/spieler/{tm_external_id}"
        )
        try:
            history = scrape_player_value_history(profile_url)
            time.sleep(_PROFILE_REQUEST_DELAY_SECONDS)
        except Exception as exc:
            LOGGER.warning(
                "TM value history scrape failed for player_id=%d: %s", player_id, exc
            )
            errors += 1
            continue

        if not history:
            continue

        total_scraped += len(history)
        rows = [
            {"player_id": player_id, "date": entry["date"], "value_eur": entry["value_eur"]}
            for entry in history
        ]
        total_upserted += upsert_rows(MarketValueHistory, rows, ["player_id", "date"])

    LOGGER.info(
        "TM value history ingest complete: scraped=%d upserted=%d errors=%d",
        total_scraped,
        total_upserted,
        errors,
    )
    return {"scraped": total_scraped, "upserted": total_upserted, "errors": errors}


# ── Internal helpers ──────────────────────────────────────────────────────────


def _estimate_annual_wage_eur(
    market_value_eur: Decimal | None,
    tier: int,
) -> Decimal | None:
    """Return an estimated annual wage in EUR from market value and league tier.

    Returns None when the market value is unknown (no basis for estimation).
    """
    if market_value_eur is None or market_value_eur <= 0:
        return None
    fraction = Decimal(str(_ANNUAL_WAGE_FRACTION_BY_TIER.get(tier, _DEFAULT_ANNUAL_WAGE_FRACTION)))
    return (market_value_eur * fraction).quantize(Decimal("0.01"))


def _infer_tier_from_slug(league_slug: str) -> int:
    """Coarsely infer English league tier from the slug or URL."""
    for keyword, tier in _SLUG_TIER_HINTS.items():
        if keyword in league_slug:
            return tier
    return 3  # Default: League One (Stockport's level)


def _build_transfermarkt_url(league_slug: str, season: str) -> str:
    if league_slug.startswith("http://") or league_slug.startswith("https://"):
        return league_slug

    base = get_settings().transfermarkt_base_url.rstrip("/")
    path = league_slug.lstrip("/")
    if "saison_id=" in path:
        return f"{base}/{path}"
    separator = "&" if "?" in path else "?"
    return f"{base}/{path}{separator}saison_id={season}"


def _fetch_transfermarkt_html(url: str) -> str:
    settings = get_settings()
    response = requests.get(url, headers={"User-Agent": settings.http_user_agent}, timeout=30)
    response.raise_for_status()
    return response.text


def _read_transfermarkt_table_with_links(html: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Parse TM squad table and extract both row data and player profile URL map.

    Returns:
        (DataFrame with columns matching table headers, dict mapping player_name → profile_path)
    """
    soup = BeautifulSoup(html, "html.parser")
    table: Tag | None = soup.select_one("table.items")
    if table is None:
        for candidate in soup.find_all("table"):
            text = candidate.get_text(" ", strip=True).lower()
            if "market value" in text and "player" in text:
                table = candidate
                break
    if table is None:
        raise ValueError("Could not find Transfermarkt player value table")

    headers = [cell.get_text(" ", strip=True) for cell in table.select("thead tr th")]

    # Build player_name → profile_path mapping from anchor hrefs in tbody.
    player_links: dict[str, str] = {}
    rows: list[list[str]] = []
    for row in table.select("tbody tr"):
        row_data = _expand_row_cells(row)
        rows.append(row_data)

        # Find player profile links within this row.
        for anchor in row.find_all("a", href=True):
            href: str = anchor["href"]
            if _TM_PLAYER_HREF_PATTERN.search(href):
                player_name = anchor.get_text(" ", strip=True)
                if player_name:
                    player_links[player_name] = href
                break  # First matching link per row is the player link.

    return pd.DataFrame(rows, columns=headers), player_links


def _flatten_columns(columns: Any) -> list[str]:
    flattened: list[str] = []
    for column in columns:
        if isinstance(column, tuple):
            parts = [str(part).strip() for part in column if str(part).strip() and "Unnamed" not in str(part)]
            name = "_".join(parts)
        else:
            name = str(column).strip()
        name = name.lower()
        name = "".join(character if character.isalnum() else "_" for character in name)
        name = "_".join(segment for segment in name.split("_") if segment)
        flattened.append(name)
    return flattened


def _resolve_column(columns: list[str], aliases: list[str]) -> str | None:
    for alias in aliases:
        for column in columns:
            if column == alias or column.endswith(f"_{alias}") or alias in column:
                return column
    return None


def _expand_row_cells(row: Tag) -> list[str]:
    values: list[str] = []
    for cell in row.find_all(["th", "td"], recursive=False):
        colspan = int(cell.get("colspan", 1))
        text = cell.get_text(" ", strip=True)
        values.extend([text] * colspan)
    return values


def _parse_contract_date(value: Any) -> date | None:
    if value in (None, "", "-", "?", "nan"):
        return None

    text = str(value).strip()
    for fmt in ("%b %d, %Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%b %d %Y", "%b %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _parse_transfer_date(value: Any) -> date | None:
    """Parse a date string as it appears in TM transfer history tables."""
    if value in (None, "", "-"):
        return None
    text = str(value).strip()
    for fmt in ("%b %d, %Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%b %d %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _load_tm_mappings(
    player_ids: list[int] | None,
) -> list[tuple[int, str, str | None, str | None]]:
    """Load (player_id, tm_external_id, source_player_name, source_team_name) for TM-mapped players."""
    from db.schema import SourcePlayerMapping

    with session_scope() as session:
        stmt = select(
            SourcePlayerMapping.player_id,
            SourcePlayerMapping.source_player_external_id,
            SourcePlayerMapping.source_player_name,
            SourcePlayerMapping.source_team_name,
        ).where(
            SourcePlayerMapping.source == "transfermarkt",
            SourcePlayerMapping.source_player_external_id.isnot(None),
        )
        if player_ids:
            stmt = stmt.where(SourcePlayerMapping.player_id.in_(player_ids))
        rows = session.execute(stmt).all()

    return [
        (row.player_id, row.source_player_external_id, row.source_player_name, row.source_team_name)
        for row in rows
        if row.source_player_external_id
    ]


def _team_name_similar(name_a: str | None, name_b: str | None) -> bool:
    """Rough team name similarity — True if one contains the other or they share significant tokens."""
    if not name_a or not name_b:
        return False
    a = name_a.lower().strip()
    b = name_b.lower().strip()
    if a == b or a in b or b in a:
        return True
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    common = tokens_a & tokens_b - {"fc", "afc", "city", "united", "town", "the"}
    return len(common) >= 1
