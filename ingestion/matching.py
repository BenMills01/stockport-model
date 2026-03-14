"""Source-to-database player matching helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import lru_cache

from sqlalchemy import select

from db.schema import Player, SourcePlayerMapping
from db.session import session_scope
from ingestion.common import normalise_text, upsert_rows


@dataclass(frozen=True)
class PlayerMatch:
    player_id: int
    score: float


@dataclass(frozen=True)
class PlayerCandidate:
    player_id: int
    player_name: str
    current_team: str | None
    normalized_name: str
    normalized_team: str
    first_initial: str
    last_token: str


def find_player_match(
    player_name: str,
    *,
    team_name: str | None = None,
    league_id: int | None = None,
    threshold: float = 0.72,
) -> PlayerMatch | None:
    """Return the best scored player match when it clears the threshold."""

    candidates = _load_candidates(league_id=league_id)
    target_name = normalise_text(player_name)
    target_team = normalise_text(team_name)
    candidate_pool = _filter_candidates(target_name, candidates)

    best_match: PlayerMatch | None = None
    for candidate in candidate_pool:
        score = _score_candidate(
            target_name,
            candidate.normalized_name,
            target_team,
            candidate.normalized_team,
        )
        if best_match is None or score > best_match.score:
            best_match = PlayerMatch(player_id=candidate.player_id, score=score)

    if best_match is None or best_match.score < threshold:
        return None
    return best_match


def match_player_id(
    player_name: str,
    *,
    team_name: str | None = None,
    league_id: int | None = None,
    threshold: float = 0.72,
) -> int | None:
    """Resolve a scraped player identity to an internal player_id."""

    match = find_player_match(
        player_name,
        team_name=team_name,
        league_id=league_id,
        threshold=threshold,
    )
    if match is None:
        return None
    return match.player_id


def build_source_lookup_key(
    source_player_name: str | None,
    *,
    source_team_name: str | None = None,
    source_player_external_id: str | None = None,
) -> str:
    """Build a stable lookup key for third-party player mappings."""

    external_id = (source_player_external_id or "").strip()
    if external_id:
        return f"id:{external_id.lower()}"

    player_name = normalise_text(source_player_name)
    if not player_name:
        raise ValueError("A source player name or external id is required to build a mapping key")

    team_name = normalise_text(source_team_name)
    if team_name:
        return f"name:{player_name}|team:{team_name}"
    return f"name:{player_name}"


def get_source_player_mapping(
    source: str,
    *,
    source_player_name: str | None = None,
    source_team_name: str | None = None,
    source_player_external_id: str | None = None,
) -> SourcePlayerMapping | None:
    """Return a previously saved source-to-player mapping when one exists."""

    lookup_key = build_source_lookup_key(
        source_player_name,
        source_team_name=source_team_name,
        source_player_external_id=source_player_external_id,
    )

    with session_scope() as session:
        statement = select(SourcePlayerMapping).where(
            SourcePlayerMapping.source == source,
            SourcePlayerMapping.source_lookup_key == lookup_key,
        )
        return session.scalar(statement)


def save_source_player_mapping(
    source: str,
    *,
    player_id: int,
    source_player_name: str | None = None,
    source_team_name: str | None = None,
    source_player_external_id: str | None = None,
    league_id: int | None = None,
    match_score: float | None = None,
    matched_by: str = "fuzzy_match",
) -> int:
    """Persist a resolved third-party player mapping for future imports."""

    lookup_key = build_source_lookup_key(
        source_player_name,
        source_team_name=source_team_name,
        source_player_external_id=source_player_external_id,
    )
    rows = [
        {
            "source": source,
            "source_lookup_key": lookup_key,
            "source_player_name": source_player_name,
            "source_team_name": source_team_name,
            "source_player_external_id": source_player_external_id,
            "league_id": league_id,
            "player_id": player_id,
            "match_score": match_score,
            "matched_by": matched_by,
            "matched_at": datetime.now(timezone.utc),
        }
    ]
    return upsert_rows(SourcePlayerMapping, rows, ["source", "source_lookup_key"])


def resolve_source_player_id(
    source: str,
    source_player_name: str | None,
    *,
    source_team_name: str | None = None,
    source_player_external_id: str | None = None,
    league_id: int | None = None,
    threshold: float = 0.72,
    persist_mapping: bool = True,
) -> int | None:
    """Resolve a source-specific player identity and optionally persist the mapping."""

    if not source_player_name and not source_player_external_id:
        return None

    mapping = get_source_player_mapping(
        source,
        source_player_name=source_player_name,
        source_team_name=source_team_name,
        source_player_external_id=source_player_external_id,
    )
    if mapping is not None:
        return mapping.player_id

    if not source_player_name:
        return None

    match = find_player_match(
        source_player_name,
        team_name=source_team_name,
        league_id=league_id,
        threshold=threshold,
    )
    matched_by = "fuzzy_match"
    if match is None and league_id is not None:
        match = find_player_match(
            source_player_name,
            team_name=source_team_name,
            league_id=None,
            threshold=threshold,
        )
        matched_by = "fuzzy_match_no_league"
    if match is None:
        return None

    if persist_mapping:
        save_source_player_mapping(
            source,
            player_id=match.player_id,
            source_player_name=source_player_name,
            source_team_name=source_team_name,
            source_player_external_id=source_player_external_id,
            league_id=league_id,
            match_score=match.score,
            matched_by=matched_by,
        )
    return match.player_id


def _score_candidate(
    target_name: str,
    candidate_name: str,
    target_team: str,
    candidate_team: str,
) -> float:
    name_score = SequenceMatcher(None, target_name, candidate_name).ratio()
    if not target_team:
        return name_score

    team_score = SequenceMatcher(None, target_team, candidate_team).ratio() if candidate_team else 0.0
    return max(name_score, (0.8 * name_score) + (0.2 * team_score))


def _filter_candidates(
    target_name: str,
    candidates: tuple[PlayerCandidate, ...],
) -> tuple[PlayerCandidate, ...]:
    tokens = target_name.split()
    if not tokens:
        return candidates

    first_initial = tokens[0][0]
    last_token = tokens[-1]
    surname_matches = tuple(candidate for candidate in candidates if candidate.last_token == last_token)
    if not surname_matches:
        return candidates

    initial_matches = tuple(candidate for candidate in surname_matches if candidate.first_initial == first_initial)
    if initial_matches:
        return initial_matches
    return surname_matches


@lru_cache(maxsize=None)
def _load_candidates(*, league_id: int | None) -> tuple[PlayerCandidate, ...]:
    with session_scope() as session:
        statement = select(Player.player_id, Player.player_name, Player.current_team)
        if league_id is not None:
            statement = statement.where(Player.current_league_id == league_id)
        rows = session.execute(statement).all()
    candidates: list[PlayerCandidate] = []
    for row in rows:
        normalized_name = normalise_text(row.player_name)
        candidates.append(
            PlayerCandidate(
                player_id=row.player_id,
                player_name=row.player_name,
                current_team=row.current_team,
                normalized_name=normalized_name,
                normalized_team=normalise_text(row.current_team),
                first_initial=normalized_name[:1],
                last_token=(normalized_name.split()[-1] if normalized_name else ""),
            )
        )
    return tuple(candidates)
