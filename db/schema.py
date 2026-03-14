"""Declarative schema for the Stockport recruitment model."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Index, Integer
from sqlalchemy import MetaData, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Shared declarative base with migration-friendly naming rules."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


Money = Numeric(14, 2)


class Player(Base):
    __tablename__ = "players"

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_name: Mapped[str] = mapped_column(Text, nullable=False)
    nationality: Mapped[str | None] = mapped_column(Text)
    nationality_secondary: Mapped[str | None] = mapped_column(Text)
    birth_date: Mapped[date | None] = mapped_column(Date)
    current_age_years: Mapped[float | None] = mapped_column(Float)
    height_cm: Mapped[int | None] = mapped_column(Integer)
    weight_kg: Mapped[int | None] = mapped_column(Integer)
    preferred_foot: Mapped[str | None] = mapped_column(String(16))
    photo_url: Mapped[str | None] = mapped_column(Text)
    current_team: Mapped[str | None] = mapped_column(Text)
    current_league_id: Mapped[int | None] = mapped_column(Integer)
    agent_name: Mapped[str | None] = mapped_column(Text)


class MatchPerformance(Base):
    __tablename__ = "match_performances"
    __table_args__ = (
        Index("ix_match_performances_player_date", "player_id", "date"),
        Index("ix_match_performances_league_season", "league_id", "season"),
    )

    fixture_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league_id: Mapped[int] = mapped_column(Integer, nullable=False)
    season: Mapped[str] = mapped_column(String(16), nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    home_team: Mapped[str] = mapped_column(Text, nullable=False)
    away_team: Mapped[str] = mapped_column(Text, nullable=False)
    team: Mapped[str] = mapped_column(Text, nullable=False)
    is_home: Mapped[bool] = mapped_column(Boolean, nullable=False)
    referee: Mapped[str | None] = mapped_column(Text)
    minutes: Mapped[int | None] = mapped_column(Integer)
    position: Mapped[str | None] = mapped_column(Text)
    rating: Mapped[float | None] = mapped_column(Float)
    is_substitute: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_captain: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    goals_scored: Mapped[int | None] = mapped_column(Integer)
    goals_conceded: Mapped[int | None] = mapped_column(Integer)
    assists: Mapped[int | None] = mapped_column(Integer)
    saves: Mapped[int | None] = mapped_column(Integer)
    shots_total: Mapped[int | None] = mapped_column(Integer)
    shots_on_target: Mapped[int | None] = mapped_column(Integer)
    passes_total: Mapped[int | None] = mapped_column(Integer)
    passes_key: Mapped[int | None] = mapped_column(Integer)
    pass_accuracy: Mapped[float | None] = mapped_column(Float)
    tackles_total: Mapped[int | None] = mapped_column(Integer)
    tackles_blocks: Mapped[int | None] = mapped_column(Integer)
    tackles_interceptions: Mapped[int | None] = mapped_column(Integer)
    duels_total: Mapped[int | None] = mapped_column(Integer)
    duels_won: Mapped[int | None] = mapped_column(Integer)
    dribbles_attempts: Mapped[int | None] = mapped_column(Integer)
    dribbles_success: Mapped[int | None] = mapped_column(Integer)
    dribbles_past: Mapped[int | None] = mapped_column(Integer)
    fouls_committed: Mapped[int | None] = mapped_column(Integer)
    fouls_drawn: Mapped[int | None] = mapped_column(Integer)
    yellow_cards: Mapped[int | None] = mapped_column(Integer)
    red_cards: Mapped[int | None] = mapped_column(Integer)
    pen_won: Mapped[int | None] = mapped_column(Integer)
    pen_committed: Mapped[int | None] = mapped_column(Integer)
    pen_scored: Mapped[int | None] = mapped_column(Integer)
    pen_missed: Mapped[int | None] = mapped_column(Integer)
    pen_saved: Mapped[int | None] = mapped_column(Integer)
    offsides: Mapped[int | None] = mapped_column(Integer)


class Fixture(Base):
    __tablename__ = "fixtures"

    fixture_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league_id: Mapped[int] = mapped_column(Integer, nullable=False)
    season: Mapped[str] = mapped_column(String(16), nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    home_team: Mapped[str] = mapped_column(Text, nullable=False)
    away_team: Mapped[str] = mapped_column(Text, nullable=False)
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    referee: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str | None] = mapped_column(String(32))


class FixtureTeamStat(Base):
    __tablename__ = "fixture_team_stats"
    __table_args__ = (Index("ix_fixture_team_stats_fixture_id", "fixture_id"),)

    fixture_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_name: Mapped[str] = mapped_column(Text, primary_key=True)
    possession: Mapped[float | None] = mapped_column(Float)
    total_shots: Mapped[int | None] = mapped_column(Integer)
    shots_on_target: Mapped[int | None] = mapped_column(Integer)
    corners: Mapped[int | None] = mapped_column(Integer)
    fouls: Mapped[int | None] = mapped_column(Integer)
    expected_goals: Mapped[float | None] = mapped_column(Float)
    passes_total: Mapped[int | None] = mapped_column(Integer)
    passes_accuracy: Mapped[float | None] = mapped_column(Float)


class MatchEvent(Base):
    __tablename__ = "match_events"
    __table_args__ = (
        UniqueConstraint(
            "fixture_id",
            "time_elapsed",
            "time_extra",
            "event_type",
            "event_detail",
            "player_id",
            "assist_player_id",
            "team",
        ),
        Index("ix_match_events_fixture_id", "fixture_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fixture_id: Mapped[int] = mapped_column(Integer, nullable=False)
    time_elapsed: Mapped[int | None] = mapped_column(Integer)
    time_extra: Mapped[int | None] = mapped_column(Integer)
    event_type: Mapped[str | None] = mapped_column(Text)
    event_detail: Mapped[str | None] = mapped_column(Text)
    player_id: Mapped[int | None] = mapped_column(Integer)
    player_name: Mapped[str | None] = mapped_column(Text)
    assist_player_id: Mapped[int | None] = mapped_column(Integer)
    assist_player_name: Mapped[str | None] = mapped_column(Text)
    team: Mapped[str | None] = mapped_column(Text)
    comments: Mapped[str | None] = mapped_column(Text)


class Lineup(Base):
    __tablename__ = "lineups"
    __table_args__ = (Index("ix_lineups_fixture_id", "fixture_id"),)

    fixture_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team: Mapped[str] = mapped_column(Text, nullable=False)
    is_starter: Mapped[bool] = mapped_column(Boolean, nullable=False)
    position_label: Mapped[str | None] = mapped_column(Text)
    grid_position: Mapped[str | None] = mapped_column(String(16))
    shirt_number: Mapped[int | None] = mapped_column(Integer)
    formation: Mapped[str | None] = mapped_column(String(16))
    coach_name: Mapped[str | None] = mapped_column(Text)
    coach_id: Mapped[int | None] = mapped_column(Integer)


class StandingsSnapshot(Base):
    __tablename__ = "standings_snapshots"

    league_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    team_name: Mapped[str] = mapped_column(Text, primary_key=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    points: Mapped[int] = mapped_column(Integer, nullable=False)
    goal_diff: Mapped[int] = mapped_column(Integer, nullable=False)
    form: Mapped[str | None] = mapped_column(String(16))
    played: Mapped[int | None] = mapped_column(Integer)
    won: Mapped[int | None] = mapped_column(Integer)
    drawn: Mapped[int | None] = mapped_column(Integer)
    lost: Mapped[int | None] = mapped_column(Integer)


class Transfer(Base):
    __tablename__ = "transfers"
    __table_args__ = (
        UniqueConstraint("player_id", "date", "type", "team_in", "team_out"),
        Index("ix_transfers_player_id", "player_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, nullable=False)
    date: Mapped[date | None] = mapped_column(Date)
    type: Mapped[str | None] = mapped_column(String(32))
    team_in: Mapped[str | None] = mapped_column(Text)
    team_in_id: Mapped[int | None] = mapped_column(Integer)
    team_out: Mapped[str | None] = mapped_column(Text)
    team_out_id: Mapped[int | None] = mapped_column(Integer)
    fee_paid: Mapped[Decimal | None] = mapped_column(Money)


class Sidelined(Base):
    __tablename__ = "sidelined"
    __table_args__ = (
        UniqueConstraint("player_id", "type", "start_date", "end_date"),
        Index("ix_sidelined_player_id", "player_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str | None] = mapped_column(Text)
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)


class Injury(Base):
    __tablename__ = "injuries"
    __table_args__ = (
        UniqueConstraint("player_id", "fixture_id", "type", "reason", "date"),
        Index("ix_injuries_player_id", "player_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, nullable=False)
    fixture_id: Mapped[int | None] = mapped_column(Integer)
    type: Mapped[str | None] = mapped_column(Text)
    reason: Mapped[str | None] = mapped_column(Text)
    date: Mapped[date | None] = mapped_column(Date)


class ExpectedMetric(Base):
    __tablename__ = "expected_metrics"
    __table_args__ = (Index("ix_expected_metrics_player_id", "player_id"),)

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[str] = mapped_column(String(16), primary_key=True)
    source: Mapped[str] = mapped_column(String(32), primary_key=True)
    league_id: Mapped[int] = mapped_column(Integer, nullable=False)
    xg: Mapped[float | None] = mapped_column(Float)
    npxg: Mapped[float | None] = mapped_column(Float)
    xa: Mapped[float | None] = mapped_column(Float)
    xg_per_shot: Mapped[float | None] = mapped_column(Float)
    goals_minus_xg: Mapped[float | None] = mapped_column(Float)
    assists_minus_xa: Mapped[float | None] = mapped_column(Float)
    progressive_passes: Mapped[float | None] = mapped_column(Float)
    progressive_carries: Mapped[float | None] = mapped_column(Float)
    progressive_receptions: Mapped[float | None] = mapped_column(Float)


class MarketValue(Base):
    __tablename__ = "market_values"
    __table_args__ = (Index("ix_market_values_player_id", "player_id"),)

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    market_value_eur: Mapped[Decimal | None] = mapped_column(Money)
    contract_expiry: Mapped[date | None] = mapped_column(Date)
    wage_estimate: Mapped[Decimal | None] = mapped_column(Money)


class MarketValueHistory(Base):
    """Time-series of Transfermarkt market value snapshots scraped from player profile pages."""

    __tablename__ = "market_value_history"
    __table_args__ = (Index("ix_market_value_history_player_id", "player_id"),)

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    value_eur: Mapped[Decimal | None] = mapped_column(Money)


class PlayerRole(Base):
    __tablename__ = "player_roles"

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[str] = mapped_column(String(16), primary_key=True)
    primary_role: Mapped[str] = mapped_column(Text, nullable=False)
    secondary_role: Mapped[str | None] = mapped_column(Text)
    cluster_confidence: Mapped[float | None] = mapped_column(Float)


class RoleTemplate(Base):
    __tablename__ = "role_templates"
    __table_args__ = (UniqueConstraint("role_name", "version"),)

    template_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role_name: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    created_by: Mapped[str | None] = mapped_column(Text)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Brief(Base):
    __tablename__ = "briefs"

    brief_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role_name: Mapped[str] = mapped_column(Text, nullable=False)
    archetype_primary: Mapped[str] = mapped_column(String(64), nullable=False)
    archetype_secondary: Mapped[str | None] = mapped_column(String(64))
    intent: Mapped[str] = mapped_column(Text, nullable=False)
    budget_max_fee: Mapped[Decimal | None] = mapped_column(Money)
    budget_max_wage: Mapped[Decimal | None] = mapped_column(Money)
    budget_max_contract_years: Mapped[int] = mapped_column(Integer, nullable=False)
    age_min: Mapped[int] = mapped_column(Integer, nullable=False)
    age_max: Mapped[int] = mapped_column(Integer, nullable=False)
    league_scope: Mapped[list[Any]] = mapped_column(JSONB, nullable=False)
    timeline: Mapped[date | None] = mapped_column(Date)
    pathway_check_done: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    pathway_player_id: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    created_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    created_by: Mapped[str | None] = mapped_column(Text)
    approved_by: Mapped[str | None] = mapped_column(Text)


class Pipeline(Base):
    __tablename__ = "pipeline"
    __table_args__ = (
        UniqueConstraint("brief_id", "player_id"),
        Index("ix_pipeline_brief_stage", "brief_id", "stage"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    brief_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("briefs.brief_id"),
        nullable=False,
    )
    player_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("players.player_id"),
        nullable=False,
    )
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    archetype_primary: Mapped[str] = mapped_column(String(64), nullable=False)
    archetype_secondary: Mapped[str | None] = mapped_column(String(64))
    intent: Mapped[str | None] = mapped_column(Text)
    added_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    stage_changed_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    stage_changed_by: Mapped[str | None] = mapped_column(Text)
    rejection_reason: Mapped[str | None] = mapped_column(Text)


class ScoutNote(Base):
    __tablename__ = "scout_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("players.player_id"),
        nullable=False,
    )
    fixture_id: Mapped[int | None] = mapped_column(Integer)
    scout_name: Mapped[str] = mapped_column(Text, nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    technical_rating: Mapped[int] = mapped_column(Integer, nullable=False)
    tactical_rating: Mapped[int] = mapped_column(Integer, nullable=False)
    physical_rating: Mapped[int] = mapped_column(Integer, nullable=False)
    mental_rating: Mapped[int] = mapped_column(Integer, nullable=False)
    system_fit_rating: Mapped[int] = mapped_column(Integer, nullable=False)
    notes_text: Mapped[str] = mapped_column(Text, nullable=False)
    video_urls: Mapped[list[str] | None] = mapped_column(JSONB)


class PredictionLog(Base):
    __tablename__ = "predictions_log"
    __table_args__ = (Index("ix_predictions_log_player_brief", "player_id", "brief_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("players.player_id"),
        nullable=False,
    )
    brief_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("briefs.brief_id"),
        nullable=False,
    )
    prediction_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    role_fit_score: Mapped[float | None] = mapped_column(Float)
    l1_performance_score: Mapped[float | None] = mapped_column(Float)
    championship_projection_50th: Mapped[float | None] = mapped_column(Float)
    championship_projection_10th: Mapped[float | None] = mapped_column(Float)
    championship_projection_90th: Mapped[float | None] = mapped_column(Float)
    projected_minutes_share: Mapped[float | None] = mapped_column(Float)
    projected_adaptation_months: Mapped[float | None] = mapped_column(Float)
    availability_risk_prob: Mapped[float | None] = mapped_column(Float)
    financial_value_band_low: Mapped[Decimal | None] = mapped_column(Money)
    financial_value_band_high: Mapped[Decimal | None] = mapped_column(Money)
    var_score: Mapped[float | None] = mapped_column(Float)
    composite_score: Mapped[float | None] = mapped_column(Float)
    archetype_weights_used: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    model_warnings: Mapped[list[Any] | None] = mapped_column(JSONB)
    component_fallbacks: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class Override(Base):
    __tablename__ = "overrides"
    __table_args__ = (Index("ix_overrides_player_brief", "player_id", "brief_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("players.player_id"),
        nullable=False,
    )
    brief_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("briefs.brief_id"),
        nullable=False,
    )
    override_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    overridden_by: Mapped[str] = mapped_column(Text, nullable=False)
    original_model_output: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    decision_made: Mapped[str] = mapped_column(Text, nullable=False)
    reason_category: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_text: Mapped[str] = mapped_column(Text, nullable=False)
    outcome: Mapped[str | None] = mapped_column(Text)


class Outcome(Base):
    __tablename__ = "outcomes"
    __table_args__ = (Index("ix_outcomes_player_brief", "player_id", "brief_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("players.player_id"),
        nullable=False,
    )
    brief_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("briefs.brief_id"),
        nullable=False,
    )
    signed_date: Mapped[date | None] = mapped_column(Date)
    fee_paid: Mapped[Decimal | None] = mapped_column(Money)
    wage_annual: Mapped[Decimal | None] = mapped_column(Money)
    contract_years: Mapped[int | None] = mapped_column(Integer)
    assessment_date_6mo: Mapped[date | None] = mapped_column(Date)
    assessment_date_12mo: Mapped[date | None] = mapped_column(Date)
    assessment_date_18mo: Mapped[date | None] = mapped_column(Date)
    performance_hit: Mapped[bool | None] = mapped_column(Boolean)
    financial_hit: Mapped[bool | None] = mapped_column(Boolean)
    availability_hit: Mapped[bool | None] = mapped_column(Boolean)
    failure_type: Mapped[str | None] = mapped_column(Text)
    failure_notes: Mapped[str | None] = mapped_column(Text)


class WyscoutZoneStat(Base):
    __tablename__ = "wyscout_zone_stats"

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[str] = mapped_column(String(16), primary_key=True)
    zone: Mapped[str] = mapped_column(String(32), primary_key=True)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    export_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class WyscoutSeasonStat(Base):
    __tablename__ = "wyscout_season_stats"
    __table_args__ = (Index("ix_wyscout_season_stats_league_season", "league_id", "season"),)

    player_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("players.player_id"),
        primary_key=True,
    )
    season: Mapped[str] = mapped_column(String(16), primary_key=True)
    league_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_player_name: Mapped[str | None] = mapped_column(Text)
    source_team_name: Mapped[str | None] = mapped_column(Text)
    current_team_name: Mapped[str | None] = mapped_column(Text)
    position: Mapped[str | None] = mapped_column(Text)
    matches_played: Mapped[int | None] = mapped_column(Integer)
    minutes_played: Mapped[int | None] = mapped_column(Integer)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    source_file: Mapped[str | None] = mapped_column(Text)
    import_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class SourcePlayerMapping(Base):
    __tablename__ = "source_player_mappings"
    __table_args__ = (
        UniqueConstraint("source", "source_lookup_key"),
        Index("ix_source_player_mappings_player_id", "player_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    source_lookup_key: Mapped[str] = mapped_column(Text, nullable=False)
    source_player_name: Mapped[str | None] = mapped_column(Text)
    source_team_name: Mapped[str | None] = mapped_column(Text)
    source_player_external_id: Mapped[str | None] = mapped_column(Text)
    league_id: Mapped[int | None] = mapped_column(Integer)
    player_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("players.player_id"),
        nullable=False,
    )
    match_score: Mapped[float | None] = mapped_column(Float)
    matched_by: Mapped[str] = mapped_column(String(32), nullable=False)
    matched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class PathwayPlayer(Base):
    __tablename__ = "pathway_players"

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_name: Mapped[str] = mapped_column(Text, nullable=False)
    birth_date: Mapped[date | None] = mapped_column(Date)
    position: Mapped[str] = mapped_column(Text, nullable=False)
    primary_role: Mapped[str] = mapped_column(Text, nullable=False)
    current_level: Mapped[str] = mapped_column(String(32), nullable=False)
    readiness_estimate_months: Mapped[int | None] = mapped_column(Integer)
    last_assessed_date: Mapped[date | None] = mapped_column(Date)
    assessed_by: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
