"""add_skillcorner_tables

Revision ID: c8e2f1a3b906
Revises: fa78f3a94f8d
Create Date: 2026-03-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8e2f1a3b906'
down_revision: Union[str, Sequence[str], None] = 'fa78f3a94f8d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add SkillCorner mapping and data tables."""

    # ------------------------------------------------------------------
    # skillcorner_match_map
    # ------------------------------------------------------------------
    op.create_table(
        'skillcorner_match_map',
        sa.Column('sc_match_id', sa.Integer(), nullable=False),
        sa.Column('fixture_id', sa.Integer(), nullable=True),
        sa.Column('sc_competition_id', sa.Integer(), nullable=True),
        sa.Column('sc_season_id', sa.Integer(), nullable=True),
        sa.Column('sc_competition_edition_id', sa.Integer(), nullable=True),
        sa.Column('match_date', sa.Date(), nullable=True),
        sa.Column('home_team_sc', sa.Text(), nullable=True),
        sa.Column('away_team_sc', sa.Text(), nullable=True),
        sa.Column('match_confidence', sa.Float(), nullable=True),
        sa.Column('matched_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('sc_match_id', name=op.f('pk_skillcorner_match_map')),
    )

    # ------------------------------------------------------------------
    # skillcorner_player_map
    # ------------------------------------------------------------------
    op.create_table(
        'skillcorner_player_map',
        sa.Column('sc_player_id', sa.Integer(), nullable=False),
        sa.Column('player_id', sa.Integer(), nullable=True),
        sa.Column('sc_first_name', sa.Text(), nullable=True),
        sa.Column('sc_last_name', sa.Text(), nullable=True),
        sa.Column('sc_short_name', sa.Text(), nullable=True),
        sa.Column('sc_birthday', sa.Date(), nullable=True),
        sa.Column('match_method', sa.String(length=32), nullable=True),
        sa.Column('match_confidence', sa.Float(), nullable=True),
        sa.Column('matched_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('sc_player_id', name=op.f('pk_skillcorner_player_map')),
    )

    # ------------------------------------------------------------------
    # skillcorner_physical
    # ------------------------------------------------------------------
    op.create_table(
        'skillcorner_physical',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('sc_match_id', sa.Integer(), nullable=False),
        sa.Column('sc_player_id', sa.Integer(), nullable=False),
        sa.Column('fixture_id', sa.Integer(), nullable=True),
        sa.Column('player_id', sa.Integer(), nullable=True),
        sa.Column('player_name', sa.Text(), nullable=True),
        sa.Column('player_birthdate', sa.Date(), nullable=True),
        sa.Column('match_name', sa.Text(), nullable=True),
        sa.Column('match_date', sa.Date(), nullable=True),
        sa.Column('team_id', sa.Integer(), nullable=True),
        sa.Column('team_name', sa.Text(), nullable=True),
        sa.Column('competition_id', sa.Integer(), nullable=True),
        sa.Column('competition_name', sa.Text(), nullable=True),
        sa.Column('season_id', sa.Integer(), nullable=True),
        sa.Column('season_name', sa.Text(), nullable=True),
        sa.Column('competition_edition_id', sa.Integer(), nullable=True),
        sa.Column('position', sa.Text(), nullable=True),
        sa.Column('group', sa.Text(), nullable=True),
        sa.Column('quality_check', sa.Boolean(), nullable=True),
        sa.Column('count_match', sa.Integer(), nullable=True),
        sa.Column('count_match_failed', sa.Integer(), nullable=True),
        sa.Column('minutes_played_per_match', sa.Float(), nullable=True),
        sa.Column('adjusted_min_tip_per_match', sa.Float(), nullable=True),
        sa.Column('adjusted_min_otip_per_match', sa.Float(), nullable=True),
        # per match
        sa.Column('dist_per_match', sa.Float(), nullable=True),
        sa.Column('hsr_dist_per_match', sa.Float(), nullable=True),
        sa.Column('sprint_dist_per_match', sa.Float(), nullable=True),
        sa.Column('count_hsr_per_match', sa.Float(), nullable=True),
        sa.Column('count_sprint_per_match', sa.Float(), nullable=True),
        sa.Column('count_high_accel_per_match', sa.Float(), nullable=True),
        sa.Column('count_high_decel_per_match', sa.Float(), nullable=True),
        sa.Column('top_speed_per_match', sa.Float(), nullable=True),
        sa.Column('dist_tip_per_match', sa.Float(), nullable=True),
        sa.Column('dist_otip_per_match', sa.Float(), nullable=True),
        sa.Column('hsr_dist_tip_per_match', sa.Float(), nullable=True),
        sa.Column('hsr_dist_otip_per_match', sa.Float(), nullable=True),
        sa.Column('sprint_dist_tip_per_match', sa.Float(), nullable=True),
        sa.Column('sprint_dist_otip_per_match', sa.Float(), nullable=True),
        # p90
        sa.Column('dist_p90', sa.Float(), nullable=True),
        sa.Column('hsr_dist_p90', sa.Float(), nullable=True),
        sa.Column('sprint_dist_p90', sa.Float(), nullable=True),
        sa.Column('count_hsr_p90', sa.Float(), nullable=True),
        sa.Column('count_sprint_p90', sa.Float(), nullable=True),
        sa.Column('count_high_accel_p90', sa.Float(), nullable=True),
        sa.Column('count_high_decel_p90', sa.Float(), nullable=True),
        # p60bip
        sa.Column('dist_p60bip', sa.Float(), nullable=True),
        sa.Column('hsr_dist_p60bip', sa.Float(), nullable=True),
        sa.Column('sprint_dist_p60bip', sa.Float(), nullable=True),
        sa.Column('count_hsr_p60bip', sa.Float(), nullable=True),
        sa.Column('count_sprint_p60bip', sa.Float(), nullable=True),
        sa.Column('count_high_accel_p60bip', sa.Float(), nullable=True),
        sa.Column('count_high_decel_p60bip', sa.Float(), nullable=True),
        # p30tip
        sa.Column('dist_p30tip', sa.Float(), nullable=True),
        sa.Column('hsr_dist_p30tip', sa.Float(), nullable=True),
        sa.Column('sprint_dist_p30tip', sa.Float(), nullable=True),
        sa.Column('dist_tip_p30tip', sa.Float(), nullable=True),
        sa.Column('hsr_dist_tip_p30tip', sa.Float(), nullable=True),
        # p30otip
        sa.Column('dist_p30otip', sa.Float(), nullable=True),
        sa.Column('hsr_dist_p30otip', sa.Float(), nullable=True),
        sa.Column('sprint_dist_p30otip', sa.Float(), nullable=True),
        sa.Column('dist_otip_p30otip', sa.Float(), nullable=True),
        sa.Column('hsr_dist_otip_p30otip', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_skillcorner_physical')),
        sa.UniqueConstraint('sc_match_id', 'sc_player_id', name=op.f('uq_skillcorner_physical_sc_match_id_sc_player_id')),
    )
    op.create_index('ix_skillcorner_physical_fixture_player', 'skillcorner_physical', ['fixture_id', 'player_id'])
    op.create_index('ix_skillcorner_physical_match_date', 'skillcorner_physical', ['match_date'])

    # ------------------------------------------------------------------
    # skillcorner_off_ball_runs
    # ------------------------------------------------------------------
    op.create_table(
        'skillcorner_off_ball_runs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('sc_match_id', sa.Integer(), nullable=False),
        sa.Column('sc_player_id', sa.Integer(), nullable=False),
        sa.Column('fixture_id', sa.Integer(), nullable=True),
        sa.Column('player_id', sa.Integer(), nullable=True),
        sa.Column('player_name', sa.Text(), nullable=True),
        sa.Column('player_birthdate', sa.Date(), nullable=True),
        sa.Column('match_name', sa.Text(), nullable=True),
        sa.Column('match_date', sa.Date(), nullable=True),
        sa.Column('team_id', sa.Integer(), nullable=True),
        sa.Column('team_name', sa.Text(), nullable=True),
        sa.Column('competition_id', sa.Integer(), nullable=True),
        sa.Column('competition_name', sa.Text(), nullable=True),
        sa.Column('season_id', sa.Integer(), nullable=True),
        sa.Column('season_name', sa.Text(), nullable=True),
        sa.Column('competition_edition_id', sa.Integer(), nullable=True),
        sa.Column('position', sa.Text(), nullable=True),
        sa.Column('group', sa.Text(), nullable=True),
        sa.Column('quality_check', sa.Boolean(), nullable=True),
        sa.Column('count_match', sa.Integer(), nullable=True),
        sa.Column('count_match_failed', sa.Integer(), nullable=True),
        sa.Column('third', sa.Text(), nullable=True),
        sa.Column('channel', sa.Text(), nullable=True),
        sa.Column('minutes_played_per_match', sa.Float(), nullable=True),
        sa.Column('adjusted_min_tip_per_match', sa.Float(), nullable=True),
        sa.Column('count_run_in_behind_in_sample', sa.Float(), nullable=True),
        sa.Column('count_dangerous_run_in_behind_per_match', sa.Float(), nullable=True),
        sa.Column('run_in_behind_threat_per_match', sa.Float(), nullable=True),
        sa.Column('count_run_in_behind_leading_to_goal_per_match', sa.Float(), nullable=True),
        sa.Column('count_run_in_behind_targeted_per_match', sa.Float(), nullable=True),
        sa.Column('count_run_in_behind_received_per_match', sa.Float(), nullable=True),
        sa.Column('count_run_in_behind_leading_to_shot_per_match', sa.Float(), nullable=True),
        sa.Column('run_in_behind_targeted_threat_per_match', sa.Float(), nullable=True),
        sa.Column('run_in_behind_received_threat_per_match', sa.Float(), nullable=True),
        sa.Column('count_dangerous_run_in_behind_targeted_per_match', sa.Float(), nullable=True),
        sa.Column('count_dangerous_run_in_behind_received_per_match', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_skillcorner_off_ball_runs')),
        sa.UniqueConstraint('sc_match_id', 'sc_player_id', name=op.f('uq_skillcorner_off_ball_runs_sc_match_id_sc_player_id')),
    )
    op.create_index('ix_skillcorner_off_ball_runs_fixture_player', 'skillcorner_off_ball_runs', ['fixture_id', 'player_id'])
    op.create_index('ix_skillcorner_off_ball_runs_match_date', 'skillcorner_off_ball_runs', ['match_date'])

    # ------------------------------------------------------------------
    # skillcorner_pressure
    # ------------------------------------------------------------------
    op.create_table(
        'skillcorner_pressure',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('sc_match_id', sa.Integer(), nullable=False),
        sa.Column('sc_player_id', sa.Integer(), nullable=False),
        sa.Column('fixture_id', sa.Integer(), nullable=True),
        sa.Column('player_id', sa.Integer(), nullable=True),
        sa.Column('player_name', sa.Text(), nullable=True),
        sa.Column('player_birthdate', sa.Date(), nullable=True),
        sa.Column('match_name', sa.Text(), nullable=True),
        sa.Column('match_date', sa.Date(), nullable=True),
        sa.Column('team_id', sa.Integer(), nullable=True),
        sa.Column('team_name', sa.Text(), nullable=True),
        sa.Column('competition_id', sa.Integer(), nullable=True),
        sa.Column('competition_name', sa.Text(), nullable=True),
        sa.Column('season_id', sa.Integer(), nullable=True),
        sa.Column('season_name', sa.Text(), nullable=True),
        sa.Column('competition_edition_id', sa.Integer(), nullable=True),
        sa.Column('position', sa.Text(), nullable=True),
        sa.Column('group', sa.Text(), nullable=True),
        sa.Column('quality_check', sa.Boolean(), nullable=True),
        sa.Column('count_match', sa.Integer(), nullable=True),
        sa.Column('count_match_failed', sa.Integer(), nullable=True),
        sa.Column('third', sa.Text(), nullable=True),
        sa.Column('channel', sa.Text(), nullable=True),
        sa.Column('minutes_played_per_match', sa.Float(), nullable=True),
        sa.Column('adjusted_min_tip_per_match', sa.Float(), nullable=True),
        sa.Column('count_high_pressure_received_in_sample', sa.Float(), nullable=True),
        sa.Column('count_high_pressure_received_per_match', sa.Float(), nullable=True),
        sa.Column('count_forced_losses_under_high_pressure_per_match', sa.Float(), nullable=True),
        sa.Column('count_ball_retention_under_high_pressure_per_match', sa.Float(), nullable=True),
        sa.Column('ball_retention_ratio_under_high_pressure', sa.Float(), nullable=True),
        sa.Column('ball_retention_added_under_high_pressure_per_match', sa.Float(), nullable=True),
        sa.Column('pass_completion_ratio_under_high_pressure', sa.Float(), nullable=True),
        sa.Column('count_pass_attempts_under_high_pressure_per_match', sa.Float(), nullable=True),
        sa.Column('count_completed_passes_under_high_pressure_per_match', sa.Float(), nullable=True),
        sa.Column('count_dangerous_pass_attemps_under_high_pressure_per_match', sa.Float(), nullable=True),
        sa.Column('count_completed_dangerous_passes_under_high_pressure_per_match', sa.Float(), nullable=True),
        sa.Column('dangerous_pass_completion_ratio_under_high_pressure', sa.Float(), nullable=True),
        sa.Column('count_difficult_pass_attempts_under_high_pressure_per_match', sa.Float(), nullable=True),
        sa.Column('count_completed_difficult_passes_under_high_pressure_per_match', sa.Float(), nullable=True),
        sa.Column('difficult_pass_completion_ratio_under_high_pressure', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_skillcorner_pressure')),
        sa.UniqueConstraint('sc_match_id', 'sc_player_id', name=op.f('uq_skillcorner_pressure_sc_match_id_sc_player_id')),
    )
    op.create_index('ix_skillcorner_pressure_fixture_player', 'skillcorner_pressure', ['fixture_id', 'player_id'])
    op.create_index('ix_skillcorner_pressure_match_date', 'skillcorner_pressure', ['match_date'])

    # ------------------------------------------------------------------
    # skillcorner_passes
    # ------------------------------------------------------------------
    op.create_table(
        'skillcorner_passes',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('sc_match_id', sa.Integer(), nullable=False),
        sa.Column('sc_player_id', sa.Integer(), nullable=False),
        sa.Column('fixture_id', sa.Integer(), nullable=True),
        sa.Column('player_id', sa.Integer(), nullable=True),
        sa.Column('player_name', sa.Text(), nullable=True),
        sa.Column('player_birthdate', sa.Date(), nullable=True),
        sa.Column('match_name', sa.Text(), nullable=True),
        sa.Column('match_date', sa.Date(), nullable=True),
        sa.Column('team_id', sa.Integer(), nullable=True),
        sa.Column('team_name', sa.Text(), nullable=True),
        sa.Column('competition_id', sa.Integer(), nullable=True),
        sa.Column('competition_name', sa.Text(), nullable=True),
        sa.Column('season_id', sa.Integer(), nullable=True),
        sa.Column('season_name', sa.Text(), nullable=True),
        sa.Column('competition_edition_id', sa.Integer(), nullable=True),
        sa.Column('position', sa.Text(), nullable=True),
        sa.Column('group', sa.Text(), nullable=True),
        sa.Column('quality_check', sa.Boolean(), nullable=True),
        sa.Column('count_match', sa.Integer(), nullable=True),
        sa.Column('count_match_failed', sa.Integer(), nullable=True),
        sa.Column('third', sa.Text(), nullable=True),
        sa.Column('channel', sa.Text(), nullable=True),
        sa.Column('minutes_played_per_match', sa.Float(), nullable=True),
        sa.Column('adjusted_min_tip_per_match', sa.Float(), nullable=True),
        sa.Column('count_opportunities_to_pass_to_run_in_behind_in_sample', sa.Float(), nullable=True),
        sa.Column('count_opportunities_to_pass_to_run_in_behind_per_match', sa.Float(), nullable=True),
        sa.Column('count_pass_attempts_to_run_in_behind_per_match', sa.Float(), nullable=True),
        sa.Column('pass_opportunities_to_run_in_behind_threat_per_match', sa.Float(), nullable=True),
        sa.Column('run_in_behind_to_which_pass_attempted_threat_per_match', sa.Float(), nullable=True),
        sa.Column('pass_completion_ratio_to_run_in_behind', sa.Float(), nullable=True),
        sa.Column('count_run_in_behind_by_teammate_per_match', sa.Float(), nullable=True),
        sa.Column('run_in_behind_to_which_pass_completed_threat_per_match', sa.Float(), nullable=True),
        sa.Column('count_completed_pass_to_run_in_behind_per_match', sa.Float(), nullable=True),
        sa.Column('count_completed_pass_to_run_in_behind_leading_to_shot_per_match', sa.Float(), nullable=True),
        sa.Column('count_completed_pass_to_run_in_behind_leading_to_goal_per_match', sa.Float(), nullable=True),
        sa.Column('count_pass_opportunities_to_dangerous_run_in_behind_per_match', sa.Float(), nullable=True),
        sa.Column('count_pass_attempts_to_dangerous_run_in_behind_per_match', sa.Float(), nullable=True),
        sa.Column('count_completed_pass_to_dangerous_run_in_behind_per_match', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_skillcorner_passes')),
        sa.UniqueConstraint('sc_match_id', 'sc_player_id', name=op.f('uq_skillcorner_passes_sc_match_id_sc_player_id')),
    )
    op.create_index('ix_skillcorner_passes_fixture_player', 'skillcorner_passes', ['fixture_id', 'player_id'])
    op.create_index('ix_skillcorner_passes_match_date', 'skillcorner_passes', ['match_date'])


def downgrade() -> None:
    """Drop SkillCorner tables."""
    op.drop_index('ix_skillcorner_passes_match_date', table_name='skillcorner_passes')
    op.drop_index('ix_skillcorner_passes_fixture_player', table_name='skillcorner_passes')
    op.drop_table('skillcorner_passes')

    op.drop_index('ix_skillcorner_pressure_match_date', table_name='skillcorner_pressure')
    op.drop_index('ix_skillcorner_pressure_fixture_player', table_name='skillcorner_pressure')
    op.drop_table('skillcorner_pressure')

    op.drop_index('ix_skillcorner_off_ball_runs_match_date', table_name='skillcorner_off_ball_runs')
    op.drop_index('ix_skillcorner_off_ball_runs_fixture_player', table_name='skillcorner_off_ball_runs')
    op.drop_table('skillcorner_off_ball_runs')

    op.drop_index('ix_skillcorner_physical_match_date', table_name='skillcorner_physical')
    op.drop_index('ix_skillcorner_physical_fixture_player', table_name='skillcorner_physical')
    op.drop_table('skillcorner_physical')

    op.drop_table('skillcorner_player_map')
    op.drop_table('skillcorner_match_map')
