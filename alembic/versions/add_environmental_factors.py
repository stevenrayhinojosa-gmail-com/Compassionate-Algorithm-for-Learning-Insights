"""Add environmental factors tables

Revision ID: 004_environmental
Revises: 003_medications
Create Date: 2025-03-17
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '004_environmental'
down_revision = '003_medications'
branch_labels = None
depends_on = None

def upgrade():
    # Create environmental_factors table
    op.create_table(
        'environmental_factors',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('student_id', sa.Integer(), nullable=True),
        sa.Column('date', sa.Date(), nullable=True),
        sa.Column('factor_type', sa.String(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('impact_level', sa.Integer(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['student_id'], ['students.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_environmental_factors_id', 'environmental_factors', ['id'], unique=False)
    op.create_index('ix_environmental_factors_date', 'environmental_factors', ['date'], unique=False)

    # Create seasonal_patterns table
    op.create_table(
        'seasonal_patterns',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('student_id', sa.Integer(), nullable=True),
        sa.Column('season', sa.String(), nullable=True),
        sa.Column('year', sa.Integer(), nullable=True),
        sa.Column('avg_behavior_score', sa.Float(), nullable=True),
        sa.Column('seasonal_impact', sa.Text(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['student_id'], ['students.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_seasonal_patterns_id', 'seasonal_patterns', ['id'], unique=False)

    # Create staff_changes table
    op.create_table(
        'staff_changes',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('student_id', sa.Integer(), nullable=True),
        sa.Column('change_date', sa.Date(), nullable=True),
        sa.Column('staff_role', sa.String(), nullable=True),
        sa.Column('change_type', sa.String(), nullable=True),
        sa.Column('adjustment_period', sa.Integer(), nullable=True),
        sa.Column('impact_observed', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['student_id'], ['students.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_staff_changes_id', 'staff_changes', ['id'], unique=False)
    op.create_index('ix_staff_changes_change_date', 'staff_changes', ['change_date'], unique=False)

    # Create learning_environments table
    op.create_table(
        'learning_environments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('student_id', sa.Integer(), nullable=True),
        sa.Column('start_date', sa.Date(), nullable=True),
        sa.Column('end_date', sa.Date(), nullable=True),
        sa.Column('classroom_type', sa.String(), nullable=True),
        sa.Column('seating_position', sa.String(), nullable=True),
        sa.Column('noise_level', sa.String(), nullable=True),
        sa.Column('lighting_type', sa.String(), nullable=True),
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['student_id'], ['students.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_learning_environments_id', 'learning_environments', ['id'], unique=False)
    op.create_index('ix_learning_environments_start_date', 'learning_environments', ['start_date'], unique=False)

    # Create nutrition_logs table
    op.create_table(
        'nutrition_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('student_id', sa.Integer(), nullable=True),
        sa.Column('date', sa.Date(), nullable=True),
        sa.Column('meal_type', sa.String(), nullable=True),
        sa.Column('food_items', sa.Text(), nullable=True),
        sa.Column('sugar_intake_level', sa.String(), nullable=True),
        sa.Column('protein_intake_level', sa.String(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['student_id'], ['students.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_nutrition_logs_id', 'nutrition_logs', ['id'], unique=False)
    op.create_index('ix_nutrition_logs_date', 'nutrition_logs', ['date'], unique=False)

    # Create routine_changes table
    op.create_table(
        'routine_changes',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('student_id', sa.Integer(), nullable=True),
        sa.Column('change_date', sa.Date(), nullable=True),
        sa.Column('change_type', sa.String(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('duration', sa.Integer(), nullable=True),
        sa.Column('adaptation_level', sa.Integer(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['student_id'], ['students.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_routine_changes_id', 'routine_changes', ['id'], unique=False)
    op.create_index('ix_routine_changes_change_date', 'routine_changes', ['change_date'], unique=False)

def downgrade():
    op.drop_table('routine_changes')
    op.drop_table('nutrition_logs')
    op.drop_table('learning_environments')
    op.drop_table('staff_changes')
    op.drop_table('seasonal_patterns')
    op.drop_table('environmental_factors')
